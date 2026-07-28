import os
import sys
import argparse
from utils.utils import (
    _extract_gpu_arg_early,
    print_cuda_info,
    get_gpu_name,
    get_gpu_memory_gb,
    reset_peak_gpu_memory,
    get_peak_gpu_memory_gb,
)
from prompts.gemma4_labeling import (
    ENTITY_SPEC, SYSTEM_A, USER_A,
    ENTITY_DEFS, SYSTEM_B, USER_B,
    ENTITIES, STATES, USER_REPAIR
)

# gpu must be specified before cuda initializes — exit early if missing
_EARLY_GPU_ID = _extract_gpu_arg_early()
if not _EARLY_GPU_ID:
    print("Error: --gpu is required for this script", file=sys.stderr)
    sys.exit(1)
os.environ["CUDA_VISIBLE_DEVICES"] = _EARLY_GPU_ID

import time
from pathlib import Path
import torch
import json
import pandas as pd
from transformers import AutoProcessor, AutoModelForMultimodalLM
from huggingface_hub import hf_hub_download


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create LLM labels from the cleaned report.")
    parser.add_argument("--input_file", type=str, required=True, help="Input Parquet file with clean_report column")
    parser.add_argument("--output_file", type=str, required=True, help="Output Parquet file path")
    parser.add_argument("--gpu", type=str, required=True, help="Physical GPU ID")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-12B-it")
    parser.add_argument("--label_strategy", nargs="+", type=str, default=["A", "B"])
    # parser.add_argument("--save_every", type=int, default=30, help="Save every N samples")
    # parser.add_argument("--max_retries", type=int, default=2, help="Maximum number of LLM generation retries if the JSON is invalid")
    return parser


max_retries=2


def generate_json(
    system_prompt: str, user_prompt: str, processor, 
    temperature: float = 1.0, max_new_tokens: int = 512, do_sample: bool = False
    ) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "{"}
    ]
    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt",
        add_generation_prompt=False,  # should be False with continue_final_message - cannot start a new tern when continuing
        continue_final_message=True,
        enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        # completly silence t during greedy decoding:
        kwargs = {"temperature": temperature} if do_sample else {}

        out = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=do_sample, 
            min_new_tokens=5,
            **kwargs
            )
        gen = out[0][input_len:]

    generated_text = processor.decode(gen, skip_special_tokens=True).strip()   
    return "{" + generated_text


def _strip_fences(s: str) -> str:
    s = re.sub(r"^```(?:json)?\s*", "", s.strip())
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    return m.group(0) if m else s


def _valid_entity_obj(o) -> bool:
    return (isinstance(o, dict)
            and o.get("state") in STATES
            and isinstance(o.get("suspected"), bool)
            and isinstance(o.get("anatomically_na"), bool)
            and isinstance(o.get("evidence"), str))


def parse_json_with_retry(system_prompt, user_prompt, processor, validator, max_retries=max_retries):
    raw = generate_json(system_prompt, user_prompt, processor)
    for attempt in range(max_retries + 1):
        try:
            obj = json.loads(_strip_fences(raw))
            if validator(obj):
                return obj, raw, attempt
        except json.JSONDecodeError:
            pass
        if attempt == max_retries:
            break
        # raw = generate_json(system_prompt, USER_REPAIR.format(bad=raw[:1000]))
        if raw.lstrip().startswith("{"):
            # looks like JSON, just broken -> repair
            raw = generate_json(system_prompt, USER_REPAIR.format(bad=raw[:1000]))
        else:
            # not JSON at all -> resend the ORIGINAL prompt, sampled to break determinism
            raw = generate_json(system_prompt, user_prompt,
                                do_sample=True, temperature=0.7)
    return None, raw, max_retries


NULL_ENT = {"state": pd.NA, "suspected": pd.NA, "anatomically_na": pd.NA, "evidence": pd.NA}


def extract_strategy_a(text, processor):
    if not isinstance(text, str) or not text.strip():
        return {f"A_{e}_{k}": v for e in ENTITIES for k, v in NULL_ENT.items()} | {"A_parse_ok": False, "A_n_retries": 0}
    validator = lambda o: isinstance(o, dict) and all(_valid_entity_obj(o.get(e)) for e in ENTITIES)
    obj, raw, n_ret = parse_json_with_retry(SYSTEM_A, USER_A.format(text=text), processor, validator)
    if obj is None:
        return {f"A_{e}_{k}": v for e in ENTITIES for k, v in NULL_ENT.items()} | {
            "A_parse_ok": False, "A_n_retries": n_ret, "A_raw": raw}
    out = {}
    for e in ENTITIES:
        for k in ["state", "suspected", "anatomically_na", "evidence"]:
            out[f"A_{e}_{k}"] = obj[e][k]
    out["A_parse_ok"] = True
    out["A_n_retries"] = n_ret
    return out


def extract_strategy_b(text, processor):
    out = {}
    if not isinstance(text, str) or not text.strip():
        return {f"B_{e}_{k}": v for e in ENTITIES for k, v in NULL_ENT.items()} | {"B_parse_ok": False, "B_n_retries": 0}
    ok, total_ret = True, 0
    for e in ENTITIES:
        up = USER_B.format(entity_name=e, entity_def=ENTITY_DEFS[e], text=text)
        obj, raw, n_ret = parse_json_with_retry(SYSTEM_B, up, processor, _valid_entity_obj)
        total_ret += n_ret
        if obj is None:
            ok = False
            out[f"B_{e}_raw"] = raw
            for k, v in NULL_ENT.items():
                out[f"B_{e}_{k}"] = v
        else:
            for k in ["state", "suspected", "anatomically_na", "evidence"]:
                out[f"B_{e}_{k}"] = obj[k]
    out["B_parse_ok"] = ok
    out["B_n_retries"] = total_ret
    return out


KEYS = ["AccessionNumber"]
METHOD_FNS = {"A": extract_strategy_a, "B": extract_strategy_b}


def run_labels_on_reports(main_df, processor, methods=("A", "B"), show_progress=True):
    unknown = set(methods) - set(METHOD_FNS)
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    dedup_subset = KEYS + ["clean_report"]
    sliced = main_df.reset_index(drop=True).copy()
    sliced["_grp"] = sliced.groupby(dedup_subset, dropna=False).ngroup()
    unique_rows = sliced.drop_duplicates("_grp").set_index("_grp")

    iterator = unique_rows.iterrows()
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=len(unique_rows))
        except ImportError:
            pass

    records = {}
    for grp, row in iterator:
        txt = row["clean_report"]
        rec = {}
        for m in methods:
            rec.update(METHOD_FNS[m](txt, processor))
        records[grp] = rec

    label_df = pd.DataFrame.from_dict(records, orient="index")
    return sliced.join(label_df, on="_grp").drop(columns="_grp")



def main() -> None:
    t_start = time.time()

    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_file)
    df_clean = pd.read_parquet(input_path)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_cuda_info()

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        )
    model.to(device)
    model.eval()
    mem_after_model = get_gpu_memory_gb()
    reset_peak_gpu_memory()

    processor = AutoProcessor.from_pretrained(args.model_id)

    if processor.chat_template is None:
        tpl = hf_hub_download(args.model_id, "chat_template.jinja")
        with open(tpl) as f:
            processor.chat_template = f.read()

    labels_df = run_labels_on_reports(df_clean, processor, methods=(args.label_strategy))

    labels_df.to_parquet(output_path)

    print("runtime_seconds:", round(time.time() - t_start, 2))
    print("gpu_name:", get_gpu_name())
    print("model_id:", args.model_id,)
    print("memory_after_model_load_gb:", mem_after_model,)
    print("peak_memory_embedding_gb:", get_peak_gpu_memory_gb(),)

if __name__ == "__main__":
    main()