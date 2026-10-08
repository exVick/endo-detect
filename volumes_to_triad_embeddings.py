"""
frozen triad swin-b (simmim) embeddings of the core-8 mri series: one 1488-d
vector per series and geometry. all core-8 series are embedded, duplicates
included; they are flagged in the keys table and resolved later.

geometries (both 96^3, canonical axes: 0 -> R, 1 -> A, 2 -> S):
  A  deform to fit the whole: p0.5/p99.5 min-max, whole volume zoomed to 96^3
  B  crop to keep proportions: zoomed to 1 mm isotropic, p0.5/p99.5 min-max over
     the whole volume, centre 96^3 crop

outputs in --out_dir:
  triad_swinb_A.npy, triad_swinb_B.npy  (n_series, 1488) float32, same row order
  triad_swinb_keys.parquet              one row per series, row order of the .npy files
  run_config.json                       arguments, versions, checkpoint hash, layout
  qc/qc_<accession>_<geometry>.png      mid-planes of the 96^3 inputs (first --n_qc accessions)
  decisions.jsonl                       one line appended per run (or --decisions)

usage:
  python volumes_to_triad_embeddings.py --frames manifest_frames.parquet --ckpt Triad-SwinB-SimMIM.pth --out_dir triad_embeddings --device cuda:3 [--limit 2]
"""

import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from scipy.ndimage import zoom
from tqdm.auto import tqdm

TARGET = 96
EMB_DIM = 1488
# column slices of the 1488-d vector: average-pooled, layer-normed hidden state of each stage
STAGE_SLICES = {"x0": [0, 48], "x1": [48, 144], "x2": [144, 336], "x3": [336, 720], "x4": [720, 1488]}
# canonical direction of each patient axis in lps coordinates: R = -x, A = -y, S = +z
WANT = np.array([-1.0, -1.0, 1.0])


# ---------------------------------------------------------------- model (QuickStart.py, args fixed)

class Swin(nn.Module):
    """QuickStart.py Swin with in_channels 1, feature_size 48, spatial_dims 3, no drop path."""

    def __init__(self):
        super().__init__()
        self.swinViT = SwinViT(
            in_chans=1, embed_dim=48, window_size=ensure_tuple_rep(7, 3), patch_size=ensure_tuple_rep(2, 3),
            depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, use_checkpoint=False,
            spatial_dims=3, use_v2=True,
        )

    def forward(self, x):
        hs = self.swinViT(x)
        return torch.cat([F.adaptive_avg_pool3d(h, 1).flatten(1) for h in hs], dim=1)


class TriadHead(nn.Module):
    """wrapper whose state-dict keys match the released checkpoint (backbone.swinViT.*)."""

    def __init__(self):
        super().__init__()
        self.backbone = Swin()

    def forward(self, x):
        return self.backbone(x)


def load_model(ckpt_path, device):
    """builds TriadHead, loads the checkpoint strictly and returns it in eval mode on the device."""
    model = TriadHead()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


@torch.no_grad()
def embed(model, vols, device):
    """embeds a list of 96^3 float32 arrays; returns (n, 1488) float32."""
    x = torch.from_numpy(np.stack(vols)[:, None]).to(device)
    return model(x).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- selection

def select_core(frames):
    """core-8 cohort: is_core rows of the accessions with exactly 8 distinct SeriesDescription values."""
    core = frames[frames["is_core"]]
    n_types = core.groupby("AccessionNumber")["SeriesDescription"].transform("nunique")
    return core[n_types == 8].copy()


def series_table(core):
    """one row per series; is_dup marks types that occur more than once in the accession."""
    ser = core.groupby(["AccessionNumber", "SeriesInstanceUID"]).agg(
        StudyInstanceUID=("StudyInstanceUID", "first"),
        SeriesDescription=("SeriesDescription", "first"),
        storage_kind=("storage_kind", "first"),
        n_slices=("slice_rank", "size"),
    ).reset_index()
    ser["is_dup"] = ser.groupby(["AccessionNumber", "SeriesDescription"])["SeriesInstanceUID"].transform("size") > 1
    return ser.sort_values(["AccessionNumber", "SeriesDescription", "SeriesInstanceUID"], ignore_index=True)


# ---------------------------------------------------------------- reading

def _fg_attr(ds, i, seq_name, attr):
    """looks an attribute up in the per-frame functional group of frame i first,
    then in the shared functional group; None if absent in both."""
    for fg_name, idx in (("PerFrameFunctionalGroupsSequence", i), ("SharedFunctionalGroupsSequence", 0)):
        fgs = getattr(ds, fg_name, None)
        if fgs is None or len(fgs) <= idx:
            continue
        seq = getattr(fgs[idx], seq_name, None)
        if seq is None or len(seq) == 0:
            continue
        val = getattr(seq[0], attr, None)
        if val is not None:
            return val
    return None


def load_series(g):
    """reads the frames of one series in slice_rank order into a float32 stack
    (slices, rows, cols). a multi-frame file is read once and indexed with
    frame_index - 1; a classic series has one file per frame. returns the stack,
    the dataset the header is taken from and the frame index within it."""
    g = g.sort_values("slice_rank")
    if g["storage_kind"].iloc[0] == "multiframe":
        assert g["FilePath"].nunique() == 1, "multi-frame series spread over several files"
        ds = pydicom.dcmread(g["FilePath"].iloc[0])
        arr = ds.pixel_array
        arr = arr[None] if arr.ndim == 2 else arr
        idx = g["frame_index"].to_numpy() - 1
        return arr[idx].astype(np.float32), ds, int(idx[0])
    dss = [pydicom.dcmread(p) for p in g["FilePath"]]
    return np.stack([d.pixel_array for d in dss]).astype(np.float32), dss[0], 0


def read_geometry(ds, frame_i, g):
    """direction vectors (lps) and spacings (mm) of the three stack axes:
    axis 0 = slices in ascending slice_pos, i.e. along the row x column normal,
    axis 1 = image rows (second iop triplet), axis 2 = image columns (first iop triplet).
    the slice spacing is the median slice_pos step."""
    if hasattr(ds, "PerFrameFunctionalGroupsSequence"):
        iop = _fg_attr(ds, frame_i, "PlaneOrientationSequence", "ImageOrientationPatient")
        psp = _fg_attr(ds, frame_i, "PixelMeasuresSequence", "PixelSpacing")
    else:
        iop, psp = ds.ImageOrientationPatient, ds.PixelSpacing
    iop = np.asarray([float(v) for v in iop])
    psp = [float(v) for v in psp]
    along_row, down_col = iop[:3], iop[3:]
    normal = np.cross(along_row, down_col)
    pos = g.sort_values("slice_rank")["slice_pos"].to_numpy(float)
    dz = float(np.median(np.diff(pos)))
    return [normal, down_col, along_row], [dz, psp[0], psp[1]]


# ---------------------------------------------------------------- geometry

def oblique_deg(dirs):
    """largest angle (degrees) between an axis direction and its nearest patient axis."""
    return max(float(np.degrees(np.arccos(np.clip(np.abs(v).max() / np.linalg.norm(v), -1, 1)))) for v in dirs)


def canonicalize(vol, dirs, spacing):
    """permutes and flips the stack so that axis 0 points to R, axis 1 to A, axis 2 to S
    (nearest patient axis per stack axis, no rotation). returns the volume, the spacing
    in the new axis order, perm (new axis i = old axis perm[i]) and the flipped old axes."""
    perm, flipped = [None] * 3, []
    for a, v in enumerate(dirs):
        p = int(np.argmax(np.abs(v)))
        if np.sign(v[p]) != WANT[p]:
            vol = np.flip(vol, axis=a)
            flipped.append(a)
        perm[p] = a
    return np.ascontiguousarray(np.transpose(vol, perm)), [spacing[a] for a in perm], perm, flipped


def minmax_p(vol, q_lo=0.5, q_hi=99.5, eps=1e-6):
    """triad normalisation: clip to the percentiles of all finite voxels, rescale to [0, 1]."""
    lo, hi = np.percentile(vol[np.isfinite(vol)], [q_lo, q_hi])
    return ((np.clip(vol, lo, hi) - lo) / (hi - lo + eps)).astype(np.float32), float(lo), float(hi)


def center_crop(vol, size=TARGET):
    """centre crop to size per axis; axes shorter than size are zero-padded symmetrically first.
    returns the crop, its start index per axis (after padding) and whether padding was needed."""
    pads = [((size - s) // 2, size - s - (size - s) // 2) if s < size else (0, 0) for s in vol.shape]
    padded = any(p != (0, 0) for p in pads)
    if padded:
        vol = np.pad(vol, pads)
    starts = [s // 2 - size // 2 for s in vol.shape]
    sl = tuple(slice(st, st + size) for st in starts)
    return vol[sl], starts, padded


def prep_a(vol, spacing):
    """geometry A: normalise, then zoom the whole volume to TARGET^3.
    returns the input, the per-axis voxel size it represents (mm) and the clip range."""
    v, lo, hi = minmax_p(vol)
    v = zoom(v, [TARGET / s for s in v.shape], order=1)
    voxel_mm = [s * sp / TARGET for s, sp in zip(vol.shape, spacing)]
    return v.astype(np.float32), voxel_mm, lo, hi


def prep_b(vol, spacing):
    """geometry B: zoom to 1 mm isotropic, normalise over the whole volume, centre crop TARGET^3.
    returns the input, the 1 mm shape, the crop start, the pad flag and the clip range."""
    v = zoom(vol, spacing, order=1)
    v, lo, hi = minmax_p(v)
    shape_1mm = list(v.shape)
    v, starts, padded = center_crop(v)
    return v.astype(np.float32), shape_1mm, starts, padded, lo, hi


# ---------------------------------------------------------------- outputs

def qc_figure(inputs, acc, geom, path):
    """saves the axial / coronal / sagittal mid-planes of the TARGET^3 inputs of one accession."""
    fig, axes = plt.subplots(len(inputs), 3, figsize=(9, 3 * len(inputs)), squeeze=False)
    m = TARGET // 2
    for r, (desc, v) in enumerate(inputs):
        views = [(v[:, :, m].T, "axial"), (v[:, m, :].T, "coronal"), (v[m, :, :].T, "sagittal")]
        for c, (img, name) in enumerate(views):
            ax = axes[r, c]
            ax.imshow(img, cmap="bone", origin="lower", vmin=0, vmax=1)
            ax.set_title(f"{desc} | {name}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"{acc} | geometry {geom}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def backup(path):
    """renames an existing output to <name>.<timestamp>.bak before it is overwritten."""
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(path), str(path.with_name(f"{path.name}.{ts}.bak")))


def sha256(path, chunk=1 << 20):
    """sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="manifest_frames parquet")
    ap.add_argument("--ckpt", required=True, help="Triad-SwinB-SimMIM.pth")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N accessions only (dry run)")
    ap.add_argument("--n_qc", type=int, default=3, help="accessions with qc figures")
    ap.add_argument("--decisions", default=None, help="decisions.jsonl to append to (default: out_dir)")
    args = ap.parse_args()

    t_start = time.time()
    out_dir = Path(args.out_dir)
    (out_dir / "qc").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # -- cohort --
    frames = pd.read_parquet(args.frames)
    core = select_core(frames)
    ser = series_table(core)
    accs = sorted(ser["AccessionNumber"].unique())
    if args.limit:
        accs = accs[: args.limit]
        ser = ser[ser["AccessionNumber"].isin(accs)].reset_index(drop=True)
    qc_accs = set(accs[: args.n_qc])
    frames_by_series = {k: g for k, g in core.groupby("SeriesInstanceUID")}
    print(f"{len(accs)} accessions, {len(ser)} series ({int(ser.is_dup.sum())} flagged as duplicates)")

    model = load_model(args.ckpt, device)

    # -- loop over series: read, canonicalise, prepare both geometries, embed in batches --
    rows, embs = [], {"A": [], "B": []}
    buf = {"A": [], "B": []}
    qc_inputs = {}
    for s in tqdm(ser.itertuples(), total=len(ser)):
        g = frames_by_series[s.SeriesInstanceUID]
        vol, ds, frame_i = load_series(g)
        dirs, spacing = read_geometry(ds, frame_i, g)
        obl = oblique_deg(dirs)
        vol_c, spacing_c, perm, flipped = canonicalize(vol, dirs, spacing)

        a, a_voxel_mm, a_lo, a_hi = prep_a(vol_c, spacing_c)
        b, b_shape_1mm, b_starts, b_padded, b_lo, b_hi = prep_b(vol_c, spacing_c)
        assert a.shape == (TARGET,) * 3 and b.shape == (TARGET,) * 3, f"input shape {a.shape} / {b.shape}"
        buf["A"].append(a)
        buf["B"].append(b)

        if s.AccessionNumber in qc_accs:
            qc_inputs.setdefault(s.AccessionNumber, []).append((s.SeriesDescription, a, b))

        rows.append(dict(
            AccessionNumber=s.AccessionNumber, StudyInstanceUID=s.StudyInstanceUID,
            SeriesInstanceUID=s.SeriesInstanceUID, SeriesDescription=s.SeriesDescription,
            SeriesNumber=int(getattr(ds, "SeriesNumber", -1) or -1), SeriesTime=str(getattr(ds, "SeriesTime", "")),
            storage_kind=s.storage_kind, n_slices=int(s.n_slices), is_dup=bool(s.is_dup),
            oblique_max_deg=round(obl, 3), native_shape=list(vol.shape), canonical_shape=list(vol_c.shape),
            canonical_spacing_mm=[round(x, 4) for x in spacing_c], perm=perm, flipped_axes=flipped,
            a_voxel_mm=[round(x, 4) for x in a_voxel_mm], a_clip_lo=a_lo, a_clip_hi=a_hi,
            b_shape_1mm=b_shape_1mm, b_crop_start=b_starts, b_padded=b_padded, b_clip_lo=b_lo, b_clip_hi=b_hi,
        ))

        if len(buf["A"]) == args.batch_size:
            for k in buf:
                embs[k].append(embed(model, buf[k], device))
                buf[k] = []
    for k in buf:
        if buf[k]:
            embs[k].append(embed(model, buf[k], device))

    emb = {k: np.concatenate(v) for k, v in embs.items()}
    keys = pd.DataFrame(rows)

    # -- final checks --
    for k, e in emb.items():
        assert e.shape == (len(keys), EMB_DIM), f"{k}: {e.shape}"
        assert np.isfinite(e).all(), f"{k}: non-finite values"

    # -- save --
    paths = {
        "A": out_dir / "triad_swinb_A.npy",
        "B": out_dir / "triad_swinb_B.npy",
        "keys": out_dir / "triad_swinb_keys.parquet",
        "config": out_dir / "run_config.json",
    }
    for p in paths.values():
        backup(p)
    np.save(paths["A"], emb["A"])
    np.save(paths["B"], emb["B"])
    keys.to_parquet(paths["keys"], index=False)

    for acc, items in qc_inputs.items():
        for gi, geom in ((1, "A"), (2, "B")):
            qc_figure([(it[0], it[gi]) for it in items], acc, geom, out_dir / "qc" / f"qc_{acc}_{geom}.png")

    config = dict(
        script="embed_triad.py", timestamp=datetime.now().isoformat(timespec="seconds"),
        args=vars(args), checkpoint_sha256=sha256(args.ckpt),
        versions=dict(python=platform.python_version(), torch=torch.__version__, monai=monai.__version__,
                      pydicom=pydicom.__version__, scipy=scipy.__version__, numpy=np.__version__),
        model="Triad-SwinB-SimMIM, QuickStart extraction (swinViT hidden states, layer-normed, avg-pooled, concat)",
        target=TARGET, emb_dim=EMB_DIM, stage_slices=STAGE_SLICES,
        canonical_axes="0 -> R, 1 -> A, 2 -> S (permute + flip to nearest patient axis, no rotation)",
        normalisation="clip to p0.5/p99.5 of all finite voxels, rescale to [0, 1]",
        geometry_A="normalise native canonical volume, zoom whole volume to 96^3 (order 1)",
        geometry_B="zoom to 1 mm isotropic (order 1), normalise whole volume, centre 96^3 crop",
        n_accessions=len(accs), n_series=len(keys), n_dup_series=int(keys.is_dup.sum()),
        elapsed_s=round(time.time() - t_start, 1),
    )
    paths["config"].write_text(json.dumps(config, indent=2))

    decisions = Path(args.decisions) if args.decisions else out_dir / "decisions.jsonl"
    with open(decisions, "a") as f:
        f.write(json.dumps(dict(
            timestamp=config["timestamp"], script="embed_triad.py", event="triad_embeddings_written",
            n_accessions=len(accs), n_series=len(keys), outputs=[str(p) for p in paths.values()],
            checkpoint_sha256=config["checkpoint_sha256"],
        )) + "\n")

    print(f"done: {len(keys)} series, A {emb['A'].shape}, B {emb['B'].shape}, "
          f"padded in B: {int(keys.b_padded.sum())}, {config['elapsed_s']} s")


if __name__ == "__main__":
    main()
