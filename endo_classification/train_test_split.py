import pandas as pd
import numpy as np


CONDITIONS = ("die", "adenomyosis", "endometrioma")
 
 
def build_accession_table(main_df, conditions=CONDITIONS):
    """
    Slice-level long frame -> one row per accession, conditions as columns
    """
    slim = (main_df[["AccessionNumber", "PatientBirth_dt", "condition", "resolved", "stratum"]]
            .drop_duplicates(subset=["AccessionNumber", "condition"]))
 
    wide = slim.pivot(index="AccessionNumber", columns="condition",
                      values=["resolved", "stratum"])
    wide.columns = [f"{cond}_{'label' if kind == 'resolved' else 'stratum'}"
                    for kind, cond in wide.columns]
 
    patients = slim.groupby("AccessionNumber")["PatientBirth_dt"].first()
    tbl = wide.join(patients).reset_index()
 
    # A missing (accession, condition) pair leaves NaN here, which would then be
    # read as "labeled but not positive" -> a phantom negative. Silent and fatal.
    assert not tbl.isna().any().any(), "NaN after pivot: an accession is missing a condition"
 
    tbl["n_labeled"] = sum((tbl[f"{c}_label"] != "not_stated").astype(int)
                           for c in conditions)
    return tbl
 
 
def make_split(
    tbl, seed, test_frac=0.2, conditions=CONDITIONS,
    w_balance=10.0, w_size=3.0, w_stratum=1.0, w_coverage=0.5,
    min_per_class=3, n_candidates=5000
    ):
    """One unified train/test split over accessions.
 
    Patients never straddle the split; accessions that are stratum 'hard' and
    labeled for any condition are held out of test. Among the candidates that
    satisfy those, the lowest-cost one wins: cost charges w_balance for drift of
    the test positive rate from the cohort rate, w_stratum for test cases drawn
    from stratum 'other' rather than 'easy', w_size for missing the target test
    size, and w_coverage for test accessions masked out of some conditions.
    Candidates with fewer than min_per_class positives or negatives in any
    condition's test set are discarded.
    """
    n = len(tbl)
    target = int(round(test_frac * n))
    n_labeled = tbl["n_labeled"].to_numpy()
 
    specs, force = [], np.zeros(n, bool)
    for c in conditions:
        lab = tbl[f"{c}_label"].to_numpy()
        stratum = tbl[f"{c}_stratum"].to_numpy()
        labeled = lab != "not_stated"
        is_pos = lab == "positive"
        specs.append((labeled, is_pos, stratum == "other", is_pos.sum() / labeled.sum()))
        force |= labeled & (stratum == "hard")
 
    pid = tbl["PatientBirth_dt"].astype(str).to_numpy()
    blocks = [np.flatnonzero(pid == p) for p in pd.unique(pid)]
    free = [b for b in blocks if not force[b].any()]
 
    rng = np.random.default_rng(seed)
    order = np.arange(len(free))
    best_cost, best_mask = np.inf, None
 
    for _ in range(n_candidates):
        rng.shuffle(order)
        mask, filled = np.zeros(n, bool), 0
        for i in order:                       # promote whole patient blocks to test
            if filled >= target:
                break
            mask[free[i]] = True
            filled += len(free[i])
 
        cost = w_size * abs(filled - target) / n
        cost += w_coverage * (len(conditions) - n_labeled[mask]).mean() / len(conditions)
        feasible = True
        for labeled, is_pos, is_other, rate in specs:
            in_test = labeled & mask
            k = in_test.sum()
            npos = (is_pos & in_test).sum()
            if npos < min_per_class or k - npos < min_per_class:
                feasible = False
                break
            cost += w_balance * abs(npos / k - rate) + w_stratum * (is_other & in_test).sum() / k
 
        if feasible and cost < best_cost:
            best_cost, best_mask = cost, mask
 
    if best_mask is None:
        raise RuntimeError(f"no feasible split in {n_candidates} draws (seed={seed})")
 
    # should be guaranteed by block construction, but just to be sure
    assert not set(pid[best_mask]) & set(pid[~best_mask]), "patient leakage"
 
    acc = tbl["AccessionNumber"].to_numpy()
    return {"seed": seed, "cost": float(best_cost),
            "train": list(acc[~best_mask]), "test": list(acc[best_mask])}
 
 
def describe_split(tbl, split, conditions=CONDITIONS):
    rows = []
    for cond in conditions:
        lab, stratum = f"{cond}_label", f"{cond}_stratum"
        cohort = (tbl.loc[tbl[lab] != "not_stated", lab] == "positive").mean()
        for part in ("train", "test"):
            sub = tbl[tbl["AccessionNumber"].isin(split[part]) & (tbl[lab] != "not_stated")]
            npos = (sub[lab] == "positive").sum()
            rows.append({"condition": cond, "part": part, "n": len(sub),
                         "pos": npos, "neg": len(sub) - npos,
                         "pos_rate": round(npos / len(sub), 3),
                         "cohort_pos_rate": round(cohort, 3),
                         **sub[stratum].value_counts().reindex(
                             ["easy", "hard", "other"], fill_value=0).to_dict()})
    return pd.DataFrame(rows)
 