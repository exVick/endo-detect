"""Fitting, evaluation and significance testing.

Every function takes a representation as its first argument and reaches it only
through the interface documented in :mod:`representations`, so the same calls
work unchanged for mean pooling, per-series pooling and top-k pooling.

Two classifiers are available, selected with ``classifier="logreg"`` (the
default, and what every earlier result was produced with) or
``classifier="mlp"``, with ``clf_kw`` carrying the chosen one's hyperparameters.
Both standardise their input and weight the classes so that zero is the balanced
threshold; see :mod:`classifiers`. `C` is the logistic probe's inverse L2
strength and is ignored by the perceptron, which takes `weight_decay` in
`clf_kw` instead.

Four levels of analysis are provided:

    run_grid, run_many      performance on the held-out portion of each split
    run_inner               cross-validation inside each split's training
                            portion, intended for model selection
    permutation_test        label permutation null, testing one representation
                            against chance
    paired_test             sign-flip null, comparing two representations that
                            were evaluated on the same splits

Module constants:
    C_HEADLINE: The pre-specified inverse regularisation strength used as the
        default everywhere, so that a single value is reported rather than one
        chosen after seeing the results.
    C_GRID: A logarithmic sweep of regularisation strengths, for inspecting how
        performance depends on it.
    METRICS: The metrics reported everywhere, in the order they appear in the
        result tables. All are oriented so that larger is better.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from .classifiers import make_classifier
from .train_test_split import CONDITIONS

C_HEADLINE = 1
C_GRID = np.logspace(-4, 2, 13)
METRICS = ("auroc", "auprc", "bacc")


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

@dataclass
class Labels:
    """Accession-aligned labels and usability masks.

    All arrays share the row order defined by `acc`.

    Attributes:
        acc (numpy.ndarray): Accession identifiers, defining the row order of
            every other array.
        pid (numpy.ndarray): Patient identifier per accession. Used to keep
            patients whole when splitting, cross-validating and permuting.
        y (dict[str, numpy.ndarray]): Binary label per condition. Entries are
            meaningless wherever the corresponding mask is False.
        mask (dict[str, numpy.ndarray]): True where the condition was stated for
            that accession. Every consumer filters on this before reading `y`.
    """
    acc: np.ndarray
    pid: np.ndarray
    y: dict
    mask: dict

    def pos(self, a):
        """Return the row index of one accession.

        Args:
            a (str): Accession identifier.

        Returns:
            int: Index into `acc`, `pid` and the per-condition arrays.
        """
        return self._pos[a]

    def __post_init__(self):
        self._pos = {a: i for i, a in enumerate(self.acc)}


def make_labels(acc_tbl, conditions=CONDITIONS):
    """Build a :class:`Labels` object from the wide accession table.

    Args:
        acc_tbl (pandas.DataFrame): One row per accession, as returned by
            ``train_test_split.build_accession_table``. Must contain
            "AccessionNumber", "PatientBirth_dt" and one "<condition>_label"
            column per condition.
        conditions (Sequence[str]): Conditions to extract.

    Returns:
        Labels: Labels and masks aligned to the rows of `acc_tbl`. A label of
        "positive" becomes 1 and anything else 0; "not_stated" additionally
        clears the mask, marking the accession unusable for that condition.
    """
    acc = acc_tbl["AccessionNumber"].to_numpy()
    y, mask = {}, {}
    for c in conditions:
        lab = acc_tbl[f"{c}_label"].to_numpy()
        mask[c] = lab != "not_stated"
        y[c] = (lab == "positive").astype(int)
    return Labels(acc=acc,
                  pid=acc_tbl["PatientBirth_dt"].astype(str).to_numpy(),
                  y=y, mask=mask)


def _arrays(labels, rep, split, cond):
    """Select the accessions of one split that are usable for one condition.

    An accession is usable when the representation covers it and the condition
    was stated for it. Both portions are filtered the same way.

    Args:
        labels (Labels): Labels and masks.
        rep (Representation): Representation supplying the coverage.
        split (dict): Split carrying "train" and "test" accession lists.
        cond (str): Condition name.

    Returns:
        tuple: ``(acc_train, y_train, acc_test, y_test)``, each a numpy array,
        with labels aligned to the accessions beside them.
    """
    have = set(rep.accessions)
    out = []
    for part in ("train", "test"):
        a = np.array([x for x in split[part]
                      if x in have and labels.mask[cond][labels.pos(x)]])
        out += [a, np.array([labels.y[cond][labels.pos(x)] for x in a])]
    return out   # acc_train, y_train, acc_test, y_test


# --------------------------------------------------------------------------
# one fit
# --------------------------------------------------------------------------

def _clf(C, classifier="logreg", clf_kw=None):
    """Build the classifier used by every function in this module.

    A thin wrapper over :func:`~.classifiers.make_classifier`, kept so that the
    evaluation code has a single place where a classifier comes into being.

    Args:
        C (float): Inverse regularisation strength. Logistic probe only.
        classifier (str): One of :data:`~.classifiers.CLASSIFIERS`.
        clf_kw (dict, optional): Extra hyperparameters for that classifier.

    Returns:
        LogReg or MLP: An unfitted classifier exposing `fit` and
        `decision_function`.
    """
    return make_classifier(C, classifier, clf_kw)


def _metric_value(y_true, scores, metric):
    """Evaluate one metric from labels and decision function values.

    The single place where a metric name is turned into a number, so that the
    held-out grid, the inner cross-validation and the permutation test can never
    disagree about what a name means.

    Args:
        y_true (numpy.ndarray): Binary labels. Must contain both classes.
        scores (numpy.ndarray): Decision function values, larger meaning more
            positive.
        metric (str): One of :data:`METRICS`. "auroc" and "auprc" use the raw
            scores and therefore depend only on their ranking, while "bacc"
            thresholds them at zero and so also depends on the intercept.

    Returns:
        float: The metric value, oriented so that larger is better.

    Raises:
        ValueError: If `metric` is not one of :data:`METRICS`.
    """
    if metric == "auroc":
        return roc_auc_score(y_true, scores)
    if metric == "auprc":
        return average_precision_score(y_true, scores)
    if metric == "bacc":
        return balanced_accuracy_score(y_true, (scores > 0).astype(int))
    raise ValueError(f"unknown metric {metric!r}; expected one of {list(METRICS)}")


def _fit_eval(Xtr, ytr, Xte, yte, clf):
    """Fit the classifier on one training set and score one test set.

    Args:
        Xtr (numpy.ndarray): Training features of shape (n_train, D).
        ytr (numpy.ndarray): Training labels.
        Xte (numpy.ndarray): Test features of shape (n_test, D).
        yte (numpy.ndarray): Test labels. Must contain both classes.
        clf (LogReg or MLP): Unfitted classifier, fitted here and discarded.

    Returns:
        dict: Metrics and counts for one fit. AUROC and AUPRC are computed on
        the raw decision function, so they depend only on the ranking; balanced
        accuracy thresholds it at zero and therefore also depends on the
        intercept. "prevalence" is the positive rate of the test set, which is
        the chance level for AUPRC.
    """
    p = clf.fit(Xtr, ytr)
    s = p.decision_function(Xte)
    return {**{m: _metric_value(yte, s, m) for m in METRICS},
            "prevalence": float(yte.mean()),
            "n_train": len(ytr), "n_test": len(yte), "n_pos_test": int(yte.sum())}


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def run_grid(rep, labels, splits, Cs=(C_HEADLINE,), conditions=CONDITIONS,
             pbar=None, classifier="logreg", clf_kw=None):
    """Evaluate one representation on the held-out portion of every split.

    For each split and condition the representation is refitted on the training
    accessions only, then used to transform both portions. The representation is
    fitted once per split and reused across all values of `Cs`.

    Conditions whose test set ends up single-class are skipped and produce no
    row, because AUROC is undefined there. The resulting row count can therefore
    be lower than ``len(splits) * len(Cs)``, which is worth checking before
    comparing representations.

    Args:
        rep (Representation): Representation to evaluate.
        labels (Labels): Labels and masks.
        splits (Sequence[dict]): Splits carrying "seed", "train" and "test".
        Cs (Sequence[float]): Inverse regularisation strengths. Pass a single
            value for a headline result, or a grid to sweep. Applies to the
            logistic probe only; the perceptron ignores it.
        conditions (Sequence[str]): Conditions to evaluate.
        pbar (tqdm.tqdm, optional): Progress bar owned by the caller, advanced
            once per (split, condition). Lets :func:`run_many` drive a single
            bar across several representations. No progress is reported when
            None.
        classifier (str): One of :data:`~.classifiers.CLASSIFIERS`.
        clf_kw (dict, optional): Extra hyperparameters for that classifier.

    Returns:
        pandas.DataFrame: One row per (split, condition, C), carrying
        "representation", "seed", "condition", "C" (NaN for the perceptron,
        which does not use it), "classifier", the entries of
        ``rep.meta`` and of the classifier's `meta`, and the metrics returned by
        :func:`_fit_eval`.
    """
    Cs = np.atleast_1d(Cs)
    n_splits = len(splits) if hasattr(splits, "__len__") else "?"
    rows = []
    for sp_idx, sp in enumerate(splits, 1):
        for c in conditions:
            if pbar is not None:
                pbar.set_postfix(rep=rep.name, split=f"{sp_idx}/{n_splits}", c=c)
            atr, ytr, ate, yte = _arrays(labels, rep, sp, c)
            # the guard is inverted rather than left as an early `continue`, so
            # that a skipped single-class test set still advances the bar and the
            # count is able to reach its total
            if len(np.unique(yte)) >= 2:
                # refit the representation per split on TRAINING accessions only
                r = rep.clone().fit(atr, ytr)
                Xtr, Xte = r.transform(atr), r.transform(ate)
                for C in Cs:
                    # a fresh classifier per (split, condition, C): nothing is
                    # carried over between fits
                    clf = _clf(C, classifier, clf_kw)
                    rows.append({"representation": rep.name, "seed": sp["seed"],
                                 "condition": c,
                                 "C": float(C) if classifier == "logreg" else np.nan,
                                 "classifier": clf.name,
                                 **rep.meta, **clf.meta,
                                 **_fit_eval(Xtr, ytr, Xte, yte, clf)})
            if pbar is not None:
                pbar.update(1)
    return pd.DataFrame(rows)


def run_many(reps, labels, splits, Cs=(C_HEADLINE,), conditions=CONDITIONS,
             progress=True, classifier="logreg", clf_kw=None):
    """Run :func:`run_grid` over several representations and concatenate.

    A single progress bar is driven across the whole sweep rather than one per
    representation, so its postfix reports which representation, split and
    condition is currently being fitted.

    Args:
        reps (Sequence[Representation]): Representations to evaluate.
        labels (Labels): Labels and masks.
        splits (Sequence[dict]): Splits carrying "seed", "train" and "test".
        Cs (Sequence[float]): Inverse regularisation strengths. Logistic probe
            only.
        conditions (Sequence[str]): Conditions to evaluate.
        progress (bool): Whether to display the progress bar. Set to False when
            this function is itself called inside a loop.
        classifier (str): One of :data:`~.classifiers.CLASSIFIERS`.
        clf_kw (dict, optional): Extra hyperparameters for that classifier.

    Returns:
        pandas.DataFrame: The concatenation of each representation's grid.
        Representations with different `meta` keys yield NaN in the columns they
        do not define.
    """
    total = (len(reps) * len(splits) * len(conditions)
             if hasattr(splits, "__len__") and hasattr(conditions, "__len__")
             else None)
    with tqdm(total=total, desc="Grid", disable=not progress) as pbar:
        out = [run_grid(r, labels, splits, Cs, conditions, pbar=pbar,
                        classifier=classifier, clf_kw=clf_kw)
               for r in reps]
    return pd.concat(out, ignore_index=True)


def summarise(df, by=("representation", "condition"), base_05=True):
    """Reduce a grid to a median and interquartile range per metric.

    The chance level of each metric is reported alongside it. For AUPRC that is
    the test prevalence, which differs per condition, so it is averaged over the
    grouped rows rather than assumed.

    Args:
        df (pandas.DataFrame): Grid produced by :func:`run_grid` or
            :func:`run_many`.
        by (Sequence[str]): Columns to group by. Add "C" to summarise a
            regularisation sweep separately per value.
        base_05 (bool): If True, also emit the 0.5 chance level for AUROC and
            balanced accuracy. Required by the plotting helpers that draw a
            reference line for those metrics.

    Returns:
        pandas.DataFrame: One row per group, indexed by `by`, with the 25th,
        50th and 75th percentile of each metric across the grouped rows, plus
        the corresponding chance levels. Values are rounded to three decimals.
    """
    by = list(by)
    g = df.groupby(by)
    out = g[["auroc", "auprc", "bacc"]].quantile([0.25, 0.5, 0.75]).unstack()
    out.columns = [f"{m}_{int(q * 100)}" for m, q in out.columns]
    out["auprc_baseline"] = g["prevalence"].mean().round(3)   # AUPRC chance = prevalence
    if base_05:
        out["auroc_baseline"] = 0.5
        out["bacc_baseline"] = 0.5
        cols = ["auroc_50", "auroc_25", "auroc_75", "auroc_baseline",
                "auprc_50", "auprc_25", "auprc_75", "auprc_baseline",
                "bacc_50", "bacc_25", "bacc_75", "bacc_baseline"]
    else:
        cols = ["auroc_50", "auroc_25", "auroc_75",
                "auprc_50", "auprc_25", "auprc_75", "auprc_baseline",
                "bacc_50", "bacc_25", "bacc_75", ]
    return out[cols].round(3)


# --------------------------------------------------------------------------
# permutation test
# --------------------------------------------------------------------------

def _blocks(pid, acc_sub, labels):
    """Group accession positions by patient and bucket them by block size.

    Bucketing by size is what allows :func:`_permute` to exchange whole patients
    without altering the label multiset.

    Args:
        pid (numpy.ndarray): Patient identifier per accession, in `labels` order.
        acc_sub (numpy.ndarray): Accessions taking part in the permutation.
        labels (Labels): Labels, used to map accessions to their row index.

    Returns:
        dict[int, list[numpy.ndarray]]: Maps a block size to the list of blocks
        of that size. Each block holds positions within `acc_sub`.
    """
    groups = defaultdict(list)
    for i, a in enumerate(acc_sub):
        groups[pid[labels.pos(a)]].append(i)
    buckets = defaultdict(list)
    for rows in groups.values():
        buckets[len(rows)].append(np.array(rows))
    return dict(buckets)


def _permute(y, buckets, rng):
    """Exchange label blocks between patients of equal block size.

    Permuting whole patients rather than individual accessions preserves the
    within-patient dependency that the split is built to respect. Restricting
    exchanges to equal-size blocks keeps the label multiset exactly intact.

    Args:
        y (numpy.ndarray): Labels aligned with the accessions the blocks index.
        buckets (dict[int, list[numpy.ndarray]]): Output of :func:`_blocks`.
        rng (numpy.random.Generator): Source of randomness.

    Returns:
        numpy.ndarray: Permuted copy of `y`.
    """
    out = y.copy()
    for blocks in buckets.values():
        order = rng.permutation(len(blocks))
        for src, dst in enumerate(order):
            out[blocks[dst]] = y[blocks[src]]
    return out


def _cache_splits(rep, labels, splits, cond):
    """Precompute the label-independent part of every split's evaluation.

    Which accessions are usable depends on the masks rather than the labels, and
    for a representation whose `fit` ignores the labels the feature matrices do
    not change either. Both can therefore be computed once and reused across
    every draw of a permutation test, which otherwise recomputes them
    identically for each draw.

    Args:
        rep (Representation): Representation with ``needs_labels`` False. The
            observed labels are still passed to `fit`, since a representation
            may fit split-dependent state without reading them.
        labels (Labels): Labels and masks.
        splits (Sequence[dict]): Splits carrying "train" and "test".
        cond (str): Condition name.

    Returns:
        list[tuple]: One ``(acc_train, acc_test, X_train, X_test)`` per split,
        in the order of `splits`. The feature matrices are None where a portion
        is empty, in which case the caller falls back to the uncached path.
    """
    cache = []
    for sp in splits:
        atr, ytr, ate, _ = _arrays(labels, rep, sp, cond)
        if len(atr) == 0 or len(ate) == 0:
            cache.append((atr, ate, None, None))
            continue
        r = rep.clone().fit(atr, ytr)
        cache.append((atr, ate, r.transform(atr), r.transform(ate)))
    return cache


def _median_metric(rep, labels, splits, cond, C, y_override=None, cache=None,
                   metric="auroc", classifier="logreg", clf_kw=None):
    """Compute the statistic the permutation test operates on.

    The statistic is the median of `metric` over the splits' held-out portions,
    so it matches the corresponding column of :func:`run_grid`.

    Args:
        rep (Representation or None): Representation to evaluate. May be None
            when `cache` supplies feature matrices for every split, since it is
            then never dereferenced.
        labels (Labels): Labels and masks.
        splits (Sequence[dict]): Splits carrying "train" and "test".
        cond (str): Condition name.
        C (float): Inverse regularisation strength. Logistic probe only.
        y_override (dict, optional): Maps accession to a replacement label. Used
            to evaluate a permuted labelling without rebuilding `labels`. The
            masks are untouched, so the same accessions stay usable.
        cache (list[tuple], optional): Output of :func:`_cache_splits`, aligned
            with `splits`. When given, the accession selection and the feature
            matrices are taken from it instead of being recomputed, and only the
            classifier is refitted. Valid only for a representation whose `fit`
            ignores the labels.
        metric (str): Metric to reduce, one of :data:`METRICS`.
        classifier (str): One of :data:`~.classifiers.CLASSIFIERS`.
        clf_kw (dict, optional): Extra hyperparameters for that classifier.

    Returns:
        tuple: ``(median_value, n_splits_used)``. The median is NaN when no
        split could be scored, which happens when a permutation leaves every
        test set single-class.

    Raises:
        ValueError: If `metric` is not one of :data:`METRICS`.
    """
    vals = []
    for i, sp in enumerate(splits):
        if cache is None:
            atr, ytr, ate, yte = _arrays(labels, rep, sp, cond)
            Xtr = Xte = None
        else:
            atr, ate, Xtr, Xte = cache[i]
            ytr = np.array([labels.y[cond][labels.pos(a)] for a in atr])
            yte = np.array([labels.y[cond][labels.pos(a)] for a in ate])
        if y_override is not None:
            ytr = np.array([y_override[a] for a in atr])
            yte = np.array([y_override[a] for a in ate])
        if len(np.unique(yte)) < 2:      # permutation left test single-class
            continue
        if Xtr is None:                  # uncached, or a portion was empty
            r = rep.clone().fit(atr, ytr)
            Xtr, Xte = r.transform(atr), r.transform(ate)
        p = _clf(C, classifier, clf_kw).fit(Xtr, ytr)
        vals.append(_metric_value(yte, p.decision_function(Xte), metric))
    return (float(np.median(vals)) if vals else np.nan), len(vals)


def _one_perm(rep, labels, splits, cond, C, acc_sub, y_sub, buckets, seed,
              cache=None, metric="auroc", classifier="logreg", clf_kw=None):
    """Draw one permutation and evaluate the statistic under it.

    Args:
        rep (Representation or None): Representation to evaluate. The caller
            passes None when `cache` is complete, which keeps the embedding
            matrix and any ranker lookup tables out of the payload pickled to
            each worker.
        labels (Labels): Labels and masks.
        splits (Sequence[dict]): Splits carrying "train" and "test".
        cond (str): Condition name.
        C (float): Inverse regularisation strength. Logistic probe only.
        acc_sub (numpy.ndarray): Accessions taking part in the permutation.
        y_sub (numpy.ndarray): Their observed labels.
        buckets (dict): Output of :func:`_blocks`.
        seed: Seed for this draw, so that each permutation is reproducible.
        cache (list[tuple], optional): Output of :func:`_cache_splits`, passed
            through to :func:`_median_metric`.
        metric (str): Metric to reduce, one of :data:`METRICS`.
        classifier (str): One of :data:`~.classifiers.CLASSIFIERS`.
        clf_kw (dict, optional): Extra hyperparameters for that classifier. A
            classifier with its own seed keeps it fixed across draws, so the
            null varies with the labels alone.

    Returns:
        float: Median value of `metric` under the permuted labelling, or NaN.
    """
    yp = _permute(y_sub, buckets, np.random.default_rng(seed))
    return _median_metric(rep, labels, splits, cond, C,
                          y_override=dict(zip(acc_sub, yp)), cache=cache,
                          metric=metric, classifier=classifier,
                          clf_kw=clf_kw)[0]


def permutation_test(rep, labels, splits, C=C_HEADLINE, n_perm=1000,
                     seed=0, n_jobs=-1, conditions=CONDITIONS, use_cache=True,
                     metric="auroc", classifier="logreg", clf_kw=None):
    """Test one representation against a group-level label permutation null.

    The null hypothesis is that the labels are unrelated to the images. It is
    realised by exchanging labels between patients of equal block size and
    re-running the whole fit-and-score procedure, so the p-value accounts for
    the split structure rather than assuming independent observations.

    Running every condition is informative even when only one is of interest:
    conditions with no expected signal act as negative controls, and their null
    medians should sit at 0.5. A different value indicates a leak or a
    structural problem affecting the condition of interest as well.

    Args:
        rep (Representation): Representation to test.
        labels (Labels): Labels and masks.
        splits (Sequence[dict]): Splits carrying "seed", "train" and "test".
        C (float): Inverse regularisation strength. Logistic probe only.
        n_perm (int): Number of permutations. The smallest reportable p-value is
            ``1 / (n_perm + 1)``, so this sets the resolution of the test.
        seed (int): Base seed. Combined with the condition name to give each
            condition an independent, reproducible stream.
        n_jobs (int): Number of parallel workers, passed to joblib.
        conditions (Sequence[str]): Conditions to test.
        use_cache (bool): Whether to precompute the label-independent part of
            each split once and reuse it across permutations. Applied only when
            ``rep.needs_labels`` is False, since a representation that learns
            from the labels must be refitted for every draw. Set to False to
            force the uncached path, which is useful for verifying that the two
            agree.

            The cache trades a larger payload per worker task against a
            recomputed transform. It is a large win when the transform is
            expensive and a small loss when it is not: at 1000 permutations over
            25 splits it takes a top-k pool from roughly seventeen minutes to
            under one, while costing mean pooling, whose vectors are already
            precomputed, about fifteen seconds. Leaving it enabled is therefore
            the better default, but it can be turned off for representations
            whose transform is a lookup.
        metric (str): Statistic to test, one of :data:`METRICS`. All are
            oriented so that larger is better, so the p-value counts null draws
            at or above the observed value regardless of the choice. The null is
            empirical, so no chance level has to be assumed: an AUROC or
            balanced-accuracy null centres near 0.5 while an AUPRC null centres
            near the prevalence, and both are handled without special casing.
        classifier (str): One of :data:`~.classifiers.CLASSIFIERS`. Note that
            the classifier is refitted on every draw whatever `use_cache` says,
            since the cache holds features and only the labels change. A
            classifier that trains by gradient descent therefore multiplies the
            cost of the test by ``n_perm * len(splits)``; start with a smaller
            `n_perm` and leave its `device` on the default "cpu", since the
            draws are dispatched across processes.
        clf_kw (dict, optional): Extra hyperparameters for that classifier.

    Returns:
        tuple:
            - pandas.DataFrame: One row per condition, with the metric tested,
              the observed statistic, summary statistics of the null, and the
              p-value. "shift" is the observed value minus the null median.
            - dict: Maps ``(rep.name, condition)`` to the array of null values,
              suitable for plotting the null distribution.

    Raises:
        ValueError: If `metric` is not one of :data:`METRICS`.

    Notes:
        The p-value adds one to both the numerator and the denominator, after
        Phipson and Smyth (2010): the observed labelling is itself one of the
        possible permutations and is given no special status. This also avoids
        reporting a p-value of exactly zero.

        The returned keys of `nulls` do not include the metric, so results for
        two metrics should be kept in separate tables rather than concatenated.
    """
    # validated here rather than inside a worker, so that a typo fails at once
    # instead of after n_perm tasks have been dispatched
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {list(METRICS)}")
    # likewise built once here so that a bad name or hyperparameter raises
    # before any work is dispatched, rather than inside every worker
    clf_name = _clf(C, classifier, clf_kw).name
    have = set(rep.accessions)
    rows, nulls = [], {}
    for c in conditions:
        acc_sub = np.array([a for a in labels.acc
                            if a in have and labels.mask[c][labels.pos(a)]])
        y_sub = np.array([labels.y[c][labels.pos(a)] for a in acc_sub])
        # the accession selection and, for a representation that ignores the
        # labels, the feature matrices are identical for every draw, so they are
        # computed once here rather than n_perm times inside the workers
        cache = (_cache_splits(rep, labels, splits, c)
                 if use_cache and not rep.needs_labels else None)
        obs, n_used = _median_metric(rep, labels, splits, c, C, cache=cache,
                                     metric=metric, classifier=classifier,
                                     clf_kw=clf_kw)
        buckets = _blocks(labels.pid, acc_sub, labels)

        # when the cache covers every split the representation is never
        # dereferenced inside a worker, so it is left out of the task payload.
        # otherwise its frame grouping, and any lookup table a ranker holds,
        # would be pickled once per permutation.
        complete = cache is not None and all(x is not None for _, _, x, _ in cache)
        rep_arg = None if complete else rep

        # zlib.crc32, not hash(): Python randomises string hashing per process
        # unless PYTHONHASHSEED is set, so hash() would give different draws on
        # every run even with the same `seed`.
        seeds = np.random.SeedSequence([seed, zlib.crc32(c.encode())]).spawn(n_perm)
        null = np.array(Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_one_perm)(rep_arg, labels, splits, c, C, acc_sub, y_sub,
                               buckets, s, cache, metric, classifier, clf_kw)
            for s in seeds))
        null = null[~np.isnan(null)]

        # +1 in both terms: the observed labelling is itself one of the
        # permutations and has no special status under the null.
        p = (1 + int((null >= obs).sum())) / (len(null) + 1)
        nulls[(rep.name, c)] = null
        rows.append({"representation": rep.name, "condition": c,
                     "metric": metric, "classifier": clf_name,
                     "observed": round(obs, 3), "splits_used": n_used,
                     "n_perm": len(null),
                     "null_median": round(float(np.median(null)), 3),
                     "null_sd": round(float(null.std(ddof=1)), 3),
                     "null_q95": round(float(np.quantile(null, 0.95)), 3),
                     "shift": round(obs - float(np.median(null)), 3),
                     "p": p})
    return pd.DataFrame(rows), nulls


# --------------------------------------------------------------------------
# held-out scores for inspection
# --------------------------------------------------------------------------

def held_out_scores(rep, labels, splits, cond, C=C_HEADLINE,
                    classifier="logreg", clf_kw=None):
    """Score every accession using only the splits where it was held out.

    Decision function values are not comparable across splits, since each split
    has its own scaler and intercept. Each split's test scores are therefore
    converted to percentile ranks before averaging, which is consistent with
    AUROC being a rank measure.

    Args:
        rep (Representation): Representation to evaluate.
        labels (Labels): Labels and masks.
        splits (Sequence[dict]): Splits carrying "train" and "test".
        cond (str): Condition name.
        C (float): Inverse regularisation strength. Logistic probe only.
        classifier (str): One of :data:`~.classifiers.CLASSIFIERS`.
        clf_kw (dict, optional): Extra hyperparameters for that classifier.

    Returns:
        pandas.DataFrame: One row per accession that appeared in at least one
        test set, with columns "AccessionNumber", "label", "score" and
        "n_held_out", sorted by descending score. "score" lies in (0, 1] and is
        the mean percentile rank across the splits counted in "n_held_out".
    """
    pct = defaultdict(list)
    for sp in splits:
        atr, ytr, ate, yte = _arrays(labels, rep, sp, cond)
        if len(np.unique(ytr)) < 2 or len(ate) == 0:
            continue
        r = rep.clone().fit(atr, ytr)
        p = _clf(C, classifier, clf_kw).fit(r.transform(atr), ytr)
        ranks = rankdata(p.decision_function(r.transform(ate))) / len(ate)
        for a, v in zip(ate, ranks):
            pct[a].append(v)

    return (pd.DataFrame([{"AccessionNumber": a,
                           "label": int(labels.y[cond][labels.pos(a)]),
                           "score": float(np.mean(v)), "n_held_out": len(v)}
                          for a, v in pct.items()])
            .sort_values("score", ascending=False)
            .reset_index(drop=True))


def four_groups(scores, k=5, min_held_out=3):
    """Extract the most informative accessions from held-out scores.

    Four groups are returned: confidently correct positives and negatives, and
    the two kinds of error. The error groups are the ones worth inspecting; the
    concordant groups mainly confirm that the pipeline behaves as expected.

    Args:
        scores (pandas.DataFrame): Output of :func:`held_out_scores`.
        k (int): Number of accessions per group.
        min_held_out (int): Minimum number of splits an accession must have been
            held out in for its mean rank to be stable enough to include.

    Returns:
        pandas.DataFrame: Up to ``4 * k`` rows with a "group" column taking the
        values "hit_confident", "reject_confident", "false_positive" and "miss".
    """
    s = scores[scores.n_held_out >= min_held_out]
    p, n = s[s.label == 1], s[s.label == 0]
    return pd.concat([
        p.nlargest(k, "score").assign(group="hit_confident"),
        n.nsmallest(k, "score").assign(group="reject_confident"),
        n.nlargest(k, "score").assign(group="false_positive"),
        p.nsmallest(k, "score").assign(group="miss"),
    ], ignore_index=True)[
        ["group", "AccessionNumber", "label", "score", "n_held_out"]]


# --------------------------------------------------------------------------
# inner cross-validation on the training portion
# --------------------------------------------------------------------------

def _train_acc(labels, rep, split, cond):
    """Select the usable training accessions of one split for one condition.

    Only ``split["train"]`` is read. No function in the inner cross-validation
    path ever references ``split["test"]``, which is what keeps the held-out
    portion untouched during model selection.

    Args:
        labels (Labels): Labels and masks.
        rep (Representation): Representation supplying the coverage.
        split (dict): Split carrying a "train" accession list.
        cond (str): Condition name.

    Returns:
        tuple: ``(accessions, labels, patient_ids)``, each a numpy array of the
        same length.
    """
    have = set(rep.accessions)
    a = np.array([x for x in split["train"]
                  if x in have and labels.mask[cond][labels.pos(x)]])
    y = np.array([labels.y[cond][labels.pos(x)] for x in a])
    g = np.array([labels.pid[labels.pos(x)] for x in a])
    return a, y, g


def _derive_seed(*parts):
    """Derive a reproducible seed from a mixture of strings and integers.

    ``zlib.crc32`` is used rather than ``hash`` because Python randomises string
    hashing per process unless PYTHONHASHSEED is set, which would make results
    differ between runs.

    Args:
        *parts: Strings and integers identifying the context, for example the
            condition, the representation name and the repeat index.

    Returns:
        int: A 31-bit seed, identical for identical arguments.
    """
    ints = [zlib.crc32(p.encode()) if isinstance(p, str) else int(p) for p in parts]
    return int(np.random.SeedSequence(ints).generate_state(1)[0] % (2 ** 31))


def inner_cv_scores(rep, labels, split, cond, C=C_HEADLINE, k=5,
                    n_repeats=10, seed=0, classifier="logreg", clf_kw=None):
    """Run repeated stratified group k-fold on one split's training portion.

    Each repeat produces a single pooled out-of-fold AUROC: every training
    accession is scored once by a model that did not see it, all scores are
    concatenated, and AUROC is computed once over the whole training portion.
    Averaging per-fold AUROCs instead would be considerably noisier, since a
    fold with few negatives yields an AUROC over very few pairs.

    Repeats differ only in the fold assignment, which changes which model scores
    which accession, so the pooled AUROC varies between them. That spread
    measures fold-assignment variability rather than anything about the data.

    Patients are kept whole across folds, mirroring the constraint the outer
    split applies.

    Args:
        rep (Representation): Representation to evaluate.
        labels (Labels): Labels and masks.
        split (dict): One outer split. Only its "train" portion is read.
        cond (str): Condition name.
        C (float): Inverse regularisation strength. Logistic probe only.
        k (int): Number of inner folds.
        n_repeats (int): Number of times the fold assignment is redrawn.
        seed (int): Base seed. Combined with the condition, the representation
            name, the split seed and the repeat index, so the same arguments
            reproduce the same folds. It seeds the fold assignment only; a
            classifier that needs a seed of its own carries it in `clf_kw` and
            holds it fixed across folds.
        classifier (str): One of :data:`~.classifiers.CLASSIFIERS`.
        clf_kw (dict, optional): Extra hyperparameters for that classifier.

    Returns:
        list[dict]: One entry per repeat with keys "repeat", "auroc", "auprc",
        "bacc", "prevalence", "n_oof" and "n_pos". The three metrics are NaN
        when the folds could not be built or when the pooled out-of-fold labels
        are single-class. "prevalence" is the positive rate of the pooled
        out-of-fold labels, which is the chance level for AUPRC.

    Notes:
        Scores are pooled across folds fitted with slightly different scalers
        and intercepts. With five folds the training sets overlap substantially,
        so the scales are close.

        That pooling affects the metrics unequally. AUROC and AUPRC depend only
        on the ranking, so a small shift between folds barely matters. Balanced
        accuracy thresholds the decision function at zero, an absolute cut, so
        it also absorbs any difference between the folds' intercepts and is
        correspondingly less stable here than on a single fitted model.
    """
    acc, y, groups = _train_acc(labels, rep, split, cond)
    out = []
    for r in range(n_repeats):
        rs = _derive_seed(cond, rep.name, int(split["seed"]), r, seed)
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs)
        s_all, y_all = [], []
        try:
            folds = list(cv.split(acc, y, groups))
        except ValueError:
            out.append({"repeat": r, "auroc": np.nan, "auprc": np.nan,
                        "bacc": np.nan, "prevalence": np.nan,
                        "n_oof": 0, "n_pos": 0})
            continue
        for itr, iva in folds:
            if len(np.unique(y[itr])) < 2:      # cannot fit on one class
                continue
            rr = rep.clone().fit(acc[itr], y[itr])   # refit inside the fold
            p = _clf(C, classifier, clf_kw).fit(rr.transform(acc[itr]), y[itr])
            s_all.append(p.decision_function(rr.transform(acc[iva])))
            y_all.append(y[iva])
        if not s_all:
            out.append({"repeat": r, "auroc": np.nan, "auprc": np.nan,
                        "bacc": np.nan, "prevalence": np.nan,
                        "n_oof": 0, "n_pos": 0})
            continue
        s_all = np.concatenate(s_all)
        y_all = np.concatenate(y_all)
        # the same metrics run_grid reports, computed once over the pooled
        # out-of-fold scores rather than averaged over the folds
        if len(np.unique(y_all)) == 2:
            m = {k: _metric_value(y_all, s_all, k) for k in METRICS}
        else:
            m = {k: np.nan for k in METRICS}
        out.append({"repeat": r, **m, "prevalence": float(y_all.mean()),
                    "n_oof": len(y_all), "n_pos": int(y_all.sum())})
    return out


def run_inner(reps, labels, splits, conditions=CONDITIONS, C=C_HEADLINE,
              k=5, n_repeats=10, seed=0, classifier="logreg", clf_kw=None,
              n_jobs=1):
    """Run :func:`inner_cv_scores` across representations, splits and conditions.

    No test accession is read at any point, so this is the appropriate place to
    compare representations and select hyperparameters.

    Args:
        reps (Representation or Sequence[Representation]): Representations to
            evaluate. A single representation is accepted and wrapped in a list.
        labels (Labels): Labels and masks.
        splits (Sequence[dict]): Outer splits. Only their "train" portions are
            read.
        conditions (Sequence[str]): Conditions to evaluate.
        C (float): Inverse regularisation strength. Logistic probe only.
        k (int): Number of inner folds.
        n_repeats (int): Number of times the fold assignment is redrawn.
        seed (int): Base seed for the fold assignments.
        classifier (str): One of :data:`~.classifiers.CLASSIFIERS`.
        clf_kw (dict, optional): Extra hyperparameters for that classifier. To
            tune one, call this once per setting and concatenate, then group on
            the corresponding "clf_" column::

                inner = pd.concat([run_inner(reps, labels, splits,
                                             classifier="mlp",
                                             clf_kw={"steps": s})
                                   for s in (100, 300, 1000)])
                summarise_inner(inner,
                                by=("representation", "condition", "clf_steps"))
        n_jobs (int): Number of parallel workers, passed to joblib. Each task is
            one (representation, split, condition), which is the coarsest unit
            that stays independent, so the grain is `n_repeats * k` fits. One by
            default, which keeps the serial path exactly as it was.

            Parallelising here rather than inside a fit is deliberate: the
            matrices are small enough that a single fit does not thread well,
            while the tasks are entirely independent. Every seed is derived from
            the task itself, so the results do not depend on how the work was
            distributed and a parallel run reproduces a serial one exactly.

    Returns:
        pandas.DataFrame: One row per (representation, condition, outer split,
        repeat), carrying "representation", "condition", "split_seed",
        "classifier", the entries of ``rep.meta`` and of the classifier's
        `meta`, and the fields returned by :func:`inner_cv_scores`.
    """
    reps = reps if isinstance(reps, (list, tuple)) else [reps]
    # built once purely to read its name and hyperparameters into every row;
    # the fits below each construct their own
    template = _clf(C, classifier, clf_kw)

    splits = list(splits)
    conditions = list(conditions)
    n_splits = len(splits)
    # the task list is materialised so that the serial and the parallel path
    # walk it in the same order, which is what makes the row order independent
    # of n_jobs
    tasks = [(rep, sp_idx, sp, c)
             for rep in reps
             for sp_idx, sp in enumerate(splits, 1)
             for c in conditions]

    with tqdm(total=len(tasks), desc="Inner CV") as pbar:
        if n_jobs == 1:
            out = []
            for rep, sp_idx, sp, c in tasks:
                pbar.set_postfix(rep=rep.name,
                                 split=f"{sp_idx}/{n_splits}", c=c)
                out.append(inner_cv_scores(
                    rep, labels, sp, c, C=C, k=k, n_repeats=n_repeats,
                    seed=seed, classifier=classifier, clf_kw=clf_kw))
                pbar.update(1)
        else:
            # return_as="generator" yields in submission order, so the bar can
            # advance as tasks land without the results being reordered
            gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(inner_cv_scores)(
                    rep, labels, sp, c, C=C, k=k, n_repeats=n_repeats,
                    seed=seed, classifier=classifier, clf_kw=clf_kw)
                for rep, _, sp, c in tasks)
            out = []
            for scores in gen:
                out.append(scores)
                pbar.update(1)

    rows = []
    for (rep, _, sp, c), scores in zip(tasks, out):
        for d in scores:
            rows.append({
                "representation": rep.name,
                "condition": c,
                "split_seed": sp["seed"],
                "classifier": template.name,
                **rep.meta,
                **template.meta,
                **d,
            })

    return pd.DataFrame(rows)


def summarise_inner(inner, by=("representation", "condition"),
                    metrics=("auroc", "auprc", "bacc")):
    """Reduce inner cross-validation results to one row per group.

    Repeats are collapsed by median within each outer split before the spread
    across splits is taken. Repeats differ only by fold assignment, so collapsing
    them first leaves one number per outer split, whose spread reflects which
    patients were trained on. Taking a single interquartile range over all
    repeats and splits together would mix the two sources of variation.

    The chance level of each metric is reported beside it, following the same
    convention as :func:`summarise`: 0.5 for AUROC and balanced accuracy, and
    the mean out-of-fold prevalence for AUPRC.

    Args:
        inner (pandas.DataFrame): Output of :func:`run_inner`.
        by (Sequence[str]): Columns to group by. Add "C" to summarise a
            regularisation sweep separately per value.
        metrics (Sequence[str]): Metrics to summarise. Each must be a column of
            `inner`. Pass a single metric to keep the table narrow.

    Returns:
        pandas.DataFrame: One row per group, indexed by `by`. For every metric
        it carries the median and interquartile range of the per-split medians,
        the chance level, and "<metric>_repeat_iqr", which is the median spread
        across repeats within a split and indicates how much the fold assignment
        alone moved the estimate. "n_splits" counts the outer splits
        contributing to the first metric.

    Raises:
        ValueError: If a requested metric is not a column of `inner`, which
            happens when the table was produced before that metric was recorded.
    """
    by = list(by)
    metrics = list(metrics)
    missing = [m for m in metrics if m not in inner.columns]
    if missing:
        raise ValueError(f"{missing} not in inner columns; available metrics: "
                         f"{[c for c in ('auroc', 'auprc', 'bacc') if c in inner.columns]}")

    parts, cols = [], []
    for i, m in enumerate(metrics):
        d = inner.dropna(subset=[m])
        per_split = (d.groupby(by + ["split_seed"])[m].median()
                     .rename("split_median").reset_index())
        g = per_split.groupby(by)["split_median"]
        q = g.quantile([0.25, 0.5, 0.75]).unstack()
        q.columns = [f"{m}_25", f"{m}_50", f"{m}_75"]
        # stability check: how much did fold assignment alone move the estimate?
        q[f"{m}_repeat_iqr"] = (d.groupby(by + ["split_seed"])[m]
                                .agg(lambda v: v.quantile(0.75) - v.quantile(0.25))
                                .groupby(by).median())
        # AUPRC chance is the prevalence, which differs per condition, so it is
        # read from the data rather than assumed
        q[f"{m}_baseline"] = (d.groupby(by)["prevalence"].mean()
                              if m == "auprc" else 0.5)
        if i == 0:
            q["n_splits"] = g.size()
        parts.append(q)
        cols += [f"{m}_50", f"{m}_25", f"{m}_75", f"{m}_baseline",
                 f"{m}_repeat_iqr"]

    out = pd.concat(parts, axis=1)
    return out[cols + ["n_splits"]].round(3)


# --------------------------------------------------------------------------
# paired comparison of two representations
# --------------------------------------------------------------------------

def paired_test(grid, rep_a, rep_b, condition, C=C_HEADLINE, metric="auroc",
                n_perm=10000, seed=0, alternative="two-sided", classifier=None):
    """Compare two representations with a sign-flip permutation test.

    Both representations were evaluated on the same splits, so their scores are
    correlated: a split that is easy is easy for both. Taking per-split
    differences cancels that shared variation, which makes the paired test
    considerably more sensitive than two independent tests.

    The null hypothesis is that the two representations perform equally, not
    that either is at chance. Under it a positive and a negative difference are
    equally likely, so flipping the sign of each observed difference at random
    generates the null distribution of the mean difference. No refitting is
    required, since the metric values already exist in `grid`.

    Two separate tests against chance cannot establish that A is better than B:
    one being significant while the other is not is not evidence of a difference
    between them. This test addresses that question directly.

    Args:
        grid (pandas.DataFrame): Output of :func:`run_grid` or
            :func:`run_many`, containing both representations.
        rep_a (str): Name of the first representation. Differences are computed
            as A minus B.
        rep_b (str): Name of the second representation.
        condition (str): Condition to compare on.
        C (float): Inverse regularisation strength to filter the grid to.
        metric (str): Column of `grid` to compare.
        n_perm (int): Number of sign-flip draws.
        seed (int): Base seed, combined with the condition and both names.
        alternative (str): "two-sided" to test for any difference, or "greater"
            to test whether A exceeds B. The direction must be chosen before
            inspecting the data.
        classifier (str, optional): Restrict the grid to one classifier before
            comparing. Required when `grid` holds more than one, since the test
            pairs rows by split and two classifiers contribute two rows per
            split. Grids produced before the classifier was selectable carry no
            such column and are unaffected.

    Returns:
        dict: Summary of the comparison, including "n_splits" (the number of
        splits both representations were scored on), "mean_diff", "median_diff",
        "n_a_wins", "null_sd" and "p". The keys "_diff" and "_null" hold the raw
        per-split differences and the null distribution, for plotting.

    Raises:
        ValueError: If fewer than two splits are shared by the two
            representations, if either side contributes more than one row per
            split, or if `alternative` is not recognised.

    Notes:
        With `n` shared splits there are only ``2 ** n`` sign patterns, which
        places a floor on the achievable p-value. At five splits the smallest
        two-sided p-value is about 0.063, so significance cannot be reached
        regardless of the effect size.
    """
    d = grid[(grid.condition == condition) & (np.isclose(grid.C, C))]
    if "classifier" in d.columns:
        if classifier is not None:
            d = d[d.classifier == classifier]
        elif d["classifier"].nunique() > 1:
            raise ValueError(
                f"grid holds several classifiers "
                f"({sorted(d['classifier'].unique())}); pass classifier= to "
                f"choose one, otherwise each split would contribute two rows")
    a = d[d.representation == rep_a].set_index("seed")[metric]
    b = d[d.representation == rep_b].set_index("seed")[metric]
    # a repeated seed means the grid still mixes settings the filters above do
    # not separate, for example two clf_kw variants concatenated. pairing them
    # would compare arbitrary rows, so it is refused rather than guessed at.
    for nm, v in ((rep_a, a), (rep_b, b)):
        if v.index.has_duplicates:
            raise ValueError(
                f"{nm} contributes several rows per split at C={C}; filter the "
                f"grid down to one setting per split before pairing")
    common = a.index.intersection(b.index)
    if len(common) < 2:
        raise ValueError(f"only {len(common)} shared splits for {rep_a} vs {rep_b}")
    diff = (a.loc[common] - b.loc[common]).to_numpy()

    rng = np.random.default_rng(_derive_seed(condition, rep_a, rep_b, seed))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = (signs * diff).mean(axis=1)
    obs = float(diff.mean())

    if alternative == "two-sided":
        hit = int((np.abs(null) >= abs(obs)).sum())
    elif alternative == "greater":
        hit = int((null >= obs).sum())
    else:
        raise ValueError("alternative must be 'two-sided' or 'greater'")
    p = (1 + hit) / (n_perm + 1)     # Phipson & Smyth (2010)

    return {"condition": condition, "rep_a": rep_a, "rep_b": rep_b,
            "n_splits": len(diff), "mean_diff": round(obs, 4),
            "median_diff": round(float(np.median(diff)), 4),
            "n_a_wins": int((diff > 0).sum()), "n_ties": int((diff == 0).sum()),
            "null_sd": round(float(null.std(ddof=1)), 4),
            "alternative": alternative, "p": p,
            "_diff": diff, "_null": null}


def paired_vs_baseline(grid, baseline, condition, C=C_HEADLINE, **kw):
    """Compare every representation in a grid against one baseline.

    Args:
        grid (pandas.DataFrame): Output of :func:`run_grid` or
            :func:`run_many`.
        baseline (str): Name of the representation every other one is compared
            against.
        condition (str): Condition to compare on.
        C (float): Inverse regularisation strength to filter the grid to.
        **kw: Forwarded to :func:`paired_test`, for example `metric`, `n_perm`,
            `alternative` or `classifier`.

    Returns:
        pandas.DataFrame: One row per comparison, sorted by descending mean
        difference, with the raw arrays dropped. "n_comparisons" and
        "bonferroni_alpha" record how many tests were run, since performing many
        comparisons inflates the chance of a significant result.
    """
    names = [n for n in grid.representation.unique() if n != baseline]
    rows = [paired_test(grid, n, baseline, condition, C=C, **kw) for n in names]
    tbl = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in rows])
    tbl["n_comparisons"] = len(rows)
    tbl["bonferroni_alpha"] = round(0.05 / max(len(rows), 1), 4)
    return tbl.sort_values("mean_diff", ascending=False).reset_index(drop=True)
