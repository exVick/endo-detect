"""Plots. All take the DataFrames produced by evaluate.py."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .evaluate import C_HEADLINE
from .train_test_split import CONDITIONS

CONDITION_MAP = {"die": "Deep Infiltrating Endometriosis", "adenomyosis":"Adenomyosis", "endometrioma":"Ovarian Endometrioma"}

P_THRESHOLDS = (0.001, 0.01, 0.05)
P_LEGEND = "* p < 0.05    ** p < 0.01    *** p < 0.001"


def _stars(p, thresholds=P_THRESHOLDS):
    """Asterisk code for one p-value, or an empty string when it is not significant.

    The usual convention is followed: three below  0.001, two below 0.01, 
    one below 0.05, and nothing at or above 0.05. None and NaN are treated 
    as "not tested" and also give an empty string, so a
    partial p-value table annotates only the bars it covers.
    """
    if p is None or not np.isfinite(p):
        return ""
    for i, t in enumerate(thresholds):
        if p < t:
            return "*" * (len(thresholds) - i)
    return ""


def _sig_label(p, thresholds=P_THRESHOLDS, p_fmt="p = {:.4f}"):
    """Asterisk code for one p-value with the value itself appended after it.

    The number is only shown where there are asterisks to show it next to, so a
    non-significant bar stays completely unannotated rather than being labelled
    with a large p. Passing p_fmt=None leaves the asterisks on their own.
    """
    st = _stars(p, thresholds)
    if not st or p_fmt is None:
        return st
    # return f"{st} {p_fmt.format(p)}"
    return f"{p_fmt.format(p)}"


def plot_grid(grid, metric="auroc", chance=0.5, conditions=CONDITIONS,
              C_headline=C_HEADLINE, reps=None, plotting_conditions=CONDITION_MAP):
    """Median + IQR across splits vs C, one panel per condition."""

    df = grid if reps is None else grid[grid.representation.isin(reps)]
    names = list(df.representation.unique())
    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(4.6 * len(conditions), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conditions):
        for i, nm in enumerate(names):
            d = df[(df.condition == c) & (df.representation == nm)]
            if d.empty:
                continue
            q = d.groupby("C")[metric].quantile([.25, .5, .75]).unstack()
            col = f"C{i}"
            ax.fill_between(q.index, q[0.25], q[0.75], alpha=0.20, color=col)
            ax.plot(q.index, q[0.5], marker="o", ms=3, color=col, label=nm)
        ax.axhline(chance, ls="--", lw=1, color="grey")
        ax.axvline(C_headline, ls=":", lw=1, color="crimson")
        ax.set_xscale("log")
        ax.set_xlabel("C")
        mapped_c = plotting_conditions.get(c, c)
        ax.set_title(mapped_c)
    axes[0].set_ylabel(f"test {metric}")

    handles = [Line2D([], [], color=f"C{i}", marker="o", ms=3, label=nm)
               for i, nm in enumerate(names)]
    handles += [
        Patch(facecolor="grey", alpha=0.20, label="IQR across splits"),
        Line2D([], [], color="grey", ls="--", lw=1, label=f"chance ({chance:.2f})"),
        Line2D([], [], color="crimson", ls=":", lw=1, label=f"pre-specified C = {C_headline}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    return fig


def plot_null(nulls, perm_tbl, conditions=CONDITIONS, plotting_conditions=CONDITION_MAP):
    """Permutation null per condition with the observed value marked."""

    keys = [k for k in nulls if k[1] in conditions]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.6))
    axes = np.atleast_1d(axes)
    idx = perm_tbl.set_index(["representation", "condition"])
    rep_name = perm_tbl.representation.unique()[0]

    for ax, k in zip(axes, keys):
        r = idx.loc[k]
        ax.hist(nulls[k], bins=40, color="tab:blue", alpha=0.55)
        ax.axvline(r["null_q95"], color="grey", ls="--", lw=1)
        ax.axvline(r["observed"], color="crimson", lw=1.5)
        mapped_c = plotting_conditions.get(k[1], k[1])
        # ax.set_title(f"{mapped_c}  (p = {r['p']:.3f})\n(represented by {rep_name})")
        ax.set_title(f"{mapped_c}  (p = {r['p']:.3f})", pad=18)
        ax.text(0.5, 1.02, f"(represented by {rep_name})", 
                transform=ax.transAxes, ha="center", va="bottom", 
                fontsize=9, color="dimgrey")
        #TODO: implement the metric
        ax.set_xlabel("median [!!metric!!] across splits")
    axes[0].set_ylabel("permutations")
    fig.legend(handles=[
        Patch(facecolor="tab:blue", alpha=0.55, label="null distribution"),
        Line2D([], [], color="grey", ls="--", lw=1, label="null 95th percentile"),
        Line2D([], [], color="crimson", lw=1.5, label="observed"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def plot_representations(summary, metric="auroc", conditions=CONDITIONS,
                         sort_by=None, plotting_conditions=CONDITION_MAP,
                         perm_tbl=None, p_fmt="p = {:.4f}"):
    """Compare representations side by side.

    `perm_tbl` is optional and takes the table permutation_test returns, or
    several of them concatenated. Wherever a (representation, condition) pair is
    found in it, the asterisk code for its p-value is drawn just past the end of
    that bar's error bar, followed by the p-value itself. Representations
    missing from the table are left unannotated, so running the test on only the
    best few is fine.

    `p_fmt` is the format applied to the number, and p_fmt=None drops it so that
    only the asterisks are drawn.
    """
    s = summary.reset_index()
    # a lookup rather than a merge, so that a partial table annotates only the
    # bars it covers and leaves every other bar untouched
    pmap = ({(r.representation, r.condition): r.p
             for r in perm_tbl.itertuples()} if perm_tbl is not None else {})
    order = None
    if sort_by is not None:
        order = (s[s.condition == sort_by]
                 .sort_values(f"{metric}_50")["representation"].tolist())

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(5.2 * len(conditions), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conditions):
        d = s[s.condition == c]
        if order is not None:
            d = d.set_index("representation").loc[order].reset_index()
        else:
            d = d.sort_values(f"{metric}_50")
        yy = np.arange(len(d))
        ax.barh(yy, d[f"{metric}_50"], color="tab:blue", alpha=0.7)
        ax.errorbar(d[f"{metric}_50"], yy, fmt="none", capsize=3, color="black",
                    xerr=[d[f"{metric}_50"] - d[f"{metric}_25"],
                          d[f"{metric}_75"] - d[f"{metric}_50"]])
        ax.axvline(d[f"{metric}_baseline"].iloc[0], color="crimson", ls="--", lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels(d["representation"], fontsize=8)
        ax.set_xlabel(f"test {metric}")
        mapped_c = plotting_conditions.get(c, c)
        ax.set_title(mapped_c)

        if pmap:
            # the label is anchored past the upper whisker rather than the bar
            # end, so that it never sits on top of the error bar
            widest = 0
            for y, nm, hi in zip(yy, d["representation"], d[f"{metric}_75"]):
                lab = _sig_label(pmap.get((nm, c)), p_fmt=p_fmt)
                if lab:
                    widest = max(widest, len(lab))
                    # the gap is given in offset points rather than data units,
                    # so it stays constant whatever the axis range is
                    ax.annotate(lab, xy=(hi, y), xytext=(5, -1),
                                textcoords="offset points", va="center",
                                ha="left", fontsize=9)
            # room is made on the right so the longest label is not clipped. the
            # allowance is scaled by the label length, since appending the
            # p-value makes it roughly four times wider than bare asterisks
            if widest:
                x0, x1 = ax.get_xlim()
                ax.set_xlim(x0, x1 + (0.05 + 0.016 * widest) * (x1 - x0))

    if pmap:
        # fig.text(0.5, -0.02, P_LEGEND, ha="center", fontsize=8)
        fig.legend(handles=[
            Line2D([], [], color="crimson", ls="--", lw=1, label="null distribution mean"),
        ], loc="lower center", ncol=3, frameon=False, fontsize=9,
            bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def plot_inner(inner_summary, conditions=CONDITIONS, inner_raw=None,
               metric="auroc", plotting_conditions=CONDITION_MAP, k_folds=5):
    """Plots inner-CV representation comparisons on the training data.

    Generates a horizontal bar chart per condition showing the median metric 
    (bars), IQR across outer splits (error bars), and optionally the 
    individual outer-split medians (dots).

    Args:
        inner_summary (pd.DataFrame): Aggregated metrics with '_25', '_50', and '_75' columns.
        conditions (list): Conditions to plot in separate subplots.
        inner_raw (pd.DataFrame, optional): Raw data to plot individual outer-split medians as dots.
        metric (str): The metric to plot (default is "auroc")
        plotting_conditions (dict): Mapping conditions to official plot-worthy names
        k_folds (int): Used for the plot title

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    s = inner_summary.reset_index()
    per_split = None
    if inner_raw is not None:
        per_split = (inner_raw.dropna(subset=[metric])
                     .groupby(["representation", "condition", "split_seed"])[metric]
                     .median().reset_index())

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(5.2 * len(conditions), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conditions):
        d = s[s.condition == c].sort_values(f"{metric}_50")
        yy = np.arange(len(d))
        ax.barh(yy, d[f"{metric}_50"], color="tab:green", alpha=0.55)
        ax.errorbar(d[f"{metric}_50"], yy, fmt="none", capsize=3, color="black",
                    xerr=[d[f"{metric}_50"] - d[f"{metric}_25"],
                          d[f"{metric}_75"] - d[f"{metric}_50"]])
        if per_split is not None:
            for i, nm in enumerate(d["representation"]):
                v = per_split[(per_split.condition == c) &
                              (per_split.representation == nm)][metric]
                ax.plot(v, np.full(len(v), i), "o", ms=3,
                        color="black", alpha=0.45, zorder=3)
        baseline = s[s["condition"]==c][f"{metric}_baseline"].iloc[0]
        ax.axvline(baseline, color="crimson", ls="--", lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels(d["representation"], fontsize=8)
        ax.set_xlabel(f"pooled out-of-fold {metric.upper()}")
        
        mapped_c = plotting_conditions.get(c, c)
        ax.set_title(mapped_c)
    plt.suptitle(f"{k_folds}-fold nested CV")
    fig.tight_layout()
    return fig


def plot_paired(res, plotting_conditions=CONDITION_MAP,):
    """Result of one paired_test: raw per-split differences + sign-flip null.
 
    The left panel is the honest evidence -- how many splits favour A, and
    whether A wins consistently or wins hugely on two splits and loses on the
    rest. A mean difference collapses those into the same number.
    """
    d, null = res["_diff"], res["_null"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
 
    x = np.arange(len(d))
    axes[0].scatter(x, d, s=26, color=np.where(d > 0, "tab:green", "tab:red"),
                    zorder=3)
    axes[0].axhline(0, color="grey", lw=1)
    axes[0].axhline(res["mean_diff"], color="crimson", ls="--", lw=1)
    axes[0].set_xlabel("split")
    axes[0].set_ylabel(f"({res['rep_a']})  minus  ({res['rep_b']})")
    axes[0].set_title(f"per-split difference "
                      f"({res['n_a_wins']}/{res['n_splits']} favour {res['rep_a']})")
 
    axes[1].hist(null, bins=50, color="tab:blue", alpha=0.55)
    axes[1].axvline(res["mean_diff"], color="crimson", lw=1.5)
    if res["alternative"] == "two-sided":
        axes[1].axvline(-res["mean_diff"], color="crimson", lw=1.5, alpha=0.4)
    axes[1].set_xlabel("mean difference under random sign flips")
    axes[1].set_ylabel("permutations")
    mapped_c = plotting_conditions.get(res['condition'], res['condition'])
    axes[1].set_title(f"{mapped_c}   p = {res['p']:.4f}")
    fig.tight_layout()
    return fig


def plot_k_sweep(grid, conditions=CONDITIONS, reference=None, metric="auroc",
                 C=None, plotting_conditions=CONDITION_MAP):
    """AUROC against k, one line per ranker, one panel per condition.

    Mean pooling is the k = N endpoint of this curve, so `reference` is not a
    separate method -- it is where the curve should land if extended.

    If dilution is the mechanism, expect a rise as k falls from all-frames
    toward some optimum, then a fall at very small k where too few frames are
    averaged to be stable. A flat line means frame selection is doing nothing.
    """
    d = grid[grid.get("k").notna()] if "k" in grid.columns else grid.iloc[0:0]
    if d.empty:
        raise ValueError("no TopKPool rows in grid (need the 'k' meta column)")
    if C is not None:
        d = d[np.isclose(d.C, C)]

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(4.8 * len(conditions), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conditions):
        sub = d[d.condition == c]
        for i, rk in enumerate(sorted(sub["ranker"].dropna().unique())):
            q = (sub[sub.ranker == rk].groupby("k")[metric]
                 .quantile([.25, .5, .75]).unstack())
            ax.fill_between(q.index, q[0.25], q[0.75], alpha=0.18, color=f"C{i}")
            ax.plot(q.index, q[0.5], marker="o", ms=4, color=f"C{i}", label=rk)
        if reference is not None:
            ref = grid[(grid.condition == c) & (grid.representation == reference)]
            if C is not None:
                ref = ref[np.isclose(ref.C, C)]
            if not ref.empty:
                ax.axhline(ref[metric].median(), color="black", ls="-.", lw=1)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("k (frames kept)")
        mapped_c = plotting_conditions.get(c, c)
        ax.set_title(mapped_c)
    axes[0].set_ylabel(f"{metric}")
    axes[0].legend(fontsize=8, frameon=False)
    fig.legend(handles=[
        Line2D([], [], color="black", ls="-.", lw=1,
               label=f"mean pooling ({reference})" if reference else "mean pooling"),
        Line2D([], [], color="grey", ls="--", lw=1, label="chance"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig



def plot_auroc_vs_steps(df, title, curve_df=None, step_col="steps", auroc_col="auroc_50",
                        lo_col="auroc_25", hi_col="auroc_75",
                        curve_step_col="step", loss_col="loss", wnorm_col="w_norm",
                        ax=None, label='pooled out-of-fold AUROC',):
    """Plot median AUROC against steps, with a fill_between band for the 25-75 range.

    The x axis uses ordinal positions (one slot per observed step value), so the
    ticks are exactly the steps present in `df` and no intermediate values appear.
    Swap `x = d[step_col].to_numpy()` and drop the `set_xticks` call if you want
    true linear spacing instead.

    If `curve_df` is given (columns `curve_step_col`, `loss_col`, `wnorm_col`), the
    training loss and weight norm are drawn on two extra y axes offset to the left.
    Their step values are mapped onto the ordinal grid by linear interpolation, so
    they stay aligned with the AUROC markers but the step spacing is warped.
    """
    d = df.sort_values(step_col)
    steps = d[step_col].to_numpy()
    x = np.arange(len(d))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.fill_between(
        x,
        d[lo_col].to_numpy(),
        d[hi_col].to_numpy(),
        color="tab:green",
        alpha=0.18,
        linewidth=0,
        label="per-split AUROC (IQR)"
    )
    ax.plot(
        x,
        d[auroc_col].to_numpy(),
        marker="o",
        markersize=5,
        linewidth=1.5,
        color="tab:green",
        label=label,
    )

    ax.axhline(0.5, ls="--", lw=1, color="crimson", label="Baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.set_xlabel("Steps")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    extra_axes = []
    if curve_df is not None:
        c = curve_df.sort_values(curve_step_col)
        xc = np.interp(c[curve_step_col].to_numpy(), steps, x)

        for col, color, ylabel, offset, rot in [
            (loss_col, "tab:blue", "loss", 1.0, 90),
            (wnorm_col, "tab:purple", r"$\|w\|$", 1.15, 0),
        ]:
            axe = ax.twinx()
            axe.spines["left"].set_visible(False)
            axe.spines["right"].set_visible(True)
            axe.spines["right"].set_position(("axes", offset))
            axe.yaxis.set_label_position("right")
            axe.yaxis.set_ticks_position("right")
            axe.plot(xc, c[col].to_numpy(), lw=1.2, color=color, label=ylabel)
            axe.set_ylabel(ylabel, color=color, rotation=rot, labelpad=4)
            axe.tick_params(axis="y", colors=color)
            axe.spines["right"].set_color(color)
            extra_axes.append(axe)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [label, "per-split AUROC (IQR)", "Baseline"]
    for axe in extra_axes:
        h, l = axe.get_legend_handles_labels()
        by_label.update(zip(l, h))
        order += l

    ax.legend(
        [by_label[l] for l in order], order,
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=3, frameon=False,
    )
    ax.figure.subplots_adjust(bottom=0.28, right=0.76 if extra_axes else 0.95)

    return ax