"""Accession-level representations.

A representation maps a list of accession identifiers to a feature matrix. It is
the only component that changes between experiments; every function in
:mod:`evaluate` operates on any representation without modification.

The interface expected by :mod:`evaluate` is::

    rep.accessions        accessions a vector can be produced for
    rep.name              unique identifier, written into every result row
    rep.meta              extra columns emitted alongside every result row
    rep.needs_labels      whether fit() consumes y_train
    rep.clone()           fresh copy carrying no fitted state
    rep.fit(acc, y)       fit on training accessions only; may be a no-op
    rep.transform(acc)    feature matrix of shape (len(acc), D)

A fit/transform pair is used rather than a precomputed feature matrix because
some pooling strategies learn from the training labels. Calling ``fit`` inside
the split loop, on training accessions only, is what keeps those strategies free
of leakage.
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import C_HEADLINE

# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------

class Representation:
    """Base class grouping frame rows by accession.

    Subclasses implement :meth:`_vec`, which turns the frames of one accession
    into a single vector. Subclasses that learn from the training data also
    override :meth:`fit`.

    Attributes:
        E (numpy.ndarray): Frame embedding matrix of shape (n_frames, D), held
            by reference and never modified.
        name (str): Identifier written into every result row.
        meta (dict): Extra key/value pairs emitted as columns alongside every
            result row. Empty in the base class.
        accessions (numpy.ndarray): Sorted accession identifiers this
            representation can produce a vector for.
        needs_labels (bool): Whether :meth:`fit` consumes `y_train`. False in
            the base class. Callers rely on this to decide whether the output of
            :meth:`transform` stays constant when only the labels change, which
            allows it to be computed once and reused across the draws of a label
            permutation test. Subclasses whose `fit` reads the labels must set
            it to True, otherwise those draws would all reuse the same features
            and the null would be invalid.
    """

    needs_labels = False

    def __init__(self, frames, E, row_col="input_row", acc_col="AccessionNumber",
                 name="base"):
        """Group the frames of `frames` by accession.

        Args:
            frames (pandas.DataFrame): One row per frame, containing at least
                `row_col` and `acc_col`. Only the frames present here are used,
                so passing a filtered subset narrows both the frames pooled and
                the accessions covered.
            E (numpy.ndarray): Frame embedding matrix of shape (n_frames, D),
                indexed by the values in `row_col`.
            row_col (str): Column holding the integer row index into `E`.
            acc_col (str): Column holding the accession identifier.
            name (str): Identifier written into every result row. Must be unique
                across the representations compared in a single analysis, since
                it is the key on which results are joined.
        Raises:
            KeyError: If `row_col` is not a column of `frames`.
            ValueError: If any value in `row_col` is out of bounds for `E`,
                which indicates that the frame table and the embedding matrix
                are misaligned.
        """
        if row_col not in frames.columns:
            raise KeyError(f"{row_col!r} not in frames columns")
        rows = frames[row_col].to_numpy()
        if rows.max() >= len(E):
            raise ValueError(f"{row_col} max {rows.max()} exceeds len(E)={len(E)}")

        self.E = E
        self.name = name
        self.meta = {}          # extra columns emitted by run_grid
        self._rows = {a: g[row_col].to_numpy()
                      for a, g in frames.groupby(acc_col, sort=True)}
        self.accessions = np.array(sorted(self._rows))

    def clone(self):
        """Return a fresh copy carrying no fitted state.

        `E` is shared by reference rather than copied. It is only ever read, so
        sharing is safe, whereas deep-copying it once per split would dominate
        the runtime of the permutation test. Everything :meth:`fit` is able to
        mutate is still deep-copied, so the caller's object is left unfitted and
        every split starts from the same state.

        Returns:
            Representation: A copy with independent fitted state and a shared
            embedding matrix.
        """
        # E is detached for the duration of the copy and restored afterwards, so
        # that deepcopy never walks it. the try/finally guarantees the original
        # is put back even if the copy raises.
        E = self.E
        self.E = None
        try:
            new = copy.deepcopy(self)
        finally:
            self.E = E
        new.E = E
        return new

    def fit(self, acc_train, y_train):
        """Fit any state the representation learns from the training data.

        A no-op in the base class. Callers invoke this with training accessions
        only, which is what keeps supervised pooling free of leakage.

        Args:
            acc_train (numpy.ndarray): Training accession identifiers.
            y_train (numpy.ndarray): Binary labels aligned with `acc_train`.
        Returns:
            Representation: self, so that callers can chain
            ``clone().fit(...)``.
        """
        return self

    def transform(self, acc):
        """Build the feature matrix for the given accessions.

        Args:
            acc (Sequence[str]): Accession identifiers, all of which must be
                present in `accessions`.
        Returns:
            numpy.ndarray: Matrix of shape (len(acc), D) and dtype float32, with
            rows in the order given by `acc`.
        """
        return np.vstack([self._vec(a) for a in acc]).astype(np.float32)

    def _vec(self, a):
        """Return the feature vector of a single accession.

        This is the only method a subclass with no fitted state has to
        implement.

        Args:
            a (str): Accession identifier.
        Returns:
            numpy.ndarray: One-dimensional feature vector. Its length must be
            the same for every accession.
        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def n_frames(self, a):
        """Return how many frames are grouped under one accession.

        Args:
            a (str): Accession identifier.
        Returns:
            int: Number of frames.
        """
        return len(self._rows[a])

    def __repr__(self):
        return f"<{type(self).__name__} {self.name} n_acc={len(self.accessions)}>"


# --------------------------------------------------------------------------
# pooling over all frames of an accession
# --------------------------------------------------------------------------

class MeanPool(Representation):
    """Mean of every frame embedding belonging to an accession.

    The vectors are precomputed at construction, so :meth:`transform` is a
    dictionary lookup. This is the reference representation the others are
    compared against.
    """

    def __init__(self, frames, E, name="mean_all", **kw):
        """Build the representation and precompute one mean vector per accession.

        Args:
            frames (pandas.DataFrame): One row per frame.
            E (numpy.ndarray): Frame embedding matrix of shape (n_frames, D).
            name (str): Identifier written into every result row.
            **kw: Forwarded to :class:`Representation`, for example `row_col`
                and `acc_col`.
        """
        super().__init__(frames, E, name=name, **kw)
        self._cache = {a: self.E[r].mean(0) for a, r in self._rows.items()}

    def _vec(self, a):
        """Return the precomputed mean vector of one accession.
        Args:
            a (str): Accession identifier.
        Returns:
            numpy.ndarray: Mean embedding of shape (D,).
        """
        return self._cache[a]


class SeriesPool(MeanPool):
    """Mean over the frames of a single imaging series.

    Frames are filtered to one series before pooling, so only accessions
    containing that series are covered. The accession count can therefore be
    lower than for :class:`MeanPool`, which matters when comparing results
    across series.
    Attributes:
        series: The value of `series_col` this representation pools over.
    """

    def __init__(self, frames, E, series, series_col="SeriesDescription", **kw):
        """Build the representation from the frames of one series.

        Args:
            frames (pandas.DataFrame): One row per frame.
            E (numpy.ndarray): Frame embedding matrix of shape (n_frames, D).
            series: Value of `series_col` selecting the frames to pool.
            series_col (str): Column identifying the series of each frame.
            **kw: Forwarded to :class:`MeanPool`.
        Raises:
            ValueError: If no frame matches `series`.
        """
        sub = frames[frames[series_col] == series]
        if sub.empty:
            raise ValueError(f"no frames with {series_col} == {series!r}")
        self.series = series
        super().__init__(sub, E, name=f"mean_{series}", **kw)


def per_series(frames, E, series_col="SeriesDescription", **kw):
    """Build one :class:`SeriesPool` per distinct series.
    Args:
        frames (pandas.DataFrame): One row per frame.
        E (numpy.ndarray): Frame embedding matrix of shape (n_frames, D).
        series_col (str): Column identifying the series of each frame.
        **kw: Forwarded to each :class:`SeriesPool`.
    Returns:
        list[SeriesPool]: One representation per distinct non-null value of
        `series_col`, ordered by that value.
    """
    return [SeriesPool(frames, E, s, series_col=series_col, **kw)
            for s in sorted(frames[series_col].dropna().unique())]


class SeriesBalancedPool(MeanPool):
    """Unweighted mean of an accession's per-series means.

    :class:`MeanPool` averages every frame equally, so each series influences
    the result in proportion to how many slices it contributes. Sequences differ
    widely in slice count, which gives the longest ones most of the weight for
    reasons unrelated to what they show. This class averages each series to a
    single vector first and then averages those vectors, so every series carries
    the same influence:

        z = (1 / T) * sum_t [ (1 / K_t) * sum_{i in t} E_i ]

    where T is the number of series the accession contains and K_t the number of
    its frames in series t. Writing mean pooling as sum_t (K_t / K) * z_t makes
    the difference explicit: both are convex combinations of the same per-series
    means, weighted by slice count in one case and uniformly in the other. They
    coincide only when every series contributes the same number of frames.

    No fitting is involved, so the vectors are precomputed at construction and
    :meth:`transform` is a dictionary lookup.

    Attributes:
        series_col (str): Column identifying the series of each frame.
    """

    def __init__(self, frames, E, series_col="SeriesDescription",
                 name="mean_balanced", row_col="input_row", **kw):
        """Build the representation and precompute one vector per accession.

        Args:
            frames (pandas.DataFrame): One row per frame, containing `row_col`,
                the accession column and `series_col`.
            E (numpy.ndarray): Frame embedding matrix of shape (n_frames, D).
            name (str): Identifier written into every result row.
            series_col (str): Column identifying the series of each frame.
            row_col (str): Column holding the integer row index into `E`.
            **kw: Forwarded to :class:`Representation`.

        Note:
            Each accession is balanced over the series it actually contains. In a
            cohort where every accession contains the same series this is a fixed
            uniform weighting; otherwise an accession holding fewer series
            weights each of them more heavily than one holding more.
        """
        # Representation is called directly rather than MeanPool, so that the
        # plain frame mean is not built and then thrown away
        Representation.__init__(self, frames, E, row_col=row_col, name=name, **kw)
        self.series_col = series_col

        # the series of each frame is resolved once, then every accession is
        # reduced by averaging within each series and averaging those means
        lut = dict(zip(frames[row_col].to_numpy(), frames[series_col].to_numpy()))
        self._cache, self._n_ser = {}, {}
        for a, rows in self._rows.items():
            ser = np.array([lut[r] for r in rows])
            uniq = np.unique(ser)
            self._n_ser[a] = len(uniq)
            self._cache[a] = np.stack(
                [self.E[rows[ser == t]].mean(0) for t in uniq]).mean(0)

    def n_series(self, a):
        """Return how many distinct series one accession contributes.

        Args:
            a (str): Accession identifier.

        Returns:
            int: Number of series averaged for that accession, the denominator
            T in the pooling formula. A cohort in which this is constant gives
            every accession the same per-series weighting.
        """
        return self._n_ser[a]


# --------------------------------------------------------------------------
# frame rankers
# --------------------------------------------------------------------------
#
# A ranker assigns a score to every frame of an accession, and TopKPool keeps
# the highest scoring ones. The protocol is:
#
#     ranker.name                 str, used in the pool's name and meta
#     ranker.needs_labels         bool, advisory only; nothing reads it
#     ranker.fit(rep, acc, y)     called from TopKPool.fit; returns self
#     ranker.score(Efr, a, rep)   one score per row of Efr; higher is kept
#
# Rankers are duck-typed; no base class is required.

class ProbeRanker:
    """Score frames by their projection onto a linear probe direction.

    The direction is fitted on mean-pooled training accessions and then applied
    to individual frames. An accession vector is the mean of its frame
    embeddings, so frames and accession means occupy the same space; a frame's
    score is therefore exactly its contribution to the accession's score.

    The fitted coefficients are divided by the scaler's per-feature scale so
    that they act on raw embeddings rather than standardised ones.

    Attributes:
        name (str): Identifier used in the pool's name and meta.
        needs_labels (bool): True, as this ranker requires training labels.
        C (float): Inverse L2 regularisation strength of the probe. Independent
            of the classifier's regularisation used in :mod:`evaluate`.
        w (numpy.ndarray or None): Fitted direction of shape (D,), None until
            :meth:`fit` has been called.
    """

    name = "probe"
    needs_labels = True

    def __init__(self, C=C_HEADLINE):
        """Configure the probe.
        Args:
            C (float): Inverse L2 regularisation strength of the probe.
        """
        self.C = C
        self.w = None

    def fit(self, rep, acc_train, y_train):
        """Fit the probe direction on mean-pooled training accessions.
        Args:
            rep (Representation): Representation supplying `E` and the frame
                grouping. Only the accessions in `acc_train` are read.
            acc_train (numpy.ndarray): Training accession identifiers.
            y_train (numpy.ndarray): Binary labels aligned with `acc_train`.
        Returns:
            ProbeRanker: self, with `w` populated.
        """
        X = np.vstack([rep.E[rep._rows[a]].mean(0) for a in acc_train])
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, l1_ratio=0, solver="lbfgs",
                               class_weight="balanced", max_iter=5000))
        pipe.fit(X, y_train)
        sc = pipe.named_steps["standardscaler"]
        lr = pipe.named_steps["logisticregression"]
        self.w = lr.coef_[0] / sc.scale_
        return self

    def score(self, Efr, a, rep):
        """Score the frames of one accession.
        Args:
            Efr (numpy.ndarray): Frame embeddings of shape (n_frames, D) for
                accession `a`.
            a (str): Accession identifier.
            rep (Representation): Representation the frames came from.
        Returns:
            numpy.ndarray: One score per frame, of shape (n_frames,).
        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.w is None:
            raise RuntimeError("ProbeRanker.fit() not called")
        return Efr @ self.w


class CentroidRanker:
    """Score frames by cosine distance from a centroid.

    Unsupervised, so no leakage is possible. The assumption is that most slices
    resemble ordinary anatomy and cluster near the centroid, while a slice
    containing a large finding sits further out. Slices that are atypical for
    other reasons, such as localizers, edge slices or motion artifacts, score
    highly as well; restricting the comparison to within a series reduces that.

    Attributes:
        name (str): Identifier used in the pool's name and meta.
        needs_labels (bool): False.
        per_series (bool): Whether the centroid is computed per series rather
            than over all frames of the accession.
        series_key (dict or None): Maps frame row index to series identifier.
            Required for the per-series centroid.
    """

    name = "centroid"
    needs_labels = False

    def __init__(self, per_series=False, series_key=None):
        """Configure the centroid comparison.

        Args:
            per_series (bool): If True, compare each frame against the centroid
                of its own series instead of the accession centroid. Falls back
                to the accession centroid when `series_key` is None.
            series_key (dict, optional): Mapping from frame row index to series
                identifier.
        """
        self.per_series = per_series
        self.series_key = series_key   # dict: input_row -> series id

    def fit(self, rep, acc_train, y_train):
        """Return self; this ranker learns nothing.

        Args:
            rep (Representation): Unused.
            acc_train (numpy.ndarray): Unused.
            y_train (numpy.ndarray): Unused.
        Returns:
            CentroidRanker: self.
        """
        return self

    def score(self, Efr, a, rep):
        """Score the frames of one accession by distance from their centroid.

        Args:
            Efr (numpy.ndarray): Frame embeddings of shape (n_frames, D) for
                accession `a`.
            a (str): Accession identifier.
            rep (Representation): Representation the frames came from, used to
                recover the series of each frame when `per_series` is set.
        Returns:
            numpy.ndarray: One score per frame, of shape (n_frames,). Larger
            values are further from the centroid.
        """
        n = Efr / (np.linalg.norm(Efr, axis=1, keepdims=True) + 1e-9)
        if not self.per_series or self.series_key is None:
            return 1 - n @ (n.mean(0) / (np.linalg.norm(n.mean(0)) + 1e-9))
        s = np.array([self.series_key[r] for r in rep._rows[a]])
        out = np.empty(len(Efr))
        for grp in np.unique(s):
            m = s == grp
            c = n[m].mean(0)
            out[m] = 1 - n[m] @ (c / (np.linalg.norm(c) + 1e-9))
        return out


class CentralityRanker:
    """Score frames by how central they are in their series stack.

    Purely geometric. Each series contributes in proportion to its share of the exam.

    Attributes:
        name (str): Identifier used in the pool's name and meta.
        needs_labels (bool): False.
    """

    name = "central"
    needs_labels = False

    def __init__(self, frames, row_col="input_row",
                 rank_col="slice_rank", n_col="n_slices_in_series"):
        """Precompute a centrality score for every frame.

        Args:
            frames (pandas.DataFrame): One row per frame. Must cover every frame
                the ranker will later be asked to score, so passing the full
                frame table is safe while passing a different subset is not.
            row_col (str): Column holding the integer row index into `E`.
            rank_col (str): Column holding the position of the slice within its
                series.
            n_col (str): Column holding the number of slices in that series.
        """
        rel = frames[rank_col].to_numpy() / np.maximum(frames[n_col].to_numpy(), 1)
        self._s = dict(zip(frames[row_col].to_numpy(), -np.abs(rel - 0.5)))

    def fit(self, rep, acc_train, y_train):
        """Return self; this ranker learns nothing.
        Args:
            rep (Representation): Unused.
            acc_train (numpy.ndarray): Unused.
            y_train (numpy.ndarray): Unused.
        Returns:
            CentralityRanker: self.
        """
        return self

    def score(self, Efr, a, rep):
        """Look up the precomputed centrality of each frame of one accession.
        Args:
            Efr (numpy.ndarray): Unused; present for protocol compatibility.
            a (str): Accession identifier.
            rep (Representation): Representation supplying the frame grouping.
        Returns:
            numpy.ndarray: One score per frame, of shape (n_frames,). Larger
            values are closer to the middle of the stack.
        Raises:
            KeyError: If a frame was not present in the table passed to
                :meth:`__init__`.
        """
        return np.array([self._s[r] for r in rep._rows[a]])


class TopKPool(Representation):
    """Mean over the `k` highest scoring frames of each accession.

    Frames are scored by the supplied ranker and the top `k` are averaged. When
    an accession has fewer than `k` frames, all of them are used, so the result
    degrades gracefully to mean pooling.

    `k` is a hyperparameter and should be selected on the training portion, for
    example with :func:`evaluate.run_inner`, rather than on test performance.

    Attributes:
        k (int): Number of frames kept.
        ranker: Object following the ranker protocol described in this module.
        needs_labels (bool): Delegated to the ranker, since the pool consumes
            the labels only by passing them on. Unknown rankers are assumed to
            need them, which is the conservative default.
    """

    @property
    def needs_labels(self):
        """bool: Whether the ranker's fit consumes the training labels."""
        return getattr(self.ranker, "needs_labels", True)

    def __init__(self, frames, E, k, ranker, **kw):
        """Build the representation.

        Args:
            frames (pandas.DataFrame): One row per frame. Passing a filtered
                subset restricts both the frames ranked and the accessions
                covered.
            E (numpy.ndarray): Frame embedding matrix of shape (n_frames, D).
            k (int): Number of highest scoring frames to average.
            ranker: Object providing `name`, `fit` and `score` as described in
                the ranker protocol. A separate instance should be used per
                pool, since `fit` mutates it.
            **kw: Forwarded to :class:`Representation`.
        """
        super().__init__(frames, E, name=f"top{k}_{ranker.name}", **kw)
        self.k = k
        self.ranker = ranker
        self.meta = {"k": k, "ranker": ranker.name}

    def fit(self, acc_train, y_train):
        """Fit the ranker on the training accessions.

        Args:
            acc_train (numpy.ndarray): Training accession identifiers.
            y_train (numpy.ndarray): Binary labels aligned with `acc_train`.
        Returns:
            TopKPool: self, with the ranker fitted.
        """
        self.ranker.fit(self, acc_train, y_train)
        return self

    def _vec(self, a):
        """Average the `k` highest scoring frames of one accession.

        Args:
            a (str): Accession identifier.
        Returns:
            numpy.ndarray: Pooled embedding of shape (D,).
        """
        rows = self._rows[a]
        k = min(self.k, len(rows))
        s = self.ranker.score(self.E[rows], a, self)
        top = rows[np.argpartition(-s, k - 1)[:k]]
        return self.E[top].mean(0)
