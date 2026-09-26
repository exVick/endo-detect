"""Attention-based pooling over the frames of an accession.

The pooled vector is a convex combination of frame embeddings: attention weights
are formed per imaging series with a softmax, and the resulting per-series
vectors are combined with a second softmax over a learned parameter. The output
therefore lies in the same space as the mean of the frames, which is what allows
this to be used wherever :class:`~.representations.MeanPool` is used and
compared against it directly.

Because the attention weights are learned by backpropagation they need a
training signal, so a linear head is trained alongside them inside
:meth:`AttentionPool.fit` and then discarded. Only the pooled vector is exposed,
and the classifier that produces the reported metrics is the same logistic probe
:mod:`evaluate` applies to every other representation. This mirrors
:class:`~.representations.ProbeRanker`, which likewise fits a classifier
internally only to shape the pooling and then throws it away, and it keeps the
comparison between representations a comparison of pooling alone.

At initialisation the module reduces to an unweighted mean of the per-series
means: the attention scores start at zero, so the within-series softmax is
uniform, and the across-series parameter starts at zero, so that softmax is
uniform too. Note that this is not the same as pooling every frame equally
unless all series contribute the same number of frames, since mean pooling
weights each series by its slice count.

Accessors on a fitted object expose what the model learned: the mixture over
series, the attention placed on individual frames, and the training history.
They read fitted state, so they must be called on the object returned by
:meth:`AttentionPool.fit` rather than on the template passed to the pipeline,
which is never fitted in place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._torch_utils import _check_deterministic, _deterministic
from .representations import Representation


# fixed rather than derived from the data, so that the series indices are
# identical for every split, every run and every cohort subset
SERIES_MAP = {
    't2_sag': 0,
    't2_tra': 1,
    't2_cor': 2,
    't1_tra': 3,
    't1_fs_cor': 4,
    't1_starvibe_tra_iso': 5,
    't1_cor': 6,
    't1_vibe_dixon_ax_bh_W': 7,
}

class peak_memory:
    """Context manager reporting the peak CUDA memory used inside the block.

    Peak statistics are reset on entry, so the reported figure covers only the
    enclosed work. Outside CUDA the manager is inert and reports zero.

    Args:
        device: CUDA device to measure. Defaults to the current device.
        label (str): Text appended to the printed line, to tell several
            measurements apart.
        verbose (bool): Whether to print on exit.

    Attributes:
        allocated (int): Peak bytes allocated by tensors.
        reserved (int): Peak bytes reserved by the caching allocator, which is
            what nvidia-smi reports.

    Example:
        >>> with peak_memory(label=" (inner CV)"):
        ...     inner = run_inner(reps, labels, splits)
        peak GPU memory (inner CV): 1423 MiB allocated, 1642 MiB reserved
    """

    def __init__(self, device=None, label="", verbose=True):
        self.device = device
        self.label = label
        self.verbose = verbose
        self.allocated = 0
        self.reserved = 0

    def _on(self):
        return torch.cuda.is_available()

    def __enter__(self):
        if self._on():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *exc):
        if self._on():
            torch.cuda.synchronize(self.device)
            self.allocated = torch.cuda.max_memory_allocated(self.device)
            self.reserved = torch.cuda.max_memory_reserved(self.device)
        if self.verbose:
            print(self)
        return False

    def __str__(self):
        if not self._on():
            return f"peak GPU memory{self.label}: no CUDA device"
        return (f"peak GPU memory{self.label}: {self.allocated / 2**20:.0f} MiB "
                f"allocated, {self.reserved / 2**20:.0f} MiB reserved")

    __repr__ = __str__


class _AttentionPoolNet(nn.Module):
    """Attention pooling with a per-series bias and a learned series mixture.

    Every accession in a call is processed in one batch. Frames are labelled
    with a segment index ``accession * n_series + series``, the per-series bias
    becomes an embedding lookup, and the per-series softmaxes become a single
    segment softmax. That removes the Python loops over accessions and over
    series, which on a GPU dominate the runtime through kernel launch overhead.

    Attributes:
        n_ser (int): Number of imaging series.
        st_emb (torch.nn.Embedding): Per-series bias added inside the
            non-linearity. Zero-initialised, so no series is favoured at first.
        W_V (torch.nn.Linear): Projection from embedding to attention hidden
            space. The only randomly initialised parameter in the model.
        W_w (torch.nn.Linear): Projection from hidden space to a scalar score.
            Zero-initialised, which makes the within-series softmax uniform.
        beta (torch.nn.Parameter): Logits of the mixture over series.
            Zero-initialised, so the mixture starts uniform.
    """

    def __init__(self, emb_dim, hidden_size, n_series=8):
        """Build the attention module.

        Args:
            emb_dim (int): Dimension of a frame embedding.
            hidden_size (int): Width of the attention hidden layer.
            n_series (int): Number of imaging series.
        """
        super().__init__()
        self.n_ser = n_series

        # learn emb for series types (st)
        self.st_emb = nn.Embedding(self.n_ser, hidden_size)
        nn.init.zeros_(self.st_emb.weight)  # start neutral, st has no effect

        self.W_V = nn.Linear(emb_dim, hidden_size, bias=False)
        nn.init.xavier_normal_(self.W_V.weight)  # if 0, no gradient

        # no bias, as softmax is shift invariant
        self.W_w = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.W_w.weight)  # MLP starts without any preference

        self.beta = nn.Parameter(torch.zeros(self.n_ser))
        # softmax(8x0s)=1/8 -> so distr over st becomes uniform

    def alpha(self, Ecat, seg, n_seg):
        """Attention weight of every frame within its own segment.

        Args:
            Ecat (torch.Tensor): Frame embeddings of shape (M, L), the frames of
                all accessions concatenated.
            seg (torch.Tensor): Segment index per frame, shape (M,).
            n_seg (int): Total number of segments.

        Returns:
            torch.Tensor: Weights of shape (M,), summing to one within each
            segment.
        """
        u = torch.tanh(self.W_V(Ecat) + self.st_emb(seg % self.n_ser)) # (M,L)@(L,D) + (M,D) -> (M,D)
        sc = self.W_w(u).squeeze(-1) # (M,D)@(D,1)->(M,1) -> (M,)
        # the max is subtracted per segment for numerical stability, the same
        # role softmax plays internally
        mx = torch.full((n_seg,), float("-inf"), device=Ecat.device,
                        dtype=sc.dtype).scatter_reduce(
            0, seg, sc, "amax", include_self=False)
        e = torch.exp(sc - mx[seg])
        den = torch.zeros(n_seg, device=Ecat.device, dtype=e.dtype).index_add_(0, seg, e)
        return e / den[seg]

    def forward(self, Ecat, seg, n_acc):
        """Pool the frames of several accessions at once.

        Args:
            Ecat (torch.Tensor): Frame embeddings of shape (M, L).
            seg (torch.Tensor): Segment index per frame, shape (M,), equal to
                ``accession_index * n_series + series_index``.
            n_acc (int): Number of accessions represented in `Ecat`.

        Returns:
            torch.Tensor: Pooled vectors of shape (n_acc, L), each a convex
            combination of that accession's frames.
        """
        n_seg = n_acc * self.n_ser
        a = self.alpha(Ecat, seg, n_seg)  # (M,)
        Z = torch.zeros(n_seg, Ecat.shape[1], device=Ecat.device,
                        dtype=Ecat.dtype).index_add_(0, seg, a.unsqueeze(1) * Ecat)  # (M,1)*(M,L) = (M,L)
        gamma = F.softmax(self.beta, dim=0).view(1, self.n_ser, 1)
        return (Z.view(n_acc, self.n_ser, -1) * gamma).sum(1)


class _LinearHead(nn.Module):
    """Linear classification head used only to train the attention module.

    It is discarded once :meth:`AttentionPool.fit` returns; the classifier that
    produces reported metrics is the logistic probe in :mod:`evaluate`.
    """

    def __init__(self, emb_dim):
        """Build the head.

        Args:
            emb_dim (int): Dimension of the pooled vector.
        """
        super().__init__()
        self.classifier = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, z):
        """Map pooled vectors to logits.

        Args:
            z (torch.Tensor): Pooled vectors of shape (N, L).

        Returns:
            torch.Tensor: Logits of shape (N, 1).
        """
        return self.classifier(z)


class AttentionPool(Representation):
    """Accession vector formed by learned attention over its frames.

    The attention module is trained inside :meth:`fit` against a linear head
    that is then discarded, so the representation exposes only the pooled
    vector. Everything downstream is unchanged, which makes this directly
    comparable with mean, per-series and top-k pooling.

    ``needs_labels`` is True because :meth:`fit` consumes `y_train`. That
    disables the split cache in the permutation test, so the module is retrained
    under every permuted labelling, as it must be for the null to be valid.

    Attributes:
        series_map (dict): Maps the series column value to a series index.
        n_series (int): Number of imaging series.
        hidden (int): Width of the attention hidden layer.
        steps (int): Number of full-batch training steps, each of which is one
            pass over the whole training set and therefore one epoch.
        lr (float): AdamW learning rate.
        weight_decay (float): AdamW weight decay, applied to non-bias
            parameters. It pulls every parameter towards zero, which is the
            initialisation, so it shrinks the model towards uniform pooling.
        device (str): Device the module is trained and evaluated on.
        seed (int): Seed for the one random initialisation in the model.
        deterministic (bool): Whether to force bit-reproducible results.
        net (_AttentionPoolNet or None): Trained module, None until :meth:`fit`
            has been called.
        history (pandas.DataFrame or None): One row per training step with the
            loss and the norm of the attention output layer. None until fitted.
        collapsed (bool): Whether training was stopped early because the loss
            became non-finite.
        stopped_at (int or None): Step at which training stopped, when it
            stopped early.
    """

    needs_labels = True

    def __init__(self, frames, E, series_col="SeriesDescription", series_map=None,
                 hidden=16, steps=300, lr=1e-3, weight_decay=1e-2,
                 device="cpu", seed=0, deterministic=True, log_every=1,
                 check_every=25, name=None, row_col="input_row", **kw):
        """Group the frames by accession and record the series index of each.

        Args:
            frames (pandas.DataFrame): One row per frame, containing `row_col`,
                the accession column and `series_col`.
            E (numpy.ndarray): Frame embedding matrix of shape (n_frames, L),
                dtype float32.
            series_col (str): Column identifying the series of each frame.
            series_map (dict, optional): Maps series names to indices. Defaults
                to :data:`SERIES_MAP`.
            hidden (int): Width of the attention hidden layer.
            steps (int): Number of full-batch training steps per fit.
            lr (float): AdamW learning rate.
            weight_decay (float): AdamW weight decay for non-bias parameters.
            device (str): Device to train and evaluate on, for example "cpu" or
                "cuda".
            seed (int): Seed for the initialisation. Held fixed across every
                fit, so that differences between splits reflect the data rather
                than the draw, and so that a permutation null varies only with
                the labels.
            deterministic (bool): If True, force deterministic kernels so that
                repeated runs agree exactly. The scatter operations used by the
                batched forward accumulate with atomics otherwise, which lets
                results drift in the last few digits. Costs roughly a tenth of
                the runtime and, on CUDA, requires the environment variable
                CUBLAS_WORKSPACE_CONFIG to be set before torch is imported.
            log_every (int): Record the loss every this many steps. Values are
                kept on the device and transferred once after training, so the
                cost is negligible. Zero disables the history.
            check_every (int): Test the loss for non-finite values every this
                many steps, restoring the last good state and stopping if one is
                found. Each test synchronises with the device, so this is not
                done every step. Zero disables the check.
            name (str, optional): Identifier. Defaults to one encoding the
                hyperparameters, so that two configurations do not collide.
            row_col (str): Column holding the integer row index into `E`.
            **kw: Forwarded to :class:`~.representations.Representation`.

        Raises:
            AssertionError: If `E` is not float32, or if a value of `series_col`
                is absent from `series_map`.
        """
        super().__init__(frames, E, row_col=row_col,
                         name=name or f"attn_h{hidden}_s{steps}", **kw)

        assert E.dtype == np.float32, (
            f"Embedding datatypes should be float32 for nn.Linear(), "
            f"but they are {E.dtype} instead")

        self.series_map = dict(SERIES_MAP if series_map is None else series_map)
        self.n_series = len(self.series_map)
        self.hidden = hidden
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.seed = seed
        self.deterministic = deterministic
        self.log_every = log_every
        self.check_every = check_every
        self.net = None
        self.history = None
        self.collapsed = False
        self.stopped_at = None
        self.meta = {"hidden": hidden, "steps": steps, "lr": lr,
                     "weight_decay": weight_decay}

        # the series index is resolved once here, per frame, and then reordered
        # to match the frame order of _rows. that alignment is what keeps the
        # segment index in step with the rows of Ecat inside the forward pass
        codes = frames[series_col].map(self.series_map)
        assert codes.notna().all(), (
            f"unmapped {series_col} values: "
            f"{sorted(frames.loc[codes.isna(), series_col].unique())}")
        lut = dict(zip(frames[row_col].to_numpy(),
                       codes.to_numpy().astype(np.int64)))
        self._sid = {a: np.array([lut[r] for r in rows], dtype=np.int64)
                     for a, rows in self._rows.items()}
        self._inv_series = {v: k for k, v in self.series_map.items()}

    # ------------------------------------------------------------------
    # packing
    # ------------------------------------------------------------------

    def _pack(self, acc, dev):
        """Concatenate the frames of several accessions into one batch.

        Args:
            acc (Sequence[str]): Accession identifiers.
            dev (torch.device): Device to place the tensors on.

        Returns:
            tuple: ``(Ecat, seg)`` of shapes (M, L) and (M,), where `seg` is
            ``accession_index * n_series + series_index``.

        Raises:
            ValueError: If an accession contains no frame of some series, which
                would leave an empty segment and an undefined softmax.
        """
        rows = np.concatenate([self._rows[a] for a in acc])
        sid = np.concatenate([self._sid[a] for a in acc])
        counts = np.array([len(self._rows[a]) for a in acc])
        seg = np.repeat(np.arange(len(acc)), counts) * self.n_series + sid

        # checked once per batch on the host, rather than per step on the
        # device, so it costs nothing inside the training loop
        n_seg = len(acc) * self.n_series
        empty = np.flatnonzero(np.bincount(seg, minlength=n_seg) == 0)
        if len(empty):
            i, t = empty[0] // self.n_series, empty[0] % self.n_series
            raise ValueError(f"accession {acc[i]!r} has no frames for series "
                             f"{self._inv_series[t]!r}")

        return (torch.from_numpy(self.E[rows]).to(dev),
                torch.from_numpy(seg).to(dev))

    def _check_deterministic(self, dev):
        """Verify that deterministic execution is actually available.

        Args:
            dev (torch.device): Device the work will run on.

        Raises:
            RuntimeError: If deterministic execution is requested on CUDA
                without CUBLAS_WORKSPACE_CONFIG having been set before torch was
                imported.
        """
        _check_deterministic(self.deterministic, dev)

    def _require_fitted(self):
        """Raise if the module has not been trained yet.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.net is None:
            raise RuntimeError("AttentionPool.fit() not called")

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, acc_train, y_train):
        """Train the attention module on the training accessions.

        A linear head is trained jointly and then discarded. Class imbalance is
        handled with a positive-class weight computed from `y_train` alone,
        which matches the ratio ``class_weight="balanced"`` gives the logistic
        probe, so both models weight the classes the same way.

        Args:
            acc_train (numpy.ndarray): Training accession identifiers.
            y_train (numpy.ndarray): Binary labels aligned with `acc_train`.

        Returns:
            AttentionPool: self, with `net`, `history`, `collapsed` and
            `stopped_at` populated.

        Raises:
            RuntimeError: If deterministic execution is requested on CUDA
                without CUBLAS_WORKSPACE_CONFIG having been set.
        """
        dev = torch.device(self.device)
        y = np.asarray(y_train)
        self._check_deterministic(dev)

        # pos_weight is the ratio class_weight="balanced" applies, computed per
        # fit from the training labels only. a degenerate fold would divide by
        # zero, so it falls back to no reweighting there.
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        pw = n_neg / n_pos if n_pos > 0 and n_neg > 0 else 1.0

        # xavier init of W_V is the single random draw in the model. fork_rng
        # makes it reproducible without disturbing the surrounding RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            net = _AttentionPoolNet(self.E.shape[1], self.hidden,
                                    self.n_series).to(dev)
            head = _LinearHead(self.E.shape[1]).to(dev)

        # the batch is packed once, before the loop, rather than per step
        Ecat, seg = self._pack(acc_train, dev)
        n_acc = len(acc_train)
        yt = torch.as_tensor(y, dtype=torch.float32, device=dev)

        decay, no_decay = [], []
        for m in (net, head):
            for pname, p in m.named_parameters():
                (no_decay if pname.endswith("bias") else decay).append(p)

        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.lr,
        )
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], dtype=torch.float32, device=dev))

        def snapshot():
            return {k: v.detach().clone() for k, v in net.state_dict().items()}

        self.collapsed = False
        self.stopped_at = None
        logs, ckpt = [], snapshot()

        with _deterministic(self.deterministic):
            net.train()
            head.train()
            for step in range(self.steps):
                opt.zero_grad()
                z = net(Ecat, seg, n_acc)                 # (N,L)
                loss = loss_fn(head(z).squeeze(-1), yt)
                loss.backward()
                opt.step()

                # values are kept on the device and moved across once after the
                # loop, so logging does not force a synchronisation per step
                if self.log_every and step % self.log_every == 0:
                    logs.append((step, loss.detach(),
                                 net.W_w.weight.detach().norm()))

                # this one does synchronise, hence not every step. a non-finite
                # loss never recovers, so checking periodically is enough.
                if self.check_every and (step + 1) % self.check_every == 0:
                    if torch.isfinite(loss).item():
                        ckpt = snapshot()
                    else:
                        net.load_state_dict(ckpt)
                        self.collapsed = True
                        self.stopped_at = step
                        break

        if logs:
            self.history = pd.DataFrame({
                "step": [s for s, _, _ in logs],
                "loss": torch.stack([l for _, l, _ in logs]).cpu().numpy(),
                "w_norm": torch.stack([w for _, _, w in logs]).cpu().numpy()})
        else:
            self.history = pd.DataFrame(columns=["step", "loss", "w_norm"])

        net.eval()
        self.net = net          # the head is deliberately not kept
        return self

    # ------------------------------------------------------------------
    # transforming
    # ------------------------------------------------------------------

    def transform(self, acc):
        """Pool the given accessions with the trained attention module.

        Overrides the base implementation so that every accession is pooled in
        one batched call rather than one at a time.

        Args:
            acc (Sequence[str]): Accession identifiers.

        Returns:
            numpy.ndarray: Matrix of shape (len(acc), L), dtype float32.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._require_fitted()
        if len(acc) == 0:
            return np.zeros((0, self.E.shape[1]), dtype=np.float32)
        dev = torch.device(self.device)
        self._check_deterministic(dev)
        with _deterministic(self.deterministic), torch.no_grad():
            Ecat, seg = self._pack(acc, dev)
            z = self.net(Ecat, seg, len(acc))
        return z.cpu().numpy().astype(np.float32)

    def _vec(self, a):
        """Pool a single accession.

        Args:
            a (str): Accession identifier.

        Returns:
            numpy.ndarray: Pooled vector of shape (L,).
        """
        return self.transform([a])[0]

    # ------------------------------------------------------------------
    # what the model learned
    # ------------------------------------------------------------------

    def series_weights(self):
        """Return the learned mixture over imaging series.

        These are the weights each series receives when its pooled vector is
        combined into the accession vector. A uniform result means the model has
        not moved away from its initialisation.

        Returns:
            pandas.Series: Weights indexed by series name, summing to one.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._require_fitted()
        with torch.no_grad():
            g = F.softmax(self.net.beta, dim=0).cpu().numpy()
        idx = [self._inv_series[i] for i in range(self.n_series)]
        return pd.Series(g, index=idx, name="gamma")

    def series_entropy(self):
        """Return the entropy of the series mixture, in nats.

        Returns:
            float: Entropy of the weights from :meth:`series_weights`. The
            maximum is ``log(n_series)``, reached when the mixture is uniform;
            smaller values mean the model concentrates on fewer series.
        """
        w = self.series_weights().to_numpy()
        return float(-(w * np.log(w + 1e-12)).sum())

    def frame_attention(self, a):
        """Return the attention placed on every frame of one accession.

        Args:
            a (str): Accession identifier.

        Returns:
            pandas.DataFrame: One row per frame, with "input_row", "series",
            "alpha" and "weight". "alpha" sums to one within each series;
            "weight" is the product of `alpha` with that series' mixture weight
            and sums to one over the whole accession, which makes it directly
            comparable to the uniform ``1 / n_frames`` of mean pooling.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._require_fitted()
        dev = torch.device(self.device)
        self._check_deterministic(dev)
        with _deterministic(self.deterministic), torch.no_grad():
            Ecat, seg = self._pack([a], dev)
            alpha = self.net.alpha(Ecat, seg, self.n_series).cpu().numpy()
        sid = self._sid[a]
        gamma = self.series_weights().to_numpy()
        return pd.DataFrame({
            "input_row": self._rows[a],
            "series": [self._inv_series[t] for t in sid],
            "alpha": alpha,
            "weight": gamma[sid] * alpha})

    def attention_concentration(self, a):
        """Return how peaked the within-series attention is for one accession.

        Args:
            a (str): Accession identifier.

        Returns:
            pandas.Series: Per series, the entropy of its attention weights
            divided by the entropy of a uniform distribution over the same
            number of frames. One means the attention is uniform, which is mean
            pooling within that series; smaller values mean it concentrates on
            fewer frames.
        """
        fa = self.frame_attention(a)

        def norm_ent(v):
            p = v.to_numpy()
            k = len(p)
            return float(-(p * np.log(p + 1e-12)).sum() / np.log(k)) if k > 1 else 1.0

        return fa.groupby("series")["alpha"].agg(norm_ent).rename("concentration")
