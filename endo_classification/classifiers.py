"""Classifiers applied on top of a representation.

Everything in :mod:`evaluate` reaches its classifier through two methods only::

    clf.fit(X, y)             fit on the training features
    clf.decision_function(X)  one score per row, larger meaning more positive

Anything providing that pair can be plugged in, which is what allows the
logistic probe and the perceptron below to be swapped without touching a single
evaluation function. Both standardise their input first, so a representation
never has to know which classifier will consume it.

The score returned by :meth:`decision_function` is a raw logit in both cases,
never a probability. :func:`evaluate._metric_value` thresholds it at zero for
balanced accuracy, so the classifiers weight the classes such that zero is the
balanced operating point: the logistic probe through ``class_weight="balanced"``
and the perceptron through a matching ``pos_weight`` on its loss.

Selecting a classifier
----------------------

Every entry point of :mod:`evaluate` takes ``classifier`` and ``clf_kw``::

    run_inner(reps, labels, splits)                       # logistic, unchanged
    run_inner(reps, labels, splits, classifier="mlp")
    run_inner(reps, labels, splits, classifier="mlp",
              clf_kw={"hidden": 32, "steps": 600, "weight_decay": 1e-2})

`C` is the logistic probe's inverse L2 strength and is ignored by the
perceptron, which is regularised through its own `weight_decay`, passed in
`clf_kw`.

Module constants:
    CLASSIFIERS: The classifier names accepted by :func:`make_classifier`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ._torch_utils import _check_deterministic, _deterministic

CLASSIFIERS = ("logreg", "mlp")


# --------------------------------------------------------------------------
# logistic probe
# --------------------------------------------------------------------------

class LogReg:
    """Standardisation followed by an L2-penalised logistic regression.

    Attributes:
        name (str): Identifier written into every result row.
        meta (dict): Extra result columns. Empty, since `C` is already recorded
            as a column in its own right.
    """

    name = "logreg"

    def __init__(self, C, max_iter=5000):
        """Record the hyperparameters. Nothing is fitted yet.

        Args:
            C (float): Inverse L2 regularisation strength. Smaller values shrink
                the coefficients more.
            max_iter (int): Iteration cap handed to the lbfgs solver.
        """
        self.C = C
        self.max_iter = max_iter
        self.meta = {}
        self.pipe_ = None

    def fit(self, X, y):
        """Fit the pipeline on one training set.

        Args:
            X (numpy.ndarray): Training features of shape (n_train, D).
            y (numpy.ndarray): Binary training labels.

        Returns:
            LogReg: self.
        """
        self.pipe_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, l1_ratio=0, solver="lbfgs",
                               class_weight="balanced",
                               max_iter=self.max_iter)).fit(X, y)
        return self

    def decision_function(self, X):
        """Score rows with the fitted pipeline.

        Args:
            X (numpy.ndarray): Features of shape (n, D).

        Returns:
            numpy.ndarray: One logit per row, larger meaning more positive.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.pipe_ is None:
            raise RuntimeError("LogReg.fit() not called")
        return self.pipe_.decision_function(X)


# --------------------------------------------------------------------------
# perceptron
# --------------------------------------------------------------------------

class _MLPNet(nn.Module):
    """Two-layer `tanh` perceptron producing a single logit.

    The output layer is initialised to zero, so at initialisation the network
    returns zero for every row. That is exactly the chance classifier, which is
    the honest place for a permutation null to start from: a randomly
    initialised output layer would instead start the model at an arbitrary
    non-chance decision boundary. The first optimiser step therefore moves only
    the output layer, since the gradient reaching the hidden layer is zero while
    the output weights are; from the second step onwards both layers learn.

    The hidden layer uses Xavier initialisation, which suits the bounded
    `tanh`.

    Attributes:
        fc1 (torch.nn.Linear): Hidden layer.
        fc2 (torch.nn.Linear): Output layer, initialised to zero.
    """

    def __init__(self, d_in, hidden, dropout):
        """Build the layers and initialise them.

        Args:
            d_in (int): Width of the standardised input.
            hidden (int): Width of the hidden layer.
            dropout (float): Dropout probability applied to the hidden
                activations. Zero disables it entirely.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # xavier keeps the variance of the activations stable through a
        # saturating nonlinearity, which he init would overshoot
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)   # start at the chance classifier
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """Map standardised features to one logit per row.

        Args:
            x (torch.Tensor): Standardised features of shape (n, d_in).

        Returns:
            torch.Tensor: Logits of shape (n,).
        """
        return self.fc2(self.drop(torch.tanh(self.fc1(x)))).squeeze(-1)


class MLP:
    """Standardisation followed by a seeded two-layer perceptron.

    The input is whatever a representation produced, standardised, so it is
    roughly zero-mean and unit-variance with a width in the hundreds or
    thousands and only a few hundred rows. The activation is fixed to `tanh`,
    which suits that regime: it is bounded, which regularises implicitly where
    the rows are few, and it cannot produce the dead units a rectifier can under
    full-batch training, which would otherwise make the result unnecessarily
    sensitive to the seed.

    Training is full batch, so no shuffling is involved and the only randomness
    is the initialisation, and optionally dropout. Both are drawn under a fixed
    seed inside a forked RNG, leaving the surrounding random state untouched.

    The seed is held fixed across every fit rather than derived from the split.
    Differences between splits then reflect the data rather than the draw, and
    in a label permutation test the null varies only with the labels. Reseeding
    per draw would fold initialisation noise into the null and inflate the
    p-value.

    Several step counts can be evaluated from a single run. Passing a sequence
    as `steps` trains once up to the largest value and, through
    :meth:`fit_path`, exposes the model at every smaller value on the way. The
    optimiser state and the RNG state are carried across those pauses, so the
    model at each checkpoint is bit-for-bit the one a separate fit with that
    many steps would produce.

    Attributes:
        name (str): Identifier written into every result row.
        meta (dict): Hyperparameters emitted as extra result columns, prefixed
            "clf_" so they never collide with a representation's own `meta`.
            Group by them to compare settings, for example
            ``summarise_inner(inner, by=("representation", "condition",
            "clf_steps"))``. "clf_steps" holds the largest step count, which
            is what :meth:`fit` trains for; :mod:`evaluate` overwrites it per
            checkpoint.
        checkpoints (tuple[int]): The step counts in `steps`, sorted and
            deduplicated.
    """

    name = "mlp"

    def __init__(self, hidden=16, steps=300, lr=1e-3, dropout=0.0,
                 weight_decay=1e-2, seed=0, device="cpu", deterministic=True,
                 threads=1):
        """Record the hyperparameters. Nothing is fitted yet.

        Args:
            hidden (int): Width of the hidden layer. Kept small by default,
                since the number of accessions is in the hundreds.
            steps (int or Sequence[int]): Number of full-batch optimiser
                steps per fit. A sequence, for example ``(800, 1400, 2000)``,
                trains once up to its largest value and lets :meth:`fit_path`
                score the model at every value on the way.
            lr (float): AdamW learning rate.
            dropout (float): Dropout probability on the hidden activations.
            weight_decay (float): AdamW decoupled weight decay, applied to the
                non-bias parameters only; biases are excluded. PyTorch scales
                it by the learning rate, so each step shrinks the weights by a
                factor of ``(1 - lr * weight_decay)``.
            seed (int): Seed for the initialisation and for dropout. Held fixed
                across every fit.
            device (str): Device to train on, for example "cpu" or "cuda". The
                default is deliberately "cpu": a permutation test dispatches
                thousands of fits across joblib workers, which would contend for
                a single device.
            deterministic (bool): Whether to force deterministic kernels. On
                CUDA this additionally requires CUBLAS_WORKSPACE_CONFIG to have
                been set before torch was imported.
            threads (int): Torch thread count to use during a fit, restored
                afterwards. One by default, so that parallel workers do not each
                spawn a full thread pool and thrash.
        """
        self.hidden = hidden
        self.checkpoints = tuple(sorted({int(s) for s in np.atleast_1d(steps)}))
        self.steps = self.checkpoints[-1]
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.deterministic = deterministic
        self.threads = threads

        self.meta = {"clf_hidden": hidden, "clf_steps": self.steps,
                     "clf_lr": lr,
                     "clf_dropout": dropout, "clf_weight_decay": weight_decay,
                     "clf_seed": seed}

        self.net_ = None
        self.scaler_ = None

    def _fork_devices(self, dev):
        """List the devices whose RNG has to be forked for reproducibility.

        Args:
            dev (torch.device): Device the work will run on.

        Returns:
            list: The CUDA device index when running on CUDA, else empty. The
            CPU generator is always forked by ``fork_rng`` itself.
        """
        if dev.type != "cuda":
            return []
        return [dev.index if dev.index is not None else torch.cuda.current_device()]

    def _rng_state(self, dev):
        """Capture the RNG state that training draws from.

        Args:
            dev (torch.device): Device the work runs on.

        Returns:
            tuple: The CPU generator state and the states of the devices listed
            by :meth:`_fork_devices`.
        """
        return (torch.get_rng_state(),
                [torch.cuda.get_rng_state(d) for d in self._fork_devices(dev)])

    def _set_rng_state(self, dev, state):
        """Restore a state captured by :meth:`_rng_state`.

        Args:
            dev (torch.device): Device the work runs on.
            state (tuple): Output of :meth:`_rng_state`.
        """
        cpu, cuda = state
        torch.set_rng_state(cpu)
        for d, st in zip(self._fork_devices(dev), cuda):
            torch.cuda.set_rng_state(st, d)

    def _setup(self, X, y):
        """Standardise, initialise the network and build the optimiser.

        Nothing is trained yet. The training state is kept on the instance so
        that :meth:`_advance` can continue from it, and the RNG state right
        after initialisation is recorded, which is exactly where a one-shot fit
        would take its first step.

        Class imbalance is handled with a positive-class weight computed from
        `y` alone, which is the ratio ``class_weight="balanced"`` gives the
        logistic probe. Both classifiers therefore weight the classes the same
        way, and zero stays the balanced threshold.

        Args:
            X (numpy.ndarray): Training features of shape (n_train, D).
            y (numpy.ndarray): Binary training labels.

        Raises:
            RuntimeError: If deterministic execution is requested on CUDA
                without CUBLAS_WORKSPACE_CONFIG having been set.
        """
        y = np.asarray(y)
        dev = torch.device(self.device)
        _check_deterministic(self.deterministic, dev)

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)

        # a degenerate fold would divide by zero, so it falls back to no
        # reweighting there, exactly as AttentionPool does
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        pw = n_neg / n_pos if n_pos > 0 and n_neg > 0 else 1.0

        wd = self.weight_decay

        self.Xt_ = torch.as_tensor(Xs, device=dev)
        self.yt_ = torch.as_tensor(y, dtype=torch.float32, device=dev)

        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            # forked so that the surrounding RNG state is left exactly as it
            # was found; the state reached inside is carried to _advance
            with torch.random.fork_rng(devices=self._fork_devices(dev)):
                torch.manual_seed(self.seed)
                net = _MLPNet(Xs.shape[1], self.hidden, self.dropout).to(dev)

                # biases are excluded from weight decay, matching AttentionPool
                decay, no_decay = [], []
                for pname, p in net.named_parameters():
                    (no_decay if pname.endswith("bias") else decay).append(p)
                self.opt_ = torch.optim.AdamW(
                    [{"params": decay, "weight_decay": wd},
                     {"params": no_decay, "weight_decay": 0.0}],
                    lr=self.lr)
                self.loss_fn_ = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pw], dtype=torch.float32,
                                            device=dev))
                self.rng_ = self._rng_state(dev)
        finally:
            torch.set_num_threads(prev_threads)
        net.eval()
        self.net_ = net
        self.step_ = 0

    def _advance(self, target):
        """Continue training from the current step up to `target`.

        The optimiser keeps its moments and step count between calls, and the
        RNG is resumed from where the previous call left it, so dropout draws
        continue the same stream. Training in several calls therefore gives
        exactly the model a single call to the final step would.

        Args:
            target (int): Total number of steps to have been taken on return.
        """
        dev = torch.device(self.device)
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            with torch.random.fork_rng(devices=self._fork_devices(dev)):
                self._set_rng_state(dev, self.rng_)
                with _deterministic(self.deterministic):
                    self.net_.train()
                    for _ in range(target - self.step_):
                        self.opt_.zero_grad()
                        loss = self.loss_fn_(self.net_(self.Xt_), self.yt_)
                        loss.backward()
                        self.opt_.step()
                self.rng_ = self._rng_state(dev)
            self.net_.eval()
            self.step_ = target
        finally:
            torch.set_num_threads(prev_threads)

    def _release(self):
        """Drop the training state, keeping only what scoring needs."""
        self.Xt_ = self.yt_ = self.opt_ = self.loss_fn_ = self.rng_ = None

    def fit(self, X, y):
        """Train the perceptron on one training set for `steps` steps.

        When `steps` was given as a sequence this trains up to its largest
        value; use :meth:`fit_path` to score the intermediate ones.

        Args:
            X (numpy.ndarray): Training features of shape (n_train, D).
            y (numpy.ndarray): Binary training labels.

        Returns:
            MLP: self, with `scaler_` and `net_` populated.

        Raises:
            RuntimeError: If deterministic execution is requested on CUDA
                without CUBLAS_WORKSPACE_CONFIG having been set.
        """
        self._setup(X, y)
        self._advance(self.steps)
        self._release()
        return self

    def fit_path(self, X, y):
        """Train once up to the largest checkpoint, pausing at every one.

        Typical use scores held-out rows at every checkpoint::

            for step, m in MLP(steps=(800, 1400, 2000)).fit_path(Xtr, ytr):
                scores[step] = m.decision_function(Xte)

        Scoring between checkpoints draws no random numbers and does not touch
        the training state, so it leaves the remaining path unchanged.

        Args:
            X (numpy.ndarray): Training features of shape (n_train, D).
            y (numpy.ndarray): Binary training labels.

        Yields:
            tuple: ``(step, self)`` at each entry of `checkpoints`, in
            increasing order, with `net_` holding the model after `step` steps.

        Raises:
            RuntimeError: If deterministic execution is requested on CUDA
                without CUBLAS_WORKSPACE_CONFIG having been set.
        """
        self._setup(X, y)
        try:
            for step in self.checkpoints:
                self._advance(step)
                yield step, self
        finally:
            self._release()

    def decision_function(self, X):
        """Score rows with the fitted perceptron.

        Args:
            X (numpy.ndarray): Features of shape (n, D).

        Returns:
            numpy.ndarray: One logit per row, larger meaning more positive.
            Zero is the balanced threshold, so balanced accuracy can be read off
            it the same way as for the logistic probe.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.net_ is None:
            raise RuntimeError("MLP.fit() not called")
        dev = torch.device(self.device)
        Xs = self.scaler_.transform(X).astype(np.float32)
        with _deterministic(self.deterministic), torch.no_grad():
            s = self.net_(torch.as_tensor(Xs, device=dev))
        return s.detach().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def make_classifier(C, classifier="logreg", clf_kw=None):
    """Build the classifier every function in :mod:`evaluate` fits.

    Args:
        C (float): Inverse L2 regularisation strength. Used by "logreg" only
            and ignored for "mlp", which takes `weight_decay` in `clf_kw`.
        classifier (str): One of :data:`CLASSIFIERS`.
        clf_kw (dict, optional): Extra hyperparameters forwarded to the chosen
            classifier, for example ``{"steps": 600, "hidden": 32}``. Ignored
            keys are not silently dropped: an unknown one raises.

    Returns:
        LogReg or MLP: An unfitted classifier exposing `fit`,
        `decision_function`, `name` and `meta`.

    Raises:
        ValueError: If `classifier` is not one of :data:`CLASSIFIERS`, or if
            `clf_kw` holds a key the classifier does not accept.
    """
    kw = dict(clf_kw or {})
    if classifier not in CLASSIFIERS:
        raise ValueError(f"unknown classifier {classifier!r}; expected one of "
                         f"{list(CLASSIFIERS)}")
    cls = {"logreg": LogReg, "mlp": MLP}[classifier]
    try:
        return cls(C=C, **kw) if classifier == "logreg" else cls(**kw)
    except TypeError as e:
        raise ValueError(f"bad clf_kw for classifier {classifier!r}: {e}") from e
