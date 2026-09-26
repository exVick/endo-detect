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
              clf_kw={"hidden": 32, "steps": 600})

`C` keeps its meaning for both. For the logistic probe it is the inverse L2
strength sklearn expects. For the perceptron it is mapped onto the weight decay
of the optimiser, so a sweep over :data:`~.evaluate.C_GRID` remains a
regularisation sweep and the plots and paired tests that read the "C" column go
on working unchanged.

Module constants:
    CLASSIFIERS: The classifier names accepted by :func:`make_classifier`.
    ACTIVATIONS: The activation names accepted by :class:`MLP`.
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
ACTIVATIONS = ("tanh", "relu", "gelu")


# --------------------------------------------------------------------------
# logistic probe
# --------------------------------------------------------------------------

class LogReg:
    """Standardisation followed by an L2-penalised logistic regression.

    This is the classifier every result in the project was produced with, kept
    unchanged so that earlier experiments reproduce exactly.

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
    """Two-layer perceptron producing a single logit.

    The output layer is initialised to zero, so at initialisation the network
    returns zero for every row. That is exactly the chance classifier, which is
    the honest place for a permutation null to start from: a randomly
    initialised output layer would instead start the model at an arbitrary
    non-chance decision boundary. The first optimiser step therefore moves only
    the output layer, since the gradient reaching the hidden layer is zero while
    the output weights are; from the second step onwards both layers learn.

    The hidden layer is initialised to match its activation: Xavier for the
    bounded `tanh`, He for the unbounded rectifiers.

    Attributes:
        fc1 (torch.nn.Linear): Hidden layer.
        fc2 (torch.nn.Linear): Output layer, initialised to zero.
    """

    def __init__(self, d_in, hidden, activation, dropout):
        """Build the layers and initialise them.

        Args:
            d_in (int): Width of the standardised input.
            hidden (int): Width of the hidden layer.
            activation (str): One of :data:`ACTIVATIONS`.
            dropout (float): Dropout probability applied to the hidden
                activations. Zero disables it entirely.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = {"tanh": torch.tanh,
                    "relu": torch.relu,
                    "gelu": nn.functional.gelu}[activation]
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if activation == "tanh":
            # xavier keeps the variance of the activations stable through a
            # saturating nonlinearity, which he init would overshoot
            nn.init.xavier_normal_(self.fc1.weight)
        else:
            # gelu is close enough to relu for the same gain to apply
            nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in",
                                    nonlinearity="relu")
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
        return self.fc2(self.drop(self.act(self.fc1(x)))).squeeze(-1)


class MLP:
    """Standardisation followed by a seeded two-layer perceptron.

    The input is whatever a representation produced, standardised, so it is
    roughly zero-mean and unit-variance with a width in the hundreds or
    thousands and only a few hundred rows. `tanh` is the default activation for
    that regime: it is bounded, which regularises implicitly where the rows are
    few, and it cannot produce the dead units a rectifier can under full-batch
    training, which would otherwise make the result unnecessarily sensitive to
    the seed.

    Training is full batch, so no shuffling is involved and the only randomness
    is the initialisation, and optionally dropout. Both are drawn under a fixed
    seed inside a forked RNG, leaving the surrounding random state untouched.

    The seed is held fixed across every fit rather than derived from the split.
    Differences between splits then reflect the data rather than the draw, and
    in a label permutation test the null varies only with the labels. Reseeding
    per draw would fold initialisation noise into the null and inflate the
    p-value.

    Attributes:
        name (str): Identifier written into every result row.
        meta (dict): Hyperparameters emitted as extra result columns, prefixed
            "clf_" so they never collide with a representation's own `meta`.
            Group by them to compare settings, for example
            ``summarise_inner(inner, by=("representation", "condition",
            "clf_steps"))``.
    """

    name = "mlp"

    def __init__(self, C, hidden=16, steps=300, lr=1e-3, activation="tanh",
                 dropout=0.0, weight_decay=None, seed=0, device="cpu",
                 deterministic=True, threads=1):
        """Record the hyperparameters. Nothing is fitted yet.

        Args:
            C (float): Inverse regularisation strength, mapped onto the weight
                decay of the optimiser as described under `weight_decay`.
            hidden (int): Width of the hidden layer. Kept small by default,
                since the number of accessions is in the hundreds.
            steps (int): Number of full-batch optimiser steps per fit.
            lr (float): AdamW learning rate.
            activation (str): One of :data:`ACTIVATIONS`.
            dropout (float): Dropout probability on the hidden activations.
            weight_decay (float, optional): AdamW weight decay, applied to
                non-bias parameters. When None it is derived from `C` as
                ``1 / (C * n_train)``. That is the value which matches what
                sklearn's `C` does: its objective is ``0.5||w||^2 + C * sum of
                losses``, which is ``mean loss + ||w||^2 / (2 * C * n)`` once
                divided through, so the two penalise the weights comparably and
                a sweep over `C` means the same thing for both classifiers.
                Passing a value here overrides that mapping, after which `C` is
                still recorded but no longer affects the fit.
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

        Raises:
            ValueError: If `activation` is not one of :data:`ACTIVATIONS`.
        """
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; expected one "
                             f"of {list(ACTIVATIONS)}")
        self.C = C
        self.hidden = hidden
        self.steps = steps
        self.lr = lr
        self.activation = activation
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.deterministic = deterministic
        self.threads = threads

        self.meta = {"clf_hidden": hidden, "clf_steps": steps, "clf_lr": lr,
                     "clf_activation": activation, "clf_dropout": dropout,
                     "clf_seed": seed}
        if weight_decay is not None:
            self.meta["clf_weight_decay"] = weight_decay

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

    def fit(self, X, y):
        """Train the perceptron on one training set.

        Class imbalance is handled with a positive-class weight computed from
        `y` alone, which is the ratio ``class_weight="balanced"`` gives the
        logistic probe. Both classifiers therefore weight the classes the same
        way, and zero stays the balanced threshold.

        Args:
            X (numpy.ndarray): Training features of shape (n_train, D).
            y (numpy.ndarray): Binary training labels.

        Returns:
            MLP: self, with `scaler_` and `net_` populated.

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

        wd = (self.weight_decay if self.weight_decay is not None
              else 1.0 / (float(self.C) * max(len(y), 1)))

        Xt = torch.as_tensor(Xs, device=dev)
        yt = torch.as_tensor(y, dtype=torch.float32, device=dev)

        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            # the fork covers the whole fit, not just the initialisation, so
            # that dropout draws are reproducible too and the surrounding RNG
            # state is left exactly as it was found
            with torch.random.fork_rng(devices=self._fork_devices(dev)):
                torch.manual_seed(self.seed)
                net = _MLPNet(Xs.shape[1], self.hidden, self.activation,
                              self.dropout).to(dev)

                # biases are excluded from weight decay, matching AttentionPool
                decay, no_decay = [], []
                for pname, p in net.named_parameters():
                    (no_decay if pname.endswith("bias") else decay).append(p)
                opt = torch.optim.AdamW(
                    [{"params": decay, "weight_decay": wd},
                     {"params": no_decay, "weight_decay": 0.0}],
                    lr=self.lr)
                loss_fn = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pw], dtype=torch.float32,
                                            device=dev))

                with _deterministic(self.deterministic):
                    net.train()
                    for _ in range(self.steps):
                        opt.zero_grad()
                        loss = loss_fn(net(Xt), yt)
                        loss.backward()
                        opt.step()
            net.eval()
            self.net_ = net
        finally:
            torch.set_num_threads(prev_threads)
        return self

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
        C (float): Inverse regularisation strength, interpreted by whichever
            classifier is selected.
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
        return cls(C=C, **kw)
    except TypeError as e:
        raise ValueError(f"bad clf_kw for classifier {classifier!r}: {e}") from e
