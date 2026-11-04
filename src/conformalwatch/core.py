"""Split-conformal classification and a separate label-free drift diagnostic."""

import math

import numpy as np


def probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or len(p) == 0 or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError("finite nonempty probability matrix required")
    if np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(1), 1, atol=1e-7):
        raise ValueError("rows must be probability distributions")
    return p


def labels(y, n, classes):
    y = np.asarray(y)
    if y.shape != (n,) or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= classes)):
        raise ValueError("integer labels outside class range")
    return y


def threshold(p, y, alpha=0.1):
    p = probabilities(p)
    y = labels(y, len(p), p.shape[1])
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    scores = 1 - p[np.arange(len(p)), y]
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    # The finite-sample rank may exceed the available sample; include every class.
    return float("inf") if rank > len(scores) else float(np.partition(scores, rank - 1)[rank - 1])


def prediction_sets(p, cutoff):
    if np.isnan(cutoff) or cutoff < 0:
        raise ValueError("invalid cutoff")
    return 1 - probabilities(p) <= cutoff


def evaluate(p, y, sets):
    p = probabilities(p)
    y = labels(y, len(p), p.shape[1])
    sets = np.asarray(sets)
    if sets.shape != p.shape or sets.dtype != bool:
        raise ValueError("boolean sets must match probabilities")
    size = sets.sum(1)
    singleton = size == 1
    correct = sets[np.arange(len(p)), y]
    return {
        "coverage": float(correct.mean()),
        "mean_size": float(size.mean()),
        "singleton_fraction": float(singleton.mean()),
        "singleton_error": float((~correct[singleton]).mean()) if singleton.any() else None,
        "accuracy": float((p.argmax(1) == y).mean()),
    }


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(1, keepdims=True)


def synthetic(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) + np.array([shift, shift])
    # P(Y|X) stays fixed, including under covariate shift.
    z = np.column_stack(
        [
            1.4 * x[:, 0],
            -x[:, 0] + 1.5 * x[:, 1],
            -0.6 * x[:, 0] - x[:, 1] + 1.2 * x[:, 0] * x[:, 1],
        ]
    )
    p = softmax(z)
    y = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(1)
    return x, y


def fit(x, y, updates=800):
    features = np.column_stack([np.ones(len(x)), x])
    w = np.zeros((3, 3))
    truth = np.eye(3)[y]
    for _ in range(updates):
        gradient = features.T @ (softmax(features @ w) - truth) / len(x)
        w -= 0.15 * (gradient + 0.001 * w)
    return w


def predict(w, x):
    return softmax(np.column_stack([np.ones(len(x)), x]) @ w)


def mmd_permutation(reference, window, seed, permutations=199, bandwidth=2.0):
    a, b = np.asarray(reference, float), np.asarray(window, float)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[1]
        or min(len(a), len(b)) < 2
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("finite feature matrices with at least two rows required")
    if permutations < 1 or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("positive bandwidth/permutations required")
    x = np.concatenate([a, b])
    distance = ((x[:, None] - x[None, :]) ** 2).sum(-1)
    kernel = np.exp(-distance / (2 * bandwidth**2))
    weights = np.r_[np.full(len(a), 1 / len(a)), np.full(len(b), -1 / len(b))]
    observed = float(weights @ kernel @ weights)  # biased MMD²; valid permutation statistic
    rng = np.random.default_rng(seed)
    permuted = np.array([rng.permutation(weights) for _ in range(permutations)])
    null = np.einsum("bi,ij,bj->b", permuted, kernel, permuted, optimize=True)
    p = (1 + np.count_nonzero(null >= observed - 1e-12)) / (permutations + 1)
    return {"mmd2": observed, "p_value": float(p)}
