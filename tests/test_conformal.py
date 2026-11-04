import numpy as np
import pytest

from conformalwatch.core import evaluate, mmd_permutation, prediction_sets, threshold


def test_correct_finite_sample_rank_and_ties():
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    # ceil((4+1)*.6)=3; true-label scores .1,.2,.3,.4
    q = threshold(p, np.zeros(4, dtype=int), alpha=0.4)
    assert q == pytest.approx(0.3)
    assert prediction_sets(np.array([[0.7, 0.3]]), q)[0, 0]


def test_small_calibration_uses_infinite_threshold():
    p = np.array([[0.6, 0.4]])
    q = threshold(p, np.array([0]), alpha=0.01)
    assert q == float("inf")
    assert prediction_sets(p, q).all()


def test_empty_and_multiple_sets_are_not_silently_argmaxed():
    p = np.array([[0.5, 0.5], [0.95, 0.05]])
    s = prediction_sets(p, 0.2)
    assert np.array_equal(s.sum(1), [0, 1])
    stats = evaluate(p, np.array([0, 1]), s)
    assert stats["coverage"] == 0 and stats["singleton_error"] == 1
    assert evaluate(p, np.array([0, 1]), prediction_sets(p, 0))["singleton_error"] is None


@pytest.mark.parametrize("p", [[[0.2, 0.2]], [[-1, 2]], [[float("nan"), 1]], []])
def test_rejects_invalid_probabilities(p):
    with pytest.raises(ValueError):
        threshold(p, np.array([0]))


def test_rejects_noninteger_labels_and_bad_alpha():
    with pytest.raises(ValueError):
        threshold([[0.5, 0.5]], np.array([0.5]))
    with pytest.raises(ValueError):
        threshold([[0.5, 0.5]], np.array([0]), 0)


def test_permutation_reproducibility_and_large_shift():
    a = np.random.default_rng(2).normal(size=(30, 2))
    b = a + 8
    r = mmd_permutation(a, b, 4, 99)
    assert r == mmd_permutation(a, b, 4, 99)
    assert r["p_value"] == 0.01
    assert mmd_permutation(a, a, 4, 99)["p_value"] == 1


def test_marginal_coverage_over_independent_trials():
    rng = np.random.default_rng(8)
    cover = []
    for _ in range(150):
        # Fixed uninformative binary classifier; iid labels and probabilities.
        p = rng.dirichlet([2, 2], size=300)
        y = rng.integers(0, 2, 300)
        q = threshold(p[:100], y[:100])
        cover.append(prediction_sets(p[100:], q)[np.arange(200), y[100:]].mean())
    assert 0.88 < np.mean(cover) < 0.94
