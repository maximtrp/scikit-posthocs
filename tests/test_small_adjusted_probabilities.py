from decimal import Decimal, localcontext

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

import scikit_posthocs as sp


def complement_power(p, exponent):
    with localcontext() as ctx:
        ctx.prec = 100
        return float(1 - (1 - Decimal(float(p))) ** Decimal(exponent))


def test_tamhane_preserves_small_welch_probabilities():
    samples = [np.arange(20, dtype=float) + shift for shift in (0, 100, 200)]
    result = sp.posthoc_tamhane(samples).to_numpy()
    expected = np.ones((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            raw = stats.ttest_ind(samples[i], samples[j], equal_var=False).pvalue
            expected[i, j] = expected[j, i] = complement_power(raw, 3)
    assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_duncan_preserves_small_range_probabilities(monkeypatch):
    # Isolate the adjustment from the range distribution's own tail accuracy.
    raw = np.array([1e-20, 1e-30, 1e-40])
    monkeypatch.setattr(stats.studentized_range, "sf", lambda *args: raw)
    result = sp.posthoc_duncan([[1, 2, 3], [5, 6, 7], [9, 10, 11]]).to_numpy()
    expected = [
        complement_power(raw[0], 1),
        complement_power(raw[1], Decimal(1) / 2),
        complement_power(raw[2], 1),
    ]
    assert_allclose(result[np.triu_indices(3, 1)], expected, rtol=1e-12, atol=0)


def test_adjustment_endpoints_do_not_warn():
    with np.errstate(all="raise"):
        for function in (sp.posthoc_tamhane, sp.posthoc_duncan):
            result = function([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
            assert_array_equal(result, np.ones((3, 3)))
