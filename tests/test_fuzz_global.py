"""Seeded p-value boundary and input-shape checks for the global tests."""

# The corpus separates valid probability boundaries from malformed input.
# Strict NumPy floating-point checks make hidden warnings reproducible failures.

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp


@pytest.mark.parametrize("seed", [0, 7, 41])
@pytest.mark.parametrize("size", [1, 2, 7, 31])
@pytest.mark.parametrize("boundary", ["interior", "zero", "tiny"])
@pytest.mark.parametrize("form", ["list", "strided", "series"])
def test_seeded_global_probabilities_match_independent_references(seed, size, boundary, form):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.01, 0.99, size=size)
    if boundary == "zero":
        values[0] = 0
    elif boundary == "tiny":
        values[0] = np.nextafter(0.0, 1.0)
    if size > 1:
        values[-1] = 1
    values = values[rng.permutation(size)]
    if form == "list":
        data = values.tolist()
    elif form == "series":
        data = pd.Series(values, index=np.arange(size) % 2)
    else:
        storage = np.full(size * 2, np.nan)
        storage[::2] = values
        data = storage[::2]
    original = np.asarray(data).copy()
    expected_simes = min(size * float(p) / rank for rank, p in enumerate(sorted(values), 1))
    with np.errstate(divide="ignore", under="ignore"):
        expected_fisher = stats.combine_pvalues(values, method="fisher")

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        simes = sp.global_simes_test(data)
        fisher = sp.global_f_test(data, stat=True)

    np.testing.assert_allclose(simes, expected_simes)
    np.testing.assert_allclose(fisher, [expected_fisher.pvalue, expected_fisher.statistic])
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("function", [sp.global_simes_test, sp.global_f_test])
@pytest.mark.parametrize(
    "data",
    [
        [],
        0.5,
        [[0.1, 0.2]],
        [[0.1], [0.2, 0.3]],
        [-0.1, 0.5],
        [0.5, 1.1],
        [0.2, np.inf],
        [0.2, np.nan],
        [0.2, pd.NA],
        pd.Series([0.2, pd.NA], dtype="Float64"),
    ],
)
def test_global_invalid_inputs_raise_guided_value_error(function, data):
    with pytest.raises(ValueError, match="finite p-values"):
        function(data)
