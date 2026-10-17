import numpy as np
import pytest

import scikit_posthocs as sp

PERMUTATION_TESTS = [
    (sp.posthoc_npm_test, "nperm"),
    (sp.test_hartley, "n_perm"),
    (sp.test_mackwolfe, "n_perm"),
]
SAMPLES = [[1, 2, 4, 7], [2, 4, 6, 9], [3, 5, 8, 10]]


@pytest.mark.parametrize(("function", "parameter"), PERMUTATION_TESTS)
@pytest.mark.parametrize(
    "count",
    [0, -1, np.int64(-3), 1.5, 2.0, np.nan, np.inf, True, False, np.bool_(True), None, "10"],
)
def test_invalid_permutation_counts_fail_before_sampling(function, parameter, count, monkeypatch):
    def unexpected_sample(*args, **kwargs):
        raise AssertionError("Invalid counts must be rejected before sampling")

    monkeypatch.setattr(np.random, "permutation", unexpected_sample)
    monkeypatch.setattr("scipy.stats.chi2.rvs", unexpected_sample)
    with (
        np.errstate(all="raise"),
        pytest.raises(ValueError, match=f"{parameter} must be a positive integer"),
    ):
        function(SAMPLES, **{parameter: count})


@pytest.mark.parametrize(("function", "parameter"), PERMUTATION_TESTS)
@pytest.mark.parametrize("count", [1, 3, np.int32(3), np.int64(3), np.uint64(3)])
def test_integer_permutation_counts_match_python_int(function, parameter, count):
    state = np.random.get_state()
    try:
        np.random.seed(491)
        expected = function(SAMPLES, **{parameter: int(count)})
        np.random.seed(491)
        actual = function(SAMPLES, **{parameter: count})
    finally:
        np.random.set_state(state)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("count", [0, -1, 1.5, None, True])
def test_known_mackwolfe_peak_does_not_validate_unused_count(count):
    expected = sp.test_mackwolfe(SAMPLES, p=1)
    actual = sp.test_mackwolfe(SAMPLES, p=1, n_perm=count)
    np.testing.assert_equal(actual, expected)
