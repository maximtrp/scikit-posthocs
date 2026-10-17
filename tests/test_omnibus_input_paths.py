import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp
from tests._input_cases import LABELS, case, independent_samples, long_data

OMNIBUS_CASES = [
    case(sp.test_mackwolfe, p=1),
    case(sp.test_mackwolfe, n_perm=100),
    case(sp.test_osrt),
    case(sp.test_jonckheere),
    case(sp.test_jonckheere, alternative="greater", continuity=True),
    case(sp.test_jonckheere, alternative="less"),
    case(sp.test_hartley, n_perm=100),
    case(sp.test_median),
    case(sp.test_median, correction=True),
]


@pytest.mark.parametrize(("function", "kwargs"), OMNIBUS_CASES)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["long", "nullable", "repeated_index"])
def test_independent_omnibus_input_forms_preserve_hypothesis_order(
    function, kwargs, sort, input_type
):
    samples = independent_samples()
    data = long_data(samples, dtype="Float64" if input_type == "nullable" else "float64")
    if input_type == "repeated_index":
        data.index = np.tile(np.arange(5), 3)
    data["irrelevant"] = np.nan
    original = data.copy(deep=True)
    order = np.argsort(LABELS) if sort else np.arange(len(LABELS))
    # Ordered omnibus tests use the order of the groups as part of the hypothesis.
    state = np.random.get_state()
    try:
        np.random.seed(271)
        expected = function([samples[i] for i in order], sort=False, **kwargs)
        np.random.seed(271)
        actual = function(data, val_col="value", group_col="group", sort=sort, **kwargs)
    finally:
        np.random.set_state(state)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(data, original)


def jonckheere_reference(samples, alternative, continuity):
    sizes = np.array(list(map(len, samples)), dtype=float)
    n = sizes.sum()
    # Enumerate comparisons directly; these samples have no ties.
    count = sum(
        x < y
        for i, j in itertools.combinations(range(len(samples)), 2)
        for x in samples[i]
        for y in samples[j]
    )
    centered = count - (n**2 - np.sum(sizes**2)) / 4
    variance = (n**2 * (2 * n + 3) - np.sum(sizes**2 * (2 * sizes + 3))) / 72
    if continuity:
        centered -= np.sign(centered) / 2
    statistic = centered / np.sqrt(variance)
    if alternative == "two-sided":
        pvalue = 2 * stats.norm.sf(abs(statistic))
    elif alternative == "greater":
        pvalue = stats.norm.sf(statistic)
    else:
        pvalue = stats.norm.cdf(statistic)
    return pvalue, statistic


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("continuity", [False, True])
def test_jonckheere_unequal_groups_match_enumerated_pair_counts(sort, alternative, continuity):
    samples = [np.arange(5) * 3.0, np.arange(7) * 3.0 + 1, np.arange(9) * 3.0 + 2]
    data = long_data(samples, dtype="Float64")
    order = np.argsort(LABELS) if sort else np.arange(len(LABELS))
    expected = jonckheere_reference([samples[i] for i in order], alternative, continuity)

    actual = sp.test_jonckheere(
        data,
        val_col="value",
        group_col="group",
        sort=sort,
        alternative=alternative,
        continuity=continuity,
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("peak", [0, 2])
@pytest.mark.parametrize("sort", [False, True])
def test_mackwolfe_endpoint_peak_matches_enumerated_ordered_alternative(peak, sort):
    samples = [np.arange(5) * 3.0, np.arange(7) * 3.0 + 1, np.arange(9) * 3.0 + 2]
    data = long_data(samples)
    order = np.argsort(LABELS) if sort else np.arange(len(LABELS))
    ordered_samples = [samples[i] for i in order]
    if peak == 0:
        ordered_samples.reverse()
    expected = jonckheere_reference(ordered_samples, "greater", False)

    actual = sp.test_mackwolfe(data, val_col="value", group_col="group", p=peak, sort=sort)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("correction", [False, True])
@pytest.mark.parametrize("sort", [False, True])
def test_omnibus_median_matches_scipy_contingency_table(correction, sort):
    samples = independent_samples()
    expected = stats.median_test(*samples, correction=correction)
    data = long_data(samples, dtype="Float64")

    actual = sp.test_median(
        data, val_col="value", group_col="group", correction=correction, sort=sort
    )

    np.testing.assert_allclose(actual, [expected.pvalue, expected.statistic, len(samples) - 1])


def test_mackwolfe_permutation_ties_have_unit_pvalue():
    pvalue, _ = sp.test_mackwolfe([[1, 1], [1, 1], [1, 1]], n_perm=19)
    np.testing.assert_equal(pvalue, 1.0)


def test_mackwolfe_permutation_tail_counts_equal_statistics(monkeypatch):
    # For two groups of two, the umbrella statistic is the maximum of the
    # standardized increasing and decreasing pair counts. The first shuffle
    # ties the observation, the second is more extreme, and the third is less.
    draws = iter([[1, 3, 2, 4], [1, 2, 3, 4], [1, 4, 2, 3]])
    monkeypatch.setattr(np.random, "permutation", lambda values: np.array(next(draws)))
    pvalue, _ = sp.test_mackwolfe([[1, 3], [2, 4]], n_perm=3)
    np.testing.assert_equal(pvalue, 3 / 4)


def test_mackwolfe_permutation_pvalue_includes_observed_order(monkeypatch):
    monkeypatch.setattr(np.random, "permutation", lambda values: np.array([1, 4, 2, 3]))
    pvalue, _ = sp.test_mackwolfe([[1, 2], [3, 4]], n_perm=9)
    np.testing.assert_equal(pvalue, 1 / 10)


def test_mackwolfe_permutation_tail_counts_numerically_close_statistics(monkeypatch):
    samples = [[1, 5, 8, 12], [10, 9, 3, 7], [6, 11, 4, 2, 14, 13]]
    draw = np.array([5, 11, 3, 9, 10, 8, 13, 4, 7, 14, 6, 12, 2, 1])
    # The observed maximum is at an endpoint; the shuffled maximum is at the
    # middle peak. Both equal sqrt(1/2), but different variances round them apart.
    observed_statistic = 6 / np.sqrt(72)
    permuted_statistic = 5 / np.sqrt(50)
    np.testing.assert_array_less(permuted_statistic, observed_statistic)
    np.testing.assert_allclose(permuted_statistic, observed_statistic, rtol=1e-15, atol=0)
    monkeypatch.setattr(np.random, "permutation", lambda values: draw.copy())

    pvalue, statistic = sp.test_mackwolfe(samples, n_perm=9)

    np.testing.assert_equal(statistic, observed_statistic)
    np.testing.assert_equal(pvalue, 1.0)


@pytest.mark.parametrize("p", [None, 0])
@pytest.mark.parametrize("input_type", ["list", "long"])
@pytest.mark.parametrize(
    "samples",
    [[[1, 2, 3]], [[1, 2], [np.nan, np.nan]], [[np.nan], [np.nan]]],
    ids=["single-group", "one-complete-group", "all-missing"],
)
def test_mackwolfe_requires_two_nonmissing_groups(samples, input_type, p):
    data = samples
    kwargs = {}
    if input_type == "long":
        data = pd.DataFrame(
            [(value, group) for group, sample in enumerate(samples) for value in sample],
            columns=["value", "group"],
        )
        kwargs = {"val_col": "value", "group_col": "group"}

    with pytest.raises(ValueError, match="at least two groups with nonmissing observations"):
        sp.test_mackwolfe(data, p=p, n_perm=9, **kwargs)


@pytest.mark.parametrize("peak", [0, 2])
@pytest.mark.parametrize("n_perm", [0, -1])
def test_mackwolfe_known_peak_ignores_permutation_count(monkeypatch, peak, n_perm):
    samples = [np.arange(5) * 3.0, np.arange(7) * 3.0 + 1, np.arange(9) * 3.0 + 2]
    ordered_samples = samples[::-1] if peak == 0 else samples
    expected = jonckheere_reference(ordered_samples, "greater", False)

    def unexpected_permutation(values):
        pytest.fail("The known-peak approximation must not draw permutations")

    monkeypatch.setattr(np.random, "permutation", unexpected_permutation)
    actual = sp.test_mackwolfe(samples, p=peak, n_perm=n_perm)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("n_perm", [0, -1])
def test_mackwolfe_unknown_peak_requires_positive_permutations(n_perm):
    with pytest.raises(ValueError, match="n_perm must be a positive integer"):
        sp.test_mackwolfe([[1, 2], [3, 4]], n_perm=n_perm)
