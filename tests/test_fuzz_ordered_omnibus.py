"""Small exact-null oracles and rank-order properties for ordered tests."""

# Seeds and input forms are pytest parameters, so failures reproduce by node ID.
# Enumerating label assignments supplies Jonckheere's null moments independently
# of the production variance formula. Page's reference uses SciPy's L statistic
# plus the specified normal continuity correction.

from functools import cache
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp


def pair_score(samples):
    return sum(
        int(first < second) + 0.5 * int(first == second)
        for i, j in combinations(range(len(samples)), 2)
        for first in samples[i]
        for second in samples[j]
    )


def allocations(remaining, sizes):
    if len(sizes) == 1:
        yield (remaining,)
        return
    for chosen in combinations(remaining, sizes[0]):
        rest = tuple(i for i in remaining if i not in chosen)
        for tail in allocations(rest, sizes[1:]):
            yield (chosen, *tail)


@cache
def exact_null_moments(samples):
    values = tuple(value for sample in samples for value in sample)
    sizes = tuple(map(len, samples))
    scores = [
        pair_score([[values[index] for index in group] for group in allocation])
        for allocation in allocations(tuple(range(len(values))), sizes)
    ]
    return float(np.mean(scores)), float(np.var(scores))


@pytest.mark.parametrize("seed", [0, 7, 41])
@pytest.mark.parametrize("sizes", [(1, 1), (1, 2), (2, 2, 2), (2, 3, 2)])
@pytest.mark.parametrize("form", ["long", "nullable", "categorical"])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("continuity", [False, True])
@pytest.mark.filterwarnings("ignore:Ties are present:UserWarning")
def test_jonckheere_seeded_ties_match_exact_null_moments(
    seed, sizes, form, sort, alternative, continuity
):
    rng = np.random.default_rng(seed)
    samples = [rng.integers(-2, 3, size=size).tolist() for size in sizes]
    labels = ["z", "a", "m"][: len(samples)]
    data = pd.DataFrame(
        [(value, label) for label, sample in zip(labels, samples) for value in sample],
        columns=["value", "group"],
    )
    if form == "nullable":
        data["value"] = data["value"].astype("Float64")
    elif form == "categorical":
        data["group"] = pd.Categorical(data["group"], categories=[*labels, "unused"])
    data.index = np.arange(len(data)) % 3
    data["ignored"] = np.nan
    original = data.copy(deep=True)
    order = np.argsort(labels) if sort and form != "categorical" else np.arange(len(labels))
    ordered = tuple(tuple(samples[index]) for index in order)
    mean, variance = exact_null_moments(ordered)
    centered = pair_score(ordered) - mean
    if continuity:
        centered -= np.sign(centered) * 0.5
    z = centered / np.sqrt(variance) if variance else 0.0
    pvalue = (
        2 * stats.norm.sf(abs(z))
        if alternative == "two-sided"
        else stats.norm.sf(z)
        if alternative == "greater"
        else stats.norm.cdf(z)
    )
    if not variance:
        pvalue = 1.0

    with np.errstate(all="raise"):
        actual = sp.test_jonckheere(
            data,
            val_col="value",
            group_col="group",
            sort=sort,
            alternative=alternative,
            continuity=continuity,
        )

    np.testing.assert_allclose(actual, [pvalue, z], rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("samples", [[], [[1, 2, 3]], [[1, 2], [np.nan, np.nan]]])
def test_jonckheere_requires_two_observed_groups(samples):
    with np.errstate(all="raise"), pytest.raises(ValueError, match="at least two groups"):
        sp.test_jonckheere(samples)


@pytest.mark.parametrize("samples", [[[1], [1]], [[3, 3], [3, 3, 3]], [[0], [0], [0]]])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("continuity", [False, True])
@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.filterwarnings("ignore:Ties are present:UserWarning")
def test_jonckheere_constant_permutation_distribution_has_no_evidence(
    samples, alternative, continuity, nullable
):
    data, kwargs = samples, {}
    if nullable:
        data = pd.DataFrame(
            [(value, group) for group, sample in enumerate(samples) for value in sample],
            columns=["value", "group"],
        )
        data["value"] = data["value"].astype("Float64")
        kwargs = {"val_col": "value", "group_col": "group"}
    mean, variance = exact_null_moments(tuple(map(tuple, samples)))
    np.testing.assert_equal([pair_score(samples), variance], [mean, 0.0])

    with np.errstate(all="raise"):
        actual = sp.test_jonckheere(data, alternative=alternative, continuity=continuity, **kwargs)

    np.testing.assert_equal(actual, [1.0, 0.0])


def block_case(values, form):
    labels = [f"g-{index}" for index in np.roll(np.arange(values.shape[1]), 1)]
    wide = pd.DataFrame(values, columns=labels, index=np.arange(len(values)) % 2)
    if form == "array":
        return values.copy(), {}, np.arange(values.shape[1])
    if form == "nullable":
        return wide.astype("Float64"), {}, labels
    if form == "wide":
        return wide, {}, labels
    data = pd.DataFrame(
        [
            (value, label, f"subject-{row % 2}", row)
            for column, label in enumerate(labels)
            for row, value in enumerate(values[:, column])
        ],
        columns=["value", "treatment", "subject", 0],
    )
    return (
        data,
        {
            "y_col": "value",
            "group_col": "treatment",
            "block_col": "subject",
            "block_id_col": 0,
            "melted": True,
        },
        labels,
    )


@pytest.mark.parametrize("seed", [0, 7, 41])
@pytest.mark.parametrize("shape", [(2, 3), (5, 4)])
@pytest.mark.parametrize("form", ["array", "wide", "nullable", "melted"])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_page_seeded_ties_match_independent_rank_sum(seed, shape, form, sort, alternative):
    values = np.random.default_rng(seed).integers(-3, 4, size=shape)
    data, kwargs, labels = block_case(values, form)
    original = data.copy(deep=True) if isinstance(data, pd.DataFrame) else data.copy()
    ordered = values[:, np.argsort(labels)] if sort else values
    n, k = shape
    # SciPy supplies L independently; our API adds a continuity correction.
    L = stats.page_trend_test(ordered, method="asymptotic").statistic
    mean = n * k * (k + 1) ** 2 / 4
    variance = n * k**2 * (k + 1) * (k**2 - 1) / 144
    centered = L - mean
    correction = (
        0.5 * np.sign(centered)
        if alternative == "two-sided"
        else (0.5 if alternative == "greater" else -0.5)
    )
    z = (centered - correction) / np.sqrt(variance)
    pvalue = (
        2 * stats.norm.sf(abs(z))
        if alternative == "two-sided"
        else stats.norm.sf(z)
        if alternative == "greater"
        else stats.norm.cdf(z)
    )

    actual = sp.test_page(data, alternative=alternative, sort=sort, **kwargs)

    np.testing.assert_allclose(actual, [pvalue, z], rtol=1e-12, atol=1e-14)
    if isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(data, original)
    else:
        np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("seed", [0, 7, 41])
@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_page_reversing_measurements_reverses_the_ordered_alternative(seed, nullable, alternative):
    values = np.random.default_rng(seed).integers(-3, 4, size=(5, 4))
    data = pd.DataFrame(values, dtype="Float64") if nullable else values
    opposite = {"two-sided": "two-sided", "greater": "less", "less": "greater"}[alternative]

    pvalue, statistic = sp.test_page(data, alternative=alternative)
    reflected_pvalue, reflected_statistic = sp.test_page(-data, alternative=opposite)

    np.testing.assert_allclose(reflected_pvalue, pvalue)
    np.testing.assert_allclose(reflected_statistic, -statistic)
