import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp
from tests._input_cases import BLOCK_POSTHOCS, block_input

COMPLETE_BLOCK_TESTS = [
    case for case in BLOCK_POSTHOCS if case.values[0] is not sp.posthoc_durbin
] + [pytest.param(sp.test_page, {}, id="test_page")]


@pytest.mark.parametrize(("function", "kwargs"), COMPLETE_BLOCK_TESTS)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["array", "wide", "nullable", "melted"])
@pytest.mark.parametrize(
    "values",
    [
        np.array([[1, np.nan, 3], [np.nan, 2, 1], [2, 3, np.nan]]),
        np.full((3, 3), np.nan),
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.arange(3, dtype=float).reshape(-1, 1),
    ],
    ids=["all-incomplete", "all-missing", "no-blocks", "no-treatments", "one-treatment"],
)
def test_complete_block_tests_reject_unusable_designs(function, kwargs, sort, input_type, values):
    data, input_kwargs = block_input(values, input_type)

    with np.errstate(all="raise"), pytest.raises(ValueError, match="complete block|two treatments"):
        function(data, sort=sort, **{**kwargs, **input_kwargs})


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["array", "wide", "nullable", "melted"])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_page_rejects_no_complete_blocks_for_every_alternative(sort, input_type, alternative):
    values = np.array([[1, np.nan, 3], [np.nan, 2, 1], [2, 3, np.nan]])
    data, kwargs = block_input(values, input_type)

    with np.errstate(all="raise"), pytest.raises(ValueError, match="complete block"):
        sp.test_page(data, alternative=alternative, sort=sort, **kwargs)


@pytest.mark.parametrize("sort", [False, True])
def test_page_dropping_first_block_preserves_original_hypothesis_order(sort):
    values = np.array([[1, 4, 2], [3, 2, 6], [8, 1, 3]])
    wide = pd.DataFrame(values, columns=["a", "b", "c"])
    wide.index.name = "subject"
    long = wide.reset_index().melt(id_vars="subject", var_name="treatment", value_name="value")
    # The first, incomplete block establishes the unsorted hypothesis c, a, b.
    first = pd.DataFrame({"subject": [-1, -1], "treatment": ["c", "a"], "value": [9, 10]})
    data = pd.concat([first, long], ignore_index=True)
    original = data.copy(deep=True)
    order = [0, 1, 2] if sort else [2, 0, 1]
    rank_sums = stats.rankdata(values[:, order], axis=1).sum(axis=0)
    n, k = values.shape
    centered = rank_sums @ np.arange(1, k + 1) - n * k * (k + 1) ** 2 / 4
    statistic = centered - 0.5 * np.sign(centered)
    statistic /= np.sqrt(n * k**2 * (k + 1) * (k**2 - 1) / 144)
    expected = [2 * stats.norm.sf(abs(statistic)), statistic]

    actual = sp.test_page(
        data,
        y_col="value",
        group_col="treatment",
        block_col="subject",
        block_id_col="subject",
        melted=True,
        sort=sort,
    )

    np.testing.assert_allclose(actual, expected)
    pd.testing.assert_frame_equal(data, original)
