import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from scikit_posthocs import posthoc_durbin
from scikit_posthocs import test_durbin as durbin
from tests._input_cases import block_input


@pytest.fixture
def balanced_pairs():
    return np.array(
        [
            [1, 2, np.nan],
            [2, 1, np.nan],
            [1, np.nan, 2],
            [2, np.nan, 1],
            [np.nan, 1, 2],
            [np.nan, 2, 1],
        ]
    )


@pytest.mark.parametrize("melted", [False, True])
@pytest.mark.parametrize("sort", [False, True])
def test_durbin_equal_rank_totals(balanced_pairs, melted, sort):
    data = balanced_pairs
    kwargs = {}
    if melted:
        data = (
            pd.DataFrame(data)
            .rename_axis("block")
            .reset_index()
            .melt(id_vars="block", var_name="treatment", value_name="value")
            .dropna()
        )
        kwargs = {
            "y_col": "value",
            "group_col": "treatment",
            "block_col": "block",
            "block_id_col": "block",
            "melted": True,
        }
    # A=30, C=27, r=4, and all three treatment rank totals equal six.
    assert_allclose(durbin(data, sort=sort, **kwargs), [1, 0, 2], atol=1e-12)
    assert_allclose(posthoc_durbin(data, sort=sort, **kwargs), np.ones((3, 3)))


def test_durbin_unequal_rank_totals(balanced_pairs):
    balanced_pairs[1, :2] = [1, 2]
    # Rank totals 5,7,6 give T1=4/3; chi-square(2) survival is exp(-T1/2).
    assert_allclose(durbin(balanced_pairs), [np.exp(-2 / 3), 4 / 3, 2])
    # NIST's pairwise formula gives df=4 and squared denominator=14/3.
    assert_allclose(
        posthoc_durbin(balanced_pairs),
        [
            [1, 0.4069401997057837, 0.667492180832647],
            [0.4069401997057837, 1, 0.667492180832647],
            [0.667492180832647, 0.667492180832647, 1],
        ],
    )


def test_durbin_nist_incomplete_example():
    # https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/durbin.htm
    data = np.array(
        [
            [2, 3, np.nan, 1, np.nan, np.nan, np.nan],
            [np.nan, 3, 1, np.nan, 2, np.nan, np.nan],
            [np.nan, np.nan, 2, 1, np.nan, 3, np.nan],
            [np.nan, np.nan, np.nan, 1, 2, np.nan, 3],
            [3, np.nan, np.nan, np.nan, 1, 2, np.nan],
            [np.nan, 3, np.nan, np.nan, np.nan, 1, 2],
            [3, np.nan, 1, np.nan, np.nan, np.nan, 2],
        ]
    )
    # NIST reports T1=12; this API uses its original chi-square approximation.
    assert_allclose(durbin(data)[1:], [12, 6])


@pytest.mark.parametrize("function", [durbin, posthoc_durbin])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["array", "wide", "nullable", "melted"])
@pytest.mark.parametrize("padding", ["blocks", "treatment", "both"])
def test_durbin_ignores_wholly_missing_blocks_and_treatments(
    balanced_pairs, function, sort, input_type, padding
):
    balanced_pairs[1, :2] = [1, 2]
    values = balanced_pairs
    if padding in {"blocks", "both"}:
        values = np.vstack([values, np.full((2, 3), np.nan)])
    if padding in {"treatment", "both"}:
        values = np.column_stack([values, np.full(len(values), np.nan)])
    data, kwargs = block_input(values, input_type)
    original = data.copy() if isinstance(data, np.ndarray) else data.copy(deep=True)

    actual = function(data, sort=sort, **kwargs)

    if function is durbin:
        assert_allclose(actual, [np.exp(-2 / 3), 4 / 3, 2])
    else:
        assert_allclose(
            actual,
            [
                [1, 0.4069401997057837, 0.667492180832647],
                [0.4069401997057837, 1, 0.667492180832647],
                [0.667492180832647, 0.667492180832647, 1],
            ],
        )
        np.testing.assert_array_equal(actual.index, [0, 1, 2])
    if isinstance(data, np.ndarray):
        np.testing.assert_array_equal(data, original)
    else:
        pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("function", [durbin, posthoc_durbin])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["array", "wide", "nullable", "melted"])
@pytest.mark.parametrize(
    "values",
    [np.full((3, 3), np.nan), np.empty((0, 3)), np.empty((3, 0)), np.arange(3).reshape(-1, 1)],
    ids=["all-missing", "no-blocks", "no-treatments", "one-treatment"],
)
def test_durbin_rejects_empty_or_single_treatment_designs(function, sort, input_type, values):
    data, kwargs = block_input(values, input_type)

    with np.errstate(all="raise"), pytest.raises(ValueError, match="block|two treatments"):
        function(data, sort=sort, **kwargs)


@pytest.mark.parametrize("function", [durbin, posthoc_durbin])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["array", "wide", "nullable", "melted"])
@pytest.mark.parametrize(
    "values",
    [
        [[1, 2, 3], [2, 1, np.nan], [1, np.nan, 2], [np.nan, 2, 1]],
        [[1, 2, np.nan], [2, 1, np.nan], [1, np.nan, 2]],
        [
            [1, 2, np.nan, np.nan],
            [np.nan, 1, 2, np.nan],
            [np.nan, np.nan, 1, 2],
            [2, np.nan, np.nan, 1],
        ],
    ],
    ids=["unequal-block-sizes", "unequal-replication", "unequal-pair-concurrence"],
)
def test_durbin_rejects_unbalanced_designs(function, sort, input_type, values):
    data, kwargs = block_input(np.asarray(values), input_type)

    with pytest.raises(ValueError, match="balanced incomplete block design"):
        function(data, sort=sort, **kwargs)


@pytest.mark.parametrize("function", [durbin, posthoc_durbin])
def test_durbin_rejects_duplicate_block_treatment_observations(balanced_pairs, function):
    data, kwargs = block_input(balanced_pairs, "melted")
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="balanced incomplete block design"):
        function(data, **kwargs)
