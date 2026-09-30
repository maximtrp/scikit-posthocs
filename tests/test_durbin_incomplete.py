import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from scikit_posthocs import posthoc_durbin
from scikit_posthocs import test_durbin as durbin


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
