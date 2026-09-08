import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from scikit_posthocs import posthoc_conover, posthoc_dunn, posthoc_nemenyi

RANK_POSTHOCS = [posthoc_conover, posthoc_dunn, posthoc_nemenyi]


@pytest.mark.parametrize("padding", [0, 1, 20])
@pytest.mark.parametrize("input_type", ["list", "array", "dataframe"])
@pytest.mark.parametrize("sort", [False, True])
def test_dunn_missing_values_do_not_change_two_group_reference(padding, input_type, sort):
    samples = [[1, 2, 2, 4, 5, 8], [3, 5, 5, 6, 9, 10]]
    reference = mannwhitneyu(*samples, method="asymptotic", use_continuity=False).pvalue
    padded = [group + [np.nan] * padding for group in samples]
    kwargs = {}
    if input_type == "dataframe":
        data = pd.DataFrame(
            {
                "value": padded[0] + padded[1],
                "group": ["b"] * len(padded[0]) + ["a"] * len(padded[1]),
            }
        )
        kwargs = {"val_col": "value", "group_col": "group"}
    else:
        data = np.asarray(padded) if input_type == "array" else padded
    actual = posthoc_dunn(data, sort=sort, **kwargs)
    np.testing.assert_allclose(actual.iloc[0, 1], reference)


def test_dunn_missing_group_is_excluded_before_ranking():
    data = pd.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, 6, 1000],
            "group": ["a", "a", "a", "b", "b", "b", None],
        }
    )
    expected = posthoc_dunn(data.dropna(), val_col="value", group_col="group", p_adjust="holm")
    actual = posthoc_dunn(data, val_col="value", group_col="group", p_adjust="holm")
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("function", RANK_POSTHOCS)
@pytest.mark.parametrize("input_type", ["list", "array", "dataframe"])
@pytest.mark.parametrize("sort", [False, True])
def test_rank_posthoc_missing_values_match_complete_input(function, input_type, sort):
    samples = [[1, 2, 2, 4, 5, 8], [3, 5, 5, 6, 9, 10], [2, 4, 7, 7, 8, 12]]
    padded = [group + [np.nan] * (i + 1) for i, group in enumerate(samples)]
    kwargs = {}
    if input_type == "dataframe":
        labels = ["b", "a", "c"]

        def to_data(groups):
            return pd.DataFrame(
                {
                    "value": [value for group in groups for value in group],
                    "group": [label for label, group in zip(labels, groups) for _ in group],
                }
            )

        complete_data = to_data(samples)
        padded_data = to_data(padded)
        kwargs = {"val_col": "value", "group_col": "group"}
    else:
        complete_data = np.asarray(samples) if input_type == "array" else samples
        padded_data = np.asarray(padded, dtype=object) if input_type == "array" else padded

    expected = function(complete_data, sort=sort, **kwargs)
    actual = function(padded_data, sort=sort, **kwargs)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("function", RANK_POSTHOCS)
def test_rank_posthoc_missing_group_matches_complete_input(function):
    data = pd.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, 6, 1000],
            "group": ["a", "a", "a", "b", "b", "b", None],
        }
    )
    expected = function(data.dropna(), val_col="value", group_col="group")
    actual = function(data, val_col="value", group_col="group")
    pd.testing.assert_frame_equal(actual, expected)
