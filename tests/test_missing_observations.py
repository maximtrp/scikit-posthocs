import numpy as np
import pandas as pd
import pytest

import scikit_posthocs as sp


@pytest.mark.parametrize(
    "function",
    [sp.posthoc_lsd, sp.posthoc_snk, sp.posthoc_duncan, sp.posthoc_vanwaerden],
)
def test_independent_posthoc_missing_values_match_complete_input(function):
    complete = [[1, 2, 4, 5, 8], [3, 5, 6, 9, 10], [2, 4, 7, 8, 12]]
    padded = [group + [np.nan] for group in complete]

    expected = function(complete)
    actual = function(padded)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(actual.index, expected.index)


def test_npm_missing_values_match_complete_input():
    complete = [[1, 2, 4, 5, 8], [3, 5, 6, 9, 10], [2, 4, 7, 8, 12]]
    padded = [group + [np.nan] for group in complete]

    np.random.seed(123)
    expected = sp.posthoc_npm_test(complete, nperm=100)
    np.random.seed(123)
    actual = sp.posthoc_npm_test(padded, nperm=100)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(actual.index, expected.index)


@pytest.mark.parametrize("function", [sp.test_mackwolfe, sp.test_osrt, sp.test_jonckheere])
def test_independent_omnibus_missing_values_match_complete_input(function):
    complete = [[1, 2, 4, 5, 8], [3, 5, 6, 9, 10], [2, 4, 7, 8, 12]]
    padded = [group + [np.nan] for group in complete]
    kwargs = {"p": 1} if function is sp.test_mackwolfe else {}

    expected = function(complete, **kwargs)
    actual = function(padded, **kwargs)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "function",
    [sp.posthoc_mannwhitney, sp.posthoc_tukey_hsd, sp.posthoc_scheffe],
)
def test_independent_posthoc_missing_group_matches_complete_input(function):
    complete = pd.DataFrame(
        {
            "value": [1, 2, 4, 5, 7, 8, 10, 12, 13],
            "group": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        }
    )
    incomplete = pd.concat(
        [complete, pd.DataFrame({"value": [1000], "group": [None]})],
        ignore_index=True,
    )

    expected = function(complete, val_col="value", group_col="group")
    actual = function(incomplete, val_col="value", group_col="group")

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("input_type", ["list", "dataframe", "array"])
def test_wilcoxon_rejects_missing_paired_observations(input_type):
    samples = [[1, 2, np.nan, 4], [1, 3, 5, 7]]
    kwargs = {}
    if input_type == "dataframe":
        data = pd.DataFrame(
            {
                "value": samples[0] + samples[1],
                "group": ["a"] * 4 + ["b"] * 4,
            }
        )
        kwargs = {"val_col": "value", "group_col": "group"}
    elif input_type == "array":
        data = np.array(
            [(value, group) for group, sample in enumerate(samples) for value in sample]
        )
    else:
        data = samples

    with pytest.raises(ValueError, match="missing paired observations"):
        sp.posthoc_wilcoxon(data, **kwargs)


COMPLETE_BLOCK_POSTHOCS = [
    (sp.posthoc_nemenyi_friedman, {}),
    (sp.posthoc_conover_friedman, {}),
    (sp.posthoc_siegel_friedman, {}),
    (sp.posthoc_miller_friedman, {}),
    (sp.posthoc_quade, {}),
    (sp.posthoc_demsar, {"control": 0}),
]


@pytest.mark.parametrize(("function", "kwargs"), COMPLETE_BLOCK_POSTHOCS)
def test_complete_block_posthoc_drops_entire_incomplete_block(function, kwargs):
    complete = np.array(
        [
            [31.0, 27.0, 24.0],
            [31.0, 28.0, 31.0],
            [45.0, 29.0, 46.0],
            [21.0, 18.0, 48.0],
            [42.0, 36.0, 46.0],
            [32.0, 17.0, 40.0],
        ]
    )
    incomplete = np.insert(complete, 2, [50.0, np.nan, 52.0], axis=0)

    expected = function(complete, **kwargs)
    actual = function(incomplete, **kwargs)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(actual.index, expected.index)


def test_page_drops_entire_incomplete_block():
    complete = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 8.0]])
    incomplete = np.insert(complete, 1, [2.0, np.nan, 6.0], axis=0)

    expected = sp.test_page(complete)
    actual = sp.test_page(incomplete)

    np.testing.assert_allclose(actual, expected)
