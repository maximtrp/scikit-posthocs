import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp

BLOCK_POSTHOCS = [
    (sp.posthoc_nemenyi_friedman, {}),
    (sp.posthoc_conover_friedman, {"p_adjust": "holm"}),
    (sp.posthoc_siegel_friedman, {"p_adjust": "holm"}),
    (sp.posthoc_miller_friedman, {}),
    (sp.posthoc_durbin, {"p_adjust": "holm"}),
    (sp.posthoc_quade, {"dist": "t"}),
    (sp.posthoc_quade, {"dist": "normal"}),
    (sp.posthoc_demsar, {"control": 0, "to_matrix": True}),
]


@pytest.mark.parametrize(("function", "kwargs"), BLOCK_POSTHOCS)
@pytest.mark.parametrize("collision", ["y", "blocks", "block_ids", "groups", "other"])
@pytest.mark.parametrize("sort", [False, True])
def test_wide_block_dataframe_preserves_treatment_labels(function, kwargs, collision, sort):
    values = np.array(
        [[31, 27, 24], [31, 28, 31], [45, 29, 46], [21, 18, 48], [42, 36, 46], [32, 17, 40]]
    )
    # Block labels may repeat, but rows still represent distinct blocks.
    data = pd.DataFrame(values, columns=["z", collision, "a"], index=[4, 2, 4, 2, 9, 9])
    data.columns.name = "treatment"
    data.index.name = "subject"
    original = data.copy(deep=True)
    expected = function(values, **kwargs)
    expected.index = data.columns
    expected.columns = data.columns
    labeled_kwargs = {**kwargs, "control": "z"} if function is sp.posthoc_demsar else kwargs

    actual = function(data, sort=sort, **labeled_kwargs)

    labels = data.columns.sort_values() if sort else data.columns
    np.testing.assert_equal(actual.shape, expected.shape)
    np.testing.assert_array_equal(actual.index, labels)
    np.testing.assert_array_equal(actual.columns, labels)
    np.testing.assert_allclose(actual.loc[data.columns, data.columns], expected, equal_nan=True)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("function", [sp.test_durbin, sp.test_page])
@pytest.mark.parametrize("collision", ["y", "blocks", "block_ids", "groups", "other"])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("dtype", ["int64", "Float64"])
def test_wide_block_omnibus_preserves_treatments(function, collision, sort, dtype):
    values = np.array(
        [[31, 27, 24], [31, 28, 31], [45, 29, 46], [21, 18, 48], [42, 36, 46], [32, 17, 40]]
    )
    # Nullable measurements exercise the converter even when sort=False.
    data = pd.DataFrame(values, columns=["z", collision, "a"], dtype=dtype)
    data.columns.name = "treatment"
    data.index.name = "subject"
    original = data.copy(deep=True)
    # Page's hypothesis follows treatment order, which sort=True changes.
    order = np.argsort(data.columns) if sort else np.arange(values.shape[1])
    expected = function(values[:, order], sort=False)

    actual = function(data, sort=sort)

    np.testing.assert_allclose(actual, expected)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("sort", [False, True])
def test_nemenyi_axis_labels_and_missing_block_match_rank_reference(sort):
    data = pd.DataFrame(
        [[31, 27, 24], [31, 28, 31], [50, np.nan, 52], [45, 29, 46], [21, 18, 48]],
        columns=["z", "y", "x"],
        dtype="Float64",
    )
    original = data.copy(deep=True)
    values = data.dropna().to_numpy(dtype=float)
    mean_ranks = stats.rankdata(values, axis=1).mean(axis=0)
    n, k = values.shape
    differences = np.abs(mean_ranks[:, None] - mean_ranks[None, :])
    expected = stats.studentized_range.sf(differences / np.sqrt(k * (k + 1) / (12 * n)), k, np.inf)
    np.fill_diagonal(expected, 1)

    actual = sp.posthoc_nemenyi_friedman(data, sort=sort)

    labels = data.columns.sort_values() if sort else data.columns
    np.testing.assert_equal(actual.shape, expected.shape)
    np.testing.assert_array_equal(actual.index, labels)
    np.testing.assert_array_equal(actual.columns, labels)
    np.testing.assert_allclose(actual.loc[data.columns, data.columns], expected)
    pd.testing.assert_frame_equal(data, original)
