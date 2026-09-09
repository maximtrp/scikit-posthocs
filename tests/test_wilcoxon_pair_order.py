import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from scikit_posthocs import posthoc_wilcoxon


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["list", "array", "dataframe"])
@pytest.mark.parametrize("method", ["auto", "exact"])
@pytest.mark.parametrize("adjust", [None, "holm"])
def test_wilcoxon_group_sort_preserves_subject_pairs(sort, input_type, method, adjust):
    x = 2 * np.arange(40)
    samples = [x, x[::-1] + 1, np.roll(x, 9) + 3]
    labels = ["c", "a", "b"] if input_type == "dataframe" else [1, 2, 3]
    kwargs = {}
    if input_type == "dataframe":
        # Interleave groups, while retaining each subject's order within a group.
        data = (
            pd.DataFrame(np.asarray(samples).T, columns=labels)
            .melt(var_name="group", value_name="value", ignore_index=False)
            .sort_index(kind="stable")
        )
        kwargs = {"val_col": "value", "group_col": "group"}
    else:
        data = np.asarray(samples) if input_type == "array" else samples
    expected = [
        wilcoxon(samples[i], samples[j], method=method).pvalue for i, j in [(0, 1), (0, 2), (1, 2)]
    ]
    if adjust:
        expected = multipletests(expected, method=adjust)[1]
    actual = posthoc_wilcoxon(data, sort=sort, method=method, p_adjust=adjust, **kwargs)
    for (i, j), p_value in zip([(0, 1), (0, 2), (1, 2)], expected):
        np.testing.assert_allclose(actual.loc[labels[i], labels[j]], p_value)
    np.testing.assert_array_equal(actual.index, sorted(labels) if sort else labels)
