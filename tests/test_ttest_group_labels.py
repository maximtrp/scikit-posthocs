import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp


@pytest.mark.parametrize("sort", [False, True])
def test_pooled_ttest_tuple_groups_match_two_sample_scipy(sort):
    samples = [[5.0, 7.0, 9.0, 10.0], [1.0, 2.0, 3.0]]
    labels = [("group", 2), ("group", 0)]
    data = pd.DataFrame({"value": sum(samples, []), "group": [labels[0]] * 4 + [labels[1]] * 3})
    original = data.copy(deep=True)

    actual = sp.posthoc_ttest(data, val_col="value", group_col="group", pool_sd=True, sort=sort)

    expected_pvalue = stats.ttest_ind(*samples, equal_var=True).pvalue
    np.testing.assert_allclose(actual.to_numpy(), [[1.0, expected_pvalue], [expected_pvalue, 1.0]])
    assert actual.index.tolist() == (sorted(labels) if sort else labels)
    assert actual.columns.tolist() == actual.index.tolist()
    assert not isinstance(actual.index, pd.MultiIndex)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("p_adjust", [None, "bonferroni"])
@pytest.mark.parametrize("labels", [[("z", 2), ("a", 0), ("m", 1)], ["z", "a", "m"], [20, 0, 10]])
def test_pooled_ttest_group_labels_match_within_group_variance(sort, p_adjust, labels):
    samples = [
        np.array([1.0, 2.0, 4.0]),
        np.array([5.0, 7.0, 9.0, 10.0]),
        np.array([2.0, 4.0, 6.0, 7.0, 11.0]),
    ]
    data = pd.DataFrame({
        "value": np.concatenate(samples),
        "group": [label for label, sample in zip(labels, samples) for _ in sample],
    })
    original = data.copy(deep=True)

    actual = sp.posthoc_ttest(
        data, val_col="value", group_col="group", pool_sd=True, p_adjust=p_adjust, sort=sort
    )

    degrees_of_freedom = sum(len(sample) - 1 for sample in samples)
    variance = sum(np.sum((sample - sample.mean()) ** 2) for sample in samples) / degrees_of_freedom
    expected = np.eye(len(samples))
    for i, j in itertools.combinations(range(len(samples)), 2):
        standard_error = np.sqrt(variance * (1 / len(samples[i]) + 1 / len(samples[j])))
        statistic = abs(samples[i].mean() - samples[j].mean()) / standard_error
        pvalue = 2 * stats.t.sf(statistic, degrees_of_freedom)
        expected[i, j] = expected[j, i] = min(pvalue * 3, 1) if p_adjust else pvalue

    np.testing.assert_allclose(actual.reindex(index=labels, columns=labels), expected)
    assert actual.index.tolist() == (sorted(labels) if sort else labels)
    assert actual.columns.tolist() == actual.index.tolist()
    assert not isinstance(actual.index, pd.MultiIndex)
    pd.testing.assert_frame_equal(data, original)
