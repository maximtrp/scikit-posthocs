"""Desired behavior for independently reproduced bugs; all xfails are strict.

Run with --runxfail to reproduce failures. Remove a mark when its production
fix lands; an unexpected pass otherwise fails CI instead of silently going stale.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

import scikit_posthocs as sp
from tests._input_cases import LABELS, assert_matrix, independent_samples, long_data


@pytest.mark.parametrize("sort", [False, True])
def test_tukey_duplicate_index_matches_scipy(sort):
    samples = independent_samples()
    data = long_data(samples)
    data.index = np.tile(np.arange(5), 3)
    expected = pd.DataFrame(stats.tukey_hsd(*samples).pvalue, index=LABELS, columns=LABELS)

    actual = sp.posthoc_tukey_hsd(data, val_col="value", group_col="group", sort=sort)

    np.testing.assert_allclose(actual.loc[LABELS, LABELS], expected)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("repeated_index", [False, True])
def test_tukey_categorical_group_order_matches_scipy(ordered, sort, repeated_index):
    samples = [[1, 2, 3], [1.1, 2.1, 3.1], [11, 12, 13]]
    data = long_data(samples)
    category_order = ["m", "z", "a"]
    data["group"] = pd.Categorical(
        data["group"], categories=[*category_order, "unused"], ordered=ordered
    )
    if repeated_index:
        data.index = np.tile(np.arange(3), 3)
    original = data.copy(deep=True)
    expected = pd.DataFrame(stats.tukey_hsd(*samples).pvalue, index=LABELS, columns=LABELS)

    actual = sp.posthoc_tukey_hsd(data, val_col="value", group_col="group", sort=sort)

    assert_matrix(actual, expected, category_order if sort else LABELS)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.xfail(
    strict=True,
    raises=(AssertionError, KeyError),
    reason="Rank helpers overwrite the user-specified grouping column with temporary scores",
)
@pytest.mark.parametrize(
    ("function", "group_col"),
    [
        (sp.posthoc_conover, "ranks"),
        (sp.posthoc_dunn, "ranks"),
        (sp.posthoc_nemenyi, "ranks"),
        (sp.posthoc_vanwaerden, "z_scores"),
    ],
)
@pytest.mark.parametrize("sort", [False, True])
def test_temporary_score_column_does_not_overwrite_group_labels(function, group_col, sort):
    data = long_data(independent_samples())
    expected = function(data, val_col="value", group_col="group", sort=sort)
    renamed = data.rename(columns={"group": group_col})

    actual = function(renamed, val_col="value", group_col=group_col, sort=sort)

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("function", [sp.posthoc_dunnett, sp.posthoc_steel])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("to_matrix", [False, True])
@pytest.mark.parametrize("columns", [(11, 17), (0, 1), (11, 0), ("value", ""), ("value", False)])
def test_many_to_one_column_names_match_external_reference(
    function, sort, to_matrix, columns, monkeypatch
):
    samples = independent_samples()
    data = long_data(samples).rename(columns={"value": columns[0], "group": columns[1]})
    original_data = data.copy(deep=True)
    kwargs = {}
    if function is sp.posthoc_dunnett:
        original_dunnett = stats.dunnett
        seed_key = (
            "rng" if "rng" in inspect.signature(original_dunnett).parameters else "random_state"
        )

        def seeded_dunnett(*args, **kwargs):
            return original_dunnett(*args, **{**kwargs, seed_key: np.random.default_rng(731)})

        monkeypatch.setattr(stats, "dunnett", seeded_dunnett)
        pvalues = seeded_dunnett(*samples[1:], control=samples[0]).pvalue
    else:
        kwargs["p_adjust"] = "holm"
        raw_pvalues = [stats.mannwhitneyu(sample, samples[0]).pvalue for sample in samples[1:]]
        pvalues = multipletests(raw_pvalues, method="holm")[1]

    actual = function(
        data,
        val_col=columns[0],
        group_col=columns[1],
        control="z",
        sort=sort,
        to_matrix=to_matrix,
        **kwargs,
    )

    if to_matrix:
        expected = pd.DataFrame(np.eye(3), index=LABELS, columns=LABELS)
        expected.loc["a", "m"] = expected.loc["m", "a"] = np.nan
        expected.loc["z", ["a", "m"]] = pvalues
        expected.loc[["a", "m"], "z"] = pvalues
        assert_matrix(actual, expected, sorted(LABELS) if sort else LABELS, many_to_one=True)
    else:
        np.testing.assert_equal(actual.shape, (2,))
        np.testing.assert_array_equal(actual.index.tolist(), [("z", "a"), ("z", "m")])
        np.testing.assert_allclose(actual, pvalues, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(data, original_data)


@pytest.mark.parametrize(
    "columns",
    [
        (0, "group"),
        ("value", 0),
        ("", "group"),
        ("value", ""),
        (False, "group"),
        ("value", False),
    ],
)
def test_falsy_dataframe_column_label_matches_named_columns(columns):
    data = long_data(independent_samples())
    expected = sp.posthoc_ttest(data, val_col="value", group_col="group")
    renamed = data.rename(columns={"value": columns[0], "group": columns[1]})

    actual = sp.posthoc_ttest(renamed, val_col=columns[0], group_col=columns[1])

    pd.testing.assert_frame_equal(actual, expected)


def test_durbin_nullable_nonpositional_blocks_match_rank_reference():
    values = np.array(
        [[31, 27, 24], [31, 28, 31], [45, 29, 46], [21, 18, 48], [42, 36, 46], [32, 17, 40]]
    )
    data = pd.DataFrame(values, index=["b4", "b1", "b4", "b2", "b3", "b3"], dtype="Float64")
    ranks = stats.rankdata(values, axis=1)
    n, k = values.shape
    correction = n * k * (k + 1) ** 2 / 4
    statistic = (k - 1) * (np.sum(ranks.sum(axis=0) ** 2) - n * correction)
    statistic /= np.sum(ranks**2) - correction
    expected = [stats.chi2.sf(statistic, k - 1), statistic, k - 1]

    actual = sp.test_durbin(data, sort=False)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="sign_array leaves p==alpha unclassified and replaces zero by a fixed 1e-10",
)
@pytest.mark.parametrize("alpha", [0.05, 1e-12])
def test_sign_array_thresholds_return_only_documented_codes(alpha):
    values = np.array([[1, alpha, 0], [alpha, 1, alpha / 2], [0, alpha / 2, 1]])
    expected = (values < alpha).astype(int)
    np.fill_diagonal(expected, -1)

    actual = sp.sign_array(values, alpha=alpha)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="sign_table leaves p==0.05 as a numeric string instead of nonsignificant",
)
def test_sign_table_classifies_exact_significance_boundaries():
    values = np.array([[1, 0.05, 0.01], [0.05, 1, 0.001], [0.01, 0.001, 1]])

    actual = sp.sign_table(values)

    np.testing.assert_array_equal(actual, [["-", "NS", "*"], ["NS", "-", "**"], ["*", "**", "-"]])


def test_two_groups_with_two_observations_match_scipy_for_array_input():
    samples = np.array([[1.0, 4.0], [2.0, 8.0]])
    expected = stats.ttest_ind(*samples).pvalue

    actual = sp.posthoc_ttest(samples)

    np.testing.assert_allclose(actual.iloc[0, 1], expected)


def test_melted_block_id_column_zero_matches_named_column():
    wide = pd.DataFrame([[1, 3, 2], [4, 2, 5], [2, 6, 1], [7, 3, 4]])
    wide.index.name = "subject"
    data = wide.reset_index().melt(id_vars="subject", var_name="treatment", value_name="value")
    data["row_id"] = data["subject"]
    expected = sp.posthoc_nemenyi_friedman(
        data,
        y_col="value",
        group_col="treatment",
        block_col="subject",
        block_id_col="row_id",
        melted=True,
    )

    actual = sp.posthoc_nemenyi_friedman(
        data.rename(columns={"row_id": 0}),
        y_col="value",
        group_col="treatment",
        block_col="subject",
        block_id_col=0,
        melted=True,
    )

    pd.testing.assert_frame_equal(actual, expected)
