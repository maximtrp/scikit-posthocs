import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

import scikit_posthocs as sp
from tests._input_cases import LABELS, assert_matrix, long_data

REFERENCES = [
    (sp.posthoc_ttest, {"pool_sd": True}, "pooled_t"),
    (sp.posthoc_lsd, {}, "pooled_t"),
    (sp.posthoc_scheffe, {}, "scheffe"),
    (sp.posthoc_tukey_hsd, {}, "tukey"),
    (sp.posthoc_tukey, {}, "tukey"),
    (sp.posthoc_games_howell, {}, "games_howell"),
    (sp.posthoc_tamhane, {}, "tamhane"),
    (sp.posthoc_median, {}, "median"),
    (sp.posthoc_median, {"p_adjust": "holm"}, "median"),
    (sp.posthoc_median, {"p_adjust": "bonferroni"}, "median"),
    (sp.posthoc_median, {"p_adjust": "fdr_bh"}, "median"),
]


def parametric_reference(samples, method):
    if method == "tukey":
        return stats.tukey_hsd(*samples).pvalue
    k = len(samples)
    n = sum(map(len, samples))
    df = n - k
    within_ss = sum(sum((x - np.mean(sample)) ** 2 for x in sample) for sample in samples)
    pooled_variance = within_ss / df
    grand_median = np.median(np.concatenate(samples))
    expected = np.eye(k)
    for i, j in itertools.combinations(range(k), 2):
        first, second = np.asarray(samples[i]), np.asarray(samples[j])
        difference = abs(first.mean() - second.mean())
        if method in ("pooled_t", "scheffe"):
            t_squared = difference**2 / (pooled_variance * (1 / len(first) + 1 / len(second)))
            if method == "pooled_t":
                pvalue = 2 * stats.t.sf(np.sqrt(t_squared), df)
            else:
                pvalue = stats.f.sf(t_squared / (k - 1), k - 1, df)
        elif method in ("games_howell", "tamhane"):
            # SciPy independently supplies Welch's statistic and Satterthwaite df.
            welch = stats.ttest_ind(first, second, equal_var=False)
            if method == "games_howell":
                # Compute the df separately to support SciPy versions predating result.df.
                v1, v2 = first.var(ddof=1) / len(first), second.var(ddof=1) / len(second)
                welch_df = (v1 + v2) ** 2 / (v1**2 / (len(first) - 1) + v2**2 / (len(second) - 1))
                pvalue = stats.studentized_range.sf(abs(welch.statistic) * np.sqrt(2), k, welch_df)
            else:
                comparisons = k * (k - 1) // 2
                pvalue = -np.expm1(comparisons * np.log1p(-welch.pvalue))
        else:
            above = [np.count_nonzero(sample > grand_median) for sample in (first, second)]
            table = [[above[0], len(first) - above[0]], [above[1], len(second) - above[1]]]
            pvalue = stats.chi2_contingency(table, correction=False).pvalue
        expected[i, j] = expected[j, i] = pvalue
    return expected


@pytest.mark.parametrize(("function", "kwargs", "method"), REFERENCES)
@pytest.mark.parametrize("seed", [5, 23, 91])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_unequal_group_sizes_and_variances_match_external_reference(
    function, kwargs, method, seed, sort, dtype
):
    rng = np.random.default_rng(seed)
    samples = [rng.normal(loc=i / 2, scale=i + 1, size=size) for i, size in enumerate([5, 8, 11])]
    data = long_data(samples, dtype=dtype)
    original = data.copy(deep=True)
    expected = pd.DataFrame(parametric_reference(samples, method), index=LABELS, columns=LABELS)
    if kwargs.get("p_adjust"):
        triangle = np.triu_indices(len(samples), 1)
        values = expected.to_numpy(copy=True)
        values[triangle] = multipletests(values[triangle], method=kwargs["p_adjust"])[1]
        values[(triangle[1], triangle[0])] = values[triangle]
        expected = pd.DataFrame(values, index=LABELS, columns=LABELS)

    actual = function(data, val_col="value", group_col="group", sort=sort, **kwargs)

    assert_matrix(actual, expected, sorted(LABELS) if sort else LABELS)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("adjustment", [None, "holm", "bonferroni", "fdr_bh"])
def test_median_adjustments_match_known_above_median_counts(sort, adjustment):
    # Exactly half of the pooled sample is above zero. The six pairwise tests
    # have distinct, nontrivial adjusted values, not just saturated ones.
    above = [1, 3, 7, 9]
    samples = [np.r_[-np.arange(1, 11 - n), np.arange(1, n + 1)] for n in above]
    labels = ["z", "a", "m", "b"]
    data = long_data(samples, labels=labels)
    triangle = np.triu_indices(4, 1)
    raw = [
        stats.chi2_contingency(
            [[above[i], 10 - above[i]], [above[j], 10 - above[j]]], correction=False
        ).pvalue
        for i, j in zip(*triangle)
    ]
    expected = np.eye(4)
    expected[triangle] = multipletests(raw, method=adjustment)[1] if adjustment else raw
    expected[(triangle[1], triangle[0])] = expected[triangle]
    expected = pd.DataFrame(expected, index=labels, columns=labels)

    actual = sp.posthoc_median(
        data, val_col="value", group_col="group", sort=sort, p_adjust=adjustment
    )

    assert_matrix(actual, expected, sorted(labels) if sort else labels)


@pytest.mark.parametrize("sort", [False, True])
def test_dscf_small_probabilities_match_pairwise_mannwhitney_reference(sort):
    samples = [np.arange(12) + shift for shift in (0, 10, 20)]
    data = long_data(samples)
    expected = np.eye(3)
    for i, j in itertools.combinations(range(3), 2):
        first, second = samples[i], samples[j]
        u = stats.mannwhitneyu(first, second, method="asymptotic").statistic
        tie_factor = stats.tiecorrect(stats.rankdata(np.concatenate([first, second])))
        variance = len(first) * len(second) * (len(first) + len(second) + 1) / 12 * tie_factor
        q = abs(u - len(first) * len(second) / 2) * np.sqrt(2 / variance)
        expected[i, j] = expected[j, i] = stats.studentized_range.sf(q, 3, np.inf)
    # The fixture must exercise positive, representable probabilities below .001.
    tails = expected[np.triu_indices(3, 1)]
    np.testing.assert_array_less(0, tails)
    np.testing.assert_array_less(tails, 0.001)

    actual = sp.posthoc_dscf(data, val_col="value", group_col="group", sort=sort)

    np.testing.assert_allclose(actual.loc[LABELS, LABELS], expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("continuity", [False, True])
def test_mannwhitney_upper_triangle_matches_directional_scipy(sort, alternative, continuity):
    samples = [[1, 2, 2, 4, 5, 8], [3, 5, 5, 6, 9, 10], [2, 4, 7, 7, 8, 12]]
    data = long_data(samples)
    grouped = dict(zip(LABELS, samples))

    actual = sp.posthoc_mannwhitney(
        data,
        val_col="value",
        group_col="group",
        sort=sort,
        alternative=alternative,
        use_continuity=continuity,
    )

    np.testing.assert_array_equal(actual.index, sorted(LABELS) if sort else LABELS)
    for i, j in itertools.combinations(range(3), 2):
        expected = stats.mannwhitneyu(
            grouped[actual.index[i]],
            grouped[actual.index[j]],
            alternative=alternative,
            use_continuity=continuity,
        ).pvalue
        np.testing.assert_allclose(actual.iloc[i, j], expected)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("correction", [False, True])
@pytest.mark.parametrize("zero_method", ["wilcox", "pratt", "zsplit"])
def test_wilcoxon_zero_differences_keep_interleaved_subject_pairs(sort, correction, zero_method):
    baseline = np.arange(12, dtype=float)
    samples = [baseline, baseline + np.resize([0, -2, 1, 3], 12), baseline[::-1] + 0.5]
    data = pd.DataFrame(np.asarray(samples).T, columns=LABELS)
    # Keep the common subject order, rather than independently shuffling paired samples.
    data = data.melt(var_name="group", value_name="value", ignore_index=False)
    data = data.sort_index(kind="stable")
    original = data.copy(deep=True)
    expected = np.eye(3)
    for i, j in itertools.combinations(range(3), 2):
        expected[i, j] = expected[j, i] = stats.wilcoxon(
            samples[i], samples[j], method="approx", correction=correction, zero_method=zero_method
        ).pvalue
    expected = pd.DataFrame(expected, index=LABELS, columns=LABELS)

    actual = sp.posthoc_wilcoxon(
        data,
        val_col="value",
        group_col="group",
        sort=sort,
        method="approx",
        correction=correction,
        zero_method=zero_method,
    )

    assert_matrix(actual, expected, sorted(LABELS) if sort else LABELS)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_dunnett_t3_two_groups_matches_rounded_df_welch_distribution(sort, dtype):
    samples = [[1, 2, 4, 8, 9], [2, 5, 6, 7, 10, 12, 15]]
    data = long_data(samples, labels=["z", "a"], dtype=dtype)
    first, second = map(np.asarray, samples)
    v1, v2 = first.var(ddof=1) / len(first), second.var(ddof=1) / len(second)
    df = (v1 + v2) ** 2 / (v1**2 / (len(first) - 1) + v2**2 / (len(second) - 1))
    t = abs(stats.ttest_ind(*samples, equal_var=False).statistic)
    # Dunnett T3 intentionally rounds Welch's df to match PMCMRplus.
    expected = 2 * stats.t.sf(t, np.round(df))

    actual = sp.posthoc_dunnett_t3(data, val_col="value", group_col="group", sort=sort)

    np.testing.assert_allclose(actual.loc["z", "a"], expected, rtol=1e-11)


def pair_count_ranks(values):
    return np.array(
        [
            [1 + sum(x < y for x in row) + (sum(x == y for x in row) - 1) / 2 for y in row]
            for row in values
        ],
        dtype=float,
    )


@pytest.mark.parametrize(
    "function",
    [sp.posthoc_nemenyi_friedman, sp.posthoc_siegel_friedman, sp.posthoc_miller_friedman],
)
@pytest.mark.parametrize("seed", [11, 37, 101])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("incomplete", [False, True])
def test_tied_block_values_match_pair_count_rank_reference(function, seed, sort, incomplete):
    values = np.random.default_rng(seed).integers(0, 5, size=(9, 3)).astype(float)
    ranks = pair_count_ranks(values)
    n, k = values.shape
    z = np.abs(ranks.mean(axis=0)[:, None] - ranks.mean(axis=0)[None, :])
    z /= np.sqrt(k * (k + 1) / (6 * n))
    if function is sp.posthoc_nemenyi_friedman:
        expected = stats.studentized_range.sf(z * np.sqrt(2), k, np.inf)
    elif function is sp.posthoc_siegel_friedman:
        expected = 2 * stats.norm.sf(z)
    else:
        expected = stats.chi2.sf(z**2, k - 1)
    np.fill_diagonal(expected, 1)
    expected = pd.DataFrame(expected, index=LABELS, columns=LABELS)
    if incomplete:
        values = np.insert(values, 3, [10000, np.nan, -10000], axis=0)
    data = pd.DataFrame(values, columns=LABELS, dtype="Float64")
    original = data.copy(deep=True)

    actual = function(data, sort=sort)

    assert_matrix(actual, expected, sorted(LABELS) if sort else LABELS)
    pd.testing.assert_frame_equal(data, original)
