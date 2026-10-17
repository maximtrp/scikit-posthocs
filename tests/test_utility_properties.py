from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp


@pytest.mark.parametrize("seed", [2, 17, 53])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("input_type", ["list", "array", "series"])
@pytest.mark.parametrize("has_outlier", [False, True])
def test_grubbs_matches_scalar_reference_and_does_not_mutate_input(
    seed, alpha, input_type, has_outlier
):
    values = np.random.default_rng(seed).normal(size=20)
    if has_outlier:
        values[7] = 12
    data = values.tolist() if input_type == "list" else values.copy()
    if input_type == "series":
        data = pd.Series(values, index=np.arange(20)[::-1])
    original = np.asarray(data).copy()
    n = len(values)
    deviations = abs(values - values.mean())
    candidate = np.argmax(deviations)
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    critical = (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))
    reject = deviations[candidate] / values.std(ddof=1) > critical
    expected = np.delete(values, candidate) if reject else values

    actual = sp.outliers_grubbs(data, alpha=alpha)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_equal(sp.outliers_grubbs(data, alpha=alpha, hypo=True), reject)
    np.testing.assert_array_equal(data, original)


def gesd_reference(values, maximum, alpha):
    remaining = list(range(len(values)))
    candidates = []
    last_rejection = 0
    for step in range(maximum):
        sample = values[remaining]
        n = len(sample)
        deviations = abs(sample - sample.mean())
        candidate = int(np.argmax(deviations))
        statistic = deviations[candidate] / sample.std(ddof=1)
        t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        critical = (n - 1) * t / np.sqrt(n * (n - 2 + t**2))
        candidates.append(remaining.pop(candidate))
        if statistic > critical:
            last_rejection = step + 1
    mask = np.zeros(len(values), dtype=bool)
    mask[candidates[:last_rejection]] = True
    return mask


@pytest.mark.parametrize("seed", [5, 31, 73])
@pytest.mark.parametrize("maximum", [1, 3, 5])
@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_gesd_scalar_reference_preserves_input_positions_and_duplicate_values(seed, maximum, alpha):
    rng = np.random.default_rng(seed)
    values = np.round(rng.normal(size=30), 1)
    values[[2, 19, 24]] = [10, -12, 10]
    values = values[rng.permutation(len(values))]
    original = values.copy()
    expected = gesd_reference(values, maximum, alpha)

    mask = sp.outliers_gesd(values, outliers=maximum, alpha=alpha, hypo=True)
    filtered = sp.outliers_gesd(values, outliers=maximum, alpha=alpha)

    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(filtered, np.sort(values[~expected]))
    transformed_mask = sp.outliers_gesd(3 * values + 7, outliers=maximum, alpha=alpha, hypo=True)
    np.testing.assert_array_equal(transformed_mask, expected)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("seed", [3, 47, 89])
@pytest.mark.parametrize("coef", [1, 1.5, 3])
def test_iqr_return_forms_partition_input_and_commute_with_positive_affine_transform(seed, coef):
    values = np.random.default_rng(seed).normal(size=30)
    values[[4, 18]] = [-15, 20]
    original = values.copy()
    keep = sp.outliers_iqr(values, ret="indices", coef=coef)
    remove = sp.outliers_iqr(values, ret="outliers_indices", coef=coef)

    ordered = sorted(values)

    def quartile(fraction):
        position = (len(ordered) - 1) * fraction
        lo = int(position)
        return ordered[lo] + (position - lo) * (ordered[lo + 1] - ordered[lo])

    q1, q3 = quartile(0.25), quartile(0.75)
    expected_keep = (values >= q1 - coef * (q3 - q1)) & (values <= q3 + coef * (q3 - q1))
    np.testing.assert_array_equal(keep, np.flatnonzero(expected_keep))
    np.testing.assert_array_equal(remove, np.flatnonzero(~expected_keep))
    np.testing.assert_array_equal(np.sort(np.concatenate([keep, remove])), np.arange(len(values)))
    np.testing.assert_array_equal(sp.outliers_iqr(values, coef=coef), values[keep])
    np.testing.assert_array_equal(
        sp.outliers_iqr(values, ret="outliers", coef=coef), values[remove]
    )
    np.testing.assert_array_equal(sp.outliers_iqr(2 * values + 11, ret="indices", coef=coef), keep)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("seed", [13, 67])
@pytest.mark.parametrize("has_outlier", [False, True])
def test_tietjen_seeded_return_modes_and_affine_transform_agree(k, seed, has_outlier):
    values = np.random.default_rng(seed).normal(size=15) if has_outlier else np.linspace(-1, 1, 15)
    if has_outlier:
        values[:k] = np.arange(k) + 20
    original = values.copy()
    state = np.random.get_state()
    try:
        np.random.seed(947)
        reject = sp.outliers_tietjen(values, k=k, hypo=True)
        np.random.seed(947)
        filtered = sp.outliers_tietjen(values, k=k)
        np.random.seed(947)
        transformed = sp.outliers_tietjen(2 * values + 3, k=k)
    finally:
        np.random.set_state(state)

    np.testing.assert_equal(reject, has_outlier)
    np.testing.assert_equal(len(filtered), len(values) - (k if reject else 0))
    np.testing.assert_allclose(transformed, 2 * filtered + 3)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("values", [[1], [1, 1, 1], [0, 0.5, 1], [0.03, 0.9, 0.03, 0.2, 0.001]])
def test_global_tests_match_external_references_and_ignore_permutation(values):
    values = np.asarray(values, dtype=float)
    original = values.copy()
    ordered = sorted(values)
    simes = min(len(values) * p / rank for rank, p in enumerate(ordered, 1))
    with np.errstate(divide="ignore"):
        fisher = stats.combine_pvalues(values, method="fisher")
        for data in [values, values[::-1], values.tolist()]:
            np.testing.assert_allclose(sp.global_simes_test(data), simes)
            pvalue, statistic = sp.global_f_test(data, stat=True)
            np.testing.assert_allclose([pvalue, statistic], [fisher.pvalue, fisher.statistic])
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("size", [1, 3, 6, 8])
@pytest.mark.parametrize("seed", [0, 3, 11, 37, 71])
def test_compact_letters_encode_exactly_the_nonsignificant_pairs(size, seed):
    rng = np.random.default_rng(seed)
    same = np.triu(rng.random((size, size)) > 0.5, 1)
    same |= same.T
    np.fill_diagonal(same, True)
    labels = [f"group-{i}" for i in rng.permutation(size)]
    values = pd.DataFrame(np.where(same, 0.5, 0.001), index=labels, columns=labels)
    original = values.copy(deep=True)

    actual = sp.compact_letter_display(values)

    np.testing.assert_array_equal(actual.index.tolist(), labels)
    letters = [set(text.replace(" ", "")) for text in actual]
    represented = np.array([[bool(first & second) for second in letters] for first in letters])
    np.testing.assert_array_equal(represented, same)
    pd.testing.assert_frame_equal(values, original)


@pytest.mark.parametrize("maxiter", [0, 100])
def test_compact_letters_reports_alphabet_exhaustion(maxiter):
    warning = (
        pytest.warns(RuntimeWarning, match="did not converge") if maxiter == 0 else nullcontext()
    )
    with warning, pytest.raises(ValueError, match="Too many letter groups"):
        sp.compact_letter_display(np.eye(53), maxiter=maxiter)


def test_compact_letters_reports_insufficient_iterations():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = sp.compact_letter_display([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]], maxiter=0)
    np.testing.assert_equal(len(result), 3)


@pytest.mark.parametrize("size", [5, 6])
def test_compact_letters_nonconvergence_preserves_all_pair_relationships(size):
    values = np.eye(size)
    for i in range(5):
        values[i, (i + 1) % 5] = values[(i + 1) % 5, i] = 0.5
    names = [f"group-{i}" for i in range(size)]
    data = pd.DataFrame(values, index=names, columns=names)
    original = data.copy(deep=True)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = sp.compact_letter_display(data, maxiter=0)
    letters = [set(text.replace(" ", "")) for text in result]
    represented = np.array([[bool(a & b) for b in letters] for a in letters])
    np.testing.assert_array_equal(represented, values >= 0.05)
    np.testing.assert_array_equal(result.index, names)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("maxiter", [0, 1, 100])
@pytest.mark.filterwarnings("ignore:compact_letter_display did not converge:RuntimeWarning")
def test_compact_letters_exact_relationships_for_every_five_group_graph(maxiter):
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    for mask in range(1 << len(pairs)):
        same = np.eye(5, dtype=bool)
        for bit, (i, j) in enumerate(pairs):
            same[i, j] = same[j, i] = bool(mask & (1 << bit))
        result = sp.compact_letter_display(np.where(same, 0.5, 0.001), maxiter=maxiter)
        letters = [set(text.replace(" ", "")) for text in result]
        represented = np.array([[bool(a & b) for b in letters] for a in letters])
        np.testing.assert_array_equal(represented, same, err_msg=f"graph mask={mask}")


@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("upper", [False, True])
def test_sign_table_retains_axes_and_does_not_mutate_pvalues(lower, upper):
    values = pd.DataFrame(
        [[1, 0, 0.02], [0, 1, 0.2], [0.02, 0.2, 1]],
        index=["z", "a", "m"],
        columns=["z", "a", "m"],
    )
    original = values.copy(deep=True)
    if not lower and not upper:
        with pytest.raises(ValueError, match="Either lower or upper"):
            sp.sign_table(values, lower=lower, upper=upper)
    else:
        actual = sp.sign_table(values, lower=lower, upper=upper)
        pd.testing.assert_index_equal(actual.index, values.index)
        pd.testing.assert_index_equal(actual.columns, values.columns)
        np.testing.assert_array_equal(np.diag(actual), ["-"] * 3)
        np.testing.assert_equal(actual.iloc[0, 1], "***" if upper else "")
        np.testing.assert_equal(actual.iloc[1, 0], "***" if lower else "")
    pd.testing.assert_frame_equal(values, original)
