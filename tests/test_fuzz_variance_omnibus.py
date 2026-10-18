"""Seeded scale, missing-group, and design checks for Hartley's test."""

# Fixed chi-square draws separate input/statistic errors from Monte Carlo noise.
# Extreme scales, ndarray shapes, nullable values, and missing-only groups are
# separate from invalid designs that must fail before sampling.

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp


@pytest.mark.parametrize("seed", [0, 7, 41])
@pytest.mark.parametrize("shape", [(2, 2), (3, 4), (4, 7)])
@pytest.mark.parametrize("form", ["list", "array", "long", "nullable"])
@pytest.mark.parametrize("scale", [1.0, -8.0, 1e-180, 1e180])
def test_hartley_seeded_samples_are_scale_and_input_form_invariant(
    seed, shape, form, scale, monkeypatch
):
    rng = np.random.default_rng(seed)
    k, n = shape
    values = rng.normal(size=shape) * np.arange(1, k + 1)[:, None]
    variances = values.var(axis=1, ddof=1)
    expected_stat = variances.max() / variances.min()
    expected_df = n - 1
    draws = rng.chisquare(expected_df, size=(113, k))
    expected_p = np.mean(draws.max(axis=1) / draws.min(axis=1) >= expected_stat)

    def fixed_draws(df, size):
        np.testing.assert_equal(df, expected_df)
        np.testing.assert_equal(size, draws.shape)
        return draws.copy()

    monkeypatch.setattr(stats.chi2, "rvs", fixed_draws)
    transformed = values * scale
    kwargs = {}
    if form == "list":
        data = transformed.tolist()
    elif form == "array":
        data = transformed.copy()
    else:
        data = pd.DataFrame(
            [
                (value, f"group-{group}")
                for group, sample in enumerate(transformed)
                for value in sample
            ],
            columns=["value", "group"],
        )
        data.index = np.arange(len(data)) % 3
        if form == "nullable":
            data["value"] = data["value"].astype("Float64")
        kwargs = {"val_col": "value", "group_col": "group"}
    original = data.copy(deep=True) if isinstance(data, pd.DataFrame) else np.asarray(data).copy()

    with np.errstate(all="raise"):
        actual = sp.test_hartley(data, n_perm=len(draws), **kwargs)

    np.testing.assert_allclose(actual, [expected_p, expected_stat, expected_df], rtol=1e-12)
    if isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(data, original)
    else:
        np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("sort", [False, True])
def test_hartley_missing_only_and_unused_groups_do_not_change_the_design(sort, monkeypatch):
    data = pd.DataFrame(
        {
            "value": [1, 2, 3, 2, 4, 6, np.nan, np.nan],
            "group": ["z"] * 3 + ["a"] * 3 + ["missing"] * 2,
        }
    )
    data["value"] = data["value"].astype("Float64")
    data["group"] = pd.Categorical(data["group"], categories=["a", "z", "missing", "unused"])
    original = data.copy(deep=True)
    draws = np.random.default_rng(731).chisquare(2, size=(19, 2))
    monkeypatch.setattr(stats.chi2, "rvs", lambda df, size: draws.copy())
    expected = sp.test_hartley([[1, 2, 3], [2, 4, 6]], n_perm=19)

    actual = sp.test_hartley(data, val_col="value", group_col="group", n_perm=19, sort=sort)

    np.testing.assert_allclose(actual, expected)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "samples",
    [[], [[1, 2, 3]], [[1], [2, 3]], [[1, 1], [2, 2]], [[np.nan], [np.nan]], [[1, 2], [3, np.inf]]],
)
def test_hartley_invalid_designs_fail_before_simulation(samples, monkeypatch):
    def unexpected_draw(*args, **kwargs):
        pytest.fail("An invalid design must not sample the null distribution")

    monkeypatch.setattr(stats.chi2, "rvs", unexpected_draw)
    with np.errstate(all="raise"), pytest.raises(ValueError, match="groups|observations|variance"):
        sp.test_hartley(samples, n_perm=19)


def test_hartley_one_zero_variance_has_infinite_ratio_without_sampling(monkeypatch):
    def unexpected_draw(*args, **kwargs):
        pytest.fail("An infinite F-ratio does not require simulation")

    monkeypatch.setattr(stats.chi2, "rvs", unexpected_draw)
    with np.errstate(all="raise"):
        actual = sp.test_hartley([[1, 1, 1], [2, 4, 6]], n_perm=19)

    np.testing.assert_equal(actual, [0.0, np.inf, 2])
