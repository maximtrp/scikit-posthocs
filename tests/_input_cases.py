"""Shared, deterministic cases for exercising public statistical input paths."""

import numpy as np
import pandas as pd
import pytest

import scikit_posthocs as sp


def case(function, **kwargs):
    suffix = "-".join(f"{key}={value}" for key, value in kwargs.items())
    return pytest.param(function, kwargs, id=function.__name__ + (f"-{suffix}" if suffix else ""))


INDEPENDENT_POSTHOCS = [
    case(sp.posthoc_conover, p_adjust="holm"),
    case(sp.posthoc_dunn, p_adjust="holm"),
    case(sp.posthoc_nemenyi),
    case(sp.posthoc_nemenyi, dist="tukey"),
    case(sp.posthoc_anderson),
    case(sp.posthoc_vanwaerden, p_adjust="holm"),
    case(sp.posthoc_ttest, p_adjust="holm"),
    case(sp.posthoc_ttest, pool_sd=True, p_adjust="holm"),
    case(sp.posthoc_ttest, equal_var=False),
    case(sp.posthoc_tukey_hsd),
    case(sp.posthoc_mannwhitney, p_adjust="holm"),
    case(sp.posthoc_scheffe),
    case(sp.posthoc_tamhane),
    case(sp.posthoc_tamhane, welch=False),
    case(sp.posthoc_tukey),
    case(sp.posthoc_dscf),
    case(sp.posthoc_games_howell),
    case(sp.posthoc_dunnett_t3),
    case(sp.posthoc_lsd, p_adjust="holm"),
    case(sp.posthoc_snk),
    case(sp.posthoc_duncan),
    case(sp.posthoc_median, p_adjust="holm"),
    case(sp.posthoc_steel, control=1, p_adjust="holm"),
    case(sp.posthoc_dunnett, control=1),
]

BLOCK_POSTHOCS = [
    case(sp.posthoc_nemenyi_friedman),
    case(sp.posthoc_conover_friedman, p_adjust="holm"),
    case(sp.posthoc_conover_friedman, p_adjust="single-step"),
    case(sp.posthoc_siegel_friedman, p_adjust="holm"),
    case(sp.posthoc_miller_friedman),
    case(sp.posthoc_durbin, p_adjust="holm"),
    case(sp.posthoc_quade, dist="t", p_adjust="holm"),
    case(sp.posthoc_quade, dist="normal"),
    case(sp.posthoc_demsar, control=0, p_adjust="holm"),
]

LABELS = ["z", "a", "m"]


def independent_samples():
    # Overlap, unequal means/variances and no ties; no random or degenerate inputs.
    return [[1.0, 3.0, 4.0, 6.0, 9.0], [2.0, 5.0, 7.0, 8.0, 10.0], [0.5, 2.5, 6.5, 9.5, 12.5]]


def long_data(samples, labels=LABELS, dtype="float64"):
    data = pd.DataFrame(
        {
            "value": [value for sample in samples for value in sample],
            "group": [label for label, sample in zip(labels, samples) for _ in sample],
        }
    )
    data["value"] = data["value"].astype(dtype)
    return data


def block_input(values, input_type):
    """Build block input forms independently of the package converters."""
    if input_type == "array":
        return values, {}
    wide = pd.DataFrame(values, dtype="Float64" if input_type == "nullable" else "float64")
    if input_type != "melted":
        return wide, {}
    wide.index.name = "subject"
    long = wide.reset_index().melt(id_vars="subject", var_name="treatment", value_name="value")
    return long, {
        "y_col": "value",
        "group_col": "treatment",
        "block_col": "subject",
        "block_id_col": "subject",
        "melted": True,
    }


def assert_matrix(actual, expected, labels, many_to_one=False):
    """Check structure before alignment, so reindexing cannot conceal lost groups."""
    np.testing.assert_equal(actual.shape, expected.shape)
    np.testing.assert_equal(expected.shape, (len(labels), len(labels)))
    np.testing.assert_array_equal(actual.index, labels)
    np.testing.assert_array_equal(actual.columns, labels)
    np.testing.assert_array_equal(np.diag(actual), np.ones(len(labels)))
    np.testing.assert_allclose(actual, actual.T, rtol=1e-11, atol=1e-12, equal_nan=True)
    finite = actual.to_numpy()[np.isfinite(actual.to_numpy())]
    np.testing.assert_equal(((finite >= 0) & (finite <= 1)).all(), True)
    if not many_to_one:
        np.testing.assert_equal(np.isfinite(actual.to_numpy()).all(), True)
    np.testing.assert_allclose(
        actual, expected.loc[labels, labels], rtol=1e-10, atol=1e-12, equal_nan=True
    )
