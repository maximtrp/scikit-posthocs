import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp
import scikit_posthocs._posthocs as posthocs


@pytest.mark.parametrize("dtype", ["Float32", "Float64", "Int64", "UInt64", "boolean"])
@pytest.mark.parametrize("missing", [False, True])
def test_nullable_numeric_array_has_native_dtype_and_preserves_missing_values(dtype, missing):
    values = [True, False, True] if dtype == "boolean" else [1, 2, 3]
    if missing:
        values[1] = None
    data = pd.Series(values, dtype=dtype)
    original = data.copy(deep=True)
    expected_dtype = data.dtype.numpy_dtype
    if missing and expected_dtype.kind in "biu":
        expected_dtype = np.dtype(float)
    expected = np.array(
        [np.nan if value is None else value for value in values], dtype=expected_dtype
    )

    actual = posthocs.__numeric_array(data)

    np.testing.assert_equal(actual.dtype, expected_dtype)
    np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_series_equal(data, original)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint64", "bool"])
def test_native_numeric_array_does_not_change_dtype(dtype):
    data = pd.Series([0, 1, 0], dtype=dtype)

    actual = posthocs.__numeric_array(data)

    np.testing.assert_equal(actual.dtype, data.dtype)
    np.testing.assert_array_equal(actual, data.to_numpy())


@pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
@pytest.mark.parametrize("sort", [False, True])
def test_nullable_large_integers_do_not_gain_ties_from_float_conversion(dtype, sort):
    offset = 2**53
    first = np.array([offset, offset + 1, offset + 4], dtype=dtype.lower())
    second = np.array([offset + 2, offset + 3, offset + 5], dtype=dtype.lower())
    data = pd.DataFrame(
        {
            "value": pd.array(np.concatenate([first, second]), dtype=dtype),
            "group": ["z"] * 3 + ["a"] * 3,
        }
    )
    original = data.copy(deep=True)

    actual = sp.posthoc_mannwhitney(data, val_col="value", group_col="group", sort=sort)

    np.testing.assert_allclose(actual.loc["z", "a"], stats.mannwhitneyu(first, second).pvalue)
    native = posthocs.__numeric_array(data["value"])
    np.testing.assert_equal(native.dtype, np.dtype(dtype.lower()))
    np.testing.assert_array_equal(native, np.concatenate([first, second]))
    pd.testing.assert_frame_equal(data, original)


def test_nullable_missing_ttest_observations_keep_finite_input_error():
    data = pd.DataFrame(
        {"value": pd.array([1, None, 3, 5, 7, 9], dtype="Float64"), "group": ["a"] * 3 + ["b"] * 3}
    )

    with pytest.raises(ValueError, match="All observations must be finite"):
        sp.posthoc_ttest(data, val_col="value", group_col="group")


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_nullable_wilcoxon_matches_scipy_and_rejects_missing_pairs(dtype):
    first, second = [1, 2, 4, 5, 8], [2, 4, 7, 8, 12]
    data = pd.DataFrame(
        {"value": pd.array(first + second, dtype=dtype), "group": ["a"] * 5 + ["b"] * 5}
    )
    original = data.copy(deep=True)

    actual = sp.posthoc_wilcoxon(data, val_col="value", group_col="group")

    np.testing.assert_allclose(actual.loc["a", "b"], stats.wilcoxon(first, second).pvalue)
    pd.testing.assert_frame_equal(data, original)
    data.loc[0, "value"] = pd.NA
    with pytest.raises(ValueError, match="missing paired observations"):
        sp.posthoc_wilcoxon(data, val_col="value", group_col="group")
