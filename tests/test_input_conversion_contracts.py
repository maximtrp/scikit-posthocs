import numpy as np
import pandas as pd
import pytest

import scikit_posthocs as sp
import scikit_posthocs._posthocs as posthocs


@pytest.mark.parametrize("value_position", [0, 1])
def test_explicit_array_column_positions_keep_zero_as_a_valid_position(value_position):
    data = np.array([[2, 20], [2, 30], [1, 40], [1, 50]], dtype=float)
    original = data.copy()

    actual, value_col, group_col = sp.__convert_to_df(
        data, val_id=value_position, group_id=1 - value_position
    )

    np.testing.assert_array_equal(actual[value_col], data[:, value_position])
    np.testing.assert_array_equal(actual[group_col], data[:, 1 - value_position])
    np.testing.assert_array_equal(data, original)


def test_ambiguous_long_array_requires_explicit_positions():
    data = np.array([[1, 11], [2, 12], [3, 13]], dtype=float)

    with pytest.raises(ValueError, match="Cannot infer input format"):
        sp.__convert_to_df(data)


@pytest.mark.parametrize("value_position", [0, 1])
def test_two_by_two_explicit_array_positions_still_select_long_input(value_position):
    data = np.array([[1, 11], [2, 12]])

    actual, value_col, group_col = sp.__convert_to_df(
        data, val_id=value_position, group_id=1 - value_position
    )

    np.testing.assert_array_equal(actual[value_col], data[:, value_position])
    np.testing.assert_array_equal(actual[group_col], data[:, 1 - value_position])


def test_dataframe_converter_reports_missing_requested_column():
    data = pd.DataFrame({"value": [1, 2], "group": ["a", "b"]})

    with pytest.raises(ValueError, match="Specify correct column names"):
        sp.__convert_to_df(data, val_col="missing", group_col="group")


@pytest.mark.parametrize("missing", ["y_col", "group_col", "block_col"])
def test_melted_block_input_requires_explicit_measurement_treatment_and_block_columns(missing):
    kwargs = {"y_col": "value", "group_col": "treatment", "block_col": "subject"}
    kwargs.pop(missing)

    with pytest.raises(ValueError, match="should be explicitly specified"):
        sp.__convert_to_block_df(pd.DataFrame(), melted=True, **kwargs)


def test_melted_repeated_subject_labels_require_distinct_block_ids():
    data = pd.DataFrame({"value": [1, 2], "treatment": ["a", "b"], "subject": ["s1", "s1"]})

    with pytest.raises(ValueError, match="block_id_col.*explicitly specified"):
        sp.__convert_to_block_df(
            data, y_col="value", group_col="treatment", block_col="subject", melted=True
        )


@pytest.mark.parametrize(
    ("data", "melted", "sort"),
    [
        (np.array([1, 2, 3]), False, False),
        ([[1, 2], [3]], False, False),
        (np.array([[1, 2], [3, 4]], dtype=object), False, False),
        (np.array([[1, np.nan], [3, 4]]), False, False),
        (np.array([[1, 2], [3, 4]]), True, False),
        (pd.DataFrame([[1, 2], [3, 4]]), False, True),
        (pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"]), False, False),
        (pd.DataFrame([[1, 2], [3, 4]], columns=["a", None]), False, False),
    ],
)
def test_optimized_block_gate_rejects_inputs_requiring_conversion(data, melted, sort):
    result = posthocs.__complete_block_matrix(data, melted=melted, sort=sort)
    np.testing.assert_equal(result is None, True)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", "Int64", "UInt64"])
def test_block_converter_uses_row_ids_without_losing_measurement_dtype(dtype):
    data = pd.DataFrame(
        {
            "y": pd.array([1, None, 3], dtype=dtype),
            "block_ids": pd.array([4, 5, 6], dtype=dtype),
        }
    )
    data.index = ["subject", "subject", "other"]
    original = data.copy(deep=True)

    actual, value_col, group_col, block_col, block_id_col = sp.__convert_to_block_df(data)

    np.testing.assert_equal(actual[value_col].dtype, pd.api.types.pandas_dtype(dtype))
    np.testing.assert_equal(len(actual), data.size)
    np.testing.assert_array_equal(actual[group_col], ["y"] * 3 + ["block_ids"] * 3)
    np.testing.assert_array_equal(actual[block_id_col], [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(actual[block_col], data.index.tolist() * 2)
    np.testing.assert_allclose(
        actual[value_col].to_numpy(dtype=float, na_value=np.nan), [1, np.nan, 3, 4, 5, 6]
    )
    pd.testing.assert_frame_equal(data, original)
