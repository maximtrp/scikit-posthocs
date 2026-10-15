import numpy as np
import pandas as pd
import pytest

import scikit_posthocs as sp
import scikit_posthocs._posthocs as posthocs
from tests._input_cases import BLOCK_POSTHOCS, LABELS, assert_matrix


@pytest.fixture
def block_values():
    return np.array(
        [
            [31, 27, 24],
            [31, 28, 31],
            [45, 29, 46],
            [21, 18, 48],
            [42, 36, 46],
            [32, 17, 40],
            [4, 4, 4],
        ],
        dtype=float,
    )


def melted_blocks(data):
    # Positional IDs distinguish repeated subject labels. Build the long input
    # independently of the package converter and retain extension measurement dtypes.
    long = data.copy(deep=True)
    long.index.name = "subject"
    long = long.reset_index()
    long["row_id"] = np.arange(len(data))
    long = long.melt(id_vars=["subject", "row_id"], var_name="treatment", value_name="value")
    return long, {
        "y_col": "value",
        "group_col": "treatment",
        "block_col": "subject",
        "block_id_col": "row_id",
        "melted": True,
    }


@pytest.mark.parametrize(("function", "kwargs"), BLOCK_POSTHOCS)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["wide", "nullable", "melted", "melted_nullable"])
@pytest.mark.parametrize("label_type", ["strings", "integers"])
def test_block_input_forms_preserve_distinct_repeated_subjects(
    function, kwargs, sort, input_type, label_type, block_values
):
    labels = LABELS if label_type == "strings" else [30, 10, 20]
    wide = pd.DataFrame(
        block_values,
        columns=labels,
        index=["s3", "s1", "s3", "s2", "s2", "s4", "s4"],
        dtype="Float64" if "nullable" in input_type else "float64",
    )
    expected = function(block_values, **kwargs)
    expected.index = labels
    expected.columns = labels
    data, input_kwargs = melted_blocks(wide) if "melted" in input_type else (wide, {})
    if "control" in kwargs:
        input_kwargs["control"] = labels[kwargs["control"]]
    original = data.copy(deep=True)

    actual = function(data, sort=sort, **{**kwargs, **input_kwargs})

    assert_matrix(actual, expected, sorted(labels) if sort else labels, "control" in kwargs)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(("function", "kwargs"), BLOCK_POSTHOCS)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("seed", [7, 41])
def test_block_row_and_treatment_permutations_preserve_alignment(
    function, kwargs, sort, seed, block_values
):
    wide = pd.DataFrame(block_values, columns=LABELS)
    input_kwargs = {**kwargs, **({"control": "z"} if "control" in kwargs else {})}
    expected = function(wide, **input_kwargs)
    rng = np.random.default_rng(seed)
    permuted = wide.iloc[rng.permutation(len(wide)), [2, 0, 1]].copy()
    original = permuted.copy(deep=True)

    actual = function(permuted, sort=sort, **input_kwargs)

    labels = sorted(LABELS) if sort else permuted.columns.tolist()
    assert_matrix(actual, expected, labels, "control" in kwargs)
    pd.testing.assert_frame_equal(permuted, original)


@pytest.mark.parametrize(
    ("function", "kwargs"), [p for p in BLOCK_POSTHOCS if p.values[0] is not sp.posthoc_durbin]
)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["wide", "nullable", "melted"])
def test_incomplete_block_is_removed_before_ranking_all_treatments(
    function, kwargs, sort, input_type, block_values
):
    values = np.insert(block_values, 2, [10000, np.nan, -10000], axis=0)
    wide = pd.DataFrame(
        values, columns=LABELS, dtype="Float64" if input_type == "nullable" else "float64"
    )
    expected = function(block_values, **kwargs)
    expected.index = LABELS
    expected.columns = LABELS
    data, input_kwargs = melted_blocks(wide) if input_type == "melted" else (wide, {})
    if "control" in kwargs:
        input_kwargs["control"] = "z"
    original = data.copy(deep=True)

    actual = function(data, sort=sort, **{**kwargs, **input_kwargs})

    assert_matrix(actual, expected, sorted(LABELS) if sort else LABELS, "control" in kwargs)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(("function", "kwargs"), BLOCK_POSTHOCS)
def test_each_block_posthoc_exercises_fast_and_fallback_paths(
    function, kwargs, block_values, monkeypatch
):
    original = posthocs.__complete_block_matrix
    selected = []

    def traced_matrix(*args, **kwargs):
        result = original(*args, **kwargs)
        selected.append(result is not None)
        return result

    monkeypatch.setattr(posthocs, "__complete_block_matrix", traced_matrix)
    wide = pd.DataFrame(block_values, columns=LABELS)
    input_kwargs = {**kwargs, **({"control": "z"} if "control" in kwargs else {})}

    dense = function(wide, **input_kwargs)
    sorted_result = function(wide, sort=True, **input_kwargs)
    nullable = function(wide.astype("Float64"), **input_kwargs)

    np.testing.assert_equal(selected, [True, False, False])
    pd.testing.assert_frame_equal(dense, sorted_result.loc[LABELS, LABELS])
    pd.testing.assert_frame_equal(dense, nullable)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("input_type", ["wide", "nullable", "melted"])
def test_page_hypothesis_uses_treatment_order_not_subject_labels(
    sort, alternative, input_type, block_values
):
    wide = pd.DataFrame(
        block_values,
        columns=LABELS,
        index=["s3", "s1", "s3", "s2", "s2", "s4", "s4"],
        dtype="Float64" if input_type == "nullable" else "float64",
    )
    data, input_kwargs = melted_blocks(wide) if input_type == "melted" else (wide, {})
    order = np.argsort(LABELS) if sort else np.arange(len(LABELS))
    expected = sp.test_page(block_values[:, order], alternative=alternative)
    original = data.copy(deep=True)

    actual = sp.test_page(data, sort=sort, alternative=alternative, **input_kwargs)

    np.testing.assert_allclose(actual, expected)
    pd.testing.assert_frame_equal(data, original)
