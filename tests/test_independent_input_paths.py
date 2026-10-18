import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scikit_posthocs as sp
from tests._input_cases import (
    INDEPENDENT_POSTHOCS,
    LABELS,
    assert_matrix,
    independent_samples,
    long_data,
)


@pytest.fixture(autouse=True)
def deterministic_dunnett(monkeypatch):
    # SciPy's multivariate integration is randomized, even for identical inputs.
    original = stats.dunnett
    key = "rng" if "rng" in inspect.signature(original).parameters else "random_state"

    def seeded_dunnett(*args, **kwargs):
        kwargs[key] = np.random.default_rng(731)
        return original(*args, **kwargs)

    monkeypatch.setattr(stats, "dunnett", seeded_dunnett)


@pytest.mark.parametrize(("function", "kwargs"), INDEPENDENT_POSTHOCS)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("input_type", ["array", "long", "nullable", "repeated_index"])
def test_independent_input_forms_match_labeled_list(function, kwargs, sort, input_type):
    samples = independent_samples()
    expected = function(samples, **kwargs)
    expected.index = LABELS
    expected.columns = LABELS
    if input_type == "array":
        data = np.asarray(samples)
        call_kwargs = kwargs
        labels = [1, 2, 3]
        expected.index = labels
        expected.columns = labels
        original = data.copy()
    else:
        data = long_data(samples, dtype="Float64" if input_type == "nullable" else "float64")
        if input_type == "repeated_index":
            data.index = np.tile(np.arange(5), 3)
        original = data.copy(deep=True)
        call_kwargs = {**kwargs, "val_col": "value", "group_col": "group"}
        if "control" in kwargs:
            call_kwargs["control"] = LABELS[kwargs["control"] - 1]
        labels = sorted(LABELS) if sort else LABELS

    actual = function(data, sort=sort, **call_kwargs)

    assert_matrix(actual, expected, labels, many_to_one="control" in kwargs)
    if isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(data, original)
    else:
        np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(("function", "kwargs"), INDEPENDENT_POSTHOCS)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("columns", [("y", "groups"), ("ranks", "group"), (11, 17)])
def test_independent_column_names_are_not_reserved(function, kwargs, sort, columns):
    data = long_data(independent_samples())
    call_kwargs = {**kwargs, "val_col": "value", "group_col": "group"}
    if "control" in kwargs:
        call_kwargs["control"] = LABELS[kwargs["control"] - 1]
    expected = function(data, sort=sort, **call_kwargs)
    renamed = data.rename(columns={"value": columns[0], "group": columns[1]})
    original = renamed.copy(deep=True)
    actual = function(
        renamed, sort=sort, **{**call_kwargs, "val_col": columns[0], "group_col": columns[1]}
    )

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(renamed, original)


@pytest.mark.parametrize(("function", "kwargs"), INDEPENDENT_POSTHOCS)
@pytest.mark.parametrize("seed", [3, 29])
def test_independent_rows_and_unused_columns_do_not_change_results(function, kwargs, seed):
    data = long_data(independent_samples())
    call_kwargs = {**kwargs, "val_col": "value", "group_col": "group", "sort": True}
    if "control" in kwargs:
        call_kwargs["control"] = LABELS[kwargs["control"] - 1]
    expected = function(data, **call_kwargs)
    permuted = data.iloc[np.random.default_rng(seed).permutation(len(data))].copy()
    permuted["unused"] = np.nan
    original = permuted.copy(deep=True)

    actual = function(permuted, **call_kwargs)

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(permuted, original)


@pytest.mark.parametrize(("function", "kwargs"), INDEPENDENT_POSTHOCS)
@pytest.mark.parametrize("sort", [False, True])
def test_unused_categories_are_not_treatments_and_ordered_categories_align_values(
    function, kwargs, sort
):
    data = long_data(independent_samples())
    call_kwargs = {**kwargs, "val_col": "value", "group_col": "group"}
    if "control" in kwargs:
        call_kwargs["control"] = LABELS[kwargs["control"] - 1]
    expected = function(data, sort=False, **call_kwargs)
    category_order = ["m", "z", "a"]
    if function is sp.posthoc_dunnett:
        # Also fix the integration coordinate order: permuting the randomized
        # multivariate integral changes its numerical approximation slightly.
        samples = independent_samples()
        expected = function([samples[2], samples[0], samples[1]], control=2)
        expected.index = category_order
        expected.columns = category_order
    data["group"] = pd.Categorical(
        data["group"], categories=[*category_order, "unused"], ordered=True
    )
    original = data.copy(deep=True)

    actual = function(data, sort=sort, **call_kwargs)

    assert_matrix(actual, expected, category_order if sort else LABELS, "control" in kwargs)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_npm_ordered_input_forms_match_seeded_reference(sort, alternative):
    samples = independent_samples()
    data = long_data(samples, dtype="Float64")
    order = np.argsort(LABELS) if sort else np.arange(len(LABELS))
    # This test is ordered: sorting changes the hypothesis, not just matrix labels.
    state = np.random.get_state()
    try:
        np.random.seed(619)
        expected = sp.posthoc_npm_test(
            [samples[i] for i in order], alternative=alternative, nperm=100
        )
        labels = [LABELS[i] for i in order]
        expected.index = labels
        expected.columns = labels
        np.random.seed(619)
        actual = sp.posthoc_npm_test(
            data,
            val_col="value",
            group_col="group",
            sort=sort,
            alternative=alternative,
            nperm=100,
        )
    finally:
        np.random.set_state(state)

    assert_matrix(actual, expected, labels)
