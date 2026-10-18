import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import scikit_posthocs as sp


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("input_type", ["array", "dataframe"])
def test_sign_plot_preserves_values_and_encodes_heatmap_categories(flat, labels, input_type):
    values = np.array([[1, 0.0005, 0.05], [0.0005, 1, 0.005], [0.05, 0.005, 1]])
    expected = np.array([[-1, 1, 0], [1, -1, 2], [0, 2, -1]])
    if flat:
        values = np.array([[-1, 1, 0], [1, -1, 1], [0, 1, -1]], dtype=int)
        expected = values.copy()
    names = ["z", "a", "m"]
    data = pd.DataFrame(values, index=names, columns=names) if input_type == "dataframe" else values
    original = copy.deepcopy(data)
    figure, ax = plt.subplots()
    try:
        result = sp.sign_plot(
            data,
            g=names,
            flat=flat,
            labels=labels,
            ax=ax,
            cbar=False,
            vmin=100,
            vmax=200,
            center=100,
        )
        returned_ax = result if flat else result[0]
        np.testing.assert_equal(returned_ax is ax, True)
        encoded = np.asarray(ax.collections[0].get_array()).reshape(3, 3)
        np.testing.assert_array_equal(encoded, expected)
        if not flat:
            np.testing.assert_equal(result[1].ax.figure is figure, True)
        if isinstance(data, pd.DataFrame):
            pd.testing.assert_frame_equal(data, original)
        else:
            np.testing.assert_array_equal(data, original)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("left_only", [False, True])
@pytest.mark.parametrize(
    "palette_type", ["estimator_dict", "estimator_list", "hue_default", "hue_list", "hue_dict"]
)
@pytest.mark.parametrize("cd", [None, 0.5])
def test_cd_diagram_cliques_colors_and_coordinates_follow_labels(left_only, palette_type, cd):
    ranks = pd.Series({"z": 2.5, "a": 1.0, "m": 4.0, "c": 3.0})
    names = ranks.index.tolist()
    matrix = pd.DataFrame(np.eye(4) + 0.001 * (1 - np.eye(4)), index=names, columns=names)
    matrix.loc["a", "z"] = matrix.loc["z", "a"] = 0.5
    matrix.loc["c", "m"] = matrix.loc["m", "c"] = 0.5
    # The significance matrix need not have the same row order as ranks.
    matrix = matrix.loc[["m", "a", "c", "z"], ["m", "a", "c", "z"]]
    original_ranks, original_matrix = ranks.copy(), matrix.copy(deep=True)
    hue = {"z": "one", "a": "one", "m": "two", "c": "two"}
    if palette_type == "estimator_dict":
        kwargs = {"color_palette": dict(zip(names, ["red", "blue", "green", "purple"]))}
        expected_colors = kwargs["color_palette"]
    elif palette_type == "estimator_list":
        kwargs = {"color_palette": ["red", "blue", "green", "purple"]}
        expected_colors = dict(zip(ranks.sort_values().index, kwargs["color_palette"]))
    elif palette_type == "hue_default":
        kwargs = {"hue": hue}
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        expected_colors = {name: colors[0 if group == "one" else 1] for name, group in hue.items()}
    else:
        palette = ["red", "blue"] if palette_type == "hue_list" else {"two": "red", "one": "blue"}
        kwargs = {"hue": hue, "hue_order": ["two", "one"], "color_palette": palette}
        expected_colors = {name: "blue" if group == "one" else "red" for name, group in hue.items()}
    original_kwargs = copy.deepcopy(kwargs)
    figure, ax = plt.subplots()
    try:
        artists = sp.critical_difference_diagram(
            ranks,
            matrix,
            ax=ax,
            cd=cd,
            left_only=left_only,
            label_fmt_left="{label}",
            label_fmt_right="{label}",
            **kwargs,
        )

        np.testing.assert_equal(
            [len(artists[key]) for key in ["markers", "elbows", "labels"]], [4, 4, 4]
        )
        np.testing.assert_equal(len(artists["crossbars"]), 2)
        for marker, label in zip(artists["markers"], artists["labels"]):
            name = label.get_text()
            np.testing.assert_allclose(marker.get_offsets(), [[ranks[name], 0]])
            np.testing.assert_allclose(marker.get_facecolors()[0], to_rgba(expected_colors[name]))
        intervals = sorted(
            (min(line[0].get_xdata()), max(line[0].get_xdata())) for line in artists["crossbars"]
        )
        np.testing.assert_equal(intervals, [(1.0, 2.5), (3.0, 4.0)])
        # Disjoint nonsignificant cliques should share a level, without overlapping.
        for line in artists["crossbars"]:
            np.testing.assert_array_equal(line[0].get_ydata(), [-1, -1])
        np.testing.assert_equal(
            any(text.get_text() == "CD = 0.5" for text in ax.texts), cd is not None
        )
        pd.testing.assert_series_equal(ranks, original_ranks)
        pd.testing.assert_frame_equal(matrix, original_matrix)
        np.testing.assert_equal(kwargs, original_kwargs)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("crossbar_props", [{}, {"marker": "o"}, {"linestyle": ":", "alpha": 0.5}])
def test_cd_diagram_crossbar_order_is_independent_of_set_iteration(monkeypatch, crossbar_props):
    clique = set("abcd")
    # Assign descending ranks in this exact set's iteration order, whatever
    # PYTHONHASHSEED is. Return the same set so an unsorted crossbar always fails.
    names = list(clique)
    ranks = pd.Series([4.0, 3.0, 2.0, 1.0], index=names)
    matrix = pd.DataFrame(1.0, index=names, columns=names)
    monkeypatch.setattr("scikit_posthocs._plotting._find_maximal_cliques", lambda _: [clique])
    figure, ax = plt.subplots()
    try:
        artists = sp.critical_difference_diagram(
            ranks, matrix, ax=ax, crossbar_props=crossbar_props
        )

        np.testing.assert_equal(len(artists["crossbars"]), 1)
        (line,) = artists["crossbars"][0]
        np.testing.assert_array_equal(line.get_xdata(), [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(line.get_ydata(), [-1, -1, -1, -1])
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hue": {"a": "one"}}, "hue index/keys must match"),
        ({"hue": {"a": "one", "b": "two"}, "hue_order": ["one"]}, "hue_order must contain"),
        ({"hue": {"a": "one", "b": "two"}, "color_palette": ["red"]}, "at least as many colors"),
        ({"hue": {"a": "one", "b": "two"}, "color_palette": {"one": "red"}}, "all group names"),
        ({"color_palette": {"other": "red"}}, "keys are not consistent"),
        ({"color_palette": ["red"]}, "keys are not consistent"),
    ],
)
def test_cd_diagram_reports_inconsistent_color_metadata(kwargs, message):
    ranks = {"a": 1.0, "b": 2.0}
    matrix = pd.DataFrame(np.eye(2), index=ranks, columns=ranks)

    with pytest.raises(ValueError, match=message):
        sp.critical_difference_diagram(ranks, matrix, **kwargs)
