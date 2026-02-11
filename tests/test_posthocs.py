import os
import sys
import unittest
import matplotlib as mpl
import scikit_posthocs._posthocs as sp
import scikit_posthocs._omnibus as som
import scikit_posthocs._outliers as so
import scikit_posthocs._plotting as splt
import scikit_posthocs._global as spg
import scikit_posthocs._grouping as spg2
import seaborn as sb
import numpy as np
import matplotlib.axes as ma
from pandas import DataFrame, Series

if os.environ.get("DISPLAY", "") == "":
    print("No display found. Using non-interactive Agg backend")
    mpl.use("Agg")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestPosthocs(unittest.TestCase):
    # Global tests
    def test_global_simes_test(self):
        a = np.array([0.9, 0.1, 0.01, 0.99, 1.0, 0.02, 0.04])
        result = spg.global_simes_test(a)
        self.assertAlmostEqual(result, 0.07)

    def test_global_f_test(self):
        a = np.array([0.9, 0.1, 0.01, 0.99, 1.0, 0.02, 0.04])
        result, _ = spg.global_f_test(a)
        self.assertAlmostEqual(result, 0.01294562)

    # Plotting tests
    def test_sign_array(self):
        p_values = np.array(
            [
                [0.0, 0.00119517, 0.00278329],
                [0.00119517, 0.0, 0.18672227],
                [0.00278329, 0.18672227, 0.0],
            ]
        )
        test_results = splt.sign_array(p_values)
        correct_results = np.array([[-1, 1, 1], [1, -1, 0], [1, 0, -1]])
        self.assertTrue(np.all(test_results == correct_results))

    def test_sign_table(self):
        p_values = np.array(
            [
                [1.0, 0.00119517, 0.00278329],
                [0.00119517, 1.0, 0.18672227],
                [0.00278329, 0.18672227, 1.0],
            ]
        )

        correct_results = np.array(
            [["-", "**", "**"], ["**", "-", "NS"], ["**", "NS", "-"]], dtype=object
        )
        correct_resultsl = np.array(
            [["-", "", ""], ["**", "-", ""], ["**", "NS", "-"]], dtype=object
        )
        correct_resultsu = np.array(
            [["-", "**", "**"], ["", "-", "NS"], ["", "", "-"]], dtype=object
        )

        with self.assertRaises(ValueError):
            splt.sign_table(p_values, lower=False, upper=False)

        self.assertTrue(
            np.all(splt.sign_table(p_values, lower=False, upper=True) == correct_resultsu)
        )
        self.assertTrue(
            np.all(splt.sign_table(p_values, lower=True, upper=False) == correct_resultsl)
        )
        self.assertTrue(
            np.all(splt.sign_table(p_values, lower=True, upper=True) == correct_results)
        )

    def test_sign_plot(self):
        x = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
        a = splt.sign_plot(x, flat=True, labels=False)
        with self.assertRaises(ValueError):
            splt.sign_plot(x.astype(float), flat=True, labels=False)
        self.assertTrue(isinstance(a, ma._axes.Axes))

    def test_sign_plot_nonflat(self):
        x = np.array(
            [
                [1.0, 0.00119517, 0.00278329],
                [0.00119517, 1.0, 0.18672227],
                [0.00278329, 0.18672227, 1.0],
            ]
        )
        a, cbar = splt.sign_plot(x, cbar=True, labels=False)

        with self.assertRaises(ValueError):
            splt.sign_plot(x, cmap=[1, 1], labels=False)
        with self.assertRaises(ValueError):
            splt.sign_plot(x.astype(np.int64), labels=False)

        self.assertTrue(
            isinstance(a, ma._axes.Axes) and isinstance(cbar, mpl.colorbar.ColorbarBase)
        )

    def test_find_maximal_cliques_input_validation(self):
        with self.assertRaisesRegex(ValueError, ".*indices do not match"):
            splt._find_maximal_cliques(
                DataFrame(
                    [[0, 1], [1, 0]],
                    index=["a", "b"],
                    columns=["b", "a"],
                )
            )
        with self.assertRaises(ValueError, msg="Input matrix must be binary"):
            splt._find_maximal_cliques(DataFrame([[0, 3], [3, 0]]))
        with self.assertRaisesRegex(ValueError, ".*empty and symmetric"):
            splt._find_maximal_cliques(DataFrame())
        with self.assertRaisesRegex(ValueError, ".*empty and symmetric"):
            splt._find_maximal_cliques(DataFrame([[1, 0], [1, 0]]))

    def test_find_maximal_cliques_1x1(self):
        adj_matrix = DataFrame([[0]], columns=["a"], index=["a"])
        expected = [{"a"}]
        self.assertEqual(splt._find_maximal_cliques(adj_matrix), expected)

    def test_find_maximal_cliques_2x2(self):
        adj_matrix = DataFrame(
            [[0, 1], [1, 0]],
            columns=["a", "b"],
            index=["a", "b"],
        )
        expected = [{"a", "b"}]
        self.assertEqual(splt._find_maximal_cliques(adj_matrix), expected)

    def test_find_maximal_cliques_3x3(self):
        adj_matrix = DataFrame(
            [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
            columns=["a", "b", "c"],
            index=["a", "b", "c"],
        )
        expected = [{"a", "c"}, {"b"}]
        self.assertEqual(
            set(map(frozenset, splt._find_maximal_cliques(adj_matrix))),
            set(map(frozenset, expected)),
        )

    def test_find_maximal_cliques_6x6(self):
        adj_matrix = DataFrame(
            [
                [0, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 0],
                [0, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        expected = [{0, 1}, {1, 2, 3, 4}, {5}]
        self.assertEqual(
            set(map(frozenset, splt._find_maximal_cliques(adj_matrix))),
            set(map(frozenset, expected)),
        )

    def test_cd_diagram_number_of_artists(self):
        index = list("abcdef")
        ranks = Series([2.1, 1.2, 4.5, 3.2, 5.7, 6.5], index=index)
        sig_matrix = DataFrame(
            [
                [0.08, 0.08, 0.01, 0.01, 0.01, 0.01],
                [0.08, 0.08, 0.08, 0.08, 0.08, 0.01],
                [0.01, 0.08, 0.08, 0.08, 0.08, 0.01],
                [0.01, 0.08, 0.08, 0.08, 0.08, 0.01],
                [0.01, 0.08, 0.08, 0.08, 0.08, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.08],
            ],
            index=index,
            columns=index,
        )

        output = splt.critical_difference_diagram(ranks, sig_matrix)
        self.assertEqual(len(output["markers"]), len(ranks))
        self.assertEqual(len(output["elbows"]), len(ranks))
        self.assertEqual(len(output["labels"]), len(ranks))
        self.assertEqual(len(output["crossbars"]), 2)

    # Outliers tests
    def test_outliers_iqr(self):
        x = np.array([4, 5, 6, 10, 12, 4, 3, 1, 2, 3, 23, 5, 3])

        x_filtered = np.array([4, 5, 6, 10, 4, 3, 1, 2, 3, 5, 3])
        indices = np.delete(np.arange(13), [4, 10])
        outliers_indices = np.array([4, 10])
        outliers = np.array([12, 23])

        test_outliers = so.outliers_iqr(x, ret="outliers")
        test_outliers_indices = so.outliers_iqr(x, ret="outliers_indices")
        test_indices = so.outliers_iqr(x, ret="indices")
        test_filtered = so.outliers_iqr(x, ret="filtered")

        self.assertTrue(
            np.all(test_outliers == outliers)
            and np.all(test_outliers_indices == outliers_indices)
            and np.all(test_indices == indices)
            and np.all(test_filtered == x_filtered)
        )

    def test_outliers_grubbs(self):
        x = np.array([199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57])
        test_results = so.outliers_grubbs(x)
        correct_results = np.array([199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18])
        self.assertTrue(so.outliers_grubbs(x, hypo=True))
        self.assertTrue(np.all(test_results == correct_results))

    def test_outliers_tietjen(self):
        x = np.array(
            [
                -1.40,
                -0.44,
                -0.30,
                -0.24,
                -0.22,
                -0.13,
                -0.05,
                0.06,
                0.10,
                0.18,
                0.20,
                0.39,
                0.48,
                0.63,
                1.01,
            ]
        )
        test_results = so.outliers_tietjen(x, 2)
        correct_results = np.array(
            [
                -0.44,
                -0.3,
                -0.24,
                -0.22,
                -0.13,
                -0.05,
                0.06,
                0.1,
                0.18,
                0.2,
                0.39,
                0.48,
                0.63,
            ]
        )
        self.assertTrue(so.outliers_tietjen(x, 2, hypo=True))
        self.assertTrue(np.all(test_results == correct_results))

    def test_outliers_gesd(self):
        x = np.array(
            [
                -0.25,
                0.68,
                0.94,
                1.15,
                1.2,
                1.26,
                1.26,
                1.34,
                1.38,
                1.43,
                1.49,
                1.49,
                1.55,
                1.56,
                1.58,
                1.65,
                1.69,
                1.7,
                1.76,
                1.77,
                1.81,
                1.91,
                1.94,
                1.96,
                1.99,
                2.06,
                2.09,
                2.1,
                2.14,
                2.15,
                2.23,
                2.24,
                2.26,
                2.35,
                2.37,
                2.4,
                2.47,
                2.54,
                2.62,
                2.64,
                2.9,
                2.92,
                2.92,
                2.93,
                3.21,
                3.26,
                3.3,
                3.59,
                3.68,
                4.3,
                4.64,
                5.34,
                5.42,
                6.01,
            ]
        )
        correct_mask = np.zeros_like(x, dtype=bool)
        correct_mask[-3:] = True
        test_results = so.outliers_gesd(x, 5)
        test_mask_results = so.outliers_gesd(x, 5, hypo=True)
        correct_results = np.array(
            [
                -0.25,
                0.68,
                0.94,
                1.15,
                1.2,
                1.26,
                1.26,
                1.34,
                1.38,
                1.43,
                1.49,
                1.49,
                1.55,
                1.56,
                1.58,
                1.65,
                1.69,
                1.7,
                1.76,
                1.77,
                1.81,
                1.91,
                1.94,
                1.96,
                1.99,
                2.06,
                2.09,
                2.1,
                2.14,
                2.15,
                2.23,
                2.24,
                2.26,
                2.35,
                2.37,
                2.4,
                2.47,
                2.54,
                2.62,
                2.64,
                2.9,
                2.92,
                2.92,
                2.93,
                3.21,
                3.26,
                3.3,
                3.59,
                3.68,
                4.3,
                4.64,
            ]
        )
        self.assertTrue(isinstance(so.outliers_gesd(x, 5, report=True), np.ndarray))
        self.assertTrue(np.array_equal(test_results, correct_results))
        self.assertTrue(np.array_equal(test_mask_results, correct_mask))
        self.assertTrue(
            np.array_equal(so.outliers_gesd(correct_results, 5, hypo=False), correct_results)
        )
        self.assertTrue(
            np.array_equal(
                so.outliers_gesd(correct_results, 5, hypo=True),
                np.zeros_like(correct_results, dtype=bool),
            )
        )

    # Statistical tests
    df = sb.load_dataset("exercise")
    df[df.columns[df.dtypes == "category"]] = df[df.columns[df.dtypes == "category"]].astype(object)
    df_bn = np.array([[4, 3, 4, 4, 5, 6, 3], [1, 2, 3, 5, 6, 7, 7], [1, 2, 6, 4, 1, 5, 1]])

    # DataFrame conversion tests
    def test_convert_to_block_df(self):
        a = np.array(
            [
                [0, 0, 0, 4],
                [1, 1, 0, 1],
                [2, 2, 0, 1],
                [0, 0, 1, 3],
                [1, 1, 1, 2],
                [2, 2, 1, 2],
                [0, 0, 2, 4],
                [1, 1, 2, 3],
                [2, 2, 2, 6],
                [0, 0, 3, 4],
                [1, 1, 3, 5],
                [2, 2, 3, 4],
                [0, 0, 4, 5],
                [1, 1, 4, 6],
                [2, 2, 4, 1],
                [0, 0, 5, 6],
                [1, 1, 5, 7],
                [2, 2, 5, 5],
                [0, 0, 6, 3],
                [1, 1, 6, 7],
                [2, 2, 6, 1],
            ],
            dtype=float,
        )
        df_a = DataFrame(a, columns=["blk_col", "blk_id_col", "grp_col", "y_col"])

        result = sp.posthoc_nemenyi_friedman(
            a, y_col=3, group_col=2, block_col=0, block_id_col=1, melted=True
        )[0].values
        result2 = sp.posthoc_nemenyi_friedman(self.df_bn)[0].values
        result3 = sp.posthoc_nemenyi_friedman(
            df_a,
            y_col="y_col",
            group_col="grp_col",
            block_col="blk_col",
            block_id_col="blk_id_col",
            melted=True,
        )[0].values
        self.assertTrue(np.allclose(result, result2))
        self.assertTrue(np.allclose(result, result3))
        self.assertTrue(np.allclose(result2, result3))

    # Omnibox tests
    def test_osrt(self):
        df = DataFrame(dict(zip(["a", "b", "c"], self.df_bn.tolist()))).melt()
        p, _, _ = som.test_osrt(df, val_col="value", group_col="variable")
        result = 0.3157646
        self.assertTrue(np.allclose(p, result, atol=1.0e-3))

    def test_durbin(self):
        r_result = np.array([0.205758, 8.468354, 6])
        result = som.test_durbin(self.df_bn)
        self.assertTrue(np.allclose(result, r_result))

    def test_mackwolfe(self):
        x = [
            [22, 23, 35],
            [60, 59, 54],
            [98, 78, 50],
            [60, 82, 59],
            [22, 44, 33],
            [23, 21, 25],
        ]
        result, _ = som.test_mackwolfe(x, p=2)
        self.assertEqual(som.test_mackwolfe(x, p=20), (np.nan, np.nan))
        self.assertEqual(som.test_mackwolfe(x, p=0), (np.nan, np.nan))
        self.assertTrue(np.allclose(result, 0.0006812725))

    def test_mackwolfe_nperm(self):
        x = [
            [22, 23, 35],
            [60, 59, 54],
            [98, 78, 50],
            [60, 82, 59],
            [22, 44, 33],
            [23, 21, 25],
        ]
        _, stat = som.test_mackwolfe(x, n_perm=50)
        self.assertTrue(np.allclose(stat, 3.2024699769846983))

    # Post hoc tests
    def test_posthoc_anderson(self):
        r_results = np.array(
            [
                [1, 1.35079e-02, 8.64418e-09],
                [1.35079e-02, 1, 1.644534e-05],
                [8.64418e-09, 1.644534e-05, 1],
            ]
        )

        results = sp.posthoc_anderson(self.df, val_col="pulse", group_col="kind", p_adjust="holm")
        self.assertTrue(np.allclose(results.values, r_results, atol=3.0e-3))

    def test_posthoc_conover(self):
        r_results = np.array(
            [
                [1, 9.354690e-11, 1.131263e-02],
                [9.354690e-11, 1, 5.496288e-06],
                [1.131263e-02, 5.496288e-06, 1],
            ]
        )

        results = sp.posthoc_conover(
            self.df, val_col="pulse", group_col="kind", p_adjust="holm"
        ).values
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_dunn(self):
        r_results = np.array(
            [
                [1, 9.570998e-09, 4.390066e-02],
                [9.570998e-09, 1, 1.873208e-04],
                [4.390066e-02, 1.873208e-04, 1],
            ]
        )

        results = sp.posthoc_dunn(
            self.df, val_col="pulse", group_col="kind", p_adjust="holm"
        ).values
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_nemenyi(self):
        r_results = np.array(
            [
                [1, 2.431833e-08, 1.313107e-01],
                [2.431833e-08, 1, 4.855675e-04],
                [1.313107e-01, 4.855675e-04, 1],
            ]
        )

        results = sp.posthoc_nemenyi(self.df, val_col="pulse", group_col="kind").values
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_nemenyi_tukey(self):
        r_results = np.array(
            [
                [1, 9.793203e-09, 1.088785e-01],
                [9.793203e-09, 1, 0.0002789016],
                [1.088785e-01, 0.0002789016, 1],
            ]
        )

        results = sp.posthoc_nemenyi(
            self.df, val_col="pulse", group_col="kind", dist="tukey"
        ).values
        self.assertTrue(np.allclose(results, r_results, atol=1.0e-3))

    def test_posthoc_nemenyi_friedman(self):
        p_results = np.array(
            [
                [
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    0.9999999,
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    0.8414506,
                    0.8833015,
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [0.9177741, 0.9449086, 0.9999962, 1.0, np.nan, np.nan, np.nan],
                [0.9177741, 0.9449086, 0.9999962, 1.0000000, 1.0, np.nan, np.nan],
                [0.2147827, 0.2597539, 0.9449086, 0.8833015, 0.8833015, 1.0, np.nan],
                [0.9976902, 0.9991770, 0.9888953, 0.9976902, 0.9976902, 0.5511935, 1.0],
            ]
        )
        tri_upper = np.triu_indices(p_results.shape[0], 1)
        p_results[tri_upper] = np.transpose(p_results)[tri_upper]
        results = sp.posthoc_nemenyi_friedman(self.df_bn)
        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_conover_friedman(self):
        results = sp.posthoc_conover_friedman(self.df_bn, p_adjust="bonferroni")
        p_results = (
            np.array(
                [
                    [1.0000000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [0.9147508, 1.00000000, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [0.1518030, 0.18071036, 1.0000000, np.nan, np.nan, np.nan, np.nan],
                    [
                        0.2140927,
                        0.25232845,
                        0.8305955,
                        1.000000,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        0.2140927,
                        0.25232845,
                        0.8305955,
                        1.000000,
                        1.000000,
                        np.nan,
                        np.nan,
                    ],
                    [
                        0.0181602,
                        0.02222747,
                        0.2523284,
                        0.1807104,
                        0.1807104,
                        1.00009000,
                        np.nan,
                    ],
                    [
                        0.5242303,
                        0.59465124,
                        0.3989535,
                        0.5242303,
                        0.5242303,
                        0.05991984,
                        1.000000,
                    ],
                ]
            )
            * 21
        )
        p_results[p_results > 1] = 1.0
        tri_upper = np.triu_indices(p_results.shape[0], 1)
        p_results[tri_upper] = np.transpose(p_results)[tri_upper]
        np.fill_diagonal(p_results, 1)
        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_conover_friedman_tukey(self):
        results = sp.posthoc_conover_friedman(self.df_bn, p_adjust="single-step")
        p_results = np.array(
            [
                [1.00000000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.99999986, 1.0000000, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.72638075, 0.7905289, 1.0000000, np.nan, np.nan, np.nan, np.nan],
                [0.84667448, 0.8934524, 0.9999910, 1.0000000, np.nan, np.nan, np.nan],
                [
                    0.84667448,
                    0.8934524,
                    0.9999910,
                    1.0000000,
                    1.0000000,
                    np.nan,
                    np.nan,
                ],
                [
                    0.09013677,
                    0.1187580,
                    0.8934524,
                    0.7905289,
                    0.7905289,
                    1.0000000,
                    np.nan,
                ],
                [
                    0.99482447,
                    0.9981178,
                    0.9763466,
                    0.9948245,
                    0.9948245,
                    0.3662675,
                    1.000000,
                ],
            ]
        )
        tri_upper = np.triu_indices(p_results.shape[0], 1)
        p_results[tri_upper] = np.transpose(p_results)[tri_upper]
        np.fill_diagonal(p_results, 1)
        self.assertTrue(np.allclose(results, p_results, atol=1e-3))

    def test_posthoc_conover_friedman_non_melted(self):
        df = DataFrame(self.df_bn)
        results = sp.posthoc_conover_friedman(df, melted=False)
        p_results = np.array(
            [
                [1.0000000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.9147508, 1.00000000, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.1518030, 0.18071036, 1.0000000, np.nan, np.nan, np.nan, np.nan],
                [0.2140927, 0.25232845, 0.8305955, 1.000000, np.nan, np.nan, np.nan],
                [0.2140927, 0.25232845, 0.8305955, 1.000000, 1.000000, np.nan, np.nan],
                [
                    0.0181602,
                    0.02222747,
                    0.2523284,
                    0.1807104,
                    0.1807104,
                    1.00009000,
                    np.nan,
                ],
                [
                    0.5242303,
                    0.59465124,
                    0.3989535,
                    0.5242303,
                    0.5242303,
                    0.05991984,
                    1.000000,
                ],
            ]
        )
        tri_upper = np.triu_indices(p_results.shape[0], 1)
        p_results[tri_upper] = np.transpose(p_results)[tri_upper]
        np.fill_diagonal(p_results, 1)
        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_miller_friedman(self):
        results = sp.posthoc_miller_friedman(self.df_bn)

        p_results = np.array(
            [
                [
                    1.0,
                    1.0,
                    0.9411963,
                    0.9724396000000001,
                    0.9724396000000001,
                    0.4717981,
                    0.9993864,
                ],
                [
                    1.0,
                    1.0,
                    0.9588993,
                    0.9823818000000001,
                    0.9823818000000001,
                    0.5256257,
                    0.9997869,
                ],
                [
                    0.9411963,
                    0.9588993,
                    1.0,
                    0.9999991,
                    0.9999991,
                    0.9823818000000001,
                    0.9968575999999999,
                ],
                [
                    0.9724396000000001,
                    0.9823818000000001,
                    0.9999991,
                    1.0,
                    1.0,
                    0.9588993,
                    0.9993864,
                ],
                [
                    0.9724396000000001,
                    0.9823818000000001,
                    0.9999991,
                    1.0,
                    1.0,
                    0.9588993,
                    0.9993864,
                ],
                [
                    0.4717981,
                    0.5256257,
                    0.9823818000000001,
                    0.9588993,
                    0.9588993,
                    1.0,
                    0.7803545999999999,
                ],
                [
                    0.9993864,
                    0.9997869,
                    0.9968575999999999,
                    0.9993864,
                    0.9993864,
                    0.7803545999999999,
                    1.0,
                ],
            ]
        )

        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_siegel_friedman(self):
        results = sp.posthoc_siegel_friedman(self.df_bn, p_adjust="bonferroni")

        p_results = (
            np.array(
                [
                    [
                        1.000000,
                        0.92471904,
                        0.18587673,
                        0.25683926,
                        0.25683926,
                        0.01816302,
                        0.57075039,
                    ],
                    [
                        0.92471904,
                        1.0000000,
                        0.2193026,
                        0.2986177,
                        0.2986177,
                        0.0233422,
                        0.6366016,
                    ],
                    [
                        0.18587673,
                        0.2193026,
                        1.0000000,
                        0.8501067,
                        0.8501067,
                        0.2986177,
                        0.4496918,
                    ],
                    [
                        0.25683926,
                        0.2986177,
                        0.8501067,
                        1.000000,
                        1.0000000,
                        0.2193026,
                        0.5707504,
                    ],
                    [
                        0.25683926,
                        0.2986177,
                        0.8501067,
                        1.0000000,
                        1.0000000,
                        0.2193026,
                        0.5707504,
                    ],
                    [
                        0.01816302,
                        0.0233422,
                        0.2986177,
                        0.2193026,
                        0.2193026,
                        1.000000,
                        0.07260094,
                    ],
                    [
                        0.57075039,
                        0.6366016,
                        0.4496918,
                        0.5707504,
                        0.5707504,
                        0.07260094,
                        1.000000,
                    ],
                ]
            )
            * 21
        )
        p_results[p_results > 1] = 1.0

        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_durbin(self):
        results = sp.posthoc_durbin(self.df_bn, p_adjust="holm")

        p_results = np.array(
            [
                [1.000000, 1.000000, 1.0, 1.0, 1.0, 0.381364, 1.0],
                [1.000000, 1.000000, 1.0, 1.0, 1.0, 0.444549, 1.0],
                [1.000000, 1.000000, 1.0, 1.0, 1.0, 1.000000, 1.0],
                [1.000000, 1.000000, 1.0, 1.0, 1.0, 1.000000, 1.0],
                [1.000000, 1.000000, 1.0, 1.0, 1.0, 1.000000, 1.0],
                [0.381364, 0.444549, 1.0, 1.0, 1.0, 1.000000, 1.0],
                [1.000000, 1.000000, 1.0, 1.0, 1.0, 1.000000, 1.0],
            ]
        )
        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_quade(self):
        results = sp.posthoc_quade(self.df_bn, p_adjust="bonferroni")

        p_results = (
            np.array(
                [
                    [
                        1.00000000,
                        0.67651326,
                        0.15432143,
                        0.17954686,
                        0.2081421,
                        0.02267043,
                        0.2081421,
                    ],
                    [
                        0.67651326,
                        1.00000000,
                        0.29595042,
                        0.33809987,
                        0.38443835,
                        0.0494024,
                        0.38443835,
                    ],
                    [
                        0.15432143,
                        0.29595042,
                        1.00000000,
                        0.92586499,
                        0.85245022,
                        0.29595042,
                        0.85245022,
                    ],
                    [
                        0.17954686,
                        0.33809987,
                        0.92586499,
                        1.00000000,
                        0.92586499,
                        0.25789648,
                        0.92586499,
                    ],
                    [
                        0.2081421,
                        0.38443835,
                        0.85245022,
                        0.92586499,
                        1.00000000,
                        0.22378308,
                        1.00000000,
                    ],
                    [
                        0.02267043,
                        0.0494024,
                        0.29595042,
                        0.25789648,
                        0.22378308,
                        1.00000000,
                        0.22378308,
                    ],
                    [
                        0.2081421,
                        0.38443835,
                        0.85245022,
                        0.92586499,
                        1.00000000,
                        0.22378308,
                        1.00000000,
                    ],
                ]
            )
            * 21
        )
        p_results[p_results > 1.0] = 1.0
        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_quade_norm(self):
        results = sp.posthoc_quade(self.df_bn, dist="normal")

        p_results = np.array(
            [
                [1.00000000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.5693540320, 1.00000000, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.0430605548, 0.145913303, 1.00000000, np.nan, np.nan, np.nan, np.nan],
                [
                    0.0578705783,
                    0.184285855,
                    0.8993796,
                    1.00000000,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    0.0766885196,
                    0.229662468,
                    0.8003530,
                    0.8993796,
                    1.00000000,
                    np.nan,
                    np.nan,
                ],
                [
                    0.0005066018,
                    0.003634715,
                    0.1459133,
                    0.1139777,
                    0.08782032,
                    1.00000000,
                    np.nan,
                ],
                [
                    0.0766885196,
                    0.229662468,
                    0.8003530,
                    0.8993796,
                    1.00000000,
                    0.08782032,
                    1.00000000,
                ],
            ]
        )
        tri_upper = np.triu_indices(p_results.shape[0], 1)
        p_results[tri_upper] = np.transpose(p_results)[tri_upper]
        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_npm_test(self):
        data = np.array(
            [
                [2.4, 3, 3, 2.2, 2.2, 2.2, 2.2, 2.8, 2, 3],
                [2.8, 2.2, 3.8, 9.4, 8.4, 3, 3.2, 4.4, 3.2, 7.4],
                [9.8, 3.2, 5.8, 7.8, 2.6, 2.2, 6.2, 9.4, 7.8, 3.4],
                [7, 9.8, 9.4, 8.8, 8.8, 3.4, 9, 8.4, 2.4, 7.8],
            ]
        )

        results = sp.posthoc_npm_test(data)

        p_results = np.array(
            [
                [1.0, 0.0077, 0.0020, 2e-16],
                [0.0077, 1.0, 0.2884, 0.0854],
                [0.0020, 0.2884, 1.0, 0.1385],
                [2e-16, 0.0854, 0.1385, 1.0],
            ]
        )

        self.assertTrue(np.allclose(results, p_results, rtol=4))

    def test_posthoc_vanwaerden(self):
        r_results = np.array(
            [
                [1, 1.054709e-02, 6.476665e-11],
                [1.054709e-02, 1, 4.433141e-06],
                [6.476665e-11, 4.433141e-06, 1],
            ]
        )

        results = sp.posthoc_vanwaerden(self.df, val_col="pulse", group_col="kind", p_adjust="holm")
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_dscf(self):
        r_results = np.array(
            [
                [1, 4.430682e-02, 9.828003e-08],
                [4.430682e-02, 1, 5.655274e-05],
                [9.828003e-08, 5.655274e-05, 1],
            ]
        )

        results = sp.posthoc_dscf(self.df, val_col="pulse", group_col="kind")
        self.assertTrue(np.allclose(results, r_results, atol=0.001))

    def test_posthoc_ttest(self):
        r_results = np.array(
            [
                [1, 9.757069e-03, 4.100954e-07],
                [9.757069e-03, 1, 1.556010e-05],
                [4.100954e-07, 1.556010e-05, 1],
            ]
        )

        results = sp.posthoc_ttest(
            self.df, val_col="pulse", group_col="kind", equal_var=False, p_adjust="holm"
        )
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_ttest_pooled(self):
        x = [[1, 2, 3, 5, 1], [12, 31, 54, 50, 40], [10, 12, 6, 74, 11]]
        r_results = np.array(
            [
                [1, 0.04226866, 0.24706893],
                [0.04226866, 1, 0.2482456],
                [0.24706893, 0.2482456, 1],
            ]
        )

        results = sp.posthoc_ttest(x, equal_var=False, p_adjust="holm", pool_sd=True)
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_tukey_hsd(self):
        x = [[1, 2, 3, 4, 5], [35, 31, 75, 40, 21], [10, 6, 9, 6, 1]]
        results = sp.posthoc_tukey_hsd(x)
        n_results = np.array(
            [
                [1.0, 0.000991287, 0.897449027],
                [0.000991287, 1.0, 0.00210909],
                [0.897449027, 0.00210909, 1.0],
            ]
        )
        self.assertTrue(np.allclose(n_results, results))

    def test_posthoc_mannwhitney(self):
        r_results = (
            np.array(
                [
                    [1, 3.420508e-08, 1.714393e-02],
                    [3.420508e-08, 1, 1.968352e-05],
                    [1.714393e-02, 1.968352e-05, 1],
                ]
            )
            * 3
        )
        np.fill_diagonal(r_results, 1)

        results = sp.posthoc_mannwhitney(
            self.df, val_col="pulse", group_col="kind", p_adjust="bonferroni"
        ).values
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_mannwhitney_ndarray(self):
        _x = [[1, 2, 3, 5, 1], [12, 31, 54, 50, 40], [10, 12, 6, 74, 11]]
        x = np.array(_x)
        g = np.repeat([0, 1, 2], 5)
        nd = np.column_stack((x.ravel(), g))
        xdf = DataFrame(dict(zip(list("abc"), _x))).melt(var_name="groups", value_name="vals")
        results = sp.posthoc_mannwhitney(xdf, val_col="vals", group_col="groups").values
        nd_results = sp.posthoc_mannwhitney(nd, val_col=0, group_col=1).values
        self.assertTrue(np.allclose(nd_results, results))

    def test_posthoc_wilcoxon(self):
        r_results = (
            np.array(
                [
                    [1, 2.337133e-03, 2.857818e-06],
                    [2.337133e-03, 1, 1.230888e-05],
                    [2.857818e-06, 1.230888e-05, 1],
                ]
            )
            * 3
        )
        np.fill_diagonal(r_results, 1)

        results = sp.posthoc_wilcoxon(
            self.df.sort_index(),
            val_col="pulse",
            group_col="kind",
            p_adjust="bonferroni",
        )
        self.assertTrue(np.allclose(results, r_results, atol=1e-4))

    def test_posthoc_scheffe(self):
        r_results = np.array(
            [
                [1.0, 3.378449e-01, 3.047472e-10],
                [3.378449e-01, 1.0, 2.173209e-07],
                [3.047472e-10, 2.173209e-07, 1.0],
            ]
        )

        results = sp.posthoc_scheffe(self.df.sort_index(), val_col="pulse", group_col="kind")
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_tamhane(self):
        r_results = np.array(
            [
                [1, 2.898653e-02, 4.100954e-07],
                [2.898653e-02, 1, 2.333996e-05],
                [4.100954e-07, 2.333996e-05, 1],
            ]
        )

        results = sp.posthoc_tamhane(self.df.sort_index(), val_col="pulse", group_col="kind")
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_tamhane_nw(self):
        r_results = np.array(
            [
                [1, 2.883219e-02, 4.780682e-08],
                [2.883219e-02, 1, 8.643683e-06],
                [4.780682e-08, 8.643683e-06, 1],
            ]
        )

        results = sp.posthoc_tamhane(
            self.df.sort_index(), val_col="pulse", group_col="kind", welch=False
        )
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_tukey(self):
        r_results = np.array(
            [
                [1, 3.042955e-01, 4.308631e-10],
                [3.042955e-01, 1, 9.946571e-08],
                [4.308631e-10, 9.946571e-08, 1],
            ]
        )

        results = sp.posthoc_tukey(self.df.sort_index(), val_col="pulse", group_col="kind")
        self.assertTrue(np.allclose(results, r_results, atol=1.0e-3))

    def test_posthoc_dunnett(self):
        r_results = [8.125844e-11, 2.427434e-01]

        # scipy use randomized Quasi-Monte Carlo integration of the multivariate-t distribution
        # to compute the p-values. The result may vary slightly from run to run.
        # we run the test 1000 times (maximum absolute tolerance = 1.e-4 for example data)
        is_close = []
        for i in range(100):
            results = sp.posthoc_dunnett(
                self.df.sort_index(),
                val_col="pulse",
                group_col="kind",
                control="rest",
                to_matrix=False,
            )
            is_close.append(np.allclose(results, r_results, atol=1e-4))

        is_close_mt = []
        for i in range(100):
            df_results = sp.posthoc_dunnett(
                self.df.sort_index(),
                val_col="pulse",
                group_col="kind",
                control="rest",
                to_matrix=True,
            )
            results = [
                df_results.loc["rest", "running"],
                df_results.loc["rest", "walking"],
            ]
            is_close_mt.append(np.allclose(results, r_results, atol=1e-4))
        self.assertTrue(sum(is_close) > 95)
        self.assertTrue(sum(is_close_mt) > 95)


class TestCompactLetterDisplay(unittest.TestCase):
    # Piepho (2004) Example 1:
    # Groups 0-3, significant pairs: (0,1), (0,2), (0,3), (1,3)
    # Expected: ['a  ', ' b ', ' bc', '  c']
    piepho1_pv = np.array([
        [-1.0, 0.01, 0.01, 0.01],
        [ 0.01, -1.0, 0.50, 0.01],
        [ 0.01,  0.50, -1.0, 0.50],
        [ 0.01,  0.01, 0.50, -1.0],
    ])

    def test_piepho_example1_letters(self):
        result = spg2.compact_letter_display(self.piepho1_pv)
        self.assertEqual(list(result), ['a  ', ' b ', ' bc', '  c'])

    def test_piepho_example1_index(self):
        result = spg2.compact_letter_display(self.piepho1_pv)
        self.assertEqual(list(result.index), [0, 1, 2, 3])

    def test_dataframe_input_preserves_names(self):
        df = DataFrame(self.piepho1_pv,
                       index=['A', 'B', 'C', 'D'],
                       columns=['A', 'B', 'C', 'D'])
        result = spg2.compact_letter_display(df)
        self.assertEqual(list(result.index), ['A', 'B', 'C', 'D'])
        self.assertEqual(list(result), ['a  ', ' b ', ' bc', '  c'])

    def test_custom_names(self):
        result = spg2.compact_letter_display(self.piepho1_pv, names=['w', 'x', 'y', 'z'])
        self.assertEqual(list(result.index), ['w', 'x', 'y', 'z'])

    def test_all_different(self):
        # All pairs significantly different -> each group gets its own letter
        pv = np.array([
            [-1.0, 0.01, 0.01],
            [ 0.01, -1.0, 0.01],
            [ 0.01,  0.01, -1.0],
        ])
        result = spg2.compact_letter_display(pv)
        # Each group belongs to exactly one unique letter group
        letters = [s.strip() for s in result]
        self.assertEqual(len(set(letters)), 3)
        self.assertTrue(all(len(s.strip()) == 1 for s in result))

    def test_none_different(self):
        # No pairs significantly different -> all groups share the same letter
        pv = np.array([
            [-1.0, 0.80, 0.90],
            [ 0.80, -1.0, 0.70],
            [ 0.90,  0.70, -1.0],
        ])
        result = spg2.compact_letter_display(pv)
        # All groups should have the same non-space letter
        self.assertEqual(len(set(result)), 1)
        self.assertEqual(result.iloc[0].strip(), 'a')

    def test_series_name(self):
        result = spg2.compact_letter_display(self.piepho1_pv)
        self.assertEqual(result.name, 'letters')


if __name__ == "__main__":
    unittest.main()
