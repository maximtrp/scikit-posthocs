import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import scikit_posthocs._posthocs as sp
import seaborn as sb
import numpy as np

class TestPosthocs(unittest.TestCase):

    df = sb.load_dataset("exercise")

    def test_posthoc_conover(self):

        r_results = np.array([[-1, 1.131263e-02, 9.354690e-11],
                              [1.131263e-02, -1, 5.496288e-06],
                              [9.354690e-11, 5.496288e-06, -1]])

        results = sp.posthoc_conover(self.df, val_col = 'pulse', group_col = 'kind', p_adjust = 'holm')
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_dunn(self):

        r_results = np.array([[-1, 4.390066e-02, 9.570998e-09],
                              [4.390066e-02, -1, 1.873208e-04],
                              [9.570998e-09, 1.873208e-04, -1]])

        results = sp.posthoc_dunn(self.df, val_col = 'pulse', group_col = 'kind', p_adjust = 'holm')
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_nemenyi(self):

        r_results = np.array([[-1, 1.313107e-01, 2.431833e-08],
                              [1.313107e-01, -1, 4.855675e-04],
                              [2.431833e-08, 4.855675e-04, -1]])

        results = sp.posthoc_nemenyi(self.df, val_col = 'pulse', group_col = 'kind')
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_nemenyi_friedman(self):
        self.assertTrue(True)

    def test_posthoc_conover_friedman(self):
        self.assertTrue(True)

    def test_posthoc_durbin(self):
        self.assertTrue(True)

    def test_posthoc_quade(self):
        self.assertTrue(True)

    def test_posthoc_vanwaerden(self):
        r_results = np.array([[-1, 1.054709e-02, 6.476665e-11],
                              [1.054709e-02, -1, 4.433141e-06],
                              [6.476665e-11, 4.433141e-06, -1]])

        results = sp.posthoc_vanwaerden(self.df, val_col = 'pulse', group_col = 'kind', p_adjust='holm')
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_ttest(self):

        r_results = np.array([[-1, 9.757069e-03, 4.100954e-07],
                              [9.757069e-03, -1, 1.556010e-05],
                              [4.100954e-07, 1.556010e-05, -1]])

        results = sp.posthoc_ttest(self.df, val_col = 'pulse', group_col = 'kind', equal_var = False, p_adjust='holm')
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_tukey_hsd(self):
        self.assertTrue(True)

    def test_posthoc_mannwhitney(self):

        r_results = np.array([[-1, 1.714393e-02, 3.420508e-08],
                              [1.714393e-02, -1, 1.968352e-05],
                              [3.420508e-08, 1.968352e-05, -1]])

        results = sp.posthoc_mannwhitney(self.df, val_col = 'pulse', group_col = 'kind')
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_wilcoxon(self):

        r_results = np.array([[-1, 2.337133e-03, 2.857818e-06],
                              [2.337133e-03, -1, 1.230888e-05],
                              [2.857818e-06, 1.230888e-05, -1]])

        results = sp.posthoc_wilcoxon(self.df.sort_index(), val_col = 'pulse', group_col = 'kind')
        self.assertTrue(np.allclose(results, r_results))

if __name__ == '__main__':
    unittest.main()
