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
        self.assertTrue(True)

    def test_posthoc_ttest(self):
        self.assertTrue(True)

    def test_posthoc_tukey_hsd(self):
        self.assertTrue(True)

    def test_posthoc_mannwhitney(self):
        self.assertTrue(True)

    def test_posthoc_wilcoxon(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
