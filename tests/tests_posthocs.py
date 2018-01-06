import unittest as ut
import scikit_posthocs as sp
import seaborn as sb
import numpy as np

class TestPosthocs(ut.TestCase):

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
        self.assertEqual()

    def test_posthoc_nemenyi_friedman(self):
        self.assertEqual()

    def test_posthoc_conover_friedman(self):
        self.assertEqual()

    def test_posthoc_durbin(self):
        self.assertEqual()

    def test_posthoc_quade(self):
        self.assertEqual()

    def test_posthoc_vanwaerden(self):
        self.assertEqual()

    def test_posthoc_ttest(self):
        self.assertEqual()

    def test_posthoc_tukey_hsd(self):
        self.assertEqual()

    def test_posthoc_mannwhitney(self):
        self.assertEqual()

    def test_posthoc_wilcoxon(self):
        self.assertEqual()


if __name__ == '__main__':
    ut.main()
