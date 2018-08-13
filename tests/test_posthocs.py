import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import scikit_posthocs._posthocs as sp
import seaborn as sb
import numpy as np

class TestPosthocs(unittest.TestCase):

    df = sb.load_dataset("exercise")
    df_bn = np.array([[4,3,4,4,5,6,3],
                      [1,2,3,5,6,7,7],
                      [1,2,6,4,1,5,1]])

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
        self.assertTrue(np.allclose(results, r_results, atol=1.e-4))

    def test_posthoc_nemenyi(self):

        r_results = np.array([[-1, 1.313107e-01, 2.431833e-08],
                              [1.313107e-01, -1, 4.855675e-04],
                              [2.431833e-08, 4.855675e-04, -1]])

        results = sp.posthoc_nemenyi(self.df, val_col = 'pulse', group_col = 'kind')
        self.assertTrue(np.allclose(results, r_results, atol=1.e-4))

    def test_posthoc_nemenyi_friedman(self):

        p_results = np.array([[-1., 0.9, 0.82163255, 0.9, 0.9, 0.21477876, 0.9],
                              [0.9, -1., 0.87719193, 0.9, 0.9, 0.25967965, 0.9],
                              [0.82163255, 0.87719193, -1., 0.9, 0.9, 0.9, 0.9],
                              [0.9, 0.9, 0.9, -1., 0.9, 0.87719193, 0.9],
                              [0.9, 0.9, 0.9, 0.9, -1., 0.87719193, 0.9],
                              [0.21477876, 0.25967965, 0.9, 0.87719193, 0.87719193, -1., 0.54381888],
                              [0.9, 0.9, 0.9, 0.9, 0.9, 0.54381888, -1.]])
        results = sp.posthoc_nemenyi_friedman(self.df_bn)
        self.assertTrue(np.allclose(results, p_results, atol=1.e-4))

    def test_posthoc_conover_friedman(self):

        results = sp.posthoc_conover_friedman(self.df_bn)
        p_results = np.array([[-1.000000, 0.935333, 0.268619, 0.339721, 0.339721, 0.060540, 0.628079],
                              [0.935333, -1.000000, 0.302605, 0.380025, 0.380025, 0.070050, 0.685981],
                              [0.268619, 0.302605, -1.000000, 0.871144, 0.871144, 0.380025, 0.519961],
                              [0.339721, 0.380025, 0.871144, -1.000000, 1.000000, 0.302605, 0.628079],
                              [0.339721, 0.380025, 0.871144, 1.000000, -1.000000, 0.302605, 0.628079],
                              [0.060540, 0.070050, 0.380025, 0.302605, 0.302605, -1.000000, 0.141412],
                              [0.628079, 0.685981, 0.519961, 0.628079, 0.628079, 0.141412, -1.000000]])
        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_miller_friedman(self):

        results = sp.posthoc_miller_friedman(self.df_bn)

        p_results = np.array([[-1.0, 1.0, 0.9411963, 0.9724396000000001, 0.9724396000000001, 0.4717981, 0.9993864],
                              [1.0, -1.0, 0.9588993, 0.9823818000000001, 0.9823818000000001, 0.5256257, 0.9997869],
                              [0.9411963, 0.9588993, -1.0, 0.9999991, 0.9999991, 0.9823818000000001, 0.9968575999999999],
                              [0.9724396000000001, 0.9823818000000001, 0.9999991, -1.0, 1.0, 0.9588993, 0.9993864],
                              [0.9724396000000001, 0.9823818000000001, 0.9999991, 1.0, -1.0, 0.9588993, 0.9993864],
                              [0.4717981, 0.5256257, 0.9823818000000001, 0.9588993, 0.9588993, -1.0, 0.7803545999999999],
                              [0.9993864, 0.9997869, 0.9968575999999999, 0.9993864, 0.9993864, 0.7803545999999999, -1.0]])

        self.assertTrue(np.allclose(results, p_results))


    def test_posthoc_siegel_friedman(self):

        results = sp.posthoc_siegel_friedman(self.df_bn)

        p_results = np.array([[-1.000000, 0.92471904, 0.18587673, 0.25683926, 0.25683926, 0.01816302, 0.57075039],
                              [0.92471904, -1.0000000, 0.2193026, 0.2986177, 0.2986177, 0.0233422, 0.6366016],
                              [0.18587673, 0.2193026, -1.0000000, 0.8501067, 0.8501067, 0.2986177, 0.4496918],
                              [0.25683926, 0.2986177, 0.8501067, -1.000000, 1.0000000, 0.2193026, 0.5707504],
                              [0.25683926, 0.2986177, 0.8501067, 1.0000000, -1.0000000, 0.2193026, 0.5707504],
                              [0.01816302, 0.0233422, 0.2986177, 0.2193026, 0.2193026, -1.000000, 0.07260094],
                              [0.57075039, 0.6366016, 0.4496918, 0.5707504, 0.5707504, 0.07260094, -1.000000]])

        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_durbin(self):
        results = sp.posthoc_durbin(self.df_bn, p_adjust = 'holm')

        p_results = np.array([[-1.000000, 1.000000, 1.0, 1.0, 1.0, 0.381364, 1.0],
                              [1.000000, -1.000000, 1.0, 1.0, 1.0, 0.444549, 1.0],
                              [1.000000, 1.000000, -1.0, 1.0, 1.0, 1.000000, 1.0],
                              [1.000000, 1.000000, 1.0, -1.0, 1.0, 1.000000, 1.0],
                              [1.000000, 1.000000, 1.0, 1.0, -1.0, 1.000000, 1.0],
                              [0.381364, 0.444549, 1.0, 1.0, 1.0, -1.000000, 1.0],
                              [1.000000, 1.000000, 1.0, 1.0, 1.0, 1.000000, -1.0]])
        self.assertTrue(np.allclose(results, p_results))

    def test_posthoc_quade(self):
        results = sp.posthoc_quade(self.df_bn)

        p_results = np.array([[-1.00000000, 0.67651326, 0.15432143, 0.17954686, 0.2081421 , 0.02267043, 0.2081421],
                              [ 0.67651326,-1.00000000, 0.29595042, 0.33809987, 0.38443835, 0.0494024 , 0.38443835],
                              [ 0.15432143, 0.29595042,-1.00000000, 0.92586499, 0.85245022, 0.29595042, 0.85245022],
                              [ 0.17954686, 0.33809987, 0.92586499,-1.00000000, 0.92586499, 0.25789648, 0.92586499],
                              [ 0.2081421 , 0.38443835, 0.85245022, 0.92586499,-1.00000000, 0.22378308, 1.00000000],
                              [ 0.02267043, 0.0494024 , 0.29595042, 0.25789648, 0.22378308,-1.00000000, 0.22378308],
                              [ 0.2081421 , 0.38443835, 0.85245022, 0.92586499, 1.00000000, 0.22378308,-1.00000000]])
        self.assertTrue(np.allclose(results, p_results))

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

        x = [[1,2,3,4,5], [35,31,75,40,21], [10,6,9,6,1]]
        g = [['a'] * 5, ['b'] * 5, ['c'] * 5]
        results = sp.posthoc_tukey_hsd(np.concatenate(x), np.concatenate(g))
        n_results = np.array([[-1,  1,  0],[ 1, -1,  1],[ 0,  1, -1]])
        self.assertTrue(np.allclose(n_results, results))

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

    def test_posthoc_scheffe(self):

        r_results = np.array([[-1, 3.378449e-01, 3.047472e-10],
                              [3.378449e-01, -1, 2.173209e-07],
                              [3.047472e-10, 2.173209e-07, -1]])

        results = sp.posthoc_scheffe(self.df.sort_index(), val_col = 'pulse', group_col = 'kind')
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_tamhane(self):

        r_results = np.array([[-1, 2.898653e-02, 4.100954e-07],
                              [2.898653e-02, -1, 2.333996e-05],
                              [4.100954e-07, 2.333996e-05, -1]])

        results = sp.posthoc_tamhane(self.df.sort_index(), val_col = 'pulse', group_col = 'kind')
        self.assertTrue(np.allclose(results, r_results))

    def test_posthoc_tukey(self):
        r_results = np.array([[-1, 3.042955e-01, 4.308631e-10],
                              [3.042955e-01, -1, 9.946571e-08],
                              [4.308631e-10, 9.946571e-08, -1]])

        results = sp.posthoc_tukey(self.df.sort_index(), val_col = 'pulse', group_col = 'kind')
        print(results)
        self.assertTrue(np.allclose(results, r_results, atol=1.e-3))



if __name__ == '__main__':
    unittest.main()
