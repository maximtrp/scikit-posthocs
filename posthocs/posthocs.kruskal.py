import numpy as np
import scipy.stats as ss
import itertools as it
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.libqsturng import psturng
from pandas.core.frame import DataFrame

def posthoc_conover(x, val_col = None, group_col = None, p_adjust = None):

    '''
    posthoc_conover(x, val_col = None, group_col = None, p_adjust = None)

        Pairwise Test for Multiple Comparisons of Mean Rank Sums (Conover's Test).

        Parameters
        ----------
        x : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas DataFrame. Array must be two-dimensional. Second dimension may vary, i.e. groups may have different lengths.

        val_col : str, optional
            Must be specified if x is a pandas DataFrame object. The name of a column that contains values.

        group_col : str, optional
            Must be specified if x is a pandas DataFrame object. The name of a column that contains group names.

        p_adjust : str, optional
            Method for adjusting p values. See statsmodels.sandbox.stats.multicomp for details. Available methods are:
                'bonferroni' : one-step correction
                'sidak' : one-step correction
                'holm-sidak' : step-down method using Sidak adjustments
                'holm' : step-down method using Bonferroni adjustments
                'simes-hochberg' : step-up method  (independent)
                'hommel' : closed method based on Simes tests (non-negative)
                'fdr_bh' : Benjamini/Hochberg  (non-negative)
                'fdr_by' : Benjamini/Yekutieli (negative)
                'fdr_tsbh' : two stage fdr correction (non-negative)
                'fdr_tsbky' : two stage fdr correction (non-negative)

        Returns
        -------
        Numpy array of p values that may be converted to a pandas DataFrame object.

        Notes
        -----
        A tie correction are employed according to Conover (1979).

        References
        ----------
        W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures, Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.

        Examples
        --------

        >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
        >>> posthoc_conover(x, p_adjust = 'holm')
        array([[ 0.        ,  0.00119517,  0.00278329],
               [ 0.00119517,  0.        ,  0.18672227],
               [ 0.00278329,  0.18672227,  0.        ]])

    '''

    def compare_conover(i, j):
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        B = (1. / x_lens[i] + 1. / x_lens[j])
        D = (x_len_overall - 1. - H) / (x_len_overall - x_len)
        t_value = diff / np.sqrt(S2 * B * D)
        p_value = 2. * (1. - ss.t.cdf(np.abs(t_value), df = x_len_overall - x_len))
        return p_value

    if isinstance(x, DataFrame):
        x.sort_values(by=[group_col, val_col], inplace=True)
        x_len = x[group_col].unique().size
        x_lens = x.groupby(by=group_col).count().values.ravel()
        x_flat = x[val_col].values

    else:
        x = np.array(x)
        x = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_flat = np.concatenate(x)
        x_len = len(x)
        x_lens = np.asarray([len(a) for a in x])

    x_len_overall = len(x_flat)

    if any(x_lens == 0):
        raise("All groups must contain data")

    x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
    x_ranks = ss.rankdata(x_flat)
    x_ranks_grouped = np.array([x_ranks[j:j + x_lens[i]] for i, j in enumerate(x_lens_cumsum)])
    x_ranks_avg = [np.mean(z) for z in x_ranks_grouped]
    x_ties = ss.tiecorrect(x_ranks)

    H = ss.kruskal(*x)[0]

    if x_ties == 1:
        S2 = x_len_overall * (x_len_overall + 1) / 12.
    else:
        S2 = (1. / (x_len_overall - 1)) * (np.sum(x_ranks ** 2) - (x_len_overall * (((x_len_overall + 1)**2) / 4.)))

    vs = np.arange(x_len, dtype=np.float)[:,None].T.repeat(x_len, axis=0)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    combs = it.combinations(range(x_len), 2)

    for i,j in combs:
        vs[i, j] = compare_conover(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]
    vs[tri_lower] = vs[tri_upper].T

    return vs



def posthoc_dunn(x, val_col = None, group_col = None, p_adjust = None):

    '''
    posthoc_dunn(x, val_col = None, group_col = None, p_adjust = None)

        Pairwise Test for Multiple Comparisons of Mean Rank Sums (Dunn's Test).

        Parameters
        ----------
        x : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas DataFrame. Array must be two-dimensional. Second dimension may vary, i.e. groups may have different lengths.

        val_col : str, optional
            Must be specified if x is a pandas DataFrame object. The name of a column that contains values.

        group_col : str, optional
            Must be specified if x is a pandas DataFrame object. The name of a column that contains group names.

        p_adjust : str, optional
            Method for adjusting p values. See statsmodels.sandbox.stats.multicomp for details. Available methods are:
                'bonferroni' : one-step correction
                'sidak' : one-step correction
                'holm-sidak' : step-down method using Sidak adjustments
                'holm' : step-down method using Bonferroni adjustments
                'simes-hochberg' : step-up method  (independent)
                'hommel' : closed method based on Simes tests (non-negative)
                'fdr_bh' : Benjamini/Hochberg  (non-negative)
                'fdr_by' : Benjamini/Yekutieli (negative)
                'fdr_tsbh' : two stage fdr correction (non-negative)
                'fdr_tsbky' : two stage fdr correction (non-negative)

        Returns
        -------
        Numpy array of p values that may be converted to a pandas DataFrame object.

        Notes
        -----
        A tie correction will be employed according to Glantz (2012).

        References
        ----------
        O.J. Dunn (1964). Multiple comparisons using rank sums. Technometrics, 6, 241-252.
        S.A. Glantz (2012), Primer of Biostatistics. New York: McGraw Hill.

        Examples
        --------

        >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
        >>> posthoc_dunn(x, p_adjust = 'holm')
        array([[ 0.          0.01764845  0.04131415]
               [ 0.01764845  0.          0.45319956]
               [ 0.04131415  0.45319956  0.        ]])

    '''

    def compare_dunn(i, j):
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        A = x_len_overall * (x_len_overall + 1.) / 12.
        B = (1. / x_lens[i] + 1. / x_lens[j])
        z_value = diff / np.sqrt((A - x_ties) * B)
        p_value = 2. * (1. - ss.norm.cdf(np.abs(z_value)))
        return p_value

    def get_ties(x):
        x_sorted = np.array(np.sort(x))
        tie_sum = 0
        for a in x_sorted:
            n_ties = len(x_sorted[x_sorted == a])
            if n_ties > 1:
                tie_sum += n_ties ** 3 - n_ties
        c = tie_sum / (12 * (x_len_overall - 1))
        return c

    if isinstance(x, DataFrame):
        x.sort_values(by=[group_col, val_col], inplace=True)
        x_len = x[group_col].unique().size
        x_lens = x.groupby(by=group_col).count().values.ravel()
        x_flat = x[val_col].values

    else:
        x = np.array(x)
        x = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_flat = np.concatenate(x)
        x_len = len(x)
        x_lens = np.asarray([len(a) for a in x])

    x_len_overall = len(x_flat)

    if any(x_lens == 0):
        raise("All groups must contain data")

    x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
    x_ranks = ss.rankdata(x_flat)
    x_ranks_grouped = np.array([x_ranks[j:j + x_lens[i]] for i, j in enumerate(x_lens_cumsum)])
    x_ranks_avg = [np.mean(z) for z in x_ranks_grouped]
    x_ties = get_ties(x_ranks)

    vs = np.arange(x_len, dtype=np.float)[:,None].T.repeat(x_len, axis=0)
    combs = it.combinations(range(x_len), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    for i,j in combs:
        vs[i, j] = compare_dunn(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs[tri_upper].T

    return vs


def posthoc_nemenyi(x, val_col = None, group_col = None,  dist = 'chi', p_adjust = None):

    '''
    posthoc_nemenyi(x, val_col = None, group_col = None,  dist = 'chi', p_adjust = None)

        Pairwise Test for Multiple Comparisons of Mean Rank Sums (Nemenyi's Test).

        Parameters
        ----------
        x : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas DataFrame. Array must be two-dimensional. Second dimension may vary, i.e. groups may have different lengths.

        val_col : str, optional
            Must be specified if x is a pandas DataFrame object. The name of a column that contains values.

        group_col : str, optional
            Must be specified if x is a pandas DataFrame object. The name of a column that contains group names.

        dist : str, optional
            Method for determining the p value. The default distribution is "chi" (chi-squared), else "tukey" (studentized range).

        p_adjust : str, optional
            Method for adjusting p values. See statsmodels.sandbox.stats.multicomp for details. Available methods are:
                'bonferroni' : one-step correction
                'sidak' : one-step correction
                'holm-sidak' : step-down method using Sidak adjustments
                'holm' : step-down method using Bonferroni adjustments
                'simes-hochberg' : step-up method  (independent)
                'hommel' : closed method based on Simes tests (non-negative)
                'fdr_bh' : Benjamini/Hochberg  (non-negative)
                'fdr_by' : Benjamini/Yekutieli (negative)
                'fdr_tsbh' : two stage fdr correction (non-negative)
                'fdr_tsbky' : two stage fdr correction (non-negative)

        Returns
        -------
        Numpy array of p values that may be converted to a pandas DataFrame object.

        Notes
        -----
        A tie correction will be employed according to Glantz (2012).

        References
        ----------
        Lothar Sachs (1997), Angewandte Statistik. Berlin: Springer. Pages: 395-397, 662-664.

        Examples
        --------

        >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
        >>> posthoc_nemenyi(x, p_adjust = 'holm')
        array([[ 0.          0.06618715  0.13541729]
               [ 0.06618715  0.          0.75361555]
               [ 0.13541729  0.75361555  0.        ]])

    '''

    def compare_stats_chi(i, j):
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        A = x_len_overall * (x_len_overall + 1) / 12.
        B = (1. / x_lens[i] + 1. / x_lens[j])
        chi = diff ** 2 / (A * B)
        return chi

    def compare_stats_tukey(i, j):
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        B = (1. / x_lens[i] + 1. / x_lens[j])
        q = diff / np.sqrt((x_len_overall * (x_len_overall + 1) / 12.) * B)
        return q

    def get_ties(x):
        x_sorted = np.array(np.sort(x))
        tie_sum = 0
        for a in x_sorted:
            n_ties = len(x_sorted[x_sorted == a])
            if n_ties > 1:
                tie_sum += n_ties ** 3 - n_ties
        c = np.min(1, 1. - tie_sum / (x_len_overall ** 3 - x_len_overall))
        return c

    if isinstance(x, DataFrame):
        x.sort_values(by=[group_col, val_col], inplace=True)
        x_len = x[group_col].unique().size
        x_lens = x.groupby(by=group_col).count().values.ravel()
        x_flat = x[val_col].values

    else:
        x = np.array(x)
        x = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_flat = np.concatenate(x)
        x_len = len(x)
        x_lens = np.asarray([len(a) for a in x])

    x_len_overall = len(x_flat)

    if any(x_lens == 0):
        raise("All groups must contain data")

    x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
    x_ranks = ss.rankdata(x_flat)
    x_ranks_grouped = np.array([x_ranks[j:j + x_lens[i]] for i, j in enumerate(x_lens_cumsum)])
    x_ranks_avg = [np.mean(z) for z in x_ranks_grouped]
    x_ties = ss.tiecorrect(x_ranks)

    vs = np.arange(x_len, dtype=np.float)[:,None].T.repeat(x_len, axis=0)
    combs = it.combinations(range(x_len), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    if dist == 'chi':
        for i,j in combs:
            vs[i, j] = compare_stats_chi(i, j) / x_ties

        vs[tri_upper] = 1 - ss.chi2.cdf(vs[tri_upper], x_len - 1)
        if p_adjust:
            vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    elif dist == 'tukey':
        for i,j in combs:
            vs[i, j] = compare_stats_tukey(i, j) * np.sqrt(2)

        vs[tri_upper] = psturng(vs[tri_upper], x_len, np.inf)
        if p_adjust:
            vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs[tri_upper].T

    return vs
