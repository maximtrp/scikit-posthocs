import numpy as np
import scipy.stats as ss
import itertools as it
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from pandas import DataFrame

def posthoc_conover(x, val_col = None, group_col = None, p_adjust = None):

    '''Pairwise Test for Multiple Comparisons of Mean Rank Sums (Conover's Test).

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
        Numpy ndarray if x is an array-like object else pandas DataFrame of p values.

        Notes
        -----
        A tie correction are employed according to Conover (1979).

        References
        ----------
        W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures, Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.

        Examples
        --------

        >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
        >>> ph.posthoc_conover(x, p_adjust = 'holm')
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
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_len = x[group_col].unique().size
        x_lens = x.groupby(by=group_col)[val_col].count().values
        x_flat = x[val_col].values
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
        x_grouped = np.array([x_flat[j:j + x_lens[i]] for i, j in enumerate(x_lens_cumsum)])

    else:
        x = np.array(x)
        x_grouped = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_flat = np.concatenate(x_grouped)
        x_len = len(x_grouped)
        x_lens = np.asarray([len(a) for a in x_grouped])
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]

    x_len_overall = len(x_flat)

    if any(x_lens == 0):
        raise("All groups must contain data")

    x_ranks = ss.rankdata(x_flat)
    x_ranks_grouped = np.array([x_ranks[j:j + x_lens[i]] for i, j in enumerate(x_lens_cumsum)])
    x_ranks_avg = [np.mean(z) for z in x_ranks_grouped]
    x_ties = ss.tiecorrect(x_ranks)

    H = ss.kruskal(*x_grouped)[0]

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
    vs[tri_lower] = vs.T[tri_lower]

    if isinstance(x, DataFrame):
        groups_unique = x[group_col].unique()
        return DataFrame(vs, index=groups_unique, columns=groups_unique)
    else:
        return vs



def posthoc_dunn(x, val_col = None, group_col = None, p_adjust = None):

    '''Pairwise Test for Multiple Comparisons of Mean Rank Sums (Dunn's Test).

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
        Numpy ndarray if x is an array-like object else pandas DataFrame of p values.

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
        >>> ph.posthoc_dunn(x, p_adjust = 'holm')
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
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_len = x[group_col].unique().size
        x_lens = x.groupby(by=group_col)[val_col].count().values
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

    vs[tri_lower] = vs.T[tri_lower]

    if isinstance(x, DataFrame):
        groups_unique = x[group_col].unique()
        return DataFrame(vs, index=groups_unique, columns=groups_unique)
    else:
        return vs


def posthoc_nemenyi(x, val_col = None, group_col = None,  dist = 'chi', p_adjust = None):

    '''Pairwise Test for Multiple Comparisons of Mean Rank Sums (Nemenyi's Test).

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
        Numpy ndarray if x is an array-like object else pandas DataFrame of p values.

        Notes
        -----
        A tie correction will be employed according to Glantz (2012).

        References
        ----------
        Lothar Sachs (1997), Angewandte Statistik. Berlin: Springer. Pages: 395-397, 662-664.

        Examples
        --------

        >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
        >>> ph.posthoc_nemenyi(x, p_adjust = 'holm')
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
        tie_sum = 0.
        for a in x_sorted:
            n_ties = len(x_sorted[x_sorted == a])
            if n_ties > 1:
                tie_sum += n_ties ** 3 - n_ties
        c = np.min([1., 1. - tie_sum / (x_len_overall ** 3 - x_len_overall)])
        return c

    if isinstance(x, DataFrame):
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_len = x[group_col].unique().size
        x_lens = x.groupby(by=group_col)[val_col].count().values
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

    vs[tri_lower] = vs.T[tri_lower]

    if isinstance(x, DataFrame):
        groups_unique = x[group_col].unique()
        return DataFrame(vs, index=groups_unique, columns=groups_unique)
    else:
        return vs


def posthoc_ttest(x, val_col = None, group_col = None, pool_sd = False, equal_var = True, p_adjust = None):

    '''Pairwise T test for Multiple Comparisons of Independent Groups.

        Parameters
        ----------
        x : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame. Array must be two-dimensional. Second dimension may
            vary, i.e. groups may have different lengths.

        val_col : str, optional
            Must be specified if x is a pandas DataFrame object.
            The name of a column that contains values.

        group_col : str, optional
            Must be specified if x is a pandas DataFrame object.
            The name of a column that contains group names.

        equal_var : bool, optional
            If True (default), perform a standard independent test
            that assumes equal population variances [1]_.
            If False, perform Welch's t-test, which does not assume equal
            population variance [2]_.

        pool_sd : bool, optional
            Calculate a common SD for all groups and use that for all
            comparisons (this can be useful if some groups are small).
            This method does not actually call scipy ttest_ind() function,
            so extra arguments are ignored. Default is False.

        p_adjust : str, optional
            Method for adjusting p values.
            See statsmodels.sandbox.stats.multicomp for details.
            Available methods are:
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
        Numpy ndarray if x is an array-like object else pandas DataFrame of p values.

        References
        ----------

        .. [1] http://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test
        .. [2] http://en.wikipedia.org/wiki/Welch%27s_t_test

        Examples
        --------

        >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
        >>> ph.posthoc_ttest(x, p_adjust = 'holm')
        array([[ 0.        ,  0.04600899,  0.31269089],
               [ 0.04600899,  0.        ,  0.6327077 ],
               [ 0.31269089,  0.6327077 ,  0.        ]])

    '''

    if isinstance(x, DataFrame):
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_lens = x.groupby(by=group_col)[val_col].count().values
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
        x_grouped = np.array([x[val_col][j:(j + x_lens[i])] for i, j in enumerate(x_lens_cumsum)])

    else:
        x = np.array(x)
        x_grouped = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_lens = np.asarray([len(a) for a in x_grouped])
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]

    if any(x_lens == 0):
        raise("All groups must contain data")

    x_len = len(x_grouped)
    vs = np.arange(x_len, dtype=np.float)[:,None].T.repeat(x_len, axis=0)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    def compare_pooled(i, j):
        diff = x_means[i] - x_means[j]
        se_diff = pooled_sd * np.sqrt(1 / x_lens[i] + 1 / x_lens[j])
        t_value = diff / se_diff
        return 2 * ss.t.cdf(-np.abs(t_value), x_totaldegf)

    combs = it.combinations(range(x_len), 2)

    if pool_sd:
        x_means = np.asarray([np.mean(xi) for xi in x_grouped])
        x_sd = np.asarray([np.std(xi, ddof=1) for xi in x_grouped])
        x_degf = x_lens - 1
        x_totaldegf = np.sum(x_degf)
        pooled_sd = np.sqrt(np.sum(x_sd ** 2 * x_degf) / x_totaldegf)

        for i, j in combs:
            vs[i, j] = compare_pooled(i, j)
    else:
        for i,j in combs:
            vs[i, j] = ss.ttest_ind(x_grouped[i], x_grouped[j], equal_var=equal_var)[1]

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]
    vs[tri_lower] = vs.T[tri_lower]

    if isinstance(x, DataFrame):
        groups_unique = x[group_col].unique()
        return DataFrame(vs, index=groups_unique, columns=groups_unique)
    else:
        return vs


def posthoc_tukey_hsd(x, g, alpha = 0.05):

    '''Pairwise comparisons with TukeyHSD confidence intervals. This is a
    convenience function to make statsmodels pairwise_tukeyhsd method more
    applicable for further use.

        Parameters
        ----------
        x : array_like or pandas Series object, 1d
            An array, any object exposing the array interface, containing
            the response variable. NaN values will cause an error. Please
            handle manually.

        g : array_like or pandas Series object, 1d
            An array, any object exposing the array interface, containing
            the groups' names. Can be string or integers.

        alpha : float, optional
            Significance level for the test. Default is 0.05.

        Returns
        -------
        Numpy ndarray where 0 is False (not significant), 1 is True (significant),
        and -1 is for diagonal elements.

        Examples
        --------

        >>> x = [[1,2,3,4,5], [35,31,75,40,21], [10,6,9,6,1]]
        >>> g = [['a'] * 5, ['b'] * 5, ['c'] * 5]
        >>> ph.posthoc_tukey_hsd(np.concatenate(x), np.concatenate(g))
        array([[-1,  1,  0],
               [ 1, -1,  1],
               [ 0,  1, -1]])

    '''

    result = pairwise_tukeyhsd(x, g, alpha=0.05)
    groups = np.array(result.groupsunique, dtype=np.str)
    groups_len = len(groups)

    vs = np.arange(groups_len, dtype=np.int)[:,None].T.repeat(groups_len, axis=0)

    for a in result.summary()[1:]:
        a0 = str(a[0])
        a1 = str(a[1])
        a0i = np.where(groups == a0)[0][0]
        a1i = np.where(groups == a1)[0][0]
        vs[a0i, a1i] = 1 if str(a[5]) == 'True' else 0

    vs = np.triu(vs)
    vs[np.diag_indices(groups_len)] = -1
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[tri_lower] = vs.T[tri_lower]

    return vs

def posthoc_mannwhitney(x, val_col = None, group_col = None, use_continuity = True, alternative = 'two-sided', p_adjust = None):

    '''Pairwise comparisons with Mann-Whitney rank test. This is also known
    as pairwise two-sample Wilcoxon test.

        Parameters
        ----------
        x : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame. Array must be two-dimensional. Second dimension may
            vary, i.e. groups may have different lengths.

        val_col : str, optional
            Must be specified if x is a pandas DataFrame object.
            The name of a column that contains values.

        group_col : str, optional
            Must be specified if x is a pandas DataFrame object.
            The name of a column that contains group names.

        use_continuity : bool, optional
            Whether a continuity correction (1/2.) should be taken into account.
            Default is True.

        alternative : {'two-sided', 'less', or ‘greater’}, optional
            Whether to get the p-value for the one-sided hypothesis
            ('less' or 'greater') or for the two-sided hypothesis ('two-sided').
            Defaults to 'two-sided'.

        p_adjust : str, optional
            Method for adjusting p values.
            See statsmodels.sandbox.stats.multicomp for details.
            Available methods are:
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
        Numpy ndarray if x is an array-like object else pandas DataFrame of p values.

        Examples
        --------

        >>> x = [[1,2,3,4,5], [35,31,75,40,21], [10,6,9,6,1]]
        >>> ph.posthoc_mannwhitney(x, p_adjust = 'holm')
        array([[-1,  1,  0],
               [ 1, -1,  1],
               [ 0,  1, -1]])

    '''

    if isinstance(x, DataFrame):
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_lens = x.groupby(by=group_col)[val_col].count().values
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
        x_grouped = np.array([x[val_col][j:(j + x_lens[i])] for i, j in enumerate(x_lens_cumsum)])

    else:
        x = np.array(x)
        x_grouped = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_lens = np.asarray([len(a) for a in x_grouped])
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]

    if any(x_lens == 0):
        raise("All groups must contain data")

    x_len = len(x_grouped)
    vs = np.arange(x_len, dtype=np.float)[:,None].T.repeat(x_len, axis=0)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    combs = it.combinations(range(x_len), 2)

    for i,j in combs:
        vs[i, j] = ss.mannwhitneyu(x_grouped[i], x_grouped[j], use_continuity=use_continuity, alternative=alternative)[1]

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]
    vs[tri_lower] = vs.T[tri_lower]

    if isinstance(x, DataFrame):
        groups_unique = x[group_col].unique()
        return DataFrame(vs, index=groups_unique, columns=groups_unique)
    else:
        return vs
