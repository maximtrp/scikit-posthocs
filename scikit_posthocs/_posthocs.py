import numpy as np
import scipy.stats as ss
import itertools as it
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from pandas import DataFrame, Categorical, Series

def posthoc_conover(a, val_col = None, group_col = None, p_adjust = None, sort = True):

    '''Post-hoc pairwise test for multiple comparisons of mean rank sums
    (Conover's test). May be used after Kruskal-Wallis one-way analysis of
    variance by ranks to do pairwise comparisons.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas DataFrame.
            Array must be two-dimensional. Second dimension may vary,
            i.e. groups may have different lengths.

        val_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains values.

        group_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        p_adjust : str, optional
            Method for adjusting p values. See statsmodels.sandbox.stats.multicomp
            for details. Available methods are:
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

        sort : bool, optional
            Specifies whether to sort DataFrame by group_col or not. Recommended
            unless you sort your data manually.

        Returns
        -------
        Numpy ndarray if `a` is an array-like object else pandas DataFrame of p values.

        Notes
        -----
        A tie correction are employed according to Conover (1979).

        References
        ----------
        W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures, Tech.
        Rep. LA-7677-MS, Los Alamos Scientific Laboratory.

        Examples
        --------

        >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
        >>> sp.posthoc_conover(x, p_adjust = 'holm')
        array([[-1.        ,  0.00119517,  0.00278329],
               [ 0.00119517, -1.        ,  0.18672227],
               [ 0.00278329,  0.18672227, -1.        ]])

    '''

    def compare_conover(i, j):
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        B = (1. / x_lens[i] + 1. / x_lens[j])
        D = (x_len_overall - 1. - H) / (x_len_overall - x_len)
        t_value = diff / np.sqrt(S2 * B * D)
        p_value = 2. * ss.t.sf(np.abs(t_value), df = x_len_overall - x_len)
        return p_value

    def get_ties(x):
        x_sorted = np.array(np.sort(x))
        tie_sum = 0
        pos = 0
        while pos < x_len_overall:
            n_ties = len(x_sorted[x_sorted == x_sorted[pos]])
            pos = pos + n_ties
            if n_ties > 1:
                tie_sum += n_ties ** 3. - n_ties
        c = np.min([1., 1. - tie_sum / (x_len_overall ** 3. - x_len_overall)])
        return c

    if isinstance(a, DataFrame):
        x = a.copy()
        if not sort:
            x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_groups_unique = np.asarray(x[group_col].unique())
        x_len = x_groups_unique.size
        x_lens = x.groupby(by=group_col)[val_col].count().values
        x_flat = x[val_col].values
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
        x_grouped = np.array([x_flat[j:j + x_lens[i]] for i, j in enumerate(x_lens_cumsum)])

    else:
        x = np.array(a)
        x_grouped = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_flat = np.concatenate(x_grouped)
        x_len = len(x_grouped)
        x_lens = np.asarray([len(a) for a in x_grouped])
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]

    x_len_overall = len(x_flat)

    if any(x_lens == 0):
        raise ValueError("All groups must contain data")

    x_ranks = ss.rankdata(x_flat)
    x_ranks_grouped = np.array([x_ranks[j:j + x_lens[i]] for i, j in enumerate(x_lens_cumsum)])
    x_ranks_avg = [np.mean(z) for z in x_ranks_grouped]
    x_ties = get_ties(x_ranks) #ss.tiecorrect(x_ranks)

    H = ss.kruskal(*x_grouped)[0]

    if x_ties == 1:
        S2 = x_len_overall * (x_len_overall + 1.) / 12.
    else:
        S2 = (1. / (x_len_overall - 1.)) * (np.sum(x_ranks ** 2.) - (x_len_overall * (((x_len_overall + 1.)**2.) / 4.)))

    vs = np.zeros((x_len, x_len), dtype=np.float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    combs = it.combinations(range(x_len), 2)

    for i,j in combs:
        vs[i, j] = compare_conover(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]
    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)

    if isinstance(x, DataFrame):
        return DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)
    else:
        return vs

def posthoc_dunn(a, val_col = None, group_col = None, p_adjust = None, sort = True):

    '''Post-hoc pairwise test for multiple comparisons of mean rank sums
    (Dunn's test). May be used after Kruskal-Wallis one-way analysis of
    variance by ranks to do pairwise comparisons.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas DataFrame.
            Array must be two-dimensional. Second dimension may vary,
            i.e. groups may have different lengths.

        val_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains values.

        group_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        p_adjust : str, optional
            Method for adjusting p values. See statsmodels.sandbox.stats.multicomp
            for details. Available methods are:
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

        sort : bool, optional
            Specifies whether to sort DataFrame by group_col or not. Recommended
            unless you sort your data manually.

        Returns
        -------
        Numpy ndarray if `a` is an array-like object else pandas DataFrame of p values.

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
        >>> sp.posthoc_dunn(x, p_adjust = 'holm')
        array([[-1.          0.01764845  0.04131415]
               [ 0.01764845 -1.          0.45319956]
               [ 0.04131415  0.45319956 -1.        ]])

    '''

    def compare_dunn(i, j):
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        A = x_len_overall * (x_len_overall + 1.) / 12.
        B = (1. / x_lens[i] + 1. / x_lens[j])
        z_value = diff / np.sqrt((A - x_ties) * B)
        p_value = 2. * ss.norm.sf(np.abs(z_value))
        return p_value

    def get_ties(x):
        x_sorted = np.array(np.sort(x))
        tie_sum = 0
        pos = 0
        while pos < x_len_overall:
            n_ties = len(x_sorted[x_sorted == x_sorted[pos]])
            pos = pos + n_ties
            if n_ties > 1:
                tie_sum += n_ties ** 3. - n_ties
        c = tie_sum / (12. * (x_len_overall - 1))
        return c

    if isinstance(a, DataFrame):
        x = a.copy()
        if not sort:
            x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)

        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_groups_unique = np.asarray(x[group_col].unique())
        x_len = x_groups_unique.size
        x_lens = x.groupby(by=group_col)[val_col].count().values
        x_flat = x[val_col].values

    else:
        x = np.array(a)
        x = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_flat = np.concatenate(x)
        x_len = len(x)
        x_lens = np.asarray([len(a) for a in x])

    x_len_overall = len(x_flat)

    if any(x_lens == 0):
        raise ValueError("All groups must contain data")

    x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
    x_ranks = ss.rankdata(x_flat)
    x_ranks_grouped = np.array([x_ranks[j:j + x_lens[i]] for i, j in enumerate(x_lens_cumsum)])
    x_ranks_avg = [np.mean(z) for z in x_ranks_grouped]
    x_ties = get_ties(x_ranks)

    vs = np.zeros((x_len, x_len), dtype=np.float)
    combs = it.combinations(range(x_len), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    for i,j in combs:
        vs[i, j] = compare_dunn(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)

    if isinstance(x, DataFrame):
        return DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)
    else:
        return vs

def posthoc_nemenyi(a, val_col = None, group_col = None,  dist = 'chi', sort = True):

    '''Post-hoc pairwise test for multiple comparisons of mean rank sums
    (Nemenyi's test). May be used after Kruskal-Wallis one-way analysis of
    variance by ranks to do pairwise comparisons.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame. Array must be two-dimensional. Second dimension may vary,
            i.e. groups may have different lengths.

        val_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains values.

        group_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        dist : str, optional
            Method for determining the p value. The default distribution is "chi"
            (chi-squared), else "tukey" (studentized range).

        sort : bool, optional
            Specifies whether to sort DataFrame by group_col or not. Recommended
            unless you sort your data manually.

        Returns
        -------
        Numpy ndarray if `a` is an array-like object else pandas DataFrame of p values.

        Notes
        -----
        A tie correction will be employed according to Glantz (2012).

        References
        ----------
        Lothar Sachs (1997), Angewandte Statistik. Berlin: Springer. Pages: 395-397, 662-664.

        Examples
        --------

        >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
        >>> sp.posthoc_nemenyi(x)
        array([[-1.        ,  0.02206238,  0.06770864],
               [ 0.02206238, -1.        ,  0.75361555],
               [ 0.06770864,  0.75361555, -1.        ]])

    '''

    def compare_stats_chi(i, j):
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        A = x_len_overall * (x_len_overall + 1.) / 12.
        B = (1. / x_lens[i] + 1. / x_lens[j])
        chi = diff ** 2. / (A * B)
        return chi

    def compare_stats_tukey(i, j):
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        B = (1. / x_lens[i] + 1. / x_lens[j])
        q = diff / np.sqrt((x_len_overall * (x_len_overall + 1.) / 12.) * B)
        return q

    def get_ties(x):
        x_sorted = np.array(np.sort(x))
        tie_sum = 0
        pos = 0
        while pos < x_len_overall:
            n_ties = len(x_sorted[x_sorted == x_sorted[pos]])
            pos = pos + n_ties
            if n_ties > 1:
                tie_sum += n_ties ** 3. - n_ties
        c = np.min([1., 1. - tie_sum / (x_len_overall ** 3. - x_len_overall)])
        return c

    def get_ties_conover(x):
        x_sorted = np.array(np.sort(x))
        tie_sum = 0
        pos = 0
        while pos < x_len_overall:
            n_ties = len(x_sorted[x_sorted == x_sorted[pos]])
            pos = pos + n_ties
            if n_ties > 1:
                tie_sum += n_ties ** 3. - n_ties
        c = np.min([1., 1. - tie_sum / (x_len_overall ** 3. - x_len_overall)])
        return c

    if isinstance(a, DataFrame):
        x = a.copy()
        if not sort:
            x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)

        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_groups_unique = np.asarray(x[group_col].unique())
        x_len = x_groups_unique.size
        x_lens = x.groupby(by=group_col)[val_col].count().values
        x_flat = x[val_col].values

    else:
        x = np.array(a)
        x = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_flat = np.concatenate(x)
        x_len = len(x)
        x_lens = np.asarray([len(a) for a in x])

    x_len_overall = len(x_flat)

    if any(x_lens == 0):
        raise ValueError("All groups must contain data")

    x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
    x_ranks = ss.rankdata(x_flat)
    x_ranks_grouped = np.array([x_ranks[j:j + x_lens[i]] for i, j in enumerate(x_lens_cumsum)])
    x_ranks_avg = [np.mean(z) for z in x_ranks_grouped]
    x_ties = get_ties(x_ranks)

    vs = np.zeros((x_len, x_len), dtype=np.float)
    combs = it.combinations(range(x_len), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    if dist == 'chi':
        for i,j in combs:
            vs[i, j] = compare_stats_chi(i, j) / x_ties

        vs[tri_upper] = ss.chi2.sf(vs[tri_upper], x_len - 1)

    elif dist == 'tukey':
        for i,j in combs:
            vs[i, j] = compare_stats_tukey(i, j) * np.sqrt(2.)

        vs[tri_upper] = psturng(vs[tri_upper], x_len, np.inf)

    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)

    if isinstance(x, DataFrame):
        return DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)
    else:
        return vs

def posthoc_nemenyi_friedman(a, y_col = None, block_col = None, group_col = None, melted = False, sort = False):

    '''Calculate pairwise comparisons using Nemenyi post-hoc test for
    unreplicated blocked data. This test is usually conducted post-hoc after
    significant results of the Friedman's test. The statistics refer to upper
    quantiles of the studentized range distribution (Tukey).

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame.

            If `melted` is set to False (default), `a` is a typical matrix of
            block design, i.e. rows are blocks, and columns are groups. In this
            case you do not need to specify col arguments.

            If `a` is an array and `melted` is set to True,
            y_col, block_col and group_col must specify the indices of columns
            containing elements of correspondary type.

            If `a` is a Pandas DataFrame and `melted` is set to True,
            y_col, block_col and group_col must specify columns names (strings).

        y_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains y data.

        block_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains block names.

        group_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        melted : bool, optional
            Specifies if data are given as melted columns "y", "blocks", and
            "groups".

        sort : bool, optional
            If True, sort data by block and group columns.

        Returns
        -------
        Pandas DataFrame containing p values.

        Notes
        -----
        A one-way ANOVA with repeated measures that is also referred to as ANOVA with
        unreplicated block design can also be conducted via Friedman's test. The
        consequent post-hoc pairwise multiple comparison test according to Nemenyi is
        conducted with this function.

        This function does not test for ties.

        References
        ----------
        Janez Demsar (2006), Statistical comparisons of classifiers over
        multiple data sets, Journal of Machine Learning Research, 7, 1-30.

        P. Nemenyi (1963) Distribution-free Multiple Comparisons. Ph.D. thesis,
        Princeton University.

        Lothar Sachs (1997), Angewandte Statistik. Berlin: Springer. Pages: 668-675.

        Examples
        --------
        >>> # Non-melted case, x is a block design matrix, i.e. rows are blocks
        >>> # and columns are groups.
        >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
        >>> sp.posthoc_nemenyi_friedman(x)

    '''

    if melted and not all([block_col, group_col, y_col]):
        raise ValueError('block_col, group_col, y_col should be explicitly specified if using melted data')

    def compare_stats(i, j):
        dif = np.abs(R[groups[i]] - R[groups[j]])
        qval = dif / np.sqrt(k * (k + 1.) / (6. * n))
        return qval

    if isinstance(a, DataFrame) and not melted:
        group_col = 'groups'
        block_col = 'blocks'
        y_col = 'y'
        x = a.melt(id_vars=block_col, var_name=group_col, value_name=y_col)

    elif not isinstance(a, DataFrame):
        x = np.array(a)
        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))

        if not melted:
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'
            x.columns.name = group_col
            x.index.name = block_col
            x = x.reset_index().melt(id_vars=block_col, var_name=group_col, value_name=y_col)

        else:
            x.columns[group_col] = 'groups'
            x.columns[block_col] = 'blocks'
            x.columns[y_col] = 'y'
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'

    #if not sort:
    #    x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
    #    x[block_col] = Categorical(x[block_col], categories=x[block_col].unique(), ordered=True)
    x.sort_values(by=[group_col, block_col], ascending=True, inplace=True)
    x.dropna(inplace=True)

    groups = x[group_col].unique()
    k = groups.size
    n = x[block_col].unique().size

    x['mat'] = x.groupby(block_col)[y_col].rank()
    R = x.groupby(group_col)['mat'].mean()
    vs = np.zeros((k, k), dtype=np.float)
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    for i, j in combs:
        vs[i, j] = compare_stats(i, j)

    vs *= np.sqrt(2.)
    vs[tri_upper] = psturng(vs[tri_upper], k, np.inf)
    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)
    return DataFrame(vs, index=groups, columns=groups)

def posthoc_conover_friedman(a, y_col = None, block_col = None, group_col = None, melted = False, sort = False, p_adjust = None):

    '''Calculate pairwise comparisons using Conover post-hoc test for unreplicated
        blocked data. This test is usually conducted post-hoc after significant results
        of the Friedman test. The statistics refer to the Student t distribution.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame.

            If `melted` is set to False (default), `a` is a typical matrix of
            block design, i.e. rows are blocks, and columns are groups. In this
            case you do not need to specify col arguments.

            If `a` is an array and `melted` is set to True,
            y_col, block_col and group_col must specify the indices of columns
            containing elements of correspondary type.

            If `a` is a Pandas DataFrame and `melted` is set to True,
            y_col, block_col and group_col must specify columns names (strings).

        y_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains y data.

        block_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains block names.

        group_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        melted : bool, optional
            Specifies if data are given as melted columns "y", "blocks", and
            "groups".

        sort : bool, optional
            If True, sort data by block and group columns.

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
        Pandas DataFrame containing p values.

        Notes
        -----
        A one-way ANOVA with repeated measures that is also referred to as ANOVA
        with unreplicated block design can also be conducted via the
        friedman.test. The consequent post-hoc pairwise multiple comparison test
        according to Conover is conducted with this function.

        If y is a matrix, than the columns refer to the treatment and the rows
        indicate the block.

        References
        ----------
        W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures,
        Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.

        W. J. Conover (1999), Practical nonparametric Statistics, 3rd. Edition,
        Wiley.

        Examples
        --------
        >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
        >>> sp.posthoc_conover_friedman(x)

    '''

    if melted and not all([block_col, group_col, y_col]):
        raise ValueError('block_col, group_col, y_col should be explicitly specified if using melted data')

    def compare_stats(i, j):
        dif = np.abs(R[groups[i]] - R[groups[j]])
        tval = dif / np.sqrt(A / B)
        pval = 2 * ss.t.sf(np.abs(tval), df = (n-1)*(k-1))
        return pval

    if isinstance(a, DataFrame) and not melted:
        group_col = 'groups'
        block_col = 'blocks'
        y_col = 'y'
        x = a.melt(id_vars=block_col, var_name=group_col, value_name=y_col)

    elif not isinstance(a, DataFrame):
        x = np.array(a)
        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))

        if not melted:
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'
            x.columns.name = group_col
            x.index.name = block_col
            x = x.reset_index().melt(id_vars=block_col, var_name=group_col, value_name=y_col)

        else:
            x.columns[group_col] = 'groups'
            x.columns[block_col] = 'blocks'
            x.columns[y_col] = 'y'
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'

    #if not sort:
    #    x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
    #    x[block_col] = Categorical(x[block_col], categories=x[block_col].unique(), ordered=True)
    x.sort_values(by=[group_col,block_col], ascending=True, inplace=True)
    x.dropna(inplace=True)

    groups = x[group_col].unique()
    k = groups.size
    n = x[block_col].unique().size

    x['mat'] = x.groupby(block_col)[y_col].rank()
    R = x.groupby(group_col)['mat'].sum()
    A1 = (x['mat'] ** 2).sum()
    C1 = (n * k * (k + 1) ** 2) / 4
    TT = np.sum([((R[g] - ((n * (k + 1))/2)) ** 2) for g in groups])
    T1 = ((k - 1) * TT) / (A1 - C1)
    A = 2 * k * (1 - T1 / (k * (n-1))) * ( A1 - C1)
    B = (n - 1) * (k - 1)

    vs = np.zeros((k, k), dtype=np.float)
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    for i, j in combs:
        vs[i, j] = compare_stats(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)
    return DataFrame(vs, index=groups, columns=groups)

def posthoc_siegel_friedman(a, y_col = None, block_col = None, group_col = None, melted = False, sort = False, p_adjust = None):

    '''Siegel and Castellan's All-Pairs Comparisons Test for Unreplicated Blocked Data.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame.

            If `melted` is set to False (default), `a` is a typical matrix of
            block design, i.e. rows are blocks, and columns are groups. In this
            case you do not need to specify col arguments.

            If `a` is an array and `melted` is set to True,
            y_col, block_col and group_col must specify the indices of columns
            containing elements of correspondary type.

            If `a` is a Pandas DataFrame and `melted` is set to True,
            y_col, block_col and group_col must specify columns names (strings).

        y_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains y data.

        block_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains block names.

        group_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        melted : bool, optional
            Specifies if data are given as melted columns "y", "blocks", and
            "groups".

        sort : bool, optional
            If True, sort data by block and group columns.

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
        Pandas DataFrame containing p values.

        Notes
        -----
        For all-pairs comparisons in a two factorial unreplicated complete block design
        with non-normally distributed residuals, Siegel and Castellan's test can be
        performed on Friedman-type ranked data.

        References
        ----------
        S. Siegel, N. J. Castellan Jr. (1988), Nonparametric Statistics for the
            Behavioral Sciences. 2nd ed. New York: McGraw-Hill.

        Examples
        --------
        >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
        >>> sp.posthoc_siegel_friedman(x)

    '''

    if melted and not all([block_col, group_col, y_col]):
        raise ValueError('block_col, group_col, y_col should be explicitly specified if using melted data')

    def compare_stats(i, j):
        dif = np.abs(R[groups[i]] - R[groups[j]])
        zval = dif / np.sqrt(k * (k + 1) / (6 * n))
        return zval

    if isinstance(a, DataFrame) and not melted:
        group_col = 'groups'
        block_col = 'blocks'
        y_col = 'y'
        x = a.melt(id_vars=block_col, var_name=group_col, value_name=y_col)

    elif not isinstance(a, DataFrame):
        x = np.array(a)
        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))

        if not melted:
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'
            x.columns.name = group_col
            x.index.name = block_col
            x = x.reset_index().melt(id_vars=block_col, var_name=group_col, value_name=y_col)

        else:
            x.columns[group_col] = 'groups'
            x.columns[block_col] = 'blocks'
            x.columns[y_col] = 'y'
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'

    #if not sort:
    #    x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
    #    x[block_col] = Categorical(x[block_col], categories=x[block_col].unique(), ordered=True)
    x.sort_values(by=[group_col,block_col], ascending=True, inplace=True)
    x.dropna(inplace=True)

    groups = x[group_col].unique()
    k = groups.size
    n = x[block_col].unique().size

    x['mat'] = x.groupby(block_col)[y_col].rank()
    R = x.groupby(group_col)['mat'].mean()

    vs = np.zeros((k, k), dtype=np.float)
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    for i, j in combs:
        vs[i, j] = compare_stats(i, j)
    vs = 2 * ss.norm.sf(np.abs(vs))
    vs[vs > 1] = 1

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_durbin(a, y_col = None, block_col = None, group_col = None, melted = False, sort = False, p_adjust = None):

    '''Pairwise post-hoc test for multiple comparisons of rank sums according to
    Durbin and Conover for a two-way balanced incomplete block design (BIBD).

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame.

            If `melted` is set to False (default), `a` is a typical matrix of block design,
            i.e. rows are blocks, and columns are groups. In this case you do
            not need to specify col arguments.

            If `a` is an array and `melted` is set to True,
            y_col, block_col and group_col must specify the indices of columns
            containing elements of correspondary type.

            If `a` is a Pandas DataFrame and `melted` is set to True,
            y_col, block_col and group_col must specify columns names (string).

        y_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains y data.

        block_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains block names.

        group_col : str or int
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        melted : bool, optional
            Specifies if data are given as melted columns "y", "blocks", and
            "groups".

        sort : bool, optional
            If True, sort data by block and group columns.

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
        Pandas DataFrame containing p values.

        References
        ----------
        W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures,
              Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.
        W. J. Conover (1999), Practical nonparametric Statistics, 3rd. Edition, Wiley.

        Examples
        --------
        >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
        >>> sp.posthoc_durbin(x)

    '''

    if melted and not all([block_col, group_col, y_col]):
        raise ValueError('block_col, group_col, y_col should be explicitly specified if using melted data')

    def compare_stats(i, j):
        dif = np.abs(Rj[groups[i]] - Rj[groups[j]])
        tval = dif / denom
        pval = 2. * ss.t.sf(np.abs(tval), df = df)
        return pval

    if isinstance(a, DataFrame) and not melted:
        group_col = 'groups'
        block_col = 'blocks'
        y_col = 'y'
        x = a.melt(id_vars=block_col, var_name=group_col, value_name=y_col)

    elif not isinstance(a, DataFrame):
        x = np.array(a)
        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))

        if not melted:
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'
            x.columns.name = group_col
            x.index.name = block_col
            x = x.reset_index().melt(id_vars=block_col, var_name=group_col, value_name=y_col)

        else:
            x.columns[group_col] = 'groups'
            x.columns[block_col] = 'blocks'
            x.columns[y_col] = 'y'
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'

    if not sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x[block_col] = Categorical(x[block_col], categories=x[block_col].unique(), ordered=True)
    x.sort_values(by=[block_col, group_col], ascending=True, inplace=True)
    x.dropna(inplace=True)

    groups = np.asarray(x[group_col].unique())
    t = len(groups)
    b = x[block_col].unique().size
    r = b
    k = t
    x['y_ranked'] = x.groupby(block_col)[y_col].rank()
    Rj = x.groupby(group_col)['y_ranked'].sum()
    A = (x['y_ranked'] ** 2).sum()
    C = (b * k * (k + 1) ** 2) / 4.
    D = (Rj ** 2).sum() - r * C
    T1 = (t - 1) / (A - C) * D
    denom = np.sqrt(((A - C) * 2 * r) / (b * k - b - t + 1) * (1 - T1 / (b * (k -1))))
    df = b * k - b - t + 1

    vs = np.zeros((t, t), dtype=np.float)
    combs = it.combinations(range(t), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    for i, j in combs:
        vs[i, j] = compare_stats(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)
    return DataFrame(vs, index=groups, columns=groups)

def posthoc_anderson(a, val_col = None, group_col = None, midrank = True, sort = False, p_adjust = None):

    '''Anderson-Darling Pairwise Test for k-samples. Tests the null hypothesis that
        k-samples are drawn from the same population without having to specify the
        distribution function of that population.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame.

        val_col : str
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains y data.

        group_col : str
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        midrank : bool, optional
            Type of Anderson-Darling test which is computed. If set to True (default), the
            midrank test applicable to continuous and discrete populations is performed. If
            False, the right side empirical distribution is used.

        sort : bool, optional
            If True, sort data by block and group columns.

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
        Pandas DataFrame containing p values.

        References
        ----------
        Scholz, F. W and Stephens, M. A. (1987), K-Sample Anderson-Darling Tests,
            Journal of the American Statistical Association, Vol. 82, pp. 918-924.

        Examples
        --------
        >>> x = np.array([[2.9, 3.0, 2.5, 2.6, 3.2], [3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]])
        >>> sp.posthoc_anderson(x)

    '''

    if isinstance(a, DataFrame):
        x = a.copy()
        if not all([group_col, val_col]):
            raise ValueError('group_col, val_col must be explicitly specified')
    else:
        x = np.array(a)

        if not all([group_col, val_col]):
            try:
                groups = np.array([len(a) * [i + 1] for i, a in enumerate(x)])
                groups = sum(groups.tolist(), [])
                x = sum(x.tolist(), [])
                x = np.column_stack([x, groups])
                val_col = 0
                group_col = 1
            except:
                raise ValueError('array cannot be processed, provide val_col and group_col args')

        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))
        x.rename(columns={group_col: 'groups', val_col: 'y'}, inplace=True)
        group_col = 'groups'
        val_col = 'y'

    if not sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
    x.sort_values(by=[group_col], ascending=True, inplace=True)

    groups = np.asarray(x[group_col].unique())
    k = groups.size
    vs = np.zeros((k, k), dtype=np.float)
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    for i, j in combs:
        vs[i, j] = ss.anderson_ksamp([x.loc[x[group_col] == groups[i], val_col], x.loc[x[group_col] == groups[j], val_col]])[2]

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)
    return DataFrame(vs, index=groups, columns=groups)

def posthoc_quade(a, y_col = None, block_col = None, group_col = None, dist = 't', melted = False, sort = False, p_adjust = None):

    '''Calculate pairwise comparisons using Quade's post-hoc test for
    unreplicated blocked data. This test is usually conducted post-hoc after
    significant results of the omnibus test.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame.

            If `melted` is set to False (default), `a` is a typical matrix of
            block design, i.e. rows are blocks, and columns are groups. In this
            case you do not need to specify col arguments.

            If `a` is an array and `melted` is set to True,
            y_col, block_col and group_col must specify the indices of columns
            containing elements of correspondary type.

            If `a` is a Pandas DataFrame and `melted` is set to True,
            y_col, block_col and group_col must specify columns names (string).

        y_col : str or int, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains y data.

        block_col : str or int, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains block names.

        group_col : str or int, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        dist : str, optional
            Method for determining p values.
            The default distribution is "t", else "normal".

        melted : bool, optional
            Specifies if data are given as melted columns "y", "blocks", and
            "groups".

        sort : bool, optional
            If True, sort data by block and group columns.

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
        Pandas DataFrame containing p values.

        References
        ----------
        W. J. Conover (1999), Practical nonparametric Statistics, 3rd. Edition, Wiley.

        N. A. Heckert and J. J. Filliben (2003). NIST Handbook 148: Dataplot Reference Manual,
        Volume 2: Let Subcommands and Library Functions.
        National Institute of Standards and Technology Handbook Series, June 2003.

        D. Quade (1979), Using weighted rankings in the analysis of complete blocks
        with additive block effects.
        Journal of the American Statistical Association, 74, 680-683.

        Examples
        --------
        >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
        >>> sp.posthoc_quade(x)

    '''

    if melted and not all([block_col, group_col, y_col]):
        raise ValueError('block_col, group_col, y_col should be explicitly specified if using melted data')

    def compare_stats_t(i, j):
        dif = np.abs(S[groups[i]] - S[groups[j]])
        tval = dif / denom
        pval = 2. * ss.t.sf(np.abs(tval), df = (b - 1) * (k - 1))
        return pval

    def compare_stats_norm(i, j):
        dif = np.abs(W[groups[i]] * ff - W[groups[j]] * ff)
        zval = dif / denom
        pval = 2. * ss.norm.sf(np.abs(zval))
        return pval

    if isinstance(a, DataFrame) and not melted:
        group_col = 'groups'
        block_col = 'blocks'
        y_col = 'y'
        x = a.melt(id_vars=block_col, var_name=group_col, value_name=y_col)

    elif not isinstance(a, DataFrame):
        x = np.array(a)
        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))

        if not melted:
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'
            x.columns.name = group_col
            x.index.name = block_col
            x = x.reset_index().melt(id_vars=block_col, var_name=group_col, value_name=y_col)

        else:
            x.columns[group_col] = 'groups'
            x.columns[block_col] = 'blocks'
            x.columns[y_col] = 'y'
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'

    if not sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x[block_col] = Categorical(x[block_col], categories=x[block_col].unique(), ordered=True)
    x.sort_values(by=[block_col, group_col], ascending=True, inplace=True)
    x.dropna(inplace=True)

    groups = np.asarray(x[group_col].unique())
    k = len(groups)
    b = x[block_col].unique().size

    x['r'] = x.groupby(block_col)[y_col].rank()
    q = (x.groupby(block_col)[y_col].max() - x.groupby(block_col)[y_col].min()).rank()
    x['rr'] = x['r'] - (k + 1)/2
    x['s'] = x.apply(lambda x, y: x['rr'] * y[x['blocks']], axis=1, args=(q,))
    x['w'] = x.apply(lambda x, y: x['r'] * y[x['blocks']], axis=1, args=(q,))
    A = (x['s'] ** 2).sum()
    S = x.groupby(group_col)['s'].sum()
    B = np.sum(S ** 2) / b
    W = x.groupby(group_col)['w'].sum()

    vs = np.zeros((k, k), dtype=np.float)
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    if dist == 't':
        denom = np.sqrt((2 * b * (A - B)) / ((b - 1) * (k - 1)))

        for i, j in combs:
            vs[i, j] = compare_stats_t(i, j)

    else:
        n = b * k
        denom = np.sqrt((k * (k + 1) * (2 * n + 1) * (k-1)) / (18 * n * (n + 1)))
        ff = 1 / (b * (b + 1)/2)

        for i, j in combs:
            vs[i, j] = compare_stats_norm(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)
    return DataFrame(vs, index=groups, columns=groups)

def posthoc_mackwolfe(a, val_col, group_col, p = None, n_perm = 100, sort = False, p_adjust = None):

    '''Mack-Wolfe Test for Umbrella Alternatives.

    In dose-finding studies one may assume an increasing treatment effect with
    increasing dose level. However, the test subject may actually succumb to
    toxic effects at high doses, which leads to decresing treatment effects.

    The scope of the Mack-Wolfe Test is to test for umbrella alternatives for
    either a known or unknown point P (i.e. dose-level), where the peak
    (umbrella point) is present.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame.

        val_col : str or int
            Name (string) or index (int) of a column in a pandas DataFrame or an
            array that contains quantitative data.

        group_col : str or int
            Name (string) or index (int) of a column in a pandas DataFrame or an
            array that contains group names.

        p : int, optional
            The a-priori known peak as an ordinal number of the treatment group
            including the zero dose level, i.e. p = {1, …, k}. Defaults to None.

        sort : bool, optional
            If True, sort data by block and group columns.

        Returns
        -------
        Pandas DataFrame containing p values.

        References
        ----------
        Chen, I.Y. (1991) Notes on the Mack-Wolfe and Chen-Wolfe Tests for
            Umbrella Alternatives. Biom. J., 33, 281-290.
        Mack, G.A., Wolfe, D. A. (1981) K-sample rank tests for umbrella
            alternatives. J. Amer. Statist. Assoc., 76, 175-181.

        Examples
        --------
        >>> x = np.array([[10,'a'], [59,'a'], [76,'b'], [10, 'b']])
        >>> sp.posthoc_mackwolfe(x, val_col = 0, group_col = 1)

    '''

    if isinstance(a, DataFrame):
        x = a.copy()
        if not all([group_col, val_col]):
            raise ValueError('group_col, val_col must be explicitly specified')
    else:
        x = np.array(a)

        if not all([group_col, val_col]):
            try:
                groups = np.array([len(a) * [i + 1] for i, a in enumerate(x)])
                groups = sum(groups.tolist(), [])
                x = sum(x.tolist(), [])
                x = np.column_stack([x, groups])
                val_col = 0
                group_col = 1
            except:
                raise ValueError('Array cannot be processed, provide val_col and group_col args')

        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))
        x.rename(columns={group_col: 'groups', val_col: 'y'}, inplace=True)
        group_col = 'groups'
        val_col = 'y'

    if not sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
    x.sort_values(by=[group_col], ascending=True, inplace=True)

    k = x[group_col].unique().size

    if p:
        if p > k:
            print("Selected 'p' > number of groups:", str(p), " > ", str(k))
            return False
        elif p < 1:
            print("Selected 'p' < 1: ", str(p))
            return False

    Rij = x[val_col].rank()
    n = x.groupby(group_col).count()

    def _fn(Ri, Rj):
        return np.sum(Ri.apply(lambda x: Rj[Rj > x].size))

    def _ustat(Rij, g, k):
        levels = np.unique(g)
        U = np.identity(k)

        for i in range(k):
            for j in range(i):
                U[i,j] = _fn(Rij[x[group_col] == levels[i]], Rij[x[group_col] == levels[j]])
                U[j,i] = _fn(Rij[x[group_col] == levels[j]], Rij[x[group_col] == levels[i]])

        return U

    def _ap(p, U):
        tmp1 = 0
        if p > 0:
            for i in range(p):
                for j in range(i+1, p+1):
                    tmp1 += U[i,j]
        tmp2 = 0
        if p < k:
            for i in range(p, k):
                for j in range(i+1, k):
                    tmp2 += U[j,i]

        return tmp1 + tmp2

    def _n1(p, n):
        return np.sum(n[:p+1])

    def _n2(p, n):
        return np.sum(n[p:k])

    def _mean_at(p, n):
        N1 = _n1(p, n)
        N2 = _n2(p, n)
        return (N1**2 + N2**2 - np.sum(n**2) - n.iloc[p]**2)/4

    def _var_at(p, n):
        N1 = _n1(p, n)
        N2 = _n2(p, n)
        N = np.sum(n)

        var = (2 * (N1**3 + N2**3) + 3 * (N1**2 + N2**2) -\
                np.sum(n**2 * (2*n + 3)) - n.iloc[p]**2 * (2 * n.iloc[p] + 3) +\
                12 * n.iloc[p] * N1 * N2 - 12 * n.iloc[p] ** 2 * N) / 72
        return var

    if p:
        if (x.groupby(val_col).count() > 1).any().any():
            print("Ties are present")
        U = _ustat(Rij, x[group_col], k)
        est = _ap(p, U)
        mean = _mean_at(p, n)
        sd = np.sqrt(_var_at(p, n))
        stat = (est - mean)/sd
        p_value = ss.norm.sf(stat)
    else:
        U = _ustat(Rij, x[group_col], k)
        Ap = np.array([_ap(i, U) for i in range(k)]).ravel()
        mean = np.array([_mean_at(i, n) for i in range(k)]).ravel()
        var = np.array([_var_at(i, n) for i in range(k)]).ravel()
        A = (Ap - mean) / np.sqrt(var)
        stat = np.max(A)
        p = A == stat
        est = None

        mt = []
        for i in range(n_perm):

            ix = Series(np.random.permutation(Rij))
            Uix = _ustat(ix, x[group_col], k)
            Apix = np.array([_ap(i, Uix) for i in range(k)])
            Astarix = (Apix - mean) / np.sqrt(var)
            mt.append(np.max(Astarix))

        mt = np.array(mt)
        p_value = mt[mt > stat] / n_perm

    return stat, p_value


def posthoc_vanwaerden(a, val_col, group_col, sort = False, p_adjust = None):

    '''Calculate pairwise multiple comparisons between group levels
    according to van der Waerden.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame.

        val_col : str or int
            Name (string) or index (int) of a column in a pandas DataFrame or an
            array that contains quantitative data.

        group_col : str or int
            Name (string) or index (int) of a column in a pandas DataFrame or an
            array that contains group names.

        sort : bool, optional
            If True, sort data by block and group columns.

        p_adjust : str, optional
            Method for adjusting p values. See statsmodels.sandbox.stats.multicomp for details.
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
        Pandas DataFrame containing p values.

        Notes
        -----
        For one-factorial designs with samples that do not meet the assumptions
        for one-way-ANOVA and subsequent post-hoc tests, the van der Waerden test
        vanWaerden.test using normal scores can be employed. Provided that
        significant differences were detected by this global test, one may be
        interested in applying post-hoc tests according to van der Waerden
        for pairwise multiple comparisons of the group levels.

        There is no tie correction applied in this function.

        References
        ----------
        W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures,
              Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.

        Examples
        --------
        >>> x = np.array([[10,'a'], [59,'a'], [76,'b'], [10, 'b']])
        >>> sp.posthoc_vanwaerden(x, val_col = 0, group_col = 1)

    '''

    def compare_stats(i, j):
        dif = np.abs(A[groups[i]] - A[groups[j]])
        B = 1. / nj[groups[i]] + 1. / nj[groups[j]]
        tval = dif / np.sqrt(s2 * (n - 1. - sts)/(n - k) * B)
        pval = 2. * ss.t.sf(np.abs(tval), df = n - k)
        return pval

    if isinstance(a, DataFrame):
        x = a.copy()
        if not all([group_col, val_col]):
            raise ValueError('group_col, val_col must be explicitly specified')
    else:
        x = np.array(a)

        if not all([group_col, val_col]):
            try:
                groups = np.array([len(a) * [i + 1] for i, a in enumerate(x)])
                groups = sum(groups.tolist(), [])
                x = sum(x.tolist(), [])
                x = np.column_stack([x, groups])
                val_col = 0
                group_col = 1
            except:
                raise ValueError('array cannot be processed, provide val_col and group_col args')

        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))
        x.rename(columns={group_col: 'groups', val_col: 'y'}, inplace=True)
        group_col = 'groups'
        val_col = 'y'

    if not sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
    x.sort_values(by=[group_col], ascending=True, inplace=True)

    groups = np.asarray(x[group_col].unique())
    n = x[val_col].size
    k = groups.size
    r = ss.rankdata(x[val_col])
    x['z_scores'] = ss.norm.ppf(r / (n + 1))

    aj = x.groupby(group_col)['z_scores'].sum()
    nj = x.groupby(group_col)['z_scores'].count()
    s2 = (1. / (n - 1.)) * (x['z_scores'] ** 2.).sum()
    sts = (1. / s2) * np.sum(aj ** 2. / nj)
    param = k - 1
    A = aj / nj

    vs = np.zeros((k, k), dtype=np.float)
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    for i, j in combs:
        vs[i, j] = compare_stats(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)
    return DataFrame(vs, index=groups, columns=groups)

def posthoc_ttest(a, val_col = None, group_col = None, pool_sd = False, equal_var = True, p_adjust = None, sort = True):

    '''Pairwise T test for multiple comparisons of independent groups. May be
    used after ordinary ANOVA to do pairwise comparisons.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame. Array must be two-dimensional.

        val_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains values.

        group_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

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

        sort : bool, optional
            Specifies whether to sort DataFrame by group_col or not. Recommended
            unless you sort your data manually.

        Returns
        -------
        Numpy ndarray if `a` is an array-like object else pandas DataFrame of p values.

        References
        ----------

        .. [1] http://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test
        .. [2] http://en.wikipedia.org/wiki/Welch%27s_t_test

        Examples
        --------

        >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
        >>> sp.posthoc_ttest(x, p_adjust = 'holm')
        array([[-1.        ,  0.04600899,  0.31269089],
               [ 0.04600899, -1.        ,  0.6327077 ],
               [ 0.31269089,  0.6327077 , -1.        ]])

    '''

    if isinstance(a, DataFrame):
        x = a.copy()
        if not sort:
            x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)

        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_lens = x.groupby(by=group_col)[val_col].count().values
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
        x_grouped = np.array([x[val_col][j:(j + x_lens[i])] for i, j in enumerate(x_lens_cumsum)])

    else:
        x = np.array(a)
        x_grouped = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_lens = np.asarray([len(a) for a in x_grouped])
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]

    if any(x_lens == 0):
        raise ValueError("All groups must contain data")

    x_len = len(x_grouped)
    vs = np.zeros((x_len, x_len), dtype=np.float)
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
    np.fill_diagonal(vs, -1)

    if isinstance(x, DataFrame):
        groups_unique = np.asarray(x[group_col].unique())
        return DataFrame(vs, index=groups_unique, columns=groups_unique)
    else:
        return vs

def posthoc_tukey_hsd(x, g, alpha = 0.05):

    '''Pairwise comparisons with TukeyHSD confidence intervals. This is a
    convenience function to make statsmodels `pairwise_tukeyhsd` method more
    applicable for further use.

        Parameters
        ----------
        x : array_like or pandas Series object, 1d
            An array, any object exposing the array interface, containing
            the response variable. NaN values will cause an error. Please
            handle manually.

        g : array_like or pandas Series object, 1d
            An array, any object exposing the array interface, containing
            groups names. Can be string or integers.

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
        >>> sp.posthoc_tukey_hsd(np.concatenate(x), np.concatenate(g))
        array([[-1,  1,  0],
               [ 1, -1,  1],
               [ 0,  1, -1]])

    '''

    result = pairwise_tukeyhsd(x, g, alpha=0.05)
    groups = np.array(result.groupsunique, dtype=np.str)
    groups_len = len(groups)

    vs = np.zeros((groups_len, groups_len), dtype=np.int)

    for a in result.summary()[1:]:
        a0 = str(a[0])
        a1 = str(a[1])
        a0i = np.where(groups == a0)[0][0]
        a1i = np.where(groups == a1)[0][0]
        vs[a0i, a1i] = 1 if str(a[5]) == 'True' else 0

    vs = np.triu(vs)
    np.fill_diagonal(vs, -1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[tri_lower] = vs.T[tri_lower]

    return vs

def posthoc_mannwhitney(a, val_col = None, group_col = None, use_continuity = True, alternative = 'two-sided', p_adjust = None, sort = True):

    '''Pairwise comparisons with Mann-Whitney rank test.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame. Array must be two-dimensional.

        val_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains values.

        group_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        use_continuity : bool, optional
            Whether a continuity correction (1/2.) should be taken into account.
            Default is True.

        alternative : ['two-sided', 'less', or 'greater'], optional
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

        sort : bool, optional
            Specifies whether to sort DataFrame by group_col or not. Recommended
            unless you sort your data manually.

        Returns
        -------
        Numpy ndarray if `a` is an array-like object else pandas DataFrame of p values.

        Notes
        -----
        Refer to `scipy.stats.mannwhitneyu` reference page for further details.

        Examples
        --------

        >>> x = [[1,2,3,4,5], [35,31,75,40,21], [10,6,9,6,1]]
        >>> sp.posthoc_mannwhitney(x, p_adjust = 'holm')
        array([[-1.       ,  0.0357757,  0.114961 ],
               [ 0.0357757, -1.       ,  0.0357757],
               [ 0.114961 ,  0.0357757, -1.       ]])

    '''

    if isinstance(a, DataFrame):
        x = a.copy()
        if not sort:
            x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)

        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
        x_lens = x.groupby(by=group_col)[val_col].count().values
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]
        x_grouped = np.array([x[val_col][j:(j + x_lens[i])] for i, j in enumerate(x_lens_cumsum)])

    else:
        x = np.array(a)
        x_grouped = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_lens = np.asarray([len(a) for a in x_grouped])
        x_lens_cumsum = np.insert(np.cumsum(x_lens), 0, 0)[:-1]

    if any(x_lens == 0):
        raise ValueError("All groups must contain data")

    x_len = len(x_grouped)
    vs = np.zeros((x_len, x_len), dtype=np.float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    combs = it.combinations(range(x_len), 2)

    for i,j in combs:
        vs[i, j] = ss.mannwhitneyu(x_grouped[i], x_grouped[j], use_continuity=use_continuity, alternative=alternative)[1]

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]
    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)

    if isinstance(x, DataFrame):
        groups_unique = np.asarray(x[group_col].unique())
        return DataFrame(vs, index=groups_unique, columns=groups_unique)
    else:
        return vs

def posthoc_wilcoxon(a, val_col = None, group_col = None, zero_method='wilcox', correction=False, p_adjust = None, sort = False):

    '''Pairwise comparisons with Wilcoxon signed-rank test. It is a non-parametric
    version of the paired T-test.

        Parameters
        ----------
        a : array_like or pandas DataFrame object
            An array, any object exposing the array interface or a pandas
            DataFrame. Array must be two-dimensional.

        val_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains values.

        group_col : str, optional
            Must be specified if `a` is a pandas DataFrame object.
            Name of the column that contains group names.

        zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
            "pratt": Pratt treatment: includes zero-differences in the ranking
                process (more conservative)
            "wilcox": Wilcox treatment: discards all zero-differences
            "zsplit": Zero rank split: just like Pratt, but spliting the zero rank
                between positive and negative ones

        correction : bool, optional
            If True, apply continuity correction by adjusting the Wilcoxon rank
            statistic by 0.5 towards the mean value when computing the z-statistic.
            Default is False.

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

        sort : bool, optional
            Specifies whether to sort DataFrame by group_col and val_col or not.
            Default is False.

        Returns
        -------
        Numpy ndarray if `a` is an array-like object else pandas DataFrame of p values.

        Notes
        -----
        Refer to `scipy.stats.wilcoxon` reference page for further details.

        Examples
        --------

        >>> x = [[1,2,3,4,5], [35,31,75,40,21], [10,6,9,6,1]]
        >>> sp.posthoc_wilcoxon(x)
        array([[-1.        ,  0.04311445,  0.1755543 ],
               [ 0.04311445, -1.        ,  0.0421682 ],
               [ 0.1755543 ,  0.0421682 , -1.        ]])

    '''

    if isinstance(a, DataFrame):
        x = a.copy()
        if sort:
            x = x.sort_values(by=[group_col, val_col])

        groups = x.groupby(group_col).groups
        groups_names = x[group_col].unique()
        x_lens = x.groupby(group_col)[val_col].count().values
        x_grouped = np.array([x.loc[groups[g].values, val_col].values for g in groups_names])

    else:
        x = np.array(a)
        x_grouped = np.array([np.asarray(a)[~np.isnan(a)] for a in x])
        x_lens = np.asarray([len(a) for a in x_grouped])

    if any(x_lens == 0):
        raise ValueError("All groups must contain data")

    x_len = len(x_grouped)
    vs = np.zeros((x_len, x_len), dtype=np.float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    combs = it.combinations(range(x_len), 2)

    for i,j in combs:
        vs[i, j] = ss.wilcoxon(x_grouped[i], x_grouped[j], zero_method=zero_method, correction=correction)[1]

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]
    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)

    if isinstance(x, DataFrame):
        groups_unique = x[group_col].unique()
        return DataFrame(vs, index=groups_unique, columns=groups_unique)
    else:
        return vs
