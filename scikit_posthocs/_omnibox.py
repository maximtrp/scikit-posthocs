# -*- coding: utf-8 -*-

import numpy as np
import itertools as it
import scipy.stats as ss
from statsmodels.stats.libqsturng import psturng
from pandas import Categorical, Series
from scikit_posthocs._posthocs import __convert_to_df, __convert_to_block_df

def test_mackwolfe(a, val_col=None, group_col=None, p=None, n_perm=100, sort=False):

    '''Mack-Wolfe Test for Umbrella Alternatives.

    In dose-finding studies one may assume an increasing treatment effect with
    increasing dose level. However, the test subject may actually succumb to
    toxic effects at high doses, which leads to decresing treatment
    effects [1]_, [2]_.

    The scope of the Mack-Wolfe Test is to test for umbrella alternatives for
    either a known or unknown point P (i.e. dose-level), where the peak
    (umbrella point) is present.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas
        DataFrame.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    p : int, optional
        The a-priori known peak as an ordinal number of the treatment group
        including the zero dose level, i.e. p = {0, ..., k-1}. Defaults to None.

    sort : bool, optional
        If True, sort data by block and group columns.

    Returns
    -------
    p : float
        P value.
    stat : float
        Statistic.

    References
    ----------
    .. [1] Chen, I.Y. (1991) Notes on the Mack-Wolfe and Chen-Wolfe Tests for
        Umbrella Alternatives. Biom. J., 33, 281-290.
    .. [2] Mack, G.A., Wolfe, D. A. (1981) K-sample rank tests for umbrella
        alternatives. J. Amer. Statist. Assoc., 76, 175-181.

    Examples
    --------
    >>> x = x = [[22, 23, 35], [60, 59, 54], [98, 78, 50], [60, 82, 59], [22, 44, 33], [23, 21, 25]]
    >>> sp.posthoc_mackwolfe(x)

    '''

    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)

    if not sort:
        x[_group_col] = Categorical(x[_group_col], categories=x[_group_col].unique(), ordered=True)
    x.sort_values(by=[_group_col], ascending=True, inplace=True)

    k = x[_group_col].unique().size

    if p:
        if p > k:
            print("Selected 'p' > number of groups:", str(p), " > ", str(k))
            return False
        elif p < 1:
            print("Selected 'p' < 1: ", str(p))
            return False

    Rij = x[_val_col].rank()
    n = x.groupby(_group_col)[_val_col].count()

    def _fn(Ri, Rj):
        return np.sum(Ri.apply(lambda x: Rj[Rj > x].size))

    def _ustat(Rij, g, k):
        levels = np.unique(g)
        U = np.identity(k)

        for i in range(k):
            for j in range(i):
                U[i,j] = _fn(Rij[x[_group_col] == levels[i]], Rij[x[_group_col] == levels[j]])
                U[j,i] = _fn(Rij[x[_group_col] == levels[j]], Rij[x[_group_col] == levels[i]])

        return U

    def _ap(p, U):
        tmp1 = 0.
        if p > 0:
            for i in range(p):
                for j in range(i+1, p+1):
                    tmp1 += U[i,j]
        tmp2 = 0.
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
        return (N1**2. + N2**2. - np.sum(n**2.) - n.iloc[p]**2.)/4.

    def _var_at(p, n):
        N1 = _n1(p, n)
        N2 = _n2(p, n)
        N = np.sum(n)

        var = (2. * (N1**3 + N2**3) + 3. * (N1**2 + N2**2) -\
                np.sum(n**2 * (2*n + 3.)) - n.iloc[p]**2 * (2. * n.iloc[p] + 3.) +\
                12. * n.iloc[p] * N1 * N2 - 12. * n.iloc[p] ** 2 * N) / 72.
        return var

    if p:
        #if (x.groupby(_val_col).count() > 1).any().any():
        #    print("Ties are present")
        U = _ustat(Rij, x[_group_col], k)
        est = _ap(p, U)
        mean = _mean_at(p, n)
        sd = np.sqrt(_var_at(p, n))
        stat = (est - mean)/sd
        p_value = ss.norm.sf(stat)
    else:
        U = _ustat(Rij, x[_group_col], k)
        Ap = np.array([_ap(i, U) for i in range(k)]).ravel()
        mean = np.array([_mean_at(i, n) for i in range(k)]).ravel()
        var = np.array([_var_at(i, n) for i in range(k)]).ravel()
        A = (Ap - mean) / np.sqrt(var)
        stat = np.max(A)
        p = A == stat
        est = None

        mt = []
        for _ in range(n_perm):

            ix = Series(np.random.permutation(Rij))
            Uix = _ustat(ix, x[_group_col], k)
            Apix = np.array([_ap(i, Uix) for i in range(k)])
            Astarix = (Apix - mean) / np.sqrt(var)
            mt.append(np.max(Astarix))

        mt = np.array(mt)
        p_value = mt[mt > stat] / n_perm

    return p_value, stat

def test_osrt(a, val_col=None, group_col=None, sort=False):

    '''Hayter's one-sided studentised range test (OSRT) against an ordered
    alternative for normal data with equal variances [1]_.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas
        DataFrame.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    sort : bool, optional
        If True, sort data by block and group columns.

    Returns
    -------
    P value, statistic.

    Notes
    -----
    P values are computed from the Tukey distribution.

    References
    ----------
    .. [1] Hayter, A.J.(1990) A One-Sided Studentised Range Test for Testing
        Against a Simple Ordered Alternative, Journal of the American
        Statistical Association, 85, 778-785.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.test_osrt(x, val_col='values', group_col='groups')

    '''

    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)

    if not sort:
        x[_group_col] = Categorical(x[_group_col], categories=x[_group_col].unique(), ordered=True)

    x.sort_values(by=[_group_col], ascending=True, inplace=True)
    groups = np.unique(x[_group_col])
    x_grouped = x.groupby(_group_col)[_val_col]

    xi = x_grouped.mean()
    ni = x_grouped.count()
    k = groups.size
    n = len(x.index)
    df = n - k

    sigma2 = 0
    c = -1

    for i in range(k):
        for j in range(ni.iloc[i]):
            c += 1
            sigma2 += (x[_val_col].iat[c] - xi[i])**2. /df

    sigma = np.sqrt(sigma2)

    def compare(i, j):
        dif = xi.loc[groups[j]] - xi.loc[groups[i]]
        A = sigma / np.sqrt(2.) * np.sqrt(1. / ni[groups[j]] + 1. / ni[groups[i]])
        qval = np.abs(dif) / A
        return qval

    vs = np.zeros((k, k), dtype=np.float)
    combs = it.combinations(range(k), 2)

    for i,j in combs:
        vs[i, j] = compare(i, j)

    stat = np.max(vs)
    pval = psturng(stat, k, df)
    return pval, stat


def test_durbin(a, y_col=None, block_col=None, group_col=None, melted=False, sort=True):

    '''Durbin's test whether k groups (or treatments) in a two-way
    balanced incomplete block design (BIBD) have identical effects. See
    references for additional information [1]_, [2]_.

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

    Returns
    -------
    pval : float


    References
    ----------
    .. [1] N. A. Heckert, J. J. Filliben. (2003) NIST Handbook 148: Dataplot Reference
        Manual, Volume 2: Let Subcommands and Library Functions. National Institute of
        Standards and Technology Handbook Series, June 2003.
    .. [2] W. J. Conover (1999), Practical nonparametric Statistics,
        3rd. edition, Wiley.

    Examples
    --------
    >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
    >>> sp.test_durbin(x)

    '''

    if melted and not all([block_col, group_col, y_col]):
        raise ValueError('block_col, group_col, y_col should be explicitly specified if using melted data')

    x, _y_col, _group_col, _block_col = __convert_to_block_df(a, y_col, group_col, block_col, melted)

    groups = x[_group_col].unique()
    blocks = x[_block_col].unique()
    if not sort:
        x[_group_col] = Categorical(x[_group_col], categories=groups, ordered=True)
        x[_block_col] = Categorical(x[_block_col], categories=blocks, ordered=True)
    x.sort_values(by=[_block_col, _group_col], ascending=True, inplace=True)
    x.dropna(inplace=True)

    t = len(groups)
    b = len(blocks)
    r = np.unique(x.groupby(_group_col).count())
    k = np.unique(x.groupby(_block_col).count())
    if r.size > 1 and k.size > 1:
        raise ValueError('Data appear to be unbalanced. Please correct your input data')
    else:
        r = r.item()
        k = k.item()
    x['y_ranks'] = x.groupby(_block_col)[_y_col].rank()
    x['y_ranks_sum'] = x.groupby(_group_col)['y_ranks'].sum()

    A = np.sum(x['y_ranks'] ** 2.)
    C = (b * k * (k + 1)**2.) / 4.
    D = np.sum(x['y_ranks_sum'] ** 2.) - r * C
    T1 = (t - 1.) / (A - C) * D

    stat = T1
    df = t - 1
    pval = ss.chi2.sf(stat, df)

    return pval, stat, df
