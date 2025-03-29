# -*- coding: utf-8 -*-

from typing import Optional, Union, List, cast
import itertools as it
import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as ss
from pandas import DataFrame, Categorical, Series
from scikit_posthocs._posthocs import __convert_to_df, __convert_to_block_df


def test_mackwolfe(
    data: Union[ArrayLike, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    p: Optional[int] = None,
    n_perm: int = 100,
    sort: bool = False,
) -> tuple[float, float]:
    """Mack-Wolfe Test for Umbrella Alternatives.

    In dose-finding studies one may assume an increasing treatment effect with
    increasing dose level. However, the test subject may actually succumb to
    toxic effects at high doses, which leads to decresing treatment
    effects [1]_, [2]_.

    The scope of the Mack-Wolfe Test is to test for umbrella alternatives for
    either a known or unknown point P (i.e. dose-level), where the peak
    (umbrella point) is present.

    Parameters
    ----------
    data : Union[List, numpy.ndarray, DataFrame]
        An array, any object exposing the array interface or a pandas
        DataFrame with data values.

    val_col : str = None
        Name of a DataFrame column that contains dependent variable values
        (test or response variable). Values should have a non-nominal scale.
        Must be specified if ``a`` is a pandas DataFrame object.

    group_col : str = None
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if ``a`` is a pandas DataFrame object.

    p : int = None
        The a priori known peak as an ordinal number of the treatment group
        including the zero dose level, i.e. p = {0, ..., k-1}.
        Defaults to None.

    n_perm: int = 100
        Permutations number.

    sort : bool = False
        If ``True``, sort data by block and group columns.

    Returns
    -------
    tuple[float, float]
        P value and statistic.

    References
    ----------
    .. [1] Chen, I.Y. (1991) Notes on the Mack-Wolfe and Chen-Wolfe Tests for
        Umbrella Alternatives. Biom. J., 33, 281-290.
    .. [2] Mack, G.A., Wolfe, D. A. (1981) K-sample rank tests for umbrella
        alternatives. J. Amer. Statist. Assoc., 76, 175-181.

    Examples
    --------
    >>> x = [[22, 23, 35], [60, 59, 54], [98, 78, 50], [60, 82, 59], [22, 44, 33], [23, 21, 25]]
    >>> sp.posthoc_mackwolfe(x)
    """
    x, _val_col, _group_col = __convert_to_df(data, val_col, group_col)

    if not sort:
        x[_group_col] = Categorical(x[_group_col], categories=x[_group_col].unique(), ordered=True)
    x.sort_values(by=[_group_col], ascending=True, inplace=True)

    k = x[_group_col].unique().size

    if p and p > k:
        print("Selected 'p' > number of groups:", str(p), " > ", str(k))
        return (np.nan, np.nan)
    elif p is not None and p < 1:
        print("Selected 'p' < 1: ", str(p))
        return (np.nan, np.nan)

    Rij = x[_val_col].rank()
    n = cast(Series, x.groupby(_group_col, observed=True)[_val_col].count())

    def _fn(Ri, Rj):
        return np.sum(Ri.apply(lambda x: Rj[Rj > x].size))

    def _ustat(Rij, g, k):
        levels = np.unique(g)
        U = np.identity(k)

        for i in range(k):
            for j in range(i):
                U[i, j] = _fn(Rij[x[_group_col] == levels[i]], Rij[x[_group_col] == levels[j]])
                U[j, i] = _fn(Rij[x[_group_col] == levels[j]], Rij[x[_group_col] == levels[i]])

        return U

    def _ap(p, U) -> float:
        tmp1 = 0.0
        if p > 0:
            for i in range(p):
                for j in range(i + 1, p + 1):
                    tmp1 += U[i, j]
        tmp2 = 0.0
        if p < k:
            for i in range(p, k):
                for j in range(i + 1, k):
                    tmp2 += U[j, i]

        return tmp1 + tmp2

    def _n1(p: int, n: Series) -> float:
        return np.sum(n[: p + 1])

    def _n2(p: int, n: Series) -> float:
        return np.sum(n[p:k])

    def _mean_at(p, n) -> float:
        N1 = _n1(p, n)
        N2 = _n2(p, n)
        return (N1**2.0 + N2**2.0 - np.sum(n**2.0) - n.iloc[p] ** 2.0) / 4.0

    def _var_at(p: int, n: Series) -> float:
        N1 = _n1(p, n)
        N2 = _n2(p, n)
        N = np.sum(n)

        var = (
            2.0 * (N1**3 + N2**3)
            + 3.0 * (N1**2 + N2**2)
            - np.sum(n**2 * (2 * n + 3.0))
            - n.iloc[p] ** 2.0 * (2.0 * n.iloc[p] + 3.0)
            + 12.0 * n.iloc[p] * N1 * N2
            - 12.0 * n.iloc[p] ** 2.0 * N
        ) / 72.0
        return var

    if p:
        # if (x.groupby(_val_col).count() > 1).any().any():
        #    print("Ties are present")
        U = _ustat(Rij, x[_group_col], k)
        est = _ap(p, U)
        mean = _mean_at(p, n)
        sd = np.sqrt(_var_at(p, n))
        stat = (est - mean) / sd
        p_value = ss.norm.sf(stat).item()
    else:
        U = _ustat(Rij, x[_group_col], k)
        Ap = np.array([_ap(i, U) for i in range(k)]).ravel()
        mean = np.array([_mean_at(i, n) for i in range(k)]).ravel()
        var = np.array([_var_at(i, n) for i in range(k)]).ravel()
        A = (Ap - mean) / np.sqrt(var)
        stat = float(np.max(A))

        mt = []
        for _ in range(n_perm):
            ix = Series(np.random.permutation(Rij))
            uix = _ustat(ix, x[_group_col], k)
            apix = np.array([_ap(i, uix) for i in range(k)])
            astarix = (apix - mean) / np.sqrt(var)
            mt.append(np.max(astarix))

        mt = np.array(mt)
        p_value = mt[mt > stat].size / n_perm

    return p_value, stat


def test_osrt(
    data: Union[List, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> tuple[float, float, int]:
    """Hayter's one-sided studentised range test (OSRT)

    Tests a hypothesis against an ordered alternative for normal data with
    equal variances [1]_.

    Parameters
    ----------
    data : Union[List, numpy.ndarray, DataFrame]
        An array, any object exposing the array interface or a pandas
        DataFrame with data values.

    val_col : str = None
        Name of a DataFrame column that contains dependent variable values
        (test or response variable). Values should have a non-nominal scale.
        Must be specified if ``a`` is a pandas DataFrame object.

    group_col : str = None
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    sort : bool = False
        If True, sort data by block and group columns.

    Returns
    -------
    tuple[float, float, int]
        P value, statistic, and number of degrees of freedom.

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
    """
    x, _val_col, _group_col = __convert_to_df(data, val_col, group_col)

    if not sort:
        x[_group_col] = Categorical(x[_group_col], categories=x[_group_col].unique(), ordered=True)

    x.sort_values(by=[_group_col], ascending=True, inplace=True)
    groups = np.unique(x[_group_col])
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]

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
            sigma2 += (x[_val_col].iloc[c] - xi.iloc[i]) ** 2.0 / df

    sigma = np.sqrt(sigma2)

    def compare(i, j):
        dif = xi.loc[groups[j]] - xi.loc[groups[i]]
        A = sigma / np.sqrt(2.0) * np.sqrt(1.0 / ni[groups[j]] + 1.0 / ni[groups[i]])
        qval = np.abs(dif) / A
        return qval

    vs = np.zeros((k, k), dtype=float)
    combs = it.combinations(range(k), 2)

    for i, j in combs:
        vs[i, j] = compare(i, j)

    stat = np.max(vs)
    pval = ss.studentized_range.sf(stat, k, df)
    return pval, stat, df


def test_durbin(
    data: Union[List, np.ndarray, DataFrame],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
    sort: bool = True,
) -> tuple[float, float, int]:
    """Durbin's test whether k groups (or treatments) in a two-way
    balanced incomplete block design (BIBD) have identical effects. See
    references for additional information [1]_, [2]_.

    Parameters
    ----------
    data : Union[List, np.ndarray, DataFrame]
        An array, any object exposing the array interface or a pandas
        DataFrame with data values.

        If ``melted`` argument is set to False (default), ``a`` is a typical
        matrix of block design, i.e. rows are blocks, and columns are groups.
        In this case, you do not need to specify col arguments.

        If ``a`` is an array and ``melted`` is set to True,
        y_col, block_col and group_col must specify the indices of columns
        containing elements of correspondary type.

        If ``a`` is a Pandas DataFrame and ``melted`` is set to True,
        y_col, block_col and group_col must specify columns names (string).

    y_col : Union[str, int] = None
        Must be specified if ``a`` is a melted pandas DataFrame object.
        Name of the column that contains y data.

    group_col : Union[str, int] = None
        Must be specified if ``a`` is a melted pandas DataFrame object.
        Name of the column that contains group names.

    block_col : Union[str, int] = None
        Must be specified if ``a`` is a melted pandas DataFrame object.
        Name of the column that contains block names.

    block_id_col : Union[str, int] = None
        Must be specified if ``a`` is a melted pandas DataFrame object.
        Name of the column that contains identifiers of block names.
        In most cases, this is the same as `block_col` except for those
        cases when you have multiple instances of the same blocks.

    melted : bool = False
        Specifies if data are given as melted columns "y", "blocks", and
        "groups".

    sort : bool = False
        If True, sort data by block and group columns.

    Returns
    -------
    tuple[float, float, int]
        P value, statistic, and number of degrees of freedom.

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
    """
    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        data, y_col, group_col, block_col, block_id_col, melted
    )

    groups = x[_group_col].unique()
    blocks = x[_block_id_col].unique()
    if not sort:
        x[_group_col] = Categorical(x[_group_col], categories=groups, ordered=True)
        x[_block_col] = Categorical(x[_block_col], categories=blocks, ordered=True)
    x.sort_values(by=[_block_col, _group_col], ascending=True, inplace=True)
    x.dropna(inplace=True)

    t = len(groups)
    b = len(blocks)
    r = float(b)
    k = float(t)

    x["y_ranks"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    rs = x.groupby(_group_col, observed=True)["y_ranks"].sum().to_numpy()

    A = float(np.sum(x["y_ranks"] ** 2.0))
    C = float(b * k * (k + 1) ** 2.0) / 4.0
    D = float(np.sum(rs**2.0)) - r * C
    T1 = (t - 1.0) / (A - C) * D
    stat = T1
    df = t - 1
    pval = ss.chi2.sf(stat, df).item()

    return pval, stat, df
