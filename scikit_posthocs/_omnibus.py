# -*- coding: utf-8 -*-

import warnings
from typing import Optional, Union, List, cast
import itertools as it
import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as ss
from pandas import DataFrame, Categorical, Series
from scikit_posthocs._posthocs import (
    __complete_block_matrix,
    __convert_to_df,
    __convert_to_block_df,
)


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

    if p is not None and not 0 <= p < k:
        raise ValueError(f"p must be between 0 and {k - 1}; got {p}")

    Rij = x[_val_col].rank()
    n = cast(Series, x.groupby(_group_col, observed=True)[_val_col].count())
    levels = x[_group_col].unique()
    group_indices = [np.flatnonzero(x[_group_col].to_numpy() == level) for level in levels]

    def _fn(Ri, Rj):
        ordered = np.sort(Rj)
        return np.sum(ordered.size - np.searchsorted(ordered, Ri, side="right"))

    def _ustat(ranks):
        grouped_ranks = [ranks[index] for index in group_indices]
        U = np.identity(k)

        for i in range(k):
            for j in range(i):
                U[i, j] = _fn(grouped_ranks[i], grouped_ranks[j])
                U[j, i] = _fn(grouped_ranks[j], grouped_ranks[i])

        return U

    def _ap(p, U) -> float:
        increasing = np.triu(U[: p + 1, : p + 1], k=1).sum()
        decreasing = np.tril(U[p:, p:], k=-1).sum()
        return float(increasing + decreasing)

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

    if p is not None:
        # if (x.groupby(_val_col).count() > 1).any().any():
        #    print("Ties are present")
        U = _ustat(Rij.to_numpy())
        est = _ap(p, U)
        mean = _mean_at(p, n)
        sd = np.sqrt(_var_at(p, n))
        stat = (est - mean) / sd
        p_value = ss.norm.sf(stat).item()
    else:
        rank_values = Rij.to_numpy()
        U = _ustat(rank_values)
        Ap = np.array([_ap(i, U) for i in range(k)]).ravel()
        mean = np.array([_mean_at(i, n) for i in range(k)]).ravel()
        var = np.array([_var_at(i, n) for i in range(k)]).ravel()
        A = (Ap - mean) / np.sqrt(var)
        stat = float(np.max(A))

        mt = []
        for _ in range(n_perm):
            ix = np.random.permutation(rank_values)
            uix = _ustat(ix)
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

    residuals = x[_val_col] - x_grouped.transform("mean")
    sigma = np.sqrt(np.sum(residuals**2.0) / df)

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
    matrix = __complete_block_matrix(data, melted, sort)
    if matrix is not None:
        values, _ = matrix
        b, t = values.shape
        ranks = ss.rankdata(values, axis=1, method="average")
        rank_sums = ranks.sum(axis=0)
        A = float(np.sum(ranks**2.0))
        C = float(b * t * (t + 1.0) ** 2.0) / 4.0
        D = float(np.sum(rank_sums**2.0)) - b * C
        stat = (t - 1.0) / (A - C) * D
        df = t - 1
        return ss.chi2.sf(stat, df).item(), stat, df

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


def test_jonckheere(
    data: Union[ArrayLike, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    alternative: str = "two-sided",
    continuity: bool = False,
    sort: bool = False,
) -> tuple[float, float]:
    """Jonckheere-Terpstra test for ordered alternatives.

    The null hypothesis, H\\ :sub:`0`: theta_1 = theta_2 = ... = theta_k, is
    tested against a simple order hypothesis,
    H\\ :sub:`A`: theta_1 <= theta_2 <= ... <= theta_k (theta_1 < theta_k), where
    group order is taken from the natural (or categorical) order of
    `group_col`, i.e. the order in which groups first appear in the data
    unless `sort` is True [1]_.

    Parameters
    ----------
    data : Union[List, numpy.ndarray, DataFrame]
        An array, any object exposing the array interface or a pandas
        DataFrame with data values.

    val_col : str = None
        Name of a DataFrame column that contains dependent variable values
        (test or response variable). Values should have a non-nominal scale.
        Must be specified if ``data`` is a pandas DataFrame object.

    group_col : str = None
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable), given in the a priori hypothesized
        order. Must be specified if ``data`` is a pandas DataFrame object.

    alternative : str = "two-sided"
        The alternative hypothesis, one of "two-sided", "greater", or
        "less".

    continuity : bool = False
        Whether to apply a continuity correction, as for Kendall's tau.

    sort : bool = False
        If True, sort data by group_col (alphabetically) instead of
        preserving the order in which groups appear in the data.

    Returns
    -------
    tuple[float, float]
        P value and z statistic.

    Notes
    -----
    P values are computed from the standard normal distribution. In the
    presence of ties, the standard deviation of the Jonckheere-Terpstra
    statistic is corrected following Kloke and McKean (2015) [2]_.

    References
    ----------
    .. [1] A. R. Jonckheere (1954), A distribution-free k-sample test
        against ordered alternatives, Biometrika, 41, 133-145.
    .. [2] J. Kloke, J. W. McKean (2015), Nonparametric statistical methods
        using R, Boca Raton, FL: Chapman & Hall/CRC.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> x = [[22, 23, 35], [60, 59, 54], [98, 78, 50]]
    >>> sp.test_jonckheere(x)
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be one of 'two-sided', 'greater', or 'less'")

    x, _val_col, _group_col = __convert_to_df(data, val_col, group_col)

    if not sort:
        x[_group_col] = Categorical(x[_group_col], categories=x[_group_col].unique(), ordered=True)
    x = x.sort_values(by=[_group_col], ascending=True)

    groups = x[_group_col].unique()
    k = groups.size
    n = len(x.index)
    nij = x.groupby(_group_col, observed=True)[_val_col].count()
    grouped_vals = [
        np.sort(x.loc[x[_group_col] == g, _val_col].dropna().to_numpy()) for g in groups
    ]

    def uij(xi, xj):
        left = np.searchsorted(xj, xi, side="left")
        right = np.searchsorted(xj, xi, side="right")
        return np.sum(xj.size - right) + 0.5 * np.sum(right - left)

    J = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            J += uij(grouped_vals[i], grouped_vals[j])

    nij_arr = nij.to_numpy().astype(float)
    mu = (n**2.0 - np.sum(nij_arr**2.0)) / 4.0
    S = J - mu

    ranks = ss.rankdata(x[_val_col])
    ties = np.unique(ranks, return_counts=True)[1]
    has_ties = np.any(ties > 1)

    if not has_ties:
        s = np.sqrt(
            (n**2.0 * (2.0 * n + 3.0) - np.sum(nij_arr**2.0 * (2.0 * nij_arr + 3.0))) / 72.0
        )
    else:
        warnings.warn(
            "Ties are present. Jonckheere z was corrected for ties.",
            UserWarning,
            stacklevel=2,
        )
        nt = np.unique(x[_val_col], return_counts=True)[1].astype(float)
        s = np.sqrt(
            (
                n * (n - 1.0) * (2.0 * n + 5.0)
                - np.sum(nij_arr * (nij_arr - 1.0) * (2.0 * nij_arr + 5.0))
                - np.sum(nt * (nt - 1.0) * (2.0 * nt + 5.0))
            )
            / 72.0
            + (
                np.sum(nij_arr * (nij_arr - 1.0) * (nij_arr - 2.0))
                * np.sum(nt * (nt - 1.0) * (nt - 2.0))
            )
            / (36.0 * n * (n - 1.0) * (n - 2.0))
            + (np.sum(nij_arr * (nij_arr - 1.0)) * np.sum(nt * (nt - 1.0))) / (8.0 * n * (n - 1.0))
        )

    if continuity:
        S = np.sign(S) * (np.abs(S) - 0.5)

    stat = S / s

    if alternative == "two-sided":
        pval = 2.0 * min(ss.norm.sf(np.abs(stat)), 0.5)
    elif alternative == "greater":
        pval = ss.norm.sf(stat)
    else:
        pval = ss.norm.cdf(stat)

    return float(pval), float(stat)


def test_page(
    data: Union[ArrayLike, DataFrame],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    alternative: str = "two-sided",
    melted: bool = False,
    sort: bool = False,
) -> tuple[float, float]:
    """Page's ordered aligned rank sum test for a randomized complete block
    design against an a priori ordered alternative (group order is taken
    from the natural order of `group_col`, i.e. the order in which groups
    appear in the data / columns, unless `sort` is True) [1]_.

    Parameters
    ----------
    data : Union[List, np.ndarray, DataFrame]
        An array, any object exposing the array interface or a pandas
        DataFrame with data values.

        If ``melted`` is set to False (default), ``data`` is a typical
        matrix of block design, i.e. rows are blocks, and columns are
        groups given in the a priori hypothesized order. In this case, you
        do not need to specify col arguments.

        If ``data`` is an array and ``melted`` is set to True, y_col,
        block_col and group_col must specify the indices of columns
        containing elements of correspondary type.

        If ``data`` is a Pandas DataFrame and ``melted`` is set to True,
        y_col, block_col and group_col must specify columns names (string).

    y_col : Union[str, int] = None
        Must be specified if ``data`` is a melted pandas DataFrame object.
        Name of the column that contains y data.

    group_col : Union[str, int] = None
        Must be specified if ``data`` is a melted pandas DataFrame object.
        Name of the column that contains group names, in the a priori
        hypothesized order.

    block_col : Union[str, int] = None
        Must be specified if ``data`` is a melted pandas DataFrame object.
        Name of the column that contains block names.

    block_id_col : Union[str, int] = None
        Must be specified if ``data`` is a melted pandas DataFrame object.
        Name of the column that contains identifiers of block names.

    alternative : str = "two-sided"
        The alternative hypothesis, one of "two-sided", "greater", or
        "less".

    melted : bool = False
        Specifies if data are given as melted columns "y", "blocks", and
        "groups".

    sort : bool = False
        If True, sort data by group_col (alphabetically) instead of
        preserving the order in which groups appear in the data.

    Returns
    -------
    tuple[float, float]
        P value and z statistic.

    Notes
    -----
    P values are computed from the standard normal distribution, with a
    continuity correction always applied (following Sachs 1997).

    References
    ----------
    .. [1] E. B. Page (1963), Ordered hypotheses for multiple treatments: A
        significance test for linear ranks, Journal of the American
        Statistical Association, 58, 216-230.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
    >>> sp.test_page(x, alternative="greater")
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be one of 'two-sided', 'greater', or 'less'")

    matrix = __complete_block_matrix(data, melted, sort)
    if matrix is not None:
        values, _ = matrix
        n, k = values.shape
        rank_sums = ss.rankdata(values, axis=1, method="average").sum(axis=0)
        L = float(np.sum(rank_sums * np.arange(1, k + 1)))
        eL = n * k * (k + 1.0) ** 2.0 / 4.0
        varL = n * k**2.0 * (k + 1.0) * (k**2.0 - 1.0) / 144.0
        stat = (L - eL - 0.5) / np.sqrt(varL)
        if alternative == "two-sided":
            pval = 2.0 * min(ss.norm.sf(np.abs(stat)), 0.5)
        elif alternative == "greater":
            pval = ss.norm.sf(stat)
        else:
            pval = ss.norm.cdf(stat)
        return float(pval), float(stat)

    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        data, y_col, group_col, block_col, block_id_col, melted
    )

    groups = x[_group_col].unique()
    blocks = x[_block_id_col].unique()
    if not sort:
        x[_group_col] = Categorical(x[_group_col], categories=groups, ordered=True)
        x[_block_col] = Categorical(x[_block_col], categories=blocks, ordered=True)
    x = x.sort_values(by=[_block_col, _group_col], ascending=True)
    x.dropna(inplace=True)

    k = len(groups)
    n = len(blocks)

    x["y_ranks"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    r_sum = x.groupby(_group_col, observed=True)["y_ranks"].sum().to_numpy()

    weights = np.arange(1, k + 1)
    L = float(np.sum(r_sum * weights))
    eL = n * k * (k + 1.0) ** 2.0 / 4.0
    varL = n * k**2.0 * (k + 1.0) * (k**2.0 - 1.0) / 144.0

    stat = (L - eL - 0.5) / np.sqrt(varL)

    if alternative == "two-sided":
        pval = 2.0 * min(ss.norm.sf(np.abs(stat)), 0.5)
    elif alternative == "greater":
        pval = ss.norm.sf(stat)
    else:
        pval = ss.norm.cdf(stat)

    return float(pval), float(stat)


def test_hartley(
    data: Union[ArrayLike, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    n_perm: int = 100000,
    sort: bool = False,
) -> tuple[float, float, int]:
    """Hartley's maximum F-ratio test of homogeneity of variances.

    Tests the null hypothesis that the variances in each of the groups are
    equal, against the alternative that at least one differs [1]_.

    Parameters
    ----------
    data : Union[List, numpy.ndarray, DataFrame]
        An array, any object exposing the array interface or a pandas
        DataFrame with data values.

    val_col : str = None
        Name of a DataFrame column that contains dependent variable values
        (test or response variable). Values should have a non-nominal scale.
        Must be specified if ``data`` is a pandas DataFrame object.

    group_col : str = None
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Must be specified if ``data`` is a
        pandas DataFrame object.

    n_perm : int = 100000
        Number of Monte Carlo samples used to approximate the null
        distribution of the maximum F-ratio statistic.

    sort : bool = False
        If True, sort data by group_col.

    Returns
    -------
    tuple[float, float, int]
        P value, F-max statistic, and degrees of freedom (of the
        minimum-variance group).

    Notes
    -----
    Hartley's test requires a (nearly) balanced design; a warning is issued
    otherwise. The p value is approximated by Monte Carlo simulation of `k`
    independent chi-squared(df) variables (df taken from the
    minimum-variance group, following PMCMRplus), rather than PMCMRplus's
    exact `pmaxFratio` distribution (which has no scipy equivalent).

    References
    ----------
    .. [1] H. O. Hartley (1950), The maximum F-ratio as a short cut test for
        heterogeneity of variance, Biometrika, 37, 308-312.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> x = [[1,2,3,5,1], [12,31,54,62,12], [10,12,6,74,11]]
    >>> sp.test_hartley(x)
    """
    x, _val_col, _group_col = __convert_to_df(data, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    var = x_grouped.var().to_numpy()
    ni = x_grouped.count().to_numpy()
    k = var.size

    if np.any(ni != ni[0]):
        warnings.warn(
            "Maximum F-ratio test is imprecise for unbalanced designs.",
            UserWarning,
            stacklevel=2,
        )

    df = int(ni[np.argmin(var)] - 1)
    stat = float(np.max(var) / np.min(var))

    exceedances = 0
    batch_size = 10000
    for start in range(0, n_perm, batch_size):
        batch = min(batch_size, n_perm - start)
        sim = ss.chi2.rvs(df, size=(batch, k))
        exceedances += np.count_nonzero(sim.max(axis=1) / sim.min(axis=1) >= stat)
    pval = float(exceedances / n_perm)

    return pval, stat, df


def test_median(
    data: Union[ArrayLike, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    correction: bool = False,
    sort: bool = False,
) -> tuple[float, float, int]:
    """Brown-Mood median test.

    Tests the null hypothesis that all groups share a common population
    median, against the alternative that at least one differs [1]_.

    Parameters
    ----------
    data : Union[List, numpy.ndarray, DataFrame]
        An array, any object exposing the array interface or a pandas
        DataFrame with data values.

    val_col : str = None
        Name of a DataFrame column that contains dependent variable values
        (test or response variable). Values should have a non-nominal scale.
        Must be specified if ``data`` is a pandas DataFrame object.

    group_col : str = None
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Must be specified if ``data`` is a
        pandas DataFrame object.

    correction : bool = False
        Whether to apply Yates' continuity correction in the underlying
        chi-squared test.

    sort : bool = False
        If True, sort data by group_col.

    Returns
    -------
    tuple[float, float, int]
        P value, chi-squared statistic, and degrees of freedom.

    Notes
    -----
    Observations are classified as above or at-or-below the grand median
    (computed once, from the pooled sample) and compared across groups with
    Pearson's chi-squared test of independence.

    References
    ----------
    .. [1] G. W. Brown, A. M. Mood (1951), On median tests for linear
        hypotheses, Proceedings of the Second Berkeley Symposium on
        Mathematical Statistics and Probability, University of California
        Press, 159-166.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> x = [[1,2,3,5,1], [12,31,54,62,12], [10,12,6,74,11]]
    >>> sp.test_median(x)
    """
    x, _val_col, _group_col = __convert_to_df(data, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    grand_median = x[_val_col].median()
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    n_gt = x[_val_col].gt(grand_median).groupby(x[_group_col], observed=True).sum().to_numpy()
    n_total = x_grouped.count().to_numpy()
    n_le = n_total - n_gt

    table = np.column_stack([n_gt, n_le])
    stat, pval, dof, _ = ss.chi2_contingency(table, correction=correction)

    return float(pval), float(stat), int(dof)
