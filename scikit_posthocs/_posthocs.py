import inspect
import itertools as it
import warnings
from typing import Optional, Union, Literal
import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests
from pandas import DataFrame, Series, MultiIndex


def __convert_to_df(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = "vals",
    group_col: Optional[str] = "groups",
    val_id: Optional[int] = None,
    group_id: Optional[int] = None,
) -> tuple[DataFrame, str, str]:
    """Hidden helper method to create a DataFrame with input data for further
    processing.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
        An array, any object exposing array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary, i.e. groups
        may have different lengths.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values
        (test or response variable). Values should have a non-nominal scale.
        Must be specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    val_id : int, optional
        Index of a column that contains dependent variable values (test or
        response variable). Should be specified if a NumPy ndarray is used as an
        input. It will be inferred from data, if not specified.

    group_id : int, optional
        Index of a column that contains independent variable values (grouping or
        predictor variable). Should be specified if a NumPy ndarray is used as
        an input. It will be inferred from data, if not specified.

    Returns
    -------
    tuple[DataFrame, str, str]
        Returns a tuple of DataFrame and two strings:
        - DataFrame with input data, `val_col` column contains numerical values
          and `group_col` column contains categorical values.
        - Name of a DataFrame column that contains dependent variable values
          (test or response variable).
        - Name of a DataFrame column that contains independent variable values
          (grouping or predictor variable).

    Notes
    -----
    Inferrence algorithm for determining `val_id` and `group_id` args is rather
    simple, so it is better to specify them explicitly to prevent errors.
    """
    if not group_col:
        group_col = "groups"
    if not val_col:
        val_col = "vals"

    if isinstance(a, DataFrame):
        if not {group_col, val_col}.issubset(a.columns):
            raise ValueError("Specify correct column names using `group_col` and `val_col` args")
        x = a.loc[:, [val_col, group_col]].copy()
        return x, val_col, group_col

    elif isinstance(a, list) or (
        isinstance(a, np.ndarray)
        and (not a.shape.count(2) or (a.shape[0] == 2 and a.shape[1] != 2))
    ):
        grps_len = map(len, a)
        grps = list(it.chain(*[[i + 1] * grp_len for i, grp_len in enumerate(grps_len)]))
        vals = list(it.chain(*a))

        return DataFrame({val_col: vals, group_col: grps}), val_col, group_col

    elif isinstance(a, np.ndarray):
        # cols ids not defined
        # trying to infer
        if val_id is None or group_id is None:
            if np.argmax(a.shape):
                a = a.T

            ax = [np.unique(a[:, 0]).size, np.unique(a[:, 1]).size]

            if np.diff(ax).item():
                __val_col = np.argmax(ax)
                __group_col = np.argmin(ax)
            else:
                raise ValueError(
                    "Cannot infer input format.\nPlease specify `val_id` and `group_id` args"
                )

            cols = {__val_col: val_col, __group_col: group_col}
        else:
            cols = {val_id: val_col, group_id: group_col}

        cols_vals = [v for _, v in sorted(cols.items())]
        return DataFrame(a, columns=cols_vals).dropna(), val_col, group_col


def __convert_to_block_df(
    a: Union[DataFrame, ArrayLike],
    y_col: Optional[Union[str, int]] = None,
    group_col: Optional[Union[str, int]] = None,
    block_col: Optional[Union[str, int]] = None,
    block_id_col: Optional[Union[str, int]] = None,
    melted: bool = False,
) -> tuple[DataFrame, str, str, str, str]:
    if melted and np.any(np.array([block_col, group_col, y_col]) == None):
        raise ValueError(
            "`block_col`, `group_col`, `y_col` should be explicitly specified if using melted data"
        )

    new_block_id_col = "block_ids"
    new_group_col = "groups"
    new_block_col = "blocks"
    new_y_col = "y"

    if isinstance(a, DataFrame) and not melted:
        x = a.copy(deep=True)
        x.columns.name = new_group_col
        x.index.name = new_block_col
        x[new_block_id_col] = np.arange(x.shape[0])
        x = x.reset_index().melt(
            id_vars=[new_block_col, new_block_id_col], var_name=new_group_col, value_name=new_y_col
        )

    elif isinstance(a, DataFrame) and melted:
        if a[block_col].duplicated().sum() > 0 and not block_id_col:
            raise ValueError(
                "`block_col` contains duplicated entries, `block_id_col` should be explicitly specified"
            )

        x = DataFrame.from_dict(
            {
                new_group_col: a[group_col],
                new_block_col: a[block_col],
                new_y_col: a[y_col],
                new_block_id_col: a[block_id_col],
            }
        )

    else:
        x = np.array(a)
        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))

        if not melted:
            x.columns.name = new_group_col
            x.index.name = new_block_col
            x[new_block_id_col] = np.arange(x.shape[0])
            x = x.reset_index().melt(
                id_vars=[new_block_col, new_block_id_col],
                var_name=new_group_col,
                value_name=new_y_col,
            )

        else:
            x.rename(
                columns={
                    group_col: new_group_col,
                    block_col: new_block_col,
                    y_col: new_y_col,
                    block_id_col: new_block_id_col,
                },
                inplace=True,
            )

    return x, new_y_col, new_group_col, new_block_col, new_block_id_col


def __complete_block_matrix(
    a: Union[DataFrame, ArrayLike], melted: bool, sort: bool
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if melted:
        return None

    if isinstance(a, DataFrame):
        if sort or not a.columns.is_unique or a.columns.hasnans:
            return None
        values = a.to_numpy()
        groups = a.columns.to_numpy(copy=True)
    else:
        try:
            values = np.asarray(a)
        except ValueError:
            return None
        if values.ndim != 2:
            return None
        groups = np.arange(values.shape[1])

    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        return None
    if np.isnan(values).any():
        return None
    return values, groups


def posthoc_conover(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    """Post hoc pairwise test for multiple comparisons of mean rank sums
    (Conover´s test). May be used after Kruskal-Wallis one-way analysis of
    variance by ranks to do pairwise comparisons [1]_.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    p_adjust : str, optional
        Method for adjusting p values. See `statsmodels.sandbox.stats.multicomp`
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
        Specifies whether to sort DataFrame by `group_col` or not. Recommended
        unless you sort your data manually.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    A tie correction are employed according to Conover [1]_.

    References
    ----------
    .. [1] W. J. Conover and R. L. Iman (1979), On multiple-comparisons
        procedures, Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.

    Examples
    --------
    >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
    >>> sp.posthoc_conover(x, p_adjust = 'holm')
    """

    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    n = len(x.index)
    x_groups_unique = x[_group_col].unique()
    x_len = x_groups_unique.size
    x_lens = x.groupby(_group_col, observed=True)[_val_col].count()

    x["ranks"] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col, observed=True)["ranks"].mean()
    x_ranks_sum = x.groupby(_group_col, observed=True)["ranks"].sum()

    # ties
    vals = x["ranks"].value_counts(sort=False).to_numpy()
    tie_sum = np.sum(vals**3 - vals)
    tie_sum = 0 if not tie_sum else tie_sum
    x_ties = np.min([1.0, 1.0 - tie_sum / (n**3.0 - n)])

    group_lens = x_lens.reindex(x_groups_unique).to_numpy()
    rank_means = x_ranks_avg.reindex(x_groups_unique).to_numpy()
    rank_sums = x_ranks_sum.reindex(x_groups_unique).to_numpy()
    h = (12.0 / (n * (n + 1.0))) * np.sum(rank_sums**2 / group_lens) - 3.0 * (n + 1.0)
    h_cor = h / x_ties

    if x_ties == 1:
        S2 = n * (n + 1.0) / 12.0
    else:
        S2 = (1.0 / (n - 1.0)) * (np.sum(x["ranks"] ** 2.0) - (n * (((n + 1.0) ** 2.0) / 4.0)))

    vs = np.zeros((x_len, x_len))
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    diff = np.abs(rank_means[tri_upper[0]] - rank_means[tri_upper[1]])
    B = 1.0 / group_lens[tri_upper[0]] + 1.0 / group_lens[tri_upper[1]]
    D = (n - 1.0 - h_cor) / (n - x_len)
    t_values = diff / np.sqrt(S2 * B * D)
    vs[tri_upper] = 2.0 * ss.t.sf(np.abs(t_values), df=n - x_len)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]
    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)

    return DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)


def posthoc_dunn(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    """Post hoc pairwise test for multiple comparisons of mean rank sums
    (Dunn's test). May be used after Kruskal-Wallis one-way analysis of
    variance by ranks to do pairwise comparisons [1]_, [2]_.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    p_adjust : str, optional
        Method for adjusting p values. See `statsmodels.sandbox.stats.multicomp`
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
    result : pandas.DataFrame
        P values.

    Notes
    -----
    A tie correction will be employed according to Glantz (2012).

    References
    ----------
    .. [1] O.J. Dunn (1964). Multiple comparisons using rank sums.
        Technometrics, 6, 241-252.
    .. [2] S.A. Glantz (2012), Primer of Biostatistics. New York: McGraw Hill.

    Examples
    --------

    >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
    >>> sp.posthoc_dunn(x, p_adjust = 'holm')
    """

    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    n = len(x.index)
    x_groups_unique = x[_group_col].unique()
    x_len = x_groups_unique.size
    x_lens = x.groupby(_group_col, observed=True)[_val_col].count()

    x["ranks"] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col, observed=True)["ranks"].mean()

    # ties
    vals = x["ranks"].value_counts(sort=False).to_numpy()
    tie_sum = np.sum(vals**3 - vals)
    tie_sum = 0 if not tie_sum else tie_sum
    x_ties = tie_sum / (12.0 * (n - 1))

    vs = np.zeros((x_len, x_len))
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    group_lens = x_lens.reindex(x_groups_unique).to_numpy()
    rank_means = x_ranks_avg.reindex(x_groups_unique).to_numpy()
    diff = np.abs(rank_means[tri_upper[0]] - rank_means[tri_upper[1]])
    B = 1.0 / group_lens[tri_upper[0]] + 1.0 / group_lens[tri_upper[1]]
    z_values = diff / np.sqrt((n * (n + 1.0) / 12.0 - x_ties) * B)
    vs[tri_upper] = 2.0 * ss.norm.sf(np.abs(z_values))

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)


def posthoc_nemenyi(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    dist: str = "chi",
    sort: bool = True,
) -> DataFrame:
    """Post hoc pairwise test for multiple comparisons of mean rank sums
    (Nemenyi's test). May be used after Kruskal-Wallis one-way analysis of
    variance by ranks to do pairwise comparisons [1]_.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas
        DataFrame. Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    dist : str, optional
        Method for determining the p value. The default distribution is "chi"
        (chi-squared), else "tukey" (studentized range).

    sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    A tie correction will be employed according to Glantz (2012).

    References
    ----------
    .. [1] Lothar Sachs (1997), Angewandte Statistik. Berlin: Springer.
        Pages: 395-397, 662-664.

    Examples
    --------
    >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
    >>> sp.posthoc_nemenyi(x)
    """

    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    n = len(x.index)
    x_groups_unique = x[_group_col].unique()
    x_len = x_groups_unique.size
    x_lens = x.groupby(_group_col, observed=True)[_val_col].count()

    x["ranks"] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col, observed=True)["ranks"].mean()

    # ties
    vals = x["ranks"].value_counts(sort=False).to_numpy()
    tie_sum = np.sum(vals**3 - vals)
    tie_sum = 0 if not tie_sum else tie_sum
    x_ties = np.min([1.0, 1.0 - tie_sum / (n**3.0 - n)])

    vs = np.zeros((x_len, x_len))
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    group_lens = x_lens.reindex(x_groups_unique).to_numpy()
    rank_means = x_ranks_avg.reindex(x_groups_unique).to_numpy()
    diff = np.abs(rank_means[tri_upper[0]] - rank_means[tri_upper[1]])
    B = 1.0 / group_lens[tri_upper[0]] + 1.0 / group_lens[tri_upper[1]]
    A = n * (n + 1.0) / 12.0

    if dist == "chi":
        vs[tri_upper] = ss.chi2.sf(diff**2.0 / (A * B * x_ties), x_len - 1)

    elif dist == "tukey":
        q_values = diff / np.sqrt(A * B)
        vs[tri_upper] = ss.studentized_range.sf(q_values * np.sqrt(2.0), x_len, np.inf)

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)

    return DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)


def posthoc_nemenyi_friedman(
    a: Union[list, np.ndarray, DataFrame],
    y_col: Optional[str] = None,
    group_col: Optional[str] = None,
    block_col: Optional[str] = None,
    block_id_col: Optional[str] = None,
    melted: bool = False,
    sort: bool = False,
) -> DataFrame:
    """Calculate pairwise comparisons using Nemenyi post hoc test for
    unreplicated blocked data. This test is usually conducted post hoc if
    significant results of the Friedman's test are obtained. The statistics
    refer to upper quantiles of the studentized range distribution (Tukey) [1]_,
    [2]_, [3]_.

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
        Name of the column that contains blocking factor values.

    group_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains treatment (group) factor values.

    melted : bool, optional
        Specifies if data are given as melted columns "y", "blocks", and
        "groups".

    sort : bool, optional
        If True, sort data by block and group columns.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    A one-way ANOVA with repeated measures that is also referred to as ANOVA
    with unreplicated block design can also be conducted via Friedman's
    test. The consequent post hoc pairwise multiple comparison test
    according to Nemenyi is conducted with this function.

    This function does not test for ties.

    References
    ----------
    .. [1] J. Demsar (2006), Statistical comparisons of classifiers over
        multiple data sets, Journal of Machine Learning Research, 7, 1-30.

    .. [2] P. Nemenyi (1963) Distribution-free Multiple Comparisons. Ph.D.
        thesis, Princeton University.

    .. [3] L. Sachs (1997), Angewandte Statistik. Berlin: Springer.
        Pages: 668-675.

    Examples
    --------
    >>> # Non-melted case, x is a block design matrix, i.e. rows are blocks
    >>> # and columns are groups.
    >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
    >>> sp.posthoc_nemenyi_friedman(x)
    """

    def compare_stats(i, j):
        dif = np.abs(R[groups[i]] - R[groups[j]])
        qval = dif / np.sqrt(k * (k + 1.0) / (6.0 * n))
        return qval

    matrix = __complete_block_matrix(a, melted, sort)
    if matrix is not None:
        values, groups = matrix
        n, k = values.shape
        ranks = ss.rankdata(values, axis=1, method="average")
        mean_ranks = ranks.mean(axis=0)
        vs = np.zeros((k, k))
        tri_upper = np.triu_indices(k, 1)
        tri_lower = np.tril_indices(k, -1)
        differences = np.abs(mean_ranks[tri_upper[0]] - mean_ranks[tri_upper[1]])
        q_values = differences / np.sqrt(k * (k + 1.0) / (6.0 * n))
        vs[tri_upper] = ss.studentized_range.sf(q_values * np.sqrt(2.0), k, np.inf)
        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)
        return DataFrame(vs, index=groups, columns=groups)

    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        a, y_col, group_col, block_col, block_id_col, melted
    )
    x = x.sort_values(by=[_group_col, _block_col], ascending=True) if sort else x
    x.dropna(inplace=True)

    groups = x[_group_col].unique()
    k = groups.size
    n = x[_block_id_col].unique().size

    x["mat"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    R = x.groupby(_group_col, observed=True)["mat"].mean()
    vs = np.zeros((k, k))
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    for i, j in combs:
        vs[i, j] = compare_stats(i, j)

    vs *= np.sqrt(2.0)
    vs[tri_upper] = ss.studentized_range.sf(vs[tri_upper], k, np.inf)
    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_conover_friedman(
    a: Union[list, np.ndarray, DataFrame],
    y_col: Optional[str] = None,
    block_col: Optional[str] = None,
    block_id_col: Optional[str] = None,
    group_col: Optional[str] = None,
    melted: bool = False,
    sort: bool = False,
    p_adjust: Optional[str] = None,
) -> DataFrame:
    """Calculate pairwise comparisons using Conover post hoc test for unreplicated
    blocked data. This test is usually conducted post hoc after
    significant results of the Friedman test. The statistics refer to
    the Student t distribution [1]_, [2]_.

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
        Must be specified if `a` is a melted pandas DataFrame object.
        Name of the column that contains blocking factor values.

    block_id_col : str or int
        Must be specified if `a` is a melted pandas DataFrame object.
        Name of the column that contains identifiers of blocking factor values.

    group_col : str or int
        Must be specified if `a` is a melted pandas DataFrame object.
        Name of the column that contains treatment (group) factor values.

    melted : bool, optional
        Specifies if data are given as melted columns "y", "blocks", and
        "groups".

    sort : bool, optional
        If True, sort data by block and group columns.

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
        'single-step' : uses Tukey distribution for multiple comparisons

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    A one-way ANOVA with repeated measures that is also referred to as ANOVA
    with unreplicated block design can also be conducted via the
    friedman.test. The consequent post hoc pairwise multiple comparison test
    according to Conover is conducted with this function.

    If y is a matrix, than the columns refer to the treatment and the rows
    indicate the block.

    References
    ----------
    .. [1] W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures,
        Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.

    .. [2] W. J. Conover (1999), Practical nonparametric Statistics, 3rd. Edition,
        Wiley.

    Examples
    --------
    >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
    >>> sp.posthoc_conover_friedman(x)
    """

    def compare_stats(i, j):
        dif = np.abs(R.loc[groups[i]] - R.loc[groups[j]])
        tval = dif / np.sqrt(A) / np.sqrt(B)
        pval = 2.0 * ss.t.sf(np.abs(tval), df=(m * n * k - k - n + 1)).item()
        return pval

    def compare_tukey(i, j):
        dif = np.abs(R.loc[groups[i]] - R.loc[groups[j]])
        qval = np.sqrt(2.0) * dif / np.sqrt(A) / np.sqrt(B)
        pval = ss.studentized_range.sf(qval, k, np.inf).item()
        return pval

    matrix = __complete_block_matrix(a, melted, sort)
    if matrix is not None:
        values, groups = matrix
        n, k = values.shape
        ranks = ss.rankdata(values, axis=1, method="average")
        rank_sums = ranks.sum(axis=0)
        A1 = np.sum(ranks**2.0)
        S2 = 1.0 / (k - 1.0) * (A1 - k * n * ((k + 1.0) ** 2.0) / 4.0)
        T2 = np.sum((rank_sums - n * (k + 1.0) / 2.0) ** 2.0) / S2
        df = n * k - k - n + 1.0
        A = S2 * (2.0 * n * (k - 1.0)) / df
        B = 1.0 - T2 / (n * (k - 1.0))

        vs = np.zeros((k, k))
        tri_upper = np.triu_indices(k, 1)
        tri_lower = np.tril_indices(k, -1)
        differences = np.abs(rank_sums[tri_upper[0]] - rank_sums[tri_upper[1]])
        if p_adjust == "single-step":
            q_values = np.sqrt(2.0) * differences / np.sqrt(A) / np.sqrt(B)
            vs[tri_upper] = ss.studentized_range.sf(q_values, k, np.inf)
        else:
            t_values = differences / np.sqrt(A) / np.sqrt(B)
            vs[tri_upper] = 2.0 * ss.t.sf(np.abs(t_values), df=df)
            if p_adjust is not None:
                vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]
        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)
        return DataFrame(vs, index=groups, columns=groups)

    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        a, y_col, group_col, block_col, block_id_col, melted
    )
    x = x.sort_values(by=[_group_col, _block_col], ascending=True) if sort else x
    x.dropna(inplace=True)

    groups = x[_group_col].unique()
    k = groups.size
    n = x[_block_id_col].unique().size

    x["mat"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    R = x.groupby(_group_col, observed=True)["mat"].sum()
    A1 = (x["mat"] ** 2).sum()
    m = 1
    S2 = m / (m * k - 1.0) * (A1 - m * k * n * ((m * k + 1.0) ** 2.0) / 4.0)
    T2 = 1.0 / S2 * (np.sum((R.to_numpy() - n * m * (m * k + 1.0) / 2.0) ** 2.0))
    A = S2 * (2.0 * n * (m * k - 1.0)) / (m * n * k - k - n + 1.0)
    B = 1.0 - T2 / (n * (m * k - 1.0))

    vs = np.zeros((k, k))
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    if p_adjust == "single-step":
        for i, j in combs:
            vs[i, j] = compare_tukey(i, j)
    else:
        for i, j in combs:
            vs[i, j] = compare_stats(i, j)

        if p_adjust is not None:
            vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_npm_test(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    alternative: Literal["greater", "less"] = "greater",
    nperm: int = 1000,
    sort: bool = False,
) -> DataFrame:
    """Calculate pairwise comparisons using Nashimoto and Wright´s all-pairs
    comparison procedure (NPM test) for simply ordered mean ranksums.

    NPM test is basically an extension of Nemenyi´s procedure for testing
    increasingly ordered alternatives [1]_.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas
        DataFrame.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values
        (test or response variable). Values should have a non-nominal scale.
        Must be specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    sort : bool, optional
        If True, sort data by block and group columns.

    alternative : str, optional
        Alternative hypothesis being tested.
        Can be either "greater" (by default) or "less".

    nperm : int, optional
        Number of permutations to perform for calculating p values.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    An asymetric permutation test is conducted for calculating p values.

    References
    ----------
    .. [1] Nashimoto, K., Wright, F.T., (2005), Multiple comparison procedures for
        detecting differences in simply ordered means. Comput. Statist. Data
        Anal. 48, 291--306.

    Examples
    --------
    >>> x = np.array([[102,109,114,120,124],
                      [110,112,123,130,145],
                      [132,141,156,160,172]])
    >>> sp.posthoc_npm_test(x)
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x
    if alternative == "less":
        x[_val_col] *= -1

    groups = x[_group_col].unique()
    k = groups.size
    n = x.shape[0]
    sigma = np.sqrt(n * (n + 1) / 12.0)
    values = x[_val_col].to_numpy()
    group_values = x[_group_col].to_numpy()
    group_codes = np.empty(n, dtype=int)
    for code, group in enumerate(groups):
        group_codes[group_values == group] = code

    def compare(permuted):
        ranks = Series(permuted).rank().to_numpy()
        valid = ~np.isnan(ranks)
        ni = np.bincount(group_codes[valid], minlength=k)
        rank_sums = np.bincount(group_codes[valid], weights=ranks[valid], minlength=k)
        ri = rank_sums / ni
        stat = np.ones((k, k))

        for j in range(1, k):
            previous = np.arange(j)
            denominator = sigma * np.sqrt(1.0 / ni[previous] + 1.0 / ni[j])
            comparisons = (ri[j] - ri[previous]) / denominator
            stat[j, :j] = np.maximum.accumulate(comparisons[::-1])[::-1]
        return stat

    stat = compare(values)

    exceedances = np.zeros((k, k), dtype=int)
    for _ in range(nperm):
        exceedances += compare(np.random.permutation(values)) >= stat

    p_values = exceedances / nperm

    tri_upper = np.triu_indices(p_values.shape[0], 1)
    p_values[tri_upper] = np.transpose(p_values)[tri_upper]
    np.fill_diagonal(p_values, 1)

    return DataFrame(p_values, index=groups, columns=groups)


def posthoc_siegel_friedman(
    a: Union[list, np.ndarray, DataFrame],
    y_col: Optional[str] = None,
    group_col: Optional[str] = None,
    block_col: Optional[str] = None,
    block_id_col: Optional[str] = None,
    p_adjust: Optional[str] = None,
    melted: bool = False,
    sort: bool = False,
) -> DataFrame:
    """Siegel and Castellan´s All-Pairs Comparisons Test for Unreplicated Blocked
    Data. See authors' paper for additional information [1]_.

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
        Must be specified if `a` is a melted pandas DataFrame object.
        Name of the column that contains y data.

    block_col : str or int
        Must be specified if `a` is a melted pandas DataFrame object.
        Name of the column that contains blocking factor values.

    group_col : str or int
        Must be specified if `a` is a melted pandas DataFrame object.
        Name of the column that contains treatment (group) factor values.

    block_id_col : str or int
        Must be specified if `a` is a melted pandas DataFrame object.
        Name of the column that contains identifiers of blocking factor values.

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
    result : pandas.DataFrame
        P values.

    Notes
    -----
    For all-pairs comparisons in a two factorial unreplicated complete block design
    with non-normally distributed residuals, Siegel and Castellan's test can be
    performed on Friedman-type ranked data.

    References
    ----------
    .. [1] S. Siegel, N. J. Castellan Jr. (1988), Nonparametric Statistics for the
        Behavioral Sciences. 2nd ed. New York: McGraw-Hill.

    Examples
    --------
    >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
    >>> sp.posthoc_siegel_friedman(x)
    """

    def compare_stats(i, j):
        dif = np.abs(R[groups[i]] - R[groups[j]])
        zval = dif / np.sqrt(k * (k + 1.0) / (6.0 * n))
        return zval

    matrix = __complete_block_matrix(a, melted, sort)
    if matrix is not None:
        values, groups = matrix
        n, k = values.shape
        mean_ranks = ss.rankdata(values, axis=1, method="average").mean(axis=0)
        vs = np.zeros((k, k))
        tri_upper = np.triu_indices(k, 1)
        tri_lower = np.tril_indices(k, -1)
        differences = np.abs(mean_ranks[tri_upper[0]] - mean_ranks[tri_upper[1]])
        z_values = differences / np.sqrt(k * (k + 1.0) / (6.0 * n))
        vs[tri_upper] = np.minimum(2.0 * ss.norm.sf(np.abs(z_values)), 1.0)
        if p_adjust:
            vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]
        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)
        return DataFrame(vs, index=groups, columns=groups)

    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        a, y_col, group_col, block_col, block_id_col, melted
    )
    x = x.sort_values(by=[_group_col, _block_col], ascending=True) if sort else x
    x.dropna(inplace=True)

    groups = x[_group_col].unique()
    k = groups.size
    n = x[_block_id_col].unique().size

    x["mat"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    R = x.groupby(_group_col, observed=True)["mat"].mean()

    vs = np.zeros((k, k), dtype=float)
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    for i, j in combs:
        vs[i, j] = compare_stats(i, j)
    vs = 2.0 * ss.norm.sf(np.abs(vs))
    vs[vs > 1] = 1.0

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_miller_friedman(
    a: Union[list, np.ndarray, DataFrame],
    y_col: Optional[str] = None,
    group_col: Optional[str] = None,
    block_col: Optional[str] = None,
    block_id_col: Optional[str] = None,
    melted: bool = False,
    sort: bool = False,
) -> DataFrame:
    """Miller´s All-Pairs Comparisons Test for Unreplicated Blocked Data.
    The p-values are computed from the chi-square distribution [1]_, [2]_,
    [3]_.

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

    group_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains treatment (group) factor values.

    block_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains blocking factor values.

    block_id_col : str or int
        Must be specified if `a` is a melted pandas DataFrame object.
        Name of the column that contains identifiers of blocking factor values.

    melted : bool, optional
        Specifies if data are given as melted columns "y", "blocks", and
        "groups".

    sort : bool, optional
        If True, sort data by block and group columns.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    For all-pairs comparisons in a two factorial unreplicated complete block
    design with non-normally distributed residuals, Miller's test can be
    performed on Friedman-type ranked data.

    References
    ----------
    .. [1] J. Bortz J, G. A. Lienert, K. Boehnke (1990), Verteilungsfreie
        Methoden in der Biostatistik. Berlin: Springerself.

    .. [2] R. G. Miller Jr. (1996), Simultaneous statistical inference. New
        York: McGraw-Hill.

    .. [3] E. L. Wike (2006), Data Analysis. A Statistical Primer for Psychology
        Students. New Brunswick: Aldine Transaction.

    Examples
    --------
    >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
    >>> sp.posthoc_miller_friedman(x)
    """

    def compare_stats(i, j):
        dif = np.abs(R[groups[i]] - R[groups[j]])
        qval = dif / np.sqrt(k * (k + 1.0) / (6.0 * n))
        return qval

    matrix = __complete_block_matrix(a, melted, sort)
    if matrix is not None:
        values, groups = matrix
        n, k = values.shape
        mean_ranks = ss.rankdata(values, axis=1, method="average").mean(axis=0)
        vs = np.zeros((k, k))
        tri_upper = np.triu_indices(k, 1)
        tri_lower = np.tril_indices(k, -1)
        differences = np.abs(mean_ranks[tri_upper[0]] - mean_ranks[tri_upper[1]])
        q_values = differences / np.sqrt(k * (k + 1.0) / (6.0 * n))
        vs[tri_upper] = ss.chi2.sf(q_values**2.0, k - 1)
        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)
        return DataFrame(vs, index=groups, columns=groups)

    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        a, y_col, group_col, block_col, block_id_col, melted
    )
    x = x.sort_values(by=[_group_col, _block_col], ascending=True) if sort else x
    x.dropna(inplace=True)

    groups = x[_group_col].unique()
    k = groups.size
    n = x[_block_id_col].unique().size

    x["mat"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    R = x.groupby(_group_col, observed=True)["mat"].mean()

    vs = np.zeros((k, k), dtype=float)
    combs = it.combinations(range(k), 2)

    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    for i, j in combs:
        vs[i, j] = compare_stats(i, j)
    vs = vs**2
    vs = ss.chi2.sf(vs, k - 1)

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_durbin(
    a: Union[list, np.ndarray, DataFrame],
    y_col: Optional[str] = None,
    group_col: Optional[str] = None,
    block_col: Optional[str] = None,
    block_id_col: Optional[str] = None,
    p_adjust: Optional[str] = None,
    melted: bool = False,
    sort: bool = False,
) -> DataFrame:
    """Pairwise post hoc test for multiple comparisons of rank sums according to
    Durbin and Conover for a two-way balanced incomplete block design (BIBD). See
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

    group_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains treatment (group) factor values.

    block_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains blocking factor values.

    block_id_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains identifiers of blocking factor values.

    melted : bool, optional
        Specifies if data are given as melted columns "y", "blocks", and
        "groups".

    sort : bool, optional
        If True, sort data by block and group columns.

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

    Returns
    -------
    result : pandas.DataFrame
        P values.

    References
    ----------
    .. [1] W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures,
        Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.
    .. [2] W. J. Conover (1999), Practical nonparametric Statistics,
        3rd. edition, Wiley.

    Examples
    --------
    >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
    >>> sp.posthoc_durbin(x)
    """
    matrix = __complete_block_matrix(a, melted, sort)
    if matrix is not None:
        values, groups = matrix
        b, t = values.shape
        ranks = ss.rankdata(values, axis=1, method="average")
        rank_sums = ranks.sum(axis=0)
        A = np.sum(ranks**2.0)
        C = b * t * (t + 1.0) ** 2.0 / 4.0
        D = np.sum(rank_sums**2.0) - b * C
        T1 = (t - 1.0) / (A - C) * D
        df = b * t - b - t + 1.0
        denom = np.sqrt((A - C) * 2.0 * b / df * (1.0 - T1 / (b * (t - 1.0))))
        vs = np.zeros((t, t))
        tri_upper = np.triu_indices(t, 1)
        tri_lower = np.tril_indices(t, -1)
        differences = np.abs(rank_sums[tri_upper[0]] - rank_sums[tri_upper[1]])
        vs[tri_upper] = 2.0 * ss.t.sf(np.abs(differences / denom), df=df)
        if p_adjust:
            vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]
        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)
        return DataFrame(vs, index=groups, columns=groups)

    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        a, y_col, group_col, block_col, block_id_col, melted
    )

    def compare_stats(i, j):
        dif = np.abs(rj[groups[i]] - rj[groups[j]])
        tval = dif / denom
        pval = 2.0 * ss.t.sf(np.abs(tval), df=df)
        return pval

    x = x.sort_values(by=[_block_col, _group_col], ascending=True) if sort else x
    x.dropna(inplace=True)

    groups = x[_group_col].unique()
    t = len(groups)
    b = x[_block_id_col].unique().size
    r = b
    k = t
    x["y_ranked"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    rj = x.groupby(_group_col, observed=True)["y_ranked"].sum()
    A = (x["y_ranked"] ** 2).sum()
    C = (b * k * (k + 1) ** 2) / 4.0
    D = (rj.to_numpy() ** 2).sum() - r * C
    T1 = (t - 1) / (A - C) * D
    denom = np.sqrt(((A - C) * 2 * r) / (b * k - b - t + 1) * (1 - T1 / (b * (k - 1))))
    df = b * k - b - t + 1

    vs = np.zeros((t, t), dtype=float)
    combs = it.combinations(range(t), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    for i, j in combs:
        vs[i, j] = compare_stats(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_anderson(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    midrank: bool = True,
    p_adjust: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Anderson-Darling Pairwise Test for k-samples. Tests the null hypothesis
    that k-samples are drawn from the same population without having to specify
    the distribution function of that population [1]_.

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
    result : pandas.DataFrame
        P values.

    References
    ----------
    .. [1] F.W. Scholz, M.A. Stephens (1987), K-Sample Anderson-Darling Tests,
        Journal of the American Statistical Association, Vol. 82, pp. 918-924.

    Examples
    --------
    >>> x = np.array([[2.9, 3.0, 2.5, 2.6, 3.2], [3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]])
    >>> sp.posthoc_anderson(x)
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    k = groups.size
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    samples = [x_grouped.get_group(group).dropna().to_numpy() for group in groups]
    vs = np.zeros((k, k), dtype=float)
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    _anderson_kwargs = (
        {"variant": "midrank" if midrank else "right"}
        if "variant" in inspect.signature(ss.anderson_ksamp).parameters
        else {"midrank": midrank}
    )

    for i, j in combs:
        result = ss.anderson_ksamp(
            [samples[i], samples[j]],
            **_anderson_kwargs,
            method=ss.PermutationMethod(),
        )
        vs[i, j] = result.pvalue if hasattr(result, "pvalue") else result[2]

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_quade(
    a: Union[list, np.ndarray, DataFrame],
    y_col: Optional[str] = None,
    group_col: Optional[str] = None,
    block_col: Optional[str] = None,
    block_id_col: Optional[str] = None,
    dist: str = "t",
    p_adjust: Optional[str] = None,
    melted: bool = False,
    sort: bool = False,
) -> DataFrame:
    """Calculate pairwise comparisons using Quade's post hoc test for
    unreplicated blocked data. This test is usually conducted if significant
    results were obtained by the omnibus test [1]_, [2]_, [3]_.

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

    block_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains blocking factor values.

    group_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains treatment (group) factor values.

    dist : str, optional
        Method for determining p values.
        The default distribution is "t", else "normal".

    melted : bool, optional
        Specifies if data are given as melted columns "y", "blocks", and
        "groups".

    sort : bool, optional
        If True, sort data by block and group columns.

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
    result : pandas.DataFrame
        P values.

    References
    ----------
    .. [1] W. J. Conover (1999), Practical nonparametric Statistics, 3rd. Edition,
        Wiley.

    .. [2] N. A. Heckert and J. J. Filliben (2003). NIST Handbook 148: Dataplot
        Reference Manual, Volume 2: Let Subcommands and Library Functions.
        National Institute of Standards and Technology Handbook Series, June 2003.

    .. [3] D. Quade (1979), Using weighted rankings in the analysis of complete
        blocks with additive block effects. Journal of the American Statistical
        Association, 74, 680-683.

    Examples
    --------
    >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
    >>> sp.posthoc_quade(x)
    """

    def compare_stats_t(i, j):
        dif = np.abs(S[groups[i]] - S[groups[j]])
        tval = dif / denom
        pval = 2.0 * ss.t.sf(np.abs(tval), df=(b - 1) * (k - 1))
        return pval

    def compare_stats_norm(i, j):
        dif = np.abs(W[groups[i]] * ff - W[groups[j]] * ff)
        zval = dif / denom
        pval = 2.0 * ss.norm.sf(np.abs(zval))
        return pval

    matrix = __complete_block_matrix(a, melted, sort)
    if matrix is not None:
        values, groups = matrix
        b, k = values.shape
        ranks = ss.rankdata(values, axis=1, method="average")
        weights = ss.rankdata(np.ptp(values, axis=1), method="average")
        centered = ranks - (k + 1.0) / 2.0
        scores = centered * weights[:, None]
        weighted_ranks = ranks * weights[:, None]
        A = np.sum(scores**2.0)
        S = scores.sum(axis=0)
        B = np.sum(S**2.0) / b
        W = weighted_ranks.sum(axis=0)
        vs = np.zeros((k, k))
        tri_upper = np.triu_indices(k, 1)
        tri_lower = np.tril_indices(k, -1)
        if dist == "t":
            denom = np.sqrt(2.0 * b * (A - B) / ((b - 1.0) * (k - 1.0)))
            statistics = np.abs(S[tri_upper[0]] - S[tri_upper[1]]) / denom
            vs[tri_upper] = 2.0 * ss.t.sf(np.abs(statistics), df=(b - 1) * (k - 1))
        else:
            n = b * k
            denom = np.sqrt(k * (k + 1.0) * (2.0 * n + 1.0) * (k - 1.0) / (18.0 * n * (n + 1.0)))
            ff = 1.0 / (b * (b + 1.0) / 2.0)
            statistics = np.abs(W[tri_upper[0]] - W[tri_upper[1]]) * ff / denom
            vs[tri_upper] = 2.0 * ss.norm.sf(np.abs(statistics))
        if p_adjust:
            vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]
        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)
        return DataFrame(vs, index=groups, columns=groups)

    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        a, y_col, group_col, block_col, block_id_col, melted
    )

    x = x.sort_values(by=[_block_col, _group_col], ascending=True) if sort else x
    x.dropna(inplace=True)

    groups = x[_group_col].unique()
    k = len(groups)
    b = x[_block_id_col].unique().size

    x["r"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    block_values = x.groupby(_block_id_col, observed=True)[_y_col]
    q = (block_values.max() - block_values.min()).rank()
    x["rr"] = x["r"] - (k + 1) / 2
    block_weights = x[_block_id_col].map(q)
    x["s"] = x["rr"] * block_weights
    x["w"] = x["r"] * block_weights

    A = (x["s"] ** 2).sum()
    S = x.groupby(_group_col, observed=True)["s"].sum()
    B = np.sum(S.to_numpy() ** 2) / b
    W = x.groupby(_group_col, observed=True)["w"].sum()

    vs = np.zeros((k, k), dtype=float)
    combs = it.combinations(range(k), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    if dist == "t":
        denom = np.sqrt((2 * b * (A - B)) / ((b - 1) * (k - 1)))

        for i, j in combs:
            vs[i, j] = compare_stats_t(i, j)

    else:
        n = b * k
        denom = np.sqrt((k * (k + 1.0) * (2.0 * n + 1.0) * (k - 1.0)) / (18.0 * n * (n + 1.0)))
        ff = 1.0 / (b * (b + 1.0) / 2.0)

        for i, j in combs:
            vs[i, j] = compare_stats_norm(i, j)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_vanwaerden(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
    p_adjust: Optional[str] = None,
) -> DataFrame:
    """Van der Waerden's test for pairwise multiple comparisons between group
    levels. See references for additional information [1]_, [2]_.

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
    result : pandas.DataFrame
        P values.

    Notes
    -----
    For one-factorial designs with samples that do not meet the assumptions for
    one-way-ANOVA and subsequent post hoc tests, the van der Waerden test using
    normal scores can be employed. Provided that significant differences were
    detected by this global test, one may be interested in applying post hoc
    tests according to van der Waerden for pairwise multiple comparisons of the
    group levels.

    There is no tie correction applied in this function.

    References
    ----------
    .. [1] W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures,
        Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.

    .. [2] B. L. van der Waerden (1952) Order tests for the two-sample problem and
        their power, Indagationes Mathematicae, 14, 453-458.

    Examples
    --------
    >>> x = np.array([[10,'a'], [59,'a'], [76,'b'], [10, 'b']])
    >>> sp.posthoc_vanwaerden(x, val_col = 0, group_col = 1)
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    n = x[_val_col].size
    k = groups.size
    r = ss.rankdata(x[_val_col])
    x["z_scores"] = ss.norm.ppf(r / (n + 1))

    grouped_scores = x.groupby(_group_col, observed=True)["z_scores"]
    aj = grouped_scores.sum().reindex(groups).to_numpy()
    nj = grouped_scores.count().reindex(groups).to_numpy()
    s2 = (1.0 / (n - 1.0)) * (x["z_scores"] ** 2.0).sum()
    sts = (1.0 / s2) * np.sum(aj**2.0 / nj)
    A = aj / nj

    vs = np.zeros((k, k), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    differences = np.abs(A[tri_upper[0]] - A[tri_upper[1]])
    sample_factor = 1.0 / nj[tri_upper[0]] + 1.0 / nj[tri_upper[1]]
    t_values = differences / np.sqrt(s2 * (n - 1.0 - sts) / (n - k) * sample_factor)
    vs[tri_upper] = 2.0 * ss.t.sf(np.abs(t_values), df=n - k)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_dunnett(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    control: Optional[str] = None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    sort: bool = False,
    to_matrix: bool = True,
) -> Union[Series, DataFrame]:
    """
    Dunnett's test [1, 2, 3] for multiple comparisons against a control group, used after parametric
    ANOVA. The control group is specified by the `control` parameter.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas
        DataFrame. Array must be two-dimensional.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    control : str, optional
        Name of the control group within the `group_col` column. Values should
        have a nominal scale (categorical). Must be specified if `a` is a pandas
        DataFrame.

    alternative : ['two-sided', 'less', or 'greater'], optional
        Whether to get the p-value for the one-sided hypothesis
        ('less' or 'greater') or for the two-sided hypothesis ('two-sided').
        Defaults to 'two-sided'.

    sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.

    to_matrix: bool, optional
        Specifies whether to return a DataFrame or a Series. If True, a DataFrame
        is returned with some NaN values since it's not pairwise comparison.
        Default is True.

    Returns
    -------
    result : pandas.Series or pandas.DataFrame
        P values.

    References
    ----------
    .. [1] Charles W. Dunnett (1955). “A Multiple Comparison Procedure for Comparing Several Treatments with a Control.”
    .. [2] https://en.wikipedia.org/wiki/Dunnett%27s_test
    .. [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dunnett.html#id1
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x
    x = x.set_index(_group_col)[_val_col]
    x_embedded = x.groupby(_group_col, observed=True).agg(lambda y: y.dropna().tolist())
    control_data = x_embedded.loc[control]
    treatment_data = x_embedded.drop(control)

    pvals = ss.dunnett(*treatment_data, control=control_data, alternative=alternative).pvalue

    multi_index = MultiIndex.from_product([[control], treatment_data.index.tolist()])
    dunnett_sr = Series(pvals, index=multi_index)

    if not to_matrix:
        return dunnett_sr

    else:
        levels = x.index.unique().to_numpy()
        result_df = DataFrame(index=levels, columns=levels, dtype=float)

        for pair in dunnett_sr.index:
            a, b = pair
            result_df.loc[a, b] = dunnett_sr[pair]
            result_df.loc[b, a] = dunnett_sr[pair]
        for level in levels:
            result_df.loc[level, level] = 1.0
        return result_df


def posthoc_ttest(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    pool_sd: bool = False,
    equal_var: bool = True,
    p_adjust: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Pairwise T test for multiple comparisons of independent groups. May be
    used after a parametric ANOVA to do pairwise comparisons.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas
        DataFrame. Array must be two-dimensional.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

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
    result : pandas.DataFrame
        P values.

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
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    if not np.isfinite(x[_val_col].to_numpy(dtype=float)).all():
        raise ValueError(
            "All observations must be finite; remove or impute missing values before calling "
            "posthoc_ttest"
        )
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    k = groups.size
    xg = x.groupby(by=_group_col, observed=True)[_val_col]
    samples = [xg.get_group(group).to_numpy() for group in groups]

    vs = np.zeros((k, k), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0
    combs = it.combinations(range(k), 2)

    if pool_sd:
        ni = xg.count()
        m = xg.mean()
        sd = xg.std(ddof=1)
        deg_f = ni - 1.0
        total_deg_f = np.sum(deg_f)
        pooled_sd = np.sqrt(np.sum(sd**2.0 * deg_f) / total_deg_f)

        def compare_pooled(i, j):
            diff = m.loc[groups[i]] - m.loc[groups[j]]
            se_diff = pooled_sd * np.sqrt(1.0 / ni.loc[groups[i]] + 1.0 / ni.loc[groups[j]])
            t_value = diff / se_diff
            return 2.0 * ss.t.cdf(-np.abs(t_value), total_deg_f)

        for i, j in combs:
            vs[i, j] = compare_pooled(i, j)
    else:
        for i, j in combs:
            vs[i, j] = ss.ttest_ind(samples[i], samples[j], equal_var=equal_var)[1]

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_tukey_hsd(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    """Pairwise comparisons with TukeyHSD confidence intervals. This is a
    convenience function to make statsmodels `pairwise_tukeyhsd` method more
    applicable for further use.

    Parameters
    ----------
    x : array_like or pandas Series object, 1d
        An array, any object exposing the array interface, containing dependent
        variable values (test or response variable). Values should have a
        non-nominal scale. NaN values will cause an error (please handle
        manually).

    g : array_like or pandas Series object, 1d
        An array, any object exposing the array interface, containing
        independent variable values (grouping or predictor variable). Values
        should have a nominal scale (categorical).

    alpha : float, optional
        Significance level for the test. Default is 0.05.

    Returns
    -------
    result : pandas.DataFrame
        DataFrame with 0, 1, and -1 values, where 0 is False (not significant),
        1 is True (significant), and -1 is for diagonal elements.

    Examples
    --------
    >>> x = [[1,2,3,4,5], [35,31,75,40,21], [10,6,9,6,1]]
    >>> g = [['a'] * 5, ['b'] * 5, ['c'] * 5]
    >>> sp.posthoc_tukey_hsd(np.concatenate(x), np.concatenate(g))
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x
    groups = x[_group_col].unique()

    results = ss.tukey_hsd(
        *[
            x.loc[idx, _val_col].to_numpy()
            for idx in x.groupby(_group_col, observed=True, sort=False).groups.values()
        ]
    )

    return DataFrame(results.pvalue, index=groups, columns=groups)


def posthoc_mannwhitney(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    use_continuity: bool = True,
    alternative: str = "two-sided",
    p_adjust: Optional[str] = None,
    sort: bool = True,
) -> DataFrame:
    """Pairwise comparisons with Mann-Whitney rank test.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas
        DataFrame. Array must be two-dimensional.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

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
    result : pandas.DataFrame
        P values.

    Notes
    -----
    Refer to `scipy.stats.mannwhitneyu` reference page for further details.

    Examples
    --------
    >>> x = [[1,2,3,4,5], [35,31,75,40,21], [10,6,9,6,1]]
    >>> sp.posthoc_mannwhitney(x, p_adjust = 'holm')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    x_len = groups.size
    vs = np.zeros((x_len, x_len))
    xg = x.groupby(_group_col, observed=True)[_val_col]
    samples = [xg.get_group(group).to_numpy() for group in groups]
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    combs = it.combinations(range(x_len), 2)

    for i, j in combs:
        vs[i, j] = ss.mannwhitneyu(
            samples[i],
            samples[j],
            use_continuity=use_continuity,
            alternative=alternative,
        )[1]

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_wilcoxon(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    method: str = "auto",
    zero_method: str = "wilcox",
    correction: bool = False,
    p_adjust: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Pairwise comparisons with Wilcoxon signed-rank test.

    It is a non-parametric version of the paired T-test for use with
    non-parametric ANOVA.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas
        DataFrame. Array must be two-dimensional.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    method : {"auto", "exact", "approx"}, optional
        Method to calculate the p-value. Default is "auto".

    zero_method : {"pratt", "wilcox", "zsplit"}, optional
        "pratt": Pratt treatment, includes zero-differences in the ranking
        process (more conservative)
        "wilcox": Wilcox treatment, discards all zero-differences
        "zsplit": Zero rank split, just like Pratt, but spliting the zero rank
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
        'simes-hochberg' : step-up method (independent)
        'hommel' : closed method based on Simes tests (non-negative)
        'fdr_bh' : Benjamini/Hochberg (non-negative)
        'fdr_by' : Benjamini/Yekutieli (negative)
        'fdr_tsbh' : two stage fdr correction (non-negative)
        'fdr_tsbky' : two stage fdr correction (non-negative)

    sort : bool, optional
        Specifies whether to sort DataFrame by group_col and val_col or not.
        Default is False.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    Refer to `scipy.stats.wilcoxon` reference page for further details [1]_.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

    Examples
    --------
    >>> x = [[1,2,3,4,5], [35,31,75,40,21], [10,6,9,6,1]]
    >>> sp.posthoc_wilcoxon(x)
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    x_len = groups.size
    vs = np.zeros((x_len, x_len))
    xg = x.groupby(_group_col, observed=True)[_val_col]
    samples = [xg.get_group(group).to_numpy() for group in groups]
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    combs = it.combinations(range(x_len), 2)

    for i, j in combs:
        vs[i, j] = ss.wilcoxon(
            samples[i],
            samples[j],
            zero_method=zero_method,
            method=method,
            correction=correction,
        )[1]

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]
    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_scheffe(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Scheffe's all-pairs comparisons test for normally distributed data with equal
    group variances. For all-pairs comparisons in an one-factorial layout with
    normally distributed residuals and equal variances Scheffe's test can be
    performed with parametric ANOVA [1]_, [2]_, [3]_.

    A total of m = k(k-1)/2 hypotheses can be tested.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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
    result : pandas.DataFrame
        P values.

    Notes
    -----
    The p values are computed from the F-distribution.

    References
    ----------
    .. [1] J. Bortz (1993) Statistik für Sozialwissenschaftler. 4. Aufl., Berlin:
        Springer.

    .. [2] L. Sachs (1997) Angewandte Statistik, New York: Springer.

    .. [3] H. Scheffe (1953) A Method for Judging all Contrasts in the Analysis
        of Variance. Biometrika 40, 87-110.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_scheffe(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    ni = x_grouped.count()
    xi = x_grouped.mean()
    si = x_grouped.var()
    n = ni.sum()
    sin = 1.0 / (n - groups.size) * np.sum(si * (ni - 1.0))

    vs = np.zeros((groups.size, groups.size), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    counts = ni.reindex(groups).to_numpy()
    means = xi.reindex(groups).to_numpy()
    dif = means[tri_upper[0]] - means[tri_upper[1]]
    A = sin * (1.0 / counts[tri_upper[0]] + 1.0 / counts[tri_upper[1]]) * (groups.size - 1.0)
    vs[tri_upper] = dif**2.0 / A

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    p_values = ss.f.sf(vs, groups.size - 1.0, n - groups.size)

    np.fill_diagonal(p_values, 1)
    return DataFrame(p_values, index=groups, columns=groups)


def posthoc_tamhane(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    welch: bool = True,
    sort: bool = False,
) -> DataFrame:
    """Tamhane's T2 all-pairs comparison test for normally distributed data with
    unequal variances. Tamhane's T2 test can be performed for all-pairs
    comparisons in an one-factorial layout with normally distributed residuals
    but unequal groups variances. A total of m = k(k-1)/2 hypotheses can be
    tested. The null hypothesis is tested in the two-tailed test against the
    alternative hypothesis [1]_.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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

    welch : bool, optional
        If True, use Welch's approximate solution for calculating the degree of
        freedom. T2 test uses the usual df = N - 2 approximation.

    sort : bool, optional
        If True, sort data by block and group columns.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    The p values are computed from the t-distribution and adjusted according to
    Dunn-Sidak.

    References
    ----------
    .. [1] A.C. Tamhane (1979), A Comparison of Procedures for Multiple Comparisons of
        Means with Unequal Variances. Journal of the American Statistical Association,
        74, 471-480.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_tamhane(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    ni = x_grouped.count()
    xi = x_grouped.mean()
    si = x_grouped.var()

    vs = np.zeros((groups.size, groups.size), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    counts = ni.reindex(groups).to_numpy()
    means = xi.reindex(groups).to_numpy()
    variances = si.reindex(groups).to_numpy()
    i, j = tri_upper
    scaled_variances = variances / counts
    A = scaled_variances[i] + scaled_variances[j]
    t_values = (means[i] - means[j]) / np.sqrt(A)
    if welch:
        df = A**2.0 / (
            variances[i] ** 2.0 / (counts[i] ** 2.0 * (counts[i] - 1.0))
            + variances[j] ** 2.0 / (counts[j] ** 2.0 * (counts[j] - 1.0))
        )
    else:
        count_ratio = counts[i] / counts[j]
        error_ratio = scaled_variances[i] / scaled_variances[j]
        ok1 = (9.0 / 10.0 <= count_ratio) & (count_ratio <= 10.0 / 9.0)
        ok2 = (9.0 / 10.0 <= error_ratio) & (error_ratio <= 10.0 / 9.0)
        ok3 = (
            (4.0 / 5.0 <= count_ratio)
            & (count_ratio <= 5.0 / 4.0)
            & (1.0 / 2.0 <= error_ratio)
            & (error_ratio <= 2.0)
        )
        ok4 = (
            (2.0 / 3.0 <= count_ratio)
            & (count_ratio <= 3.0 / 2.0)
            & (3.0 / 4.0 <= error_ratio)
            & (error_ratio <= 4.0 / 3.0)
        )
        for _ in np.flatnonzero(~(ok1 | ok2 | ok3 | ok4)):
            warnings.warn(
                "Sample sizes or standard errors are not balanced. T2 test is recommended.",
                UserWarning,
                stacklevel=2,
            )
        df = counts[i] + counts[j] - 2.0
    vs[tri_upper] = 2.0 * ss.t.sf(np.abs(t_values), df=df)

    vs[tri_upper] = 1.0 - (1.0 - vs[tri_upper]) ** groups.size
    vs[tri_lower] = np.transpose(vs)[tri_lower]
    vs[vs > 1] = 1

    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_tukey(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Performs Tukey's all-pairs comparisons test for normally distributed data
    with equal group variances. For all-pairs comparisons in an
    one-factorial layout with normally distributed residuals and equal variances
    Tukey's test can be performed. A total of m = k(k-1)/2 hypotheses can be
    tested. The null hypothesis is tested in the two-tailed test against
    the alternative hypothesis [1]_, [2]_.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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
    result : pandas.DataFrame
        P values.

    Notes
    -----
    The p values are computed from the Tukey-distribution.

    References
    ----------
    .. [1] L. Sachs (1997) Angewandte Statistik, New York: Springer.
    .. [2] J. Tukey (1949) Comparing Individual Means in the Analysis of Variance,
        Biometrics 5, 99-114.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_tukey(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x
    groups = x[_group_col].unique()
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]

    ni = x_grouped.count()
    n = ni.sum()
    xi = x_grouped.mean()
    si = x_grouped.var()
    sin = 1.0 / (n - groups.size) * np.sum(si * (ni - 1))

    vs = np.zeros((groups.size, groups.size), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    counts = ni.reindex(groups).to_numpy()
    means = xi.reindex(groups).to_numpy()
    scale = sin * 0.5 * (1.0 / counts[tri_upper[0]] + 1.0 / counts[tri_upper[1]])
    q_values = (means[tri_upper[0]] - means[tri_upper[1]]) / np.sqrt(scale)
    vs[tri_upper] = ss.studentized_range.sf(np.abs(q_values), groups.size, n - groups.size)
    vs[tri_lower] = np.transpose(vs)[tri_lower]

    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_dscf(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Dwass, Steel, Critchlow and Fligner all-pairs comparison test for a
    one-factorial layout with non-normally distributed residuals. As opposed to
    the all-pairs comparison procedures that depend on Kruskal ranks, the DSCF
    test is basically an extension of the U-test as re-ranking is conducted for
    each pairwise test [1]_, [2]_, [3]_.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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
    result : pandas.DataFrame
        P values.

    Notes
    -----
    The p values are computed from the Tukey-distribution.

    References
    ----------
    .. [1] Douglas, C. E., Fligner, A. M. (1991) On distribution-free multiple
        comparisons in the one-way analysis of variance, Communications in
        Statistics - Theory and Methods, 20, 127-139.

    .. [2] Dwass, M. (1960) Some k-sample rank-order tests. In Contributions to
        Probability and Statistics, Edited by: I. Olkin, Stanford: Stanford
        University Press.

    .. [3] Steel, R. G. D. (1960) A rank sum test for comparing all pairs of
        treatments, Technometrics, 2, 197-207.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_dscf(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x
    groups = x[_group_col].unique()
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    samples = [x_grouped.get_group(group).dropna().to_numpy() for group in groups]
    k = groups.size

    def compare(i, j):
        left = samples[i]
        right = samples[j]
        ni = left.size
        nj = right.size
        ranks = ss.rankdata(np.concatenate((left, right)), method="average")
        ri = ranks[:ni].sum()
        rj = ranks[ni:].sum()
        u = np.array(
            [
                nj * ni + nj * (nj + 1) / 2.0 - rj,
                nj * ni + ni * (ni + 1) / 2.0 - ri,
            ]
        )
        u_min = np.min(u)
        s = ni + nj
        tie_counts = np.unique(ranks, return_counts=True)[1]
        ties = np.sum((tie_counts**3 - tie_counts) / 12.0)
        var = (nj * ni / (s * (s - 1.0))) * ((s**3 - s) / 12.0 - ties)
        p = np.sqrt(2.0) * (u_min - nj * ni / 2.0) / np.sqrt(var)
        return p

    vs = np.zeros((k, k))
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    combs = it.combinations(range(k), 2)

    for i, j in combs:
        vs[i, j] = compare(i, j)

    vs[tri_upper] = ss.studentized_range.sf(np.abs(vs[tri_upper]), k, np.inf)
    vs[tri_lower] = np.transpose(vs)[tri_lower]

    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_games_howell(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Games-Howell all-pairs comparison test for normally distributed data
    with unequal group variances. A total of m = k(k-1)/2 hypotheses can be
    tested. The null hypothesis is tested in the two-tailed test against the
    alternative hypothesis [1]_.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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
        If True, sort data by group columns.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    Games-Howell is the most common all-pairs alternative to Tukey's test
    when group variances cannot be assumed equal. The p values are computed
    from the studentized range (Tukey) distribution using Welch's
    approximate degrees of freedom, same as `posthoc_tamhane` with
    `welch=True` but using the studentized range rather than the
    t-distribution.

    References
    ----------
    .. [1] L. Sachs (1997), Angewandte Statistik. Berlin: Springer. Pages: 396.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_games_howell(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    k = groups.size
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    ni = x_grouped.count()
    xi = x_grouped.mean()
    si = x_grouped.var()

    vs = np.zeros((k, k), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    counts = ni.reindex(groups).to_numpy()
    means = xi.reindex(groups).to_numpy()
    variances = si.reindex(groups).to_numpy()
    i, j = tri_upper
    A = variances[i] / counts[i] + variances[j] / counts[j]
    q_values = (means[i] - means[j]) / np.sqrt(A) * np.sqrt(2.0)
    df = A**2.0 / (
        variances[i] ** 2.0 / (counts[i] ** 2.0 * (counts[i] - 1.0))
        + variances[j] ** 2.0 / (counts[j] ** 2.0 * (counts[j] - 1.0))
    )
    vs[tri_upper] = ss.studentized_range.sf(np.abs(q_values), k, df)

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_dunnett_t3(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Dunnett's T3 all-pairs comparison test for normally distributed data
    with unequal group variances. A total of m = k(k-1)/2 hypotheses can be
    tested. The null hypothesis is tested in the two-tailed test against the
    alternative hypothesis [1]_.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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
        If True, sort data by group columns.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    Test statistics are Welch t values, with degrees of freedom rounded to
    the nearest integer (matching PMCMRplus; this is what numerically
    distinguishes T3 from `posthoc_tamhane`, whose Welch df are not
    rounded). P values are obtained with a single-step Dunn-Sidak-type
    adjustment, `1 - (1 - p) ** m` with `m = k * (k - 1) / 2` the number of
    pairwise comparisons, applied to the raw two-sided Welch t p value of
    each pair. This reproduces PMCMRplus's studentized maximum modulus
    computation (an equicorrelated multivariate t distribution with zero
    off-diagonal correlation reduces exactly to this closed form), without
    requiring a multivariate-t dependency.

    References
    ----------
    .. [1] C. W. Dunnett (1980), Pair wise multiple comparisons in the
        unequal variance case, Journal of the American Statistical
        Association, 75, 796-800.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_dunnett_t3(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    k = groups.size
    m = k * (k - 1) / 2.0
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    ni = x_grouped.count()
    xi = x_grouped.mean()
    si = x_grouped.var()

    vs = np.zeros((k, k), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    counts = ni.reindex(groups).to_numpy()
    means = xi.reindex(groups).to_numpy()
    variances = si.reindex(groups).to_numpy()
    i, j = tri_upper
    A = variances[i] / counts[i] + variances[j] / counts[j]
    t_values = (means[i] - means[j]) / np.sqrt(A)
    df = A**2.0 / (
        variances[i] ** 2.0 / (counts[i] ** 2.0 * (counts[i] - 1.0))
        + variances[j] ** 2.0 / (counts[j] ** 2.0 * (counts[j] - 1.0))
    )
    p_raw = 2.0 * ss.t.sf(np.abs(t_values), df=np.round(df))
    vs[tri_upper] = 1.0 - (1.0 - p_raw) ** m

    vs[vs > 1] = 1.0
    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_lsd(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    p_adjust: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Fisher's Least Significant Difference (LSD) all-pairs comparison test
    for normally distributed data with equal group variances, following a
    parametric ANOVA [1]_.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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

    p_adjust : str, optional
        Method for adjusting p values. See `statsmodels.sandbox.stats.multicomp`
        for details. Left as `None` (the default), this reproduces Fisher's
        *protected* LSD test, which is only valid when applied after a
        significant omnibus ANOVA F-test.

    sort : bool, optional
        If True, sort data by group columns.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    Test statistics use the pooled within-group variance, equivalent to
    `posthoc_ttest` with `pool_sd=True` and no p value adjustment.

    References
    ----------
    .. [1] L. Sachs (1997), Angewandte Statistik. Berlin: Springer.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_lsd(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    k = groups.size
    n = len(x.index)
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    ni = x_grouped.count()
    xi = x_grouped.mean()
    si = x_grouped.var()
    s2in = (1.0 / (n - k)) * np.sum(si * (ni - 1.0))

    vs = np.zeros((k, k), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    counts = ni.reindex(groups).to_numpy()
    means = xi.reindex(groups).to_numpy()
    i, j = tri_upper
    A = s2in * (1.0 / counts[i] + 1.0 / counts[j])
    t_values = (means[i] - means[j]) / np.sqrt(A)
    vs[tri_upper] = 2.0 * ss.t.sf(np.abs(t_values), df=n - k)

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_snk(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Student-Newman-Keuls (SNK) all-pairs comparison test for normally
    distributed data with equal group variances, following a parametric
    ANOVA [1]_, [2]_, [3]_.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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
        If True, sort data by group columns.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    Unlike single-step procedures (Tukey, Games-Howell), SNK is a stepwise
    range test: the p value for a pair of group means is computed from the
    studentized range distribution using as `nmeans` the number of ordered
    means the pair spans (i.e. 1 + the difference between their ranks when
    means are sorted in decreasing order), not the total number of groups.
    There is no separate p value adjustment argument, since the step-down
    procedure is itself the adjustment.

    References
    ----------
    .. [1] Newman, D. (1939), The distribution of range in samples from a
        normal population, expressed in terms of an independent estimate of
        standard deviation, Biometrika, 31, 20-30.
    .. [2] Keuls, M. (1952), The use of the "studentized range" in connection
        with an analysis of variance, Euphytica, 1, 112-122.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_snk(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    k = groups.size
    n = len(x.index)
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    ni = x_grouped.count()
    xi = x_grouped.mean()
    si = x_grouped.var()
    s2in = (1.0 / (n - k)) * np.sum(si * (ni - 1.0))
    df = n - k

    # rank position of each group when means are sorted in decreasing order
    counts = ni.reindex(groups).to_numpy()
    means = xi.reindex(groups).to_numpy()
    order = means.argsort()[::-1]
    rank_pos = np.empty(k, dtype=int)
    rank_pos[order] = np.arange(k)

    vs = np.zeros((k, k), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    i, j = tri_upper
    q_values = (means[i] - means[j]) / np.sqrt((s2in / 2.0) * (1.0 / counts[i] + 1.0 / counts[j]))
    nmeans = 1 + np.abs(rank_pos[i] - rank_pos[j])
    vs[tri_upper] = ss.studentized_range.sf(np.abs(q_values), nmeans, df)

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_duncan(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Duncan's multiple range test for normally distributed data with equal
    group variances, following a parametric ANOVA [1]_.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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
        If True, sort data by group columns.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    Like `posthoc_snk`, this is a stepwise range test using as `nmeans` the
    number of ordered means a pair spans. The studentized-range p value for
    each pair is further Bonferroni-adjusted as
    `1 - (1 - p) ** (1 / (nmeans - 1))`, following Duncan (1955). There is no
    separate p value adjustment argument, since the step-down procedure is
    itself the adjustment.

    References
    ----------
    .. [1] Duncan, D. B. (1955), Multiple range and multiple F tests,
        Biometrics, 11, 1-42.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_duncan(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    k = groups.size
    n = len(x.index)
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    ni = x_grouped.count()
    xi = x_grouped.mean()
    si = x_grouped.var()
    s2in = (1.0 / (n - k)) * np.sum(si * (ni - 1.0))
    df = n - k
    r = k / np.sum(1.0 / ni)

    counts = ni.reindex(groups).to_numpy()
    means = xi.reindex(groups).to_numpy()
    order = means.argsort()[::-1]
    rank_pos = np.empty(k, dtype=int)
    rank_pos[order] = np.arange(k)

    vs = np.zeros((k, k), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    i, j = tri_upper
    t_values = (means[i] - means[j]) / np.sqrt(s2in / r)
    nmeans = 1 + np.abs(rank_pos[i] - rank_pos[j])
    p_values = ss.studentized_range.sf(np.abs(t_values), nmeans, df)
    vs[tri_upper] = 1.0 - (1.0 - p_values) ** (1.0 / (nmeans - 1))

    vs[vs > 1] = 1.0
    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_median(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    p_adjust: Optional[str] = None,
    sort: bool = False,
) -> DataFrame:
    """Brown-Mood all-pairs median test, a nonparametric alternative to
    Kruskal-Wallis-type all-pairs tests robust to outliers and skew [1]_.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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

    p_adjust : str, optional
        Method for adjusting p values. See `statsmodels.sandbox.stats.multicomp`
        for details.

    sort : bool, optional
        If True, sort data by group columns.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    For each pair of groups, observations are classified as above or at-or-below
    the grand median (computed once, from the pooled sample of both groups'
    parent dataset) and compared with Pearson's chi-squared test (no continuity
    correction), following PMCMRplus's `medianAllPairsTest`.

    References
    ----------
    .. [1] G. W. Brown, A. M. Mood (1951), On median tests for linear
        hypotheses, Proceedings of the Second Berkeley Symposium on
        Mathematical Statistics and Probability, University of California
        Press, 159-166.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_median(x, val_col='values', group_col='groups')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    k = groups.size
    grand_median = x[_val_col].median()
    x_grouped = x.groupby(_group_col, observed=True)[_val_col]
    n_gt = x[_val_col].gt(grand_median).groupby(x[_group_col], observed=True).sum()
    n_total = x_grouped.count()
    n_le = n_total - n_gt

    def compare(i, j):
        table = np.array([[n_gt[i], n_le[i]], [n_gt[j], n_le[j]]])
        _, p_val, _, _ = ss.chi2_contingency(table, correction=False)
        return p_val

    vs = np.zeros((k, k), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    combs = it.combinations(range(k), 2)

    for i, j in combs:
        vs[i, j] = compare(groups[i], groups[j])

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=groups, columns=groups)


def posthoc_steel(
    a: Union[list, np.ndarray, DataFrame],
    val_col: Optional[str] = None,
    group_col: Optional[str] = None,
    control: Optional[str] = None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    p_adjust: Optional[str] = None,
    sort: bool = False,
    to_matrix: bool = True,
) -> Union[Series, DataFrame]:
    """Steel's many-to-one rank test [1]_, a nonparametric alternative to
    Dunnett's test for comparisons of several treatment groups against one
    control group.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
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

    control : str, optional
        Name of the control group within the `group_col` column. Must be
        specified if `a` is a pandas DataFrame.

    alternative : ['two-sided', 'less', or 'greater'], optional
        Whether to get the p-value for the one-sided hypothesis
        ('less' or 'greater') or for the two-sided hypothesis ('two-sided').
        Defaults to 'two-sided'.

    p_adjust : str, optional
        Method for adjusting p values across the treatment-vs-control
        comparisons. See `statsmodels.sandbox.stats.multicomp` for details.

    sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not.

    to_matrix : bool, optional
        Specifies whether to return a DataFrame or a Series. If True, a
        DataFrame is returned with some NaN values since it's not a pairwise
        comparison. Default is True.

    Returns
    -------
    result : pandas.Series or pandas.DataFrame
        P values.

    Notes
    -----
    This implements the normal-approximation variant of Steel's test: each
    treatment group is compared to the control with a Mann-Whitney U test
    (`scipy.stats.mannwhitneyu`), and the resulting p values across
    treatments may be combined with `p_adjust`. PMCMRplus's `steelTest`
    instead looks up exact critical rank-sum values from a fixed table
    (balanced designs only, n <= 20, k <= 9, alpha = 0.05); that table is not
    reproduced here.

    References
    ----------
    .. [1] R. G. D. Steel (1959), A multiple comparison rank sum test:
        treatments versus control, Biometrics, 15, 560-572.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import pandas as pd
    >>> x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
    >>> x = x.melt(var_name='groups', value_name='values')
    >>> sp.posthoc_steel(x, val_col='values', group_col='groups', control='a')
    """
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x
    x = x.set_index(_group_col)[_val_col]
    x_embedded = x.groupby(_group_col, observed=True).agg(lambda y: y.dropna().tolist())
    control_data = x_embedded.loc[control]
    treatment_data = x_embedded.drop(control)

    pvals = np.array(
        [ss.mannwhitneyu(t, control_data, alternative=alternative).pvalue for t in treatment_data]
    )

    if p_adjust:
        pvals = multipletests(pvals, method=p_adjust)[1]

    multi_index = MultiIndex.from_product([[control], treatment_data.index.tolist()])
    steel_sr = Series(pvals, index=multi_index)

    if not to_matrix:
        return steel_sr

    else:
        levels = x.index.unique().to_numpy()
        result_df = DataFrame(index=levels, columns=levels, dtype=float)

        for pair in steel_sr.index:
            ctl, trt = pair
            result_df.loc[ctl, trt] = steel_sr[pair]
            result_df.loc[trt, ctl] = steel_sr[pair]
        for level in levels:
            result_df.loc[level, level] = 1.0
        return result_df


def posthoc_demsar(
    a: Union[list, np.ndarray, DataFrame],
    y_col: Optional[str] = None,
    group_col: Optional[str] = None,
    block_col: Optional[str] = None,
    block_id_col: Optional[str] = None,
    control: Optional[str] = None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    p_adjust: Optional[str] = None,
    melted: bool = False,
    sort: bool = False,
    to_matrix: bool = True,
) -> Union[Series, DataFrame]:
    """Demsar's many-to-one test for unreplicated blocked data [1]_, comparing
    several treatments against one control (e.g. a baseline algorithm)
    across a set of blocks (e.g. datasets), based on Friedman-type ranks.

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

    group_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains treatment (group) factor values.

    block_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains blocking factor values.

    block_id_col : str or int
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains identifiers of blocking factor values.

    control : str, optional
        Name of the control group within the `group_col` column. Must be
        specified if `a` is a pandas DataFrame.

    alternative : ['two-sided', 'less', or 'greater'], optional
        Whether to get the p-value for the one-sided hypothesis
        ('less' or 'greater') or for the two-sided hypothesis ('two-sided').
        Defaults to 'two-sided'.

    p_adjust : str, optional
        Method for adjusting p values across the treatment-vs-control
        comparisons. See `statsmodels.sandbox.stats.multicomp` for details.

    melted : bool, optional
        Specifies if data are given as melted columns "y", "blocks", and
        "groups".

    sort : bool, optional
        If True, sort data by block and group columns.

    to_matrix : bool, optional
        Specifies whether to return a DataFrame or a Series. If True, a
        DataFrame is returned with some NaN values since it's not a pairwise
        comparison. Default is True.

    Returns
    -------
    result : pandas.Series or pandas.DataFrame
        P values.

    Notes
    -----
    P values are computed from the standard normal distribution applied to
    the difference in mean Friedman-type ranks between each treatment and
    the control. This is the standard procedure for comparing multiple
    classifiers/algorithms against a baseline across multiple datasets [1]_.

    References
    ----------
    .. [1] J. Demsar (2006), Statistical comparisons of classifiers over
        multiple data sets, Journal of Machine Learning Research, 7, 1-30.

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> import numpy as np
    >>> x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
    >>> sp.posthoc_demsar(x, control=0)
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be one of 'two-sided', 'greater', or 'less'")

    matrix = __complete_block_matrix(a, melted, sort)
    if matrix is not None:
        values, groups = matrix
        control_indices = np.flatnonzero(groups == control)
        if control_indices.size:
            n, k = values.shape
            mean_ranks = ss.rankdata(values, axis=1, method="average").mean(axis=0)
            treatment_indices = np.flatnonzero(groups != control)
            treatments = groups[treatment_indices]
            zvals = (mean_ranks[treatment_indices] - mean_ranks[control_indices[0]]) / np.sqrt(
                k * (k + 1.0) / (6.0 * n)
            )
            if alternative == "two-sided":
                pvals = 2.0 * ss.norm.sf(np.abs(zvals))
            elif alternative == "greater":
                pvals = ss.norm.sf(zvals)
            else:
                pvals = ss.norm.cdf(zvals)
            if p_adjust:
                pvals = multipletests(pvals, method=p_adjust)[1]
            demsar_sr = Series(pvals, index=MultiIndex.from_product([[control], treatments]))
            if not to_matrix:
                return demsar_sr
            result_df = DataFrame(index=groups, columns=groups, dtype=float)
            result_df.loc[control, treatments] = pvals
            result_df.loc[treatments, control] = pvals
            np.fill_diagonal(result_df.values, 1.0)
            return result_df

    x, _y_col, _group_col, _block_col, _block_id_col = __convert_to_block_df(
        a, y_col, group_col, block_col, block_id_col, melted
    )
    x = x.sort_values(by=[_group_col, _block_col], ascending=True) if sort else x
    x.dropna(inplace=True)

    groups = x[_group_col].unique()
    k = groups.size
    n = x[_block_id_col].unique().size

    x["mat"] = x.groupby(_block_id_col, observed=True)[_y_col].rank()
    R = x.groupby(_group_col, observed=True)["mat"].mean()

    denom = np.sqrt(k * (k + 1.0) / (6.0 * n))
    treatments = [g for g in groups if g != control]
    zvals = np.array([(R[t] - R[control]) / denom for t in treatments])

    if alternative == "two-sided":
        pvals = 2.0 * ss.norm.sf(np.abs(zvals))
    elif alternative == "greater":
        pvals = ss.norm.sf(zvals)
    else:
        pvals = ss.norm.cdf(zvals)

    if p_adjust:
        pvals = multipletests(pvals, method=p_adjust)[1]

    multi_index = MultiIndex.from_product([[control], treatments])
    demsar_sr = Series(pvals, index=multi_index)

    if not to_matrix:
        return demsar_sr

    else:
        levels = groups
        result_df = DataFrame(index=levels, columns=levels, dtype=float)

        for pair in demsar_sr.index:
            ctl, trt = pair
            result_df.loc[ctl, trt] = demsar_sr[pair]
            result_df.loc[trt, ctl] = demsar_sr[pair]
        for level in levels:
            result_df.loc[level, level] = 1.0
        return result_df
