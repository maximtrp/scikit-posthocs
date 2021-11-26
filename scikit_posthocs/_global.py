from typing import Union, List, Tuple
from numpy import array, ndarray, log
from scipy.stats import rankdata, chi2


def global_simes_test(p_vals: Union[List, ndarray]) -> float:
    '''Global Simes test of the intersection null hypothesis.

    Computes the combined p value as min(np(i)/i), where p(1), ..., p(n) are
    the ordered p values [1]_.

    Parameters
    ----------
    p_vals : Union[List, ndarray]
        An array of p values.

    Returns
    -------
    p_value : float
        Global p value.

    References
    ----------
    .. [1] Simes, R. J. (1986). An improved Bonferroni procedure for multiple
        tests of significance. Biometrika, 73(3):751-754.

    Examples
    --------
    >>> arr = [0.04, 0.03, 0.98, 0.01, 0.43, 0.99, 1.0, 0.002]
    >>> sp.global_simes_test(arr)
    '''
    arr = array(p_vals)
    ranks = rankdata(arr)
    p_value = min(arr.size * arr / ranks)
    return p_value


def global_f_test(
        p_vals: Union[List, ndarray],
        stat: bool = False) -> Union[float, Tuple[float, float]]:
    '''Fisher's combination test for global null hypothesis.

    Computes the combined p value using chi-squared distribution and T
    statistic: -2 * sum(log(x)) [1]_.

    Parameters
    ----------
    p_vals : Union[List, ndarray]
        An array or a list of p values.
    stat : bool
        Defines if statistic should be returned.

    Returns
    -------
    p_value : float
        Global p value.
    t_stat : float
        Statistic.

    References
    ----------
    .. [1] Fisher RA. Statistical methods for research workers,
        London: Oliver and Boyd, 1932.

    Examples
    --------
    >>> x = [0.04, 0.03, 0.98, 0.01, 0.43, 0.99, 1.0, 0.002]
    >>> sp.global_f_test(x)
    '''
    arr = array(p_vals)
    t_stat = -2 * sum(log(arr))
    p_value = chi2.sf(t_stat, df=2 * len(arr))
    return p_value, t_stat if stat else p_value

