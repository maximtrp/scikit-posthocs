from numpy import array, ndarray, log
from scipy.stats import rankdata, chi2
from typing import Union, List, Tuple


def global_simes_test(x: Union[List, ndarray]) -> float:
    '''Global Simes test of the intersection null hypothesis. Computes
    the combined p value as min(np(i)/i), where p(1), ..., p(n)
    are the ordered p values [1]_.

    Parameters
    ----------
    x : Union[List, ndarray]
        An array of p values.

    Returns
    -------
    p : float
        Global p value.

    References
    ----------
    .. [1] Simes, R. J. (1986). An improved Bonferroni procedure for multiple
        tests of significance. Biometrika, 73(3):751-754.

    Examples
    --------
    >>> x = [0.04, 0.03, 0.98, 0.01, 0.43, 0.99, 1.0, 0.002]
    >>> sp.global_simes_test(x)
    '''

    arr = array(x)
    ranks = rankdata(arr)
    p = min(arr.size * arr / ranks)
    return p


def global_f_test(x: Union[List, ndarray],
                  stat: bool = False) -> Tuple[float, float]:
    '''Fisher's combination test for global null hypothesis. Computes
    the combined p value using chi-squared distribution and
    T statistic: -2 * sum(log(x)) [1]_.

    Parameters
    ----------
    x : Union[List, ndarray]
        An array or a list of p values.
    stat : bool
        Defines if statistic should be returned.

    Returns
    -------
    p : float
        Global p value.
    T : float
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

    arr = array(x)
    T = -2 * sum(log(arr))
    p = chi2.sf(T, df=2 * len(arr))
    return p, T
