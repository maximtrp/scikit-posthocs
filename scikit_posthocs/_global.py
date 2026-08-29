from typing import Union, List, Tuple
from numpy import array, ndarray, log, isfinite, sort
from scipy.stats import chi2


def _validate_pvalues(p_vals: Union[List, ndarray]) -> ndarray:
    arr = array(p_vals, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
        raise ValueError(
            "p_vals must be a non-empty one-dimensional sequence of finite p-values between "
            "0 and 1; remove invalid values before calling this function"
        )
    return arr


def global_simes_test(p_vals: Union[List, ndarray]) -> float:
    """Global Simes test of the intersection null hypothesis.

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
    """
    arr = sort(_validate_pvalues(p_vals))
    return float(min(arr.size * arr / array(range(1, arr.size + 1))))


def global_f_test(
    p_vals: Union[List, ndarray], stat: bool = False
) -> Union[float, Tuple[float, float]]:
    """Fisher's combination test for global null hypothesis.

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
    """
    arr = _validate_pvalues(p_vals)
    t_stat = -2 * sum(log(arr))
    p_value = chi2.sf(t_stat, df=2 * len(arr))
    return (p_value, t_stat) if stat else p_value
