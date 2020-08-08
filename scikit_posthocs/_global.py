import numpy as np
import itertools as it
import scipy.stats as ss


def global_simes_test(x):
    '''Global Simes test of the intersection null hypothesis. Computes
    the combined p value as min(np(i)/i), where p(1), ..., p(n)
    are the ordered p values.

    Parameters
    ----------
    x : array_like
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

    arr = np.array(x)
    ranks = ss.rankdata(arr)
    p = np.min(arr.size * arr / ranks)
    return p


def global_f_test(x, stat=False):
    '''Fisher's combination test for global null hypothesis. Computes
    the combined p value using chi-squared distribution and
    T statistic: -2 * sum(log(x)).

    Parameters
    ----------
    x : array_like
        An array of p values.
    stat : bool
        Defines if statistic should be returned.

    Returns
    -------
    p : float
        Global p value.
    T : float
        Statistic (optional).

    References
    ----------
    .. [1] Fisher RA. Statistical methods for research workers,
        London: Oliver and Boyd, 1932.

    Examples
    --------
    >>> x = [0.04, 0.03, 0.98, 0.01, 0.43, 0.99, 1.0, 0.002]
    >>> sp.global_f_test(x)

    '''

    arr = np.array(x)
    T = -2 * np.sum(np.log(arr))
    p = ss.chi2.sf(T, df=2*len(arr))
    return p, T if stat else p
