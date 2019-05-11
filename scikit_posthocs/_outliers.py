# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import t

def outliers_iqr(x, ret='filtered', coef = 1.5):

    """
    Simple detection of potential outliers based on interquartile range (IQR).

    Data that lie within the lower and upper limits are considered
    non-outliers. The lower limit is the number that lies 1.5 IQRs below
    (coefficient may be changed with an argument, see Parameters)
    the first quartile; the upper limit is the number that lies 1.5 IQRs
    above the third quartile.

    Parameters
    ----------
    x : array_like or ndarray, 1d
        An array, any object exposing the array interface, containing
        p values.

    ret : str, optional
        Specifies object to be returned. Available options are:
        'filtered' : return a filtered array (default)
        'outliers' : return outliers
        'indices' : return indices of non-outliers
        'outliers_indices' : return indices of outliers

    coef : float, optional
        Coefficient by which IQR is multiplied. Default is 1.5.

    Returns
    -------
    Numpy array where 0 is False (not significant), 1 is True (significant),
    and -1 is for diagonal elements.

    Examples
    --------
    >>> x = np.array([4,5,6,10,12,4,3,1,2,3,23,5,3])
    >>> outliers_iqr(x, ret = 'outliers')
    array([12, 23])

    """

    x = np.asarray(x)

    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    ll = q1 - iqr * coef
    ul = q3 + iqr * coef

    if ret == 'indices':
        return np.where((x > ll) & (x < ul))[0]
    elif ret == 'outliers':
        return x[(x < ll) | (x > ul)]
    elif ret == 'outliers_indices':
        return np.where((x < ll) | (x > ul))[0]
    else:
        return x[(x > ll) & (x < ul)]

def outliers_grubbs(x, hypo = False, alpha = 0.05):

    """
    Grubbs' Test for Outliers [1]_. This is the two-sided version of the test.

    The null hypothesis implies that there are no outliers in the data set.

    Parameters
    ----------
    x : array_like or ndarray, 1d
        An array, any object exposing the array interface, containing
        data to test for an outlier in.

    hypo : bool, optional
        Specifies whether to return a bool value of a hypothesis test result.
        Returns True when we can reject the null hypothesis. Otherwise, False.
        Available options are:
        1) True - return a hypothesis test result
        2) False - return a filtered array without an outlier (default)

    alpha : float, optional
        Significance level for a hypothesis test. Default is 0.05.

    Returns
    -------
    Numpy array if hypo is False or a bool value of a hypothesis test result.

    Notes
    -----
    .. [1] http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm

    Examples
    --------

    >>> x = np.array([199.31,199.53,200.19,200.82,201.92,201.95,202.18,245.57])
    >>> ph.outliers_grubbs(x)
    array([ 199.31,  199.53,  200.19,  200.82,  201.92,  201.95,  202.18])
    """

    val = np.max(np.abs(x - np.mean(x)))
    ind = np.argmax(np.abs(x - np.mean(x)))
    G = val / np.std(x, ddof=1)
    N = len(x)
    result = G > (N - 1)/np.sqrt(N) * np.sqrt((t.ppf(1-alpha/(2*N), N-2) ** 2) / (N - 2 + t.ppf(1-alpha/(2*N), N-2) ** 2 ))

    if hypo:
        return result
    else:
        if result:
            return np.delete(x, ind)
        else:
            return x

def outliers_tietjen(x, k, hypo = False, alpha = 0.05):

    """
    Tietjen-Moore test [1]_ to detect multiple outliers in a univariate
    data set that follows an approximately normal distribution.
    The Tietjen-Moore test [2]_ is a generalization of the Grubbs' test to
    the case of multiple outliers. If testing for a single outlier,
    the Tietjen-Moore test is equivalent to the Grubbs' test.

    The null hypothesis implies that there are no outliers in the data set.

    Parameters
    ----------
    x : array_like or ndarray, 1d
        An array, any object exposing the array interface, containing
        data to test for an outlier in.

    k : int
        Number of potential outliers to test for. Function tests for
        outliers in both tails.

    hypo : bool, optional
        Specifies whether to return a bool value of a hypothesis test result.
        Returns True when we can reject the null hypothesis. Otherwise, False.
        Available options are:
        1) True - return a hypothesis test result
        2) False - return a filtered array without outliers (default)

    alpha : float, optional
        Significance level for a hypothesis test. Default is 0.05.

    Returns
    -------
    Numpy array if hypo is False or a bool value of a hypothesis test result.

    Notes
    -----
    .. [1] Tietjen and Moore (August 1972), Some Grubbs-Type Statistics
        for the Detection of Outliers, Technometrics, 14(3), pp. 583-597.
    .. [2] http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h2.htm

    Examples
    --------
    >>> x = np.array([-1.40, -0.44, -0.30, -0.24, -0.22, -0.13, -0.05, 0.06,
    0.10, 0.18, 0.20, 0.39, 0.48, 0.63, 1.01])
    >>> outliers_tietjen(x, 2)
    array([-0.44, -0.3 , -0.24, -0.22, -0.13, -0.05,  0.06,  0.1 ,  0.18,
    0.2 ,  0.39,  0.48,  0.63])

    """

    n = x.size
    def tietjen(x_, k_):
        x_mean = x_.mean()
        r = np.abs(x_ - x_mean)
        z = x_[r.argsort()]
        E = np.sum((z[:-k_] - z[:-k_].mean()) ** 2) / np.sum((z - x_mean) ** 2)
        return E

    E_x = tietjen(x, k)
    E_norm = np.zeros(10000)

    for i in np.arange(10000):
        norm = np.random.normal(size=n)
        E_norm[i] = tietjen(norm, k)

    CV = np.percentile(E_norm, alpha * 100)
    result = E_x < CV

    if hypo:
        return result
    else:
        if result:
            ind = np.argpartition(np.abs(x - x.mean()), -k)[-k:]
            return np.delete(x, ind)
        else:
            return x

def outliers_gesd(data, outliers = 5, report = False, alpha=0.05):

    """
    The generalized (Extreme Studentized Deviate) ESD test is used
    to detect one or more outliers in a univariate data set that follows
    an approximately normal distribution [1]_.

    Parameters
    ----------
    data : array_like or ndarray, 1d
        An array, any object exposing the array interface, containing
        data to test for outliers.

    outliers : int, optional
        Number of potential outliers to test for. Test is two-tailed, i.e.
        maximum and minimum values are checked for potential outliers.

    report : bool, optional
        Specifies whether to return a summary table of the test.
        Available options are:
        1) True - return a summary table
        2) False - return the array with outliers removed. (default)

    alpha : float, optional
        Significance level for a hypothesis test. Default is 0.05.

    Returns
    -------
    Numpy array if hypo is False or a bool value of a hypothesis test result.

    Notes
    -----
    .. [1] Rosner, Bernard (May 1983), Percentage Points for a Generalized
        ESD Many-Outlier Procedure,Technometrics, 25(2), pp. 165-172.

    Examples
    --------
    >>> data = np.array([-0.25, 0.68, 0.94, 1.15, 1.2, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69, 1.7, 1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.1, 2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37, 2.4, 2.47, 2.54, 2.62, 2.64, 2.9, 2.92, 2.92, 2.93, 3.21, 3.26, 3.3, 3.59, 3.68, 4.3, 4.64, 5.34, 5.42, 6.01])

    >>> outliers_gesd(data, 5)
    array([-0.25,  0.68,  0.94,  1.15,  1.2 ,  1.26,  1.26,  1.34,  1.38,
            1.43,  1.49,  1.49,  1.55,  1.56,  1.58,  1.65,  1.69,  1.7 ,
            1.76,  1.77,  1.81,  1.91,  1.94,  1.96,  1.99,  2.06,  2.09,
            2.1 ,  2.14,  2.15,  2.23,  2.24,  2.26,  2.35,  2.37,  2.4 ,
            2.47,  2.54,  2.62,  2.64,  2.9 ,  2.92,  2.92,  2.93,  3.21,
            3.26,  3.3 ,  3.59,  3.68,  4.3 ,  4.64])

    >>> outliers_gesd(data, outliers = 5, report = True)
    H0: no outliers in the data
    Ha: up to 5 outliers in the data
    Significance level:  α = 0.05
    Reject H0 if Ri > Critical Value (λi)

    Summary Table for Two-Tailed Test
    ---------------------------------------
          Exact           Test     Critical
      Number of      Statistic    Value, λi
    Outliers, i      Value, Ri          5 %
    ---------------------------------------
              1          3.119        3.159
              2          2.943        3.151
              3          3.179        3.144 *
              4           2.81        3.136
              5          2.816        3.128

    """

    Rs, ls = np.zeros(outliers, dtype = np.float), np.zeros(outliers, dtype = np.float)
    ms = []

    data = np.sort(np.array(data))
    data_proc = np.copy(data)
    n = data_proc.size

    for i in np.arange(outliers):

        abs_d = np.abs(data_proc - np.mean(data_proc))

        # R-value calculation
        R = np.max(abs_d) / np.std(data_proc, ddof=1)
        Rs[i] = R

        # Masked values
        lms = ms[-1] if len(ms) > 0 else []
        ms.append(lms + [np.argmax(abs_d)])

        # Lambdas calculation
        p = 1 - alpha / (2 * (n - i))
        df = n - i - 2
        t_ppr = t.ppf(p, df)
        lambd = ((n - i - 1) * t_ppr) / np.sqrt((n - i - 2 + t_ppr**2) * (n - i))
        ls[i] = lambd

        # Remove the observation that maximizes |xi − xmean|
        data_proc = np.delete(data_proc, np.argmax(abs_d))

    if report:

        report = ["H0: no outliers in the data",
                  "Ha: up to " + str(outliers) + " outliers in the data",
                  "Significance level:  α = " + str(alpha),
                  "Reject H0 if Ri > Critical Value (λi)", "",
                  "Summary Table for Two-Tailed Test",
                  "---------------------------------------",
                  "      Exact           Test     Critical",
                  "  Number of      Statistic    Value, λi",
                  "Outliers, i      Value, Ri          5 %",
                  "---------------------------------------"]

        for i, (r, l) in enumerate(zip(Rs, ls)):
            report.append('{: >11s}'.format(str(i+1)) + \
                          '{: >15s}'.format(str(np.round(r, 3))) + \
                          '{: >13s}'.format(str(np.round(l, 3))) + (" *" if r > l else ""))

        print("\n".join(report))

    else:
        # Remove masked values
        # for which the test statistic is greater
        # than the critical value and return the result

        if any(Rs > ls):
            data = np.delete(data, ms[np.max(np.where(Rs > ls))])

        return data
