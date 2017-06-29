import numpy as np

def outliers_iqr(x, ret = 'filtered', coef = 1.5):

    '''Simple detection of potential outliers based on interquartile range (IQR).

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

        return : str, optional
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
        >>> ph.outliers_iqr(x, ret = 'outliers')
        array([12, 23])
    '''

    x = np.asarray(x)

    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    ll = q1 - iqr * coef
    ul = q3 + iqr * coef

    if ret == 'indices':
        return np.where((x >= ll) & (x <= ul))[0]
    elif ret == 'outliers':
        return x[(x < ll) | (x > ul)]
    elif ret == 'outliers_indices':
        return np.where((x < ll) | (x > ul))[0]
    else:
        return x[(x >= ll) & (x <= ul)]

def outliers_grubbs(x, hypo = False, alpha = 0.05):

    '''Grubbs' Test for Outliers [1]_. This is the two-sided version of the test.

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
                True : return a hypothesis test result
                False : return a filtered array without an outlier (default)

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
    '''

    val = np.max(np.abs(x - np.mean(x)))
    ind = np.argmax(np.abs(x - np.mean(x)))
    G = val / np.std(x, ddof=1)
    result = G > (N - 1)/np.sqrt(N) * np.sqrt((ss.t.ppf(1-alpha/(2*N), N-2) ** 2) / (N - 2 + ss.t.ppf(1-alpha/(2*N), N-2) ** 2 ))

    if hypo:
        return result
    else:
        if result:
            return np.delete(x, ind)
        else:
            return x


def outliers_tietjen(x, k, hypo = False, alpha = 0.05):

    '''Tietjen-Moore test [1]_ to detect multiple outliers in a univariate
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
                True : return a hypothesis test result
                False : return a filtered array without outliers (default)

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
        >>> ph.outliers_tietjen(x, 2)
        array([-0.44, -0.3 , -0.24, -0.22, -0.13, -0.05,  0.06,  0.1 ,  0.18,
        0.2 ,  0.39,  0.48,  0.63])
    '''

    n = x.size
    def tietjen(x, k):
        x_mean = x.mean()
        r = np.abs(x - x_mean)
        z = x[r.argsort()]
        E = np.sum((z[:-k] - z[:-k].mean()) ** 2) / np.sum((z - x_mean) ** 2)
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
