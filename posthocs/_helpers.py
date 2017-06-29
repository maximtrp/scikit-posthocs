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
        x : array_like or ndarray
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
        x : array_like or ndarray
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
