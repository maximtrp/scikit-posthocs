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
