import numpy as np
from matplotlib.colors import ListedColormap
from seaborn import heatmap
from pandas import DataFrame

def sign_array(a, alpha = 0.05):

    '''Significance matrix

        Converts an array with p values to a significance matrix where
        0 is False (not significant), 1 is True (significant),
        and -1 is for diagonal elements.

        Parameters
        ----------
        x : array_like or ndarray
            An array, any object exposing the array interface, containing
            p values.

        alpha : float, optional
            Significance level. Default is 0.05.

        Returns
        -------
        Numpy array where 0 is False (not significant), 1 is True (significant),
        and -1 is for diagonal elements.

        Examples
        --------

        >>> x = np.array([[ 0.        ,  0.00119517,  0.00278329],
                          [ 0.00119517,  0.        ,  0.18672227],
                          [ 0.00278329,  0.18672227,  0.        ]])
        >>> ph.sign_array(x)
        array([[-1,  1,  1],
               [ 1, -1,  0],
               [ 1,  0, -1]])

    '''

    a = np.array(a)
    a_ = np.copy(a)
    a_[a > alpha] = 0
    a_[a <= alpha] = 1
    a_[np.diag_indices(a_.shape[0])] = -1
    return a_


def sign_plot(x, g = None, cmap = None, **kwargs):

    '''Significance plot

        Plots a significance array as a heatmap using seaborn.

        Parameters
        ----------
        x : array_like or ndarray
            An array, any object exposing the array interface, containing
            p values.

        g : array_like or Numpy array, optional
            An array, any object exposing the array interface, containing
            group names.

        cmap : list, optional
            List consisting of three elements, that will be exported to
            ListedColormap method of matplotlib.

        kwargs : other keyword arguments
            All other keyword arguments are passed to seaborn heatmap method.

        Returns
        -------
        Numpy ndarray where 0 is False (not significant), 1 is True (significant),
        and -1 is for diagonal elements.

        Examples
        --------

        >>> x = np.array([[-1,  1,  1],
                          [ 1, -1,  0],
                          [ 1,  0, -1]])
        >>> ph.sign_plot(x)
    '''

    if g is None:
        g = np.arange(x.shape[0])

    if cmap is None:
        cmap = ['#f0f0f0', '#d73027', '#1a9641']

    df = DataFrame(x, index=g, columns=g, dtype=np.int)
    return heatmap(df, vmin=-1, vmax=1, cmap=ListedColormap(cmap), **kwargs)
