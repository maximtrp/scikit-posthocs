import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
from seaborn import heatmap
from pandas import DataFrame

def sign_array(a, alpha = 0.05):

    '''Significance array

        Converts an array with p values to a significance array where
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
    a[a > alpha] = 0
    a[(a < alpha) & (a > 0)] = 1
    np.fill_diagonal(a, -1)
    return a


def sign_table(a, lower = True, upper = True):

    '''Significance table

        Returns table that can be used in a publication. P values are replaced
        with asterisks: * - p < 0.05, ** - p < 0.01, *** - p < 0.001.

        Parameters
        ----------
        x : array_like, or ndarray, or pandas DataFrame
            An array, any object exposing the array interface, containing
            p values.

        lower : bool, optional
            Defines whether to return the lower triangle.

        upper : bool, optional
            Defines whether to return the upper triangle.

        Returns
        -------
        Numpy array or pandas DataFrame with asterisks masked p values.

        Examples
        --------

        >>> x = np.array([[-1.        ,  0.00119517,  0.00278329],
                          [ 0.00119517, -1.        ,  0.18672227],
                          [ 0.00278329,  0.18672227, -1.        ]])
        >>> ph.sign_table(x)
        array([['-', '**', '**'],
               ['**', '-', 'NS'],
               ['**', 'NS', '-']], dtype=object)

    '''
    if not any([lower, upper]):
        raise ValueError("Either lower or upper triangle must be returned")

    if not isinstance(a, DataFrame):
        a = np.array(a, dtype = np.float)

    ns = a > 0.05
    three = (a < 0.001) & (a >= 0)
    two = (a < 0.01) & (a >= 0.001)
    one = (a < 0.05) & (a >= 0.01)

    a = a.astype(object)
    a[ns] = 'NS'
    a[three] = '***'
    a[two] = '**'
    a[one] = '*'

    if not isinstance(a, DataFrame):
        np.fill_diagonal(a, '-')
        if not lower:
            a[np.tril_indices(a.shape[0], -1)] = ''
        elif not upper:
            a[np.triu_indices(a.shape[0], 1)] = ''
    else:
        np.fill_diagonal(a.values, '-')
        if not lower:
            a.values[np.tril_indices(a.shape[0], -1)] = ''
        elif not upper:
            a.values[np.triu_indices(a.shape[0], 1)] = ''

    return a

def sign_plot(x, g = None, flat = False, cmap = None, cbar_ax_bbox = None,\
    ax = None, **kwargs):

    '''Significance plot, a heatmap of p values.

        Parameters
        ----------
        x : array_like, ndarray or DataFrame
            If flat is False (default), x must be an array, any object exposing
            the array interface, containing p values. If flat is True, x must be
            a sign_array (returned by `scikit_posthocs.sign_array` function)

        g : array_like or Numpy array, optional
            An array, any object exposing the array interface, containing
            group names.

        flat : bool, optional
            If `flat` is True, plots a significance array as a heatmap using
            seaborn. If `flat` is False (default), plots an array of p values
            as a heatmap using seaborn. Non-flat mode is useful if you need to
            differentiate significance levels visually. It is the preferred mode.

        cmap : list, optional
            If flat is False (default):
                List consisting of five elements, that will be exported to
                ListedColormap method of matplotlib. First is for diagonal
                elements, second is for non-significant elements, third is for
                p < 0.001, fourth is for p < 0.01, fifth is for p < 0.05.

            If flat is True:
                List consisting of three elements, that will be exported to
                ListedColormap method of matplotlib. First is for diagonal
                elements, second is for non-significant elements, third is for
                significant ones.

            If not defined, default colormaps will be used.

        cbar_ax_bbox : list, optional
            Colorbar axes position rect [left, bottom, width, height] where
            all quantities are in fractions of figure width and height.
            Refer to `matplotlib.figure.Figure.add_axes` for more information.
            Default is [0.95, 0.35, 0.04, 0.3].

        kwargs : other keyword arguments
            Keyword arguments to be passed to seaborn heatmap method. These
            keyword args cannot be used: cbar, vmin, vmax, center.

        Returns
        -------
        Numpy ndarray where 0 is False (not significant), 1 is True (significant),
        and -1 is for diagonal elements.

        Examples
        --------

        >>> x = np.array([[-1,  1,  1],
                          [ 1, -1,  0],
                          [ 1,  0, -1]])
        >>> ph.sign_plot(x, flat = True)
    '''
    try:
        del kwargs['cbar'], kwargs['vmin'], kwargs['vmax'], kwargs['center']
    except:
        pass

    if isinstance(x, DataFrame):
        df = x.copy()
    else:
        x = np.array(x)
        g = g or np.arange(x.shape[0])
        df = DataFrame(x, index=g, columns=g)

    dtype = df.values.dtype

    if not np.issubdtype(dtype, np.integer) and flat:
        raise ValueError("X should be a sign_array or DataFrame of integer values")
    elif not np.issubdtype(dtype, np.floating) and not flat:
        raise ValueError("X should be an array or DataFrame of float p values")

    if not cmap and flat:
        # format: diagonal, non-significant, significant
        cmap = ['1', '#d73027', '#1a9641']
    elif not cmap and not flat:
        # format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
        cmap = ['1', '#ef3b2c',  '#005a32',  '#238b45', '#a1d99b']

    if flat:
        return heatmap(df, vmin=-1, vmax=1, cmap=ListedColormap(cmap), cbar=False, ax=ax, **kwargs)

    else:
        df[(df > 0.05)] = 0
        df[(df < 0.001) & (df > 0)] = 1
        df[(df < 0.01)  & (df > 0.001)] = 2
        df[(df < 0.05)  & (df > 0.01)] = 3
        np.fill_diagonal(df.values, -1)

        if len(cmap) != 5:
            raise ValueError("Cmap list must contain 5 items")

        g = heatmap(df, vmin=-1, vmax=3, cmap=ListedColormap(cmap), center=1, cbar=False, ax=ax, **kwargs)

        cbar_ax = g.figure.add_axes(cbar_ax_bbox or [0.95, 0.35, 0.04, 0.3])
        cbar = ColorbarBase(cbar_ax, cmap=ListedColormap(cmap[1:]), boundaries=[0,1,2,3,4])
        cbar.set_ticks(np.linspace(0.5, 3.5, 4))
        cbar.set_ticklabels(['NS', 'p < 0.001', 'p < 0.01', 'p < 0.05'])

        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor('0.5')
        cbar.ax.tick_params(size=0)

        return g, cbar
