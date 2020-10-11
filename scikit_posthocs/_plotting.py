import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
from seaborn import heatmap
from pandas import DataFrame

def sign_array(p_values, alpha=0.05):
    """
    Significance array

    Converts an array with p values to a significance array where
    0 is False (not significant), 1 is True (significant),
    and -1 is for diagonal elements.

    Parameters
    ----------
    p_values : array_like or ndarray
        An array, any object exposing the array interface, containing
        p values.

    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    result : numpy.ndarray
        Array where 0 is False (not significant), 1 is True (significant),
        and -1 is for diagonal elements.

    Examples
    --------

    >>> p_values = np.array([[ 1.        ,  0.00119517,  0.00278329],
                             [ 0.00119517,  1.        ,  0.18672227],
                             [ 0.00278329,  0.18672227,  1.        ]])
    >>> ph.sign_array(p_values)
    array([[1, 1, 1],
           [1, 1, 0],
           [1, 0, 1]])

    """

    p_values = np.array(p_values)
    p_values[p_values > alpha] = 0
    p_values[(p_values < alpha) & (p_values > 0)] = 1
    np.fill_diagonal(p_values, 1)
    return p_values


def sign_table(p_values, lower=True, upper=True):
    """
    Significance table

    Returns table that can be used in a publication. P values are replaced
    with asterisks: * - p < 0.05, ** - p < 0.01, *** - p < 0.001.

    Parameters
    ----------
    p_values : array_like, or numpy.ndarray, or pandas.DataFrame
        An array, any object exposing the array interface, containing
        p values.

    lower : bool, optional
        Defines whether to return the lower triangle.

    upper : bool, optional
        Defines whether to return the upper triangle.

    Returns
    -------
    result : numpy.ndarray or pandas.DataFrame
        P values masked with asterisks.

    Examples
    --------

    >>> p_values = np.array([[-1.        ,  0.00119517,  0.00278329],
                      [ 0.00119517, -1.        ,  0.18672227],
                      [ 0.00278329,  0.18672227, -1.        ]])
    >>> ph.sign_table(p_values)
    array([['-', '**', '**'],
           ['**', '-', 'NS'],
           ['**', 'NS', '-']], dtype=object)

    """
    if not any([lower, upper]):
        raise ValueError("Either lower or upper triangle must be returned")

    if not isinstance(p_values, DataFrame):
        p_values = np.array(p_values, dtype=np.float)

    ns = p_values > 0.05
    three = (p_values < 0.001) & (p_values >= 0)
    two = (p_values < 0.01) & (p_values >= 0.001)
    one = (p_values < 0.05) & (p_values >= 0.01)

    p_values = p_values.astype(object)
    p_values[ns] = 'NS'
    p_values[three] = '***'
    p_values[two] = '**'
    p_values[one] = '*'

    pv = p_values.values if isinstance(p_values, DataFrame) else p_values
    np.fill_diagonal(pv, '-')
    if not lower:
        pv[np.tril_indices(pv.shape[0], -1)] = ''
    elif not upper:
        pv[np.triu_indices(pv.shape[0], 1)] = ''

    return DataFrame(pv, index=p_values.index, columns=p_values.columns)\
        if isinstance(p_values, DataFrame) else pv


def sign_plot(
    x, g=None, flat=False, labels=True, cmap=None,
    cbar_ax_bbox=None, ax=None, **kwargs
):
    """
    Significance plot, a heatmap of p values (based on Seaborn).

    Parameters
    ----------
    x : array_like, ndarray or DataFrame
        If flat is False (default), x must be an array, any object exposing
        the array interface, containing p values. If flat is True, x must be
        a sign_array (returned by `scikit_posthocs.sign_array` function)

    g : array_like or ndarray, optional
        An array, any object exposing the array interface, containing
        group names.

    flat : bool, optional
        If `flat` is True, plots a significance array as a heatmap using
        seaborn. If `flat` is False (default), plots an array of p values.
        Non-flat mode is useful if you need to  differentiate significance
        levels visually. It is the preferred mode.

    labels : bool, optional
        Plot axes labels (default) or not.

    cmap : list, optional
        1) If flat is False (default):
        List consisting of five elements, that will be exported to
        ListedColormap method of matplotlib. First is for diagonal
        elements, second is for non-significant elements, third is for
        p < 0.001, fourth is for p < 0.01, fifth is for p < 0.05.

        2) If flat is True:
        List consisting of three elements, that will be exported to
        ListedColormap method of matplotlib. First is for diagonal
        elements, second is for non-significant elements, third is for
        significant ones.
        3) If not defined, default colormaps will be used.

    cbar_ax_bbox : list, optional
        Colorbar axes position rect [left, bottom, width, height] where
        all quantities are in fractions of figure width and height.
        Refer to `matplotlib.figure.Figure.add_axes` for more information.
        Default is [0.95, 0.35, 0.04, 0.3].

    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    kwargs
        Keyword arguments to be passed to seaborn heatmap method. These
        keyword args cannot be used: cbar, vmin, vmax, center.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object with the heatmap

    cbar : matplotlib.colorbar.Colorbar, optional
        ColorBar object of cbar if `flat` is set to False.

    Examples
    --------
    >>> x = np.array([[ 1, 1, 1],
                      [ 1, 1, 0],
                      [ 1, 0, 1]])
    >>> ph.sign_plot(x, flat = True)

    """

    for key in ['cbar', 'vmin', 'vmax', 'center']:
        if key in kwargs:
            del kwargs[key]

    if isinstance(x, DataFrame):
        df = x.copy()
    else:
        x = np.array(x)
        g = g or np.arange(x.shape[0])
        df = DataFrame(np.copy(x), index=g, columns=g)

    dtype = df.values.dtype

    if not np.issubdtype(dtype, np.integer) and flat:
        raise ValueError("X should be a sign_array or DataFrame of integers")
    elif not np.issubdtype(dtype, np.floating) and not flat:
        raise ValueError("X should be an array or DataFrame of float p values")

    if not cmap and flat:
        # format: diagonal, non-significant, significant
        cmap = ['1', '#fbd7d4', '#1a9641']
    elif not cmap and not flat:
        # format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
        cmap = ['1', '#fbd7d4', '#005a32', '#238b45', '#a1d99b']

    if flat:
        np.fill_diagonal(df.values, -1)
        hax = heatmap(df, vmin=-1, vmax=1, cmap=ListedColormap(cmap), cbar=False, ax=ax,
                    **kwargs)
        if not labels:
            hax.set_xlabel('')
            hax.set_ylabel('')
        return hax

    else:
        df[(x < 0.001) & (x >= 0)] = 1
        df[(x < 0.01) & (x >= 0.001)] = 2
        df[(x < 0.05) & (x >= 0.01)] = 3
        df[(x >= 0.05)] = 0

        np.fill_diagonal(df.values, -1)

        if len(cmap) != 5:
            raise ValueError("Cmap list must contain 5 items")

        print(df)
        print('-----------')

        hax = heatmap(
            df, vmin=-1, vmax=3, cmap=ListedColormap(cmap), center=1, cbar=False,
            ax=ax, **kwargs
        )
        if not labels:
            hax.set_xlabel('')
            hax.set_ylabel('')

        #hax.set_xticks(np.arange(df.shape[1]) + 0.5)
        #hax.set_yticks(np.arange(df.shape[0]) + 0.5)
        #hax.set_xticklabels(df.columns.values)
        #hax.set_yticklabels(df.index.values)

        cbar_ax = hax.figure.add_axes(cbar_ax_bbox or [0.95, 0.35, 0.04, 0.3])
        cbar = ColorbarBase(cbar_ax, cmap=ListedColormap(cmap[2:] + [cmap[1]]),
                            boundaries=[0, 1, 2, 3, 4])
        cbar.set_ticks(np.linspace(0.5, 3.5, 4))
        cbar.set_ticklabels(['p < 0.001', 'p < 0.01', 'p < 0.05', 'NS'])

        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor('0.5')
        cbar.ax.tick_params(size=0)

        return hax, cbar
