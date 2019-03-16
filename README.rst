===============
scikit-posthocs
===============

.. image:: https://travis-ci.org/maximtrp/scikit-posthocs.svg?branch=master
    :target: https://travis-ci.org/maximtrp/scikit-posthocs
.. image:: https://img.shields.io/readthedocs/scikit-posthocs.svg
    :target: https://scikit-posthocs.readthedocs.io
.. image:: https://img.shields.io/github/issues/maximtrp/scikit-posthocs.svg
    :target: https://github.com/maximtrp/scikit-posthocs/issues
.. image:: https://img.shields.io/pypi/v/scikit-posthocs.svg
    :target: https://pypi.python.org/pypi/scikit-posthocs/
.. image:: https://img.shields.io/badge/donate-PayPal-blue.svg
    :target: https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=K5J3W3WUQ754U&lc=US&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted

*scikit-posthocs* is a Python package which provides post hoc tests for pairwise multiple comparisons that are usually performed in statistical data analysis to assess the differences between group levels if a statistically significant result of ANOVA test has been obtained. *scikit-posthocs* is tightly integrated with Pandas DataFrames and NumPy arrays to ensure fast computations and convenient data import and storage.

.. contents:: Contents:

Background
----------

Python statistical ecosystem is comprised of multiple packages. However, it still has numerous gaps and is surpassed by R packages and capabilities.

`SciPy <https://www.scipy.org/>`_ (version 1.2.0) offers *Student*, *Wilcoxon*, and *Mann-Whitney* tests which are not adapted to multiple pairwise comparisons. `Statsmodels <http://statsmodels.sourceforge.net/>`_ (version 0.9.0) features *TukeyHSD* test which needs some extra actions to be fluently integrated into a data analysis pipeline. `Statsmodels <http://statsmodels.sourceforge.net/>`_ also has good helper methods: ``allpairtest`` (adapts an external function such as ``scipy.stats.ttest_ind`` to multiple pairwise comparisons) and ``multipletests`` (adjusts *p* values to minimize type I and II errors). `PMCMRplus <https://rdrr.io/cran/PMCMRplus/>`_ is a very good R package which has no rivals in Python as it offers more than 40 various tests (including post hoc tests) for factorial and block design data. PMCMRplus was an inspiration and a reference for *scikit-posthocs*.

*scikit-posthocs* attempts to improve Python statistical capabilities by offering a lot of parametric and nonparametric post hoc tests along with outliers detection and basic plotting methods.


Features
--------

- Pairwise multiple comparisons parametric and nonparametric tests:

  - Conover, Dunn, and Nemenyi tests for use with Kruskal-Wallis test.
  - Conover, Nemenyi, Siegel, and Miller tests for use with Friedman test.
  - Quade, van Waerden, and Durbin tests.
  - Student, Mann-Whitney, Wilcoxon, and TukeyHSD tests.
  - Anderson-Darling test.
  - Mack-Wolfe test.
  - Nashimoto and Wright's test (NPM test).
  - Scheffe test.
  - Tamhane T2 test.

- Outliers detection tests:

  - Simple test based on interquartile range (IQR).
  - Grubbs test.
  - Tietjen-Moore test.
  - Generalized Extreme Studentized Deviate test (ESD test).

- Plotting functionality (e.g. significance plots).

All post hoc tests are capable of p adjustments for multiple pairwise comparisons.

Dependencies
------------

- `NumPy and SciPy packages <https://www.scipy.org/>`_
- `Statsmodels <http://statsmodels.sourceforge.net/>`_
- `Pandas <http://pandas.pydata.org/>`_
- `Matplotlib <https://matplotlib.org/>`_
- `Seaborn <https://seaborn.pydata.org/>`_

Compatibility
-------------

Package is compatible with Python 2 and Python 3.

Install
-------

You can install the package with:
``pip install scikit-posthocs``

Examples
--------

List or NumPy array
~~~~~~~~~~~~~~~~~~~

.. code:: python

  import scikit_posthocs as sp
  x = [[1,2,3,4,5], [10,11,12], [21,22,23,24,25]]
  sp.posthoc_conover(x, p_adjust = 'holm')

::

  array([[-1.        ,  0.00119517,  0.00278329],
         [ 0.00119517, -1.        ,  0.18672227],
         [ 0.00278329,  0.18672227, -1.        ]])

Pandas DataFrame
~~~~~~~~~~~~~~~~

Columns specified with ``val_col`` and ``group_col`` args must be melted prior to making comparisons.

.. code:: python

  import scikit_posthocs as sp
  import pandas as pd
  x = pd.DataFrame({"a": [1,2,3,5,1], "b": [12,31,54,62,12], "c": [10,12,6,74,11]})
  x = x.melt(var_name='groups', value_name='values')

::

     groups  values
  0       a       1
  1       a       2
  2       a       3
  3       a       5
  4       a       1
  5       b      12
  6       b      31
  7       b      54
  8       b      62
  9       b      12
  10      c      10
  11      c      12
  12      c       6
  13      c      74
  14      c      11

.. code:: python

  sp.posthoc_conover(x, val_col='values', group_col='groups', p_adjust = 'fdr_bh')

::

            a         b         c
  a -1.000000  0.000328  0.002780
  b  0.000328 -1.000000  0.121659
  c  0.002780  0.121659 -1.000000

Significance plots
~~~~~~~~~~~~~~~~~~

P values can be plotted using a heatmap:

.. code:: python

  pc = sp.posthoc_conover(x, val_col='values', group_col='groups')
  heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
  sp.sign_plot(pc, **heatmap_args)

.. image:: images/plot-conover.png

Custom colormap applied to a plot:

.. code:: python

  pc = sp.posthoc_conover(x, val_col='values', group_col='groups')
  # Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
  cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
  heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
  sp.sign_plot(pc, **heatmap_args)

.. image:: images/plot-conover-custom-cmap.png

Acknowledgement
---------------

Thorsten Pohlert, PMCMR author and maintainer
