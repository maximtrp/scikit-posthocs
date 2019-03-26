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

You can install the package using ``pip`` :

.. code:: bash

  pip install scikit-posthocs

Input data types
----------------

Python lists, NumPy ndarrays and pandas DataFrames are supported as input data types.

Examples
--------

Parametric ANOVA and post hoc test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a simple example of the one-way analysis of variance (ANOVA) with post hoc tests used to compare *sepal length* means of three groups (three iris species) in *iris* dataset.

To begin, we will import the dataset using statsmodels ``get_rdataset()`` method.

.. code:: python

  >>> import statsmodels.api as sa
  >>> import statsmodels.formula.api as sfa
  >>> import scikit_posthocs as sp
  >>> df = sm.datasets.get_rdataset('iris').data
  >>> df.head()
     sepal_length  sepal_width  petal_length  petal_width species
  0           5.1          3.5           1.4          0.2  setosa
  1           4.9          3.0           1.4          0.2  setosa
  2           4.7          3.2           1.3          0.2  setosa
  3           4.6          3.1           1.5          0.2  setosa
  4           5.0          3.6           1.4          0.2  setosa

Now, we will build a model and run ANOVA using statsmodels ``ols()`` and ``anova_lm()`` methods. Columns ``species`` and ``sepal_length`` contain independent (predictor) and dependent (response) variable values, correspondingly.

.. code:: python

  >>> lm = sfa.ols('sepal_width ~ C(species)', data=df).fit()
  >>> anova = sm.stats.anova_lm(lm)
  >>> print(anova)
                 df     sum_sq   mean_sq         F        PR(>F)
  C(species)    2.0  11.344933  5.672467  49.16004  4.492017e-17
  Residual    147.0  16.962000  0.115388       NaN           NaN

The results tell us that there is a significant difference between groups means (p = 4.49e-17), but does not tell us the exact group pairs which are different by means. To obtain pairwise group differences, we will carry out a posteriori (post hoc) analysis using ``scikits-posthocs`` package. Student T test applied pairwisely gives us the following p values:

.. code:: python

  >>> sp.posthoc_ttest(df, val_col='sepal_length', group_col='species')
                    setosa    versicolor     virginica
  setosa     -1.000000e+00  8.985235e-18  6.892546e-28
  versicolor  8.985235e-18 -1.000000e+00  1.724856e-07
  virginica   6.892546e-28  1.724856e-07 -1.000000e+00

As seen from this table, significant differences in group means are obtained for all group pairs.

Non-parametric ANOVA and post hoc test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If normality and other `assumptions <https://en.wikipedia.org/wiki/One-way_analysis_of_variance>`_ are violated, one can use a non-parametric Kruskall-Wallis H test (one-way non-parametric ANOVA) to test if samples came from the same distribution.



Significance plots
------------------

P values can be plotted using a heatmap:

.. code:: python

  >>> pc = sp.posthoc_conover(x, val_col='values', group_col='groups')
  >>> heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
  >>> sp.sign_plot(pc, **heatmap_args)

.. image:: images/plot-conover.png

Custom colormap applied to a plot:

.. code:: python

  >>> pc = sp.posthoc_conover(x, val_col='values', group_col='groups')
  >>> # Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
  >>> cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
  >>> heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
  >>> sp.sign_plot(pc, **heatmap_args)

.. image:: images/plot-conover-custom-cmap.png

Acknowledgement
---------------

Thorsten Pohlert, PMCMR author and maintainer
