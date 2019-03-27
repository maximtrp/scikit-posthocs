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

*scikit-posthocs* is a Python package which provides post hoc tests for pairwise multiple comparisons that are usually performed in statistical data analysis to assess the differences between group levels if a statistically significant result of ANOVA test has been obtained.

*scikit-posthocs* is tightly integrated with Pandas DataFrames and NumPy arrays to ensure fast computations and convenient data import and storage.

This package will be useful for statisticians, data analysts, and researchers who use Python in their work.


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


Examples
--------

Parametric ANOVA with post hoc tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a simple example of the one-way analysis of variance (ANOVA) with post hoc tests used to compare *sepal width* means of three groups (three iris species) in *iris* dataset.

To begin, we will import the dataset using statsmodels ``get_rdataset()`` method.

.. code:: python

  >>> import statsmodels.api as sa
  >>> import statsmodels.formula.api as sfa
  >>> import scikit_posthocs as sp
  >>> df = sa.datasets.get_rdataset('iris').data
  >>> df.head()
     sepal_length  sepal_width  petal_length  petal_width species
  0           5.1          3.5           1.4          0.2  setosa
  1           4.9          3.0           1.4          0.2  setosa
  2           4.7          3.2           1.3          0.2  setosa
  3           4.6          3.1           1.5          0.2  setosa
  4           5.0          3.6           1.4          0.2  setosa

Now, we will build a model and run ANOVA using statsmodels ``ols()`` and ``anova_lm()`` methods. Columns ``species`` and ``sepal_width`` contain independent (predictor) and dependent (response) variable values, correspondingly.

.. code:: python

  >>> lm = sfa.ols('sepal_width ~ C(species)', data=df).fit()
  >>> anova = sa.stats.anova_lm(lm)
  >>> print(anova)
                 df     sum_sq   mean_sq         F        PR(>F)
  C(species)    2.0  11.344933  5.672467  49.16004  4.492017e-17
  Residual    147.0  16.962000  0.115388       NaN           NaN

The results tell us that there is a significant difference between groups means (p = 4.49e-17), but does not tell us the exact group pairs which are different in means. To obtain pairwise group differences, we will carry out a posteriori (post hoc) analysis using ``scikits-posthocs`` package. Student T test applied pairwisely gives us the following p values:

.. code:: python

  >>> sp.posthoc_ttest(df, val_col='sepal_width', group_col='species', p_adjust='holm')
                    setosa    versicolor     virginica
  setosa     -1.000000e+00  5.535780e-15  8.492711e-09
  versicolor  5.535780e-15 -1.000000e+00  1.819100e-03
  virginica   8.492711e-09  1.819100e-03 -1.000000e+00

Remember to use a `FWER controlling procedure <https://en.wikipedia.org/wiki/Family-wise_error_rate#Controlling_procedures>`_, such as Holm procedure, when making multiple comparisons. As seen from this table, significant differences in group means are obtained for all group pairs.

Non-parametric ANOVA with post hoc tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If normality and other `assumptions <https://en.wikipedia.org/wiki/One-way_analysis_of_variance>`_ are violated, one can use a non-parametric Kruskal-Wallis H test (one-way non-parametric ANOVA) to test if samples came from the same distribution.

Let's use the same dataset just to demonstrate the procedure. Kruskal-Wallis test is implemented in SciPy package. ``scipy.stats.kruskal`` method accepts array-like structures, but not DataFrames.

.. code:: python

  >>> import scipy.stats as ss
  >>> import statsmodels.api as sa
  >>> import scikit_posthocs as sp
  >>> df = sa.datasets.get_rdataset('iris').data
  >>> data = [df.loc[ids, 'sepal_width'].values for ids in df.groupby('species').groups.values()]

``data`` is a list of 1D arrays containing *sepal width* values, one array per each species. Now we can run Kruskal-Wallis analysis of variance.

.. code:: python

  >>> H, p = ss.kruskal(*data)
  >>> p
  1.5692820940316782e-14

P value tells us we may reject the null hypothesis that the population medians of all of the groups are equal. To learn what groups (species) differ in their medians we need to run post hoc tests. ``scikit-posthocs`` provides a lot of non-parametric tests mentioned above. Let's choose Conover's test.

.. code:: python

  >>> print(sp.posthoc_conover(df, val_col='sepal_width', group_col='species', p_adjust = 'holm'))
                    setosa    versicolor     virginica
  setosa     -1.000000e+00  2.278515e-18  1.293888e-10
  versicolor  2.278515e-18 -1.000000e+00  1.881294e-03
  virginica   1.293888e-10  1.881294e-03 -1.000000e+00

Pairwise comparisons show that we may reject the null hypothesis (p < 0.01) for each pair of species and conclude that all groups (species) differ in their sepal widths.

Data types
~~~~~~~~~~

Internally, ``scikit-posthocs`` uses pandas DataFrames to store and process data, but python lists, NumPy ndarrays, and pandas DataFrames are supported as input data types. Below are usage examples of various input data structures.

Lists and arrays
^^^^^^^^^^^^^^^^

.. code:: python

  >>> x = [[1,2,1,3,1,4], [12,3,11,9,3,8,1], [10,22,12,9,8,3]]
  >>> sp.posthoc_conover(x, p_adjust='holm')
            1         2         3
  1 -1.000000  0.057606  0.007888
  2  0.057606 -1.000000  0.215761
  3  0.007888  0.215761 -1.000000

You can check how it is processed with a hidden function ``__convert_to_df()``:

.. code:: python

  >>> sp.__convert_to_df(x)

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
