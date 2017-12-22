=========
Post-hocs
=========

This Python package provides statistical post-hoc tests for pairwise multiple comparisons.

Features
--------

- Multiple comparison post-hoc tests (ported from R's
  `PMCMR <https://cran.r-project.org/web/packages/PMCMR/index.html>`_ package):
  * Conover's, Dunn's, Nemenyi's tests for use with Kruskal-Wallis test
  * Quade's, van Waerden's, Durbin's tests
  * Conover's, Nemenyi's tests for use with Friedman test
  * Student's T test, Mann-Whitney U test, and TukeyHSD test.

  All tests are capable of p adjustments for multiple pairwise comparisons.

- Plotting functionality (e.g. significance plots).

- Outlier detection algorithms: simple test based on interquartile range (IQR),
  Grubbs test, Tietjen-Moore test, and generalized extreme Studentized deviate test (ESD test).

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
``pip install posthocs``

Example
-------

  >>> import posthocs as ph
  >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
  >>> # This will return a symmetric array of p values
  >>> ph.posthoc_conover(x, p_adjust = 'holm')
  array([[ 0.        ,  0.00119517,  0.00278329],
         [ 0.00119517,  0.        ,  0.18672227],
         [ 0.00278329,  0.18672227,  0.        ]])

Credits
-------

Thorsten Pohlert, PMCMR's author and maintainer
