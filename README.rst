==============
Post-hoc tests
==============

This Python package provides statistical post-hoc tests for pairwise multiple comparisons. Currently, three non-parametric post-hoc tests are ported from R's `PMCMR <https://cran.r-project.org/web/packages/PMCMR/index.html>`_ package: Conover's, Dunn's, and Nemenyi's tests.

Dependencies
------------

- `NumPy and SciPy packages <https://www.scipy.org/>`_
- `Statsmodels <http://statsmodels.sourceforge.net/>`_
- `Pandas <http://pandas.pydata.org>`_

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
  >>> ph.posthoc_conover(x, p_adjust = 'holm')
  array([[ 0.        ,  0.00119517,  0.00278329],
         [ 0.00119517,  0.        ,  0.18672227],
         [ 0.00278329,  0.18672227,  0.        ]])``

Credits
-------

Thorsten Pohlert, PMCMR's author and maintainer
