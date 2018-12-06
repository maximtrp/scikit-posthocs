---
title: 'scikit-posthocs: Pairwise multiple comparison tests in Python'
tags:
  - Python
  - statistics
  - post hoc
authors:
  - name: Maksim A. Terpilowski
    orcid: 0000-0003-2586-4633
    affiliation: 1
affiliations:
 - name: Institute of Evolutionary Physiology and Biochemistry, Saint Petersburg, Russia
   index: 1
date: 6 December 2018
bibliography: paper.bib
---

# Summary

Python currently lacks implementation of many multiple pairwise (post hoc) comparison tests that are routinely performed following a statistically significant result of a parametric or nonparametric analysis of variance (ANOVA) test to assess the differences between group levels. 

The ``scikit-posthocs`` package is aimed at filling this gap by providing a number of nonparametric and parametric pairwise comparisons tests as well as outlier detection algorithms implemented in Python: 

1. Conover, Dunn, and Nemenyi tests (for use with Kruskal-Wallis test)
2. Conover, Nemenyi, Siegel, and Miller tests (for use with Friedman test)
3. Quade, van Waerden, Durbin, Student, Mann-Whitney, Wilcoxon, TukeyHSD, Anderson-Darling, Mack-Wolfe, Nashimoto and Wright (NPM), Scheffe, and Tamhane T2 tests.
4. Interquartile range (IQR), Grubbs, Tietjen-Moore, and Generalized Extreme Studentized Deviate (ESD) tests.

It also has plotting functionality to present the results of pairwise comparisons as a heatmap (significance plot).

This package is compatible with Python 2 and 3 versions, relies heavily and extends the functionality of ``statsmodels``, ``SciPy`` and ``PMCMRplus`` packages [@Seabold2010], [@Jones2001], [Pohlert2018]. It is also integrated with ``Pandas`` [@McKinney2010] and ``Numpy`` [@Oliphant2006] for efficient computations and data analysis. The package is fully documented and comes with a Jupyter notebook example.

# Figures

![Significance plot](figure.png)

# References
