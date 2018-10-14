Introduction
============

*scikit-posthocs* provides statistical post-hoc tests for pairwise multiple comparisons and outliers detection algorithms.

Features
--------

- Parametric and nonparametric pairwise multiple comparisons tests:

  - Conover, Dunn, and Nemenyi tests for use with Kruskal-Wallis test.
  - Conover, Nemenyi, Siegel, and Miller tests for use with Friedman test.
  - Quade, van Waerden, and Durbin tests.
  - Student, Mann-Whitney, Wilcoxon, and TukeyHSD tests.
  - Anderson-Darling test.
  - Mack-Wolfe test.
  - Nashimoto and Wright's test (NPM test).
  - Scheffe test.
  - Tamhane T2 test.

- Plotting (significance plots).

- Outliers detection algorithms:

  - Simple interquartile range (IQR) test.
  - Grubbs test.
  - Tietjen-Moore test.
  - Generalized Extreme Studentized Deviate test (ESD test).

  All pairwise tests are capable of p adjustments for multiple pairwise comparisons.
