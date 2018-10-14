__version__ = '0.4.0'

from scikit_posthocs._posthocs \
    import posthoc_anderson, posthoc_conover, posthoc_conover_friedman,\
            posthoc_dscf, posthoc_dunn, posthoc_durbin, posthoc_mackwolfe,\
            posthoc_mannwhitney, posthoc_miller_friedman, posthoc_nemenyi,\
            posthoc_nemenyi_friedman, posthoc_npm_test, posthoc_quade,\
            posthoc_scheffe, posthoc_siegel_friedman, posthoc_tamhane,\
            posthoc_ttest, posthoc_tukey, posthoc_tukey_hsd,\
            posthoc_vanwaerden, posthoc_wilcoxon

from scikit_posthocs._plotting \
    import sign_array, sign_plot, sign_table

from scikit_posthocs._outliers \
    import outliers_gesd, outliers_grubbs, outliers_iqr, outliers_tietjen
