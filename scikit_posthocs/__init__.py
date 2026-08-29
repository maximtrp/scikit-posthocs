__version__ = "0.16.1"

from scikit_posthocs._global import global_simes_test, global_f_test
from scikit_posthocs._omnibus import (
    test_osrt,
    test_durbin,
    test_mackwolfe,
    test_jonckheere,
    test_page,
    test_hartley,
    test_median,
)

from scikit_posthocs._posthocs import (
    posthoc_anderson,
    posthoc_conover,
    posthoc_conover_friedman,
    posthoc_demsar,
    posthoc_dscf,
    posthoc_duncan,
    posthoc_dunn,
    posthoc_dunnett_t3,
    posthoc_durbin,
    posthoc_games_howell,
    posthoc_lsd,
    posthoc_mannwhitney,
    posthoc_median,
    posthoc_miller_friedman,
    posthoc_nemenyi,
    posthoc_nemenyi_friedman,
    posthoc_npm_test,
    posthoc_quade,
    posthoc_scheffe,
    posthoc_siegel_friedman,
    posthoc_snk,
    posthoc_steel,
    posthoc_tamhane,
    posthoc_ttest,
    posthoc_tukey,
    posthoc_tukey_hsd,
    posthoc_vanwaerden,
    posthoc_wilcoxon,
    posthoc_dunnett,
    __convert_to_df,
    __convert_to_block_df,
)

from scikit_posthocs._plotting import (
    sign_array,
    sign_plot,
    sign_table,
    critical_difference_diagram,
)
from scikit_posthocs._grouping import (
    compact_letter_display,
)
from scikit_posthocs._outliers import (
    outliers_gesd,
    outliers_grubbs,
    outliers_iqr,
    outliers_tietjen,
)
