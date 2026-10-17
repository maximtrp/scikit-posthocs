import numpy as np
from scipy import stats

import scikit_posthocs as sp
import scikit_posthocs._posthocs as posthocs


def test_dunnett_t3_two_groups_do_not_use_quadrature(monkeypatch):
    def unexpected_quadrature(*args, **kwargs):
        raise AssertionError("A single maximum-modulus variate has an analytic survival function")

    monkeypatch.setattr(posthocs.integrate, "quad", unexpected_quadrature)
    samples = [np.array([1, 2, 4, 8, 9]), np.array([2, 5, 6, 7, 10, 12, 15])]
    variances = [sample.var(ddof=1) / len(sample) for sample in samples]
    df = sum(variances) ** 2 / sum(
        variance**2 / (len(sample) - 1) for variance, sample in zip(variances, samples)
    )
    statistic = abs(stats.ttest_ind(*samples, equal_var=False).statistic)
    expected = 2 * stats.t.sf(statistic, np.round(df))

    result = sp.posthoc_dunnett_t3(samples)

    np.testing.assert_allclose(result.iloc[0, 1], expected, rtol=1e-14, atol=0)
