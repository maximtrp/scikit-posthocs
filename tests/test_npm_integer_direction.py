import itertools
import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import scikit_posthocs as sp


@pytest.mark.parametrize(
    "dtype,offset", [("uint8", 0), ("int8", -128), ("int16", -32768), ("uint64", 2**63)]
)
def test_less_with_bounded_integer_observations(dtype, offset):
    samples = [np.array([offset + i, offset + i + 6], dtype=dtype) for i in range(3)]
    saved = [sample.copy() for sample in samples]
    np.random.seed(17)
    result = sp.posthoc_npm_test(samples, alternative="less", nperm=50)
    np.random.seed(17)
    expected = sp.posthoc_npm_test([[i, i + 6] for i in range(3)], alternative="less", nperm=50)
    assert_array_equal(result, expected)
    for sample, original in zip(samples, saved):
        assert_array_equal(sample, original)


@pytest.mark.parametrize("alternative", ["less", "greater"])
@pytest.mark.parametrize("values", [[0, 255, 1, 254, 2, 253], [0, 0, 1, 1, 2, 3]])
def test_npm_matches_exhaustive_pair_count_reference(monkeypatch, alternative, values):
    permutations = list(itertools.permutations(values))
    iterator = iter(permutations)
    monkeypatch.setattr(np.random, "permutation", lambda _: np.array(next(iterator), dtype="uint8"))

    def statistic(v, i, j):
        # Midranks defined by pair counts, independent of pandas rank.
        ranks = [
            1
            + sum((x > y if alternative == "less" else x < y) for x in v)
            + (sum(x == y for x in v) - 1) / 2
            for y in v
        ]
        means = [sum(ranks[2 * g : 2 * g + 2]) / 2 for g in range(3)]
        return max((means[j] - means[g]) / math.sqrt(3.5) for g in range(i, j))

    expected = np.ones((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            observed = statistic(values, i, j)
            p = sum(statistic(p, i, j) >= observed for p in permutations) / len(permutations)
            expected[i, j] = expected[j, i] = p
    result = sp.posthoc_npm_test(
        [np.array(values[i : i + 2], dtype="uint8") for i in range(0, 6, 2)],
        alternative=alternative,
        nperm=len(permutations),
    )
    assert_array_equal(result, expected)
