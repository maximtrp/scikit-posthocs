import warnings
from typing import List, Optional, Union

import numpy as np
from pandas import DataFrame, Index, Series


def compact_letter_display(
    p_values: Union[DataFrame, np.ndarray],
    alpha: float = 0.05,
    names: Optional[List] = None,
    maxiter: int = 100,
) -> Series:
    """Compact Letter Display (CLD) for pairwise comparison results.

    Assigns letters to groups based on pairwise significance. Groups sharing
    at least one letter are not significantly different from each other.
    This provides a compact summary of pairwise comparison results that is
    commonly used in publications (e.g. as annotations on bar plots).

    Parameters
    ----------
    p_values : Union[DataFrame, np.ndarray]
        Symmetric matrix of p-values from any ``posthoc_*`` function.
        Diagonal values are ignored (treated as non-significant).

    alpha : float = 0.05
        Significance level. Pairs with p-values >= alpha are considered
        not significantly different.

    names : Optional[List] = None
        Group names used as the index of the returned Series. If ``None``
        and ``p_values`` is a DataFrame, its index is used; otherwise
        integer indices ``0, 1, ..., k-1`` are used.

    maxiter : int = 100
        Maximum number of iterations for the set-intersection step.

    Returns
    -------
    result : pandas.Series
        Series mapping each group name to a letter string. Groups sharing
        a letter are not significantly different. Spaces indicate that a
        group does not belong to that letter group.

    Raises
    ------
    ValueError
        If the number of letter groups exceeds the available alphabet
        length (52: a-z then A-Z).

    Warns
    -----
    RuntimeWarning
        If the algorithm does not converge within ``maxiter`` iterations.

    Notes
    -----
    Uses the set-intersection algorithm described by Perktold (unpublished)
    which is equivalent to the sweep-and-absorb algorithm of Piepho (2004)
    for standard use cases. Groups are sorted by their smallest member
    index to ensure a deterministic letter assignment.

    References
    ----------
    Piepho, Hans-Peter (2004). An Algorithm for a Letter-Based
    Representation of All-Pairwise Comparisons. Journal of Computational
    and Graphical Statistics, 13(2), 456-466.
    https://doi.org/10.1198/1061860043515

    Examples
    --------
    >>> import scikit_posthocs as sp
    >>> x = [[1, 2, 1, 3, 1, 4], [12, 3, 11, 9, 3, 8, 1],
    ...      [10, 22, 12, 9, 8, 3], [14, 12, 16, 17, 5, 9]]
    >>> pc = sp.posthoc_dunn(x, p_adjust='holm')
    >>> sp.compact_letter_display(pc, alpha=0.05)
    1    a
    2    ab
    3     b
    4     b
    Name: letters, dtype: str
    """
    pv = np.asarray(p_values, dtype=float)
    k = pv.shape[0]

    if names is None:
        names = list(p_values.index) if isinstance(p_values, DataFrame) else list(range(k))

    # Build "not significantly different" boolean matrix.
    # Diagonal and negative sentinels (posthoc_* use -1 on diagonal) are non-significant.
    same = (pv >= alpha) | (pv < 0)
    np.fill_diagonal(same, True)

    # Precompute each group's neighborhood as a frozenset.
    neighbors = [frozenset(np.nonzero(same[i])[0]) for i in range(k)]

    # Start: one set per group containing its full neighborhood.
    allsets: set = set(neighbors)

    # Iterate: intersect each set with the neighborhoods of its members
    # until convergence, ensuring every set forms a clique.
    converged = False
    for _ in range(maxiter):
        allsets_new = {s.intersection(neighbors[i]) for s in allsets for i in s} - {frozenset()}
        if allsets_new == allsets:
            converged = True
            break
        allsets = allsets_new

    if not converged:
        warnings.warn(
            "compact_letter_display did not converge after %d iterations." % maxiter,
            RuntimeWarning,
            stacklevel=2,
        )

    # Keep only sets where every member's neighborhood contains the full set.
    valid = [s for s in allsets if all(s <= neighbors[i] for i in s)]

    if not valid:
        # All groups are mutually significantly different; one letter each.
        valid = [frozenset({i}) for i in range(k)]

    # Deterministic ordering: primary key = smallest member, secondary = size (larger first).
    valid.sort(key=len, reverse=True)
    valid.sort(key=min)

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(valid) > len(alphabet):
        raise ValueError(
            "Too many letter groups (%d); only %d letters available." % (len(valid), len(alphabet))
        )

    letters = [
        "".join(alphabet[j] if i in group else " " for j, group in enumerate(valid))
        for i in range(k)
    ]

    return Series(letters, index=Index(names), name="letters")
