"""
DTW core — Open-end Dynamic Time Warping with adaptive step penalty.

This is the live implementation used by Phase A.2 of the hybrid pipeline
(`hybrid_alignment.py`) and by the diagnostic subsystem
(`dtw_diagnostic/`).

Originally lived in `Old_version/new_method/dtw_alignment.py`. Moved here
on 2026-05 so that the active pipeline does not depend on `Old_version/`.
A shim is kept in the old location for backward compatibility with tests
and external scripts that may still reference the historical path.
"""

from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def compute_dtw(
    features1: np.ndarray,
    features2: np.ndarray,
    metric: str = "euclidean",
    step_penalty: float = 1.0,
    open_end: bool = False,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Compute the Dynamic Time Warping path between two feature sequences.

    Adds a small penalty to non-diagonal moves to discourage stuttering
    (one frame of one sequence aligned to many frames of the other). The
    penalty is adaptive: it scales with the average pairwise distance, so
    the same `step_penalty` value behaves consistently across feature
    spaces of different magnitudes.

    Args:
        features1:    Reference feature matrix, shape (N, F).
        features2:    Target feature matrix, shape (M, F).
        metric:       Pairwise distance metric (passed to `scipy.cdist`).
        step_penalty: Multiplier on the mean distance, applied to every
                      non-diagonal step. Higher = stronger 1:1 preference.
        open_end:     If True, run Open-End DTW (a.k.a. Subsequence DTW):
                      the path may terminate at the minimum of either the
                      last row or the last column, allowing one sequence
                      to be a strict prefix of the other in the geographic
                      sense. If False, the path must reach (N-1, M-1).

    Returns:
        path:         Optimal warping path as a list of (i, j) index pairs,
                      ordered from (0, 0) to the chosen endpoint.
        acc_cost:     Accumulated cost matrix, shape (N, M).
        dist_matrix:  Raw pairwise distance matrix, shape (N, M). Useful
                      for downstream confidence scoring.
    """
    n = len(features1)
    m = len(features2)

    # Pairwise distance matrix.
    dist_matrix = cdist(features1, features2, metric=metric)

    # Adaptive penalty: scale by the mean distance so the same nominal
    # value of `step_penalty` is meaningful across feature spaces.
    avg_dist = float(np.mean(dist_matrix))
    penalty = avg_dist * step_penalty

    # Accumulated cost matrix.
    acc_cost = np.zeros((n, m))
    acc_cost[0, 0] = dist_matrix[0, 0]

    # First row and column: only non-diagonal steps are possible, so each
    # cell pays the step penalty.
    for i in range(1, n):
        acc_cost[i, 0] = acc_cost[i - 1, 0] + dist_matrix[i, 0] + penalty
    for j in range(1, m):
        acc_cost[0, j] = acc_cost[0, j - 1] + dist_matrix[0, j] + penalty

    # Standard DTW recurrence.
    for i in range(1, n):
        for j in range(1, m):
            cost_insertion = acc_cost[i - 1, j] + penalty   # vertical
            cost_deletion = acc_cost[i, j - 1] + penalty    # horizontal
            cost_match = acc_cost[i - 1, j - 1]             # diagonal
            acc_cost[i, j] = dist_matrix[i, j] + min(
                cost_insertion, cost_deletion, cost_match
            )

    # Choose the endpoint of the path.
    if open_end:
        # Find the minimum on the last row (V1 ends, V2 may end earlier)
        # and on the last column (V2 ends, V1 may end earlier). Pick the
        # cheaper of the two.
        last_row_min_j = int(np.argmin(acc_cost[n - 1, :]))
        last_row_min_cost = acc_cost[n - 1, last_row_min_j]

        last_col_min_i = int(np.argmin(acc_cost[:, m - 1]))
        last_col_min_cost = acc_cost[last_col_min_i, m - 1]

        if last_row_min_cost <= last_col_min_cost:
            i, j = n - 1, last_row_min_j
        else:
            i, j = last_col_min_i, m - 1
    else:
        # Classical DTW: forced endpoint at the corner.
        i, j = n - 1, m - 1

    # Backtrack along the optimal path. Tie-break in favour of the diagonal
    # step to encourage 1:1 sync where the costs allow it.
    path: List[Tuple[int, int]] = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            cost_insertion = acc_cost[i - 1, j] + penalty
            cost_deletion = acc_cost[i, j - 1] + penalty
            cost_match = acc_cost[i - 1, j - 1]
            min_val = min(cost_insertion, cost_deletion, cost_match)

            if cost_match == min_val:
                i -= 1
                j -= 1
            elif cost_insertion == min_val:
                i -= 1
            else:
                j -= 1

        path.append((i, j))

    return path[::-1], acc_cost, dist_matrix
