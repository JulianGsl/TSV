import numpy as np
from scipy.spatial.distance import cdist


def compute_dtw(features1, features2, metric='euclidean', step_penalty=1.0, open_end=False):
    """
    Computes the Dynamic Time Warping (DTW) path and distance between two feature sequences.
    Adds a step penalty to non-diagonal moves to encourage 1:1 mapping (constant speed).

    Args:
        features1 (np.ndarray): Feature matrix 1 (N x F).
        features2 (np.ndarray): Feature matrix 2 (M x F).
        metric (str): Distance metric to use (default: 'euclidean').
        step_penalty (float): Penalty added to vertical/horizontal steps (insertions/deletions)
                              to reduce stuttering in alignment.
        open_end (bool): If True, implements Open-End DTW (Subsequence DTW).
                         The algorithm finds the optimal endpoint on the last row OR last column,
                         allowing one video to end earlier geographically.

    Returns:
        path (list of tuples): Optimal warping path [(i, j), ...].
        cost_matrix (np.ndarray): Accumulated cost matrix.
    """
    n = len(features1)
    m = len(features2)

    # Compute pairwise distance matrix
    dist_matrix = cdist(features1, features2, metric=metric)

    # Calculate an adaptive penalty based on the average distance
    # so the penalty is somewhat scale invariant.
    avg_dist = np.mean(dist_matrix)
    penalty = avg_dist * step_penalty

    # Initialize accumulated cost matrix
    acc_cost = np.zeros((n, m))
    acc_cost[0, 0] = dist_matrix[0, 0]

    # Fill first row and column (all non-diagonal steps, so apply penalty)
    for i in range(1, n):
        acc_cost[i, 0] = acc_cost[i-1, 0] + dist_matrix[i, 0] + penalty
    for j in range(1, m):
        acc_cost[0, j] = acc_cost[0, j-1] + dist_matrix[0, j] + penalty

    # Fill the rest
    for i in range(1, n):
        for j in range(1, m):
            # Insertion (moving down) or Deletion (moving right) incurs penalty
            cost_insertion = acc_cost[i-1, j] + penalty
            cost_deletion = acc_cost[i, j-1] + penalty
            cost_match = acc_cost[i-1, j-1]  # Diagonal move

            acc_cost[i, j] = dist_matrix[i, j] + min(cost_insertion, cost_deletion, cost_match)

    # Determine starting point for backtracking
    if open_end:
        # Open-End DTW: Find minimum cost on last row OR last column
        # This allows one video to end earlier than the other

        # Minimum on last row (video1 ends, video2 may end earlier)
        last_row_min_j = np.argmin(acc_cost[n-1, :])
        last_row_min_cost = acc_cost[n-1, last_row_min_j]

        # Minimum on last column (video2 ends, video1 may end earlier)
        last_col_min_i = np.argmin(acc_cost[:, m-1])
        last_col_min_cost = acc_cost[last_col_min_i, m-1]

        # Choose the endpoint with minimum cost
        if last_row_min_cost <= last_col_min_cost:
            i, j = n-1, last_row_min_j
        else:
            i, j = last_col_min_i, m-1
    else:
        # Standard DTW: Start from bottom-right corner
        i, j = n-1, m-1

    # Backtrack to find the path
    path = []
    path.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Backtrack with the same penalty logic to follow the true min cost path
            cost_insertion = acc_cost[i-1, j] + penalty
            cost_deletion = acc_cost[i, j-1] + penalty
            cost_match = acc_cost[i-1, j-1]

            # Prefer diagonal (match) if costs are roughly equal to encourage 1:1 sync
            min_val = min(cost_insertion, cost_deletion, cost_match)

            if cost_match == min_val:
                i -= 1
                j -= 1
            elif cost_insertion == min_val:
                i -= 1
            else:
                j -= 1

        path.append((i, j))

    return path[::-1], acc_cost
