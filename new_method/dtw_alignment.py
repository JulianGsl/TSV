import numpy as np
from scipy.spatial.distance import cdist

def compute_dtw(features1, features2, metric='euclidean'):
    """
    Computes the Dynamic Time Warping (DTW) path and distance between two feature sequences.

    Args:
        features1 (np.ndarray): Feature matrix 1 (N x F).
        features2 (np.ndarray): Feature matrix 2 (M x F).
        metric (str): Distance metric to use (default: 'euclidean').

    Returns:
        path (list of tuples): Optimal warping path [(i, j), ...].
        cost_matrix (np.ndarray): Accumulated cost matrix.
    """
    n = len(features1)
    m = len(features2)

    # Compute pairwise distance matrix
    dist_matrix = cdist(features1, features2, metric=metric)

    # Initialize accumulated cost matrix
    acc_cost = np.zeros((n, m))
    acc_cost[0, 0] = dist_matrix[0, 0]

    # Fill first row and column
    for i in range(1, n):
        acc_cost[i, 0] = acc_cost[i-1, 0] + dist_matrix[i, 0]
    for j in range(1, m):
        acc_cost[0, j] = acc_cost[0, j-1] + dist_matrix[0, j]

    # Fill the rest
    for i in range(1, n):
        for j in range(1, m):
            acc_cost[i, j] = dist_matrix[i, j] + min(
                acc_cost[i-1, j],    # Insertion
                acc_cost[i, j-1],    # Deletion
                acc_cost[i-1, j-1]   # Match
            )

    # Backtrack to find the path
    path = []
    i, j = n-1, m-1
    path.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Prefer diagonal (match) if costs are equal to encourage synchronization
            # But strictly follow the min cost path
            min_val = min(acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1])

            if acc_cost[i-1, j-1] == min_val:
                i -= 1
                j -= 1
            elif acc_cost[i-1, j] == min_val:
                i -= 1
            else:
                j -= 1

        path.append((i, j))

    return path[::-1], acc_cost
