"""
Per-frame DTW confidence scoring.

For every V1 sampled frame i, we look at:
- dist_at_dtw_pick   = dist[i, j_dtw]            (what DTW chose)
- row_min            = min over all j of dist[i, j]   (the cheapest match for i)
- row_mean / std     = baseline noise level of the row

A *confident* DTW pick at frame i means: the distance at the DTW pick is
close to the row minimum, and the row has clear structure (high std). A
*low-confidence* pick is one that is far above the minimum, possibly close
to the mean — DTW is essentially guessing.

We translate this into a scalar `confidence ∈ [0, 1]`:
    confidence = clip( (row_mean - dist_at_dtw_pick) / (row_mean - row_min + ε), 0, 1 )

Properties:
    confidence = 1  if DTW picked exactly the row minimum
    confidence = 0  if DTW picked something at or above the row mean (noise)
    confidence = 0.5 if DTW is halfway between min and mean

Why this metric is principled
-----------------------------
The row-mean is a reasonable baseline for "this V1 frame matches a random
V2 frame about this well" — i.e. the noise floor. Anything above it is
*worse* than chance. Anything close to the row minimum is the best the
features can offer. This makes the score directly interpretable as
"how much better than random did DTW do here?".

The scoring is computed on the *raw* dist matrix, not the accumulated
acc_cost — we want to know "is DTW's pick locally good?", not "does DTW's
pick contribute to the cheapest global path?".

Output of `compute_dtw_confidence` matches the DTW path: one score per
DTW waypoint.
"""

from __future__ import annotations

import numpy as np


def compute_dtw_confidence(dist: np.ndarray,
                            path: np.ndarray) -> np.ndarray:
    """Return per-waypoint DTW confidence scores in [0, 1].

    Args:
        dist: (n, m) raw pairwise distance matrix (sampled space).
        path: (L, 2) DTW path, integer indices into dist.

    Returns:
        conf: (L,) float32 array of confidence scores.
    """
    n, m = dist.shape
    path = np.asarray(path, dtype=np.int64)
    if path.size == 0:
        return np.zeros(0, dtype=np.float32)

    # Row statistics, one row at a time but vectorised by waypoint
    rows = dist[path[:, 0], :]                  # (L, m)
    row_min = rows.min(axis=1)                  # (L,)
    row_mean = rows.mean(axis=1)                # (L,)

    dist_pick = dist[path[:, 0], path[:, 1]]    # (L,)
    denom = (row_mean - row_min) + 1e-6
    conf = (row_mean - dist_pick) / denom
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------
# Per-frame projection
# ----------------------------------------------------------------------

def project_to_frames(path: np.ndarray,
                       conf: np.ndarray,
                       indices1: np.ndarray,
                       total_frames_v1: int) -> np.ndarray:
    """Spread per-waypoint confidence into a per-V1-frame array.

    DTW operates in sampled space; we typically need a confidence value for
    every original V1 frame. This function:
      1. Maps each DTW waypoint's V1 index back to original-frame space.
      2. For every frame `f` in [0, total_frames_v1), uses the confidence
         of the nearest DTW waypoint (in V1 original-frame distance).

    Args:
        path:             (L, 2) DTW path in sampled space.
        conf:             (L,) confidence per waypoint.
        indices1:         (n,) original-V1-frame of each sampled V1 row.
        total_frames_v1:  total number of frames in V1.

    Returns:
        per_frame: (total_frames_v1,) float32 confidence aligned to V1 frames.
    """
    path = np.asarray(path, dtype=np.int64)
    indices1 = np.asarray(indices1, dtype=np.int64)
    v1_orig_at_waypoints = indices1[path[:, 0]]
    # Sort waypoints by V1 frame for monotone interpolation
    order = np.argsort(v1_orig_at_waypoints)
    v1_sorted = v1_orig_at_waypoints[order]
    conf_sorted = conf[order]
    # For every frame, find nearest waypoint V1 frame via searchsorted
    frames = np.arange(total_frames_v1, dtype=np.int64)
    pos = np.searchsorted(v1_sorted, frames)
    pos = np.clip(pos, 0, len(v1_sorted) - 1)
    # Consider neighbour on the left too
    left = np.clip(pos - 1, 0, len(v1_sorted) - 1)
    pick_right = np.abs(v1_sorted[pos] - frames) <= np.abs(v1_sorted[left] - frames)
    chosen = np.where(pick_right, pos, left)
    return conf_sorted[chosen].astype(np.float32)


# ----------------------------------------------------------------------
# Rescue zone detection
# ----------------------------------------------------------------------

def find_rescue_zones(per_frame_conf: np.ndarray,
                       threshold: float = 0.25,
                       min_length: int = 20) -> list[tuple[int, int]]:
    """Identify contiguous runs of low-confidence V1 frames.

    A rescue zone is a stretch of `min_length` or more frames where DTW
    confidence is below `threshold`. Phase B can apply special treatment
    inside these zones (e.g. wider search window + consistency check).

    Args:
        per_frame_conf: (N,) confidence per V1 frame, in [0, 1].
        threshold:      frames below this are "low confidence".
        min_length:     minimum run length to qualify as a zone.

    Returns:
        zones: list of (start, end) inclusive frame index pairs.
    """
    low = per_frame_conf < threshold
    zones = []
    in_zone = False
    start = 0
    for i, lo in enumerate(low):
        if lo and not in_zone:
            in_zone = True
            start = i
        elif not lo and in_zone:
            in_zone = False
            if i - start >= min_length:
                zones.append((start, i - 1))
    if in_zone and len(low) - start >= min_length:
        zones.append((start, len(low) - 1))
    return zones
