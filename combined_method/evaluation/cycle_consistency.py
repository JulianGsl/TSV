"""
Cycle consistency error — self-supervised alignment quality metric.

Inspired by CycleGAN (Zhu et al. 2017). Idea:

  Forward pass:  f(i) = j      (V1 frame i  →  best V2 frame j)
  Backward pass: g(j) = i'     (V2 frame j  →  best V1 frame i')
  Error:         |i − g(f(i))|  (in frames)

A perfect aligner round-trips exactly. Any drift, ambiguity or asymmetry
in the algorithm shows up as cycle error. Lower is better.
"""

import bisect
from typing import Dict, List, Optional

import numpy as np

try:
    from ..hybrid_alignment import align_videos_hybrid
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from combined_method.hybrid_alignment import align_videos_hybrid


def _matches_to_dict(matches: List[Dict], key: str = "v1_frame", value: str = "v2_frame") -> Dict[int, int]:
    """Convert list of match dicts to a {key_frame: value_frame} dictionary."""
    return {int(m[key]): int(m[value]) for m in matches}


def _nearest_lookup(mapping: Dict[int, int], key: int, sorted_keys: List[int]) -> int:
    """
    Look up `key` in `mapping`. If absent, fall back to the nearest key.
    `sorted_keys` is the pre-sorted list of mapping.keys() for binary search.
    """
    if key in mapping:
        return mapping[key]

    # Binary search for nearest key
    idx = bisect.bisect_left(sorted_keys, key)
    if idx == 0:
        nearest = sorted_keys[0]
    elif idx == len(sorted_keys):
        nearest = sorted_keys[-1]
    else:
        before = sorted_keys[idx - 1]
        after = sorted_keys[idx]
        nearest = before if (key - before) <= (after - key) else after

    return mapping[nearest]


def compute_cycle_consistency(
    video_v1_path: str,
    video_v2_path: str,
    sample_every: int = 30,
    algorithm: str = "AKAZE",
    forward_matches: Optional[List[Dict]] = None,
    backward_matches: Optional[List[Dict]] = None,
    align_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compute cycle-consistency error between V1 and V2 alignments.

    Runs the alignment twice (V1→V2 and V2→V1) unless mappings are provided.

    Args:
        video_v1_path:    Reference video.
        video_v2_path:    Target video.
        sample_every:     Sample one V1 frame every N frames (30 → 10 % at 30 fps).
        algorithm:        Feature matching algorithm (AKAZE / BRISK / ORB).
        forward_matches:  Optional pre-computed V1→V2 matches (avoids recomputation).
        backward_matches: Optional pre-computed V2→V1 matches.
        align_kwargs:     Extra kwargs forwarded to align_videos_hybrid.
        verbose:          Print progress.

    Returns:
        {
            "n_sampled":         int,
            "sample_every":      int,
            "errors":            list of int (per-sample |i - i'|),
            "mean_error":        float,
            "std_error":         float,
            "median_error":      float,
            "max_error":         int,
            "perfect_count":     int,    # samples with error == 0
            "perfect_percent":   float,
            "within_1_percent":  float,  # % of samples with error <= 1
            "within_5_percent":  float,  # % of samples with error <= 5
        }
    """
    align_kwargs = align_kwargs or {}

    # Forward: V1 → V2
    if forward_matches is None:
        if verbose:
            print(f"  Running forward alignment V1 → V2 ({algorithm})...")
        forward_matches = align_videos_hybrid(
            video_v1_path, video_v2_path,
            algorithm=algorithm, verbose=verbose, **align_kwargs
        )

    # Backward: V2 → V1 (swap arguments)
    if backward_matches is None:
        if verbose:
            print(f"  Running backward alignment V2 → V1 ({algorithm})...")
        backward_matches = align_videos_hybrid(
            video_v2_path, video_v1_path,
            algorithm=algorithm, verbose=verbose, **align_kwargs
        )

    # Forward map: f(i) = j  (i = V1, j = V2)
    f_map = _matches_to_dict(forward_matches, "v1_frame", "v2_frame")

    # Backward map: in the backward run, the function received V2 first, so
    # the dict entry's "v1_frame" is actually a V2 frame, and "v2_frame" is a V1 frame.
    g_map = _matches_to_dict(backward_matches, "v1_frame", "v2_frame")
    g_keys_sorted = sorted(g_map.keys())

    # Sample every Nth V1 frame from the forward map
    f_keys_sorted = sorted(f_map.keys())
    sampled_v1 = [k for k in f_keys_sorted if k % sample_every == 0]

    if not sampled_v1:
        raise RuntimeError("No sampled V1 frames — check sample_every vs. video length.")

    errors = []
    for i, v1_idx in enumerate(sampled_v1):
        j = f_map[v1_idx]                                # f(i)
        i_prime = _nearest_lookup(g_map, j, g_keys_sorted)  # g(f(i))
        errors.append(abs(int(v1_idx) - int(i_prime)))

        if verbose and (i % 50 == 0 or i == len(sampled_v1) - 1):
            print(f"    [{i + 1}/{len(sampled_v1)}] i={v1_idx} → j={j} → i'={i_prime} | err={errors[-1]}")

    arr = np.array(errors, dtype=int)
    n = len(arr)

    return {
        "n_sampled":        n,
        "sample_every":     sample_every,
        "errors":           arr.tolist(),
        "mean_error":       float(np.mean(arr)),
        "std_error":        float(np.std(arr)),
        "median_error":     float(np.median(arr)),
        "max_error":        int(np.max(arr)),
        "perfect_count":    int(np.sum(arr == 0)),
        "perfect_percent":  float(np.mean(arr == 0) * 100.0),
        "within_1_percent": float(np.mean(arr <= 1) * 100.0),
        "within_5_percent": float(np.mean(arr <= 5) * 100.0),
        "higher_is_better": False,
    }
