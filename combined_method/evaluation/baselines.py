"""
Baseline alignments for evaluation comparison.

Each generator returns a match list in the SAME format as the real algorithm:
    [{"v1_frame": int, "v2_frame": int}, ...]

Available baselines:
    - random_alignment    : each V1 frame mapped to a uniformly random V2 frame
    - linear_alignment    : v2 = round(v1 * N2 / N1) — naive proportional stretch
    - offset_alignment    : real alignment shifted by ±k frames

For cycle consistency, every baseline can also produce a "backward" version
(V2 → V1), simply by swapping n parameters.

Important: the cycle consistency metric is degenerate for `linear_alignment`
(it round-trips to ≈ 0 by construction) and for `offset_alignment` if the
same offset is applied to both directions. This is itself an interesting
finding to discuss in the report — cycle catches noise, not systematic bias.
"""

from typing import List, Dict
import numpy as np


def random_alignment(n_src: int, n_dst: int, seed: int = 42) -> List[Dict]:
    """
    For each src frame i in [0, n_src), pick a uniform random dst frame in [0, n_dst).

    Args:
        n_src: number of frames in the source video (V1 for forward, V2 for backward).
        n_dst: number of frames in the destination video.
        seed:  RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    dst_indices = rng.integers(0, n_dst, size=n_src)
    return [{"v1_frame": int(i), "v2_frame": int(dst_indices[i])} for i in range(n_src)]


def linear_alignment(n_src: int, n_dst: int) -> List[Dict]:
    """
    Naive proportional stretch:  dst = round(src * n_dst / n_src).

    This is the "do nothing intelligent" baseline — the value-add of any
    real alignment algorithm should beat this.
    """
    if n_src <= 0:
        return []
    matches = []
    ratio = (n_dst - 1) / max(n_src - 1, 1)
    for i in range(n_src):
        j = int(round(i * ratio))
        j = max(0, min(n_dst - 1, j))
        matches.append({"v1_frame": int(i), "v2_frame": int(j)})
    return matches


def offset_alignment(base_matches: List[Dict], offset: int, n_dst: int) -> List[Dict]:
    """
    Take a real alignment and shift the destination index by `offset` frames.

    Args:
        base_matches: real alignment to perturb.
        offset:       frames to add to each v2_frame (can be negative).
        n_dst:        max destination frame count, for clipping.
    """
    out = []
    for m in base_matches:
        j = int(m["v2_frame"]) + int(offset)
        j = max(0, min(n_dst - 1, j))
        out.append({"v1_frame": int(m["v1_frame"]), "v2_frame": j})
    return out


def identity_alignment(n_src: int, n_dst: int) -> List[Dict]:
    """
    Trivial baseline: v2 = v1 (clipped to n_dst). Useful when videos are
    nearly the same length — shows what "doing nothing" gives.
    """
    matches = []
    for i in range(n_src):
        j = max(0, min(n_dst - 1, i))
        matches.append({"v1_frame": int(i), "v2_frame": int(j)})
    return matches
