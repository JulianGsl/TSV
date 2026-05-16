"""
Phase B rescue mode.

Premise: in zones where DTW is unconfident (per `dtw_confidence`), the
feature matcher (AKAZE/BRISK/ORB) has the *opportunity* to over-rule DTW —
but only if it produces *coherent, consistent* evidence of a different
alignment. We don't want phase B to flip individual frames on noise; we
require a run of consecutive matches that *all* deviate from the DTW
prediction in the same direction by a similar amount.

Detection: a "consistent deviation" over N frames is established when:
  - At least `min_inliers_per_frame` are obtained for ≥ N frames in a row
  - The signed offsets (matched_v2 - dtw_predicted_v2) have the same sign
  - The mean absolute deviation across the run is bounded (so we trust the
    median offset)

When detected, the DTW predictions in the run (and a small look-ahead
window after) are shifted by the median offset before being passed to the
later phases. This is a **principled** override: it does not require
ground truth, it only acts on signal that the matcher itself produced.

Public API
----------
    RescueConfig                — knobs grouped together
    detect_rescue_corrections   — analyses a list of per-frame matches +
                                   confidence and returns a frame→Δ dict
    apply_rescue_corrections    — applies the offset dict in-place onto
                                   the dtw_prediction field of match dicts
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RescueConfig:
    """Parameters for the rescue logic.

    All conservative by default — the rescue is meant to be the exception,
    not the rule. We bias toward keeping the original DTW prediction unless
    the matcher gives strong evidence of a consistent shift.
    """
    # Trigger condition
    low_confidence_threshold: float = 0.25     # frames below this are candidates
    min_zone_length: int = 20                  # need this many low-conf frames in a row

    # Evidence required from phase B inside a zone
    min_consistent_frames: int = 5             # consecutive frames w/ same-sign deviation
    min_inliers_per_frame: int = 6             # AKAZE inlier count to "count" a frame
    max_offset_stddev: float = 8.0             # frame-units; spread of offsets in the run
    min_abs_median_offset: int = 12            # only correct if median |offset| ≥ this

    # Correction scope
    extend_after: int = 30                     # propagate correction this many frames past zone


def _runs_of_consistent_offsets(offsets: np.ndarray,
                                 valid: np.ndarray,
                                 cfg: RescueConfig):
    """Yield (start, end, median_offset) for each run of length
    ≥ cfg.min_consistent_frames where:
      - valid[i] is True
      - offsets[i] all share the same sign and ≈ same magnitude
    """
    n = len(offsets)
    if n == 0:
        return
    sign = np.sign(offsets)
    i = 0
    while i < n:
        if not valid[i] or sign[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and valid[j + 1] and sign[j + 1] == sign[i]:
            j += 1
        if j - i + 1 >= cfg.min_consistent_frames:
            run = offsets[i:j + 1]
            if run.std() <= cfg.max_offset_stddev:
                med = int(np.median(run))
                if abs(med) >= cfg.min_abs_median_offset:
                    yield (i, j, med)
        i = j + 1


def detect_rescue_corrections(matches: list[dict],
                               per_frame_conf: np.ndarray,
                               cfg: RescueConfig | None = None) -> dict[int, int]:
    """Return a frame→offset dict describing what DTW predictions should be shifted.

    Args:
        matches: list of per-frame dicts. Each must contain at least:
            'v1_frame', 'dtw_prediction', 'v2_frame_raw' (matcher's best
            candidate before fallback), 'score' (matcher inlier count).
        per_frame_conf: confidence per V1 frame.
        cfg: rescue config (defaults if None).

    Returns:
        corrections: {v1_frame: signed_offset_to_add_to_dtw_prediction}.
        Frames not in the dict are left untouched.
    """
    cfg = cfg or RescueConfig()
    if not matches:
        return {}

    v1s = np.asarray([m["v1_frame"] for m in matches])
    dtw_pred = np.asarray([m.get("dtw_prediction", m.get("v2_frame", 0)) for m in matches])
    v2_raw = np.asarray([m.get("v2_frame_raw", m.get("v2_frame", 0)) for m in matches])
    scores = np.asarray([m.get("score", 0) for m in matches])
    offsets = v2_raw - dtw_pred

    # A frame is "valid evidence" if (a) the matcher had enough inliers AND
    # (b) the matcher's pick is not literally the DTW prediction.
    valid = (scores >= cfg.min_inliers_per_frame) & (offsets != 0)

    # Restrict to low-confidence regions. We need per_frame_conf interpolated
    # at v1 frames.
    confs = per_frame_conf[np.clip(v1s, 0, len(per_frame_conf) - 1)]
    in_zone = confs < cfg.low_confidence_threshold

    # Mark frames inside long-enough low-confidence runs
    zone_mask = np.zeros_like(in_zone)
    i = 0
    while i < len(in_zone):
        if in_zone[i]:
            j = i
            while j + 1 < len(in_zone) and in_zone[j + 1]:
                j += 1
            if j - i + 1 >= cfg.min_zone_length:
                zone_mask[i:j + 1] = True
            i = j + 1
        else:
            i += 1

    valid &= zone_mask

    corrections: dict[int, int] = {}
    for start, end, med in _runs_of_consistent_offsets(offsets, valid, cfg):
        # Apply the median offset to every frame in the run, plus extend_after
        extended_end = min(len(v1s) - 1, end + cfg.extend_after)
        for k in range(start, extended_end + 1):
            corrections[int(v1s[k])] = med
    return corrections


def apply_rescue_corrections(matches: list[dict],
                              corrections: dict[int, int]) -> int:
    """Shift `dtw_prediction` in-place for frames in `corrections`. Returns
    the number of frames actually corrected."""
    if not corrections:
        return 0
    n_applied = 0
    for m in matches:
        v1 = m.get("v1_frame")
        if v1 in corrections:
            offset = corrections[v1]
            m["dtw_prediction"] = int(m.get("dtw_prediction", m.get("v2_frame", 0)) + offset)
            m["rescue_applied"] = True
            m["rescue_offset"] = offset
            n_applied += 1
    return n_applied
