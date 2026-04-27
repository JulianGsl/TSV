"""
Image operations shared by the perceptual metrics (SSIM, LPIPS).

Two preprocessing tools that drastically reduce the noise floor of the
metrics in this dataset:

  - match_histograms(src, ref):
        Photometric normalization. Maps `src` to have the same per-channel
        intensity distribution as `ref`. Removes most of the global
        illumination/exposure gap between V1 and V2 so the metric mostly
        reflects geometric alignment quality.

  - apply_roi(frame, roi):
        Crops a region of interest. The default presets focus on the rail
        area at the bottom of the frame, where the alignment matters most;
        the sky and distant background are removed to improve sensitivity.

Both are pure functions and have no extra dependencies (NumPy + OpenCV only).
"""

from typing import Tuple, Union

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Photometric normalization (per-channel histogram matching)
# -----------------------------------------------------------------------------

def _match_one_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """CDF-based histogram matching for a single 2-D uint8 channel."""
    src_hist, _ = np.histogram(src.ravel(), bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(ref.ravel(), bins=256, range=(0, 256))

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    src_cdf /= max(src_cdf[-1], 1.0)
    ref_cdf /= max(ref_cdf[-1], 1.0)

    # For each src intensity, find the ref intensity with the closest CDF value.
    lut = np.searchsorted(ref_cdf, src_cdf).astype(np.uint8)
    return lut[src]


def match_histograms(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """
    Match the per-channel intensity histogram of `src_bgr` to `ref_bgr`.

    Both inputs must be uint8 BGR frames of any (matching or non-matching) shape;
    `src_bgr` is resized to `ref_bgr` if needed.

    Returns a uint8 BGR frame the same shape as `ref_bgr`.
    """
    if src_bgr.shape != ref_bgr.shape:
        h, w = ref_bgr.shape[:2]
        src_bgr = cv2.resize(src_bgr, (w, h))

    out = np.empty_like(src_bgr)
    for c in range(src_bgr.shape[2]):
        out[..., c] = _match_one_channel(src_bgr[..., c], ref_bgr[..., c])
    return out


# -----------------------------------------------------------------------------
# Region of interest
# -----------------------------------------------------------------------------

# Presets tuned for forward-facing rail-track footage. Fractions are
# (top, bottom, left, right) — the kept region is frame[top*H:bottom*H,
# left*W:right*W].
ROI_PRESETS = {
    # Bottom 60 % of the frame — discards sky and far horizon, keeps the
    # rails and immediate trackside objects.
    "rail_lower": (0.40, 1.00, 0.00, 1.00),
    # Central column where the rails converge — even more focused.
    "rail_center": (0.40, 1.00, 0.20, 0.80),
}


def apply_roi(
    frame: np.ndarray,
    roi: Union[str, Tuple[float, float, float, float]],
) -> np.ndarray:
    """
    Crop `frame` to the requested region of interest.

    Args:
        frame: BGR (or grayscale) image of shape (H, W, ...).
        roi:   Either a preset name in ROI_PRESETS, or a 4-tuple of fractions
               (top, bottom, left, right) in [0, 1].

    Returns:
        The cropped view (no copy unless cv2 needs one downstream).
    """
    if isinstance(roi, str):
        if roi not in ROI_PRESETS:
            raise ValueError(
                f"Unknown ROI preset '{roi}'. Available: {list(ROI_PRESETS)}"
            )
        roi = ROI_PRESETS[roi]

    top_f, bottom_f, left_f, right_f = roi
    if not (0.0 <= top_f < bottom_f <= 1.0 and 0.0 <= left_f < right_f <= 1.0):
        raise ValueError(f"Invalid ROI fractions: {roi}")

    h, w = frame.shape[:2]
    top = int(round(top_f * h))
    bottom = int(round(bottom_f * h))
    left = int(round(left_f * w))
    right = int(round(right_f * w))

    return frame[top:bottom, left:right]
