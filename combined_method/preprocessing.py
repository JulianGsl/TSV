"""
Video preprocessing for DTW alignment.

Goal: reduce the per-video systematic differences (exposure, white balance,
gain, sensor gamma) so feature extractors see a more comparable signal across
V1 and V2.

Technique: global luminance histogram matching. We build a per-video LUT
that maps V2's luminance distribution onto V1's, then apply that LUT to V2
frames before feature extraction. The LUT is derived from a uniform sample
of frames in each video — identical inputs always produce identical LUTs
(no fitting on the ground truth, no Plan-specific tweaking).

Why this is principled
----------------------
Any two recordings of the same trajectory under different conditions
(sun angle, camera settings, time of day) shift the global luminance
distribution. Feature extractors that quantize the luminance signal
(intensity histograms, gradient magnitudes via Sobel, …) inherit that
shift as a *systematic* bias in the cost matrix. Histogram matching
removes the first-order bias before features are computed.

Public API
----------
    build_luminance_lut(reference_path, target_path, n_samples=60)
        → np.ndarray of shape (256,), uint8

    apply_luminance_lut(frame_bgr, lut)
        → BGR frame with adjusted luminance

    Preprocessor
        Stateful helper that bundles "build LUT once, apply to every frame".

Notes
-----
- Matching is done in CIE Lab on the L channel only — preserves the colour
  balance of the target video, only its luminance distribution is rewritten.
- Sample frames are drawn uniformly from each video to capture the full
  range of lighting along the trajectory, not just the opening seconds.
- The reference video is by convention V1 (the older / reference take).
"""

from __future__ import annotations

import cv2
import numpy as np


# ----------------------------------------------------------------------
# LUT construction
# ----------------------------------------------------------------------

def _sample_luminance_hist(video_path: str, n_samples: int = 60) -> np.ndarray:
    """Return a 256-bin histogram of the L channel from `n_samples` uniformly
    spaced frames of the video. Counts, not probabilities (caller normalises)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Empty video: {video_path}")

    indices = np.linspace(0, total - 1, num=min(n_samples, total), dtype=int)
    hist = np.zeros(256, dtype=np.float64)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        # Crop sky band (same convention as feature extractors)
        h = frame.shape[0]
        crop = frame[h // 3:, :]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        L = lab[..., 0]
        h_i, _ = np.histogram(L, bins=256, range=(0, 256))
        hist += h_i
    cap.release()
    return hist


def _cdf(hist: np.ndarray) -> np.ndarray:
    """Normalised cumulative distribution function from a histogram."""
    h = hist.astype(np.float64)
    s = h.sum()
    if s <= 0:
        return np.linspace(0.0, 1.0, num=len(h))
    return np.cumsum(h) / s


def build_luminance_lut(reference_path: str,
                         target_path: str,
                         n_samples: int = 60) -> np.ndarray:
    """Build a 256-entry uint8 LUT mapping target's luminance distribution
    onto reference's.

    Args:
        reference_path: path to the reference video (the one we keep as-is).
        target_path:    path to the video that will be remapped.
        n_samples:      number of frames sampled per video to estimate the
                        luminance histogram. Default 60 covers a long
                        trajectory without crippling startup time.

    Returns:
        lut: uint8 array of shape (256,). `lut[L_target_pixel]` gives the
             new L value such that target's distribution matches reference's.
    """
    ref_hist = _sample_luminance_hist(reference_path, n_samples)
    tgt_hist = _sample_luminance_hist(target_path, n_samples)
    ref_cdf = _cdf(ref_hist)
    tgt_cdf = _cdf(tgt_hist)
    # For each target intensity v, find the reference intensity u such that
    # ref_cdf[u] ≥ tgt_cdf[v]. np.searchsorted does this efficiently.
    lut = np.searchsorted(ref_cdf, tgt_cdf, side="left")
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut


# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------

def apply_luminance_lut(frame_bgr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a luminance LUT to a BGR frame and return the BGR-adjusted frame."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------
# Stateful helper
# ----------------------------------------------------------------------

class Preprocessor:
    """Build a LUT once for a (reference, target) video pair, then apply it
    to frames on demand. Cheap to keep around for the whole pipeline.

    By convention, V1 is the reference (we leave it untouched) and V2 is the
    target (we remap its luminance). This keeps V1 features stable across
    re-runs while making V2 more comparable to V1.

    Usage:
        pp = Preprocessor(v1_path, v2_path)
        adjusted = pp.apply_to_v2(frame_v2)
        # V1 frames are returned unchanged by pp.apply_to_v1(frame_v1)
    """

    def __init__(self, v1_path: str, v2_path: str,
                 n_samples: int = 60, enabled: bool = True):
        self.enabled = enabled
        self.v1_path = v1_path
        self.v2_path = v2_path
        self.lut: np.ndarray | None = None
        if enabled:
            self.lut = build_luminance_lut(v1_path, v2_path, n_samples=n_samples)

    def apply_to_v1(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def apply_to_v2(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or self.lut is None:
            return frame
        return apply_luminance_lut(frame, self.lut)

    def histograms(self, n_samples: int = 60):
        """Return (ref_hist, target_hist_original, target_hist_after_lut) for diagnostics."""
        ref = _sample_luminance_hist(self.v1_path, n_samples)
        tgt = _sample_luminance_hist(self.v2_path, n_samples)
        if self.lut is None:
            return ref, tgt, tgt
        # Apply lut bin-wise to the target histogram for visualisation
        remapped = np.zeros_like(tgt)
        for v, count in enumerate(tgt):
            remapped[int(self.lut[v])] += count
        return ref, tgt, remapped
