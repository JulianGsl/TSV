"""
SSIM (Structural Similarity Index Measure) metric.

Wang et al. 2004 — measures luminance, contrast and structure similarity.
Higher SSIM = more similar frames. Range: [-1, 1], where 1 means identical.

Implemented with OpenCV Gaussian filtering — no extra dependency.

Improvements over the naive implementation:
  - Grayscale by default (canonical Wang 2004 protocol; B/G/R averaging
    triples chromatic noise without a perceptual gain).
  - Optional photometric normalization (histogram matching V2 → V1) so the
    score isolates ALIGNMENT quality from inherent illumination/exposure
    differences between the two recordings.
  - Optional ROI mask (e.g. lower 60 % of the frame for the rail area) so
    sky/distant background does not dilute the discriminative pixels.
"""

import cv2
import numpy as np

try:
    from .base import BaseMetric
    from .image_ops import match_histograms, apply_roi
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from combined_method.evaluation.base import BaseMetric
    from combined_method.evaluation.image_ops import match_histograms, apply_roi


# Standard SSIM constants for 8-bit images (Wang et al. 2004)
_K1 = 0.01
_K2 = 0.03
_L = 255.0
_C1 = (_K1 * _L) ** 2
_C2 = (_K2 * _L) ** 2
_WINDOW = (11, 11)
_SIGMA = 1.5


def _ssim_single_channel(x: np.ndarray, y: np.ndarray) -> float:
    """SSIM on a single 2-D float channel."""
    mu_x = cv2.GaussianBlur(x, _WINDOW, _SIGMA)
    mu_y = cv2.GaussianBlur(y, _WINDOW, _SIGMA)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, _WINDOW, _SIGMA) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, _WINDOW, _SIGMA) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, _WINDOW, _SIGMA) - mu_xy

    num = (2.0 * mu_xy + _C1) * (2.0 * sigma_xy + _C2)
    den = (mu_x2 + mu_y2 + _C1) * (sigma_x2 + sigma_y2 + _C2)

    return float(np.mean(num / den))


class SSIMMetric(BaseMetric):
    """
    Structural Similarity (Wang et al. 2004).

    Args:
        multichannel: If True, average SSIM over B/G/R channels. Default False
                      (canonical grayscale, lower noise).
        photometric_normalize: If True, apply histogram matching V2 → V1 before
                               computing the score. Removes the bulk of the
                               illumination gap between recordings, leaving the
                               metric mostly sensitive to the alignment.
        roi: Optional region-of-interest. Either:
             - None (use the whole frame, default)
             - "rail_lower" (use the bottom 60 % of the frame)
             - tuple (top, bottom, left, right) of fractions in [0, 1]
    """

    def __init__(
        self,
        multichannel: bool = False,
        photometric_normalize: bool = True,
        roi=None,
    ):
        self.multichannel = multichannel
        self.photometric_normalize = photometric_normalize
        self.roi = roi

    @property
    def name(self) -> str:
        # Tag the variant in the name so JSON outputs are self-describing.
        tag = []
        if self.photometric_normalize:
            tag.append("photo")
        if self.roi is not None:
            tag.append("roi")
        return "SSIM" + ("[" + "+".join(tag) + "]" if tag else "")

    def higher_is_better(self) -> bool:
        return True

    def compute(self, frame_v1: np.ndarray, frame_v2: np.ndarray) -> float:
        if frame_v1.shape != frame_v2.shape:
            h, w = frame_v1.shape[:2]
            frame_v2 = cv2.resize(frame_v2, (w, h))

        if self.photometric_normalize:
            frame_v2 = match_histograms(frame_v2, frame_v1)

        if self.roi is not None:
            frame_v1 = apply_roi(frame_v1, self.roi)
            frame_v2 = apply_roi(frame_v2, self.roi)

        x = frame_v1.astype(np.float32)
        y = frame_v2.astype(np.float32)

        if not self.multichannel:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
            return _ssim_single_channel(x, y)

        scores = [_ssim_single_channel(x[..., c], y[..., c]) for c in range(3)]
        return float(np.mean(scores))
