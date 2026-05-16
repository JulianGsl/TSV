"""
LPIPS (Learned Perceptual Image Patch Similarity) metric.

Lower LPIPS = more perceptually similar frames.
Range: [0, ~1], where 0 means identical.

Requires: pip install lpips torch torchvision

Improvements over the naive implementation:
  - Optional photometric normalization (histogram matching V2 → V1) so the
    score isolates ALIGNMENT quality from inherent illumination/exposure
    differences between the two recordings.
  - Optional ROI mask (e.g. lower 60 % of the frame) so sky/distant
    background does not dilute the discriminative pixels.
  - VGG backbone is the recommended option for evaluation runs (more
    discriminative); AlexNet remains the default for speed.
"""

import numpy as np
import cv2

try:
    from .base import BaseMetric
    from .image_ops import match_histograms, apply_roi
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from combined_method.evaluation.base import BaseMetric
    from combined_method.evaluation.image_ops import match_histograms, apply_roi


class LPIPSMetric(BaseMetric):
    """
    Perceptual similarity using deep VGG/AlexNet features.

    Uses the `lpips` package (Zhang et al., 2018).

    Args:
        net: Backbone network — 'alex' (fast, default) or 'vgg' (slower, more
             discriminative; preferred for thesis-quality evaluation).
        photometric_normalize: Match V2's histogram to V1's before computing
             the score. Removes illumination differences that would otherwise
             dominate.
        roi: Optional region-of-interest. Either:
             - None (use the whole frame, default)
             - "rail_lower" (use the bottom 60 % of the frame)
             - tuple (top, bottom, left, right) of fractions in [0, 1]
    """

    def __init__(
        self,
        net: str = "alex",
        photometric_normalize: bool = True,
        roi=None,
    ):
        import lpips  # deferred import: only required when this metric is used
        import torch

        self._torch = torch
        self._loss_fn = lpips.LPIPS(net=net)
        self._loss_fn.eval()

        # Use GPU if available
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loss_fn = self._loss_fn.to(self._device)

        self._net = net
        self.photometric_normalize = photometric_normalize
        self.roi = roi

    @property
    def name(self) -> str:
        # Tag the variant in the name so JSON outputs are self-describing.
        tag = [self._net]
        if self.photometric_normalize:
            tag.append("photo")
        if self.roi is not None:
            tag.append("roi")
        return "LPIPS[" + "+".join(tag) + "]"

    def higher_is_better(self) -> bool:
        return False  # lower LPIPS = better perceptual match

    def _to_tensor(self, frame_bgr: np.ndarray) -> "torch.Tensor":
        """Convert a BGR uint8 frame to a normalised [-1, 1] RGB tensor (1, 3, H, W)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._torch.from_numpy(rgb).float().permute(2, 0, 1)  # (3, H, W)
        tensor = tensor / 127.5 - 1.0  # [0, 255] → [-1, 1]
        return tensor.unsqueeze(0).to(self._device)  # (1, 3, H, W)

    def compute(self, frame_v1: np.ndarray, frame_v2: np.ndarray) -> float:
        """
        Compute LPIPS between two BGR frames.

        Frames are resized to the same dimensions if they differ (uses frame_v1 size).
        """
        if frame_v1.shape != frame_v2.shape:
            h, w = frame_v1.shape[:2]
            frame_v2 = cv2.resize(frame_v2, (w, h))
        # Normalize photometric
        if self.photometric_normalize:
            frame_v2 = match_histograms(frame_v2, frame_v1)
        # Apply ROI
        if self.roi is not None:
            frame_v1 = apply_roi(frame_v1, self.roi)
            frame_v2 = apply_roi(frame_v2, self.roi)
        # Convert into tensors
        t1 = self._to_tensor(frame_v1)
        t2 = self._to_tensor(frame_v2)
        # Compute LPIPS
        with self._torch.no_grad():
            score = self._loss_fn(t1, t2)

        return float(score.item())
