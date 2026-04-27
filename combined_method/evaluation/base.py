"""
Abstract base class for perceptual video alignment evaluation metrics.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseMetric(ABC):
    """
    Base class for all alignment quality metrics.

    Each metric compares a pair of aligned frames (V1, V2) and returns a scalar score.
    Subclasses must implement `name` and `compute`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name (e.g. 'LPIPS', 'SSIM')."""

    @abstractmethod
    def compute(self, frame_v1: np.ndarray, frame_v2: np.ndarray) -> float:
        """
        Compute the metric for a single aligned frame pair.

        Args:
            frame_v1: BGR frame from the reference video (V1), shape (H, W, 3), uint8.
            frame_v2: BGR frame from the aligned video (V2), shape (H, W, 3), uint8.

        Returns:
            Scalar score. Convention varies per metric (lower = more similar for LPIPS,
            higher = more similar for SSIM).
        """

    def higher_is_better(self) -> bool:
        """Return True if a higher score means better alignment."""
        return False
