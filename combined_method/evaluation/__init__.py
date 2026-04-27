"""
Evaluation Module for Video Alignment Quality Assessment

Three perceptual / consistency metrics for the hybrid alignment:
- LPIPSMetric              — Learned Perceptual Image Patch Similarity (deep features, lower = better)
- SSIMMetric               — Structural Similarity Index Measure   (Wang et al. 2004, higher = better)
- compute_cycle_consistency — Self-supervised cycle error f∘g       (CycleGAN-inspired, lower = better)

Usage:
    from combined_method.evaluation import (
        evaluate_alignment, LPIPSMetric, SSIMMetric, compute_cycle_consistency,
    )

    perceptual = evaluate_alignment("v1.mp4", "v2.mp4", matches,
                                     metrics=[LPIPSMetric(), SSIMMetric()],
                                     sample_rate=0.10)

    cycle = compute_cycle_consistency("v1.mp4", "v2.mp4",
                                      sample_every=30, algorithm="AKAZE")
"""

from .base import BaseMetric
from .lpips_metric import LPIPSMetric
from .ssim_metric import SSIMMetric
from .cycle_consistency import compute_cycle_consistency
from .evaluate import evaluate_alignment, sample_frame_pairs
from .baselines import (
    random_alignment,
    linear_alignment,
    offset_alignment,
    identity_alignment,
)

__all__ = [
    "BaseMetric",
    "LPIPSMetric",
    "SSIMMetric",
    "compute_cycle_consistency",
    "evaluate_alignment",
    "sample_frame_pairs",
    "random_alignment",
    "linear_alignment",
    "offset_alignment",
    "identity_alignment",
]
