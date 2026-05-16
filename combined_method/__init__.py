"""
Combined Method — Coarse-to-Fine Video Alignment

This package contains the active alignment pipeline. It aligns two railway
videos taken on different days at different speeds, using a six-phase
hybrid approach:

    Phase 0  : Preprocessing             (luminance histogram matching V2 -> V1)
    Phase A.1: Feature extraction        (per-frame gradient orientation
                                          histogram, `grad_hist_16`)
    Phase A.2: Open-end DTW              (adaptive step penalty, confidence
                                          side-car)
    Phase B  : Feature matching          (AKAZE/BRISK/ORB + KNN + Lowe +
                                          spatial filter + RANSAC)
    Phase B' : Rescue                    (opportunistic DTW override in
                                          low-confidence zones)
    Phase C  : Global optimization       (top-K candidates, monotonic DP)
    Phase D  : Smoothing                 (weighted moving average +
                                          monotonicity re-enforcement)

The DTW provides a globally robust skeleton; the feature matcher refines
it locally; the DP step enforces a monotonic non-trembling output.

Layout
------
- `hybrid_alignment.py`  HybridAligner class + align_videos_hybrid wrapper
- `preprocessing.py`     Luminance LUT (Phase 0)
- `dtw_confidence.py`    Per-waypoint and per-frame DTW confidence (Phase A.2)
- `phase_b_rescue.py`    Opportunistic override of DTW predictions (Phase B')
- `visualization.py`     Plots, side-by-side videos, HTML report
- `evaluation/`          SSIM, LPIPS, cycle consistency, baselines
- `dtw_diagnostic/`      Standalone diagnostic subsystem (separate, not part
                         of the production pipeline)

Entry-point scripts
-------------------
- `run_alignment_hybrid.py`  Run the alignment (interactive or CLI).
- `run_evaluation.py`        Evaluate a cached alignment (LPIPS / SSIM /
                             cycle consistency).
- `run_full_evaluation.py`   Sweep all algorithms, all metrics, all
                             baselines.
"""

from .hybrid_alignment import align_videos_hybrid, HybridAligner, SUPPORTED_ALGORITHMS
from .preprocessing import (
    Preprocessor,
    build_luminance_lut,
    apply_luminance_lut,
)
from .dtw_confidence import (
    compute_dtw_confidence,
    project_to_frames,
    find_rescue_zones,
)
from .phase_b_rescue import (
    RescueConfig,
    detect_rescue_corrections,
    apply_rescue_corrections,
)
from .visualization import (
    create_aligned_video,
    plot_alignment_scatter,
    plot_alignment_difference,
    plot_source_distribution,
    plot_velocity_analysis,
    plot_dtw_cost_matrix,
    compute_alignment_metrics,
    generate_html_report,
    save_alignment_csv,
    save_metrics_json,
)

__all__ = [
    # Core alignment
    "align_videos_hybrid",
    "HybridAligner",
    "SUPPORTED_ALGORITHMS",
    # Phase 0
    "Preprocessor",
    "build_luminance_lut",
    "apply_luminance_lut",
    # Phase A.2 confidence
    "compute_dtw_confidence",
    "project_to_frames",
    "find_rescue_zones",
    # Phase B' rescue
    "RescueConfig",
    "detect_rescue_corrections",
    "apply_rescue_corrections",
    # Visualization & I/O
    "create_aligned_video",
    "plot_alignment_scatter",
    "plot_alignment_difference",
    "plot_source_distribution",
    "plot_velocity_analysis",
    "plot_dtw_cost_matrix",
    "compute_alignment_metrics",
    "generate_html_report",
    "save_alignment_csv",
    "save_metrics_json",
]
