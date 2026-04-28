"""
Combined Method Module - Coarse-to-Fine Video Alignment

This module combines the robustness of DTW-based optical flow alignment (macro)
with the precision of AKAZE/BRISK/ORB feature matching (micro).

Main components:
- HybridAligner / align_videos_hybrid : alignment engine (hybrid_alignment.py)
- visualization                       : plots, videos, HTML report
- evaluation/                         : SSIM, LPIPS, cycle consistency, baselines

Entry-point scripts:
- run_alignment_hybrid.py  : run the alignment (interactive or CLI)
- run_evaluation.py        : evaluate a cached alignment (LPIPS / SSIM / cycle)
- run_full_evaluation.py   : evaluate all algos × all metrics × all baselines
"""

from .hybrid_alignment import align_videos_hybrid, HybridAligner
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
    save_metrics_json
)

__all__ = [
    'align_videos_hybrid',
    'HybridAligner',
    'create_aligned_video',
    'plot_alignment_scatter',
    'plot_alignment_difference',
    'plot_source_distribution',
    'plot_velocity_analysis',
    'plot_dtw_cost_matrix',
    'compute_alignment_metrics',
    'generate_html_report',
    'save_alignment_csv',
    'save_metrics_json'
]
