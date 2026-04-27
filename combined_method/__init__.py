"""
Combined Method Module - Coarse-to-Fine Video Alignment

This module combines the robustness of DTW-based optical flow alignment (macro)
with the precision of AKAZE feature matching (micro) for optimal video alignment.

Main components:
- HybridAligner: Main alignment class
- align_videos_hybrid: Convenience function for alignment
- visualization: Comprehensive output generation
- run_hybrid: Full pipeline script
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
