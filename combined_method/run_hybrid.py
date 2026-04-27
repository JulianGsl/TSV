#!/usr/bin/env python3
"""
A RUN SI ON VEUT DES PARAMETRES PERSONNALISABLES POUR LE PIPELINE COMPLET

Run Hybrid Video Alignment - Complete Pipeline

This script runs the full Coarse-to-Fine alignment pipeline and generates
comprehensive outputs including:
- Multiple video modes (simple, features, optical flow)
- Alignment visualizations
- Quality metrics
- HTML report
- CSV/JSON exports
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combined_method.hybrid_alignment import HybridAligner
from combined_method.visualization import (
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
from new_method.feature_extraction import VideoFeatureExtractor
from new_method.dtw_alignment import compute_dtw
import numpy as np


def run_full_pipeline(
    video1_path: str,
    video2_path: str,
    output_dir: str,
    dtw_sample_rate: int = 5,
    akaze_sample_rate: int = 1,
    search_window: int = 10,
    min_inliers: int = 4,
    dtw_step_penalty: float = 1.5,
    max_video_frames: int = None,
    generate_all_videos: bool = True,
    verbose: bool = True
):
    """
    Run the complete hybrid alignment pipeline with all outputs.

    Args:
        video1_path: Path to reference video
        video2_path: Path to target video
        output_dir: Directory for all outputs
        dtw_sample_rate: Sample rate for DTW extraction
        akaze_sample_rate: Sample rate for AKAZE refinement
        search_window: Search window around DTW prediction
        min_inliers: Minimum inliers threshold for AKAZE
        dtw_step_penalty: DTW step penalty
        max_video_frames: Limit video output frames
        generate_all_videos: Generate all video modes
        verbose: Print progress
    """
    start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("HYBRID VIDEO ALIGNMENT - FULL PIPELINE")
    print("=" * 70)
    print(f"\nVideo 1: {video1_path}")
    print(f"Video 2: {video2_path}")
    print(f"Output:  {output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    config = {
        "dtw_sample_rate": dtw_sample_rate,
        "akaze_sample_rate": akaze_sample_rate,
        "search_window": search_window,
        "min_inliers": min_inliers,
        "dtw_step_penalty": dtw_step_penalty,
        "max_video_frames": max_video_frames,
    }

    # ================================================================
    # STEP 1: Run Hybrid Alignment
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 1: Running Hybrid Alignment")
    print("=" * 50)

    aligner = HybridAligner(
        dtw_sample_rate=dtw_sample_rate,
        dtw_step_penalty=dtw_step_penalty,
        akaze_sample_rate=akaze_sample_rate,
        search_window=search_window,
        min_inliers_threshold=min_inliers,
        verbose=verbose
    )

    matches = aligner.align_videos(video1_path, video2_path)

    if not matches:
        print("ERROR: No matches found!")
        return

    # ================================================================
    # STEP 2: Compute DTW for visualization
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 2: Computing DTW for Cost Matrix Visualization")
    print("=" * 50)

    extractor = VideoFeatureExtractor(
        resize_dim=(320, 240),
        sample_rate=dtw_sample_rate,
        ignore_sky=True
    )

    features1, indices1 = extractor.extract_features(video1_path)
    features2, indices2 = extractor.extract_features(video2_path)

    # Normalize
    mean1, std1 = np.mean(features1, axis=0), np.std(features1, axis=0)
    mean2, std2 = np.mean(features2, axis=0), np.std(features2, axis=0)
    features1_norm = (features1 - mean1) / (std1 + 1e-6)
    features2_norm = (features2 - mean2) / (std2 + 1e-6)

    path, cost_matrix = compute_dtw(
        features1_norm, features2_norm,
        step_penalty=dtw_step_penalty,
        open_end=True
    )

    print(f"  DTW path length: {len(path)}")
    print(f"  Cost matrix shape: {cost_matrix.shape}")

    # ================================================================
    # STEP 3: Compute Metrics
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 3: Computing Alignment Metrics")
    print("=" * 50)

    metrics = compute_alignment_metrics(matches)

    print(f"  Total matches:        {metrics['total_matches']}")
    print(f"  AKAZE refined:        {metrics['akaze_refined_count']} ({metrics['akaze_refined_percent']:.1f}%)")
    print(f"  DTW fallback:         {metrics['dtw_fallback_count']} ({metrics['dtw_fallback_percent']:.1f}%)")
    print(f"  Monotonicity score:   {metrics['monotonicity_score']:.1f}%")
    print(f"  Mean velocity ratio:  {metrics['velocity_mean']:.3f}")
    print(f"  Mean AKAZE inliers:   {metrics['akaze_score_mean']:.1f}")

    # ================================================================
    # STEP 4: Generate Visualizations
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 4: Generating Visualizations")
    print("=" * 50)

    # Alignment scatter plot
    print("  [1/5] Alignment scatter plot...")
    plot_alignment_scatter(
        matches,
        os.path.join(output_dir, "alignment_scatter.png"),
        title="Hybrid Video Alignment (DTW + AKAZE)"
    )

    # Alignment difference plot
    print("  [2/5] Alignment difference plot...")
    plot_alignment_difference(
        matches,
        os.path.join(output_dir, "alignment_difference.png")
    )

    # Source distribution
    print("  [3/5] Source distribution...")
    plot_source_distribution(
        matches,
        os.path.join(output_dir, "source_distribution.png")
    )

    # Velocity analysis
    print("  [4/5] Velocity analysis...")
    plot_velocity_analysis(
        matches,
        os.path.join(output_dir, "velocity_analysis.png")
    )

    # DTW cost matrix
    print("  [5/5] DTW cost matrix...")
    plot_dtw_cost_matrix(
        cost_matrix, path,
        os.path.join(output_dir, "dtw_cost_matrix.png")
    )

    # ================================================================
    # STEP 5: Generate Videos
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 5: Generating Aligned Videos")
    print("=" * 50)

    video_modes = ["simple"]
    if generate_all_videos:
        video_modes = ["simple", "features", "flow"]

    for mode in video_modes:
        video_output = os.path.join(output_dir, f"aligned_video_{mode}.mp4")
        print(f"  Generating {mode} video: {video_output}")

        try:
            stats = create_aligned_video(
                video1_path, video2_path, matches,
                video_output,
                mode=mode,
                max_frames=max_video_frames
            )
            print(f"    -> {stats['frames_written']} frames, {stats['resolution']}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # ================================================================
    # STEP 6: Export Data
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 6: Exporting Data Files")
    print("=" * 50)

    # CSV export
    csv_path = os.path.join(output_dir, "alignment_results.csv")
    save_alignment_csv(matches, csv_path)
    print(f"  CSV: {csv_path}")

    # JSON metrics
    json_path = os.path.join(output_dir, "metrics.json")
    save_metrics_json(metrics, json_path)
    print(f"  JSON: {json_path}")

    # ================================================================
    # STEP 7: Generate HTML Report
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 7: Generating HTML Report")
    print("=" * 50)

    report_path = generate_html_report(
        matches, metrics, output_dir,
        video1_path, video2_path,
        config=config
    )
    print(f"  Report: {report_path}")

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nOutputs generated in: {output_dir}")
    print("\nFiles created:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  - {f} ({size_str})")

    print(f"\nOpen the report: file://{os.path.abspath(report_path)}")

    return matches, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Video Alignment - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_hybrid.py video1.mp4 video2.mp4

  # Custom output directory
  python run_hybrid.py video1.mp4 video2.mp4 -o results/my_alignment

  # Fast mode (larger sample rates)
  python run_hybrid.py video1.mp4 video2.mp4 --dtw-sample-rate 10 --akaze-sample-rate 5

  # High precision mode
  python run_hybrid.py video1.mp4 video2.mp4 --dtw-sample-rate 2 --search-window 15
        """
    )

    parser.add_argument("video1", help="Path to reference video (J-1)")
    parser.add_argument("video2", help="Path to target video (J)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: ./hybrid_output_<timestamp>)")

    # DTW parameters
    parser.add_argument("--dtw-sample-rate", type=int, default=5,
                        help="Sample rate for DTW feature extraction (default: 5)")
    parser.add_argument("--dtw-penalty", type=float, default=1.5,
                        help="DTW step penalty (default: 1.5)")

    # AKAZE parameters
    parser.add_argument("--akaze-sample-rate", type=int, default=1,
                        help="Sample rate for AKAZE refinement (default: 1)")
    parser.add_argument("--search-window", type=int, default=10,
                        help="Search window around DTW prediction (default: 10)")
    parser.add_argument("--min-inliers", type=int, default=4,
                        help="Minimum RANSAC inliers for AKAZE (default: 4)")

    # Output options
    parser.add_argument("--max-video-frames", type=int, default=None,
                        help="Limit video output frames (default: all)")
    parser.add_argument("--simple-videos-only", action="store_true",
                        help="Only generate simple side-by-side video (faster)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"hybrid_output_{timestamp}"

    # Run pipeline
    run_full_pipeline(
        video1_path=args.video1,
        video2_path=args.video2,
        output_dir=args.output,
        dtw_sample_rate=args.dtw_sample_rate,
        akaze_sample_rate=args.akaze_sample_rate,
        search_window=args.search_window,
        min_inliers=args.min_inliers,
        dtw_step_penalty=args.dtw_penalty,
        max_video_frames=args.max_video_frames,
        generate_all_videos=not args.simple_videos_only,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
