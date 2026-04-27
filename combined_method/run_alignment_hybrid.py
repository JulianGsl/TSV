#!/usr/bin/env python3
"""
Hybrid Video Alignment Runner

Main script for running Coarse-to-Fine video alignment (DTW + Feature Matching)
on rail track videos organized in a dataset structure.

Supports multiple feature matching algorithms: AKAZE, BRISK, ORB

This mirrors the feature_matching workflow for consistency.
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path to allow imports
# Script is in: TSV/combined_method/run_alignment_hybrid.py
# We need: TSV/ as the root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level to TSV/
sys.path.insert(0, project_root)

from combined_method.hybrid_alignment import HybridAligner, SUPPORTED_ALGORITHMS
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

# Dataset directory relative to current working directory
DATASET_DIR = "../dataset"


def get_available_plans():
    """Returns a list of Plan directories in the dataset folder."""
    if not os.path.exists(DATASET_DIR):
        print(f"Warning: Dataset directory '{DATASET_DIR}' not found.")
        return []

    plans = []
    for d in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, d)
        if os.path.isdir(path) and d.lower().startswith("plan"):
            plans.append(d)
    return sorted(plans)


def select_algorithm():
    """
    Display algorithm selection menu and return selected algorithm.

    Returns:
        str: Selected algorithm name ("AKAZE", "BRISK", or "ORB")
    """
    print("\n🔧 Select Feature Matching Algorithm:")
    for i, algo in enumerate(SUPPORTED_ALGORITHMS, 1):
        desc = {
            "AKAZE": "Recommended - Good balance of speed and accuracy",
            "BRISK": "Fast - Good for real-time applications",
            "ORB": "Fastest - More features, less precise"
        }
        print(f"   {i}. {algo} - {desc.get(algo, '')}")

    while True:
        choice = input("\n➜ Select algorithm (1-3) [default: 1 AKAZE]: ").strip()

        if choice == "" or choice == "1":
            return "AKAZE"
        elif choice == "2":
            return "BRISK"
        elif choice == "3":
            return "ORB"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def process_plan_hybrid(plan_name, algorithm="AKAZE"):
    """
    Runs hybrid alignment on the specified plan.

    Args:
        plan_name: Name of the plan folder (e.g., "Plan1")
        algorithm: Feature matching algorithm ("AKAZE", "BRISK", or "ORB")

    Generates:
    - Aligned videos (simple, features, flow modes)
    - Alignment visualizations
    - Quality metrics
    - HTML report
    - CSV/JSON exports
    """
    print(f"\n{'=' * 70}")
    print(f"Processing: {plan_name} with {algorithm}")
    print(f"{'=' * 70}")

    plan_dir = os.path.join(DATASET_DIR, plan_name)
    video1_path = os.path.join(plan_dir, "video1.mp4")
    video2_path = os.path.join(plan_dir, "video2.mp4")

    # Validate videos exist
    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        print(f"ERROR: Missing videos in {plan_dir}")
        print(f"  Expected: video1.mp4 and video2.mp4")
        return

    # Create output directory (includes algorithm name)
    output_dir = os.path.join(plan_dir, f"hybrid_{algorithm.lower()}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    start_time = time.time()

    # ================================================================
    # PHASE 1: Hybrid Alignment
    # ================================================================
    print("\n[Phase 1] Running hybrid alignment...")

    aligner = HybridAligner(
        algorithm=algorithm,
        dtw_sample_rate=5,
        dtw_step_penalty=1.5,
        feature_sample_rate=1,
        search_window=10,
        min_inliers_threshold=4,
        verbose=True
    )

    matches = aligner.align_videos(video1_path, video2_path)

    if not matches:
        print("ERROR: No matches found!")
        return

    # ================================================================
    # PHASE 2: DTW for visualization
    # ================================================================
    print("\n[Phase 2] Computing DTW cost matrix...")

    extractor = VideoFeatureExtractor(
        resize_dim=(320, 240),
        sample_rate=5,
        ignore_sky=True
    )

    features1, indices1 = extractor.extract_features(video1_path)
    features2, indices2 = extractor.extract_features(video2_path)

    mean1, std1 = np.mean(features1, axis=0), np.std(features1, axis=0)
    mean2, std2 = np.mean(features2, axis=0), np.std(features2, axis=0)
    features1_norm = (features1 - mean1) / (std1 + 1e-6)
    features2_norm = (features2 - mean2) / (std2 + 1e-6)

    path, cost_matrix = compute_dtw(
        features1_norm, features2_norm,
        step_penalty=1.5,
        open_end=True
    )

    print(f"  DTW path: {len(path)} waypoints")
    print(f"  Cost matrix: {cost_matrix.shape}")

    # ================================================================
    # PHASE 3: Compute metrics
    # ================================================================
    print("\n[Phase 3] Computing alignment metrics...")

    metrics = compute_alignment_metrics(matches)

    print(f"  Total matches:      {metrics['total_matches']}")
    print(f"  AKAZE refined:      {metrics['akaze_refined_count']} ({metrics['akaze_refined_percent']:.1f}%)")
    print(f"  DTW fallback:       {metrics['dtw_fallback_count']} ({metrics['dtw_fallback_percent']:.1f}%)")
    print(f"  Monotonicity:       {metrics['monotonicity_score']:.1f}%")
    print(f"  Velocity ratio:     {metrics['velocity_mean']:.3f} ± {metrics['velocity_std']:.3f}")
    print(f"  AKAZE inliers:      {metrics['akaze_score_mean']:.1f} ± {metrics['akaze_score_std']:.1f}")

    # ================================================================
    # PHASE 4: Generate visualizations
    # ================================================================
    print("\n[Phase 4] Generating visualizations...")

    print("  [1/5] Alignment scatter plot...")
    plot_alignment_scatter(
        matches,
        os.path.join(output_dir, "alignment_scatter.png"),
        title=f"Hybrid Alignment - {plan_name}"
    )

    print("  [2/5] Alignment difference plot...")
    plot_alignment_difference(
        matches,
        os.path.join(output_dir, "alignment_difference.png")
    )

    print("  [3/5] Source distribution...")
    plot_source_distribution(
        matches,
        os.path.join(output_dir, "source_distribution.png")
    )

    print("  [4/5] Velocity analysis...")
    plot_velocity_analysis(
        matches,
        os.path.join(output_dir, "velocity_analysis.png")
    )

    print("  [5/5] DTW cost matrix...")
    plot_dtw_cost_matrix(
        cost_matrix, path,
        os.path.join(output_dir, "dtw_cost_matrix.png")
    )

    # ================================================================
    # PHASE 5: Generate aligned videos
    # ================================================================
    print("\n[Phase 5] Generating aligned videos...")

    video_configs = [
        ("simple", "simple side-by-side (fastest)"),
        ("features", "with AKAZE keypoints"),
        ("flow", "with optical flow visualization")
    ]

    for mode, description in video_configs:
        video_path = os.path.join(output_dir, f"aligned_{mode}.mp4")
        print(f"  Generating {mode} video ({description})...")

        try:
            stats = create_aligned_video(
                video1_path, video2_path, matches,
                video_path,
                mode=mode,
                max_frames=None
            )
            file_size = os.path.getsize(video_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"    ✓ {stats['frames_written']} frames, {stats['resolution']}, {file_size_mb:.1f} MB")
        except Exception as e:
            print(f"    ✗ ERROR: {e}")

    # ================================================================
    # PHASE 6: Export data
    # ================================================================
    print("\n[Phase 6] Exporting data...")

    csv_path = os.path.join(output_dir, "alignment_results.csv")
    save_alignment_csv(matches, csv_path)
    print(f"  CSV: alignment_results.csv ({len(matches)} matches)")

    json_path = os.path.join(output_dir, "metrics.json")
    save_metrics_json(metrics, json_path)
    print(f"  JSON: metrics.json ({len(metrics)} metrics)")

    # ================================================================
    # PHASE 7: Generate HTML report
    # ================================================================
    print("\n[Phase 7] Generating HTML report...")

    config = {
        "algorithm": algorithm,
        "dtw_sample_rate": 5,
        "feature_sample_rate": 1,
        "search_window": 10,
        "min_inliers": 4,
    }

    report_path = generate_html_report(
        matches, metrics, output_dir,
        video1_path, video2_path,
        config=config
    )

    print(f"  Report: report.html")

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 70}")
    print(f"\nOutput files in: {output_dir}")

    file_list = sorted(os.listdir(output_dir))
    for filename in file_list:
        filepath = os.path.join(output_dir, filename)
        size = os.path.getsize(filepath)
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  • {filename:<35} {size_str:>10}")

    print(f"\n✓ Open report: file://{os.path.abspath(report_path)}")


def main():
    """Main entry point with interactive plan and algorithm selection."""
    print("=" * 70)
    print("HYBRID VIDEO ALIGNMENT (Coarse-to-Fine DTW + Feature Matching)")
    print("Supports: AKAZE, BRISK, ORB")
    print("=" * 70)

    plans = get_available_plans()

    if not plans:
        print(f"\nERROR: No plans found in '{DATASET_DIR}'")
        print(f"Please ensure dataset structure:")
        print(f"  {DATASET_DIR}/")
        print(f"  ├── Plan1/")
        print(f"  │   ├── video1.mp4")
        print(f"  │   └── video2.mp4")
        print(f"  ├── Plan2/")
        print(f"  │   ├── video1.mp4")
        print(f"  │   └── video2.mp4")
        print(f"  └── ...")
        return

    # Argument handling for automation
    selected_plans = []
    selected_algorithm = "AKAZE"  # Default

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() == "all":
            selected_plans = plans
        elif arg in plans:
            selected_plans = [arg]
        else:
            print(f"\nERROR: Plan '{arg}' not found.")
            print(f"Available plans: {', '.join(plans)}")
            return

        # Check for algorithm argument
        if len(sys.argv) > 2:
            algo_arg = sys.argv[2].upper()
            if algo_arg in SUPPORTED_ALGORITHMS:
                selected_algorithm = algo_arg
            else:
                print(f"WARNING: Unknown algorithm '{sys.argv[2]}'. Using AKAZE.")
    else:
        # Interactive mode - Select Plan
        print("\n📁 Available Plans:")
        for i, plan in enumerate(plans, 1):
            print(f"   {i}. {plan}")

        print("\n🎛️  Options:")
        print("   a. Process All")
        print("   q. Quit")

        choice = input("\n➜ Select plan number or option: ").strip().lower()

        if choice == "a":
            selected_plans = plans
        elif choice == "q":
            print("\nExiting.")
            return
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(plans):
                selected_plans = [plans[idx]]
            else:
                print("ERROR: Invalid selection.")
                return
        else:
            print("ERROR: Invalid input.")
            return

        # Interactive mode - Select Algorithm
        selected_algorithm = select_algorithm()

    # Process selected plans
    total_start = time.time()

    for i, plan in enumerate(selected_plans, 1):
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(selected_plans)}] {plan}")
        print(f"{'─' * 70}")

        process_plan_hybrid(plan, algorithm=selected_algorithm)

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
