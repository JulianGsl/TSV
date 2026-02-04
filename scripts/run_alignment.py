"""
Main Module for Image Alignment on Rail Tracks
This module orchestrates the alignment of videos using ORB, BRISK, and AKAZE.
"""

import sys
import os
import csv

# Add project root to path to allow importing src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.alignment.core import align_videos
from src.alignment.utils import filter_outliers_and_smooth
from src.alignment.visualization import plot_alignment, create_side_by_side_video, generate_html_report

DATASET_DIR = "../dataset"
ALGORITHMS = ["ORB", "BRISK", "AKAZE"]

def get_available_plans():
    """Returns a list of Plan directories in the dataset folder."""
    if not os.path.exists(DATASET_DIR):
        return []

    plans = []
    for d in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, d)
        if os.path.isdir(path) and d.lower().startswith("plan"):
            plans.append(d)
    return sorted(plans)

def save_results_to_csv(filepath, results):
    """Saves alignment results to a CSV file."""
    if not results:
        return

    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["v1_frame", "v2_frame", "score"])
            for row in results:
                writer.writerow([row["v1_frame"], row["v2_frame"], row.get("score", 0)])
        print(f"Saved results to {filepath}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

def process_plan(plan_name):
    """Runs alignment algorithms on the specified plan."""
    print(f"\nProcessing {plan_name}...")

    plan_dir = os.path.join(DATASET_DIR, plan_name)
    video1_path = os.path.join(plan_dir, "video1.mp4")
    video2_path = os.path.join(plan_dir, "video2.mp4")

    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        print(f"Error: Missing videos in {plan_dir}. Expected video1.mp4 and video2.mp4.")
        return

    for algo in ALGORITHMS:
        print(f"\n--- Running {algo} ---")

        # Create algorithm-specific folder
        algo_dir = os.path.join(plan_dir, algo.lower())
        os.makedirs(algo_dir, exist_ok=True)

        # 1. Alignment
        # Sample rate 15 (process every 15th frame of video1)
        # Search window 150, search_step 10 (test frames 0,10,20,... up to 150)
        raw_results = align_videos(video1_path, video2_path, algo_name=algo,
                                   sample_rate=15, search_window=150, search_step=10)

        # Save Raw CSV
        raw_csv_path = os.path.join(algo_dir, "alignment_raw.csv")
        save_results_to_csv(raw_csv_path, raw_results)

        if not raw_results:
            print("No matches found.")
            continue

        # 2. Post-Processing
        print("  > Post-processing (smoothing)...")
        clean_results = filter_outliers_and_smooth(raw_results)

        # Save Clean CSV
        clean_csv_path = os.path.join(algo_dir, "alignment_clean.csv")
        save_results_to_csv(clean_csv_path, clean_results)

        # 3. Visualization
        print("  > Generating visualizations...")

        # Plot
        plot_path = os.path.join(algo_dir, "plot.png")
        plot_alignment(clean_results, plot_path)

        # Generate 3 types of videos
        # Video 1: Simple side-by-side (frames only)
        video_simple_path = os.path.join(algo_dir, "comparison_simple.mp4")
        create_side_by_side_video(video1_path, video2_path, clean_results,
                                  video_simple_path, max_frames=2700,
                                  algorithm=algo, mode="simple")

        # Video 2: All features detected
        video_all_features_path = os.path.join(algo_dir, "comparison_all_features.mp4")
        create_side_by_side_video(video1_path, video2_path, clean_results,
                                  video_all_features_path, max_frames=2700,
                                  algorithm=algo, mode="all_features")

        # Video 3: Only matched features
        video_matched_path = os.path.join(algo_dir, "comparison_matched_only.mp4")
        create_side_by_side_video(video1_path, video2_path, clean_results,
                                  video_matched_path, max_frames=2700,
                                  algorithm=algo, mode="matched_only")

        # HTML Report
        report_path = os.path.join(algo_dir, "report.html")
        stats = {
            "algorithm": algo,
            "avg_score": sum(r["score"] for r in raw_results) / len(raw_results) if raw_results else 0
        }
        generate_html_report(clean_results, stats, report_path)

def main():
    print("=" * 60)
    print("Video Alignment on Rail Tracks")
    print("=" * 60)

    plans = get_available_plans()

    if not plans:
        print(f"No plans found in {DATASET_DIR}. Please create folders like Plan1, Plan2...")
        return

    # Argument handling for automation
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() == "all":
            selected_plans = plans
        elif arg in plans:
            selected_plans = [arg]
        else:
            print(f"Plan '{arg}' not found.")
            return
    else:
        # Interactive Mode
        print("\nAvailable Plans:")
        for i, plan in enumerate(plans):
            print(f"{i+1}. {plan}")

        print("\nOptions:")
        print("a. Process All")
        print("q. Quit")

        choice = input("\nSelect a plan number or option: ").strip().lower()

        selected_plans = []
        if choice == 'a':
            selected_plans = plans
        elif choice == 'q':
            print("Exiting.")
            return
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(plans):
                selected_plans = [plans[idx]]
            else:
                print("Invalid selection.")
                return
        else:
            print("Invalid input.")
            return

    # Process selected plans
    for plan in selected_plans:
        process_plan(plan)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
