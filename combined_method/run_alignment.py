"""
Execution module for testing the combined alignment approaches.
"""

import sys
import os
import csv
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from combined_method.core import align_coarse_to_fine, align_fusion
from feature_matching.src.alignment.utils import filter_outliers_and_smooth
from feature_matching.src.alignment.visualization import plot_alignment, create_side_by_side_video, generate_html_report

DATASET_DIR = "dataset"
ALGORITHMS = ["ORB", "BRISK", "AKAZE"]

def get_available_plans():
    if not os.path.exists(DATASET_DIR):
        print(f"Warning: Dataset directory '{DATASET_DIR}' not found.")
        return []

    plans = []
    for d in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, d)
        if os.path.isdir(path) and d.lower().startswith("plan"):
            plans.append(d)
    return sorted(plans)

def save_results_to_csv(filepath, results):
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

def run_alignment_and_save(video1_path, video2_path, algo_name, plan_dir, method_name, align_func, **kwargs):
    print(f"\n--- Running {method_name} with {algo_name} ---")

    algo_dir = os.path.join(plan_dir, f"{method_name}_{algo_name.lower()}")
    os.makedirs(algo_dir, exist_ok=True)

    raw_results = align_func(video1_path, video2_path, algo_name=algo_name, **kwargs)

    raw_csv_path = os.path.join(algo_dir, "alignment_raw.csv")
    save_results_to_csv(raw_csv_path, raw_results)

    if not raw_results:
        print("No matches found.")
        return

    print("  > Post-processing (smoothing)...")
    clean_results = filter_outliers_and_smooth(raw_results)

    clean_csv_path = os.path.join(algo_dir, "alignment_clean.csv")
    save_results_to_csv(clean_csv_path, clean_results)

    print("  > Generating visualizations...")
    plot_path = os.path.join(algo_dir, "plot.png")
    plot_alignment(clean_results, plot_path)

    # Simplified video generation for speed when testing multiple methods
    video_simple_path = os.path.join(algo_dir, "comparison_simple.mp4")
    create_side_by_side_video(video1_path, video2_path, clean_results,
                              video_simple_path, max_frames=2700,
                              algorithm=algo_name, mode="simple")

    report_path = os.path.join(algo_dir, "report.html")
    stats = {
        "algorithm": f"{method_name}_{algo_name}",
        "avg_score": sum(r["score"] for r in raw_results) / len(raw_results) if raw_results else 0
    }
    generate_html_report(clean_results, stats, report_path)


def process_plan(plan_name, skip_video=False):
    print(f"\nProcessing {plan_name}...")

    plan_dir = os.path.join(DATASET_DIR, plan_name)
    video1_path = os.path.join(plan_dir, "video1.mp4")
    video2_path = os.path.join(plan_dir, "video2.mp4")

    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        print(f"Error: Missing videos in {plan_dir}. Expected video1.mp4 and video2.mp4.")
        return

    for algo in ALGORITHMS:
        # Run Coarse to Fine
        run_alignment_and_save(video1_path, video2_path, algo, plan_dir, "coarse_to_fine", align_coarse_to_fine, sample_rate=15, window_size=15)

        # Run Fusion
        run_alignment_and_save(video1_path, video2_path, algo, plan_dir, "fusion", align_fusion, sample_rate=15, penalty=1.5)


def main():
    print("=" * 60)
    print("Combined Video Alignment (Coarse-to-Fine & Fusion)")
    print("=" * 60)

    plans = get_available_plans()

    if not plans:
        print(f"No plans found in {DATASET_DIR}. Please create folders like Plan1, Plan2...")
        return

    selected_plans = []
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
        print("\nAvailable Plans:")
        for i, plan in enumerate(plans):
            print(f"{i+1}. {plan}")

        print("\nOptions:")
        print("a. Process All")
        print("q. Quit")

        choice = input("\nSelect a plan number or option: ").strip().lower()

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

    for plan in selected_plans:
        process_plan(plan)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
