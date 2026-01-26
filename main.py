"""
Main Module for Image Alignment on Rail Tracks
This module orchestrates the alignment of videos using ORB, BRISK, and AKAZE.
"""

import sys
import os
import csv
from alignment import align_videos

DATASET_DIR = "./dataset"
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

def save_results_to_csv(plan_name, algo_name, results):
    """Saves alignment results to a CSV file."""
    if not results:
        print(f"No results to save for {plan_name} - {algo_name}")
        return

    output_dir = os.path.join(DATASET_DIR, plan_name)
    filename = f"alignment_{algo_name.lower()}.csv"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["v1_frame", "v2_frame", "score", "algorithm"])
            for row in results:
                writer.writerow([row["v1_frame"], row["v2_frame"], row["score"], row["algorithm"]])
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
        print(f"  > Running {algo}...")
        # Sample rate 30 (1 per second assuming 30fps) to make it faster for demo
        # Search window 150 frames (5 seconds)
        results = align_videos(video1_path, video2_path, algo_name=algo, sample_rate=30, search_window=150)
        save_results_to_csv(plan_name, algo, results)

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
