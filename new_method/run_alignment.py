import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from new_method.feature_extraction import VideoFeatureExtractor
from new_method.dtw_alignment import compute_dtw

def main():
    parser = argparse.ArgumentParser(description="Video Temporal Alignment using DTW")
    parser.add_argument("video1", nargs="?", default="dataset/PlanTest/video1.mp4", help="Path to the first video (default: dataset/PlanTest/video1.mp4)")
    parser.add_argument("video2", nargs="?", default="dataset/PlanTest/video2.mp4", help="Path to the second video (default: dataset/PlanTest/video2.mp4)")
    parser.add_argument("--output", default="alignment_result.png", help="Path to save the alignment plot")

    args = parser.parse_args()

    # Check if default paths exist if no arguments provided
    if not os.path.exists(args.video1):
        # Fallback to current directory videos if dataset not found, or raise error
        print(f"Error: Video file not found: {args.video1}")
        print("Please provide video paths as arguments: python -m new_method.run_alignment path/to/video1.mp4 path/to/video2.mp4")
        return
    if not os.path.exists(args.video2):
        print(f"Error: Video file not found: {args.video2}")
        return

    print(f"Processing: {args.video1} and {args.video2}")

    print("Extracting features from Video 1...")
    extractor = VideoFeatureExtractor()
    features1 = extractor.extract_features(args.video1)
    print(f"Features 1 shape: {features1.shape}")

    print("Extracting features from Video 2...")
    features2 = extractor.extract_features(args.video2)
    print(f"Features 2 shape: {features2.shape}")

    print("Computing DTW alignment...")

    # Z-score normalization to ensure features are comparable
    f1_mean = np.mean(features1, axis=0)
    f1_std = np.std(features1, axis=0) + 1e-6
    features1_norm = (features1 - f1_mean) / f1_std

    f2_mean = np.mean(features2, axis=0)
    f2_std = np.std(features2, axis=0) + 1e-6
    features2_norm = (features2 - f2_mean) / f2_std

    path, cost_matrix = compute_dtw(features1_norm, features2_norm)

    print(f"Alignment complete. Path length: {len(path)}")
    print(f"Final cost: {cost_matrix[-1, -1]}")

    # Visualization
    # Unzip path into x and y coordinates
    path_x, path_y = zip(*path)

    plt.figure(figsize=(10, 8))
    # Display cost matrix. Transpose to match X-axis=Video1, Y-axis=Video2
    plt.imshow(cost_matrix.T, origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')
    plt.plot(path_x, path_y, 'r-', linewidth=2, label='Alignment Path')
    plt.colorbar(label='Accumulated Cost')
    plt.xlabel('Video 1 Frame Index')
    plt.ylabel('Video 2 Frame Index')
    plt.title(f'DTW Alignment Cost Matrix\n{os.path.basename(args.video1)} vs {os.path.basename(args.video2)}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Alignment plot saved to {args.output}")

if __name__ == "__main__":
    main()
