import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from new_method.feature_extraction import VideoFeatureExtractor
from new_method.dtw_alignment import compute_dtw

def create_aligned_video(video1_path, video2_path, path, output_video_path):
    """
    Creates a side-by-side video of the two videos aligned according to the DTW path.

    Args:
        video1_path (str): Path to the first video.
        video2_path (str): Path to the second video.
        path (list of tuples): The DTW alignment path [(idx1, idx2), ...].
        output_video_path (str): Path to save the output video.
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open source videos for alignment generation.")
        return

    # Get video properties
    fps = cap1.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Target height for side-by-side (use max height)
    target_height = max(height1, height2)

    # Resize width proportionally
    target_width1 = int(width1 * (target_height / height1))
    target_width2 = int(width2 * (target_height / height2))

    total_width = target_width1 + target_width2

    # Using 'avc1' (H.264) for QuickTime compatibility on Mac.
    # Note: 'mp4v' was used previously, but QuickTime on macOS drops support for older 'mp4v' codecs.
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (total_width, target_height))

    # Check if VideoWriter opened successfully. Sometimes 'avc1' requires openh264 in some envs.
    if not out.isOpened():
        print("Warning: avc1 codec failed to initialize. Falling back to mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (total_width, target_height))

    print(f"Generating aligned video: {output_video_path}")
    print(f"Total frames: {len(path)}")

    # Cache frames if video is small enough, otherwise seek (slower)
    # For efficiency with large videos and random access, seeking might be slow.
    # However, DTW path is usually somewhat monotonic, so seeking forward is okay.
    # But seeking backward is very slow.
    # Since DTW path is monotonic in time (indices increase), we can just iterate.
    # BUT, DTW can map one frame to multiple (stutter).

    # Optimization: Read all frames into memory if feasible?
    # No, videos can be large.
    # Better approach: Keep track of current frame index and read next frame when needed.

    curr_idx1 = -1
    curr_idx2 = -1
    frame1 = None
    frame2 = None

    # Pre-read check: The path is ordered.
    # (0,0), (1,0), (2,1)...

    for i, (idx1, idx2) in enumerate(path):
        if i % 50 == 0:
            print(f"Processing frame {i}/{len(path)}...", end='\r')

        # Get Frame 1
        if idx1 != curr_idx1:
            if idx1 == curr_idx1 + 1:
                ret, frame1 = cap1.read()
            else:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, idx1)
                ret, frame1 = cap1.read()
            curr_idx1 = idx1

            if not ret:
                frame1 = np.zeros((height1, width1, 3), dtype=np.uint8) # Blank if failed

        # Get Frame 2
        if idx2 != curr_idx2:
            if idx2 == curr_idx2 + 1:
                ret, frame2 = cap2.read()
            else:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)
                ret, frame2 = cap2.read()
            curr_idx2 = idx2

            if not ret:
                frame2 = np.zeros((height2, width2, 3), dtype=np.uint8)

        if frame1 is None or frame2 is None:
            continue

        # Resize to target height
        f1_resized = cv2.resize(frame1, (target_width1, target_height))
        f2_resized = cv2.resize(frame2, (target_width2, target_height))

        # Concatenate
        combined = np.hstack((f1_resized, f2_resized))

        # Add frame indices text
        cv2.putText(combined, f"Frame {idx1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, f"Frame {idx2}", (target_width1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(combined)

    print(f"\nVideo saved to {output_video_path}")
    cap1.release()
    cap2.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="Video Temporal Alignment using DTW")
    parser.add_argument("video1", nargs="?", default="dataset/PlanTest/video1.mp4", help="Path to the first video (default: dataset/PlanTest/video1.mp4)")
    parser.add_argument("video2", nargs="?", default="dataset/PlanTest/video2.mp4", help="Path to the second video (default: dataset/PlanTest/video2.mp4)")
    parser.add_argument("--output", default="alignment_result.png", help="Path to save the alignment plot")
    parser.add_argument("--video-output", default="aligned_video.mp4", help="Path to save the aligned side-by-side video")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")

    args = parser.parse_args()

    # Check if default paths exist if no arguments provided
    if not os.path.exists(args.video1):
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

    if not args.no_video:
        create_aligned_video(args.video1, args.video2, path, args.video_output)

if __name__ == "__main__":
    main()
