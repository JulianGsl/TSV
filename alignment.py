"""
Video Alignment Module
This module handles the logic for aligning two videos using feature matching.
"""

import cv2
import numpy as np
import os

# Import our algorithms
import orb
import brisk
import akaze

ALGORITHMS = {
    "ORB": orb,
    "BRISK": brisk,
    "AKAZE": akaze
}

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    props = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    cap.release()
    return props

def align_videos(video1_path, video2_path, algo_name="ORB", sample_rate=1, search_window=150):
    """
    Aligns two videos by finding corresponding frames.

    Args:
        video1_path (str): Path to the first video (reference).
        video2_path (str): Path to the second video.
        algo_name (str): The algorithm to use ("ORB", "BRISK", "AKAZE").
        sample_rate (int): Process every Nth frame of video1.
        search_window (int): Number of frames to search in video2 ahead of the last match.

    Returns:
        list: A list of matches dictionaries.
    """
    print(f"Aligning {video1_path} and {video2_path} using {algo_name}...")

    if algo_name not in ALGORITHMS:
        print(f"Error: Unknown algorithm {algo_name}")
        return []

    algo_module = ALGORITHMS[algo_name]

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos.")
        return []

    # Initialize Matcher
    # Use Hamming distance for binary descriptors (ORB, BRISK, AKAZE)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    results = []

    # State for Video 2 search
    v2_current_index = 0
    v2_frame_cache = {} # Cache frames or features if needed?
                        # For now, just seek. Seeking might be slow but simple.

    # Pre-load video 2 frames? No, too much memory.
    # We will assume monotonic alignment.
    # For a frame V1[i], we search V2[j] where j is close to last_j.

    last_best_v2_frame = 0

    # Loop through Video 1
    v1_frame_idx = 0

    # We'll read Video 1 sequentially
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break

        if v1_frame_idx % sample_rate != 0:
            v1_frame_idx += 1
            continue

        # Compute features for Frame 1
        kp1, des1 = algo_module.compute_features(frame1)

        if des1 is None or len(kp1) < 10:
            print(f"Frame {v1_frame_idx}: Not enough features in Video 1.")
            v1_frame_idx += 1
            continue

        # Search in Video 2
        # We start searching from `last_best_v2_frame`
        # But we might need to look slightly behind if we made a mistake?
        # For simplicity: search from last_best_v2_frame to last_best_v2_frame + search_window

        best_match_score = -1
        best_v2_idx = -1

        # Seek to start of window
        cap2.set(cv2.CAP_PROP_POS_FRAMES, last_best_v2_frame)

        current_search_idx = last_best_v2_frame

        frames_scanned = 0
        while frames_scanned < search_window:
            ret2, frame2 = cap2.read()
            if not ret2:
                break

            # Compute features for Frame 2
            kp2, des2 = algo_module.compute_features(frame2)

            if des2 is not None and len(kp2) >= 10:
                # Match
                matches = bf.match(des1, des2)

                # Simple metric: number of matches
                # Better metric: sort by distance, take top N, sum distances?
                # Or just count matches with distance < threshold?
                # crossCheck=True returns only consistent matches.
                # So len(matches) is a good proxy for similarity.

                score = len(matches)

                if score > best_match_score:
                    best_match_score = score
                    best_v2_idx = current_search_idx

            current_search_idx += 1
            frames_scanned += 1

        # Record result
        if best_v2_idx != -1:
            print(f"V1 Frame {v1_frame_idx} matches V2 Frame {best_v2_idx} (Score: {best_match_score})")
            results.append({
                "v1_frame": v1_frame_idx,
                "v2_frame": best_v2_idx,
                "score": best_match_score,
                "algorithm": algo_name
            })

            # Update the start point for next search
            # We assume we won't go backward.
            # But we shouldn't jump too far if it's a false positive?
            # Let's trust the max score for now.
            last_best_v2_frame = best_v2_idx
        else:
            print(f"Frame {v1_frame_idx}: No good match found in search window.")

        v1_frame_idx += 1

    cap1.release()
    cap2.release()

    return results

if __name__ == "__main__":
    # Test stub
    pass
