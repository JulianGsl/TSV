"""
Video Alignment Module
This module handles the logic for aligning two videos using feature matching.
"""

import cv2
import numpy as np
import sys
import os

# Import our algorithms
from .algorithms import orb
from .algorithms import brisk
from .algorithms import akaze

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

    last_best_v2_frame = 0

    # We estimate velocity to predict the center of the search window
    # Initial velocity = 1.0 (assuming same speed)
    estimated_velocity = 1.0

    # Loop through Video 1
    v1_frame_idx = 0

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
            # print(f"Frame {v1_frame_idx}: Not enough features in Video 1.")
            v1_frame_idx += 1
            continue

        # Dynamic Search Window
        # Instead of just searching from `last_best_v2_frame`, we predict where we *should* be.
        # Predicted V2 = Last V2 + (Delta V1 * Velocity)

        if len(results) > 1:
            # Update velocity based on last few matches?
            # Simple approach: just use the last confirmed match
            # But we want to be robust.
            pass

        # Center the window around the expected position
        # We moved `sample_rate` frames in V1.
        # predicted_move = int(sample_rate * estimated_velocity) # Unused for now

        search_start = max(0, last_best_v2_frame) # Don't go back too much?
        # Actually, let's strictly go forward from last known good position?
        # Or allow a small backward look in case of jitter?
        # Let's say we look from `last_best` to `last_best + search_window`.

        # Robustness: Look at TOP N matches, not just the single best.
        # But here we are comparing Frame1 against MANY Frame2 candidates.
        # We want the Frame2 that has the most matches with Frame1.

        best_match_score = -1
        best_v2_idx = -1

        # Seek to start of search
        cap2.set(cv2.CAP_PROP_POS_FRAMES, search_start)

        current_search_idx = search_start
        frames_scanned = 0

        candidates = []

        while frames_scanned < search_window:
            ret2, frame2 = cap2.read()
            if not ret2:
                break

            # Compute features for Frame 2
            kp2, des2 = algo_module.compute_features(frame2)

            if des2 is not None and len(kp2) >= 10:
                matches = bf.match(des1, des2)
                score = len(matches)

                # Heuristic: Penalize distance from expected position?
                # This helps if we have multiple similar frames (e.g. repeated patterns)
                # We prefer the one closer to where we expect to be.
                # For now, let's just use raw score.

                if score > 10: # Minimum threshold
                    candidates.append((current_search_idx, score))

            current_search_idx += 1
            frames_scanned += 1

        # Select best candidate
        if candidates:
            # Sort by score descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_v2_idx, best_match_score = candidates[0]

            # Simple velocity update?
            # if best_v2_idx > last_best_v2_frame:
            #     current_vel = (best_v2_idx - last_best_v2_frame) / sample_rate
            #     estimated_velocity = 0.9 * estimated_velocity + 0.1 * current_vel

            print(f"V1 {v1_frame_idx} -> V2 {best_v2_idx} (Score: {best_match_score})")

            results.append({
                "v1_frame": v1_frame_idx,
                "v2_frame": best_v2_idx,
                "score": best_match_score,
                "algorithm": algo_name
            })

            last_best_v2_frame = best_v2_idx
        else:
            # print(f"Frame {v1_frame_idx}: No good match found.")
            pass

        v1_frame_idx += 1

    cap1.release()
    cap2.release()

    return results
