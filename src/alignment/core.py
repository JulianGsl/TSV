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

def align_videos(video1_path, video2_path, algo_name="ORB", sample_rate=1, search_window=150, use_velocity=True):
    """
    Aligns two videos by finding corresponding frames.

    Args:
        video1_path (str): Path to the first video (reference).
        video2_path (str): Path to the second video.
        algo_name (str): The algorithm to use ("ORB", "BRISK", "AKAZE").
        sample_rate (int): Process every Nth frame of video1.
        search_window (int): Number of frames to search in video2 ahead of the last match.
        use_velocity (bool): If True, uses velocity estimation to predict search window.
                             If False, uses a fixed forward window from the last match.

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
    # crossCheck=False to enable ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Force Frame 0 -> Frame 0 match as per requirements
    results = [{
        "v1_frame": 0,
        "v2_frame": 0,
        "score": 100,
        "algorithm": algo_name,
        "kp1": [],
        "kp2": [],
        "matches": []
    }]

    last_best_v2_frame = 0

    # We estimate velocity to predict the center of the search window
    # Initial velocity = 1.0 (assuming same speed)
    estimated_velocity = 1.0
    velocity_confidence = 0.0  # Confidence in velocity estimate (0-1)

    # Loop through Video 1
    # Start at 0, but since we forced 0, we can technically skip it if sample_rate aligns,
    # but the loop logic below checks modulo.
    # If sample_rate=1, loop 0 is processed. We should avoid duplicating.
    # Let's start v1_frame_idx at 0, but skip if we already have it?
    # No, cleaner to just let the loop run but maybe skip 0 explicitly if present?
    # Actually, let's start after 0.
    v1_frame_idx = 0

    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break

        # Skip frame 0 as we forced it
        if v1_frame_idx == 0:
            v1_frame_idx += 1
            continue

        if v1_frame_idx % sample_rate != 0:
            v1_frame_idx += 1
            continue

        # Compute features for Frame 1
        kp1, des1 = algo_module.compute_features(frame1)

        if des1 is None or len(kp1) < 10:
            # print(f"Frame {v1_frame_idx}: Not enough features in Video 1.")
            v1_frame_idx += 1
            continue

        # Update velocity estimate if enabled
        if use_velocity and len(results) >= 2:
            # Calculate velocity from last few matches
            recent_window = min(5, len(results))
            recent_results = results[-recent_window:]
            velocities = []
            for i in range(1, len(recent_results)):
                dv1 = recent_results[i]['v1_frame'] - recent_results[i-1]['v1_frame']
                dv2 = recent_results[i]['v2_frame'] - recent_results[i-1]['v2_frame']
                if dv1 > 0:
                    velocities.append(dv2 / dv1)
            
            if velocities:
                # Use exponential moving average with recent velocity
                new_velocity = np.mean(velocities)
                estimated_velocity = 0.7 * estimated_velocity + 0.3 * new_velocity
                
                # Update velocity confidence based on consistency
                velocity_std = np.std(velocities) if len(velocities) > 1 else 0.5
                mean_velocity = np.mean(velocities)
                if mean_velocity > 0:
                    coeff_of_variation = velocity_std / mean_velocity
                    velocity_confidence = max(0.0, min(1.0, 1.0 - coeff_of_variation))
                else:
                    velocity_confidence = 0.0

        # Define Search Window
        if use_velocity:
            # Adaptive search window based on velocity confidence
            adaptive_window = search_window
            if velocity_confidence > 0.7:
                adaptive_window = int(search_window * 0.7)
            elif velocity_confidence < 0.3:
                adaptive_window = int(search_window * 1.3)

            # Predict center of search window
            predicted_v2_frame = last_best_v2_frame + int(sample_rate * estimated_velocity)

            # Allow absolute backward search for robustness
            backward_margin = max(15, int(adaptive_window * 0.1))
            search_start = max(0, predicted_v2_frame - backward_margin)
            search_end = search_start + adaptive_window
        else:
            # Fixed forward search from last match
            # "Search 102 (V1) in 125 to 135 (V2)" logic
            # Start slightly behind last match to allow for small errors
            backward_margin = 5
            search_start = max(0, last_best_v2_frame - backward_margin)
            search_end = search_start + search_window

        # Robustness: Use ratio test and RANSAC for better matching quality
        # Compare Frame1 against MANY Frame2 candidates to find best match

        best_match_score = -1
        best_v2_idx = -1
        best_candidate = None

        # Seek to start of search
        cap2.set(cv2.CAP_PROP_POS_FRAMES, search_start)

        current_search_idx = search_start

        raw_candidates = []

        while current_search_idx < search_end:
            ret2, frame2 = cap2.read()
            if not ret2:
                break

            # Compute features for Frame 2
            kp2, des2 = algo_module.compute_features(frame2)

            if des2 is not None and len(kp2) >= 10:
                # Use knnMatch for ratio test (Lowe's ratio test)
                # k=2 to get the two best matches for each descriptor
                knn_matches = bf.knnMatch(des1, des2, k=2)
                
                # Apply Lowe's ratio test to filter good matches
                good_matches = []
                for match_pair in knn_matches:
                    # Ensure we have two matches
                    if len(match_pair) == 2:
                        m, n = match_pair
                        # If best match is significantly better than second best
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                    elif len(match_pair) == 1:
                        # If only one match, check if distance is reasonable
                        m = match_pair[0]
                        if m.distance < 50:
                            good_matches.append(m)
                
                # Filter by match count - use adaptive threshold based on detected features
                min_matches = max(8, min(10, len(kp1) // 20))
                if len(good_matches) >= min_matches:
                    raw_candidates.append({
                        "idx": current_search_idx,
                        "matches": good_matches,
                        "kp2": kp2
                    })

            current_search_idx += 1

        # Process candidates with Branch and Bound optimization
        # We want to find the candidate with max (inliers - penalty).
        # We visit candidates closest to expected_pos first.
        # We skip RANSAC if (raw_matches - penalty) <= current_best_score.

        # If using velocity, bias towards predicted position.
        # If not, bias towards start of search window (closest to last match).
        if use_velocity:
            expected_pos = predicted_v2_frame
        else:
            expected_pos = search_start

        # Sort by distance from expected position (closest first)
        raw_candidates.sort(key=lambda x: abs(x["idx"] - expected_pos))

        best_weighted_score = -float('inf')

        # Also keep track of the raw match score for logging
        best_match_raw_score = 0
        
        # Adaptive penalty based on velocity confidence
        # Higher confidence = higher penalty (more strict about position)
        # Lower confidence = lower penalty (more flexible search)
        # Penalty factor ranges from 2.0 (low confidence) to 6.0 (high confidence)
        base_penalty = 4.0
        penalty_factor = base_penalty * (0.5 + velocity_confidence)

        for cand in raw_candidates:
            idx = cand["idx"]
            good_matches = cand["matches"]
            kp2 = cand["kp2"]

            dist = abs(idx - expected_pos)
            penalty = dist * penalty_factor

            # Upper bound: even if all good matches are inliers
            max_possible_score = len(good_matches) - penalty

            if max_possible_score <= best_weighted_score:
                continue

            # Run RANSAC with improved parameters
            inlier_count = 0
            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Improved RANSAC parameters:
                # - ransacReprojThreshold: 3.0 (tighter than default 5.0)
                # - maxIters: 1000 (balance between accuracy and performance)
                # - confidence: 0.995 (higher confidence)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                            ransacReprojThreshold=3.0,
                                            maxIters=1000,
                                            confidence=0.995)
                if mask is not None:
                    inlier_count = int(np.sum(mask))

            weighted = inlier_count - penalty

            if weighted > best_weighted_score:
                best_weighted_score = weighted
                best_v2_idx = idx
                best_match_raw_score = inlier_count # We return the raw score (inliers) for display
                best_candidate = cand

        # Adaptive threshold based on match quality and velocity confidence
        # Higher velocity confidence = stricter threshold (more negative, harder to pass)
        # Lower confidence = more lenient threshold (less negative, easier to pass)
        acceptance_threshold = -100 - (velocity_confidence * 50)
        
        if best_v2_idx != -1 and best_weighted_score > acceptance_threshold:
            best_match_score = best_match_raw_score

            print(f"V1 {v1_frame_idx} -> V2 {best_v2_idx} (Score: {best_match_score}, Weighted: {best_weighted_score:.1f}, Vel: {estimated_velocity:.2f}, Conf: {velocity_confidence:.2f})")

            results.append({
                "v1_frame": v1_frame_idx,
                "v2_frame": best_v2_idx,
                "score": best_match_score,
                "algorithm": algo_name,
                "kp1": kp1,
                "kp2": best_candidate["kp2"],
                "matches": best_candidate["matches"]
            })

            last_best_v2_frame = best_v2_idx
        else:
            # print(f"Frame {v1_frame_idx}: No good match found.")
            pass

        v1_frame_idx += 1

    cap1.release()
    cap2.release()

    return results
