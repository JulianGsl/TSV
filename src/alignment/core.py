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

def align_videos(video1_path, video2_path, algo_name="ORB", sample_rate=1, search_window=150, search_step=1):
    """
    Aligns two videos by finding corresponding frames.

    Args:
        video1_path (str): Path to the first video (reference).
        video2_path (str): The path to the second video.
        algo_name (str): The algorithm to use ("ORB", "BRISK", "AKAZE").
        sample_rate (int): Process every Nth frame of video1.
        search_window (int): Number of frames to search in video2 ahead of the last match.
        search_step (int): Step size between candidate frames in video2 when searching (e.g., 10 to test 0,10,20...)

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

    results = []

    last_best_v2_frame = 0
    last_matched_v1_frame = 0  # Track last V1 frame that had a match

    # We estimate velocity to predict the center of the search window
    # Initial velocity = 1.0 (assuming same speed)
    # Can be < 1 if video2 is faster, > 1 if video2 is slower
    estimated_velocity = 1.0
    velocity_confidence = 0.0  # Confidence in velocity estimate (0-1)
    
    # Kalman filter state for velocity estimation
    # state: [velocity, velocity_change_rate]
    velocity_variance = 1.0  # Initial uncertainty in velocity estimate
    process_noise = 0.01  # How much we expect velocity to change (Q)
    measurement_noise = 0.1  # Uncertainty in velocity measurements (R)

    # Track consecutive failures to expand search
    consecutive_failures = 0

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

        # Dynamic Search Window with Velocity Prediction
        # Update velocity estimate based on recent matches using Kalman filter
        if len(results) >= 2:
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
                # Measure current velocity (median for robustness against outliers)
                measured_velocity = np.median(velocities)
                velocity_std = np.std(velocities) if len(velocities) > 1 else 0.5
                
                # Kalman filter update
                # Prediction step (we assume velocity stays constant)
                predicted_variance = velocity_variance + process_noise
                
                # Update step: blend prediction with measurement
                # Kalman gain: how much to trust the measurement vs prediction
                kalman_gain = predicted_variance / (predicted_variance + measurement_noise + velocity_std**2)
                
                # Update velocity estimate
                estimated_velocity = estimated_velocity + kalman_gain * (measured_velocity - estimated_velocity)
                
                # Update variance (uncertainty decreases with each measurement)
                velocity_variance = (1 - kalman_gain) * predicted_variance
                
                # Clamp velocity to reasonable bounds to prevent runaway drift
                # Trains typically have similar speeds (0.5x to 2.0x speed ratio)
                estimated_velocity = max(0.5, min(2.0, estimated_velocity))
                
                # Update velocity confidence based on:
                # 1. Low variance (high certainty in estimate)
                # 2. Consistent measurements (low std)
                # 3. Reasonable velocity range
                variance_confidence = max(0.0, min(1.0, 1.0 - velocity_variance))
                consistency_confidence = max(0.0, min(1.0, 1.0 - velocity_std))
                velocity_confidence = 0.7 * variance_confidence + 0.3 * consistency_confidence
                
                # Periodic drift correction: reset if we detect systematic drift
                # Check if recent velocities are consistently different from estimate
                if len(velocities) >= 3:
                    recent_bias = measured_velocity - estimated_velocity
                    # If bias is significant and consistent, recalibrate
                    if abs(recent_bias) > 0.2 and velocity_std < 0.1:
                        # Strong evidence of systematic drift, reset with higher process noise
                        velocity_variance = min(velocity_variance + 0.5, 1.0)
                        process_noise = min(process_noise * 1.5, 0.1)  # Increase adaptability temporarily

        # Adaptive search window based on velocity confidence
        # Higher confidence = smaller window, faster processing
        adaptive_window = search_window
        if velocity_confidence > 0.7:
            adaptive_window = int(search_window * 0.7)
        elif velocity_confidence < 0.3:
            adaptive_window = int(search_window * 1.3)

        # Expand search window on consecutive failures
        if consecutive_failures > 0:
            expansion_factor = 1.0 + (consecutive_failures * 0.5)
            adaptive_window = int(adaptive_window * min(expansion_factor, 3.0))

        # Predict center of search window based on frames elapsed since last match
        frames_since_last_match = v1_frame_idx - last_matched_v1_frame
        predicted_v2_frame = last_best_v2_frame + int(frames_since_last_match * estimated_velocity)

        # Allow backward search but don't go below last matched position (monotonicity)
        # Use smaller backward margin to encourage forward progress
        backward_margin = max(5, int(adaptive_window * 0.05))
        search_start = max(last_best_v2_frame, predicted_v2_frame - backward_margin)

        # Search ahead from predicted position
        search_end = search_start + adaptive_window

        # Robustness: Use ratio test and RANSAC for better matching quality
        # Compare Frame1 against MANY Frame2 candidates to find best match

        best_match_score = -1
        best_v2_idx = -1

        # Seek to start of search
        current_search_idx = search_start

        raw_candidates = []

        # Iterate through candidate frames in video2 with given step (e.g., every 10 frames)
        while current_search_idx < search_end:
            # Seek to the candidate frame index explicitly before reading (handles stepping)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, current_search_idx)
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
                # Lowered minimum threshold to 5 to accommodate sparse features (e.g. AKAZE)
                min_matches = max(5, min(10, len(kp1) // 20))
                if len(good_matches) >= min_matches:
                    raw_candidates.append({
                        "idx": current_search_idx,
                        "matches": good_matches,
                        "kp2": kp2
                    })

            # Advance by the configured step (allows sparse candidate sampling)
            current_search_idx += search_step

        # Process candidates with Branch and Bound optimization
        # We want to find the candidate with max (inliers - penalty).
        # We visit candidates closest to expected_pos first.
        # We skip RANSAC if (raw_matches - penalty) <= current_best_score.

        expected_pos = predicted_v2_frame

        # Sort by distance from expected position (closest first)
        raw_candidates.sort(key=lambda x: abs(x["idx"] - expected_pos))

        best_weighted_score = -float('inf')

        # Also keep track of the raw match score for logging
        best_match_raw_score = 0
        
        # Adaptive penalty based on velocity confidence
        # Higher confidence = higher penalty (more strict about position)
        # Lower confidence = lower penalty (more flexible search)
        # Penalty factor ranges from 0.5 (low confidence) to 2.0 (high confidence)
        # Reduced from previous values to be less conservative
        base_penalty = 1.0
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
                # - ransacReprojThreshold: 2.5 (tighter than before for rail tracks)
                # - maxIters: 2000 (increased for better reliability)
                # - confidence: 0.999 (very high confidence)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                            ransacReprojThreshold=2.5,
                                            maxIters=2000,
                                            confidence=0.999)
                if mask is not None:
                    raw_inlier_count = int(np.sum(mask))
                    inlier_count = raw_inlier_count
                    
                    # Additional quality check: verify homography is reasonable
                    # For rail track videos, we expect mostly translation with minimal rotation/scaling
                    if M is not None and inlier_count > 0:
                        # Check that transformation is not too extreme
                        # Decompose to check if scale and rotation are reasonable
                        try:
                            # Extract scale from homography
                            scale_x = np.sqrt(M[0,0]**2 + M[1,0]**2)
                            scale_y = np.sqrt(M[0,1]**2 + M[1,1]**2)

                            # Extract rotation roughly (assuming small rotation)
                            rotation_rad = np.arctan2(M[1,0], M[0,0])
                            rotation_deg = np.degrees(rotation_rad)

                            # For rail tracks, scale should be close to 1.0 and rotation small
                            # Relaxed constraints to handle curves and speed differences better
                            if scale_x < 0.6 or scale_x > 1.5 or scale_y < 0.6 or scale_y > 1.5:
                                # Suspicious scale, reduce confidence slightly
                                inlier_count = int(inlier_count * 0.8)

                            if abs(rotation_deg) > 20.0:
                                # Suspicious rotation (> 20 degrees), reduce confidence
                                # Relaxed from 10 deg to 20 deg to account for curved tracks
                                inlier_count = int(inlier_count * 0.5)

                        except (ValueError, ZeroDivisionError, IndexError):
                            # If decomposition fails, reduce confidence
                            inlier_count = int(inlier_count * 0.8)

            weighted = inlier_count - penalty

            if weighted > best_weighted_score:
                best_weighted_score = weighted
                best_v2_idx = idx
                best_match_raw_score = raw_inlier_count # Store the true raw inlier count

        # Adaptive threshold based on match quality and velocity confidence
        # Higher velocity confidence = stricter threshold (more negative, harder to pass)
        # Lower confidence = more lenient threshold (less negative, easier to pass)
        acceptance_threshold = -100 - (velocity_confidence * 50)
        
        # Require at least 4 inliers to consider it a valid geometric match
        # Use best_match_raw_score (unpenalized) for the hard gate
        if best_v2_idx != -1 and best_weighted_score > acceptance_threshold and best_match_raw_score >= 4:
            best_match_score = best_match_raw_score

            print(f"V1 {v1_frame_idx} -> V2 {best_v2_idx} (Score: {best_match_score}, Weighted: {best_weighted_score:.1f}, Vel: {estimated_velocity:.2f}, Conf: {velocity_confidence:.2f})")

            results.append({
                "v1_frame": v1_frame_idx,
                "v2_frame": best_v2_idx,
                "score": best_match_score,
                "algorithm": algo_name
            })

            last_best_v2_frame = best_v2_idx
            last_matched_v1_frame = v1_frame_idx
            consecutive_failures = 0  # Reset on success
        else:
            # print(f"Frame {v1_frame_idx}: No good match found.")
            consecutive_failures += 1

        v1_frame_idx += 1

    cap1.release()
    cap2.release()

    return results
