import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from new_method.feature_extraction import VideoFeatureExtractor
from new_method.dtw_alignment import compute_dtw
from feature_matching.src.alignment.core import ALGORITHMS

def get_dtw_path(video1_path, video2_path, sample_rate_dtw=15, penalty=1.5):
    """Computes the global DTW path using optical flow features."""
    extractor = VideoFeatureExtractor(sample_rate=sample_rate_dtw)

    # Extract features
    features1, indices1 = extractor.extract_features(video1_path)
    features2, indices2 = extractor.extract_features(video2_path)

    # Z-score normalization
    f1_mean = np.mean(features1, axis=0)
    f1_std = np.std(features1, axis=0) + 1e-6
    features1_norm = (features1 - f1_mean) / f1_std

    f2_mean = np.mean(features2, axis=0)
    f2_std = np.std(features2, axis=0) + 1e-6
    features2_norm = (features2 - f2_mean) / f2_std

    path, cost_matrix = compute_dtw(features1_norm, features2_norm, step_penalty=penalty)

    # Map DTW path back to original frame indices
    real_path = [(indices1[i], indices2[j]) for i, j in path]
    return real_path

def align_coarse_to_fine(video1_path, video2_path, algo_name="ORB", sample_rate=15, window_size=15):
    """
    Combines Fast DTW with Keypoint matching.
    1. Uses DTW on Optical Flow features to get a global 'coarse' alignment path.
    2. For each sampled frame in Video 1, uses the predicted frame from Video 2 as a center point,
       and searches locally (+/- window_size) using exact Keypoint Matching (ORB/BRISK/AKAZE)
       and RANSAC for a 'fine' pixel-perfect alignment.
    """
    print(f"Aligning {video1_path} and {video2_path} using Coarse-to-Fine ({algo_name})...")

    if algo_name not in ALGORITHMS:
        print(f"Error: Unknown algorithm {algo_name}")
        return []

    algo_module = ALGORITHMS[algo_name]

    # Step 1: Coarse Global Alignment with DTW
    print("  > Computing global coarse alignment via DTW...")
    dtw_path = get_dtw_path(video1_path, video2_path, sample_rate_dtw=sample_rate)

    # Create a lookup table for coarse predictions: v1_frame -> v2_frame prediction
    # If multiple v2 frames map to one v1 frame (stuttering), take the median or first
    coarse_map = {}
    for v1, v2 in dtw_path:
        if v1 not in coarse_map:
            coarse_map[v1] = []
        coarse_map[v1].append(v2)

    for k in coarse_map:
        coarse_map[k] = int(np.median(coarse_map[k]))

    # Step 2: Fine Local Alignment using Feature Matching
    print(f"  > Refining alignment locally using {algo_name}...")
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos.")
        return []

    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    results = []

    for v1_idx in sorted(coarse_map.keys()):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, v1_idx)
        ret1, frame1 = cap1.read()
        if not ret1:
            continue

        kp1, des1 = algo_module.compute_features(frame1)
        if des1 is None or len(kp1) < 10:
            continue

        predicted_v2_idx = coarse_map[v1_idx]

        best_match_score = -1
        best_v2_idx = -1

        # Search locally around predicted frame
        search_start = max(0, predicted_v2_idx - window_size)
        search_end = predicted_v2_idx + window_size

        current_search_idx = search_start
        while current_search_idx <= search_end:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, current_search_idx)
            ret2, frame2 = cap2.read()
            if not ret2:
                break

            kp2, des2 = algo_module.compute_features(frame2)
            if des2 is not None and len(kp2) >= 10:
                knn_matches = bf.knnMatch(des1, des2, k=2)

                good_matches = []
                for match_pair in knn_matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                    elif len(match_pair) == 1:
                        m = match_pair[0]
                        if m.distance < 50:
                            good_matches.append(m)

                if len(good_matches) >= 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                                ransacReprojThreshold=2.5,
                                                maxIters=2000,
                                                confidence=0.999)
                    if mask is not None:
                        inlier_count = int(np.sum(mask))

                        # Scale verification
                        if M is not None and inlier_count > 0:
                            try:
                                scale_x = np.sqrt(M[0,0]**2 + M[1,0]**2)
                                scale_y = np.sqrt(M[0,1]**2 + M[1,1]**2)
                                if scale_x < 0.7 or scale_x > 1.3 or scale_y < 0.7 or scale_y > 1.3:
                                    inlier_count = int(inlier_count * 0.5)
                            except:
                                inlier_count = int(inlier_count * 0.7)

                        if inlier_count > best_match_score:
                            best_match_score = inlier_count
                            best_v2_idx = current_search_idx

            # Use step size of 1 for precision during local search, or higher for speed
            # Since window is small (e.g. 30 frames total), step 1 is fine.
            current_search_idx += 1

        if best_v2_idx != -1 and best_match_score >= 4:
            print(f"V1 {v1_idx} -> V2 {best_v2_idx} (Score: {best_match_score}) - DTW Predicted: {predicted_v2_idx}")
            results.append({
                "v1_frame": v1_idx,
                "v2_frame": best_v2_idx,
                "score": best_match_score,
                "algorithm": f"CoarseFine_{algo_name}"
            })

    cap1.release()
    cap2.release()
    return results

def align_fusion(video1_path, video2_path, algo_name="ORB", sample_rate=15, penalty=1.5):
    """
    Combines Fast DTW with Keypoint matching using a fused distance matrix.
    1. Extracts Optical Flow features.
    2. Extracts Keypoints for sampled frames.
    3. Computes a fused cost matrix: distance = w1 * (optical_flow_dist) + w2 * (1 / (1 + keypoint_inliers)).
    4. Runs DTW on the fused matrix.
    """
    print(f"Aligning {video1_path} and {video2_path} using Fusion ({algo_name})...")

    if algo_name not in ALGORITHMS:
        print(f"Error: Unknown algorithm {algo_name}")
        return []

    algo_module = ALGORITHMS[algo_name]

    # Step 1: Extract Optical Flow Features
    print("  > Extracting optical flow features...")
    extractor = VideoFeatureExtractor(sample_rate=sample_rate)

    features1, indices1 = extractor.extract_features(video1_path)
    features2, indices2 = extractor.extract_features(video2_path)

    f1_mean = np.mean(features1, axis=0)
    f1_std = np.std(features1, axis=0) + 1e-6
    features1_norm = (features1 - f1_mean) / f1_std

    f2_mean = np.mean(features2, axis=0)
    f2_std = np.std(features2, axis=0) + 1e-6
    features2_norm = (features2 - f2_mean) / f2_std

    # Compute Euclidean distance matrix for optical flow
    from scipy.spatial.distance import cdist
    dist_matrix_flow = cdist(features1_norm, features2_norm, metric='euclidean')

    # Step 2: Extract Keypoints
    print("  > Extracting keypoints for sampled frames...")

    def extract_all_keypoints(video_path, indices):
        cap = cv2.VideoCapture(video_path)
        kp_dict = {}
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            kp, des = algo_module.compute_features(frame)
            kp_dict[idx] = (kp, des)
        cap.release()
        return kp_dict

    kp_dict1 = extract_all_keypoints(video1_path, indices1)
    kp_dict2 = extract_all_keypoints(video2_path, indices2)

    # Step 3: Compute Keypoint Distance Matrix
    print("  > Computing keypoint matches (this may take a while)...")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    n, m = len(indices1), len(indices2)
    dist_matrix_kp = np.full((n, m), 10.0) # Default high penalty for no matches

    # Optimization: only compute keypoint matches where optical flow distance is relatively low
    # to save computation time (e.g. bottom 20% of distances per row)
    for i in range(n):
        idx1 = indices1[i]
        if idx1 not in kp_dict1:
            continue
        kp1, des1 = kp_dict1[idx1]
        if des1 is None or len(kp1) < 10:
            continue

        # Get threshold for top candidates based on flow distance
        flow_row = dist_matrix_flow[i]
        threshold = np.percentile(flow_row, 30) # Test top 30% candidates

        for j in range(m):
            if flow_row[j] > threshold:
                continue # Skip bad candidates to save time

            idx2 = indices2[j]
            if idx2 not in kp_dict2:
                continue
            kp2, des2 = kp_dict2[idx2]

            if des2 is not None and len(kp2) >= 10:
                knn_matches = bf.knnMatch(des1, des2, k=2)
                good_matches = []
                for match_pair in knn_matches:
                    if len(match_pair) == 2:
                        m_match, n_match = match_pair
                        if m_match.distance < 0.75 * n_match.distance:
                            good_matches.append(m_match)
                    elif len(match_pair) == 1:
                        m_match = match_pair[0]
                        if m_match.distance < 50:
                            good_matches.append(m_match)

                inlier_count = 0
                if len(good_matches) >= 4:
                    src_pts = np.float32([kp1[m_match.queryIdx].pt for m_match in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m_match.trainIdx].pt for m_match in good_matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                                ransacReprojThreshold=2.5,
                                                maxIters=500, # Lower iterations for speed in fusion
                                                confidence=0.99)
                    if mask is not None:
                        inlier_count = int(np.sum(mask))

                # Convert inliers to a distance metric (more inliers -> lower distance)
                # Max expected inliers usually around 100-200. Let's cap at 100.
                dist_matrix_kp[i, j] = max(0.0, 10.0 - (inlier_count / 10.0))

    # Normalize matrices to ensure they have similar scale before fusion
    dist_matrix_flow = dist_matrix_flow / (np.max(dist_matrix_flow) + 1e-6)
    dist_matrix_kp = dist_matrix_kp / (np.max(dist_matrix_kp) + 1e-6)

    # Fused Matrix
    w_flow = 0.4
    w_kp = 0.6
    fused_matrix = w_flow * dist_matrix_flow + w_kp * dist_matrix_kp

    print("  > Running DTW on fused cost matrix...")
    avg_dist = np.mean(fused_matrix)
    dyn_penalty = avg_dist * penalty

    acc_cost = np.zeros((n, m))
    acc_cost[0, 0] = fused_matrix[0, 0]

    for i in range(1, n):
        acc_cost[i, 0] = acc_cost[i-1, 0] + fused_matrix[i, 0] + dyn_penalty
    for j in range(1, m):
        acc_cost[0, j] = acc_cost[0, j-1] + fused_matrix[0, j] + dyn_penalty

    for i in range(1, n):
        for j in range(1, m):
            cost_insertion = acc_cost[i-1, j] + dyn_penalty
            cost_deletion = acc_cost[i, j-1] + dyn_penalty
            cost_match = acc_cost[i-1, j-1]
            acc_cost[i, j] = fused_matrix[i, j] + min(cost_insertion, cost_deletion, cost_match)

    path = []
    i, j = n-1, m-1
    path.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            cost_insertion = acc_cost[i-1, j] + dyn_penalty
            cost_deletion = acc_cost[i, j-1] + dyn_penalty
            cost_match = acc_cost[i-1, j-1]

            min_val = min(cost_insertion, cost_deletion, cost_match)
            if cost_match == min_val:
                i -= 1
                j -= 1
            elif cost_insertion == min_val:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    dtw_path = path[::-1]

    # Format results
    results = []
    for idx_tuple in dtw_path:
        i, j = idx_tuple
        v1_idx = indices1[i]
        v2_idx = indices2[j]

        score = 1.0 / (1.0 + fused_matrix[i, j]) # Higher score is better

        results.append({
            "v1_frame": v1_idx,
            "v2_frame": v2_idx,
            "score": score,
            "algorithm": f"Fusion_{algo_name}"
        })

    return results
