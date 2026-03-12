"""
Combined Video Alignment - Improved Implementation

Strategy: DTW-Guided Feature Matching with Adaptive Fallback

The key insight is that each algorithm covers the other's weaknesses:
- DTW (optical flow): global alignment, robust to low-texture, finds overall structure
- Feature matching (ORB/BRISK/AKAZE + RANSAC): precise per-frame accuracy

Pipeline:
  Phase 1 – Coarse Global Alignment (DTW on Optical Flow):
    Extract optical flow features at a moderate sample rate and run DTW to obtain
    a globally optimal V1→V2 mapping. An interpolated prediction function is built
    so that any V1 frame index can be mapped to a predicted V2 frame index.

  Phase 2 – Fine Frame-by-Frame Alignment (Feature Matching):
    For each V1 frame, a blended prediction is computed from the DTW prediction and a
    Kalman-filter velocity estimate. Feature matching searches in a tight window around
    that blended prediction (rather than a large blind window). Branch-and-Bound RANSAC
    finds the best candidate efficiently.

  Phase 3 – Robust Fallback:
    When feature matching fails (score below threshold), the DTW prediction is used
    as the matched V2 frame (score=0). This guarantees full temporal coverage even
    in low-texture segments.
"""

import cv2
import numpy as np
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from new_method.feature_extraction import VideoFeatureExtractor
from new_method.dtw_alignment import compute_dtw
from feature_matching.src.alignment.core import ALGORITHMS


# ---------------------------------------------------------------------------
# Tuning constants shared across the combined alignment functions
# ---------------------------------------------------------------------------

# Exponential decay factor used in branch-and-bound scoring.
# Controls how quickly the score decays with distance from the predicted frame.
# Higher values = stricter temporal locality. Range roughly 0.05 – 0.3.
_BASE_PENALTY_FACTOR = 0.15

# Offset added to velocity_confidence when computing the decay rate.
# Ensures that even with zero confidence there is some temporal penalty.
_CONFIDENCE_OFFSET = 0.5

# Minimum inliers (before confidence weighting) to accept a feature match.
# Multiplied by (0.5 + velocity_confidence) at run-time.
_BASE_ACCEPTANCE_THRESHOLD = 4.0

# Hard bounds on the estimated V2/V1 velocity ratio.
# Prevents the Kalman filter from drifting to physically impossible values.
_MIN_VELOCITY = 0.5
_MAX_VELOCITY = 2.0

# Search window expansion per consecutive failure frame.
# e.g. after 2 failures the window grows to 1.8× its current size.
_FAILURE_EXPANSION_RATE = 0.4
_MAX_WINDOW_EXPANSION = 3.0  # cap the expansion multiplier

# Fraction of the adaptive window used as a backward search margin.
# Allows a small backward search to correct minor forward over-shoot.
_BACKWARD_MARGIN_RATIO = 0.1
_MIN_BACKWARD_MARGIN = 5  # frames

# How much the DTW weight is discounted by velocity_confidence.
# At full confidence (1.0), the effective DTW weight is halved.
_CONFIDENCE_DISCOUNT_FACTOR = 0.5

# Kalman filter weights for combining variance and consistency confidence.
_VARIANCE_CONFIDENCE_WEIGHT = 0.7
_CONSISTENCY_CONFIDENCE_WEIGHT = 0.3

# Thresholds for detecting systematic Kalman drift and triggering recalibration.
_DRIFT_BIAS_THRESHOLD = 0.2   # minimum bias magnitude to trigger recalibration
_DRIFT_STD_THRESHOLD = 0.1    # only recalibrate when measurements are consistent

# Search window size per V1 sample step.
# The effective search window is auto-scaled as: search_window = sample_rate * this value.
# This means that if you process every frame (sample_rate=1) you get a tight window
# (20 V2 frames to check), while sampling every 15th frame gives a proportionally
# wider window (300 V2 frames) so the correct match is never missed.
# Can be overridden by passing an explicit search_window argument.
_WINDOW_FRAMES_PER_SAMPLE_STEP = 20

def _build_dtw_predictor(video1_path, video2_path, dtw_sample_rate=10, penalty=1.5):
    """
    Runs DTW on optical flow features and returns:
      - predict_fn(v1_frame) -> int   : interpolated V2 prediction for any V1 frame
      - global_velocity: float        : average V2/V1 speed ratio from the DTW path
    """
    extractor = VideoFeatureExtractor(sample_rate=dtw_sample_rate)

    print("  [DTW] Extracting optical flow features from Video 1...")
    features1, indices1 = extractor.extract_features(video1_path)
    print("  [DTW] Extracting optical flow features from Video 2...")
    features2, indices2 = extractor.extract_features(video2_path)

    # Z-score normalization per feature dimension
    features1_norm = (features1 - np.mean(features1, axis=0)) / (np.std(features1, axis=0) + 1e-6)
    features2_norm = (features2 - np.mean(features2, axis=0)) / (np.std(features2, axis=0) + 1e-6)

    print("  [DTW] Running DTW alignment...")
    path, _ = compute_dtw(features1_norm, features2_norm, step_penalty=penalty)

    # Map sampled indices back to original frame numbers
    real_path = [(indices1[i], indices2[j]) for i, j in path]

    # Collapse stuttering: for each V1 frame, take the median mapped V2 frame
    v1_to_v2 = {}
    for v1, v2 in real_path:
        v1_to_v2.setdefault(v1, []).append(v2)

    anchor_points = sorted([(v1, int(np.median(v2s))) for v1, v2s in v1_to_v2.items()])

    # Estimate global velocity (V2 frames / V1 frames)
    if len(anchor_points) >= 2:
        v1_first, v2_first = anchor_points[0]
        v1_last, v2_last = anchor_points[-1]
        dv1 = v1_last - v1_first
        global_velocity = (v2_last - v2_first) / dv1 if dv1 > 0 else 1.0
    else:
        global_velocity = 1.0

    v1_anchors = np.array([p[0] for p in anchor_points], dtype=float)
    v2_anchors = np.array([p[1] for p in anchor_points], dtype=float)

    def predict_fn(v1_frame):
        if len(v1_anchors) == 0:
            return v1_frame
        return int(np.interp(v1_frame, v1_anchors, v2_anchors))

    return predict_fn, global_velocity


def _run_ransac(kp1, kp2, good_matches):
    """
    Runs RANSAC homography estimation and returns the inlier count.
    Returns a penalised count when the inferred scale is unreasonable.
    """
    if len(good_matches) < 4:
        return 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=2000,
        confidence=0.999,
    )
    if mask is None:
        return 0

    inlier_count = int(np.sum(mask))
    if M is not None and inlier_count > 0:
        try:
            scale_x = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
            scale_y = np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
            if scale_x < 0.7 or scale_x > 1.3 or scale_y < 0.7 or scale_y > 1.3:
                inlier_count = int(inlier_count * 0.5)
        except (ValueError, ZeroDivisionError, IndexError):
            inlier_count = int(inlier_count * 0.7)

    return inlier_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_coarse_to_fine(
    video1_path,
    video2_path,
    algo_name="ORB",
    sample_rate=1,
    dtw_sample_rate=10,
    search_window=None,
    search_step=1,
    dtw_penalty=1.5,
    dtw_weight=0.5,
):
    """
    Improved coarse-to-fine alignment.

    Phase 1 – DTW on optical flow gives a dense, interpolated global prediction.
    Phase 2 – Feature matching refines each frame within a focused search window
              centred on a blend of the DTW prediction and a Kalman velocity estimate.
    Phase 3 – DTW prediction used as fallback when feature matching fails.

    Args:
        video1_path (str):     Path to the reference video.
        video2_path (str):     Path to the second video.
        algo_name (str):       Feature detector ("ORB", "BRISK", "AKAZE").
        sample_rate (int):     Process every Nth V1 frame for feature matching.
                               Controls the speed/accuracy trade-off:
                               - sample_rate=1  → every frame, slow but dense output
                               - sample_rate=15 → every 15th frame, fast but sparser
        dtw_sample_rate (int): Frame sampling rate for the DTW optical-flow phase.
        search_window (int|None): Total width of the V2 search window in frames.
                               When None (default), auto-computed as:
                               ``sample_rate * _WINDOW_FRAMES_PER_SAMPLE_STEP``
                               so that larger sample steps get proportionally wider
                               windows (e.g. sample_rate=1 → 20 frames,
                               sample_rate=15 → 300 frames).
        search_step (int):     Step size inside the search window.
        dtw_penalty (float):   Step penalty passed to DTW (reduces stuttering).
        dtw_weight (float):    Initial weight for DTW prediction vs Kalman prediction
                               (0 = Kalman only, 1 = DTW only). Decreases as local
                               confidence grows.

    Returns:
        list[dict]: Alignment results with keys 'v1_frame', 'v2_frame', 'score',
                    'algorithm'.  Fallback frames have score=0.
    """
    # Auto-scale search window: tighter window for dense sampling, wider for sparse.
    if search_window is None:
        search_window = sample_rate * _WINDOW_FRAMES_PER_SAMPLE_STEP

    print(
        f"Combined alignment: {video1_path} + {video2_path}  algo={algo_name}"
        f"  sample_rate={sample_rate}  search_window={search_window}"
    )

    if algo_name not in ALGORITHMS:
        print(f"Error: Unknown algorithm '{algo_name}'")
        return []

    algo_module = ALGORITHMS[algo_name]

    # ------------------------------------------------------------------
    # Phase 1: Coarse Global Alignment via DTW
    # ------------------------------------------------------------------
    print("Phase 1: Computing coarse global alignment via DTW...")
    dtw_predict, global_velocity = _build_dtw_predictor(
        video1_path, video2_path,
        dtw_sample_rate=dtw_sample_rate,
        penalty=dtw_penalty,
    )
    print(f"  Global velocity estimate: {global_velocity:.3f}")

    # ------------------------------------------------------------------
    # Phase 2: Fine Feature Matching with Blended Prediction
    # ------------------------------------------------------------------
    print(f"Phase 2: Fine feature matching using {algo_name}...")

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos.")
        return []

    total_v2_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Kalman filter state – initialised from the DTW global velocity
    estimated_velocity = global_velocity
    velocity_variance = 0.5      # Moderate initial uncertainty (DTW provides a prior)
    process_noise = 0.01
    measurement_noise = 0.1
    velocity_confidence = 0.3    # Some initial confidence thanks to DTW

    last_kalman_v2_frame = None
    last_matched_v1_frame = 0
    consecutive_failures = 0

    results = []
    v1_frame_idx = 0

    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break

        if v1_frame_idx % sample_rate != 0:
            v1_frame_idx += 1
            continue

        kp1, des1 = algo_module.compute_features(frame1)
        if des1 is None or len(kp1) < 10:
            v1_frame_idx += 1
            continue

        # ---------------------------------------------------------------
        # Blended prediction: DTW + Kalman
        # ---------------------------------------------------------------
        dtw_pred = dtw_predict(v1_frame_idx)

        if last_kalman_v2_frame is not None:
            frames_since_last = v1_frame_idx - last_matched_v1_frame
            kalman_pred = last_kalman_v2_frame + int(frames_since_last * estimated_velocity)
        else:
            kalman_pred = dtw_pred

        # As local confidence grows, rely more on Kalman and less on DTW
        effective_dtw_weight = dtw_weight * (1.0 - velocity_confidence * _CONFIDENCE_DISCOUNT_FACTOR)
        effective_kalman_weight = 1.0 - effective_dtw_weight
        blended_pred = int(effective_dtw_weight * dtw_pred + effective_kalman_weight * kalman_pred)
        blended_pred = max(0, min(total_v2_frames - 1, blended_pred))

        # ---------------------------------------------------------------
        # Adaptive search window
        # ---------------------------------------------------------------
        adaptive_window = search_window
        if velocity_confidence > 0.7:
            adaptive_window = int(search_window * 0.7)
        elif velocity_confidence < 0.3:
            adaptive_window = int(search_window * 1.3)

        if consecutive_failures > 0:
            expansion_factor = 1.0 + consecutive_failures * _FAILURE_EXPANSION_RATE
            adaptive_window = int(adaptive_window * min(expansion_factor, _MAX_WINDOW_EXPANSION))

        # Allow a small backward margin but enforce monotonicity
        backward_margin = max(_MIN_BACKWARD_MARGIN, int(adaptive_window * _BACKWARD_MARGIN_RATIO))
        if last_kalman_v2_frame is not None:
            search_start = max(last_kalman_v2_frame, blended_pred - backward_margin)
        else:
            search_start = max(0, blended_pred - backward_margin)
        search_end = min(total_v2_frames, search_start + adaptive_window)

        # ---------------------------------------------------------------
        # Gather candidates
        # ---------------------------------------------------------------
        raw_candidates = []
        current_search_idx = search_start

        while current_search_idx < search_end:
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

                min_matches = max(8, min(10, len(kp1) // 20))
                if len(good_matches) >= min_matches:
                    raw_candidates.append({"idx": current_search_idx, "matches": good_matches, "kp2": kp2})

            current_search_idx += search_step

        # ---------------------------------------------------------------
        # Branch-and-Bound RANSAC (sorted by distance from prediction)
        # ---------------------------------------------------------------
        raw_candidates.sort(key=lambda x: abs(x["idx"] - blended_pred))

        base_k = _BASE_PENALTY_FACTOR
        k = base_k * (_CONFIDENCE_OFFSET + velocity_confidence)

        best_weighted_score = -float("inf")
        best_v2_idx = -1
        best_match_raw_score = 0

        for cand in raw_candidates:
            idx = cand["idx"]
            good_matches = cand["matches"]
            kp2 = cand["kp2"]

            dist = abs(idx - blended_pred)
            penalty_multiplier = math.exp(-k * max(0, dist - 1))
            max_possible_score = len(good_matches) * penalty_multiplier

            if max_possible_score <= best_weighted_score:
                continue  # Branch-and-Bound pruning

            inlier_count = _run_ransac(kp1, kp2, good_matches)
            weighted = inlier_count * penalty_multiplier

            if weighted > best_weighted_score:
                best_weighted_score = weighted
                best_v2_idx = idx
                best_match_raw_score = inlier_count

        # ---------------------------------------------------------------
        # Accept or fall back to DTW prediction
        # ---------------------------------------------------------------
        acceptance_threshold = _BASE_ACCEPTANCE_THRESHOLD * (_CONFIDENCE_OFFSET + velocity_confidence)

        if best_v2_idx != -1 and best_weighted_score >= acceptance_threshold:
            matched_v2 = best_v2_idx
            score = best_match_raw_score

            # Update Kalman filter from recent verified matches
            if results:
                recent = results[-min(5, len(results)):]
                velocities = []
                for i in range(1, len(recent)):
                    dv1 = recent[i]["v1_frame"] - recent[i - 1]["v1_frame"]
                    dv2 = recent[i]["v2_frame"] - recent[i - 1]["v2_frame"]
                    if dv1 > 0:
                        velocities.append(dv2 / dv1)

                if velocities:
                    measured_velocity = np.mean(velocities)
                    velocity_std = np.std(velocities) if len(velocities) > 1 else 0.5

                    predicted_variance = velocity_variance + process_noise
                    kalman_gain = predicted_variance / (predicted_variance + measurement_noise + velocity_std ** 2)
                    estimated_velocity = estimated_velocity + kalman_gain * (measured_velocity - estimated_velocity)
                    velocity_variance = (1 - kalman_gain) * predicted_variance
                    estimated_velocity = max(_MIN_VELOCITY, min(_MAX_VELOCITY, estimated_velocity))

                    variance_confidence = max(0.0, min(1.0, 1.0 - velocity_variance))
                    consistency_confidence = max(0.0, min(1.0, 1.0 - velocity_std))
                    velocity_confidence = (
                        _VARIANCE_CONFIDENCE_WEIGHT * variance_confidence
                        + _CONSISTENCY_CONFIDENCE_WEIGHT * consistency_confidence
                    )

                    if len(velocities) >= 3:
                        recent_bias = measured_velocity - estimated_velocity
                        if abs(recent_bias) > _DRIFT_BIAS_THRESHOLD and velocity_std < _DRIFT_STD_THRESHOLD:
                            velocity_variance = min(velocity_variance + 0.5, 1.0)
                            process_noise = min(process_noise * 1.5, 0.1)

            last_kalman_v2_frame = matched_v2
            last_matched_v1_frame = v1_frame_idx
            consecutive_failures = 0

            print(
                f"V1 {v1_frame_idx} -> V2 {matched_v2} [FEATURE]"
                f"  score={score}  dtw={dtw_pred}  blend={blended_pred}"
                f"  vel={estimated_velocity:.2f}  conf={velocity_confidence:.2f}"
            )
        else:
            # Phase 3: DTW fallback – guarantees full coverage
            matched_v2 = dtw_pred
            score = 0
            consecutive_failures += 1

            print(f"V1 {v1_frame_idx} -> V2 {matched_v2} [DTW FALLBACK]  dtw={dtw_pred}")

        results.append({
            "v1_frame": v1_frame_idx,
            "v2_frame": matched_v2,
            "score": score,
            "algorithm": f"CoarseFine_{algo_name}",
        })

        v1_frame_idx += 1

    cap1.release()
    cap2.release()

    feature_count = sum(1 for r in results if r["score"] > 0)
    total = len(results)
    print(
        f"Alignment complete: {total} frames total, "
        f"{feature_count} feature matches ({100 * feature_count // max(1, total)}%), "
        f"{total - feature_count} DTW fallbacks."
    )
    return results


def align_fusion(video1_path, video2_path, algo_name="ORB", sample_rate=10, penalty=1.5):
    """
    Fused DTW alignment: combines optical-flow distances and keypoint-inlier
    distances into a single cost matrix, then runs DTW on it.

    Improvements over the original:
    - Uses a tighter candidate filter (top 20 % by flow distance) to reduce
      the expensive O(N×M) keypoint computation.
    - Delegates RANSAC to the shared ``_run_ransac`` helper.
    - Normalises each distance matrix independently before fusion so neither
      modality dominates due to scale differences.

    Args:
        video1_path (str):   Path to the reference video.
        video2_path (str):   Path to the second video.
        algo_name (str):     Feature detector ("ORB", "BRISK", "AKAZE").
        sample_rate (int):   Frame sampling rate for both optical-flow and keypoint
                             extraction.
        penalty (float):     DTW step penalty (reduces stuttering).

    Returns:
        list[dict]: Alignment results with keys 'v1_frame', 'v2_frame', 'score',
                    'algorithm'.
    """
    from scipy.spatial.distance import cdist

    print(f"Fusion alignment: {video1_path} + {video2_path}  algo={algo_name}")

    if algo_name not in ALGORITHMS:
        print(f"Error: Unknown algorithm '{algo_name}'")
        return []

    algo_module = ALGORITHMS[algo_name]

    # ------------------------------------------------------------------
    # Step 1: Optical-flow feature extraction
    # ------------------------------------------------------------------
    print("  [Fusion] Extracting optical flow features...")
    extractor = VideoFeatureExtractor(sample_rate=sample_rate)
    features1, indices1 = extractor.extract_features(video1_path)
    features2, indices2 = extractor.extract_features(video2_path)

    features1_norm = (features1 - np.mean(features1, axis=0)) / (np.std(features1, axis=0) + 1e-6)
    features2_norm = (features2 - np.mean(features2, axis=0)) / (np.std(features2, axis=0) + 1e-6)

    dist_matrix_flow = cdist(features1_norm, features2_norm, metric="euclidean")

    # ------------------------------------------------------------------
    # Step 2: Keypoint extraction for sampled frames
    # ------------------------------------------------------------------
    print("  [Fusion] Extracting keypoints...")

    def _extract_keypoints(video_path, frame_indices):
        cap = cv2.VideoCapture(video_path)
        kp_dict = {}
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            kp, des = algo_module.compute_features(frame)
            kp_dict[idx] = (kp, des)
        cap.release()
        return kp_dict

    kp_dict1 = _extract_keypoints(video1_path, indices1)
    kp_dict2 = _extract_keypoints(video2_path, indices2)

    # ------------------------------------------------------------------
    # Step 3: Sparse keypoint distance matrix (top-20% flow candidates)
    # ------------------------------------------------------------------
    print("  [Fusion] Computing keypoint distances (sparse)...")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    n, m_len = len(indices1), len(indices2)
    dist_matrix_kp = np.full((n, m_len), 10.0)

    for i in range(n):
        idx1 = indices1[i]
        kp1, des1 = kp_dict1.get(idx1, ([], None))
        if des1 is None or len(kp1) < 10:
            continue

        flow_row = dist_matrix_flow[i]
        threshold = np.percentile(flow_row, 20)  # Tighter than before (20 % vs 30 %)

        for j in range(m_len):
            if flow_row[j] > threshold:
                continue

            idx2 = indices2[j]
            kp2, des2 = kp_dict2.get(idx2, ([], None))
            if des2 is None or len(kp2) < 10:
                continue

            knn_matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m_m, n_m = match_pair
                    if m_m.distance < 0.75 * n_m.distance:
                        good_matches.append(m_m)
                elif len(match_pair) == 1:
                    m_m = match_pair[0]
                    if m_m.distance < 50:
                        good_matches.append(m_m)

            inlier_count = _run_ransac(kp1, kp2, good_matches) if len(good_matches) >= 4 else 0
            dist_matrix_kp[i, j] = max(0.0, 10.0 - inlier_count / 10.0)

    # ------------------------------------------------------------------
    # Step 4: Normalise and fuse distance matrices
    # ------------------------------------------------------------------
    dist_matrix_flow /= np.max(dist_matrix_flow) + 1e-6
    dist_matrix_kp /= np.max(dist_matrix_kp) + 1e-6

    w_flow, w_kp = 0.4, 0.6
    fused_matrix = w_flow * dist_matrix_flow + w_kp * dist_matrix_kp

    # ------------------------------------------------------------------
    # Step 5: DTW on the fused matrix
    # ------------------------------------------------------------------
    print("  [Fusion] Running DTW on fused cost matrix...")
    path, _ = _dtw_on_matrix(fused_matrix, penalty)

    results = []
    for raw_i, raw_j in path:
        v1_idx = indices1[raw_i]
        v2_idx = indices2[raw_j]
        score = 1.0 / (1.0 + fused_matrix[raw_i, raw_j])
        results.append({
            "v1_frame": v1_idx,
            "v2_frame": v2_idx,
            "score": float(score),
            "algorithm": f"Fusion_{algo_name}",
        })

    return results


def _dtw_on_matrix(dist_matrix, step_penalty=1.5):
    """
    Runs DTW directly on a pre-computed distance matrix.
    Mirrors the logic in new_method/dtw_alignment.py but operates on a matrix
    that has already been computed.

    Returns:
        path (list of (i, j) tuples): optimal warping path.
        acc_cost (np.ndarray): accumulated cost matrix.
    """
    n, m = dist_matrix.shape
    avg_dist = np.mean(dist_matrix)
    penalty = avg_dist * step_penalty

    acc_cost = np.zeros((n, m))
    acc_cost[0, 0] = dist_matrix[0, 0]

    for i in range(1, n):
        acc_cost[i, 0] = acc_cost[i - 1, 0] + dist_matrix[i, 0] + penalty
    for j in range(1, m):
        acc_cost[0, j] = acc_cost[0, j - 1] + dist_matrix[0, j] + penalty

    for i in range(1, n):
        for j in range(1, m):
            acc_cost[i, j] = dist_matrix[i, j] + min(
                acc_cost[i - 1, j] + penalty,
                acc_cost[i, j - 1] + penalty,
                acc_cost[i - 1, j - 1],
            )

    # Backtrack
    path = []
    i, j = n - 1, m - 1
    path.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            cost_up = acc_cost[i - 1, j] + penalty
            cost_left = acc_cost[i, j - 1] + penalty
            cost_diag = acc_cost[i - 1, j - 1]
            min_val = min(cost_up, cost_left, cost_diag)
            if cost_diag == min_val:
                i -= 1
                j -= 1
            elif cost_up == min_val:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    return path[::-1], acc_cost
