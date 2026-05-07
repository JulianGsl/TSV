"""
Hybrid Video Alignment - Coarse-to-Fine Approach (v2 - Anti-Trembling)

This module combines:
- Phase A (Coarse): DTW-based optical flow alignment for robust macro-synchronization
- Phase B (Fine): Feature matching (AKAZE/BRISK/ORB) for precise micro-alignment
- Phase C (Fallback): DTW prediction as safety net when feature matching fails
- Phase D (NEW): Global optimization with monotonicity constraint to prevent trembling

Key improvements in v2:
- DTW distance penalty: Scores are weighted by proximity to DTW prediction
- Top-K candidates: Keep best K candidates per frame for global optimization
- Monotonic path optimization: Dynamic Programming to find smooth, monotonic alignment
- Post-processing smoothing: Final pass to eliminate residual jitter
- Multi-algorithm support: AKAZE, BRISK, or ORB for feature matching

The DTW provides an "unbreakable skeleton" that prevents drift,
while feature matching acts as a "magnifying glass" for pixel-precise alignment.
"""

import cv2
import numpy as np
import sys
import os
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Old_version.new_method.feature_extraction import VideoFeatureExtractor
from Old_version.new_method.dtw_alignment import compute_dtw

# Supported feature matching algorithms
SUPPORTED_ALGORITHMS = ["AKAZE", "BRISK", "ORB"]


class HybridAligner:
    """
    Coarse-to-Fine video alignment combining DTW (macro) and feature matching (micro).
    Supports AKAZE, BRISK, and ORB algorithms.
    """

    def __init__(
        self,
        algorithm: str = "AKAZE",
        dtw_sample_rate: int = 5,
        dtw_step_penalty: float = 1.5,
        feature_sample_rate: int = 1,
        search_window: int = 10,
        min_inliers_threshold: int = 4,
        lowe_ratio: float = 0.75,
        ransac_reproj_threshold: float = 2.5,
        # New v2 parameters for anti-trembling
        dtw_distance_penalty_k: float = 0.15,
        top_k_candidates: int = 5,
        enable_global_optimization: bool = True,
        enable_smoothing: bool = True,
        smoothing_window: int = 5,
        verbose: bool = True
    ):
        """
        Initialize the hybrid aligner.

        Args:
            algorithm: Feature matching algorithm ("AKAZE", "BRISK", or "ORB")
            dtw_sample_rate: Sample rate for DTW feature extraction (higher = faster but coarser)
            dtw_step_penalty: Penalty for non-diagonal DTW steps
            feature_sample_rate: Sample rate for feature matching refinement phase
            search_window: Half-window size for search around DTW prediction (e.g., 10 = [-10, +10])
            min_inliers_threshold: Minimum RANSAC inliers to accept match (fallback to DTW if below)
            lowe_ratio: Lowe's ratio test threshold for feature matching
            ransac_reproj_threshold: RANSAC reprojection threshold

            # New v2 parameters:
            dtw_distance_penalty_k: Exponential decay factor for DTW distance penalty.
                                    Higher = more trust in DTW, lower = more freedom for feature matching.
                                    Formula: score = inliers * exp(-k * |candidate - dtw_prediction|)
            top_k_candidates: Number of best candidates to keep per frame for global optimization.
            enable_global_optimization: If True, use Dynamic Programming to find monotonic path.
            enable_smoothing: If True, apply post-processing smoothing to remove residual jitter.
            smoothing_window: Window size for median/weighted smoothing.
            verbose: Print progress information
        """
        # Validate algorithm
        self.algorithm = algorithm.upper()
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from: {SUPPORTED_ALGORITHMS}")

        self.dtw_sample_rate = dtw_sample_rate
        self.dtw_step_penalty = dtw_step_penalty
        self.feature_sample_rate = feature_sample_rate
        self.search_window = search_window
        self.min_inliers_threshold = min_inliers_threshold
        self.lowe_ratio = lowe_ratio
        self.ransac_reproj_threshold = ransac_reproj_threshold

        # New v2 parameters
        self.dtw_distance_penalty_k = dtw_distance_penalty_k
        self.top_k_candidates = top_k_candidates
        self.enable_global_optimization = enable_global_optimization
        self.enable_smoothing = enable_smoothing
        self.smoothing_window = smoothing_window

        self.verbose = verbose

        # Initialize feature detector based on algorithm
        self._init_feature_detector()

        # Initialize BFMatcher for binary descriptors (works for all 3 algorithms)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _init_feature_detector(self):
        """Initialize the appropriate feature detector based on selected algorithm."""
        if self.algorithm == "AKAZE":
            self.detector = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                threshold=0.001
            )
        elif self.algorithm == "BRISK":
            self.detector = cv2.BRISK_create(
                thresh=30,
                octaves=4,
                patternScale=1.0
            )
        elif self.algorithm == "ORB":
            self.detector = cv2.ORB_create(
                nfeatures=5000,
                scaleFactor=1.2,
                nlevels=8
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _log(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    def _extract_dtw_features(self, video_path: str) -> tuple:
        """
        Phase A.1: Extract optical flow features for DTW.

        Returns:
            features: numpy array of optical flow features
            frame_indices: list of original frame indices
        """
        extractor = VideoFeatureExtractor(
            resize_dim=(320, 240),
            sample_rate=self.dtw_sample_rate,
            ignore_sky=True
        )
        features, frame_indices = extractor.extract_features(video_path)

        # Z-score normalization for better DTW performance
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        features_norm = (features - mean) / (std + 1e-6)

        return features_norm, frame_indices

    def _build_dtw_mapping(self, path: list, indices1: list, indices2: list, total_frames_v1: int) -> dict:
        """
        Phase A.3: Convert DTW path to a frame mapping dictionary with interpolation.

        Args:
            path: DTW path as list of (i, j) tuples (sampled indices)
            indices1: Original frame indices for video 1
            indices2: Original frame indices for video 2
            total_frames_v1: Total number of frames in video 1

        Returns:
            Dictionary mapping v1_frame -> predicted_v2_frame for all frames
        """
        # Convert sampled path to actual frame indices
        actual_path = [(indices1[i], indices2[j]) for i, j in path]

        # Build sparse mapping from actual path
        sparse_mapping = {}
        for v1_idx, v2_idx in actual_path:
            sparse_mapping[v1_idx] = v2_idx

        # Interpolate missing frames (linear interpolation)
        dtw_mapping = {}
        sorted_v1_frames = sorted(sparse_mapping.keys())

        for frame_idx in range(total_frames_v1):
            if frame_idx in sparse_mapping:
                dtw_mapping[frame_idx] = sparse_mapping[frame_idx]
            else:
                # Find surrounding anchor points for interpolation
                lower_anchor = None
                upper_anchor = None

                for anchor in sorted_v1_frames:
                    if anchor <= frame_idx:
                        lower_anchor = anchor
                    if anchor >= frame_idx and upper_anchor is None:
                        upper_anchor = anchor

                if lower_anchor is None and upper_anchor is not None:
                    # Before first anchor: extrapolate from start
                    dtw_mapping[frame_idx] = sparse_mapping[upper_anchor]
                elif upper_anchor is None and lower_anchor is not None:
                    # After last anchor: extrapolate from end
                    dtw_mapping[frame_idx] = sparse_mapping[lower_anchor]
                elif lower_anchor is not None and upper_anchor is not None:
                    # Linear interpolation between anchors
                    v2_lower = sparse_mapping[lower_anchor]
                    v2_upper = sparse_mapping[upper_anchor]
                    if upper_anchor == lower_anchor:
                        dtw_mapping[frame_idx] = v2_lower
                    else:
                        ratio = (frame_idx - lower_anchor) / (upper_anchor - lower_anchor)
                        dtw_mapping[frame_idx] = int(v2_lower + ratio * (v2_upper - v2_lower))
                else:
                    # Fallback: use frame index directly
                    dtw_mapping[frame_idx] = frame_idx

        return dtw_mapping

    def _compute_features(self, frame: np.ndarray) -> tuple:
        """
        Extract features from a frame using the selected algorithm.

        Returns:
            keypoints: list of cv2.KeyPoint
            descriptors: numpy array of descriptors (or None)
        """
        if frame is None:
            return [], None

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def _filter_matches_spatial(self, matches: list, kp1: list, kp2: list, frame_width: int) -> list:
        """
        Filter matches by spatial coherence (left/right half consistency).

        For rail track videos, matching features should generally be on the same
        side of the image (parallax effect). This filters out physically impossible matches.

        Args:
            matches: List of cv2.DMatch objects
            kp1: Keypoints from frame 1
            kp2: Keypoints from frame 2
            frame_width: Width of the frames

        Returns:
            Filtered list of matches with spatial coherence
        """
        if not matches:
            return matches

        mid_x = frame_width / 2
        filtered = []

        for m in matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt

            # Check if both points are on the same side (left or right)
            pt1_left = pt1[0] < mid_x
            pt2_left = pt2[0] < mid_x

            if pt1_left == pt2_left:
                filtered.append(m)

        return filtered

    def _match_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        kp1: list,
        des1: np.ndarray
    ) -> int:
        """
        Match two frames using AKAZE and return inlier count.

        Args:
            frame1: Reference frame (from video 1)
            frame2: Candidate frame (from video 2)
            kp1: Pre-computed keypoints for frame1
            des1: Pre-computed descriptors for frame1

        Returns:
            Number of RANSAC inliers (0 if matching failed)
        """
        if des1 is None or len(kp1) < 4:
            return 0

        # Extract features from frame2
        kp2, des2 = self._compute_features(frame2)

        if des2 is None or len(kp2) < 4:
            return 0

        # KNN matching with Lowe's ratio test
        try:
            knn_matches = self.bf.knnMatch(des1, des2, k=2)
        except cv2.error:
            return 0

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.lowe_ratio * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                m = match_pair[0]
                if m.distance < 50:  # Absolute threshold for single matches
                    good_matches.append(m)

        if len(good_matches) < 4:
            return 0

        # Apply spatial coherence filter
        frame_width = frame1.shape[1] if len(frame1.shape) >= 2 else 640
        good_matches = self._filter_matches_spatial(good_matches, kp1, kp2, frame_width)

        if len(good_matches) < 4:
            return 0

        # RANSAC homography estimation
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        try:
            M, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
                maxIters=2000,
                confidence=0.999
            )
        except cv2.error:
            return 0

        if mask is None:
            return 0

        inlier_count = int(np.sum(mask))

        # Validate homography is reasonable (not too extreme transformation)
        if M is not None and inlier_count > 0:
            try:
                scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
                scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)
                # For rail tracks, scale should be close to 1.0
                if scale_x < 0.7 or scale_x > 1.3 or scale_y < 0.7 or scale_y > 1.3:
                    inlier_count = int(inlier_count * 0.5)
            except (ValueError, ZeroDivisionError, IndexError):
                inlier_count = int(inlier_count * 0.7)

        return inlier_count

    def _compute_weighted_score(self, inliers: int, candidate_frame: int, dtw_prediction: int) -> float:
        """
        Compute score with DTW distance penalty.

        The score is weighted by how far the candidate is from the DTW prediction.
        Formula: score = inliers * exp(-k * |candidate - dtw_prediction|)

        This encourages AKAZE to stay close to DTW while allowing small corrections.

        Args:
            inliers: Raw RANSAC inlier count
            candidate_frame: The candidate frame index in V2
            dtw_prediction: The DTW-predicted frame index

        Returns:
            Weighted score (float)
        """
        if inliers == 0:
            return 0.0

        distance = abs(candidate_frame - dtw_prediction)
        penalty = np.exp(-self.dtw_distance_penalty_k * distance)

        return inliers * penalty

    def _find_monotonic_path(self, candidates_per_frame: List[List[Tuple[int, float, int]]]) -> List[Dict]:
        """
        Find the optimal monotonic path through all frames using Dynamic Programming.

        This ensures V2 frames always increase (or stay same), preventing "trembling".

        Algorithm:
        - For each frame i, and each candidate c in top-K:
          - best_score[i][c] = max over all valid predecessors of (best_score[i-1][p] + score[i][c])
          - Valid predecessor: V2[p] <= V2[c] (monotonicity constraint)

        Args:
            candidates_per_frame: List of lists, each containing (v2_frame, weighted_score, raw_inliers)
                                  for top-K candidates per V1 frame.

        Returns:
            Optimized list of match dictionaries with monotonic V2 progression.
        """
        n_frames = len(candidates_per_frame)
        if n_frames == 0:
            return []

        # DP tables
        # best_score[i][j] = best cumulative score ending at frame i with candidate j
        # best_prev[i][j] = which candidate at frame i-1 led to best_score[i][j]
        best_score = []
        best_prev = []

        # Initialize first frame
        first_candidates = candidates_per_frame[0]
        best_score.append([c[1] for c in first_candidates])  # weighted scores
        best_prev.append([-1] * len(first_candidates))  # no predecessor

        # Fill DP table
        for i in range(1, n_frames):
            current_candidates = candidates_per_frame[i]
            n_curr = len(current_candidates)
            n_prev = len(candidates_per_frame[i-1])

            curr_best_score = [-float('inf')] * n_curr
            curr_best_prev = [-1] * n_curr

            for j, (v2_curr, score_curr, _) in enumerate(current_candidates):
                # Find best predecessor that satisfies monotonicity
                for p, (v2_prev, _, _) in enumerate(candidates_per_frame[i-1]):
                    # Monotonicity: V2 must not decrease
                    if v2_prev <= v2_curr:
                        candidate_score = best_score[i-1][p] + score_curr
                        if candidate_score > curr_best_score[j]:
                            curr_best_score[j] = candidate_score
                            curr_best_prev[j] = p

                # If no valid predecessor found (all violate monotonicity),
                # accept DTW prediction with penalty
                if curr_best_prev[j] == -1:
                    # Find the closest valid predecessor
                    for p, (v2_prev, _, _) in enumerate(candidates_per_frame[i-1]):
                        candidate_score = best_score[i-1][p] + score_curr * 0.5  # penalty
                        if candidate_score > curr_best_score[j]:
                            curr_best_score[j] = candidate_score
                            curr_best_prev[j] = p

            best_score.append(curr_best_score)
            best_prev.append(curr_best_prev)

        # Backtrack to find optimal path
        path_indices = []

        # Find best ending candidate
        last_scores = best_score[-1]
        best_end_idx = np.argmax(last_scores) if last_scores else 0
        path_indices.append(best_end_idx)

        # Backtrack
        for i in range(n_frames - 1, 0, -1):
            prev_idx = best_prev[i][path_indices[-1]]
            if prev_idx == -1:
                prev_idx = 0  # fallback
            path_indices.append(prev_idx)

        path_indices.reverse()

        # Build result
        results = []
        for i, candidate_idx in enumerate(path_indices):
            if candidate_idx < len(candidates_per_frame[i]):
                v2_frame, weighted_score, raw_inliers = candidates_per_frame[i][candidate_idx]
            else:
                # Fallback to first candidate
                v2_frame, weighted_score, raw_inliers = candidates_per_frame[i][0]

            results.append({
                'v2_frame': v2_frame,
                'score': raw_inliers,
                'weighted_score': weighted_score
            })

        return results

    def _apply_smoothing(self, results: List[Dict]) -> List[Dict]:
        """
        Apply post-processing smoothing to remove residual jitter.

        Uses weighted median filter that respects:
        - Monotonicity (V2 should not decrease)
        - Score-based weighting (high-confidence matches weighted more)

        Args:
            results: List of match dictionaries

        Returns:
            Smoothed results
        """
        if len(results) < 3:
            return results

        v2_frames = np.array([r['v2_frame'] for r in results], dtype=float)
        scores = np.array([r.get('score', 1) for r in results], dtype=float)

        # Normalize scores for weighting
        scores = np.maximum(scores, 1)  # Avoid zero weights

        smoothed_v2 = v2_frames.copy()
        half_window = self.smoothing_window // 2

        for i in range(len(v2_frames)):
            start = max(0, i - half_window)
            end = min(len(v2_frames), i + half_window + 1)

            window_v2 = v2_frames[start:end]
            window_scores = scores[start:end]

            # Weighted average instead of median (smoother)
            weights = window_scores / np.sum(window_scores)
            smoothed_value = np.sum(window_v2 * weights)

            smoothed_v2[i] = smoothed_value

        # Enforce monotonicity after smoothing
        for i in range(1, len(smoothed_v2)):
            if smoothed_v2[i] < smoothed_v2[i-1]:
                smoothed_v2[i] = smoothed_v2[i-1]

        # Round to integers
        smoothed_v2 = np.round(smoothed_v2).astype(int)

        # Update results
        for i, r in enumerate(results):
            r['v2_frame_raw'] = r['v2_frame']  # Keep original
            r['v2_frame'] = int(smoothed_v2[i])

        return results

    def align_videos(self, video1_path: str, video2_path: str) -> list:
        """
        Main alignment function using Coarse-to-Fine approach.

        Args:
            video1_path: Path to reference video (J-1)
            video2_path: Path to target video (J)

        Returns:
            List of match dictionaries: [{'v1_frame': int, 'v2_frame': int, 'score': int, 'source': str}, ...]
        """
        self._log("=" * 60)
        self._log("HYBRID ALIGNMENT: Coarse-to-Fine Approach")
        self._log("=" * 60)

        # ============================================================
        # PHASE A: MACRO-SYNCHRONIZATION (DTW Skeleton)
        # ============================================================
        self._log("\n[Phase A] Extracting optical flow features for DTW...")

        # A.1: Extract features
        features1, indices1 = self._extract_dtw_features(video1_path)
        features2, indices2 = self._extract_dtw_features(video2_path)

        self._log(f"  Video 1: {len(features1)} sampled frames")
        self._log(f"  Video 2: {len(features2)} sampled frames")

        # A.2: Compute Open-End DTW
        self._log("\n[Phase A] Computing Open-End DTW alignment...")
        path, cost_matrix = compute_dtw(
            features1, features2,
            metric='euclidean',
            step_penalty=self.dtw_step_penalty,
            open_end=True
        )

        # Get actual endpoint
        end_i, end_j = path[-1]
        self._log(f"  DTW path length: {len(path)} points")
        self._log(f"  DTW endpoint: V1[{indices1[end_i]}] <-> V2[{indices2[end_j]}]")

        # Get total frame counts
        cap1_temp = cv2.VideoCapture(video1_path)
        cap2_temp = cv2.VideoCapture(video2_path)
        total_frames_v1 = int(cap1_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_v2 = int(cap2_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap1_temp.release()
        cap2_temp.release()

        self._log(f"  Total frames: V1={total_frames_v1}, V2={total_frames_v2}")

        # A.3: Build interpolated mapping
        dtw_mapping = self._build_dtw_mapping(path, indices1, indices2, total_frames_v1)
        self._log(f"  DTW mapping built for {len(dtw_mapping)} frames")

        # ============================================================
        # PHASE B: MICRO-ALIGNMENT (AKAZE Refinement with Top-K)
        # ============================================================
        self._log(f"\n[Phase B] {self.algorithm} micro-alignment with DTW distance penalty...")
        self._log(f"  DTW distance penalty k={self.dtw_distance_penalty_k}")
        self._log(f"  Top-K candidates: {self.top_k_candidates}")

        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)

        # Store all candidates for global optimization
        all_candidates = []  # List of lists: [(v2_frame, weighted_score, raw_inliers), ...]
        frame_info = []  # Store v1_frame and dtw_prediction for each processed frame

        v1_frame_idx = 0
        processed_count = 0

        while True:
            ret1, frame1 = cap1.read()
            if not ret1:
                break

            # Apply sample rate
            if v1_frame_idx % self.feature_sample_rate != 0:
                v1_frame_idx += 1
                continue

            # Check if this frame is within the DTW-aligned region
            if v1_frame_idx not in dtw_mapping:
                v1_frame_idx += 1
                continue

            # Get DTW prediction
            expected_v2_frame = dtw_mapping[v1_frame_idx]

            # Define strict search window around DTW prediction
            search_start = max(0, expected_v2_frame - self.search_window)
            search_end = min(total_frames_v2 - 1, expected_v2_frame + self.search_window)

            # Extract AKAZE features for frame1 (once)
            kp1, des1 = self._compute_features(frame1)

            # Collect ALL candidates with their weighted scores
            candidates = []

            if des1 is not None and len(kp1) >= 4:
                for candidate_idx in range(search_start, search_end + 1):
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
                    ret2, frame2 = cap2.read()
                    if not ret2:
                        continue

                    raw_inliers = self._match_frames(frame1, frame2, kp1, des1)
                    weighted_score = self._compute_weighted_score(
                        raw_inliers, candidate_idx, expected_v2_frame
                    )

                    candidates.append((candidate_idx, weighted_score, raw_inliers))

            # Sort by weighted score (descending) and keep top-K
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:self.top_k_candidates]

            # If no good candidates, add DTW prediction as fallback
            if not top_candidates or top_candidates[0][2] < self.min_inliers_threshold:
                # Add DTW prediction with score 0
                top_candidates.insert(0, (expected_v2_frame, 0.0, 0))

            all_candidates.append(top_candidates)
            frame_info.append({
                'v1_frame': v1_frame_idx,
                'dtw_prediction': expected_v2_frame
            })

            processed_count += 1
            if self.verbose and processed_count % 100 == 0:
                self._log(f"  Processed {processed_count} frames...")

            v1_frame_idx += 1

        cap1.release()
        cap2.release()

        # ============================================================
        # PHASE C: GLOBAL OPTIMIZATION (Monotonic Path)
        # ============================================================
        if self.enable_global_optimization and len(all_candidates) > 0:
            self._log("\n[Phase C] Global optimization with monotonicity constraint...")

            optimized = self._find_monotonic_path(all_candidates)

            # Merge with frame info
            results = []
            for i, opt in enumerate(optimized):
                info = frame_info[i]
                raw_inliers = opt['score']

                # Determine source
                if raw_inliers < self.min_inliers_threshold:
                    source = "dtw_fallback"
                else:
                    source = "akaze_refined"

                results.append({
                    'v1_frame': info['v1_frame'],
                    'v2_frame': opt['v2_frame'],
                    'score': raw_inliers,
                    'source': source,
                    'dtw_prediction': info['dtw_prediction'],
                    'weighted_score': opt.get('weighted_score', 0)
                })
        else:
            # Fallback: use best candidate directly (old behavior)
            self._log("\n[Phase C] Selecting best candidates (no global optimization)...")

            results = []
            for i, candidates in enumerate(all_candidates):
                info = frame_info[i]

                if candidates and candidates[0][2] >= self.min_inliers_threshold:
                    best = candidates[0]
                    source = "akaze_refined"
                else:
                    best = (info['dtw_prediction'], 0.0, 0)
                    source = "dtw_fallback"

                results.append({
                    'v1_frame': info['v1_frame'],
                    'v2_frame': best[0],
                    'score': best[2],
                    'source': source,
                    'dtw_prediction': info['dtw_prediction'],
                    'weighted_score': best[1]
                })

        # ============================================================
        # PHASE D: POST-PROCESSING (Smoothing)
        # ============================================================
        if self.enable_smoothing and len(results) > 0:
            self._log("\n[Phase D] Applying smoothing to remove residual jitter...")
            results = self._apply_smoothing(results)

        # ============================================================
        # STATISTICS
        # ============================================================
        dtw_fallback_count = sum(1 for r in results if r['source'] == 'dtw_fallback')
        akaze_refined_count = len(results) - dtw_fallback_count

        # Check monotonicity
        monotonic_violations = 0
        for i in range(1, len(results)):
            if results[i]['v2_frame'] < results[i-1]['v2_frame']:
                monotonic_violations += 1

        # ============================================================
        # SUMMARY
        # ============================================================
        self._log("\n" + "=" * 60)
        self._log("ALIGNMENT COMPLETE (v2 - Anti-Trembling)")
        self._log("=" * 60)
        self._log(f"  Total matches: {len(results)}")
        self._log(f"  AKAZE refined: {akaze_refined_count} ({100*akaze_refined_count/max(1,len(results)):.1f}%)")
        self._log(f"  DTW fallback:  {dtw_fallback_count} ({100*dtw_fallback_count/max(1,len(results)):.1f}%)")
        self._log(f"  Monotonicity violations: {monotonic_violations}")
        if self.enable_global_optimization:
            self._log(f"  Global optimization: ENABLED")
        if self.enable_smoothing:
            self._log(f"  Smoothing: ENABLED (window={self.smoothing_window})")

        return results


def align_videos_hybrid(
    video1_path: str,
    video2_path: str,
    algorithm: str = "AKAZE",
    dtw_sample_rate: int = 5,
    feature_sample_rate: int = 1,
    search_window: int = 10,
    min_inliers: int = 4,
    # New v2 parameters
    dtw_distance_penalty_k: float = 0.15,
    top_k_candidates: int = 5,
    enable_global_optimization: bool = True,
    enable_smoothing: bool = True,
    smoothing_window: int = 5,
    verbose: bool = True
) -> list:
    """
    Convenience function for hybrid video alignment (v2 - Anti-Trembling).

    Args:
        video1_path: Path to reference video (J-1)
        video2_path: Path to target video (J)
        algorithm: Feature matching algorithm ("AKAZE", "BRISK", or "ORB")
        dtw_sample_rate: Sample rate for DTW phase (higher = faster)
        feature_sample_rate: Sample rate for feature matching phase
        search_window: Half-window around DTW prediction for feature search
        min_inliers: Minimum RANSAC inliers to accept match (else fallback to DTW)

        # New v2 parameters:
        dtw_distance_penalty_k: Exponential decay for DTW distance (0.1-0.3 recommended)
        top_k_candidates: Number of candidates to keep per frame
        enable_global_optimization: Use DP for monotonic path optimization
        enable_smoothing: Apply post-processing smoothing
        smoothing_window: Window size for smoothing
        verbose: Print progress

    Returns:
        List of match dictionaries
    """
    aligner = HybridAligner(
        algorithm=algorithm,
        dtw_sample_rate=dtw_sample_rate,
        feature_sample_rate=feature_sample_rate,
        search_window=search_window,
        min_inliers_threshold=min_inliers,
        dtw_distance_penalty_k=dtw_distance_penalty_k,
        top_k_candidates=top_k_candidates,
        enable_global_optimization=enable_global_optimization,
        enable_smoothing=enable_smoothing,
        smoothing_window=smoothing_window,
        verbose=verbose
    )
    return aligner.align_videos(video1_path, video2_path)


if __name__ == "__main__":
    import argparse

    # Check if being run with video arguments (legacy/direct usage)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # User provided video arguments - use direct alignment mode
        parser = argparse.ArgumentParser(description="Hybrid Coarse-to-Fine Video Alignment (Direct Mode)")
        parser.add_argument("video1", help="Path to reference video (J-1)")
        parser.add_argument("video2", help="Path to target video (J)")
        parser.add_argument("--algorithm", choices=["AKAZE", "BRISK", "ORB"], default="AKAZE",
                            help="Feature matching algorithm (default: AKAZE)")
        parser.add_argument("--dtw-sample-rate", type=int, default=5, help="Sample rate for DTW (default: 5)")
        parser.add_argument("--feature-sample-rate", type=int, default=1, help="Sample rate for feature matching (default: 1)")
        parser.add_argument("--search-window", type=int, default=10, help="Search window around DTW prediction (default: 10)")
        parser.add_argument("--min-inliers", type=int, default=4, help="Min RANSAC inliers (default: 4)")
        parser.add_argument("--output", default="hybrid_alignment.csv", help="Output CSV file")
        parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

        args = parser.parse_args()

        results = align_videos_hybrid(
            args.video1,
            args.video2,
            algorithm=args.algorithm,
            dtw_sample_rate=args.dtw_sample_rate,
            feature_sample_rate=args.feature_sample_rate,
            search_window=args.search_window,
            min_inliers=args.min_inliers,
            verbose=not args.quiet
        )

        # Save results to CSV
        import csv
        with open(args.output, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                print(f"\nResults saved to {args.output}")
            else:
                print("\nNo matches found.")
    else:
        # No video arguments provided - show helpful message
        print("=" * 70)
        print("Hybrid Video Alignment - Module")
        print("=" * 70)
        print()
        print("ℹ️  This is the internal alignment module.")
        print()
        print("📌 To use the interactive interface with dataset folder:")
        print()
        print("   python combined_method/run_alignment_hybrid.py")
        print()
        print("   This will guide you through selecting which Plan to process.")
        print()
        print("─" * 70)
        print()
        print("💡 Alternative: Direct mode (for single video pairs)")
        print()
        print("   python -m combined_method.hybrid_alignment <video1> <video2>")
        print()
        print("   Example:")
        print("   python -m combined_method.hybrid_alignment video1.mp4 video2.mp4")
        print()
        print("─" * 70)
        print()
        print("📖 For more information, see: combined_method/README.md")
        print()
