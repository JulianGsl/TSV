"""
Hybrid Video Alignment — Coarse-to-Fine Pipeline

Aligns two videos of the same railway trajectory recorded on different
days at different speeds. The pipeline runs six phases:

    Phase 0  : Preprocessing             (luminance histogram matching V2 -> V1)
    Phase A.1: Feature extraction        (per-frame gradient orientation
                                          histogram, `grad_hist_16`)
    Phase A.2: Open-end DTW              (adaptive step penalty, plus a
                                          per-waypoint confidence side-car)
    Phase B  : Feature matching          (AKAZE/BRISK/ORB + KNN + Lowe's
                                          ratio + spatial-coherence filter
                                          + RANSAC homography)
    Phase B' : Rescue                    (opportunistic override of the
                                          DTW prediction in low-confidence
                                          zones, when phase B produces a
                                          consistent contradicting signal)
    Phase C  : Global optimization       (top-K candidates per frame, then
                                          monotonic DP through the trellis)
    Phase D  : Smoothing                 (weighted moving average +
                                          monotonicity re-enforcement)

Design principles
-----------------
- DTW provides a globally robust skeleton; feature matching refines it
  locally; the DP step enforces a monotonic non-trembling output.
- The strict ±N search window in phase B is never widened on failure.
  When phase B cannot find a match, the DTW prediction is kept as-is
  (`source = "dtw_fallback"`).
- The rescue (phase B') is the single carefully circumscribed exception
  to that rule: it requires multiple consecutive frames of consistent
  evidence before overriding DTW.

Module organisation
-------------------
- This file holds the `HybridAligner` class and the `align_videos_hybrid`
  thin wrapper.
- Side modules: `preprocessing.py`, `dtw_confidence.py`,
  `phase_b_rescue.py`, all in this package.
- DTW core (`compute_dtw`) is imported from `combined_method.dtw_core`
  (see import below); `Old_version/new_method/` only re-exports it as a
  backward-compatibility shim.
"""

import cv2
import numpy as np
import sys
import os
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combined_method.dtw_core import compute_dtw

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
        dtw_step_penalty: float = 0.3,
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
        # Preprocessing & rescue (new) ---------------------------------
        enable_preprocessing: bool = True,
        enable_phase_b_rescue: bool = True,
        # ----------------------------------------------------------------
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

        # Preprocessing + rescue knobs
        self.enable_preprocessing = enable_preprocessing
        self.enable_phase_b_rescue = enable_phase_b_rescue
        self.preprocessor = None              # built lazily once video paths are known
        self.dtw_confidence_per_frame = None  # filled after Phase A
        self.rescue_corrections = None        # filled after rescue analysis

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

    def _maybe_preprocess(self, frame: np.ndarray, video_path: str) -> np.ndarray:
        """Apply luminance LUT if preprocessor is set up. V1 is left untouched
        by convention; V2 is remapped onto V1's luminance distribution."""
        if self.preprocessor is None:
            return frame
        if video_path == self.preprocessor.v1_path:
            return self.preprocessor.apply_to_v1(frame)
        if video_path == self.preprocessor.v2_path:
            return self.preprocessor.apply_to_v2(frame)
        return frame

    def _extract_dtw_features_raw(self, video_path: str) -> tuple:
        """
        Phase A.1: Extract per-frame gradient-orientation histograms for DTW.

        Selected by sweep over (intensity, hue, RGB, grid-intensity, gradient,
        combined, lowres-gray) × (raw, per-video z-score, joint z-score) ×
        step-penalty on Plan2 ground truth: gradient histograms with per-video
        z-score gave mean abs error ≈ 33 frames, vs ≈ 128 for intensity. See
        combined_method/dtw_diagnostic/sweep.py.

        Why gradients beat intensity: gradient orientation captures geometric
        structure (rail edges, sleepers, fixed lineside structures) which is
        intrinsically tied to a geographic position. Raw intensity also
        encodes lighting/exposure, which differs between V1 and V2 takes.

        We compute Sobel gx, gy on the lower 2/3 of the frame (sky cropped),
        accumulate magnitude into a 16-bin orientation histogram over [0, 360),
        and L1-normalise the histogram per frame.

        Returns:
            features: float32 array of shape (n_sampled, 16)
            frame_indices: list of original frame indices
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        feats = []
        frame_indices = []
        idx = -1
        while True:
            ret, frame = cap.read()
            idx += 1
            if not ret:
                break
            if idx % self.dtw_sample_rate != 0:
                continue
            frame = self._maybe_preprocess(frame, video_path)     # luminance LUT (V2 only)
            h = frame.shape[0]
            crop = frame[h // 3:, :]                              # drop sky band
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            ang = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)  # 0..360
            hist, _ = np.histogram(ang, bins=16, range=(0, 360), weights=mag)
            hist = hist.astype(np.float32)
            s = hist.sum()
            if s > 1e-6:
                hist = hist / s                                    # L1-normalise per frame
            feats.append(hist)
            frame_indices.append(idx)

        cap.release()
        return np.asarray(feats, dtype=np.float32), frame_indices

    @staticmethod
    def _to_cumulative_features(features1: np.ndarray, features2: np.ndarray) -> tuple:
        """
        Per-video z-score normalisation.

        Each video is z-scored using its own per-dimension mean/std. The point
        is to remove the per-video baseline (lighting, exposure, gain settings
        that differ between recording days) while preserving the *relative*
        variations of the descriptor along the trajectory — which are what
        DTW should align.

        Joint z-score (using stats from the concatenation of both videos) was
        also tested and consistently underperforms per-video z-score on the
        Plan2 ground truth. See sweep.py for the full evaluation.
        """
        def z(x: np.ndarray) -> np.ndarray:
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True) + 1e-6
            return (x - mu) / sd
        return z(features1), z(features2)

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

        Returns lightweight (N, 2) float32 keypoint coordinates rather than
        cv2.KeyPoint objects, which lets us cache features for thousands of
        V2 frames without exploding memory.

        Returns:
            kp_pts: np.ndarray of shape (N, 2) with (x, y) coordinates
            descriptors: numpy array of descriptors (or None)
        """
        if frame is None:
            return np.empty((0, 2), dtype=np.float32), None

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if keypoints:
            kp_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        else:
            kp_pts = np.empty((0, 2), dtype=np.float32)
        return kp_pts, descriptors

    def _precompute_v2_features(self, video_path: str, max_frame_idx: int) -> list:
        """
        Pre-extract feature descriptors for V2 by sequential decode.

        Random seeking in H.264 forces the decoder back to the previous keyframe;
        with a search window of W, each V2 frame would otherwise be decoded and
        re-described O(W) times across overlapping windows. Reading V2 once
        end-to-end and caching (kp_pts, des) per frame eliminates both costs.

        Args:
            video_path: Path to V2
            max_frame_idx: Last frame index we will need (inclusive)

        Returns:
            List indexed by frame number; entries are (kp_pts, des) tuples.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        features = []
        idx = 0
        while idx <= max_frame_idx:
            ret, frame = cap.read()
            if not ret:
                break
            features.append(self._compute_features(frame))
            idx += 1

        cap.release()
        return features

    def _filter_matches_spatial(self, matches: list, kp1: np.ndarray, kp2: np.ndarray, frame_width: int) -> list:
        """
        Filter matches by spatial coherence (left/right half consistency).

        For rail track videos, matching features should generally be on the same
        side of the image (parallax effect). This filters out physically impossible matches.

        Args:
            matches: List of cv2.DMatch objects
            kp1: (N1, 2) keypoint coordinates from frame 1
            kp2: (N2, 2) keypoint coordinates from frame 2
            frame_width: Width of the frames

        Returns:
            Filtered list of matches with spatial coherence
        """
        if not matches:
            return matches

        mid_x = frame_width / 2
        filtered = []

        for m in matches:
            pt1_left = kp1[m.queryIdx, 0] < mid_x
            pt2_left = kp2[m.trainIdx, 0] < mid_x
            if pt1_left == pt2_left:
                filtered.append(m)

        return filtered

    def _match_frames(
        self,
        kp1: np.ndarray,
        des1: np.ndarray,
        kp2: np.ndarray,
        des2: np.ndarray,
        frame_width: int
    ) -> int:
        """
        Match two pre-extracted feature sets and return inlier count.

        Args:
            kp1: (N1, 2) keypoint coords for frame1
            des1: descriptors for frame1
            kp2: (N2, 2) keypoint coords for frame2
            des2: descriptors for frame2
            frame_width: Width of the frames (for spatial filter)

        Returns:
            Number of RANSAC inliers (0 if matching failed)
        """
        if des1 is None or des2 is None:
            return 0
        if len(kp1) < 4 or len(kp2) < 4:
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
        good_matches = self._filter_matches_spatial(good_matches, kp1, kp2, frame_width)

        if len(good_matches) < 4:
            return 0

        # RANSAC homography estimation
        src_pts = kp1[[m.queryIdx for m in good_matches]].reshape(-1, 1, 2)
        dst_pts = kp2[[m.trainIdx for m in good_matches]].reshape(-1, 1, 2)

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

        Uses a weighted average filter that respects:
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
        # PHASE 0: PREPROCESSING (luminance histogram matching)
        # ============================================================
        if self.enable_preprocessing:
            self._log("\n[Phase 0] Building luminance LUT (V2 → V1)...")
            from combined_method.preprocessing import Preprocessor
            self.preprocessor = Preprocessor(video1_path, video2_path,
                                             n_samples=60, enabled=True)
            self._log("  LUT built. V2 luminance will be matched to V1's distribution.")

        # ============================================================
        # PHASE A: MACRO-SYNCHRONIZATION (DTW Skeleton)
        # ============================================================
        self._log("\n[Phase A] Extracting gradient-orientation histogram features for DTW...")

        # A.1: Extract raw per-frame gradient-histogram features, then apply
        # per-video z-score normalisation. Z-scoring removes the per-video
        # baseline (lighting, exposure, gain differences between recording
        # days) while preserving the relative descriptor variations along the
        # track, which are what DTW should align.
        features1_raw, indices1 = self._extract_dtw_features_raw(video1_path)
        features2_raw, indices2 = self._extract_dtw_features_raw(video2_path)
        features1, features2 = self._to_cumulative_features(features1_raw, features2_raw)

        self._log(f"  Video 1: {len(features1)} sampled frames")
        self._log(f"  Video 2: {len(features2)} sampled frames")

        # A.2: Compute Open-End DTW
        self._log("\n[Phase A] Computing Open-End DTW alignment...")
        path, cost_matrix, dist_matrix = compute_dtw(
            features1, features2,
            metric='euclidean',
            step_penalty=self.dtw_step_penalty,
            open_end=True
        )

        # Expose DTW artefacts so callers (e.g. the runner's visualisation
        # phase) can reuse them instead of recomputing the whole DTW.
        self.dtw_cost_matrix = cost_matrix      # accumulated cost (DP result)
        self.dtw_dist_matrix = dist_matrix      # raw pairwise distance matrix
        self.dtw_path = path
        self.dtw_indices1 = indices1
        self.dtw_indices2 = indices2
        self.dtw_features1 = features1
        self.dtw_features2 = features2

        # Per-waypoint and per-V1-frame DTW confidence. Cheap to compute and
        # used both for diagnostics and by the optional Phase B rescue logic.
        from combined_method.dtw_confidence import (
            compute_dtw_confidence, project_to_frames,
        )
        self.dtw_confidence_per_waypoint = compute_dtw_confidence(
            dist_matrix, np.asarray(path, dtype=np.int32)
        )

        # Get actual endpoint
        end_i, end_j = path[-1]
        v1_endpoint = indices1[end_i]   # last V1 frame in the DTW path (original index)
        v2_endpoint = indices2[end_j]   # last V2 frame in the DTW path (original index)
        self._log(f"  DTW path length: {len(path)} points")
        self._log(f"  DTW endpoint: V1[{v1_endpoint}] <-> V2[{v2_endpoint}]")

        # Get total frame counts
        cap1_temp = cv2.VideoCapture(video1_path)
        cap2_temp = cv2.VideoCapture(video2_path)
        total_frames_v1 = int(cap1_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_v2 = int(cap2_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap1_temp.release()
        cap2_temp.release()

        self._log(f"  Total frames: V1={total_frames_v1}, V2={total_frames_v2}")

        # Project per-waypoint confidence onto every original V1 frame; used by
        # the rescue logic and exposed for diagnostics.
        self.dtw_confidence_per_frame = project_to_frames(
            np.asarray(path, dtype=np.int32),
            self.dtw_confidence_per_waypoint,
            np.asarray(indices1, dtype=np.int32),
            total_frames_v1,
        )
        low_conf = float((self.dtw_confidence_per_frame < 0.25).mean())
        self._log(f"  DTW confidence: mean={self.dtw_confidence_per_frame.mean():.3f}, "
                  f"{low_conf * 100:.1f}% of V1 frames below 0.25")

        # Detect which video's geography terminated first (open-end DTW).
        # If the path stopped on the last column (V2 ran out of geography),
        # we should not process V1 frames beyond v1_endpoint, since V2 has
        # no more geographic content to match against.
        v2_finished_first = v2_endpoint >= total_frames_v2 - self.dtw_sample_rate
        v1_finished_first = v1_endpoint >= total_frames_v1 - self.dtw_sample_rate

        if v2_finished_first and not v1_finished_first:
            self._log(f"  V2 reached its end first (at V1 frame {v1_endpoint}). "
                      f"Will stop V1 iteration there.")
        elif v1_finished_first and not v2_finished_first:
            self._log(f"  V1 reached its end first (at V2 frame {v2_endpoint}). "
                      f"V2 has more geographic content beyond.")
        else:
            self._log(f"  Both videos reached their ends simultaneously.")

        # A.3: Build interpolated mapping (only for V1 frames within the DTW path)
        # Cap the mapping at v1_endpoint to avoid extrapolating beyond the
        # geographic overlap.
        max_v1_to_map = min(total_frames_v1, v1_endpoint + 1)
        dtw_mapping = self._build_dtw_mapping(path, indices1, indices2, max_v1_to_map)
        self._log(f"  DTW mapping built for {len(dtw_mapping)} frames "
                  f"(out of {total_frames_v1} V1 frames)")

        # ============================================================
        # PHASE B: MICRO-ALIGNMENT (AKAZE Refinement with Top-K)
        # ============================================================
        self._log(f"\n[Phase B] {self.algorithm} micro-alignment with DTW distance penalty...")
        self._log(f"  DTW distance penalty k={self.dtw_distance_penalty_k}")
        self._log(f"  Top-K candidates: {self.top_k_candidates}")

        # Pre-extract V2 features by sequential read. The fine phase will look
        # up cached (kp, des) by frame index rather than seeking and re-running
        # the detector on the same frame across overlapping windows.
        max_v2_needed = min(
            total_frames_v2 - 1,
            max(dtw_mapping.values()) + self.search_window
        )
        self._log(f"\n[Phase B] Pre-extracting V2 features for frames "
                  f"[0..{max_v2_needed}] (sequential read)...")
        v2_features = self._precompute_v2_features(video2_path, max_v2_needed)
        self._log(f"  Cached features for {len(v2_features)} V2 frames")

        cap1 = cv2.VideoCapture(video1_path)

        # Store all candidates for global optimization
        all_candidates = []  # List of lists: [(v2_frame, weighted_score, raw_inliers), ...]
        frame_info = []  # Store v1_frame and dtw_prediction for each processed frame

        v1_frame_idx = 0
        processed_count = 0
        n_v2_cached = len(v2_features)

        while True:
            ret1, frame1 = cap1.read()
            if not ret1:
                break

            # Stop iterating V1 once we have left the geographic overlap
            # (i.e. V2 has no more matching content beyond v1_endpoint).
            if v1_frame_idx > v1_endpoint:
                self._log(f"  Reached DTW V1 endpoint ({v1_endpoint}); "
                          f"stopping at V1 frame {v1_frame_idx}.")
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
            search_end = min(n_v2_cached - 1, expected_v2_frame + self.search_window)

            # Extract features for frame1 (once)
            kp1, des1 = self._compute_features(frame1)
            frame_width = frame1.shape[1] if frame1.ndim >= 2 else 640

            # Collect ALL candidates with their weighted scores
            candidates = []

            if des1 is not None and len(kp1) >= 4:
                for candidate_idx in range(search_start, search_end + 1):
                    kp2, des2 = v2_features[candidate_idx]

                    raw_inliers = self._match_frames(kp1, des1, kp2, des2, frame_width)
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
        # Free the V2 feature cache before the heavy DP phase.
        del v2_features

        # Snapshot the alignment as "DTW only": the raw Phase-A prediction with
        # no influence from feature matching. Built on the same V1 frames as the
        # other snapshots so the rendered videos line up frame-for-frame and the
        # user can compare DTW vs DTW+features vs final side by side.
        self.matches_dtw_only = [{
            'v1_frame': info['v1_frame'],
            'v2_frame': info['dtw_prediction'],
            'score': 0,
            'source': 'dtw',
            'dtw_prediction': info['dtw_prediction'],
            'weighted_score': 0.0,
        } for info in frame_info]

        # Snapshot the alignment as "DTW + feature matching, pre-rectification":
        # the best candidate per frame from the fine phase, with a DTW fallback
        # when no candidate hits min_inliers_threshold. Exposed on the instance
        # so the runner can render an intermediate video for visual comparison
        # against the post-Phase-C/D output.
        self.matches_pre_rectification = []
        for i, candidates in enumerate(all_candidates):
            info = frame_info[i]
            if candidates and candidates[0][2] >= self.min_inliers_threshold:
                best = candidates[0]
                source = f"{self.algorithm.lower()}_refined"
            else:
                best = (info['dtw_prediction'], 0.0, 0)
                source = "dtw_fallback"
            self.matches_pre_rectification.append({
                'v1_frame': info['v1_frame'],
                'v2_frame': best[0],
                'score': best[2],
                'source': source,
                'dtw_prediction': info['dtw_prediction'],
                'weighted_score': best[1],
            })

        # ============================================================
        # PHASE B': RESCUE — override DTW where the matcher gave coherent
        # contradicting evidence inside a low-confidence zone
        # ============================================================
        if self.enable_phase_b_rescue and self.dtw_confidence_per_frame is not None:
            from combined_method.phase_b_rescue import (
                RescueConfig, detect_rescue_corrections,
            )
            rescue_inputs = []
            for i, candidates in enumerate(all_candidates):
                info = frame_info[i]
                if candidates and candidates[0][2] >= 1:
                    v2_raw = int(candidates[0][0])
                    score = int(candidates[0][2])
                else:
                    v2_raw = int(info['dtw_prediction'])
                    score = 0
                rescue_inputs.append({
                    'v1_frame': info['v1_frame'],
                    'dtw_prediction': int(info['dtw_prediction']),
                    'v2_frame_raw': v2_raw,
                    'score': score,
                })

            corrections = detect_rescue_corrections(
                rescue_inputs, self.dtw_confidence_per_frame,
            )
            self.rescue_corrections = dict(corrections)

            if corrections:
                # Shift DTW prediction in frame_info so Phase C sees the
                # corrected anchor when it weighs candidates.
                for info in frame_info:
                    v1 = info['v1_frame']
                    if v1 in corrections:
                        info['dtw_prediction'] = int(info['dtw_prediction'] + corrections[v1])
                # Re-emit matches_pre_rectification with corrected predictions
                # so diagnostic videos reflect the rescue.
                for i, m in enumerate(self.matches_pre_rectification):
                    info = frame_info[i]
                    m['dtw_prediction'] = info['dtw_prediction']
                self._log(f"\n[Phase B'] Rescue corrected {len(corrections)} V1 frames "
                          f"(consistent matcher deviation in low-confidence zones)")
            else:
                self._log("\n[Phase B'] No rescue applied — no consistent matcher deviation detected")

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
                    source = f"{self.algorithm.lower()}_refined"

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
                    source = f"{self.algorithm.lower()}_refined"
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
        refined_count = len(results) - dtw_fallback_count

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
        self._log(f"  {self.algorithm} refined: {refined_count} ({100*refined_count/max(1,len(results)):.1f}%)")
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
