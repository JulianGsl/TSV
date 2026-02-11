import cv2
import numpy as np
import sys
import os
import random

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.alignment.algorithms import orb, brisk, akaze

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths to the videos
VIDEO1_PATH = "./data/PlanTest/video1.mp4"
VIDEO2_PATH = "./data/PlanTest/video2.mp4"

# --- Ranking Benchmark Configuration ---
# Focus on testing one frame from Video 1 against multiple candidates in Video 2

REFERENCE_FRAME_IDX = 30     # The frame in Video 1 we want to match
CORRECT_MATCH_IDX = 30       # The correct corresponding frame in Video 2

# Distractor Configuration
# We will generate random distractors for each run
NUM_CLOSE_DISTRACTORS = 5   # Number of frames close to the correct match
CLOSE_RANGE = 10            # +/- frames for close distractors (e.g., +/- 10)
NUM_FAR_DISTRACTORS = 5     # Number of frames far from the correct match
FAR_RANGE = 100             # +/- frames for far distractors (e.g., +/- 100)

# Scoring Parameters (from src/alignment/core.py)
# We disable the distance penalty for benchmarking to test pure visual discrimination.
# If an algorithm ranks a distractor higher than the correct frame, it has failed visually.
BASE_PENALTY = 1.0
PENALTY_FACTOR = 0.0 # Disabled penalty (was BASE_PENALTY * 1.0)

# Algorithms to test
ALGORITHMS = {
    "ORB": orb,
    "BRISK": brisk,
    "AKAZE": akaze
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_frame(video_path, frame_idx):
    """Loads a specific frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_idx} from {video_path}")
        return None
    return frame

def evaluate_match(algo_module, img1, img2, v2_idx, expected_pos):
    """
    Computes matching score between two images using the specified algorithm.
    Includes Homography Scale Penalty and Distance Penalty logic from core.py.
    """
    # 1. Compute Features
    kp1, des1 = algo_module.compute_features(img1)
    kp2, des2 = algo_module.compute_features(img2)

    empty_result = {
        "inliers": 0,
        "weighted_score": -float('inf')
    }

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return empty_result

    # 2. Match Features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        knn_matches = bf.knnMatch(des1, des2, k=2)
    except Exception as e:
        print(f"Matching error: {e}")
        return empty_result

    # 3. Ratio Test
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

    match_count = len(good_matches)
    inlier_count = 0

    # 4. RANSAC (Homography)
    if match_count >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask_arr = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                     ransacReprojThreshold=2.5,
                                     maxIters=2000,
                                     confidence=0.999)
        if mask_arr is not None:
            inlier_count = int(np.sum(mask_arr))

            # --- HOMOGRAPHY SCALE PENALTY (from core.py) ---
            if M is not None and inlier_count > 0:
                try:
                    scale_x = np.sqrt(M[0,0]**2 + M[1,0]**2)
                    scale_y = np.sqrt(M[0,1]**2 + M[1,1]**2)
                    if scale_x < 0.7 or scale_x > 1.3 or scale_y < 0.7 or scale_y > 1.3:
                        print(f"  [Penalty] Suspicious Scale: x={scale_x:.2f}, y={scale_y:.2f}")
                        inlier_count = int(inlier_count * 0.5)
                except (ValueError, ZeroDivisionError, IndexError):
                    inlier_count = int(inlier_count * 0.7)

    # --- DISTANCE PENALTY (from core.py) ---
    # weighted = inlier_count - penalty
    dist = abs(v2_idx - expected_pos)
    penalty = dist * PENALTY_FACTOR
    weighted_score = inlier_count - penalty

    return {
        "inliers": inlier_count, # This is the penalized inlier count
        "weighted_score": weighted_score
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Set random seed for reproducibility across runs and algorithms
    random.seed(42)

    print(f"Benchmark Script: Discrimination Ranking (Exact Scoring)")
    print(f"Video 1: {VIDEO1_PATH}")
    print(f"Video 2: {VIDEO2_PATH}")
    print(f"Reference Frame (V1): {REFERENCE_FRAME_IDX}")
    print(f"Expected Match (V2):  {CORRECT_MATCH_IDX}")
    print(f"Penalty Factor:       {PENALTY_FACTOR:.2f}")
    print("=" * 60)

    if not os.path.exists(VIDEO1_PATH) or not os.path.exists(VIDEO2_PATH):
        print("Error: Videos not found.")
        return

    ref_img = get_frame(VIDEO1_PATH, REFERENCE_FRAME_IDX)
    if ref_img is None:
        print("Error: Could not load reference frame.")
        return

    # Build candidates
    candidates = []
    # 1. Add Correct Match
    candidates.append({"offset": 0, "idx": CORRECT_MATCH_IDX, "type": "CORRECT"})

    # 2. Add Random Close Distractors
    used_offsets = {0}
    for _ in range(NUM_CLOSE_DISTRACTORS):
        while True:
            offset = random.randint(-CLOSE_RANGE, CLOSE_RANGE)
            if offset not in used_offsets:
                used_offsets.add(offset)
                idx = CORRECT_MATCH_IDX + offset
                if idx >= 0:
                    candidates.append({"offset": offset, "idx": idx, "type": "DISTRACTOR (Close)"})
                break

    # 3. Add Random Far Distractors
    for _ in range(NUM_FAR_DISTRACTORS):
        while True:
            # Generate offset in [-FAR_RANGE, -CLOSE_RANGE] U [CLOSE_RANGE, FAR_RANGE]
            if random.random() < 0.5:
                offset = random.randint(-FAR_RANGE, -CLOSE_RANGE - 1)
            else:
                offset = random.randint(CLOSE_RANGE + 1, FAR_RANGE)

            if offset not in used_offsets:
                used_offsets.add(offset)
                idx = CORRECT_MATCH_IDX + offset
                if idx >= 0:
                    candidates.append({"offset": offset, "idx": idx, "type": "DISTRACTOR (Far)"})
                break

    for algo_name, algo_module in ALGORITHMS.items():
        print(f"\n--- Testing Algorithm: {algo_name} ---")

        algo_results = []

        for cand in candidates:
            v2_idx = cand["idx"]
            cand_img = get_frame(VIDEO2_PATH, v2_idx)

            if cand_img is None:
                continue

            # Pass v2_idx and expected_pos for weighted score calculation
            res = evaluate_match(algo_module, ref_img, cand_img, v2_idx, CORRECT_MATCH_IDX)

            cand_result = {
                "v2_idx": v2_idx,
                "type": cand["type"],
                "inliers": res["inliers"],
                "weighted_score": res["weighted_score"]
            }
            algo_results.append(cand_result)
            print(f"  > V2[{v2_idx:<3}] ({cand['type']:<10}): Inliers={res['inliers']}, Weighted={res['weighted_score']:.1f}")

        # Sort results by Weighted Score (descending)
        algo_results.sort(key=lambda x: x["weighted_score"], reverse=True)

        print(f"\n  Ranking for {algo_name}:")
        print(f"  {'Rank':<5} | {'Frame':<5} | {'Type':<10} | {'Inliers':<7} | {'Weighted':<8}")
        print("  " + "-" * 60)

        correct_found_at_rank = -1

        for rank, res in enumerate(algo_results, 1):
            print(f"  {rank:<5} | {res['v2_idx']:<5} | {res['type']:<10} | {res['inliers']:<7} | {res['weighted_score']:<8.1f}")
            if res["type"] == "CORRECT":
                correct_found_at_rank = rank

        if correct_found_at_rank == 1:
            print(f"\n  [SUCCESS] {algo_name} ranked correct match #1.")
        else:
            print(f"\n  [FAILURE] {algo_name} ranked correct match #{correct_found_at_rank}.")

if __name__ == "__main__":
    main()
