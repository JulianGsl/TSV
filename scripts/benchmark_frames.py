import cv2
import numpy as np
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.alignment.algorithms import orb, brisk, akaze

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths to the videos
VIDEO1_PATH = "./data/PlanTest/video1.mp4"
VIDEO2_PATH = "./data/PlanTest/video2.mp4"

# Output directory for visualizations
BENCHMARK_OUTPUT_DIR = "./benchmark_results"

# --- Ranking Benchmark Configuration ---
# Focus on testing one frame from Video 1 against multiple candidates in Video 2

REFERENCE_FRAME_IDX = 30     # The frame in Video 1 we want to match
CORRECT_MATCH_IDX = 30       # The correct corresponding frame in Video 2

# Distractor offsets: Frames relative to CORRECT_MATCH_IDX to test against
# We want to see if the algorithm scores the correct frame (offset 0) higher than these.
DISTRACTOR_OFFSETS = [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]

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

def save_visualization(img1, kp1, img2, kp2, good_matches, mask, algo_name, v1_idx, v2_idx, rank, is_correct):
    """Generates and saves a side-by-side comparison image with matches."""
    if not os.path.exists(BENCHMARK_OUTPUT_DIR):
        os.makedirs(BENCHMARK_OUTPUT_DIR)

    # Prepare mask for drawing (convert RANSAC mask to list of matches)
    matches_mask = None
    if mask is not None:
        matches_mask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0), # Green for matches
                       singlePointColor=None,
                       matchesMask=matches_mask, # Draw only inliers
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Draw matches
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    # Add text label
    status = "CORRECT" if is_correct else "DISTRACTOR"
    color = (0, 255, 0) if is_correct else (0, 0, 255)

    text = f"{algo_name}: V1[{v1_idx}] - V2[{v2_idx}] | Rank: {rank} | Inliers: {np.sum(mask) if mask is not None else 0} | {status}"
    cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Save
    filename = f"{algo_name}_rank{rank:02d}_{status}_v2-{v2_idx}.jpg"
    filepath = os.path.join(BENCHMARK_OUTPUT_DIR, filename)
    cv2.imwrite(filepath, result_img)

def evaluate_match(algo_module, img1, img2):
    """
    Computes matching score between two images using the specified algorithm.
    Returns a dictionary with metrics and data for visualization.
    """
    # 1. Compute Features
    kp1, des1 = algo_module.compute_features(img1)
    kp2, des2 = algo_module.compute_features(img2)

    empty_result = {
        "kp1": len(kp1),
        "kp2": len(kp2),
        "matches": 0,
        "inliers": 0,
        "score": 0.0,
        "good_matches": [],
        "mask": None,
        "keypoints": (kp1, kp2)
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
    mask = None

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
            mask = mask_arr

    return {
        "kp1": len(kp1),
        "kp2": len(kp2),
        "matches": match_count,
        "inliers": inlier_count,
        "score": inlier_count,
        "good_matches": good_matches,
        "mask": mask,
        "keypoints": (kp1, kp2)
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(f"Benchmark Script: Discrimination Ranking")
    print(f"Video 1: {VIDEO1_PATH}")
    print(f"Video 2: {VIDEO2_PATH}")
    print(f"Output:  {BENCHMARK_OUTPUT_DIR}")
    print(f"Reference Frame (V1): {REFERENCE_FRAME_IDX}")
    print(f"Correct Match (V2):   {CORRECT_MATCH_IDX}")
    print("=" * 60)

    if not os.path.exists(VIDEO1_PATH) or not os.path.exists(VIDEO2_PATH):
        print("Error: Videos not found.")
        return

    # Load Reference Frame
    ref_img = get_frame(VIDEO1_PATH, REFERENCE_FRAME_IDX)
    if ref_img is None:
        print("Error: Could not load reference frame.")
        return

    # Build list of candidate frames (Correct + Distractors)
    candidates = []
    # Add correct match
    candidates.append({"offset": 0, "idx": CORRECT_MATCH_IDX, "type": "CORRECT"})
    # Add distractors
    for offset in DISTRACTOR_OFFSETS:
        idx = CORRECT_MATCH_IDX + offset
        if idx >= 0: # Ensure valid frame index
            candidates.append({"offset": offset, "idx": idx, "type": "DISTRACTOR"})

    # Clean output dir
    if os.path.exists(BENCHMARK_OUTPUT_DIR):
        for f in os.listdir(BENCHMARK_OUTPUT_DIR):
            os.remove(os.path.join(BENCHMARK_OUTPUT_DIR, f))

    # Run each algorithm
    for algo_name, algo_module in ALGORITHMS.items():
        print(f"\n--- Testing Algorithm: {algo_name} ---")

        algo_results = []

        for cand in candidates:
            v2_idx = cand["idx"]
            cand_img = get_frame(VIDEO2_PATH, v2_idx)

            if cand_img is None:
                continue

            res = evaluate_match(algo_module, ref_img, cand_img)

            # Store full result
            cand_result = {
                "v2_idx": v2_idx,
                "type": cand["type"],
                "offset": cand["offset"],
                "inliers": res["inliers"],
                "data": res,
                "img": cand_img
            }
            algo_results.append(cand_result)
            print(f"  > V2[{v2_idx:<3}] ({cand['type']:<10}): {res['inliers']} inliers")

        # Sort results by score (descending)
        algo_results.sort(key=lambda x: x["inliers"], reverse=True)

        print(f"\n  Ranking for {algo_name}:")
        print(f"  {'Rank':<5} | {'Frame':<5} | {'Type':<10} | {'Score':<5} | {'Delta Score'}")
        print("  " + "-" * 50)

        # Calculate score difference from top match
        top_score = algo_results[0]["inliers"] if algo_results else 0

        correct_found_at_rank = -1

        for rank, res in enumerate(algo_results, 1):
            delta = res["inliers"] - top_score
            print(f"  {rank:<5} | {res['v2_idx']:<5} | {res['type']:<10} | {res['inliers']:<5} | {delta}")

            if res["type"] == "CORRECT":
                correct_found_at_rank = rank

            # Visualize Top 3 and the Correct one (if not in top 3)
            should_visualize = (rank <= 3) or (res["type"] == "CORRECT")

            if should_visualize and res["inliers"] > 0:
                kp1, kp2 = res["data"]["keypoints"]
                save_visualization(ref_img, kp1, res["img"], kp2,
                                   res["data"]["good_matches"], res["data"]["mask"],
                                   algo_name, REFERENCE_FRAME_IDX, res["v2_idx"],
                                   rank, res["type"] == "CORRECT")

        if correct_found_at_rank == 1:
            print(f"\n  [SUCCESS] {algo_name} correctly identified the true match as Rank 1.")
        else:
            print(f"\n  [FAILURE] {algo_name} ranked true match at {correct_found_at_rank}.")

    print(f"\nVisualizations saved to {BENCHMARK_OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
