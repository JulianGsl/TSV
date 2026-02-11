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
# You can change these paths to your specific video files
VIDEO1_PATH = "./data/PlanTest/video1.mp4"
VIDEO2_PATH = "./data/PlanTest/video2.mp4"

# Manual Test Cases
# Format: (frame_video1, frame_video2, description)
# Add your manual frame pairs here
TEST_CASES = [
    (10, 10, "Same frame index (should match if synced)"),
    (30, 30, "Frame 30 vs Frame 30"),
    (30, 45, "Frame 30 vs Frame 45 (Simulating offset)"),
    (30, 100, "Large offset (Expected no match)")
]

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

def evaluate_match(algo_name, algo_module, img1, img2):
    """
    Computes matching score between two images using the specified algorithm.
    Returns a dictionary with metrics.
    """
    # 1. Compute Features
    kp1, des1 = algo_module.compute_features(img1)
    kp2, des2 = algo_module.compute_features(img2)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return {
            "kp1": len(kp1),
            "kp2": len(kp2),
            "matches": 0,
            "inliers": 0,
            "score": 0.0
        }

    # 2. Match Features
    # Use Hamming distance for binary descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        knn_matches = bf.knnMatch(des1, des2, k=2)
    except Exception as e:
        print(f"Matching error: {e}")
        return {
            "kp1": len(kp1),
            "kp2": len(kp2),
            "matches": 0,
            "inliers": 0,
            "score": 0.0
        }

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

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                     ransacReprojThreshold=2.5,
                                     maxIters=2000,
                                     confidence=0.999)
        if mask is not None:
            inlier_count = int(np.sum(mask))

    return {
        "kp1": len(kp1),
        "kp2": len(kp2),
        "matches": match_count,
        "inliers": inlier_count,
        "score": inlier_count # Using inliers as the primary score
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(f"Benchmark Script for Video Alignment Algorithms")
    print(f"Video 1: {VIDEO1_PATH}")
    print(f"Video 2: {VIDEO2_PATH}")
    print("=" * 60)

    # Validate videos exist
    if not os.path.exists(VIDEO1_PATH) or not os.path.exists(VIDEO2_PATH):
        print("Error: One or both video files not found.")
        print("Please check paths or run scripts/generate_test_data.py")
        return

    print(f"{'Case':<40} | {'Algo':<8} | {'KP1':<5} | {'KP2':<5} | {'Match':<5} | {'Inliers':<7}")
    print("-" * 85)

    for v1_idx, v2_idx, description in TEST_CASES:
        print(f"\nTest Case: V1[{v1_idx}] vs V2[{v2_idx}] - {description}")

        img1 = get_frame(VIDEO1_PATH, v1_idx)
        img2 = get_frame(VIDEO2_PATH, v2_idx)

        if img1 is None or img2 is None:
            continue

        best_algo = None
        best_score = -1

        for algo_name, algo_module in ALGORITHMS.items():
            result = evaluate_match(algo_name, algo_module, img1, img2)

            print(f"{'':<40} | {algo_name:<8} | {result['kp1']:<5} | {result['kp2']:<5} | {result['matches']:<5} | {result['inliers']:<7}")

            if result['inliers'] > best_score:
                best_score = result['inliers']
                best_algo = algo_name

        print("-" * 85)
        if best_score > 10:
            print(f"Winner: {best_algo} with {best_score} inliers.")
        else:
            print("Result: No significant match found.")

if __name__ == "__main__":
    main()
