"""
Visualization Module
Generates plots and videos to visualize the alignment results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Import algorithms for feature detection
from .algorithms import orb, brisk, akaze

ALGORITHMS = {
    "ORB": orb,
    "BRISK": brisk,
    "AKAZE": akaze
}


def draw_simple_side_by_side(frame1, frame2):
    """
    Creates a simple side-by-side view of two frames without any feature visualization.

    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)

    Returns:
        Combined frame (side by side)
    """
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    # Resize frame2 if dimensions don't match
    if h1 != h2 or w1 != w2:
        frame2 = cv2.resize(frame2, (w1, h1))

    return np.hstack((frame1, frame2))


def draw_all_features(frame1, frame2, algorithm="ORB"):
    """
    Draws ALL detected keypoints on both frames (no matching, just feature detection).

    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)
        algorithm: Algorithm name to use for feature detection ("ORB", "BRISK", "AKAZE")

    Returns:
        Combined frame with all keypoints drawn, keypoint counts
    """
    algo_module = ALGORITHMS.get(algorithm.upper(), orb)

    # Compute features for both frames
    kp1, des1 = algo_module.compute_features(frame1)
    kp2, des2 = algo_module.compute_features(frame2)

    # Create copies to draw on
    frame1_with_kp = frame1.copy()
    frame2_with_kp = frame2.copy()

    # Draw ALL keypoints on each frame (green circles)
    frame1_with_kp = cv2.drawKeypoints(frame1_with_kp, kp1, None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    frame2_with_kp = cv2.drawKeypoints(frame2_with_kp, kp2, None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    h1, w1 = frame1_with_kp.shape[:2]
    h2, w2 = frame2_with_kp.shape[:2]

    # Resize frame2 if dimensions don't match
    if h1 != h2 or w1 != w2:
        frame2_with_kp = cv2.resize(frame2_with_kp, (w1, h1))

    combined = np.hstack((frame1_with_kp, frame2_with_kp))

    # Add keypoint count info
    cv2.putText(combined, f"Keypoints V1: {len(kp1)}", (10, combined.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, f"Keypoints V2: {len(kp2)}", (w1 + 10, combined.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return combined, len(kp1), len(kp2)


def draw_matched_features_only(frame1, frame2, algorithm="ORB"):
    """
    Draws ONLY the matched keypoints on both frames with lines connecting them.
    Non-matched keypoints are not displayed.

    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)
        algorithm: Algorithm name to use for feature detection ("ORB", "BRISK", "AKAZE")

    Returns:
        Combined frame with matched features only, match count
    """
    algo_module = ALGORITHMS.get(algorithm.upper(), orb)

    # Compute features for both frames
    kp1, des1 = algo_module.compute_features(frame1)
    kp2, des2 = algo_module.compute_features(frame2)

    # Create copies to draw on
    frame1_draw = frame1.copy()
    frame2_draw = frame2.copy()

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    # Resize frame2 if dimensions don't match
    if h1 != h2 or w1 != w2:
        frame2_draw = cv2.resize(frame2_draw, (w1, h1))

    # Combine frames first
    combined = np.hstack((frame1_draw, frame2_draw))

    # Find good matches
    good_matches = []
    if des1 is not None and des2 is not None and len(kp1) >= 2 and len(kp2) >= 2:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        try:
            knn_matches = bf.knnMatch(des1, des2, k=2)

            # Apply Lowe's ratio test
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                elif len(match_pair) == 1:
                    m = match_pair[0]
                    if m.distance < 50:
                        good_matches.append(m)
        except cv2.error:
            pass

    # Draw only matched keypoints and their connections
    max_lines = 100  # Limit for visualization clarity
    if good_matches:
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_lines]

        for match in good_matches:
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            pt2_offset = (pt2[0] + w1, pt2[1])

            # Color based on match quality
            if match.distance < 30:
                color = (0, 255, 0)  # Green - excellent
            elif match.distance < 50:
                color = (0, 255, 255)  # Yellow - good
            else:
                color = (0, 165, 255)  # Orange - moderate

            # Draw keypoint circles ONLY for matched points
            cv2.circle(combined, pt1, 6, color, 2)  # Circle on frame1
            cv2.circle(combined, pt2_offset, 6, color, 2)  # Circle on frame2

            # Draw connection line
            cv2.line(combined, pt1, pt2_offset, color, 1, cv2.LINE_AA)

    # Add match count info
    cv2.putText(combined, f"Matched Features: {len(good_matches)}", (10, combined.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return combined, len(good_matches)


def draw_features_and_matches(frame1, frame2, algorithm="ORB"):
    """
    Draws keypoints on both frames and draws match lines between corresponding features.
    Shows ALL keypoints + match lines (legacy function for compatibility).

    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)
        algorithm: Algorithm name to use for feature detection ("ORB", "BRISK", "AKAZE")

    Returns:
        Combined frame with features and matches drawn
    """
    algo_module = ALGORITHMS.get(algorithm.upper(), orb)

    # Compute features for both frames
    kp1, des1 = algo_module.compute_features(frame1)
    kp2, des2 = algo_module.compute_features(frame2)

    # Create copies to draw on
    frame1_with_kp = frame1.copy()
    frame2_with_kp = frame2.copy()

    # Draw keypoints on each frame
    # Green circles for keypoints
    frame1_with_kp = cv2.drawKeypoints(frame1_with_kp, kp1, None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    frame2_with_kp = cv2.drawKeypoints(frame2_with_kp, kp2, None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Match features if descriptors are available
    good_matches = []
    if des1 is not None and des2 is not None and len(kp1) >= 2 and len(kp2) >= 2:
        # Use BFMatcher with Hamming distance for binary descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        try:
            knn_matches = bf.knnMatch(des1, des2, k=2)

            # Apply Lowe's ratio test
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                elif len(match_pair) == 1:
                    m = match_pair[0]
                    if m.distance < 50:
                        good_matches.append(m)
        except cv2.error:
            pass  # If matching fails, just show keypoints without matches

    # Combine the two frames side by side
    h1, w1 = frame1_with_kp.shape[:2]
    h2, w2 = frame2_with_kp.shape[:2]

    # Resize frame2 if dimensions don't match
    if h1 != h2 or w1 != w2:
        frame2_with_kp = cv2.resize(frame2_with_kp, (w1, h1))

    combined = np.hstack((frame1_with_kp, frame2_with_kp))

    # Draw match lines between corresponding keypoints
    # Limit to top matches for cleaner visualization
    max_lines = 50  # Limit number of lines to draw
    if good_matches:
        # Sort by distance and take best matches
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_lines]

        for match in good_matches:
            # Get keypoint coordinates
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))

            # Offset pt2 by the width of frame1 (since it's on the right side)
            pt2_offset = (pt2[0] + w1, pt2[1])

            # Draw line with color based on match quality (green = good, yellow = moderate)
            # Use match distance to determine color
            if match.distance < 30:
                color = (0, 255, 0)  # Green - excellent match
            elif match.distance < 50:
                color = (0, 255, 255)  # Yellow - good match
            else:
                color = (0, 165, 255)  # Orange - moderate match

            cv2.line(combined, pt1, pt2_offset, color, 1, cv2.LINE_AA)

            # Draw small circles at match points
            cv2.circle(combined, pt1, 4, (255, 0, 0), -1)  # Blue circle on frame1
            cv2.circle(combined, pt2_offset, 4, (255, 0, 0), -1)  # Blue circle on frame2

    # Add match count info
    cv2.putText(combined, f"Keypoints: V1={len(kp1)}, V2={len(kp2)}", (10, combined.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, f"Good Matches: {len(good_matches)}", (10, combined.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return combined, len(kp1), len(kp2), len(good_matches)


def plot_alignment(matches, output_path):
    """
    Generates a plot of V1 Frame vs V2 Frame.
    """
    v1_frames = [m["v1_frame"] for m in matches]
    v2_frames = [m["v2_frame"] for m in matches]
    scores = [m["score"] for m in matches]

    plt.figure(figsize=(10, 6))
    plt.scatter(v1_frames, v2_frames, c=scores, cmap='viridis', s=10, alpha=0.8)
    plt.colorbar(label='Match Score')
    plt.xlabel('Video 1 Frame')
    plt.ylabel('Video 2 Frame')
    plt.title('Video Alignment Path')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add an ideal diagonal line for reference (assuming roughly same speed)
    if v1_frames:
        min_v1, max_v1 = min(v1_frames), max(v1_frames)
        min_v2, max_v2 = min(v2_frames), max(v2_frames)
        plt.plot([min_v1, max_v1], [min_v2, max_v2], 'r--', alpha=0.3, label='Linear Reference')
        plt.legend()

    plt.savefig(output_path)
    plt.close()
    print(f"Saved alignment plot to {output_path}")

def create_side_by_side_video(video1_path, video2_path, matches, output_path, max_frames=None, algorithm="ORB", mode="simple"):
    """
    Creates a side-by-side video showing the aligned frames.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        matches: List of match dictionaries with v1_frame and v2_frame keys
        output_path: Path for output video
        max_frames: Maximum number of frames to process (None for all)
        algorithm: Algorithm name to use for feature detection ("ORB", "BRISK", "AKAZE")
        mode: Visualization mode:
              - "simple": Just side-by-side frames, no features
              - "all_features": Show all detected keypoints on both frames
              - "matched_only": Show only matched features with connection lines
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open videos for visualization.")
        return

    # Get properties
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # Output writer
    out_width = width * 2
    out_height = height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # Convert matches to a lookup dict for fast access: v1_frame -> v2_frame
    match_lookup = {m["v1_frame"]: m["v2_frame"] for m in matches}
    sorted_v1_frames = sorted(match_lookup.keys())

    if not sorted_v1_frames:
        print("No matches to visualize.")
        return

    min_frame = sorted_v1_frames[0]
    max_frame = sorted_v1_frames[-1]

    if max_frames:
        max_frame = min(max_frame, min_frame + max_frames)

    mode_names = {
        "simple": "Frames Only",
        "all_features": "All Features",
        "matched_only": "Matched Features"
    }
    print(f"Generating visualization video ({mode_names.get(mode, mode)}): {output_path}...")

    current_v1_frame = 0

    # Let's iterate linearly through the matched range
    cap1.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
    current_v1_frame = min_frame

    last_v2_frame = -1

    count = 0
    while current_v1_frame <= max_frame:
        ret1, frame1 = cap1.read()
        if not ret1:
            break

        # Determine which V2 frame to show
        target_v2_frame = match_lookup.get(current_v1_frame)

        if target_v2_frame is None:
            current_v1_frame += 1
            continue

        # Seek V2
        if target_v2_frame != last_v2_frame:
             cap2.set(cv2.CAP_PROP_POS_FRAMES, target_v2_frame)
             last_v2_frame = target_v2_frame

        ret2, frame2 = cap2.read()
        if not ret2:
            break # V2 ended

        # Resize if needed
        if frame2.shape != frame1.shape:
             frame2 = cv2.resize(frame2, (width, height))

        # Create combined frame based on mode
        if mode == "all_features":
            # Show ALL detected keypoints
            combined, n_kp1, n_kp2 = draw_all_features(frame1, frame2, algorithm)
        elif mode == "matched_only":
            # Show ONLY matched features with lines
            combined, n_matches = draw_matched_features_only(frame1, frame2, algorithm)
        else:
            # Simple mode - just frames
            combined = draw_simple_side_by_side(frame1, frame2)

        # Add frame info text (common to all modes)
        cv2.putText(combined, f"V1 Frame: {current_v1_frame}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, f"V2 Frame: {target_v2_frame}", (width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Add mode and algorithm info for feature modes
        if mode in ["all_features", "matched_only"]:
            cv2.putText(combined, f"Algorithm: {algorithm.upper()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out.write(combined)

        current_v1_frame += 1
        count += 1
        if count % 100 == 0:
            print(f"  Processed {count} frames...")

    cap1.release()
    cap2.release()
    out.release()
    print("Video generation complete.")

def generate_html_report(matches, stats, output_path):
    """
    Generates a simple HTML report.
    """
    html_content = f"""
    <html>
    <head>
        <title>Alignment Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; }}
            .stats {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
            .chart {{ margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h1>Video Alignment Report</h1>
        <div class="stats">
            <h2>Statistics</h2>
            <p><strong>Total Matches:</strong> {len(matches)}</p>
            <p><strong>Average Score:</strong> {stats.get('avg_score', 0):.2f}</p>
            <p><strong>Algorithm:</strong> {stats.get('algorithm', 'Unknown')}</p>
        </div>

        <div class="chart">
            <h2>Alignment Path</h2>
            <img src="plot_{stats.get('algorithm', '').lower()}.png" width="800" />
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Saved HTML report to {output_path}")
