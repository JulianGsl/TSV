"""
Visualization Module
Generates plots and videos to visualize the alignment results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

def create_side_by_side_video(video1_path, video2_path, matches, output_path, max_frames=None):
    """
    Creates a side-by-side video showing the aligned frames.
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

    # Convert matches to a lookup dict for fast access
    # We store the full match object to access keypoints
    match_lookup = {m["v1_frame"]: m for m in matches}
    sorted_v1_frames = sorted(match_lookup.keys())

    if not sorted_v1_frames:
        print("No matches to visualize.")
        return

    # Check if we have keypoints to visualize
    has_keypoints = "kp1" in matches[0] if matches else False

    # Output writer
    # If using drawMatches, the width might be different if images differ in size,
    # but we will resize to fit the side-by-side view.
    out_width = width * 2
    out_height = height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    min_frame = sorted_v1_frames[0]
    max_frame = sorted_v1_frames[-1]

    if max_frames:
        max_frame = min(max_frame, min_frame + max_frames)

    print(f"Generating visualization video {output_path}...")

    current_v1_frame = 0

    # We iterate through V1 frames.
    # If we have a match, we seek V2 to that frame.
    # If we don't have a specific match, we might interpolate or just hold the last one?
    # For simplicity, let's only visualize the matched frames or linear interpolation.

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
        match_obj = match_lookup.get(current_v1_frame)

        # Skip frames that aren't in the match list
        if match_obj is None:
            current_v1_frame += 1
            continue

        target_v2_frame = match_obj["v2_frame"]

        # Seek V2
        if target_v2_frame != last_v2_frame:
             cap2.set(cv2.CAP_PROP_POS_FRAMES, target_v2_frame)
             last_v2_frame = target_v2_frame

        ret2, frame2 = cap2.read()
        if not ret2:
            break # V2 ended

        # Visualization
        combined = None

        if has_keypoints and "matches" in match_obj and match_obj["matches"]:
            # Use drawMatches to show the links
            # We don't resize frame2 before this to ensure keypoints align
            kp1 = match_obj["kp1"]
            kp2 = match_obj["kp2"]
            good_matches = match_obj["matches"]

            # drawMatches creates the combined image
            combined = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        else:
            # Fallback to simple stacking
            if frame2.shape != frame1.shape:
                 frame2 = cv2.resize(frame2, (width, height))
            combined = np.hstack((frame1, frame2))

        # Resize combined image to match the video writer output size
        if combined.shape[0] != out_height or combined.shape[1] != out_width:
            combined = cv2.resize(combined, (out_width, out_height))

        # Add text
        cv2.putText(combined, f"V1 Frame: {current_v1_frame}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, f"V2 Frame: {target_v2_frame}", (width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
