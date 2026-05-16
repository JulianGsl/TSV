"""
Visualization Module for Hybrid Video Alignment

Generates comprehensive visual outputs:
- Side-by-side aligned videos (multiple modes)
- Alignment plots and diagnostics
- DTW cost matrix heatmaps
- Quality metrics visualization
- HTML reports
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
import json


def draw_simple_side_by_side(frame1, frame2, v1_idx, v2_idx, source="", score=0, algorithm="AKAZE"):
    """
    Create a simple side-by-side view with frame info overlay.

    Args:
        algorithm: Feature matching algorithm name (AKAZE, BRISK, ORB) for display
    """
    if frame1 is None or frame2 is None:
        return None

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    # Resize to same height
    target_h = max(h1, h2)
    if h1 != target_h:
        scale = target_h / h1
        frame1 = cv2.resize(frame1, (int(w1 * scale), target_h))
    if h2 != target_h:
        scale = target_h / h2
        frame2 = cv2.resize(frame2, (int(w2 * scale), target_h))

    # Concatenate horizontally
    combined = np.hstack([frame1, frame2])

    # Add info overlay
    h, w = combined.shape[:2]

    # Semi-transparent info bar at bottom
    overlay = combined.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    combined = cv2.addWeighted(overlay, 0.7, combined, 0.3, 0)

    # Text info
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Video 1 info (left side)
    cv2.putText(combined, f"Video 1 - Frame {v1_idx}", (10, h - 35),
                font, font_scale, (255, 255, 255), thickness)

    # Video 2 info (right side)
    cv2.putText(combined, f"Video 2 - Frame {v2_idx}", (frame1.shape[1] + 10, h - 35),
                font, font_scale, (255, 255, 255), thickness)

    # Source indicator with color coding
    if source != "dtw_fallback":
        color = (0, 255, 0)  # Green for feature matching refined
        text = f"{algorithm} ({score} inliers)"
    else:
        color = (0, 165, 255)  # Orange for DTW fallback
        text = "DTW Fallback"

    cv2.putText(combined, text, (10, h - 10), font, font_scale, color, thickness)

    return combined


def draw_features_side_by_side(frame1, frame2, v1_idx, v2_idx, source="", score=0, algorithm="AKAZE"):
    """
    Create side-by-side view with feature keypoints visualized.

    Args:
        algorithm: Feature matching algorithm (AKAZE, BRISK, ORB) to use for visualization
    """
    if frame1 is None or frame2 is None:
        return None

    # Initialize detector based on algorithm
    if algorithm.upper() == "BRISK":
        detector = cv2.BRISK_create()
    elif algorithm.upper() == "ORB":
        detector = cv2.ORB_create(nfeatures=5000)
    else:  # Default to AKAZE
        detector = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, threshold=0.001)

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

    kp1, _ = detector.detectAndCompute(gray1, None)
    kp2, _ = detector.detectAndCompute(gray2, None)

    # Draw keypoints
    frame1_kp = cv2.drawKeypoints(frame1, kp1, None, color=(0, 255, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    frame2_kp = cv2.drawKeypoints(frame2, kp2, None, color=(0, 255, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Combine
    combined = draw_simple_side_by_side(frame1_kp, frame2_kp, v1_idx, v2_idx, source, score, algorithm)

    # Add keypoint counts
    if combined is not None:
        h = combined.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f"KP: {len(kp1)}", (10, 30), font, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, f"KP: {len(kp2)}", (frame1.shape[1] + 10, 30), font, 0.6, (0, 255, 0), 2)

    return combined


def draw_optical_flow_side_by_side(frame1, frame2, prev_frame1, prev_frame2, v1_idx, v2_idx, source="", score=0, algorithm="AKAZE"):
    """
    Create side-by-side view with optical flow visualization.

    Args:
        algorithm: Feature matching algorithm name for display in overlay
    """
    if frame1 is None or frame2 is None:
        return None

    def compute_flow_viz(curr, prev):
        if prev is None:
            return curr.copy()

        gray_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) if len(curr.shape) == 3 else curr
        gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if len(prev.shape) == 3 else prev

        # Resize for speed
        scale = 0.5
        gray_curr_s = cv2.resize(gray_curr, None, fx=scale, fy=scale)
        gray_prev_s = cv2.resize(gray_prev, None, fx=scale, fy=scale)

        flow = cv2.calcOpticalFlowFarneback(gray_prev_s, gray_curr_s, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Create HSV visualization
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((gray_curr_s.shape[0], gray_curr_s.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flow_rgb = cv2.resize(flow_rgb, (curr.shape[1], curr.shape[0]))

        # Blend with original
        return cv2.addWeighted(curr, 0.6, flow_rgb, 0.4, 0)

    frame1_flow = compute_flow_viz(frame1, prev_frame1)
    frame2_flow = compute_flow_viz(frame2, prev_frame2)

    return draw_simple_side_by_side(frame1_flow, frame2_flow, v1_idx, v2_idx, source, score, algorithm)


def create_aligned_video(
    video1_path: str,
    video2_path: str,
    matches: list,
    output_path: str,
    mode: str = "simple",
    max_frames: int = None,
    fps: float = None,
    algorithm: str = "AKAZE"
):
    """
    Create side-by-side aligned video.

    Args:
        video1_path: Path to video 1
        video2_path: Path to video 2
        matches: List of match dictionaries from hybrid alignment
        output_path: Output video path
        mode: "simple", "features", or "flow"
        max_frames: Limit number of frames (None for all)
        fps: Output FPS (None to use video1's FPS)
        algorithm: Feature matching algorithm name (AKAZE, BRISK, ORB) for display

    Returns:
        dict with video statistics
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Could not open one or both videos")

    # Get properties
    if fps is None:
        fps = cap1.get(cv2.CAP_PROP_FPS) or 30.0

    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output dimensions (side by side)
    out_h = max(h1, h2)
    scale1 = out_h / h1
    scale2 = out_h / h2
    out_w = int(w1 * scale1) + int(w2 * scale2)

    # Try avc1 first (Mac compatible), fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Build frame lookup
    match_lookup = {m['v1_frame']: m for m in matches}

    prev_frame1 = None
    prev_frame2 = None
    frames_written = 0

    for match in matches:
        if max_frames and frames_written >= max_frames:
            break

        v1_idx = match['v1_frame']
        v2_idx = match['v2_frame']
        source = match.get('source', 'unknown')
        score = match.get('score', 0)

        # Read frames
        cap1.set(cv2.CAP_PROP_POS_FRAMES, v1_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, v2_idx)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            continue

        # Generate visualization based on mode
        if mode == "features":
            combined = draw_features_side_by_side(frame1, frame2, v1_idx, v2_idx, source, score, algorithm)
        elif mode == "flow":
            combined = draw_optical_flow_side_by_side(
                frame1, frame2, prev_frame1, prev_frame2, v1_idx, v2_idx, source, score, algorithm
            )
            prev_frame1 = frame1.copy()
            prev_frame2 = frame2.copy()
        else:  # simple
            combined = draw_simple_side_by_side(frame1, frame2, v1_idx, v2_idx, source, score, algorithm)

        if combined is not None:
            # Resize to output dimensions
            combined = cv2.resize(combined, (out_w, out_h))
            out.write(combined)
            frames_written += 1

    cap1.release()
    cap2.release()
    out.release()

    return {
        "frames_written": frames_written,
        "output_path": output_path,
        "fps": fps,
        "resolution": f"{out_w}x{out_h}"
    }


def plot_alignment_scatter(matches: list, output_path: str, title: str = "Video Alignment"):
    """
    Create scatter plot of V1 frames vs V2 frames with source color coding.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    v1_frames = [m['v1_frame'] for m in matches]
    v2_frames = [m['v2_frame'] for m in matches]
    sources = [m.get('source', 'unknown') for m in matches]

    # Color by source
    colors = ['green' if s != 'dtw_fallback' else 'orange' for s in sources]

    scatter = ax.scatter(v1_frames, v2_frames, c=colors, alpha=0.6, s=10)

    # Add trend line
    if len(v1_frames) > 1:
        z = np.polyfit(v1_frames, v2_frames, 1)
        p = np.poly1d(z)
        ax.plot(v1_frames, p(v1_frames), "r--", alpha=0.8, linewidth=2, label=f"Trend: y = {z[0]:.3f}x + {z[1]:.1f}")

    # Add diagonal reference (1:1 mapping)
    max_frame = max(max(v1_frames), max(v2_frames))
    ax.plot([0, max_frame], [0, max_frame], 'b:', alpha=0.5, label="1:1 Reference")

    ax.set_xlabel("Video 1 Frame", fontsize=12)
    ax.set_ylabel("Video 2 Frame", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Legend
    refined_patch = mpatches.Patch(color='green', label='Refined')
    dtw_patch = mpatches.Patch(color='orange', label='DTW Fallback')
    ax.legend(handles=[refined_patch, dtw_patch, ax.lines[0], ax.lines[1]], loc='upper left')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_alignment_difference(matches: list, output_path: str):
    """
    Plot the difference between the feature matcher's refinement and the
    DTW prediction. Shows how much the matcher corrects the DTW estimate.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    v1_frames = [m['v1_frame'] for m in matches]
    v2_frames = [m['v2_frame'] for m in matches]
    dtw_predictions = [m.get('dtw_prediction', m['v2_frame']) for m in matches]
    sources = [m.get('source', 'unknown') for m in matches]

    # Calculate differences
    differences = [v2 - dtw for v2, dtw in zip(v2_frames, dtw_predictions)]

    # Top plot: V2 frame and DTW prediction over time
    ax1 = axes[0]
    ax1.plot(v1_frames, v2_frames, 'g-', alpha=0.7, linewidth=1, label='Final V2 Frame')
    ax1.plot(v1_frames, dtw_predictions, 'b--', alpha=0.5, linewidth=1, label='DTW Prediction')
    ax1.set_xlabel("Video 1 Frame")
    ax1.set_ylabel("Video 2 Frame")
    ax1.set_title("Alignment: Final vs DTW Prediction")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Difference (correction) over time
    ax2 = axes[1]
    colors = ['green' if s != 'dtw_fallback' else 'orange' for s in sources]
    ax2.scatter(v1_frames, differences, c=colors, alpha=0.6, s=15)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No Correction')
    ax2.set_xlabel("Video 1 Frame")
    ax2.set_ylabel("Correction (frames)")
    ax2.set_title("Refinement Correction vs DTW Prediction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add statistics
    refined_diffs = [d for d, s in zip(differences, sources) if s != 'dtw_fallback']
    if refined_diffs:
        stats_text = f"Refinement corrections: mean={np.mean(refined_diffs):.2f}, std={np.std(refined_diffs):.2f}, max={np.max(np.abs(refined_diffs)):.0f}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_source_distribution(matches: list, output_path: str):
    """
    Create pie chart and histogram showing refined matches vs DTW fallback.
    "Refined" covers any feature-matcher backed match (AKAZE / BRISK / ORB).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sources = [m.get('source', 'unknown') for m in matches]
    scores = [m.get('score', 0) for m in matches]

    refined_count = sum(1 for s in sources if s != 'dtw_fallback')
    dtw_count = sum(1 for s in sources if s == 'dtw_fallback')

    # Pie chart
    ax1 = axes[0]
    sizes = [refined_count, dtw_count]
    labels = [f'Refined\n({refined_count})', f'DTW Fallback\n({dtw_count})']
    colors = ['#2ecc71', '#e67e22']
    explode = (0.05, 0)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title("Alignment Source Distribution")

    # Score histogram
    ax2 = axes[1]
    refined_scores = [s for s, src in zip(scores, sources) if src != 'dtw_fallback' and s > 0]

    if refined_scores:
        ax2.hist(refined_scores, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(refined_scores), color='red', linestyle='--',
                    label=f'Mean: {np.mean(refined_scores):.1f}')
        ax2.axvline(x=np.median(refined_scores), color='blue', linestyle='--',
                    label=f'Median: {np.median(refined_scores):.1f}')
        ax2.set_xlabel("RANSAC Inliers")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Match Quality (RANSAC Inliers)")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No refined matches", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Match Quality")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_velocity_analysis(matches: list, output_path: str):
    """
    Analyze and plot the relative velocity between videos.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    v1_frames = np.array([m['v1_frame'] for m in matches])
    v2_frames = np.array([m['v2_frame'] for m in matches])

    # Calculate instantaneous velocity
    if len(v1_frames) > 1:
        dv1 = np.diff(v1_frames)
        dv2 = np.diff(v2_frames)

        # Avoid division by zero
        valid_mask = dv1 > 0
        velocities = np.zeros(len(dv1))
        velocities[valid_mask] = dv2[valid_mask] / dv1[valid_mask]

        # Top: Velocity over time
        ax1 = axes[0]
        ax1.plot(v1_frames[1:], velocities, 'b-', alpha=0.5, linewidth=1)

        # Smoothed velocity
        window = min(20, len(velocities) // 10) if len(velocities) > 20 else 3
        if window > 1:
            smoothed = np.convolve(velocities, np.ones(window)/window, mode='valid')
            ax1.plot(v1_frames[window:], smoothed, 'r-', linewidth=2, label='Smoothed')

        ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Constant Speed (1:1)')
        ax1.set_xlabel("Video 1 Frame")
        ax1.set_ylabel("Relative Velocity (dV2/dV1)")
        ax1.set_title("Relative Velocity Analysis")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 3])

        # Bottom: Velocity histogram
        ax2 = axes[1]
        valid_velocities = velocities[(velocities > 0) & (velocities < 5)]
        if len(valid_velocities) > 0:
            ax2.hist(valid_velocities, bins=50, color='blue', alpha=0.7, edgecolor='black')
            ax2.axvline(x=np.mean(valid_velocities), color='red', linestyle='--',
                        label=f'Mean: {np.mean(valid_velocities):.3f}')
            ax2.axvline(x=1.0, color='green', linestyle='--', label='1:1 Reference')
            ax2.set_xlabel("Relative Velocity")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Velocity Distribution")
            ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_dtw_cost_matrix(cost_matrix: np.ndarray, path: list, output_path: str,
                          ground_truth_sampled: list = None):
    """
    Visualize DTW cost matrix with optimal path and (optional) ground truth.

    Args:
        cost_matrix: accumulated cost matrix (sampled space).
        path: list of (i, j) DTW waypoints in sampled space.
        output_path: where to write the PNG.
        ground_truth_sampled: optional list of (i, j) anchors in sampled space
            — when given, drawn as a piecewise-linear "manual ground truth"
            line for visual comparison against the DTW path.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('dtw', ['#2c3e50', '#3498db', '#2ecc71', '#f1c40f', '#e74c3c'])

    # Normalize and display
    im = ax.imshow(cost_matrix, cmap=cmap, aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax, label='Accumulated Cost')

    # Draw path
    if path:
        path_i = [p[0] for p in path]
        path_j = [p[1] for p in path]
        ax.plot(path_j, path_i, 'w-', linewidth=2, label='DTW path')
        ax.scatter([path_j[0]], [path_i[0]], c='lime', s=100, marker='o', label='Start', zorder=5)
        ax.scatter([path_j[-1]], [path_i[-1]], c='red', s=100, marker='s', label='End', zorder=5)

    # Optional: piecewise-linear ground truth between manual anchors
    if ground_truth_sampled:
        gt = sorted(ground_truth_sampled)
        gt_i = [a[0] for a in gt]
        gt_j = [a[1] for a in gt]
        ax.plot(gt_j, gt_i, '-', color='#ff00ff', linewidth=2.2,
                label='Manual ground truth (lerp)', alpha=0.95)
        ax.scatter(gt_j, gt_i, c='#ff00ff', edgecolors='white', linewidths=1.5,
                   s=70, marker='o', zorder=6)

    ax.set_xlabel("Video 2 Frame (sampled)")
    ax.set_ylabel("Video 1 Frame (sampled)")
    title = "DTW Cost Matrix with Optimal Path"
    if ground_truth_sampled:
        title += " vs Manual Ground Truth"
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def compute_alignment_metrics(matches: list) -> dict:
    """
    Compute comprehensive alignment quality metrics.
    """
    if not matches:
        return {}

    v1_frames = np.array([m['v1_frame'] for m in matches])
    v2_frames = np.array([m['v2_frame'] for m in matches])
    scores = np.array([m.get('score', 0) for m in matches])
    sources = [m.get('source', 'unknown') for m in matches]
    dtw_preds = np.array([m.get('dtw_prediction', m['v2_frame']) for m in matches])

    # Basic counts. "Refined" = match accepted by phase B (any algorithm),
    # as opposed to "dtw_fallback" = the DTW prediction was kept as-is.
    total = len(matches)
    refined_count = sum(1 for s in sources if s != 'dtw_fallback')
    dtw_count = total - refined_count

    # Monotonicity check
    monotonic_violations = sum(1 for i in range(1, len(v2_frames)) if v2_frames[i] < v2_frames[i-1])

    # Velocity statistics
    velocities = []
    for i in range(1, len(matches)):
        dv1 = v1_frames[i] - v1_frames[i-1]
        dv2 = v2_frames[i] - v2_frames[i-1]
        if dv1 > 0:
            velocities.append(dv2 / dv1)

    # Phase B correction statistics: by how much does the matcher pull
    # the final V2 away from the DTW prediction, where it succeeds?
    corrections = v2_frames - dtw_preds
    refined_corrections = [c for c, s in zip(corrections, sources) if s != 'dtw_fallback']

    # Inlier-score statistics for refined matches only.
    refined_scores = [s for s, src in zip(scores, sources) if src != 'dtw_fallback' and s > 0]

    metrics = {
        "total_matches": total,
        # Canonical keys (algorithm-agnostic).
        "refined_count": refined_count,
        "refined_percent": 100 * refined_count / total if total > 0 else 0,
        "dtw_fallback_count": dtw_count,
        "dtw_fallback_percent": 100 * dtw_count / total if total > 0 else 0,
        "monotonicity_violations": monotonic_violations,
        "monotonicity_score": 100 * (1 - monotonic_violations / max(1, total - 1)),
        "v1_frame_range": [int(v1_frames.min()), int(v1_frames.max())],
        "v2_frame_range": [int(v2_frames.min()), int(v2_frames.max())],
        "velocity_mean": float(np.mean(velocities)) if velocities else 1.0,
        "velocity_std": float(np.std(velocities)) if velocities else 0.0,
        "velocity_range": [float(min(velocities)), float(max(velocities))] if velocities else [1.0, 1.0],
        "refined_score_mean": float(np.mean(refined_scores)) if refined_scores else 0.0,
        "refined_score_std": float(np.std(refined_scores)) if refined_scores else 0.0,
        "refined_score_max": float(max(refined_scores)) if refined_scores else 0,
        "refined_correction_mean": float(np.mean(refined_corrections)) if refined_corrections else 0.0,
        "refined_correction_std": float(np.std(refined_corrections)) if refined_corrections else 0.0,
        "refined_correction_max": float(np.max(np.abs(refined_corrections))) if refined_corrections else 0.0,
    }

    # Backward-compatible aliases. Older consumers (HTML reports rendered
    # from previously saved JSONs, third-party analysis scripts) still read
    # the legacy keys; we duplicate the values so nothing breaks. Drop these
    # aliases once all consumers have migrated.
    metrics["akaze_refined_count"]      = metrics["refined_count"]
    metrics["akaze_refined_percent"]    = metrics["refined_percent"]
    metrics["akaze_score_mean"]         = metrics["refined_score_mean"]
    metrics["akaze_score_std"]          = metrics["refined_score_std"]
    metrics["akaze_score_max"]          = metrics["refined_score_max"]
    metrics["akaze_correction_mean"]    = metrics["refined_correction_mean"]
    metrics["akaze_correction_std"]     = metrics["refined_correction_std"]
    metrics["akaze_correction_max"]     = metrics["refined_correction_max"]

    return metrics


def generate_html_report(
    matches: list,
    metrics: dict,
    output_dir: str,
    video1_path: str,
    video2_path: str,
    config: dict = None
) -> str:
    """
    Generate comprehensive HTML report with all visualizations.
    """
    report_path = os.path.join(output_dir, "report.html")

    # Algorithm name for human-readable labels in the report.
    algo_name = (config or {}).get("algorithm", "feature matcher")

    # Get relative paths for images
    def rel_path(filename):
        return os.path.basename(filename)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid Video Alignment Report</title>
    <style>
        :root {{
            --primary: #3498db;
            --success: #2ecc71;
            --warning: #f39c12;
            --danger: #e74c3c;
            --dark: #2c3e50;
            --light: #ecf0f1;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--light);
            color: var(--dark);
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, var(--dark), var(--primary));
            color: white;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header p {{ opacity: 0.9; }}
        .card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            overflow: hidden;
        }}
        .card-header {{
            background: var(--dark);
            color: white;
            padding: 15px 20px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .card-body {{ padding: 20px; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric-box {{
            background: var(--light);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--primary);
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .success {{ color: var(--success); }}
        .warning {{ color: var(--warning); }}
        .danger {{ color: var(--danger); }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: var(--dark); color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge-success {{ background: var(--success); color: white; }}
        .badge-warning {{ background: var(--warning); color: white; }}
        .progress-bar {{
            background: #ddd;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Hybrid Video Alignment Report</h1>
            <p>Coarse-to-Fine Alignment using DTW + {algo_name}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="card">
            <div class="card-header">Input Videos</div>
            <div class="card-body">
                <table>
                    <tr><th>Property</th><th>Video 1 (Reference)</th><th>Video 2 (Target)</th></tr>
                    <tr><td>Path</td><td>{os.path.basename(video1_path)}</td><td>{os.path.basename(video2_path)}</td></tr>
                    <tr><td>Frame Range</td><td>{metrics.get('v1_frame_range', ['?', '?'])[0]} - {metrics.get('v1_frame_range', ['?', '?'])[1]}</td>
                        <td>{metrics.get('v2_frame_range', ['?', '?'])[0]} - {metrics.get('v2_frame_range', ['?', '?'])[1]}</td></tr>
                </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Alignment Quality Metrics</div>
            <div class="card-body">
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('total_matches', 0)}</div>
                        <div class="metric-label">Total Matches</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value success">{metrics.get('refined_percent', metrics.get('akaze_refined_percent', 0)):.1f}%</div>
                        <div class="metric-label">{algo_name} Refined</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value warning">{metrics.get('dtw_fallback_percent', 0):.1f}%</div>
                        <div class="metric-label">DTW Fallback</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value {'success' if metrics.get('monotonicity_score', 0) > 95 else 'warning'}">{metrics.get('monotonicity_score', 0):.1f}%</div>
                        <div class="metric-label">Monotonicity Score</div>
                    </div>
                </div>

                <h3 style="margin-top: 30px;">Velocity Analysis</h3>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('velocity_mean', 1.0):.3f}</div>
                        <div class="metric-label">Mean Velocity Ratio</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('velocity_std', 0):.3f}</div>
                        <div class="metric-label">Velocity Std Dev</div>
                    </div>
                </div>

                <h3 style="margin-top: 30px;">{algo_name} Quality</h3>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('refined_score_mean', metrics.get('akaze_score_mean', 0)):.1f}</div>
                        <div class="metric-label">Mean Inliers</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('refined_correction_mean', metrics.get('akaze_correction_mean', 0)):.2f}</div>
                        <div class="metric-label">Mean Correction (frames)</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('refined_correction_max', metrics.get('akaze_correction_max', 0)):.0f}</div>
                        <div class="metric-label">Max Correction</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Alignment Visualization</div>
            <div class="card-body">
                <div class="image-container">
                    <img src="alignment_scatter.png" alt="Alignment Scatter Plot">
                    <p>Scatter plot showing frame correspondences (green={algo_name}, orange=DTW)</p>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">{algo_name} Correction Analysis</div>
            <div class="card-body">
                <div class="image-container">
                    <img src="alignment_difference.png" alt="Alignment Difference">
                    <p>How much {algo_name} corrects the DTW predictions</p>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Source Distribution</div>
            <div class="card-body">
                <div class="image-container">
                    <img src="source_distribution.png" alt="Source Distribution">
                    <p>Distribution of alignment sources and {algo_name} match quality</p>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Velocity Analysis</div>
            <div class="card-body">
                <div class="image-container">
                    <img src="velocity_analysis.png" alt="Velocity Analysis">
                    <p>Relative velocity between videos over time</p>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">DTW Cost Matrix</div>
            <div class="card-body">
                <div class="image-container">
                    <img src="dtw_cost_matrix.png" alt="DTW Cost Matrix">
                    <p>DTW accumulated cost matrix with optimal path</p>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Configuration</div>
            <div class="card-body">
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    {"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in (config or {}).items())}
                </table>
            </div>
        </div>

        <footer>
            <p>Generated by Hybrid Video Alignment System</p>
            <p>Coarse-to-Fine: DTW (macro) + {algo_name} (micro)</p>
        </footer>
    </div>
</body>
</html>
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return report_path


def save_alignment_csv(matches: list, output_path: str):
    """
    Save alignment results to CSV with all details.
    """
    import csv

    if not matches:
        return

    # Determine all keys
    all_keys = set()
    for m in matches:
        all_keys.update(m.keys())

    fieldnames = sorted(all_keys)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)


def save_metrics_json(metrics: dict, output_path: str):
    """
    Save metrics to JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
