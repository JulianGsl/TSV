"""
Post-processing Module
Handles smoothing, filtering, and interpolation of alignment results.
"""

import numpy as np

def filter_outliers_and_smooth(matches, window_size=5):
    """
    Filters outliers from the matches and smooths the path.

    Args:
        matches (list): List of dicts {'v1_frame', 'v2_frame', 'score'}
        window_size (int): Size of the smoothing window.

    Returns:
        list: Filtered and smoothed matches.
    """
    if not matches:
        return []

    # Sort by V1 frame
    sorted_matches = sorted(matches, key=lambda x: x['v1_frame'])

    # 1. Monotonicity Constraint & Outlier Filtering
    # We assume V2 frame generally increases as V1 increases.
    # We can use a simple heuristic: the velocity (dv2/dv1) should be positive and relatively stable.

    filtered = []
    if len(sorted_matches) > 0:
        filtered.append(sorted_matches[0])

    for i in range(1, len(sorted_matches)):
        curr = sorted_matches[i]
        prev = filtered[-1]

        dv1 = curr['v1_frame'] - prev['v1_frame']
        dv2 = curr['v2_frame'] - prev['v2_frame']

        # Simple Logic:
        # 1. dv2 must be >= 0 (No going backward significantly, allow small jitter? No, let's enforce monotonic)
        # 2. Velocity shouldn't be insanely high (e.g. jumping 1000 frames in 1 frame).

        velocity = dv2 / dv1 if dv1 > 0 else 0

        # Thresholds can be tuned.
        # Expected velocity is roughly 1.0 if speeds are similar.
        # Let's allow 0 <= velocity <= 5.0 (V2 moves up to 5x faster than V1)
        if 0 <= velocity <= 5.0:
            filtered.append(curr)
        else:
            # If rejected, we might want to see if this point fits better with the *next* valid point?
            # For this simple implementation, we just drop it.
            pass

    # 2. Smoothing (Moving Average)
    # We smooth the V2 frames.

    v1_data = [m['v1_frame'] for m in filtered]
    v2_data = [m['v2_frame'] for m in filtered]
    scores = [m['score'] for m in filtered]

    if len(v2_data) < window_size:
        return filtered

    # Simple moving average on V2 frames
    v2_smooth = np.convolve(v2_data, np.ones(window_size)/window_size, mode='valid')

    # Pad the result to match length (lost frames at edges)
    pad_start = (window_size - 1) // 2
    pad_end = window_size - 1 - pad_start

    final_matches = []

    # Add start unsmoothed
    for i in range(pad_start):
        final_matches.append({
            "v1_frame": v1_data[i],
            "v2_frame": v2_data[i],
            "score": scores[i]
        })

    # Add smoothed
    for i in range(len(v2_smooth)):
        idx = i + pad_start
        final_matches.append({
            "v1_frame": v1_data[idx],
            "v2_frame": int(v2_smooth[i]),
            "score": scores[idx]
        })

    # Add end unsmoothed
    for i in range(len(v2_data) - pad_end, len(v2_data)):
        final_matches.append({
            "v1_frame": v1_data[i],
            "v2_frame": v2_data[i],
            "score": scores[i]
        })

    return final_matches
