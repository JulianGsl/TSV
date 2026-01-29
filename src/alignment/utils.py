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

    # 1. Monotonicity Constraint & Improved Outlier Filtering
    # We assume V2 frame generally increases as V1 increases.
    # We use velocity analysis with statistical outlier detection.

    filtered = []
    if len(sorted_matches) > 0:
        filtered.append(sorted_matches[0])

    # Calculate local velocities for outlier detection
    velocities = []
    for i in range(1, len(sorted_matches)):
        curr = sorted_matches[i]
        prev = sorted_matches[i-1]
        
        dv1 = curr['v1_frame'] - prev['v1_frame']
        dv2 = curr['v2_frame'] - prev['v2_frame']
        
        if dv1 > 0:
            velocity = dv2 / dv1
            velocities.append(velocity)

    # Calculate median and MAD (Median Absolute Deviation) for robust outlier detection
    if len(velocities) > 3:
        median_velocity = np.median(velocities)
        mad = np.median([abs(v - median_velocity) for v in velocities])
        # Use MAD-based threshold (more robust than std for outliers)
        velocity_threshold_low = max(0, median_velocity - 3 * mad)
        velocity_threshold_high = median_velocity + 3 * mad
    else:
        # Fallback to simple thresholds
        velocity_threshold_low = 0
        velocity_threshold_high = 5.0

    for i in range(1, len(sorted_matches)):
        curr = sorted_matches[i]
        prev = filtered[-1]

        dv1 = curr['v1_frame'] - prev['v1_frame']
        dv2 = curr['v2_frame'] - prev['v2_frame']

        # Monotonicity: enforce forward motion
        if dv2 < 0:
            continue
            
        velocity = dv2 / dv1 if dv1 > 0 else 0

        # Use adaptive thresholds based on velocity distribution
        if velocity_threshold_low <= velocity <= velocity_threshold_high:
            filtered.append(curr)
        else:
            # Outlier detected - skip it
            pass

    # 2. Interpolation for missing frames (optional, if gaps are small)
    # Fill small gaps (< 5 frames) with linear interpolation
    interpolated = []
    for i in range(len(filtered)):
        interpolated.append(filtered[i])
        
        if i < len(filtered) - 1:
            v1_gap = filtered[i+1]['v1_frame'] - filtered[i]['v1_frame']
            # If gap is reasonable, interpolate
            if 1 < v1_gap <= 5:
                v2_start = filtered[i]['v2_frame']
                v2_end = filtered[i+1]['v2_frame']
                v2_step = (v2_end - v2_start) / v1_gap
                
                for j in range(1, v1_gap):
                    interp_v1 = filtered[i]['v1_frame'] + j
                    interp_v2 = int(v2_start + j * v2_step)
                    interpolated.append({
                        'v1_frame': interp_v1,
                        'v2_frame': interp_v2,
                        'score': 0  # Mark as interpolated
                    })
    
    filtered = interpolated

    # 3. Smoothing (Moving Average)
    # We smooth the V2 frames.

    v1_data = [m['v1_frame'] for m in filtered]
    v2_data = [m['v2_frame'] for m in filtered]
    scores = [m['score'] for m in filtered]

    if len(v2_data) < window_size:
        return filtered

    # Gaussian-weighted moving average for smoother results
    # Create gaussian kernel
    sigma = window_size / 4.0
    x = np.arange(window_size) - window_size // 2
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    v2_smooth = np.convolve(v2_data, gaussian_kernel, mode='valid')

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
