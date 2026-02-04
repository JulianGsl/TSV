"""
Post-processing Module
Handles smoothing, filtering, and interpolation of alignment results.
"""

import numpy as np

def filter_outliers_and_smooth(matches, window_size=5):
    """
    Filters outliers from the matches and smooths the path.
    Uses robust statistical methods to detect and correct drift.

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

    # Calculate all pairwise velocities for robust baseline estimation
    all_velocities = []
    for i in range(1, len(sorted_matches)):
        curr = sorted_matches[i]
        prev = sorted_matches[i-1]
        
        dv1 = curr['v1_frame'] - prev['v1_frame']
        dv2 = curr['v2_frame'] - prev['v2_frame']
        
        if dv1 > 0 and dv2 >= 0:  # Only positive velocities
            velocity = dv2 / dv1
            all_velocities.append(velocity)

    # Calculate median and MAD (Median Absolute Deviation) for robust outlier detection
    if len(all_velocities) >= 2:
        median_velocity = np.median(all_velocities)
        deviations = [abs(v - median_velocity) for v in all_velocities]
        mad = np.median(deviations)
        
        # Handle zero MAD case (all velocities identical) by using a small tolerance
        if mad < 0.01:
            mad = 0.1  # Minimum tolerance to allow slight variations
        
        # Use MAD-based threshold (more robust than std for outliers)
        # Tighten thresholds from 3*MAD to 2.5*MAD for better outlier rejection
        velocity_threshold_low = max(0.1, median_velocity - 2.5 * mad)
        velocity_threshold_high = median_velocity + 2.5 * mad
        
        # Also set absolute bounds based on physical constraints
        velocity_threshold_low = max(0.1, velocity_threshold_low)
        velocity_threshold_high = min(3.0, velocity_threshold_high)
    else:
        # Fallback to simple thresholds for first match
        velocity_threshold_low = 0.1
        velocity_threshold_high = 3.0

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
    
    # Additional pass: detect and correct systematic drift
    # Look for gradual velocity changes that might indicate accumulated error
    if len(filtered) >= 10:
        # Divide matches into segments and check for velocity drift
        segment_size = len(filtered) // 3
        if segment_size >= 3:
            segments = [
                filtered[0:segment_size],
                filtered[segment_size:2*segment_size],
                filtered[2*segment_size:]
            ]
            
            segment_velocities = []
            for segment in segments:
                seg_vels = []
                for i in range(1, len(segment)):
                    dv1 = segment[i]['v1_frame'] - segment[i-1]['v1_frame']
                    dv2 = segment[i]['v2_frame'] - segment[i-1]['v2_frame']
                    if dv1 > 0:
                        seg_vels.append(dv2 / dv1)
                if seg_vels:
                    segment_velocities.append(np.median(seg_vels))
            
            # If there's a consistent trend (drift), apply correction to later segments
            if len(segment_velocities) == 3:
                drift_trend = segment_velocities[-1] - segment_velocities[0]
                # If drift is significant (>10% change), apply linear correction
                if abs(drift_trend) > 0.1:
                    target_velocity = segment_velocities[0]  # Use first segment as reference
                    # Apply gradual correction to last segment
                    correction_needed = drift_trend * segment_size
                    for j, idx in enumerate(range(2*segment_size, len(filtered))):
                        progress = j / segment_size if segment_size > 0 else 0
                        filtered[idx]['v2_frame'] = int(filtered[idx]['v2_frame'] - progress * correction_needed)

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
                
                for j in range(1, int(v1_gap)):  # Explicit int conversion for type safety
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
    # Create gaussian kernel with minimum sigma for numerical stability
    sigma = max(window_size / 4.0, 0.5)  # Prevent very small sigma values
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
