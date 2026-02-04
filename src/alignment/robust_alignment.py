"""
Robust Video Alignment Module
Implements robust alignment with temporal continuity constraints and bidirectional search.
"""

import numpy as np
from collections import deque

# Constants
DEFAULT_MAX_VELOCITY = 5
DEFAULT_CONF_THRESHOLD = 0.3
DEFAULT_HISTORY_SIZE = 10
SEARCH_RADIUS_MULTIPLIER = 10  # Multiplier for adaptive search radius
SINGLE_MATCH_DISTANCE_THRESHOLD = 50  # Distance threshold for accepting single matches
DISTANCE_PENALTY_FACTOR = 0.1  # Factor for exponential distance penalty


class RobustAligner:
    """
    Robust video alignment with temporal constraints and bidirectional search.
    
    This class implements:
    - Confidence scoring based on match quality distribution
    - Velocity tracking for temporal continuity
    - Weighted scoring combining multiple factors
    - Bidirectional search window (forward AND backward) to correct past errors
    """
    
    def __init__(self, max_velocity=DEFAULT_MAX_VELOCITY, conf_threshold=DEFAULT_CONF_THRESHOLD, 
                 history_size=DEFAULT_HISTORY_SIZE):
        """
        Initialize the RobustAligner.
        
        Args:
            max_velocity (int): Maximum allowed velocity change (frames/frame)
            conf_threshold (float): Minimum confidence threshold (0-1)
            history_size (int): Number of recent matches to track
        """
        self.max_velocity = max_velocity
        self.conf_threshold = conf_threshold
        self.history = deque(maxlen=history_size)
        self.last_offset = 0
        
    def compute_confidence(self, scores, best_idx):
        """
        Compute confidence score based on match quality distribution.
        
        Confidence measures how distinctly better the best match is compared to alternatives.
        Formula: conf = 1 - (second_best / best)
        
        Args:
            scores (list): List of tuples (score, frame_idx) for all candidates
            best_idx (int): Index of the best match in scores list
            
        Returns:
            float: Confidence score (0-1), where 1 means very confident
        """
        if len(scores) < 2:
            return 0.5  # Low confidence if not enough candidates
            
        # Sort scores in descending order
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        
        best_score = sorted_scores[0][0]
        second_best_score = sorted_scores[1][0]
        
        # Avoid division by zero
        if best_score <= 0:
            return 0.0
            
        # Confidence: how much better is the best vs second best
        conf = 1.0 - (second_best_score / best_score)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, conf))
    
    def compute_velocity(self):
        """
        Compute average velocity from recent history.
        
        Velocity is the rate of change of offset (dV2/dV1).
        
        Returns:
            float: Average velocity, or 1.0 if insufficient history
        """
        if len(self.history) < 2:
            return 1.0
            
        velocities = []
        for i in range(1, len(self.history)):
            prev = self.history[i-1]
            curr = self.history[i]
            
            dv1 = curr['v1_frame'] - prev['v1_frame']
            dv2 = curr['v2_frame'] - prev['v2_frame']
            
            if dv1 > 0:
                velocities.append(dv2 / dv1)
        
        if not velocities:
            return 1.0
            
        return np.mean(velocities)
    
    def compute_weighted_score(self, raw_score, distance_from_prediction, conf):
        """
        Compute weighted score combining raw score, confidence, and distance penalty.
        
        Formula: weighted = raw_score * conf * exp(-DISTANCE_PENALTY_FACTOR * |distance|)
        
        Args:
            raw_score (float): Raw match score (e.g., number of inliers)
            distance_from_prediction (int): Distance from predicted frame position
            conf (float): Confidence score (0-1)
            
        Returns:
            float: Weighted score
        """
        # Exponential decay based on distance from prediction
        distance_penalty = np.exp(-DISTANCE_PENALTY_FACTOR * abs(distance_from_prediction))
        
        # Combine all factors
        weighted = raw_score * conf * distance_penalty
        
        return weighted
    
    def find_best_match(self, v1_frame_idx, video2_frames, algo_module, 
                       bf_matcher, kp1, des1, predicted_offset=None):
        """
        Find best matching frame with bidirectional search and temporal constraints.
        
        CRITICAL: This implements BIDIRECTIONAL search - searches both forward AND backward
        from the predicted position to allow correcting errors from previous frames.
        
        Args:
            v1_frame_idx (int): Current frame index in video 1
            video2_frames (list): List of available frames from video 2
            algo_module: Algorithm module (orb, brisk, or akaze)
            bf_matcher: BFMatcher instance for feature matching
            kp1: Keypoints from video 1 current frame
            des1: Descriptors from video 1 current frame
            predicted_offset (int): Predicted frame offset in video 2
            
        Returns:
            dict: Best match result with keys:
                - v2_frame (int): Matched frame index in video 2
                - score (float): Raw match score
                - weighted (float): Weighted score
                - conf (float): Confidence score
                - velocity (float): Current velocity estimate
        """
        if predicted_offset is None:
            predicted_offset = self.last_offset
        
        # Compute current velocity estimate
        velocity = self.compute_velocity()
        
        # Predict center of search window
        predicted_v2_frame = predicted_offset
        
        # BIDIRECTIONAL search window
        # Search both FORWARD and BACKWARD from prediction
        # This is critical to allow correcting errors from previous frames
        search_radius = int(self.max_velocity * SEARCH_RADIUS_MULTIPLIER)  # Adaptive radius
        
        # Define bidirectional search range
        search_start = max(0, predicted_v2_frame - search_radius)
        search_end = min(len(video2_frames), predicted_v2_frame + search_radius)
        
        # Collect all candidate scores
        candidates = []
        
        for v2_idx in range(search_start, search_end):
            if v2_idx >= len(video2_frames):
                break
                
            frame2 = video2_frames[v2_idx]
            kp2, des2 = algo_module.compute_features(frame2)
            
            if des2 is None or len(kp2) < 10:
                continue
            
            # Match features using ratio test
            knn_matches = bf_matcher.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                elif len(match_pair) == 1:
                    m = match_pair[0]
                    if m.distance < SINGLE_MATCH_DISTANCE_THRESHOLD:
                        good_matches.append(m)
            
            if len(good_matches) >= 8:
                # Compute raw score (number of good matches)
                raw_score = len(good_matches)
                candidates.append((raw_score, v2_idx, good_matches, kp2))
        
        if not candidates:
            return None
        
        # Compute confidence based on score distribution
        scores_for_conf = [(score, idx) for score, idx, _, _ in candidates]
        best_candidate_idx = max(range(len(candidates)), key=lambda i: candidates[i][0])
        conf = self.compute_confidence(scores_for_conf, best_candidate_idx)
        
        # Find best match using weighted scoring
        best_weighted_score = -float('inf')
        best_match = None
        
        for raw_score, v2_idx, good_matches, kp2 in candidates:
            distance_from_prediction = abs(v2_idx - predicted_v2_frame)
            weighted_score = self.compute_weighted_score(raw_score, distance_from_prediction, conf)
            
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_match = {
                    'v2_frame': v2_idx,
                    'score': raw_score,
                    'weighted': weighted_score,
                    'conf': conf,
                    'velocity': velocity
                }
        
        # Update history and last offset if match is acceptable
        if best_match and conf >= self.conf_threshold:
            self.history.append({
                'v1_frame': v1_frame_idx,
                'v2_frame': best_match['v2_frame']
            })
            self.last_offset = best_match['v2_frame']
        
        return best_match


def load_video_frames(video_path, start_frame=0, end_frame=None, step=1):
    """
    Load frames from a video file.
    
    Args:
        video_path (str): Path to video file
        start_frame (int): Starting frame index
        end_frame (int): Ending frame index (None for all frames)
        step (int): Step between frames
        
    Returns:
        list: List of frames as numpy arrays
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= start_frame and (end_frame is None or frame_idx < end_frame):
            if (frame_idx - start_frame) % step == 0:
                frames.append(frame)
                
        if end_frame is not None and frame_idx >= end_frame:
            break
            
        frame_idx += 1
    
    cap.release()
    return frames
