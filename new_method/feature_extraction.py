import cv2
import numpy as np

class VideoFeatureExtractor:
    def __init__(self, resize_dim=(320, 240)):
        self.resize_dim = resize_dim

    def extract_features(self, video_path):
        """
        Extracts features from a video file using Dense Optical Flow.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: Feature matrix of shape (n_frames, n_features).
                        Features: [mean_dx, mean_dy, std_dx, std_dy, mean_mag]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        features = []
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert to grayscale for efficiency
            if self.resize_dim:
                frame_resized = cv2.resize(frame, self.resize_dim)
            else:
                frame_resized = frame

            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            if prev_frame is None:
                prev_frame = gray
                # For the first frame, use zero features to maintain length
                features.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            # Compute Dense Optical Flow using Farneback algorithm
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # flow is (H, W, 2)
            dx = flow[..., 0]
            dy = flow[..., 1]
            mag, _ = cv2.cartToPolar(dx, dy)

            # Compute statistics
            mean_dx = np.mean(dx)
            mean_dy = np.mean(dy)
            std_dx = np.std(dx)
            std_dy = np.std(dy)
            mean_mag = np.mean(mag)

            features.append([mean_dx, mean_dy, std_dx, std_dy, mean_mag])
            prev_frame = gray

        cap.release()
        return np.array(features)
