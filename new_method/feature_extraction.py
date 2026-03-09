import cv2
import numpy as np

class VideoFeatureExtractor:
    def __init__(self, resize_dim=(320, 240), sample_rate=1, ignore_sky=True):
        self.resize_dim = resize_dim
        self.sample_rate = sample_rate
        self.ignore_sky = ignore_sky

    def extract_features(self, video_path):
        """
        Extracts features from a video file using Dense Optical Flow.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: Feature matrix of shape (n_frames, n_features).
            list: Original frame indices corresponding to each feature row.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        features = []
        frame_indices = []
        prev_frame = None
        frame_idx = -1

        while True:
            ret, frame = cap.read()
            frame_idx += 1

            if not ret:
                break

            if frame_idx % self.sample_rate != 0:
                continue

            # Resize and convert to grayscale for efficiency
            if self.resize_dim:
                frame_resized = cv2.resize(frame, self.resize_dim)
            else:
                frame_resized = frame

            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # Apply spatial filter: ignore top 33% (sky) where clouds/lighting cause noise
            # Just like the legacy code logic does implicitly or explicitly.
            if self.ignore_sky:
                h = gray.shape[0]
                gray_roi = gray[int(h/3):, :]
            else:
                gray_roi = gray

            if prev_frame is None:
                prev_frame = gray_roi
                # First frame features: 12 elements (3 regions * 4 stats)
                features.append(np.zeros(12))
                frame_indices.append(frame_idx)
                continue

            # Compute Dense Optical Flow using Farneback algorithm
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # flow is (H, W, 2)
            dx = flow[..., 0]
            dy = flow[..., 1]
            mag, ang = cv2.cartToPolar(dx, dy)

            # Spatial pooling: divide the ROI into 3 vertical regions (Left, Center, Right)
            # This captures the parallax effect (edges move faster than center on a train)
            w = gray_roi.shape[1]
            w3 = w // 3

            regions = [
                (0, w3),          # Left
                (w3, 2*w3),       # Center
                (2*w3, w)         # Right
            ]

            frame_features = []

            for (x_start, x_end) in regions:
                r_dx = dx[:, x_start:x_end]
                r_dy = dy[:, x_start:x_end]
                r_mag = mag[:, x_start:x_end]

                # Compute statistics for this region
                mean_dx = np.mean(r_dx)
                mean_dy = np.mean(r_dy)
                std_mag = np.std(r_mag)
                mean_mag = np.mean(r_mag)

                frame_features.extend([mean_dx, mean_dy, std_mag, mean_mag])

            features.append(frame_features)
            frame_indices.append(frame_idx)
            prev_frame = gray_roi

        cap.release()
        return np.array(features), frame_indices
