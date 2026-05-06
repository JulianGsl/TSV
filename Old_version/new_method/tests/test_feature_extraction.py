import cv2
import numpy as np
import unittest
import os
from Old_version.new_method.feature_extraction import VideoFeatureExtractor

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        # Create a dummy video file
        self.video_path = "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 20.0, (640, 480))
        for _ in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def tearDown(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)

    def test_extract_features(self):
        extractor = VideoFeatureExtractor(resize_dim=(160, 120))
        features, indices = extractor.extract_features(self.video_path)

        # Check shape (10 frames, 12 features: 3 regions * 4 stats)
        self.assertEqual(features.shape, (10, 12))
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices, list(range(10)))

        # Check that features are not all zero (except maybe first frame)
        self.assertTrue(np.any(features[1:]))

    def test_sample_rate(self):
        extractor = VideoFeatureExtractor(resize_dim=(160, 120), sample_rate=2)
        features, indices = extractor.extract_features(self.video_path)

        # Should process frames 0, 2, 4, 6, 8
        self.assertEqual(features.shape, (5, 12))
        self.assertEqual(indices, [0, 2, 4, 6, 8])

if __name__ == "__main__":
    unittest.main()
