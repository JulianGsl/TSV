import unittest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.alignment.image_utils import preprocess_image
from src.alignment.algorithms import orb, brisk, akaze

class TestAlignment(unittest.TestCase):
    def test_preprocess_image(self):
        # Create a dummy image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = preprocess_image(img)

        self.assertIsNotNone(processed)
        self.assertEqual(len(processed.shape), 2) # Should be grayscale
        self.assertEqual(processed.shape, (100, 100))

    def test_algorithms(self):
        # Create a dummy image with some structure (random noise)
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        for name, module in [("ORB", orb), ("BRISK", brisk), ("AKAZE", akaze)]:
            kp, des = module.compute_features(img)
            # We don't expect matches on random noise, but it shouldn't crash
            # And it should return a tuple
            self.assertIsInstance(kp, tuple)
            # des can be None if no features found
            if des is not None:
                self.assertIsInstance(des, np.ndarray)

if __name__ == '__main__':
    unittest.main()
