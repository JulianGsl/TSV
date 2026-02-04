"""
ORB Algorithm Implementation
This module implements ORB (Oriented FAST and Rotated BRIEF) for image alignment.
"""

import cv2
import numpy as np
from ..image_utils import preprocess_image


def compute_features(image):
    """
    Computes features (keypoints and descriptors) for the given image using ORB.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: (keypoints, descriptors)
    """
    if image is None:
        return [], None

    # Initialize ORB detector with increased feature count for better precision
    # nfeatures: 5000 features (up from 500 default) for more reliable matching
    # scaleFactor: 1.2 for better scale invariance
    # nlevels: 8 pyramid levels for multi-scale detection
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)

    # Preprocess image (Grayscale + CLAHE)
    processed_image = preprocess_image(image)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(processed_image, None)

    return keypoints, descriptors


def run_orb(image_path=None):
    """
    Legacy wrapper for standalone execution/testing.
    """
    print("Running ORB Algorithm...")
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "message": "Failed to load image"}

        kp, desc = compute_features(image)
        print(f"Number of keypoints detected: {len(kp)}")
        return {
            "algorithm": "ORB",
            "status": "success",
            "keypoints_count": len(kp)
        }

    return {"status": "no_image"}

if __name__ == "__main__":
    pass
