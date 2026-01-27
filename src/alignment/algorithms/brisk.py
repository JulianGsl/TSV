"""
BRISK Algorithm Implementation
This module implements BRISK (Binary Robust Invariant Scalable Keypoints) for image alignment.
"""

import cv2
import numpy as np


def compute_features(image):
    """
    Computes features (keypoints and descriptors) for the given image using BRISK.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: (keypoints, descriptors)
    """
    if image is None:
        return [], None

    # Initialize BRISK detector
    brisk = cv2.BRISK_create()

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect keypoints and compute descriptors
    keypoints, descriptors = brisk.detectAndCompute(gray, None)

    return keypoints, descriptors


def run_brisk(image_path=None):
    """
    Legacy wrapper for standalone execution/testing.
    """
    print("Running BRISK Algorithm...")
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "message": "Failed to load image"}

        kp, desc = compute_features(image)
        print(f"Number of keypoints detected: {len(kp)}")
        return {
            "algorithm": "BRISK",
            "status": "success",
            "keypoints_count": len(kp)
        }

    return {"status": "no_image"}


if __name__ == "__main__":
    pass
