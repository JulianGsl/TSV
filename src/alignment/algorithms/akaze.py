"""
AKAZE Algorithm Implementation
This module implements AKAZE (Accelerated-KAZE) for image alignment.
"""

import cv2
import numpy as np
from ..image_utils import preprocess_image


def compute_features(image):
    """
    Computes features (keypoints and descriptors) for the given image using AKAZE.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: (keypoints, descriptors)
    """
    if image is None:
        return [], None

    # Initialize AKAZE detector with optimized parameters
    # descriptor_type: DESCRIPTOR_MLDB for better performance
    # descriptor_size: 0 (full size) for maximum precision
    # threshold: Lowered to 0.0001 to detect more features in low-texture areas
    akaze = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size=0,
        threshold=0.0001
    )

    # Preprocess image (Grayscale + CLAHE)
    processed_image = preprocess_image(image)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = akaze.detectAndCompute(processed_image, None)

    return keypoints, descriptors


def run_akaze(image_path=None):
    """
    Legacy wrapper for standalone execution/testing.
    """
    print("Running AKAZE Algorithm...")
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "message": "Failed to load image"}

        kp, desc = compute_features(image)
        print(f"Number of keypoints detected: {len(kp)}")
        return {
            "algorithm": "AKAZE",
            "status": "success",
            "keypoints_count": len(kp)
        }

    return {"status": "no_image"}


if __name__ == "__main__":
    pass
