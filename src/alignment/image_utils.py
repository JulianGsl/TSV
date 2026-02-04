"""
Image Processing Utilities
"""

import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocesses the image for feature detection.
    Applies Grayscale conversion and CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image (numpy.ndarray): Input image (BGR or Grayscale).

    Returns:
        numpy.ndarray: Preprocessed grayscale image.
    """
    if image is None:
        return None

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply CLAHE
    # clipLimit: 2.0 (default is usually fine, 2-4 is good for contrast)
    # tileGridSize: (8, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced
