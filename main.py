"""
Main Module for Image Alignment on Rail Tracks
This module calls the three algorithms: OCR, BRISK, and AKAZE
"""

import sys
import os
from ocr import run_ocr
from brisk import run_brisk
from akaze import run_akaze


def main():
    """
    Main function that calls all three algorithms.
    """
    print("=" * 60)
    print("Image Alignment on Rail Tracks - Master Thesis Project")
    print("=" * 60)
    print()
    
    # Check if an image path is provided as argument
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
        print(f"Processing image: {image_path}")
    else:
        print("No image path provided. Running algorithms in test mode.")
    
    print()
    
    # Run OCR Algorithm
    print("-" * 60)
    ocr_results = run_ocr(image_path)
    print(f"OCR Results: {ocr_results}")
    print()
    
    # Run BRISK Algorithm
    print("-" * 60)
    brisk_results = run_brisk(image_path)
    print(f"BRISK Results: {brisk_results}")
    print()
    
    # Run AKAZE Algorithm
    print("-" * 60)
    akaze_results = run_akaze(image_path)
    print(f"AKAZE Results: {akaze_results}")
    print()
    
    # Summary
    print("=" * 60)
    print("All algorithms executed successfully!")
    print("=" * 60)
    
    return {
        "ocr": ocr_results,
        "brisk": brisk_results,
        "akaze": akaze_results
    }


if __name__ == "__main__":
    results = main()
