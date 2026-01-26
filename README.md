# TSV
Master Thesis job - Image alignment on rail track

## Description

This project implements three different algorithms for image alignment on rail tracks:
- **OCR** (Optical Character Recognition)
- **BRISK** (Binary Robust Invariant Scalable Keypoints)
- **AKAZE** (Accelerated-KAZE)

## Project Structure

```
TSV/
├── main.py           # Main file that calls all three algorithms
├── ocr.py            # OCR algorithm implementation
├── brisk.py          # BRISK algorithm implementation
├── akaze.py          # AKAZE algorithm implementation
├── dataset/          # Directory for videos and images
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run all three algorithms

```bash
python main.py
```

Or with an image:
```bash
python main.py path/to/image.jpg
```

### Run individual algorithms

```bash
python ocr.py
python brisk.py
python akaze.py
```

## Dataset

Place your videos and images in the `dataset/` directory. The user is responsible for configuring the photos.
