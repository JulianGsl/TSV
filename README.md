# TSV
Master Thesis job - Image alignment on rail track

## Description

This project implements three different algorithms for image alignment on rail tracks:
- **ORB** (Oriented FAST and Rotated BRIEF)
- **BRISK** (Binary Robust Invariant Scalable Keypoints)
- **AKAZE** (Accelerated-KAZE)

The goal is to align two videos of the same track taken at different speeds.

## Project Structure

```
.
├── data/               # Directory containing Plan folders (videos)
├── scripts/            # Executable scripts
│   ├── run_alignment.py      # Main entry point
│   └── generate_test_data.py # Helper to create dummy videos
├── src/                # Source code
│   └── alignment/      # Core logic package
│       ├── algorithms/ # ORB, BRISK, AKAZE implementations
│       ├── core.py     # Alignment logic
│       ├── utils.py    # Post-processing
│       └── visualization.py
└── requirements.txt    # Dependencies
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
Place your videos in the `data/` directory using the following structure:
```
data/
  Plan1/
    video1.mp4
    video2.mp4
  Plan2/
    ...
```

If you don't have data, you can generate a test plan:
```bash
python scripts/generate_test_data.py
```

### 2. Run Alignment
Execute the main script:
```bash
python scripts/run_alignment.py
```
Follow the interactive prompts to select a plan, or run with arguments:
```bash
python scripts/run_alignment.py PlanTest
```

## Outputs

For each algorithm (ORB, BRISK, AKAZE), the script generates:
- `alignment_<algo>_raw.csv`: Raw alignment data.
- `alignment_<algo>_clean.csv`: Post-processed (smoothed) alignment data.
- `plot_<algo>.png`: Visualization of the alignment path.
- `comparison_<algo>.mp4`: Side-by-side video comparison.
- `report_<algo>.html`: Summary report.
