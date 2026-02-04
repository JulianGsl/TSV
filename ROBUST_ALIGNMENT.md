# Robust Video Alignment Implementation

## Overview

This implementation adds robust video alignment capabilities with bidirectional search to the TSV project. The key improvement is the ability to search both forward AND backward from the predicted frame position, allowing the algorithm to correct errors made in previous frames.

## Key Components

### 1. RobustAligner Class (`src/alignment/robust_alignment.py`)

The `RobustAligner` class implements:

- **Confidence scoring**: Measures how distinctly better the best match is compared to alternatives
  - Formula: `conf = 1 - (second_best / best)`
  - Higher confidence means the match is more reliable

- **Velocity tracking**: Tracks the rate of change of frame offsets
  - Computed from recent match history
  - Used to predict the next frame position

- **Weighted scoring**: Combines multiple factors for robust matching
  - Formula: `weighted = raw_score × conf × exp(-0.1 × |distance|)`
  - Balances match quality, confidence, and proximity to prediction

- **Bidirectional search**: Critical feature that searches both forward and backward
  - Searches around the predicted position in BOTH directions
  - Allows correcting errors from previous frames
  - Does not limit search to only forward frames

### 2. Enhanced Core Alignment (`src/alignment/core.py`)

Added `align_videos_robust()` function that:

- Uses the `RobustAligner` class
- Implements bidirectional search window
- Returns additional metrics in results:
  - `score`: Raw match score (number of inliers)
  - `weighted`: Weighted score combining all factors
  - `velocity`: Current velocity estimate
  - `conf`: Confidence score (0-1)

### 3. Updated CSV Export (`scripts/run_alignment.py`)

Modified to:

- Use `align_videos_robust()` instead of `align_videos()`
- Export new columns: `weighted`, `velocity`, `conf`
- Maintain backward compatibility with existing columns

## Metrics Explanation

| Metric | Formula | Usage |
|--------|---------|-------|
| `score` | Number of RANSAC inliers | Raw quality of feature matches |
| `conf` | 1 - (2nd_best / best) | Reliability, avoids ambiguity |
| `velocity` | mean(Δoffset) | Detects drift, predicts next frame |
| `weighted` | score × conf × exp(-0.1×\|dist\|) | Combined final score |

## Critical Feature: Bidirectional Search

The most important aspect of this implementation is the **bidirectional search window**:

```python
# Predict center of search based on velocity
predicted_v2_frame = last_best_v2_frame + int(sample_rate * velocity)

# BIDIRECTIONAL search: both forward AND backward
search_radius = search_window // 2
search_start = max(0, predicted_v2_frame - search_radius)
search_end = predicted_v2_frame + search_radius
```

This allows the algorithm to:
1. Search forward from the prediction (normal case)
2. Search backward from the prediction (correction case)
3. Correct errors from previous frames
4. Avoid accumulating drift over time

## Usage Example

```python
from src.alignment.core import align_videos_robust

results = align_videos_robust(
    video1_path="path/to/video1.mp4",
    video2_path="path/to/video2.mp4",
    algo_name="BRISK",
    sample_rate=15,
    search_window=150,
    search_step=10
)

# Each result contains:
# - v1_frame: Frame index in video 1
# - v2_frame: Matched frame index in video 2
# - score: Raw match score
# - weighted: Weighted score
# - velocity: Velocity estimate
# - conf: Confidence score
# - algorithm: Algorithm name
```

## Testing

Run tests with:
```bash
python /tmp/test_robust_aligner.py
```

Generate test data and run alignment:
```bash
python scripts/generate_test_data.py
cd scripts
python run_alignment.py PlanTest
```

## Output Files

For each algorithm, the following files are generated:

- `alignment_raw.csv`: Raw alignment data with all metrics
- `alignment_clean.csv`: Post-processed (smoothed) alignment data
- `plot.png`: Visualization of the alignment path
- `comparison_*.mp4`: Side-by-side video comparisons
- `report.html`: Summary report

The CSV files now include the new columns: `weighted`, `velocity`, and `conf`.
