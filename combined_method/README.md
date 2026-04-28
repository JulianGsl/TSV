# Hybrid Video Alignment (Coarse-to-Fine: DTW + Feature Matching)

Temporal alignment of rail-track videos using a coarse-to-fine pipeline:
- **Macro phase (DTW)** — robust global alignment over dense optical flow
- **Micro phase (AKAZE / BRISK / ORB)** — local refinement via feature matching
- **Fallback** — DTW prediction is kept whenever RANSAC inliers fall below `min_inliers`

The strict `±search_window` bound around the DTW prediction is **never** widened —
this is intentional to prevent drift accumulation.

## Dataset layout

```
dataset/
└── Plan1/
    ├── video1.mp4         # reference (J-1)
    ├── video2.mp4         # target (J)
    └── hybrid_<algo>/     # auto-created outputs (one folder per algorithm)
```

## Running the alignment

### Interactive

```bash
cd Projet/TSV
python combined_method/run_alignment_hybrid.py
```

### CLI

```bash
# One plan, default algorithm (AKAZE)
python combined_method/run_alignment_hybrid.py Plan1

# All plans, BRISK
python combined_method/run_alignment_hybrid.py all BRISK

# Speed / quality knobs
python combined_method/run_alignment_hybrid.py Plan1 \
    --dtw-sample-rate 2 --feature-sample-rate 1 --search-window 15   # slower, more precise

python combined_method/run_alignment_hybrid.py Plan1 \
    --dtw-sample-rate 10 --feature-sample-rate 5 --search-window 8   # faster, less precise
```

Available flags: `--dtw-sample-rate`, `--feature-sample-rate` (alias
`--akaze-sample-rate`), `--search-window`, `--min-inliers`,
`--dtw-step-penalty`.

## Outputs (per plan, per algorithm)

`dataset/Plan1/hybrid_akaze/` (or `hybrid_brisk/`, `hybrid_orb/`):

**Videos**
- `aligned_simple.mp4` — side-by-side
- `aligned_features.mp4` — keypoint overlay
- `aligned_flow.mp4` — optical flow overlay

**Plots**
- `alignment_scatter.png`, `alignment_difference.png`,
  `source_distribution.png`, `velocity_analysis.png`,
  `dtw_cost_matrix.png`

**Data**
- `alignment_results.csv` — forward V1 → V2
- `alignment_results_backward.csv` — cached after first cycle-consistency run
- `metrics.json` — alignment metrics
- `evaluation.json` — perceptual / cycle metrics (after `run_evaluation.py`)
- `report.html` — combined HTML report

## Evaluating an alignment

```bash
python combined_method/run_evaluation.py Plan1 --algorithm AKAZE --metrics all
python combined_method/run_full_evaluation.py Plan1   # all algos × all metrics × all baselines
```

See `../SSIM_LPIPS_Analysis.pdf` for the rationale behind photometric
normalization and ROI options on the perceptual metrics.

## Python API

```python
from combined_method import align_videos_hybrid

matches = align_videos_hybrid(
    "video1.mp4", "video2.mp4",
    algorithm="AKAZE",
    dtw_sample_rate=5,
    feature_sample_rate=1,
    search_window=10,
    min_inliers=4,
)

for m in matches:
    print(f"V1[{m['v1_frame']}] -> V2[{m['v2_frame']}] "
          f"source={m['source']} score={m['score']}")
```

## Module layout

```
combined_method/
├── __init__.py
├── hybrid_alignment.py        # HybridAligner + align_videos_hybrid
├── visualization.py           # plots, videos, HTML report
├── _cli_helpers.py            # shared CLI utilities (plan/algo selection, CSV I/O)
├── evaluation/                # SSIM, LPIPS, cycle consistency, baselines
├── run_alignment_hybrid.py    # alignment entry point
├── run_evaluation.py          # single-plan evaluation
└── run_full_evaluation.py     # full sweep evaluation
```

## Key metrics

| Metric | Meaning |
|---|---|
| Total matches | Frames matched (= length of V1 stream considered) |
| Refined % | Fraction refined by feature matching (vs. DTW fallback) |
| DTW fallback % | Fraction kept at the DTW prediction |
| Monotonicity | % of matches respecting forward progression |
| Velocity ratio | Mean V2/V1 frame-rate ratio (1.0 = same speed) |
| Mean inliers | Average RANSAC inliers in the fine phase |
| Max correction | Largest delta between DTW prediction and final match |
