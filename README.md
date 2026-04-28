# TSV — Temporal Synchronization of Videos

Master thesis project: temporal alignment of two rail-track videos filmed at
different speeds.

## Algorithms

Three generations of alignment algorithm, each in its own package:

| Package | Approach | Status |
|---|---|---|
| `feature_matching/` | Greedy frame-by-frame ORB / BRISK / AKAZE + Kalman filter | Baseline |
| `new_method/`       | DTW skeleton over dense optical flow features (open-end mode) | Backbone |
| `combined_method/`  | Hybrid coarse-to-fine — DTW (macro) + feature matching (micro) | **Current** |

## Setup

```bash
cd Projet/TSV
pip install -r requirements.txt
```

Dependencies: `opencv-python`, `numpy`, `matplotlib`, `scipy`. The combined
method's evaluation suite additionally uses `lpips` (PyTorch).

## Data layout

```
Projet/TSV/dataset/
└── Plan1/
    ├── video1.mp4    # reference (J-1)
    ├── video2.mp4    # target (J)
    └── hybrid_<algo>/  # auto-created outputs
```

## Running the alignment

### Recommended (hybrid coarse-to-fine)

```bash
cd Projet/TSV
python combined_method/run_alignment_hybrid.py            # interactive
python combined_method/run_alignment_hybrid.py Plan1      # one plan
python combined_method/run_alignment_hybrid.py all AKAZE  # all plans, AKAZE
```

Speed / quality knobs:

```bash
# Faster, less precise
python combined_method/run_alignment_hybrid.py Plan1 \
    --dtw-sample-rate 10 --feature-sample-rate 5 --search-window 8

# Slower, more precise
python combined_method/run_alignment_hybrid.py Plan1 \
    --dtw-sample-rate 2  --feature-sample-rate 1 --search-window 15
```

### Feature-matching only (baseline)

```bash
python feature_matching/scripts/run_alignment.py            # interactive
python feature_matching/scripts/run_alignment.py PlanTest   # one plan
python feature_matching/scripts/generate_test_data.py       # dummy videos
```

## Evaluating an alignment

The combined method ships with a perceptual + cycle-consistency evaluation
suite under `combined_method/evaluation/`.

```bash
# Evaluate a single cached alignment
python combined_method/run_evaluation.py Plan1 --algorithm AKAZE --metrics all

# Full sweep: all algos × all metrics × all baselines (random, linear, offset±k)
python combined_method/run_full_evaluation.py Plan1
```

Metrics:
- **SSIM** — structural similarity (Wang 2004), with photometric normalization
  enabled by default to isolate alignment quality from illumination.
- **LPIPS** — learned perceptual similarity (Zhang 2018), AlexNet by default,
  VGG available for higher discrimination.
- **Cycle consistency** — round-trip frame error `|i − g(f(i))|`, fully
  content-agnostic.

See `SSIM_LPIPS_Analysis.pdf` for the design rationale of the photometric
normalization and ROI options.

## Tests

```bash
cd Projet/TSV
python -m pytest new_method/tests/
```

## Project layout

```
Projet/TSV/
├── combined_method/        # current pipeline (hybrid)
│   ├── hybrid_alignment.py
│   ├── visualization.py
│   ├── evaluation/         # SSIM, LPIPS, cycle, baselines
│   ├── run_alignment_hybrid.py
│   ├── run_evaluation.py
│   └── run_full_evaluation.py
├── new_method/             # DTW backbone (library + tests)
├── feature_matching/       # baseline (greedy + Kalman)
├── dataset/                # input videos + per-plan outputs
└── requirements.txt
```
