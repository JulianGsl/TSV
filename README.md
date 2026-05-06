# TSV — Temporal Synchronization of Videos

Master thesis project: temporal alignment of two rail-track videos filmed at
different speeds. The active pipeline lives in `combined_method/`; earlier
generations are archived under `Old_version/` for reference.

## Repository layout

```
Projet/TSV/
├── combined_method/       # Active pipeline — hybrid coarse-to-fine
│   ├── hybrid_alignment.py
│   ├── visualization.py
│   ├── _cli_helpers.py
│   ├── evaluation/        # SSIM, LPIPS, cycle, baselines
│   ├── run_alignment_hybrid.py
│   ├── run_evaluation.py
│   └── run_full_evaluation.py
│
├── Old_version/           # Archived earlier generations
│   ├── DTW.ipynb          # Early DTW exploration notebook
│   ├── new_method/        # DTW-only library (still imported by combined_method)
│   └── feature_matching/  # Greedy + Kalman baseline
│
├── Documentation/         # Reference papers, design notes, pipeline docs
│   ├── notes_tsv.pdf      # Full literature review (15 algorithms)
│   ├── Pipeline_EN.pdf    # Pipeline description (English)
│   ├── Pipeline_FR.pdf    # Pipeline description (French)
│   ├── SSIM_LPIPS_Analysis.pdf
│   └── ... (algorithm papers, metric papers)
│
├── dataset/               # Input videos + per-plan outputs
│   ├── Plan1/
│   │   ├── video1.mp4     # reference (J-1)
│   │   ├── video2.mp4     # target (J)
│   │   └── hybrid_<algo>/ # auto-created outputs (akaze / brisk / orb)
│   ├── Plan2/
│   └── Plan3/
│
└── requirements.txt
```

## Generations of the algorithm

Three generations were developed during the project. Only the third is active.

| Package | Approach | Status | Location |
|---|---|---|---|
| `feature_matching/` | Greedy frame-by-frame ORB / BRISK / AKAZE + Kalman filter | Archived | `Old_version/` |
| `new_method/` | DTW skeleton over dense optical flow features (open-end mode) | Archived (still imported by `combined_method` for the DTW core) | `Old_version/` |
| `combined_method/` | Hybrid coarse-to-fine — DTW (macro) + feature matching (micro) | **Active** | Root |

> Note: `combined_method/hybrid_alignment.py` imports `VideoFeatureExtractor` and
> `compute_dtw` from `Old_version/new_method/`. The `Old_version/` folder is not
> deprecated code — its DTW utilities remain part of the live pipeline.

## Setup

```bash
cd Projet/TSV
pip install -r requirements.txt
```

Core dependencies: `opencv-python`, `numpy`, `matplotlib`, `scipy`.
The evaluation suite additionally uses `lpips` (PyTorch) for the perceptual metric.

## Running the alignment (current pipeline)

```bash
cd Projet/TSV

# Interactive (asks for plan and algorithm)
python combined_method/run_alignment_hybrid.py

# One plan, default algorithm (AKAZE)
python combined_method/run_alignment_hybrid.py Plan1

# One plan, specific algorithm
python combined_method/run_alignment_hybrid.py Plan1 BRISK

# All plans, ORB
python combined_method/run_alignment_hybrid.py all ORB
```

### Speed / quality knobs

```bash
# Faster, less precise
python combined_method/run_alignment_hybrid.py Plan1 \
    --dtw-sample-rate 10 --feature-sample-rate 5 --search-window 8

# Slower, more precise
python combined_method/run_alignment_hybrid.py Plan1 \
    --dtw-sample-rate 2  --feature-sample-rate 1 --search-window 15
```

Available flags: `--dtw-sample-rate`, `--feature-sample-rate` (alias
`--akaze-sample-rate`), `--search-window`, `--min-inliers`,
`--dtw-step-penalty`. See `python combined_method/run_alignment_hybrid.py --help`.

## Running the archived baselines (for comparison only)

```bash
# Greedy feature-matching baseline (Old_version/feature_matching/)
python Old_version/feature_matching/scripts/run_alignment.py            # interactive
python Old_version/feature_matching/scripts/run_alignment.py PlanTest   # one plan
python Old_version/feature_matching/scripts/generate_test_data.py       # dummy videos
```

These scripts are kept to reproduce the early-experiment results discussed in
chapter 4 of the thesis. They are not part of the active pipeline.

## Evaluating an alignment

The active pipeline ships a perceptual + cycle-consistency evaluation suite under
`combined_method/evaluation/`. It loads a previously-computed alignment from
`alignment_results.csv` (so you must have run the alignment first).

```bash
# Evaluate a single cached alignment
python combined_method/run_evaluation.py Plan1 --algorithm AKAZE --metrics all
python combined_method/run_evaluation.py Plan1 --metrics lpips,ssim
python combined_method/run_evaluation.py all   --metrics cycle --algorithm BRISK

# Full sweep: all algos × all metrics × all baselines (random, linear, offset±k)
python combined_method/run_full_evaluation.py Plan1
```

### Metrics

- **SSIM** — Structural Similarity (Wang 2004), with photometric normalization
  enabled by default to isolate alignment quality from illumination differences.
- **LPIPS** — Learned Perceptual Image Patch Similarity (Zhang 2018), AlexNet
  backbone by default; VGG available for higher discrimination.
- **Cycle consistency** — Round-trip frame error `|i − g(f(i))|`,
  fully content-agnostic. The most discriminating metric on this dataset.

See `Documentation/SSIM_LPIPS_Analysis.pdf` for the rationale behind the
photometric normalization and ROI options.

## Outputs (per plan, per algorithm)

For `python combined_method/run_alignment_hybrid.py Plan1 AKAZE`, the directory
`dataset/Plan1/hybrid_akaze/` is created and populated with:

```
hybrid_akaze/
├── alignment_results.csv          # forward V1 → V2 mapping (canonical output)
├── alignment_results_backward.csv # backward V2 → V1 (cached after first cycle eval)
├── metrics.json                   # alignment metrics (refinement %, monotonicity, ...)
├── evaluation_akaze.json          # SSIM/LPIPS/cycle results (after run_evaluation)
├── alignment_scatter.png          # V1 vs V2 frame scatter
├── alignment_difference.png       # corrections from DTW prediction
├── source_distribution.png        # refined vs fallback breakdown
├── velocity_analysis.png          # velocity ratio over time
├── dtw_cost_matrix.png            # DTW cost matrix + path
├── aligned_simple.mp4             # side-by-side video
├── aligned_features.mp4           # with keypoints overlaid
├── aligned_flow.mp4               # with optical flow overlay
└── report.html                    # combined HTML report
```

## Tests

```bash
cd Projet/TSV
python -m pytest Old_version/new_method/tests/
```

(Tests cover the DTW core; they are kept under `Old_version/` along with the
module they test.)

## Documentation

The `Documentation/` folder contains:

- **`notes_tsv.pdf`** — full literature review of 15 feature-detection algorithms
  (classical, binary, deep learning) with the elimination tree leading to the
  retained AKAZE / BRISK / ORB.
- **`Pipeline_EN.pdf` / `Pipeline_FR.pdf`** — narrative description of the
  4-phase hybrid pipeline.
- **`SSIM_LPIPS_Analysis.pdf`** — design rationale for photometric normalization
  and ROI in the perceptual metrics.
- Reference papers for each retained algorithm (AKAZE, BRISK, ORB) and for the
  evaluation metrics (SSIM, LPIPS).
- Subfolder `metrics/` — per-metric notes (SSIM and LPIPS, French).
- Subfolder `tsv/` — domain-specific papers (rail-track inspection, defect
  detection, image registration in repetitive environments).
