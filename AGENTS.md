# AGENTS Guide

## Big Picture (what matters first)
- This repo has **3 alignment pipelines**: `feature_matching/` (keypoint-only), `new_method/` (optical-flow + DTW), and `combined_method/` (DTW-guided keypoint + fusion).
- Core data flow is always: **video pair -> frame/frame-feature alignment path -> post-processing -> artifacts** (`*.csv`, `plot.png`, comparison videos, `report.html`).
- `feature_matching/src/alignment/core.py` is the baseline matcher (ORB/BRISK/AKAZE + ratio test + RANSAC + adaptive search).
- `new_method/feature_extraction.py` + `new_method/dtw_alignment.py` provide optical-flow descriptors and DTW pathing (used standalone and reused by combined methods).
- `combined_method/core.py` is the integration layer: (1) coarse DTW predictor, (2) local feature refinement, (3) DTW fallback when features fail.

## Where to run commands (important in this repo)
- Many scripts use **cwd-relative** dataset paths (`../dataset`, `../../dataset`), not `__file__`-relative resolution.
- Run baseline pipeline from `feature_matching/scripts/`:
  - `python run_alignment.py Plan1` or `python run_alignment.py all`
- Run combined pipeline from `combined_method/`:
  - `python run_alignment.py Plan1`
- Run DTW pipeline from `new_method/` (or pass explicit video paths if run elsewhere):
  - `python run_alignment.py ../dataset/Plan2/video1.mp4 ../dataset/Plan2/video2.mp4 --sample-rate 2 --penalty 1.5`
- Unit tests currently live in `new_method/tests/` and are `unittest`-style:
  - `python -m unittest discover -s new_method/tests -p "test_*.py"`

## Project-Specific Conventions
- Algorithm contract is module-based: each algorithm in `feature_matching/src/alignment/algorithms/*.py` exposes `compute_features(image) -> (keypoints, descriptors)`.
- Algorithm names are uppercase in code (`"ORB"`, `"BRISK"`, `"AKAZE"`) but output folders are lowercase (`orb/`, `brisk/`, `akaze/`).
- Alignment records are dicts with stable keys: `v1_frame`, `v2_frame`, `score` (plus optional `algorithm`). Keep this schema to stay compatible with CSV/report/video utilities.
- Matching logic consistently uses BFMatcher Hamming + Lowe ratio (`0.75`) + RANSAC homography inlier count as score.
- Post-processing is centralized in `feature_matching/src/alignment/utils.py::filter_outliers_and_smooth`; avoid duplicating smoothing rules elsewhere.

## Integration Points You Should Reuse
- Reuse `feature_matching/src/alignment/visualization.py` for plots/videos/HTML reports instead of new rendering code.
- `combined_method/core.py` imports both `new_method` and `feature_matching` components; treat it as the reference for cross-pipeline composition.
- `combined_method/run_alignment.py` already orchestrates per-plan outputs and should be extended before creating new top-level runners.

## Known Frictions / Gotchas
- `README.md` documents a `data/` layout, but active runners use `dataset/` in practice.
- Root `src/alignment/` exists but active implementations are under `feature_matching/src/alignment/`.
- `new_method/run_alignment.py` writes H.264 (`avc1`) then falls back to `mp4v`; codec availability can vary by machine.
- Output naming is mostly stable (`alignment_raw.csv`, `alignment_clean.csv`, `plot.png`, `comparison_*.mp4`, `report.html`) and downstream analysis assumes these names.

## Fast Onboarding Checklist for Agents
- Inspect one real plan folder under `dataset/Plan*/` before changing output formats.
- When changing alignment scoring, update both baseline (`feature_matching/.../core.py`) and combined logic if behavior should stay comparable.
- Keep sample-rate/search-window trade-offs explicit in CLI args; defaults encode speed-vs-accuracy assumptions used in existing experiments.

