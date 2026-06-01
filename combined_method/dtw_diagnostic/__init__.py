"""
DTW diagnostic toolkit.

Standalone subsystem to investigate DTW behaviour against ground-truth
anchors. Not imported by the production pipeline; meant to be run as
scripts. See README.md in this directory for the per-script breakdown.

The five entry-point scripts are:
- analyze.py            full diagnostic for one configuration
- sweep.py              multi-configuration leaderboard
- compute_dtw_only.py   fast iteration helper (DTW only, no AKAZE)
- diagnose.py           post-mortem on an existing hybrid run

Reusable building blocks:
- ground_truth.py       per-plan anchors + interpolation
- extractors.py         per-frame feature extractors
- transforms.py         sequence-level transforms (z-score variants)
"""
