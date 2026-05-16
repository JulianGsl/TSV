#!/usr/bin/env python3
"""
DTW Full Diagnostic for Plan2
=============================

Runs one configuration end-to-end and produces a structured diagnostic
folder with every intermediate artefact and plot needed to understand
where and why DTW diverges from the manual ground truth.

Output layout
-------------
    dataset/Plan2/dtw_analysis_<timestamp>/
        REPORT.md                              ← main analysis (read this first)
        config.json                            ← what was tested
        01_features/
            v1_raw.npy, v2_raw.npy
            v1_transformed.npy, v2_transformed.npy
            v1_indices.npy, v2_indices.npy
            feature_heatmap_v1.png             ← features over time (V1)
            feature_heatmap_v2.png
            feature_norm_over_time.png         ← per-frame energy
            feature_pca_2d.png                 ← PCA scatter w/ anchors
        02_matrices/
            dist_matrix.npy, acc_cost.npy
            dist_matrix.png                    ← heatmap + truth + DTW path
            acc_cost.png
            dist_along_truth.png               ← cost values along truth
            zoomed_anchors.png                 ← dist around each anchor
        03_path/
            dtw_path.npy
            path_vs_truth_frame.png            ← key visual: paths in frame coords
            path_vs_truth_sampled.png
            path_error_along_v1.png            ← signed error
            step_distribution.png              ← diagonal vs ins/del counts
        04_anchors/
            per_anchor.csv                     ← detailed table
            anchor_rank_in_dist.png            ← percentile in dist row
            anchor_neighborhood.png            ← dist[v1_idx, near v2_truth]
            anchor_feature_similarity.png      ← dist truth / dist to noise
        05_diagnostics/
            cost_along_paths.png               ← cumulative + per-step
            divergence_map.png                 ← where & how far DTW drifts

Usage
-----
    # default config (matches current pipeline baseline)
    python combined_method/dtw_diagnostic/analyze.py

    # try another config
    python combined_method/dtw_diagnostic/analyze.py \
        --extractor grad_hist_3x3_grid --transform per_video_zscore \
        --sample-rate 5 --step-penalty 0.3
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # TSV/
sys.path.insert(0, project_root)

from combined_method.dtw_core import compute_dtw
from combined_method.dtw_diagnostic.ground_truth import (
    PLAN2_ANCHORS, get_anchors, interpolate_truth,
)
from combined_method.dtw_diagnostic.extractors import EXTRACTORS, extract_features
from combined_method.dtw_diagnostic.transforms import TRANSFORMS
from combined_method.dtw_confidence import (
    compute_dtw_confidence, project_to_frames, find_rescue_zones,
)
from combined_method.preprocessing import Preprocessor


DATASET_DIR = os.path.join(project_root, "dataset")
PLAN = "Plan2"


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def map_original_to_sampled(idx_array, target_frame):
    return int(np.argmin(np.abs(np.asarray(idx_array) - target_frame)))


def anchors_to_sampled(anchors, idx1, idx2):
    return [
        (map_original_to_sampled(idx1, v1), map_original_to_sampled(idx2, v2))
        for v1, v2 in anchors
    ]


def truth_pairs_in_sampled(anchors_sampled):
    """Piecewise-linear truth path in sampled-index space."""
    a = sorted(anchors_sampled)
    out = []
    for k in range(len(a) - 1):
        i0, j0 = a[k]
        i1, j1 = a[k + 1]
        steps = max(abs(i1 - i0), abs(j1 - j0))
        if steps == 0:
            out.append((i0, j0))
            continue
        for s in range(steps + 1):
            t = s / steps
            out.append((int(round(i0 + t * (i1 - i0))),
                        int(round(j0 + t * (j1 - j0)))))
    # dedupe preserving order
    seen = set()
    dedup = []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def cost_along(dist, pairs):
    n, m = dist.shape
    vals = []
    for i, j in pairs:
        if 0 <= i < n and 0 <= j < m:
            vals.append(float(dist[i, j]))
    return np.asarray(vals)


def extract_features_maybe_preprocessed(video_path, extractor_name, sample_rate, pp, is_v2):
    """Wrap extract_features with optional luminance LUT applied frame-by-frame.

    If pp is None or the video is V1 (is_v2=False), behaves identically to
    extract_features. If pp is set and is_v2=True, every read frame goes
    through the LUT before the extractor sees it.
    """
    if pp is None:
        return extract_features(video_path, extractor_name, sample_rate)

    import cv2
    from combined_method.dtw_diagnostic.extractors import EXTRACTORS
    fn = EXTRACTORS[extractor_name]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    feats, idx = [], []
    i = -1
    while True:
        ret, frame = cap.read()
        i += 1
        if not ret:
            break
        if i % sample_rate != 0:
            continue
        if is_v2:
            frame = pp.apply_to_v2(frame)
        v = fn(frame).astype(np.float32)
        s = v.sum()
        if s > 1e-6:
            v = v / s
        feats.append(v)
        idx.append(i)
    cap.release()
    return np.asarray(feats, dtype=np.float32), idx


def save_preprocessing_artefacts(out_dir, pp):
    """Plot the V1, V2-original, V2-after-LUT luminance histograms side by side."""
    pdir = os.path.join(out_dir, "00_preprocessing")
    os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, "luminance_lut.npy"), pp.lut)
    ref, tgt_orig, tgt_remap = pp.histograms(n_samples=60)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ref / ref.sum(), color="steelblue", lw=1.5, label="V1 (reference)")
    axes[0].plot(tgt_orig / tgt_orig.sum(), color="darkorange", lw=1.5,
                 label="V2 (original)")
    axes[0].plot(tgt_remap / tgt_remap.sum(), color="green", lw=1.5,
                 label="V2 (after LUT)")
    axes[0].set_xlabel("L channel value (0-255)")
    axes[0].set_ylabel("frequency")
    axes[0].set_title("Luminance distributions before/after histogram matching")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(pp.lut, color="purple", lw=1.5)
    axes[1].plot([0, 255], [0, 255], "--", color="gray", alpha=0.5, label="identity")
    axes[1].set_xlabel("L (V2 original)")
    axes[1].set_ylabel("L (V2 remapped)")
    axes[1].set_title("LUT applied to V2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "histogram_matching.png"), dpi=130)
    plt.close(fig)


def save_confidence_artefacts(out_dir, dist, path_arr, idx1, idx2, total_frames_v1, anchors):
    """Plot DTW confidence per V1 frame + the detected rescue zones."""
    cdir = os.path.join(out_dir, "06_confidence")
    os.makedirs(cdir, exist_ok=True)
    conf_wp = compute_dtw_confidence(dist, path_arr)
    conf_pf = project_to_frames(path_arr, conf_wp, np.asarray(idx1), total_frames_v1)
    zones = find_rescue_zones(conf_pf, threshold=0.25, min_length=20)
    np.save(os.path.join(cdir, "confidence_per_waypoint.npy"), conf_wp)
    np.save(os.path.join(cdir, "confidence_per_frame.npy"), conf_pf)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(conf_pf, color="steelblue", lw=1.2, label="DTW confidence")
    ax.axhline(0.25, color="red", ls="--", alpha=0.5, label="rescue threshold (0.25)")
    for z0, z1 in zones:
        ax.axvspan(z0, z1, color="red", alpha=0.15)
    for v1, _ in anchors:
        ax.axvline(v1, color="#ff00ff", lw=0.5, alpha=0.5)
    ax.set_xlabel("V1 frame")
    ax.set_ylabel("confidence (0 = noise, 1 = row min)")
    ax.set_title(f"DTW confidence along V1 — {len(zones)} rescue zone(s) (red), "
                 "ground-truth anchors (magenta)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cdir, "confidence_per_frame.png"), dpi=130)
    plt.close(fig)

    # Confidence histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(conf_pf, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(0.25, color="red", ls="--", alpha=0.5, label="rescue threshold")
    ax.set_xlabel("confidence value")
    ax.set_ylabel("# V1 frames")
    ax.set_title("Distribution of DTW confidence across V1 frames")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cdir, "confidence_histogram.png"), dpi=130)
    plt.close(fig)

    return {"n_zones": len(zones),
            "mean_conf": float(conf_pf.mean()),
            "low_conf_pct": float((conf_pf < 0.25).mean() * 100),
            "zones": zones}


# ----------------------------------------------------------------------
# Phase 1: Features
# ----------------------------------------------------------------------

def save_feature_artefacts(out_dir, f1_raw, f2_raw, f1, f2, idx1, idx2, anchors):
    """Save .npy + plots for the feature phase."""
    fdir = os.path.join(out_dir, "01_features")
    os.makedirs(fdir, exist_ok=True)
    np.save(os.path.join(fdir, "v1_raw.npy"), f1_raw)
    np.save(os.path.join(fdir, "v2_raw.npy"), f2_raw)
    np.save(os.path.join(fdir, "v1_transformed.npy"), f1)
    np.save(os.path.join(fdir, "v2_transformed.npy"), f2)
    np.save(os.path.join(fdir, "v1_indices.npy"), np.asarray(idx1, dtype=np.int32))
    np.save(os.path.join(fdir, "v2_indices.npy"), np.asarray(idx2, dtype=np.int32))

    # Heatmaps (one per video)
    for label, feats, idx, fname in [
        ("V1", f1, idx1, "feature_heatmap_v1.png"),
        ("V2", f2, idx2, "feature_heatmap_v2.png"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(feats.T, aspect="auto", origin="lower", cmap="viridis",
                       extent=[idx[0], idx[-1], 0, feats.shape[1]])
        plt.colorbar(im, ax=ax, label="feature value")
        # mark anchors on V1 heatmap
        if label == "V1":
            for v1, _ in anchors:
                if idx[0] <= v1 <= idx[-1]:
                    ax.axvline(v1, color="#ff00ff", lw=0.7, alpha=0.7)
        else:
            for _, v2 in anchors:
                if idx[0] <= v2 <= idx[-1]:
                    ax.axvline(v2, color="#ff00ff", lw=0.7, alpha=0.7)
        ax.set_xlabel(f"{label} frame")
        ax.set_ylabel("feature dim")
        ax.set_title(f"{label} features over time (vertical lines = ground-truth anchors)")
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, fname), dpi=130)
        plt.close(fig)

    # Per-frame norm
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(idx1, np.linalg.norm(f1, axis=1), color="steelblue", label="V1 ‖feat‖")
    ax.plot(idx2, np.linalg.norm(f2, axis=1), color="darkorange", label="V2 ‖feat‖", alpha=0.8)
    for v1, _ in anchors:
        ax.axvline(v1, color="#ff00ff", lw=0.5, alpha=0.4)
    ax.set_xlabel("frame")
    ax.set_ylabel("L2 norm of transformed feature")
    ax.set_title("Feature energy over time (vertical pink = V1 anchors)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "feature_norm_over_time.png"), dpi=130)
    plt.close(fig)

    # PCA 2D scatter
    try:
        cat = np.vstack([f1, f2])
        cat = cat - cat.mean(axis=0)
        # SVD-based PCA
        U, S, Vt = np.linalg.svd(cat, full_matrices=False)
        proj = cat @ Vt[:2].T   # (N, 2)
        n1 = len(f1)
        p1 = proj[:n1]
        p2 = proj[n1:]

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.scatter(p1[:, 0], p1[:, 1], s=12, c="steelblue", alpha=0.6, label="V1 frames")
        ax.scatter(p2[:, 0], p2[:, 1], s=12, c="darkorange", alpha=0.6, label="V2 frames")
        # Connect anchor pairs
        for v1, v2 in anchors:
            i = map_original_to_sampled(idx1, v1)
            j = map_original_to_sampled(idx2, v2)
            ax.plot([p1[i, 0], p2[j, 0]], [p1[i, 1], p2[j, 1]],
                    "-", color="#ff00ff", lw=1.2, alpha=0.8)
            ax.scatter([p1[i, 0]], [p1[i, 1]], s=40, c="#ff00ff", edgecolor="white", zorder=5)
            ax.scatter([p2[j, 0]], [p2[j, 1]], s=40, c="#ff00ff", edgecolor="white", zorder=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA 2D — magenta lines link V1↔V2 truth anchors\n(if features are good, lines should be short)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, "feature_pca_2d.png"), dpi=130)
        plt.close(fig)

        anchor_dists = []
        for v1, v2 in anchors:
            i = map_original_to_sampled(idx1, v1)
            j = map_original_to_sampled(idx2, v2)
            anchor_dists.append(float(np.linalg.norm(p1[i] - p2[j])))
        return anchor_dists
    except Exception:
        return []


# ----------------------------------------------------------------------
# Phase 2: Matrices
# ----------------------------------------------------------------------

def save_matrix_artefacts(out_dir, dist, acc, path_arr, anchors_sampled):
    mdir = os.path.join(out_dir, "02_matrices")
    os.makedirs(mdir, exist_ok=True)
    np.save(os.path.join(mdir, "dist_matrix.npy"), dist)
    np.save(os.path.join(mdir, "acc_cost.npy"), acc)

    def _heatmap(matrix, title, fname, log=False):
        fig, ax = plt.subplots(figsize=(12, 9))
        kw = {"aspect": "auto", "origin": "lower", "cmap": "viridis"}
        if log:
            kw["norm"] = LogNorm(vmin=max(matrix.min(), 1e-3), vmax=matrix.max())
        im = ax.imshow(matrix, **kw)
        plt.colorbar(im, ax=ax, label="cost" + (" (log)" if log else ""))
        # Diagonal
        n, m = matrix.shape
        ax.plot([0, m - 1], [0, n - 1], "--", color="gray", alpha=0.5, label="diagonal")
        # DTW path
        ax.plot(path_arr[:, 1], path_arr[:, 0], "-", color="orange", lw=2.0, label="DTW path")
        # Ground truth
        gt = sorted(anchors_sampled)
        ax.plot([a[1] for a in gt], [a[0] for a in gt], "-",
                color="#ff00ff", lw=2.2, label="Manual ground truth (lerp)")
        ax.scatter([a[1] for a in gt], [a[0] for a in gt], s=70,
                   c="#ff00ff", edgecolors="white", linewidths=1.4, zorder=5)
        ax.set_xlabel("V2 sampled idx")
        ax.set_ylabel("V1 sampled idx")
        ax.set_title(title)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(mdir, fname), dpi=130)
        plt.close(fig)

    _heatmap(dist, "Raw distance matrix — DTW (orange) vs truth (magenta)",
             "dist_matrix.png", log=True)
    _heatmap(acc, "Accumulated cost (DP) — DTW (orange) vs truth (magenta)",
             "acc_cost.png", log=True)

    # Cost along truth path
    truth_pairs = truth_pairs_in_sampled(anchors_sampled)
    cv = cost_along(dist, truth_pairs)
    dtw_pairs = [(int(i), int(j)) for i, j in path_arr]
    dv = cost_along(dist, dtw_pairs)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(cv, color="#ff00ff", lw=1.3, label=f"along truth path (n={len(cv)})")
    ax.plot(dv, color="orange", lw=1.3, label=f"along DTW path  (n={len(dv)})", alpha=0.8)
    ax.axhline(float(dist.mean()), color="gray", ls="--", alpha=0.5,
               label=f"matrix mean = {dist.mean():.3f}")
    ax.set_xlabel("step along path")
    ax.set_ylabel("dist value at (i, j)")
    ax.set_title("Raw distance values along truth path vs DTW path\n"
                 "(if truth is much above mean → features don't make truth a minimum)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(mdir, "dist_along_truth.png"), dpi=130)
    plt.close(fig)

    # Zoomed anchor crosshairs (grid of 14)
    n_anchors = len(anchors_sampled)
    ncols = 4
    nrows = (n_anchors + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0))
    axes = np.array(axes).reshape(-1)
    n, m = dist.shape
    R = 12  # zoom radius (sampled idx)
    for k, (i, j) in enumerate(anchors_sampled):
        ax = axes[k]
        i0, i1 = max(0, i - R), min(n, i + R + 1)
        j0, j1 = max(0, j - R), min(m, j + R + 1)
        sub = dist[i0:i1, j0:j1]
        ax.imshow(sub, aspect="auto", origin="lower",
                  extent=[j0, j1, i0, i1], cmap="viridis")
        ax.scatter([j], [i], s=80, c="#ff00ff", edgecolors="white", lw=1.5, zorder=5)
        # mark row minimum
        rmin = int(np.argmin(dist[i, j0:j1])) + j0
        ax.scatter([rmin], [i], s=60, c="red", marker="x", lw=2, zorder=4)
        ax.set_title(f"#{k} V1={PLAN2_ANCHORS[k][0]} V2={PLAN2_ANCHORS[k][1]}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n_anchors, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Zoomed dist matrix around each anchor — pink = truth, red ✗ = row minimum",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(mdir, "zoomed_anchors.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# Phase 3: Path
# ----------------------------------------------------------------------

def save_path_artefacts(out_dir, path_arr, idx1, idx2, anchors):
    pdir = os.path.join(out_dir, "03_path")
    os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, "dtw_path.npy"), path_arr)

    v1_path = np.asarray(idx1)[path_arr[:, 0]]
    v2_path = np.asarray(idx2)[path_arr[:, 1]]

    # Frame-space
    fig, ax = plt.subplots(figsize=(11, 8))
    sa = sorted(anchors)
    lim = max(max(idx1), max(idx2), max(a[0] for a in sa), max(a[1] for a in sa))
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.4, label="diagonal")
    ax.plot([a[0] for a in sa], [a[1] for a in sa], "-",
            color="#ff00ff", lw=2.5, label="Manual ground truth", zorder=4)
    ax.scatter([a[0] for a in sa], [a[1] for a in sa], s=80,
               c="#ff00ff", edgecolors="white", lw=1.5, zorder=5)
    ax.plot(v1_path, v2_path, "-", color="orange", lw=2.0, label="DTW path")
    ax.scatter([v1_path[0]], [v2_path[0]], s=80, c="lime", edgecolors="black", label="DTW start", zorder=5)
    ax.scatter([v1_path[-1]], [v2_path[-1]], s=80, c="red", edgecolors="black", label="DTW end", zorder=5)
    ax.set_xlabel("V1 frame")
    ax.set_ylabel("V2 frame")
    ax.set_title("DTW path vs manual ground truth (frame coordinates)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "path_vs_truth_frame.png"), dpi=130)
    plt.close(fig)

    # Sampled-space (raw indices)
    fig, ax = plt.subplots(figsize=(11, 8))
    n = max(path_arr[:, 0].max(), 1)
    m = max(path_arr[:, 1].max(), 1)
    sa_s = sorted(anchors_to_sampled(anchors, idx1, idx2))
    ax.plot(path_arr[:, 1], path_arr[:, 0], "-", color="orange", lw=2, label="DTW path")
    ax.plot([a[1] for a in sa_s], [a[0] for a in sa_s], "-",
            color="#ff00ff", lw=2, label="truth (sampled)")
    ax.scatter([a[1] for a in sa_s], [a[0] for a in sa_s], c="#ff00ff",
               edgecolors="white", lw=1.5, s=70, zorder=5)
    ax.set_xlabel("V2 sampled idx")
    ax.set_ylabel("V1 sampled idx")
    ax.set_title("Same plot in sampled-index space")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "path_vs_truth_sampled.png"), dpi=130)
    plt.close(fig)

    # Path error along V1
    errs, v1_eval = [], []
    for v1, v2 in zip(v1_path, v2_path):
        t = interpolate_truth(int(v1))
        if t is None:
            continue
        errs.append(int(v2) - t)
        v1_eval.append(int(v1))
    if errs:
        errs = np.asarray(errs)
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.axhspan(-30, 30, color="green", alpha=0.10, label="±30 frames")
        ax.axhline(0, color="black", lw=0.7)
        ax.plot(v1_eval, errs, "-", color="orange", lw=1.6, label="DTW − truth")
        for v1, v2 in anchors:
            k = int(np.argmin(np.abs(np.asarray(v1_eval) - v1)))
            if 0 <= k < len(v1_eval):
                ax.scatter([v1_eval[k]], [errs[k]], s=60, c="#ff00ff",
                           edgecolors="white", lw=1.4, zorder=5)
        ax.set_xlabel("V1 frame")
        ax.set_ylabel("DTW − truth (frames)")
        ax.set_title("Signed frame error along V1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, "path_error_along_v1.png"), dpi=130)
        plt.close(fig)

    # Step distribution
    di = np.diff(path_arr[:, 0])
    dj = np.diff(path_arr[:, 1])
    diag = int(((di == 1) & (dj == 1)).sum())
    insert = int(((di == 1) & (dj == 0)).sum())   # V1 advances, V2 stays
    delete = int(((di == 0) & (dj == 1)).sum())   # V2 advances, V1 stays
    other = len(di) - diag - insert - delete

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["diagonal", "V1+ only\n(V2 waits)", "V2+ only\n(V1 waits)", "other"],
           [diag, insert, delete, other],
           color=["steelblue", "darkorange", "purple", "gray"])
    for k, v in enumerate([diag, insert, delete, other]):
        ax.text(k, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("count of steps")
    ax.set_title(f"DTW step type distribution (total = {len(di)})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "step_distribution.png"), dpi=130)
    plt.close(fig)
    return {"diag": diag, "insert": insert, "delete": delete, "other": other}


# ----------------------------------------------------------------------
# Phase 4: Anchors
# ----------------------------------------------------------------------

def save_anchor_artefacts(out_dir, dist, path_arr, idx1, idx2, anchors):
    adir = os.path.join(out_dir, "04_anchors")
    os.makedirs(adir, exist_ok=True)

    anchors_s = anchors_to_sampled(anchors, idx1, idx2)
    v1_path = np.asarray(idx1)[path_arr[:, 0]]
    v2_path = np.asarray(idx2)[path_arr[:, 1]]

    # Build per-anchor table
    rows = []
    for k, ((v1, v2), (i, j)) in enumerate(zip(anchors, anchors_s)):
        row = dist[i]
        rank_strict = int((row < row[j]).sum())
        pct = 100.0 * rank_strict / max(1, len(row) - 1)
        row_min = float(row.min())
        row_min_j = int(np.argmin(row))
        truth_dist = float(row[j])
        # DTW prediction nearest to this V1 frame
        kp = int(np.argmin(np.abs(v1_path - v1)))
        v2_pred = int(v2_path[kp])
        err = v2_pred - v2
        rows.append({
            "anchor": k,
            "v1_truth": v1, "v2_truth": v2,
            "v1_sampled_idx": i, "v2_sampled_idx": j,
            "v2_dtw_pred": v2_pred, "abs_err": abs(err), "signed_err": err,
            "dist_at_truth": truth_dist,
            "row_min_dist": row_min,
            "row_min_v2_sampled": row_min_j,
            "truth_pct_in_row": round(pct, 2),
            "rank_strict": rank_strict,
        })

    with open(os.path.join(adir, "per_anchor.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plot: percentile of truth in each row
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    labels = [f"#{r['anchor']}\nV1={r['v1_truth']}" for r in rows]
    pcts = [r["truth_pct_in_row"] for r in rows]
    axes[0].bar(labels, pcts, color="steelblue")
    axes[0].axhline(50, color="red", ls="--", alpha=0.6, label="random (50%)")
    axes[0].axhline(5, color="green", ls="--", alpha=0.6, label="discriminative (<5%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percentile of truth in row")
    axes[0].set_title("How discriminative is each row? "
                      "(0% = truth is the row min → DTW would pick it)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    x = np.arange(len(labels))
    truth_d = [r["dist_at_truth"] for r in rows]
    min_d = [r["row_min_dist"] for r in rows]
    axes[1].bar(x - 0.2, truth_d, 0.4, color="orange", label="dist at truth")
    axes[1].bar(x + 0.2, min_d, 0.4, color="darkgreen", label="row min")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("distance")
    axes[1].set_title("Truth's distance vs the cheapest distance in the row")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(adir, "anchor_rank_in_dist.png"), dpi=130)
    plt.close(fig)

    # Anchor neighborhood — for each anchor, plot dist[v1_idx, :] zoomed
    n_anchors = len(anchors_s)
    ncols = 2
    nrows = (n_anchors + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 2.3))
    axes = np.array(axes).reshape(-1)
    n, m = dist.shape
    for k, (i, j) in enumerate(anchors_s):
        ax = axes[k]
        ax.plot(dist[i], color="steelblue", lw=0.8)
        ax.axvline(j, color="#ff00ff", lw=2, label="truth")
        ax.axvline(int(np.argmin(dist[i])), color="red", ls=":", lw=1.5, label="row min")
        ax.set_title(f"#{k} V1={PLAN2_ANCHORS[k][0]} | dist row, truth pct = {rows[k]['truth_pct_in_row']}%",
                     fontsize=9)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(fontsize=8)
    for k in range(n_anchors, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Distance row at each V1 anchor — is truth (magenta) a clear minimum?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(adir, "anchor_neighborhood.png"), dpi=130)
    plt.close(fig)

    # Anchor feature similarity: dist at truth vs distribution of dists from same row
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, (i, j) in enumerate(anchors_s):
        row = dist[i]
        ax.boxplot(row, positions=[k], widths=0.6, showfliers=False,
                   medianprops=dict(color="black"))
        ax.scatter([k], [row[j]], s=60, c="#ff00ff", edgecolors="white", lw=1.4, zorder=5)
    ax.set_xticks(range(len(anchors_s)))
    ax.set_xticklabels([f"#{k}" for k in range(len(anchors_s))])
    ax.set_ylabel("distance value")
    ax.set_title("For each anchor row: distribution of all distances (box) vs distance at truth (magenta ●)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(adir, "anchor_feature_similarity.png"), dpi=130)
    plt.close(fig)

    return rows


# ----------------------------------------------------------------------
# Phase 5: Diagnostics (path-cost reasoning)
# ----------------------------------------------------------------------

def save_diagnostic_artefacts(out_dir, dist, path_arr, anchors_sampled, idx1, idx2, anchors):
    ddir = os.path.join(out_dir, "05_diagnostics")
    os.makedirs(ddir, exist_ok=True)

    dtw_pairs = [(int(i), int(j)) for i, j in path_arr]
    truth_pairs = truth_pairs_in_sampled(anchors_sampled)
    L = min(dist.shape)
    diag_pairs = [(i, i) for i in range(L)]

    dtw_v = cost_along(dist, dtw_pairs)
    truth_v = cost_along(dist, truth_pairs)
    diag_v = cost_along(dist, diag_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(np.cumsum(dtw_v),   color="orange",  lw=1.7, label=f"DTW (n={len(dtw_v)})")
    axes[0].plot(np.cumsum(truth_v), color="#ff00ff", lw=1.7, label=f"truth (n={len(truth_v)})")
    axes[0].plot(np.cumsum(diag_v),  color="gray",    lw=1.3, label=f"diagonal (n={len(diag_v)})", alpha=0.7)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cumulative dist")
    axes[0].set_title("Cumulative cost along candidate paths")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    means = {
        "DTW": dtw_v.mean() if dtw_v.size else 0.0,
        "truth": truth_v.mean() if truth_v.size else 0.0,
        "diagonal": diag_v.mean() if diag_v.size else 0.0,
    }
    bars = axes[1].bar(list(means.keys()), list(means.values()),
                       color=["orange", "#ff00ff", "gray"])
    for k, v in means.items():
        axes[1].text(k, v, f"{v:.4f}", ha="center", va="bottom")
    axes[1].set_ylabel("mean dist / step")
    axes[1].set_title("Mean per-step cost (DTW vs truth vs diagonal)\nLower = path is cheaper")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Path-cost diagnostic — does the cost function reward the truth?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ddir, "cost_along_paths.png"), dpi=130)
    plt.close(fig)

    # Divergence map: scatter v1 → abs error
    v1_path = np.asarray(idx1)[path_arr[:, 0]]
    v2_path = np.asarray(idx2)[path_arr[:, 1]]
    pts = []
    for v1, v2 in zip(v1_path, v2_path):
        t = interpolate_truth(int(v1))
        if t is None:
            continue
        pts.append((int(v1), abs(int(v2) - t)))
    if pts:
        pts = np.asarray(pts)
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c=pts[:, 1], cmap="Reds", alpha=0.7)
        ax.axhline(10, color="green", ls="--", alpha=0.5, label="±10 (AKAZE window)")
        ax.axhline(30, color="orange", ls="--", alpha=0.5, label="±30")
        for v1, _ in anchors:
            ax.axvline(v1, color="#ff00ff", lw=0.5, alpha=0.4)
        ax.set_xlabel("V1 frame")
        ax.set_ylabel("|DTW − truth|  (frames)")
        ax.set_title("Where DTW drifts the most along V1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(ddir, "divergence_map.png"), dpi=130)
        plt.close(fig)

    return means


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def write_report(out_dir, cfg, summary, anchor_rows, step_counts, means, pca_anchor_dists,
                 conf_summary=None):
    rep = []
    rep.append(f"# DTW Full Diagnostic — Plan2\n")
    rep.append(f"- Generated : {datetime.now().isoformat(timespec='seconds')}")
    rep.append(f"- Folder    : `{out_dir}`\n")

    rep.append("## Configuration\n")
    for k, v in cfg.items():
        rep.append(f"- **{k}** : {v}")
    rep.append("")

    rep.append("## Quick stats\n")
    rep.append(f"- Mean abs error  : **{summary['mean_abs']:.1f} frames**")
    rep.append(f"- Median abs err  : {summary['median_abs']:.1f} frames")
    rep.append(f"- Max abs error   : {summary['max_abs']} frames")
    rep.append(f"- Anchors ≤30     : {summary['within_30']}/14")
    rep.append(f"- Anchors ≤50     : {summary['within_50']}/14")
    rep.append(f"- DTW endpoint    : V1[{summary['endpoint_v1']}] ↔ V2[{summary['endpoint_v2']}]")
    rep.append(f"- Truth endpoint  : V1[{PLAN2_ANCHORS[-1][0]}] ↔ V2[{PLAN2_ANCHORS[-1][1]}]")
    rep.append("")

    rep.append("### DTW step types\n")
    total = sum(step_counts.values())
    for k, v in step_counts.items():
        pct = 100.0 * v / max(1, total)
        rep.append(f"- {k:20s} : {v:5d}  ({pct:5.1f}%)")
    rep.append("")

    rep.append("### Path cost comparison (mean dist / step)\n")
    for k, v in means.items():
        rep.append(f"- {k:8s} : {v:.4f}")
    if means.get("DTW", 0) < means.get("truth", 0):
        rep.append("\n> **DTW cost < truth cost** — the cost function does NOT reward the truth.\n"
                   "> Improving features (or filtering noise) is the path forward.\n")
    else:
        rep.append("\n> DTW cost ≥ truth cost — truth is the cheapest path. If DTW still misses,\n"
                   "> the issue is open-end / step-penalty / monotonicity bias, not features.\n")

    rep.append("## Per-anchor table\n")
    rep.append("| # | V1 truth | V2 truth | V2 DTW | abs err | dist at truth | row min | truth pct |")
    rep.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in anchor_rows:
        rep.append(
            f"| {r['anchor']} | {r['v1_truth']} | {r['v2_truth']} | {r['v2_dtw_pred']} | "
            f"{r['abs_err']} | {r['dist_at_truth']:.4f} | {r['row_min_dist']:.4f} | "
            f"{r['truth_pct_in_row']:.1f}% |"
        )
    rep.append("")

    pcts = [r["truth_pct_in_row"] for r in anchor_rows]
    rep.append("## Hypothesis testing\n")

    rep.append("### H1 — Are features discriminative at anchor rows?\n")
    rep.append(f"- Mean percentile of truth in its dist row : **{np.mean(pcts):.1f}%**")
    rep.append(f"- Anchors below 5%  (clearly discriminative) : {sum(1 for p in pcts if p < 5)}/14")
    rep.append(f"- Anchors above 30% (noise-level)            : {sum(1 for p in pcts if p > 30)}/14")
    if np.mean(pcts) > 25:
        rep.append("\n→ **Features are weak.** Truth is not a clear minimum in its row. "
                   "Try better extractors or normalisation. See `04_anchors/anchor_neighborhood.png`.")
    elif np.mean(pcts) < 10:
        rep.append("\n→ **Features are strong** at anchors. Investigate why DTW still misses them "
                   "(open-end endpoint, step penalty, sample rate).")
    rep.append("")

    rep.append("### H2 — Is the truth path globally cheap?\n")
    rep.append(f"- DTW mean cost/step   : {means.get('DTW', 0):.4f}")
    rep.append(f"- Truth mean cost/step : {means.get('truth', 0):.4f}")
    rep.append(f"- Ratio truth / DTW    : {means.get('truth', 0) / max(means.get('DTW', 1e-9), 1e-9):.3f}")
    if means.get("DTW", 0) < means.get("truth", 0):
        rep.append("\n→ DTW finds a path cheaper than the truth. The cost function "
                   "is **misaligned with the truth**; no DTW tweak alone will fix this.")
    rep.append("")

    rep.append("### H3 — Is DTW terminating at the right endpoint?\n")
    dv1 = summary["endpoint_v1"] - PLAN2_ANCHORS[-1][0]
    dv2 = summary["endpoint_v2"] - PLAN2_ANCHORS[-1][1]
    rep.append(f"- V1 endpoint offset : {dv1:+d}")
    rep.append(f"- V2 endpoint offset : {dv2:+d}")
    if abs(dv1) > 50 or abs(dv2) > 50:
        rep.append("\n→ Open-end DTW terminates far from truth. The DP may be choosing "
                   "a cheap-but-wrong endpoint; consider constrained-end DTW or a "
                   "termination penalty.")
    rep.append("")

    rep.append("### H4 — Does the DTW step distribution look healthy?\n")
    diag_pct = 100.0 * step_counts["diag"] / max(1, total)
    rep.append(f"- Diagonal steps : {diag_pct:.1f}%")
    if diag_pct < 30:
        rep.append("→ Very few diagonal steps — DTW is making lots of unit ins/del moves. "
                   "Increase step_penalty to bias toward diagonal.")
    elif diag_pct > 90:
        rep.append("→ Almost all-diagonal — DTW is essentially returning the naive diagonal. "
                   "Decrease step_penalty or check features.")
    else:
        rep.append("→ Step distribution looks reasonable.")
    rep.append("")

    rep.append("### H5 — PCA sanity check\n")
    if pca_anchor_dists:
        m = float(np.mean(pca_anchor_dists))
        rep.append(f"- Mean PC1-PC2 distance between V1 and V2 truth pairs : {m:.3f}")
        rep.append("- See `01_features/feature_pca_2d.png`. Short magenta lines = good.")
    rep.append("")

    if conf_summary is not None:
        rep.append("### H6 — DTW confidence & rescue zones\n")
        rep.append(f"- Mean confidence across V1 frames : {conf_summary['mean_conf']:.3f}")
        rep.append(f"- % of frames below 0.25 (rescue threshold) : {conf_summary['low_conf_pct']:.1f}%")
        rep.append(f"- Rescue zones detected (≥20 contiguous low-conf frames) : {conf_summary['n_zones']}")
        for k, (z0, z1) in enumerate(conf_summary.get("zones", [])):
            rep.append(f"  - zone #{k}: V1 frames [{z0}, {z1}]  ({z1 - z0 + 1} frames)")
        rep.append("- See `06_confidence/confidence_per_frame.png` — red bands = rescue zones.")
        rep.append("")

    rep.append("## Files in this folder\n")
    rep.append("```")
    for root, _, files in os.walk(out_dir):
        rel = os.path.relpath(root, out_dir)
        prefix = "" if rel == "." else f"{rel}/"
        for f in sorted(files):
            rep.append(f"  {prefix}{f}")
    rep.append("```")

    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(rep))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DTW full diagnostic for Plan2.")
    p.add_argument("--extractor", default="grad_hist_16",
                   help=f"Feature extractor. Default matches pipeline baseline.")
    p.add_argument("--transform", default="per_video_zscore",
                   help="Feature transform.")
    p.add_argument("--sample-rate", type=int, default=10,
                   help="DTW sample rate.")
    p.add_argument("--step-penalty", type=float, default=0.3,
                   help="DTW step penalty.")
    p.add_argument("--out", default=None, help="Output dir (default: auto-timestamp).")
    p.add_argument("--preprocess", action="store_true",
                   help="Apply luminance histogram matching to V2 before feature extraction.")
    args = p.parse_args()

    if args.extractor not in EXTRACTORS:
        print(f"ERROR: unknown extractor '{args.extractor}'. Available: {list(EXTRACTORS)}")
        sys.exit(1)
    if args.transform not in TRANSFORMS:
        print(f"ERROR: unknown transform '{args.transform}'. Available: {list(TRANSFORMS)}")
        sys.exit(1)

    plan_dir = os.path.join(DATASET_DIR, PLAN)
    v1_path = os.path.join(plan_dir, "video1.mp4")
    v2_path = os.path.join(plan_dir, "video2.mp4")
    if not (os.path.exists(v1_path) and os.path.exists(v2_path)):
        print(f"ERROR: missing videos in {plan_dir}")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(plan_dir, f"dtw_analysis_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = {
        "plan": PLAN,
        "extractor": args.extractor,
        "transform": args.transform,
        "sample_rate": args.sample_rate,
        "step_penalty": args.step_penalty,
        "preprocess": args.preprocess,
        "n_anchors": 14,
        "timestamp": ts,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"DTW Full Diagnostic — Plan2")
    print(f"  extractor    : {args.extractor}")
    print(f"  transform    : {args.transform}")
    print(f"  sample rate  : {args.sample_rate}")
    print(f"  step penalty : {args.step_penalty}")
    print(f"  output       : {out_dir}\n")

    anchors = get_anchors(PLAN)

    # Optional preprocessing: build the LUT, save the before/after histogram plot,
    # and patch extract_features so V2 frames go through the LUT.
    pp = None
    if args.preprocess:
        print("[0/5] Building luminance LUT...")
        t0 = time.time()
        pp = Preprocessor(v1_path, v2_path, n_samples=60, enabled=True)
        print(f"     LUT built in {time.time() - t0:.1f}s")
        save_preprocessing_artefacts(out_dir, pp)

    # Phase 1: features
    print("[1/5] Extracting features...")
    t0 = time.time()
    f1_raw, idx1 = extract_features_maybe_preprocessed(v1_path, args.extractor, args.sample_rate, pp, is_v2=False)
    f2_raw, idx2 = extract_features_maybe_preprocessed(v2_path, args.extractor, args.sample_rate, pp, is_v2=True)
    print(f"     V1 {f1_raw.shape}, V2 {f2_raw.shape}  ({time.time() - t0:.1f}s)")
    f1, f2 = TRANSFORMS[args.transform](f1_raw, f2_raw)

    # Phase 2: DTW
    print("[2/5] Running Open-End DTW...")
    t0 = time.time()
    path, acc, dist = compute_dtw(f1, f2, metric="euclidean",
                                   step_penalty=args.step_penalty, open_end=True)
    print(f"     dist {dist.shape}, path {len(path)}  ({time.time() - t0:.1f}s)")
    path_arr = np.asarray(path, dtype=np.int32)
    anchors_sampled = anchors_to_sampled(anchors, idx1, idx2)

    # Summary
    v1_p = np.asarray(idx1)[path_arr[:, 0]]
    v2_p = np.asarray(idx2)[path_arr[:, 1]]
    abs_errs = []
    for v1t, v2t in anchors:
        k = int(np.argmin(np.abs(v1_p - v1t)))
        abs_errs.append(abs(int(v2_p[k]) - v2t))
    summary = {
        "mean_abs": float(np.mean(abs_errs)),
        "median_abs": float(np.median(abs_errs)),
        "max_abs": int(np.max(abs_errs)),
        "within_30": int(sum(1 for e in abs_errs if e <= 30)),
        "within_50": int(sum(1 for e in abs_errs if e <= 50)),
        "endpoint_v1": int(idx1[path_arr[-1, 0]]),
        "endpoint_v2": int(idx2[path_arr[-1, 1]]),
    }

    print(f"\n>>> mean abs err: {summary['mean_abs']:.1f}  | "
          f"endpoint V1[{summary['endpoint_v1']}] ↔ V2[{summary['endpoint_v2']}]\n")

    # Phase 3 — Save artefacts
    print("[3/5] Saving feature artefacts + plots...")
    pca_dists = save_feature_artefacts(out_dir, f1_raw, f2_raw, f1, f2, idx1, idx2, anchors)

    print("[4/5] Saving matrices + plots...")
    save_matrix_artefacts(out_dir, dist, acc, path_arr, anchors_sampled)

    print("[4/5] Saving path + anchor + diagnostic plots...")
    step_counts = save_path_artefacts(out_dir, path_arr, idx1, idx2, anchors)
    anchor_rows = save_anchor_artefacts(out_dir, dist, path_arr, idx1, idx2, anchors)
    means = save_diagnostic_artefacts(out_dir, dist, path_arr, anchors_sampled,
                                       idx1, idx2, anchors)

    # Compute total V1 frames for the per-frame confidence projection.
    import cv2 as _cv2
    cap = _cv2.VideoCapture(v1_path)
    total_v1 = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    conf_summary = save_confidence_artefacts(out_dir, dist, path_arr,
                                              idx1, idx2, total_v1, anchors)

    print("[5/5] Writing REPORT.md...")
    write_report(out_dir, cfg, summary, anchor_rows, step_counts, means, pca_dists,
                 conf_summary=conf_summary)

    print(f"\n✓ DONE")
    print(f"  Open report: file://{os.path.abspath(os.path.join(out_dir, 'REPORT.md'))}")
    print(f"  Folder     : {out_dir}")


if __name__ == "__main__":
    main()
