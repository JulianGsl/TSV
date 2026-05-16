#!/usr/bin/env python3
"""
DTW Sweep
=========

Iterates over combinations of (feature extractor × transform × step_penalty),
runs DTW for each, scores against the manual ground truth, and ranks them.

Folder layout produced
----------------------
    dataset/<Plan>/dtw_sweep_<timestamp>/
        results.csv                  # full ranked table (machine-readable)
        results.md                   # ranked table (human-readable)
        leaderboard.png              # bar chart of mean abs error per config
        config_costs.png             # DTW vs truth cost-per-step per config
        top_<rank>_<config_name>/    # full artefacts + plots, top-K only
            meta.json
            dtw_acc_cost.npy
            dtw_dist_matrix.npy
            dtw_path.npy
            dtw_indices1.npy
            dtw_indices2.npy
            02_path_in_frame_space.png

Usage
-----
    python combined_method/dtw_diagnostic/sweep.py Plan2
    python combined_method/dtw_diagnostic/sweep.py Plan2 --sample-rate 5 --top-k 10

Add or remove configs by editing EXTRACTORS / TRANSFORMS / STEP_PENALTIES below.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # TSV/
sys.path.insert(0, project_root)

from combined_method.dtw_core import compute_dtw
from combined_method.dtw_diagnostic.ground_truth import get_anchors, interpolate_truth
from combined_method.dtw_diagnostic.extractors import EXTRACTORS, extract_features
from combined_method.dtw_diagnostic.transforms import TRANSFORMS
from combined_method.dtw_diagnostic.diagnose import (
    plot_path_in_frame_space,
    map_original_to_sampled,
)


DATASET_DIR = os.path.join(project_root, "dataset")

STEP_PENALTIES = [0.0, 0.1, 0.3, 0.5]


def extract_features_for_video(video_path, extractor_fn, sample_rate):
    """Legacy wrapper kept for the sweep loop below (uses extractor_fn directly)."""
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
        v = extractor_fn(frame).astype(np.float32)
        s = v.sum()
        if s > 1e-6:
            v = v / s
        feats.append(v)
        idx.append(i)
    cap.release()
    return np.asarray(feats, dtype=np.float32), idx


def evaluate_path(path, idx1, idx2, anchors):
    """Mean abs error of DTW path against piecewise-linear ground truth."""
    v1_path = np.asarray(idx1, dtype=np.int32)[path[:, 0]]
    v2_path = np.asarray(idx2, dtype=np.int32)[path[:, 1]]
    abs_errs = []
    per_anchor = []
    for v1t, v2t in anchors:
        if len(v1_path) == 0:
            per_anchor.append(None)
            continue
        k = int(np.argmin(np.abs(v1_path - v1t)))
        v2p = int(v2_path[k])
        d = v2p - v2t
        per_anchor.append((v1t, v2t, v2p, d))
        abs_errs.append(abs(d))
    if not abs_errs:
        return float("inf"), float("inf"), 0, per_anchor
    return float(np.mean(abs_errs)), float(np.median(abs_errs)), int(np.max(abs_errs)), per_anchor


def cost_per_step(dist, pairs):
    n, m = dist.shape
    s, c = 0.0, 0
    for i, j in pairs:
        if 0 <= i < n and 0 <= j < m:
            s += dist[i, j]
            c += 1
    return s / c if c else float("inf")


def truth_pairs_in_sampled_space(anchors, idx1, idx2):
    """Piecewise-linear ground-truth path in sampled-index space."""
    a = sorted([(map_original_to_sampled(np.asarray(idx1), v1),
                 map_original_to_sampled(np.asarray(idx2), v2)) for v1, v2 in anchors])
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
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def write_results_csv(out_path, rows):
    cols = ["rank", "config", "extractor", "transform", "step_penalty",
            "mean_abs_err", "median_abs_err", "max_abs_err",
            "anchors_within_30", "anchors_within_50",
            "dtw_cost_per_step", "truth_cost_per_step",
            "endpoint_v1", "endpoint_v2", "feat_dim", "elapsed_s"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_results_md(out_path, plan, rows, params):
    lines = []
    lines.append(f"# DTW Sweep — {plan}\n")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Sample rate: {params['sample_rate']}")
    lines.append(f"- Configurations tested: {len(rows)}")
    lines.append(f"- Anchors: {params['n_anchors']}\n")

    lines.append("## Top 20 by mean absolute error\n")
    lines.append("| Rank | Extractor | Transform | sp | mean | median | max | ≤30 | ≤50 | DTW cost | truth cost | endpoint V1<->V2 |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows[:20]:
        lines.append(
            f"| {r['rank']} | {r['extractor']} | {r['transform']} | {r['step_penalty']} | "
            f"{r['mean_abs_err']:.1f} | {r['median_abs_err']:.1f} | {r['max_abs_err']} | "
            f"{r['anchors_within_30']} | {r['anchors_within_50']} | "
            f"{r['dtw_cost_per_step']:.4f} | {r['truth_cost_per_step']:.4f} | "
            f"V1[{r['endpoint_v1']}]<->V2[{r['endpoint_v2']}] |"
        )
    lines.append("")

    # Best per (extractor, transform), grouped
    lines.append("## Best step-penalty per (extractor × transform)\n")
    best = {}
    for r in rows:
        key = (r["extractor"], r["transform"])
        if key not in best or r["mean_abs_err"] < best[key]["mean_abs_err"]:
            best[key] = r
    grouped = sorted(best.values(), key=lambda x: x["mean_abs_err"])
    lines.append("| Extractor | Transform | best sp | mean | endpoint |")
    lines.append("|---|---|---:|---:|---|")
    for r in grouped:
        lines.append(
            f"| {r['extractor']} | {r['transform']} | {r['step_penalty']} | "
            f"{r['mean_abs_err']:.1f} | V1[{r['endpoint_v1']}]<->V2[{r['endpoint_v2']}] |"
        )
    lines.append("")

    lines.append("## Reading guide\n")
    lines.append("- **mean** : mean absolute error in frames at the 14 ground-truth anchors (lower = better).")
    lines.append("- **DTW cost / step** vs **truth cost / step** : if DTW < truth, the cost function rewards the wrong path → features fail.")
    lines.append("- **endpoint** : V1 should end near 2273, V2 near 2681 for Plan2.")
    lines.append("- Full artefacts for the top configs are in `top_<rank>_<config>/`. Run `diagnose.py` on those for the full visual report.\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def plot_topk_paths_vs_truth(top_rows, anchors, out_path):
    """All top-K DTW paths overlaid with the manual ground-truth line.

    Both DTW paths and ground truth are plotted in original frame coordinates
    so they can be compared directly. Use this to spot which config tracks the
    truth most closely along the entire trajectory (not just at anchors).
    """
    if not top_rows:
        return

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(12, 8))

    # diagonal reference
    max_v1 = max((np.asarray(r["_artefacts"][3]).max() for r in top_rows),
                 default=0)
    max_v2 = max((np.asarray(r["_artefacts"][4]).max() for r in top_rows),
                 default=0)
    max_v1 = max(max_v1, max(a[0] for a in anchors))
    max_v2 = max(max_v2, max(a[1] for a in anchors))
    lim = max(max_v1, max_v2)
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.4, label="diagonal (1:1)")

    # ground truth in magenta (manual measurements)
    sa = sorted(anchors)
    ax.plot([a[0] for a in sa], [a[1] for a in sa], "-", color="#ff00ff", lw=3.0,
            label="Manual ground truth", alpha=0.95, zorder=4)
    ax.scatter([a[0] for a in sa], [a[1] for a in sa], s=80, c="#ff00ff",
               edgecolors="white", linewidths=1.5, zorder=5)

    # top-K DTW paths
    for k, r in enumerate(top_rows):
        _dist, _acc, path_arr, idx1, idx2 = r["_artefacts"]
        v1_path = np.asarray(idx1)[path_arr[:, 0]]
        v2_path = np.asarray(idx2)[path_arr[:, 1]]
        color = cmap(k % 10)
        label = f"#{r['rank']} {r['extractor']}|{r['transform']}|sp{r['step_penalty']} (mean={r['mean_abs_err']:.1f})"
        ax.plot(v1_path, v2_path, "-", color=color, lw=1.6, alpha=0.85, label=label)

    ax.set_xlabel("V1 frame")
    ax.set_ylabel("V2 frame")
    ax.set_title(f"Top-{len(top_rows)} DTW paths vs manual ground truth (Plan2)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_topk_error_curves(top_rows, anchors, out_path):
    """Frame-by-frame error (DTW − truth) for each top-K config, on one chart.

    Lets you see *where along V1* each config diverges from the truth, which
    is hard to read from a single mean-abs-error number.
    """
    if not top_rows:
        return

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhspan(-30, 30, color="green", alpha=0.10, label="±30 frames")
    ax.axhline(0, color="black", lw=0.7)

    for k, r in enumerate(top_rows):
        _, _, path_arr, idx1, idx2 = r["_artefacts"]
        v1_path = np.asarray(idx1)[path_arr[:, 0]]
        v2_path = np.asarray(idx2)[path_arr[:, 1]]
        v1_eval, errs = [], []
        for v1, v2 in zip(v1_path, v2_path):
            t = interpolate_truth(int(v1), anchors)
            if t is None:
                continue
            errs.append(int(v2) - t)
            v1_eval.append(int(v1))
        if not errs:
            continue
        color = cmap(k % 10)
        label = f"#{r['rank']} {r['extractor']}|{r['transform']}|sp{r['step_penalty']}"
        ax.plot(v1_eval, errs, "-", color=color, lw=1.4, alpha=0.85, label=label)

    ax.set_xlabel("V1 frame")
    ax.set_ylabel("DTW − truth (frames)")
    ax.set_title(f"Top-{len(top_rows)} per-frame error vs manual ground truth")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_leaderboard(rows, out_path):
    n = min(20, len(rows))
    top = rows[:n]
    labels = [f"{r['rank']:2d} {r['extractor'][:18]} | {r['transform'][:8]} | sp{r['step_penalty']}"
              for r in top]
    errs = [r["mean_abs_err"] for r in top]
    colors = ["green" if r["dtw_cost_per_step"] >= r["truth_cost_per_step"] else "orange"
              for r in top]

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.3)))
    ax.barh(range(n), errs, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean abs error (frames)")
    ax.set_title(f"Leaderboard — top {n} configs (green = truth ≤ DTW cost; orange = features fail)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_config_costs(rows, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    dtw_costs = [r["dtw_cost_per_step"] for r in rows]
    truth_costs = [r["truth_cost_per_step"] for r in rows]
    errs = [r["mean_abs_err"] for r in rows]
    sc = ax.scatter(dtw_costs, truth_costs, c=errs, cmap="viridis_r", s=40,
                    edgecolors="black", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="mean abs error (frames)")
    lim = max(max(dtw_costs), max(truth_costs)) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="red", alpha=0.5, label="DTW cost = truth cost")
    ax.set_xlabel("DTW path cost / step")
    ax.set_ylabel("truth path cost / step")
    ax.set_title("Per-config: cost of DTW path vs cost of truth path\n"
                 "Points above the red line = truth is cheaper than DTW = DTW found a wrong cheap path")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DTW configuration sweep with ground-truth scoring.")
    p.add_argument("plan", help="Plan name (e.g. Plan2)")
    p.add_argument("--sample-rate", type=int, default=10,
                   help="DTW sample rate (default: 10).")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of top configs to keep full artefacts for (default: 5).")
    p.add_argument("--extractors", default=None,
                   help="Comma-separated subset of extractor names. Default: all.")
    p.add_argument("--transforms", default=None,
                   help="Comma-separated subset of transform names. Default: all.")
    p.add_argument("--step-penalties", default=None,
                   help="Comma-separated penalties (e.g. '0.0,0.3'). Default: 0.0,0.1,0.3,0.5.")
    p.add_argument("--out", default=None, help="Output dir (default: <plan>/dtw_sweep_<ts>).")
    args = p.parse_args()

    plan_dir = os.path.join(DATASET_DIR, args.plan)
    v1 = os.path.join(plan_dir, "video1.mp4")
    v2 = os.path.join(plan_dir, "video2.mp4")
    if not (os.path.exists(v1) and os.path.exists(v2)):
        print(f"ERROR: missing videos in {plan_dir}")
        sys.exit(1)

    anchors = get_anchors(args.plan)
    if not anchors:
        print(f"ERROR: no ground truth registered for {args.plan}")
        sys.exit(1)

    extractors = (args.extractors.split(",") if args.extractors
                  else list(EXTRACTORS.keys()))
    transforms = (args.transforms.split(",") if args.transforms
                  else list(TRANSFORMS.keys()))
    step_penalties = ([float(x) for x in args.step_penalties.split(",")]
                      if args.step_penalties else STEP_PENALTIES)

    for e in extractors:
        if e not in EXTRACTORS:
            print(f"ERROR: unknown extractor '{e}'. Available: {list(EXTRACTORS)}")
            sys.exit(1)
    for t in transforms:
        if t not in TRANSFORMS:
            print(f"ERROR: unknown transform '{t}'. Available: {list(TRANSFORMS)}")
            sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(plan_dir, f"dtw_sweep_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    n_total = len(extractors) * len(transforms) * len(step_penalties)
    print(f"Plan:           {args.plan}")
    print(f"Sample rate:    {args.sample_rate}")
    print(f"Extractors ({len(extractors)}): {', '.join(extractors)}")
    print(f"Transforms ({len(transforms)}): {', '.join(transforms)}")
    print(f"Step penalties ({len(step_penalties)}): {step_penalties}")
    print(f"Total configs:  {n_total}")
    print(f"Top-K full:     {args.top_k}")
    print(f"Output dir:     {out_dir}\n")

    # Cache features per extractor (computed once for V1 and V2)
    feat_cache = {}
    for e_name in extractors:
        print(f"[features] {e_name} ...", end="", flush=True)
        t0 = time.time()
        f1, idx1 = extract_features_for_video(v1, EXTRACTORS[e_name], args.sample_rate)
        f2, idx2 = extract_features_for_video(v2, EXTRACTORS[e_name], args.sample_rate)
        feat_cache[e_name] = (f1, f2, idx1, idx2)
        print(f" V1 {f1.shape}, V2 {f2.shape}  ({time.time() - t0:.1f}s)")
    print()

    # Run sweep
    rows = []
    sweep_start = time.time()
    n = 0
    for e_name in extractors:
        f1, f2, idx1, idx2 = feat_cache[e_name]
        for t_name in transforms:
            f1t, f2t = TRANSFORMS[t_name](f1, f2)
            truth_pairs = truth_pairs_in_sampled_space(anchors, idx1, idx2)
            for sp in step_penalties:
                n += 1
                cfg_name = f"{e_name}__{t_name}__sp{sp}"
                t0 = time.time()
                try:
                    path, acc, dist = compute_dtw(f1t, f2t,
                                                   metric="euclidean",
                                                   step_penalty=sp,
                                                   open_end=True)
                except Exception as ex:
                    print(f"  [{n:3d}/{n_total}] {cfg_name}  ✗ {ex}")
                    continue
                elapsed = time.time() - t0
                path_arr = np.asarray(path, dtype=np.int32)
                mean_e, med_e, max_e, _ = evaluate_path(path_arr, idx1, idx2, anchors)
                dtw_cost = cost_per_step(dist, [(int(i), int(j)) for i, j in path_arr])
                truth_cost = cost_per_step(dist, truth_pairs)
                end_i, end_j = path_arr[-1]

                # Per-anchor counters
                v1_path = np.asarray(idx1)[path_arr[:, 0]]
                v2_path = np.asarray(idx2)[path_arr[:, 1]]
                w30 = w50 = 0
                for v1t_, v2t_ in anchors:
                    k = int(np.argmin(np.abs(v1_path - v1t_)))
                    d = abs(int(v2_path[k]) - v2t_)
                    if d <= 30: w30 += 1
                    if d <= 50: w50 += 1

                rows.append({
                    "config": cfg_name,
                    "extractor": e_name,
                    "transform": t_name,
                    "step_penalty": sp,
                    "mean_abs_err": mean_e,
                    "median_abs_err": med_e,
                    "max_abs_err": max_e,
                    "anchors_within_30": w30,
                    "anchors_within_50": w50,
                    "dtw_cost_per_step": dtw_cost,
                    "truth_cost_per_step": truth_cost,
                    "endpoint_v1": int(idx1[end_i]),
                    "endpoint_v2": int(idx2[end_j]),
                    "feat_dim": int(f1t.shape[1]),
                    "elapsed_s": round(elapsed, 2),
                    "_artefacts": (dist, acc, path_arr, idx1, idx2),
                })
                marker = "✓" if dtw_cost >= truth_cost else "↓"
                print(f"  [{n:3d}/{n_total}] {cfg_name:<55} mean={mean_e:6.1f}  "
                      f"DTW/truth cost: {dtw_cost:.3f}/{truth_cost:.3f} {marker}  ({elapsed:.1f}s)")

    sweep_elapsed = time.time() - sweep_start

    # Rank
    rows.sort(key=lambda r: (r["mean_abs_err"], r["max_abs_err"]))
    for k, r in enumerate(rows, 1):
        r["rank"] = k

    # Save full artefacts for top-K
    print(f"\n[save] Top-{args.top_k} artefacts...")
    for r in rows[:args.top_k]:
        sub = os.path.join(out_dir, f"top_{r['rank']:02d}_{r['config']}")
        os.makedirs(sub, exist_ok=True)
        dist, acc, path_arr, idx1, idx2 = r["_artefacts"]
        np.save(os.path.join(sub, "dtw_dist_matrix.npy"), dist)
        np.save(os.path.join(sub, "dtw_acc_cost.npy"), acc)
        np.save(os.path.join(sub, "dtw_path.npy"), path_arr)
        np.save(os.path.join(sub, "dtw_indices1.npy"), np.asarray(idx1, dtype=np.int32))
        np.save(os.path.join(sub, "dtw_indices2.npy"), np.asarray(idx2, dtype=np.int32))
        meta = {k: v for k, v in r.items() if not k.startswith("_")}
        with open(os.path.join(sub, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        # Quick frame-space plot
        try:
            plot_path_in_frame_space(
                {"path": path_arr, "idx1": np.asarray(idx1), "idx2": np.asarray(idx2)},
                anchors,
                os.path.join(sub, "02_path_in_frame_space.png"),
            )
        except Exception as ex:
            print(f"    (plot failed for {r['config']}: {ex})")
        print(f"  rank {r['rank']:2d}: {r['config']}  → {sub}")

    # Comparison plots: top-K paths overlaid with manual ground truth
    print(f"\n[plots] Top-{args.top_k} comparison plots vs manual ground truth...")
    plot_topk_paths_vs_truth(
        rows[:args.top_k], anchors,
        os.path.join(out_dir, "topk_paths_vs_truth.png"),
    )
    plot_topk_error_curves(
        rows[:args.top_k], anchors,
        os.path.join(out_dir, "topk_error_curves.png"),
    )

    # Drop in-memory artefacts before writing the table
    for r in rows:
        r.pop("_artefacts", None)

    # Reports
    write_results_csv(os.path.join(out_dir, "results.csv"), rows)
    write_results_md(os.path.join(out_dir, "results.md"),
                     args.plan, rows,
                     {"sample_rate": args.sample_rate, "n_anchors": len(anchors)})
    plot_leaderboard(rows, os.path.join(out_dir, "leaderboard.png"))
    plot_config_costs(rows, os.path.join(out_dir, "config_costs.png"))

    print(f"\n{'=' * 70}")
    print(f"DONE — {len(rows)} configs in {sweep_elapsed:.1f}s ({sweep_elapsed/60:.1f}m)")
    print(f"{'=' * 70}")
    print(f"Best: {rows[0]['config']}  → mean abs err = {rows[0]['mean_abs_err']:.1f} frames")
    print(f"Worst: {rows[-1]['config']}  → {rows[-1]['mean_abs_err']:.1f}")
    print(f"\nResults  : {os.path.join(out_dir, 'results.md')}")
    print(f"CSV      : {os.path.join(out_dir, 'results.csv')}")
    print(f"Plots    : {os.path.join(out_dir, 'leaderboard.png')}")
    print(f"           {os.path.join(out_dir, 'config_costs.png')}")
    print(f"Top-K    : {os.path.join(out_dir, 'top_01_*')}")


if __name__ == "__main__":
    main()
