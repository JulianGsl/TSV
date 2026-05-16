#!/usr/bin/env python3
"""
DTW Diagnostic Tool
===================

Analyses the DTW output of a hybrid run against manually-collected ground-truth
anchors. Generates a self-contained diagnostic folder with plots and a summary.

Usage
-----
    python combined_method/dtw_diagnostic/diagnose.py dataset/Plan2/hybrid_akaze
    python combined_method/dtw_diagnostic/diagnose.py dataset/Plan2/hybrid_akaze --plan Plan2

Inputs (must exist in the hybrid output directory)
    - dtw_acc_cost.npy        accumulated cost (DP) matrix
    - dtw_dist_matrix.npy     raw pairwise distances
    - dtw_path.npy            (i, j) waypoints in *sampled* index space
    - dtw_indices1.npy        sampled-idx → V1 original frame
    - dtw_indices2.npy        sampled-idx → V2 original frame

The plan name is inferred from the parent directory unless --plan is given.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # TSV/
sys.path.insert(0, project_root)

from combined_method.dtw_diagnostic.ground_truth import (
    get_anchors, interpolate_truth,
)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

REQUIRED_FILES = [
    "dtw_acc_cost.npy", "dtw_dist_matrix.npy",
    "dtw_path.npy", "dtw_indices1.npy", "dtw_indices2.npy",
]


def load_artefacts(hybrid_dir):
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(hybrid_dir, f))]
    if missing:
        print(f"ERROR: Missing artefacts in {hybrid_dir}:")
        for f in missing:
            print(f"  - {f}")
        print("Re-run the hybrid pipeline first to generate them.")
        sys.exit(1)

    return {
        "acc":      np.load(os.path.join(hybrid_dir, "dtw_acc_cost.npy")),
        "dist":     np.load(os.path.join(hybrid_dir, "dtw_dist_matrix.npy")),
        "path":     np.load(os.path.join(hybrid_dir, "dtw_path.npy")),
        "idx1":     np.load(os.path.join(hybrid_dir, "dtw_indices1.npy")),
        "idx2":     np.load(os.path.join(hybrid_dir, "dtw_indices2.npy")),
    }


def map_original_to_sampled(idx_array, target_frame):
    """Return the sampled-space index whose original frame is closest to target_frame."""
    return int(np.argmin(np.abs(idx_array - target_frame)))


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_path_on_matrices(art, anchors_sampled, out_path):
    """Side-by-side: dist_matrix (left) and acc_cost (right) with DTW path + anchors."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, mat, label in [
        (axes[0], art["dist"], "Raw distance matrix"),
        (axes[1], art["acc"],  "Accumulated cost (DP)"),
    ]:
        # imshow: rows = V1 sampled, cols = V2 sampled
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                       norm=LogNorm(vmin=max(mat.min(), 1e-3), vmax=mat.max()))
        plt.colorbar(im, ax=ax, label="cost (log)")

        # naive diagonal
        n, m = mat.shape
        ax.plot([0, m - 1], [0, n - 1], "--", color="gray", alpha=0.5, label="diagonal")

        # DTW path
        path = art["path"]
        ax.plot(path[:, 1], path[:, 0], "-", color="orange", lw=2, label="DTW path")

        # ground-truth anchors (in sampled space)
        if anchors_sampled:
            i_arr = [a[0] for a in anchors_sampled]
            j_arr = [a[1] for a in anchors_sampled]
            ax.scatter(j_arr, i_arr, s=80, c="red", edgecolors="white",
                       linewidths=1.5, zorder=5, label="ground truth")

        ax.set_title(label)
        ax.set_xlabel("V2 (sampled idx)")
        ax.set_ylabel("V1 (sampled idx)")
        ax.legend(loc="upper left")

    fig.suptitle("DTW path vs ground truth on cost matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_path_in_frame_space(art, anchors, out_path):
    """Frame-space view: DTW path (orange) vs piecewise-linear ground truth (green)."""
    fig, ax = plt.subplots(figsize=(11, 7))

    # diagonal
    max_v1 = max(art["idx1"].max(), max(a[0] for a in anchors))
    max_v2 = max(art["idx2"].max(), max(a[1] for a in anchors))
    lim = max(max_v1, max_v2)
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, label="diagonal (1:1)")

    # DTW path in original frame coords
    path = art["path"]
    v1_path = art["idx1"][path[:, 0]]
    v2_path = art["idx2"][path[:, 1]]
    ax.plot(v1_path, v2_path, "-", color="orange", lw=2.2, label="DTW path")

    # ground truth (piecewise-linear between anchors)
    sa = sorted(anchors)
    ax.plot([a[0] for a in sa], [a[1] for a in sa], "-", color="green",
            lw=2, alpha=0.7, label="ground truth (lerp)")
    ax.scatter([a[0] for a in sa], [a[1] for a in sa], s=70, c="red",
               edgecolors="white", linewidths=1.4, zorder=5, label="anchors")

    ax.set_xlabel("V1 frame (original)")
    ax.set_ylabel("V2 frame (original)")
    ax.set_title("DTW path vs ground truth in original frame coordinates")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_path_error(art, anchors, out_path):
    """Error of DTW prediction vs interpolated ground truth, along V1 frames."""
    path = art["path"]
    v1_path = art["idx1"][path[:, 0]]
    v2_path = art["idx2"][path[:, 1]]

    # For each V1 frame on the DTW path, get truth (interp); compute err
    errs, v1_eval = [], []
    for v1, v2 in zip(v1_path, v2_path):
        truth = interpolate_truth(int(v1), anchors)
        if truth is None:
            continue
        errs.append(int(v2) - truth)
        v1_eval.append(int(v1))

    errs = np.asarray(errs)
    v1_eval = np.asarray(v1_eval)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhspan(-50, 50, color="green", alpha=0.10, label="±50 frames (acceptable)")
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(v1_eval, errs, "-", color="orange", lw=1.6, label="DTW − truth")

    # Mark anchors (where err == 0 by construction since we interpolate through them)
    for v1, v2 in anchors:
        truth = interpolate_truth(v1, anchors)
        if truth is None:
            continue
        # find nearest dtw point at this v1
        if len(v1_path) == 0:
            continue
        k = int(np.argmin(np.abs(v1_path - v1)))
        ax.scatter([v1], [int(v2_path[k]) - truth], s=70, c="red",
                   edgecolors="white", linewidths=1.4, zorder=5)

    ax.set_xlabel("V1 frame")
    ax.set_ylabel("DTW prediction − truth (frames)")
    ax.set_title("DTW prediction error along V1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_anchor_ranks(art, anchors_sampled, out_path):
    """For each anchor (i, j_truth), where does j_truth rank in dist[i, :]?
    0% = lowest distance (best), 100% = highest distance (worst)."""
    dist = art["dist"]
    n, m = dist.shape

    pcts, raw_dists, min_dists, anchor_labels = [], [], [], []
    for k, (i, j) in enumerate(anchors_sampled):
        if not (0 <= i < n and 0 <= j < m):
            continue
        row = dist[i]
        rank = (row < row[j]).sum()      # number of cells strictly cheaper than truth
        pct = 100.0 * rank / (m - 1)
        pcts.append(pct)
        raw_dists.append(row[j])
        min_dists.append(row.min())
        anchor_labels.append(f"#{k}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].bar(anchor_labels, pcts, color="steelblue")
    axes[0].axhline(50, color="red", ls="--", alpha=0.6, label="random (50%)")
    axes[0].axhline(5,  color="green", ls="--", alpha=0.6, label="discriminative (<5%)")
    axes[0].set_ylabel("Percentile of truth in row")
    axes[0].set_title("Where ground-truth V2 ranks in its distance row\n"
                      "(0% = truth is the cheapest cell — what DTW would pick)")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    x = np.arange(len(anchor_labels))
    width = 0.4
    axes[1].bar(x - width/2, raw_dists, width, color="orange", label="distance at truth")
    axes[1].bar(x + width/2, min_dists, width, color="darkgreen", label="row minimum")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(anchor_labels)
    axes[1].set_ylabel("Distance value")
    axes[1].set_title("Truth distance vs the cheapest distance in the row")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return pcts, raw_dists, min_dists


def plot_cost_along_paths(art, anchors, anchors_sampled, out_path):
    """Cumulative raw distance along DTW path vs ground-truth path vs diagonal."""
    dist = art["dist"]
    n, m = dist.shape

    def cumcost(ij_pairs):
        cs = []
        s = 0.0
        for i, j in ij_pairs:
            if 0 <= i < n and 0 <= j < m:
                s += dist[i, j]
                cs.append(s)
        return np.asarray(cs)

    # DTW path
    dtw_pairs = [(int(i), int(j)) for i, j in art["path"]]
    dtw_cum = cumcost(dtw_pairs)

    # Ground-truth path (piecewise linear in sampled space, between anchors)
    sa = sorted(anchors_sampled)
    truth_pairs = []
    for k in range(len(sa) - 1):
        i0, j0 = sa[k]
        i1, j1 = sa[k + 1]
        steps = max(abs(i1 - i0), abs(j1 - j0))
        if steps == 0:
            truth_pairs.append((i0, j0))
            continue
        for s in range(steps + 1):
            t = s / steps
            truth_pairs.append((int(round(i0 + t * (i1 - i0))),
                                int(round(j0 + t * (j1 - j0)))))
    truth_cum = cumcost(truth_pairs)

    # Naive diagonal up to the shorter of n, m
    L = min(n, m)
    diag_pairs = [(i, i) for i in range(L)]
    diag_cum = cumcost(diag_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # raw cumulative
    axes[0].plot(dtw_cum,   "-", color="orange",     lw=1.8, label=f"DTW path  (n={len(dtw_cum)})")
    axes[0].plot(truth_cum, "-", color="green",      lw=1.8, label=f"truth path (n={len(truth_cum)})")
    axes[0].plot(diag_cum,  "-", color="gray",       lw=1.3, label=f"diagonal  (n={len(diag_cum)})", alpha=0.7)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cumulative raw distance")
    axes[0].set_title("Cumulative cost along candidate paths")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # per-step (mean cost)
    means = {
        "DTW":   dtw_cum[-1] / len(dtw_cum)   if len(dtw_cum)   else 0.0,
        "truth": truth_cum[-1] / len(truth_cum) if len(truth_cum) else 0.0,
        "diagonal": diag_cum[-1] / len(diag_cum) if len(diag_cum) else 0.0,
    }
    axes[1].bar(list(means.keys()), list(means.values()),
                color=["orange", "green", "gray"])
    axes[1].set_ylabel("mean cost / step")
    axes[1].set_title("Mean per-step cost (lower = path is cheaper)")
    axes[1].grid(True, alpha=0.3, axis="y")
    for k, v in means.items():
        axes[1].text(k, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Path-cost analysis — does DTW pick a cheaper path than the truth?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return means


# ----------------------------------------------------------------------
# Per-anchor table & summary
# ----------------------------------------------------------------------

def per_anchor_table(art, anchors):
    """For each ground-truth anchor, find the DTW prediction at that V1 frame."""
    path = art["path"]
    v1_path = art["idx1"][path[:, 0]]
    v2_path = art["idx2"][path[:, 1]]
    rows = []
    for v1_truth, v2_truth in anchors:
        if len(v1_path) == 0:
            rows.append((v1_truth, v2_truth, None, None))
            continue
        # nearest waypoint (DTW path may not contain exact V1 frame after resampling)
        k = int(np.argmin(np.abs(v1_path - v1_truth)))
        v2_pred = int(v2_path[k])
        rows.append((v1_truth, v2_truth, v2_pred, v2_pred - v2_truth))
    return rows


def write_summary(out_path, plan, hybrid_dir, anchors, table, pcts, means):
    errs = [r[3] for r in table if r[3] is not None]
    abs_errs = [abs(e) for e in errs]
    lines = []
    lines.append(f"# DTW Diagnostic — {plan}\n")
    lines.append(f"Source directory: `{hybrid_dir}`\n")

    lines.append("## Per-anchor predictions\n")
    lines.append("| V1 (truth) | V2 (truth) | V2 (DTW) | Δ |")
    lines.append("|---:|---:|---:|---:|")
    for v1t, v2t, v2p, d in table:
        if v2p is None:
            lines.append(f"| {v1t} | {v2t} | — | — |")
        else:
            sign = "+" if d >= 0 else ""
            lines.append(f"| {v1t} | {v2t} | {v2p} | {sign}{d} |")
    lines.append("")

    if abs_errs:
        lines.append("## Error stats\n")
        lines.append(f"- Mean absolute error : **{np.mean(abs_errs):.1f} frames**")
        lines.append(f"- Median absolute err : **{np.median(abs_errs):.1f} frames**")
        lines.append(f"- Max absolute error  : **{np.max(abs_errs)} frames**")
        lines.append(f"- Anchors within ±30  : {sum(1 for e in abs_errs if e <= 30)}/{len(abs_errs)}")
        lines.append(f"- Anchors within ±50  : {sum(1 for e in abs_errs if e <= 50)}/{len(abs_errs)}")
        lines.append("")

    if pcts:
        lines.append("## Distance-row percentile of truth\n")
        lines.append("How discriminative the features are: 0% = truth is the cheapest cell"
                     " in its row (DTW would pick it). 50% = random.\n")
        lines.append(f"- Mean percentile : **{np.mean(pcts):.1f}%**")
        lines.append(f"- Median percentile : {np.median(pcts):.1f}%")
        lines.append(f"- Anchors below 5%  : {sum(1 for p in pcts if p < 5)}/{len(pcts)}  (clearly discriminative)")
        lines.append(f"- Anchors above 30% : {sum(1 for p in pcts if p > 30)}/{len(pcts)}  (noise-level)")
        lines.append("")

    lines.append("## Path-cost comparison\n")
    lines.append(f"Mean raw distance per step along each candidate path "
                 "(lower = DP would prefer it):\n")
    for k, v in means.items():
        lines.append(f"- {k:8s} : {v:.4f}")
    lines.append("")

    if means.get("DTW", 0) < means.get("truth", float("inf")):
        lines.append("**Diagnosis:** DTW finds a *cheaper* path than the truth — the cost "
                     "function (current features) does not reward the geographic answer. "
                     "Improving features, not DTW, is the path forward.\n")
    else:
        lines.append("**Diagnosis:** Truth is at least as cheap as the DTW path; if DTW "
                     "still misses, the issue is monotonicity / step-penalty / open-end "
                     "bias rather than features.\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DTW diagnostic: compare DTW path against manual ground truth.")
    p.add_argument("hybrid_dir", help="Path to a hybrid output dir (e.g. dataset/Plan2/hybrid_akaze)")
    p.add_argument("--plan", default=None, help="Plan name (default: parent dir name).")
    p.add_argument("--out", default=None, help="Output dir (default: <hybrid_dir>/dtw_diagnostic).")
    args = p.parse_args()

    hybrid_dir = os.path.abspath(args.hybrid_dir)
    plan = args.plan or os.path.basename(os.path.dirname(hybrid_dir))
    out_dir = args.out or os.path.join(hybrid_dir, "dtw_diagnostic")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Plan:           {plan}")
    print(f"Hybrid dir:     {hybrid_dir}")
    print(f"Output dir:     {out_dir}\n")

    anchors = get_anchors(plan)
    if not anchors:
        print(f"ERROR: No ground-truth anchors registered for plan '{plan}'.")
        print(f"Add them in {os.path.relpath(__file__)} → GROUND_TRUTH dict.")
        sys.exit(1)
    print(f"Loaded {len(anchors)} ground-truth anchors.")

    art = load_artefacts(hybrid_dir)
    print(f"Loaded artefacts: dist {art['dist'].shape}, "
          f"acc {art['acc'].shape}, path {art['path'].shape}")

    # Map anchors from frame space to sampled-index space
    anchors_sampled = []
    for v1, v2 in anchors:
        i = map_original_to_sampled(art["idx1"], v1)
        j = map_original_to_sampled(art["idx2"], v2)
        anchors_sampled.append((i, j))

    print("\nGenerating plots...")
    plot_path_on_matrices(art, anchors_sampled,
                          os.path.join(out_dir, "01_path_on_cost_matrices.png"))
    print("  [1/5] 01_path_on_cost_matrices.png")

    plot_path_in_frame_space(art, anchors,
                             os.path.join(out_dir, "02_path_in_frame_space.png"))
    print("  [2/5] 02_path_in_frame_space.png")

    plot_path_error(art, anchors,
                    os.path.join(out_dir, "03_path_error.png"))
    print("  [3/5] 03_path_error.png")

    pcts, raw_d, min_d = plot_anchor_ranks(art, anchors_sampled,
                                           os.path.join(out_dir, "04_anchor_ranks.png"))
    print("  [4/5] 04_anchor_ranks.png")

    means = plot_cost_along_paths(art, anchors, anchors_sampled,
                                  os.path.join(out_dir, "05_cost_along_paths.png"))
    print("  [5/5] 05_cost_along_paths.png")

    print("\nGenerating summary...")
    table = per_anchor_table(art, anchors)
    summary_path = os.path.join(out_dir, "summary.md")
    write_summary(summary_path, plan, hybrid_dir, anchors, table, pcts, means)
    print(f"  summary.md written ({summary_path})")

    abs_errs = [abs(r[3]) for r in table if r[3] is not None]
    if abs_errs:
        print(f"\n>>> Mean abs error: {np.mean(abs_errs):.1f} frames | "
              f"median: {np.median(abs_errs):.0f} | max: {np.max(abs_errs)}")
    print(f">>> Truth path mean cost/step: {means['truth']:.4f}")
    print(f">>> DTW   path mean cost/step: {means['DTW']:.4f}  "
          f"({'cheaper than truth — features fail' if means['DTW'] < means['truth'] else 'truth is cheaper'})")

    print(f"\n✓ Open: file://{os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
