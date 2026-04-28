#!/usr/bin/env python3
"""
Full Evaluation Runner — comprehensive baseline comparison.

Asks for ONE plan name, then automatically runs:
  - All 3 algorithms      : AKAZE, BRISK, ORB
  - All 3 metrics         : LPIPS, SSIM, cycle consistency
  - All input variants    : real algorithm + 4 baselines (random, linear, offset±k)

For each (algorithm × variant) combination, computes the 3 metrics.
Saves everything (JSON + CSV + plots) under:

    dataset/<Plan>/full_evaluation_<timestamp>/
        ├── full_evaluation.json    # complete numerical data
        ├── results_table.csv       # flat table for plotting / analysis
        ├── summary.md              # human-readable summary
        └── plots/
            ├── perceptual_comparison.png
            ├── cycle_comparison.png
            ├── sensitivity_curve.png
            └── algorithm_comparison.png

Usage:
    cd Projet/TSV
    python combined_method/run_full_evaluation.py            # interactive
    python combined_method/run_full_evaluation.py Plan1      # CLI
    python combined_method/run_full_evaluation.py Plan1 --offsets 1,5,10,30
    python combined_method/run_full_evaluation.py Plan1 --skip-lpips
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root (TSV/) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from combined_method.hybrid_alignment import align_videos_hybrid, SUPPORTED_ALGORITHMS
from combined_method.evaluation import (
    SSIMMetric,
    compute_cycle_consistency,
    sample_frame_pairs,
    random_alignment,
    linear_alignment,
    offset_alignment,
)
from combined_method._cli_helpers import (
    get_available_plans as _list_plans,
    select_plan as _pick_plan,
    load_matches_from_csv,
    save_matches_to_csv,
)

DATASET_DIR = os.path.join(project_root, "dataset")
FORWARD_CSV = "alignment_results.csv"
BACKWARD_CSV = "alignment_results_backward.csv"


# =============================================================================
# Discovery / interactive (thin wrappers over the shared helpers)
# =============================================================================

def get_available_plans():
    return _list_plans(DATASET_DIR)


def select_plan(plans):
    return _pick_plan(plans, allow_all=False)[0]


# =============================================================================
# Optimised metric computation with V1 frame caching
# =============================================================================

def evaluate_variants_for_algo(
    v1_path: str,
    v2_path: str,
    variants: dict,            # {variant_name: {"forward": [...], "backward": [...]}}
    metrics: list,             # list of BaseMetric instances
    sample_rate: float,
    cycle_sample_every: int,
    algorithm: str,
    progress_prefix: str = "",
) -> dict:
    """
    For one algorithm, evaluate every variant on every metric.

    Optimisation: V1 frames are read ONCE and reused across all variants
    (since they share the same V1 sample indices).
    """
    # Use the REAL forward to derive the sampling indices (other variants reuse them)
    real_forward = variants["real"]["forward"]
    pairs_real = sample_frame_pairs(real_forward, sample_rate)
    sampled_v1_indices = [v1 for v1, _ in pairs_real]
    n_pairs = len(sampled_v1_indices)

    # ---- Pre-load V1 frames once -----------------------------------------
    print(f"{progress_prefix}Pre-loading {n_pairs} V1 frames…", end="", flush=True)
    cap1 = cv2.VideoCapture(v1_path)
    v1_frames = {}
    for v1 in sampled_v1_indices:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, v1)
        ok, frame = cap1.read()
        if ok:
            v1_frames[v1] = frame
    cap1.release()
    print(f" ✓ ({len(v1_frames)} cached)")

    # ---- Build per-variant pair lists (reuse V1 indices) ------------------
    # For each variant, we need v2 indices. Build a lookup from V1→V2 from the variant's forward.
    variant_pairs = {}
    for vname, vmatches in variants.items():
        fwd_map = {int(m["v1_frame"]): int(m["v2_frame"]) for m in vmatches["forward"]}
        pairs = []
        for v1 in sampled_v1_indices:
            if v1 in fwd_map:
                pairs.append((v1, fwd_map[v1]))
        variant_pairs[vname] = pairs

    # ---- Run metrics for each variant -------------------------------------
    results = {}
    cap2 = cv2.VideoCapture(v2_path)
    for v_idx, (vname, pairs) in enumerate(variant_pairs.items()):
        scores = {m.name: [] for m in metrics}
        bar_width = 25
        t_var = time.time()

        for i, (v1, v2) in enumerate(pairs):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, v2)
            ok, frame2 = cap2.read()
            if not ok or v1 not in v1_frames:
                continue
            frame1 = v1_frames[v1]
            for m in metrics:
                scores[m.name].append(m.compute(frame1, frame2))

            done = i + 1
            filled = int(bar_width * done / len(pairs))
            bar = "█" * filled + "░" * (bar_width - filled)
            elapsed = time.time() - t_var
            eta = (elapsed / done) * (len(pairs) - done) if done > 0 else 0
            live = "  ".join(
                f"{m.name}={np.mean(scores[m.name]):.3f}" if scores[m.name] else f"{m.name}=--"
                for m in metrics
            )
            print(f"\r{progress_prefix}[{vname:>10}] [{bar}] {done}/{len(pairs)}  "
                  f"{live}  ETA {eta:.0f}s   ", end="", flush=True)
        print()

        per_metric = {}
        for m in metrics:
            arr = np.array(scores[m.name], dtype=float)
            per_metric[m.name] = {
                "mean": float(np.mean(arr)) if len(arr) else float("nan"),
                "std":  float(np.std(arr))  if len(arr) else float("nan"),
                "min":  float(np.min(arr))  if len(arr) else float("nan"),
                "max":  float(np.max(arr))  if len(arr) else float("nan"),
                "n":    int(len(arr)),
                "higher_is_better": m.higher_is_better(),
            }

        # Cycle consistency for this variant (uses both fwd & bwd matches, no frame I/O)
        cycle = compute_cycle_consistency(
            v1_path, v2_path,
            sample_every=cycle_sample_every,
            algorithm=algorithm,
            forward_matches=variants[vname]["forward"],
            backward_matches=variants[vname]["backward"],
            verbose=False,
        )
        per_metric["CYCLE"] = {
            "mean":   cycle["mean_error"],
            "std":    cycle["std_error"],
            "median": cycle["median_error"],
            "max":    cycle["max_error"],
            "perfect_percent":  cycle["perfect_percent"],
            "within_1_percent": cycle["within_1_percent"],
            "within_5_percent": cycle["within_5_percent"],
            "n":      cycle["n_sampled"],
            "higher_is_better": False,
        }

        results[vname] = per_metric

    cap2.release()
    return results


# =============================================================================
# Variant generation
# =============================================================================

def build_variants(real_forward, real_backward, n_v1, n_v2, offsets, seed=42):
    """
    Return dict {variant_name: {"forward": [...], "backward": [...]}} for:
      - real
      - random
      - linear
      - offset_+k (for each k in offsets)
    """
    variants = {
        "real":   {"forward": real_forward, "backward": real_backward},
        "random": {
            "forward":  random_alignment(n_v1, n_v2, seed=seed),
            "backward": random_alignment(n_v2, n_v1, seed=seed + 1),
        },
        "linear": {
            "forward":  linear_alignment(n_v1, n_v2),
            "backward": linear_alignment(n_v2, n_v1),
        },
    }
    for k in offsets:
        variants[f"offset_+{k}"] = {
            "forward":  offset_alignment(real_forward, +k, n_v2),
            "backward": real_backward,  # only forward perturbed → cycle ≈ k frames
        }
    return variants


# =============================================================================
# Output: tables, plots
# =============================================================================

def flatten_to_table(results: dict) -> list:
    """Flatten {algo: {variant: {metric: {stats}}}} into a list of rows for CSV."""
    rows = []
    for algo, variants in results.items():
        for vname, metrics in variants.items():
            row = {"algorithm": algo, "variant": vname}
            for mname, stats in metrics.items():
                for k in ("mean", "std", "min", "max", "median"):
                    if k in stats:
                        row[f"{mname}_{k}"] = stats[k]
            rows.append(row)
    return rows


def save_csv_table(rows: list, csv_path: str):
    if not rows:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    cols = ["algorithm", "variant"] + [c for c in cols if c not in ("algorithm", "variant")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_summary_md(plan, results, config, md_path):
    lines = [f"# Full Evaluation — {plan}", ""]
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Sample rate (LPIPS/SSIM): {config['sample_rate']*100:.0f}%")
    lines.append(f"- Cycle sample-every: {config['cycle_sample_every']}")
    lines.append(f"- Offsets tested: {config['offsets']}")
    lines.append(f"- Random seed: {config['random_seed']}")
    lines.append("")

    metric_names = set()
    for algo in results.values():
        for variant in algo.values():
            metric_names.update(variant.keys())
    metric_names = [m for m in ("LPIPS", "SSIM", "CYCLE") if m in metric_names]

    for algo, variants in results.items():
        lines.append(f"## {algo}")
        header = "| Variant | " + " | ".join(metric_names) + " |"
        sep = "|---" * (len(metric_names) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for vname, metrics in variants.items():
            cells = []
            for mname in metric_names:
                if mname in metrics:
                    s = metrics[mname]
                    if mname == "CYCLE":
                        cells.append(f"{s['mean']:.2f} (med={s.get('median', 0):.0f})")
                    else:
                        cells.append(f"{s['mean']:.4f} ± {s['std']:.4f}")
                else:
                    cells.append("--")
            lines.append(f"| {vname} | " + " | ".join(cells) + " |")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def plot_perceptual_comparison(results, config, out_path):
    """Bar chart per algorithm: LPIPS and SSIM across all variants."""
    algos = list(results.keys())
    metrics_to_plot = ["LPIPS", "SSIM"]
    fig, axes = plt.subplots(len(algos), len(metrics_to_plot),
                             figsize=(13, 3.5 * len(algos)), squeeze=False)
    for ai, algo in enumerate(algos):
        variants = list(results[algo].keys())
        for mi, mname in enumerate(metrics_to_plot):
            ax = axes[ai][mi]
            means = [results[algo][v].get(mname, {}).get("mean", 0) for v in variants]
            stds = [results[algo][v].get(mname, {}).get("std", 0) for v in variants]
            colors = ["#2E86AB" if v == "real" else "#A23B72" for v in variants]
            ax.bar(variants, means, yerr=stds, capsize=3, color=colors, alpha=0.85)
            arrow = "↑" if mname == "SSIM" else "↓"
            ax.set_title(f"{algo} — {mname} {arrow}")
            ax.set_ylabel(mname)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_cycle_comparison(results, config, out_path):
    algos = list(results.keys())
    fig, axes = plt.subplots(1, len(algos), figsize=(5 * len(algos), 4), squeeze=False)
    for ai, algo in enumerate(algos):
        ax = axes[0][ai]
        variants = list(results[algo].keys())
        means = [results[algo][v].get("CYCLE", {}).get("mean", 0) for v in variants]
        colors = ["#2E86AB" if v == "real" else "#A23B72" for v in variants]
        ax.bar(variants, means, color=colors, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"{algo} — Cycle error ↓ (log)")
        ax.set_ylabel("Mean |i - g(f(i))| (frames)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sensitivity_curve(results, config, out_path):
    """Show how each metric reacts to known offsets (real + offset_+k)."""
    offsets = [0] + list(config["offsets"])  # real = 0
    metrics_to_plot = ["LPIPS", "SSIM", "CYCLE"]
    algos = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mi, mname in enumerate(metrics_to_plot):
        ax = axes[mi]
        for algo in algos:
            ys = []
            for k in offsets:
                vname = "real" if k == 0 else f"offset_+{k}"
                ys.append(results[algo].get(vname, {}).get(mname, {}).get("mean", np.nan))
            ax.plot(offsets, ys, marker="o", label=algo)
        arrow = "↑" if mname == "SSIM" else "↓"
        ax.set_title(f"{mname} {arrow} vs offset")
        ax.set_xlabel("Offset (frames added to real V2 index)")
        ax.set_ylabel(mname)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_algorithm_comparison(results, out_path):
    """Compare AKAZE / BRISK / ORB on the REAL pipeline only."""
    algos = list(results.keys())
    metrics_to_plot = ["LPIPS", "SSIM", "CYCLE"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mi, mname in enumerate(metrics_to_plot):
        ax = axes[mi]
        means = [results[a]["real"].get(mname, {}).get("mean", 0) for a in algos]
        stds  = [results[a]["real"].get(mname, {}).get("std", 0)  for a in algos]
        ax.bar(algos, means, yerr=stds, capsize=4, color=["#2E86AB", "#A23B72", "#F18F01"])
        arrow = "↑" if mname == "SSIM" else "↓"
        ax.set_title(f"REAL — {mname} {arrow}")
        ax.set_ylabel(mname)
        if mname == "CYCLE":
            ax.set_ylabel("Cycle error (frames)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def get_video_frame_count(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def run_full_evaluation(plan, offsets, sample_rate, cycle_sample_every,
                        skip_lpips, random_seed):
    plan_dir = os.path.join(DATASET_DIR, plan)
    v1_path = os.path.join(plan_dir, "video1.mp4")
    v2_path = os.path.join(plan_dir, "video2.mp4")

    for path in (v1_path, v2_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing video: {path}")

    n_v1 = get_video_frame_count(v1_path)
    n_v2 = get_video_frame_count(v2_path)
    print(f"\n📹 V1: {n_v1} frames   V2: {n_v2} frames")

    # Build per-frame metric instances ONCE (LPIPS loads a heavy model)
    metrics = [SSIMMetric()]
    if not skip_lpips:
        try:
            from combined_method.evaluation import LPIPSMetric
            print("  Loading LPIPS model…")
            metrics.insert(0, LPIPSMetric())
        except ImportError as e:
            print(f"  ⚠️  LPIPS unavailable ({e}). Continuing with SSIM only.")

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(plan_dir, f"full_evaluation_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\n💾 Output directory: {out_dir}")

    all_results = {}
    t_total = time.time()

    for ai, algo in enumerate(SUPPORTED_ALGORITHMS, 1):
        print(f"\n{'═' * 70}")
        print(f"  ALGORITHM {ai}/{len(SUPPORTED_ALGORITHMS)}: {algo}")
        print(f"{'═' * 70}")

        algo_dir = os.path.join(plan_dir, f"hybrid_{algo.lower()}")
        forward_csv = os.path.join(algo_dir, FORWARD_CSV)
        backward_csv = os.path.join(algo_dir, BACKWARD_CSV)

        if not os.path.exists(forward_csv):
            print(f"⚠️  Missing {forward_csv}. Skipping {algo}.")
            print(f"   Run first:  python combined_method/run_alignment_hybrid.py {plan}")
            continue

        real_forward = load_matches_from_csv(forward_csv)
        print(f"  ✓ Loaded forward alignment ({len(real_forward)} matches)")

        if os.path.exists(backward_csv):
            real_backward = load_matches_from_csv(backward_csv)
            print(f"  ✓ Loaded backward alignment ({len(real_backward)} matches)")
        else:
            print(f"  ⚙  Computing backward alignment (one-time, cached)…")
            t0 = time.time()
            real_backward = align_videos_hybrid(v2_path, v1_path,
                                                algorithm=algo, verbose=False)
            print(f"  ✓ Backward computed in {time.time() - t0:.1f}s")
            save_matches_to_csv(real_backward, backward_csv)
            print(f"  💾 Cached → {backward_csv}")

        variants = build_variants(real_forward, real_backward, n_v1, n_v2,
                                  offsets, seed=random_seed)
        print(f"  Variants: {list(variants.keys())}\n")

        algo_results = evaluate_variants_for_algo(
            v1_path, v2_path, variants, metrics,
            sample_rate=sample_rate, cycle_sample_every=cycle_sample_every,
            algorithm=algo, progress_prefix="  ",
        )
        all_results[algo] = algo_results

        # ----- Incremental save after each algorithm ----------------------
        # so that interrupting mid-way still preserves partial results
        partial_json = os.path.join(out_dir, "full_evaluation.json")
        with open(partial_json, "w") as f:
            json.dump({
                "plan": plan,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "video_v1": os.path.abspath(v1_path),
                "video_v2": os.path.abspath(v2_path),
                "n_v1_frames": n_v1,
                "n_v2_frames": n_v2,
                "config": {
                    "sample_rate": sample_rate,
                    "cycle_sample_every": cycle_sample_every,
                    "offsets": offsets,
                    "random_seed": random_seed,
                    "skip_lpips": skip_lpips,
                },
                "results": all_results,
                "_partial": algo != SUPPORTED_ALGORITHMS[-1],
            }, f, indent=2)
        print(f"  💾 Incremental save → {partial_json}")

    # ----- Persist everything ----------------------------------------------
    config = {
        "sample_rate": sample_rate,
        "cycle_sample_every": cycle_sample_every,
        "offsets": offsets,
        "random_seed": random_seed,
        "skip_lpips": skip_lpips,
    }

    json_path = os.path.join(out_dir, "full_evaluation.json")
    with open(json_path, "w") as f:
        json.dump({
            "plan": plan,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "video_v1": os.path.abspath(v1_path),
            "video_v2": os.path.abspath(v2_path),
            "n_v1_frames": n_v1,
            "n_v2_frames": n_v2,
            "config": config,
            "results": all_results,
        }, f, indent=2)

    csv_path = os.path.join(out_dir, "results_table.csv")
    save_csv_table(flatten_to_table(all_results), csv_path)

    md_path = os.path.join(out_dir, "summary.md")
    save_summary_md(plan, all_results, config, md_path)

    # ----- Plots ------------------------------------------------------------
    if all_results:
        try:
            plot_perceptual_comparison(all_results, config,
                                       os.path.join(plots_dir, "perceptual_comparison.png"))
            plot_cycle_comparison(all_results, config,
                                  os.path.join(plots_dir, "cycle_comparison.png"))
            plot_sensitivity_curve(all_results, config,
                                   os.path.join(plots_dir, "sensitivity_curve.png"))
            plot_algorithm_comparison(all_results,
                                      os.path.join(plots_dir, "algorithm_comparison.png"))
        except Exception as e:
            print(f"⚠️  Plotting error: {e}")

    print(f"\n{'═' * 70}")
    print(f"✅ Done in {(time.time() - t_total) / 60:.1f} min")
    print(f"   Results : {json_path}")
    print(f"   Table   : {csv_path}")
    print(f"   Summary : {md_path}")
    print(f"   Plots   : {plots_dir}/")
    print(f"{'═' * 70}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Full evaluation — all algorithms × all metrics × all baselines.")
    p.add_argument("plan", nargs="?", help="Plan name (omit for interactive).")
    p.add_argument("--offsets", default="1,5,30",
                   help="Comma-separated frame offsets for sensitivity test (default: 1,5,30).")
    p.add_argument("--sample-rate", type=float, default=0.10,
                   help="Sample fraction for LPIPS/SSIM (default 0.10).")
    p.add_argument("--cycle-sample-every", type=int, default=30,
                   help="Cycle consistency sample interval in frames (default 30).")
    p.add_argument("--skip-lpips", action="store_true",
                   help="Skip LPIPS (avoids torch dependency).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random baseline.")
    return p.parse_args()


def main():
    args = parse_args()
    plans = get_available_plans()
    if not plans:
        print("No plans found in dataset directory.")
        return 1

    if args.plan:
        if args.plan not in plans:
            print(f"❌ Plan '{args.plan}' not found. Available: {plans}")
            return 1
        plan = args.plan
    else:
        plan = select_plan(plans)

    offsets = [int(x.strip()) for x in args.offsets.split(",") if x.strip()]
    run_full_evaluation(
        plan,
        offsets=offsets,
        sample_rate=args.sample_rate,
        cycle_sample_every=args.cycle_sample_every,
        skip_lpips=args.skip_lpips,
        random_seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
