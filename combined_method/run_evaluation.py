#!/usr/bin/env python3
"""
Hybrid Alignment Evaluation Runner — load-only mode.

Reads alignment data already produced by `run_alignment_hybrid.py` and computes:
  1. LPIPS — perceptual similarity (deep features, lower is better)
  2. SSIM  — structural similarity (Wang et al. 2004, higher is better)
  3. Cycle consistency — round-trip frame error (CycleGAN-inspired, lower is better)

This script does NOT re-run the hybrid alignment. It only loads cached data:
    dataset/<Plan>/hybrid_<algo>/alignment_results.csv          (forward V1 → V2)
    dataset/<Plan>/hybrid_<algo>/alignment_results_backward.csv (backward V2 → V1, cycle only)

If the backward CSV is missing, it will be computed once and cached
(only the cycle metric requires it).

Usage (interactive):
    cd Projet/TSV
    python combined_method/run_evaluation.py

Usage (CLI):
    python combined_method/run_evaluation.py Plan1 --metrics all --algorithm AKAZE
    python combined_method/run_evaluation.py Plan1 --metrics lpips,ssim
    python combined_method/run_evaluation.py all   --metrics cycle --algorithm BRISK
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

# Add project root (TSV/) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from combined_method.hybrid_alignment import align_videos_hybrid, SUPPORTED_ALGORITHMS
from combined_method.evaluation import (
    evaluate_alignment,
    compute_cycle_consistency,
    SSIMMetric,
)

# Resolve dataset path relative to this script (TSV/dataset),
# so it works regardless of the caller's cwd.
DATASET_DIR = os.path.join(project_root, "dataset")
AVAILABLE_METRICS = ["lpips", "ssim", "cycle"]

FORWARD_CSV  = "alignment_results.csv"
BACKWARD_CSV = "alignment_results_backward.csv"


# ---------------------------------------------------------------------------
# Data loading (no algorithm reruns)
# ---------------------------------------------------------------------------

def load_matches_from_csv(csv_path: str) -> list:
    """Load alignment matches from any CSV that has v1_frame and v2_frame columns."""
    matches = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            matches.append({
                "v1_frame": int(row["v1_frame"]),
                "v2_frame": int(row["v2_frame"]),
            })
    return matches


def save_matches_to_csv(matches: list, csv_path: str):
    """Save a minimal v1_frame/v2_frame CSV (used to cache the backward alignment)."""
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["v1_frame", "v2_frame"])
        w.writeheader()
        for m in matches:
            w.writerow({"v1_frame": int(m["v1_frame"]), "v2_frame": int(m["v2_frame"])})


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def get_available_plans():
    if not os.path.exists(DATASET_DIR):
        print(f"⚠️  Dataset directory '{DATASET_DIR}' not found.")
        return []
    plans = [d for d in os.listdir(DATASET_DIR)
             if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.lower().startswith("plan")]
    return sorted(plans)


def output_dir_for(plan: str, algorithm: str) -> str:
    return os.path.join(DATASET_DIR, plan, f"hybrid_{algorithm.lower()}")


# ---------------------------------------------------------------------------
# Interactive selectors
# ---------------------------------------------------------------------------

def select_plan(plans):
    print("\n📁 Available plans:")
    for i, p in enumerate(plans, 1):
        print(f"   {i}. {p}")
    print(f"   {len(plans) + 1}. all")
    while True:
        choice = input(f"\n➜ Select plan (1-{len(plans) + 1}) [default: 1]: ").strip() or "1"
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(plans):
                return [plans[n - 1]]
            if n == len(plans) + 1:
                return plans
        print("   Invalid choice.")


def select_algorithm():
    print("\n🔧 Feature matching algorithm:")
    for i, algo in enumerate(SUPPORTED_ALGORITHMS, 1):
        print(f"   {i}. {algo}")
    while True:
        choice = input(f"\n➜ Select algorithm (1-{len(SUPPORTED_ALGORITHMS)}) [default: 1 AKAZE]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(SUPPORTED_ALGORITHMS):
            return SUPPORTED_ALGORITHMS[int(choice) - 1]
        print("   Invalid choice.")


def select_metrics():
    print("\n📊 Evaluation metrics:")
    print("   1. LPIPS  (perceptual similarity, lower = better)")
    print("   2. SSIM   (structural similarity, higher = better)")
    print("   3. Cycle consistency (round-trip frame error, lower = better)")
    print("   4. All")
    while True:
        choice = input("\n➜ Select metric(s) (1-4) [default: 4 all]: ").strip() or "4"
        mapping = {"1": ["lpips"], "2": ["ssim"], "3": ["cycle"], "4": AVAILABLE_METRICS}
        if choice in mapping:
            return mapping[choice]
        print("   Invalid choice.")


# ---------------------------------------------------------------------------
# Per-plan evaluation (load-only)
# ---------------------------------------------------------------------------

def evaluate_plan(plan: str, algorithm: str, metrics_to_run: list, sample_rate: float) -> dict | None:
    plan_dir = os.path.join(DATASET_DIR, plan)
    v1_path = os.path.join(plan_dir, "video1.mp4")
    v2_path = os.path.join(plan_dir, "video2.mp4")

    for path in (v1_path, v2_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing video: {path}")

    out_dir = output_dir_for(plan, algorithm)
    forward_csv  = os.path.join(out_dir, FORWARD_CSV)
    backward_csv = os.path.join(out_dir, BACKWARD_CSV)

    print(f"\n{'=' * 70}")
    print(f"  PLAN: {plan}   ALGORITHM: {algorithm}")
    print(f"  Metrics: {', '.join(m.upper() for m in metrics_to_run)}")
    print(f"{'=' * 70}")

    # --- Forward alignment: REQUIRED, must be pre-computed -----------------
    if not os.path.exists(forward_csv):
        print(f"\n❌  No alignment data found at:\n      {forward_csv}\n"
              f"   Run the alignment first:\n"
              f"      python combined_method/run_alignment_hybrid.py {plan}\n"
              f"   Skipping {plan}.")
        return None

    print(f"\n[1] Loading forward alignment V1 → V2…")
    forward = load_matches_from_csv(forward_csv)
    print(f"    ✓ {len(forward)} matches loaded from {os.path.basename(forward_csv)}")

    results = {
        "plan": plan,
        "algorithm": algorithm,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "video_v1": os.path.abspath(v1_path),
        "video_v2": os.path.abspath(v2_path),
        "sample_rate": sample_rate,
        "n_forward_matches": len(forward),
    }

    # --- LPIPS / SSIM (per-frame metrics) ----------------------------------
    perframe_metrics = []
    if "lpips" in metrics_to_run:
        try:
            from combined_method.evaluation import LPIPSMetric
            perframe_metrics.append(LPIPSMetric())
        except ImportError as e:
            print(f"\n⚠️  LPIPS unavailable ({e}). Run:  pip install lpips torch torchvision")
            results["lpips_error"] = str(e)
    if "ssim" in metrics_to_run:
        perframe_metrics.append(SSIMMetric())

    if perframe_metrics:
        names = ", ".join(m.name for m in perframe_metrics)
        print(f"\n[2] Computing per-frame metrics ({names}) — sampling {sample_rate * 100:.0f}%…")
        t0 = time.time()
        perceptual = evaluate_alignment(
            v1_path, v2_path, forward,
            metrics=perframe_metrics, sample_rate=sample_rate, verbose=True,
        )
        print(f"    ✓ Done in {time.time() - t0:.1f}s on {perceptual['n_sampled']} pairs")

        # Strip per-pair scores from the saved JSON to keep it small
        compact = {"n_sampled": perceptual["n_sampled"], "sample_rate": perceptual["sample_rate"]}
        for name, stats in perceptual["per_metric"].items():
            compact[name] = {k: v for k, v in stats.items() if k != "scores"}
        results["per_frame"] = compact

    # --- Cycle consistency -------------------------------------------------
    if "cycle" in metrics_to_run:
        print(f"\n[3] Cycle consistency…")

        # Load or compute backward alignment (cached after first run)
        if os.path.exists(backward_csv):
            backward = load_matches_from_csv(backward_csv)
            print(f"    ✓ Backward alignment loaded from {os.path.basename(backward_csv)} "
                  f"({len(backward)} matches)")
        else:
            print(f"    ⚙  No backward alignment cached at {os.path.basename(backward_csv)}")
            print(f"       Computing it now (one-time cost — required for cycle metric)…")
            t0 = time.time()
            backward = align_videos_hybrid(v2_path, v1_path,
                                           algorithm=algorithm, verbose=False)
            print(f"    ✓ Backward alignment computed in {time.time() - t0:.1f}s")
            save_matches_to_csv(backward, backward_csv)
            print(f"    💾 Cached to {backward_csv}")

        cycle = compute_cycle_consistency(
            v1_path, v2_path,
            sample_every=30,
            algorithm=algorithm,
            forward_matches=forward,
            backward_matches=backward,
            verbose=False,
        )
        results["cycle_consistency"] = {k: v for k, v in cycle.items() if k != "errors"}

    # --- Save & summarise --------------------------------------------------
    out_path = os.path.join(out_dir, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved: {out_path}")
    print_summary(results)
    return results


def print_summary(results: dict):
    print(f"\n  ── Summary ──")
    if "per_frame" in results:
        for name in ("LPIPS", "SSIM"):
            if name in results["per_frame"]:
                s = results["per_frame"][name]
                arrow = "↑" if s.get("higher_is_better") else "↓"
                print(f"    {name} {arrow}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
                      f"min={s['min']:.4f}  max={s['max']:.4f}")
    if "cycle_consistency" in results:
        c = results["cycle_consistency"]
        print(f"    Cycle ↓  mean={c['mean_error']:.2f} frames  "
              f"median={c['median_error']:.0f}  max={c['max_error']}  "
              f"perfect={c['perfect_percent']:.1f}%  ≤5frames={c['within_5_percent']:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate hybrid alignment quality (load-only).")
    p.add_argument("plan", nargs="?", help="Plan name (e.g. Plan1) or 'all'. Omit for interactive mode.")
    p.add_argument("--algorithm", "-a", choices=SUPPORTED_ALGORITHMS, default=None,
                   help="Feature matching algorithm.")
    p.add_argument("--metrics", "-m", default=None,
                   help="Comma-separated list: lpips,ssim,cycle  or  'all'.")
    p.add_argument("--sample-rate", type=float, default=0.10,
                   help="Fraction sampled for LPIPS/SSIM (default 0.10 = every ~10th frame).")
    return p.parse_args()


def resolve_metrics(arg: str) -> list:
    if arg in (None, "", "all"):
        return AVAILABLE_METRICS
    chosen = [m.strip().lower() for m in arg.split(",")]
    invalid = [m for m in chosen if m not in AVAILABLE_METRICS]
    if invalid:
        raise SystemExit(f"Unknown metric(s): {invalid}. Choose from: {AVAILABLE_METRICS}")
    return chosen


def main():
    args = parse_args()
    available = get_available_plans()
    if not available:
        print("No plans found in dataset directory.")
        return 1

    if args.plan:
        if args.plan.lower() == "all":
            plans = available
        elif args.plan in available:
            plans = [args.plan]
        else:
            print(f"❌  Plan '{args.plan}' not found. Available: {available}")
            return 1
    else:
        plans = select_plan(available)

    algorithm = args.algorithm or select_algorithm()

    metrics_to_run = resolve_metrics(args.metrics) if args.metrics is not None else select_metrics()

    sample_rate = args.sample_rate

    print(f"\n🚀 Evaluating {len(plans)} plan(s) with {algorithm} | "
          f"metrics={metrics_to_run} | sample_rate={sample_rate}")
    t_total = time.time()
    for plan in plans:
        try:
            evaluate_plan(plan, algorithm, metrics_to_run, sample_rate)
        except Exception as e:
            print(f"\n❌  {plan} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ All done in {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
