"""
Shared helpers for the run_*.py CLI scripts.

Centralises plan discovery, interactive selection menus and the minimal
CSV I/O used to cache forward / backward alignments. Each run_*.py script
used to define its own copy of these — kept here once.
"""

import csv
import os
from typing import Dict, List, Sequence


# ---------------------------------------------------------------------------
# Plan discovery
# ---------------------------------------------------------------------------

def get_available_plans(dataset_dir: str) -> List[str]:
    """
    Return sorted list of plan-folder names inside `dataset_dir`.

    A plan folder is any subdirectory whose name starts with 'plan' (case-insensitive).
    Returns [] if the dataset directory is missing.
    """
    if not os.path.isdir(dataset_dir):
        return []
    return sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d.lower().startswith("plan")
    )


# ---------------------------------------------------------------------------
# Interactive selectors
# ---------------------------------------------------------------------------

def select_plan(plans: Sequence[str], allow_all: bool = True) -> List[str]:
    """
    Prompt the user to pick a plan (or 'all'). Returns a list of selected plans.
    """
    print("\n📁 Available plans:")
    for i, p in enumerate(plans, 1):
        print(f"   {i}. {p}")
    if allow_all:
        print(f"   {len(plans) + 1}. all")
    upper = len(plans) + (1 if allow_all else 0)
    while True:
        choice = input(f"\n➜ Select plan (1-{upper}) [default: 1]: ").strip() or "1"
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(plans):
                return [plans[n - 1]]
            if allow_all and n == len(plans) + 1:
                return list(plans)
        print("   Invalid choice.")


def select_algorithm(supported: Sequence[str]) -> str:
    """
    Prompt the user to pick a feature-matching algorithm. Default is the first.
    """
    print("\n🔧 Feature matching algorithm:")
    for i, algo in enumerate(supported, 1):
        print(f"   {i}. {algo}")
    while True:
        choice = input(f"\n➜ Select algorithm (1-{len(supported)}) [default: 1 {supported[0]}]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(supported):
            return supported[int(choice) - 1]
        print("   Invalid choice.")


# ---------------------------------------------------------------------------
# Minimal CSV I/O for cached alignments
# ---------------------------------------------------------------------------

def load_matches_from_csv(csv_path: str) -> List[Dict]:
    """Load alignment matches from any CSV that has v1_frame and v2_frame columns."""
    matches = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            matches.append({
                "v1_frame": int(row["v1_frame"]),
                "v2_frame": int(row["v2_frame"]),
            })
    return matches


def save_matches_to_csv(matches: List[Dict], csv_path: str) -> None:
    """Save a minimal v1_frame/v2_frame CSV (used to cache the backward alignment)."""
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["v1_frame", "v2_frame"])
        w.writeheader()
        for m in matches:
            w.writerow({"v1_frame": int(m["v1_frame"]), "v2_frame": int(m["v2_frame"])})
