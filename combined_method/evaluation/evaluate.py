"""
Frame-pair evaluation runner for video alignment quality assessment.

Given the alignment mapping produced by HybridAligner, this module:
1. Samples a fraction of frame pairs uniformly across the video.
2. Extracts the corresponding frames from V1 and V2.
3. Runs each metric on every sampled pair.
4. Returns per-pair scores and their averages.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Sequence

try:
    from .base import BaseMetric
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from combined_method.evaluation.base import BaseMetric


def sample_frame_pairs(
    matches: List[Dict],
    sample_rate: float = 0.10,
    dedup_v2: bool = False,
) -> List[Tuple[int, int]]:
    """
    Uniformly sample a fraction of the alignment mapping.

    Args:
        matches: List of dicts with at least 'v1_frame' and 'v2_frame' keys,
                 as returned by HybridAligner.align_videos().
        sample_rate: Fraction of pairs to sample (0.10 = 10 %).
        dedup_v2: If True, skip pairs that target an already-seen V2 frame.
                  Default False — DTW plateaus (one V2 mapped to many V1) are
                  legitimate alignment data; deduplicating biases the sample
                  toward the start of each plateau and silently shrinks the
                  effective sample size.

    Returns:
        List of (v1_frame_index, v2_frame_index) tuples, evenly spaced.
    """
    n_total = len(matches)
    n_sample = max(1, round(n_total * sample_rate))

    # Evenly spaced indices into the matches list
    indices = np.linspace(0, n_total - 1, n_sample, dtype=int)

    pairs = []
    seen_v2 = set()
    for idx in indices:
        m = matches[int(idx)]
        v1 = int(m["v1_frame"])
        v2 = int(m["v2_frame"])
        if dedup_v2 and v2 in seen_v2:
            continue
        pairs.append((v1, v2))
        seen_v2.add(v2)

    return pairs


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def evaluate_alignment(
    video_v1_path: str,
    video_v2_path: str,
    matches: List[Dict],
    metrics: Sequence[BaseMetric],
    sample_rate: float = 0.10,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate alignment quality by comparing sampled frame pairs.

    Args:
        video_v1_path: Path to the reference video (V1).
        video_v2_path: Path to the aligned video (V2).
        matches: Alignment mapping from HybridAligner.align_videos().
        metrics: List of BaseMetric instances to evaluate.
        sample_rate: Fraction of frame pairs to sample (default 10 %).
        verbose: Print progress if True.

    Returns:
        Dict with structure::

            {
                "n_sampled": 300,
                "sample_rate": 0.10,
                "per_metric": {
                    "LPIPS": {
                        "mean": 0.123,
                        "std":  0.045,
                        "min":  0.012,
                        "max":  0.287,
                        "scores": [0.123, ...]   # one per sampled pair
                    },
                    ...
                },
                "pairs": [(v1_idx, v2_idx), ...]  # sampled pairs
            }
    """
    pairs = sample_frame_pairs(matches, sample_rate)

    cap1 = cv2.VideoCapture(video_v1_path)
    cap2 = cv2.VideoCapture(video_v2_path)

    if not cap1.isOpened():
        raise RuntimeError(f"Cannot open V1 video: {video_v1_path}")
    if not cap2.isOpened():
        raise RuntimeError(f"Cannot open V2 video: {video_v2_path}")

    # Initialise per-metric score lists
    scores: Dict[str, List[float]] = {m.name: [] for m in metrics}
    skipped = 0
    n_pairs = len(pairs)
    bar_width = 30
    t_start = time.time()

    for i, (v1_idx, v2_idx) in enumerate(pairs):
        frame1 = _read_frame(cap1, v1_idx)
        frame2 = _read_frame(cap2, v2_idx)

        if frame1 is None or frame2 is None:
            skipped += 1
        else:
            for metric in metrics:
                score = metric.compute(frame1, frame2)
                scores[metric.name].append(score)

        if verbose:
            done = i + 1
            filled = int(bar_width * done / n_pairs)
            bar = "█" * filled + "░" * (bar_width - filled)
            elapsed = time.time() - t_start
            eta = (elapsed / done) * (n_pairs - done) if done > 0 else 0
            # Live scores for each metric
            live = "  ".join(
                f"{m.name}={np.mean(scores[m.name]):.4f}" if scores[m.name] else f"{m.name}=--"
                for m in metrics
            )
            print(f"\r    [{bar}] {done}/{n_pairs}  {live}  ETA {eta:.0f}s  ",
                  end="", flush=True)

    if verbose:
        print()  # newline after progress bar

    cap1.release()
    cap2.release()

    if skipped > 0 and verbose:
        print(f"    ⚠  {skipped} frame pairs skipped (unreadable frames).")

    # Aggregate
    per_metric = {}
    for metric in metrics:
        s = np.array(scores[metric.name], dtype=float)
        per_metric[metric.name] = {
            "mean": float(np.mean(s)) if len(s) else float("nan"),
            "std":  float(np.std(s))  if len(s) else float("nan"),
            "min":  float(np.min(s))  if len(s) else float("nan"),
            "max":  float(np.max(s))  if len(s) else float("nan"),
            "scores": s.tolist(),
            "higher_is_better": metric.higher_is_better(),
        }

    return {
        "n_sampled": len(pairs) - skipped,
        "sample_rate": sample_rate,
        "per_metric": per_metric,
        "pairs": pairs,
    }
