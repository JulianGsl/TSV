"""
Feature transforms applied between extraction and DTW. Each takes the two
feature matrices (V1 and V2) and returns the two transformed matrices.
"""

import numpy as np


def t_raw(f1, f2):
    return f1, f2


def t_per_video_zscore(f1, f2):
    """Each video z-scored using its own mean/std (per-dim)."""
    def z(x):
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True) + 1e-6
        return (x - mu) / sd
    return z(f1), z(f2)


def t_joint_zscore(f1, f2):
    """Z-scored using stats from the concatenation of both videos."""
    cat = np.vstack([f1, f2])
    mu = cat.mean(axis=0, keepdims=True)
    sd = cat.std(axis=0, keepdims=True) + 1e-6
    return (f1 - mu) / sd, (f2 - mu) / sd


TRANSFORMS = {
    "raw":              t_raw,
    "per_video_zscore": t_per_video_zscore,
    "joint_zscore":     t_joint_zscore,
}
