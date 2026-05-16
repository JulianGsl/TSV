"""
Feature extractors for DTW. Each function takes a BGR frame and returns a
1-D float32 vector. The driver L1-normalizes per-frame at the end.
"""

import cv2
import numpy as np


def feat_intensity_hist_32(frame):
    h = frame.shape[0]
    crop = frame[h // 3:, :]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.calcHist([g], [0], None, [32], [0, 256]).flatten()


def feat_intensity_hist_full_64(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.calcHist([g], [0], None, [64], [0, 256]).flatten()


def feat_hue_hist_32(frame):
    h = frame.shape[0]
    crop = frame[h // 3:, :]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()


def feat_rgb_hist_48(frame):
    h = frame.shape[0]
    crop = frame[h // 3:, :]
    feats = []
    for c in range(3):
        feats.append(cv2.calcHist([crop], [c], None, [16], [0, 256]).flatten())
    return np.concatenate(feats)


def feat_grid_intensity_3x3(frame):
    h = frame.shape[0]
    crop = frame[h // 3:, :]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    feats = []
    for r in range(3):
        for c in range(3):
            cell = g[r*H//3:(r+1)*H//3, c*W//3:(c+1)*W//3]
            feats.append(cv2.calcHist([cell], [0], None, [16], [0, 256]).flatten())
    return np.concatenate(feats)


def feat_grad_hist_16(frame):
    h = frame.shape[0]
    crop = frame[h // 3:, :]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)
    hist, _ = np.histogram(ang, bins=16, range=(0, 360), weights=mag)
    return hist.astype(np.float32)


def feat_grad_hist_3x3_grid(frame):
    """HOG-style: grid of gradient orientation histograms (3x3 cells × 16 bins = 144-dim)."""
    h = frame.shape[0]
    crop = frame[h // 3:, :]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)
    H, W = g.shape
    feats = []
    for r in range(3):
        for c in range(3):
            m = mag[r*H//3:(r+1)*H//3, c*W//3:(c+1)*W//3]
            a = ang[r*H//3:(r+1)*H//3, c*W//3:(c+1)*W//3]
            hist, _ = np.histogram(a, bins=16, range=(0, 360), weights=m)
            feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


def feat_combined_grid_color(frame):
    """grid_intensity_3x3 (144) + hue_hist_32 (32) → 176-dim."""
    a = feat_grid_intensity_3x3(frame)
    b = feat_hue_hist_32(frame)
    return np.concatenate([a, b])


def feat_lowres_gray_64(frame):
    h = frame.shape[0]
    crop = frame[h // 3:, :]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    return small.flatten()


EXTRACTORS = {
    "intensity_hist_32":      feat_intensity_hist_32,
    "intensity_hist_full_64": feat_intensity_hist_full_64,
    "hue_hist_32":            feat_hue_hist_32,
    "rgb_hist_48":            feat_rgb_hist_48,
    "grid_intensity_3x3":     feat_grid_intensity_3x3,
    "grad_hist_16":           feat_grad_hist_16,
    "grad_hist_3x3_grid":     feat_grad_hist_3x3_grid,
    "combined_grid_color":    feat_combined_grid_color,
    "lowres_gray_64":         feat_lowres_gray_64,
}


def extract_features(video_path: str, extractor_name: str, sample_rate: int):
    """Apply extractor over a video at sample_rate, L1-normalize per frame.

    Returns:
        features: float32 (n_sampled, d)
        indices: list of original frame indices
    """
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
        v = fn(frame).astype(np.float32)
        s = v.sum()
        if s > 1e-6:
            v = v / s
        feats.append(v)
        idx.append(i)
    cap.release()
    return np.asarray(feats, dtype=np.float32), idx
