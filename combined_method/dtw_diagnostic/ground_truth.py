"""
Hardcoded ground-truth anchors for Plan2 DTW evaluation.

Each entry is (v1_frame, v2_frame): V1 and V2 frame numbers corresponding
to the same geographic location on the rail track. Manually identified by
the experimenter on Plan2 — the only plan we evaluate against ground truth.
"""

PLAN2_ANCHORS = [
    (0, 0),
    (128, 114),
    (264, 237),
    (526, 483),
    (708, 673),
    (937, 915),
    (1045, 1039),
    (1201, 1221),
    (1368, 1489),
    (1582, 1693),
    (1765, 1941),
    (1882, 2163),
    (2170, 2525),
    (2273, 2681),
]


def get_anchors(plan_name: str = "Plan2"):
    """Return the Plan2 anchors (hardcoded). Other plans return None."""
    if plan_name != "Plan2":
        return None
    return list(PLAN2_ANCHORS)


def interpolate_truth(v1_frame: int, anchors=None):
    """Linear interpolation of the ground-truth V2 frame for any V1 frame.

    Returns None outside the anchor range (extrapolation is unreliable).
    `anchors` arg kept for API compat with earlier callers.
    """
    a = anchors or PLAN2_ANCHORS
    a = sorted(a)
    if v1_frame < a[0][0] or v1_frame > a[-1][0]:
        return None
    for k in range(len(a) - 1):
        v1a, v2a = a[k]
        v1b, v2b = a[k + 1]
        if v1a <= v1_frame <= v1b:
            if v1b == v1a:
                return v2a
            t = (v1_frame - v1a) / (v1b - v1a)
            return v2a + t * (v2b - v2a)
    return None
