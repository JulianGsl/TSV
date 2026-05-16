"""
Backward-compatible shim.

The DTW core moved to `combined_method/dtw_core.py` on 2026-05 so that
the active pipeline does not import from `Old_version/`. This file
re-exports `compute_dtw` from the new location, so any test or external
script that still references `Old_version.new_method.dtw_alignment`
keeps working without modification.

New code should import directly from `combined_method.dtw_core`.
"""

import os
import sys

# Make `combined_method` importable from this nested location.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TSV_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _TSV_ROOT not in sys.path:
    sys.path.insert(0, _TSV_ROOT)

from combined_method.dtw_core import compute_dtw  # noqa: E402, F401

__all__ = ["compute_dtw"]
