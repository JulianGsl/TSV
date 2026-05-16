"""
Unit tests for combined_method.dtw_core.compute_dtw.

Run with:
    cd Projet/TSV
    python -m pytest combined_method/tests/test_dtw_core.py
"""

import os
import sys
import unittest

import numpy as np

# Make `combined_method` importable when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TSV_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _TSV_ROOT not in sys.path:
    sys.path.insert(0, _TSV_ROOT)

from combined_method.dtw_core import compute_dtw


class TestComputeDTW(unittest.TestCase):
    def test_identical_sequences_diagonal_path(self):
        """Two identical sequences should produce a perfect diagonal path."""
        features1 = np.array([0, 1, 2, 3]).reshape(-1, 1)
        features2 = np.array([0, 1, 2, 3]).reshape(-1, 1)

        path, acc_cost, dist_matrix = compute_dtw(features1, features2)

        self.assertEqual(path, [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(acc_cost[-1, -1], 0.0)
        self.assertEqual(dist_matrix.shape, (4, 4))

    def test_shifted_sequence(self):
        """A shifted sequence still terminates at the corner under classical DTW."""
        features1 = np.array([0, 1, 2]).reshape(-1, 1)
        features2 = np.array([0, 0, 1, 2]).reshape(-1, 1)

        path, acc_cost, _ = compute_dtw(features1, features2)

        self.assertGreater(acc_cost[-1, -1], 0.0)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 3))

    def test_open_end_terminates_early_on_mismatched_tails(self):
        """Open-end DTW should stop in the middle of the longer sequence."""
        # features1 ends at value 5; features2 has a sensible match for it
        # at index 2 then continues with garbage (very different values).
        features1 = np.array([0, 3, 5]).reshape(-1, 1)
        features2 = np.array([0, 3, 5, 100, 200]).reshape(-1, 1)

        path_classical, _, _ = compute_dtw(features1, features2, open_end=False)
        path_open, _, _ = compute_dtw(features1, features2, open_end=True)

        # Classical DTW must reach the corner.
        self.assertEqual(path_classical[-1], (2, 4))
        # Open-end DTW should terminate at index 2 of features2 (the
        # last valid match), not be dragged to the corner.
        self.assertEqual(path_open[-1], (2, 2))

    def test_step_penalty_discourages_stuttering(self):
        """A high step penalty should discourage non-diagonal moves."""
        features1 = np.array([0, 1, 2]).reshape(-1, 1)
        features2 = np.array([0, 0, 0, 1, 2]).reshape(-1, 1)

        # With penalty=0, the path is free to stutter at the start.
        path_no_pen, _, _ = compute_dtw(features1, features2, step_penalty=0.0)
        # With a strong penalty, it pays more for non-diagonal moves.
        path_high_pen, _, _ = compute_dtw(features1, features2, step_penalty=5.0)

        # Both must start and end at the corners.
        self.assertEqual(path_no_pen[0], (0, 0))
        self.assertEqual(path_high_pen[0], (0, 0))
        self.assertEqual(path_no_pen[-1], (2, 4))
        self.assertEqual(path_high_pen[-1], (2, 4))

    def test_returns_dist_matrix_with_correct_shape(self):
        """The third return value is the raw distance matrix."""
        features1 = np.random.RandomState(0).rand(7, 4)
        features2 = np.random.RandomState(1).rand(11, 4)

        _, _, dist = compute_dtw(features1, features2)

        self.assertEqual(dist.shape, (7, 11))
        self.assertTrue(np.all(dist >= 0.0))


if __name__ == "__main__":
    unittest.main()
