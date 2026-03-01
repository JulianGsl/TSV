import unittest
import numpy as np
from new_method.dtw_alignment import compute_dtw

class TestDTW(unittest.TestCase):
    def test_simple_sequence(self):
        features1 = np.array([0, 1, 2, 3]).reshape(-1, 1)
        features2 = np.array([0, 1, 2, 3]).reshape(-1, 1)

        path, cost = compute_dtw(features1, features2)

        # Should be diagonal path: (0,0), (1,1), (2,2), (3,3)
        self.assertEqual(path, [(0,0), (1,1), (2,2), (3,3)])
        self.assertEqual(cost[-1, -1], 0.0)

    def test_shifted_sequence(self):
        # features1 = [0, 1, 2]
        # features2 = [0, 0, 1, 2] (shifted by 1)
        features1 = np.array([0, 1, 2]).reshape(-1, 1)
        features2 = np.array([0, 0, 1, 2]).reshape(-1, 1)

        path, cost = compute_dtw(features1, features2)

        # Path should verify minimal cost (with penalty, it's > 0)
        self.assertTrue(cost[-1, -1] > 0.0)
        # Expected path could vary slightly depending on implementation preference for diagonal
        # But generally: (0,0), (0,1), (1,2), (2,3)
        self.assertEqual(path[-1], (2, 3))
        self.assertEqual(path[0], (0, 0))

if __name__ == '__main__':
    unittest.main()
