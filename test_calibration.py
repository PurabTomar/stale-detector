"""
test_calibration.py - Unit tests for benchmark calibration curve calculations.

Run with:
    python -m unittest test_calibration.py -v
"""

from __future__ import annotations

import unittest

from calibration import compute_calibration


class TestCalibration(unittest.TestCase):
    def test_calibration_not_enough_data(self):
        # Empty input
        res = compute_calibration([])
        self.assertEqual(res["status"], "not_enough_data")
        self.assertIn("Not enough", res["message"])

        # Fewer than 5 claims
        res_few = compute_calibration([
            {"score": 90, "correct": True},
            {"score": 20, "correct": True},
        ])
        self.assertEqual(res_few["status"], "not_enough_data")

    def test_calibration_binned_calculation(self):
        sample_results = [
            {"score": 10, "correct": True},
            {"score": 15, "correct": True},
            {"score": 20, "correct": False},
            {"score": 35, "correct": True},
            {"score": 40, "correct": False},
            {"score": 60, "correct": True},
            {"score": 80, "correct": True},
            {"score": 85, "correct": True},
            {"score": 95, "correct": True},
            {"score": 100, "correct": False},
        ]
        cal = compute_calibration(sample_results)
        self.assertEqual(cal["status"], "ok")
        self.assertEqual(len(cal["bins"]), 4)

        # Check bin 0-25: 3 items (10, 15, 20), 2 correct -> 66.7%
        bin_0_25 = cal["bins"][0]
        self.assertEqual(bin_0_25["bin"], "0–25")
        self.assertEqual(bin_0_25["n"], 3)
        self.assertEqual(bin_0_25["correct"], 2)
        self.assertAlmostEqual(bin_0_25["accuracy_pct"], 66.7, places=1)

        # Check bin 76-100: 4 items (80, 85, 95, 100), 3 correct -> 75.0%
        bin_76_100 = cal["bins"][3]
        self.assertEqual(bin_76_100["bin"], "76–100")
        self.assertEqual(bin_76_100["n"], 4)
        self.assertEqual(bin_76_100["correct"], 3)
        self.assertEqual(bin_76_100["accuracy_pct"], 75.0)

        # Check plain language reading
        self.assertIn("Scores above 75 were right 75.0% of the time (n=4)", cal["reading"])


if __name__ == "__main__":
    unittest.main()
