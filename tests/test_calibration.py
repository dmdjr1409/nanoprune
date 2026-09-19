import unittest
import numpy as np
from nanoprune.core.calibration import compute_ece, TemperatureScaler

class TestCalibration(unittest.TestCase):
    def test_compute_ece(self):
        probs = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        metrics = compute_ece(probs, labels)
        self.assertIn("ece", metrics)
        self.assertIn("brier_score", metrics)
        self.assertLess(metrics["ece"], 0.20)
        self.assertLess(metrics["brier_score"], 0.10)

    def test_temperature_scaler(self):
        logits = np.array([-5.0, -2.0, 2.0, 5.0])
        labels = np.array([0, 0, 1, 1])
        scaler = TemperatureScaler(temperature=1.0)
        scaler.fit(logits, labels)
        probs = scaler.transform(logits)
        self.assertEqual(len(probs), 4)
        self.assertLess(probs[0], 0.1)
        self.assertGreater(probs[3], 0.9)

if __name__ == "__main__":
    unittest.main()
