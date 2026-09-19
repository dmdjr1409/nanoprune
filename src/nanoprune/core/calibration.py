from typing import List, Tuple, Dict
import numpy as np

def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    A well-calibrated model has ECE < 0.05 (5%).
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total_samples = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            diff = np.abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)

    brier_score = float(np.mean((probs - labels) ** 2))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
    }


class TemperatureScaler:
    """
    Post-hoc temperature scaling to calibrate model logits.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-4, float(temperature))

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 100):
        """
        Gradient descent on negative log-likelihood over validation set to optimize temperature T.
        """
        logits = np.asarray(logits, dtype=np.float64).flatten()
        labels = np.asarray(labels, dtype=np.float64).flatten()

        t = self.temperature
        for _ in range(max_iter):
            scaled = logits / t
            # Stable sigmoid
            p = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))
            # Gradient of NLL w.r.t temperature T
            grad = -np.sum((labels - p) * (logits / (t ** 2))) / len(logits)
            t = max(0.01, t - lr * grad)

        self.temperature = float(t)
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits) / self.temperature
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))
