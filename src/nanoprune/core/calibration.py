import math
from typing import Tuple, Dict
import numpy as np

def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    A well-calibrated model has ECE < 0.05 (5%).
    """
    probs = np.asarray(probs, dtype=np.float64).flatten()
    labels = np.asarray(labels, dtype=np.float64).flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0

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


def fit_platt(scores, labels, l2: float = 1.0, iterations: int = 100) -> Tuple[float, float]:
    """Platt scaling: (a, b) such that sigmoid(a * score + b) estimates P(relevant).

    Fitted by L2-regularised logistic regression on standardised scores (Newton
    steps), which stays finite even when the calibration set is separable.
    """
    x = np.asarray(scores, dtype=np.float64).flatten()
    y = np.asarray(labels, dtype=np.float64).flatten()
    if x.size == 0 or len(set(y.tolist())) < 2:
        raise ValueError("Calibration needs both relevant and irrelevant examples")
    mu, sd = float(x.mean()), float(x.std()) or 1.0
    design = np.c_[(x - mu) / sd, np.ones_like(x)]
    penalty = np.diag([l2, 0.0])
    w = np.zeros(2)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-design @ w))
        gradient = design.T @ (p - y) + penalty @ w
        hessian = design.T @ (design * (p * (1.0 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0] / sd), float(w[1] - w[0] * mu / sd)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    # log(1 + exp(-|z|)) formulation, stable for large |z|
    log_p = -np.logaddexp(0.0, -z)
    log_not_p = -np.logaddexp(0.0, z)
    return float(-np.mean(labels * log_p + (1.0 - labels) * log_not_p))


class TemperatureScaler:
    """
    Post-hoc temperature scaling to calibrate model logits.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-4, float(temperature))

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
        bounds: Tuple[float, float] = (0.05, 20.0),
    ) -> float:
        """
        Finds the temperature T minimising the negative log-likelihood of
        sigmoid(logits / T) on a validation set.

        NLL is unimodal in log T, so a golden-section search over ``bounds``
        converges reliably (``lr`` is kept for backwards compatibility).
        """
        logits = np.asarray(logits, dtype=np.float64).flatten()
        labels = np.asarray(labels, dtype=np.float64).flatten()
        if logits.size == 0:
            return self.temperature

        lo, hi = math.log(bounds[0]), math.log(bounds[1])
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        fa, fb = _nll(logits, labels, math.exp(a)), _nll(logits, labels, math.exp(b))
        for _ in range(max(10, max_iter)):
            if fa <= fb:
                hi, b, fb = b, a, fa
                a = hi - ratio * (hi - lo)
                fa = _nll(logits, labels, math.exp(a))
            else:
                lo, a, fa = a, b, fb
                b = lo + ratio * (hi - lo)
                fb = _nll(logits, labels, math.exp(b))
            if hi - lo < 1e-6:
                break

        self.temperature = float(math.exp((lo + hi) / 2.0))
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits) / self.temperature
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))
