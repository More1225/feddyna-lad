from __future__ import annotations

from typing import Tuple

import numpy as np

from .exceptions import ValidationError


def entropy_weight_method(
    X: np.ndarray,
    epsilon: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    m, n = X.shape

    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < epsilon] = 1.0
    X_norm = (X - X_min) / X_range

    X_shifted = X_norm + epsilon
    P = X_shifted / np.clip(X_shifted.sum(axis=0, keepdims=True), epsilon, None)

    k = 1.0 / np.log(m) if m > 1 else 1.0
    E = -k * np.sum(P * np.log(np.clip(P, epsilon, None)), axis=0)

    d = 1.0 - E
    d_sum = float(np.sum(d))
    if (not np.isfinite(d_sum)) or d_sum <= epsilon:
        weights = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        weights = d / d_sum

    scores = np.sum(X_norm * weights, axis=1)
    return weights, scores


def compute_universality_score(
    delta_f1: float,
    alpha: float = 10.0,
    optimal: float = 0.02,
    penalty_threshold: float = 0.0,
) -> float:
    universality = np.exp(-alpha * (delta_f1 - optimal) ** 2)

    if delta_f1 < penalty_threshold:
        universality *= 0.5

    return float(np.clip(universality, 0.0, 1.0))


def map_to_pruning_rate(
    scores: np.ndarray,
    p_min: float = 0.2,
    p_max: float = 0.8,
    epsilon: float = 1e-12,
) -> np.ndarray:
    if p_min > p_max:
        raise ValidationError(f"p_min ({p_min}) cannot be greater than p_max ({p_max})")

    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores

    s_min = float(scores.min())
    s_max = float(scores.max())

    if abs(s_max - s_min) < epsilon:
        return np.full_like(scores, fill_value=(p_min + p_max) / 2.0, dtype=np.float64)

    scaled = (scores - s_min) / (s_max - s_min)
    global_ratios = p_max - (p_max - p_min) * scaled
    return np.clip(global_ratios, p_min, p_max)
