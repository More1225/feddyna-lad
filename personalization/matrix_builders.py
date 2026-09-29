from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.math_utils import compute_universality_score


def compute_anomaly_contribution(
    stats: Dict[str, float],
    method: str = "weighted_log",
) -> float:
    normal_count = stats.get("normal_count", 0)
    anomaly_count = stats.get("anomaly_count", 0)
    total_count = normal_count + anomaly_count

    if total_count == 0:
        return 0.0

    anomaly_ratio = anomaly_count / total_count if total_count > 0 else 0.0

    if method == "weighted_log":
        anomaly_score = np.log(anomaly_count + 1)
        ratio_weight = 0.5 + 0.5 * min(anomaly_ratio * 10, 1.0)
        return float(anomaly_score * ratio_weight)

    if method == "scarcity_weighted":
        scarcity = 1.0 / (anomaly_ratio + 0.01)
        contribution = anomaly_count * np.log(scarcity + 1)
        return float(np.log(contribution + 1))

    if method == "information_theory":
        if anomaly_ratio > 0:
            info_per_sample = -np.log(anomaly_ratio + 1e-6)
            total_info = anomaly_count * info_per_sample
            return float(np.log(total_info + 1))
        return 0.0

    return float(np.log(anomaly_count + 1))


def compute_aggregation_factors_matrix(
    client_ids: List[int],
    delta_f1_dict: Dict[int, float],
    data_stats_dict: Dict[int, Dict[str, float]],
    update_alignment_dict: Dict[int, float],
    update_stability_dict: Dict[int, float],
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Build the entropy-weighting matrix from client contribution indicators.

    Columns:
      1. universality derived from the local/global F1 improvement;
      2. log sample count as statistical support;
      3. scope-aware leave-one-out update alignment;
      4. scope-aware temporal update stability.
    """
    rows = []
    for cid in client_ids:
        delta_f1 = float(delta_f1_dict.get(cid, 0.0))
        stats = data_stats_dict.get(cid, {})

        universality = compute_universality_score(
            delta_f1=delta_f1,
            alpha=10.0,
            optimal=0.02,
            penalty_threshold=0.0,
        )
        sample_count = max(float(stats.get("sample_count", 0.0)), 0.0)
        log_sample_count = float(np.log1p(sample_count))
        update_alignment = float(update_alignment_dict.get(cid, 0.5))
        update_stability = float(update_stability_dict.get(cid, 0.5))

        rows.append(
            [
                universality,
                log_sample_count,
                update_alignment,
                update_stability,
            ]
        )

    X = np.asarray(rows, dtype=np.float64)
    if X.size == 0:
        return X

    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < epsilon] = 1.0
    return (X - X_min) / X_range


def compute_coordination_analysis(
    agg_weights: Dict[int, float],
    personalization_rates: Dict[int, float],
    client_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    coordination = {}

    weights = [agg_weights.get(cid, 0.0) for cid in client_ids]
    rates = [personalization_rates.get(cid, 0.5) for cid in client_ids]

    weight_median = np.median(weights) if weights else 0.1
    rate_median = np.median(rates) if rates else 0.5

    for cid in client_ids:
        w = agg_weights.get(cid, 0.0)
        p = personalization_rates.get(cid, 0.5)

        high_weight = w > weight_median
        high_personalization = p < (1 - rate_median)

        if high_weight and not high_personalization:
            client_type = "global_leader"
        elif high_weight and high_personalization:
            client_type = "strong_personalized"
        elif not high_weight and high_personalization:
            client_type = "weak_contributor"
        else:
            client_type = "follower"

        coordination[cid] = {
            "agg_weight": float(w),
            "global_ratio": float(p),
            "local_ratio": float(1 - p),
            "client_type": client_type,
            "coordination_ratio": float(p / (w + 1e-6)),
        }

    return coordination
