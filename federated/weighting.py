import numpy as np

from core.math_utils import compute_universality_score


def _uniform_weights(client_ids):
    if not client_ids:
        return {}
    uniform_weight = 1.0 / len(client_ids)
    return {cid: float(uniform_weight) for cid in client_ids}


def _protect_weights_array(weights, min_weight=0.06, smooth_alpha=0.10, eps=1e-12):
    """
    Apply safeguards to client weights:
    1) Smooth slightly toward a uniform distribution
    2) Enforce a minimum weight floor
    3) Renormalize
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        return weights

    n = weights.size

    s = float(weights.sum())
    if (not np.isfinite(s)) or s <= eps:
        weights = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        weights = weights / s

    if smooth_alpha is not None and smooth_alpha > 0:
        uniform = np.full(n, 1.0 / n, dtype=np.float64)
        weights = (1.0 - smooth_alpha) * weights + smooth_alpha * uniform

    if min_weight is not None and min_weight > 0:
        max_allowed_floor = (1.0 / n) - 1e-6
        if min_weight >= max_allowed_floor:
            raise ValueError(
                f"min_weight={min_weight} is too large; with {n} clients it must be below {max_allowed_floor:.6f}"
            )
        weights = np.clip(weights, min_weight, None)

    s = float(weights.sum())
    if (not np.isfinite(s)) or s <= eps:
        return np.full(n, 1.0 / n, dtype=np.float64)

    return weights / s


def _protect_weights_dict(
    weight_dict,
    enable=True,
    min_weight=0.06,
    smooth_alpha=0.10,
):
    if not enable:
        return weight_dict

    client_ids = sorted(weight_dict.keys())
    if not client_ids:
        return {}

    raw = np.array([float(weight_dict[cid]) for cid in client_ids], dtype=np.float64)
    protected = _protect_weights_array(
        raw,
        min_weight=min_weight,
        smooth_alpha=smooth_alpha,
    )
    return {cid: float(w) for cid, w in zip(client_ids, protected)}


def compute_client_weights(
    method,
    client_metrics,
    personalization_controller=None,
    args=None,
    return_diagnostics=False,
):
    client_ids = sorted(client_metrics.keys())
    if not client_ids:
        return ({}, None) if return_diagnostics else {}

    def _return(weights, diagnostics=None):
        if return_diagnostics:
            return weights, diagnostics
        return weights

    if method == "fedavg":
        sample_counts = np.array(
            [
                client_metrics[cid]["data_stats"].get("sample_count", 0)
                for cid in client_ids
            ],
            dtype=np.float64,
        )
        total_samples = sample_counts.sum()

        if total_samples <= 0:
            raw_weights = _uniform_weights(client_ids)
        else:
            weights = sample_counts / total_samples
            raw_weights = {cid: float(w) for cid, w in zip(client_ids, weights)}

        protected_weights = _protect_weights_dict(
            raw_weights,
            enable=getattr(args, "enable_weight_protection", False),
            min_weight=getattr(args, "agg_weight_min", 0.06),
            smooth_alpha=getattr(args, "agg_weight_smooth", 0.10),
        )
        return _return(protected_weights)

    delta_f1_dict = {cid: client_metrics[cid]["delta_f1"] for cid in client_ids}

    if method == "entropy":
        if personalization_controller is None:
            raise ValueError("personalization_controller is required for weighting method 'entropy'")
        update_alignment_dict = {
            cid: float(client_metrics[cid].get("update_alignment", 0.5))
            for cid in client_ids
        }
        update_stability_dict = {
            cid: float(client_metrics[cid].get("update_stability", 0.5))
            for cid in client_ids
        }
        weights, scores, feature_weights, factor_matrix = (
            personalization_controller.compute_entropy_aggregation(
                delta_f1_dict,
                update_alignment_dict=update_alignment_dict,
                update_stability_dict=update_stability_dict,
            )
        )
        raw_weights = {cid: float(w) for cid, w in zip(client_ids, weights)}
        protected_weights = _protect_weights_dict(
            raw_weights,
            enable=getattr(args, "enable_weight_protection", False),
            min_weight=getattr(args, "agg_weight_min", 0.06),
            smooth_alpha=getattr(args, "agg_weight_smooth", 0.10),
        )
        factor_names = list(personalization_controller.ENTROPY_FACTOR_NAMES)
        diagnostics = {
            "factor_names": factor_names,
            "factor_weights": {
                name: float(weight)
                for name, weight in zip(factor_names, feature_weights)
            },
            "client_scores": {
                cid: float(score)
                for cid, score in zip(client_ids, scores)
            },
            "client_factors": {
                cid: {
                    name: float(value)
                    for name, value in zip(factor_names, row)
                }
                for cid, row in zip(client_ids, factor_matrix)
            },
            "raw_client_factors": {
                cid: {
                    "universality": compute_universality_score(
                        float(client_metrics[cid]["delta_f1"]),
                        alpha=10.0,
                        optimal=0.02,
                        penalty_threshold=0.0,
                    ),
                    "log_sample_count": float(
                        np.log1p(
                            max(
                                float(
                                    client_metrics[cid]["data_stats"].get(
                                        "sample_count", 0.0
                                    )
                                ),
                                0.0,
                            )
                        )
                    ),
                    "update_alignment": update_alignment_dict[cid],
                    "update_stability": update_stability_dict[cid],
                }
                for cid in client_ids
            },
            "raw_weights": raw_weights,
            "protected_weights": protected_weights,
        }
        return _return(protected_weights, diagnostics)

    raise ValueError(f"Unsupported weighting method: {method}")
