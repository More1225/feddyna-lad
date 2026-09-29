from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch


EPSILON = 1e-12


def _iter_trainable_names(*state_dicts: Mapping[str, torch.Tensor]) -> Iterable[str]:
    if not state_dicts:
        return []

    common_names = set(state_dicts[0].keys())
    for state in state_dicts[1:]:
        common_names &= set(state.keys())

    return (
        name for name in state_dicts[0].keys()
        if name in common_names and (name.endswith(".weight") or name.endswith(".bias"))
    )


def compute_knowledge_drift(
    client_state_dict: Mapping[str, torch.Tensor],
    global_state_dict: Mapping[str, torch.Tensor],
    epsilon: float = EPSILON,
) -> float:
    """Compute ||theta_i - theta_g||_2 / (||theta_g||_2 + epsilon)."""
    diff_sq_sum = 0.0
    global_sq_sum = 0.0

    with torch.no_grad():
        for name in _iter_trainable_names(client_state_dict, global_state_dict):
            client_param = client_state_dict[name].detach().double().cpu()
            global_param = global_state_dict[name].detach().double().cpu()
            diff = client_param - global_param
            diff_sq_sum += float(torch.sum(diff * diff).item())
            global_sq_sum += float(torch.sum(global_param * global_param).item())

    drift_norm = float(np.sqrt(max(diff_sq_sum, 0.0)))
    global_norm = float(np.sqrt(max(global_sq_sum, 0.0)))
    return drift_norm / (global_norm + epsilon)


def compute_mean_rate(
    current_state_dict: Mapping[str, torch.Tensor],
    prev_state_dict: Mapping[str, torch.Tensor],
    epsilon: float = EPSILON,
) -> float:
    """Compute the mean element-wise relative parameter change rate."""
    rate_sum = 0.0
    total_params = 0

    with torch.no_grad():
        for name in _iter_trainable_names(current_state_dict, prev_state_dict):
            current_param = current_state_dict[name].detach().double().cpu()
            prev_param = prev_state_dict[name].detach().double().cpu()
            rate = torch.abs(current_param - prev_param) / (torch.abs(prev_param) + epsilon)
            rate_sum += float(torch.sum(rate).item())
            total_params += int(rate.numel())

    if total_params == 0:
        return 0.0
    return rate_sum / total_params


def _min_max_normalize(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values

    v_min = float(np.min(values))
    v_max = float(np.max(values))
    normalized = (values - v_min) / (v_max - v_min + epsilon)
    return np.clip(normalized, 0.0, 1.0)


def compute_kdi(
    drift_values: np.ndarray,
    mean_rate_values: np.ndarray,
    alpha: float = 0.7,
    epsilon: float = EPSILON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute KDI = alpha * normalized drift + (1-alpha) * normalized mean rate."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    drift_values = np.asarray(drift_values, dtype=np.float64)
    mean_rate_values = np.asarray(mean_rate_values, dtype=np.float64)

    if drift_values.shape != mean_rate_values.shape:
        raise ValueError(
            f"drift_values and mean_rate_values must have the same shape: "
            f"{drift_values.shape} != {mean_rate_values.shape}"
        )

    drift_hat = _min_max_normalize(drift_values, epsilon=epsilon)
    mean_rate_hat = _min_max_normalize(mean_rate_values, epsilon=epsilon)
    kdi = alpha * drift_hat + (1.0 - alpha) * mean_rate_hat
    return np.clip(kdi, 0.0, 1.0), drift_hat, mean_rate_hat


def compute_personalization_ratio(
    kdi_values: np.ndarray,
    p_min: float = 0.2,
    p_max: float = 0.8,
) -> np.ndarray:
    """Map KDI values to per-update personalization rates in [p_min, p_max]."""
    if p_min > p_max:
        raise ValueError(f"p_min ({p_min}) cannot be greater than p_max ({p_max})")

    kdi_values = np.asarray(kdi_values, dtype=np.float64)
    if kdi_values.size == 0:
        return kdi_values

    ratios = p_min + (p_max - p_min) * np.clip(kdi_values, 0.0, 1.0)
    return np.clip(ratios, p_min, p_max)


def compute_kdars_ratios(
    client_ids: List[int],
    client_state_dicts: Dict[int, OrderedDict],
    prev_state_dicts: Dict[int, OrderedDict],
    global_state_dict: OrderedDict,
    alpha: float = 0.7,
    personal_min_ratio: float = 0.2,
    personal_max_ratio: float = 0.8,
    epsilon: float = EPSILON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute incremental retention ratios, KDI scores, and diagnostics."""
    drift_values = np.asarray(
        [
            compute_knowledge_drift(client_state_dicts[cid], global_state_dict, epsilon=epsilon)
            for cid in client_ids
        ],
        dtype=np.float64,
    )
    mean_rate_values = np.asarray(
        [
            compute_mean_rate(client_state_dicts[cid], prev_state_dicts[cid], epsilon=epsilon)
            for cid in client_ids
        ],
        dtype=np.float64,
    )

    kdi_values, drift_hat, mean_rate_hat = compute_kdi(
        drift_values,
        mean_rate_values,
        alpha=alpha,
        epsilon=epsilon,
    )
    step_personal_ratios = compute_personalization_ratio(
        kdi_values,
        p_min=personal_min_ratio,
        p_max=personal_max_ratio,
    )
    step_global_retention_ratios = np.clip(
        1.0 - step_personal_ratios,
        0.0,
        1.0,
    )

    diagnostics = np.column_stack(
        [
            drift_values,
            mean_rate_values,
            drift_hat,
            mean_rate_hat,
            kdi_values,
            step_personal_ratios,
            step_global_retention_ratios,
        ]
    )
    return step_global_retention_ratios, kdi_values, diagnostics
