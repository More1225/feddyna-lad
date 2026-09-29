from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.math_utils import entropy_weight_method
from .kdars import compute_kdars_ratios
from .matrix_builders import (
    compute_aggregation_factors_matrix,
    compute_coordination_analysis,
)
from .metrics_tracker import ClientMetricsTracker, GlobalMetricsTracker


class PersonalizationController:
    """Controller for aggregation weights and KDARS-based mask ratios."""

    ENTROPY_FACTOR_NAMES = (
        "universality",
        "log_sample_count",
        "update_alignment",
        "update_stability",
    )

    def __init__(self, client_ids: List[int], args: Any) -> None:
        self.client_tracker = ClientMetricsTracker(client_ids, max_history=args.max_metrics_history)
        self.global_tracker = GlobalMetricsTracker()
        self.args = args

    def update_client(
        self,
        client_id: int,
        metrics: Dict[str, float],
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client_tracker.update_round(client_id, metrics, data_stats)

    def update_global(self, metrics: Dict[str, float]) -> None:
        self.global_tracker.update_round(metrics)

    def compute_entropy_aggregation(
        self,
        delta_f1_dict: Dict[int, float],
        update_alignment_dict: Optional[Dict[int, float]] = None,
        update_stability_dict: Optional[Dict[int, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        client_ids = self.client_tracker.get_all_client_ids()

        update_alignment_dict = update_alignment_dict or {}
        update_stability_dict = update_stability_dict or {}
        X = compute_aggregation_factors_matrix(
            client_ids=client_ids,
            delta_f1_dict=delta_f1_dict,
            data_stats_dict=self.client_tracker.data_stats,
            update_alignment_dict=update_alignment_dict,
            update_stability_dict=update_stability_dict,
        )

        feature_weights, client_scores = entropy_weight_method(X)

        num_clients = len(client_scores)
        if num_clients == 0:
            agg_weights = client_scores
        else:
            score_sum = float(np.sum(client_scores))
            if (not np.isfinite(score_sum)) or score_sum <= 0.0:
                agg_weights = np.full(num_clients, 1.0 / num_clients, dtype=np.float64)
            else:
                agg_weights = client_scores / score_sum

        return agg_weights, client_scores, feature_weights, X

    def compute_pruning_rates(
        self,
        client_state_dicts: Optional[Dict[int, Any]] = None,
        prev_state_dicts: Optional[Dict[int, Any]] = None,
        global_state_dict: Optional[Any] = None,
        use_legacy: bool = False,
        **compat_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-update mask-retention ratios with KDARS.

        Returns:
            global_ratios: fraction of the currently shared parameters retained
                for this update. The mask layer applies it cumulatively and
                enforces the final personalization bounds.
            kdi_scores: KDARS KDI values.
            diagnostics: columns are drift, mean_rate, drift_hat, mean_rate_hat,
                kdi, step_personal_ratio, step_global_retention_ratio.

        The legacy keyword arguments are kept for call-site compatibility and are
        intentionally ignored by KDARS.
        """
        del use_legacy, compat_kwargs

        client_ids = self.client_tracker.get_all_client_ids()
        if not client_ids:
            empty = np.asarray([], dtype=np.float64)
            return empty, empty, np.empty((0, 7), dtype=np.float64)

        missing_state = (
            client_state_dicts is None
            or prev_state_dicts is None
            or global_state_dict is None
        )
        if missing_state:
            personal_min = float(getattr(self.args, "personal_min_ratio", 0.2))
            personal_max = float(getattr(self.args, "personal_max_ratio", 0.8))
            personal_mid = (personal_min + personal_max) / 2.0
            global_mid = 1.0 - personal_mid
            global_ratios = np.full(len(client_ids), global_mid, dtype=np.float64)
            kdi_scores = np.zeros(len(client_ids), dtype=np.float64)
            diagnostics = np.column_stack(
                [
                    np.zeros(len(client_ids), dtype=np.float64),
                    np.zeros(len(client_ids), dtype=np.float64),
                    np.zeros(len(client_ids), dtype=np.float64),
                    np.zeros(len(client_ids), dtype=np.float64),
                    kdi_scores,
                    np.full(len(client_ids), personal_mid, dtype=np.float64),
                    global_ratios,
                ]
            )
            return global_ratios, kdi_scores, diagnostics

        return compute_kdars_ratios(
            client_ids=client_ids,
            client_state_dicts=client_state_dicts,
            prev_state_dicts=prev_state_dicts,
            global_state_dict=global_state_dict,
            alpha=float(getattr(self.args, "kdi_alpha", 0.7)),
            personal_min_ratio=float(getattr(self.args, "personal_min_ratio", 0.2)),
            personal_max_ratio=float(getattr(self.args, "personal_max_ratio", 0.8)),
            epsilon=1e-12,
        )

    def compute_coordination(
        self,
        agg_weights: Dict[int, float],
        global_ratios: np.ndarray,
    ) -> Dict[int, Dict[str, float]]:
        client_ids = self.client_tracker.get_all_client_ids()
        personalization_rates = {
            cid: float(global_ratios[idx])
            for idx, cid in enumerate(client_ids)
        }

        return compute_coordination_analysis(
            agg_weights=agg_weights,
            personalization_rates=personalization_rates,
            client_ids=client_ids,
        )

    def get_decision_summary(
        self,
        delta_f1_dict: Dict[int, float],
        update_alignment_dict: Optional[Dict[int, float]] = None,
        update_stability_dict: Optional[Dict[int, float]] = None,
        client_state_dicts: Optional[Dict[int, Any]] = None,
        prev_state_dicts: Optional[Dict[int, Any]] = None,
        global_state_dict: Optional[Any] = None,
    ) -> Dict[str, Any]:
        client_ids = self.client_tracker.get_all_client_ids()

        agg_weights, scores, feature_weights, X_agg = self.compute_entropy_aggregation(
            delta_f1_dict=delta_f1_dict,
            update_alignment_dict=update_alignment_dict,
            update_stability_dict=update_stability_dict,
        )

        global_ratios, kdi_scores, kdars_matrix = self.compute_pruning_rates(
            client_state_dicts=client_state_dicts,
            prev_state_dicts=prev_state_dicts,
            global_state_dict=global_state_dict,
        )

        agg_weights_dict = {
            cid: float(agg_weights[idx])
            for idx, cid in enumerate(client_ids)
        }
        global_ratios_dict = {
            cid: float(global_ratios[idx])
            for idx, cid in enumerate(client_ids)
        }

        coordination = self.compute_coordination(agg_weights_dict, global_ratios)

        return {
            "client_ids": client_ids,
            "aggregation": {
                "weights": agg_weights_dict,
                "scores": scores.tolist(),
                "feature_weights": feature_weights.tolist(),
                "matrix": X_agg.tolist(),
                "factor_names": list(self.ENTROPY_FACTOR_NAMES),
            },
            "personalization": {
                "global_ratios": global_ratios_dict,
                "kdi_scores": kdi_scores.tolist(),
                "matrix": kdars_matrix.tolist(),
                "matrix_columns": [
                    "drift",
                    "mean_rate",
                    "drift_hat",
                    "mean_rate_hat",
                    "kdi",
                    "step_personal_ratio",
                    "step_global_retention_ratio",
                ],
            },
            "coordination": coordination,
        }
