# coding: utf-8
"""
Debug utilities for personalized methods

Provide consistent diagnostic output for inspecting personalized methods.
"""

import torch
import numpy as np
from typing import Dict, List, Union


class PersonalizedMethodDebugger:
    """Personalized-method debugger"""

    @staticmethod
    def compare_models(
        model1_state: Dict[str, torch.Tensor],
        model2_state: Dict[str, torch.Tensor],
        method_name: str = "",
        client_id: int = -1,
        round_num: int = -1,
    ) -> Dict[str, float]:
        """
        Compare two model states.

        Args:
            model1_state: State dictionary for model 1
            model2_state: State dictionary for model 2
            method_name: Method name (for display)
            client_id: Client ID (for display)
            round_num: Round number (for display)

        Returns:
            Difference statistics
        """
        if not model1_state or not model2_state:
            return {"error": "Empty model state"}

        stats = {
            "num_params": 0,
            "num_diff_params": 0,
            "avg_diff": 0.0,
            "max_diff": 0.0,
            "total_change": 0.0,
        }

        all_diffs = []
        param_names = sorted(set(model1_state.keys()) & set(model2_state.keys()))

        for name in param_names:
            param1 = model1_state[name].flatten().cpu()
            param2 = model2_state[name].flatten().cpu()

            # Compute differences.
            diff = torch.abs(param1 - param2)
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            total_diff = diff.sum().item()

            all_diffs.append(mean_diff)

            stats["num_params"] += 1
            stats["total_change"] += total_diff
            if mean_diff > 1e-6:  # Count parameters with a significant difference.
                stats["num_diff_params"] += 1

            if max_diff > stats["max_diff"]:
                stats["max_diff"] = max_diff

        if all_diffs:
            stats["avg_diff"] = np.mean(all_diffs)

        # Print diagnostic information.
        if stats["num_params"] > 0:
            diff_ratio = stats["num_diff_params"] / stats["num_params"] * 100
            if method_name:
                print(f"[{method_name}] Model comparison (Client {client_id}, Round {round_num}):")
                print(f"  Total params: {stats['num_params']}")
                print(f"  Different params: {stats['num_diff_params']} ({diff_ratio:.2f}%)")
                print(f"  Average diff: {stats['avg_diff']:.6f}")
                print(f"  Max diff: {stats['max_diff']:.6f}")
                print(f"  Total change: {stats['total_change']:.6f}")

        return stats

    @staticmethod
    def compute_personalization_statistics(
        global_state: Dict[str, torch.Tensor],
        personalized_state: Dict[str, torch.Tensor],
        method_name: str = "",
        client_id: int = -1,
    ) -> Dict[str, float]:
        """
        Compute personalization statistics.

        Args:
            global_state: Global model state
            personalized_state: Personalized model state
            method_name: Method name
            client_id: Client ID

        Returns:
            Personalization statistics
        """
        stats = PersonalizedMethodDebugger.compare_models(
            global_state, personalized_state, method_name, client_id
        )

        # Compute the degree of personalization.
        if "num_params" in stats and stats["num_params"] > 0:
            personalized_ratio = stats["num_diff_params"] / stats["num_params"]
            stats["personalization_ratio"] = personalized_ratio

            if method_name:
                print(f"[{method_name}] Personalization ratio for Client {client_id}: {personalized_ratio:.2%}")

        return stats

    @staticmethod
    def log_weight_statistics(
        weights: Union[Dict, List[torch.Tensor], torch.Tensor],
        method_name: str,
        client_id: int = -1,
        round_num: int = -1,
        weight_name: str = "weights",
    ):
        """
        Record weight statistics.

        Args:
            weights: Weights (dictionary, list, or tensor)
            method_name: Method name
            client_id: Client ID
            round_num: Round number
            weight_name: Weight name (for display)
        """
        if weights is None:
            print(f"[{method_name}] {weight_name} is None")
            return

        if isinstance(weights, dict):
            # Dictionary of weights.
            weight_values = []
            for key, value in weights.items():
                if isinstance(value, (dict, list)):
                    continue
                elif torch.is_tensor(value):
                    weight_values.extend(value.flatten().cpu().tolist())
                else:
                    weight_values.append(float(value))

            if weight_values:
                values = np.array(weight_values)
                stats = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            else:
                stats = {"count": 0}
        elif torch.is_tensor(weights):
            # Tensor weights.
            values = weights.flatten().cpu().numpy()
            stats = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        elif isinstance(weights, list) and weights and torch.is_tensor(weights[0]):
            # List of tensors.
            all_values = []
            for tensor in weights:
                all_values.extend(tensor.flatten().cpu().numpy())
            values = np.array(all_values)
            stats = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            print(f"[{method_name}] Unknown weight type: {type(weights)}")
            return

        if stats["count"] > 0:
            print(f"[{method_name}] {weight_name} statistics (Client {client_id}, Round {round_num}):")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    @staticmethod
    def track_metric_differences(
        global_metrics: Dict[str, float],
        personalized_metrics: Dict[str, float],
        method_name: str,
        round_num: int = -1,
    ) -> Dict[str, float]:
        """
        Track metric differences.

        Args:
            global_metrics: Global model metrics
            personalized_metrics: Personalized model metrics
            method_name: Method name
            round_num: Round number

        Returns:
            Metric differences
        """
        differences = {}
        common_keys = set(global_metrics.keys()) & set(personalized_metrics.keys())

        for key in common_keys:
            diff = personalized_metrics[key] - global_metrics[key]
            differences[key] = diff

        # Print key metric differences.
        if "f1" in differences:
            print(f"[{method_name}] Round {round_num} metric differences:")
            for key in ["accuracy", "precision", "recall", "f1", "prauc"]:
                if key in differences:
                    diff = differences[key]
                    sign = "+" if diff >= 0 else ""
                    print(f"  {key}: {global_metrics.get(key, 0):.4f} → {personalized_metrics.get(key, 0):.4f} ({sign}{diff:.4f})")

        return differences