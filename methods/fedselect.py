# coding: utf-8
"""
FedSelect independent baseline adapter.

This adapter keeps FedSelect out of the FedDyna pruning path. It owns client
masks, updates them from local parameter deltas, and performs mask-aware
aggregation inside the method implementation.
"""

from collections import OrderedDict
from typing import Dict
import copy

import numpy as np
import torch
import torch.nn as nn

from .base_method import BaseFLMethod
from .registry import register_method


def _init_zero_mask(model: nn.Module) -> OrderedDict:
    mask = OrderedDict()
    for name, param in model.state_dict().items():
        if name.endswith(".weight") or name.endswith(".bias"):
            mask[name] = torch.zeros_like(param, dtype=torch.float32)
    return mask


def _compute_fedselect_mask(
    current_state: Dict[str, torch.Tensor],
    previous_state: Dict[str, torch.Tensor],
    current_mask: OrderedDict,
    keep_ratio: float,
) -> OrderedDict:
    """Keep the smallest-delta shared parameters as mask=0; move others to mask=1."""
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    new_mask = copy.deepcopy(current_mask)
    delta_values = []
    positions = []

    for name, mask_tensor in current_mask.items():
        if name not in current_state or name not in previous_state:
            continue
        global_positions = (mask_tensor == 0).cpu().numpy()
        if not np.any(global_positions):
            continue

        delta = torch.abs(
            current_state[name].detach().cpu() - previous_state[name].detach().cpu()
        ).numpy()
        flat_indices = np.where(global_positions.flatten())[0]
        for flat_idx, value in zip(flat_indices, delta[global_positions].flatten()):
            delta_values.append(float(value))
            positions.append((name, int(flat_idx)))

    if not delta_values:
        return new_mask

    keep_count = int(round(keep_ratio * len(delta_values)))
    keep_count = max(0, min(keep_count, len(delta_values)))
    if keep_count == 0:
        threshold = -np.inf
    elif keep_count == len(delta_values):
        threshold = np.inf
    else:
        threshold = np.sort(delta_values)[keep_count - 1]

    updates_by_name = {}
    for (name, flat_idx), value in zip(positions, delta_values):
        updates_by_name.setdefault(name, []).append((flat_idx, value))

    for name, updates in updates_by_name.items():
        mask_tensor = new_mask[name]
        shape = mask_tensor.shape
        flat_mask = mask_tensor.flatten().cpu().numpy()
        for flat_idx, value in updates:
            if value > threshold:
                flat_mask[flat_idx] = 1.0
        new_mask[name] = torch.from_numpy(flat_mask.reshape(shape)).to(mask_tensor.device)

    return new_mask


@register_method("fedselect")
class FedSelectMethod(BaseFLMethod):
    def __init__(self, args, model: nn.Module, device: torch.device):
        super().__init__(args, model, device)
        self.keep_ratio = getattr(args, "fedselect_keep_ratio", 0.8)
        self.update_interval = getattr(args, "fedselect_update_interval", 3)
        self.default_mask = _init_zero_mask(model)
        self.pre_train_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.latest_client_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.latest_round = -1

    def set_client_mask(self, client_id: int, mask):
        self.client_masks[client_id] = mask

    def _ensure_client_mask(self, client_id: int) -> OrderedDict:
        if client_id not in self.client_masks:
            self.client_masks[client_id] = copy.deepcopy(self.default_mask)
        return self.client_masks[client_id]

    def pre_train_hook(
        self,
        client_id,
        local_model,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        criterion=None,
        round_num=0,
    ):
        self._ensure_client_mask(client_id)
        self.pre_train_states[client_id] = {
            k: v.detach().cpu().clone()
            for k, v in local_model.state_dict().items()
        }
        self.latest_round = round_num

    def post_train_hook(
        self,
        client_id: int,
        trained_model: nn.Module,
        train_result: Dict,
        round_num: int = 0,
    ):
        self.latest_client_states[client_id] = {
            k: v.detach().cpu().clone()
            for k, v in trained_model.state_dict().items()
        }
        self.latest_round = round_num

    def update_all_weights(self, client_states: Dict[int, Dict[str, torch.Tensor]]):
        if self.latest_round <= 0:
            return
        if self.update_interval <= 0:
            return
        if self.latest_round % self.update_interval != 0:
            return

        for cid, current_state in client_states.items():
            previous_state = self.pre_train_states.get(cid)
            if previous_state is None:
                continue

            shared_mask = self._ensure_client_mask(cid)
            new_mask = _compute_fedselect_mask(
                current_state=current_state,
                previous_state=previous_state,
                current_mask=shared_mask,
                keep_ratio=self.keep_ratio,
            )

            # Mutate the existing mask object so round_runner.client_masks stays in sync.
            shared_mask.clear()
            shared_mask.update(new_mask)

    def aggregate(
        self,
        client_states: Dict[int, Dict[str, torch.Tensor]],
        client_weights: Dict[int, float],
        global_state: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        new_global_state = {}
        weight_sum = sum(client_weights.values())

        for key, global_tensor in global_state.items():
            if not global_tensor.is_floating_point():
                first_cid = next(iter(client_states))
                new_global_state[key] = client_states[first_cid][key].clone()
                continue

            weighted_sum = torch.zeros_like(global_tensor)
            denominator = torch.zeros_like(global_tensor, dtype=torch.float32)

            for cid, state in client_states.items():
                weight = client_weights[cid] / weight_sum if weight_sum > 0 else 0.0
                mask = self._ensure_client_mask(cid)
                if key in mask:
                    participation = (mask[key].to(global_tensor.device) == 0).float()
                else:
                    participation = torch.ones_like(global_tensor, dtype=torch.float32)
                weighted_sum += weight * participation * state[key].to(global_tensor.device)
                denominator += weight * participation

            aggregated = global_tensor.clone()
            valid = denominator != 0
            aggregated[valid] = weighted_sum[valid] / denominator[valid]
            new_global_state[key] = aggregated

        return new_global_state

    def requires_personalized_inference(self) -> bool:
        return True

    def get_personalized_model(
        self,
        client_id: int,
        global_state: Dict[str, torch.Tensor],
        client_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mask = self._ensure_client_mask(client_id)
        personalized_state = {}
        for key, global_tensor in global_state.items():
            if key not in mask or key not in client_state:
                personalized_state[key] = global_tensor
                continue
            mask_tensor = mask[key].to(global_tensor.device)
            local_tensor = client_state[key].to(global_tensor.device)
            combined = global_tensor.clone()
            combined[mask_tensor == 1] = local_tensor[mask_tensor == 1]
            personalized_state[key] = combined
        return personalized_state
