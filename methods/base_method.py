# coding: utf-8
"""Base class defining the common interface for federated-learning methods"""

from abc import ABC
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class BaseFLMethod(ABC):

    def __init__(self, args, model: nn.Module, device: torch.device):
        self.args = args
        self.model = model
        self.device = device
        self.global_state = None
        self.client_masks: Dict[int, Any] = {}

    def set_global_state(self, global_state: Dict[str, torch.Tensor]):
        self.global_state = global_state

    def set_client_mask(self, client_id: int, mask: Any):
        self.client_masks[client_id] = mask

    def get_client_mask(self, client_id: int) -> Optional[Any]:
        return self.client_masks.get(client_id)

    # ==================== Training hooks ====================

    def pre_train_hook(self, client_id, local_model, train_loader=None,
                       val_loader=None, optimizer=None, criterion=None, round_num=0):
        pass

    def post_train_hook(self, client_id, trained_model, train_result, round_num=0):
        pass

    def compute_extra_loss(self, client_id, model, outputs, targets, inputs=None):
        return torch.tensor(0.0, device=self.device)

    def compute_loss(self, model, output, target, criterion, client_id, inputs=None):
        base_loss = criterion(output, target)
        extra_loss = self.compute_extra_loss(
            client_id=client_id, model=model, outputs=output, targets=target, inputs=inputs)
        return base_loss + extra_loss

    # ==================== Aggregation hooks ====================

    def pre_aggregate_hook(self, client_states):
        pass

    def aggregate(self, client_states, client_weights, global_state, **kwargs):
        new_global_state = {}
        weight_sum = sum(client_weights.values())
        for key in global_state.keys():
            weighted_sum = torch.zeros_like(global_state[key])
            for cid, state in client_states.items():
                weighted_sum += client_weights[cid] / weight_sum * state[key]
            new_global_state[key] = weighted_sum
        return new_global_state

    def post_aggregate_hook(self, global_state):
        pass

    # ==================== Personalization ====================

    def requires_personalized_inference(self) -> bool:
        return False

    def get_personalized_model(self, client_id, global_state, client_state):
        return global_state

    def train_personal_model(self, client_id, global_model, train_loader, device, epochs=None):
        return {}

    # ==================== Alternating training ====================

    def should_alternate_training(self) -> bool:
        return False

    # ==================== Pruning ====================

    def post_pruning_update(self, current_state_dict, current_mask, val_accuracy):
        return current_mask

    # ==================== Weight updates ====================

    def update_all_weights(self, client_states):
        pass
