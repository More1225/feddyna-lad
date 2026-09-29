# coding: utf-8
"""
FedLTN: Federated Learning with Local Test-time Normalization
Aggregate the feature extractor globally and train the classifier locally. Persist the classifier across rounds.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any

from .base_method import BaseFLMethod
from .registry import register_method


@register_method("fedltn")
class FedLTNMethod(BaseFLMethod):

    def __init__(self, args, model: nn.Module, device: torch.device):
        super().__init__(args, model, device)
        self.num_classes = getattr(args, 'num_classes', 2)
        self.client_prototypes: Dict[int, Dict[int, torch.Tensor]] = {}
        self.global_prototypes: Dict[int, torch.Tensor] = {}
        self.classifier_keys = self._identify_classifier_keys(model)
        self.client_classifier_states: Dict[int, Dict[str, torch.Tensor]] = {}

    def _identify_classifier_keys(self, model: nn.Module) -> list:
        if hasattr(model, 'classifier_weight_keys'):
            return model.classifier_weight_keys

        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)

        classifier_keys = []
        if linear_layers:
            last_linear_name = linear_layers[-1]
            for name, param in model.named_parameters():
                if name.startswith(last_linear_name):
                    classifier_keys.append(name)

        return classifier_keys

    def post_train_hook(
        self,
        client_id: int,
        trained_model: nn.Module,
        train_result: Dict,
        round_num: int = 0,
    ):
        if 'prototypes' in train_result:
            self.client_prototypes[client_id] = train_result['prototypes']

        current_state = trained_model.state_dict()
        self.client_classifier_states[client_id] = {
            k: v.clone() for k, v in current_state.items() if k in self.classifier_keys
        }

    def pre_aggregate_hook(self, client_states: Dict[int, Dict[str, torch.Tensor]]):
        class_prototypes: Dict[int, list] = {c: [] for c in range(self.num_classes)}
        for cid, prototypes in self.client_prototypes.items():
            for class_id, proto in prototypes.items():
                if isinstance(proto, torch.Tensor):
                    class_prototypes[class_id].append(proto)
        for class_id, proto_list in class_prototypes.items():
            if proto_list:
                self.global_prototypes[class_id] = torch.stack(proto_list).mean(dim=0)

    def requires_personalized_inference(self) -> bool:
        return True

    def get_personalized_model(
        self,
        client_id: int,
        global_state: Dict[str, torch.Tensor],
        client_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        personalized_state = {}
        saved_cls = self.client_classifier_states.get(client_id, {})
        for key in global_state:
            if key in self.classifier_keys:
                if key in saved_cls:
                    personalized_state[key] = saved_cls[key]
                else:
                    personalized_state[key] = client_state.get(key, global_state[key])
            else:
                personalized_state[key] = global_state[key]
        return personalized_state

    def aggregate(
        self,
        client_states: Dict[int, Dict[str, torch.Tensor]],
        client_weights: Dict[int, float],
        global_state: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        new_global_state = {}
        weight_sum = sum(client_weights.values())

        for key in global_state.keys():
            if key in self.classifier_keys:
                new_global_state[key] = global_state[key]
            else:
                weighted_sum = torch.zeros_like(global_state[key])
                for cid, state in client_states.items():
                    weighted_sum += client_weights[cid] / weight_sum * state[key]
                new_global_state[key] = weighted_sum

        return new_global_state

    def post_pruning_update(
        self,
        current_state_dict: Dict[str, torch.Tensor],
        current_mask: Any,
        val_accuracy: float,
    ) -> Any:
        return current_mask
