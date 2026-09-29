# coding: utf-8
"""
FedAS: Bridging Inconsistency in Personalized Federated Learning
1. Parameter-alignment regularization: keep shared parameters close to each client's history.
2. FIM-trace-weighted aggregation: assign lower weights to parameters with higher uncertainty.
3. Keep the classifier local.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import numpy as np

from .base_method import BaseFLMethod
from .registry import register_method


@register_method("fedas")
class FedASMethod(BaseFLMethod):

    def __init__(self, args, model: nn.Module, device: torch.device):
        super().__init__(args, model, device)
        self.align_weight = getattr(args, 'fedas_align_weight', 0.5)
        self.fim_temperature = getattr(args, 'yxy_hyper', 1.0)
        self.client_history: Dict[int, Dict] = {}
        self.client_fim: Dict[int, float] = {}
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

    def compute_extra_loss(
        self,
        client_id: int,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor = None,
    ) -> torch.Tensor:
        if client_id not in self.client_history:
            return torch.tensor(0.0, device=self.device)

        history_state = self.client_history[client_id]
        align_loss = torch.tensor(0.0, device=self.device)
        n_shared = 0

        for name, param in model.named_parameters():
            if name in self.classifier_keys:
                continue
            if name in history_state and param.requires_grad:
                history_param = history_state[name].to(self.device)
                align_loss += F.mse_loss(param, history_param.detach())
                n_shared += 1

        if n_shared > 0:
            align_loss = align_loss / n_shared

        return self.align_weight * align_loss

    def post_train_hook(
        self,
        client_id: int,
        trained_model: nn.Module,
        train_result: Dict,
        round_num: int = 0,
    ):
        self.client_history[client_id] = {
            k: v.cpu().clone() for k, v in trained_model.state_dict().items()
        }

        current_state = trained_model.state_dict()
        self.client_classifier_states[client_id] = {
            k: v.clone() for k, v in current_state.items() if k in self.classifier_keys
        }

        fim = self._compute_fim_trace(trained_model)
        if fim is not None:
            self.client_fim[client_id] = fim

    def _compute_fim_trace(self, model: nn.Module) -> Optional[float]:
        """
        Diagonal approximation of the FIM trace using parameter variance.
        Exact computation requires log-likelihood gradients over training data, but post_train_hook does not receive a train_loader.
        """
        model = model.to(self.device)
        model.eval()

        fim_trace = 0.0
        n_params = 0

        try:
            for name, param in model.named_parameters():
                if param.requires_grad and name not in self.classifier_keys:
                    param_mean = param.data.mean()
                    param_var = ((param.data - param_mean) ** 2).mean()
                    fim_trace += param_var.item()
                    n_params += 1

            if n_params > 0:
                fim_trace = fim_trace / n_params
        except Exception:
            return None

        return fim_trace

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
                    personalized_state[key] = saved_cls[key].to(self.device)
                else:
                    personalized_state[key] = client_state.get(key, global_state[key]).to(self.device)
            else:
                personalized_state[key] = global_state[key].to(self.device)
        return personalized_state

    def aggregate(
        self,
        client_states: Dict[int, Dict[str, torch.Tensor]],
        client_weights: Dict[int, float],
        global_state: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        new_global_state = {}

        fim_weights = self._compute_fim_weights(client_states.keys())

        combined_weights = {}
        total_weight = 0.0
        for cid in client_states:
            w = client_weights[cid] * fim_weights.get(cid, 1.0)
            combined_weights[cid] = w
            total_weight += w

        if total_weight <= 0:
            total_weight = sum(client_weights.values())
            combined_weights = dict(client_weights)

        for key in global_state.keys():
            if key in self.classifier_keys:
                new_global_state[key] = global_state[key]
            else:
                weighted_sum = torch.zeros_like(global_state[key].to(self.device))
                for cid, state in client_states.items():
                    w = combined_weights[cid] / total_weight
                    weighted_sum += w * state[key].to(self.device)
                new_global_state[key] = weighted_sum

        return new_global_state

    def _compute_fim_weights(self, client_ids) -> Dict[int, float]:
        if not self.client_fim:
            return {cid: 1.0 for cid in client_ids}

        fim_values = []
        cids = []
        for cid in client_ids:
            fim_values.append(self.client_fim.get(cid, 0.0))
            cids.append(cid)

        if all(v == 0 for v in fim_values):
            return {cid: 1.0 for cid in client_ids}

        fim_array = np.array(fim_values, dtype=np.float64)
        epsilon = 1e-8
        inv_fim = 1.0 / (fim_array + epsilon)
        inv_fim_norm = inv_fim / (np.max(inv_fim) + epsilon)
        temperature = max(self.fim_temperature, 0.01)

        weights = np.exp(inv_fim_norm / temperature)
        weights = weights / weights.sum()

        return {cid: float(w) for cid, w in zip(cids, weights)}
