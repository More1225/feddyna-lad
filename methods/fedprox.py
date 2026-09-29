# coding: utf-8
"""
FedProx: Federated Optimization in Heterogeneous Networks
Add a proximal term to local training: min F_k(w) + (mu/2) * ||w - w_t||^2.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any

from .base_method import BaseFLMethod
from .registry import register_method


@register_method("fedprox")
class FedProxMethod(BaseFLMethod):

    def __init__(self, args, model: nn.Module, device: torch.device):
        super().__init__(args, model, device)
        self.mu = getattr(args, 'fedprox_mu', 0.01)
        self.global_params = None

    def pre_train_hook(
        self,
        client_id: int,
        local_model: nn.Module,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        criterion=None,
        round_num: int = 0,
    ):
        self.global_params = {}
        for name, param in local_model.named_parameters():
            if param.requires_grad:
                self.global_params[name] = param.data.clone().to(self.device)

    def compute_extra_loss(
        self,
        client_id: int,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.global_params is None:
            return torch.tensor(0.0, device=self.device)

        proximal_term = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.global_params:
                global_param = self.global_params[name].to(param.device)
                proximal_term = proximal_term.to(param.device) + torch.sum((param - global_param) ** 2)

        return 0.5 * self.mu * proximal_term

    def post_train_hook(
        self,
        client_id: int,
        trained_model: nn.Module,
        train_result: Dict,
        round_num: int = 0,
    ):
        self.global_params = None
