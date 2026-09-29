# coding: utf-8
"""
Ditto: Fair and Robust Federated Learning Through Personalization
Global branch plus personalized model; personalization objective: min F_k(v_k) + λ/2 * ||v_k - w_t||^2.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Optional, Any

from .base_method import BaseFLMethod
from .registry import register_method


@register_method("ditto")
class DittoMethod(BaseFLMethod):

    def __init__(self, args, model: nn.Module, device: torch.device):
        super().__init__(args, model, device)
        self.lambda_reg = getattr(args, 'ditto_lambda', 1.0)
        self.personal_epochs = getattr(args, 'ditto_personal_epochs', args.local_epochs)
        self.personal_models: Dict[int, nn.Module] = {}
        self.criterion = nn.CrossEntropyLoss()

    def requires_personalized_inference(self) -> bool:
        return True

    def get_personalized_model(
        self,
        client_id: int,
        global_state: Dict[str, torch.Tensor],
        client_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if client_id in self.personal_models:
            return self.personal_models[client_id].state_dict()
        return global_state

    def train_personal_model(
        self,
        client_id: int,
        global_model: nn.Module,
        train_loader,
        device: torch.device,
        epochs: int = None,
    ) -> Dict:
        if epochs is None:
            epochs = self.personal_epochs

        if client_id not in self.personal_models:
            self.personal_models[client_id] = copy.deepcopy(global_model)

        personal_model = self.personal_models[client_id]
        personal_model.train()

        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}

        optimizer = torch.optim.SGD(
            personal_model.parameters(),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=getattr(self.args, 'weight_decay', 0.0005)
        )

        total_loss = 0.0
        batch_count = 0

        for ep in range(epochs):
            for batch in train_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                optimizer.zero_grad()
                outputs = personal_model(inputs)
                if isinstance(outputs, tuple):
                    _, outputs = outputs

                cls_loss = self.criterion(outputs, labels)

                reg_loss = 0.0
                for p_name, p_param in personal_model.named_parameters():
                    if p_param.requires_grad and p_name in global_state:
                        reg_loss += torch.sum((p_param - global_state[p_name]) ** 2)
                reg_loss = 0.5 * self.lambda_reg * reg_loss

                total = cls_loss + reg_loss
                total.backward()
                optimizer.step()

                total_loss += total.item()
                batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

        return {
            "state_dict": personal_model.state_dict(),
            "loss": avg_loss,
        }

    def post_aggregate_hook(self, global_state: Dict[str, torch.Tensor]):
        self.global_state = global_state
