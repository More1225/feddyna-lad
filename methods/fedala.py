# coding: utf-8
"""
FedALA: Federated Adaptive Local Aggregation
Learn local aggregation weights α for each client and compute v_k = w_k + α * (w_t - w_k).
Start local training from the fused model. ALA is applied before training through train_alternating.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List

from .base_method import BaseFLMethod
from .registry import register_method


@register_method("fedala")
class FedALAMethod(BaseFLMethod):

    def __init__(self, args, model: nn.Module, device: torch.device):
        super().__init__(args, model, device)
        self.eta = getattr(args, 'fedala_eta', 0.1)
        self.rand_percent = getattr(args, 'fedala_rand_percent', 80)
        self.layer_idx = getattr(args, 'fedala_layer_idx', -1)
        self.threshold = getattr(args, 'fedala_threshold', 0.1)
        self.num_pre_loss = getattr(args, 'fedala_num_pre_loss', 5)
        self.adaptive_lr = getattr(args, 'fedala_adaptive_lr', 0.01)

        self.client_weights: Dict[int, List[torch.Tensor]] = {}
        self.criterion = nn.CrossEntropyLoss()
        self.client_local_states: Dict[int, Dict[str, torch.Tensor]] = {}

    def _init_weights(self, client_id: int, model: nn.Module):
        if client_id not in self.client_weights:
            self.client_weights[client_id] = [
                torch.ones_like(p.data).to(self.device) for p in model.parameters()
            ]

    def should_alternate_training(self) -> bool:
        return True

    def _apply_ala(
        self,
        client_id: int,
        global_model: nn.Module,
        local_model: nn.Module,
        train_loader: DataLoader,
    ):
        self._init_weights(client_id, local_model)

        params_g = list(global_model.parameters())
        params = list(local_model.parameters())
        weights = self.client_weights[client_id]

        total_diff = sum(
            torch.sum(torch.abs(pg.data - p.data)).item()
            for pg, p in zip(params_g, params)
        )
        if total_diff < 1e-10:
            return

        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        if self.layer_idx <= 0:
            layer_idx = len(params)
        else:
            layer_idx = min(self.layer_idx, len(params))
        if layer_idx == 0:
            return

        for i, (pt, p, pg, w) in enumerate(zip(params_t, params, params_g, weights)):
            if i >= len(params) - layer_idx:
                pt.data = p + (pg - p) * w.to(p.device)

        for i, pt in enumerate(params_t):
            if i < len(params_t) - layer_idx:
                pt.requires_grad = False

        optimizer = torch.optim.SGD(params_t[-layer_idx:], lr=self.adaptive_lr)

        all_data = list(train_loader)
        if not all_data:
            return

        total_samples = sum(len(b[0]) for b in all_data)
        rand_num = max(1, int(self.rand_percent / 100 * total_samples))

        x_all = torch.cat([b[0] for b in all_data], dim=0)[:rand_num]
        y_all = torch.cat([b[1] for b in all_data], dim=0)[:rand_num]

        losses = []
        for iteration in range(50):
            indices = torch.randperm(len(x_all))[:min(32, len(x_all))]
            x = x_all[indices].to(self.device)
            y = y_all[indices].to(self.device)

            optimizer.zero_grad()
            output = model_t(x)
            if isinstance(output, tuple):
                _, output = output
            loss = self.criterion(output, y)
            loss.backward()

            for pt, p, pg, w in zip(
                params_t[-layer_idx:], params[-layer_idx:],
                params_g[-layer_idx:], weights[-layer_idx:]
            ):
                if pt.grad is not None:
                    w_grad = pt.grad * (pg.to(p.device) - p)
                    w.data = torch.clamp(w.to(p.device) - self.adaptive_lr * w_grad, 0, 1).to(w.device)

            for pt, p, pg, w in zip(
                params_t[-layer_idx:], params[-layer_idx:],
                params_g[-layer_idx:], weights[-layer_idx:]
            ):
                pt.data = p + (pg.to(p.device) - p) * w.to(p.device)

            losses.append(loss.item())

            if len(losses) >= self.num_pre_loss:
                recent = losses[-self.num_pre_loss:]
                if len(recent) >= 2 and torch.std(torch.tensor(recent)).item() < self.threshold:
                    break

        for i, (p, pg, w) in enumerate(zip(params, params_g, weights)):
            if i >= len(params) - layer_idx:
                p.data = p + (pg - p) * w

    def train_alternating(
        self,
        model: nn.Module,
        train_loader,
        criterion,
        optimizer_class,
        lr: float,
        weight_decay: float,
        round_num: int = 0,
        client_id: int = -1,
        verbose: int = 1,
    ) -> Dict:
        device = self.device
        model.to(device)

        global_model_copy = copy.deepcopy(model)

        if client_id in self.client_local_states:
            model.load_state_dict(self.client_local_states[client_id])

        self._apply_ala(client_id, global_model_copy, model, train_loader)

        model.train()
        if optimizer_class == torch.optim.SGD:
            optimizer = optimizer_class(
                model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        else:
            optimizer = optimizer_class(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        local_epochs = getattr(self.args, 'local_epochs', 5)
        epoch_losses = []
        epoch_accs = []

        for epoch in range(local_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0

            for batch in train_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    _, outputs = outputs

                loss = criterion(outputs, labels)
                loss.backward()

                if getattr(self.args, "clipgradnorm", None):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.clipgradnorm
                    )

                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_losses.append(total_loss / max(batch_count, 1))
            epoch_accs.append(100.0 * correct / max(total, 1))

        self.client_local_states[client_id] = {
            k: v.clone() for k, v in model.state_dict().items()
        }

        return {
            "model": model,
            "train_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "train_acc": epoch_accs[-1] if epoch_accs else 0.0,
            "epoch_losses": epoch_losses,
            "epoch_accs": epoch_accs,
            "grad_norms": [],
        }

    def adaptive_local_aggregation(
        self,
        client_id: int,
        global_model: nn.Module,
        local_model: nn.Module,
        train_loader: DataLoader,
    ):
        pass  # ALA has already been applied in train_alternating.

    def requires_personalized_inference(self) -> bool:
        return True

    def get_personalized_model(
        self,
        client_id: int,
        global_state: Dict[str, torch.Tensor],
        client_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if client_id not in self.client_weights:
            return client_state

        weights = self.client_weights[client_id]
        personalized_state = {}

        param_idx = 0
        for key in global_state:
            global_tensor = global_state[key].to(self.device)
            client_tensor = client_state.get(key, global_tensor).to(self.device)

            personalized_state[key] = client_tensor
            if param_idx < len(weights):
                w = weights[param_idx].to(self.device)
                personalized_state[key] = client_tensor + w * (global_tensor - client_tensor)
                param_idx += 1

        return personalized_state
