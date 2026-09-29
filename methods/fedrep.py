# coding: utf-8
"""
FedRep: Federated Learning with Representation Learning
Separate representation learning (global aggregation) from classifier learning (local training).
Persist the classifier across rounds; use fedrep_head_epochs / fedrep_rep_epochs to configure the epoch allocation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any

from .base_method import BaseFLMethod
from .registry import register_method


@register_method("fedrep")
class FedRepMethod(BaseFLMethod):

    def __init__(self, args, model: nn.Module, device: torch.device):
        super().__init__(args, model, device)
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

    def should_alternate_training(self) -> bool:
        return True

    def _make_optimizer(self, optimizer_class, params, lr, weight_decay):
        if optimizer_class == torch.optim.SGD:
            return optimizer_class(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        return optimizer_class(params, lr=lr, weight_decay=weight_decay)

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
        model.train()
        model.to(device)

        if client_id in self.client_classifier_states:
            saved_cls = self.client_classifier_states[client_id]
            model_state = model.state_dict()
            for key in self.classifier_keys:
                if key in saved_cls:
                    model_state[key] = saved_cls[key]
            model.load_state_dict(model_state)

        classifier_epochs = getattr(self.args, 'fedrep_head_epochs', 1)
        feature_epochs = getattr(self.args, 'fedrep_rep_epochs', 4)

        classifier_params = []
        feature_params = []
        for name, param in model.named_parameters():
            if name in self.classifier_keys:
                classifier_params.append(param)
            else:
                feature_params.append(param)

        epoch_losses = []
        epoch_accs = []

        # Phase 1: Train the classifier with the feature extractor frozen.
        cls_optimizer = self._make_optimizer(optimizer_class, classifier_params, lr, weight_decay)
        for param in feature_params:
            param.requires_grad = False
        for param in classifier_params:
            param.requires_grad = True

        for epoch in range(classifier_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                cls_optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    _, outputs = outputs

                loss = criterion(outputs, labels)
                loss.backward()
                cls_optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_losses.append(total_loss / len(train_loader))
            epoch_accs.append(100. * correct / total)

        # Phase 2: Train the feature extractor with the classifier frozen.
        feat_optimizer = self._make_optimizer(optimizer_class, feature_params, lr, weight_decay)
        for param in classifier_params:
            param.requires_grad = False
        for param in feature_params:
            param.requires_grad = True

        for epoch in range(feature_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                feat_optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    _, outputs = outputs

                loss = criterion(outputs, labels)
                loss.backward()
                feat_optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_losses.append(total_loss / len(train_loader))
            epoch_accs.append(100. * correct / total)

        for param in model.parameters():
            param.requires_grad = True

        # Save the classifier state.
        current_state = model.state_dict()
        self.client_classifier_states[client_id] = {
            k: v.clone() for k, v in current_state.items() if k in self.classifier_keys
        }

        return {
            "model": model,
            "train_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "train_acc": epoch_accs[-1] if epoch_accs else 0.0,
            "epoch_losses": epoch_losses,
            "epoch_accs": epoch_accs,
            "grad_norms": [],
        }
