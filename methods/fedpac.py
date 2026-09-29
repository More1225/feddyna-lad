# coding: utf-8
"""
FedPAC: Federated Learning with Personalized Aggregation Coefficients
Train the classifier locally, aggregate the feature extractor globally, and use a prototype-alignment loss.
Persist the classifier across rounds and implement alternating training through train_alternating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any

from .base_method import BaseFLMethod
from .registry import register_method


@register_method("fedpac")
class FedPACMethod(BaseFLMethod):

    def __init__(self, args, model: nn.Module, device: torch.device):
        super().__init__(args, model, device)
        self.num_classes = getattr(args, 'num_classes', 2)
        self.lam = getattr(args, 'fedpac_lam', getattr(args, 'lam', 1.0))
        self.local_protos: Dict[int, Dict[int, torch.Tensor]] = {}
        self.global_protos: Dict[int, torch.Tensor] = {}
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

    def should_alternate_training(self) -> bool:
        return True

    def _extract_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(model, 'tcn'):
            tcn = model.tcn
            if hasattr(tcn, 'compress') and not isinstance(tcn.compress, nn.Identity):
                x = tcn.compress(x)
            if x.dim() == 3 and hasattr(tcn, 'blocks'):
                x = x.permute(0, 2, 1)
            if hasattr(tcn, 'blocks'):
                for block in tcn.blocks:
                    x = block(x)
            if hasattr(tcn, 'pool'):
                x = tcn.pool(x).squeeze(-1)
            return x
        return model(x)

    def _compute_local_prototypes(
        self, model: nn.Module, train_loader, device: torch.device
    ) -> Dict[int, torch.Tensor]:
        model.eval()
        class_features: Dict[int, list] = {c: [] for c in range(self.num_classes)}

        with torch.no_grad():
            for batch in train_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                features = self._extract_features(model, inputs)
                for c in range(self.num_classes):
                    mask = labels == c
                    if mask.any():
                        class_features[c].append(features[mask])

        prototypes = {}
        for c in range(self.num_classes):
            if class_features[c]:
                prototypes[c] = torch.cat(class_features[c], dim=0).mean(dim=0)
        return prototypes

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

        # Restore the classifier from the previous round.
        if client_id in self.client_classifier_states:
            saved_cls = self.client_classifier_states[client_id]
            model_state = model.state_dict()
            for key in self.classifier_keys:
                if key in saved_cls:
                    model_state[key] = saved_cls[key]
            model.load_state_dict(model_state)

        head_epochs = getattr(self.args, 'fedpac_head_epochs', 1)
        feature_epochs = getattr(self.args, 'fedpac_feature_epochs', 4)

        classifier_params = []
        feature_params = []
        for name, param in model.named_parameters():
            if name in self.classifier_keys:
                classifier_params.append(param)
            else:
                feature_params.append(param)

        epoch_losses = []
        epoch_accs = []

        # Phase 1: Train the classifier.
        cls_optimizer = self._make_optimizer(optimizer_class, classifier_params, lr, weight_decay)
        for param in feature_params:
            param.requires_grad = False
        for param in classifier_params:
            param.requires_grad = True

        for epoch in range(head_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            num_samples = 0
            batch_count = 0

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
                batch_count += 1
                _, predicted = outputs.max(1)
                num_samples += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_losses.append(total_loss / max(batch_count, 1))
            epoch_accs.append(100.0 * correct / max(num_samples, 1))

        # Phase 2: Train the feature extractor with prototype-alignment loss.
        feat_optimizer = self._make_optimizer(optimizer_class, feature_params, lr, weight_decay)
        for param in classifier_params:
            param.requires_grad = False
        for param in feature_params:
            param.requires_grad = True

        has_global_protos = bool(self.global_protos)

        for epoch in range(feature_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            num_samples = 0
            batch_count = 0

            for batch in train_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                feat_optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    _, outputs = outputs
                cls_loss = criterion(outputs, labels)

                proto_loss = torch.tensor(0.0, device=device)
                if has_global_protos:
                    features = self._extract_features(model, inputs)
                    for c in range(self.num_classes):
                        if c not in self.global_protos:
                            continue
                        mask = labels == c
                        if not mask.any():
                            continue
                        local_proto = features[mask].mean(dim=0)
                        global_proto = self.global_protos[c].to(device)
                        proto_loss += F.mse_loss(local_proto, global_proto)

                loss = cls_loss + self.lam * proto_loss
                loss.backward()
                feat_optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                _, predicted = outputs.max(1)
                num_samples += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_losses.append(total_loss / max(batch_count, 1))
            epoch_accs.append(100.0 * correct / max(num_samples, 1))

        for param in model.parameters():
            param.requires_grad = True

        # Save prototypes and classifier state.
        self.local_protos[client_id] = self._compute_local_prototypes(model, train_loader, device)
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

    def pre_aggregate_hook(self, client_states: Dict[int, Dict[str, torch.Tensor]]):
        class_prototypes: Dict[int, list] = {c: [] for c in range(self.num_classes)}
        for cid, prototypes in self.local_protos.items():
            for class_id, proto in prototypes.items():
                if isinstance(proto, torch.Tensor):
                    class_prototypes[class_id].append(proto.to(self.device))
        for class_id, proto_list in class_prototypes.items():
            if proto_list:
                self.global_protos[class_id] = torch.stack(proto_list).mean(dim=0)
