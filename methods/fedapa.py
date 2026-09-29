# coding: utf-8
"""
FedAPA: Server-side Gradient-Based Adaptive Personalized Aggregation

Core idea: maintain a learnable weight vector for each client and optimize the weight matrix with autograd gradients,
so each client trains from a personalized aggregated model rather than a single global model.

Algorithm workflow:
  1. Before each round, use the weight matrix to aggregate the previous-round client states into a personalized model for each client.
  2. Each client performs local training from its personalized model.
  3. After aggregation, update the weight matrix by backpropagating the MSE loss between the personalized aggregate and local update.
  4. Global-model aggregation remains FedAvg; personalization affects the training starting point, not the aggregation rule.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any

from .base_method import BaseFLMethod
from .registry import register_method


@register_method("fedapa")
class FedAPAMethod(BaseFLMethod):

    def __init__(self, args, model: nn.Module, device: torch.device):
        super().__init__(args, model, device)
        self.server_lr = getattr(args, 'fedapa_server_lr', 0.01)
        self.clamp_min = getattr(args, 'fedapa_clamp_min', 0.0)
        self.clamp_max = getattr(args, 'fedapa_clamp_max', 1.0)

        # Learnable weight matrix (N x N), initialized on the first call to update_all_weights.
        self.weight_param = None
        self.weight_optimizer = None
        self.client_ids = []
        self.client_idx = {}  # cid -> row index
        self.initialized = False

        # Cache previous-round client states for (1) personalized-model construction and (2) weight-gradient computation.
        self.prev_round_client_states = None

        # Store personalized model states after training for evaluation.
        self.personal_models: Dict[int, Dict[str, torch.Tensor]] = {}

    # ==================== Weight matrix initialization ====================

    def _initialize_weight_matrix(self, client_ids: list):
        self.client_ids = sorted(client_ids)
        self.client_idx = {cid: i for i, cid in enumerate(self.client_ids)}
        n = len(self.client_ids)

        # Initialize uniformly with slightly higher diagonal entries.
        initial = torch.full((n, n), 1.0 / n, device=self.device)
        for i in range(n):
            initial[i, i] = 1.5 / n

        self.weight_param = nn.Parameter(initial, requires_grad=True)
        self.weight_optimizer = torch.optim.SGD(
            [self.weight_param], lr=self.server_lr, momentum=0.9
        )
        self.initialized = True

    def _get_weight_vector(self, client_id: int) -> torch.Tensor:
        idx = self.client_idx[client_id]
        w = self.weight_param[idx].clone()
        w[idx] = 1.0  # Keep the self-weight fixed at 1.
        w = torch.clamp(w, self.clamp_min, self.clamp_max)
        w = w / (torch.sum(w) + 1e-8)
        return w

    # ==================== Alternating training (core) ====================

    def should_alternate_training(self) -> bool:
        return True

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

        # ---- Load the personalized model ----
        if (
            self.prev_round_client_states is not None
            and client_id in self.client_idx
        ):
            weight_vector = self._get_weight_vector(client_id).detach().cpu()
            personalized_state = {}
            for key in self.prev_round_client_states[self.client_ids[0]].keys():
                params = torch.stack(
                    [self.prev_round_client_states[cid][key] for cid in self.client_ids],
                    dim=0,
                )
                shape = [-1] + [1] * (len(params.shape) - 1)
                personalized_state[key] = torch.sum(
                    params * weight_vector.reshape(shape), dim=0
                )
            model.load_state_dict(personalized_state)
            # Statistics only: the server-produced state consumed by this client.
            if hasattr(self, "communication_tracker"):
                self.communication_tracker.record_personalized_state(
                    round_num, client_id, personalized_state,
                )
        # On the first round, no previous client state is available; use the global model provided by the framework.

        # ---- Standard local training ----
        model.train()
        model.to(device)

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

                if getattr(self.args, 'clipgradnorm', None):
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

        # Save the personalized model state after training.
        self.personal_models[client_id] = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

        return {
            "model": model,
            "train_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "train_acc": epoch_accs[-1] if epoch_accs else 0.0,
            "epoch_losses": epoch_losses,
            "epoch_accs": epoch_accs,
            "grad_norms": [],
        }

    # ==================== Weight-matrix gradient optimization ====================

    def update_all_weights(self, client_states: Dict[int, Dict[str, torch.Tensor]]):
        if not self.initialized:
            self._initialize_weight_matrix(sorted(client_states.keys()))

        # First round: cache states only; do not update gradients.
        if self.prev_round_client_states is None:
            self.prev_round_client_states = {
                cid: {k: v.clone() for k, v in state.items()}
                for cid, state in client_states.items()
            }
            return

        # Order matters: compute gradients using the old prev_round states before updating the cache.

        # Update the weight matrix with autograd for each participating client.
        for cid in self.client_ids:
            if cid not in client_states:
                continue

            self.weight_optimizer.zero_grad()

            idx = self.client_idx[cid]
            weight_vector = self.weight_param[idx]

            # Keep the self-weight fixed at 1.0 (detach the difference to preserve gradients for other weights).
            with torch.no_grad():
                w_clone = weight_vector.clone()
                w_clone[idx] = 1.0
            weight_vector_with_self = weight_vector + (w_clone - weight_vector).detach()

            # Clamp and normalize while preserving gradients.
            weight_clamped = torch.clamp(weight_vector_with_self, self.clamp_min, self.clamp_max)
            weight_normalized = weight_clamped / (torch.sum(weight_clamped) + 1e-8)

            # Build the personalized aggregate from previous-round client states while retaining the autograd graph.
            loss = torch.tensor(0.0, device=self.device)
            num_params = 0

            for key in self.prev_round_client_states[self.client_ids[0]].keys():
                if key not in client_states[cid]:
                    continue

                params = torch.stack(
                    [
                        self.prev_round_client_states[other_cid][key].to(self.device)
                        for other_cid in self.client_ids
                    ],
                    dim=0,
                )
                shape = [-1] + [1] * (len(params.shape) - 1)
                aggregated = torch.sum(
                    params * weight_normalized.reshape(shape), dim=0
                )

                diff = aggregated - client_states[cid][key].to(self.device)
                loss = loss + torch.sum(diff * diff)
                num_params += diff.numel()

            if num_params > 0:
                loss = loss / num_params
                loss.backward()
                self.weight_optimizer.step()

        # After the gradient update, refresh the cache for participating clients; retain prior states for clients that did not participate.
        for cid, state in client_states.items():
            self.prev_round_client_states[cid] = {
                k: v.clone() for k, v in state.items()
            }

    # ==================== Aggregation ====================

    def pre_aggregate_hook(self, client_states):
        if not self.initialized:
            self._initialize_weight_matrix(sorted(client_states.keys()))

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
            weighted_sum = torch.zeros_like(global_state[key])
            for cid, state in client_states.items():
                weighted_sum += client_weights[cid] / weight_sum * state[key]
            new_global_state[key] = weighted_sum

        return new_global_state

    # ==================== Personalization ====================

    def requires_personalized_inference(self) -> bool:
        return True

    def get_personalized_model(
        self,
        client_id: int,
        global_state: Dict[str, torch.Tensor],
        client_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if client_id in self.personal_models:
            return self.personal_models[client_id]
        return client_state if client_state is not None else global_state

    def train_personal_model(
        self,
        client_id: int,
        global_model: nn.Module,
        train_loader,
        device: torch.device,
        epochs: int = None,
    ) -> Dict:
        # Personalization is handled in train_alternating; no additional training is needed.
        return {"loss": 0.0}
