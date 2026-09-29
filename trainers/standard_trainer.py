from __future__ import annotations

from typing import Any, Optional
from collections import OrderedDict

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.debug_tools import log, print_section, compute_grad_norm, save_client_training_plot


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


def get_optimizer_class(args: Any):
    """Get the optimizer class"""
    opt_name = args.optimizer.lower()
    if opt_name == "adam":
        return torch.optim.Adam
    if opt_name == "adamw":
        return torch.optim.AdamW
    if opt_name == "sgd":
        return torch.optim.SGD
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_optimizer(model: nn.Module, args: Any) -> torch.optim.Optimizer:
    opt_name = args.optimizer.lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_criterion(class_counts: np.ndarray, device: torch.device, args: Any) -> nn.Module:
    if args.loss_mode == "ce":
        return nn.CrossEntropyLoss()

    if args.loss_mode == "weighted_ce":
        if class_counts[1] == 0:
            return nn.CrossEntropyLoss()

        if getattr(args, "use_auto_class_weight", True):
            pos_w = class_counts[0] / max(class_counts[1], 1)
            pos_w = pos_w * getattr(args, "pos_weight_scale", 1.0)
            pos_w = min(pos_w, getattr(args, "max_pos_weight", 10.0))
        else:
            pos_w = getattr(args, "fixed_pos_weight", 8.0)

        weights = torch.tensor(
            [1.0, pos_w],
            dtype=torch.float,
            device=device
        )
        weights = weights / weights.mean()
        return nn.CrossEntropyLoss(weight=weights)


    if args.loss_mode == "focal":
        if class_counts[1] == 0:
            return nn.CrossEntropyLoss()
        alpha = torch.tensor(
            [1.0, class_counts[0] / max(class_counts[1], 1)],
            dtype=torch.float,
            device=device
        )
        alpha = alpha / alpha.mean()
        return FocalLoss(gamma=2.0, alpha=alpha)

    raise ValueError(f"Unsupported loss_mode: {args.loss_mode}")


def summarize_mask(mask: OrderedDict) -> dict[str, Any]:
    total_params = 0
    global_params = 0
    local_params = 0

    for _, mask_param in mask.items():
        total = mask_param.numel()
        ones = int(mask_param.sum().item())
        zeros = total - ones
        total_params += total
        global_params += zeros
        local_params += ones

    return {
        "total_params": total_params,
        "global_params": global_params,
        "local_params": local_params,
        "global_ratio": global_params / total_params if total_params > 0 else 0.0,
        "local_ratio": local_params / total_params if total_params > 0 else 0.0,
    }


def evaluate_on_loader(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            # Handle both (inputs, labels) and (inputs, labels, lengths)
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs)
            loss = ce(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += float(loss.item()) * labels.size(0)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    return {
        "acc": correct / total if total > 0 else 0.0,
        "loss": total_loss / total if total > 0 else 0.0,
    }


def train_local_standard(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    mask: OrderedDict,
    args: Any,
    client_id: int = -1,
    round_num: int = -1,
    initialization: Optional[OrderedDict] = None,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    fl_method: Any = None,  # Federated-learning method instance
) -> dict[str, Any]:
    if initialization is not None:
        model.load_state_dict(initialization)

    device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = model.to(device)
    model.train()

    dataset = train_loader.dataset
    # Handle datasets that return (feature, label, length) or (feature, label)
    if hasattr(dataset, "labels"):
        all_labels = dataset.labels
    else:
        all_labels = []
        for item in dataset:
            # item can be (feature, label, length) or (feature, label)
            if len(item) >= 2:
                all_labels.append(item[1])
            else:
                all_labels.append(item[-1])
    class_counts = np.bincount(all_labels, minlength=2)

    # Check whether alternating training is required (e.g., FedRep or FedPAC).
    if fl_method is not None and hasattr(fl_method, 'should_alternate_training') and fl_method.should_alternate_training():
        # Use alternating training.
        optimizer_class = get_optimizer_class(args)

        # Get the method name for logging.
        method_name = type(fl_method).__name__.replace('Method', '')

        alternating_result = fl_method.train_alternating(
            model=model,
            train_loader=train_loader,
            criterion=build_criterion(class_counts, device, args),
            optimizer_class=optimizer_class,
            lr=args.lr,
            weight_decay=args.weight_decay,
            round_num=round_num,
            client_id=client_id,
            verbose=getattr(args, 'verbose', 1),
        )
        # Populate the return values.
        epoch_losses = alternating_result.get("epoch_losses", [alternating_result.get("train_loss", 0.0)])
        epoch_accs = alternating_result.get("epoch_accs", [alternating_result.get("train_acc", 0.0)])
        grad_norms = alternating_result.get("grad_norms", [])

        # Evaluate after training.
        model.eval()
        train_eval = evaluate_on_loader(model, train_loader, device)

        # Prediction statistics.
        pred_stats = {}
        with torch.no_grad():
            for batch in train_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                probs = F.softmax(outputs, dim=1)
                pred_stats = {
                    "pred_distribution": torch.bincount(preds).cpu().tolist(),
                    "prob_std_class1": float(probs.std(dim=0)[1].cpu().item()) if probs.shape[1] > 1 else 0.0,
                }
                break

        # Plot output path.
        plot_path = None
        if getattr(args, "save_plots", False):
            plot_dir = os.path.join(args.plot_dir, f"round_{round_num}")
            plot_path = save_client_training_plot(
                round_num=round_num,
                client_id=client_id,
                epoch_losses=epoch_losses,
                epoch_accs=epoch_accs,
                grad_norms=grad_norms,
                save_dir=plot_dir,
            )
            log(f"[Round {round_num}] [Client {client_id}] Plot Saved: {plot_path}", 1, args)

        log(
            f"[Round {round_num}] [Client {client_id}] {method_name} Alternating Training Finished "
            f"| loss={train_eval['loss']:.4f} | acc={train_eval['acc']:.4f} "
            f"| pred_dist={pred_stats.get('pred_distribution', [])}",
            1,
            args,
        )

        return {
            "model": model,
            "train_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "epoch_losses": epoch_losses,
            "epoch_accs": epoch_accs,
            "grad_norms": grad_norms,
            "data_stats": {
                "sample_count": len(dataset),
                "class_counts": class_counts.tolist(),
                "normal_count": int(class_counts[0]),
                "anomaly_count": int(class_counts[1]),
                "anomaly_ratio": float(class_counts[1] / max(class_counts.sum(), 1)),
            },
            "train_eval": train_eval,
            "plot_path": plot_path,
            "client_id": client_id,
            "round_num": round_num,
        }

    criterion = build_criterion(class_counts, device, args)
    optimizer = build_optimizer(model, args)
    mask_stats = summarize_mask(mask)

    if getattr(args, "verbose", 1) >= 1:
        print_section(f"[Round {round_num}] [Client {client_id}] Local Training Start", "-")
        log(f"Device                : {device}", 1, args)
        log(f"Optimizer             : {args.optimizer}", 1, args)
        log(f"Loss Mode             : {args.loss_mode}", 1, args)
        log(f"Local Epochs          : {args.local_epochs}", 1, args)
        log(f"Learning Rate         : {args.lr}", 1, args)
        log(f"Weight Decay          : {args.weight_decay}", 1, args)
        log(f"Samples               : {len(dataset)}", 1, args)
        log(f"Class Counts          : normal={int(class_counts[0])}, anomaly={int(class_counts[1])}", 1, args)
        log(f"Anomaly Ratio         : {class_counts[1] / max(class_counts.sum(), 1):.4f}", 1, args)
        log(
            f"Mask Ratio            : global={mask_stats['global_ratio']:.4f}, local={mask_stats['local_ratio']:.4f}",
            1,
            args,
        )

    epoch_losses = []
    epoch_accs = []
    grad_norms = []
    val_losses = [] if val_loader is not None else None

    # Early stopping configuration
    early_stopping_enabled = val_loader is not None
    early_stopping_patience = getattr(args, "early_stopping_patience", 5)
    early_stopping_delta = getattr(args, "early_stopping_delta", 1e-4)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    best_epoch = -1

    use_tqdm = getattr(args, "verbose", 1) >= 1
    pbar = tqdm(
        total=args.local_epochs,
        disable=not use_tqdm,
        ncols=110,
        desc=f"Round {round_num} Client {client_id}"
    )

    if early_stopping_enabled:
        log(f"[Round {round_num}] [Client {client_id}] Early stopping enabled with patience={early_stopping_patience}, delta={early_stopping_delta}", 1, args)

    for epoch in range(args.local_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        correct = 0
        total = 0
        epoch_grad_norms = []
        current_val_loss = 0.0  # Initialize for each epoch

        for batch_idx, batch in enumerate(train_loader):
            # Handle both (inputs, labels) and (inputs, labels, lengths)
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # If a specific federated-learning method is active, call its loss computation.
            if fl_method is not None:
                loss = fl_method.compute_loss(
                    model=model,
                    output=outputs,
                    target=labels,
                    criterion=criterion,
                    client_id=client_id,
                    inputs=inputs,  # Pass inputs to methods that require feature extraction.
                )
            else:
                loss = criterion(outputs, labels)

            loss.backward()

            grad_norm = compute_grad_norm(model)
            epoch_grad_norms.append(grad_norm)

            if getattr(args, "clipgradnorm", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipgradnorm)

            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

            running_loss += float(loss.item())
            batch_count += 1

            if getattr(args, "verbose", 1) >= 2 and ((batch_idx + 1) % args.log_batch_interval == 0 or (batch_idx + 1) == len(train_loader)):
                tqdm.write(
                    f"[Round {round_num}] [Client {client_id}] "
                    f"Epoch {epoch+1}/{args.local_epochs} "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} | grad_norm={grad_norm:.4f}"
                )

        epoch_loss = running_loss / max(batch_count, 1)
        epoch_acc = correct / max(total, 1)
        avg_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)
        grad_norms.append(avg_grad_norm)

        # Validation evaluation for early stopping
        if early_stopping_enabled:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    # Handle both (inputs, labels) and (inputs, labels, lengths)
                    val_inputs = batch[0].to(device)
                    val_labels = batch[1].to(device)
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_labels).item() * val_labels.size(0)
                    val_count += val_labels.size(0)
            current_val_loss = val_loss / max(val_count, 1)
            if val_losses is not None:
                val_losses.append(current_val_loss)

            # Early stopping check
            if current_val_loss < best_val_loss - early_stopping_delta:
                best_val_loss = current_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
                log(f"[Round {round_num}] [Client {client_id}] Epoch {epoch+1}: Validation loss improved to {current_val_loss:.4f}", 1, args)
            else:
                epochs_no_improve += 1
                log(f"[Round {round_num}] [Client {client_id}] Epoch {epoch+1}: Validation loss {current_val_loss:.4f} (no improvement for {epochs_no_improve}/{early_stopping_patience})", 1, args)

            if epochs_no_improve >= early_stopping_patience:
                log(f"[Round {round_num}] [Client {client_id}] Early stopping triggered at epoch {epoch+1}", 1, args)
                break
        # If no validation set, early stopping is disabled and all epochs will run

        if use_tqdm:
            pbar.update(1)
            postfix_dict = {
                "Loss": f"{epoch_loss:.4f}",
                "Acc": f"{epoch_acc:.4f}",
                "Grad": f"{avg_grad_norm:.4f}",
            }
            if early_stopping_enabled:
                postfix_dict["ValLoss"] = f"{current_val_loss:.4f}"
            pbar.set_postfix(postfix_dict)

    if use_tqdm:
        pbar.close()

    # Load best model if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log(f"[Round {round_num}] [Client {client_id}] Loaded best model from epoch {best_epoch+1} (val_loss={best_val_loss:.4f})", 1, args)

    pred_stats = {}
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            # Handle both (inputs, labels) and (inputs, labels, lengths)
            inputs = batch[0].to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            probs = F.softmax(outputs, dim=1)
            pred_stats = {
                "pred_distribution": torch.bincount(preds).cpu().tolist(),
                "prob_std_class1": float(probs.std(dim=0)[1].cpu().item()) if probs.shape[1] > 1 else 0.0,
            }
            break

    train_eval = evaluate_on_loader(model, train_loader, device)

    plot_path = None
    if getattr(args, "save_plots", False):
        plot_dir = os.path.join(args.plot_dir, f"round_{round_num}")
        plot_path = save_client_training_plot(
            round_num=round_num,
            client_id=client_id,
            epoch_losses=epoch_losses,
            epoch_accs=epoch_accs,
            grad_norms=grad_norms,
            save_dir=plot_dir,
        )
        log(f"[Round {round_num}] [Client {client_id}] Plot Saved: {plot_path}", 1, args)

    log(
        f"[Round {round_num}] [Client {client_id}] Final Train Eval "
        f"| loss={train_eval['loss']:.4f} | acc={train_eval['acc']:.4f} "
        f"| pred_dist={pred_stats.get('pred_distribution', [])}",
        1,
        args,
    )

    return {
        "model": model,
        "train_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "epoch_losses": epoch_losses,
        "epoch_accs": epoch_accs,
        "grad_norms": grad_norms,
        "data_stats": {
            "sample_count": len(dataset),  # type: ignore
            "class_counts": class_counts.tolist(),
            "normal_count": int(class_counts[0]),
            "anomaly_count": int(class_counts[1]),
            "anomaly_ratio": float(class_counts[1] / max(class_counts.sum(), 1)),
        },
        "mask_stats": mask_stats,
        "pred_stats": pred_stats,
        "train_eval": train_eval,
        "plot_path": plot_path,
        "client_id": client_id,
        "round_num": round_num,
    }