"""
Local-training runner

Train each client independently without federated aggregation, then report mean test metrics.
"""

import copy
import time
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.train_utils import DatasetSplit, collate_fn_with_lengths
from federated.round_runner import evaluate_model
from trainers.standard_trainer import build_criterion, build_optimizer
from utils.debug_tools import log


def print_section(title: str, sep: str = "="):
    """Print a section heading."""
    print(f"\n{sep * 80}")
    print(f" {title}")
    print(f"{sep * 80}")


def format_mean_std(mean: float, std: float) -> str:
    """Format a mean ± standard deviation string."""
    return f"{mean:.4f} ± {std:.4f}"


def _collect_class_counts(train_loader: DataLoader) -> np.ndarray:
    labels = []
    for item in train_loader.dataset:
        labels.append(int(item[1]))
    return np.bincount(labels, minlength=2)


def train_local_client(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args,
    client_id: int,
) -> Dict[str, Any]:
    """
    Train one client locally.

    Args:
        model: model
        train_loader: training data loader
        val_loader: validation data loader
        test_loader: test data loader
        device: device
        args: argument configuration
        client_id: client ID

    Returns:
        Dictionary containing training results and metrics.
    """
    model = model.to(device)
    class_counts = _collect_class_counts(train_loader)
    criterion = build_criterion(class_counts, device, args)
    optimizer = build_optimizer(model, args)

    patience = getattr(args, 'local_patience', 5)
    num_epochs = args.local_epochs

    train_losses = []
    val_metrics_history = []

    best_val_auc = 0.5
    best_epoch = -1
    best_model_state = None
    epochs_no_improve = 0

    log(f"[Client {client_id}] starting training with {num_epochs} epochs and early-stopping patience {patience}", 1, args)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            # Handle batches that may include sequence lengths.
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if getattr(args, "clipgradnorm", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipgradnorm)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)

        # Evaluate on the validation set.
        val_metrics = evaluate_model(model, val_loader, device)
        val_metrics_history.append(val_metrics)

        current_val_auc = val_metrics["auroc"] if "auroc" in val_metrics else val_metrics.get("auc", 0.5)
        current_val_f1 = val_metrics["f1"]

        log(
            f"[Client {client_id}] Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {current_val_f1:.4f}, Val AUC: {current_val_auc:.4f}",
            1, args
        )

        # Check early stopping.
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            log(f"[Client {client_id}] Updated best model (Val AUC: {best_val_auc:.4f}, Val F1: {current_val_f1:.4f})", 1, args)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log(f"[Client {client_id}] Early stopping triggered at epoch {epoch+1}", 1, args)
                break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log(f"[Client {client_id}] Loading best model from epoch {best_epoch+1}best model at epoch (Val AUC: {best_val_auc:.4f})", 1, args)

    # Evaluate on the test set.
    test_metrics = evaluate_model(model, test_loader, device)

    return {
        "model_state": best_model_state if best_model_state else model.state_dict(),
        "train_losses": train_losses,
        "val_metrics_history": val_metrics_history,
        "test_metrics": test_metrics,
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else 0,
        "best_val_auc": best_val_auc,
        "best_val_f1": val_metrics_history[best_epoch]["f1"] if best_epoch >= 0 else 0.0,
    }


def run_local_training(
    model_template: nn.Module,
    dataset_train,
    dataset_val,
    dataset_test,
    dict_users_train: Dict[int, List[int]],
    dict_users_val: Dict[int, List[int]],
    dict_users_test: Dict[int, List[int]],
    args,
) -> Dict[str, Any]:
    """
    Run Local training: train each client independently.

    Args:
        model_template: model template
        dataset_train: training dataset
        dataset_val: validation dataset
        dataset_test: test dataset
        dict_users_train: client training indices
        dict_users_val: client validation indices
        dict_users_test: client test indices
        args: argument configuration

    Returns:
        Dictionary containing results for all clients.
    """
    client_ids = sorted(dict_users_train.keys())
    if args.debug_client_limit > 0:
        client_ids = client_ids[:args.debug_client_limit]

    device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    print_section("Local Training Start", "=")
    print(f"Number of clients: {len(client_ids)}")
    print(f"client ID: {client_ids}")
    print(f"Local-training epochs: {args.local_epochs}")
    print(f"Early-stopping patience: {getattr(args, 'local_patience', 5)}")
    print(f"Learning rate: {args.lr}")
    print(f"device: {device}")

    # Store results.
    client_results = {}
    final_local_state_dicts = {}

    for cid in client_ids:
        print_section(f"Client {cid} Training", "-")

        # Prepare data loaders.
        train_subset = DatasetSplit(dataset_train, dict_users_train[cid])
        val_subset = DatasetSplit(dataset_val, dict_users_val[cid])
        test_subset = DatasetSplit(dataset_test, dict_users_test[cid])

        train_loader = DataLoader(
            train_subset,
            batch_size=args.local_bs,
            shuffle=True,
            num_workers=getattr(args, 'num_workers', 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn_with_lengths,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.local_bs,
            shuffle=False,
            num_workers=getattr(args, 'num_workers', 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn_with_lengths,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=args.local_bs,
            shuffle=False,
            num_workers=getattr(args, 'num_workers', 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn_with_lengths,
        )

        log(f"[Client {cid}] Train samples: {len(train_subset)}, Validation samples: {len(val_subset)}, Test samples: {len(test_subset)}", 1, args)

        # Create a new model instance
        local_model = copy.deepcopy(model_template)

        # Training
        start_time = time.time()
        result = train_local_client(
            model=local_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            args=args,
            client_id=cid,
        )
        training_time = time.time() - start_time

        # Save results.
        client_results[cid] = {
            **result,
            "training_time": training_time,
            "train_samples": len(train_subset),
            "val_samples": len(val_subset),
            "test_samples": len(test_subset),
        }
        final_local_state_dicts[cid] = result["model_state"]

        # Print test results.
        test_metrics = result["test_metrics"]
        log(f"[Client {cid}] training completed in  {training_time:.2f}s", 1, args)
        log(
            f"[Client {cid}] Test metrics: Acc={test_metrics['accuracy']:.4f}, "
            f"Prec={test_metrics['precision']:.4f}, Rec={test_metrics['recall']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, PRAUC={test_metrics['prauc']:.4f}, "
            f"AUROC={test_metrics.get('auroc', 0.0):.4f}",
            1, args
        )

        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute summary statistics
    print_section("Local Training Summary", "=")

    # Compute total training time
    total_training_time = sum(client_results[cid]["training_time"] for cid in client_ids)
    avg_training_time = total_training_time / len(client_ids) if client_ids else 0

    all_test_metrics = {cid: client_results[cid]["test_metrics"] for cid in client_ids}

    metric_names = ["accuracy", "precision", "recall", "f1", "prauc", "auroc"]
    summary = {}

    for metric in metric_names:
        values = [all_test_metrics[cid].get(metric, 0.0) for cid in client_ids]
        summary[metric] = np.mean(values)
        summary[f"{metric}_std"] = np.std(values)
        summary[f"{metric}_max"] = np.max(values)
        summary[f"{metric}_min"] = np.min(values)

    # Add training time to the summary
    summary["total_training_time_sec"] = total_training_time
    summary["avg_training_time_sec"] = avg_training_time
    summary["total_training_time_min"] = total_training_time / 60.0
    summary["avg_training_time_min"] = avg_training_time / 60.0

    print("\n[local_avg] Mean client metrics:")
    print(f"  Accuracy:  {format_mean_std(summary['accuracy'], summary['accuracy_std'])}")
    print(f"  Precision: {format_mean_std(summary['precision'], summary['precision_std'])}")
    print(f"  Recall:    {format_mean_std(summary['recall'], summary['recall_std'])}")
    print(f"  F1:        {format_mean_std(summary['f1'], summary['f1_std'])}")
    print(f"  PRAUC:     {format_mean_std(summary['prauc'], summary['prauc_std'])}")
    print(f"  AUROC:     {format_mean_std(summary['auroc'], summary['auroc_std'])}")

    print(f"\n[local_time] Training time summary:")
    print(f"  Total training time: {summary['total_training_time_sec']:.2f} s ({summary['total_training_time_min']:.2f} min)")
    print(f"  Average training time: {summary['avg_training_time_sec']:.2f} s ({summary['avg_training_time_min']:.2f} min)")

    print(f"\n[local_per] Across-client F1 distribution:")
    print(f"  F1 Std:   {summary['f1_std']:.4f}")
    print(f"  Best F1:  {summary['f1_max']:.4f}")
    print(f"  Worst F1: {summary['f1_min']:.4f}")
    print(f"  Range:    {summary['f1_max'] - summary['f1_min']:.4f}")

    # Print the detailed results table.
    print("\nDetailed test metrics by client:")
    print("-" * 120)
    print(f"{'Client':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'PRAUC':<12} {'AUROC':<12}")
    print("-" * 120)
    for cid in client_ids:
        r = client_results[cid]
        m = r["test_metrics"]
        print(
            f"{cid:<10} {r['train_samples']:<10} {r['val_samples']:<10} {r['test_samples']:<10} "
            f"{m['accuracy']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f} "
            f"{m['f1']:<12.4f} {m['prauc']:<12.4f} {m.get('auroc', 0.0):<12.4f}"
        )
    print("-" * 120)

    return {
        "client_results": client_results,
        "final_local_state_dicts": final_local_state_dicts,
        "test_metrics": all_test_metrics,
        "summary": summary,
    }
