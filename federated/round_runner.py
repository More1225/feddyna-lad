import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data.train_utils import prepare_dataloaders
from federated.aggregation_ops import (
    add_masks_weighted,
    add_server_weights_weighted,
    div_server_weights,
    init_server_like,
)
from federated.broadcast import broadcast_server_to_client_initialization
from federated.mask_manager import MaskManager
from federated.weighting import compute_client_weights
from personalization.controller import PersonalizationController
from personalization.update_metrics import (
    build_scope_aware_client_updates,
    compute_scope_aware_update_metrics,
)
from trainers.standard_trainer import summarize_mask, train_local_standard
from utils.debug_tools import log, print_section, print_state_summary

# Auxiliary one-time aggregation-scope selector
from federated.heterogeneity_aware import (
    flatten_model_update,
    compute_pairwise_cosine_stats,
    compute_heterogeneity_score,
    select_scope,
    append_ha_selection_log,
    append_ha_parameter_update_stats,
    append_ha_scope_history,
    print_ha_selection,
    print_ha_scope_round,
    save_ha_round_analysis,
)

# Import federated-learning method modules
from methods import get_method, MethodRegistry


from federated.communication import CommunicationTracker


def format_mean_std(mean, std):
    """
    Format the mean ± standard deviation string.
    """
    return f"{mean:.4f} ± {std:.4f}"


def compute_metrics_statistics(metrics_dict):
    """
    Compute summary statistics (mean, standard deviation, minimum, and maximum) for a set of metrics.

    Args:
        metrics_dict: Metric mapping in the form {client_id: metrics_dict}

    Returns:
        dict: Mapping containing summary statistics for each metric.
    """
    if not metrics_dict:
        return {}

    metric_names = ["accuracy", "precision", "recall", "f1", "prauc", "auroc"]
    stats = {}

    for metric in metric_names:
        values = [m.get(metric, 0.0) for m in metrics_dict.values() if m is not None]
        if values:
            stats[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "max": np.max(values),
                "min": np.min(values),
            }
        else:
            stats[metric] = {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

    return stats


def evaluate_model(model, data_loader, device):
    """
    Evaluate model performance.

    Metrics: accuracy, precision, recall, f1, prauc, auroc
    """
    model.eval()
    preds_all, labels_all, probs_all = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            # Handle both (inputs, labels) and (inputs, labels, lengths)
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())

    if not labels_all:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "prauc": 0.0,
            "auroc": 0.0,
        }

    # Compute PR-AUC (Average Precision).
    prauc = average_precision_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0

    # Compute ROC-AUC.
    auroc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0

    return {
        "accuracy": accuracy_score(labels_all, preds_all),
        "precision": precision_score(labels_all, preds_all, zero_division=0.0),
        "recall": recall_score(labels_all, preds_all, zero_division=0.0),
        "f1": f1_score(labels_all, preds_all, zero_division=0.0),
        "prauc": prauc,
        "auroc": auroc,
    }


def build_client_loaders(dataset_train, dataset_val, user_train_idx, user_val_idx, args):
    train_loader, val_loader = prepare_dataloaders(
        dataset_train=dataset_train,
        dict_users_train=user_train_idx,
        dataset_test=dataset_val,
        dict_users_test=user_val_idx,
        args=args,
    )
    return train_loader, val_loader


def evaluate_global_after_aggregation(
    model_template,
    global_state,
    dataset_train,
    dataset_val,
    dict_users_train,
    dict_users_val,
    client_ids,
    args,
    fl_method=None,
    client_states=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    post_global_metrics = {}
    for cid in client_ids:
        _, val_loader = build_client_loaders(
            dataset_train,
            dataset_val,
            dict_users_train[cid],
            dict_users_val[cid],
            args,
        )

        eval_global_model = copy.deepcopy(model_template).to(device)

        # Use the personalized model for evaluation when the method supports it.
        if fl_method is not None and fl_method.requires_personalized_inference() and client_states is not None:
            personal_state = fl_method.get_personalized_model(
                cid, global_state, client_states.get(cid, global_state)
            )
            eval_global_model.load_state_dict(personal_state)

            # Diagnostic: compare personalized and global models.
            if args.verbose >= 2:
                try:
                    from methods.debug_utils import PersonalizedMethodDebugger

                    # Create a global model for comparison.
                    global_model_for_comparison = copy.deepcopy(model_template).to(device)
                    global_model_for_comparison.load_state_dict(global_state)

                    # Compute model differences.
                    debugger = PersonalizedMethodDebugger()
                    method_name = getattr(args, 'fl_method', 'unknown')
                    stats = debugger.compare_models(
                        global_state,
                        personal_state,
                        method_name,
                        cid,
                        round_num
                    )

                    # Report key differences.
                    if stats.get('num_params', 0) > 0 and cid == 0:  # report only for the first client
                        diff_ratio = stats.get('num_diff_params', 0) / stats['num_params']
                        if diff_ratio > 0.1:  # report only when the difference is substantial
                            log(f"[Debug] Client {cid} personalization: {diff_ratio:.1%} params changed", 2, args)
                            log(f"[Debug] Avg param change: {stats.get('avg_diff', 0):.6f}", 2, args)
                except ImportError:
                    pass
                except Exception as e:
                    log(f"[Warning] Debug comparison failed: {e}", 1, args)
        else:
            eval_global_model.load_state_dict(global_state)

        post_global_metrics[cid] = evaluate_model(eval_global_model, val_loader, device)

    return post_global_metrics


def summarize_global_metrics(post_global_metrics):
    """
    Summarize global metrics by computing the mean and standard deviation.

    Returns:
        dict containing the mean and standard deviation for each metric
    """
    client_ids = sorted(post_global_metrics.keys())
    if not client_ids:
        return {}

    metric_names = ["accuracy", "precision", "recall", "f1", "prauc", "auroc"]
    result = {}

    for metric in metric_names:
        values = [post_global_metrics[cid].get(metric, 0.0) for cid in client_ids]
        result[metric] = np.mean(values)
        result[f"{metric}_std"] = np.std(values)

    return result


def print_round_summary(round_num, client_local_metrics, client_loss_storage, comm_stats=None):
    """
    Print the per-round training summary as mean ± standard deviation.

    Args:
        round_num: current round
        client_local_metrics: client-local metric mapping
        client_loss_storage: client-loss mapping
        comm_stats: optional communication statistics
    """
    print_section(f"Round {round_num} Summary", "-")
    print(f"{'Client':<10} {'Loss':<12} {'Acc':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'PRAUC':<10} {'AUROC':<10}")
    print(f"{'-' * 100}")

    for cid in sorted(client_local_metrics.keys()):
        metrics = client_local_metrics[cid]
        loss_val = client_loss_storage[cid]
        print(
            f"Client {cid:<5} {loss_val:<12.4f} {metrics['accuracy']:<10.4f} "
            f"{metrics['precision']:<12.4f} {metrics['recall']:<10.4f} "
            f"{metrics['f1']:<10.4f} {metrics['prauc']:<10.4f} {metrics.get('auroc', 0.0):<10.4f}"
        )

    print(f"{'-' * 100}")

    # Compute summary statistics.
    losses = [client_loss_storage[cid] for cid in client_local_metrics]
    accs = [metrics['accuracy'] for metrics in client_local_metrics.values()]
    precs = [metrics['precision'] for metrics in client_local_metrics.values()]
    recalls = [metrics['recall'] for metrics in client_local_metrics.values()]
    f1s = [metrics['f1'] for metrics in client_local_metrics.values()]
    praucs = [metrics['prauc'] for metrics in client_local_metrics.values()]
    aurocs = [metrics.get('auroc', 0.0) for metrics in client_local_metrics.values()]

    print(f"\n  Metric summary (mean ± standard deviation):")
    print(f"    Loss:     {format_mean_std(np.mean(losses), np.std(losses))}")
    print(f"    Accuracy: {format_mean_std(np.mean(accs), np.std(accs))}")
    print(f"    Precision:{format_mean_std(np.mean(precs), np.std(precs))}")
    print(f"    Recall:   {format_mean_std(np.mean(recalls), np.std(recalls))}")
    print(f"    F1:       {format_mean_std(np.mean(f1s), np.std(f1s))}")
    print(f"    PRAUC:    {format_mean_std(np.mean(praucs), np.std(praucs))}")
    print(f"    AUROC:    {format_mean_std(np.mean(aurocs), np.std(aurocs))}")

    # Print communication costs.
    if comm_stats:
        print(f"\n  Communication costs:")
        print(f"    Round uplink: {comm_stats['uplink_mb']:.2f} MB")
        print(f"    Round downlink: {comm_stats['downlink_mb']:.2f} MB")
        print(f"    Round total: {comm_stats['total_mb']:.2f} MB")
        print(f"    Per-client communication: {comm_stats['avg_uplink_per_client_mb']:.2f} MB (uplink), "
              f"{comm_stats['avg_downlink_per_client_mb']:.2f} MB (downlink)")


def aggregate_baseline_path(
    fl_method,
    fl_method_name,
    client_results,
    client_ids,
    client_weights,
    global_state,
    round_num,
    args,
):
    client_states = {cid: client_results[cid]["state_dict"] for cid in client_ids}

    if hasattr(fl_method, 'update_all_weights'):
        fl_method.update_all_weights(client_states)

    fl_method.pre_aggregate_hook(client_states)
    new_global_state = fl_method.aggregate(
        client_states=client_states,
        client_weights=client_weights,
        global_state=global_state,
    )
    fl_method.post_aggregate_hook(new_global_state)

    log(f"[Round {round_num}] Aggregation completed using {fl_method_name}", 1, args)
    return new_global_state


def aggregate_feddyna_path(
    client_results,
    client_ids,
    client_weights,
    client_masks,
    global_state,
    current_aggregation_scope,
    args,
):
    print(
        f"[Aggregation] weighting_method={args.weighting_method}, "
        f"current_scope={current_aggregation_scope}"
    )

    server_state, server_mask = init_server_like(global_state)
    for cid in client_ids:
        weight = client_weights[cid]
        server_mask = add_masks_weighted(
            server_mask,
            client_masks[cid],
            weight,
            aggregation_scope=current_aggregation_scope,
        )
        server_state = add_server_weights_weighted(
            server_state,
            client_results[cid]["state_dict"],
            client_masks[cid],
            weight,
            aggregation_scope=current_aggregation_scope,
        )

    new_global_state, zero_denom_ratio = div_server_weights(
        server_state,
        server_mask,
        prev_global_state=global_state,
    )

    if current_aggregation_scope == "masked_only":
        print(
            f"[Aggregation][masked_only] zero_denominator_ratio="
            f"{zero_denom_ratio:.4f}"
        )

    return new_global_state, zero_denom_ratio


def resolve_feddyna_scope_for_round(
    ha_applies_this_round,
    is_parameter_update_round,
    ha_selected_scope,
    ha_H_select,
    round_num,
    client_ids,
    client_results,
    global_state,
    args,
    client_updates=None,
):
    current_aggregation_scope = (
        ha_selected_scope
        if ha_selected_scope is not None
        else args.aggregation_scope
    )
    selection_log = None

    if not ha_applies_this_round or not is_parameter_update_round:
        if (
            ha_applies_this_round
            and ha_selected_scope is not None
            and getattr(args, "ha_debug", False)
        ):
            print_ha_scope_round(round_num, ha_selected_scope)
        return current_aggregation_scope, ha_selected_scope, ha_H_select, selection_log

    client_update_norms = {}
    if client_updates is None:
        client_updates = {}
        for cid in client_ids:
            client_updates[cid] = flatten_model_update(
                client_results[cid]["state_dict"],
                global_state,
            )
    for cid in client_ids:
        client_update_norms[cid] = client_updates[cid].norm(2).item()

    stats_dict, pairwise_records = compute_pairwise_cosine_stats(
        client_updates,
        client_ids=client_ids,
        quantile=args.ha_quantile,
    )
    agreement_score, current_H_select = compute_heterogeneity_score(
        stats_dict,
        mode=args.ha_scope_mode,
        lambda_std=args.ha_lambda_std,
        lambda_q=args.ha_lambda_q,
        lambda_neg=args.ha_lambda_neg,
    )
    candidate_scope = select_scope(current_H_select, args.ha_select_threshold)

    is_lock_round = ha_selected_scope is None

    if is_lock_round:
        ha_selected_scope = candidate_scope
        ha_H_select = current_H_select
        current_aggregation_scope = ha_selected_scope
        print_ha_selection(
            round_num, stats_dict, current_H_select,
            agreement_score, ha_selected_scope, args,
        )
    else:
        current_aggregation_scope = ha_selected_scope
        print(
            f"[FedDyna-HA][Round {round_num}] DIAGNOSTIC: "
            f"candidate_scope={candidate_scope}, "
            f"locked_scope={ha_selected_scope}, "
            f"H_select={current_H_select:.4f}, decision_applied=False"
        )

    selection_log = {
        "stats_dict": stats_dict,
        "pairwise_records": pairwise_records,
        "agreement_score": agreement_score,
        "client_update_norms": client_update_norms,
        "current_H_select": current_H_select,
        "candidate_scope": candidate_scope,
        "locked_scope": ha_selected_scope,
        "is_lock_round": is_lock_round,
        "decision_applied": is_lock_round,
    }

    return current_aggregation_scope, ha_selected_scope, ha_H_select, selection_log


def run_federated_training(model, dataset_train, dataset_val, dict_users_train, dict_users_val, args):
    client_ids = sorted(dict_users_train.keys())
    if args.debug_client_limit > 0:
        client_ids = client_ids[:args.debug_client_limit]

    client_init_state_dicts = {cid: copy.deepcopy(model.state_dict()) for cid in client_ids}
    final_local_state_dicts = {cid: copy.deepcopy(model.state_dict()) for cid in client_ids}

    # Initialize best-model tracking.
    best_client_local_f1 = {cid: -1.0 for cid in client_ids}
    best_client_local_state = {cid: None for cid in client_ids}
    best_client_round = {cid: -1 for cid in client_ids}

    mask_manager = MaskManager(model, args)
    client_masks = mask_manager.init_client_masks(client_ids)

    personalization = PersonalizationController(client_ids, args)

    global_state = copy.deepcopy(model.state_dict())
    prev_global_model = copy.deepcopy(model)
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    history = []

    # Initialize communication-cost tracking.
    comm_tracker = CommunicationTracker(args, num_clients=len(dict_users_train))
    comm_tracker.start_training()  # Start timing.

    # Initialize the federated-learning method.
    fl_method = None
    fl_method_name = getattr(args, 'fl_method', 'fedavg')
    comm_tracker.record_initialization(global_state, client_ids)
    if fl_method_name != 'fedavg':
        try:
            fl_method = get_method(fl_method_name, args, model, device)
            fl_method.communication_tracker = comm_tracker
            fl_method.set_global_state(global_state)
            log(f"FL Method initialized: {fl_method_name}", 1, args)

            # Sync initial masks with FL method if supported
            if hasattr(fl_method, 'set_client_mask'):
                for cid in client_ids:
                    fl_method.set_client_mask(cid, client_masks[cid])
        except Exception as e:
            log(f"Warning: Failed to initialize FL method '{fl_method_name}': {e}. Using FedAvg.", 1, args)
            fl_method = None

    print_section("Federated Training Start", "=")
    log(f"Client IDs             : {client_ids}", 1, args)
    log(f"Communication Rounds   : {args.com_rounds}", 1, args)
    log(f"Weighting Method       : {args.weighting_method}", 1, args)
    log(f"Pruning Method         : {args.pruning_method}", 1, args)
    log(f"Local Epochs           : {args.local_epochs}", 1, args)
    log(f"Aggregation Scope      : {args.aggregation_scope}", 1, args)
    log(f"FL Method              : {fl_method_name}", 1, args)

    # FedDyna-HA initialization and configuration logging
    ha_enabled = getattr(args, "enable_ha_scope", False)
    ha_selected_scope = None   # Scope locked by the selector; None means no scope has been selected.
    ha_H_select = None         # Locked update-direction disagreement score
    ha_run_id = None            # Shared run_id, generated when first written to the log.
    previous_entropy_updates = None
    previous_entropy_participation = None
    if ha_enabled:
        log("[FedDyna-HA] One-time aggregation-scope selector enabled", 1, args)
        log("  ha_lock_policy      : first parameter-update round", 1, args)
        log(f"  parameter interval  : {args.lth_epoch_iters}", 1, args)
        log(f"  ha_select_threshold : {args.ha_select_threshold}", 1, args)
        log(f"  ha_update_basis     : {args.ha_update_basis}", 1, args)
        log(f"  ha_scope_mode       : {args.ha_scope_mode}", 1, args)
        log(f"  ha_quantile         : {args.ha_quantile}", 1, args)
        log(f"  ha_lambda_std       : {args.ha_lambda_std}", 1, args)
        log(f"  ha_lambda_q         : {args.ha_lambda_q}", 1, args)
        log(f"  ha_lambda_neg       : {args.ha_lambda_neg}", 1, args)
        log(f"  ha_save_pairwise    : {args.ha_save_pairwise}", 1, args)
        log(f"  ha_debug            : {args.ha_debug}", 1, args)
        log(f"  Default scope (pre-selection): {args.aggregation_scope}", 1, args)
        if args.pruning_method == "none":
            log(
                "  WARNING: pruning_method=none, so no parameter-update round "
                "can trigger the one-time aggregation-scope selector.",
                1,
                args,
            )
    else:
        log(f"[FedDyna-HA] Disabled (use --enable_ha_scope to enable)", 1, args)

    for round_num in range(args.com_rounds):
        print_section(f"Round {round_num} Start", "=")
        client_results = {}

        # Update the method's global state.
        if fl_method is not None:
            fl_method.set_global_state(global_state)

        comm_tracker.record_pre_round(round_num, client_ids, fl_method)

        for cid in client_ids:
            local_model = copy.deepcopy(model)
            local_model.load_state_dict(client_init_state_dicts[cid])

            train_loader, val_loader = build_client_loaders(
                dataset_train,
                dataset_val,
                dict_users_train[cid],
                dict_users_val[cid],
                args,
            )

            # Call the method's pre-training hook.
            if fl_method is not None:
                fl_method.pre_train_hook(
                    cid, local_model,
                    train_loader=train_loader,
                    criterion=nn.CrossEntropyLoss(),  # Pass criterion for method-specific initialization.
                )

            train_result = train_local_standard(
                model=local_model,
                train_loader=train_loader,
                mask=client_masks[cid],
                args=args,
                client_id=cid,
                round_num=round_num,
                initialization=client_init_state_dicts[cid],
                val_loader=val_loader,
                fl_method=fl_method,  # Pass the method instance to the trainer.
            )

            local_state = {
                k: v.detach().cpu().clone()
                for k, v in train_result["model"].state_dict().items()
            }

            # Call the method's post-training hook.
            if fl_method is not None:
                fl_method.post_train_hook(
                    cid,
                    train_result["model"],
                    train_result,
                    round_num,
                )

            # Special case: run adaptive local aggregation for FedALA.
            if fl_method is not None and hasattr(fl_method, 'adaptive_local_aggregation'):
                try:
                    if prev_global_model is not None:
                        # Create a copy of the global model.
                        global_model_copy = copy.deepcopy(prev_global_model).to(device)
                        local_model_copy = copy.deepcopy(train_result["model"]).to(device)

                        # Call adaptive local aggregation.
                        fl_method.adaptive_local_aggregation(
                            cid,
                            global_model_copy,
                            local_model_copy,
                            train_loader
                        )

                        # Update the local state.
                        local_state = {
                            k: v.detach().cpu().clone()
                            for k, v in local_model_copy.state_dict().items()
                        }

                        if args.verbose >= 2:
                            log(f"[FedALA] Client {cid} adaptive local aggregation applied", 1, args)
                except Exception as e:
                    log(f"[Warning] FedALA adaptive_local_aggregation failed for client {cid}: {e}", 1, args)

            eval_local_model = copy.deepcopy(model).to(device)
            eval_local_model.load_state_dict(local_state)

            eval_global_model = copy.deepcopy(prev_global_model).to(device)

            local_metrics = evaluate_model(eval_local_model, val_loader, device)
            global_prev_metrics = evaluate_model(eval_global_model, val_loader, device)
            delta_f1 = local_metrics["f1"] - global_prev_metrics["f1"]

            personalization.update_client(cid, local_metrics, train_result["data_stats"])

            log(
                f"[Round {round_num}] [Client {cid}] "
                f"Local Metrics: acc={local_metrics['accuracy']:.4f}, "
                f"prec={local_metrics['precision']:.4f}, rec={local_metrics['recall']:.4f}, "
                f"f1={local_metrics['f1']:.4f}, prauc={local_metrics['prauc']:.4f}, "
                f"auroc={local_metrics.get('auroc', 0.0):.4f}",
                1,
                args,
            )
            log(
                f"[Round {round_num}] [Client {cid}] "
                f"Prev Global Metrics: acc={global_prev_metrics['accuracy']:.4f}, "
                f"prec={global_prev_metrics['precision']:.4f}, rec={global_prev_metrics['recall']:.4f}, "
                f"f1={global_prev_metrics['f1']:.4f}, prauc={global_prev_metrics['prauc']:.4f}, "
                f"auroc={global_prev_metrics.get('auroc', 0.0):.4f}",
                1,
                args,
            )
            log(f"[Round {round_num}] [Client {cid}] Delta F1 = {delta_f1:.6f}", 1, args)

            client_results[cid] = {
                "state_dict": local_state,
                "train_result": train_result,
                "local_metrics": local_metrics,
                "global_prev_metrics": global_prev_metrics,
                "delta_f1": delta_f1,
            }

        client_local_metrics = {cid: client_results[cid]["local_metrics"] for cid in client_ids}
        client_loss_storage = {cid: client_results[cid]["train_result"].get("train_loss", 0.0) for cid in client_ids}
        print_round_summary(round_num, client_local_metrics, client_loss_storage)

        avg_prev_global_metrics = {
            "accuracy": sum(client_results[cid]["global_prev_metrics"]["accuracy"] for cid in client_ids) / len(client_ids),
            "precision": sum(client_results[cid]["global_prev_metrics"]["precision"] for cid in client_ids) / len(client_ids),
            "recall": sum(client_results[cid]["global_prev_metrics"]["recall"] for cid in client_ids) / len(client_ids),
            "f1": sum(client_results[cid]["global_prev_metrics"]["f1"] for cid in client_ids) / len(client_ids),
            "prauc": sum(client_results[cid]["global_prev_metrics"]["prauc"] for cid in client_ids) / len(client_ids),
            "auroc": sum(client_results[cid]["global_prev_metrics"].get("auroc", 0.0) for cid in client_ids) / len(client_ids),
        }
        personalization.update_global(avg_prev_global_metrics)

        # =====================================================
        # Auxiliary one-time aggregation-scope selector (selector mode)
        # =====================================================
        # Determine aggregation_scope for this round:
        #   - If enable_ha_scope=False, use args.aggregation_scope.
        #   - If enable_ha_scope=True on the FedDyna main path (fl_method=None):
        #     use cross-client update-direction disagreement to select and lock the scope at the first scheduled mask-update round; later rounds cannot change it.
        #   - If enable_ha_scope=True with a baseline method (fl_method!=None):
        #     HA does not intervene; the baseline uses its own aggregation logic.
        # HA applies only on the FedDyna main path (fl_method=None, i.e. --fl_method fedavg).
        ha_applies_this_round = ha_enabled and (fl_method is None)
        is_parameter_update_round = mask_manager.should_update(round_num)

        client_state_dicts_for_metrics = {
            cid: client_results[cid]["state_dict"]
            for cid in client_ids
        }
        full_client_updates = None
        full_participation = None
        if ha_applies_this_round or args.weighting_method == "entropy":
            (
                full_client_updates,
                full_participation,
                _,
            ) = build_scope_aware_client_updates(
                client_ids=client_ids,
                client_state_dicts=client_state_dicts_for_metrics,
                global_state_dict=global_state,
                client_masks=client_masks,
                aggregation_scope="all",
            )

        if ha_enabled and fl_method is not None and round_num == 0:
            log(
                f"[FedDyna-HA] WARNING: --enable_ha_scope is set but fl_method={fl_method_name} "
                f"uses its own aggregation logic. HA scope control will NOT take effect.",
                1, args,
            )

        (
            current_aggregation_scope,
            ha_selected_scope,
            ha_H_select,
            ha_selection_log,
        ) = resolve_feddyna_scope_for_round(
            ha_applies_this_round=ha_applies_this_round,
            is_parameter_update_round=is_parameter_update_round,
            ha_selected_scope=ha_selected_scope,
            ha_H_select=ha_H_select,
            round_num=round_num,
            client_ids=client_ids,
            client_results=client_results,
            global_state=global_state,
            args=args,
            client_updates=full_client_updates,
        )

        entropy_diagnostics = None
        update_metric_values = None
        current_entropy_updates = None
        current_entropy_participation = None

        if args.weighting_method == "entropy":
            entropy_update_view = getattr(
                args, "entropy_update_view", "scope_aware"
            )
            if entropy_update_view == "shared_mask0":
                # Keep the unmodified train delta as the canonical update and
                # evaluate reliability on one fixed shared channel.  HA then
                # changes only parameter participation, not client weights.
                current_entropy_updates = full_client_updates
                (
                    _,
                    current_entropy_participation,
                    _,
                ) = build_scope_aware_client_updates(
                    client_ids=client_ids,
                    client_state_dicts=client_state_dicts_for_metrics,
                    global_state_dict=global_state,
                    client_masks=client_masks,
                    aggregation_scope="masked_only",
                )
            elif current_aggregation_scope == "all":
                # Exact legacy entropy behavior for reproducibility.
                current_entropy_updates = full_client_updates
                current_entropy_participation = full_participation
            else:
                (
                    current_entropy_updates,
                    current_entropy_participation,
                    _,
                ) = build_scope_aware_client_updates(
                    client_ids=client_ids,
                    client_state_dicts=client_state_dicts_for_metrics,
                    global_state_dict=global_state,
                    client_masks=client_masks,
                    aggregation_scope=current_aggregation_scope,
                )

            update_metric_values = compute_scope_aware_update_metrics(
                client_ids=client_ids,
                client_updates=current_entropy_updates,
                participation=current_entropy_participation,
                previous_updates=previous_entropy_updates,
                previous_participation=previous_entropy_participation,
            )

        client_metrics_for_weight = {
            cid: {
                "delta_f1": client_results[cid]["delta_f1"],
                "data_stats": client_results[cid]["train_result"]["data_stats"],
                "update_alignment": (
                    update_metric_values["update_alignment"][cid]
                    if update_metric_values is not None
                    else 0.5
                ),
                "update_stability": (
                    update_metric_values["update_stability"][cid]
                    if update_metric_values is not None
                    else 0.5
                ),
            }
            for cid in client_ids
        }

        client_weights, entropy_diagnostics = compute_client_weights(
            method=args.weighting_method,
            client_metrics=client_metrics_for_weight,
            personalization_controller=personalization,
            args=args,
            return_diagnostics=True,
        )

        if entropy_diagnostics is not None:
            entropy_diagnostics["aggregation_scope"] = current_aggregation_scope
            entropy_diagnostics["update_view"] = entropy_update_view
            entropy_diagnostics["alignment_coverage"] = (
                update_metric_values["alignment_coverage"]
            )
            entropy_diagnostics["stability_coverage"] = (
                update_metric_values["stability_coverage"]
            )

            print_section(f"Round {round_num} Entropy Factor Weights", "-")
            for factor_name, factor_weight in entropy_diagnostics[
                "factor_weights"
            ].items():
                log(
                    f"{factor_name}: weight={factor_weight:.6f}",
                    1,
                    args,
                )

        print_section(f"Round {round_num} Aggregation Weights", "-")
        for cid in client_ids:
            log(f"Client {cid}: weight={client_weights[cid]:.6f}", 1, args)

        if getattr(args, "save_ha_analysis", False):
            analysis_dir = getattr(args, "ha_analysis_dir", None)
            if not analysis_dir:
                analysis_dir = os.path.join(args.save_dir, args.exp_name, "ha_analysis")
            save_ha_round_analysis(
                round_id=round_num,
                client_results=client_results,
                client_ids=client_ids,
                global_state=global_state,
                client_masks=client_masks,
                selected_scope=current_aggregation_scope,
                analysis_dir=analysis_dir,
                args=args,
            )

        # =====================================================
        # Aggregate.
        # =====================================================
        zero_denom_ratio = 0.0  # Initialize; updated only for masked_only.

        if fl_method is not None:
            global_state = aggregate_baseline_path(
                fl_method=fl_method,
                fl_method_name=fl_method_name,
                client_results=client_results,
                client_ids=client_ids,
                client_weights=client_weights,
                global_state=global_state,
                round_num=round_num,
                args=args,
            )
        else:
            global_state, zero_denom_ratio = aggregate_feddyna_path(
                client_results=client_results,
                client_ids=client_ids,
                client_weights=client_weights,
                client_masks=client_masks,
                global_state=global_state,
                current_aggregation_scope=current_aggregation_scope,
                args=args,
            )

        # Save HA log data after aggregation because zero_denom_ratio is required.
        if ha_applies_this_round:
            # Generate run_id once; reuse it for the entire experiment.
            if ha_run_id is None:
                from datetime import datetime as _dt
                ha_run_id = f"{args.exp_name}_seed{args.seed}_{_dt.now().strftime('%Y%m%d_%H%M%S')}"

            # Record scope history for each round.
            append_ha_scope_history(
                save_dir=args.save_dir,
                exp_name=args.exp_name,
                round_idx=round_num,
                selected_scope=current_aggregation_scope,
                H_select=ha_H_select,
                zero_denom_ratio=zero_denom_ratio,
                args=args,
                run_id=ha_run_id,
            )
            # Selection round: Save additional selection details.
            if ha_selection_log is not None and ha_selection_log["is_lock_round"]:
                append_ha_selection_log(
                    save_dir=args.save_dir,
                    exp_name=args.exp_name,
                    select_round=round_num,
                    stats_dict=ha_selection_log["stats_dict"],
                    H_select=ha_selection_log["current_H_select"],
                    agreement_score=ha_selection_log["agreement_score"],
                    selected_scope=ha_selected_scope,
                    args=args,
                    pairwise_records=ha_selection_log["pairwise_records"] if getattr(args, "ha_save_pairwise", False) else None,
                    client_update_norms=ha_selection_log["client_update_norms"],
                    run_id=ha_run_id,
                )

        # Train personalized models after aggregation using the updated global model.
        # Run personalized training only for methods that use personalized inference.
        if fl_method is not None and fl_method.requires_personalized_inference():
            print_section(f"Round {round_num} Personal Model Training (Ditto)", "-")
            personal_epochs = getattr(fl_method, 'personal_epochs', None) or args.local_epochs
            log(f"Training personal models for {personal_epochs} epochs per client...", 1, args)

            # Create a new global model for personalized training.
            global_model_for_personal = copy.deepcopy(model)
            global_model_for_personal.load_state_dict(global_state)
            global_model_for_personal.to(device)

            for cid in client_ids:
                train_loader, val_loader = build_client_loaders(
                    dataset_train,
                    dataset_val,
                    dict_users_train[cid],
                    dict_users_val[cid],
                    args,
                )

                log(f"[Round {round_num}] [Client {cid}] Training personal model...", 1, args)

                personal_result = fl_method.train_personal_model(
                    client_id=cid,
                    global_model=global_model_for_personal,
                    train_loader=train_loader,
                    device=device,
                    epochs=personal_epochs,
                )

                log(
                    f"[Round {round_num}] [Client {cid}] Personal model trained: "
                    f"loss={personal_result.get('loss', personal_result.get('train_loss', 0.0)):.4f}",
                    1, args
                )

        if args.print_param_stats:
            print_state_summary(global_state, title=f"Round {round_num} Aggregated Global State", topk=8)

        print_section(f"Round {round_num} Post-Aggregation Global Evaluation", "-")
        client_states_for_eval = {cid: client_results[cid]["state_dict"] for cid in client_ids}
        post_global_metrics = evaluate_global_after_aggregation(
            model_template=model,
            global_state=global_state,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dict_users_train=dict_users_train,
            dict_users_val=dict_users_val,
            client_ids=client_ids,
            args=args,
            fl_method=fl_method,
            client_states=client_states_for_eval,
        )
        avg_post_global_metrics = summarize_global_metrics(post_global_metrics)

        for cid in client_ids:
            m = post_global_metrics[cid]
            log(
                f"[Round {round_num}] [Client {cid}] "
                f"Post Global Metrics: acc={m['accuracy']:.4f}, "
                f"prec={m['precision']:.4f}, rec={m['recall']:.4f}, "
                f"f1={m['f1']:.4f}, prauc={m['prauc']:.4f}, auroc={m.get('auroc', 0.0):.4f}",
                1,
                args,
            )

        log(
            f"[Round {round_num}] [Global Avg After Aggregation] "
            f"acc={format_mean_std(avg_post_global_metrics.get('accuracy', 0.0), avg_post_global_metrics.get('accuracy_std', 0.0))}, "
            f"prec={format_mean_std(avg_post_global_metrics.get('precision', 0.0), avg_post_global_metrics.get('precision_std', 0.0))}, "
            f"rec={format_mean_std(avg_post_global_metrics.get('recall', 0.0), avg_post_global_metrics.get('recall_std', 0.0))}, "
            f"f1={format_mean_std(avg_post_global_metrics.get('f1', 0.0), avg_post_global_metrics.get('f1_std', 0.0))}, "
            f"prauc={format_mean_std(avg_post_global_metrics.get('prauc', 0.0), avg_post_global_metrics.get('prauc_std', 0.0))}, "
            f"auroc={format_mean_std(avg_post_global_metrics.get('auroc', 0.0), avg_post_global_metrics.get('auroc_std', 0.0))}",
            1,
            args,
        )

        if is_parameter_update_round:
            mean_local_ratio_before = float(np.mean([
                summarize_mask(client_masks[cid])["local_ratio"]
                for cid in client_ids
            ]))
            log(
                f"[Round {round_num}] Starting pruning update "
                f"(method: {args.pruning_method}, interval: {args.lth_epoch_iters})",
                1,
                args,
            )

            pruning_rates = None
            if args.pruning_method == "kdars":
                pruning_rates_array, _, _ = personalization.compute_pruning_rates(
                    client_state_dicts={
                        cid: client_results[cid]["state_dict"]
                        for cid in client_ids
                    },
                    prev_state_dicts=client_init_state_dicts,
                    global_state_dict=global_state,
                )
                pruning_rates = {
                    cid: float(pruning_rates_array[idx])
                    for idx, cid in enumerate(client_ids)
                }
                log(
                    f"[Round {round_num}] KDARS step global-retention "
                    f"ratios computed: {pruning_rates}",
                    1,
                    args,
                )

            for cid in client_ids:
                current_state_dict = client_results[cid]["state_dict"]
                prev_state_dict = client_init_state_dicts[cid]

                current_model = copy.deepcopy(model).to(device)
                current_model.load_state_dict(current_state_dict)
                prev_model = copy.deepcopy(model).to(device)
                prev_model.load_state_dict(prev_state_dict)

                local_metrics = client_results[cid]["local_metrics"]
                global_metrics = client_results[cid]["global_prev_metrics"]
                global_ratio_override = pruning_rates[cid] if pruning_rates and cid in pruning_rates else None

                mask_stats_before = summarize_mask(client_masks[cid])

                # FedLTN-specific pruning based on validation accuracy
                if fl_method is not None and hasattr(fl_method, 'post_pruning_update'):
                    val_accuracy = local_metrics["accuracy"]
                    new_mask = fl_method.post_pruning_update(
                        current_state_dict=current_state_dict,
                        current_mask=client_masks[cid],
                        val_accuracy=val_accuracy,
                    )
                    # Update FedLTN's internal mask storage
                    fl_method.set_client_mask(cid, new_mask)
                else:
                    # Standard mask manager pruning
                    new_mask = mask_manager.update_client_mask(
                        current_model=current_model,
                        prev_model=prev_model,
                        current_mask=client_masks[cid],
                        local_metrics=local_metrics,
                        global_metrics=global_metrics,
                        global_ratio_override=global_ratio_override,
                    )
                mask_stats_after = summarize_mask(new_mask)

                client_masks[cid] = new_mask

                # Sync mask with FL method if supported
                if fl_method is not None and hasattr(fl_method, 'set_client_mask'):
                    fl_method.set_client_mask(cid, new_mask)

                log(f"[Round {round_num}] [Client {cid}] Mask updated:", 1, args)
                log(
                    f"  Before: global={mask_stats_before['global_ratio']:.4f}, "
                    f"local={mask_stats_before['local_ratio']:.4f}",
                    1,
                    args,
                )
                log(
                    f"  After:  global={mask_stats_after['global_ratio']:.4f}, "
                    f"local={mask_stats_after['local_ratio']:.4f}",
                    1,
                    args,
                )
                log(
                    f"  Change: global_ratio Δ="
                    f"{mask_stats_after['global_ratio'] - mask_stats_before['global_ratio']:+.4f}",
                    1,
                    args,
                )

                if args.pruning_method == "kdars" and pruning_rates and cid in pruning_rates:
                    log(
                        f"  KDARS step global-retention ratio: "
                        f"{pruning_rates[cid]:.4f}",
                        1,
                        args,
                    )

            log(f"[Round {round_num}] Pruning update completed for all clients", 1, args)

            if ha_applies_this_round and ha_selection_log is not None:
                mean_local_ratio_after = float(np.mean([
                    summarize_mask(client_masks[cid])["local_ratio"]
                    for cid in client_ids
                ]))
                append_ha_parameter_update_stats(
                    save_dir=args.save_dir,
                    exp_name=args.exp_name,
                    round_idx=round_num,
                    stats_dict=ha_selection_log["stats_dict"],
                    H_select=ha_selection_log["current_H_select"],
                    agreement_score=ha_selection_log["agreement_score"],
                    candidate_scope=ha_selection_log["candidate_scope"],
                    locked_scope=ha_selection_log["locked_scope"],
                    is_lock_round=ha_selection_log["is_lock_round"],
                    decision_applied=ha_selection_log["decision_applied"],
                    mean_local_ratio_before=mean_local_ratio_before,
                    mean_local_ratio_after=mean_local_ratio_after,
                    args=args,
                    run_id=ha_run_id,
                )
        else:
            if args.pruning_method != "none":
                log(
                    f"[Round {round_num}] Skipping pruning update "
                    f"(not at update interval: round_num={round_num}, lth_epoch_iters={args.lth_epoch_iters})",
                    1,
                    args,
                )

        # Account after mask updates, without altering any training messages.
        select_mask_update = (
            fl_method_name == "fedselect" and fl_method is not None
            and fl_method.latest_round > 0
            and fl_method.latest_round % fl_method.update_interval == 0
        )
        comm_stats = comm_tracker.record_round(
            round_num=round_num,
            client_state_dicts={cid: client_results[cid]["state_dict"] for cid in client_ids},
            global_state=global_state,
            client_masks=client_masks,
            client_weights=client_weights,
            fl_method_name=fl_method_name,
            client_results=client_results,
            fl_method=fl_method,
            mask_updated=is_parameter_update_round or select_mask_update,
        )
        # Persist each completed round, including partial runs if later work fails.
        comm_tracker.export(os.path.join(args.run_dir or args.save_dir, "results"))
        log(
            f"[Round {round_num}] [Communication: estimated communication payload] "
            f"upload={comm_stats['upload_bytes']} bytes, "
            f"download={comm_stats['download_bytes']} bytes, "
            f"total={comm_stats['total_bytes']} bytes", 1, args,
        )

        round_item = {
            "round": round_num,
            "client_metrics": {},
            "client_weights": client_weights,
            "entropy_aggregation": entropy_diagnostics,
            "avg_post_global_metrics": avg_post_global_metrics,
            "communication_stats": comm_stats,
            "aggregation_scope": current_aggregation_scope,
            "zero_denom_ratio": zero_denom_ratio,
        }

        # FedDyna-HA: save HA state to history.
        if ha_enabled:
            round_item["ha_state"] = {
                "selected_scope": ha_selected_scope,
                "H_select": ha_H_select,
            }
            if ha_selection_log is not None:
                round_item["ha_state"].update({
                    "round_H_select": ha_selection_log["current_H_select"],
                    "candidate_scope": ha_selection_log["candidate_scope"],
                    "is_lock_round": ha_selection_log["is_lock_round"],
                    "decision_applied": ha_selection_log["decision_applied"],
                })

        for cid in client_ids:
            client_result = client_results[cid]
            train_result = client_result["train_result"]
            round_item["client_metrics"][cid] = {
                "local": client_result["local_metrics"],
                "global_prev": client_result["global_prev_metrics"],
                "post_global": post_global_metrics.get(cid, {}),
                "train_loss": train_result.get("train_loss", None),
                "train_eval": train_result.get("train_eval", {}),
                "mask_stats": train_result.get("mask_stats", {}),
                "data_stats": train_result.get("data_stats", {}),
                "delta_f1": client_result["delta_f1"],
                "epoch_losses": train_result.get("epoch_losses", []),
                "epoch_accs": train_result.get("epoch_accs", []),
                "grad_norms": train_result.get("grad_norms", []),
                "plot_path": train_result.get("plot_path", ""),
            }

        history.append(round_item)

        if (
            current_entropy_updates is not None
            and current_entropy_participation is not None
        ):
            previous_entropy_updates = {
                cid: update.detach().cpu().clone()
                for cid, update in current_entropy_updates.items()
            }
            previous_entropy_participation = {
                cid: participate.detach().cpu().clone()
                for cid, participate in current_entropy_participation.items()
            }

        for cid in client_ids:
            final_local_state_dicts[cid] = copy.deepcopy(client_results[cid]["state_dict"])

            # Track the best local model by validation F1.
            current_f1 = client_results[cid]["local_metrics"]["f1"]
            if current_f1 > best_client_local_f1[cid]:
                best_client_local_f1[cid] = current_f1
                best_client_local_state[cid] = copy.deepcopy(client_results[cid]["state_dict"])
                best_client_round[cid] = round_num
                log(f"[Round {round_num}] [Client {cid}] New best local F1: {current_f1:.4f}", 1, args)

            client_init_state_dicts[cid] = broadcast_server_to_client_initialization(
                global_state,
                client_masks[cid],
                final_local_state_dicts[cid],
            )
            log(
                f"[Round {round_num}] [Client {cid}] Model broadcast completed "
                f"(mask=0: global, mask=1: local)",
                2,
                args,
            )

        prev_global_model = copy.deepcopy(model)
        prev_global_model.load_state_dict(global_state)

        print_section(f"Round {round_num} End", "=")

    # Use the best local model to replace the final-round state according to save_best_local_model.
    save_best = getattr(args, "save_best_local_model", True)
    if save_best:
        # Replace the final-round state with the best local-model state (default behavior).
        for cid in client_ids:
            if best_client_local_state[cid] is not None:
                final_local_state_dicts[cid] = best_client_local_state[cid]
        log("Use the historical best-F1 model as the final local model.", 1, args)
    else:
        # Keep the final trained model state without replacement.
        log("Use the final trained model as the final local model.", 1, args)

    # Print the best local-model summary.
    print_section("Local Models Summary", "=")
    for cid in client_ids:
        if best_client_round[cid] >= 0:
            log(f"Client {cid}: Best F1={best_client_local_f1[cid]:.4f} at Round {best_client_round[cid]}", 1, args)

    # Stop timing and print the communication summary.
    comm_tracker.end_training()
    comm_tracker.export(os.path.join(args.run_dir or args.save_dir, "results"))
    comm_summary = comm_tracker.get_summary()
    print_section("Communication Overhead Summary", "=")
    log(f"Total training time:   {comm_summary['total_training_time_sec']:.2f} s ({comm_summary['total_training_time_min']:.2f} min)", 1, args)
    log(f"Total uplink: {comm_summary['total_uplink_mb']:.2f} MB", 1, args)
    log(f"Total downlink: {comm_summary['total_downlink_mb']:.2f} MB", 1, args)
    log(f"Total communication:   {comm_summary['total_communication_mb']:.2f} MB", 1, args)
    log(f"Average uplink per round: {comm_summary['avg_uplink_per_round_mb']:.2f} MB", 1, args)
    log(f"Average downlink per round: {comm_summary['avg_downlink_per_round_mb']:.2f} MB", 1, args)

    final_global_model = copy.deepcopy(model)
    final_global_model.load_state_dict(global_state)

    print_section("Federated Training Finished", "=")
    return final_global_model, final_local_state_dicts, client_init_state_dicts, client_masks, history, comm_tracker
