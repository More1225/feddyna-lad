# coding: utf-8
"""
FedDyna Auxiliary One-Time Aggregation-Scope Selector

Core idea:
  Use cross-client update-direction disagreement to select and lock the aggregation scope once, at the first scheduled KDARS/mask-update round:
    - H_select < threshold → selected_scope = "all"  (lower update-direction disagreement, aggregate all parameters)
    - H_select >= threshold → selected_scope = "masked_only" (higher update-direction disagreement, aggregate shared parameters)

Mathematical definition:
  At round t, the global model is θ^t, and client i's locally trained model is θ_i^{t+1/2}.
  Update vector: Δ_i = θ_i^{t+1/2} - θ^t
  cosine similarity: c_ij = <Δ_i, Δ_j> / (||Δ_i||_2 ||Δ_j||_2 + eps)

Decision process (selector mode):
  1. Compute cosine similarity for all client pairs at the first parameter-update round.
  2. Compute summary statistics (mean, std, negative_ratio, low_quantile, ...).
  3. Compute an update-direction agreement score and derive H_select.
  4. H_select >= ha_select_threshold selects masked_only; otherwise, select all.
  5. Lock selected_scope; later parameter-update rounds record candidate scopes without changing the locked selection.
"""

import os
import csv
import json
import torch
import numpy as np
from collections import OrderedDict
from datetime import datetime


# ===========================================================================
# 0. Utility: client source resolution
# ===========================================================================

_KNOWN_PREFIXES = ["hdfs", "bgl", "tb", "thunderbird"]


def resolve_client_source(client_id, args=None):
    """
    Resolve a client's data-source label.

    Precedence:
      1. args.client_source_map lookup
      2. parse the client_id string prefix (e.g.  "hdfs_0" → "hdfs")
      3. fallback to  "unknown"
    """
    if args is not None and hasattr(args, "client_source_map"):
        src_map = args.client_source_map
        if src_map and client_id in src_map:
            return src_map[client_id]

    cid_str = str(client_id)
    for prefix in _KNOWN_PREFIXES:
        if cid_str.lower().startswith(prefix):
            return prefix

    return "unknown"


# ===========================================================================
# 1. Flatten model update
# ===========================================================================

def flatten_model_update(local_model_or_state, global_model_or_state):
    """
    Compute local - global and flatten the result into a 1D tensor.

    - Skip non-floating-point parameters (such as num_batches_tracked).
    - Return the vector on CPU to avoid GPU-memory and device conflicts.
    """
    if isinstance(local_model_or_state, torch.nn.Module):
        local_sd = local_model_or_state.state_dict()
    elif isinstance(local_model_or_state, (dict, OrderedDict)):
        local_sd = local_model_or_state
    else:
        raise TypeError(f"Unsupported local_model_or_state type: {type(local_model_or_state)}")

    if isinstance(global_model_or_state, torch.nn.Module):
        global_sd = global_model_or_state.state_dict()
    elif isinstance(global_model_or_state, (dict, OrderedDict)):
        global_sd = global_model_or_state
    else:
        raise TypeError(f"Unsupported global_model_or_state type: {type(global_model_or_state)}")

    diffs = []
    for key in local_sd:
        if key not in global_sd:
            continue
        local_val = local_sd[key]
        global_val = global_sd[key]

        if not local_val.is_floating_point():
            continue

        diff = local_val.float().cpu() - global_val.float().cpu()
        diffs.append(diff.flatten())

    if not diffs:
        raise ValueError("No floating-point parameters are available to compute the update.")

    return torch.cat(diffs)


# ===========================================================================
# 2. Pairwise cosine similarity statistics
# ===========================================================================

def compute_pairwise_cosine_stats(client_updates, client_ids=None, quantile=0.1, eps=1e-12):
    """
    Compute pairwise cosine similarities and summary statistics for client update vectors.

    Returns:
        stats_dict: Contains mean_cos, std_cos, negative_ratio, low_quantile_cos,
                    min_cos, max_cos, median_cos, num_pairs
        pairwise_records: list[dict], each item has {client_i, client_j, cosine}
    """
    if isinstance(client_updates, dict):
        if client_ids is None:
            client_ids = sorted(client_updates.keys())
        updates = [client_updates[cid] for cid in client_ids]
    else:
        updates = list(client_updates)
        if client_ids is None:
            client_ids = list(range(len(updates)))

    n = len(updates)
    if n < 2:
        stats_dict = {
            "mean_cos": 1.0,
            "std_cos": 0.0,
            "negative_ratio": 0.0,
            "low_quantile_cos": 1.0,
            "min_cos": 1.0,
            "max_cos": 1.0,
            "median_cos": 1.0,
            "num_pairs": 0,
        }
        return stats_dict, []

    norms = []
    for u in updates:
        if isinstance(u, torch.Tensor):
            norms.append(u.float().norm(2).item())
        else:
            norms.append(float(np.linalg.norm(u)))
    norms = np.array(norms, dtype=np.float64)

    cosines = []
    pairwise_records = []

    for i in range(n):
        for j in range(i + 1, n):
            ui = updates[i].float() if isinstance(updates[i], torch.Tensor) else torch.tensor(updates[i], dtype=torch.float32)
            uj = updates[j].float() if isinstance(updates[j], torch.Tensor) else torch.tensor(updates[j], dtype=torch.float32)

            dot = torch.dot(ui, uj).item()
            denom = norms[i] * norms[j] + eps
            cos_val = dot / denom
            cos_val = max(-1.0, min(1.0, cos_val))

            cosines.append(cos_val)
            pairwise_records.append({
                "client_i": client_ids[i],
                "client_j": client_ids[j],
                "cosine": cos_val,
            })

    cosines_arr = np.array(cosines, dtype=np.float64)

    stats_dict = {
        "mean_cos": float(np.mean(cosines_arr)),
        "std_cos": float(np.std(cosines_arr)),
        "negative_ratio": float(np.sum(cosines_arr < 0) / len(cosines_arr)),
        "low_quantile_cos": float(np.quantile(cosines_arr, quantile)),
        "min_cos": float(np.min(cosines_arr)),
        "max_cos": float(np.max(cosines_arr)),
        "median_cos": float(np.median(cosines_arr)),
        "num_pairs": len(cosines),
    }

    return stats_dict, pairwise_records


def compute_pairwise_cosine_matrix(client_updates, client_ids=None, eps=1e-12):
    """
    Compute the full pairwise cosine similarity matrix for flattened updates.

    Returns:
        matrix: np.ndarray with shape [num_clients, num_clients]
        client_ids: list in the same order as matrix rows/columns
        update_matrix: np.ndarray with shape [num_clients, num_params]
    """
    if isinstance(client_updates, dict):
        if client_ids is None:
            client_ids = sorted(client_updates.keys())
        updates = [client_updates[cid] for cid in client_ids]
    else:
        updates = list(client_updates)
        if client_ids is None:
            client_ids = list(range(len(updates)))

    if not updates:
        return np.zeros((0, 0), dtype=np.float32), list(client_ids), np.zeros((0, 0), dtype=np.float32)

    arrays = []
    for update in updates:
        if isinstance(update, torch.Tensor):
            arr = update.detach().float().cpu().numpy()
        else:
            arr = np.asarray(update, dtype=np.float32)
        arrays.append(arr.reshape(-1).astype(np.float32, copy=False))

    update_matrix = np.vstack(arrays).astype(np.float32, copy=False)
    norms = np.linalg.norm(update_matrix, axis=1, keepdims=True)
    normalized = update_matrix / np.maximum(norms, eps)
    cosine_matrix = np.matmul(normalized, normalized.T)
    cosine_matrix = np.clip(cosine_matrix, -1.0, 1.0).astype(np.float32, copy=False)
    np.fill_diagonal(cosine_matrix, 1.0)
    return cosine_matrix, list(client_ids), update_matrix


def mean_upper_triangle(matrix, default=0.0):
    """Mean over i < j entries."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return float(default)
    tri = np.triu_indices(matrix.shape[0], k=1)
    if len(tri[0]) == 0:
        return float(default)
    return float(np.mean(matrix[tri]))


def flatten_mask(mask_dict, key_order=None):
    """
    Flatten a client mask dictionary into a 1D boolean numpy array.
    mask=True corresponds to mask value != 0.
    """
    if not mask_dict:
        return None, key_order or []

    if key_order is None:
        key_order = [key for key in sorted(mask_dict.keys())]

    parts = []
    for key in key_order:
        if key not in mask_dict:
            continue
        val = mask_dict[key]
        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().numpy()
        else:
            arr = np.asarray(val)
        parts.append((arr.reshape(-1) != 0))

    if not parts:
        return None, key_order
    return np.concatenate(parts).astype(bool, copy=False), key_order


def compute_pairwise_mask_disagreement_matrix(client_masks, client_ids=None):
    """
    Compute D_ij = mean(mask_i != mask_j) for available client masks.
    Returns None when masks are unavailable or incompatible.
    """
    if not client_masks:
        return None, []

    if client_ids is None:
        client_ids = sorted(client_masks.keys())

    available_ids = [cid for cid in client_ids if cid in client_masks and client_masks[cid]]
    if len(available_ids) < 2:
        return None, available_ids

    first_mask = client_masks[available_ids[0]]
    key_order = [key for key in sorted(first_mask.keys())]
    flat_masks = []
    for cid in available_ids:
        flat, _ = flatten_mask(client_masks[cid], key_order=key_order)
        if flat is None:
            return None, available_ids
        flat_masks.append(flat)

    lengths = {mask.shape[0] for mask in flat_masks}
    if len(lengths) != 1:
        return None, available_ids

    n = len(available_ids)
    disagreement = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(np.mean(flat_masks[i] != flat_masks[j]))
            disagreement[i, j] = value
            disagreement[j, i] = value
    return disagreement, available_ids


def save_labeled_matrix_csv(path, matrix, labels, value_format="{:.8f}"):
    """Save a square matrix with client ids as row/column labels."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id"] + [str(label) for label in labels])
        for label, row in zip(labels, matrix):
            writer.writerow([str(label)] + [value_format.format(float(value)) for value in row])


def append_ha_score_row(path, row):
    header = ["round_id", "H_cos", "H_mask", "H_total", "mean_cosine", "selected_scope"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in header})


def save_ha_round_analysis(round_id, client_results, client_ids, global_state,
                           client_masks=None, selected_scope="",
                           analysis_dir=None, args=None):
    """
    Save per-round HA analysis artifacts before server aggregation.

    This is a passive analysis hook: it does not change aggregation, masks, or
    any training state.
    """
    if analysis_dir is None:
        base_dir = getattr(args, "save_dir", "./save") if args is not None else "./save"
        exp_name = getattr(args, "exp_name", "fed_lightlog_exp") if args is not None else "fed_lightlog_exp"
        analysis_dir = os.path.join(base_dir, exp_name, "ha_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    round_dir = os.path.join(analysis_dir, f"round_{int(round_id):03d}")
    os.makedirs(round_dir, exist_ok=True)

    client_updates = {}
    for cid in client_ids:
        client_updates[cid] = flatten_model_update(
            client_results[cid]["state_dict"],
            global_state,
        )

    cosine_matrix, ordered_ids, update_matrix = compute_pairwise_cosine_matrix(
        client_updates,
        client_ids=client_ids,
    )
    mean_cosine = mean_upper_triangle(cosine_matrix, default=1.0)
    H_cos = (1.0 - mean_cosine) / 2.0
    H_cos = max(0.0, min(1.0, H_cos))

    cos_npy = os.path.join(round_dir, "pairwise_cosine.npy")
    cos_csv = os.path.join(round_dir, "pairwise_cosine.csv")
    np.save(cos_npy, cosine_matrix)
    save_labeled_matrix_csv(cos_csv, cosine_matrix, ordered_ids)

    np.save(os.path.join(round_dir, "client_update_vectors.npy"), update_matrix)

    sources = [resolve_client_source(cid, args) for cid in ordered_ids]
    with open(os.path.join(round_dir, "client_metadata.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "source"])
        for cid, source in zip(ordered_ids, sources):
            writer.writerow([cid, source])

    mask_status = "unavailable"
    H_mask = 0.0
    mask_matrix, mask_ids = compute_pairwise_mask_disagreement_matrix(client_masks, client_ids=client_ids)
    if mask_matrix is not None:
        mask_status = "available"
        H_mask = mean_upper_triangle(mask_matrix, default=0.0)
        np.save(os.path.join(round_dir, "pairwise_mask_disagreement.npy"), mask_matrix)
        save_labeled_matrix_csv(
            os.path.join(round_dir, "pairwise_mask_disagreement.csv"),
            mask_matrix,
            mask_ids,
        )

    H_total = 0.7 * H_cos + 0.3 * H_mask
    scores_path = os.path.join(analysis_dir, "ha_scores.csv")
    append_ha_score_row(scores_path, {
        "round_id": int(round_id),
        "H_cos": f"{H_cos:.8f}",
        "H_mask": f"{H_mask:.8f}",
        "H_total": f"{H_total:.8f}",
        "mean_cosine": f"{mean_cosine:.8f}",
        "selected_scope": selected_scope,
    })

    summary = {
        "round_id": int(round_id),
        "num_clients": len(ordered_ids),
        "H_cos": float(H_cos),
        "H_mask": float(H_mask),
        "H_total": float(H_total),
        "mean_cosine": float(mean_cosine),
        "selected_scope": selected_scope,
        "mask_status": mask_status,
        "cosine_npy": cos_npy,
        "cosine_csv": cos_csv,
    }
    with open(os.path.join(round_dir, "ha_round_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if mask_status != "available":
        print(f"[HA Analysis][Round {round_id}] mask unavailable; H_mask set to 0.0")
    print(
        f"[HA Analysis][Round {round_id}] "
        f"H_cos={H_cos:.4f}, H_mask={H_mask:.4f}, "
        f"H_total={H_total:.4f}, mean_cosine={mean_cosine:.4f}, "
        f"scope={selected_scope}"
    )
    return summary


# ===========================================================================
# 3. Update-direction disagreement score computation
# ===========================================================================

def compute_heterogeneity_score(stats_dict, mode="robust",
                                lambda_std=0.3, lambda_q=0.3, lambda_neg=0.4):
    """
    Compute agreement_score and H_select from pairwise cosine statistics.

    robust mode:
        agreement_score = mean_cos - λ_std*std - λ_q*(mean-q) - λ_neg*neg_ratio

    H_select = clip((1 - agreement_score) / 2, 0, 1)
    """
    mean_cos = stats_dict["mean_cos"]

    if mode == "robust":
        std_cos = stats_dict["std_cos"]
        low_q_cos = stats_dict["low_quantile_cos"]
        neg_ratio = stats_dict["negative_ratio"]

        agreement_score = (
            mean_cos
            - lambda_std * std_cos
            - lambda_q * (mean_cos - low_q_cos)
            - lambda_neg * neg_ratio
        )
    else:
        raise ValueError(f"Unsupported ha_scope_mode: {mode}")

    H_select = (1.0 - agreement_score) / 2.0
    H_select = max(0.0, min(1.0, H_select))

    return agreement_score, H_select


# ===========================================================================
# 4. Aggregation-scope selector decision
# ===========================================================================

def select_scope(H_select, threshold):
    """
    Select selected_scope from H_select and the threshold.

    Args:
        H_select:  update-direction disagreement score, clipped to [0, 1]
        threshold: ha_select_threshold

    Returns:
        str: "all" or "masked_only"
    """
    if H_select >= threshold:
        return "masked_only"
    return "all"


# ===========================================================================
# 5. HA data logging / persistence (selector mode)
# ===========================================================================

def append_ha_selection_log(save_dir, exp_name, select_round, stats_dict,
                            H_select, agreement_score, selected_scope,
                            args, pairwise_records=None, client_update_norms=None,
                            run_id=None):
    """
    Save HA data for the selection round to CSV (called once at select_round).

    Files written:
        - {save_dir}/{exp_name}/ha_logs/ha_scope_selection.csv  (one row)
        - {save_dir}/{exp_name}/ha_logs/pairwise_cosines.csv    (when ha_save_pairwise is enabled)
        - {save_dir}/{exp_name}/ha_logs/client_update_norms.csv
    """
    ha_dir = os.path.join(save_dir, exp_name, "ha_logs")
    os.makedirs(ha_dir, exist_ok=True)

    seed = getattr(args, "seed", -1)
    dataset = getattr(args, "dataset", "unknown")
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{exp_name}_seed{seed}_{timestamp}"

    # ---- 1. ha_scope_selection.csv ----
    # header: 25 fields
    sel_path = os.path.join(ha_dir, "ha_scope_selection.csv")
    sel_header = [
        "run_id", "exp_name", "dataset", "seed", "select_round",
        "selected_scope", "H_select", "agreement_score",
        "mean_cos", "std_cos", "negative_ratio", "low_quantile_cos",
        "min_cos", "max_cos", "median_cos", "num_clients", "num_pairs",
        "ha_scope_mode", "ha_select_threshold",
        "ha_quantile", "ha_lambda_std", "ha_lambda_q", "ha_lambda_neg",
        "ha_update_basis",
    ]
    import math
    num_clients = 0
    np_val = stats_dict.get("num_pairs", 0)
    if np_val > 0:
        num_clients = int((1 + math.sqrt(1 + 8 * np_val)) / 2)

    sel_row = [
        run_id,                                   # 1  run_id
        exp_name,                                 # 2  exp_name
        dataset,                                  # 3  dataset
        seed,                                     # 4  seed
        select_round,                             # 5  select_round
        selected_scope,                           # 6  selected_scope
        f"{H_select:.6f}",                        # 7  H_select
        f"{agreement_score:.6f}",                 # 8  agreement_score
        f"{stats_dict.get('mean_cos', 0.0):.6f}",       # 9  mean_cos
        f"{stats_dict.get('std_cos', 0.0):.6f}",        # 10 std_cos
        f"{stats_dict.get('negative_ratio', 0.0):.6f}", # 11 negative_ratio
        f"{stats_dict.get('low_quantile_cos', 0.0):.6f}", # 12 low_quantile_cos
        f"{stats_dict.get('min_cos', 0.0):.6f}",        # 13 min_cos
        f"{stats_dict.get('max_cos', 0.0):.6f}",        # 14 max_cos
        f"{stats_dict.get('median_cos', 0.0):.6f}",     # 15 median_cos
        num_clients,                              # 16 num_clients
        stats_dict.get("num_pairs", 0),           # 17 num_pairs
        getattr(args, "ha_scope_mode", "robust"),  # 18 ha_scope_mode
        getattr(args, "ha_select_threshold", 0.55), # 19 ha_select_threshold
        getattr(args, "ha_quantile", 0.1),         # 20 ha_quantile
        getattr(args, "ha_lambda_std", 0.3),       # 21 ha_lambda_std
        getattr(args, "ha_lambda_q", 0.3),         # 22 ha_lambda_q
        getattr(args, "ha_lambda_neg", 0.4),       # 23 ha_lambda_neg
        getattr(args, "ha_update_basis", "train_delta"), # 24 ha_update_basis
    ]

    write_header = not os.path.exists(sel_path)
    with open(sel_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(sel_header)
        writer.writerow(sel_row)

    # ---- 2. pairwise_cosines.csv (only when ha_save_pairwise=True) ----
    if getattr(args, "ha_save_pairwise", False) and pairwise_records:
        pairwise_path = os.path.join(ha_dir, "pairwise_cosines.csv")
        pairwise_header = [
            "run_id", "exp_name", "dataset", "seed", "select_round",
            "client_i", "client_i_source", "client_j", "client_j_source",
            "cosine",
        ]
        write_header = not os.path.exists(pairwise_path)
        with open(pairwise_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(pairwise_header)
            for rec in pairwise_records:
                ci = rec["client_i"]
                cj = rec["client_j"]
                writer.writerow([
                    run_id, exp_name, dataset, seed, select_round,
                    ci, resolve_client_source(ci, args),
                    cj, resolve_client_source(cj, args),
                    f"{rec['cosine']:.8f}",
                ])

    # ---- 3. client_update_norms.csv ----
    if client_update_norms:
        norms_path = os.path.join(ha_dir, "client_update_norms.csv")
        norms_header = [
            "run_id", "exp_name", "dataset", "seed", "select_round",
            "client_id", "client_source", "update_l2_norm",
        ]
        write_header = not os.path.exists(norms_path)
        with open(norms_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(norms_header)
            for cid, norm_val in client_update_norms.items():
                writer.writerow([
                    run_id, exp_name, dataset, seed, select_round,
                    cid, resolve_client_source(cid, args),
                    f"{norm_val:.8f}",
                ])


def append_ha_scope_history(save_dir, exp_name, round_idx, selected_scope,
                            H_select, zero_denom_ratio, args, run_id=None):
    """
    Append the current round to aggregation_scope_history.csv.

    Record the scope used in each round (constant across rounds in selector mode).
    """
    ha_dir = os.path.join(save_dir, exp_name, "ha_logs")
    os.makedirs(ha_dir, exist_ok=True)

    seed = getattr(args, "seed", -1)
    dataset = getattr(args, "dataset", "unknown")
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{exp_name}_seed{seed}_{timestamp}"

    scope_path = os.path.join(ha_dir, "aggregation_scope_history.csv")
    scope_header = [
        "run_id", "exp_name", "dataset", "seed", "round",
        "selected_scope", "H_select", "zero_denom_ratio",
    ]

    scope_row = [
        run_id,
        exp_name,
        dataset,
        seed,
        round_idx,
        selected_scope,
        f"{H_select:.6f}" if H_select is not None else "",
        f"{zero_denom_ratio:.6f}" if zero_denom_ratio is not None else "",
    ]

    write_header = not os.path.exists(scope_path)
    with open(scope_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(scope_header)
        writer.writerow(scope_row)


def append_ha_parameter_update_stats(
    save_dir,
    exp_name,
    round_idx,
    stats_dict,
    H_select,
    agreement_score,
    candidate_scope,
    locked_scope,
    is_lock_round,
    decision_applied,
    mean_local_ratio_before,
    mean_local_ratio_after,
    args,
    run_id=None,
):
    """Record HA diagnostics for every KDARS/mask parameter-update round."""
    ha_dir = os.path.join(save_dir, exp_name, "ha_logs")
    os.makedirs(ha_dir, exist_ok=True)

    seed = getattr(args, "seed", -1)
    dataset = getattr(args, "dataset", "unknown")
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{exp_name}_seed{seed}_{timestamp}"

    path = os.path.join(ha_dir, "ha_parameter_update_stats.csv")
    header = [
        "run_id", "exp_name", "dataset", "seed", "round",
        "H_select", "candidate_scope", "locked_scope",
        "is_lock_round", "decision_applied", "agreement_score",
        "mean_cos", "std_cos", "negative_ratio", "low_quantile_cos",
        "mean_local_ratio_before", "mean_local_ratio_after",
    ]
    row = [
        run_id, exp_name, dataset, seed, round_idx,
        f"{H_select:.6f}", candidate_scope, locked_scope,
        bool(is_lock_round), bool(decision_applied),
        f"{agreement_score:.6f}",
        f"{stats_dict.get('mean_cos', 0.0):.6f}",
        f"{stats_dict.get('std_cos', 0.0):.6f}",
        f"{stats_dict.get('negative_ratio', 0.0):.6f}",
        f"{stats_dict.get('low_quantile_cos', 0.0):.6f}",
        f"{mean_local_ratio_before:.6f}",
        f"{mean_local_ratio_after:.6f}",
    ]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# ===========================================================================
# 6. HA debug print helper
# ===========================================================================

def print_ha_selection(round_idx, stats_dict, H_select, agreement_score,
                       selected_scope, args):
    """
    Print HA decision details for the selection round.

    Format:
        [FedDyna-HA][Round t] SELECTED: scope=..., H_select=..., threshold=...
    """
    msg = (
        f"[FedDyna-HA][Round {round_idx}] SELECTED: "
        f"scope={selected_scope}, "
        f"H_select={H_select:.4f}, threshold={args.ha_select_threshold:.4f}, "
        f"agreement_score={agreement_score:.4f}, "
        f"mean_cos={stats_dict['mean_cos']:.4f}, "
        f"std_cos={stats_dict['std_cos']:.4f}, "
        f"neg_ratio={stats_dict['negative_ratio']:.4f}, "
        f"q{int(args.ha_quantile*100)}={stats_dict['low_quantile_cos']:.4f}"
    )
    print(msg)


def print_ha_scope_round(round_idx, selected_scope):
    """
    Print the current scope briefly for non-selection rounds.

    Format:
        [FedDyna-HA][Round t] scope=... (locked)
    """
    print(f"[FedDyna-HA][Round {round_idx}] scope={selected_scope} (locked)")
