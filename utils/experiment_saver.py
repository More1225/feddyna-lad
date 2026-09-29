import os
import re
import hashlib
import json

import pandas as pd
import torch

from config.options import dump_effective_args
from utils.save_results import save_results_to_excel


_BASELINE_METHODS = {
    "fedprox",
    "ditto",
    "fedrep",
    "fedpac",
    "fedala",
    "fedapa",
    "fedas",
    "fedltn",
}


def _slug(value, default="na", max_len=80):
    text = str(value) if value is not None else default
    text = text.strip()
    if not text:
        text = default
    text = os.path.splitext(os.path.basename(text))[0] if os.path.sep in text or "/" in text else text
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    if not text:
        text = default
    return text[:max_len].strip("-._") or default


def _compact_name(name, max_len=140):
    if len(name) <= max_len:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    keep = max_len - len(digest) - 1
    return f"{name[:keep].rstrip('-._')}-{digest}"


def _uses_strict_baseline_controls(args):
    return (
        getattr(args, "weighting_method", "fedavg") == "fedavg"
        and getattr(args, "aggregation_scope", "all") == "all"
        and getattr(args, "pruning_method", "none") == "none"
        and not getattr(args, "enable_ha_scope", False)
    )


def _is_plain_fedavg_baseline(args):
    fl_method = getattr(args, "fl_method", "fedavg").lower()
    return fl_method == "fedavg" and _uses_strict_baseline_controls(args)


def infer_experiment_role(args):
    fl_method = getattr(args, "fl_method", "fedavg").lower()
    if fl_method == "local":
        return "local"
    if getattr(args, "enable_ha_scope", False):
        return "ha"
    if fl_method in _BASELINE_METHODS or _is_plain_fedavg_baseline(args):
        return "baseline"
    return "main"


def _method_param_tags(args, role):
    fl_method = getattr(args, "fl_method", "fedavg").lower()
    tags = []

    if role in {"main", "ha"}:
        tags.extend([
            f"agg-{_slug(getattr(args, 'weighting_method', 'unknown'))}",
            f"scope-{_slug(getattr(args, 'aggregation_scope', 'unknown'))}",
            f"prune-{_slug(getattr(args, 'pruning_method', 'unknown'))}",
        ])

    if role == "ha":
        tags.extend([
            f"ha-{_slug(getattr(args, 'ha_scope_mode', 'robust'))}",
            f"thr{getattr(args, 'ha_select_threshold', 'na')}",
            f"halock-first-update{getattr(args, 'lth_epoch_iters', 'na')}",
        ])

    if role == "baseline":
        strict_tag = "strict" if _uses_strict_baseline_controls(args) else "method-only"
        tags.append(strict_tag)

    method_specific = {
        "fedprox": [f"mu{getattr(args, 'fedprox_mu', 'na')}"],
        "ditto": [f"lam{getattr(args, 'ditto_lambda', 'na')}"],
        "fedrep": [
            f"head{getattr(args, 'fedrep_head_epochs', 'na')}",
            f"rep{getattr(args, 'fedrep_rep_epochs', 'na')}",
        ],
        "fedpac": [f"lam{getattr(args, 'fedpac_lam', 'na')}"],
        "fedala": [f"eta{getattr(args, 'fedala_eta', 'na')}"],
        "fedapa": [f"slr{getattr(args, 'fedapa_server_lr', 'na')}"],
        "fedas": [f"align{getattr(args, 'fedas_align_weight', 'na')}"],
        "fedltn": [f"target{getattr(args, 'lth_target_sparsity', 'na')}"],
    }
    tags.extend(method_specific.get(fl_method, []))
    return [_slug(tag, max_len=40) for tag in tags if tag]


def build_run_name(args):
    role = infer_experiment_role(args)
    fl_method = getattr(args, "fl_method", "fedavg").lower()
    method_label = f"base-{fl_method}" if role == "baseline" else f"method-{fl_method}"

    tags = [
        f"ds-{_slug(getattr(args, 'dataset', 'unknown'))}",
        f"data-{_slug(getattr(args, 'data_path', 'unknown'))}",
        f"role-{role}",
        method_label,
    ]
    tags.extend(_method_param_tags(args, role))
    tags.extend([
        f"c{getattr(args, 'num_users', 'na')}",
        f"r{getattr(args, 'com_rounds', 'na')}",
        f"le{getattr(args, 'local_epochs', 'na')}",
        f"seed{getattr(args, 'seed', 'na')}",
    ])

    return _compact_name("_".join(_slug(tag, max_len=80) for tag in tags if tag), max_len=96)


def build_result_file_stem(args):
    role = infer_experiment_role(args)
    fl_method = getattr(args, "fl_method", "fedavg").lower()
    method_label = f"base-{fl_method}" if role == "baseline" else f"method-{fl_method}"
    run_name = getattr(args, "run_name", "") or build_run_name(args)
    digest = hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:8]

    tags = [
        "results",
        f"ds-{_slug(getattr(args, 'dataset', 'unknown'), max_len=20)}",
        f"role-{role}",
        method_label,
    ]
    if role in {"main", "ha"}:
        tags.extend([
            f"agg-{_slug(getattr(args, 'weighting_method', 'unknown'), max_len=20)}",
            f"scope-{_slug(getattr(args, 'aggregation_scope', 'unknown'), max_len=20)}",
            f"prune-{_slug(getattr(args, 'pruning_method', 'unknown'), max_len=24)}",
        ])
    if role == "ha":
        tags.extend([
            f"ha-{_slug(getattr(args, 'ha_scope_mode', 'robust'), max_len=16)}",
            f"thr{getattr(args, 'ha_select_threshold', 'na')}",
            f"halock-first-update{getattr(args, 'lth_epoch_iters', 'na')}",
        ])

    tags.extend([
        f"r{getattr(args, 'com_rounds', 'na')}",
        f"le{getattr(args, 'local_epochs', 'na')}",
        f"seed{getattr(args, 'seed', 'na')}",
        digest,
    ])
    return _compact_name("_".join(_slug(tag, max_len=32) for tag in tags if tag), max_len=72)


def _excel_stem(args):
    return getattr(args, "result_file_stem", None) or build_result_file_stem(args)


def _warn_if_long_path(path, label, max_len=240):
    path_len = len(os.path.abspath(path))
    if path_len > max_len:
        print(f"[ExperimentSaver] WARNING: {label} path length is {path_len}, which may exceed Windows limits: {path}")


def _unique_run_dir(save_root, run_name):
    candidate = os.path.join(save_root, run_name)
    if not os.path.exists(candidate):
        return run_name, candidate

    for index in range(2, 1000):
        candidate_name = f"{run_name}_run{index:03d}"
        candidate = os.path.join(save_root, candidate_name)
        if not os.path.exists(candidate):
            return candidate_name, candidate

    raise RuntimeError(f"Unable to create a unique run directory under {save_root}")


def _build_paths(run_dir):
    return {
        "run_dir": run_dir,
        "models_dir": os.path.join(run_dir, "models"),
        "history_dir": os.path.join(run_dir, "history"),
        "results_dir": os.path.join(run_dir, "results"),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _final_metrics_item(history):
    if not isinstance(history, list):
        return {}
    for item in reversed(history):
        if isinstance(item, dict) and (
            item.get("avg_test_local")
            or item.get("test_metrics_local")
            or item.get("global_test_metrics")
        ):
            return item
    return {}


def _save_run_metric_artifacts(history, args, paths):
    final_item = _final_metrics_item(history)
    avg_local = final_item.get("avg_test_local", {}) or {}
    client_metrics = final_item.get("test_metrics_local", {}) or {}
    global_test_metrics = final_item.get("global_test_metrics", {}) or {}

    metric_names = ["accuracy", "precision", "recall", "f1", "prauc", "auroc"]
    scenario = getattr(args, "scenario_tag", "") or getattr(args, "dataset", "")
    method = getattr(args, "method_tag", "") or getattr(args, "fl_method", "")

    run_metrics = {
        "status": "completed",
        "scenario": scenario,
        "method": method,
        "fl_method": getattr(args, "fl_method", ""),
        "seed": getattr(args, "seed", None),
        "run_dir": paths["run_dir"],
        "num_clients": getattr(args, "num_users", None),
    }

    for metric in metric_names:
        value = avg_local.get(metric)
        run_metrics[metric] = value
        run_metrics[f"mean_client_{metric}"] = value
        run_metrics[f"client_std_{metric}"] = avg_local.get(f"{metric}_std")
        run_metrics[f"client_min_{metric}"] = avg_local.get(f"{metric}_min")
        run_metrics[f"client_max_{metric}"] = avg_local.get(f"{metric}_max")
        if global_test_metrics:
            run_metrics[f"global_test_{metric}"] = global_test_metrics.get(metric)

    run_metrics_path = os.path.join(paths["results_dir"], "run_metrics.json")
    with open(run_metrics_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(run_metrics), f, ensure_ascii=False, indent=2)

    client_rows = []
    for cid, metrics in sorted(client_metrics.items(), key=lambda x: str(x[0])):
        if metrics is None:
            continue
        row = {"Client_ID": cid}
        for metric in metric_names:
            row[metric] = metrics.get(metric)
        client_rows.append(row)
    client_metrics_path = os.path.join(paths["results_dir"], "client_metrics.csv")
    pd.DataFrame(client_rows).to_csv(client_metrics_path, index=False)

    completion_path = os.path.join(paths["run_dir"], "completion.json")
    with open(completion_path, "w", encoding="utf-8") as f:
        json.dump(
            _json_ready({
                "status": "completed",
                "run_metrics": run_metrics_path,
                "client_metrics": client_metrics_path,
            }),
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "run_metrics": run_metrics_path,
        "client_metrics": client_metrics_path,
        "completion": completion_path,
    }


def prepare_experiment_save(args):
    exact_run_dir = getattr(args, "run_dir", None)
    if exact_run_dir:
        run_dir = os.path.abspath(exact_run_dir)
        save_root = os.path.abspath(os.path.dirname(run_dir))
        run_name = os.path.basename(run_dir.rstrip(os.path.sep))
    else:
        save_root = os.path.abspath(getattr(args, "save_dir", "./save"))
        run_name = build_run_name(args)
        run_name, run_dir = _unique_run_dir(save_root, run_name)
    paths = _build_paths(run_dir)

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    args.save_root = save_root
    args.run_name = run_name
    args.result_file_stem = build_result_file_stem(args)
    args.run_dir = run_dir
    args.save_dir = run_dir

    config_path = dump_effective_args(args, run_dir)
    args_path = dump_effective_args(args, run_dir, filename="args.json")
    paths["config_path"] = config_path
    paths["args_path"] = args_path
    _warn_if_long_path(run_dir, "run_dir")
    print(f"[ExperimentSaver] Run name: {run_name}")
    print(f"[ExperimentSaver] Save directory: {run_dir}")
    return paths


def save_local_artifacts(final_local_state_dicts, history, args):
    paths = _build_paths(getattr(args, "run_dir", getattr(args, "save_dir", ".")))
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    model_path = os.path.join(paths["models_dir"], "client_models_final_local.pt")
    history_path = os.path.join(paths["history_dir"], "train_history.pt")

    torch.save(final_local_state_dicts, model_path)
    torch.save(history, history_path)

    excel_path = save_results_to_excel(
        history=history,
        client_masks={},
        args=args,
        save_dir=paths["results_dir"],
        run_name=_excel_stem(args),
    )

    paths.update({
        "client_models_final_local": model_path,
        "train_history": history_path,
        "excel_path": excel_path,
    })
    paths.update(_save_run_metric_artifacts(history, args, paths))
    return paths


def save_federated_artifacts(
    final_global_model,
    final_local_state_dicts,
    client_post_broadcast_state_dicts,
    client_masks,
    history,
    args,
):
    paths = _build_paths(getattr(args, "run_dir", getattr(args, "save_dir", ".")))
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    global_model_path = os.path.join(paths["models_dir"], "global_model.pt")
    local_models_path = os.path.join(paths["models_dir"], "client_models_final_local.pt")
    post_broadcast_path = os.path.join(paths["models_dir"], "client_models_post_broadcast.pt")
    masks_path = os.path.join(paths["models_dir"], "client_masks.pt")
    history_path = os.path.join(paths["history_dir"], "train_history.pt")

    torch.save(final_global_model.state_dict(), global_model_path)
    torch.save(final_local_state_dicts, local_models_path)
    torch.save(client_post_broadcast_state_dicts, post_broadcast_path)
    torch.save(client_masks, masks_path)
    torch.save(history, history_path)

    excel_path = save_results_to_excel(
        history=history,
        client_masks=client_masks,
        args=args,
        save_dir=paths["results_dir"],
        run_name=_excel_stem(args),
    )

    paths.update({
        "global_model": global_model_path,
        "client_models_final_local": local_models_path,
        "client_models_post_broadcast": post_broadcast_path,
        "client_masks": masks_path,
        "train_history": history_path,
        "excel_path": excel_path,
    })
    paths.update(_save_run_metric_artifacts(history, args, paths))
    return paths
