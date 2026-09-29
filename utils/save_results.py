import os
import json
import pandas as pd
import numpy as np


def _safe_get(d: dict, key: str, default=np.nan):
    if d is None:
        return default
    return d.get(key, default)


def _build_round_client_metrics_df(history):
    rows = []

    for round_item in history:
        round_num = round_item.get("round", -1)
        client_metrics = round_item.get("client_metrics", {})
        # FedDyna-HA: extract aggregation_scope from the history.
        ha_scope = round_item.get("aggregation_scope", None)

        for client_id, metric_pack in client_metrics.items():
            local_metrics = metric_pack.get("local", {})
            global_prev_metrics = metric_pack.get("global_prev", {})
            post_global_metrics = metric_pack.get("post_global", {})
            train_eval = metric_pack.get("train_eval", {})
            mask_stats = metric_pack.get("mask_stats", {})
            data_stats = metric_pack.get("data_stats", {})

            rows.append({
                "Round": round_num,
                "Client_ID": client_id,

                "Train_Loss": _safe_get(metric_pack, "train_loss"),
                "Train_Eval_Loss": _safe_get(train_eval, "loss"),
                "Train_Eval_Acc": _safe_get(train_eval, "acc"),

                "Local_Accuracy": _safe_get(local_metrics, "accuracy"),
                "Local_Precision": _safe_get(local_metrics, "precision"),
                "Local_Recall": _safe_get(local_metrics, "recall"),
                "Local_F1": _safe_get(local_metrics, "f1"),
                "Local_PRAUC": _safe_get(local_metrics, "prauc"),
                "Local_AUROC": _safe_get(local_metrics, "auroc"),

                "GlobalPrev_Accuracy": _safe_get(global_prev_metrics, "accuracy"),
                "GlobalPrev_Precision": _safe_get(global_prev_metrics, "precision"),
                "GlobalPrev_Recall": _safe_get(global_prev_metrics, "recall"),
                "GlobalPrev_F1": _safe_get(global_prev_metrics, "f1"),
                "GlobalPrev_PRAUC": _safe_get(global_prev_metrics, "prauc"),
                "GlobalPrev_AUROC": _safe_get(global_prev_metrics, "auroc"),

                "PostGlobal_Accuracy": _safe_get(post_global_metrics, "accuracy"),
                "PostGlobal_Precision": _safe_get(post_global_metrics, "precision"),
                "PostGlobal_Recall": _safe_get(post_global_metrics, "recall"),
                "PostGlobal_F1": _safe_get(post_global_metrics, "f1"),
                "PostGlobal_PRAUC": _safe_get(post_global_metrics, "prauc"),
                "PostGlobal_AUROC": _safe_get(post_global_metrics, "auroc"),

                "Delta_F1": _safe_get(metric_pack, "delta_f1"),

                "Mask_Global_Params": _safe_get(mask_stats, "global_params"),
                "Mask_Local_Params": _safe_get(mask_stats, "local_params"),
                "Mask_Global_Ratio": _safe_get(mask_stats, "global_ratio"),
                "Mask_Local_Ratio": _safe_get(mask_stats, "local_ratio"),

                "Sample_Count": _safe_get(data_stats, "sample_count"),
                "Normal_Count": _safe_get(data_stats, "normal_count"),
                "Anomaly_Count": _safe_get(data_stats, "anomaly_count"),
                "Anomaly_Ratio": _safe_get(data_stats, "anomaly_ratio"),

                "Aggregation_Scope": ha_scope if ha_scope is not None else "",

                "Plot_Path": _safe_get(metric_pack, "plot_path", ""),
            })

    return pd.DataFrame(rows)


def _build_epoch_curve_df(history):
    rows = []

    for round_item in history:
        round_num = round_item.get("round", -1)
        client_metrics = round_item.get("client_metrics", {})

        for client_id, metric_pack in client_metrics.items():
            losses = metric_pack.get("epoch_losses", [])
            accs = metric_pack.get("epoch_accs", [])
            grads = metric_pack.get("grad_norms", [])

            max_len = max(len(losses), len(accs), len(grads), 0)

            for idx in range(max_len):
                rows.append({
                    "Round": round_num,
                    "Client_ID": client_id,
                    "Epoch": idx + 1,
                    "Epoch_Loss": losses[idx] if idx < len(losses) else np.nan,
                    "Epoch_Acc": accs[idx] if idx < len(accs) else np.nan,
                    "Grad_Norm": grads[idx] if idx < len(grads) else np.nan,
                })

    return pd.DataFrame(rows)


def _build_round_summary_df(df_metrics: pd.DataFrame):
    if df_metrics.empty:
        return pd.DataFrame()

    rows = []
    for round_num in sorted(df_metrics["Round"].unique()):
        round_df = df_metrics[df_metrics["Round"] == round_num]

        rows.append({
            "Round": round_num,

            "Avg_Train_Loss": round_df["Train_Loss"].mean(),
            "Std_Train_Loss": round_df["Train_Loss"].std(),

            "Avg_Local_Accuracy": round_df["Local_Accuracy"].mean(),
            "Std_Local_Accuracy": round_df["Local_Accuracy"].std(),
            "Avg_Local_Precision": round_df["Local_Precision"].mean(),
            "Std_Local_Precision": round_df["Local_Precision"].std(),
            "Avg_Local_Recall": round_df["Local_Recall"].mean(),
            "Std_Local_Recall": round_df["Local_Recall"].std(),
            "Avg_Local_F1": round_df["Local_F1"].mean(),
            "Std_Local_F1": round_df["Local_F1"].std(),
            "Avg_Local_PRAUC": round_df["Local_PRAUC"].mean(),
            "Std_Local_PRAUC": round_df["Local_PRAUC"].std(),
            "Avg_Local_AUROC": round_df["Local_AUROC"].mean(),
            "Std_Local_AUROC": round_df["Local_AUROC"].std(),

            "Avg_GlobalPrev_F1": round_df["GlobalPrev_F1"].mean(),
            "Avg_PostGlobal_F1": round_df["PostGlobal_F1"].mean(),
            "Std_PostGlobal_F1": round_df["PostGlobal_F1"].std(),
            "Avg_PostGlobal_AUROC": round_df["PostGlobal_AUROC"].mean(),
            "Std_PostGlobal_AUROC": round_df["PostGlobal_AUROC"].std(),

            "Avg_Delta_F1": round_df["Delta_F1"].mean(),
            "Std_Delta_F1": round_df["Delta_F1"].std(),

            "Avg_Mask_Global_Ratio": round_df["Mask_Global_Ratio"].mean(),
            "Avg_Mask_Local_Ratio": round_df["Mask_Local_Ratio"].mean(),

            "Best_Client_By_Local_F1": (
                round_df.loc[round_df["Local_F1"].idxmax(), "Client_ID"]
                if not round_df["Local_F1"].isna().all() else np.nan
            ),
            "Best_Local_F1": round_df["Local_F1"].max(),

            "Best_Client_By_PostGlobal_F1": (
                round_df.loc[round_df["PostGlobal_F1"].idxmax(), "Client_ID"]
                if not round_df["PostGlobal_F1"].isna().all() else np.nan
            ),
            "Best_PostGlobal_F1": round_df["PostGlobal_F1"].max(),
        })

    return pd.DataFrame(rows)


def _build_client_summary_df(df_metrics: pd.DataFrame):
    if df_metrics.empty:
        return pd.DataFrame()

    rows = []
    for client_id in sorted(df_metrics["Client_ID"].unique()):
        client_df = df_metrics[df_metrics["Client_ID"] == client_id]

        rows.append({
            "Client_ID": client_id,

            "Avg_Train_Loss": client_df["Train_Loss"].mean(),
            "Std_Train_Loss": client_df["Train_Loss"].std(),

            "Avg_Local_Accuracy": client_df["Local_Accuracy"].mean(),
            "Std_Local_Accuracy": client_df["Local_Accuracy"].std(),
            "Avg_Local_Precision": client_df["Local_Precision"].mean(),
            "Std_Local_Precision": client_df["Local_Precision"].std(),
            "Avg_Local_Recall": client_df["Local_Recall"].mean(),
            "Std_Local_Recall": client_df["Local_Recall"].std(),
            "Avg_Local_F1": client_df["Local_F1"].mean(),
            "Std_Local_F1": client_df["Local_F1"].std(),
            "Avg_Local_PRAUC": client_df["Local_PRAUC"].mean(),
            "Std_Local_PRAUC": client_df["Local_PRAUC"].std(),
            "Avg_Local_AUROC": client_df["Local_AUROC"].mean(),
            "Std_Local_AUROC": client_df["Local_AUROC"].std(),

            "Avg_GlobalPrev_F1": client_df["GlobalPrev_F1"].mean(),
            "Avg_PostGlobal_F1": client_df["PostGlobal_F1"].mean(),
            "Std_PostGlobal_F1": client_df["PostGlobal_F1"].std(),
            "Avg_PostGlobal_AUROC": client_df["PostGlobal_AUROC"].mean(),
            "Std_PostGlobal_AUROC": client_df["PostGlobal_AUROC"].std(),

            "Avg_Delta_F1": client_df["Delta_F1"].mean(),
            "Std_Delta_F1": client_df["Delta_F1"].std(),

            "Avg_Mask_Global_Ratio": client_df["Mask_Global_Ratio"].mean(),
            "Avg_Mask_Local_Ratio": client_df["Mask_Local_Ratio"].mean(),

            "Best_Round_By_Local_F1": (
                client_df.loc[client_df["Local_F1"].idxmax(), "Round"]
                if not client_df["Local_F1"].isna().all() else np.nan
            ),
            "Best_Local_F1": client_df["Local_F1"].max(),

            "Best_Round_By_PostGlobal_F1": (
                client_df.loc[client_df["PostGlobal_F1"].idxmax(), "Round"]
                if not client_df["PostGlobal_F1"].isna().all() else np.nan
            ),
            "Best_PostGlobal_F1": client_df["PostGlobal_F1"].max(),
        })

    return pd.DataFrame(rows)


def _build_client_weights_df(history):
    rows = []
    for round_item in history:
        round_num = round_item.get("round", -1)
        client_weights = round_item.get("client_weights", {})
        for client_id, weight in client_weights.items():
            rows.append({
                "Round": round_num,
                "Client_ID": client_id,
                "Weight": weight,
            })
    return pd.DataFrame(rows)


def _build_post_global_avg_df(history):
    rows = []
    for round_item in history:
        round_num = round_item.get("round", -1)
        avg_post_global = round_item.get("avg_post_global_metrics", {})
        rows.append({
            "Round": round_num,
            "Avg_PostGlobal_Accuracy": _safe_get(avg_post_global, "accuracy"),
            "Std_PostGlobal_Accuracy": _safe_get(avg_post_global, "accuracy_std"),
            "Avg_PostGlobal_Precision": _safe_get(avg_post_global, "precision"),
            "Std_PostGlobal_Precision": _safe_get(avg_post_global, "precision_std"),
            "Avg_PostGlobal_Recall": _safe_get(avg_post_global, "recall"),
            "Std_PostGlobal_Recall": _safe_get(avg_post_global, "recall_std"),
            "Avg_PostGlobal_F1": _safe_get(avg_post_global, "f1"),
            "Std_PostGlobal_F1": _safe_get(avg_post_global, "f1_std"),
            "Avg_PostGlobal_PRAUC": _safe_get(avg_post_global, "prauc"),
            "Std_PostGlobal_PRAUC": _safe_get(avg_post_global, "prauc_std"),
            "Avg_PostGlobal_AUROC": _safe_get(avg_post_global, "auroc"),
            "Std_PostGlobal_AUROC": _safe_get(avg_post_global, "auroc_std"),
        })
    return pd.DataFrame(rows)


def _build_mask_summary_df(client_masks):
    rows = []

    for client_id, mask_dict in client_masks.items():
        client_row = {"Client_ID": client_id}

        total_global = 0
        total_local = 0
        total_params = 0

        for layer_name, mask_tensor in mask_dict.items():
            total = mask_tensor.numel()
            local_count = int(mask_tensor.sum().item())
            global_count = total - local_count

            total_params += total
            total_global += global_count
            total_local += local_count

            safe_layer_name = str(layer_name).replace("/", "_")

            client_row[f"{safe_layer_name}_Total"] = total
            client_row[f"{safe_layer_name}_Global"] = global_count
            client_row[f"{safe_layer_name}_Local"] = local_count
            client_row[f"{safe_layer_name}_Global_Ratio"] = global_count / total if total > 0 else 0.0
            client_row[f"{safe_layer_name}_Local_Ratio"] = local_count / total if total > 0 else 0.0

        client_row["All_Total"] = total_params
        client_row["All_Global"] = total_global
        client_row["All_Local"] = total_local
        client_row["All_Global_Ratio"] = total_global / total_params if total_params > 0 else 0.0
        client_row["All_Local_Ratio"] = total_local / total_params if total_params > 0 else 0.0

        rows.append(client_row)

    return pd.DataFrame(rows)


def _build_config_df(args):
    config_dict = vars(args).copy()
    rows = [{"Key": k, "Value": json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}
            for k, v in config_dict.items()]
    return pd.DataFrame(rows)


def _build_test_metrics_df(history):
    """Build final test metrics table."""
    if not history:
        return pd.DataFrame()

    last_item = history[-1]

    test_metrics_global = last_item.get("test_metrics_global", {})
    test_metrics_local = last_item.get("test_metrics_local", {})
    avg_test_global = last_item.get("avg_test_global", {})
    avg_test_local = last_item.get("avg_test_local", {})

    if not test_metrics_local and not test_metrics_global and not avg_test_local and not avg_test_global:
        return pd.DataFrame()

    client_ids = sorted(set(test_metrics_local.keys()) | set(test_metrics_global.keys()))
    rows = []

    has_global_per_client = bool(test_metrics_global)

    for cid in client_ids:
        local_metrics = test_metrics_local.get(cid, {})
        global_metrics = test_metrics_global.get(cid, {})

        row = {
            "Client_ID": cid,
            "Local_Accuracy": _safe_get(local_metrics, "accuracy") if local_metrics else np.nan,
            "Local_Precision": _safe_get(local_metrics, "precision") if local_metrics else np.nan,
            "Local_Recall": _safe_get(local_metrics, "recall") if local_metrics else np.nan,
            "Local_F1": _safe_get(local_metrics, "f1") if local_metrics else np.nan,
            "Local_PRAUC": _safe_get(local_metrics, "prauc") if local_metrics else np.nan,
            "Local_AUROC": _safe_get(local_metrics, "auroc") if local_metrics else np.nan,
        }

        if has_global_per_client:
            row.update({
                "Global_Accuracy": _safe_get(global_metrics, "accuracy") if global_metrics else np.nan,
                "Global_Precision": _safe_get(global_metrics, "precision") if global_metrics else np.nan,
                "Global_Recall": _safe_get(global_metrics, "recall") if global_metrics else np.nan,
                "Global_F1": _safe_get(global_metrics, "f1") if global_metrics else np.nan,
                "Global_PRAUC": _safe_get(global_metrics, "prauc") if global_metrics else np.nan,
                "Global_AUROC": _safe_get(global_metrics, "auroc") if global_metrics else np.nan,
            })

        rows.append(row)

    if avg_test_local or avg_test_global:
        row = {"Client_ID": "Average"}
        if avg_test_local:
            row.update({
                "Local_Accuracy": avg_test_local.get("accuracy", np.nan),
                "Local_Precision": avg_test_local.get("precision", np.nan),
                "Local_Recall": avg_test_local.get("recall", np.nan),
                "Local_F1": avg_test_local.get("f1", np.nan),
                "Local_PRAUC": avg_test_local.get("prauc", np.nan),
                "Local_AUROC": avg_test_local.get("auroc", np.nan),
            })
        if has_global_per_client and avg_test_global:
            row.update({
                "Global_Accuracy": avg_test_global.get("accuracy", np.nan),
                "Global_Precision": avg_test_global.get("precision", np.nan),
                "Global_Recall": avg_test_global.get("recall", np.nan),
                "Global_F1": avg_test_global.get("f1", np.nan),
                "Global_PRAUC": avg_test_global.get("prauc", np.nan),
                "Global_AUROC": avg_test_global.get("auroc", np.nan),
            })
        rows.append(row)

    if avg_test_local:
        f1_val = avg_test_local.get('f1', np.nan)
        f1_std = avg_test_local.get('f1_std', np.nan)
        auroc_val = avg_test_local.get('auroc', np.nan)
        auroc_std = avg_test_local.get('auroc_std', np.nan)
        prauc_val = avg_test_local.get('prauc', np.nan)
        prauc_std = avg_test_local.get('prauc_std', np.nan)

        rows.extend([
            {"Client_ID": "Local F1 (mean +/- std)", "Local_F1": f"{f1_val:.4f} +/- {f1_std:.4f}" if not np.isnan(f1_val) else np.nan},
            {"Client_ID": "Local AUROC (mean +/- std)", "Local_AUROC": f"{auroc_val:.4f} +/- {auroc_std:.4f}" if not np.isnan(auroc_val) else np.nan},
            {"Client_ID": "Local PRAUC (mean +/- std)", "Local_PRAUC": f"{prauc_val:.4f} +/- {prauc_std:.4f}" if not np.isnan(prauc_val) else np.nan},
            {"Client_ID": "Local Best F1", "Local_F1": avg_test_local.get("f1_max", np.nan)},
            {"Client_ID": "Local Worst F1", "Local_F1": avg_test_local.get("f1_min", np.nan)},
            {"Client_ID": "Local Best AUROC", "Local_AUROC": avg_test_local.get("auroc_max", np.nan)},
            {"Client_ID": "Local Worst AUROC", "Local_AUROC": avg_test_local.get("auroc_min", np.nan)},
        ])

    if has_global_per_client and avg_test_global:
        g_f1_val = avg_test_global.get('f1', np.nan)
        g_f1_std = avg_test_global.get('f1_std', np.nan)
        g_auroc_val = avg_test_global.get('auroc', np.nan)
        g_auroc_std = avg_test_global.get('auroc_std', np.nan)

        rows.extend([
            {"Client_ID": "Global F1 (mean +/- std)", "Global_F1": f"{g_f1_val:.4f} +/- {g_f1_std:.4f}" if not np.isnan(g_f1_val) else np.nan},
            {"Client_ID": "Global AUROC (mean +/- std)", "Global_AUROC": f"{g_auroc_val:.4f} +/- {g_auroc_std:.4f}" if not np.isnan(g_auroc_val) else np.nan},
            {"Client_ID": "Global Best F1", "Global_F1": avg_test_global.get("f1_max", np.nan)},
            {"Client_ID": "Global Worst F1", "Global_F1": avg_test_global.get("f1_min", np.nan)},
            {"Client_ID": "Global Best AUROC", "Global_AUROC": avg_test_global.get("auroc_max", np.nan)},
            {"Client_ID": "Global Worst AUROC", "Global_AUROC": avg_test_global.get("auroc_min", np.nan)},
        ])

    return pd.DataFrame(rows)


def _build_global_test_metrics_df(history):
    if not history:
        return pd.DataFrame()

    last_item = history[-1]
    if "global_test_metrics" not in last_item:
        return pd.DataFrame()

    global_test_metrics = last_item.get("global_test_metrics", {})
    global_test_size = last_item.get("global_test_size", 0)

    rows = [{
        "Test_Set": "Global_Test_Set",
        "Sample_Size": global_test_size,
        "Accuracy": _safe_get(global_test_metrics, "accuracy"),
        "Precision": _safe_get(global_test_metrics, "precision"),
        "Recall": _safe_get(global_test_metrics, "recall"),
        "F1": _safe_get(global_test_metrics, "f1"),
        "PRAUC": _safe_get(global_test_metrics, "prauc"),
        "AUROC": _safe_get(global_test_metrics, "auroc"),
    }]

    return pd.DataFrame(rows)


def _build_communication_df(history):
    """Build communication overhead statistics table."""
    if not history:
        return pd.DataFrame(), pd.DataFrame()

    round_rows = []
    for round_item in history:
        comm_stats = round_item.get("communication_stats", {})
        if comm_stats:
            round_rows.append({
                "Accounting_Basis": comm_stats.get("accounting_basis", "legacy tensor estimate"),
                "Accounting_Version": comm_stats.get("accounting_version", "legacy_v1"),
                "MB_Divisor": comm_stats.get("mb_divisor", 1024 * 1024),
                "Upload_Bytes": comm_stats.get("upload_bytes"),
                "Download_Bytes": comm_stats.get("download_bytes"),
                "Total_Bytes": comm_stats.get("total_bytes"),
                "Round": comm_stats.get("round", -1),
                "FL_Method": comm_stats.get("fl_method", ""),
                "Uplink_MB": comm_stats.get("uplink_mb", 0.0),
                "Downlink_MB": comm_stats.get("downlink_mb", 0.0),
                "Total_MB": comm_stats.get("total_mb", 0.0),
                "Num_Clients": comm_stats.get("num_clients", 0),
                "Avg_Uplink_Per_Client_MB": comm_stats.get("avg_uplink_per_client_mb", 0.0),
                "Avg_Downlink_Per_Client_MB": comm_stats.get("avg_downlink_per_client_mb", 0.0),
            })

    summary_rows = []
    last_item = history[-1] if history else {}
    comm_summary = last_item.get("communication_summary", {})
    if comm_summary:
        summary_rows.append({"Metric": "Total Training Time (sec)", "Value": comm_summary.get("total_training_time_sec", 0.0)})
        summary_rows.append({"Metric": "Total Training Time (min)", "Value": comm_summary.get("total_training_time_min", 0.0)})
        summary_rows.append({"Metric": "Total Uplink (MB)", "Value": comm_summary.get("total_uplink_mb", 0.0)})
        summary_rows.append({"Metric": "Total Downlink (MB)", "Value": comm_summary.get("total_downlink_mb", 0.0)})
        summary_rows.append({"Metric": "Total Communication (MB)", "Value": comm_summary.get("total_communication_mb", 0.0)})
        summary_rows.append({"Metric": "Number of Rounds", "Value": comm_summary.get("num_rounds", 0)})
        summary_rows.append({"Metric": "Avg Uplink Per Round (MB)", "Value": comm_summary.get("avg_uplink_per_round_mb", 0.0)})
        summary_rows.append({"Metric": "Avg Downlink Per Round (MB)", "Value": comm_summary.get("avg_downlink_per_round_mb", 0.0)})

    if comm_summary.get("accounting_version"):
        for key in ("accounting_basis", "accounting_version", "mb_divisor",
                    "upload_bytes", "download_bytes", "total_bytes",
                    "initialization_policy", "final_download_included"):
            summary_rows.append({"Metric": key, "Value": comm_summary[key]})
        for key, value in comm_summary["initialization"].items():
            summary_rows.append({"Metric": "initialization_" + key, "Value": value})

    return pd.DataFrame(round_rows), pd.DataFrame(summary_rows)


def _build_training_time_df(history):
    """Build training time statistics table."""
    if not history:
        return pd.DataFrame()

    rows = []

    # Check whether this is the local method.
    for i, item in enumerate(history):
        if item.get("method") == "local":
            # Add total training-time information.
            rows.append({
                "Method": "local",
                "Round": i,
                "Client_ID": "Overall",
                "Total_Training_Time_Sec": item.get("total_training_time_sec", 0),
                "Avg_Training_Time_Sec": item.get("avg_training_time_sec", 0),
                "Total_Training_Time_Min": item.get("total_training_time_min", 0),
                "Avg_Training_Time_Min": item.get("avg_training_time_min", 0),
            })

            # Add per-client training times.
            client_times = item.get("client_training_times", {})
            for client_id, training_time in client_times.items():
                rows.append({
                    "Method": "local",
                    "Round": i,
                    "Client_ID": client_id,
                    "Training_Time_Sec": training_time,
                    "Training_Time_Min": training_time / 60.0,
                })

    return pd.DataFrame(rows)


def _add_pivot_sheets(writer, df_metrics: pd.DataFrame):
    if df_metrics.empty:
        return

    pivot_metrics = [
        "Local_Accuracy",
        "Local_Precision",
        "Local_Recall",
        "Local_F1",
        "Local_AUROC",
        "GlobalPrev_F1",
        "PostGlobal_F1",
        "Train_Loss",
        "Delta_F1",
        "Mask_Global_Ratio",
        "Mask_Local_Ratio",
    ]

    for metric in pivot_metrics:
        if metric not in df_metrics.columns:
            continue

        pivot_df = df_metrics.pivot_table(
            index="Round",
            columns="Client_ID",
            values=metric,
            aggfunc="mean"
        )
        pivot_df.columns = [f"Client_{int(col)}" for col in pivot_df.columns]
        sheet_name = metric[:31]
        pivot_df.to_excel(writer, sheet_name=sheet_name)


def _build_ha_state_df(history):
    """Build a summary table of aggregation-scope selector states from the history."""
    rows = []
    for round_item in history:
        ha_state = round_item.get("ha_state", None)
        if ha_state is None:
            continue
        rows.append({
            "Round": round_item.get("round", -1),
            "HA_Selected_Scope": ha_state.get("selected_scope", ""),
            "HA_H_Select": ha_state.get("H_select", np.nan),
            "Aggregation_Scope": round_item.get("aggregation_scope", ""),
            "Zero_Denom_Ratio": round_item.get("zero_denom_ratio", np.nan),
        })
    return pd.DataFrame(rows)


def save_results_to_excel(history, client_masks, args, save_dir=None, run_name=None):
    """Save results to Excel file."""
    if save_dir is None:
        save_dir = getattr(args, "save_dir", ".")

    os.makedirs(save_dir, exist_ok=True)

    df_metrics = _build_round_client_metrics_df(history)
    df_epoch = _build_epoch_curve_df(history)
    df_round_summary = _build_round_summary_df(df_metrics)
    df_client_summary = _build_client_summary_df(df_metrics)
    df_weights = _build_client_weights_df(history)
    df_post_global_avg = _build_post_global_avg_df(history)
    df_masks = _build_mask_summary_df(client_masks)
    df_config = _build_config_df(args)
    df_test = _build_test_metrics_df(history)
    df_global_test = _build_global_test_metrics_df(history)
    df_comm_rounds, df_comm_summary = _build_communication_df(history)
    df_training_time = _build_training_time_df(history)
    df_ha_state = _build_ha_state_df(history)

    num_clients = getattr(args, "num_users", len(client_masks))
    fl_method = getattr(args, "fl_method", "fedavg")

    # HA marker: append _HA to the filename when enabled.
    ha_tag = "_HA" if getattr(args, "enable_ha_scope", False) else ""

    if run_name:
        safe_run_name = str(run_name).replace(os.path.sep, "_")
        if os.path.altsep:
            safe_run_name = safe_run_name.replace(os.path.altsep, "_")
        excel_name = safe_run_name if safe_run_name.lower().endswith(".xlsx") else f"{safe_run_name}.xlsx"
        filename = os.path.join(save_dir, excel_name)
    else:
        filename = os.path.join(
            save_dir,
            f"{fl_method}_results_{args.dataset}"
            f"_clients{num_clients}"
            f"_agg-{getattr(args, 'weighting_method', 'unknown')}"
            f"_scope-{getattr(args, 'aggregation_scope', 'unknown')}"
            f"_prune-{getattr(args, 'pruning_method', 'unknown')}"
            f"_rounds{args.com_rounds}{ha_tag}.xlsx"
        )

    print(f"[SaveResults] Saving to file: {filename}")
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        if not df_metrics.empty:
            df_metrics.to_excel(writer, sheet_name="Round_Client_Metrics", index=False)

        if not df_epoch.empty:
            df_epoch.to_excel(writer, sheet_name="Epoch_Curves", index=False)

        if not df_round_summary.empty:
            df_round_summary.to_excel(writer, sheet_name="Round_Summary", index=False)

        if not df_client_summary.empty:
            df_client_summary.to_excel(writer, sheet_name="Client_Summary", index=False)

        if not df_weights.empty:
            df_weights.to_excel(writer, sheet_name="Client_Weights", index=False)

        if not df_post_global_avg.empty:
            df_post_global_avg.to_excel(writer, sheet_name="PostGlobal_Avg", index=False)

        if not df_masks.empty:
            df_masks.to_excel(writer, sheet_name="Mask_Summary", index=False)

        if not df_config.empty:
            df_config.to_excel(writer, sheet_name="Config", index=False)

        if not df_test.empty:
            df_test.to_excel(writer, sheet_name="Test_Metrics", index=False)

        if not df_global_test.empty:
            df_global_test.to_excel(writer, sheet_name="Global_Test_Metrics", index=False)

        if not df_comm_rounds.empty:
            df_comm_rounds.to_excel(writer, sheet_name="Communication_Rounds", index=False)

        if not df_comm_summary.empty:
            df_comm_summary.to_excel(writer, sheet_name="Communication_Summary", index=False)

        if not df_training_time.empty:
            df_training_time.to_excel(writer, sheet_name="Training_Time", index=False)

        if not df_ha_state.empty:
            df_ha_state.to_excel(writer, sheet_name="HA_State", index=False)

        _add_pivot_sheets(writer, df_metrics)

    print(f"Results saved to: {filename}")
    return filename
