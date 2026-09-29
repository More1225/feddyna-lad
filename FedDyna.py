import copy
import glob
import json
import os
import random

# CuBLAS requires this setting before PyTorch initializes CUDA in order to
# provide deterministic kernels. Preserve an explicit user-provided override.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config.options import parse_args
from data.train_utils import DatasetSplit, get_data, collate_fn_with_lengths
from federated.round_runner import evaluate_model, run_federated_training
from federated.local_runner import run_local_training
from models.lightlog import LightLogModel
from utils.experiment_saver import (
    prepare_experiment_save,
    save_federated_artifacts,
    save_local_artifacts,
)


def setup_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels improve reproducibility but may slow CUDA training.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pick_existing_path(explicit_path, candidates):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_global_testset(args):
    data_dir = os.path.dirname(args.data_path) if getattr(args, "data_path", "") else os.path.join("data", args.dataset.upper(), "fed_data")

    vocab_path = _pick_existing_path(
        getattr(args, "vocab_path", None),
        sorted(glob.glob(os.path.join(data_dir, "vocab*.json"))),
    )
    embed_path = _pick_existing_path(
        getattr(args, "embed_path", None),
        sorted(glob.glob(os.path.join(data_dir, "event_vectors*.npy"))),
    )
    npz_path = _pick_existing_path(
        getattr(args, "global_test_npz_path", None),
        [getattr(args, "data_path", None)] + sorted(glob.glob(os.path.join(data_dir, "*_fed_unified*.npz"))),
    )
    test_csv_path = _pick_existing_path(
        getattr(args, "global_test_csv_path", None),
        [os.path.join(data_dir, "log_test.csv")],
    )

    required_paths = {
        "vocab": vocab_path,
        "embedding": embed_path,
        "npz": npz_path,
        "global_test_csv": test_csv_path,
    }
    missing = [name for name, path in required_paths.items() if path is None]
    if missing:
        print(f"Warning: Missing files required for global test evaluation: {missing}. Skipping evaluation.")
        return None, 0

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    event_embeddings = np.load(embed_path)
    pad_vector = np.zeros((1, event_embeddings.shape[1]), dtype=np.float32)
    event_embeddings_with_pad = np.vstack([pad_vector, event_embeddings])

    npz_data = np.load(npz_path, allow_pickle=True)
    max_seq_len = int(npz_data.get("max_seq_len", 50))

    import pandas as pd

    test_df = pd.read_csv(test_csv_path)
    required_columns = {"sequence", "label"}
    if not required_columns.issubset(set(test_df.columns)):
        print(f"Warning: {test_csv_path} is missing columns {required_columns}. Skipping global test evaluation.")
        return None, 0

    def seq_to_tensor(seq_str):
        if pd.isna(seq_str) or seq_str == "":
            events = []
        else:
            events = list(map(int, str(seq_str).split()))

        if len(events) > max_seq_len:
            events = events[:max_seq_len]
        else:
            events = events + [0] * (max_seq_len - len(events))

        indices = [vocab.get(str(e), 0) for e in events]
        tensor = event_embeddings_with_pad[indices]
        return torch.from_numpy(tensor).float()

    class GlobalTestDataset(Dataset):
        def __init__(self, dataframe):
            self.sequences = dataframe["sequence"].tolist()
            self.labels = dataframe["label"].values.astype(np.int64)

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return seq_to_tensor(self.sequences[idx]), self.labels[idx]

    dataset = GlobalTestDataset(test_df)
    batch_size = getattr(args, "local_bs", getattr(args, "batch_size", 64))
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"Loaded global test set with {len(dataset)} samples.")
    return test_loader, len(dataset)


def load_model(args):
    if args.model.lower() != "lightlog":
        raise ValueError(f"Unsupported model: {args.model}")

    # Prefer the FedPAC-compatible model for FedPAC and FedAS comparisons.
    fl_method = getattr(args, 'fl_method', '').lower()
    if fl_method in ['fedpac', 'fedas']:
        try:
            from models.fedpac_lightlog import create_fedpac_model
            print(f"Using the FedPAC-compatible model for the {fl_method} method.")
            return create_fedpac_model(args)
        except ImportError:
            print("Warning: FedPAC-compatible model unavailable; using the base model with automatic enhancements.")

    return LightLogModel(
        num_classes=args.num_classes,
        filters_list=args.filters_list,
        input_dim=args.input_dim,
        dropout=args.dropout,
        compress_dim=args.compress_dim,
    )


def compute_statistics_metrics(metrics_dict):
    """
    Compute summary statistics (mean, standard deviation, minimum, and maximum)

    Supported metrics: accuracy, precision, recall, f1, prauc, and auroc.
    """
    if not metrics_dict:
        return {
            "accuracy": 0.0, "accuracy_std": 0.0, "accuracy_max": 0.0, "accuracy_min": 0.0,
            "precision": 0.0, "precision_std": 0.0, "precision_max": 0.0, "precision_min": 0.0,
            "recall": 0.0, "recall_std": 0.0, "recall_max": 0.0, "recall_min": 0.0,
            "f1": 0.0, "f1_std": 0.0, "f1_max": 0.0, "f1_min": 0.0,
            "prauc": 0.0, "prauc_std": 0.0, "prauc_max": 0.0, "prauc_min": 0.0,
            "auroc": 0.0, "auroc_std": 0.0, "auroc_max": 0.0, "auroc_min": 0.0,
        }

    metric_names = ["accuracy", "precision", "recall", "f1", "prauc", "auroc"]
    result = {}

    for metric in metric_names:
        values = [m.get(metric, 0.0) for m in metrics_dict.values() if m is not None]
        if values:
            result[metric] = np.mean(values)
            result[f"{metric}_std"] = np.std(values)
            result[f"{metric}_max"] = np.max(values)
            result[f"{metric}_min"] = np.min(values)
        else:
            result[metric] = 0.0
            result[f"{metric}_std"] = 0.0
            result[f"{metric}_max"] = 0.0
            result[f"{metric}_min"] = 0.0

    return result


def format_mean_std(mean, std):
    """Format a mean and standard deviation as a compact string."""
    return f"{mean:.4f} ± {std:.4f}"


def evaluate_final_local_models(model_template, dataset_test, dict_users_test, final_local_state_dicts, args):
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    client_ids = sorted(dict_users_test.keys())
    test_metrics_local = {}

    print("\n" + "=" * 80)
    print("Final test evaluation (final local models only)")
    print("=" * 80)

    for cid in client_ids:
        test_subset = DatasetSplit(dataset_test, dict_users_test[cid])
        test_loader = DataLoader(
            test_subset,
            batch_size=getattr(args, "local_bs", getattr(args, "batch_size", 64)),
            shuffle=False,
            num_workers=getattr(args, "num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn_with_lengths,
        )

        if cid not in final_local_state_dicts:
            test_metrics_local[cid] = None
            continue

        local_model = copy.deepcopy(model_template)
        local_model.load_state_dict(final_local_state_dicts[cid])
        local_model.eval()
        local_model.to(device)
        test_metrics_local[cid] = evaluate_model(local_model, test_loader, device)

    print("\nPer-client test metrics:")
    print("-" * 120)
    print(f"{'Client':<10} {'Model':<18} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'PRAUC':<12} {'AUROC':<12}")
    print("-" * 120)
    for cid in client_ids:
        local_metrics = test_metrics_local.get(cid)
        if local_metrics is None:
            print(f"{cid:<10} {'Final Local':<18} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue

        print(
            f"{cid:<10} {'Final Local':<18} {local_metrics.get('accuracy', 0.0):<12.4f} "
            f"{local_metrics.get('precision', 0.0):<12.4f} {local_metrics.get('recall', 0.0):<12.4f} "
            f"{local_metrics.get('f1', 0.0):<12.4f} {local_metrics.get('prauc', 0.0):<12.4f} "
            f"{local_metrics.get('auroc', 0.0):<12.4f}"
        )
    print("-" * 120)

    avg_local = compute_statistics_metrics(test_metrics_local)
    print("\nMean local-model test metrics (mean ± standard deviation):")
    print(f"  Accuracy:  {format_mean_std(avg_local['accuracy'], avg_local['accuracy_std'])}")
    print(f"  Precision: {format_mean_std(avg_local['precision'], avg_local['precision_std'])}")
    print(f"  Recall:    {format_mean_std(avg_local['recall'], avg_local['recall_std'])}")
    print(f"  F1:        {format_mean_std(avg_local['f1'], avg_local['f1_std'])}")
    print(f"  PRAUC:              {format_mean_std(avg_local['prauc'], avg_local['prauc_std'])}")
    print(f"  AUROC:              {format_mean_std(avg_local['auroc'], avg_local['auroc_std'])}")
    print(f"\n  Best F1: {avg_local['f1_max']:.4f}  |  Worst F1: {avg_local['f1_min']:.4f}")
    print(f"  Best AUROC: {avg_local['auroc_max']:.4f}  |  Worst AUROC: {avg_local['auroc_min']:.4f}")

    return test_metrics_local, avg_local


def main():
    args = parse_args()
    setup_seed(args.seed)

    # Append the _HA suffix to exp_name when FedDyna-HA is enabled.
    if getattr(args, "enable_ha_scope", False):
        if not args.exp_name.endswith("_HA"):
            args.exp_name = args.exp_name + "_HA"

    dataset_train, dataset_val, dataset_test, dict_users_train, dict_users_val, dict_users_test, _ = get_data(args)
    save_paths = prepare_experiment_save(args)
    model = load_model(args)

    # Select the training path based on fl_method.
    if args.fl_method == "local":
        # Train each client independently, without federated aggregation.
        local_result = run_local_training(
            model_template=model,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            dict_users_train=dict_users_train,
            dict_users_val=dict_users_val,
            dict_users_test=dict_users_test,
            args=args,
        )

        final_local_state_dicts = local_result["final_local_state_dicts"]
        test_metrics = local_result["test_metrics"]
        summary = local_result["summary"]

        # Save results.
        history = [{
            "test_metrics_local": test_metrics,
            "avg_test_local": summary,
            "method": "local",
            # Add training-time information.
            "total_training_time_sec": summary.get("total_training_time_sec", 0),
            "avg_training_time_sec": summary.get("avg_training_time_sec", 0),
            "total_training_time_min": summary.get("total_training_time_min", 0),
            "avg_training_time_min": summary.get("avg_training_time_min", 0),
            # Add per-client training times.
            "client_training_times": {cid: local_result["client_results"][cid]["training_time"] for cid in local_result["client_results"]}
        }]

        save_paths = save_local_artifacts(
            final_local_state_dicts=final_local_state_dicts,
            history=history,
            args=args,
        )
        excel_path = save_paths["excel_path"]

        print("\nTraining complete. Results saved.")
        print(f"Excel results: {excel_path}")
        print("Model files:")
        print("  - client_models_final_local.pt: best independently trained model for each client")

        return

    # Standard federated training path.
    training_result = run_federated_training(
        model=model,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dict_users_train=dict_users_train,
        dict_users_val=dict_users_val,
        args=args,
    )
    # Unpack the result, including communication statistics.
    final_global_model = training_result[0]
    final_local_state_dicts = training_result[1]
    client_post_broadcast_state_dicts = training_result[2]
    client_masks = training_result[3]
    history = training_result[4]
    comm_tracker = training_result[5] if len(training_result) > 5 else None

    test_metrics_local, avg_local = evaluate_final_local_models(
        model_template=model,
        dataset_test=dataset_test,
        dict_users_test=dict_users_test,
        final_local_state_dicts=final_local_state_dicts,
        args=args,
    )

    if isinstance(history, list):
        history.append({
            "test_metrics_local": test_metrics_local,
            "avg_test_local": avg_local,
        })

    print("\n" + "=" * 80)
    print("Global test evaluation (log_test.csv)")
    print("=" * 80)
    global_test_loader, global_test_size = load_global_testset(args)
    if global_test_loader is not None and global_test_size > 0:
        device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        final_global_model.eval()
        final_global_model.to(device)
        global_test_metrics = evaluate_model(final_global_model, global_test_loader, device)

        print("Global model metrics on the global test set:")
        print(f"  Samples: {global_test_size}")
        print(f"  Accuracy:  {global_test_metrics.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {global_test_metrics.get('precision', 0.0):.4f}")
        print(f"  Recall:    {global_test_metrics.get('recall', 0.0):.4f}")
        print(f"  F1:        {global_test_metrics.get('f1', 0.0):.4f}")
        print(f"  PRAUC:              {global_test_metrics.get('prauc', 0.0):.4f}")
        print(f"  AUROC:              {global_test_metrics.get('auroc', 0.0):.4f}")

        if isinstance(history, list):
            history[-1]["global_test_metrics"] = global_test_metrics
            history[-1]["global_test_size"] = global_test_size
    else:
        print("Warning: Failed to load the global test set. Skipping evaluation.")

    # Print the communication summary.
    if comm_tracker is not None:
        comm_summary = comm_tracker.get_summary()
        print("\n" + "=" * 80)
        print("Training complete - communication summary")
        print("=" * 80)
        print(f"Total training time:   {comm_summary['total_training_time_sec']:.2f} s ({comm_summary['total_training_time_min']:.2f} min)")
        print(f"Total uplink: {comm_summary['total_uplink_mb']:.2f} MB")
        print(f"Total downlink: {comm_summary['total_downlink_mb']:.2f} MB")
        print(f"Total communication:   {comm_summary['total_communication_mb']:.2f} MB")
        print(f"Average uplink per round: {comm_summary['avg_uplink_per_round_mb']:.2f} MB")
        print(f"Average downlink per round: {comm_summary['avg_downlink_per_round_mb']:.2f} MB")
        print("=" * 80)

    # Save communication statistics to history.
    if comm_tracker is not None and isinstance(history, list):
        history[-1]["communication_summary"] = comm_tracker.get_summary()
        history[-1]["communication_rounds"] = comm_tracker.round_communication

    save_paths = save_federated_artifacts(
        final_global_model=final_global_model,
        final_local_state_dicts=final_local_state_dicts,
        client_post_broadcast_state_dicts=client_post_broadcast_state_dicts,
        client_masks=client_masks,
        history=history,
        args=args,
    )
    excel_path = save_paths["excel_path"]

    print("Training complete. Results saved.")
    print(f"Excel results: {excel_path}")
    print("Model files:")
    print("  - global_model.pt: final global model")
    print("  - client_models.pt / client_models_final_local.pt: best local models by validation F1")
    print("  - client_models_post_broadcast.pt: client initialization states after the final broadcast")


if __name__ == "__main__":
    main()
