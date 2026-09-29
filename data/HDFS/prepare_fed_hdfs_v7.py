# coding: utf-8
"""Prepare HDFS client splits and dense/compact NPZ inputs.

The current v7 workflow constructs balanced candidate clients, selects valid
clients by anomaly-ratio quantiles, and writes preparation metadata.
Inputs are Event_traces.csv and anomaly_label.csv. Historical producer identity
is not certified; see docs/reproducibility_notes.md.
"""

import argparse
import ast
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm


# ============================================================
# Default configuration
# ============================================================
RANDOM_SEED = 42

EVENT_TRACES_CSV = "Event_traces.csv"
LABEL_FILE = "anomaly_label.csv"

OUTPUT_DIR = "fed_data"
CLIENT_CSV_DIR = os.path.join(OUTPUT_DIR, "client_data_v7")
VERSION_TAG = "v7"

USE_ALL_VALID_BLOCKS = True
SAMPLE_BLOCKS = None

NUM_CANDIDATE_CLIENTS = 100
NUM_SELECTED_CLIENTS = 10
CLIENT_SELECTION_RULE = "quantile"
SELECTION_QUANTILES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

TRAIN_RATIO = 0.5
VAL_RATIO = 0.2
TEST_RATIO = 0.3

MIN_CLIENT_BLOCKS = 1000
MIN_CLIENT_ANOMALIES = 50
MIN_TRAIN_ANOMALIES = 20
MIN_VAL_ANOMALIES = 8
MIN_TEST_ANOMALIES = 12

W2V_DIM = 300
MAX_SEQ_LEN = 100
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"
PAD_IDX = 0
OOV_IDX = 1


# ============================================================
# Utilities
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def script_dir():
    return Path(__file__).resolve().parent


def resolve_path(path_value, base_dir):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ratio(numer, denom):
    return float(numer) / float(denom) if denom else 0.0


def label_counts(blocks, sessions):
    anomaly = sum(1 for blk in blocks if sessions[blk]["label"] == 1)
    normal = len(blocks) - anomaly
    return normal, anomaly


def sequence_stats(blocks, sessions):
    if not blocks:
        return {"avg_sequence_length": 0.0, "max_sequence_length": 0, "unique_event_count": 0}
    lengths = [len(sessions[blk]["event_ids"]) for blk in blocks]
    unique_events = set()
    for blk in blocks:
        unique_events.update(sessions[blk]["event_ids"])
    return {
        "avg_sequence_length": float(np.mean(lengths)),
        "max_sequence_length": int(max(lengths)),
        "unique_event_count": int(len(unique_events)),
    }


def summarize_blocks(blocks, sessions, client_id=None):
    normal, anomaly = label_counts(blocks, sessions)
    summary = {
        "total_blocks": int(len(blocks)),
        "normal_blocks": int(normal),
        "anomaly_blocks": int(anomaly),
        "anomaly_ratio": ratio(anomaly, len(blocks)),
    }
    summary.update(sequence_stats(blocks, sessions))
    if client_id is not None:
        summary = {"client_id": int(client_id), **summary}
    return summary


def summarize_split(blocks, sessions):
    normal, anomaly = label_counts(blocks, sessions)
    return {
        "total": int(len(blocks)),
        "normal": int(normal),
        "anomaly": int(anomaly),
        "anomaly_ratio": ratio(anomaly, len(blocks)),
    }


def parse_features(features_value):
    """Parse Event_traces.csv Features values such as [E12, E5] to integer ids."""
    if pd.isna(features_value):
        return []

    text = str(features_value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    raw_items = None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            raw_items = parsed
    except Exception:
        raw_items = None

    if raw_items is None:
        raw_items = text.strip().strip("[]").split(",")

    events = []
    for item in raw_items:
        token = str(item).strip().strip("'\"")
        if not token:
            continue
        if token.startswith("E") and token[1:].isdigit():
            events.append(int(token[1:]))
        elif token.isdigit():
            events.append(int(token))
        else:
            return None
    return events


def load_labels(label_path):
    label_df = pd.read_csv(label_path, header=None, names=["blk_id", "label"])
    label_map = {"Normal": 0, "Anomaly": 1}
    labels = {}
    for _, row in label_df.iterrows():
        label = label_map.get(str(row["label"]).strip())
        if label is not None:
            labels[str(row["blk_id"]).strip()] = label
    return labels


def load_event_sessions(event_path, label_dict):
    traces_df = pd.read_csv(event_path)
    required_cols = {"BlockId", "Features"}
    missing = required_cols.difference(traces_df.columns)
    if missing:
        raise ValueError(f"{event_path} is missing required columns: {sorted(missing)}")

    valid_sessions = {}
    parse_failures = 0
    missing_label = 0
    empty_sequence = 0

    for _, row in tqdm(traces_df.iterrows(), total=len(traces_df), desc="Parsing Event_traces.csv"):
        blk_id = str(row["BlockId"]).strip()
        if blk_id not in label_dict:
            missing_label += 1
            continue

        events = parse_features(row["Features"])
        if events is None:
            parse_failures += 1
            continue
        if len(events) == 0:
            empty_sequence += 1
            continue

        valid_sessions[blk_id] = {
            "event_ids": events,
            "label": int(label_dict[blk_id]),
            "log_count": int(len(events)),
        }

    read_stats = {
        "event_trace_rows": int(len(traces_df)),
        "valid_rows": int(len(valid_sessions)),
        "missing_label_rows": int(missing_label),
        "empty_sequence_rows": int(empty_sequence),
        "parse_failure_rows": int(parse_failures),
    }
    return valid_sessions, read_stats


def split_by_label(valid_sessions):
    normal_blks, anomaly_blks = [], []
    for blk, session in valid_sessions.items():
        if session["label"] == 1:
            anomaly_blks.append(blk)
        else:
            normal_blks.append(blk)
    return normal_blks, anomaly_blks


def select_working_blocks(normal_blks, anomaly_blks, use_all_valid, sample_blocks, seed):
    all_before = list(normal_blks) + list(anomaly_blks)
    before_stats = {
        "total": int(len(all_before)),
        "normal": int(len(normal_blks)),
        "anomaly": int(len(anomaly_blks)),
        "anomaly_ratio": ratio(len(anomaly_blks), len(all_before)),
    }

    if use_all_valid:
        selected_normal = list(normal_blks)
        selected_anomaly = list(anomaly_blks)
    else:
        if sample_blocks is None:
            selected_normal = list(normal_blks)
            selected_anomaly = list(anomaly_blks)
        else:
            if sample_blocks <= 0:
                raise ValueError("sample_blocks must be a positive integer or None")
            rng = np.random.default_rng(seed)
            total = len(all_before)
            target_total = min(int(sample_blocks), total)
            target_anomaly = int(round(target_total * before_stats["anomaly_ratio"]))
            target_anomaly = min(target_anomaly, len(anomaly_blks))
            target_normal = min(target_total - target_anomaly, len(normal_blks))
            selected_normal = list(rng.choice(normal_blks, size=target_normal, replace=False))
            selected_anomaly = list(rng.choice(anomaly_blks, size=target_anomaly, replace=False))

    selected = selected_normal + selected_anomaly
    after_stats = {
        "total": int(len(selected)),
        "normal": int(len(selected_normal)),
        "anomaly": int(len(selected_anomaly)),
        "anomaly_ratio": ratio(len(selected_anomaly), len(selected)),
    }
    return selected, before_stats, after_stats


def build_candidate_clients(block_ids, num_clients, seed):
    rng = np.random.default_rng(seed)
    shuffled = np.array(list(block_ids), dtype=object)
    rng.shuffle(shuffled)
    chunks = np.array_split(shuffled, num_clients)
    return {cid: [str(x) for x in chunks[cid].tolist()] for cid in range(num_clients)}


def stratified_split_blocks(blocks, sessions, train_ratio, val_ratio, test_ratio, seed):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1")

    rng = np.random.default_rng(seed)
    normal_blks = np.array([b for b in blocks if sessions[b]["label"] == 0], dtype=object)
    anomaly_blks = np.array([b for b in blocks if sessions[b]["label"] == 1], dtype=object)
    rng.shuffle(normal_blks)
    rng.shuffle(anomaly_blks)

    def split_one_class(items):
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train = items[:n_train].tolist()
        val = items[n_train:n_train + n_val].tolist()
        test = items[n_train + n_val:].tolist()
        return train, val, test

    train_n, val_n, test_n = split_one_class(normal_blks)
    train_a, val_a, test_a = split_one_class(anomaly_blks)

    train = train_n + train_a
    val = val_n + val_a
    test = test_n + test_a
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def validate_client(blocks, train_blks, val_blks, test_blks, sessions, constraints):
    reasons = []
    total_summary = summarize_blocks(blocks, sessions)
    train_summary = summarize_split(train_blks, sessions)
    val_summary = summarize_split(val_blks, sessions)
    test_summary = summarize_split(test_blks, sessions)

    if total_summary["total_blocks"] < constraints["min_client_blocks"]:
        reasons.append(
            f"client_blocks<{constraints['min_client_blocks']} "
            f"({total_summary['total_blocks']})"
        )
    if total_summary["anomaly_blocks"] < constraints["min_client_anomalies"]:
        reasons.append(
            f"client_anomalies<{constraints['min_client_anomalies']} "
            f"({total_summary['anomaly_blocks']})"
        )
    if train_summary["anomaly"] < constraints["min_train_anomalies"]:
        reasons.append(
            f"train_anomalies<{constraints['min_train_anomalies']} "
            f"({train_summary['anomaly']})"
        )
    if val_summary["anomaly"] < constraints["min_val_anomalies"]:
        reasons.append(
            f"val_anomalies<{constraints['min_val_anomalies']} "
            f"({val_summary['anomaly']})"
        )
    if test_summary["anomaly"] < constraints["min_test_anomalies"]:
        reasons.append(
            f"test_anomalies<{constraints['min_test_anomalies']} "
            f"({test_summary['anomaly']})"
        )
    return reasons, {
        "overall": total_summary,
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
    }


def select_clients_by_quantiles(valid_client_stats, num_selected, quantiles):
    if len(valid_client_stats) < num_selected:
        raise RuntimeError(
            f"Only {len(valid_client_stats)} valid clients are available, "
            f"but {num_selected} are required."
        )

    sorted_clients = sorted(
        valid_client_stats,
        key=lambda item: (item["anomaly_ratio"], item["client_id"]),
    )
    if len(quantiles) != num_selected:
        quantiles = [(i + 0.5) / num_selected for i in range(num_selected)]

    selected = []
    used = set()
    n = len(sorted_clients)
    for q in quantiles:
        target = int(round(q * (n - 1)))
        target = max(0, min(n - 1, target))
        candidates = sorted(range(n), key=lambda idx: (abs(idx - target), idx))
        chosen_idx = None
        for idx in candidates:
            cid = sorted_clients[idx]["client_id"]
            if cid not in used:
                chosen_idx = idx
                break
        if chosen_idx is None:
            raise RuntimeError("Failed to select unique quantile clients")
        selected.append(sorted_clients[chosen_idx])
        used.add(sorted_clients[chosen_idx]["client_id"])
    return [item["client_id"] for item in selected]


def select_clients_random(valid_client_stats, num_selected, seed):
    if len(valid_client_stats) < num_selected:
        raise RuntimeError(
            f"Only {len(valid_client_stats)} valid clients are available, "
            f"but {num_selected} are required."
        )
    rng = np.random.default_rng(seed)
    ids = np.array([item["client_id"] for item in valid_client_stats], dtype=np.int64)
    selected = rng.choice(ids, size=num_selected, replace=False).tolist()
    return sorted(int(x) for x in selected)


def save_client_csv(client_id, train_blks, val_blks, test_blks, sessions, output_dir):
    ensure_dir(output_dir)

    def to_df(blocks):
        return pd.DataFrame([
            {
                "block_id": blk,
                "event_ids": " ".join(map(str, sessions[blk]["event_ids"])),
                "label": int(sessions[blk]["label"]),
            }
            for blk in blocks
        ])

    paths = {}
    for split_name, split_blocks in (
        ("train", train_blks),
        ("val", val_blks),
        ("test", test_blks),
    ):
        path = Path(output_dir) / f"client_{client_id}_{split_name}.csv"
        to_df(split_blocks).to_csv(path, index=False)
        paths[split_name] = str(path)
    return paths


def train_word2vec(train_sequences, embedding_dim, seed):
    if not train_sequences:
        raise RuntimeError("No train sequences available for Word2Vec")

    train_sequences_str = [[str(e) for e in seq] for seq in train_sequences]
    all_events = sorted({int(e) for seq in train_sequences for e in seq})

    vocab = {PAD_TOKEN: PAD_IDX, OOV_TOKEN: OOV_IDX}
    for event in all_events:
        vocab[int(event)] = len(vocab)

    model = Word2Vec(
        sentences=train_sequences_str,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        workers=4,
        seed=seed,
        sg=1,
    )

    rng = np.random.default_rng(seed)
    embeddings = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    embeddings[OOV_IDX] = rng.normal(0.0, 0.1, size=embedding_dim).astype(np.float32)

    for event in all_events:
        idx = vocab[event]
        key = str(event)
        if key in model.wv:
            embeddings[idx] = model.wv[key]
        else:
            embeddings[idx] = embeddings[OOV_IDX]
    return vocab, embeddings


def sequence_to_ids(events, vocab, max_seq_len):
    truncated = list(events[:max_seq_len])
    length = len(truncated)
    ids = [vocab.get(int(e), OOV_IDX) for e in truncated]
    ids = ids + [PAD_IDX] * (max_seq_len - len(ids))
    return np.asarray(ids, dtype=np.int32), int(length)


def build_arrays(selected_client_ids, split_blocks_by_client, sessions, vocab, embeddings, max_seq_len):
    split_arrays = {}
    dict_users = {}

    for split_name in ("train", "val", "test"):
        all_ids = []
        all_labels = []
        all_lengths = []
        dict_users[split_name] = {}
        cursor = 0

        for cid in selected_client_ids:
            blocks = split_blocks_by_client[split_name][cid]
            start = cursor
            for blk in blocks:
                ids, length = sequence_to_ids(sessions[blk]["event_ids"], vocab, max_seq_len)
                all_ids.append(ids)
                all_labels.append(int(sessions[blk]["label"]))
                all_lengths.append(length)
                cursor += 1
            dict_users[split_name][cid] = np.arange(start, cursor, dtype=np.int64)

        id_array = np.stack(all_ids).astype(np.int32) if all_ids else np.empty((0, max_seq_len), dtype=np.int32)
        label_array = np.asarray(all_labels, dtype=np.int64)
        length_array = np.asarray(all_lengths, dtype=np.int32)
        feature_array = embeddings[id_array].astype(np.float32, copy=False)

        split_arrays[split_name] = {
            "ids": id_array,
            "features": feature_array,
            "labels": label_array,
            "lengths": length_array,
        }

    return split_arrays, dict_users


def save_npz_outputs(output_dir, version_tag, split_arrays, dict_users, embeddings,
                     selected_client_ids, max_seq_len):
    dense_path = Path(output_dir) / f"hdfs_fed_unified_{version_tag}.npz"
    ids_path = Path(output_dir) / f"hdfs_fed_unified_{version_tag}_ids.npz"

    np.savez(
        dense_path,
        train_features=split_arrays["train"]["features"],
        train_labels=split_arrays["train"]["labels"],
        train_lengths=split_arrays["train"]["lengths"],
        val_features=split_arrays["val"]["features"],
        val_labels=split_arrays["val"]["labels"],
        val_lengths=split_arrays["val"]["lengths"],
        test_features=split_arrays["test"]["features"],
        test_labels=split_arrays["test"]["labels"],
        test_lengths=split_arrays["test"]["lengths"],
        dict_users_train=dict_users["train"],
        dict_users_val=dict_users["val"],
        dict_users_test=dict_users["test"],
        client_ids=np.asarray(selected_client_ids, dtype=np.int32),
        max_seq_len=int(max_seq_len),
        feature_dim=int(embeddings.shape[1]),
        vocab_size=int(embeddings.shape[0]),
        has_dynamic_padding=True,
    )

    np.savez(
        ids_path,
        train_ids=split_arrays["train"]["ids"],
        val_ids=split_arrays["val"]["ids"],
        test_ids=split_arrays["test"]["ids"],
        train_labels=split_arrays["train"]["labels"],
        val_labels=split_arrays["val"]["labels"],
        test_labels=split_arrays["test"]["labels"],
        train_lengths=split_arrays["train"]["lengths"],
        val_lengths=split_arrays["val"]["lengths"],
        test_lengths=split_arrays["test"]["lengths"],
        dict_users_train=dict_users["train"],
        dict_users_val=dict_users["val"],
        dict_users_test=dict_users["test"],
        embedding_matrix=embeddings.astype(np.float32),
        client_ids=np.asarray(selected_client_ids, dtype=np.int32),
        max_seq_len=int(max_seq_len),
        vocab_size=int(embeddings.shape[0]),
        has_dynamic_padding=True,
    )

    return dense_path, ids_path


def save_vocab_and_embeddings(output_dir, version_tag, vocab, embeddings, embedding_dim):
    vocab_path = Path(output_dir) / f"vocab_{version_tag}.json"
    embed_path = Path(output_dir) / f"event_vectors_w2v{embedding_dim}_{version_tag}.npy"
    semantic_path = Path(output_dir) / f"hdfs_semantic_vec_{version_tag}.json"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in vocab.items()}, f, indent=2)
    np.save(embed_path, embeddings.astype(np.float32))

    semantic_vectors = {}
    for token, idx in vocab.items():
        if isinstance(token, int):
            semantic_vectors[str(token)] = embeddings[idx].tolist()
    with open(semantic_path, "w", encoding="utf-8") as f:
        json.dump(semantic_vectors, f, indent=2)

    return vocab_path, embed_path, semantic_path


def enrich_selected_stats(client_id, candidate_summary, split_stats):
    return {
        "client_id": int(client_id),
        "total_blocks": candidate_summary["total_blocks"],
        "normal_blocks": candidate_summary["normal_blocks"],
        "anomaly_blocks": candidate_summary["anomaly_blocks"],
        "anomaly_ratio": candidate_summary["anomaly_ratio"],
        "train_total": split_stats["train"]["total"],
        "train_normal": split_stats["train"]["normal"],
        "train_anomaly": split_stats["train"]["anomaly"],
        "train_anomaly_ratio": split_stats["train"]["anomaly_ratio"],
        "val_total": split_stats["val"]["total"],
        "val_normal": split_stats["val"]["normal"],
        "val_anomaly": split_stats["val"]["anomaly"],
        "val_anomaly_ratio": split_stats["val"]["anomaly_ratio"],
        "test_total": split_stats["test"]["total"],
        "test_normal": split_stats["test"]["normal"],
        "test_anomaly": split_stats["test"]["anomaly"],
        "test_anomaly_ratio": split_stats["test"]["anomaly_ratio"],
        "avg_sequence_length": candidate_summary["avg_sequence_length"],
        "max_sequence_length": candidate_summary["max_sequence_length"],
        "unique_event_count": candidate_summary["unique_event_count"],
    }


def make_jsonable(value):
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare HDFS federated data V7.")
    parser.add_argument("--event_traces_csv", default=EVENT_TRACES_CSV)
    parser.add_argument("--label_file", default=LABEL_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--client_csv_dir", default=CLIENT_CSV_DIR)
    parser.add_argument("--version_tag", default=VERSION_TAG)
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED)

    parser.add_argument("--use_all_valid_blocks", action="store_true", default=USE_ALL_VALID_BLOCKS)
    parser.add_argument("--no_use_all_valid_blocks", action="store_false", dest="use_all_valid_blocks")
    parser.add_argument("--sample_blocks", type=int, default=SAMPLE_BLOCKS)

    parser.add_argument("--num_candidate_clients", type=int, default=NUM_CANDIDATE_CLIENTS)
    parser.add_argument("--num_selected_clients", type=int, default=NUM_SELECTED_CLIENTS)
    parser.add_argument("--client_selection_rule", choices=["quantile", "random"], default=CLIENT_SELECTION_RULE)

    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO)

    parser.add_argument("--min_client_blocks", type=int, default=MIN_CLIENT_BLOCKS)
    parser.add_argument("--min_client_anomalies", type=int, default=MIN_CLIENT_ANOMALIES)
    parser.add_argument("--min_train_anomalies", type=int, default=MIN_TRAIN_ANOMALIES)
    parser.add_argument("--min_val_anomalies", type=int, default=MIN_VAL_ANOMALIES)
    parser.add_argument("--min_test_anomalies", type=int, default=MIN_TEST_ANOMALIES)

    parser.add_argument("--embedding_dim", type=int, default=W2V_DIM)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--skip_dense_npz", action="store_true", help="Debug option: save compact NPZ only.")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.random_seed)

    base_dir = script_dir()
    event_path = resolve_path(args.event_traces_csv, base_dir)
    label_path = resolve_path(args.label_file, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    client_csv_dir = resolve_path(args.client_csv_dir, base_dir)
    ensure_dir(output_dir)
    ensure_dir(client_csv_dir)

    constraints = {
        "min_client_blocks": args.min_client_blocks,
        "min_client_anomalies": args.min_client_anomalies,
        "min_train_anomalies": args.min_train_anomalies,
        "min_val_anomalies": args.min_val_anomalies,
        "min_test_anomalies": args.min_test_anomalies,
    }

    print("=" * 78)
    print("HDFS federated preprocessing V7")
    print("=" * 78)
    print(f"Inputs: {event_path}, {label_path}")
    print(f"Output dir: {output_dir}")
    print(f"Client CSV dir: {client_csv_dir}")
    print(f"use_all_valid_blocks={args.use_all_valid_blocks}, sample_blocks={args.sample_blocks}")
    print(f"candidate_clients={args.num_candidate_clients}, selected_clients={args.num_selected_clients}")

    print("\n[1/9] Loading labels")
    label_dict = load_labels(label_path)
    print(f"  Labels loaded: {len(label_dict):,}")

    print("\n[2/9] Loading and parsing Event_traces.csv")
    valid_sessions, read_stats = load_event_sessions(event_path, label_dict)
    normal_blks, anomaly_blks = split_by_label(valid_sessions)
    print(f"  Valid block sessions: {len(valid_sessions):,}")
    print(f"  Normal: {len(normal_blks):,}, anomaly: {len(anomaly_blks):,}, ratio={ratio(len(anomaly_blks), len(valid_sessions)):.4%}")

    print("\n[3/9] Selecting working blocks")
    working_blocks, pre_sample_stats, post_sample_stats = select_working_blocks(
        normal_blks,
        anomaly_blks,
        args.use_all_valid_blocks,
        args.sample_blocks,
        args.random_seed,
    )
    print(f"  Before: {pre_sample_stats}")
    print(f"  After : {post_sample_stats}")

    print("\n[4/9] Building candidate clients")
    candidate_blocks = build_candidate_clients(working_blocks, args.num_candidate_clients, args.random_seed)

    candidate_stats = []
    split_blocks_by_client = {"train": {}, "val": {}, "test": {}}
    split_stats_by_client = {}
    invalid_client_reasons = {}
    valid_client_stats = []

    for cid in tqdm(range(args.num_candidate_clients), desc="Splitting and validating clients"):
        blocks = candidate_blocks[cid]
        train_blks, val_blks, test_blks = stratified_split_blocks(
            blocks,
            valid_sessions,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.random_seed + cid,
        )
        reasons, validation_stats = validate_client(
            blocks,
            train_blks,
            val_blks,
            test_blks,
            valid_sessions,
            constraints,
        )

        summary = summarize_blocks(blocks, valid_sessions, client_id=cid)
        candidate_stats.append(summary)
        split_stats_by_client[cid] = {
            "train": validation_stats["train"],
            "val": validation_stats["val"],
            "test": validation_stats["test"],
        }

        if reasons:
            invalid_client_reasons[cid] = reasons
            continue

        split_blocks_by_client["train"][cid] = train_blks
        split_blocks_by_client["val"][cid] = val_blks
        split_blocks_by_client["test"][cid] = test_blks
        valid_client_stats.append({
            "client_id": int(cid),
            "anomaly_ratio": summary["anomaly_ratio"],
            "total_blocks": summary["total_blocks"],
            "anomaly_blocks": summary["anomaly_blocks"],
        })

    print(f"  Candidate clients: {args.num_candidate_clients}")
    print(f"  Valid candidates : {len(valid_client_stats)}")
    print(f"  Invalid candidates: {len(invalid_client_reasons)}")
    if len(valid_client_stats) < args.num_selected_clients:
        raise RuntimeError(
            "Not enough valid candidate clients. For debug runs, lower the min_* constraints."
        )

    print("\n[5/9] Selecting clients")
    if args.client_selection_rule == "quantile":
        selected_client_ids = select_clients_by_quantiles(
            valid_client_stats,
            args.num_selected_clients,
            SELECTION_QUANTILES[:args.num_selected_clients],
        )
    else:
        selected_client_ids = select_clients_random(
            valid_client_stats,
            args.num_selected_clients,
            args.random_seed,
        )
    selected_client_ids = [int(cid) for cid in selected_client_ids]
    print(f"  selected_client_ids={selected_client_ids}")

    print("\n[6/9] Saving selected client CSV files")
    csv_paths = {}
    selected_stats = []
    per_split_statistics = {}
    for cid in selected_client_ids:
        paths = save_client_csv(
            cid,
            split_blocks_by_client["train"][cid],
            split_blocks_by_client["val"][cid],
            split_blocks_by_client["test"][cid],
            valid_sessions,
            client_csv_dir,
        )
        csv_paths[cid] = paths
        selected_summary = next(item for item in candidate_stats if item["client_id"] == cid)
        selected_stats.append(enrich_selected_stats(cid, selected_summary, split_stats_by_client[cid]))
        per_split_statistics[cid] = split_stats_by_client[cid]

    print("\n[7/9] Training Word2Vec from selected train splits only")
    train_sequences = []
    for cid in selected_client_ids:
        for blk in split_blocks_by_client["train"][cid]:
            train_sequences.append(valid_sessions[blk]["event_ids"])
    vocab, embeddings = train_word2vec(train_sequences, args.embedding_dim, args.random_seed)
    vocab_path, embed_path, semantic_path = save_vocab_and_embeddings(
        output_dir,
        args.version_tag,
        vocab,
        embeddings,
        args.embedding_dim,
    )
    print(f"  vocab_size={len(vocab)}, embedding_matrix={embeddings.shape}")

    print("\n[8/9] Building NPZ arrays")
    selected_split_blocks = {
        split_name: {cid: split_blocks_by_client[split_name][cid] for cid in selected_client_ids}
        for split_name in ("train", "val", "test")
    }
    split_arrays, dict_users = build_arrays(
        selected_client_ids,
        selected_split_blocks,
        valid_sessions,
        vocab,
        embeddings,
        args.max_seq_len,
    )

    dense_path = Path(output_dir) / f"hdfs_fed_unified_{args.version_tag}.npz"
    ids_path = Path(output_dir) / f"hdfs_fed_unified_{args.version_tag}_ids.npz"
    if args.skip_dense_npz:
        np.savez(
            ids_path,
            train_ids=split_arrays["train"]["ids"],
            val_ids=split_arrays["val"]["ids"],
            test_ids=split_arrays["test"]["ids"],
            train_labels=split_arrays["train"]["labels"],
            val_labels=split_arrays["val"]["labels"],
            test_labels=split_arrays["test"]["labels"],
            train_lengths=split_arrays["train"]["lengths"],
            val_lengths=split_arrays["val"]["lengths"],
            test_lengths=split_arrays["test"]["lengths"],
            dict_users_train=dict_users["train"],
            dict_users_val=dict_users["val"],
            dict_users_test=dict_users["test"],
            embedding_matrix=embeddings.astype(np.float32),
            client_ids=np.asarray(selected_client_ids, dtype=np.int32),
            max_seq_len=int(args.max_seq_len),
            vocab_size=int(embeddings.shape[0]),
            has_dynamic_padding=True,
        )
        dense_path = None
    else:
        dense_path, ids_path = save_npz_outputs(
            output_dir,
            args.version_tag,
            split_arrays,
            dict_users,
            embeddings,
            selected_client_ids,
            args.max_seq_len,
        )

    print(f"  train_features shape: {split_arrays['train']['features'].shape}")
    print(f"  val_features shape  : {split_arrays['val']['features'].shape}")
    print(f"  test_features shape : {split_arrays['test']['features'].shape}")
    print(f"  train_ids shape     : {split_arrays['train']['ids'].shape}")
    print(f"  val_ids shape       : {split_arrays['val']['ids'].shape}")
    print(f"  test_ids shape      : {split_arrays['test']['ids'].shape}")

    print("\n[9/9] Saving metadata")
    output_files = {
        "client_csv_dir": str(client_csv_dir),
        "vocab": str(vocab_path),
        "event_vectors": str(embed_path),
        "semantic_vectors": str(semantic_path),
        "dense_npz": str(dense_path) if dense_path is not None else None,
        "compact_ids_npz": str(ids_path),
    }
    metadata = {
        "random_seed": args.random_seed,
        "input_files": {
            "event_traces_csv": str(event_path),
            "anomaly_label_csv": str(label_path),
            "hdfs_log_structured_csv_used": False,
        },
        "read_stats": read_stats,
        "use_all_valid_blocks": bool(args.use_all_valid_blocks),
        "sample_blocks": args.sample_blocks,
        "num_valid_blocks": post_sample_stats["total"],
        "num_normal_blocks": post_sample_stats["normal"],
        "num_anomaly_blocks": post_sample_stats["anomaly"],
        "global_anomaly_ratio": post_sample_stats["anomaly_ratio"],
        "pre_sampling_statistics": pre_sample_stats,
        "post_sampling_statistics": post_sample_stats,
        "num_candidate_clients": args.num_candidate_clients,
        "num_valid_candidate_clients": len(valid_client_stats),
        "num_invalid_candidate_clients": len(invalid_client_reasons),
        "selected_client_ids": selected_client_ids,
        "client_selection_rule": args.client_selection_rule,
        "selection_quantiles": SELECTION_QUANTILES[:args.num_selected_clients],
        "split_ratio": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "min_anomaly_constraints": constraints,
        "max_seq_len": args.max_seq_len,
        "embedding_dim": args.embedding_dim,
        "vocab_size": int(len(vocab)),
        "per_candidate_client_statistics": candidate_stats,
        "per_selected_client_statistics": selected_stats,
        "per_split_statistics": per_split_statistics,
        "invalid_client_reasons": invalid_client_reasons,
        "client_csv_files": csv_paths,
        "output_files": output_files,
        "npz_shapes": {
            "train_features": list(split_arrays["train"]["features"].shape),
            "val_features": list(split_arrays["val"]["features"].shape),
            "test_features": list(split_arrays["test"]["features"].shape),
            "train_ids": list(split_arrays["train"]["ids"].shape),
            "val_ids": list(split_arrays["val"]["ids"].shape),
            "test_ids": list(split_arrays["test"]["ids"].shape),
        },
    }
    metadata_path = Path(output_dir) / f"hdfs_fed_unified_{args.version_tag}_metadata.json"
    output_files["metadata"] = str(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(make_jsonable(metadata), f, indent=2)

    print("\nDone.")
    print(f"  metadata: {metadata_path}")
    if dense_path is not None:
        print(f"  dense npz: {dense_path} ({dense_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  compact npz: {ids_path} ({ids_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  selected_client_ids: {selected_client_ids}")


if __name__ == "__main__":
    main()
