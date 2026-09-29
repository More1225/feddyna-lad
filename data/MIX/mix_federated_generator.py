#!/usr/bin/env python3
"""Build a unified federated NPZ from HDFS, BGL and Thunderbird client CSV triplets. Detect supported filename formats, fit shared Word2Vec embeddings on training sequences, and preserve each input split."""












import os
import sys
import json
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
from gensim.models import Word2Vec
import argparse

# ============================================================

# ============================================================
DEFAULT_INPUT_DIR = "fed_data_5HDFS+2BGL+3TB"
DEFAULT_OUTPUT_DIR = "fed_data_5HDFS+2BGL+3TB"
DEFAULT_OUTPUT_NAME = "mixed_fed_unified"


W2V_DIM = 300
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_WORKERS = 4
W2V_SEED = 42


MAX_SEQ_LEN = 100
FEATURE_DIM = W2V_DIM

# ============================================================

# ============================================================

def scan_all_clients(input_dir):
    """Discover client IDs from supported training CSV filenames in the input directory."""








    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return []

    client_ids = set()

    for filename in os.listdir(input_dir):

        match_hdfs = re.match(r'client_(\d+)_train\.csv$', filename)
        if match_hdfs:
            client_ids.add(('hdfs', int(match_hdfs.group(1))))
            continue


        match_shuffle = re.match(r'client(\d+)_train_shuffle\.csv$', filename)
        if match_shuffle:
            client_ids.add(('shuffle', int(match_shuffle.group(1))))
            continue


        match_bgl = re.match(r'client(\d+)_train\.csv$', filename)
        if match_bgl:
            client_ids.add(('bgl', int(match_bgl.group(1))))
            continue


        match_tb = re.match(r'client(\d+)_train_sequential\.csv$', filename)
        if match_tb:
            client_ids.add(('tb', int(match_tb.group(1))))
            continue


    sorted_clients = sorted(list(client_ids), key=lambda x: x[1])
    return sorted_clients

def load_client_data(input_dir, dataset_type, client_id):
    """Load train, validation and test DataFrames for the requested dataset type and client."""










    data = {}

    if dataset_type == 'hdfs':
        filename_template = f"client_{client_id}_{{split}}.csv"
    elif dataset_type == 'bgl':
        filename_template = f"client{client_id}_{{split}}.csv"
    elif dataset_type == 'shuffle':
        filename_template = f"client{client_id}_{{split}}_shuffle.csv"
    else:  # tb (Thunderbird)
        filename_template = f"client{client_id}_{{split}}_sequential.csv"

    for split in ['train', 'val', 'test']:
        filename = filename_template.format(split=split)
        filepath = os.path.join(input_dir, filename)

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)


                if 'event_ids' in df.columns and 'sequence' not in df.columns:
                    df = df.rename(columns={'event_ids': 'sequence'})


                if 'sequence' not in df.columns or 'label' not in df.columns:
                    print(f"[WARN] File {filename} is missing required columns")
                    data[split] = None
                    continue

                data[split] = df


            except Exception as e:
                print(f"[ERROR] Failed to load file {filename}: {e}")
                data[split] = None
        else:

            data[split] = None

    return data

def parse_sequence(seq_str):
    """Parse an event sequence into a list of tokens."""
    if pd.isna(seq_str) or seq_str == '':
        return []
    return [str(e) for e in str(seq_str).split()]

def extract_all_train_sequences(client_data_list):
    """Collect training sequences from all loaded clients."""
    all_sequences = []

    for dataset_type, client_id, data in tqdm(client_data_list, desc="Extracting training sequences"):
        if 'train' in data and data['train'] is not None:
            train_df = data['train']
            for seq_str in train_df['sequence']:
                events = parse_sequence(seq_str)
                if events:
                    all_sequences.append(events)

    print(f"  Extracted {len(all_sequences)} training sequences")
    return all_sequences

def train_word2vec_model(all_sequences, w2v_dim, seed):
    """Train Word2Vec on the supplied training sequences."""
    print("Training Word2Vec...")

    if len(all_sequences) == 0:
        raise ValueError("No training sequences available for Word2Vec")

    model = Word2Vec(
        sentences=all_sequences,
        vector_size=w2v_dim,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=W2V_WORKERS,
        seed=seed,
        sg=1  # skip-gram
    )

    print(f"  Word2Vec vocabulary size: {len(model.wv)}")
    return model

def create_vocab_and_embeddings(word2vec_model):
    """Create the vocabulary and embedding matrix, including PAD and OOV entries."""

    event_strings = sorted(word2vec_model.wv.key_to_index.keys(), key=lambda x: int(x))


    vocab = {'<PAD>': 0, '<OOV>': 1}
    for idx, event_str in enumerate(event_strings, start=2):
        vocab[event_str] = idx


    feature_dim = word2vec_model.vector_size
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, feature_dim), dtype=np.float32)


    embeddings[0] = np.zeros(feature_dim, dtype=np.float32)
    np.random.seed(W2V_SEED)
    embeddings[1] = np.random.randn(feature_dim).astype(np.float32) * 0.1


    for event_str, idx in vocab.items():
        if event_str not in ['<PAD>', '<OOV>']:
            embeddings[idx] = word2vec_model.wv[event_str]

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding matrix shape: {embeddings.shape}")

    return vocab, embeddings

def calculate_max_sequence_length(client_data_list):
    """Find the largest sequence length across all client splits."""
    max_len = 0

    for dataset_type, client_id, data in client_data_list:
        for split in ['train', 'val', 'test']:
            if split in data and data[split] is not None:
                df = data[split]
                for seq_str in df['sequence']:
                    events = parse_sequence(seq_str)
                    max_len = max(max_len, len(events))

    return max(1, max_len)

def sequence_to_tensor(seq_str, vocab, embeddings, max_len):
    """Return the padded sequence tensor and its actual length."""






    events = parse_sequence(seq_str)
    actual_len = min(len(events), max_len)


    if len(events) > max_len:
        events = events[:max_len]
    else:
        events = events + ['<PAD>'] * (max_len - len(events))


    indices = [vocab.get(e, 1) for e in events]


    tensor = embeddings[indices]

    return tensor, actual_len

def build_federated_npz(client_data_list, vocab, embeddings, max_seq_len, output_path):
    """Build the unified NPZ and return its path and per-client statistics."""






    print(f"\nBuilding federated NPZ file (Maximum sequence length: {max_seq_len})...")

    all_train_features = []
    all_train_labels = []
    all_train_lengths = []
    all_val_features = []
    all_val_labels = []
    all_val_lengths = []
    all_test_features = []
    all_test_labels = []
    all_test_lengths = []

    dict_users_train = {}
    dict_users_val = {}
    dict_users_test = {}

    current_train_idx = 0
    current_val_idx = 0
    current_test_idx = 0

    client_ids = []
    client_stats = []


    for idx, (dataset_type, orig_client_id, data) in enumerate(tqdm(client_data_list, desc="Processing clients")):
        new_cid = idx
        client_ids.append(new_cid)


        client_info = f"{dataset_type.upper()}-{orig_client_id}"


        client_stat = {
            'new_id': new_cid,
            'original_id': orig_client_id,
            'dataset_type': dataset_type,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'train_anomaly': 0,
            'val_anomaly': 0,
            'test_anomaly': 0,
            'train_anomaly_ratio': 0.0,
            'val_anomaly_ratio': 0.0,
            'test_anomaly_ratio': 0.0
        }


        if 'train' in data and data['train'] is not None:
            df = data['train']
            features, labels, lengths = [], [], []

            for _, row in df.iterrows():
                tensor, length = sequence_to_tensor(row['sequence'], vocab, embeddings, max_seq_len)
                features.append(tensor)
                labels.append(row['label'])
                lengths.append(length)

            if features:
                X_train = np.stack(features)
                y_train = np.array(labels, dtype=np.int64)
                l_train = np.array(lengths, dtype=np.int32)

                all_train_features.append(X_train)
                all_train_labels.append(y_train)
                all_train_lengths.append(l_train)

                dict_users_train[new_cid] = np.arange(current_train_idx, current_train_idx + len(X_train))
                current_train_idx += len(X_train)


                client_stat['train_samples'] = len(X_train)
                client_stat['train_anomaly'] = int(y_train.sum())
                client_stat['train_anomaly_ratio'] = float(y_train.mean()) if len(y_train) > 0 else 0.0

                anomaly_ratio = y_train.mean() if len(y_train) > 0 else 0



        if 'val' in data and data['val'] is not None:
            df = data['val']
            features, labels, lengths = [], [], []

            for _, row in df.iterrows():
                tensor, length = sequence_to_tensor(row['sequence'], vocab, embeddings, max_seq_len)
                features.append(tensor)
                labels.append(row['label'])
                lengths.append(length)

            if features:
                X_val = np.stack(features)
                y_val = np.array(labels, dtype=np.int64)
                l_val = np.array(lengths, dtype=np.int32)

                all_val_features.append(X_val)
                all_val_labels.append(y_val)
                all_val_lengths.append(l_val)

                dict_users_val[new_cid] = np.arange(current_val_idx, current_val_idx + len(X_val))
                current_val_idx += len(X_val)


                client_stat['val_samples'] = len(X_val)
                client_stat['val_anomaly'] = int(y_val.sum())
                client_stat['val_anomaly_ratio'] = float(y_val.mean()) if len(y_val) > 0 else 0.0


        if 'test' in data and data['test'] is not None:
            df = data['test']
            features, labels, lengths = [], [], []

            for _, row in df.iterrows():
                tensor, length = sequence_to_tensor(row['sequence'], vocab, embeddings, max_seq_len)
                features.append(tensor)
                labels.append(row['label'])
                lengths.append(length)

            if features:
                X_test = np.stack(features)
                y_test = np.array(labels, dtype=np.int64)
                l_test = np.array(lengths, dtype=np.int32)

                all_test_features.append(X_test)
                all_test_labels.append(y_test)
                all_test_lengths.append(l_test)

                dict_users_test[new_cid] = np.arange(current_test_idx, current_test_idx + len(X_test))
                current_test_idx += len(X_test)


                client_stat['test_samples'] = len(X_test)
                client_stat['test_anomaly'] = int(y_test.sum())
                client_stat['test_anomaly_ratio'] = float(y_test.mean()) if len(y_test) > 0 else 0.0

                anomaly_ratio = y_test.mean() if len(y_test) > 0 else 0



        client_stats.append(client_stat)


    def safe_stack(arrays):
        if not arrays:
            return np.array([])
        return np.vstack(arrays)

    def safe_concatenate(arrays):
        if not arrays:
            return np.array([])
        return np.concatenate(arrays)

    X_train_all = safe_stack(all_train_features)
    y_train_all = safe_concatenate(all_train_labels)
    l_train_all = safe_concatenate(all_train_lengths)

    X_val_all = safe_stack(all_val_features)
    y_val_all = safe_concatenate(all_val_labels)
    l_val_all = safe_concatenate(all_val_lengths)

    X_test_all = safe_stack(all_test_features)
    y_test_all = safe_concatenate(all_test_labels)
    l_test_all = safe_concatenate(all_test_lengths)


    print(f"\nData statistics:")
    print(f"  Total clients: {len(client_ids)}")
    print(f"  Training samples: {len(X_train_all)}")
    print(f"  Validation samples: {len(X_val_all)}")
    print(f"  Test samples: {len(X_test_all)}")

    if len(y_train_all) > 0:
        print(f"  Training anomaly ratio: {y_train_all.mean():.2%}")
    if len(y_test_all) > 0:
        print(f"  Test anomaly ratio: {y_test_all.mean():.2%}")


    print(f"\nClient details (total {len(client_stats)} clients):")
    print("-" * 100)
    print(f"{'ID':<4} {'Type':<8} {'Original ID':<8} {'Training samples':<10} {'Validation samples':<10} {'Test samples':<10} {'Train anomaly rate':<12} {'Validation anomaly rate':<12} {'Test anomaly rate':<12}")
    print("-" * 100)
    for stat in client_stats:
        print(f"{stat['new_id']:<4} {stat['dataset_type'].upper():<8} {stat['original_id']:<8} "
              f"{stat['train_samples']:<10} {stat['val_samples']:<10} {stat['test_samples']:<10} "
              f"{stat['train_anomaly_ratio']:>10.2%} {stat['val_anomaly_ratio']:>10.2%} {stat['test_anomaly_ratio']:>10.2%}")
    print("-" * 100)


    np.savez(
        output_path,
        train_features=X_train_all.astype(np.float32) if len(X_train_all) > 0 else np.array([]),
        train_labels=y_train_all.astype(np.int64) if len(y_train_all) > 0 else np.array([]),
        train_lengths=l_train_all.astype(np.int32) if len(l_train_all) > 0 else np.array([]),

        val_features=X_val_all.astype(np.float32) if len(X_val_all) > 0 else np.array([]),
        val_labels=y_val_all.astype(np.int64) if len(y_val_all) > 0 else np.array([]),
        val_lengths=l_val_all.astype(np.int32) if len(l_val_all) > 0 else np.array([]),

        test_features=X_test_all.astype(np.float32) if len(X_test_all) > 0 else np.array([]),
        test_labels=y_test_all.astype(np.int64) if len(y_test_all) > 0 else np.array([]),
        test_lengths=l_test_all.astype(np.int32) if len(l_test_all) > 0 else np.array([]),

        dict_users_train=dict_users_train,
        dict_users_val=dict_users_val,
        dict_users_test=dict_users_test,
        client_ids=np.array(client_ids, dtype=np.int32),

        max_seq_len=max_seq_len,
        feature_dim=embeddings.shape[1],
        vocab_size=len(vocab),
        has_dynamic_padding=True,
        dataset_composition=f"{len([c for c in client_data_list if c[0]=='hdfs'])}_HDFS_{len([c for c in client_data_list if c[0]=='bgl'])}_BGL_{len([c for c in client_data_list if c[0]=='tb'])}_TB_{len([c for c in client_data_list if c[0]=='shuffle'])}_SHUFFLE",
    )

    print(f"\nNPZ file saved: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return output_path, client_stats

def save_vocab_and_embeddings(vocab, embeddings, output_dir, output_name):
    """Save the vocabulary and embedding matrix."""
    vocab_path = os.path.join(output_dir, f'{output_name}_vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)

    embeddings_path = os.path.join(output_dir, f'{output_name}_vectors.npy')
    np.save(embeddings_path, embeddings)

    print(f"  Vocabulary saved: {vocab_path}")
    print(f"  Embeddings saved: {embeddings_path}")

    return vocab_path, embeddings_path

# ============================================================

# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Federated dataset generator for HDFS, BGL and Thunderbird',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mix_federated_generator.py
  python mix_federated_generator.py --input_dir my_data --output_name custom_mixed
  python mix_federated_generator.py --max_seq_len 50 --w2v_dim 200
        """
    )


    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                        help=f'Client CSV directory (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--output_name', type=str, default=DEFAULT_OUTPUT_NAME,
                        help=f'Output filename prefix (default: {DEFAULT_OUTPUT_NAME})')


    parser.add_argument('--w2v_dim', type=int, default=W2V_DIM,
                        help=f'Word2Vec vector dimension (default: {W2V_DIM})')
    parser.add_argument('--w2v_seed', type=int, default=W2V_SEED,
                        help=f'Word2Vec random seed (default: {W2V_SEED})')


    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN,
                        help=f'Maximum sequence length; use 0 for automatic detection (default: {MAX_SEQ_LEN})')


    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip generation when the output file exists')

    args = parser.parse_args()


    np.random.seed(args.w2v_seed)

    print("=" * 70)
    print("Federated dataset generator for HDFS, BGL and Thunderbird")
    print("=" * 70)


    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    print(f"Input directory: {os.path.abspath(input_dir)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Output name: {args.output_name}")
    print(f"Word2Vec dimension: {args.w2v_dim}")
    print(f"Random seed: {args.w2v_seed}")


    npz_path = os.path.join(args.output_dir, f'{args.output_name}.npz')
    if args.skip_existing and os.path.exists(npz_path):
        print(f"\nOutput already exists: {npz_path}")
        print("Output exists and --skip_existing is set; skipping generation.")
        sys.exit(0)


    print("\n[1/6] Scanning client files...")
    clients = scan_all_clients(input_dir)

    if not clients:
        print(f"[ERROR] Directory {input_dir} has no client files")
        print("  Supported file formats:")
        print("  - HDFS format: client_{id}_train.csv")
        print("  - BGL format: client{id}_train.csv")
        print("  - Thunderbird format: client{id}_train_sequential.csv")
        print("  - Shuffle format: client{id}_train_shuffle.csv")
        sys.exit(1)

    hdfs_clients = [c for c in clients if c[0] == 'hdfs']
    bgl_clients = [c for c in clients if c[0] == 'bgl']
    tb_clients = [c for c in clients if c[0] == 'tb']
    shuffle_clients = [c for c in clients if c[0] == 'shuffle']

    print(f"  Found {len(clients)} clients:")
    print(f"    - HDFS client: {len(hdfs_clients)} ")
    print(f"    - BGL client: {len(bgl_clients)} ")
    print(f"    - Thunderbird client: {len(tb_clients)} ")
    print(f"    - Shuffle client: {len(shuffle_clients)} ")

    if hdfs_clients:
        hdfs_ids = [cid for _, cid in hdfs_clients]
        print(f"      HDFS client ID: {sorted(hdfs_ids)}")

    if bgl_clients:
        bgl_ids = [cid for _, cid in bgl_clients]
        print(f"      BGL client ID: {sorted(bgl_ids)}")

    if tb_clients:
        tb_ids = [cid for _, cid in tb_clients]
        print(f"      Thunderbird client ID: {sorted(tb_ids)}")

    if shuffle_clients:
        shuffle_ids = [cid for _, cid in shuffle_clients]
        print(f"      Shuffle client ID: {sorted(shuffle_ids)}")


    print("\n[2/6] Loading client data...")
    client_data_list = []

    for dataset_type, client_id in tqdm(clients, desc="Loading clients"):
        data = load_client_data(input_dir, dataset_type, client_id)
        client_data_list.append((dataset_type, client_id, data))

    print(f"  Loaded {len(client_data_list)} clients")


    print("\n[3/6] Extracting sequences and training Word2Vec...")
    all_train_sequences = extract_all_train_sequences(client_data_list)

    if len(all_train_sequences) == 0:
        print("[ERROR] No training sequences available")
        sys.exit(1)

    word2vec_model = train_word2vec_model(all_train_sequences, args.w2v_dim, args.w2v_seed)


    print("\n[4/6] Creating vocabulary and embeddings...")
    vocab, embeddings = create_vocab_and_embeddings(word2vec_model)


    print("\n[5/6] Determining sequence length...")
    if args.max_seq_len <= 0:
        max_seq_len = calculate_max_sequence_length(client_data_list)
        print(f"  Detected maximum sequence length: {max_seq_len}")
    else:
        max_seq_len = args.max_seq_len
        print(f"  Configured maximum sequence length: {max_seq_len}")


    os.makedirs(args.output_dir, exist_ok=True)


    print("\n[6/6] Building federated NPZ file...")
    npz_path, client_stats = build_federated_npz(client_data_list, vocab, embeddings, max_seq_len, npz_path)


    vocab_path, embeddings_path = save_vocab_and_embeddings(
        vocab, embeddings, args.output_dir, args.output_name
    )


    print("\n" + "=" * 70)
    print("Mixed federated dataset generation complete!")
    print("=" * 70)
    print(f"Output files:")
    print(f"  - {npz_path}")
    print(f"  - {vocab_path}")
    print(f"  - {embeddings_path}")
    print(f"\nDataset composition:")
    print(f"  - HDFS client: {len(hdfs_clients)} ")
    print(f"  - BGL client: {len(bgl_clients)} ")
    print(f"  - Thunderbird client: {len(tb_clients)} ")
    print(f"  - Shuffle client: {len(shuffle_clients)} ")
    print(f"  - Total clients: {len(clients)} ")
    print(f"  - Maximum sequence length: {max_seq_len}")
    print(f"  - Feature dimension: {embeddings.shape[1]}")
    print(f"  - Vocabulary size: {len(vocab)}")
    print("=" * 70)


    usage_file = os.path.join(args.output_dir, f"{args.output_name}_usage.txt")
    with open(usage_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Mixed federated dataset usage\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset composition: {len(hdfs_clients)} HDFS client + {len(bgl_clients)} BGL client + {len(tb_clients)} Thunderbird client + {len(shuffle_clients)} Shuffle client\n")
        f.write(f"Total clients: {len(clients)}\n")
        f.write(f"Maximum sequence length: {max_seq_len}\n")
        f.write(f"Feature dimension: {embeddings.shape[1]}\n\n")
        f.write("Loading example:\n")
        f.write("```python\n")
        f.write("import numpy as np\n\n")
        f.write("# Load data\n")
        f.write("data = np.load('{}', allow_pickle=True)\n".format(os.path.basename(npz_path)))
        f.write("\n# Get data\n")
        f.write("X_train = data['train_features']  # Training features\n")
        f.write("y_train = data['train_labels']    # Training labels\n")
        f.write("train_lengths = data['train_lengths']  # Actual sequence length\n")
        f.write("\n# Get client partitions\n")
        f.write("dict_users_train = data['dict_users_train'].item()\n")
        f.write("client_ids = data['client_ids']\n")
        f.write("```\n")

    print(f"Usage instructions saved: {usage_file}")

if __name__ == "__main__":
    main()
