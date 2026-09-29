# prepare_fed_thunderbird_10m_logs_shuffle.py


import os
import pandas as pd
import numpy as np
import json
import sys
import hashlib
import re
from collections import defaultdict
from tqdm import tqdm
from gensim.models import Word2Vec
import random
from datetime import datetime, timedelta
from Drain import LogParser, Node, Logcluster

# ----------------------------

# ----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


THUNDERBIRD_LOG_FILE = 'Thunderbird.log'
DRAIN_CONFIG = {
    'log_format': '<Content>',
    'depth': 4,
    'st': 0.4,
    'maxChild': 100,
    'rex': []
}


OUTPUT_DIR = 'fed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


NUM_CLIENTS_TOTAL = 100
NUM_CLIENTS_SELECTED = 10
NUM_CLIENTS = NUM_CLIENTS_SELECTED


TARGET_LOG_COUNT = 10_000_000


TIME_WINDOW_SECONDS = 60
TIME_STEP_SECONDS = 30
MAX_WINDOW_EVENTS = 20


TRAIN_RATIO = 0.5
VAL_RATIO = 0.2
TEST_RATIO = 0.3

# Word2Vec
W2V_DIM = 300
MAX_SEQ_LEN = MAX_WINDOW_EVENTS
FEATURE_DIM = 300

# ----------------------------

# ----------------------------
def parse_timestamp_thunderbird(ts_str):
    """Parse a Thunderbird timestamp such as Nov 10 00:05:01 using year 2005."""
    if not ts_str or ts_str.strip() == '':
        return None
    try:
        ts_with_year = f"2005 {ts_str}"
        return datetime.strptime(ts_with_year, "%Y %b %d %H:%M:%S")
    except ValueError:
        return None

# ----------------------------

# ----------------------------
print("[OK] Thunderbird log sampling...")
print(f"  Target log entries: {TARGET_LOG_COUNT:,}")
print(f"  Time window: {TIME_WINDOW_SECONDS} seconds, Stride: {TIME_STEP_SECONDS} seconds")

timestamp_pattern = re.compile(r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})')

log_entries = []
total_anomaly = 0

print(f"\n[INFO] Reading first {TARGET_LOG_COUNT:,} log entries...")

with open(THUNDERBIRD_LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    for line_idx, line in enumerate(tqdm(f, desc="Reading logs", total=TARGET_LOG_COUNT)):
        if line_idx >= TARGET_LOG_COUNT:
            break

        line = line.strip()
        if not line:
            continue


        match = timestamp_pattern.search(line)
        if not match:
            continue

        ts_str = match.group(1)
        timestamp = parse_timestamp_thunderbird(ts_str)
        if timestamp is None:
            continue


        label = 0 if line.startswith("-") else 1
        if label == 1:
            total_anomaly += 1

        log_entries.append({
            'content': line,
            'label': label,
            'timestamp': timestamp
        })

print(f"\n[RESULT] Log reading complete:")
print(f"  Log entries read: {len(log_entries):,}")
print(f"  Anomalous logs: {total_anomaly:,}")
print(f"  Anomaly ratio: {total_anomaly/len(log_entries):.2%}")


print("\n[OK] Sorting logs by timestamp...")
log_entries.sort(key=lambda x: x['timestamp'])


first_ts = log_entries[0]['timestamp']
last_ts = log_entries[-1]['timestamp']
time_span = (last_ts - first_ts).total_seconds()
print(f"  Duration: {time_span:.0f} seconds ({time_span/3600:.1f} hours, {time_span/86400:.1f} days)")
print(f"  Start time: {first_ts}")
print(f"  End time: {last_ts}")

# ----------------------------

# ----------------------------
print(f"\n[OK] Parsing log templates with Drain...")

all_lines = [e['content'] for e in log_entries]
all_labels = [e['label'] for e in log_entries]
all_timestamps = [e['timestamp'] for e in log_entries]


print("[OK] Deduplicating log messages...")
line_to_indices = defaultdict(list)
for i, line in enumerate(all_lines):
    line_to_indices[line].append(i)

unique_lines = list(line_to_indices.keys())
unique_count = len(unique_lines)
original_count = len(all_lines)

print(f"  Original log count: {original_count:,}")
print(f"  Unique log count: {unique_count:,}")
print(f"  Duplicate fraction: {(1 - unique_count/original_count):.1%}")


df_log = pd.DataFrame({"LineId": range(1, unique_count+1), "Content": unique_lines})
parser = LogParser(log_format=DRAIN_CONFIG["log_format"],
                   indir=".", outdir=OUTPUT_DIR,
                   depth=DRAIN_CONFIG["depth"], st=DRAIN_CONFIG["st"],
                   maxChild=DRAIN_CONFIG["maxChild"], rex=DRAIN_CONFIG["rex"],
                   keep_para=False)
parser.df_log = df_log
parser.logName = "Thunderbird"
parser.savePath = OUTPUT_DIR

rootNode = Node()
logCluL = []
for idx, row in tqdm(df_log.iterrows(), total=len(df_log), desc="Drain parsing"):
    logID = row["LineId"]
    logmessageL = parser.preprocess(row["Content"]).strip().split()
    matchCluster = parser.treeSearch(rootNode, logmessageL)
    if matchCluster is None:
        newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
        logCluL.append(newCluster)
        parser.addSeqToPrefixTree(rootNode, newCluster)
    else:
        newTemplate = parser.getTemplate(logmessageL, matchCluster.logTemplate)
        matchCluster.logIDL.append(logID)
        if " ".join(newTemplate) != " ".join(matchCluster.logTemplate):
            matchCluster.logTemplate = newTemplate


unique_template_ids = [0] * unique_count
for logClust in logCluL:
    template_str = " ".join(logClust.logTemplate)
    template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]
    for logID in logClust.logIDL:
        unique_template_ids[logID-1] = template_id

print(f"  Unique templates: {len(set(unique_template_ids))}")


template_ids = [None] * original_count
for unique_idx, template_id in enumerate(unique_template_ids):
    original_line = unique_lines[unique_idx]
    for original_idx in line_to_indices[original_line]:
        template_ids[original_idx] = template_id


unique_templates = sorted(set(template_ids))
template_to_idx = {tid: idx+1 for idx, tid in enumerate(unique_templates)}
template_indices = [template_to_idx[tid] for tid in template_ids]

# ----------------------------

# ----------------------------
print(f"\n[OK] Creating time windows...")

block_events = {}
block_log_counts = {}
label_dict = {}
all_windows = []


window_starts = []
current = first_ts
while current + timedelta(seconds=TIME_WINDOW_SECONDS) <= last_ts:
    window_starts.append(current)
    current += timedelta(seconds=TIME_STEP_SECONDS)

print(f"  Available windows: {len(window_starts):,}")


time_data = list(zip(all_timestamps, template_indices, all_labels))

block_id_counter = 0
data_idx = 0

for w_start in tqdm(window_starts, desc="Generating windows"):
    w_end = w_start + timedelta(seconds=TIME_WINDOW_SECONDS)


    while data_idx < len(time_data) and time_data[data_idx][0] < w_start:
        data_idx += 1


    window_indices = []
    window_labels = []
    temp_idx = data_idx
    while temp_idx < len(time_data) and time_data[temp_idx][0] < w_end:
        _, idx, label = time_data[temp_idx]
        window_indices.append(idx)
        window_labels.append(label)
        temp_idx += 1


    if len(window_indices) == 0:
        continue


    if len(window_indices) > MAX_WINDOW_EVENTS:
        window_indices = window_indices[:MAX_WINDOW_EVENTS]
        window_labels = window_labels[:MAX_WINDOW_EVENTS]


    window_label = 1 if any(l == 1 for l in window_labels) else 0

    blk_id = str(block_id_counter)
    block_events[blk_id] = window_indices
    block_log_counts[blk_id] = len(window_indices)
    label_dict[blk_id] = window_label
    all_windows.append(blk_id)
    block_id_counter += 1

print(f"  Generated windows: {block_id_counter:,}")


print(f"\n{'='*60}")
print("[STAT] Stage 1: Window generation statistics")
print(f"{'='*60}")
normal_windows = sum(1 for l in label_dict.values() if l == 0)
anomaly_windows = sum(1 for l in label_dict.values() if l == 1)
print(f"  Total windows: {block_id_counter:,}")
print(f"  Normal windows: {normal_windows:,} ({normal_windows/block_id_counter:.2%})")
print(f"  Anomalous windows: {anomaly_windows:,} ({anomaly_windows/block_id_counter:.2%})")
print(f"{'='*60}")

# ----------------------------

# ----------------------------
valid_sessions = {}
for blk, events in block_events.items():
    if len(events) > 0:
        valid_sessions[blk] = {
            'event_ids': events,
            'label': label_dict[blk],
            'log_count': block_log_counts[blk]
        }

# ----------------------------

# ----------------------------
print(f"\n[OK] Assigning windows to {NUM_CLIENTS_TOTAL} clients...")

client_blocks = defaultdict(list)
for i, blk_id in enumerate(all_windows):
    cid = i % NUM_CLIENTS_TOTAL
    client_blocks[cid].append(blk_id)


client_anomaly_ratios = {}
client_stats_all = {}

for cid in range(NUM_CLIENTS_TOTAL):
    block_ids = client_blocks[cid]
    if not block_ids:
        client_anomaly_ratios[cid] = 0
        client_stats_all[cid] = {'normal_blocks': 0, 'anomaly_blocks': 0, 'total_blocks': 0, 'anomaly_ratio': 0}
        continue

    anomaly_count = sum(1 for blk in block_ids if label_dict[blk] == 1)
    ratio = anomaly_count / len(block_ids)
    client_anomaly_ratios[cid] = ratio
    client_stats_all[cid] = {
        'normal_blocks': len(block_ids) - anomaly_count,
        'anomaly_blocks': anomaly_count,
        'total_blocks': len(block_ids),
        'anomaly_ratio': ratio
    }


valid_clients = [(cid, ratio) for cid, ratio in client_anomaly_ratios.items()
                  if 0.05 <= ratio <= 0.50 and client_stats_all[cid]['total_blocks'] > 0]
valid_clients.sort(key=lambda x: x[1])


print(f"\n{'='*60}")
print("[STAT] Stage 2: 100-client allocation statistics")
print(f"{'='*60}")
total_assigned = sum(client_stats_all[cid]['total_blocks'] for cid in range(NUM_CLIENTS_TOTAL))
total_anomaly_assigned = sum(client_stats_all[cid]['anomaly_blocks'] for cid in range(NUM_CLIENTS_TOTAL))
print(f"  Total assigned windows: {total_assigned:,}")
print(f"  Normal windows: {total_assigned - total_anomaly_assigned:,} ({(total_assigned-total_anomaly_assigned)/total_assigned:.2%})")
print(f"  Anomalous windows: {total_anomaly_assigned:,} ({total_anomaly_assigned/total_assigned:.2%})")
print(f"  Eligible clients (5%-50% anomalies): {len(valid_clients)}")
if valid_clients:
    print(f"  Anomaly-ratio range: {valid_clients[0][1]:.2%} - {valid_clients[-1][1]:.2%}")
print(f"{'='*60}")


if len(valid_clients) >= NUM_CLIENTS_SELECTED:
    indices = np.linspace(0, len(valid_clients)-1, NUM_CLIENTS_SELECTED, dtype=int)
    selected_client_ids = sorted([valid_clients[i][0] for i in indices])
else:
    selected_client_ids = sorted([cid for cid, _ in valid_clients])
    NUM_CLIENTS_SELECTED = len(selected_client_ids)

NUM_CLIENTS = NUM_CLIENTS_SELECTED
print(f"\n[OK] Select {NUM_CLIENTS_SELECTED} clients: {selected_client_ids}")

# ----------------------------

# ----------------------------
print(f"\n[OK] Within-client 5:2:3 split...")

client_train_blks = defaultdict(list)
client_val_blks = defaultdict(list)
client_test_blks = defaultdict(list)
client_stats = {}

for cid in selected_client_ids:
    blks = client_blocks[cid].copy()
    np.random.shuffle(blks)

    n_total = len(blks)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    client_train_blks[cid] = blks[:n_train]
    client_val_blks[cid] = blks[n_train:n_train + n_val]
    client_test_blks[cid] = blks[n_train + n_val:]

    train_anomaly = sum(1 for blk in client_train_blks[cid] if label_dict[blk] == 1)
    val_anomaly = sum(1 for blk in client_val_blks[cid] if label_dict[blk] == 1)
    test_anomaly = sum(1 for blk in client_test_blks[cid] if label_dict[blk] == 1)

    client_stats[cid] = {
        'num_blocks': n_total,
        'train_blocks': n_train,
        'val_blocks': n_val,
        'test_blocks': n_total - n_train - n_val,
        'train_anomaly': train_anomaly,
        'val_anomaly': val_anomaly,
        'test_anomaly': test_anomaly,
        'anomaly_ratio': (train_anomaly + val_anomaly + test_anomaly) / n_total if n_total > 0 else 0
    }


print(f"\n{'='*60}")
print("[STAT] Stage 3: Training/validation/test split statistics")
print(f"{'='*60}")

total_train = sum(client_stats[cid]['train_blocks'] for cid in selected_client_ids)
total_val = sum(client_stats[cid]['val_blocks'] for cid in selected_client_ids)
total_test = sum(client_stats[cid]['test_blocks'] for cid in selected_client_ids)
total_train_anomaly = sum(client_stats[cid]['train_anomaly'] for cid in selected_client_ids)
total_val_anomaly = sum(client_stats[cid]['val_anomaly'] for cid in selected_client_ids)
total_test_anomaly = sum(client_stats[cid]['test_anomaly'] for cid in selected_client_ids)

print(f"\n  [Global statistics]")
print(f"  Training set: {total_train:,} windows, anomalous {total_train_anomaly:,} ({total_train_anomaly/total_train:.2%})")
print(f"  Validation set: {total_val:,} windows, anomalous {total_val_anomaly:,} ({total_val_anomaly/total_val:.2%})")
print(f"  Test set: {total_test:,} windows, anomalous {total_test_anomaly:,} ({total_test_anomaly/total_test:.2%})")

print(f"\n  [Detailed client statistics]")
print(f"  {'Client':<8} {'Total windows':<8} {'Training set':<8} {'Train anomalies':<10} {'Validation set':<8} {'Validation anomalies':<10} {'Test set':<8} {'Test anomalies':<10} {'Anomaly ratio':<10}")
print(f"  {'-'*90}")
for cid in selected_client_ids:
    s = client_stats[cid]
    print(f"  {cid:<8} {s['num_blocks']:<8} {s['train_blocks']:<8} {s['train_anomaly']:<10} "
          f"{s['val_blocks']:<8} {s['val_anomaly']:<10} {s['test_blocks']:<8} {s['test_anomaly']:<10} {s['anomaly_ratio']:.2%}")
print(f"{'='*60}")

# ----------------------------

# ----------------------------
print(f"\n[OK] Building vocabulary and training Word2Vec...")

train_sequences = []
for cid in selected_client_ids:
    for blk in client_train_blks[cid]:
        if blk in valid_sessions:
            train_sequences.append([str(e) for e in valid_sessions[blk]['event_ids']])

all_events = set(e for seq in train_sequences for e in map(int, seq))
vocab = {'<PAD>': 0}
for idx, event in enumerate(sorted(all_events), 1):
    vocab[event] = idx

with open(os.path.join(OUTPUT_DIR, 'vocab_10client_5-2-3_shuffle.json'), 'w') as f:
    json.dump(vocab, f, indent=2)

w2v_model = Word2Vec(sentences=train_sequences, vector_size=W2V_DIM, window=5, min_count=1, workers=4, seed=RANDOM_SEED, sg=1)

vocab_events = sorted([eid for eid in vocab.keys() if eid != '<PAD>'], key=int)
final_vectors = np.array([w2v_model.wv[str(e)] for e in vocab_events], dtype=np.float32)
np.save(os.path.join(OUTPUT_DIR, 'event_vectors_w2v300_10client_5-2-3_shuffle.npy'), final_vectors)


semantic_vectors = {str(e): w2v_model.wv[str(e)].tolist() for e in vocab_events}
with open(os.path.join(OUTPUT_DIR, 'thunderbird_semantic_vec_shuffle.json'), 'w') as f:
    json.dump(semantic_vectors, f, indent=2)

print(f"  Vocabulary size: {len(vocab)}")

# ----------------------------

# ----------------------------
print(f"\n[OK] Saving CSV file...")

def blocks_to_minimal_df(blk_list):
    sequences, labels = [], []
    for blk in blk_list:
        if blk in valid_sessions:
            sequences.append(' '.join(map(str, valid_sessions[blk]['event_ids'])))
            labels.append(valid_sessions[blk]['label'])
    return pd.DataFrame({'sequence': sequences, 'label': labels})


blocks_to_minimal_df(sum([client_train_blks[cid] for cid in selected_client_ids], [])).to_csv(
    os.path.join(OUTPUT_DIR, 'log_train_shuffle.csv'), index=False)
blocks_to_minimal_df(sum([client_val_blks[cid] for cid in selected_client_ids], [])).to_csv(
    os.path.join(OUTPUT_DIR, 'log_val_shuffle.csv'), index=False)
blocks_to_minimal_df(sum([client_test_blks[cid] for cid in selected_client_ids], [])).to_csv(
    os.path.join(OUTPUT_DIR, 'log_test_shuffle.csv'), index=False)


for cid in selected_client_ids:
    blocks_to_minimal_df(client_train_blks[cid]).to_csv(os.path.join(OUTPUT_DIR, f'client{cid}_train_shuffle.csv'), index=False)
    blocks_to_minimal_df(client_val_blks[cid]).to_csv(os.path.join(OUTPUT_DIR, f'client{cid}_val_shuffle.csv'), index=False)
    blocks_to_minimal_df(client_test_blks[cid]).to_csv(os.path.join(OUTPUT_DIR, f'client{cid}_test_shuffle.csv'), index=False)

print(f"  Training set: {total_train:,}, Validation set: {total_val:,}, Test set: {total_test:,}")

# ----------------------------

# ----------------------------
print(f"\n[OK] Building federated .npz file...")

event_embeddings = np.load(os.path.join(OUTPUT_DIR, 'event_vectors_w2v300_10client_5-2-3_shuffle.npy'))
pad_vector = np.zeros((1, event_embeddings.shape[1]), dtype=np.float32)
event_embeddings_with_pad = np.vstack([pad_vector, event_embeddings])

def seq_to_tensor(seq_str, max_len=MAX_SEQ_LEN):
    events = list(map(int, str(seq_str).split())) if seq_str else []
    events = (events[:max_len] + [0] * max_len)[:max_len]
    indices = [vocab.get(e, 0) for e in events]
    return event_embeddings_with_pad[indices]

def blocks_to_df(blk_list):
    seqs, labels = [], []
    for blk in blk_list:
        if blk in valid_sessions:
            seqs.append(' '.join(map(str, valid_sessions[blk]['event_ids'])))
            labels.append(valid_sessions[blk]['label'])
    return pd.DataFrame({'sequence': seqs, 'label': labels})

all_train_features, all_train_labels = [], []
all_val_features, all_val_labels = [], []
all_test_features, all_test_labels = [], []
dict_users_train, dict_users_val, dict_users_test = {}, {}, {}
current_train_idx, current_val_idx, current_test_idx = 0, 0, 0

for cid in selected_client_ids:
    train_df = blocks_to_df(client_train_blks[cid])
    if len(train_df) > 0:
        X = np.stack([seq_to_tensor(s) for s in train_df['sequence']])
        y = train_df['label'].values.astype(np.int64)
        all_train_features.append(X)
        all_train_labels.append(y)
        dict_users_train[cid] = np.arange(current_train_idx, current_train_idx + len(X))
        current_train_idx += len(X)

    val_df = blocks_to_df(client_val_blks[cid])
    if len(val_df) > 0:
        X = np.stack([seq_to_tensor(s) for s in val_df['sequence']])
        y = val_df['label'].values.astype(np.int64)
        all_val_features.append(X)
        all_val_labels.append(y)
        dict_users_val[cid] = np.arange(current_val_idx, current_val_idx + len(X))
        current_val_idx += len(X)

    test_df = blocks_to_df(client_test_blks[cid])
    if len(test_df) > 0:
        X = np.stack([seq_to_tensor(s) for s in test_df['sequence']])
        y = test_df['label'].values.astype(np.int64)
        all_test_features.append(X)
        all_test_labels.append(y)
        dict_users_test[cid] = np.arange(current_test_idx, current_test_idx + len(X))
        current_test_idx += len(X)

X_train_all = np.vstack(all_train_features) if all_train_features else np.array([])
y_train_all = np.concatenate(all_train_labels) if all_train_labels else np.array([])
X_val_all = np.vstack(all_val_features) if all_val_features else np.array([])
y_val_all = np.concatenate(all_val_labels) if all_val_labels else np.array([])
X_test_all = np.vstack(all_test_features) if all_test_features else np.array([])
y_test_all = np.concatenate(all_test_labels) if all_test_labels else np.array([])

np.savez(os.path.join(OUTPUT_DIR, 'thunderbird_fed_unified_10client_5-2-3_shuffle.npz'),
         train_features=X_train_all.astype(np.float32) if len(X_train_all) > 0 else np.array([]),
         train_labels=y_train_all.astype(np.int64) if len(y_train_all) > 0 else np.array([]),
         val_features=X_val_all.astype(np.float32) if len(X_val_all) > 0 else np.array([]),
         val_labels=y_val_all.astype(np.int64) if len(y_val_all) > 0 else np.array([]),
         test_features=X_test_all.astype(np.float32) if len(X_test_all) > 0 else np.array([]),
         test_labels=y_test_all.astype(np.int64) if len(y_test_all) > 0 else np.array([]),
         dict_users_train=dict_users_train,
         dict_users_val=dict_users_val,
         dict_users_test=dict_users_test,
         client_ids=np.array(selected_client_ids, dtype=np.int32),
         max_seq_len=MAX_SEQ_LEN,
         feature_dim=FEATURE_DIM)


print(f"\n{'='*60}")
print("[STAT] Stage 4: Final NPZ statistics")
print(f"{'='*60}")

train_anomaly = int(y_train_all.sum()) if len(y_train_all) > 0 else 0
val_anomaly = int(y_val_all.sum()) if len(y_val_all) > 0 else 0
test_anomaly = int(y_test_all.sum()) if len(y_test_all) > 0 else 0

print(f"  Training set: {len(X_train_all):,} samples, anomalous {train_anomaly:,} ({train_anomaly/len(X_train_all):.2%})" if len(X_train_all) > 0 else "  Training set: empty")
print(f"  Validation set: {len(X_val_all):,} samples, anomalous {val_anomaly:,} ({val_anomaly/len(X_val_all):.2%})" if len(X_val_all) > 0 else "  Validation set: empty")
print(f"  Test set: {len(X_test_all):,} samples, anomalous {test_anomaly:,} ({test_anomaly/len(X_test_all):.2%})" if len(X_test_all) > 0 else "  Test set: empty")

total_samples = len(X_train_all) + len(X_val_all) + len(X_test_all)
total_anomaly_final = train_anomaly + val_anomaly + test_anomaly
print(f"\n  Total: {total_samples:,} samples, anomalous {total_anomaly_final:,} ({total_anomaly_final/total_samples:.2%})" if total_samples > 0 else "  Total: empty")

print(f"\n  NPZ size: {os.path.getsize(os.path.join(OUTPUT_DIR, 'thunderbird_fed_unified_10client_5-2-3_shuffle.npz')) / 1024 / 1024:.2f} MB")
print(f"{'='*60}")

print("\n[CELEBRATE] Thunderbird preprocessing complete!")
