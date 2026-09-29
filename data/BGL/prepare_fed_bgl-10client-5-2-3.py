# prepare_fed_bgl_10client_5-2-3.py
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


BGL_LOG_FILE = 'BGL.log'
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


SAMPLES_FOR_FEDERATION = 100000


TIME_WINDOW_SECONDS = 60
TIME_STEP_SECONDS = 20
MAX_WINDOW_EVENTS = 20


TARGET_TOTAL_WINDOWS = 100000


TRAIN_RATIO = 0.5
VAL_RATIO = 0.2
TEST_RATIO = 0.3


W2V_DIM = 300


MAX_SEQ_LEN = MAX_WINDOW_EVENTS
FEATURE_DIM = 300

# ----------------------------

# ----------------------------
print("[OK] Reading BGL logs and parsing timestamps...")

with open(BGL_LOG_FILE, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)


timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6}')

log_entries = []
with open(BGL_LOG_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading log lines", total=total_lines):
        line = line.strip()
        if not line:
            continue

        label = 0 if line.startswith("-") else 1

        match = timestamp_pattern.search(line)
        if match:
            timestamp_str = match.group(0)

            log_entries.append({
                'content': line,
                'label': label,
                'timestamp': timestamp_str
            })
        else:

            log_entries.append({
                'content': line,
                'label': label,
                'timestamp': ''
            })

print(f"  Total log lines: {len(log_entries)}")
print(f"  Anomalous line fraction: {sum(entry['label'] for entry in log_entries)/len(log_entries):.2%}")


print("[OK] Sorting logs by timestamp...")
log_entries.sort(key=lambda x: x['timestamp'])


print("[OK] Using the full log sequence...")
selected_entries = log_entries
print(f"  Full-data log entries: {len(selected_entries)}")
print(f"  Full-data anomaly ratio: {sum(entry['label'] for entry in selected_entries)/len(selected_entries):.2%}")


print("[OK] Preparing global log data...")
NUM_CLIENTS_TOTAL = 100

all_selected_lines = []
all_labels = []
all_timestamps = []

for entry in selected_entries:
    all_selected_lines.append(entry['content'])
    all_labels.append(entry['label'])
    all_timestamps.append(entry['timestamp'])

print(f"  Selected log entries: {len(all_selected_lines)}")
print(f"  Global anomaly ratio: {sum(all_labels)/len(all_labels):.2%}")


print("[OK] Parsing log templates with Drain...")
df_log = pd.DataFrame({"LineId": range(1, len(all_selected_lines)+1), "Content": all_selected_lines})
parser = LogParser(log_format=DRAIN_CONFIG["log_format"],
                   indir=".",
                   outdir=OUTPUT_DIR,
                   depth=DRAIN_CONFIG["depth"],
                   st=DRAIN_CONFIG["st"],
                   maxChild=DRAIN_CONFIG["maxChild"],
                   rex=DRAIN_CONFIG["rex"],
                   keep_para=False)

parser.df_log = df_log
parser.logName = "BGL"
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


template_ids = [0] * len(df_log)
for logClust in logCluL:
    template_str = " ".join(logClust.logTemplate)
    template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]
    for logID in logClust.logIDL:
        template_ids[logID-1] = template_id

print(f"  Unique templates: {len(set(template_ids))}")


unique_templates = sorted(set(template_ids))
template_to_idx = {tid: idx+1 for idx, tid in enumerate(unique_templates)}
template_indices = [template_to_idx[tid] for tid in template_ids]


print("[OK] Preparing global template indices...")

global_indices = template_indices
global_labels = all_labels
global_timestamps = all_timestamps

print(f"  Global template index count: {len(global_indices)}")
print(f"  Global anomaly ratio: {sum(global_labels)/len(global_labels):.2%}")


print("[OK] Creating global time windows...")
block_events = {}
block_log_counts = {}
label_dict = {}
client_blocks = defaultdict(list)


def parse_timestamp(ts_str):


    if not ts_str or ts_str.strip() == '':
        return datetime.min
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d-%H.%M.%S.%f")
    except ValueError:

        if '.' in ts_str and ts_str.count('.') == 3:

            parts = ts_str.split('.')
            ts_str_no_us = '.'.join(parts[:3])
            return datetime.strptime(ts_str_no_us, "%Y-%m-%d-%H.%M.%S")
        else:

            return datetime.min

print("[STAT] Using global log data...")
print(f"  Global log entries: {len(global_indices)}")
print(f"  Global anomaly ratio: {sum(global_labels)/len(global_labels):.2%}")

print("[OK] Parsing timestamps and creating windows...")

dt_list = [parse_timestamp(ts) for ts in tqdm(global_timestamps, desc="Parsing timestamps")]

valid_dt_list = [dt for dt in dt_list if dt != datetime.min]
if len(valid_dt_list) == 0:
    print("[ERROR] No valid timestamps; cannot create windows")
    sys.exit(1)

start_time = min(valid_dt_list)
end_time = max(valid_dt_list)
duration_seconds = (end_time - start_time).total_seconds()

print(f"  Time range: {start_time} to {end_time}")
print(f"  Total duration: {duration_seconds:.0f} seconds ({duration_seconds/3600:.1f} hours)")


window_starts = []
current = start_time
while current + timedelta(seconds=TIME_WINDOW_SECONDS) <= end_time:
    window_starts.append(current)
    current += timedelta(seconds=TIME_STEP_SECONDS)

print(f"  Available: {len(window_starts)} time windows (start points)")


print("[OK] Building timestamp index...")

time_data = list(zip(dt_list, global_indices, global_labels))

print("[OK] Generating time windows...")
block_id_counter = 0
all_windows = []


data_idx = 0
for w_start in tqdm(window_starts, desc="Generating windows"):
    w_end = w_start + timedelta(seconds=TIME_WINDOW_SECONDS)


    while data_idx < len(time_data) and time_data[data_idx][0] < w_start:
        data_idx += 1


    window_indices = []
    window_labels = []
    temp_idx = data_idx
    while temp_idx < len(time_data) and time_data[temp_idx][0] < w_end:
        dt, idx, label = time_data[temp_idx]
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

print(f"  Generated {block_id_counter} nonempty time windows")
print(f"  Window anomaly ratio: {sum(label_dict.values())/len(label_dict):.2%}")


normal_windows = sum(1 for label in label_dict.values() if label == 0)
anomaly_windows = sum(1 for label in label_dict.values() if label == 1)
print(f"[STAT] Window distribution:")
print(f"  Normal windows: {normal_windows} ({normal_windows/len(label_dict):.2%})")
print(f"  Anomalous windows: {anomaly_windows} ({anomaly_windows/len(label_dict):.2%})")


if block_id_counter > TARGET_TOTAL_WINDOWS:
    print(f"[WARN] Total windows {block_id_counter} exceeds target {TARGET_TOTAL_WINDOWS}, sampling randomly...")
    np.random.shuffle(all_windows)
    keep_windows = all_windows[:TARGET_TOTAL_WINDOWS]

    new_block_events = {}
    new_block_log_counts = {}
    new_label_dict = {}
    new_all_windows = []
    new_block_id_counter = 0

    old_to_new = {}
    for old_blk_id in keep_windows:
        new_blk_id = str(new_block_id_counter)
        old_to_new[old_blk_id] = new_blk_id
        new_block_events[new_blk_id] = block_events[old_blk_id]
        new_block_log_counts[new_blk_id] = block_log_counts[old_blk_id]
        new_label_dict[new_blk_id] = label_dict[old_blk_id]
        new_all_windows.append(new_blk_id)
        new_block_id_counter += 1

    block_events = new_block_events
    block_log_counts = new_block_log_counts
    label_dict = new_label_dict
    all_windows = new_all_windows
    block_id_counter = new_block_id_counter
    print(f"  Retained after sampling: {block_id_counter} windows")
elif block_id_counter < TARGET_TOTAL_WINDOWS:
    print(f"[INFO] Total windows {block_id_counter} below target {TARGET_TOTAL_WINDOWS}, using all windows")

print(f"[OK] Final window count: {block_id_counter}")
print(f"  Final window anomaly ratio: {sum(label_dict.values())/len(label_dict):.2%}")

# ----------------------------

# ----------------------------
print("\n[OK] Assigning windows to 100 clients...")

client_blocks = defaultdict(list)




for i, blk_id in enumerate(all_windows):

    cid = i % NUM_CLIENTS_TOTAL
    client_blocks[cid].append(blk_id)

print(f"  Assigned {block_id_counter} windows to {NUM_CLIENTS_TOTAL} clients")
print(f"  Mean windows per client: {block_id_counter / NUM_CLIENTS_TOTAL:.1f}")


print("\n[STAT] Client anomaly-ratio statistics:")
client_anomaly_ratios = {}
client_stats_all = {}
total_assigned_normal = 0
total_assigned_anomaly = 0

for cid in range(NUM_CLIENTS_TOTAL):
    block_ids = client_blocks[cid]
    if not block_ids:
        client_anomaly_ratios[cid] = 0
        client_stats_all[cid] = {
            'normal_blocks': 0,
            'anomaly_blocks': 0,
            'total_blocks': 0,
            'anomaly_ratio': 0
        }
        continue

    anomaly_count = sum(1 for blk in block_ids if label_dict[blk] == 1)
    normal_count = len(block_ids) - anomaly_count
    ratio = anomaly_count / len(block_ids)

    client_anomaly_ratios[cid] = ratio
    client_stats_all[cid] = {
        'normal_blocks': normal_count,
        'anomaly_blocks': anomaly_count,
        'total_blocks': len(block_ids),
        'anomaly_ratio': ratio
    }

    total_assigned_normal += normal_count
    total_assigned_anomaly += anomaly_count

total_assigned = total_assigned_normal + total_assigned_anomaly

print(f"[STAT] 100-client allocation summary:")
print(f"  Total assigned windows: {total_assigned:,}")
print(f"  Assigned normal windows: {total_assigned_normal:,} ({total_assigned_normal/total_assigned:.2%})")
print(f"  Assigned anomalous windows: {total_assigned_anomaly:,} ({total_assigned_anomaly/total_assigned:.2%})")


print("\n[OK] Filtering clients with anomaly ratios between 5% and 50%...")
valid_clients = [(cid, ratio) for cid, ratio in client_anomaly_ratios.items()
                  if 0.05 <= ratio <= 0.50 and client_stats_all[cid]['total_blocks'] > 0]
valid_clients.sort(key=lambda x: x[1])

print(f"  Eligible clients (5%-50% anomaly ratio): {len(valid_clients)}")
if len(valid_clients) > 0:
    print(f"  Anomaly-ratio range: {valid_clients[0][1]:.2%} - {valid_clients[-1][1]:.2%}")


    print("\n[STAT] First 10 eligible clients:")
    print("-" * 80)
    print(f"{'client ID':<10} {'Anomaly ratio':<15} {'Normal windows':<15} {'Anomalous windows':<15} {'Total windows':<15}")
    print("-" * 80)
    for cid, ratio in valid_clients[:10]:
        stats = client_stats_all[cid]
        print(f"{cid:<10} {ratio:<15.2%} {stats['normal_blocks']:<15} {stats['anomaly_blocks']:<15} {stats['total_blocks']:<15}")
    print("-" * 80)


NUM_CLIENTS_SELECTED = 10
if len(valid_clients) >= NUM_CLIENTS_SELECTED:

    indices = np.linspace(0, len(valid_clients)-1, NUM_CLIENTS_SELECTED, dtype=int)
    selected_client_ids = sorted([valid_clients[i][0] for i in indices])
    print(f"\n[OK] Selecting evenly across eligible clients: {NUM_CLIENTS_SELECTED} clients:")
    print(f"  Selected client ID: {selected_client_ids}")
    print(f"  Anomaly-ratio distribution: {[f'{client_anomaly_ratios[cid]:.2%}' for cid in selected_client_ids]}")
else:

    selected_client_ids = sorted([cid for cid, _ in valid_clients])
    print(f"\n[WARN] Fewer than {NUM_CLIENTS_SELECTED} eligible clients; using {len(selected_client_ids)} clients")
    print(f"  Selected client ID: {selected_client_ids}")
    print(f"  Anomaly-ratio distribution: {[f'{client_anomaly_ratios[cid]:.2%}' for cid in selected_client_ids]}")
    NUM_CLIENTS_SELECTED = len(selected_client_ids)


NUM_CLIENTS = NUM_CLIENTS_SELECTED


print(f"\n[OK] Window generation and client allocation complete!")
print(f"  Generated Block count: {len(block_events)}")
print(f"  Selected clients: {NUM_CLIENTS}")
print(f"  Selected client ID: {selected_client_ids}")



# ----------------------------

# ----------------------------
print("\n[STAT] Computing block type statistics...")
valid_sessions = {}
normal_blks = []
anomaly_blks = []

for blk, events in block_events.items():
    if blk in label_dict and len(events) > 0:
        valid_sessions[blk] = {
            'event_ids': events,
            'label': label_dict[blk],
            'log_count': block_log_counts.get(blk, 0)
        }
        if label_dict[blk] == 0:
            normal_blks.append(blk)
        else:
            anomaly_blks.append(blk)

print(f"  Normal blocks: {len(normal_blks)}")
print(f"  Anomalous blocks: {len(anomaly_blks)}")
print(f"  Valid blocks: {len(valid_sessions)}")


original_anomaly_ratio = len(anomaly_blks) / len(valid_sessions)
print(f"\n[STAT] Raw-data anomaly ratio: {original_anomaly_ratio:.2%}")

# ----------------------------

# ----------------------------
print("\n[STAT] Raw log BGL.log analysis:")

total_log_lines = sum(block_log_counts.values())
print(f"  Log lines across all blocks: {total_log_lines:,}")

normal_log_lines = sum(block_log_counts.get(blk, 0) for blk in normal_blks)
anomaly_log_lines = sum(block_log_counts.get(blk, 0) for blk in anomaly_blks)
print(f"  Log lines in normal blocks: {normal_log_lines:,}")
print(f"  Log lines in anomalous blocks: {anomaly_log_lines:,}")

log_anomaly_ratio = anomaly_log_lines / total_log_lines if total_log_lines > 0 else 0
print(f"  Raw-log anomaly ratio: {log_anomaly_ratio:.2%}")
print(f"  Normal blocks: {len(normal_blks):,}")
print(f"  Anomalous blocks: {len(anomaly_blks):,}")
print(f"  Block-level anomaly ratio: {original_anomaly_ratio:.2%}")



# ----------------------------

# ----------------------------
print("\n[RULER] Within-client 5:2:3 split...")

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
    n_test = n_total - n_train - n_val


    if n_train == 0 or n_val == 0 or n_test == 0:
        print(f"[WARN] Warning: Client {cid} has too few samples; adjusting the split...")
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n_total - n_train - n_val)

    client_train_blks[cid] = blks[:n_train]
    client_val_blks[cid] = blks[n_train:n_train + n_val]
    client_test_blks[cid] = blks[n_train + n_val:]


    train_anomaly = sum(1 for blk in client_train_blks[cid] if valid_sessions[blk]['label'] == 1)
    val_anomaly = sum(1 for blk in client_val_blks[cid] if valid_sessions[blk]['label'] == 1)
    test_anomaly = sum(1 for blk in client_test_blks[cid] if valid_sessions[blk]['label'] == 1)

    train_labels = [valid_sessions[blk]['label'] for blk in client_train_blks[cid]]
    val_labels = [valid_sessions[blk]['label'] for blk in client_val_blks[cid]]
    test_labels = [valid_sessions[blk]['label'] for blk in client_test_blks[cid]]
    all_labels = train_labels + val_labels + test_labels

    total_logs = sum(valid_sessions[blk]['log_count'] for blk in blks)

    client_stats[cid] = {
        'num_blocks': n_total,
        'total_logs': total_logs,
        'train_blocks': n_train,
        'val_blocks': n_val,
        'test_blocks': n_test,
        'train_anomaly': train_anomaly,
        'val_anomaly': val_anomaly,
        'test_anomaly': test_anomaly,
        'normal_logs': sum(1 for blk in blks if valid_sessions[blk]['label'] == 0),
        'anomaly_logs': sum(1 for blk in blks if valid_sessions[blk]['label'] == 1),
        'anomaly_ratio': sum(all_labels) / len(all_labels) if len(all_labels) > 0 else 0
    }


print("\n[STAT] Per-client statistics:")
print("-" * 120)


# ----------------------------

# ----------------------------
def blocks_to_df(blk_list):
    seqs, labels, log_counts, blk_ids = [], [], [], []
    for blk in blk_list:
        if blk in valid_sessions:
            seq_str = ' '.join(map(str, valid_sessions[blk]['event_ids']))
            seqs.append(seq_str)
            labels.append(valid_sessions[blk]['label'])
            log_counts.append(valid_sessions[blk]['log_count'])
            blk_ids.append(blk)
    return pd.DataFrame({'sequence': seqs, 'label': labels, 'log_count': log_counts, 'blk_id': blk_ids})

# ----------------------------

# ----------------------------
print("\n[ABC] Building vocabulary and training Word2Vec...")

train_sequences = []
for cid in selected_client_ids:
    for blk in client_train_blks[cid]:
        if blk in valid_sessions:
            train_sequences.append([str(e) for e in valid_sessions[blk]['event_ids']])

all_events = set(e for seq in train_sequences for e in map(int, seq))
vocab = {'<PAD>': 0}
for idx, event in enumerate(sorted(all_events), 1):
    vocab[event] = idx

with open(os.path.join(OUTPUT_DIR, 'vocab_10client_5-2-3.json'), 'w') as f:
    json.dump(vocab, f, indent=2)

w2v_model = Word2Vec(
    sentences=train_sequences,
    vector_size=W2V_DIM,
    window=5,
    min_count=1,
    workers=4,
    seed=RANDOM_SEED,
    sg=1
)

vocab_events = sorted([eid for eid in vocab.keys() if eid != '<PAD>'], key=int)
final_vectors = np.array([w2v_model.wv[str(e)] for e in vocab_events], dtype=np.float32)
np.save(os.path.join(OUTPUT_DIR, 'event_vectors_w2v300_10client_5-2-3.npy'), final_vectors)

print(f"  Vocabulary size: {len(vocab)}")
print(f"  Word2Vec vector shape: {final_vectors.shape}")

# ----------------------------

# ----------------------------
print("\n[SAVE] Saving semantic event vectors to JSON...")

semantic_vectors = {}
for event_id in vocab_events:

    semantic_vectors[str(event_id)] = w2v_model.wv[str(event_id)].tolist()


json_output_path = os.path.join(OUTPUT_DIR, 'bgl_semantic_vec.json')
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(semantic_vectors, f, indent=2)

print(f"  -> {json_output_path}")
print(f"  Number of events: {len(semantic_vectors)}")
print(f"  Vector dimension: {len(list(semantic_vectors.values())[0])}")

# ----------------------------

# ----------------------------
vocab_events = sorted([eid for eid in vocab.keys() if eid != '<PAD>'], key=int)
event_vectors_300d = np.array([w2v_model.wv[str(e)] for e in vocab_events], dtype=np.float32)

print(f"\n[ABC] Event vector shape: {event_vectors_300d.shape}")
print(f"[OK] 300-dimensional Word2Vec vectors are ready!")

# ----------------------------

# ----------------------------
print("\n[SAVE] Saving CSV files (compact format: sequence + label)...")

def blocks_to_minimal_df(blk_list):
    """Return a DataFrame containing sequence and label columns."""
    sequences = []
    labels = []

    for blk in blk_list:
        if blk in valid_sessions:
            events = valid_sessions[blk]['event_ids']

            seq_str = ' '.join(map(str, events))
            sequences.append(seq_str)
            labels.append(valid_sessions[blk]['label'])

    return pd.DataFrame({'sequence': sequences, 'label': labels})


minimal_train_df = blocks_to_minimal_df(
    sum([client_train_blks[cid] for cid in selected_client_ids], [])
)
minimal_train_df.to_csv(os.path.join(OUTPUT_DIR, 'log_train.csv'), index=False)
print(f"  -> log_train.csv ({len(minimal_train_df)} rows)")


minimal_val_df = blocks_to_minimal_df(
    sum([client_val_blks[cid] for cid in selected_client_ids], [])
)
minimal_val_df.to_csv(os.path.join(OUTPUT_DIR, 'log_val.csv'), index=False)
print(f"  -> log_val.csv ({len(minimal_val_df)} rows)")


minimal_test_df = blocks_to_minimal_df(
    sum([client_test_blks[cid] for cid in selected_client_ids], [])
)
minimal_test_df.to_csv(os.path.join(OUTPUT_DIR, 'log_test.csv'), index=False)
print(f"  -> log_test.csv ({len(minimal_test_df)} rows)")


CLIENT_DATA_DIR = os.path.join(OUTPUT_DIR, 'client_data')
os.makedirs(CLIENT_DATA_DIR, exist_ok=True)


for cid in selected_client_ids:

    client_train_df = blocks_to_minimal_df(client_train_blks[cid])
    client_train_df.to_csv(os.path.join(OUTPUT_DIR, f'client{cid}_train.csv'), index=False)
    client_train_df.to_csv(os.path.join(CLIENT_DATA_DIR, f'client{cid}_train.csv'), index=False)


    client_val_df = blocks_to_minimal_df(client_val_blks[cid])
    client_val_df.to_csv(os.path.join(OUTPUT_DIR, f'client{cid}_val.csv'), index=False)
    client_val_df.to_csv(os.path.join(CLIENT_DATA_DIR, f'client{cid}_val.csv'), index=False)


    client_test_df = blocks_to_minimal_df(client_test_blks[cid])
    client_test_df.to_csv(os.path.join(OUTPUT_DIR, f'client{cid}_test.csv'), index=False)
    client_test_df.to_csv(os.path.join(CLIENT_DATA_DIR, f'client{cid}_test.csv'), index=False)

    print(f"  -> client{cid}_train.csv ({len(client_train_df)} rows), client{cid}_val.csv ({len(client_val_df)} rows), client{cid}_test.csv ({len(client_test_df)} rows)")

# ----------------------------

# ----------------------------
print("\n[CALC] Building unified federated .npz file...")


event_embed_path = os.path.join(OUTPUT_DIR, 'event_vectors_w2v300_10client_5-2-3.npy')
event_embeddings = np.load(event_embed_path)
pad_vector = np.zeros((1, event_embeddings.shape[1]), dtype=np.float32)
event_embeddings_with_pad = np.vstack([pad_vector, event_embeddings])

print(f"Event embedding shape: {event_embeddings_with_pad.shape}")

def seq_to_tensor(seq_str, max_len=MAX_SEQ_LEN):
    """Convert a space-separated event sequence to a padded (max_len, embed_dim) array."""










    if pd.isna(seq_str) or seq_str == '':
        events = []
    else:
        events = list(map(int, str(seq_str).split()))


    if len(events) > max_len:
        events = events[:max_len]
    else:
        events = events + [0] * (max_len - len(events))


    indices = [vocab.get(e, 0) for e in events]


    tensor = event_embeddings_with_pad[indices]  # shape: (max_len, 300)

    return tensor


all_train_features = []
all_train_labels = []
all_val_features = []
all_val_labels = []
all_test_features = []
all_test_labels = []


dict_users_train = {}
dict_users_val = {}
dict_users_test = {}
current_train_idx = 0
current_val_idx = 0
current_test_idx = 0


for cid in selected_client_ids:
    print(f"  Client {cid}: Processing data...")


    train_df_cid = blocks_to_df(client_train_blks[cid])


    if len(train_df_cid) > 0:
        X_train_list = []
        for seq in train_df_cid['sequence']:
            tensor = seq_to_tensor(seq, max_len=MAX_SEQ_LEN)
            X_train_list.append(tensor)

        X_train = np.stack(X_train_list)  # (n_samples, 300, {MAX_SEQ_LEN})
        y_train = train_df_cid['label'].values.astype(np.int64)

        all_train_features.append(X_train)
        all_train_labels.append(y_train)


        dict_users_train[cid] = np.arange(current_train_idx, current_train_idx + len(X_train))
        current_train_idx += len(X_train)

        print(f"    Training set: {X_train.shape}, Anomaly ratio: {y_train.mean():.2%}")
    else:
        print(f"    WARN: Client {cid} has no training data")
        dict_users_train[cid] = np.array([])


    val_df_cid = blocks_to_df(client_val_blks[cid])

    if len(val_df_cid) > 0:
        X_val_list = []
        for seq in val_df_cid['sequence']:
            tensor = seq_to_tensor(seq, max_len=MAX_SEQ_LEN)
            X_val_list.append(tensor)

        X_val = np.stack(X_val_list)  # (n_samples, 300, {MAX_SEQ_LEN})
        y_val = val_df_cid['label'].values.astype(np.int64)

        all_val_features.append(X_val)
        all_val_labels.append(y_val)


        dict_users_val[cid] = np.arange(current_val_idx, current_val_idx + len(X_val))
        current_val_idx += len(X_val)

        print(f"    Validation set: {X_val.shape}, Anomaly ratio: {y_val.mean():.2%}")
    else:
        print(f"    WARN: Client {cid} has no validation data")
        dict_users_val[cid] = np.array([])


    test_df_cid = blocks_to_df(client_test_blks[cid])

    if len(test_df_cid) > 0:
        X_test_list = []
        for seq in test_df_cid['sequence']:
            tensor = seq_to_tensor(seq, max_len=MAX_SEQ_LEN)
            X_test_list.append(tensor)

        X_test = np.stack(X_test_list)  # (n_samples, 300, {MAX_SEQ_LEN})
        y_test = test_df_cid['label'].values.astype(np.int64)

        all_test_features.append(X_test)
        all_test_labels.append(y_test)


        dict_users_test[cid] = np.arange(current_test_idx, current_test_idx + len(X_test))
        current_test_idx += len(X_test)

        print(f"    Test set: {X_test.shape}, Anomaly ratio: {y_test.mean():.2%}")
    else:
        print(f"    WARN: Client {cid} has no test data")
        dict_users_test[cid] = np.array([])


if len(all_train_features) > 0:
    X_train_all = np.vstack(all_train_features)
    y_train_all = np.concatenate(all_train_labels)
else:
    X_train_all = np.array([])
    y_train_all = np.array([])

if len(all_val_features) > 0:
    X_val_all = np.vstack(all_val_features)
    y_val_all = np.concatenate(all_val_labels)
else:
    X_val_all = np.array([])
    y_val_all = np.array([])

if len(all_test_features) > 0:
    X_test_all = np.vstack(all_test_features)
    y_test_all = np.concatenate(all_test_labels)
else:
    X_test_all = np.array([])
    y_test_all = np.array([])

print(f"\n  Total training samples: {len(X_train_all)}, Data shape: {X_train_all.shape if len(X_train_all) > 0 else 'empty'}")
print(f"  Total validation samples: {len(X_val_all)}, Data shape: {X_val_all.shape if len(X_val_all) > 0 else 'empty'}")
print(f"  Total test samples: {len(X_test_all)}, Data shape: {X_test_all.shape if len(X_test_all) > 0 else 'empty'}")


np.savez(os.path.join(OUTPUT_DIR, 'bgl_fed_unified_10client_5-2-3.npz'),
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

print(f"\n[OK] Unified federated .npz saved:")
print(f"   - {OUTPUT_DIR}/bgl_fed_unified_10client_5-2-3.npz")
print(f"   - Sequence length: {MAX_SEQ_LEN}, Feature dimension: {FEATURE_DIM}")
print(f"   - File size: {os.path.getsize(os.path.join(OUTPUT_DIR, 'bgl_fed_unified_10client_5-2-3.npz')) / 1024 / 1024:.2f} MB")

# ----------------------------

# ----------------------------
print("\n" + "=" * 120)
print("[STAT] Final client statistics")
print("=" * 120)
print(f"{'Client':<10} {'Num_Blocks':<12} {'Total_Logs':<14} {'Normal':<10} {'Anomaly':<10} {'Anomaly_Ratio':<15} {'Train_Anom':<12} {'Val_Anom':<10} {'Test_Anom':<10}")
print("=" * 120)
for cid in selected_client_ids:
    stats = client_stats[cid]
    print(f"Client {cid:<5} {stats['num_blocks']:<12} {stats['total_logs']:<14} "
          f"{stats['normal_logs']:<10} {stats['anomaly_logs']:<10} {stats['anomaly_ratio']:.2%}      "
          f"{stats['train_anomaly']:<12} {stats['val_anomaly']:<10} {stats['test_anomaly']:<10}")
print("=" * 120)

# ----------------------------

# ----------------------------
print(f"\n[FOLDER] Output directory: {OUTPUT_DIR}")
print("   - vocab_10client_5-2-3.json")
print("   - event_vectors_w2v300_10client_5-2-3.npy (Word2Vec 300d)")
print("   - bgl_semantic_vec.json")
print("   - log_train.csv, log_val.csv, log_test.csv")
print("   - client{cid}_train.csv, client{cid}_val.csv, client{cid}_test.csv (10 clients, each with 3 files)")
print("   - bgl_fed_unified_10client_5-2-3.npz (Word2Vec 300-dimensional vectors for federated learning)")
print(f"\n[FOLDER] Client data directory: {CLIENT_DATA_DIR}")
print("   - client{cid}_train.csv, client{cid}_val.csv, client{cid}_test.csv (10 clients, each with 3 files)")

print("\n[CELEBRATE] Preprocessing complete: 10 clients, 5:2:3 splits!")
