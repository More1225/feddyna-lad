# Current preparation and execution workflow

Start with a compatible dense NPZ file to run **the current public reference implementation**. The optional preparation recipes below support that interface. See [provenance limitations](reproducibility_notes.md). Run training commands from the repository root and supply the input path explicitly.

## Dataset producers and consumers

| Dataset | Public producer | Required input | Generated output | Consumer |
| --- | --- | --- | --- | --- |
| HDFS | data/HDFS/prepare_fed_hdfs_v7.py | Event_traces.csv with BlockId/Features columns; anomaly_label.csv with block ID and Normal/Anomaly label | fed_data/hdfs_fed_unified_v7.npz, compact IDs NPZ, client_data_v7 CSV triplets, vocabulary/embeddings and metadata | Dense NPZ: data/train_utils.py; CSVs: MIX generator |
| BGL | data/BGL/prepare_fed_bgl-10client-5-2-3.py | BGL.log | fed_data/bgl_fed_unified_10client_5-2-3.npz, client CSVs, vocabulary and event vectors | Dense NPZ loader or MIX generator |
| Thunderbird | data/TB/prepare_fed_thunderbird_10m_logs_shuffle.py | Thunderbird.log | fed_data/thunderbird_fed_unified_10client_5-2-3_shuffle.npz, client*_shuffle.csv and embeddings | Dense NPZ loader or MIX generator |
| Mixed systems | data/MIX/mix_federated_generator.py | Selected client train/val/test CSV triplets in one directory | mixed_fed_unified.npz, vocabulary, embeddings, client statistics and usage text | FedDyna.py via data/train_utils.py |

These are current reference producers for the [documented dense NPZ interface](../README.md#expected-data-interface). Historical HDFS producer identity and Thunderbird raw-to-final lineage are `DOCUMENTED_LIMITATION`, not release blockers by themselves. Current reference scripts retain their existing behavior, including Thunderbird's first-10-million-entry limit. No byte-identical historical regeneration is claimed.

### HDFS

Place the two required CSVs under data/HDFS. Features contains a serialized event sequence such as `[E12, E5]`; block IDs must match the label file. Labels are read as two columns, with recognized values Normal and Anomaly. Preparing block/event traces from raw HDFS logs is an upstream input requirement; an exact historical raw-log-to-trace converter is not supplied.

```console
python data/HDFS/prepare_fed_hdfs_v7.py --help
python data/HDFS/prepare_fed_hdfs_v7.py
```

Relative HDFS input/output arguments resolve against the script directory. Use the dense `hdfs_fed_unified_v7.npz` for training; the compact `_ids.npz` is not a drop-in replacement for the current dense loader. The legacy training default still names a v6 input, so pass the v7 path explicitly.

### BGL and Thunderbird

These two scripts run preprocessing at module top level. They have no safe help-only CLI; inspect their configuration constants in source. Place each raw log beside its preparation script and run from that directory:

```console
cd data/BGL
python prepare_fed_bgl-10client-5-2-3.py
cd ../..
cd data/TB
python prepare_fed_thunderbird_10m_logs_shuffle.py
cd ../..
```

The local Drain.py beside each script is required. Both write into a relative fed_data directory. Run in a fresh workspace/output context when preserving prior experiment artifacts. Dataset preparation can be memory-intensive and was not executed during release preparation.

### Assemble a mixed dataset

Create `data/MIX/current` and copy your chosen clients' complete train/val/test CSV triplets into it. Choose and record client identities explicitly; this release does not silently select historical clients. HDFS filenames are `client_<id>_<split>.csv`; BGL uses `client<id>_<split>.csv`; Thunderbird shuffle uses `client<id>_<split>_shuffle.csv`. HDFS event_ids/label and BGL/TB sequence/label columns are accepted by the loader in the generator.

```console
python data/MIX/mix_federated_generator.py --input_dir data/MIX/current --output_dir data/MIX/current --output_name mixed_fed_unified
```

The generator discovers all matching clients in the directory, preserves supplied splits, and fits shared embeddings using their training sequences. Avoid mixing unrelated clients or competing same-type IDs in this folder. It does not merge already embedded standalone NPZ arrays. All generated artifacts stay local.

Historical scenario source combinations are HDFS+TB, HDFS+BGL and HDFS+BGL+TB. Identified final Scenario 3 inputs used 5/2/3 clients; do not confuse this with the general launcher's unchanged 4/3/3 path. Current experiments may use explicitly chosen cohorts, but must record them accurately.

## Run the current implementation

Use the commands in the [README](../README.md#running-feddyna). FedDyna is a configuration of the shared root entry, not a separate fl_method. Its options are fedavg + entropy + kdars + enable_ha_scope + enable_weight_protection. Configuration comes from config/options.py and explicit CLI overrides. --run_dir selects an exact output directory; without it, the configured save/output mechanism chooses the run location.

Generated outputs include effective_config.json, completion.json and results/run_metrics.json / client_metrics.csv. Batch runs additionally record launch.json and log.txt. Preserve these together with an environment capture and source/input hashes for future provenance.

## Current multi-seed runs

Create a local `current_run.json` with this structure, changing only the input path and scenario label to describe your prepared cohort:

```json
{
  "seeds": [2024, 2025, 2026, 2027, 2028],
  "scenarios": {
    "current_mix": {
      "args": ["--dataset", "mix", "--data_path", "data/MIX/current/mixed_fed_unified.npz"]
    }
  }
}
```

Omitting methods uses the launcher's built-in method definitions. Select one explicitly:

```console
python scripts/run_5seed_experiments.py --config current_run.json --methods FedDyna --dry-run
python scripts/run_5seed_experiments.py --config current_run.json --methods FedDyna --results-root results/current_multiseed --summarize
```

Outputs are organized as results-root/dataset-id/scenario/method/seed_N. The derived dataset-id is computed from the input path. Use the same JSON with --methods FedAvg,FedProx to run those adapters. The default seeds are 2024–2028; defaults are current settings, not recovered historical source/environment evidence.

For ablations, copy scripts/ablation_5seed_config.json to a local configuration and add an explicit scenarios object as above. Run with --config pointing to that local file. Compatibility variant A2_w_o_HA means without the aggregation-scope selector; A1_w_o_EW uses sample-count weights. Do not relabel it uniform weighting.

## Auxiliary tools

- scripts/run_rounds.py previews commands unless --execute is supplied. It expects its named prepared mixed inputs for S1/S2/S3 and runs 6/10/20/30-round budgets. Its FedDyna entropy view is explicitly shared_mask0.
- scripts/run_sensitivity.py requires --scenario, --dataset, --data-path and --experiment. It retains its configured grids; inspect help before selecting a current experiment.
- scripts/summarize_5seed_results.py reads completed run outputs and produces seed-level and mean/std summaries; use --results-root and --output-dir for your new cohort.
- tools/plot_ha_analysis.py plots generated selector diagnostics; it does not supply or recreate historical paper figures.

See [baseline behavior](baseline_reproduction_status.md) for method-specific optimization/stopping and [dependency notes](dependency_audit.md) for environment limitations.
