# FedDyna

**Adaptive Personalized Federated Learning for Cross-System Log Anomaly Detection**

FedDyna coordinates federated knowledge sharing with client-specific adaptive personalization. Clients can originate from different log systems with different log patterns and anomaly characteristics. This repository provides a minimal runnable reference implementation with LightTCN and supported baseline adapters; datasets and historical experiment outputs are not bundled.

## Core mechanisms

- **Knowledge Drift-Aware Adaptive Personalization (KDARS)** uses Knowledge Drift and MeanRate signals for adaptive parameter-level personalization.
- **Entropy-based adaptive client weighting** combines `universality`, `log_sample_count`, `update_alignment` and `update_stability`.
- The **aggregation-scope selector** makes a one-time selection between `all` and `masked_only` from cross-client update-direction disagreement at the first scheduled mask-update round. The selected scope remains fixed thereafter.

Mask semantics are `0 = shared` and `1 = personalized`. The detector is LightTCN; its CLI model name is `lightlog`.

## Repository structure

| Path | Purpose |
| --- | --- |
| `FedDyna.py`, `config/` | Main entry point and CLI configuration |
| `core/`, `federated/` | Shared computations, federated orchestration, aggregation and scope selection |
| `personalization/` | KDARS, personalization control and weighting factors |
| `models/` | LightTCN and the FedPAC model variant |
| `methods/`, `trainers/` | Baseline adapters and client training |
| `data/` | Dense NPZ loading and preprocessing references |
| `utils/` | Evaluation output and experiment serialization |

Key sources: [KDARS](personalization/kdars.py), [entropy weighting](federated/weighting.py), [scope selector](federated/heterogeneity_aware.py), [round orchestration and selector locking](federated/round_runner.py), and [LightTCN](models/lightlog.py).

## Installation

```console
git clone https://github.com/More1225/feddyna-lad.git
cd feddyna-lad
```

Create and activate a virtual environment with Python 3.10 or newer, then install:

```console
python -m pip install -r requirements.txt
```

Choose a PyTorch build appropriate to your CPU/CUDA environment. Python 3.10 is the source syntax floor, not a tested compatibility matrix. Dependencies are unpinned; validate and record the environment used for your experiments.

## Data preparation / Input format

Obtain raw data separately under its applicable terms, or supply a compatible dense NPZ. `FedDyna.py` loads the NPZ through `data/train_utils.py::get_data()`:

| NPZ key | Expected value |
| --- | --- |
| `train_features`, `val_features`, `test_features` | Dense numeric arrays `(N_split, sequence_length, 300)` for the default model, padded to a common sequence length within each split |
| `train_labels`, `val_labels`, `test_labels` | Integer binary arrays `(N_split,)`: `0` normal, `1` anomaly |
| `dict_users_train`, `dict_users_val`, `dict_users_test` | Scalar object arrays containing dictionaries from client IDs to split row indices; matching client IDs across splits |
| `train_lengths`, `val_lengths`, `test_lengths` | Optional integer arrays `(N_split,)` containing valid unpadded lengths |
| `client_ids` | Optional client IDs; otherwise inferred from the training partition |

Features are converted to float32 and labels to int64. Feature width must match `--input_dim` (default `300`). Supply separate train/validation/test splits: the loader falls back to test data when validation arrays or partitions are absent. Partition dictionaries require `allow_pickle=True`, so load only trusted NPZ files. Token-ID-only NPZ files are not supported by this dense interface.

The retained preprocessing references produce dense NPZ files and client CSV splits:

| Dataset | Reference | Required input and execution context |
| --- | --- | --- |
| HDFS | [prepare_fed_hdfs_v7.py](data/HDFS/prepare_fed_hdfs_v7.py) | `Event_traces.csv` with `BlockId`/`Features` columns and `anomaly_label.csv`; relative paths resolve against the script directory |
| BGL | [prepare_fed_bgl-10client-5-2-3.py](data/BGL/prepare_fed_bgl-10client-5-2-3.py) | `BGL.log` beside the script; run from `data/BGL/` |
| Thunderbird | [prepare_fed_thunderbird_10m_logs_shuffle.py](data/TB/prepare_fed_thunderbird_10m_logs_shuffle.py) | `Thunderbird.log` beside the script; run from `data/TB/`; processes the first 10 million entries |

HDFS requires prepared block/event traces and matching block labels; a raw-log-to-trace converter is not supplied. Use its dense v7 output rather than the compact ID output. BGL and Thunderbird execute preprocessing at module top level: inspect configuration constants before running; they do not offer a safe help-only CLI. Keep each bundled `Drain.py` beside its script. Preparation can be memory-intensive.

For cross-system inputs, the [MIX preparation reference](data/MIX/README.md) combines explicitly selected client train/val/test CSV triplets and fits shared embeddings from training sequences. It does not merge independently embedded NPZ arrays. These references do not guarantee exact regeneration of historical paper inputs.

## Quick Start

Run from the repository root after preparing your NPZ. Replace the example data path with your input:

```console
python FedDyna.py --dataset mix --data_path data/MIX/current/mixed_fed_unified.npz --model lightlog --fl_method fedavg --weighting_method entropy --aggregation_scope all --enable_ha_scope --pruning_method kdars --enable_weight_protection --seed 2024 --run_dir results/current_feddyna/seed_2024
```

`--fl_method fedavg` is intentional: FedDyna uses the common FedAvg-compatible orchestration with entropy weighting, KDARS, the aggregation-scope selector and weight protection. This configuration runs FedDyna rather than plain FedAvg.

Use `--device cpu` when appropriate and `python FedDyna.py --help` to inspect options. Defaults include six rounds, five local epochs, batch size 32, AdamW, learning rate `5e-4` and weight decay `5e-4`. Always pass `--data_path`; the legacy default names an unbundled HDFS v6 input.

### Supported baselines

Example: run FedProx on the same input, without FedDyna-only enable flags:

```console
python FedDyna.py --dataset mix --data_path data/MIX/current/mixed_fed_unified.npz --fl_method fedprox --weighting_method fedavg --aggregation_scope all --pruning_method none --seed 2024 --run_dir results/current_fedprox/seed_2024
```

The 11 supported comparison methods and `--fl_method` values are Local-only (`local`), FedAvg (`fedavg`), FedProx (`fedprox`), Ditto (`ditto`), FedRep (`fedrep`), FedALA (`fedala`), FedPAC (`fedpac`), FedAS (`fedas`), FedAPA (`fedapa`), FedLTN (`fedltn`) and FedSelect (`fedselect`). Replace `fedprox` in the baseline command to select one.

These are current adapters, not certifications of exact original-paper implementations. FedPAC lacks classifier collaboration; this FedLTN baseline configuration disables pruning. FedSelect manages its own masks despite `--pruning_method none`. Stopping and model selection differ: the shared trainer restores the best validation-loss state within each fit, Local-only selects by validation AUROC, and fixed-epoch/alternating or personalized branches use method-specific policies. A top-level optimizer setting does not apply uniformly to every branch.

## Outputs

The FedDyna command writes its configuration and, on completion, metrics and a completion record under:

```text
results/current_feddyna/seed_2024/
├── effective_config.json
├── completion.json
└── results/
    ├── run_metrics.json
    └── client_metrics.csv
```

Reported metrics include F1, precision, recall and AUPRC (sklearn average precision from class-1 scores). Run-level metrics are equal-client means.

## Reproducibility note

This is a reference implementation for current runs on compatible inputs. Raw datasets and historical experiment outputs are not included; preprocessing provenance is partial, and exact historical numerical reproduction is not guaranteed. The current weight-decay default is `5e-4`, while the manuscript reports `2e-4`; historical KDARS integer-rounding provenance is incomplete. Record the source revision, command, environment, input hashes, client splits and seed with each run.

## Citation

The associated FedDyna paper has not yet been published. Citation metadata will be added when the final public bibliographic record is available.

## License

[MIT License](LICENSE), Copyright (c) 2026 Yingwu Fu. Existing third-party notices, including those in the bundled Drain parsers, continue to apply.
