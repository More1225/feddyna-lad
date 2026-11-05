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
| `data/` | Runtime unified-NPZ input loading |
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

## NPZ input contract

FedDyna operates on preprocessed federated data supplied as a unified NPZ file. Dataset-specific preprocessing and client-construction pipelines are not included in this repository.

`--data_path PATH` is required. Specify one unified NPZ containing all client partitions, rather than one file per client. No particular directory or filename is required. The runtime loader is `data/train_utils.py::get_data()`.

| NPZ key | Requirement and value |
| --- | --- |
| `train_features`, `test_features` | Required dense numeric arrays `(N_split, sequence_length, input_dim)`; default `input_dim=300`, with a common padded sequence length within each split |
| `train_labels`, `test_labels` | Required arrays `(N_split,)`; binary labels `0` normal and `1` anomaly |
| `val_features`, `val_labels` | Recommended separate validation arrays, with the same shape conventions as training/test arrays |
| `dict_users_train`, `dict_users_test` | Required scalar object arrays containing dictionaries from client IDs to row indices in the corresponding split |
| `dict_users_val` | Recommended validation partition dictionary, using the same convention |
| `train_lengths`, `val_lengths`, `test_lengths` | Optional integer arrays `(N_split,)` containing valid unpadded sequence lengths |
| `client_ids` | Optional client ID array; otherwise IDs are inferred from the training partition |

Features are converted to float32, labels to int64 and supplied lengths to int64. Feature width must match `--input_dim`. Use matching client IDs across partitions and valid indices for each split. Token-ID-only NPZ files are not supported. Partition dictionaries require `allow_pickle=True`, so load only trusted NPZ files.

Validation fallbacks are independent: missing `val_features` uses test arrays; missing `dict_users_val` uses the test partition. Provide `val_features`, `val_labels` and `dict_users_val` together to use a separate validation split and avoid mixing arrays with an unrelated partition.

## Quick Start

Run from the repository root after preparing your NPZ. Replace the example data path with your input:

```console
python FedDyna.py --dataset mix --data_path inputs/federated.npz --model lightlog --fl_method fedavg --weighting_method entropy --aggregation_scope all --enable_ha_scope --pruning_method kdars --enable_weight_protection --seed 2024 --run_dir results/current_feddyna/seed_2024
```

`--fl_method fedavg` is intentional: FedDyna uses the common FedAvg-compatible orchestration with entropy weighting, KDARS, the aggregation-scope selector and weight protection. This configuration runs FedDyna rather than plain FedAvg.

Use `--device cpu` when appropriate and `python FedDyna.py --help` to inspect options. Defaults include six rounds, five local epochs, batch size 32, AdamW, learning rate `5e-4` and weight decay `5e-4`. Always pass `--data_path`; omitting it produces an argparse error.

### Supported baselines

Example: run FedProx on the same input, without FedDyna-only enable flags:

```console
python FedDyna.py --dataset mix --data_path inputs/federated.npz --fl_method fedprox --weighting_method fedavg --aggregation_scope all --pruning_method none --seed 2024 --run_dir results/current_fedprox/seed_2024
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

This is a reference implementation for current runs on compatible inputs. Raw datasets, dataset-specific preparation pipelines and historical experiment outputs are not included; exact historical numerical reproduction is not guaranteed. The current weight-decay default is `5e-4`, while the manuscript reports `2e-4`; historical KDARS integer-rounding provenance is incomplete. Record the source revision, command, environment, input hashes, client splits and seed with each run.

## Citation

The associated FedDyna paper has not yet been published. Citation metadata will be added when the final public bibliographic record is available.

## License

[MIT License](LICENSE), Copyright (c) 2026 Yingwu Fu.
