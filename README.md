# FedDyna

**Adaptive Personalized Federated Learning for Log Anomaly Detection Across Systems**

**FedDyna** targets **cross-system log anomaly detection** under federated constraints. Clients can originate from different log systems with different log patterns and anomaly characteristics. FedDyna coordinates federated knowledge sharing with client-specific adaptive personalization.

This repository provides the reference implementation, federated training and evaluation pipeline, LightTCN model, and supported baseline adapters. Reference preprocessing scripts help prepare compatible inputs; datasets are not bundled. See [reproducibility notes](docs/reproducibility_notes.md) for limits on comparisons with historical paper results.

## FedDyna implementation

FedDyna uses a LightTCN detector and combines:

- **Knowledge Drift-Aware Adaptive Personalization (KDARS)** uses Knowledge Drift and MeanRate signals to guide adaptive, parameter-level personalization.
- **Entropy-based adaptive client weighting** integrates multiple client-side/server-observable factors: universality, log sample count, update alignment and update stability.
- An **auxiliary one-time aggregation-scope selector** based on cross-client update-direction disagreement, activated at the first scheduled mask-update round. The selected scope then remains fixed.

Mask semantics are `0 = shared` and `1 = personalized`. Runtime aggregation scopes are `all` and `masked_only`. The CLI model name is `lightlog`.

## Repository structure

| Path | Purpose |
| --- | --- |
| `FedDyna.py`, `config/` | Entry point and CLI configuration |
| `models/` | LightTCN models; CLI model name `lightlog` |
| `federated/`, `personalization/`, `core/` | Federated rounds, aggregation, masks and KDARS |
| `methods/`, `trainers/` | Baseline adapters and training utilities |
| `data/` | Input loader and selected current preparation scripts |
| `scripts/` | Multi-seed, ablation, round-budget and sensitivity launchers; result statistics |
| `utils/` | Experiment output management and result serialization |
| `tools/` | Plotting of generated selector diagnostics |
| `docs/` | Current workflow, baseline behavior, dependencies and release scope |

See the [execution map](docs/execution_flow.md) for the call flow and the [repository manifest](docs/public_release_allowlist.md) for the maintained source boundary. Manuscripts, datasets, checkpoints and experiment archives are excluded.

| Component | Source |
| --- | --- |
| KDARS | [personalization/kdars.py](personalization/kdars.py) |
| Entropy weighting | [federated/weighting.py](federated/weighting.py) → [controller](personalization/controller.py) → [factor matrix](personalization/matrix_builders.py) → [entropy weights](core/math_utils.py) |
| Aggregation-scope selector | [federated/heterogeneity_aware.py](federated/heterogeneity_aware.py); [round runner](federated/round_runner.py) controls activation at the first scheduled mask-update round and fixes the selected scope afterward |
| LightTCN | [models/lightlog.py](models/lightlog.py) |
| Baselines | [Supported adapters and implementation boundaries](docs/baseline_reproduction_status.md) |

## Installation

Start by cloning the repository:

```console
git clone https://github.com/More1225/feddyna-lad.git
cd feddyna-lad
```

The basic workflow is: install dependencies → [prepare a dense NPZ](#dataset-preparation) → [run FedDyna](#running-feddyna) → [inspect outputs](#outputs).

Use Python 3.10 or newer (required by the source syntax), with packages available for your Python/platform combination. This is a syntax floor, not a tested compatibility matrix. Create and activate a virtual environment, then run:

```console
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate to your CPU/CUDA environment. Versions are deliberately unpinned because a validated historical environment is unavailable; no historical version equivalence is claimed. See [dependency details](docs/dependency_audit.md).

## Expected data interface

`FedDyna.py` loads one dense NPZ file through `data/train_utils.py::get_data()`. It contains split-level arrays and dictionaries assigning rows to clients:

| NPZ key | Expected value |
| --- | --- |
| `train_features`, `val_features`, `test_features` | Dense numeric arrays shaped `(N_split, sequence_length, 300)` for the default model; padded to a common sequence length within each split |
| `train_labels`, `val_labels`, `test_labels` | Arrays shaped `(N_split,)`; integer binary labels `0` (normal) and `1` (anomaly) |
| `dict_users_train`, `dict_users_val`, `dict_users_test` | Scalar object arrays containing dictionaries: client ID to row indices in the corresponding split; use matching client IDs across splits |
| `train_lengths`, `val_lengths`, `test_lengths` | Optional integer arrays shaped `(N_split,)` with valid unpadded sequence lengths |
| `client_ids` | Optional client ID array; when omitted, IDs are inferred from the training partition |

Features are converted to float32 and labels to int64. The feature dimension must match `--input_dim` (default `300`). Provide separate validation and test splits: the legacy loader falls back to test data when validation arrays or partitions are missing. The loader uses `allow_pickle=True` for partition dictionaries, so load only trusted NPZ files. Token-ID-only NPZ files are not accepted by this dense interface.

## Dataset preparation

Obtain the relevant dataset separately under its applicable terms. The retained scripts are current reference workflows:

| Dataset | Reference script | Expected input |
| --- | --- | --- |
| HDFS (Hadoop distributed file system logs) | `data/HDFS/prepare_fed_hdfs_v7.py` | Prepared block-event traces in `Event_traces.csv` and block labels in `anomaly_label.csv` |
| BGL (Blue Gene/L system logs) | `data/BGL/prepare_fed_bgl-10client-5-2-3.py` | `BGL.log` |
| Thunderbird system logs | `data/TB/prepare_fed_thunderbird_10m_logs_shuffle.py` | `Thunderbird.log` |

Each writes a dense federated NPZ and client CSV splits. `data/MIX/mix_federated_generator.py` combines selected client CSV splits into compatible mixed-system inputs. See [preparation commands and current runs](docs/canonical_reproduction_path.md).

Historical paper inputs were prepared during the original experiment workflow; complete byte-identical preprocessing provenance is unavailable for every dataset. These scripts support current experiments on compatible inputs, but do not claim exact regeneration of historical artifacts.

## Running FedDyna

Run from the repository root after preparing the input. Replace `data/MIX/current/mixed_fed_unified.npz` with your actual dense NPZ path. The example starts a new current experiment; it is not a historical-result recovery command.

```console
python FedDyna.py --dataset mix --data_path data/MIX/current/mixed_fed_unified.npz --model lightlog --fl_method fedavg --weighting_method entropy --aggregation_scope all --enable_ha_scope --pruning_method kdars --enable_weight_protection --seed 2024 --run_dir results/current_feddyna/seed_2024
```

`--fl_method fedavg` is intentional: FedDyna uses the common FedAvg-compatible orchestration path together with entropy-based adaptive client weighting, KDARS, the aggregation-scope selector and weight protection. This configuration runs FedDyna, rather than plain FedAvg.

Use `--device cpu` when appropriate. Inspect `python FedDyna.py --help` for options. Default training settings include six rounds, five local epochs, batch size 32, AdamW, learning rate `5e-4` and weight decay `5e-4`. Always set `--data_path`: the unchanged legacy default references an unbundled HDFS v6 artifact. Effective configuration and generated metrics are saved under the selected run directory.

### Outputs

The command above writes its configuration and, on completion, metrics and completion record under:

```text
results/current_feddyna/seed_2024/
├── effective_config.json
├── completion.json
└── results/
    ├── run_metrics.json
    └── client_metrics.csv
```

## Running supported baseline adapters

For example, run the current FedProx adapter on the same prepared input:

```console
python FedDyna.py --dataset mix --data_path data/MIX/current/mixed_fed_unified.npz --fl_method fedprox --weighting_method fedavg --aggregation_scope all --pruning_method none --seed 2024 --run_dir results/current_fedprox/seed_2024
```

Replace `fedprox` with `local`, `fedavg`, `ditto`, `fedrep`, `fedala`, `fedpac`, `fedas`, `fedapa`, `fedltn` or `fedselect`. Baseline behavior is adapter-specific; these are not blanket claims of exact original-paper implementations. FedPAC lacks classifier collaboration; the root FedLTN baseline runs with pruning disabled. FedSelect owns its mask update internally despite `--pruning_method none`. See [baseline instructions and stopping behavior](docs/baseline_reproduction_status.md).

## Experiment utilities

The [workflow guide](docs/canonical_reproduction_path.md#current-multi-seed-runs) provides a JSON scenario example with an explicit input. The launcher uses seeds 2024–2028 by default, records commands/configurations and can aggregate completed runs:

```console
python scripts/run_5seed_experiments.py --config current_run.json --methods FedDyna --dry-run
python scripts/run_5seed_experiments.py --config current_run.json --methods FedDyna --results-root results/current_multiseed --summarize
```

Create `current_run.json` from the documented example first. Additional workflows:

- **Multi-seed:** [launcher](scripts/run_5seed_experiments.py) and [scenario configuration guide](docs/canonical_reproduction_path.md#current-multi-seed-runs).
- **Sensitivity:** [launcher](scripts/run_sensitivity.py) and [required inputs and grids](docs/canonical_reproduction_path.md#auxiliary-tools).
- **Round budgets:** [launcher](scripts/run_rounds.py) and [scenario requirements](docs/canonical_reproduction_path.md#auxiliary-tools). Prepare the appropriate Scenario 1/2/3 input paths first. Its `shared_mask0` entropy update view differs from the main configuration's `scope_aware`; six-round results are not interchangeable with main runs.

Preview modes skip training, but may inspect inputs or write configuration artifacts; see the workflow guide for launcher-specific behavior.

## Evaluation metrics

The implementation reports F1, precision, recall and AUPRC. AUPRC is sklearn **average precision** computed from class-1 softmax scores for each client's full evaluation split. Run-level metrics are equal-client means; five-seed summaries use sample standard deviation. Empty or single-class evaluation sets have explicit zero-valued AP handling. Training and stopping policies differ by method; see the baseline guide.

## Reproducibility limitations

Exact historical numerical reproduction is not guaranteed. Current defaults, baseline-adapter differences and material historical limitations are summarized in [reproducibility_notes.md](docs/reproducibility_notes.md).

## Citation

The associated FedDyna paper has not yet been published. Citation metadata will be added when the final public bibliographic record is available.

## License

The project license is the [MIT License](LICENSE). Copyright (c) 2026 Yingwu Fu. Existing third-party copyright/license notices, including those in the bundled Drain parsers, remain intact and continue to apply to those components.
