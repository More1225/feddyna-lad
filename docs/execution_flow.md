# Execution Flow

This map follows the current reference implementation: the root entry point, common federated orchestration, configured FedDyna mechanisms, server aggregation and baseline adapters. See the [README](../README.md#running-feddyna) for the main run command.

## Main experiment flow

1. Entry and options: FedDyna.py::main() calls config/options.py::parse_args(), validates the selected method/options, and initializes the configured random seed.
2. Data loading: data/train_utils.py::get_data() reads the configured unified NPZ dataset and constructs client-specific train, validation, and test loaders.
3. Model and output setup: the entry point constructs the configured LightTCN model (and a FedPAC-compatible variant for the relevant adapter), then prepares the run directory through utils/experiment_saver.py.
4. Training dispatch: --fl_method local calls federated/local_runner.py::run_local_training(). Other methods call federated/round_runner.py::run_federated_training().
5. Client work: the federated runner selects participating clients and calls trainers/standard_trainer.py::train_local_standard() or a selected method's training hook/alternating-training implementation in methods/.
6. Server update: client states flow through the selected aggregation path in federated/round_runner.py and federated/aggregation_ops.py. Method hooks can adjust baseline behavior.
7. FedDyna mechanisms: for the FedDyna experiment configuration, the round path obtains client weights from federated/weighting.py, uses federated/heterogeneity_aware.py for the auxiliary one-time aggregation-scope selector based on cross-client update-direction disagreement at the first scheduled mask-update round, after which the selected scope remains fixed. The round also updates personalization state through personalization/controller.py and personalization/kdars.py, and routes mask operations through federated/mask_manager.py and federated/lottery_ticket.py. Client initialization/broadcast is handled in federated/broadcast.py.
8. Evaluation and serialization: the round and entry paths evaluate validation/test splits and collect metrics. utils/experiment_saver.py and utils/save_results.py serialize configurations, history, checkpoints, metrics, and spreadsheet outputs.

For exact ordering of state updates and calls, consult federated/round_runner.py. This document summarizes the call graph and does not redefine algorithm semantics.

## FedDyna-specific modules

| File | Class/function | Caller | Responsibility |
| --- | --- | --- | --- |
| personalization/kdars.py | KDARS scoring/update helpers | personalization/controller.py | Compute configured drift-aware personalization signals |
| personalization/controller.py | Personalization controller | Federated round path | Coordinate personalization-state updates |
| federated/lottery_ticket.py | Mask construction/update helpers | Mask manager and personalization flow | Map parameter-change signals and ratios to mask states |
| federated/mask_manager.py | Mask manager | Federated round path | Own client mask state and update routing |
| federated/weighting.py | Entropy-based weighting helpers | Federated round path | Produce adaptive client aggregation weights |
| federated/heterogeneity_aware.py | Aggregation-scope selector | Federated round path | Select the scope once from cross-client update-direction disagreement at the first scheduled mask-update round, then keep it fixed |
| federated/aggregation_ops.py | Aggregation operations | Federated round path | Aggregate client states under selected scope and masks |
| federated/broadcast.py | Broadcast helpers | Federated round path | Initialize client models from server state subject to mask semantics |

The principal orchestrator is [federated/round_runner.py](../federated/round_runner.py), which defines the invocation order and selector activation/locking behavior.

## Baselines

The methods are imported by methods/__init__.py and registered through methods/registry.py. They share the main entry point and federated infrastructure unless stated otherwise.

| Method | Entry path | Implementation |
| --- | --- | --- |
| FedAvg | FedDyna.py to federated runner | Shared aggregation path; no dedicated adapter class |
| FedProx | --fl_method fedprox | methods/fedprox.py |
| Ditto | --fl_method ditto | methods/ditto.py |
| FedRep | --fl_method fedrep | methods/fedrep.py |
| FedALA | --fl_method fedala | methods/fedala.py |
| FedPAC | --fl_method fedpac | methods/fedpac.py and models/fedpac_lightlog.py |
| FedAS | --fl_method fedas | methods/fedas.py |
| FedAPA | --fl_method fedapa | methods/fedapa.py |
| FedLTN | --fl_method fedltn | methods/fedltn.py |
| FedSelect | --fl_method fedselect (adapter-owned masks) | methods/fedselect.py |
| Local | --fl_method local | federated/local_runner.py |

FedDyna is not a separate --fl_method adapter in this roster. Its main experiment combines the FedAvg path with configured weighting, scope-selection, and KDARS options. Baseline fidelity limitations are documented in docs/baseline_reproduction_status.md.
