# Supported baseline adapters

The root FedDyna.py entry runs the current adapters below. This document describes executable paths, not a certification of fidelity to every original baseline paper or historical result. All baseline examples use `--weighting_method fedavg --aggregation_scope all --pruning_method none` and an explicit --data_path. Omit FedDyna-only enable flags. See the [README](../README.md#running-supported-baseline-adapters) for a full command.

| Method | --fl_method | Main implementation | Current local stopping/model selection |
| --- | --- | --- | --- |
| Local-only | local | federated/local_runner.py | Validation AUROC, patience 5; restore improving best state; five total epochs by default |
| FedAvg | fedavg | Shared trainer and aggregation path | Validation loss, patience 3; restore best within fit |
| FedProx | fedprox | methods/fedprox.py | Shared validation-loss path with proximal loss term |
| Ditto | ditto | methods/ditto.py | Global validation-loss path; personalized fixed-epoch loop uses its own optimizer/loss |
| FedRep | fedrep | methods/fedrep.py | Fixed alternating head/representation phases |
| FedALA | fedala | methods/fedala.py | Fixed local epochs; adaptive initialization convergence is separate |
| FedPAC | fedpac | methods/fedpac.py and models/fedpac_lightlog.py | Fixed head/feature phases; local classifiers, prototype alignment, no classifier collaboration |
| FedAS | fedas | methods/fedas.py | Shared validation-loss path |
| FedAPA | fedapa | methods/fedapa.py | Fixed local epochs without validation-selected restoration |
| FedLTN | fedltn | methods/fedltn.py | Shared validation-loss path; root adapter retains local classifiers, pruning disabled in baseline launch |
| FedSelect | fedselect | methods/fedselect.py | Shared validation-loss path; adapter owns mask updates even with outer pruning_method=none |

These paths share data loading, model selection and result serialization where applicable. Alternating and personalized branches have method-specific operations; a top-level optimizer setting does not imply every branch uses that optimizer. `save_best_local_model=False` does not disable the standard trainer's intra-fit restoration of the lowest validation-loss state.

Local-only training uses `federated/local_runner.py`. Historical standalone baseline trees are not distributed.

Known final-associated FedPAC/FedLTN evidence is strongest for surviving Scenario 3 records; some original Scenario 1/2 records are missing. See [provenance notes](reproducibility_notes.md) for qualifications, numerical exceptions and the limits of historical comparisons.
