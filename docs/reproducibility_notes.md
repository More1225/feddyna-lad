# Reproducibility notes

This is the current official public reference implementation of FedDyna. It supports inspection and execution of the method on compatible prepared data; it does not guarantee exact numerical or byte-identical reproduction of every historical paper experiment.

## Current behavior relevant to comparisons

- The current weight-decay default is `5e-4`, consistent with identified historical runtime configurations; the final manuscript reports `2e-4`.
- KDARS rounds the retained shared-parameter count before enforcing cumulative personalization bounds. The exact historical source for integer rounding is unavailable.
- Baselines are supported adapters sharing the root pipeline. FedPAC does not implement classifier collaboration, and the root FedLTN adapter runs with pruning disabled. Method-specific training and stopping policies are documented in [baseline instructions](baseline_reproduction_status.md).
- The auxiliary round-budget runner uses `entropy_update_view=shared_mask0`; the identified main Scenario 3 runs use `scope_aware`. Auxiliary six-round runs should not be treated as identical main runs.
- The `w/o EW` variant uses sample-count weighting, not uniform weighting.
- The general five-seed launcher's default mixed-data path names a 4/3/3 cohort. The identified final Scenario 3 cohort and the MIX/round-budget paths use five HDFS, two BGL and three Thunderbird clients. Supply an explicit input path/scenario configuration for current experiments.

These implementation and historical configuration differences should be considered when comparing new runs with manuscript results.

## Documented historical limitations

| Item | Classification | Consequence |
| --- | --- | --- |
| HDFS historical producer identity and client-file lineage | Known limitation | The retained v7 script is a current reference workflow, not a certified historical producer. |
| Thunderbird historical raw-to-final lineage | Known limitation | The retained shuffle script defines its own current input processing; it does not recover every original raw-log limit or historical cohort. |
| Complete preprocessing provenance for every paper input | Known limitation | Historical paper inputs were produced during the original experiment workflow. Exact regeneration of those artifacts is not claimed. |
| Incomplete historical environment/source records and some baseline execution records | Known limitation | Package versions are unpinned and exact historical execution equivalence cannot be certified. |
| Historical numerical exceptions | Known limitation | Reported AUPRC values for Scenario 3 FedALA, Scenario 3 FedSelect, Scenario 1 w/o EW and Scenario 3 w/o EW do not all match the identified raw cohorts. Do not claim those reported values were reproduced. |

Historical datasets, outputs, manuscripts and detailed numerical provenance tables are not distributed here. The limitations above describe the resulting boundaries on comparisons with historical results.

The documented input interface has been checked against the source. This static check does not establish end-to-end preprocessing, training or evaluation success. For current experiments, verify that prepared inputs meet the documented format and that preparation scripts work with your raw data; historical provenance limitations do not replace those checks.

## Record current experiments

Follow the [current workflow](canonical_reproduction_path.md), supply compatible dense NPZ input, and use a new output directory. Record the source revision, full command, effective configuration, software/CUDA versions, input hashes, client IDs, splits and seeds. Validate and pin an environment for your experiment; the requirements file is not a recovered historical lockfile.
