# Public release allowlist

The public repository centers on the FedDyna implementation, its training/evaluation pipeline, model, supported baseline adapters and current execution utilities. Preprocessing is secondary: one reference workflow per dataset plus mixed-client assembly documents how to prepare compatible inputs.

Only the **71 files** individually marked YES are intended for publication. Directory names do not include unlisted descendants. Historical preprocessing provenance is a `DOCUMENTED_LIMITATION`; it is not a release blocker unless the current documented workflow is broken, indeterminate or misleading. Detailed audit evidence remains local.

| Path | Include | Reason |
| --- | --- | --- |
| `.gitignore` | YES | Publication boundary and local/generated file exclusions. |
| `FedDyna.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `LICENSE` | YES | MIT License, Copyright (c) 2026 Yingwu Fu. |
| `README.md` | YES | FedDyna overview, data interface and run instructions. |
| `config/__init__.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `config/options.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `core/__init__.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `core/exceptions.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `core/mask_semantics.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `core/math_utils.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `data/BGL/Drain.py` | YES | Selected current reference preparation workflow or required local dependency; historical variants excluded. |
| `data/BGL/prepare_fed_bgl-10client-5-2-3.py` | YES | Selected current reference preparation workflow or required local dependency; historical variants excluded. |
| `data/HDFS/prepare_fed_hdfs_v7.py` | YES | Selected current reference preparation workflow or required local dependency; historical variants excluded. |
| `data/MIX/README.md` | YES | Selected current reference preparation workflow or required local dependency; historical variants excluded. |
| `data/MIX/mix_federated_generator.py` | YES | Selected current reference preparation workflow or required local dependency; historical variants excluded. |
| `data/TB/Drain.py` | YES | Selected current reference preparation workflow or required local dependency; historical variants excluded. |
| `data/TB/prepare_fed_thunderbird_10m_logs_shuffle.py` | YES | Selected current reference preparation workflow or required local dependency; historical variants excluded. |
| `data/__init__.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `data/train_utils.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `docs/baseline_reproduction_status.md` | YES | Essential execution, baseline, reproducibility or release-scope documentation. |
| `docs/canonical_reproduction_path.md` | YES | Essential execution, baseline, reproducibility or release-scope documentation. |
| `docs/dependency_audit.md` | YES | Essential execution, baseline, reproducibility or release-scope documentation. |
| `docs/execution_flow.md` | YES | Essential execution, baseline, reproducibility or release-scope documentation. |
| `docs/final_release_tree.md` | YES | Essential execution, baseline, reproducibility or release-scope documentation. |
| `docs/public_release_allowlist.md` | YES | Essential execution, baseline, reproducibility or release-scope documentation. |
| `docs/reproducibility_notes.md` | YES | Essential execution, baseline, reproducibility or release-scope documentation. |
| `federated/__init__.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `federated/aggregation_ops.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `federated/broadcast.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `federated/communication.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `federated/heterogeneity_aware.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `federated/local_runner.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `federated/lottery_ticket.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `federated/mask_manager.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `federated/round_runner.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `federated/weighting.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/__init__.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/base_method.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/debug_utils.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/ditto.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/fedala.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/fedapa.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/fedas.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/fedltn.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/fedpac.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/fedprox.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/fedrep.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/fedselect.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `methods/registry.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `models/__init__.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `models/fedpac_lightlog.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `models/lightlog.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `personalization/__init__.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `personalization/controller.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `personalization/kdars.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `personalization/matrix_builders.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `personalization/metrics_tracker.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `personalization/update_metrics.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `requirements.txt` | YES | Dependencies for the selected implementation and reference preparation tools. |
| `scripts/ablation_5seed_config.json` | YES | Current experiment runner, configuration or result aggregation utility. |
| `scripts/run_5seed_experiments.py` | YES | Current experiment runner, configuration or result aggregation utility. |
| `scripts/run_rounds.py` | YES | Current experiment runner, configuration or result aggregation utility. |
| `scripts/run_sensitivity.py` | YES | Current experiment runner, configuration or result aggregation utility. |
| `scripts/summarize_5seed_results.py` | YES | Current experiment runner, configuration or result aggregation utility. |
| `tools/plot_ha_analysis.py` | YES | Plot diagnostics generated by the current selector. |
| `trainers/__init__.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `trainers/standard_trainer.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `utils/__init__.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `utils/debug_tools.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `utils/experiment_saver.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |
| `utils/save_results.py` | YES | FedDyna core, model, training/evaluation infrastructure or supported baseline adapter dependency. |

## Exclusions

| Path/category | Include | Reason |
| --- | --- | --- |
| `paper/` | NO | Manuscript artifacts are intentionally excluded from the public code release. |
| `data/HDFS/Drain.py` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `data/HDFS/prepare_fed_hdfs_v6.py` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `data/MIX/client_selector.py` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `data/TB/prepare_fed_thunderbird_10m_logs_sequential.py` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `data/config_merge.yaml` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `data/merge_hdfs_bgl_dynamic.py` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `data/merge_hdfs_bgl_federated.py` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `docs/experiment_commands_kdars.md` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `docs/feddyna_code_logic.md` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `scripts/summarize_communication_cost.py` | NO | Historical, alternate or redundant workflow/documentation outside the current minimal release; preserved locally. |
| `Controlled Experiment/`, backups and historical preprocessing variants | NO | Independent archives and superseded variants are not required by the current public pipeline. |
| Internal provenance, terminology, paper-code and release-preparation audit documents | NO | Detailed historical evidence remains local; user-relevant limitations are summarized in reproducibility_notes.md. |
| Raw/prepared datasets, experiment archives, checkpoints, temporary results and caches | NO | Local inputs and generated artifacts are not source-release files. |
| `analysis/`, `figures/`, submission materials and temporary plotting scripts | NO | Local-only or manuscript-oriented artifacts. |
| Any file not individually listed YES | NO | Explicit allowlist; no blanket directory inclusion. |

## Git publication boundary

This list defines the intended source snapshot, not the current Git index or history. Existing tracked files and historical commits may still contain excluded material; `.gitignore` does not remove them. A separate final Git-boundary check is required before publishing. Excluded files must remain on disk. No remote configuration or push is part of this scope adjustment.
