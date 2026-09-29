# Proposed public release tree

Only these 71 files are intended for GitHub publication. This is the intended working-tree snapshot; the changes have not yet been committed. The release includes the MIT License, Copyright (c) 2026 Yingwu Fu.

```text
FedDyna/
|-- config/
|   |-- __init__.py
|   `-- options.py
|-- core/
|   |-- __init__.py
|   |-- exceptions.py
|   |-- mask_semantics.py
|   `-- math_utils.py
|-- data/
|   |-- BGL/
|   |   |-- Drain.py
|   |   `-- prepare_fed_bgl-10client-5-2-3.py
|   |-- HDFS/
|   |   `-- prepare_fed_hdfs_v7.py
|   |-- MIX/
|   |   |-- mix_federated_generator.py
|   |   `-- README.md
|   |-- TB/
|   |   |-- Drain.py
|   |   `-- prepare_fed_thunderbird_10m_logs_shuffle.py
|   |-- __init__.py
|   `-- train_utils.py
|-- docs/
|   |-- baseline_reproduction_status.md
|   |-- canonical_reproduction_path.md
|   |-- dependency_audit.md
|   |-- execution_flow.md
|   |-- final_release_tree.md
|   |-- public_release_allowlist.md
|   `-- reproducibility_notes.md
|-- federated/
|   |-- __init__.py
|   |-- aggregation_ops.py
|   |-- broadcast.py
|   |-- communication.py
|   |-- heterogeneity_aware.py
|   |-- local_runner.py
|   |-- lottery_ticket.py
|   |-- mask_manager.py
|   |-- round_runner.py
|   `-- weighting.py
|-- methods/
|   |-- __init__.py
|   |-- base_method.py
|   |-- debug_utils.py
|   |-- ditto.py
|   |-- fedala.py
|   |-- fedapa.py
|   |-- fedas.py
|   |-- fedltn.py
|   |-- fedpac.py
|   |-- fedprox.py
|   |-- fedrep.py
|   |-- fedselect.py
|   `-- registry.py
|-- models/
|   |-- __init__.py
|   |-- fedpac_lightlog.py
|   `-- lightlog.py
|-- personalization/
|   |-- __init__.py
|   |-- controller.py
|   |-- kdars.py
|   |-- matrix_builders.py
|   |-- metrics_tracker.py
|   `-- update_metrics.py
|-- scripts/
|   |-- ablation_5seed_config.json
|   |-- run_5seed_experiments.py
|   |-- run_rounds.py
|   |-- run_sensitivity.py
|   `-- summarize_5seed_results.py
|-- tools/
|   `-- plot_ha_analysis.py
|-- trainers/
|   |-- __init__.py
|   `-- standard_trainer.py
|-- utils/
|   |-- __init__.py
|   |-- debug_tools.py
|   |-- experiment_saver.py
|   `-- save_results.py
|-- .gitignore
|-- FedDyna.py
|-- LICENSE
|-- README.md
`-- requirements.txt
```

The tree includes the current FedDyna implementation, supported adapters and essential user documentation. Reference preparation is limited to HDFS v7, BGL, Thunderbird shuffle, their required Drain modules and MIX assembly. Historical variants and all unlisted files remain local. Manuscripts, raw/prepared data, results, checkpoints and internal audit evidence are excluded.
