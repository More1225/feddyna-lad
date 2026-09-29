"""Single-factor sensitivity commands; dry-run unless --execute is supplied."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GRIDS = {
    "alpha": ("kdi_alpha", [0, 0.3, 0.5, 0.7, 1.0]),
    "rho_max": ("personal_max_ratio", [0.4, 0.6, 0.8, 0.9]),
    "xi": ("agg_weight_smooth", [0, 0.05, 0.1, 0.2, 0.3]),
    "ha_threshold": ("ha_select_threshold", [0.3, 0.4, 0.5, 0.6, 0.7]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=["S1", "S2", "S3"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--experiment", required=True, choices=[*GRIDS, "main", "all"])
    parser.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    parser.add_argument("--values", type=float, nargs="+", help="Subset of the selected single experiment grid")
    parser.add_argument("--results-root", default=str(ROOT / "results" / "sensitivity_v1"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    data_path = Path(args.data_path).resolve()
    if not data_path.is_file():
        parser.error(f"Data file does not exist: {data_path}")
    if args.values is not None and args.experiment in ("main", "all"):
        parser.error("--values requires a single experiment")
    experiments = list(GRIDS) if args.experiment == "all" else list(GRIDS)[:3] if args.experiment == "main" else [args.experiment]
    for experiment in experiments:
        parameter, grid = GRIDS[experiment]
        values = args.values if args.values is not None else grid
        if any(value not in grid for value in values):
            parser.error(f"--values must be drawn from {grid}")
        root = Path(args.results_root).resolve() / experiment
        root.mkdir(parents=True, exist_ok=True)
        defaults = {
            "kdi_alpha": "0.7", "personal_min_ratio": "0.2",
            "personal_max_ratio": "0.8", "agg_weight_min": "0.03",
            "agg_weight_smooth": "0.1", "ha_select_threshold": "0.5",
            "aggregation_scope": "all",
            "ha_scope_mode": "robust", "ha_update_basis": "train_delta",
            "ha_quantile": "0.1", "ha_lambda_std": "0.3",
            "ha_lambda_q": "0.3", "ha_lambda_neg": "0.4",
            "com_rounds": "6", "lth_epoch_iters": "2",
            "local_epochs": "5", "local_bs": "32",
            "lr": "0.0005", "dropout": "0.3", "weight_decay": "0.0005",
            "optimizer": "adamw", "verbose": "1",
        }
        methods = {}
        for value in values:
            settings = dict(defaults, **{parameter: str(value)})
            name = f"{experiment}_{value:g}".replace(".", "p")
            command_args = ["--exp_name", "sensitivity", "--fl_method", "fedavg",
                            "--weighting_method", "entropy", "--pruning_method", "kdars",
                            "--enable_ha_scope", "--enable_weight_protection"]
            for key, setting in settings.items():
                command_args.extend([f"--{key}", setting])
            methods[name] = {"args": command_args}
        config = {
            "seeds": [int(seed) for seed in args.seeds.split(",")],
            "scenarios": {args.scenario: {"args": ["--dataset", args.dataset, "--data_path", str(data_path)]}},
            "methods": methods,
        }
        config_path = root / f"config_{args.scenario}.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous["scenarios"] != config["scenarios"]:
                parser.error(f"Scenario data changed; use a new --results-root: {config_path}")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        command = [sys.executable, str(ROOT / "scripts" / "run_5seed_experiments.py"),
                   "--config", str(config_path), "--results-root", str(root),
                   "--skip-completed", "--stop-on-failure"]
        command += ["--summarize"] if args.execute else ["--dry-run"]
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
