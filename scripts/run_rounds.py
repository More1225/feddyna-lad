"""FedDyna and baseline round-budget study. Preview unless --execute is supplied."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "S1": "5HDFS+5TB",
    "S2": "5HDFS+5BGL",
    "S3": "5HDFS+2BGL+3TB",
}


METHODS = ["FedDyna", "FedAvg", "FedProx", "FedAS", "FedSelect", "FedAPA"]
BASELINE_SETTINGS = {
    "FedAvg": {},
    "FedProx": {"fedprox_mu": 0.01},
    "FedAS": {"fedas_align_weight": 0.5, "fedas_sync_threshold": 0.1},
    "FedSelect": {"fedselect_keep_ratio": 0.8, "fedselect_update_interval": 3},
    "FedAPA": {"fedapa_server_lr": 0.1, "fedapa_clamp_min": 0.5, "fedapa_clamp_max": 1.0},
}


def method_args(rounds, method="FedDyna"):
    if method != "FedDyna":
        settings = {
            "exp_name": f"rounds_{method}_R{rounds:02d}",
            "fl_method": method.lower(), "weighting_method": "fedavg",
            "aggregation_scope": "all", "pruning_method": "none",
            "com_rounds": rounds, "lth_epoch_iters": 2,
            "model": "lightlog", "loss_mode": "weighted_ce",
            "local_epochs": 5, "local_bs": 32, "lr": 0.0005,
            "dropout": 0.3, "weight_decay": 0.0005, "optimizer": "adamw",
            "early_stopping_patience": 3, "early_stopping_delta": 0.0,
            "verbose": 1, **BASELINE_SETTINGS[method],
        }
        return [token for key, value in settings.items() for token in (f"--{key}", str(value))]

    settings = {
        "exp_name": f"rounds_FedDyna_R{rounds:02d}",
        "fl_method": "fedavg", "weighting_method": "entropy",
        "pruning_method": "kdars", "entropy_update_view": "shared_mask0",
        "kdi_alpha": 0.7, "personal_min_ratio": 0.2, "personal_max_ratio": 0.8,
        "agg_weight_min": 0.03, "agg_weight_smooth": 0.1,
        "ha_select_threshold": 0.5,
        "aggregation_scope": "all", "ha_scope_mode": "robust",
        "ha_update_basis": "train_delta", "ha_quantile": 0.1,
        "ha_lambda_std": 0.3, "ha_lambda_q": 0.3, "ha_lambda_neg": 0.4,
        "com_rounds": rounds, "lth_epoch_iters": 2,
        "local_epochs": 5, "local_bs": 32, "lr": 0.0005,
        "dropout": 0.3, "weight_decay": 0.0005, "optimizer": "adamw",
        "early_stopping_patience": 3, "early_stopping_delta": 0.0,
        "verbose": 1,
    }
    args = ["--enable_ha_scope", "--enable_weight_protection"]
    for key, value in settings.items():
        args.extend([f"--{key}", str(value)])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["FedDyna"])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--rounds", nargs="+", type=int, choices=[6, 10, 20, 30], default=[6, 10, 20, 30])
    parser.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    parser.add_argument("--results-root", default=str(ROOT / "results" / "rounds_v1"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("Seeds must be nonempty and unique")
    if len(set(args.rounds)) != len(args.rounds) or len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Scenarios and rounds must be unique")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be unique")
    # Parse every setting before creating files or starting expensive work.
    sys.path.insert(0, str(ROOT))
    from config.options import build_parser, validate_args
    from run_5seed_experiments import extract_dataset_identifier, is_completed
    plans = []
    for scenario in args.scenarios:
        data = ROOT / "data" / "MIX" / f"fed_data_{SCENARIOS[scenario]}" / "mixed_fed_unified.npz"
        if not data.is_file():
            parser.error(f"Missing data: {data}")
        scenario_args = ["--dataset", "mix", "--data_path", str(data)]
        methods = {f"{method}_R{r:02d}": {"args": method_args(r, method)}
                   for method in args.methods for r in args.rounds}
        for item in methods.values():
            parsed = build_parser().parse_args(scenario_args + item["args"])
            validate_args(parsed)
            if parsed.save_best_local_model:
                parser.error("This study requires save_best_local_model=False")
        config = {"seeds": seeds, "scenarios": {scenario: {"args": scenario_args}}, "methods": methods}
        plans.append((scenario, extract_dataset_identifier(scenario_args), config))
    output = Path(args.results_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for scenario, dataset, config in plans:
        label = f"rounds_{scenario}_R{'-'.join(f'{r:02d}' for r in args.rounds)}_seeds{'-'.join(map(str, seeds))}"
        if args.methods != ["FedDyna"]:
            label = f"{'-'.join(args.methods)}_{label}"
        config_path = output / f"config_{label}.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
            parser.error(f"Config changed; choose a new --results-root: {config_path}")
        # Prevent --skip-completed from silently mixing settings in the same directory.
        for method, item in config["methods"].items():
            for seed in seeds:
                run_dir = output / dataset / scenario / method / f"seed_{seed}"
                launch_path = run_dir / "launch.json"
                if run_dir.exists():
                    if not launch_path.is_file():
                        parser.error(f"Existing run lacks launch metadata; use a new root: {run_dir}")
                    old = json.loads(launch_path.read_text(encoding="utf-8"))["command"]
                    from run_5seed_experiments import build_command
                    expected = build_command(sys.executable, scenario, config["scenarios"][scenario]["args"], method, item["args"], seed, run_dir)
                    if old[2:] != expected[2:]:
                        parser.error(f"Existing run settings differ; use a new root: {run_dir}")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        command = [sys.executable, str(ROOT / "scripts" / "run_5seed_experiments.py"),
                   "--config", str(config_path), "--results-root", str(output),
                   "--skip-completed", "--stop-on-failure"]
        if not args.execute:
            command.append("--dry-run")
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode
        if args.execute:
            for method in config["methods"]:
                for seed in seeds:
                    run_dir = output / dataset / scenario / method / f"seed_{seed}"
                    if not is_completed(run_dir):
                        raise RuntimeError(f"Incomplete run: {run_dir}")
                    stem = f"{method}_{scenario}_{dataset}_seed{seed}"
                    for source_name in ("client_metrics.csv", "run_metrics.json"):
                        shutil.copy2(run_dir / "results" / source_name, run_dir / "results" / f"{stem}_{source_name}")
                    named_xlsx = run_dir / "results" / f"{stem}.xlsx"
                    originals = [p for p in (run_dir / "results").glob("*.xlsx") if p != named_xlsx]
                    if len(originals) != 1:
                        raise RuntimeError(f"Expected one original Excel file: {run_dir}")
                    shutil.copy2(originals[0], named_xlsx)
            summary = output / f"summary_{label}"
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_5seed_results.py"),
                                     "--results-root", str(output), "--output-dir", str(summary),
                                     "--scenarios", scenario, "--methods", ",".join(config["methods"])], cwd=ROOT)
            if result.returncode:
                return result.returncode
    print(f"{'EXECUTED' if args.execute else 'PREVIEW'}: {len(plans) * len(args.methods) * len(args.rounds) * len(seeds)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
