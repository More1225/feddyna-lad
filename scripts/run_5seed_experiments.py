"""Run reproducible multi-seed FedDyna experiments.

Default output layout:
  results/multi_seed/{scenario}/{method}/seed_{seed}/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [2024, 2025, 2026, 2027, 2028]

DEFAULT_SCENARIOS: dict[str, dict[str, list[str]]] = {
    "HDFS+BGL": {
        "args": [
            "--dataset", "mix",
            "--data_path", str(ROOT / "data" / "MIX" / "fed_data_5HDFS+5BGL" / "mixed_fed_unified.npz"),
        ],
    },
    "HDFS+Thunderbird": {
        "args": [
            "--dataset", "mix",
            "--data_path", str(ROOT / "data" / "MIX" / "fed_data_5HDFS+5TB" / "mixed_fed_unified.npz"),
        ],
    },
    "HDFS+BGL+Thunderbird": {
        "args": [
            "--dataset", "mix",
            "--data_path", str(ROOT / "data" / "MIX" / "fed_data_4HDFS+3BGL+3TB" / "mixed_fed_unified.npz"),
        ],
    },
}

STRICT_BASELINE_ARGS = [
    "--weighting_method", "fedavg",
    "--aggregation_scope", "all",
    "--pruning_method", "none",
]

DEFAULT_METHODS: dict[str, dict[str, list[str]]] = {
    "Local_only": {
        "args": ["--exp_name", "base_Local_only", "--fl_method", "local", *STRICT_BASELINE_ARGS],
    },
    "FedAvg": {
        "args": ["--exp_name", "base_FedAvg", "--fl_method", "fedavg", *STRICT_BASELINE_ARGS],
    },
    "FedProx": {
        "args": ["--exp_name", "base_FedProx", "--fl_method", "fedprox", *STRICT_BASELINE_ARGS],
    },
    "Ditto": {
        "args": ["--exp_name", "base_Ditto", "--fl_method", "ditto", *STRICT_BASELINE_ARGS],
    },
    "FedRep": {
        "args": ["--exp_name", "base_FedRep", "--fl_method", "fedrep", *STRICT_BASELINE_ARGS],
    },
    "FedALA": {
        "args": ["--exp_name", "base_FedALA", "--fl_method", "fedala", *STRICT_BASELINE_ARGS],
    },
    "FedPAC": {
        "args": ["--exp_name", "base_FedPAC", "--fl_method", "fedpac", *STRICT_BASELINE_ARGS],
    },
    "FedAS": {
        "args": ["--exp_name", "base_FedAS", "--fl_method", "fedas", *STRICT_BASELINE_ARGS],
    },
    "FedAPA": {
        "args": ["--exp_name", "base_FedAPA", "--fl_method", "fedapa", *STRICT_BASELINE_ARGS],
    },
    "FedLTN": {
        "args": ["--exp_name", "base_FedLTN", "--fl_method", "fedltn", *STRICT_BASELINE_ARGS],
    },
    "FedSelect": {
        "args": [
            "--exp_name", "cmp_FedSelect",
            "--fl_method", "fedselect",
            "--weighting_method", "fedavg",
            "--aggregation_scope", "all",
            "--pruning_method", "none",
        ],
    },
    "FedDyna": {
        "args": [
            "--exp_name", "main_HA_Entropy_KDARS",
            "--fl_method", "fedavg",
            "--weighting_method", "entropy",
            "--enable_ha_scope",
            "--pruning_method", "kdars",
            "--enable_weight_protection",
        ],
    },
}


def parse_csv_or_default(value: str | None, default: list[Any]) -> list[Any]:
    if not value:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items


def load_config(path: str | None) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
    if not path:
        return DEFAULT_SEEDS, DEFAULT_SCENARIOS, DEFAULT_METHODS

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seeds = cfg.get("seeds", DEFAULT_SEEDS)
    scenarios = cfg.get("scenarios", DEFAULT_SCENARIOS)
    methods = cfg.get("methods", DEFAULT_METHODS)
    return [int(seed) for seed in seeds], scenarios, methods


def ensure_list_args(config_item: dict[str, Any], name: str) -> list[str]:
    args = config_item.get("args", [])
    if not isinstance(args, list):
        raise TypeError(f"{name}.args must be a list")
    return [str(item) for item in args]


def is_completed(run_dir: Path) -> bool:
    completion_path = run_dir / "completion.json"
    run_metrics_path = run_dir / "results" / "run_metrics.json"
    if not completion_path.exists() or not run_metrics_path.exists():
        return False
    try:
        with completion_path.open("r", encoding="utf-8") as f:
            completion = json.load(f)
    except json.JSONDecodeError:
        return False
    return completion.get("status") == "completed"


def build_command(
    python_executable: str,
    scenario: str,
    scenario_args: list[str],
    method: str,
    method_args: list[str],
    seed: int,
    run_dir: Path,
) -> list[str]:
    return [
        python_executable,
        str(ROOT / "FedDyna.py"),
        *scenario_args,
        *method_args,
        "--seed", str(seed),
        "--scenario_tag", scenario,
        "--method_tag", method,
        "--run_dir", str(run_dir),
    ]


def extract_dataset_identifier(args_list: list[str], use_options_scenario: bool = False) -> str:
    """Extract the dataset identifier from an argument list.

    Prefer extracting the dataset name from `--data_path`.
    Fall back to `--dataset` when `--data_path` is not provided.
    In `use-options-scenario` mode, try to infer the identifier from defaults in `options.py`.
    The returned identifier is normalized to contain only letters, digits, and underscores.
    """
    dataset_name = ""

    # Try to infer the dataset identifier from --data_path.
    for i, arg in enumerate(args_list):
        if arg == "--data_path" and i + 1 < len(args_list):
            data_path = args_list[i + 1]
            # Extract the dataset identifier from the path.
            import re
            # Try common naming patterns.
            patterns = [
                r"fed_data_(\d+HDFS\+\d+BGL\+\d+TB)",  # Mixed-dataset pattern
                r"fed_data_(\d+HDFS\+\d+BGL)",  # HDFS+BGL mixture
                r"fed_data_(\d+HDFS\+\d+TB)",   # HDFS+TB mixture
                r"fed_data_(\d+HDFS)",          # HDFS dataset
                r"fed_data_(\d+BGL)",           # BGL dataset
                r"fed_data_(\d+TB)",            # TB dataset
                r"(hdfs|bgl|thunderbird|tb|mix)",  # Base dataset name (last resort)
            ]

            for pattern in patterns:
                match = re.search(pattern, data_path, re.IGNORECASE)
                if match:
                    dataset_name = match.group(1)
                    break

            # Use the filename if no known pattern matches.
            if not dataset_name:
                # Extract the last non-generic path component.
                import os
                dir_name = os.path.dirname(data_path)
                base_name = os.path.basename(dir_name)
                if base_name and base_name not in ["fed_data", "data"]:
                    dataset_name = base_name
                else:
                    # Try to infer it from the directory structure.
                    parts = [p for p in dir_name.split(os.sep) if p and p.lower() not in ["", "data", "fed_data", "hdfs", "bgl", "tb", "thunderbird", "mix"]]
                    if parts:
                        dataset_name = parts[-1]

            if dataset_name:
                break

    # If data_path does not provide an identifier, try --dataset.
    if not dataset_name:
        for i, arg in enumerate(args_list):
            if arg == "--dataset" and i + 1 < len(args_list):
                dataset_name = args_list[i + 1]
                break

    # Use "options" in use-options-scenario mode if no identifier is found.
    if not dataset_name:
        if use_options_scenario:
            dataset_name = "options"
        else:
            dataset_name = "unknown_dataset"

    # Normalize the identifier to lowercase letters, digits, and underscores.
    import re
    dataset_name = re.sub(r'[^a-zA-Z0-9_]', '_', dataset_name)
    dataset_name = dataset_name.lower()

    # Remove redundant underscores.
    dataset_name = re.sub(r'_+', '_', dataset_name)
    dataset_name = dataset_name.strip('_')

    # Add a prefix if the identifier is too short.
    if len(dataset_name) < 3:
        dataset_name = f"dataset_{dataset_name}"

    return dataset_name

def write_launch_metadata(run_dir: Path, command: list[str], scenario: str, method: str, seed: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "command": command,
        "cwd": str(ROOT),
    }
    with (run_dir / "launch.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_one(command: list[str], run_dir: Path) -> int:
    write_mode = "w"
    log_path = run_dir / "log.txt"
    with log_path.open(write_mode, encoding="utf-8", errors="replace") as log_file:
        log_file.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log_file.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        with (run_dir / "failed.json").open("w", encoding="utf-8") as f:
            json.dump({"status": "failed", "returncode": result.returncode}, f, indent=2)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 5-seed FedDyna experiments.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config overriding seeds/scenarios/methods.")
    parser.add_argument("--results-root", type=str, default=str(ROOT / "results" / "multi_seed"))
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method names.")
    parser.add_argument("--method", type=str, default=None, help="Single method name; shorthand for --methods.")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenario names.")
    parser.add_argument(
        "--use-options-scenario",
        action="store_true",
        help="Do not pass dataset/data_path from this script; use config/options.py for the scenario.",
    )
    parser.add_argument(
        "--scenario-label",
        type=str,
        default="options",
        help="Folder/table label used with --use-options-scenario.",
    )
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip runs with completion.json and run_metrics.json.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed run.")
    parser.add_argument("--summarize", action="store_true", help="Summarize selected completed runs after execution.")
    parser.add_argument(
        "--summary-current-method-only",
        action="store_true",
        help="With --summarize, include only the methods selected in this invocation. By default all completed methods in the selected scenario are summarized.",
    )
    args = parser.parse_args()

    cfg_seeds, cfg_scenarios, cfg_methods = load_config(args.config)
    if args.use_options_scenario:
        cfg_scenarios = {args.scenario_label: {"args": []}}

    seeds = [int(seed) for seed in parse_csv_or_default(args.seeds, cfg_seeds)]
    methods_arg = args.methods or args.method
    selected_methods = parse_csv_or_default(methods_arg, list(cfg_methods.keys()))
    selected_scenarios = parse_csv_or_default(args.scenarios, list(cfg_scenarios.keys()))
    results_root = Path(args.results_root).resolve()

    failures: list[dict[str, Any]] = []
    total = 0
    # Collect identifiers for all datasets used in runs.
    all_datasets: set[str] = set()

    for scenario in selected_scenarios:
        if scenario not in cfg_scenarios:
            raise KeyError(f"Unknown scenario: {scenario}")
        scenario_args = ensure_list_args(cfg_scenarios[scenario], f"scenario {scenario}")

        # Infer the dataset identifier at the scenario level.
        dataset_id = extract_dataset_identifier(scenario_args, use_options_scenario=args.use_options_scenario)
        all_datasets.add(dataset_id)

        for method in selected_methods:
            if method not in cfg_methods:
                raise KeyError(f"Unknown method: {method}")
            method_args = ensure_list_args(cfg_methods[method], f"method {method}")

            for seed in seeds:
                # Build the result path: dataset_id/scenario/method/seed_{seed}.
                if args.use_options_scenario:
                    # In use-options-scenario mode, the scenario name itself is the label.
                    run_dir = results_root / dataset_id / scenario / method / f"seed_{seed}"
                else:
                    run_dir = results_root / dataset_id / scenario / method / f"seed_{seed}"

                command = build_command(
                    args.python_executable,
                    scenario,
                    scenario_args,
                    method,
                    method_args,
                    seed,
                    run_dir,
                )
                total += 1

                if args.skip_completed and is_completed(run_dir):
                    print(f"[SKIP] {dataset_id} / {scenario} / {method} / seed {seed}: already completed")
                    continue

                printable = subprocess.list2cmdline(command)
                if args.dry_run:
                    print(printable)
                    continue

                write_launch_metadata(run_dir, command, scenario, method, seed)
                print(f"[RUN] {dataset_id} / {scenario} / {method} / seed {seed}")
                returncode = run_one(command, run_dir)
                if returncode != 0:
                    failure = {
                        "dataset": dataset_id,
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "returncode": returncode,
                        "log": str(run_dir / "log.txt"),
                    }
                    failures.append(failure)
                    print(f"[FAIL] {failure}")
                    if args.stop_on_failure:
                        break
            if failures and args.stop_on_failure:
                break
        if failures and args.stop_on_failure:
            break

    print(f"[DONE] planned={total}, failures={len(failures)}")
    if args.summarize and not args.dry_run and not failures:
        summary_command = [
            args.python_executable,
            str(ROOT / "scripts" / "summarize_5seed_results.py"),
            "--results-root", str(results_root),
            "--output-dir", str(results_root),
        ]
        # Pass all dataset identifiers.
        if all_datasets:
            summary_command.extend(["--datasets", ",".join(sorted(all_datasets))])
        # Pass all scenarios.
        summary_command.extend(["--scenarios", ",".join(selected_scenarios)])

        if args.summary_current_method_only:
            summary_command.extend(["--methods", ",".join(selected_methods)])
        print("[SUMMARY] " + subprocess.list2cmdline(summary_command))
        summary_result = subprocess.run(summary_command, cwd=ROOT, text=True)
        if summary_result.returncode != 0:
            return summary_result.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
