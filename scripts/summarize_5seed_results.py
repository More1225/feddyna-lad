"""Summarize multi-seed FedDyna results.

The statistical unit is one completed seed run. Each run contributes one
run-level mean client metric; seed-level std is computed across those runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "multi_seed"
METRICS = ["accuracy", "precision", "recall", "f1", "prauc", "auroc"]


def parse_csv_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_from_path(run_metrics_path: Path, results_root: Path) -> dict[str, Any]:
    rel = run_metrics_path.relative_to(results_root)
    parts = rel.parts
    inferred: dict[str, Any] = {}
    # Directory layout: dataset_id/scenario/method/seed_{seed}.
    if len(parts) >= 5:
        inferred["dataset"] = parts[0]  # Dataset identifier
        inferred["scenario"] = parts[1]  # Scenario
        inferred["method"] = parts[2]  # Method
        match = re.match(r"seed_(\d+)", parts[3])
        if match:
            inferred["seed"] = int(match.group(1))
    return inferred


def collect_runs(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Directory layout: dataset_id/scenario/method/seed_*/results/run_metrics.json.
    for metrics_path in sorted(results_root.glob("*/*/*/seed_*/results/run_metrics.json")):
        data = read_json(metrics_path)
        inferred = infer_from_path(metrics_path, results_root)
        row = {
            "dataset": inferred.get("dataset", ""),  # Dataset identifier
            "scenario": data.get("scenario") or inferred.get("scenario", ""),
            "method": data.get("method") or inferred.get("method", ""),
            "seed": data.get("seed", inferred.get("seed")),
            "run_dir": str(metrics_path.parents[1]),
            "run_metrics_path": str(metrics_path),
            "status": data.get("status", ""),
        }
        for metric in METRICS:
            row[metric] = data.get(f"mean_client_{metric}", data.get(metric))
        rows.append(row)
    return pd.DataFrame(rows)


def filter_runs(raw_df: pd.DataFrame, datasets: set[str] | None, scenarios: set[str] | None, methods: set[str] | None) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    filtered = raw_df
    if datasets is not None:
        filtered = filtered[filtered["dataset"].isin(datasets)]
    if scenarios is not None:
        filtered = filtered[filtered["scenario"].isin(scenarios)]
    if methods is not None:
        filtered = filtered[filtered["method"].isin(methods)]
    return filtered.copy()


def mean_pm_std(mean_value: float, std_value: float, digits: int) -> str:
    if pd.isna(mean_value):
        return ""
    if pd.isna(std_value):
        return f"{mean_value:.{digits}f} +/- nan"
    return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"


def latex_pm_std(mean_value: float, std_value: float, digits: int) -> str:
    if pd.isna(mean_value):
        return ""
    if pd.isna(std_value):
        return rf"${mean_value:.{digits}f} \pm \mathrm{{nan}}$"
    return rf"${mean_value:.{digits}f} \pm {std_value:.{digits}f}$"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def build_summary(raw_df: pd.DataFrame, digits: int) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    # Group by dataset, scenario, and method.
    for (dataset, scenario, method), group in raw_df.groupby(["dataset", "scenario", "method"], sort=True):
        row: dict[str, Any] = {
            "dataset": dataset,  # Dataset identifier
            "scenario": scenario,
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"].dropna().astype(int).unique())),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean_value = values.mean()
            std_value = values.std(ddof=1)
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_mean_pm_std"] = mean_pm_std(mean_value, std_value, digits)
        rows.append(row)
    return pd.DataFrame(rows)


def build_latex(summary_df: pd.DataFrame, digits: int) -> str:
    # Add the dataset column.
    columns = ["Dataset", "Scenario", "Method", "Acc.", "Prec.", "Recall", "F1", "PRAUC", "AUROC"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Five-seed results reported as mean $\pm$ sample std.}",
        r"\label{tab:multi_seed_results}",
        # Arrange columns as dataset, scenario, method, followed by six metric columns.
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        values = [
            latex_escape(row["dataset"]),  # Dataset
            latex_escape(row["scenario"]),
            latex_escape(row["method"]),
        ]
        for metric in METRICS:
            values.append(latex_pm_std(row[f"{metric}_mean"], row[f"{metric}_std"], digits))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize 5-seed FedDyna experiment results.")
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset identifiers to include.")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenario labels to include.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method labels to include.")
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = filter_runs(
        collect_runs(results_root),
        datasets=parse_csv_filter(args.datasets),
        scenarios=parse_csv_filter(args.scenarios),
        methods=parse_csv_filter(args.methods),
    )
    raw_path = output_dir / "summary_raw_runs.csv"
    summary_path = output_dir / "summary_mean_std.csv"
    latex_path = output_dir / "summary_latex_table.tex"

    if raw_df.empty:
        raw_df.to_csv(raw_path, index=False)
        pd.DataFrame().to_csv(summary_path, index=False)
        latex_path.write_text("% No completed runs found.\n", encoding="utf-8")
        print(f"No completed runs found under {results_root}")
        return 0

    raw_df = raw_df.sort_values(["scenario", "method", "seed"])
    summary_df = build_summary(raw_df, args.digits)

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    latex_path.write_text(build_latex(summary_df, args.digits), encoding="utf-8")

    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
