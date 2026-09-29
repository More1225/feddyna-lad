#!/usr/bin/env python
# coding: utf-8
"""
Plot passive aggregation-scope diagnostics saved by FedDyna.

Example:
  python tools/plot_ha_analysis.py \
      --input_dir save/exp/ha_analysis \
      --output_dir save/exp/ha_figures \
      --round latest \
      --tau 0.55
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


SOURCE_COLORS = {
    "hdfs": "#1f77b4",
    "bgl": "#d62728",
    "tb": "#2ca02c",
    "thunderbird": "#2ca02c",
    "unknown": "#7f7f7f",
}
SOURCE_MARKERS = {
    "hdfs": "o",
    "bgl": "s",
    "tb": "^",
    "thunderbird": "^",
    "unknown": "D",
}


def set_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
    })


def save_figure(fig, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def discover_round_dirs(input_dir):
    input_dir = Path(input_dir)
    round_dirs = []
    for child in input_dir.iterdir():
        if child.is_dir():
            match = re.fullmatch(r"round_(\d+)", child.name)
            if match:
                round_dirs.append((int(match.group(1)), child))
    return sorted(round_dirs)


def select_round_dir(input_dir, round_arg):
    rounds = discover_round_dirs(input_dir)
    if not rounds:
        raise FileNotFoundError(f"No round_XXX directories found in {input_dir}")
    if str(round_arg).lower() == "latest":
        return rounds[-1]
    round_id = int(round_arg)
    for rid, path in rounds:
        if rid == round_id:
            return rid, path
    raise FileNotFoundError(f"Round {round_id} not found in {input_dir}")


def read_labeled_matrix(csv_path):
    labels = []
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_labels = header[1:]
        for row in reader:
            labels.append(row[0])
            rows.append([float(value) for value in row[1:]])
    if not labels:
        labels = col_labels
    return np.asarray(rows, dtype=np.float32), labels


def read_metadata(round_dir):
    path = Path(round_dir) / "client_metadata.csv"
    metadata = {}
    if not path.exists():
        return metadata
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[str(row["client_id"])] = row.get("source", "unknown").lower()
    return metadata


def plot_heatmap(matrix, labels, title, output_dir, stem, vmin=None, vmax=None, cmap="viridis"):
    width = max(3.4, min(7.2, 0.28 * len(labels) + 1.4))
    fig, ax = plt.subplots(figsize=(width, width * 0.86))
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    ax.set_title(title, pad=4)
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Client ID")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(width=0.5, length=2)
    ax.tick_params(width=0.5, length=2)
    save_figure(fig, output_dir, stem)


def plot_pca(round_dir, labels, metadata, output_dir, stem):
    update_path = Path(round_dir) / "client_update_vectors.npy"
    if not update_path.exists():
        print(f"[skip] Missing {update_path}")
        return

    updates = np.load(update_path)
    if updates.shape[0] < 2:
        print("[skip] PCA needs at least two clients")
        return

    n_components = 2 if updates.shape[0] >= 2 else 1
    coords = PCA(n_components=n_components, random_state=0).fit_transform(updates)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    sources = [metadata.get(str(label), "unknown").lower() for label in labels]
    for source in sorted(set(sources)):
        idx = [i for i, value in enumerate(sources) if value == source]
        color = SOURCE_COLORS.get(source, SOURCE_COLORS["unknown"])
        marker = SOURCE_MARKERS.get(source, SOURCE_MARKERS["unknown"])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=24,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.35,
            label=source.upper() if source != "tb" else "Thunderbird",
        )

    for idx, label in enumerate(labels):
        ax.annotate(str(label), (coords[idx, 0], coords[idx, 1]), xytext=(2, 2),
                    textcoords="offset points", fontsize=6)

    ax.set_title("PCA of Client Update Vectors", pad=4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, output_dir, stem)


def read_scores(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "round_id": int(row["round_id"]),
                "H_cos": float(row["H_cos"]),
                "H_mask": float(row["H_mask"]),
                "H_total": float(row["H_total"]),
            })
    return rows


def plot_score_curves(input_dir, output_dir, tau):
    scores_path = Path(input_dir) / "ha_scores.csv"
    if not scores_path.exists():
        print(f"[skip] Missing {scores_path}")
        return
    rows = read_scores(scores_path)
    if not rows:
        print("[skip] ha_scores.csv is empty")
        return

    x = np.array([row["round_id"] for row in rows])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(x, [row["H_cos"] for row in rows], marker="o", markersize=3, label=r"$H_{cos}$")
    ax.plot(x, [row["H_mask"] for row in rows], marker="s", markersize=3, label=r"$H_{mask}$")
    ax.plot(x, [row["H_total"] for row in rows], marker="^", markersize=3, label=r"$H_{total}$")
    ax.axhline(tau, color="#444444", linestyle="--", linewidth=0.8, label=rf"$\tau={tau:.2f}$")
    ax.set_title("Round-wise Update-Disagreement Scores", pad=4)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Update-Direction Disagreement Score")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output_dir, "ha_scores_curve")


def main():
    parser = argparse.ArgumentParser(description="Plot FedDyna aggregation-scope diagnostics.")
    parser.add_argument("--input_dir", required=True, help="Directory produced by --save_ha_analysis.")
    parser.add_argument("--output_dir", required=True, help="Directory for generated PNG/PDF figures.")
    parser.add_argument("--round", default="latest", help="Round id to plot, or 'latest'.")
    parser.add_argument("--tau", type=float, default=0.55, help="Display threshold for the update-disagreement score curve.")
    args = parser.parse_args()

    set_ieee_style()
    round_id, round_dir = select_round_dir(args.input_dir, args.round)
    metadata = read_metadata(round_dir)

    cosine_csv = Path(round_dir) / "pairwise_cosine.csv"
    if cosine_csv.exists():
        cosine_matrix, labels = read_labeled_matrix(cosine_csv)
        plot_heatmap(
            cosine_matrix,
            labels,
            "Client-update Cosine Similarity",
            args.output_dir,
            f"round_{round_id:03d}_cosine_heatmap",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        plot_pca(round_dir, labels, metadata, args.output_dir, f"round_{round_id:03d}_update_pca")
    else:
        print(f"[skip] Missing {cosine_csv}")

    mask_csv = Path(round_dir) / "pairwise_mask_disagreement.csv"
    if mask_csv.exists():
        mask_matrix, mask_labels = read_labeled_matrix(mask_csv)
        plot_heatmap(
            mask_matrix,
            mask_labels,
            "Client Mask Disagreement",
            args.output_dir,
            f"round_{round_id:03d}_mask_disagreement_heatmap",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
    else:
        print(f"[skip] Missing {mask_csv}")

    plot_score_curves(args.input_dir, args.output_dir, args.tau)
    print(f"Aggregation-scope diagnostic figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
