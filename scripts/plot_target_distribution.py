"""Plot distribution of the binary target variable and save as PDF.

Reads: stroke_risk_dataset.csv
Target: 'At Risk (Binary)' where 0 = Not At Risk, 1 = At Risk

Usage:
  python3 scripts/plot_target_distribution.py \
    --input stroke_risk_dataset.csv \
    --output figures/target_distribution.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="stroke_risk_dataset.csv", help="Path to input CSV")
    p.add_argument(
        "--output",
        default="figures/target_distribution.pdf",
        help="Path to output PDF",
    )
    p.add_argument(
        "--target",
        default="At Risk (Binary)",
        help="Target column name (binary: 0/1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.target not in df.columns:
        raise KeyError(
            f"Target column '{args.target}' not found. Available columns: {list(df.columns)}"
        )

    # Count and map labels
    counts = df[args.target].value_counts().sort_index()
    label_map = {0: "Not At Risk", 1: "At Risk"}
    labels = [label_map.get(int(k), str(k)) for k in counts.index]
    total = int(counts.sum())
    perc = (counts / total * 100.0).round(1)

    # Use a non-interactive backend suitable for headless environments
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#F58518"], width=0.6)

    ax.set_title('Distribution of binary target: "At Risk" vs "Not At Risk"')
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("")
    ax.set_ylim(0, max(counts.values) * 1.15)

    # Annotate with counts and percentages
    for b, c, p in zip(bars, counts.values, perc.values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{int(c):,}\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
