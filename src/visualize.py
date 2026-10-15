"""Visualize RFM distributions and customer segments.

Saves:
  figures/rfm_distributions.png  - histograms of Recency/Frequency/Monetary
  figures/clusters_2d.png         - Recency vs Monetary scatter colored by segment
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

plt.style.use("seaborn-v0_8")


def main() -> None:
    rfm = pd.read_csv(ROOT / "data/labeled_rfm.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
        ax.hist(rfm[col], bins=30, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel(col)
    fig.suptitle("RFM Feature Distributions")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/rfm_distributions.png", dpi=120)
    print(f"Saved {ROOT / 'figures/rfm_distributions.png'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for segment, group in rfm.groupby("segment"):
        ax.scatter(group["Recency"], group["Monetary"], s=12, alpha=0.6, label=segment)
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary (total spend)")
    ax.set_title("Customer Segments: Recency vs Monetary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures/clusters_2d.png", dpi=120)
    print(f"Saved {ROOT / 'figures/clusters_2d.png'}")


if __name__ == "__main__":
    main()
