"""Visualisations exploratoires (cellules 10, 11 et 13 du notebook)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR

sns.set_theme(style="whitegrid", context="notebook")


def plot_spread_correlation_heatmap(correl: pd.DataFrame, title: str, filename: str) -> Path:
    """Heatmap verticale des corrélations avec 'spread' / 'log_spread'."""
    spread_corr = correl[["spread", "log_spread"]].sort_values(by="spread", ascending=False)

    plt.figure(figsize=(10, 12))
    sns.heatmap(
        spread_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
    )
    plt.title(title, fontsize=14, pad=15)
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_log_spread_distribution(final_data: pd.DataFrame, filename: str = "log_spread_distribution.png") -> Path:
    """Distribution (KDE) de log_spread une fois les valeurs extrêmes retirées."""
    plt.figure(figsize=(9, 6))
    sns.kdeplot(data=final_data, x="log_spread", fill=True, color="#2c7fb8")
    plt.title("Distribution de log(Fee / Valeur Transfermarkt)")
    plt.xlabel("log_spread")
    plt.tight_layout()

    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
