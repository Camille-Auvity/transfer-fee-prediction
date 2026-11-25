"""Analyse d'erreurs du meilleur modèle (XGBoost) au-delà de la simple
segmentation par tranche de valorisation déjà faite dans `evaluation.py` :
position, âge, championnat de destination et saison du transfert — plus le
biais directionnel (sur-estimation vs sous-estimation systématique), pas
seulement l'ampleur de l'erreur.

Module séparé : ne modifie rien au pipeline principal ni à `tune_models.py`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR

sns.set_theme(style="whitegrid", context="notebook")

AGE_BINS = [0, 21, 24, 27, 30, 33, 100]
AGE_LABELS = ["<21", "21-23", "24-26", "27-29", "30-32", "33+"]


def build_error_frame(
    transfer_full_raw: pd.DataFrame, df_model: pd.DataFrame,
    X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray,
) -> pd.DataFrame:
    """Reconstruit une ligne par transfert du test set avec l'erreur du modèle
    et les variables catégorielles brutes (position, championnat...) qui ont
    été one-hot encodées avant la modélisation."""
    meta_cols = [
        "player_id", "date", "transfer_season", "position", "foot",
        "domestic_competition_id_to", "age_at_transfer",
    ]
    test_keys = df_model.loc[X_test.index, ["player_id", "date"]]
    meta = test_keys.merge(transfer_full_raw[meta_cols], on=["player_id", "date"], how="left")
    meta.index = X_test.index

    tm_val = np.exp(X_test["log_curr_market_value"].values)
    actual_fee = np.exp(X_test["log_curr_market_value"].values + y_test.values)
    pred_fee = np.exp(X_test["log_curr_market_value"].values + y_pred)

    err = pd.DataFrame({
        "date": meta["date"].values,
        "position": meta["position"].values,
        "foot": meta["foot"].values,
        "league_to": meta["domestic_competition_id_to"].values,
        "age_at_transfer": meta["age_at_transfer"].values,
        "transfer_season": meta["transfer_season"].values,
        "tm_val": tm_val,
        "actual_fee": actual_fee,
        "pred_fee": pred_fee,
    }, index=X_test.index)

    err["age_bucket"] = pd.cut(err["age_at_transfer"], bins=AGE_BINS, labels=AGE_LABELS)
    err["abs_pct_err"] = np.abs(err["pred_fee"] - err["actual_fee"]) / err["actual_fee"] * 100
    # Biais signé : positif = le modèle SUR-estime le montant réel, négatif = il SOUS-estime
    err["signed_pct_err"] = (err["pred_fee"] - err["actual_fee"]) / err["actual_fee"] * 100
    err["tm_abs_pct_err"] = np.abs(err["tm_val"] - err["actual_fee"]) / err["actual_fee"] * 100
    return err


def _grouped_summary(err: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = err.groupby(group_col, observed=False).agg(
        N=("abs_pct_err", "count"),
        Median_Abs_Pct_Err=("abs_pct_err", "median"),
        Mean_Signed_Pct_Err=("signed_pct_err", "mean"),
    ).reset_index()
    return summary[summary["N"] > 0]


def build_breakdown_tables(err: pd.DataFrame) -> dict:
    """Une table par dimension : position, âge, championnat de destination, saison."""
    return {
        "position": _grouped_summary(err, "position"),
        "age_bucket": _grouped_summary(err, "age_bucket"),
        "league_to": _grouped_summary(err, "league_to"),
        "transfer_season": _grouped_summary(err, "transfer_season"),
    }


def bucket_by_time(err: pd.DataFrame, n_buckets: int = 10) -> pd.DataFrame:
    """Découpe le jeu de test en N tranches chronologiques de taille égale
    (plutôt que par mois civil, trop irrégulier à cause des fenêtres de
    transfert) et calcule l'erreur du modèle et de Transfermarkt seul sur
    chacune — pour voir si l'erreur est stable dans le temps ou concentrée
    sur certaines périodes."""
    err_sorted = err.sort_values("date").reset_index(drop=True)
    err_sorted["time_bucket"] = pd.qcut(err_sorted.index, q=n_buckets, duplicates="drop")

    summary = err_sorted.groupby("time_bucket", observed=True).agg(
        N=("abs_pct_err", "count"),
        Date_Min=("date", "min"),
        Date_Max=("date", "max"),
        Model_Median_Abs_Pct_Err=("abs_pct_err", "median"),
        Model_Mean_Signed_Pct_Err=("signed_pct_err", "mean"),
        TM_Median_Abs_Pct_Err=("tm_abs_pct_err", "median"),
    ).reset_index(drop=True)

    summary["Period"] = summary["Date_Min"].dt.strftime("%d/%m/%y") + " – " + summary["Date_Max"].dt.strftime("%d/%m/%y")
    return summary


def plot_error_over_time(time_summary: pd.DataFrame, filename: str = "error_over_time.png"):
    """Évolution de l'erreur médiane (modèle vs. Transfermarkt seul) et du
    biais signé du modèle, tranche chronologique par tranche chronologique."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    x = range(len(time_summary))

    ax = axes[0]
    ax.plot(x, time_summary["Model_Median_Abs_Pct_Err"], marker="o", color="#2ecc71", lw=2, label="Modèle (XGBoost)")
    ax.plot(x, time_summary["TM_Median_Abs_Pct_Err"], marker="o", color="#c0392b", lw=2, ls="--", label="Transfermarkt seul")
    for i, n in enumerate(time_summary["N"]):
        ax.annotate(f"n={n}", (i, time_summary["Model_Median_Abs_Pct_Err"].iloc[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="dimgray")
    ax.set_ylabel("Erreur absolue médiane (%)")
    ax.set_title("Évolution de l'erreur dans le temps (tranches chronologiques de taille égale)")
    ax.legend()

    ax2 = axes[1]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in time_summary["Model_Mean_Signed_Pct_Err"]]
    ax2.bar(x, time_summary["Model_Mean_Signed_Pct_Err"], color=colors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Biais signé moyen (%)\n(+ = sur-estime)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(time_summary["Period"], rotation=35, ha="right", fontsize=8)
    ax2.set_xlabel("Période (date du transfert)")

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_error_magnitude(breakdowns: dict, filename: str = "error_analysis_magnitude.png"):
    """Ampleur de l'erreur (erreur absolue médiane en %) par dimension."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    specs = [
        ("position", "Position", axes[0, 0]),
        ("age_bucket", "Tranche d'âge au transfert", axes[0, 1]),
        ("league_to", "Championnat du club acheteur", axes[1, 0]),
        ("transfer_season", "Saison du transfert", axes[1, 1]),
    ]
    for key, title, ax in specs:
        df = breakdowns[key]
        sns.barplot(data=df, x=df.columns[0], y="Median_Abs_Pct_Err", hue=df.columns[0], palette="mako", legend=False, ax=ax)
        for i, (n, v) in enumerate(zip(df["N"], df["Median_Abs_Pct_Err"])):
            ax.text(i, v + 0.5, f"n={n}", ha="center", fontsize=8, color="dimgray")
        ax.set_title(f"Erreur médiane (%) par {title}")
        ax.set_xlabel("")
        ax.set_ylabel("Erreur absolue médiane (%)")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_error_bias(breakdowns: dict, filename: str = "error_analysis_bias.png"):
    """Biais directionnel (erreur signée moyenne en %) : positif = le modèle
    paie le joueur plus cher que le montant réel (sur-estimation), négatif =
    moins cher (sous-estimation)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    specs = [
        ("position", "Position", axes[0, 0]),
        ("age_bucket", "Tranche d'âge au transfert", axes[0, 1]),
        ("league_to", "Championnat du club acheteur", axes[1, 0]),
        ("transfer_season", "Saison du transfert", axes[1, 1]),
    ]
    for key, title, ax in specs:
        df = breakdowns[key]
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in df["Mean_Signed_Pct_Err"]]
        ax.bar(df[df.columns[0]].astype(str), df["Mean_Signed_Pct_Err"], color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"Biais (%) par {title}\n(+ = sur-estime, − = sous-estime)")
        ax.set_xlabel("")
        ax.set_ylabel("Erreur signée moyenne (%)")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
