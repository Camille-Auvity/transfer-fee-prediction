"""Comparaison finale 'Transfermarkt seul' vs. modèles, et exemples concrets de
joueurs/transferts dans le jeu de test (cellules 28, 29 et 31 du notebook)."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .config import FIGURES_DIR, PLAYERS_CSV

SHOW_COLS = [
    "Player", "transfer_season", "name_from", "name_to",
    "TM_Val (€M)", "Pred_Fee (€M)", "Actual_Fee (€M)",
    "TM_Err (%)", "Model_Err (%)", "Beat_TM",
]


def compare_models_vs_transfermarkt(X_test: pd.DataFrame, y_test: pd.Series, predictions: dict) -> pd.DataFrame:
    """Reconstruit le montant réel et comparé au montant Transfermarkt seul (cellule 28)."""
    if "log_curr_market_value" in X_test.columns:
        log_mv_test = X_test["log_curr_market_value"].values
    else:
        raise KeyError("log_curr_market_value introuvable dans X_test")

    actual_log_fee = log_mv_test + y_test.values
    actual_fee = np.exp(actual_log_fee)

    all_preds = {"Transfermarkt Alone (spread = 0)": np.zeros(len(y_test)), **predictions}

    fee_rows = []
    for name, pred_log_spread in all_preds.items():
        pred_log_fee = log_mv_test + pred_log_spread
        pred_fee = np.exp(pred_log_fee)

        abs_err_eur = np.abs(pred_fee - actual_fee)
        pct_err = abs_err_eur / actual_fee

        fee_rows.append({
            "Model": name,
            "R2 (log Fee)": r2_score(actual_log_fee, pred_log_fee),
            "R2 (EUR Fee)": r2_score(actual_fee, pred_fee),
            "MAE (€M)": mean_absolute_error(actual_fee, pred_fee) / 1e6,
            "RMSE (€M)": root_mean_squared_error(actual_fee, pred_fee) / 1e6,
            "Median % Error": np.median(pct_err) * 100,
            "Within ±20% (%)": (pct_err <= 0.20).mean() * 100,
            "Within ±30% (%)": (pct_err <= 0.30).mean() * 100,
        })

    return (
        pd.DataFrame(fee_rows)
        .sort_values("R2 (log Fee)", ascending=False)
        .reset_index(drop=True)
    )


def plot_fee_comparison(fee_comparison_df: pd.DataFrame, filename: str = "final_fee_comparison.png"):
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.barplot(
        data=fee_comparison_df, x="R2 (log Fee)", y="Model",
        hue="Model", palette="viridis", legend=False, ax=axes[0]
    )
    axes[0].set_title("Final Transfer Fee — R² on log(Fee) (Higher is Better)")
    axes[0].set_xlim(left=max(0, fee_comparison_df["R2 (log Fee)"].min() - 0.05))
    axes[0].set_ylabel("")

    sns.barplot(
        data=fee_comparison_df.sort_values("Median % Error"), x="Median % Error", y="Model",
        hue="Model", palette="mako", legend=False, ax=axes[1]
    )
    axes[1].set_title("Final Transfer Fee — Median % Error (Lower is Better)")
    axes[1].set_ylabel("")

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def build_test_players_table(
    transfer_full: pd.DataFrame, df_model: pd.DataFrame, X_test: pd.DataFrame,
    y_test: pd.Series, predictions: dict, fee_comparison_df: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Table détaillée par transfert du jeu de test, avec le meilleur modèle (cellule 29)."""
    # Pick the best model by R2 on log(Fee) (excluding the TM benchmark)
    best_model_name = fee_comparison_df.loc[
        fee_comparison_df["Model"] != "Transfermarkt Alone (spread = 0)", "Model"
    ].iloc[0]
    best_pred_spread = predictions[best_model_name]

    meta_cols = ["player_id", "date", "transfer_season", "name_from", "name_to", "transfer_fee", "market_value_in_eur"]

    if set(X_test.index).issubset(set(transfer_full.index)):
        test_players = transfer_full.loc[X_test.index, meta_cols].copy()
    else:
        test_players = df_model.loc[X_test.index, ["player_id", "date"]].merge(
            transfer_full[meta_cols], on=["player_id", "date"], how="left"
        )

    # Attach player names from players.csv if present
    if os.path.exists(PLAYERS_CSV):
        players_df = pd.read_csv(PLAYERS_CSV)
        name_col = "name" if "name" in players_df.columns else [c for c in players_df.columns if "name" in c.lower()][0]
        test_players = test_players.merge(
            players_df[["player_id", name_col]].drop_duplicates("player_id"),
            on="player_id", how="left"
        ).rename(columns={name_col: "Player"})
    else:
        test_players["Player"] = "ID_" + test_players["player_id"].astype(str)

    # Compute predicted fee and percentage errors
    test_players["TM_Val (€M)"] = test_players["market_value_in_eur"] / 1e6
    test_players["Pred_Fee (€M)"] = (test_players["market_value_in_eur"] * np.exp(best_pred_spread)) / 1e6
    test_players["Actual_Fee (€M)"] = test_players["transfer_fee"] / 1e6

    test_players["TM_Err (%)"] = np.abs(test_players["TM_Val (€M)"] - test_players["Actual_Fee (€M)"]) / test_players["Actual_Fee (€M)"] * 100
    test_players["Model_Err (%)"] = np.abs(test_players["Pred_Fee (€M)"] - test_players["Actual_Fee (€M)"]) / test_players["Actual_Fee (€M)"] * 100
    test_players["Beat_TM"] = test_players["Model_Err (%)"] < test_players["TM_Err (%)"]

    print(f"Best model evaluated: {best_model_name}")
    print(f"Overall test set — Model beats Transfermarkt on {test_players['Beat_TM'].mean():.1%} of transfers\n")

    return best_model_name, test_players


def top_mid_sample_tables(test_players: pd.DataFrame) -> dict:
    """Table A (top 15), table intermédiaire et échantillon aléatoire (cellule 29)."""
    top15_test = test_players.sort_values("Actual_Fee (€M)", ascending=False).head(15)
    mid15_test = (
        test_players[test_players["Actual_Fee (€M)"].between(1, 10)]
        .sample(15, random_state=42)
        .sort_values("Actual_Fee (€M)", ascending=False)
    )
    sample10_test = test_players.sample(10, random_state=42).sort_values("Actual_Fee (€M)", ascending=False)

    return {
        "top15": top15_test[SHOW_COLS].round(2).reset_index(drop=True),
        "mid15": mid15_test[SHOW_COLS].round(2).reset_index(drop=True),
        "sample10": sample10_test[SHOW_COLS].round(2).reset_index(drop=True),
    }, top15_test


def plot_top_transfers(top15_test: pd.DataFrame, best_model_name: str, filename: str = "top15_transfers.png"):
    """Barres comparant TM / prédiction / montant réel pour les 12 plus gros transferts (cellule 31)."""
    plot_df = top15_test.head(12).iloc[::-1].copy()
    labels = (
        plot_df["Player"].fillna("Unknown")
        + " (" + plot_df["transfer_season"].astype(str)
        + " → " + plot_df["name_to"].fillna("Unknown").str.slice(0, 14) + ")"
    )

    y_pos = np.arange(len(plot_df))
    h = 0.26

    plt.figure(figsize=(12, 7))
    plt.barh(y_pos + h, plot_df["TM_Val (€M)"], height=h, label="Transfermarkt Value", color="#95a5a6")
    plt.barh(y_pos, plot_df["Pred_Fee (€M)"], height=h, label=f"Predicted Fee ({best_model_name})", color="#2ecc71")
    plt.barh(y_pos - h, plot_df["Actual_Fee (€M)"], height=h, label="Actual Transfer Fee", color="#2c3e50")

    plt.yticks(y_pos, labels, fontsize=9)
    plt.xlabel("Transfer Amount (€ Millions)")
    plt.title(f"Out-of-Sample Test Transfers: Transfermarkt vs. {best_model_name} vs. Actual Fee")
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
