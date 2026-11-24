"""Évaluation des modèles de régression (cellules 18 à 24 du notebook) :
comparaison des erreurs, importances de variables, métriques de
classification binaire (prime vs décote) et erreur sur le montant final
du transfert, par tranche de valorisation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from .config import FIGURES_DIR

sns.set_theme(style="whitegrid", context="notebook")


def evaluate_regression_models(y_test: pd.Series, predictions: dict) -> pd.DataFrame:
    """Tableau MAE / RMSE / R2 / erreur médiane en % sur le montant du transfert (cellule 18)."""
    rows = []
    actual_ratio = np.exp(y_test)

    for name, y_pred in predictions.items():
        pred_ratio = np.exp(y_pred)
        median_pct_err = np.median(np.abs(pred_ratio - actual_ratio) / actual_ratio) * 100

        rows.append({
            "Model": name,
            "MAE (log_spread)": mean_absolute_error(y_test, y_pred),
            "RMSE (log_spread)": root_mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "Median Fee % Error": median_pct_err,
        })

    return pd.DataFrame(rows).sort_values("MAE (log_spread)").reset_index(drop=True)


def plot_mae_comparison(results_df: pd.DataFrame, filename: str = "mae_comparison.png"):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="MAE (log_spread)", y="Model", hue="Model", palette="viridis", legend=False)
    plt.title("Chronological Test Set Comparison — MAE on log_spread (Lower is Better)")
    plt.xlabel("MAE (log_spread)")
    plt.ylabel("")
    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_feature_importance(
    results_df: pd.DataFrame, fitted_models: dict, linear_models: dict,
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series,
    filename: str = "feature_importance.png",
):
    """Importances de variables du meilleur modèle d'arbres (cellule 19)."""
    tree_candidates = [m for m in results_df["Model"] if m in fitted_models and m not in linear_models]
    best_tree_name = tree_candidates[0]
    best_tree = fitted_models[best_tree_name]

    if best_tree_name == "RandomForest":
        importances = best_tree.named_steps["model"].feature_importances_
    elif hasattr(best_tree, "feature_importances_"):
        importances = best_tree.feature_importances_
    else:
        # Fallback to permutation importance for HistGradientBoosting
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            best_tree, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=5, random_state=42
        )
        importances = perm.importances_mean

    feat_imp = (
        pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(15)
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette="mako", legend=False)
    plt.title(f"Top 15 Feature Importances — {best_tree_name}")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return best_tree_name, feat_imp, out_path


def print_tm_vs_model_r2(y_test: pd.Series, X_test: pd.DataFrame, predictions: dict):
    """Compare le R2 de Transfermarkt seul vs. Transfermarkt + XGBoost (cellule 20)."""
    actual_log_fee = y_test + X_test["log_curr_market_value"]
    pred_log_fee = predictions["XGBoost (Pseudo-Huber)"] + X_test["log_curr_market_value"]
    print("TM alone R2:", round(r2_score(actual_log_fee, X_test["log_curr_market_value"]), 4))
    print("TM + XGBoost R2:", round(r2_score(actual_log_fee, pred_log_fee), 4))


def evaluate_classification_metrics(y_test: pd.Series, predictions: dict) -> pd.DataFrame:
    """Métriques de classification binaire : fee payé > valeur Transfermarkt ? (cellule 21)."""
    y_true_dir = (y_test.values > 0).astype(int)

    spread_rows = []
    for name, y_pred in predictions.items():
        y_pred_dir = (y_pred > 0).astype(int)
        spearman_r = spearmanr(y_test.values, y_pred).statistic

        spread_rows.append({
            "Model": name,
            "R2 (log_spread)": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "Spearman r": spearman_r,
            "Dir. Accuracy": accuracy_score(y_true_dir, y_pred_dir),
            "ROC-AUC": roc_auc_score(y_true_dir, y_pred),
            "Precision": precision_score(y_true_dir, y_pred_dir, zero_division=0),
            "Recall": recall_score(y_true_dir, y_pred_dir, zero_division=0),
            "F1": f1_score(y_true_dir, y_pred_dir, zero_division=0),
        })

    print(f"Base rate (actual Fee > TM Value): {y_true_dir.mean():.1%}")
    return pd.DataFrame(spread_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def evaluate_fee_level_metrics(y_test: pd.Series, X_test: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """Erreur sur le montant réel du transfert, en euros (cellule 22)."""
    tm_val = np.exp(X_test["log_curr_market_value"].values)
    actual_log_fee = y_test.values + X_test["log_curr_market_value"].values
    actual_fee = np.exp(actual_log_fee)

    fee_rows = []
    all_eval = {"Transfermarkt Alone (spread=0)": np.zeros(len(y_test)), **predictions}

    for name, y_pred in all_eval.items():
        pred_log_fee = y_pred + X_test["log_curr_market_value"].values
        pred_fee = np.exp(pred_log_fee)

        pct_err = np.abs(pred_fee - actual_fee) / actual_fee

        fee_rows.append({
            "Model": name,
            "R2 (log_fee)": r2_score(actual_log_fee, pred_log_fee),
            "R2 (EUR fee)": r2_score(actual_fee, pred_fee),
            "MAE (EUR M)": mean_absolute_error(actual_fee, pred_fee) / 1e6,
            "RMSE (EUR M)": root_mean_squared_error(actual_fee, pred_fee) / 1e6,
            "Median % Err": np.median(pct_err) * 100,
            "Within ±20%": (pct_err <= 0.20).mean() * 100,
            "Within ±30%": (pct_err <= 0.30).mean() * 100,
        })

    return pd.DataFrame(fee_rows).sort_values("R2 (log_fee)", ascending=False).reset_index(drop=True)


def plot_diagnostics(
    y_test: pd.Series, X_test: pd.DataFrame, predictions: dict, results_df: pd.DataFrame,
    filename: str = "diagnostics_panel.png",
):
    """Panel 2x2 : ROC, matrice de confusion, actual vs pred (log_spread et log_fee) (cellule 23)."""
    y_true_dir = (y_test.values > 0).astype(int)
    best_model_name = results_df.iloc[0]["Model"]
    best_pred_spread = predictions[best_model_name]
    best_pred_log_fee = best_pred_spread + X_test["log_curr_market_value"].values
    actual_log_fee = y_test.values + X_test["log_curr_market_value"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. ROC Curves (Premium vs. Discount: Fee > TM Value)
    ax = axes[0, 0]
    models_to_plot = [m for m in [best_model_name, "LightGBM (Huber)", "RidgeCV"] if m in predictions]
    for m in dict.fromkeys(models_to_plot):  # preserve order, remove duplicates
        RocCurveDisplay.from_predictions(y_true_dir, predictions[m], name=m, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance (AUC = 0.50)")
    ax.set_title("ROC Curve — Predicting Premium (Fee > TM Value)")
    ax.legend(loc="lower right")

    # 2. Confusion Matrix for Best Model
    ax = axes[0, 1]
    ConfusionMatrixDisplay.from_predictions(
        y_true_dir,
        (best_pred_spread > 0).astype(int),
        display_labels=["Discount / Par\n(Fee ≤ TM)", "Premium\n(Fee > TM)"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix — {best_model_name}")

    # 3. Actual vs. Predicted log_spread
    ax = axes[1, 0]
    ax.scatter(y_test, best_pred_spread, alpha=0.25, s=15, color="steelblue")
    lims = [min(y_test.min(), best_pred_spread.min()), max(y_test.max(), best_pred_spread.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect Prediction")
    ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
    ax.set_title(f"Actual vs. Predicted log_spread ({best_model_name})")
    ax.set_xlabel("Actual log_spread")
    ax.set_ylabel("Predicted log_spread")
    ax.legend()

    # 4. Actual vs. Predicted Final Fee (Log Scale)
    ax = axes[1, 1]
    ax.scatter(actual_log_fee, best_pred_log_fee, alpha=0.25, s=15, color="darkorange")
    fee_lims = [min(actual_log_fee.min(), best_pred_log_fee.min()), max(actual_log_fee.max(), best_pred_log_fee.max())]
    ax.plot(fee_lims, fee_lims, "r--", lw=1.5, label="Perfect Prediction")
    ax.set_title(f"Actual vs. Predicted log(Final Fee) ({best_model_name})")
    ax.set_xlabel("Actual log(Fee)")
    ax.set_ylabel("Predicted log(Fee)")
    ax.legend()

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return best_model_name, out_path


def tier_error_table_and_plot(
    X_test: pd.DataFrame, y_test: pd.Series, predictions: dict, results_df: pd.DataFrame,
    filename: str = "fee_error_by_valuation_tier.png",
):
    """Erreur médiane (%) par tranche de valeur Transfermarkt : TM seul vs. meilleur modèle (cellule 24)."""
    best_model_name = results_df.iloc[0]["Model"]
    best_pred_spread = predictions[best_model_name]
    best_pred_log_fee = best_pred_spread + X_test["log_curr_market_value"].values

    tm_val = np.exp(X_test["log_curr_market_value"].values)
    actual_fee = np.exp(y_test.values + X_test["log_curr_market_value"].values)

    tier_df = pd.DataFrame({
        "tm_val": tm_val,
        "actual_fee": actual_fee,
        "pred_fee": np.exp(best_pred_log_fee),
    })

    tier_df["tm_pct_err"] = np.abs(tier_df["tm_val"] - tier_df["actual_fee"]) / tier_df["actual_fee"] * 100
    tier_df["model_pct_err"] = np.abs(tier_df["pred_fee"] - tier_df["actual_fee"]) / tier_df["actual_fee"] * 100

    bins = [0, 1e6, 5e6, 15e6, np.inf]
    labels = ["< €1M", "€1M – €5M", "€5M – €15M", "> €15M"]
    tier_df["tier"] = pd.cut(tier_df["tm_val"], bins=bins, labels=labels)

    summary = tier_df.groupby("tier", observed=False).agg(
        Count=("actual_fee", "count"),
        TM_Median_Pct_Err=("tm_pct_err", "median"),
        Model_Median_Pct_Err=("model_pct_err", "median"),
        TM_Within_30pct=("tm_pct_err", lambda s: (s <= 30).mean() * 100),
        Model_Within_30pct=("model_pct_err", lambda s: (s <= 30).mean() * 100),
    )

    # Side-by-side bar chart of Median % Error by Market Value Tier
    plot_df = summary[["TM_Median_Pct_Err", "Model_Median_Pct_Err"]].reset_index().melt(
        id_vars="tier", var_name="Method", value_name="Median Fee % Error"
    )
    plot_df["Method"] = plot_df["Method"].replace({
        "TM_Median_Pct_Err": "Transfermarkt Alone",
        "Model_Median_Pct_Err": f"TM + {best_model_name}",
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="tier", y="Median Fee % Error", hue="Method", palette="Set2")
    plt.title("Median Transfer Fee % Error by Player Valuation Tier")
    plt.xlabel("Transfermarkt Market Value Tier")
    plt.ylabel("Median Absolute % Error (%)")
    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return summary, out_path
