"""Modèles XGBoost spécifiques par tranche de valorisation, comparés à un
modèle global (cellules 26 et 27 du notebook)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from xgboost import XGBRegressor

from .config import FIGURES_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid", context="notebook")

TIER_BINS = [0, 2e6, 10e6, 25e6, np.inf]
TIER_LABELS = ["Small (< €2M)", "Mid (€2M – €10M)", "Starter (€10M – €25M)", "Star (> €25M)"]


def make_xgb(n_rows: int) -> XGBRegressor:
    """Slightly shallower/more regularized trees when a tier has fewer rows."""
    return XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=300 if n_rows < 1500 else 400,
        learning_rate=0.03,
        max_depth=3 if n_rows < 1500 else 4,
        min_child_weight=10 if n_rows < 1500 else 20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def run_tier_comparison(X: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame,
                         y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Entraîne un XGBoost par tranche de valorisation et le compare au modèle global."""
    tm_all = np.exp(X["log_curr_market_value"])
    tiers = pd.cut(tm_all, bins=TIER_BINS, labels=TIER_LABELS)

    # Fit a clean Global XGBoost on the full 80/20 split for apples-to-apples comparison
    global_xgb = make_xgb(len(X_train))
    global_xgb.fit(X_train, y_train)

    tier_results = []

    for tier_name in TIER_LABELS:
        # Restrict to this tier's rows inside the SAME global train/test split (zero leakage)
        tr_mask = tiers.loc[X_train.index] == tier_name
        te_mask = tiers.loc[X_test.index] == tier_name

        X_tr_t, y_tr_t = X_train[tr_mask], y_train[tr_mask]
        X_te_t, y_te_t = X_test[te_mask], y_test[te_mask]

        # Train Tier-Specific XGBoost
        tier_xgb = make_xgb(len(X_tr_t))
        tier_xgb.fit(X_tr_t, y_tr_t)

        pred_tier = tier_xgb.predict(X_te_t)
        pred_global = global_xgb.predict(X_te_t)

        # Convert to actual and predicted fees (€)
        tm_te = np.exp(X_te_t["log_curr_market_value"].values)
        actual_fee = np.exp(X_te_t["log_curr_market_value"].values + y_te_t.values)
        fee_tier = np.exp(X_te_t["log_curr_market_value"].values + pred_tier)
        fee_global = np.exp(X_te_t["log_curr_market_value"].values + pred_global)

        tm_med_err = np.median(np.abs(tm_te - actual_fee) / actual_fee) * 100
        tier_med_err = np.median(np.abs(fee_tier - actual_fee) / actual_fee) * 100
        glob_med_err = np.median(np.abs(fee_global - actual_fee) / actual_fee) * 100

        y_dir = (y_te_t.values > 0).astype(int)

        tier_results.append({
            "Tier": tier_name,
            "N_Train": len(X_tr_t),
            "N_Test": len(X_te_t),
            "Target Std": y_te_t.std(),
            "TM Med % Err": tm_med_err,
            "Tier-XGB R2": r2_score(y_te_t, pred_tier),
            "Tier-XGB MAE": mean_absolute_error(y_te_t, pred_tier),
            "Tier-XGB AUC": roc_auc_score(y_dir, pred_tier),
            "Tier-XGB Med % Err": tier_med_err,
            "Global-XGB R2": r2_score(y_te_t, pred_global),
            "Global-XGB MAE": mean_absolute_error(y_te_t, pred_global),
            "Global-XGB Med % Err": glob_med_err,
            "Lift vs TM (pp)": tm_med_err - glob_med_err,
        })

    return pd.DataFrame(tier_results)


def plot_tier_comparison(tier_comp_df: pd.DataFrame, filename: str = "tier_vs_global_xgb.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. R2 on log_spread by Tier (Tier-Specific vs. Global)
    r2_plot = tier_comp_df[["Tier", "Tier-XGB R2", "Global-XGB R2"]].melt(
        id_vars="Tier", var_name="Model", value_name="R2"
    )
    sns.barplot(data=r2_plot, x="Tier", y="R2", hue="Model", ax=axes[0], palette="viridis")
    axes[0].set_title("R² on log_spread by Valuation Tier")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_xlabel("")

    # 2. MAE on log_spread by Tier
    mae_plot = tier_comp_df[["Tier", "Tier-XGB MAE", "Global-XGB MAE"]].melt(
        id_vars="Tier", var_name="Model", value_name="MAE"
    )
    sns.barplot(data=mae_plot, x="Tier", y="MAE", hue="Model", ax=axes[1], palette="mako")
    axes[1].set_title("MAE (log_spread) by Valuation Tier (Lower = Better)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_xlabel("")

    # 3. Median Fee % Error: Transfermarkt Alone vs. Tier-XGB vs. Global-XGB
    err_plot = tier_comp_df[["Tier", "TM Med % Err", "Tier-XGB Med % Err", "Global-XGB Med % Err"]].melt(
        id_vars="Tier", var_name="Method", value_name="Median Fee % Error"
    )
    sns.barplot(data=err_plot, x="Tier", y="Median Fee % Error", hue="Method", ax=axes[2], palette="Set2")
    axes[2].set_title("Median Fee % Error vs. Transfermarkt")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set_xlabel("")

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
