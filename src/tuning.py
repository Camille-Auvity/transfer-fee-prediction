"""Recherche d'hyperparamètres (RandomizedSearchCV) pour les meilleurs modèles
d'arbres/boosting du projet : XGBoost, LightGBM, HistGradientBoosting et
RandomForest.

Ce module est volontairement séparé du pipeline principal (`main.py` /
`src/models.py`) : il ne modifie rien à la logique existante, il ajoute juste
une étape optionnelle de tuning. La recherche se fait uniquement sur le train
set (avec un `TimeSeriesSplit`, cohérent avec le split chronologique utilisé
partout ailleurs dans le projet) ; le jeu de test chronologique (les 20% de
transferts les plus récents) n'est utilisé qu'une seule fois, à la fin, pour
vérifier que le gain de performance tient vraiment "out of sample" et n'est
pas un artefact du cross-validation.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from .config import FIGURES_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid", context="notebook")

N_CV_SPLITS = 5
SCORING = "neg_mean_absolute_error"

MODEL_SHORT_NAMES = ["XGBoost", "LightGBM", "HistGradientBoosting", "RandomForest"]

# Noms des modèles baseline tels qu'ils apparaissent dans src/models.py / outputs existants
BASELINE_FULL_NAMES = {
    "XGBoost": "XGBoost (Pseudo-Huber)",
    "LightGBM": "LightGBM (Huber)",
    "HistGradientBoosting": "HistGradientBoosting (MAE)",
    "RandomForest": "RandomForest",
}

# Hyperparamètres actuels du pipeline principal pour XGBoost (src/models.py),
# utilisés uniquement pour visualiser l'évolution avant/après tuning.
BASELINE_XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
}


def _xgb_search_space():
    estimator = XGBRegressor(
        objective="reg:pseudohubererror",
        random_state=RANDOM_STATE,
        n_jobs=1,  # évite la contention de threads avec le parallélisme du RandomizedSearchCV
    )
    param_distributions = {
        "n_estimators": randint(200, 700),
        "max_depth": randint(3, 7),
        "learning_rate": loguniform(0.01, 0.12),
        "min_child_weight": randint(5, 40),
        "subsample": uniform(0.6, 0.4),       # [0.6, 1.0]
        "colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0]
        "reg_lambda": loguniform(0.5, 25),
    }
    return estimator, param_distributions, None  # pas de prefixe "model__" (pas de pipeline)


def _lightgbm_search_space():
    estimator = LGBMRegressor(
        objective="huber",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    param_distributions = {
        "n_estimators": randint(200, 700),
        "max_depth": randint(3, 7),
        "num_leaves": randint(7, 63),
        "learning_rate": loguniform(0.01, 0.12),
        "min_child_samples": randint(10, 60),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_lambda": loguniform(0.5, 25),
    }
    return estimator, param_distributions, None


def _hgb_search_space():
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        random_state=RANDOM_STATE,
    )
    param_distributions = {
        "max_depth": randint(3, 8),
        "max_iter": randint(150, 500),
        "learning_rate": loguniform(0.01, 0.12),
        "min_samples_leaf": randint(10, 60),
        "l2_regularization": loguniform(0.1, 25),
    }
    return estimator, param_distributions, None


def _random_forest_search_space():
    # RandomForest ne gère pas nativement les NaN -> même imputation médiane que dans src/models.py
    estimator = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
    ])
    param_distributions = {
        "model__n_estimators": randint(200, 700),
        "model__max_depth": randint(5, 16),
        "model__min_samples_leaf": randint(5, 40),
        "model__max_features": uniform(0.3, 0.7),  # [0.3, 1.0]
    }
    return estimator, param_distributions, "model__"


SEARCH_SPACES = {
    "XGBoost": _xgb_search_space,
    "LightGBM": _lightgbm_search_space,
    "HistGradientBoosting": _hgb_search_space,
    "RandomForest": _random_forest_search_space,
}


@dataclass
class TuningResult:
    model_name: str
    search: RandomizedSearchCV
    best_params: dict
    best_cv_mae: float  # MAE moyen en cross-validation (train set uniquement)
    test_mae: float
    test_rmse: float
    test_r2: float
    test_median_fee_pct_err: float
    y_pred_test: np.ndarray  # prédictions (log_spread) sur le jeu de test, pour comparaisons ultérieures


def tune_model(
    model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series, n_iter: int = 25,
) -> TuningResult:
    """Lance un RandomizedSearchCV (TimeSeriesSplit) pour un modèle, puis évalue
    le meilleur estimateur trouvé sur le jeu de test chronologique (out-of-sample)."""
    estimator, param_distributions, _ = SEARCH_SPACES[model_name]()
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=SCORING,
        cv=tscv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train, y_train)

    y_pred_test = search.predict(X_test)
    actual_ratio = np.exp(y_test)
    pred_ratio = np.exp(y_pred_test)
    median_fee_pct_err = np.median(np.abs(pred_ratio - actual_ratio) / actual_ratio) * 100

    return TuningResult(
        model_name=model_name,
        search=search,
        best_params=search.best_params_,
        best_cv_mae=-search.best_score_,  # neg_mean_absolute_error -> on repasse en positif
        test_mae=mean_absolute_error(y_test, y_pred_test),
        test_rmse=root_mean_squared_error(y_test, y_pred_test),
        test_r2=r2_score(y_test, y_pred_test),
        test_median_fee_pct_err=median_fee_pct_err,
        y_pred_test=y_pred_test,
    )


def tune_all_models(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
    n_iter: int = 25,
) -> dict:
    """Lance le tuning pour les 4 modèles et renvoie {nom: TuningResult}."""
    results = {}
    for model_name in SEARCH_SPACES:
        print(f"Tuning {model_name} ({n_iter} tirages, TimeSeriesSplit à {N_CV_SPLITS} folds)...")
        results[model_name] = tune_model(model_name, X_train, y_train, X_test, y_test, n_iter=n_iter)
        r = results[model_name]
        print(
            f"  -> best CV MAE = {r.best_cv_mae:.4f} | "
            f"test MAE = {r.test_mae:.4f} | test R2 = {r.test_r2:.4f}"
        )
    return results


def build_comparison_table(tuning_results: dict, baseline_results_df: pd.DataFrame) -> pd.DataFrame:
    """Compare, pour chaque modèle tuné, la performance de base (pipeline
    principal) à la performance après tuning — et calcule l'écart CV vs. test
    pour repérer un éventuel sur-ajustement au cross-validation."""
    rows = []
    baseline_by_name = baseline_results_df.set_index("Model")

    for model_name, r in tuning_results.items():
        baseline_row = None
        for baseline_name in baseline_by_name.index:
            if baseline_name.startswith(model_name):
                baseline_row = baseline_by_name.loc[baseline_name]
                break

        rows.append({
            "Model": model_name,
            "Baseline Test MAE": baseline_row["MAE (log_spread)"] if baseline_row is not None else np.nan,
            "Tuned CV MAE (train)": r.best_cv_mae,
            "Tuned Test MAE": r.test_mae,
            "Tuned Test RMSE": r.test_rmse,
            "Tuned Test R2": r.test_r2,
            "Tuned Median Fee % Error": r.test_median_fee_pct_err,
            "CV vs Test MAE Gap": r.test_mae - r.best_cv_mae,
            "Improved vs Baseline": (
                r.test_mae < baseline_row["MAE (log_spread)"] if baseline_row is not None else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values("Tuned Test MAE").reset_index(drop=True)


def plot_tuning_vs_benchmark(
    fee_level_df: pd.DataFrame, filename: str = "tuning_vs_benchmark.png",
):
    """Compare, pour chaque modèle, la performance avant/après tuning au
    benchmark Transfermarkt (ligne pointillée) — sur l'erreur médiane (%) et
    le R² (log fee), les deux métriques out-of-sample les plus parlantes."""
    tm_row = fee_level_df[fee_level_df["Model"] == "Transfermarkt Alone (spread=0)"].iloc[0]

    rows = []
    for short_name in MODEL_SHORT_NAMES:
        baseline_row = fee_level_df[fee_level_df["Model"] == BASELINE_FULL_NAMES[short_name]].iloc[0]
        tuned_row = fee_level_df[fee_level_df["Model"] == f"{short_name} (tuned)"].iloc[0]
        rows.append({"Model": short_name, "Version": "Avant tuning", "Median % Err": baseline_row["Median % Err"], "R2 (log_fee)": baseline_row["R2 (log_fee)"]})
        rows.append({"Model": short_name, "Version": "Après tuning", "Median % Err": tuned_row["Median % Err"], "R2 (log_fee)": tuned_row["R2 (log_fee)"]})

    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.barplot(
        data=plot_df, x="Model", y="Median % Err", hue="Version",
        palette={"Avant tuning": "#95a5a6", "Après tuning": "#2ecc71"}, ax=axes[0],
    )
    axes[0].axhline(tm_row["Median % Err"], color="#c0392b", ls="--", lw=1.5, label="Transfermarkt seul (benchmark)")
    axes[0].set_title("Erreur médiane (%) sur le montant du transfert\n(plus bas = meilleur)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Erreur médiane (%)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(
        data=plot_df, x="Model", y="R2 (log_fee)", hue="Version",
        palette={"Avant tuning": "#95a5a6", "Après tuning": "#2ecc71"}, ax=axes[1],
    )
    axes[1].axhline(tm_row["R2 (log_fee)"], color="#c0392b", ls="--", lw=1.5, label="Transfermarkt seul (benchmark)")
    axes[1].set_title("R² sur log(Montant du transfert)\n(plus haut = meilleur)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R² (log fee)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_xgboost_hyperparam_evolution(
    tuned_params: dict, filename: str = "xgboost_hyperparam_evolution.png",
):
    """Visualise, pour XGBoost, l'évolution de chaque hyperparamètre entre les
    valeurs actuelles du pipeline (src/models.py) et celles trouvées par le
    tuning, en variation relative (%) — avec les valeurs avant/après en annotation."""
    params = list(BASELINE_XGB_PARAMS.keys())
    baseline_vals = [BASELINE_XGB_PARAMS[p] for p in params]
    tuned_vals = [tuned_params[p] for p in params]
    pct_change = [(t - b) / b * 100 for b, t in zip(baseline_vals, tuned_vals)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in pct_change]
    bars = ax.barh(params, pct_change, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Variation après tuning (%)")
    ax.set_title("Évolution des hyperparamètres XGBoost (pipeline actuel → tuné)")

    x_span = max(abs(min(pct_change, default=0)), abs(max(pct_change, default=0)), 10)
    ax.set_xlim(-x_span * 1.35, x_span * 1.35)

    for bar, b, t in zip(bars, baseline_vals, tuned_vals):
        width = bar.get_width()
        label = f"{b:g} → {t:.3g}"
        offset = x_span * 0.03
        if width >= 0:
            ax.text(width + offset, bar.get_y() + bar.get_height() / 2, label, va="center", ha="left", fontsize=9)
        else:
            ax.text(width - offset, bar.get_y() + bar.get_height() / 2, label, va="center", ha="right", fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
