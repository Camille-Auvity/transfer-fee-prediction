"""Entraînement des modèles (cellules 16, 17 et 18 du notebook) :
régressions linéaires régularisées, puis modèles d'arbres/boosting, puis un
blend Ridge + meilleur GBDT."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import RANDOM_STATE

# Drop 1 reference dummy per one-hot group + curr_val_vs_fee_paid (near-identity) for linear models
LINEAR_DROP_COLS = [
    "curr_val_vs_fee_paid", "position_Goalkeeper",
    "domestic_competition_id_to_Other", "domestic_competition_id_from_Other",
]
ZERO_IMPUTE_COLS = ["prev_log_spread", "value_momentum"]


@dataclass
class TrainedModels:
    predictions: dict = field(default_factory=dict)
    fitted_models: dict = field(default_factory=dict)
    linear_models: dict = field(default_factory=dict)


def train_linear_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> TrainedModels:
    linear_cols = [c for c in X_train.columns if c not in LINEAR_DROP_COLS]

    # Zero-impute log-ratio history features; median-impute everything else
    zero_impute_cols = [c for c in ZERO_IMPUTE_COLS if c in linear_cols]
    median_impute_cols = [c for c in linear_cols if c not in zero_impute_cols]

    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("zero_imp", SimpleImputer(strategy="constant", fill_value=0.0), zero_impute_cols),
            ("med_imp", SimpleImputer(strategy="median"), median_impute_cols),
        ]
    )

    tscv = TimeSeriesSplit(n_splits=5)

    linear_models = {
        "Dummy (Median)": DummyRegressor(strategy="median"),
        "RidgeCV": RidgeCV(alphas=np.logspace(-2, 4, 25), cv=tscv),
        "ElasticNetCV": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, 1, 20), cv=tscv,
            max_iter=5000, random_state=RANDOM_STATE,
        ),
        "HuberRegressor": HuberRegressor(max_iter=500, alpha=1.0),
    }

    result = TrainedModels(linear_models=linear_models)

    for name, estimator in linear_models.items():
        pipe = Pipeline([
            ("prep", linear_preprocessor),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipe.fit(X_train[linear_cols], y_train)
        result.predictions[name] = pipe.predict(X_test[linear_cols])
        result.fitted_models[name] = pipe
        print(f"Trained {name}")

    return result


def train_tree_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, result: TrainedModels) -> None:
    """Entraîne les modèles d'arbres/boosting et ajoute leurs prédictions à `result` (in place)."""
    # 1. Scikit-learn tree models (always available)
    # HistGradientBoosting handles NaNs natively
    # Hyperparamètres tunés via tune_models.py (RandomizedSearchCV, TimeSeriesSplit,
    # gain confirmé out-of-sample sur le jeu de test chronologique) — voir
    # outputs/tables/best_hyperparameters.json et tuned_vs_baseline.csv
    hgb = HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=6, learning_rate=0.03683895479362275, max_iter=494,
        min_samples_leaf=58, l2_regularization=0.5364813719603044, random_state=RANDOM_STATE,
    )
    hgb.fit(X_train, y_train)
    result.predictions["HistGradientBoosting (MAE)"] = hgb.predict(X_test)
    result.fitted_models["HistGradientBoosting (MAE)"] = hgb
    print("Trained HistGradientBoosting (MAE)")

    # Random Forest (median imputation inside pipeline) — hyperparamètres tunés (voir ci-dessus)
    rf_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=539, max_depth=14, min_samples_leaf=6, max_features=0.5132296384213595,
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    result.predictions["RandomForest"] = rf_pipe.predict(X_test)
    result.fitted_models["RandomForest"] = rf_pipe
    print("Trained RandomForest")

    # 2. LightGBM (handles NaNs natively)
    lgbm = LGBMRegressor(
        objective="huber", n_estimators=400, learning_rate=0.03, max_depth=4,
        num_leaves=15, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, random_state=RANDOM_STATE, verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    result.predictions["LightGBM (Huber)"] = lgbm.predict(X_test)
    result.fitted_models["LightGBM (Huber)"] = lgbm
    print("Trained LightGBM (Huber)")

    # 3. XGBoost (handles NaNs natively) — hyperparamètres tunés (voir ci-dessus)
    xgb = XGBRegressor(
        objective="reg:pseudohubererror", n_estimators=684, learning_rate=0.016424199098700288, max_depth=5,
        min_child_weight=7, subsample=0.6682096494749166, colsample_bytree=0.9140703845572055,
        reg_lambda=5.3848112391315635, random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    result.predictions["XGBoost (Pseudo-Huber)"] = xgb.predict(X_test)
    result.fitted_models["XGBoost (Pseudo-Huber)"] = xgb
    print("Trained XGBoost (Pseudo-Huber)")

    # 4. CatBoost (handles NaNs natively)
    cb = CatBoostRegressor(
        loss_function="Huber:delta=1.0", iterations=500, learning_rate=0.04, depth=5,
        l2_leaf_reg=5.0, random_seed=RANDOM_STATE, verbose=0,
    )
    cb.fit(X_train, y_train)
    result.predictions["CatBoost (Huber)"] = cb.predict(X_test)
    result.fitted_models["CatBoost (Huber)"] = cb
    print("Trained CatBoost (Huber)")


def add_blend(result: TrainedModels) -> None:
    """Ajoute un blend 30% RidgeCV + 70% meilleur GBDT disponible (cellule 18)."""
    best_gbdt_name = next(
        (m for m in ["LightGBM (Huber)", "CatBoost (Huber)", "XGBoost (Pseudo-Huber)", "HistGradientBoosting (MAE)"]
         if m in result.predictions),
        None,
    )
    if best_gbdt_name:
        result.predictions[f"Blend (30% Ridge + 70% {best_gbdt_name.split()[0]})"] = (
            0.3 * result.predictions["RidgeCV"] + 0.7 * result.predictions[best_gbdt_name]
        )


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> TrainedModels:
    result = train_linear_models(X_train, y_train, X_test)
    train_tree_models(X_train, y_train, X_test, result)
    add_blend(result)
    return result
