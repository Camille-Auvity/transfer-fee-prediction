"""Calibration par régression isotonique des prédictions de XGBoost, pour
corriger le biais de sur-estimation systématique mis en évidence dans
`error_analysis.py` — sans juste décaler par une moyenne globale, mais en
apprenant une correction qui peut varier selon le niveau de la prédiction.

Méthodologie, pour rester cohérent avec le reste du projet (toujours vérifier
hors échantillon) :
1. Le train set (déjà le plus ancien chronologiquement) est lui-même redécoupé
   en deux : une portion "fit" (entraînement du modèle) et une portion
   "calib" (les transferts juste avant le test set), utilisée uniquement pour
   apprendre la correction isotonique.
2. Le test set final (le plus récent, jamais touché jusqu'ici) sert une seule
   fois à vérifier si la calibration apporte un vrai gain hors échantillon.

Module séparé : ne modifie rien au pipeline principal.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor

from .config import FIGURES_DIR
from .models import XGBOOST_PARAMS

sns.set_theme(style="whitegrid", context="notebook")

CALIB_FRACTION = 0.15  # part du train set (la plus récente) réservée à la calibration


@dataclass
class CalibrationSplit:
    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_calib: pd.DataFrame
    y_calib: pd.Series


def chronological_calib_split(X_train: pd.DataFrame, y_train: pd.Series, calib_fraction: float = CALIB_FRACTION) -> CalibrationSplit:
    """Redécoupe le train set chronologique : les (1 - calib_fraction) premiers
    transferts pour entraîner le modèle, les plus récents pour calibrer."""
    split_idx = int(len(X_train) * (1 - calib_fraction))
    return CalibrationSplit(
        X_fit=X_train.iloc[:split_idx].copy(),
        y_fit=y_train.iloc[:split_idx].copy(),
        X_calib=X_train.iloc[split_idx:].copy(),
        y_calib=y_train.iloc[split_idx:].copy(),
    )


def fit_model_and_calibrator(split: CalibrationSplit):
    """Entraîne XGBoost sur la portion 'fit', puis une régression isotonique
    (prédiction brute -> cible réelle) sur la portion 'calib'."""
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(split.X_fit, split.y_fit)

    raw_calib_preds = model.predict(split.X_calib)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calib_preds, split.y_calib)

    return model, calibrator


def predict_raw_and_calibrated(model: XGBRegressor, calibrator: IsotonicRegression, X: pd.DataFrame):
    raw_preds = model.predict(X)
    calibrated_preds = calibrator.predict(raw_preds)
    return raw_preds, calibrated_preds


def plot_isotonic_curve(calibrator: IsotonicRegression, raw_calib_preds: np.ndarray, filename: str = "isotonic_calibration_curve.png"):
    """Visualise la fonction de correction apprise : prédiction brute -> prédiction calibrée."""
    x_grid = np.linspace(raw_calib_preds.min(), raw_calib_preds.max(), 200)
    y_grid = calibrator.predict(x_grid)

    plt.figure(figsize=(8, 6))
    plt.plot([x_grid.min(), x_grid.max()], [x_grid.min(), x_grid.max()], "k--", lw=1, label="Pas de correction (y = x)")
    plt.plot(x_grid, y_grid, color="#2ecc71", lw=2.5, label="Correction isotonique apprise")
    plt.xlabel("Prédiction brute de XGBoost (log_spread)")
    plt.ylabel("Prédiction calibrée (log_spread)")
    plt.title("Fonction de calibration isotonique\n(appris sur la portion 'calib' du train set)")
    plt.legend()
    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_bias_before_after(err_raw: pd.DataFrame, err_calibrated: pd.DataFrame, group_col: str, title: str, filename: str):
    """Compare le biais directionnel moyen (%) avant/après calibration, pour une dimension donnée."""
    def grouped(err, label):
        g = err.groupby(group_col, observed=False)["signed_pct_err"].mean().reset_index()
        g["Version"] = label
        return g

    plot_df = pd.concat([grouped(err_raw, "Avant calibration"), grouped(err_calibrated, "Après calibration")])

    plt.figure(figsize=(9, 5.5))
    sns.barplot(
        data=plot_df, x=group_col, y="signed_pct_err", hue="Version",
        palette={"Avant calibration": "#e67e22", "Après calibration": "#2ecc71"},
    )
    plt.axhline(0, color="black", lw=0.8)
    plt.title(f"Biais (%) avant/après calibration — {title}\n(+ = sur-estime, − = sous-estime)")
    plt.xlabel("")
    plt.ylabel("Erreur signée moyenne (%)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
