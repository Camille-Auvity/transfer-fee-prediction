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


def bias_before_after_table(
    err_a: pd.DataFrame, err_b: pd.DataFrame, group_col: str,
    label_a: str = "Avant", label_b: str = "Après",
) -> pd.DataFrame:
    """Biais moyen ET médian pour deux variantes d'un modèle (ex: avant/après
    calibration, ou XGBoost vs. NN), pour une dimension donnée. La moyenne est
    sensible aux gros % d'erreur sur les transferts bon marché (cf.
    error_density.png) ; la médiane est plus robuste et donne le verdict le
    plus fiable sur l'effet réel observé."""
    a = err_a.groupby(group_col, observed=False)["signed_pct_err"].agg(["mean", "median"])
    b = err_b.groupby(group_col, observed=False)["signed_pct_err"].agg(["mean", "median"])
    out = pd.concat(
        [a["mean"].rename(f"{label_a} - Moyenne"), a["median"].rename(f"{label_a} - Médiane"),
         b["mean"].rename(f"{label_b} - Moyenne"), b["median"].rename(f"{label_b} - Médiane")],
        axis=1,
    )
    return out.round(2)


def plot_bias_before_after(
    err_a: pd.DataFrame, err_b: pd.DataFrame, group_col: str, title: str, filename: str,
    label_a: str = "Avant calibration", label_b: str = "Après calibration",
):
    """Compare le biais directionnel entre deux variantes d'un modèle, pour une
    dimension donnée — moyenne (sensible aux outliers) et médiane (robuste)
    côte à côte."""
    def grouped(err, label, stat):
        g = err.groupby(group_col, observed=False)["signed_pct_err"].agg(stat).reset_index()
        g["Version"] = label
        return g

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, stat, stat_label in [(axes[0], "mean", "Moyenne"), (axes[1], "median", "Médiane")]:
        plot_df = pd.concat([
            grouped(err_a, label_a, stat),
            grouped(err_b, label_b, stat),
        ])
        sns.barplot(
            data=plot_df, x=group_col, y="signed_pct_err", hue="Version", ax=ax,
            palette={label_a: "#e67e22", label_b: "#2ecc71"},
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{stat_label} — {title}")
        ax.set_xlabel("")
        ax.set_ylabel(f"Erreur signée ({stat_label.lower()}, %)")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(f"{label_a} vs. {label_b} (+ = sur-estime, − = sous-estime)")
    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
