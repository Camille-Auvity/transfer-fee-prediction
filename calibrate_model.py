"""Script optionnel : calibration par régression isotonique des prédictions
de XGBoost, pour corriger le biais de sur-estimation systématique mis en
évidence par `error_analysis.py`.

Le modèle est ré-entraîné sur une portion du train set uniquement (les
transferts les plus récents du train set servent à apprendre la correction,
pas le test set final) ; le test set chronologique n'est utilisé qu'une
seule fois, à la fin, pour vérifier que la calibration aide vraiment hors
échantillon. Ne modifie rien au pipeline principal.

Usage:
    python calibrate_model.py
"""

import pandas as pd

from src import calibration, data_loading, dataset, error_analysis, evaluation, features
from src.config import TABLES_DIR


def main():
    print("=" * 70)
    print("1. Reconstruction des données et du split chronologique")
    print("=" * 70)
    transfer_full_raw = data_loading.build_transfer_full()
    transfer_full = features.build_feature_table(transfer_full_raw.copy())
    final_data = dataset.build_final_data(transfer_full)
    split = dataset.build_model_split(final_data, transfer_full)

    print("\n" + "=" * 70)
    print("2. Split fit / calibration à l'intérieur du train set (chronologique)")
    print("=" * 70)
    calib_split = calibration.chronological_calib_split(split.X_train, split.y_train)
    print(f"Fit: {len(calib_split.X_fit):,} transferts | Calibration: {len(calib_split.X_calib):,} transferts")

    print("\n" + "=" * 70)
    print("3. Entraînement de XGBoost (fit) + régression isotonique (calib)")
    print("=" * 70)
    model, calibrator = calibration.fit_model_and_calibrator(calib_split)

    print("\n" + "=" * 70)
    print("4. Évaluation sur le jeu de test final (jamais touché jusqu'ici)")
    print("=" * 70)
    raw_preds, calibrated_preds = calibration.predict_raw_and_calibrated(model, calibrator, split.X_test)

    predictions = {
        "XGBoost (raw)": raw_preds,
        "XGBoost (calibrated)": calibrated_preds,
    }
    fee_level_df = evaluation.evaluate_fee_level_metrics(split.y_test, split.X_test, predictions)
    print("\n--- Erreur sur le montant final (€), avec benchmark Transfermarkt ---")
    print(fee_level_df.round(3))
    fee_level_df.round(3).to_csv(TABLES_DIR / "calibration_fee_level_comparison.csv", index=False)

    classification_df = evaluation.evaluate_classification_metrics(split.y_test, predictions)
    print("\n--- Classification binaire (prime vs. décote) ---")
    print(classification_df.round(4))
    classification_df.round(4).to_csv(TABLES_DIR / "calibration_classification_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("5. Biais par segment (La Liga, gardiens, jeunes...) : avant vs. après")
    print("=" * 70)
    err_raw = error_analysis.build_error_frame(transfer_full_raw, split.df_model, split.X_test, split.y_test, raw_preds)
    err_calibrated = error_analysis.build_error_frame(transfer_full_raw, split.df_model, split.X_test, split.y_test, calibrated_preds)

    for group_col in ["position", "age_bucket", "league_to"]:
        raw_bias = err_raw.groupby(group_col, observed=False)["signed_pct_err"].mean()
        calib_bias = err_calibrated.groupby(group_col, observed=False)["signed_pct_err"].mean()
        comparison = pd.concat(
            [raw_bias.rename("Biais avant (%)"), calib_bias.rename("Biais après (%)")], axis=1
        ).round(2)
        print(f"\n--- Biais par {group_col} ---")
        print(comparison)
        comparison.to_csv(TABLES_DIR / f"calibration_bias_by_{group_col}.csv")

    print("\n" + "=" * 70)
    print("6. Graphiques")
    print("=" * 70)
    raw_calib_preds = model.predict(calib_split.X_calib)
    calibration.plot_isotonic_curve(calibrator, raw_calib_preds)
    calibration.plot_bias_before_after(err_raw, err_calibrated, "league_to", "Championnat du club acheteur", "calibration_bias_by_league.png")
    calibration.plot_bias_before_after(err_raw, err_calibrated, "position", "Position", "calibration_bias_by_position.png")

    print("\nTerminé.")
    print("  Tableaux -> outputs/tables/calibration_*.csv")
    print("  Graphiques -> outputs/figures/isotonic_calibration_curve.png, calibration_bias_by_*.png")


if __name__ == "__main__":
    main()
