"""Script optionnel : teste un réseau de neurones avec embeddings de club
(from_club_id / to_club_id) comme alternative à XGBoost, et le compare au
benchmark Transfermarkt ainsi qu'au meilleur modèle du pipeline principal.

Entièrement séparé du pipeline principal et des autres scripts optionnels
(`tune_models.py`, `error_analysis.py`, `calibrate_model.py`) : rien n'est
modifié ailleurs. Le modèle est entraîné sur une portion du train set
uniquement (le reste sert à l'arrêt anticipé) ; le jeu de test chronologique
n'est utilisé qu'une seule fois, à la fin, pour une comparaison hors
échantillon honnête.

Usage:
    python club_embedding_nn.py
"""

from src import calibration, data_loading, dataset, error_analysis, evaluation, features, models, neural_net
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
    print("2. Split fit / validation à l'intérieur du train set (chronologique)")
    print("=" * 70)
    nn_split = calibration.chronological_calib_split(split.X_train, split.y_train, calib_fraction=0.15)
    print(f"Fit: {len(nn_split.X_fit):,} transferts | Validation: {len(nn_split.X_calib):,} transferts")
    n_clubs_fit = neural_net.get_club_ids(transfer_full_raw, split.df_model, nn_split.X_fit)
    print(f"Clubs distincts vus en fit: {len(set(n_clubs_fit['from_club_id']) | set(n_clubs_fit['to_club_id']))}")

    print("\n" + "=" * 70)
    print("3. Entraînement du réseau de neurones (embeddings de club)")
    print("=" * 70)
    artifacts = neural_net.train_club_embedding_model(
        transfer_full_raw, split.df_model, nn_split.X_fit, nn_split.y_fit, nn_split.X_calib, nn_split.y_calib,
    )

    print("\n" + "=" * 70)
    print("4. Baseline : XGBoost (hyperparamètres tunés, entraîné sur 100% du train)")
    print("=" * 70)
    baseline = models.train_all_models(split.X_train, split.y_train, split.X_test)

    print("\n" + "=" * 70)
    print("5. Évaluation sur le jeu de test final (jamais touché jusqu'ici)")
    print("=" * 70)
    nn_preds = neural_net.predict(artifacts, transfer_full_raw, split.df_model, split.X_test)

    predictions = {
        "XGBoost (Pseudo-Huber)": baseline.predictions["XGBoost (Pseudo-Huber)"],
        "NN (embeddings club)": nn_preds,
    }
    fee_level_df = evaluation.evaluate_fee_level_metrics(split.y_test, split.X_test, predictions)
    print("\n--- Erreur sur le montant final (€), avec benchmark Transfermarkt ---")
    print(fee_level_df.round(3))
    fee_level_df.round(3).to_csv(TABLES_DIR / "nn_fee_level_comparison.csv", index=False)

    classification_df = evaluation.evaluate_classification_metrics(split.y_test, predictions)
    print("\n--- Classification binaire (prime vs. décote) ---")
    print(classification_df.round(4))
    classification_df.round(4).to_csv(TABLES_DIR / "nn_classification_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("6. Biais par segment : XGBoost vs. NN (moyenne vs. médiane)")
    print("=" * 70)
    err_xgb = error_analysis.build_error_frame(transfer_full_raw, split.df_model, split.X_test, split.y_test, baseline.predictions["XGBoost (Pseudo-Huber)"])
    err_nn = error_analysis.build_error_frame(transfer_full_raw, split.df_model, split.X_test, split.y_test, nn_preds)

    for group_col in ["position", "league_to"]:
        comparison = calibration.bias_before_after_table(err_xgb, err_nn, group_col, label_a="XGBoost", label_b="NN")
        print(f"\n--- Biais par {group_col} ---")
        print(comparison)
        comparison.to_csv(TABLES_DIR / f"nn_bias_by_{group_col}.csv")

    print("\nTerminé. Tableaux -> outputs/tables/nn_*.csv")


if __name__ == "__main__":
    main()
