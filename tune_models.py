"""Script optionnel : tuning d'hyperparamètres (RandomizedSearchCV) pour les
meilleurs modèles d'arbres/boosting du projet (XGBoost, LightGBM,
HistGradientBoosting, RandomForest).

Ne modifie rien au pipeline principal (`main.py`) : il reconstruit les mêmes
données et le même split chronologique via `src/`, lance la recherche
d'hyperparamètres uniquement sur le train set, puis vérifie le gain sur le
jeu de test chronologique (out-of-sample) avant de l'afficher.

Usage:
    python tune_models.py [--n-iter 25]
"""

import argparse
import json

from src import data_loading, dataset, evaluation, features, models, tuning
from src.config import TABLES_DIR


def main(n_iter: int):
    print("=" * 70)
    print("1. Reconstruction des données et du split chronologique (identique à main.py)")
    print("=" * 70)
    transfer_full = data_loading.build_transfer_full()
    transfer_full = features.build_feature_table(transfer_full)
    final_data = dataset.build_final_data(transfer_full)
    split = dataset.build_model_split(final_data, transfer_full)

    print("\n" + "=" * 70)
    print("2. Baseline : entraînement des modèles avec les hyperparamètres actuels")
    print("=" * 70)
    baseline = models.train_all_models(split.X_train, split.y_train, split.X_test)
    baseline_results_df = evaluation.evaluate_regression_models(split.y_test, baseline.predictions)
    print(baseline_results_df.round(4))

    print("\n" + "=" * 70)
    print(f"3. RandomizedSearchCV ({n_iter} tirages, TimeSeriesSplit 5 folds, train set uniquement)")
    print("=" * 70)
    tuning_results = tuning.tune_all_models(
        split.X_train, split.y_train, split.X_test, split.y_test, n_iter=n_iter
    )

    print("\n" + "=" * 70)
    print("4. Comparaison baseline vs. tuné — vérification out-of-sample")
    print("=" * 70)
    comparison_df = tuning.build_comparison_table(tuning_results, baseline_results_df)
    print(comparison_df.round(4))
    comparison_df.round(4).to_csv(TABLES_DIR / "tuned_vs_baseline.csv", index=False)

    # Alerte explicite si le CV est optimiste par rapport au test set chronologique :
    # c'est exactement le genre de sur-ajustement qu'une vérification out-of-sample doit attraper.
    for _, row in comparison_df.iterrows():
        gap = row["CV vs Test MAE Gap"]
        if gap > 0.02:
            print(
                f"⚠️  {row['Model']} : l'écart entre MAE de cross-validation et MAE de test "
                f"({gap:+.4f}) suggère un possible sur-ajustement au CV — "
                "le résultat sur le test set chronologique prime toujours."
            )
        if not row["Improved vs Baseline"]:
            print(
                f"ℹ️  {row['Model']} : le tuning n'améliore pas la MAE de test par rapport "
                "aux hyperparamètres actuels du pipeline — garder les hyperparamètres actuels."
            )

    print("\n" + "=" * 70)
    print("5. Comparaison complète : baseline vs. tuné vs. benchmark Transfermarkt")
    print("=" * 70)
    baseline_name_map = {
        "XGBoost": "XGBoost (Pseudo-Huber)",
        "LightGBM": "LightGBM (Huber)",
        "HistGradientBoosting": "HistGradientBoosting (MAE)",
        "RandomForest": "RandomForest",
    }
    combined_predictions = {}
    for short_name, full_name in baseline_name_map.items():
        combined_predictions[full_name] = baseline.predictions[full_name]
        combined_predictions[f"{short_name} (tuned)"] = tuning_results[short_name].y_pred_test

    # evaluate_fee_level_metrics ajoute automatiquement la ligne "Transfermarkt Alone (spread=0)"
    fee_level_df = evaluation.evaluate_fee_level_metrics(split.y_test, split.X_test, combined_predictions)
    print("\n--- Erreur sur le montant final (€), avec benchmark Transfermarkt ---")
    print(fee_level_df.round(3))
    fee_level_df.round(3).to_csv(TABLES_DIR / "tuned_fee_level_comparison.csv", index=False)

    classification_df = evaluation.evaluate_classification_metrics(split.y_test, combined_predictions)
    print("\n--- Classification binaire (prime vs. décote), avec Spearman / ROC-AUC / F1 ---")
    print(classification_df.round(4))
    classification_df.round(4).to_csv(TABLES_DIR / "tuned_classification_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("6. Graphiques : avant/après tuning vs. benchmark, évolution des hyperparamètres XGBoost")
    print("=" * 70)
    tuning.plot_tuning_vs_benchmark(fee_level_df)

    best_params_per_model = {name: r.best_params for name, r in tuning_results.items()}
    with open(TABLES_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_params_per_model, f, indent=2, default=str)

    tuning.plot_xgboost_hyperparam_evolution(tuning_results["XGBoost"].best_params)

    print("\nMeilleurs hyperparamètres trouvés :")
    for model_name, result in tuning_results.items():
        print(f"\n{model_name}:")
        for param, value in result.best_params.items():
            print(f"  {param}: {value}")

    print("\nTerminé.")
    print("  Tableaux -> outputs/tables/tuned_*.csv, best_hyperparameters.json")
    print("  Graphiques -> outputs/figures/tuning_vs_benchmark.png, xgboost_hyperparam_evolution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-iter", type=int, default=25,
        help="Nombre de combinaisons d'hyperparamètres testées par modèle (défaut: 25)",
    )
    args = parser.parse_args()
    main(args.n_iter)
