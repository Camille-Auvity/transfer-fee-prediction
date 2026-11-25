"""Script optionnel : analyse d'erreurs du meilleur modèle (XGBoost) au-delà de
la segmentation par tranche de valorisation déjà faite dans `main.py`.

Segmente l'erreur sur le jeu de test chronologique par position, âge au
transfert, championnat du club acheteur et saison du transfert, et regarde le
biais directionnel (sur- ou sous-estimation systématique) en plus de l'ampleur
de l'erreur. Ne modifie rien au pipeline principal.

Usage:
    python error_analysis.py
"""

from src import data_loading, dataset, error_analysis, features, models
from src.config import TABLES_DIR


def main():
    print("=" * 70)
    print("1. Reconstruction des données (brutes + encodées) et du split chronologique")
    print("=" * 70)
    # Version brute (avant one-hot encoding) : on a besoin des colonnes catégorielles
    # d'origine (position, championnat...) pour segmenter l'erreur.
    transfer_full_raw = data_loading.build_transfer_full()
    transfer_full = features.build_feature_table(transfer_full_raw.copy())

    final_data = dataset.build_final_data(transfer_full)
    split = dataset.build_model_split(final_data, transfer_full)

    print("\n" + "=" * 70)
    print("2. Entraînement des modèles (on utilise XGBoost, le meilleur modèle)")
    print("=" * 70)
    trained = models.train_all_models(split.X_train, split.y_train, split.X_test)
    y_pred = trained.predictions["XGBoost (Pseudo-Huber)"]

    print("\n" + "=" * 70)
    print("3. Construction du tableau d'erreurs par transfert")
    print("=" * 70)
    err = error_analysis.build_error_frame(
        transfer_full_raw, split.df_model, split.X_test, split.y_test, y_pred
    )
    print(f"{len(err)} transferts analysés dans le jeu de test.")

    print("\n" + "=" * 70)
    print("4. Tableaux de synthèse par dimension")
    print("=" * 70)
    breakdowns = error_analysis.build_breakdown_tables(err)
    for name, df in breakdowns.items():
        print(f"\n--- Par {name} ---")
        print(df.round(2))
        df.round(2).to_csv(TABLES_DIR / f"error_analysis_{name}.csv", index=False)

    print("\n" + "=" * 70)
    print("5. Évolution de l'erreur dans le temps (modèle vs. Transfermarkt seul)")
    print("=" * 70)
    time_summary = error_analysis.bucket_by_time(err, n_buckets=10)
    print(time_summary[["Period", "N", "Model_Median_Abs_Pct_Err", "TM_Median_Abs_Pct_Err", "Model_Mean_Signed_Pct_Err"]].round(2))
    time_summary.round(2).to_csv(TABLES_DIR / "error_analysis_over_time.csv", index=False)

    print("\n" + "=" * 70)
    print("6. Graphiques")
    print("=" * 70)
    error_analysis.plot_error_magnitude(breakdowns)
    error_analysis.plot_error_bias(breakdowns)
    error_analysis.plot_error_over_time(time_summary)
    error_analysis.plot_error_density(err)

    print(f"\nErreur signée : médiane={err['signed_pct_err'].median():.1f}% | moyenne={err['signed_pct_err'].mean():.1f}%")
    print("(la moyenne est tirée vers le haut par de gros % d'erreur sur des transferts bon marché)")

    print("\nTerminé.")
    print("  Tableaux -> outputs/tables/error_analysis_*.csv")
    print("  Graphiques -> outputs/figures/error_analysis_magnitude.png, error_analysis_bias.png, error_over_time.png, error_density.png")


if __name__ == "__main__":
    main()
