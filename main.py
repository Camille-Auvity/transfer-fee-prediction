"""Point d'entrée du projet : reproduit de bout en bout le notebook
`players data.ipynb` (prédiction du montant des transferts de football à
partir des données Transfermarkt), et sauvegarde les tableaux et graphiques
de résultats dans `outputs/`.

Usage:
    python main.py
"""

import pandas as pd

from src import data_loading, dataset, eda, evaluation, examples, features, models, tier_models
from src.config import TABLES_DIR


def main():
    print("=" * 70)
    print("1. Chargement des données et construction de 'transfer_full'")
    print("=" * 70)
    transfer_full = data_loading.build_transfer_full()
    transfer_full = features.build_feature_table(transfer_full)

    print("\n" + "=" * 70)
    print("2. Analyse exploratoire (corrélations, distribution de log_spread)")
    print("=" * 70)
    correl = features.compute_spread_correlations(transfer_full)
    eda.plot_spread_correlation_heatmap(
        correl, "Correlation with Spread (full feature set)", "correlation_heatmap_full.png"
    )

    final_data = dataset.build_final_data(transfer_full)
    eda.plot_log_spread_distribution(final_data)
    eda.plot_spread_correlation_heatmap(
        final_data.corr(), "Correlation with Spread (final modeling set)", "correlation_heatmap_final.png"
    )

    print("\n" + "=" * 70)
    print("3. Construction de X / y et split train/test chronologique")
    print("=" * 70)
    split = dataset.build_model_split(final_data, transfer_full)

    print("\n" + "=" * 70)
    print("4. Entraînement des modèles (linéaires + arbres/boosting + blend)")
    print("=" * 70)
    trained = models.train_all_models(split.X_train, split.y_train, split.X_test)

    print("\n" + "=" * 70)
    print("5. Évaluation des modèles")
    print("=" * 70)
    results_df = evaluation.evaluate_regression_models(split.y_test, trained.predictions)
    print(results_df.round(4))
    results_df.round(4).to_csv(TABLES_DIR / "results_regression.csv", index=False)
    evaluation.plot_mae_comparison(results_df)

    evaluation.plot_feature_importance(
        results_df, trained.fitted_models, trained.linear_models,
        split.X_train, split.X_test, split.y_test,
    )

    evaluation.print_tm_vs_model_r2(split.y_test, split.X_test, trained.predictions)

    spread_metrics_df = evaluation.evaluate_classification_metrics(split.y_test, trained.predictions)
    print(spread_metrics_df.round(4))
    spread_metrics_df.round(4).to_csv(TABLES_DIR / "results_classification.csv", index=False)

    fee_metrics_df = evaluation.evaluate_fee_level_metrics(split.y_test, split.X_test, trained.predictions)
    print(fee_metrics_df.round(3))
    fee_metrics_df.round(3).to_csv(TABLES_DIR / "fee_level_metrics.csv", index=False)

    evaluation.plot_diagnostics(split.y_test, split.X_test, trained.predictions, results_df)

    tier_summary, _ = evaluation.tier_error_table_and_plot(
        split.X_test, split.y_test, trained.predictions, results_df
    )
    print(tier_summary.round(2))
    tier_summary.round(2).to_csv(TABLES_DIR / "fee_error_by_valuation_tier.csv")

    print("\n" + "=" * 70)
    print("6. Modèles XGBoost par tranche de valorisation vs. modèle global")
    print("=" * 70)
    X_all = pd.concat([split.X_train, split.X_test])
    tier_comp_df = tier_models.run_tier_comparison(
        X_all, split.X_train, split.X_test, split.y_train, split.y_test
    )
    print(tier_comp_df.round(3))
    tier_comp_df.round(3).to_csv(TABLES_DIR / "tier_vs_global_xgb.csv", index=False)
    tier_models.plot_tier_comparison(tier_comp_df)

    print("\n" + "=" * 70)
    print("7. Comparaison finale et exemples de transferts du jeu de test")
    print("=" * 70)
    fee_comparison_df = examples.compare_models_vs_transfermarkt(split.X_test, split.y_test, trained.predictions)
    print(fee_comparison_df.round(3))
    fee_comparison_df.round(3).to_csv(TABLES_DIR / "final_fee_comparison.csv", index=False)
    examples.plot_fee_comparison(fee_comparison_df)

    best_model_name, test_players = examples.build_test_players_table(
        transfer_full, split.df_model, split.X_test, split.y_test, trained.predictions, fee_comparison_df
    )
    sample_tables, top15_test = examples.top_mid_sample_tables(test_players)
    for table_name, table_df in sample_tables.items():
        print(f"\n--- {table_name} ---")
        print(table_df)
        table_df.to_csv(TABLES_DIR / f"test_transfers_{table_name}.csv", index=False)

    examples.plot_top_transfers(top15_test, best_model_name)

    print("\nTerminé. Tableaux -> outputs/tables/ | Graphiques -> outputs/figures/")


if __name__ == "__main__":
    main()
