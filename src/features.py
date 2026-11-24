"""Nettoyage et feature engineering sur 'transfer_full' (cellules 6 à 9 du notebook)."""

import numpy as np
import pandas as pd


def parse_to_int(val):
    """Convertit une valeur Transfermarkt du type '12.5m' / '750k' / '-' en entier (euros)."""
    if pd.isna(val):
        return np.nan

    # Clean string
    val = str(val).replace("€", "").strip()
    if val in ["-", "+-0", "0", ""]:
        return 0

    # Apply multiplier based on suffix
    multiplier = 1
    if val.endswith("m"):
        multiplier = 1_000_000
        val = val[:-1]
    elif val.endswith("k"):
        multiplier = 1_000
        val = val[:-1]

    try:
        return int(round(float(val) * multiplier))
    except ValueError:
        return np.nan


def clean_net_transfer_record(transfer_full: pd.DataFrame) -> pd.DataFrame:
    # Convert column to nullable integer ('Int64')
    transfer_full["net_transfer_record_from"] = (
        transfer_full["net_transfer_record_from"].apply(parse_to_int).astype("Int64")
    )

    # Convert column to nullable integer ('Int64')
    transfer_full["net_transfer_record_to"] = (
        transfer_full["net_transfer_record_to"].apply(parse_to_int).astype("Int64")
    )
    return transfer_full


def one_hot_encode(transfer_full: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        transfer_full,
        dtype="bool",
        columns=[
            "position", "foot", "domestic_competition_id_to",
            "from_confederation", "to_confederation", "domestic_competition_id_from",
        ],
    )


def add_engineered_features(transfer_full: pd.DataFrame) -> pd.DataFrame:
    transfer_full["spread"] = transfer_full.transfer_fee - transfer_full.market_value_in_eur
    transfer_full["log_spread"] = np.log(transfer_full.transfer_fee / transfer_full.market_value_in_eur)
    transfer_full["prev_spread"] = transfer_full.prev_transfer_fee - transfer_full.prev_market_value
    transfer_full["prev_log_spread"] = np.log(transfer_full.prev_transfer_fee / transfer_full.prev_market_value)
    transfer_full["new_club_bigger_squad"] = transfer_full["to_club_squad_size"] > transfer_full["from_club_squad_size"]
    transfer_full["new_club_younger_squad"] = transfer_full["to_club_average_age"] < transfer_full["from_club_average_age"]
    transfer_full["new_club_more_foreigners"] = transfer_full["to_club_foreigners_pct"] > transfer_full["from_club_foreigners_pct"]
    transfer_full["new_club_more_national_team_players"] = (
        transfer_full["to_club_national_team_players"] > transfer_full["from_club_national_team_players"]
    )
    transfer_full["log_curr_market_value"] = np.log(transfer_full.market_value_in_eur)
    transfer_full["value_momentum"] = np.log(transfer_full.market_value_in_eur / transfer_full.prev_market_value)
    transfer_full["curr_val_vs_fee_paid"] = np.log(transfer_full.market_value_in_eur / transfer_full.prev_transfer_fee)
    return transfer_full


# Colonnes à exclure du calcul de corrélation (identifiants, doublons, cibles redondantes)
CORR_DROP_COLS = [
    "date", "player_id", "has_fee", "has_recent_appearances",
    "from_club_id", "to_club_id", "club_id_to", "club_id_from",
    "squad_size_from", "foreigners_percentage_from", "national_team_players_from",
    "squad_size_to", "average_age_to", "foreigners_percentage_to", "national_team_players_to",
    "average_age_from", "transfer_fee", "market_value_in_eur",
    "to_club_squad_size", "from_club_squad_size", "to_club_average_age", "from_club_average_age",
    "to_club_foreigners_pct", "from_club_foreigners_pct",
    "to_club_national_team_players", "from_club_national_team_players",
]


def compute_spread_correlations(transfer_full: pd.DataFrame) -> pd.DataFrame:
    """Matrice de corrélation (toutes variables numériques) sur les lignes avec fee connu."""
    correl = (
        transfer_full[transfer_full.has_fee == 1]
        .select_dtypes(exclude=["object"])  # équivalent à exclude=["str"], compatible pandas >= 2
        .drop(columns=CORR_DROP_COLS)
        .corr()
    )
    return correl


def build_feature_table(transfer_full: pd.DataFrame) -> pd.DataFrame:
    """Applique tout le feature engineering (cellules 6 à 9) à 'transfer_full'."""
    transfer_full = clean_net_transfer_record(transfer_full)
    transfer_full = one_hot_encode(transfer_full)
    transfer_full = add_engineered_features(transfer_full)
    return transfer_full
