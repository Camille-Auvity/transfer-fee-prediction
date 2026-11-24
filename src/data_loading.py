"""Chargement des données brutes et construction du tableau 'transfer_full'.

Reprend tel quel le pipeline des premières cellules du notebook
(`players data.ipynb`) : fusion avec les clubs, calcul de la valeur
marchande précédente, filtrage des transferts avec un montant connu,
et regroupement des championnats hors Big 5 sous "Other".
"""

import pandas as pd

from .config import BIG5_LEAGUES, CLUBS_CSV, TRANSFER_PANEL_CSV


def load_clubs() -> pd.DataFrame:
    return pd.read_csv(CLUBS_CSV)


def load_transfer_panel() -> pd.DataFrame:
    return pd.read_csv(TRANSFER_PANEL_CSV, parse_dates=["date"])


def merge_transfer_with_clubs(transfer_panel: pd.DataFrame, clubs: pd.DataFrame) -> pd.DataFrame:
    cols_to_keep = [
        "club_id", "name", "domestic_competition_id", "squad_size", "average_age",
        "foreigners_percentage", "national_team_players", "net_transfer_record",
    ]
    transfer_full = transfer_panel.merge(
        clubs[cols_to_keep], left_on="from_club_id", right_on="club_id", how="left"
    ).merge(
        clubs[cols_to_keep], left_on="to_club_id", right_on="club_id", how="left",
        suffixes=("_from", "_to"),
    )
    return transfer_full


def add_previous_market_value(transfer_full: pd.DataFrame) -> pd.DataFrame:
    transfer_full = transfer_full.sort_values(
        by=["player_id", "date", "num_prior_transfers"]
    ).reset_index(drop=True)
    # Récupère la valeur marchande au transfert précédent du même joueur
    transfer_full["prev_market_value"] = transfer_full.groupby("player_id")["market_value_in_eur"].shift(1)
    return transfer_full


def filter_known_fees(transfer_full: pd.DataFrame) -> pd.DataFrame:
    return transfer_full[
        (transfer_full.transfer_fee > 0)
        & (transfer_full.transfer_fee.notna())
        & (transfer_full.market_value_in_eur.notna())
    ]


def bucket_other_leagues(transfer_full: pd.DataFrame) -> pd.DataFrame:
    comp_cols = ["domestic_competition_id_from", "domestic_competition_id_to"]
    for col in comp_cols:
        transfer_full[col] = transfer_full[col].apply(lambda x: x if x in BIG5_LEAGUES else "Other")
    return transfer_full


def build_transfer_full() -> pd.DataFrame:
    """Reconstitue le dataframe 'transfer_full' utilisé dans tout le reste du projet."""
    clubs = load_clubs()
    transfer_panel = load_transfer_panel()

    transfer_full = merge_transfer_with_clubs(transfer_panel, clubs)
    transfer_full = add_previous_market_value(transfer_full)
    transfer_full = filter_known_fees(transfer_full)
    transfer_full = bucket_other_leagues(transfer_full)
    return transfer_full
