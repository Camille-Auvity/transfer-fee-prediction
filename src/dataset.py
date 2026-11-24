"""Construction de 'final_data', puis de la matrice de features (X, y) et du
split train/test chronologique (cellules 11, 12, 14 et 15 du notebook)."""

from dataclasses import dataclass

import pandas as pd

from .features import CORR_DROP_COLS

# Même liste que pour la matrice de corrélation, mais on garde 'date' et
# 'player_id' car on en a encore besoin pour le tri chronologique et le suivi des joueurs.
FINAL_DATA_DROP_COLS = [c for c in CORR_DROP_COLS if c not in ("date", "player_id")]


def build_final_data(transfer_full: pd.DataFrame) -> pd.DataFrame:
    """Reproduit 'final_data' : lignes avec fee connu, colonnes redondantes retirées,
    et valeurs extrêmes de log_spread (2%/98%) écartées."""
    final_data = (
        transfer_full[transfer_full.has_fee == 1]
        .select_dtypes(exclude=["object"])  # équivalent à exclude=["str"], compatible pandas >= 2
        .drop(columns=FINAL_DATA_DROP_COLS)
    )
    final_data = final_data[
        (final_data.log_spread < final_data.log_spread.quantile(0.98))
        & (final_data.log_spread > final_data.log_spread.quantile(0.02))
    ]

    final_data["has_prior_paid_transfer"] = (
        (final_data.prev_transfer_fee != 0) & (final_data.prev_transfer_fee.notna())
    ).astype("bool")

    final_data["is_missing_value_momentum"] = final_data.value_momentum.isna()
    final_data["is_missing_contract_exp"] = final_data.years_to_contract_expiration.isna()
    final_data["is_missing_net_transfer_record_to"] = final_data.net_transfer_record_to.isna()

    return final_data


# Colonnes retirées de X avant modélisation (identifiants, cibles, doublons de 'spread')
X_DROP_COLS = [
    "player_id", "from_club_id", "to_club_id", "club_id_from", "club_id_to",
    "date", "spread", "log_spread",
    "prev_transfer_fee", "prev_market_value", "prev_spread",
    "last12m_app_count", "last12m_goals", "last12m_assists",
    "last12m_yellow_cards", "last12m_red_cards",
    "net_transfer_record_from", "foot_left", "foot_right", "foot_both",
    "new_club_bigger_squad", "new_club_younger_squad",
]

# Colonnes pour lesquelles on ajoute un flag binaire "is_missing_<col>"
MISSINGNESS_FLAG_COLS = ["years_to_contract_expiration", "value_momentum", "net_transfer_record_to"]

TRAIN_FRACTION = 0.80


@dataclass
class ModelSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    player_ids: pd.Series
    df_model: pd.DataFrame
    cutoff_date: pd.Timestamp


def build_model_split(final_data: pd.DataFrame, transfer_full: pd.DataFrame) -> ModelSplit:
    """Construit X/y et effectue le split 80/20 chronologique (cellule 15)."""
    df_model = final_data.copy()
    df_model = df_model.sort_values("date").reset_index(drop=True)

    # Save player_id separately (keeps exact same index as X)
    player_ids = (
        df_model["player_id"].copy()
        if "player_id" in df_model.columns
        else transfer_full.loc[df_model.index, "player_id"].copy()
    )

    # Target
    y = df_model["log_spread"].astype(float)

    X = df_model.drop(columns=[c for c in X_DROP_COLS if c in df_model.columns]).copy()

    # Missingness flags
    for col in MISSINGNESS_FLAG_COLS:
        if col in X.columns:
            X[f"is_missing_{col}"] = X[col].isna().astype(float)

    X = X.apply(pd.to_numeric, errors="coerce").astype(float)

    assert "player_id" not in X.columns, "player_id is still in X!"

    # Chronological 80/20 train-test split
    split_idx = int(len(df_model) * TRAIN_FRACTION)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    cutoff_date = df_model["date"].iloc[split_idx]
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,} | Split date: {cutoff_date.date()}")
    print(f"Total features: {X_train.shape[1]}")

    return ModelSplit(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        player_ids=player_ids, df_model=df_model, cutoff_date=cutoff_date,
    )
