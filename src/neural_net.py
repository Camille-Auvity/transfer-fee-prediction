"""Réseau de neurones avec embeddings de club (from_club_id / to_club_id),
testé comme alternative aux GBDT du pipeline principal.

Idée : `from_club_id` / `to_club_id` sont actuellement ignorés comme features
(seule la ligue agrégée est gardée). Un embedding par club pourrait capter des
effets spécifiques (style "ce club négocie dur" / "ce club paie plein tarif")
qui recouperaient le biais par championnat déjà observé dans
`error_analysis.py` — sans avoir à construire la feature financière à la main.

Module entièrement séparé du pipeline principal et des autres scripts
(`tune_models.py`, `error_analysis.py`, `calibrate_model.py`) : rien n'est
modifié ailleurs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import RANDOM_STATE

UNKNOWN_CLUB_IDX = 0
EMBEDDING_DIM = 12
HIDDEN_DIMS = (64, 32)
DROPOUT = 0.3


def get_club_ids(transfer_full: pd.DataFrame, df_model: pd.DataFrame, X_subset: pd.DataFrame) -> pd.DataFrame:
    """Récupère from_club_id / to_club_id (absents de X, retirés avant la
    modélisation) pour les lignes de X_subset, via un merge player_id+date —
    même logique que error_analysis.build_error_frame / examples.py."""
    keys = df_model.loc[X_subset.index, ["player_id", "date"]]
    merged = keys.merge(transfer_full[["player_id", "date", "from_club_id", "to_club_id"]], on=["player_id", "date"], how="left")
    merged.index = X_subset.index
    return merged[["from_club_id", "to_club_id"]]


class ClubVocab:
    """Associe chaque club_id vu dans les données d'entraînement à un indice
    d'embedding. Tout club inédit (vu seulement en validation/test) tombe sur
    l'indice 'inconnu' (0) plutôt que de faire planter l'encodage."""

    def __init__(self, club_ids: pd.Series):
        unique_ids = sorted(club_ids.dropna().unique())
        self.id_to_idx = {cid: i + 1 for i, cid in enumerate(unique_ids)}  # 0 = inconnu
        self.size = len(unique_ids) + 1

    def encode(self, club_ids: pd.Series) -> np.ndarray:
        return club_ids.map(self.id_to_idx).fillna(UNKNOWN_CLUB_IDX).astype(int).to_numpy()


class ClubEmbeddingNet(nn.Module):
    def __init__(self, n_clubs: int, n_numeric: int, embedding_dim: int = EMBEDDING_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.from_embed = nn.Embedding(n_clubs, embedding_dim)
        self.to_embed = nn.Embedding(n_clubs, embedding_dim)

        input_dim = 2 * embedding_dim + n_numeric
        h1, h2 = HIDDEN_DIMS
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.BatchNorm1d(h1), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.BatchNorm1d(h2), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, from_idx: torch.Tensor, to_idx: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        f = self.from_embed(from_idx)
        t = self.to_embed(to_idx)
        x = torch.cat([f, t, numeric], dim=1)
        return self.net(x).squeeze(-1)


@dataclass
class PreparedData:
    from_idx: torch.Tensor
    to_idx: torch.Tensor
    numeric: torch.Tensor
    y: torch.Tensor


@dataclass
class NNArtifacts:
    model: ClubEmbeddingNet
    vocab: ClubVocab
    imputer: SimpleImputer
    scaler: StandardScaler
    history: dict


def _prepare_split(
    transfer_full: pd.DataFrame, df_model: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
    vocab: ClubVocab, imputer: SimpleImputer, scaler: StandardScaler,
) -> PreparedData:
    club_ids = get_club_ids(transfer_full, df_model, X)
    from_idx = torch.as_tensor(vocab.encode(club_ids["from_club_id"]), dtype=torch.long)
    to_idx = torch.as_tensor(vocab.encode(club_ids["to_club_id"]), dtype=torch.long)

    numeric = imputer.transform(X)
    numeric = scaler.transform(numeric)
    numeric_t = torch.as_tensor(numeric, dtype=torch.float32)

    y_t = torch.as_tensor(y.to_numpy(), dtype=torch.float32)
    return PreparedData(from_idx=from_idx, to_idx=to_idx, numeric=numeric_t, y=y_t)


def train_club_embedding_model(
    transfer_full: pd.DataFrame, df_model: pd.DataFrame,
    X_fit: pd.DataFrame, y_fit: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
    max_epochs: int = 200, patience: int = 20, batch_size: int = 128, lr: float = 1e-3,
) -> NNArtifacts:
    """Entraîne le réseau sur (X_fit, y_fit), avec arrêt anticipé sur (X_val, y_val)."""
    torch.manual_seed(RANDOM_STATE)

    # Vocabulaire et préprocesseurs numériques appris UNIQUEMENT sur le fit set
    club_ids_fit = get_club_ids(transfer_full, df_model, X_fit)
    vocab = ClubVocab(pd.concat([club_ids_fit["from_club_id"], club_ids_fit["to_club_id"]]))

    imputer = SimpleImputer(strategy="median").fit(X_fit)
    scaler = StandardScaler().fit(imputer.transform(X_fit))

    fit_data = _prepare_split(transfer_full, df_model, X_fit, y_fit, vocab, imputer, scaler)
    val_data = _prepare_split(transfer_full, df_model, X_val, y_val, vocab, imputer, scaler)

    model = ClubEmbeddingNet(n_clubs=vocab.size, n_numeric=fit_data.numeric.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)

    loader = DataLoader(
        TensorDataset(fit_data.from_idx, fit_data.to_idx, fit_data.numeric, fit_data.y),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        for from_b, to_b, num_b, y_b in loader:
            optimizer.zero_grad()
            pred = model(from_b, to_b, num_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(val_data.from_idx, val_data.to_idx, val_data.numeric)
            val_loss = loss_fn(val_pred, val_data.y).item()

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping à l'epoch {epoch + 1} (meilleure val loss: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return NNArtifacts(model=model, vocab=vocab, imputer=imputer, scaler=scaler, history=history)


def predict(artifacts: NNArtifacts, transfer_full: pd.DataFrame, df_model: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    data = _prepare_split(transfer_full, df_model, X, pd.Series(np.zeros(len(X)), index=X.index), artifacts.vocab, artifacts.imputer, artifacts.scaler)
    artifacts.model.eval()
    with torch.no_grad():
        preds = artifacts.model(data.from_idx, data.to_idx, data.numeric)
    return preds.numpy()
