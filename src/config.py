"""Chemins et constantes partagés par tous les modules du projet."""

from pathlib import Path

# Racine du projet (dossier contenant main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "transfermarkt data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

CLUBS_CSV = DATA_DIR / "clubs.csv"
TRANSFER_PANEL_CSV = DATA_DIR / "transfer_panel.csv"
PLAYERS_CSV = DATA_DIR / "players.csv"

RANDOM_STATE = 42

# Ligues "Big 5" conservées telles quelles ; toutes les autres sont regroupées sous "Other"
BIG5_LEAGUES = ["L1", "GB1", "IT1", "ES1", "FR1"]

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
