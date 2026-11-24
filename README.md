# ⚽ Transfer Fee Prediction — Transfermarkt

Projet de Machine Learning (Edhec) qui tente de répondre à une question simple :
**peut-on prédire le montant réel d'un transfert de football mieux que ne le fait
la valeur marchande Transfermarkt ?**

On modélise le ratio `log(Montant du transfert / Valeur Transfermarkt)` ("spread")
à partir de caractéristiques du joueur, de son historique récent et des clubs
vendeur/acheteur, puis on compare plusieurs modèles (régressions linéaires
régularisées, Random Forest, HistGradientBoosting, LightGBM, XGBoost, CatBoost)
à la simple valeur Transfermarkt prise comme estimation du montant du transfert.

Ce dépôt reprend à l'identique le pipeline développé dans le notebook original
(`players data.ipynb`), réorganisé en modules `.py` pour pouvoir être relancé
facilement et partagé sur Git.

## Résultats en un coup d'œil

| Approche | R² sur log(Montant) | Erreur médiane sur le montant |
|---|---|---|
| Transfermarkt seul | 0.738 | ~47.1 % |
| **XGBoost (meilleur modèle)** | **0.823** | **~38.2 %** |

(voir le détail complet, modèle par modèle, dans `outputs/tables/final_fee_comparison.csv`)

Le modèle bat la valeur Transfermarkt sur **~60 % des transferts** du jeu de test
(chronologique, 20 % des transferts les plus récents). Le gain est le plus net
sur les gros transferts (> 15 M€) et les transferts "stars" (> 25 M€).

Toutes les figures sont dans [`outputs/figures/`](outputs/figures) et tous les
tableaux de résultats dans [`outputs/tables/`](outputs/tables), régénérés à
chaque exécution de `main.py`. Exemples :

![Comparaison des modèles sur le montant final](outputs/figures/final_fee_comparison.png)
![Transferts les plus chers du jeu de test](outputs/figures/top15_transfers.png)
![Diagnostics (ROC, matrice de confusion, actual vs. predicted)](outputs/figures/diagnostics_panel.png)

## Structure du projet

```
.
├── main.py                 # Point d'entrée : exécute tout le pipeline
├── src/
│   ├── config.py            # Chemins et constantes
│   ├── data_loading.py      # Chargement + fusion clubs / transfer_panel
│   ├── features.py          # Nettoyage, one-hot encoding, feature engineering
│   ├── eda.py                # Corrélations, distribution de log_spread
│   ├── dataset.py            # Construction de X/y, split train/test chronologique
│   ├── models.py             # Entraînement des modèles (linéaires + arbres/boosting)
│   ├── evaluation.py         # Métriques de régression/classification, diagnostics
│   ├── tier_models.py        # XGBoost par tranche de valorisation vs. modèle global
│   └── examples.py           # Exemples de transferts du jeu de test, graphiques finaux
├── outputs/
│   ├── figures/              # PNG générés par le pipeline
│   └── tables/                # CSV des tableaux de résultats
├── transfermarkt data/       # Données (voir ci-dessous)
├── players data.ipynb         # Notebook original (exploration)
└── requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Données

Le pipeline a besoin de 3 fichiers dans `transfermarkt data/` :

- `clubs.csv` et `players.csv` — issus du dataset Kaggle
  [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores)
  (données Transfermarkt).
- `transfer_panel.csv` — un **panel de features déjà construit** (un transfert
  par ligne : fee, valeur marchande, historique du joueur sur 12 mois,
  caractéristiques des clubs vendeur/acheteur, etc.) à partir de ce même
  dataset. C'est le fichier d'entrée principal du pipeline.

Ces trois fichiers sont inclus dans le dépôt (ils ne pèsent que quelques
dizaines de Mo au total). Les autres fichiers bruts de `transfermarkt data/`
(appearances, game_events, game_lineups, transfers, valuations...) ne sont
**pas utilisés** par ce pipeline et sont exclus du dépôt (voir `.gitignore`) —
ils peuvent être retéléchargés via `kagglehub.dataset_download("davidcariboo/player-scores")`
si besoin pour reconstruire `transfer_panel.csv` à partir de zéro.

## Lancer le projet

```bash
python main.py
```

Le script affiche la progression dans la console, puis écrit :
- les figures dans `outputs/figures/*.png`
- les tableaux de résultats dans `outputs/tables/*.csv`

Temps d'exécution : quelques minutes sur un ordinateur portable (le jeu de
données fait ~17 000 transferts avec un montant connu).

## Méthodologie (résumé)

1. **Préparation** : fusion des transferts avec les informations des clubs
   vendeur/acheteur, récupération de la valeur marchande au transfert
   précédent, filtrage des transferts avec montant connu.
2. **Feature engineering** : encodage one-hot (poste, pied, championnat,
   confédération), variables dérivées (spread, momentum de valorisation,
   indicateurs de comparaison entre clubs), nettoyage du "net transfer record".
3. **Split chronologique** 80/20 (entraînement sur les transferts les plus
   anciens, test sur les plus récents) pour éviter toute fuite d'information
   dans le temps.
4. **Modélisation** : régressions régularisées (Ridge, ElasticNet, Huber) et
   modèles d'arbres/boosting (Random Forest, HistGradientBoosting, LightGBM,
   XGBoost, CatBoost), plus un blend Ridge/GBDT.
5. **Évaluation** : MAE/RMSE/R² sur `log_spread`, classification binaire
   (prime vs. décote par rapport à la valeur Transfermarkt), erreur en € sur
   le montant final, et analyse par tranche de valorisation du joueur.

## Notebook original

Le notebook `players data.ipynb` reste disponible dans le dépôt pour
l'exploration interactive — le code de `src/` en est une traduction fidèle,
sans changement de logique ni de paramètres.
