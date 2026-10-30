# Prédiction du risque d’impayé télécom avec MLOps

## Présentation

Ce projet met en place une chaîne MLOps pour prédire le niveau de risque d’impayé d’un client télécom au Nigeria. Il couvre l’entraînement et l’évaluation d’un modèle XGBoost, une API de prédiction, le suivi des expériences avec MLflow, un registre de modèles et une première surveillance de la dérive des données.

Le projet est réalisé dans le cadre d’un travail de groupe. Le dépôt GitHub sert à partager le code ; DVC sert à versionner et à partager le jeu de données.

## Jeu de données

Le projet utilise le jeu public [Telecom Credit Scoring Data (Nigeria)](https://huggingface.co/datasets/electricsheepafrica/africa-synth-telecom-telecom-credit-scoring-data-nigeria).

Le fichier contient 500 000 lignes et 15 colonnes. La variable cible est `default_risk`, avec trois classes : `low`, `medium` et `high`. Le problème est donc une classification à trois classes.

Les variables `customer_id` et `phone_number` ne sont pas utilisées par le modèle. Les variables utilisées sont :

- `operator`
- `credit_score`
- `credit_rating`
- `tenure_months`
- `avg_monthly_spend_ngn`
- `recharge_frequency_per_month`
- `payment_consistency_score`
- `late_payment_count`
- `account_type`

Les données étant synthétiques, les résultats obtenus ne peuvent pas être directement généralisés à une clientèle réelle.

## Modèle et résultats de référence

Le modèle de référence est un classificateur XGBoost, intégré à un pipeline qui prépare les variables catégorielles et numériques.

L’entraînement de référence utilise une séparation stratifiée reproductible : 80 % des lignes pour l’entraînement et 20 % pour le test, avec `random_state=42`. Les métriques sont enregistrées dans `reports/metrics.json`.

Résultats de référence sur 100 000 lignes de test :

| Mesure | Résultat |
|---|---:|
| Accuracy | 0,637 |
| Précision macro | 0,486 |
| Rappel macro | 0,526 |
| F1-score macro | 0,456 |
| AUC macro (One-vs-Rest) | 0,656 |

La classe `medium` reste difficile à détecter : son rappel de référence est très faible. L’accuracy seule ne suffit donc pas à juger ce modèle. Les mesures par classe et la matrice de confusion doivent aussi être prises en compte.

Le notebook `notebooks/xgboost_evaluation.ipynb` contient une exploration et une évaluation complémentaire sur un échantillon. Ses résultats ne remplacent pas les résultats de référence et ne signifient pas automatiquement que le modèle sérialisé a été remplacé.

## Fonctionnalités présentes

- Jeu de données suivi avec DVC et stockage distant configuré sur Google Drive
- Entraînement et évaluation du modèle XGBoost
- Modèle sérialisé dans `models/default_risk_model.joblib`
- Métriques de référence dans `reports/metrics.json`
- API FastAPI avec endpoints de santé et de prédiction
- Interface web simple pour essayer l’API
- Tests automatisés de l’API et de fonctions de surveillance
- Suivi des entraînements et des métriques avec MLflow
- Registre MLflow avec versions candidates, validation et retour arrière
- Surveillance par PSI et test de Kolmogorov-Smirnov
- Documentation de l’API, du registre et de la surveillance dans `docs/`

## État des livrables

| Livrable | État |
|---|---|
| Dépôt Git documenté | En place ; le README doit rester synchronisé avec l’avancement |
| Données versionnées avec DVC | Configuré avec un remote Google Drive |
| Entraînement et évaluation XGBoost | En place |
| API de prédiction | En place |
| Tests automatisés | Écrits ; leur exécution doit être vérifiée dans l’environnement du projet |
| Suivi et registre MLflow | En place en local |
| Surveillance de la dérive et de la performance | Première version en place, sur des fichiers CSV fournis |
| Image Docker | À réaliser |
| CI/CD et entraînement continu | À réaliser |
| Rapport final et architecture | À compléter avec les décisions et résultats définitifs |

## Organisation du dépôt

```text
.
├── .dvc/                         # Configuration versionnée du remote DVC
├── .github/workflows/            # Emplacement prévu pour les workflows GitHub Actions
├── docs/                         # Guides API, MLflow et surveillance
├── model_registry/                # Emplacement prévu pour les éléments du registre
├── models/                        # Modèle et fichiers de référence locaux
├── notebooks/                     # Exploration et évaluation
├── reports/                       # Métriques, matrices et rapports
├── src/telecom_mlops/
│   ├── api.py                     # API FastAPI
│   ├── config.py                  # Chemins, cible et variables du modèle
│   ├── monitoring.py              # Surveillance de la dérive et des performances
│   ├── mlflow_settings.py         # Configuration MLflow et seuils
│   ├── registry.py                # Validation et retour arrière du modèle
│   ├── train.py                   # Entraînement de référence XGBoost
│   ├── train_mlflow.py            # Entraînement suivi par MLflow
│   └── static/index.html          # Interface web de l’API
├── tests/                         # Tests automatisés
├── requirements-api.txt           # Dépendances API et tests API
├── requirements-mlops.txt          # Dépendances MLflow et surveillance
├── telecom_credit_scoring_data.csv.dvc
└── README.md
