# Prédiction du risque d’impayé télécom avec MLOps

## Objectif

Construire une chaîne MLOps reproductible pour entraîner, évaluer, déployer et surveiller un modèle qui prédit la classe de risque d’impayé d’un client télécom.

## Jeu de données

Le projet utilise le jeu public [Telecom Credit Scoring Data (Nigeria)](https://huggingface.co/datasets/electricsheepafrica/africa-synth-telecom-telecom-credit-scoring-data-nigeria).

L’exploration initiale a relevé 500 000 lignes, 15 colonnes et aucune valeur manquante. La cible `default_risk` comporte trois classes : `low`, `medium` et `high`. Le problème est donc une classification à trois classes.

Le fichier CSV est suivi localement avec DVC. Le dépôt contient un fichier `.dvc` de référence. Aucun stockage distant DVC n’est encore configuré.

## Préparation des données

Les identifiants client et numéro de téléphone ne seront pas utilisés comme variables du modèle. Les autres variables seront examinées avant l’entraînement afin d’écarter les informations qui pourraient révéler indirectement la cible.

## Livrables du projet

- Dépôt Git organisé et documenté
- Données versionnées avec DVC
- Modèle entraîné, évalué et sérialisé
- API de prédiction
- Image Docker
- Automatisation CI/CD et entraînement continu
- Registre de modèles
- Surveillance des performances et de la dérive
- Rapport final avec l’architecture et les choix réalisés

## Structure actuelle

- `src/telecom_mlops/` : code source du projet
- `tests/` : tests automatisés
- `notebooks/` : exploration des données
- `models/` : modèles entraînés
- `model_registry/` : versions et statut des modèles
- `reports/` : métriques et rapport
- `monitoring/` : éléments de surveillance
- `.github/workflows/` : automatisation GitHub Actions

Les dossiers sont créés. Le code d’entraînement, l’API, le modèle et le pipeline automatisé restent à réaliser.
