# Suivi MLflow et registre de modèles

## Objectif
Garder une trace de chaque entraînement (paramètres, métriques, fichiers) et savoir à tout moment
**quelle version du modèle est candidate et laquelle est validée**.

## Fichiers et scripts

| Élément | Rôle |
|---|---|
| `src/telecom_mlops/mlflow_settings.py` | Noms, alias et seuils de validation (un seul endroit à modifier) |
| `src/telecom_mlops/train_mlflow.py` | Entraîne XGBoost comme `train.py`, suit l'essai dans MLflow et crée une version candidate |
| `src/telecom_mlops/registry.py` | `status`, `promote` (validation) et `rollback` (retour arrière) |
| `mlflow.db`, `mlartifacts/` | Base et fichiers locaux de MLflow. **Non versionnés dans Git** |

## Ce que MLflow conserve pour chaque essai
- **Paramètres** : paramètres XGBoost, taille du test, nombre de lignes, liste des variables.
- **Métriques** : accuracy, précision / rappel / F1 macro, AUC macro, et précision / rappel / F1 de chaque classe (`recall_medium`, etc.).
- **Étiquettes (tags)** : commit Git, empreinte MD5 du fichier de données (à comparer avec celle du fichier `.dvc`), classes dans l'ordre du modèle, versions de scikit-learn et xgboost.
- **Fichiers** : matrice de confusion, métriques en JSON, échantillon de référence pour la surveillance, modèle complet (prétraitement + XGBoost).

## Version candidate et version validée

| | Version candidate | Version validée | Version précédente |
|---|---|---|---|
| Alias MLflow | `candidate` | `validated` | `previous` |
| Étape (tag `stage`) | `staging` | `production` | `archived` |
| Signification | Issue d'un entraînement, pas encore approuvée | A passé les critères et a été approuvée par une personne | Ancienne version validée, gardée pour revenir en arrière |
| Utilisée par l'API | Non | **Oui, la seule** | Non |

Un alias ne désigne qu'une seule version à la fois. Une version refusée reste « candidate ».
Une version annulée par un retour arrière reçoit `stage = rejected`.

## Critères de validation (modifiables dans `mlflow_settings.py`)

| Critère | Seuil | Effet |
|---|---|---|
| F1 macro | ≥ 0,45 | Bloquant |
| AUC macro (One-vs-Rest) | ≥ 0,65 | Bloquant |
| Pas de régression par rapport à la version validée | F1 macro ne baisse pas de plus de 0,01 | Bloquant |
| Rappel de chaque classe | < 0,10 : avertissement | Non bloquant, à documenter (voir l'issue #2 : la classe `medium` est difficile à détecter) |

Les seuils minimaux sont tirés du premier entraînement (F1 macro 0,456 ; AUC 0,656). Ils sont modestes :
à relever si le modèle s'améliore. La validation n'est **jamais automatique** : une personne lance `promote`
après avoir lu les métriques.

## Commandes (depuis le dossier du projet)

```bash
pip install -r requirements-mlops.txt

# 1. Entraîner et créer une version candidate
python src/telecom_mlops/train_mlflow.py
python src/telecom_mlops/train_mlflow.py --max-depth 4 --learning-rate 0.05 --run-name depth4

# 2. Comparer les essais dans l'interface
mlflow ui --backend-store-uri sqlite:///mlflow.db        # puis http://localhost:5000

# 3. Voir l'état du registre, valider ou revenir en arrière
python src/telecom_mlops/registry.py status
python src/telecom_mlops/registry.py promote
python src/telecom_mlops/registry.py rollback
```

## Lien avec l'API
Le modèle validé se charge avec `mlflow.sklearn.load_model("models:/default_risk_xgboost@validated")`.
Le modèle prédit des nombres ; leur correspondance avec les libellés est dans l'étiquette `classes` de la
version (ordre : `high`, `low`, `medium`). Pour l'instant, l'API lit `models/default_risk_model.joblib`.
Passer à `@validated` est un choix à discuter avec la personne qui s'occupe de l'API.

## Limites
- La base MLflow est **locale** : chaque personne du groupe a son propre registre. Pour partager un registre
  commun, il faudrait un serveur MLflow (par exemple dans le `docker-compose` de l'issue #5).
- Un modèle enregistré doit être rechargé avec les mêmes versions de scikit-learn et xgboost que celles de
  l'entraînement (voir `requirements-mlops.txt`).
