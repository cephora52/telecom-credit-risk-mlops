# API de prédiction du risque d'impayé

API FastAPI qui charge le modèle XGBoost enregistré (`models/default_risk_model.joblib`)
et renvoie la classe de risque d'un client : `low`, `medium` ou `high`.

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements-api.txt
```

Les versions de `scikit-learn` et `xgboost` sont fixées pour correspondre à celles
utilisées à l'entraînement : le fichier du modèle ne se charge pas toujours avec
d'autres versions.

## Lancer l'API

```bash
uvicorn telecom_mlops.api:app --app-dir src --reload
```

- API : http://127.0.0.1:8000
- Documentation interactive (Swagger) : http://127.0.0.1:8000/docs

Pour utiliser un autre fichier modèle : `MODEL_PATH=chemin/vers/modele.joblib` (variable
d'environnement) avant la commande `uvicorn`.

## Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "operator": "MTN",
    "credit_score": 720,
    "credit_rating": "good",
    "tenure_months": 36,
    "avg_monthly_spend_ngn": 25000,
    "recharge_frequency_per_month": 15,
    "payment_consistency_score": 0.55,
    "late_payment_count": 2,
    "account_type": "prepaid"
  }'
```
```json
{"risk_class": "low", "probabilities": {"high": 0.2, "low": 0.5, "medium": 0.3}}
```
(valeurs de probabilités données à titre d'exemple)

| Variable | Type | Contraintes |
|---|---|---|
| `operator` | texte | `9mobile`, `Airtel`, `Glo`, `MTN` |
| `credit_score` | nombre | 300 à 1000 |
| `credit_rating` | texte | `poor`, `fair`, `good`, `excellent` |
| `tenure_months` | entier | 0 à 1200 |
| `avg_monthly_spend_ngn` | nombre | ≥ 0 |
| `recharge_frequency_per_month` | nombre | ≥ 0 |
| `payment_consistency_score` | nombre | 0 à 1 |
| `late_payment_count` | entier | ≥ 0 |
| `account_type` | texte | `prepaid`, `postpaid` |

### Erreurs

| Code | Cas |
|---|---|
| 422 | Données invalides (champ manquant, hors limites, mauvais type, catégorie inconnue, champ en trop) |
| 503 | Modèle introuvable ou non chargé |

Exemple de réponse 422 :

```json
{
  "detail": "Données invalides.",
  "errors": [
    {"field": "credit_score", "message": "La valeur doit être inférieure ou égale à 1000."}
  ]
}
```

## Lancer les tests

```bash
pytest
```
