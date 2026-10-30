"""
API FastAPI de prédiction du risque d'impayé (default_risk).

Endpoints :
- GET  /health  : vérifie que l'API fonctionne et que le modèle est chargé.
- POST /predict : reçoit les variables d'un client, renvoie la classe de risque.

Lancement (depuis la racine du projet) :
    uvicorn telecom_mlops.api:app --app-dir src --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from telecom_mlops.config import MODEL_PATH

logger = logging.getLogger("telecom_mlops.api")

# Le chemin du modèle peut être remplacé par la variable d'environnement
# MODEL_PATH (utile pour Docker, la CI ou les tests).
def get_model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", MODEL_PATH))


# --------------------------------------------------------------------------
# Schémas d'entrée / sortie
# --------------------------------------------------------------------------
class ClientFeatures(BaseModel):
    """Variables nécessaires pour prédire la classe de risque d'un client."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "operator": "MTN",
                "credit_score": 720,
                "credit_rating": "good",
                "tenure_months": 36,
                "avg_monthly_spend_ngn": 25000,
                "recharge_frequency_per_month": 15,
                "payment_consistency_score": 0.55,
                "late_payment_count": 2,
                "account_type": "prepaid",
            }
        },
    )

    operator: str = Field(min_length=1, description="Opérateur (ex. MTN, Airtel, Glo, 9mobile)")
    credit_score: float = Field(ge=300, le=1000, description="Score de crédit (300 à 1000)")
    credit_rating: str = Field(min_length=1, description="Note de crédit : poor, fair, good ou excellent")
    tenure_months: int = Field(ge=0, le=1200, description="Ancienneté du client en mois")
    avg_monthly_spend_ngn: float = Field(ge=0, description="Dépense mensuelle moyenne (NGN)")
    recharge_frequency_per_month: float = Field(ge=0, description="Nombre de recharges par mois")
    payment_consistency_score: float = Field(ge=0, le=1, description="Régularité des paiements (0 à 1)")
    late_payment_count: int = Field(ge=0, description="Nombre de retards de paiement")
    account_type: str = Field(min_length=1, description="Type de compte : prepaid ou postpaid")


class PredictionResponse(BaseModel):
    risk_class: str = Field(description="Classe de risque prédite : low, medium ou high")
    probabilities: dict[str, float] = Field(description="Probabilité estimée pour chaque classe")


# --------------------------------------------------------------------------
# Chargement du modèle
# --------------------------------------------------------------------------
def load_model_bundle(path: Path) -> dict:
    """Charge le modèle enregistré par train.py et en extrait les infos utiles."""
    bundle = joblib.load(path)
    allowed_values: dict[str, list[str]] = {}
    try:
        # Valeurs catégorielles connues du modèle (utilisées pour valider l'entrée).
        preprocessing = bundle["model"].named_steps["preprocessing"]
        name, _, columns = preprocessing.transformers_[0]
        onehot = preprocessing.named_transformers_[name].named_steps["onehot"]
        for column, categories in zip(columns, onehot.categories_):
            allowed_values[column] = sorted(str(c) for c in categories)
    except Exception:  # noqa: BLE001 - la validation fine est optionnelle
        logger.warning("Valeurs catégorielles du modèle introuvables : validation allégée.")
    return {
        "pipeline": bundle["model"],
        "features": list(bundle["features"]),
        "class_names": list(bundle["class_names"]),
        "allowed_values": allowed_values,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle est rangé dans l'état propre à ce démarrage (request.state),
    # et non dans l'objet `app` partagé : deux démarrages ne s'écrasent pas.
    model = None
    path = get_model_path()
    try:
        model = load_model_bundle(path)
        logger.info("Modèle chargé depuis %s", path)
    except Exception as error:  # noqa: BLE001
        logger.error("Impossible de charger le modèle (%s) : %s", path, error)
    yield {"model": model}


app = FastAPI(
    title="Telecom credit risk API",
    description="Prédit la classe de risque d'impayé (low, medium, high) d'un client télécom.",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Erreurs de validation compréhensibles
# --------------------------------------------------------------------------
def _friendly_message(error: dict) -> str:
    kind = error.get("type", "")
    context = error.get("ctx", {})
    if kind == "missing":
        return "Champ obligatoire manquant."
    if kind == "extra_forbidden":
        return "Champ inconnu : il ne fait pas partie des variables attendues."
    if kind in ("greater_than_equal", "greater_than"):
        return f"La valeur doit être supérieure ou égale à {context.get('ge', context.get('gt'))}."
    if kind in ("less_than_equal", "less_than"):
        return f"La valeur doit être inférieure ou égale à {context.get('le', context.get('lt'))}."
    if kind in ("int_parsing", "int_from_float", "int_type"):
        return "La valeur doit être un nombre entier."
    if kind in ("float_parsing", "float_type"):
        return "La valeur doit être un nombre."
    if kind in ("string_type", "string_too_short"):
        return "La valeur doit être une chaîne de caractères non vide."
    if kind == "json_invalid":
        return "Le corps de la requête n'est pas un JSON valide."
    return error.get("msg", "Valeur invalide.")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = [
        {
            "field": ".".join(str(part) for part in error["loc"] if part != "body"),
            "message": _friendly_message(error),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": "Données invalides.", "errors": errors},
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Interface web : formulaire pour tester la prédiction dans le navigateur."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health(request: Request) -> dict:
    """Indique que l'API répond et si le modèle est chargé."""
    return {"status": "ok", "model_loaded": getattr(request.state, "model", None) is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures, request: Request) -> PredictionResponse:
    """Renvoie la classe de risque prédite pour un client."""
    model = getattr(request.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas disponible. Vérifiez que le fichier du modèle existe.",
        )

    data = client.model_dump()

    # Vérifier les valeurs catégorielles par rapport à celles connues du modèle.
    errors = []
    for column, allowed in model["allowed_values"].items():
        if data.get(column) not in allowed:
            errors.append(
                {
                    "field": column,
                    "message": f"Valeur '{data.get(column)}' inconnue. Valeurs acceptées : {', '.join(allowed)}.",
                }
            )
    if errors:
        return JSONResponse(
            status_code=422,
            content={"detail": "Données invalides.", "errors": errors},
        )

    row = pd.DataFrame([data])[model["features"]]
    probabilities = model["pipeline"].predict_proba(row)[0]
    best = int(probabilities.argmax())

    return PredictionResponse(
        risk_class=model["class_names"][best],
        probabilities={
            name: round(float(p), 4) for name, p in zip(model["class_names"], probabilities)
        },
    )
