"""
Entraînement XGBoost suivi avec MLflow, puis enregistrement dans le registre.

Ce script reprend la méthode de train.py (mêmes variables, même séparation
80/20 stratifiée, même graine) et ajoute :
1. le suivi des paramètres, des métriques et des fichiers dans MLflow ;
2. l'enregistrement du modèle dans le registre, avec l'alias « candidate ».

Exécution (depuis le dossier du projet) :
    python src/telecom_mlops/train_mlflow.py
    python src/telecom_mlops/train_mlflow.py --max-depth 4 --learning-rate 0.05
    python src/telecom_mlops/train_mlflow.py --sample-rows 100000   # essai rapide

Consulter les essais :
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import hashlib
import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import DATA_PATH, FEATURES, ROOT, TARGET
from mlflow_settings import (
    ALIAS_CANDIDATE,
    ARTIFACT_ROOT,
    EXPERIMENT_NAME,
    REFERENCE_SAMPLE_PATH,
    REGISTERED_MODEL_NAME,
    TRACKING_URI,
)
from train import RANDOM_STATE, TEST_SIZE, XGBOOST_PARAMETERS, build_preprocessor


def parse_arguments():
    """Lit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run-name", default=None, help="Nom de l'essai dans MLflow.")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample-bytree", type=float, default=None)
    parser.add_argument(
        "--sample-rows", type=int, default=None,
        help="Utiliser seulement N lignes (essai rapide).",
    )
    parser.add_argument(
        "--no-register", action="store_true",
        help="Suivre l'essai dans MLflow sans l'enregistrer dans le registre.",
    )
    return parser.parse_args()


def git_commit():
    """Retourne le commit Git courant, pour savoir quel code a produit le modèle."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "inconnu"


def file_md5(path):
    """Empreinte MD5 du fichier de données (identique à celle du fichier .dvc)."""
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def get_or_create_experiment():
    """Crée l'expérience MLflow avec un dossier d'artefacts stable."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT)
    return experiment.experiment_id


def main():
    args = parse_arguments()

    # 1. Données et séparation : identiques à train.py.
    print(f"Chargement du jeu de données : {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    if args.sample_rows:
        data = data.sample(args.sample_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    X = data[FEATURES].copy()
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(data[TARGET].astype(str))
    class_names = target_encoder.classes_.tolist()
    class_ids = list(range(len(class_names)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 2. Paramètres du modèle (ceux de train.py, modifiables en ligne de commande).
    parameters = XGBOOST_PARAMETERS.copy()
    overrides = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }
    parameters.update({name: value for name, value in overrides.items() if value is not None})
    parameters["num_class"] = len(class_names)

    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor(categorical, numeric)),
            ("classifier", XGBClassifier(**parameters)),
        ]
    )

    # 3. Entraînement (poids équilibrés, comme train.py).
    print("Entraînement de XGBoost...")
    pipeline.fit(
        X_train, y_train,
        classifier__sample_weight=compute_sample_weight("balanced", y_train),
    )

    # 4. Évaluation sur le jeu de test.
    predictions = pipeline.predict(X_test).astype(int)
    probabilities = pipeline.predict_proba(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "auc_macro_ovr": roc_auc_score(
            y_test, probabilities, labels=class_ids, multi_class="ovr", average="macro"
        ),
    }
    report = classification_report(
        y_test, predictions, labels=class_ids, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    for name in class_names:
        metrics[f"precision_{name}"] = report[name]["precision"]
        metrics[f"recall_{name}"] = report[name]["recall"]
        metrics[f"f1_{name}"] = report[name]["f1-score"]
    matrix = confusion_matrix(y_test, predictions, labels=class_ids)

    print("\n=== Métriques sur le jeu de test ===")
    for name, value in metrics.items():
        print(f"{name:22s}: {value:.4f}")

    # 5. Fichiers à conserver avec l'essai.
    (ROOT / "reports").mkdir(exist_ok=True)
    REFERENCE_SAMPLE_PATH.parent.mkdir(exist_ok=True)
    # Échantillon de référence : sert plus tard à détecter la dérive des données.
    reference = X_train.sample(min(5000, len(X_train)), random_state=RANDOM_STATE)
    reference.to_csv(REFERENCE_SAMPLE_PATH, index=False)

    figure, axis = plt.subplots(figsize=(5, 4.2))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(class_ids, class_names)
    axis.set_yticks(class_ids, class_names)
    axis.set_xlabel("classe prédite")
    axis.set_ylabel("classe réelle")
    for i in class_ids:
        for j in class_ids:
            axis.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    confusion_path = ROOT / "reports" / "mlflow_confusion_matrix.png"
    figure.savefig(confusion_path, dpi=120, bbox_inches="tight")
    plt.close(figure)

    metrics_path = ROOT / "reports" / "mlflow_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"metrics": metrics, "confusion_matrix": matrix.tolist(), "classes": class_names},
            indent=2, default=float,
        ),
        encoding="utf-8",
    )

    # 6. Suivi MLflow : paramètres, métriques, fichiers, modèle.
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_id = get_or_create_experiment()
    with mlflow.start_run(experiment_id=experiment_id, run_name=args.run_name) as run:
        mlflow.log_params(
            {key: value for key, value in parameters.items() if key != "n_jobs"}
            | {"test_size": TEST_SIZE, "n_train": len(X_train), "n_test": len(X_test),
               "features": ",".join(FEATURES)}
        )
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "data_md5": file_md5(DATA_PATH) if DATA_PATH.exists() else "inconnu",
                "classes": ",".join(class_names),
                "xgboost": xgboost.__version__,
                "scikit_learn": sklearn.__version__,
            }
        )
        mlflow.log_metrics({name: float(value) for name, value in metrics.items()})
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(REFERENCE_SAMPLE_PATH))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
            input_example=X_test.head(3),
        )
        run_id = run.info.run_id
    print(f"\nEssai MLflow enregistré : {run_id}")

    # 7. Enregistrement dans le registre : la nouvelle version est « candidate ».
    if args.no_register:
        print("Enregistrement dans le registre ignoré (--no-register).")
        return

    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CANDIDATE, version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "stage", "staging")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "classes", ",".join(class_names))
    print(f"Version {version.version} enregistrée sous l'alias « {ALIAS_CANDIDATE} ».")
    print("Étape suivante : python src/telecom_mlops/registry.py promote")


if __name__ == "__main__":
    main()
