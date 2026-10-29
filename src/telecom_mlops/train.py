"""
Entraînement et évaluation d'un modèle XGBoost pour prédire default_risk.

Le script :
1. Charge le jeu de données CSV.
2. Sépare les données en ensembles d'entraînement et de test.
3. Prépare les variables numériques et catégorielles.
4. Entraîne XGBoost.
5. Calcule les métriques d'évaluation.
6. Enregistre le modèle et les métriques.

Exécution dans Spyder :
    Ouvrir ce fichier et cliquer sur le bouton Exécuter.
"""

import json

import joblib
import pandas as pd
import sklearn
import xgboost

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from config import DATA_PATH, FEATURES, METRICS_PATH, MODEL_PATH, TARGET


# Paramètres utilisés pour rendre l'entraînement reproductible.
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Paramètres principaux du modèle XGBoost.
XGBOOST_PARAMETERS = {
    "objective": "multi:softprob",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def build_preprocessor(categorical_features, numeric_features):
    """
    Prépare les colonnes avant l'entraînement.

    - Variables catégorielles : remplace les valeurs manquantes par la valeur
      la plus fréquente, puis transforme les catégories en colonnes numériques.
    - Variables numériques : remplace les valeurs manquantes par la médiane.

    XGBoost n'a pas besoin de standardiser les variables numériques.
    """

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )

    return preprocessor


def main():
    """Charge les données, entraîne le modèle et écrit les résultats."""

    # 1. Charger le fichier de données.
    print(f"Chargement du jeu de données : {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)

    # Vérifier que les colonnes nécessaires sont présentes.
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le CSV : "
            + ", ".join(missing_columns)
        )

    # X contient les variables d'entrée; y contient la cible à prédire.
    X = data[FEATURES].copy()
    y_text = data[TARGET].astype(str)

    # Transformer les classes texte en nombres pour XGBoost.
    # L'encodeur conserve la correspondance entre nombres et classes.
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_text)
    class_names = target_encoder.classes_.tolist()
    class_ids = list(range(len(class_names)))

    print(f"Nombre de lignes : {len(data):,}")
    print(f"Cible à prédire : {TARGET}")
    print(f"Classes : {class_names}")
    print(f"Variables utilisées : {FEATURES}")

    # 2. Repérer les variables catégorielles et numériques.
    categorical_features = (
        X.select_dtypes(include=["object", "category", "bool"])
        .columns
        .tolist()
    )
    numeric_features = (
        X.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    print(f"Variables catégorielles : {categorical_features}")
    print(f"Variables numériques : {numeric_features}")

    # 3. Séparer les données.
    # La stratification conserve des proportions semblables de chaque classe
    # dans les ensembles d'entraînement et de test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Lignes d'entraînement : {len(X_train):,}")
    print(f"Lignes de test : {len(X_test):,}")

    # 4. Donner un poids plus élevé aux classes moins fréquentes.
    # Cela aide l'algorithme à ne pas privilégier uniquement la classe majoritaire.
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    # 5. Construire le pipeline complet.
    # Le prétraitement est ajusté uniquement sur les données d'entraînement,
    # afin d'éviter que le jeu de test influence l'entraînement.
    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    classifier_parameters = XGBOOST_PARAMETERS.copy()
    classifier_parameters["num_class"] = len(class_names)

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", XGBClassifier(**classifier_parameters)),
        ]
    )

    # 6. Entraîner XGBoost.
    print("\nEntraînement de XGBoost...")
    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights,
    )

    # 7. Faire des prédictions sur les données de test.
    # Ces lignes n'ont pas été utilisées pendant l'entraînement.
    predictions = model.predict(X_test).astype(int)
    probabilities = model.predict_proba(X_test)

    # 8. Calculer les métriques globales.
    accuracy = accuracy_score(y_test, predictions)
    precision_macro = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    recall_macro = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    # AUC multiclasse calculée en One-vs-Rest, puis moyennée entre les classes.
    auc_macro_ovr = roc_auc_score(
        y_test,
        probabilities,
        labels=class_ids,
        multi_class="ovr",
        average="macro",
    )

    # 9. Calculer le rapport par classe et la matrice de confusion.
    report = classification_report(
        y_test,
        predictions,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=class_ids,
    )

    # 10. Afficher les résultats.
    print("\n=== Résultats globaux de XGBoost ===")
    print(f"Accuracy              : {accuracy:.4f}")
    print(f"Précision macro       : {precision_macro:.4f}")
    print(f"Rappel macro          : {recall_macro:.4f}")
    print(f"F1-score macro        : {f1_macro:.4f}")
    print(f"AUC macro (One-vs-Rest): {auc_macro_ovr:.4f}")

    print("\n=== Précision, rappel et F1-score par classe ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=class_ids,
            target_names=class_names,
            zero_division=0,
        )
    )

    print(
        "=== Matrice de confusion "
        "(lignes = classes réelles, colonnes = classes prédites) ==="
    )
    print(
        pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names,
        )
    )

    # 11. Créer les dossiers de sortie s'ils n'existent pas.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Enregistrer le pipeline et l'encodeur ensemble.
    # L'encodeur permettra plus tard de reconvertir les prédictions numériques
    # en libellés compréhensibles : high, low ou medium.
    model_bundle = {
        "model": model,
        "target_encoder": target_encoder,
        "target": TARGET,
        "features": FEATURES,
        "class_names": class_names,
        "algorithm": "XGBoost",
        "parameters": XGBOOST_PARAMETERS,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(model_bundle, MODEL_PATH)

    # Enregistrer les métriques dans un fichier JSON lisible.
    metrics = {
        "algorithm": "XGBoost",
        "target": TARGET,
        "features": FEATURES,
        "classes": class_names,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "parameters": XGBOOST_PARAMETERS,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "auc_macro_ovr": auc_macro_ovr,
        "classification_report": report,
        "confusion_matrix": {
            "labels": class_names,
            "values": matrix.tolist(),
        },
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"\nModèle XGBoost enregistré dans : {MODEL_PATH}")
    print(f"Métriques enregistrées dans : {METRICS_PATH}")


if __name__ == "__main__":
    main()