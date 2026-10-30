"""
Surveillance du modèle : dérive des données (data drift) et performances.

Compare des données récentes à l'échantillon de référence (celui de
l'entraînement) et, si la vraie classe est connue, mesure la performance de la
version validée du registre.

Exécutions possibles (depuis le dossier du projet) :
    python src/telecom_mlops/monitoring.py --current donnees_recentes.csv
    python src/telecom_mlops/monitoring.py --simulate-drift          # démonstration
    python src/telecom_mlops/monitoring.py --simulate-drift --simulate-concept-drift

Code de retour : 0 = tout va bien, 1 = avertissement, 2 = alerte critique.
"""

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DATA_PATH, FEATURES, TARGET
from mlflow_settings import (
    ALIAS_CANDIDATE,
    ALIAS_VALIDATED,
    F1_DROP_CRITICAL,
    F1_DROP_WARNING,
    KS_P_VALUE,
    MONITORING_JSON,
    MONITORING_MD,
    MONITORING_PNG,
    PSI_CRITICAL,
    PSI_WARNING,
    REFERENCE_SAMPLE_PATH,
    REGISTERED_MODEL_NAME,
    TRACKING_URI,
)

SEVERITY = {"ok": 0, "warning": 1, "critical": 2}


def psi(reference, current, bins=10):
    """
    Population Stability Index : mesure de combien une variable a changé.
    < 0,10 stable ; 0,10 à 0,20 dérive légère ; > 0,20 dérive forte.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()
    if pd.api.types.is_numeric_dtype(reference) and reference.nunique() > bins:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_share = np.histogram(reference, edges)[0] / len(reference)
        cur_share = np.histogram(current, edges)[0] / len(current)
    else:  # variable catégorielle ou à peu de valeurs
        categories = sorted(set(reference.unique()) | set(current.unique()), key=str)
        ref_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0).values
        cur_share = current.value_counts(normalize=True).reindex(categories, fill_value=0).values
    ref_share = ref_share + 1e-6
    cur_share = cur_share + 1e-6
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def level_from_psi(value):
    if value >= PSI_CRITICAL:
        return "critical"
    if value >= PSI_WARNING:
        return "warning"
    return "ok"


def level_from_drop(drop):
    if drop >= F1_DROP_CRITICAL:
        return "critical"
    if drop >= F1_DROP_WARNING:
        return "warning"
    return "ok"


def data_drift(reference, current):
    """PSI et test KS pour chaque variable utilisée par le modèle."""
    rows = []
    for column in FEATURES:
        value = psi(reference[column], current[column])
        ks_p = None
        if pd.api.types.is_numeric_dtype(reference[column]):
            ks_p = float(ks_2samp(reference[column].dropna(), current[column].dropna()).pvalue)
        rows.append(
            {
                "variable": column,
                "psi": round(value, 4),
                "ks_p_value": None if ks_p is None else float(f"{ks_p:.3g}"),
                "ks_significatif": bool(ks_p is not None and ks_p < KS_P_VALUE),
                "niveau": level_from_psi(value),
            }
        )
    return rows


def load_reference():
    if not REFERENCE_SAMPLE_PATH.exists():
        raise SystemExit(
            f"Échantillon de référence introuvable : {REFERENCE_SAMPLE_PATH}\n"
            "Lancez d'abord : python src/telecom_mlops/train_mlflow.py"
        )
    return pd.read_csv(REFERENCE_SAMPLE_PATH)


def simulated_current_data(concept_drift):
    """
    Données de démonstration : un échantillon du jeu de test dont on décale
    volontairement certaines variables (inflation, moins de recharges...).
    Avec concept_drift, une partie des vraies classes est aussi mélangée.
    """
    from train import RANDOM_STATE, TEST_SIZE

    data = pd.read_csv(DATA_PATH)
    y = LabelEncoder().fit_transform(data[TARGET].astype(str))
    _, test_data = train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    current = test_data.sample(min(20000, len(test_data)), random_state=0).copy()
    for column, factor in {"avg_monthly_spend_ngn": 1.6, "recharge_frequency_per_month": 0.7}.items():
        if column in current:
            current[column] = current[column] * factor
    if "credit_score" in current:
        current["credit_score"] = (current["credit_score"] - 40).clip(lower=300)
    if concept_drift:
        rng = np.random.RandomState(0)
        index = current.sample(frac=0.30, random_state=0).index
        current.loc[index, TARGET] = rng.permutation(current.loc[index, TARGET].values)
    return current.reset_index(drop=True)


def model_performance(current):
    """F1 macro de la version validée sur les données récentes, si la classe est connue."""
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    version, alias = None, ALIAS_VALIDATED
    for alias in (ALIAS_VALIDATED, ALIAS_CANDIDATE):
        try:
            version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
            break
        except MlflowException:
            continue
    if version is None:
        return {"disponible": False, "raison": "aucun modèle dans le registre"}

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}")
    classes = version.tags.get("classes", "").split(",")
    mapping = {name: index for index, name in enumerate(classes)}
    labelled = current[current[TARGET].astype(str).isin(mapping)]
    y_true = labelled[TARGET].astype(str).map(mapping).values
    y_pred = model.predict(labelled[FEATURES]).astype(int)

    reference_f1 = client.get_run(version.run_id).data.metrics["f1_macro"]
    current_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    recalls = recall_score(y_true, y_pred, labels=list(range(len(classes))), average=None, zero_division=0)
    drop = reference_f1 - current_f1
    return {
        "disponible": True,
        "version": version.version,
        "alias": alias,
        "f1_macro_reference": round(reference_f1, 4),
        "f1_macro_actuel": round(current_f1, 4),
        "baisse": round(drop, 4),
        "rappel_par_classe": {name: round(float(value), 4) for name, value in zip(classes, recalls)},
        "lignes_evaluees": int(len(labelled)),
        "niveau": level_from_drop(drop),
    }


ACTIONS = {
    "ok": "Aucune action. Continuer la surveillance au rythme prévu.",
    "warning": (
        "Surveiller de plus près (relancer la surveillance sous 7 jours), identifier les "
        "variables concernées et vérifier la qualité des données entrantes."
    ),
    "critical": (
        "Analyser d'abord la cause (erreur de collecte ou vrai changement de comportement). "
        "Si le changement est réel : réentraîner avec train_mlflow.py sur des données récentes, "
        "comparer au modèle validé, puis promouvoir seulement si les critères sont respectés. "
        "En cas de baisse de performance sévère, revenir à la version précédente "
        "(registry.py rollback). Ne jamais remplacer le modèle automatiquement."
    ),
}


def write_reports(report):
    MONITORING_JSON.parent.mkdir(exist_ok=True)
    MONITORING_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7, 3.8))
    rows = sorted(report["derive_donnees"], key=lambda row: row["psi"])
    colors = {"ok": "#4c9a2a", "warning": "#e8a317", "critical": "#c0392b"}
    axis.barh([row["variable"] for row in rows], [row["psi"] for row in rows],
              color=[colors[row["niveau"]] for row in rows])
    axis.axvline(PSI_WARNING, color="#e8a317", linestyle="--", label=f"avertissement ({PSI_WARNING})")
    axis.axvline(PSI_CRITICAL, color="#c0392b", linestyle="--", label=f"critique ({PSI_CRITICAL})")
    axis.set_xlabel("PSI (dérive des données)")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(MONITORING_PNG, dpi=130)
    plt.close(figure)

    lines = [
        "# Rapport de surveillance du modèle",
        "",
        f"**Statut global : {report['statut_global'].upper()}**",
        "",
        "## Dérive des données",
        "",
        "| Variable | PSI | p-value KS | Niveau |",
        "|---|---|---|---|",
    ]
    for row in report["derive_donnees"]:
        ks = "—" if row["ks_p_value"] is None else row["ks_p_value"]
        lines.append(f"| {row['variable']} | {row['psi']} | {ks} | {row['niveau']} |")
    perf = report["performance"]
    lines += ["", "## Performance", ""]
    if perf.get("disponible"):
        lines += [
            f"- Version évaluée : {perf['version']} (alias {perf['alias']})",
            f"- F1 macro de référence : {perf['f1_macro_reference']}",
            f"- F1 macro actuel : {perf['f1_macro_actuel']} (baisse : {perf['baisse']})",
            f"- Rappel par classe : {perf['rappel_par_classe']}",
            f"- Niveau : {perf['niveau']}",
        ]
    else:
        lines.append(f"Non évaluée : {perf.get('raison', 'classe réelle absente des données')}.")
    lines += ["", "## Action recommandée", "", ACTIONS[report["statut_global"]], ""]
    MONITORING_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Surveillance du modèle de risque d'impayé")
    parser.add_argument("--current", help="CSV de données récentes (colonnes du modèle).")
    parser.add_argument("--simulate-drift", action="store_true", help="Démonstration avec dérive simulée.")
    parser.add_argument("--simulate-concept-drift", action="store_true",
                        help="Avec --simulate-drift : mélange aussi une partie des vraies classes.")
    args = parser.parse_args()

    reference = load_reference()
    if args.simulate_drift:
        current = simulated_current_data(args.simulate_concept_drift)
    elif args.current:
        current = pd.read_csv(args.current)
    else:
        parser.error("Indiquez --current fichier.csv ou --simulate-drift")

    manquantes = [column for column in FEATURES if column not in current.columns]
    if manquantes:
        raise SystemExit(f"Colonnes absentes des données récentes : {manquantes}")

    drift = data_drift(reference, current)
    if TARGET in current.columns:
        performance = model_performance(current)
    else:
        performance = {"disponible": False, "raison": "la colonne de la vraie classe est absente"}

    levels = [row["niveau"] for row in drift]
    if performance.get("disponible"):
        levels.append(performance["niveau"])
    global_level = max(levels, key=lambda level: SEVERITY[level])

    report = {
        "lignes_recentes": int(len(current)),
        "derive_donnees": drift,
        "performance": performance,
        "statut_global": global_level,
        "action_recommandee": ACTIONS[global_level],
    }
    write_reports(report)

    print(f"Statut global : {global_level.upper()}")
    for row in sorted(drift, key=lambda item: -item["psi"]):
        print(f"  {row['variable']:32s} PSI = {row['psi']:.3f}  [{row['niveau']}]")
    if performance.get("disponible"):
        print(f"  F1 macro : {performance['f1_macro_reference']} -> {performance['f1_macro_actuel']} "
              f"[{performance['niveau']}]")
    print(f"Action : {ACTIONS[global_level]}")
    print(f"Rapports écrits : {MONITORING_MD.name}, {MONITORING_JSON.name}, {MONITORING_PNG.name}")
    return SEVERITY[global_level]


if __name__ == "__main__":
    sys.exit(main())
