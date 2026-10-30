"""
Gestion du registre de modèles MLflow : version candidate, version validée.

Règles
- Chaque entraînement enregistré crée une nouvelle version, alias « candidate ».
- Une version devient « validated » seulement si elle passe les critères
  de validation (voir mlflow_settings.py) : F1 macro, AUC, et pas de régression
  par rapport à la version déjà validée.
- L'ancienne version validée reçoit l'alias « previous » : on peut y revenir.

Commandes (depuis le dossier du projet) :
    python src/telecom_mlops/registry.py status
    python src/telecom_mlops/registry.py promote              # promeut la version « candidate »
    python src/telecom_mlops/registry.py promote --version 3  # promeut une version précise
    python src/telecom_mlops/registry.py rollback             # revient à la version « previous »
"""

import argparse
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlflow_settings import (
    ALIAS_CANDIDATE,
    ALIAS_PREVIOUS,
    ALIAS_VALIDATED,
    MAX_F1_REGRESSION,
    MIN_AUC_MACRO,
    MIN_F1_MACRO,
    RECALL_ALERT_PER_CLASS,
    REGISTERED_MODEL_NAME,
    TRACKING_URI,
)


def get_client():
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def version_by_alias(client, alias):
    """Retourne la version qui porte l'alias, ou None."""
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    except MlflowException:
        return None


def run_metrics(client, version):
    return client.get_run(version.run_id).data.metrics


def check_gate(candidate_metrics, validated_metrics=None):
    """
    Applique les critères de validation.
    Retourne (ok, blocages, avertissements).
    """
    blockers, warnings = [], []
    f1 = candidate_metrics.get("f1_macro", 0.0)
    auc = candidate_metrics.get("auc_macro_ovr", 0.0)

    if f1 < MIN_F1_MACRO:
        blockers.append(f"F1 macro {f1:.3f} < seuil minimal {MIN_F1_MACRO:.2f}")
    if auc < MIN_AUC_MACRO:
        blockers.append(f"AUC macro {auc:.3f} < seuil minimal {MIN_AUC_MACRO:.2f}")
    if validated_metrics:
        reference = validated_metrics.get("f1_macro", 0.0)
        if f1 < reference - MAX_F1_REGRESSION:
            blockers.append(
                f"régression : F1 macro {f1:.3f} contre {reference:.3f} pour la version validée"
            )

    # Avertissements : n'empêchent pas la validation mais doivent être documentés.
    for name, value in candidate_metrics.items():
        if name.startswith("recall_") and name != "recall_macro" and value < RECALL_ALERT_PER_CLASS:
            warnings.append(f"{name} = {value:.3f} : classe mal détectée (limite connue)")
    return (not blockers), blockers, warnings


def show_status():
    client = get_client()
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    if not versions:
        print("Aucune version dans le registre. Lancez d'abord train_mlflow.py.")
        return
    aliases = {}
    for alias in (ALIAS_CANDIDATE, ALIAS_VALIDATED, ALIAS_PREVIOUS):
        found = version_by_alias(client, alias)
        if found:
            aliases.setdefault(str(found.version), []).append(alias)

    print(f"Modèle : {REGISTERED_MODEL_NAME}")
    print(f"{'version':8s}{'alias':22s}{'stage':12s}{'f1_macro':10s}{'auc':8s}{'recall medium':14s}")
    for version in sorted(versions, key=lambda item: int(item.version)):
        metrics = run_metrics(client, version)
        print(
            f"{str(version.version):8s}"
            f"{','.join(aliases.get(str(version.version), [])) or '-':22s}"
            f"{version.tags.get('stage', '-'):12s}"
            f"{metrics.get('f1_macro', float('nan')):<10.4f}"
            f"{metrics.get('auc_macro_ovr', float('nan')):<8.4f}"
            f"{metrics.get('recall_medium', float('nan')):<14.4f}"
        )


def promote(version_number=None):
    client = get_client()
    if version_number:
        candidate = client.get_model_version(REGISTERED_MODEL_NAME, str(version_number))
    else:
        candidate = version_by_alias(client, ALIAS_CANDIDATE)
    if candidate is None:
        print("Aucune version candidate. Lancez train_mlflow.py.")
        return 1

    validated = version_by_alias(client, ALIAS_VALIDATED)
    if validated and str(validated.version) == str(candidate.version):
        print(f"La version {candidate.version} est déjà validée.")
        return 0

    ok, blockers, warnings = check_gate(
        run_metrics(client, candidate),
        run_metrics(client, validated) if validated else None,
    )
    print(f"Évaluation de la version {candidate.version} :")
    for message in warnings:
        print(f"  [avertissement] {message}")
    if not ok:
        for message in blockers:
            print(f"  [REFUSÉ] {message}")
        print("La version reste « candidate ».")
        return 1

    if validated:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_PREVIOUS, validated.version)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, validated.version, "stage", "archived")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_VALIDATED, candidate.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, candidate.version, "stage", "production")
    still_candidate = version_by_alias(client, ALIAS_CANDIDATE)
    if still_candidate and str(still_candidate.version) == str(candidate.version):
        client.delete_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CANDIDATE)
    print(f"Version {candidate.version} VALIDÉE (alias « {ALIAS_VALIDATED} », stage production).")
    return 0


def rollback():
    client = get_client()
    previous = version_by_alias(client, ALIAS_PREVIOUS)
    validated = version_by_alias(client, ALIAS_VALIDATED)
    if previous is None:
        print("Pas de version « previous » : retour arrière impossible.")
        return 1
    if validated:
        client.set_model_version_tag(REGISTERED_MODEL_NAME, validated.version, "stage", "rejected")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_VALIDATED, previous.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, previous.version, "stage", "production")
    client.delete_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_PREVIOUS)
    print(f"Retour arrière : la version {previous.version} est de nouveau validée.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Registre de modèles MLflow")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Afficher les versions et leurs alias.")
    promote_parser = sub.add_parser("promote", help="Valider une version candidate.")
    promote_parser.add_argument("--version", type=int, default=None)
    sub.add_parser("rollback", help="Revenir à la version précédente.")
    args = parser.parse_args()

    if args.command == "status":
        show_status()
        return 0
    if args.command == "promote":
        return promote(args.version)
    return rollback()


if __name__ == "__main__":
    sys.exit(main())
