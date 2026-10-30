"""
Paramètres de suivi MLflow, de registre et de surveillance.

Tout ce qui peut être ajusté (noms, seuils) est regroupé ici, pour ne pas
modifier les scripts. Ce fichier complète config.py sans le modifier.
"""

from config import ROOT

# --- MLflow : suivi des entraînements --------------------------------------
# Base SQLite locale (non versionnée dans Git) et dossier des artefacts.
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()
EXPERIMENT_NAME = "impayes-telecom-xgboost"

# --- Registre de modèles ----------------------------------------------------
REGISTERED_MODEL_NAME = "default_risk_xgboost"

# Un alias désigne la version qui joue un rôle. Une version peut en avoir un seul
# de chaque ; un alias ne pointe que vers une seule version à la fois.
ALIAS_CANDIDATE = "candidate"   # version entraînée, pas encore validée (staging)
ALIAS_VALIDATED = "validated"   # version validée, utilisée en production
ALIAS_PREVIOUS = "previous"     # ancienne version validée, gardée pour le retour arrière

# --- Critères de validation (quality gate) ---------------------------------
# Ces seuils sont tirés du premier entraînement (F1 macro 0,456 ; AUC 0,656).
# Ils sont volontairement modestes : à relever si l'issue #2 améliore le modèle.
MIN_F1_MACRO = 0.45
MIN_AUC_MACRO = 0.65
MAX_F1_REGRESSION = 0.01        # le candidat ne doit pas perdre plus de 0,01 de F1 macro
RECALL_ALERT_PER_CLASS = 0.10   # simple avertissement (ne bloque pas) : voir issue #2

# --- Surveillance -----------------------------------------------------------
REFERENCE_SAMPLE_PATH = ROOT / "models" / "reference_sample.csv"
MONITORING_JSON = ROOT / "reports" / "monitoring_report.json"
MONITORING_MD = ROOT / "reports" / "monitoring_report.md"
MONITORING_PNG = ROOT / "reports" / "monitoring_psi.png"

PSI_WARNING = 0.10              # dérive légère
PSI_CRITICAL = 0.20             # dérive forte
KS_P_VALUE = 0.01               # test de Kolmogorov-Smirnov (variables numériques)
F1_DROP_WARNING = 0.03          # baisse du F1 macro par rapport à la version validée
F1_DROP_CRITICAL = 0.06
