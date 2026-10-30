"""Tests de la surveillance (PSI, niveaux) et des critères de validation du registre."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Les scripts MLOps s'importent comme train.py : depuis src/telecom_mlops.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "telecom_mlops"))

from monitoring import level_from_drop, level_from_psi, psi  # noqa: E402
from registry import check_gate  # noqa: E402


def test_psi_is_near_zero_for_identical_data():
    values = np.random.RandomState(0).normal(size=5000)
    assert psi(values, values) < 0.01


def test_psi_detects_a_strong_shift():
    rng = np.random.RandomState(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)) > 0.2


def test_psi_works_for_categories():
    reference = pd.Series(["MTN"] * 70 + ["Glo"] * 30)
    current = pd.Series(["MTN"] * 20 + ["Glo"] * 80)
    assert psi(reference, current) > 0.2


def test_levels():
    assert level_from_psi(0.05) == "ok"
    assert level_from_psi(0.15) == "warning"
    assert level_from_psi(0.35) == "critical"
    assert level_from_drop(0.01) == "ok"
    assert level_from_drop(0.04) == "warning"
    assert level_from_drop(0.10) == "critical"


def test_gate_accepts_a_good_candidate():
    ok, blockers, _ = check_gate({"f1_macro": 0.50, "auc_macro_ovr": 0.70, "recall_medium": 0.30})
    assert ok and not blockers


def test_gate_refuses_low_f1_or_auc():
    ok, blockers, _ = check_gate({"f1_macro": 0.30, "auc_macro_ovr": 0.60})
    assert not ok and len(blockers) == 2


def test_gate_refuses_a_regression():
    ok, blockers, _ = check_gate(
        {"f1_macro": 0.47, "auc_macro_ovr": 0.70},
        validated_metrics={"f1_macro": 0.50},
    )
    assert not ok and any("régression" in message for message in blockers)


def test_gate_warns_about_a_poorly_detected_class_without_blocking():
    ok, _, warnings = check_gate({"f1_macro": 0.50, "auc_macro_ovr": 0.70, "recall_medium": 0.003})
    assert ok and any("recall_medium" in message for message in warnings)
