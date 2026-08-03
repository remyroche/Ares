import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "data_perp/artifacts/pre2026_regime_overlay_gamma_hpo_20260730_v2"
)


def test_gamma_hpo_v2_is_sealed_and_fail_closed():
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    expected = (ARTIFACT / "manifest.sha256").read_text().split()[0]
    actual = hashlib.sha256((ARTIFACT / "manifest.json").read_bytes()).hexdigest()

    assert actual == expected
    assert manifest["promotion_eligible"] is False
    assert manifest["contract"]["selected_gamma"] is None
    assert manifest["contract"]["authorized_for_2026"] is False
    assert manifest["contract"]["decision_cadence"] == "1h"
    assert manifest["contract"]["exact_replay_bar_cadence"] == "1m_labels_only"

    eligibility = pd.read_csv(ARTIFACT / "eligibility.csv")
    assert eligibility["gamma"].tolist() == [0.125, 0.25, 0.5]
    assert not eligibility["eligible"].any()
    assert (eligibility["positive_auc_fraction"] >= 0.75).all()
    assert (eligibility["economic_improvement_fraction"] < 0.625).all()


def test_gamma_half_exactly_reproduces_audited_overlay():
    parity = json.loads((ARTIFACT / "parity_audit.json").read_text())
    assert parity["rows"] == 68_234
    assert parity["candidate_sets_equal"] is True
    assert parity["core_max_abs_error"] == 0.0
    assert parity["gamma_half_max_abs_error"] == 0.0
