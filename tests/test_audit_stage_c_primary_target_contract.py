"""Focused checks for the Stage-C H0 conditional-target readiness audit."""

from __future__ import annotations

import hashlib

import pandas as pd

from scripts.audit_stage_c_primary_target_contract import (
    EXPECTED_ARMS,
    PRIMARY_TARGET,
    _id_hash,
    audit_frames,
)


def _fixture() -> dict[str, object]:
    decision = pd.Timestamp("2024-04-01T00:00:00Z")
    candidate = "c1"
    ids = {
        "target_id": "exact_h12_net_current_frozen_spread_counterfactual_v1",
        "execution_policy_id": "historical_current_frozen_spread_counterfactual_h12_v1",
        "cost_model_id": "current_frozen_spread_counterfactual_row_cost_v1",
    }
    common = {
        "candidate_id": [candidate], "side": ["long"], "decision_ts": [decision],
        "label_end_ts": [decision + pd.Timedelta(hours=12)],
        "label_available_ts": [decision + pd.Timedelta(hours=12)],
    }
    panel = pd.DataFrame({
        **common, **{name: [value] for name, value in ids.items()},
        PRIMARY_TARGET: [1.0], f"{PRIMARY_TARGET}__valid": [1],
        f"{PRIMARY_TARGET}__condition_met": [1],
        f"{PRIMARY_TARGET}__support_side": ["long"],
        f"{PRIMARY_TARGET}__support_month": ["2024-04"],
    })
    persistence = pd.DataFrame({
        **common, "exact_h12_net_bps": [12.0],
        "postcost_h0_clear_first": [1], "postcost_h0_persistence_target_valid": [1],
        "postcost_h0_retained_net": [1], **{name: [value] for name, value in ids.items() if name != "target_id"},
    })
    events = pd.DataFrame({"candidate_id": [candidate], "postcost_h0_event": ["clear_cost_first"]})
    alignment = pd.DataFrame({"candidate_id": [candidate], "exact_h12_net_bps": [12.0], **{name: [value] for name, value in ids.items()}})
    predictions = pd.DataFrame([
        {**common, "month": "2024-04", "label": 1, "exact_h12_net_bps": 12.0,
         "arm": arm, "split": "development_oof", "fold": "2024-04", "prediction": 0.7}
        for arm in EXPECTED_ARMS
    ]).explode(["candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts"], ignore_index=True)
    # Common fields were one-element lists for DataFrame construction above;
    # the remaining arms are already individual rows.
    for field in ("candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts"):
        predictions[field] = predictions[field].astype(object)
    identity = pd.DataFrame([
        {"split": "development_oof", "fold": "2024-04", "arm": arm, "rows": 1,
         "candidate_id_sha256": _id_hash([candidate]), "identical_to_c0": True}
        for arm in EXPECTED_ARMS
    ])
    stability = pd.DataFrame([
        {"arm": arm, "side": "long", "split": "development_oof", "fold": "2024-04",
         "fold_start_utc": decision, "purge_embargo_hours": 12,
         "train_decision_ts_max": decision - pd.Timedelta(hours=13),
         "train_label_available_ts_max": decision - pd.Timedelta(seconds=1),
         "final_oos_labels_used": False, "base_features": "[]", "incremental_selected": "[]", "model_features": "[\"causal_feature\"]"}
        for arm in EXPECTED_ARMS
    ])
    return {
        "panel": panel, "persistence": persistence, "events": events, "alignment": alignment,
        "predictions": predictions, "evaluation_ids": identity, "stability": stability,
        "manifest": {"target": PRIMARY_TARGET, "population": "exact H0 clear-first support only"},
    }


def test_recomputed_h0_target_and_identical_arm_ledger_pass() -> None:
    readiness, coverage = audit_frames(**_fixture())
    assert readiness.status.eq("PASS").all(), readiness.loc[readiness.status.ne("PASS")].to_dict("records")
    assert len(coverage) == len(EXPECTED_ARMS)


def test_target_formula_fails_if_a_clear_positive_net_row_is_relabeled() -> None:
    values = _fixture()
    values["panel"].loc[0, PRIMARY_TARGET] = 0.0
    readiness, _ = audit_frames(**values)
    assert readiness.loc[readiness.check.eq("retain_h0_formula_exact_net_positive_on_clear_support"), "status"].item() == "BLOCKED"


def test_h25_or_continuous_target_cannot_enter_retention_feature_matrix() -> None:
    values = _fixture()
    values["stability"].loc[0, "model_features"] = '["retain_h25_given_clear"]'
    readiness, _ = audit_frames(**values)
    assert readiness.loc[readiness.check.eq("target_columns_never_enter_feature_matrix"), "status"].item() == "BLOCKED"
