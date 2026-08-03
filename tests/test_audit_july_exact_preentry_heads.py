import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_july_exact_preentry_heads import (
    IDENTITY,
    evaluate,
    prepare_joined,
    sha256,
    validate_manifest_hash,
)


def _frames(rows: int = 40):
    index = np.arange(rows)
    side = np.where(index < rows // 2, "long", "short")
    timestamp = pd.date_range("2026-07-20", periods=rows // 2, freq="h", tz="UTC")
    timestamp = np.concatenate([timestamp, timestamp])
    candidates = pd.DataFrame(
        {
            "candidate_id": [f"candidate-{value}" for value in index],
            "__ts__": timestamp,
            "__symbol__": [f"ASSET{value % 7}/USD:USD" for value in index],
            "side_name": side,
        }
    )
    packb = candidates.assign(
        execution_decision_utc=pd.to_datetime(timestamp, utc=True) + pd.Timedelta(hours=1),
        base_prediction=index / rows,
        base_alpha_ev=(index / rows - 0.5) / 100,
        residual_delta_ev=(index / rows - 0.4) / 200,
        existing_alpha_ev=(index / rows - 0.45) / 100,
    )
    path_probability = np.tile(
        np.array([0.10, 0.08, 0.18, 0.16, 0.20, 0.12, 0.16]), (rows, 1)
    )
    path_probability[:, 2] += index / rows * 0.05
    path_probability[:, 6] -= index / rows * 0.05
    preentry = candidates.assign(
        oof_clean_favorable_probability=np.linspace(0.1, 0.9, rows),
        pred_peak_MFE_12h_ATR=np.linspace(0.2, 3.0, rows),
        catboost_archetype=np.where(index % 2, "slow_grinder", "dead_timeout"),
        **{
            f"catboost_p_{position}": path_probability[:, position]
            for position in range(7)
        },
    )
    net = np.linspace(-0.03, 0.03, rows)
    scored = candidates.assign(
        final_direct_net_raw=net * 0.8,
        final_capture_probability=np.linspace(0.05, 0.95, rows),
        mapped_execution_ev=net * 0.9,
    )
    labels = candidates.assign(
        execution_gross_ev_12h=net + 0.01,
        execution_cost_return=0.01,
        execution_net_ev_12h=net,
        execution_exit_reason=np.where(index % 3, "trailing", "timeout"),
        execution_exit_hour=1 + index % 12,
        execution_mfe_return_12h=np.linspace(0.0, 0.05, rows),
        execution_mae_return_12h=np.linspace(0.03, 0.0, rows),
        execution_label_end_utc=pd.to_datetime(timestamp, utc=True)
        + pd.Timedelta(hours=13),
    )
    geometry = candidates.assign(
        __barrier_pct__=0.01,
        __path_auxiliary_atr_fraction__=0.01,
    )
    return packb, preentry, scored, labels, geometry


def test_prepare_joined_uses_exact_atr_and_keeps_path_truth_unfabricated():
    joined = prepare_joined(*_frames())
    assert len(joined) == 40
    assert "path_class_truth" not in joined
    assert "exact_peak_mfe_atr_raw" in joined
    assert "meaningful_mfe_reached" in joined
    expected = (
        joined["execution_mfe_return_12h"]
        >= np.maximum(1.5 * joined["__path_auxiliary_atr_fraction__"], 0.015)
    ).astype(np.int8)
    np.testing.assert_array_equal(joined["meaningful_mfe_reached"], expected)
    np.testing.assert_allclose(
        joined["exact_peak_mfe_atr_canonical"],
        np.where(
            expected > 0,
            np.clip(
                joined["execution_mfe_return_12h"]
                / joined["__path_auxiliary_atr_fraction__"],
                0,
                10,
            ),
            0,
        ),
    )


def test_prepare_joined_fails_closed_on_identity_or_probability_errors():
    frames = list(_frames())
    frames[1] = frames[1].iloc[:-1].copy()
    with pytest.raises(ValueError, match="identities differ"):
        prepare_joined(*frames)

    frames = list(_frames())
    frames[1].loc[0, "catboost_p_0"] = 0.9
    with pytest.raises(ValueError, match="do not sum"):
        prepare_joined(*frames)


def test_evaluate_reports_global_top10_and_conditional_peak_diagnostics():
    joined = prepare_joined(*_frames())
    result = evaluate(joined)
    metrics = result["head_metrics"]
    pooled = metrics.loc[metrics["scope"].eq("pooled")]
    assert {
        "base_raw_alpha_score",
        "residual_delta_ev",
        "existing_alpha_ev",
        "clean_meaningful_event_probability",
        "peak_mfe_unconditional",
        "peak_mfe_conditional_magnitude",
        "direct_execution_ev",
        "capture_probability",
        "mapped_execution_ev",
        "path_probability__immediate_adverse_path",
    }.issubset(set(pooled["head"]))
    base = pooled.loc[pooled["head"].eq("base_raw_alpha_score")].iloc[0]
    assert base["top10_rows"] == 4
    assert base["top10_selection_scope"] == "one_global_pool_within_reported_scope"
    capture = pooled.loc[pooled["head"].eq("capture_probability")].iloc[0]
    assert capture["binary_target"] == "exact_net_positive"
    assert capture["auc"] > 0.99
    diagnostics = result["diagnostics"]
    assert diagnostics["diagnosis"].str.len().gt(0).all()
    assert set(result["daily_stability"]["days"]) == {1}


def test_manifest_binding_fails_on_changed_input(tmp_path: Path):
    data = tmp_path / "input.bin"
    data.write_bytes(b"bound")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "test_schema", "output": {"sha256": sha256(data)}}),
        encoding="utf-8",
    )
    binding = validate_manifest_hash(
        data, manifest, ("output", "sha256"), expected_schema="test_schema"
    )
    assert binding["sha256"] == sha256(data)
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest_hash(
            data, manifest, ("output", "sha256"), expected_schema="test_schema"
        )
