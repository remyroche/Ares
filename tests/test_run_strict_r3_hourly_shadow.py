from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from extreme_price_movements.strict_r3_inference_bundle import StrictR3InferenceBundle
from scripts.run_strict_r3_hourly_shadow import _commands, _utc_hour


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_strict_r3_hourly_shadow.py"


def test_hourly_orchestrator_has_no_exchange_or_order_imports() -> None:
    tree = ast.parse(PATH.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("ccxt", "kraken", "exchange", "order", "portfolio_manager")
    assert not any(token in name.lower() for name in imported for token in forbidden)


def test_hourly_commands_generate_features_before_filtering_actionable_rows(
    tmp_path,
) -> None:
    payload = json.loads(
        (ROOT / "config/strict_r3_inference_bundle_long_20260716_v2.json").read_text(),
    )
    bundle = StrictR3InferenceBundle(root=ROOT, payload=payload)
    commands = _commands(
        bundle_path=ROOT / "config/strict_r3_inference_bundle_long_20260716_v2.json",
        bundle=bundle,
        state_path=tmp_path / "state.json",
        decision=pd.Timestamp("2026-08-12T09:00:00Z"),
        out_dir=tmp_path / "cycle",
    )
    by_name = {name: command for name, command in commands}
    feature_command = by_name["features"]
    score_command = by_name["shadow_cycle"]
    assert str(tmp_path / "cycle/candidate_grid/target_free_candidate_population.parquet") in feature_command
    assert str(tmp_path / "cycle/candidate_grid/eligible_candidates.parquet") in score_command
    assert "2026-02-01T00:00:00+00:00" in feature_command


def test_hourly_decision_must_be_an_exact_utc_hour() -> None:
    assert _utc_hour("2026-08-12T09:00:00Z") == pd.Timestamp("2026-08-12T09:00:00Z")
    try:
        _utc_hour("2026-08-12T09:01:00Z")
    except ValueError as exc:
        assert "exact UTC hour" in str(exc)
    else:
        raise AssertionError("non-hourly decision timestamp was accepted")


def test_feature_parity_gate_precedes_shadow_scoring() -> None:
    source = PATH.read_text()
    assert source.index("feature_parity_audit =") < source.index(
        "name, command = stages[2]",
    )
