"""Regression checks for the corrected robust-control support-head screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts import run_bounded_robust_auxiliary_contribution_ablation as subject


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_perp/artifacts/bounded_robust_auxiliary_contribution_ablation_20260730_v2"
SIDE = ROOT / "data_perp/artifacts/bounded_robust_auxiliary_contribution_ablation_20260730_v2_provenance_20260730_v1"
GATES = ROOT / "data_perp/artifacts/bounded_robust_auxiliary_contribution_ablation_20260730_v2_gates_20260730_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def test_control_is_bit_identical_to_v2_and_weight_selection_never_uses_map() -> None:
    parity = json.loads((OUT / "control_parity.json").read_text())
    assert parity["bit_identical"] is True
    assert parity["max_abs_delta"] == 0.0
    metrics = pd.read_csv(OUT / "global_metrics.csv")
    development = metrics.loc[metrics["stage"].eq("development_oof_raw_only")]
    assert set(development["score_kind"]) == {"raw"}
    choice = pd.read_csv(OUT / "march_oof_raw_weight_selection.csv")
    assert "march_oof_raw_top10_net_bps" in choice
    assert not any("mapped" in column for column in choice.columns)


def test_march_cutoff_is_causal_and_slope_is_sealed() -> None:
    frame = subject.load(type("Args", (), {"source": subject.SRC, "peak": subject.PEAK, "slope": subject.SLOPE})())
    cutoff = pd.Timestamp("2025-04-01T01:00:00Z")
    development = frame.loc[frame.candidate_month.eq("2025-03") & frame[subject.END].lt(cutoff)]
    assert len(development) > 0
    assert development[subject.END].max() < cutoff
    seal = json.loads((SIDE / "slope_detached_seal.json").read_text())
    slope_root = ROOT / "data_perp/artifacts/febapr2025_historical_future_slope_fixed_geometry_oof_20260730_v1"
    assert seal["oof_predictions_sha256"] == _sha(slope_root / "oof_predictions.parquet")
    assert seal["artifact_manifest_sha256"] == _sha(slope_root / "manifest.json")
    assert len(seal["folds_sha256"]) == 2


def test_mapped_tie_gate_uses_expected_not_deterministic_outcome() -> None:
    ties = pd.read_csv(GATES / "tie_bounds.csv")
    mapped_top1 = ties.loc[(ties["score_kind"].eq("mapped")) & (ties["top_fraction"].eq(0.01))].iloc[0]
    assert mapped_top1["cutoff_tie_rows"] > mapped_top1["rows"]
    assert mapped_top1["deterministic_net_bps"] > 0.0
    assert mapped_top1["random_tie_expected_net_bps"] < 0.0
