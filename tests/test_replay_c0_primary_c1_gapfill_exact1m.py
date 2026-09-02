from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/replay_c0_primary_c1_gapfill_exact1m.py"
SPEC = importlib.util.spec_from_file_location("gapfill", MODULE_PATH)
assert SPEC and SPEC.loader
gapfill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gapfill)


def _candidate(candidate_id: str, decision: str, priority: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "decision_timestamp": pd.Timestamp(decision, tz="UTC"),
        "timestamp": pd.Timestamp(decision, tz="UTC") + pd.Timedelta(minutes=5),
        "symbol": candidate_id.split("|")[0],
        "side": "long",
        "portfolio_priority_adjustment": priority,
    }


def test_c1_only_fills_c0_empty_target_free_timestamps() -> None:
    c0_target_free = pd.DataFrame({
        "candidate_id": ["A|long|0"],
        "timestamp": [pd.Timestamp("2026-05-01T00:00Z")],
    })
    c0 = pd.DataFrame([_candidate("A|long|0", "2026-05-01T00:00", 80.0)])
    c1 = pd.DataFrame([
        _candidate("B|long|0", "2026-05-01T00:00", 100.0),
        _candidate("C|long|1", "2026-05-01T01:00", 90.0),
    ])
    route, audit = gapfill.select_c0_primary_c1_gapfill(
        c0_target_free=c0_target_free,
        c1_target_free=pd.DataFrame({
            "candidate_id": ["B|long|0", "C|long|1"],
            "timestamp": [pd.Timestamp("2026-05-01T00:00Z"), pd.Timestamp("2026-05-01T01:00Z")],
        }),
        c0_exact_candidates=c0,
        c1_exact_candidates=c1,
    )
    assert set(route["candidate_id"]) == {"A|long|0", "C|long|1"}
    assert audit["c1_gapfill_evaluable_rows"] == 1


def test_rejects_c0_candidate_not_in_target_free_admissions() -> None:
    c0_target_free = pd.DataFrame({
        "candidate_id": ["A|long|0"],
        "timestamp": [pd.Timestamp("2026-05-01T00:00Z")],
    })
    c0 = pd.DataFrame([_candidate("B|long|0", "2026-05-01T00:00", 80.0)])
    c1 = pd.DataFrame()
    try:
        gapfill.select_c0_primary_c1_gapfill(
            c0_target_free=c0_target_free,
            c1_target_free=pd.DataFrame(columns=["candidate_id", "timestamp"]),
            c0_exact_candidates=c0,
            c1_exact_candidates=c1,
        )
    except AssertionError as exc:
        assert "subset" in str(exc)
    else:
        raise AssertionError("expected a target-free identity failure")


def test_resolves_current_direct_c1_candidate_alias(tmp_path: Path) -> None:
    c0 = tmp_path / "C0_refit_core_postfeb_portfolio_candidates.parquet"
    c1 = tmp_path / "C1_refit_core_plus_causal_sr_portfolio_candidates.parquet"
    c0.touch()
    c1.touch()

    assert gapfill._resolve_exact_candidate_panel(tmp_path, arm="C0") == c0
    assert gapfill._resolve_exact_candidate_panel(tmp_path, arm="C1") == c1


def test_rejects_ambiguous_c1_candidate_aliases(tmp_path: Path) -> None:
    (tmp_path / "C1_refit_core_plus_causal_sr_portfolio_candidates.parquet").touch()
    (tmp_path / "C1_LVA_refit_core_plus_causal_sr_portfolio_candidates.parquet").touch()

    try:
        gapfill._resolve_exact_candidate_panel(tmp_path, arm="C1")
    except FileNotFoundError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("expected ambiguous C1 alias to fail closed")
