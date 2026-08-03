from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_hurdle_cross_side_common_unit_mapping.py"
SPEC = importlib.util.spec_from_file_location("hurdle_common_unit", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(rows: int = 600, start: str = "2026-05-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    score = np.linspace(-.02, .02, rows)
    return pd.DataFrame({"candidate_id": [f"id-{i:04d}" for i in range(rows)], "__ts__": ts, "__symbol__": ["A" if i % 3 else "B" for i in range(rows)], "side_name": np.where(np.arange(rows) % 2, "long", "short"), "execution_decision_utc": ts, "support_label_available_utc": ts + pd.Timedelta(hours=12), "execution_net_ev_12h": score + np.where(np.arange(rows) % 2, .001, -.001), "gross_cost_hurdle_ev": score})


def test_causal_map_excludes_same_day_and_unresolved_rows() -> None:
    frame = _frame()
    target = frame.loc[frame.__ts__.dt.floor("D").eq(pd.Timestamp("2026-05-20", tz="UTC"))].copy()
    mapped, audit = MODULE.causal_map(frame, target, shrink=1000.0)
    assert np.isfinite(mapped.mapped_score).all()
    row = audit.iloc[0]
    assert row.reference_rows < len(frame.loc[frame.__ts__.lt(pd.Timestamp("2026-05-20", tz="UTC"))])
    assert mapped.map_status.isin(["pooled_plus_shrunk_side_residual", "pooled_anchor", "zero_fallback_weak_global"]).all()


def test_weak_global_support_is_exact_zero_not_raw_fallback() -> None:
    frame = _frame(20)
    target = frame.iloc[-3:].copy()
    mapped, _ = MODULE.causal_map(frame, target, shrink=None)
    assert np.array_equal(mapped.mapped_score.to_numpy(float), np.zeros(len(target)))
    assert mapped.map_status.eq("zero_fallback_weak_global").all()


def test_global_top_has_no_side_quota_and_reports_ties() -> None:
    frame = _frame(100)
    frame["mapped_score"] = 1.0
    selected = MODULE.stable_top(frame, "mapped_score", .1)
    assert selected.candidate_id.tolist() == sorted(frame.candidate_id.tolist())[:10]
    tie = MODULE._tie_metrics(frame, "mapped_score", .1)
    assert tie["cutoff_tie_ambiguous"] and tie["cutoff_plateau_rows"] == 100
