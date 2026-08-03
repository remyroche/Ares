from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_exact12h_legacy24h_base_label_parity.py"
SPEC = importlib.util.spec_from_file_location("exact12h_legacy24h_parity", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sources(rows: int = 80) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    legacy_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    exact_rows: list[dict[str, object]] = []
    for number in range(rows):
        ts = pd.Timestamp("2025-02-01T00:00:00Z") + pd.Timedelta(hours=number)
        decision = ts + pd.Timedelta(hours=1)
        side = "long" if number % 2 else "short"
        identity = {"candidate_id": f"id-{number:03d}", "side_name": side, "__symbol__": f"A{number % 3}", "__ts__": ts}
        target = number / rows
        gross = (number - 30) / 1_000
        cost = .01
        legacy_rows.append({**identity, "__decision_ts__": decision, "base_label_resolution_utc": decision + pd.Timedelta(hours=24), "__first_touch_target_soft__": target, "base_oof_score": target})
        native_rows.append({**identity, "__decision_ts__": decision, "base_label_resolution_utc": decision + pd.Timedelta(hours=12), "target_12h": target, "base_oof_score": target + .01})
        exact_rows.append({**identity, "execution_decision_utc": decision, "execution_label_end_utc": decision + pd.Timedelta(hours=12), "execution_label_available_at": decision + pd.Timedelta(hours=12), "execution_gross_ev_12h": gross, "execution_cost_return": cost, "execution_net_ev_12h": gross - cost})
    return pd.DataFrame(legacy_rows), pd.DataFrame(native_rows), pd.DataFrame(exact_rows)


def test_identical_panel_asserts_horizons_and_economics() -> None:
    legacy, native, exact = _sources()
    panel = MODULE.build_identical_panel(MODULE._load_legacy_frame(legacy), MODULE._load_native_frame(native), MODULE._load_exact_frame(exact))
    assert len(panel) == 80
    assert panel.candidate_month.tolist()[0] == "2025-02"
    assert panel.exact12h_opportunity.sum() > 0
    assert np.allclose(panel.exact12h_gross_return - panel.exact12h_cost_return, panel.exact12h_net_return)


def test_identical_panel_rejects_candidate_or_label_horizon_mismatch() -> None:
    legacy, native, exact = _sources()
    with pytest.raises(ValueError, match="candidate IDs"):
        MODULE.build_identical_panel(MODULE._load_legacy_frame(legacy), MODULE._load_native_frame(native.iloc[1:].copy()), MODULE._load_exact_frame(exact))
    bad_native = native.copy()
    bad_native.loc[0, "base_label_resolution_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="native target"):
        MODULE.build_identical_panel(MODULE._load_legacy_frame(legacy), MODULE._load_native_frame(bad_native), MODULE._load_exact_frame(exact))


def test_tail_selection_is_pooled_global_and_side_is_only_attribution() -> None:
    legacy, native, exact = _sources()
    panel = MODULE.build_identical_panel(MODULE._load_legacy_frame(legacy), MODULE._load_native_frame(native), MODULE._load_exact_frame(exact))
    tails = MODULE.tail_table(panel)
    pooled = tails.query("scope == 'monthly' and score == 'legacy24h_oof_score' and fraction == .1 and side_name == 'all'")
    assert len(pooled) == 1 and pooled.selected_rows.iloc[0] == 8
    side = tails.query("scope == 'monthly' and score == 'legacy24h_oof_score' and fraction == .1 and side_name != 'all'")
    assert side.selected_rows.sum() == pooled.selected_rows.iloc[0]
    assert set(tails.fraction.unique()) == {0.01, 0.05, 0.1, 0.2}


def test_ventiles_include_explicit_cost_opportunity_and_conditional_payoffs() -> None:
    legacy, native, exact = _sources()
    panel = MODULE.build_identical_panel(MODULE._load_legacy_frame(legacy), MODULE._load_native_frame(native), MODULE._load_exact_frame(exact))
    ventiles = MODULE.ventile_table(panel)
    monthly = ventiles.query("scope == 'monthly' and side_name == 'all' and score == 'native12h_oof_score'")
    assert monthly.score_ventile.nunique() == 20
    assert {"mean_cost_bps", "opportunity_recall", "conditional_favorable_net_bps", "conditional_adverse_net_bps"}.issubset(ventiles.columns)
