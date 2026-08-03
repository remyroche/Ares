from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_canonical_execution_reliability_ic_ev_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("reliability_ic_ev_diagnostic", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    rows = []
    for month in ("2025-03", "2025-04"):
        for index in range(20):
            gross = index / 1_000
            rows.append({
                "candidate_id": f"{month}-{index:02d}", "side_name": "long" if index % 2 else "short",
                "__symbol__": f"A{index % 3}", "__ts__": pd.Timestamp(f"{month}-01", tz="UTC") + pd.Timedelta(hours=index),
                "candidate_month": month, "score_base_alpha": float(index), "score_residual_expected_ev": index / 1_000,
                "direct_q25_return": index / 2_000, "raw_score": index / 3_000,
                "causal_pooled_21d": index / 4_000 if month == "2025-04" else np.nan,
                "causal_pooled_21d_eligible": month == "2025-04", "__first_touch_target_soft__": index / 20,
                "execution_mfe_return_12h": gross + .02, "execution_gross_ev_12h": gross,
                "execution_cost_return": .01, "execution_net_ev_12h": gross - .01,
                "target_economic_opportunity_hard": int(gross > .01), "target_net_positive": int(gross > .01),
                "target_positive_net_magnitude": max(gross - .01, 0.), "target_adverse_loss_magnitude": max(.01 - gross, 0.),
                "exit_is_full_stop": index % 5 == 0, "exit_is_timeout": index % 7 == 0,
                "execution_exit_class": "timeout" if index % 7 == 0 else "trailing",
                "__regime_source_execution_risk_score__": float(index),
            })
    return pd.DataFrame(rows)


def test_stable_top_is_one_pooled_global_book_with_total_ties() -> None:
    frame = _frame().iloc[:3].copy()
    frame["score_base_alpha"] = 1.0
    frame.loc[frame.index[0], "candidate_id"] = "z"
    frame.loc[frame.index[1], "candidate_id"] = "a"
    frame.loc[frame.index[2], "candidate_id"] = "b"
    selected = MODULE.stable_top(frame, "score_base_alpha", .34)
    assert selected.candidate_id.astype(str).tolist() == ["a", "b"]


def test_tail_cost_counterfactuals_use_the_same_frozen_selected_book() -> None:
    frame = _frame().loc[lambda x: x.candidate_month.eq("2025-03")].copy()
    rows, books = MODULE.tail_rows(frame, "base_alpha", "2025-03")
    top10 = next(row for row in rows if row["top_fraction"] == .10)
    selected = next(book for book in books if book.top_fraction.iloc[0] == .10)
    assert top10["selected_rows"] == len(selected)
    assert top10["zero_cost_hurdle_net_bps"] - top10["fixed_100bps_hurdle_net_bps"] == pytest.approx(100.0)
    assert len(top10["selected_identity_sha256"]) == 64


def test_frozen_numeric_bands_do_not_recompute_april_cutoffs() -> None:
    result = MODULE.frozen_numeric_bands(_frame(), "base_alpha")
    source = result.loc[result.role.eq("source_definition")].sort_values("band")
    target = result.loc[result.role.eq("target_application_of_frozen_source_cutoffs")].sort_values("band")
    assert set(source.band) == set(target.band)
    assert source.lower_inclusive_score.tolist() == target.lower_inclusive_score.tolist()
    assert source.upper_score.tolist() == target.upper_score.tolist()


def test_score_cutoff_migration_is_pooled_global_and_does_not_fit_labels() -> None:
    result = MODULE.score_scale_and_cutoff_migration(_frame(), "base_alpha")
    assert set(result.top_fraction) == set(MODULE.TOP_FRACTIONS)
    assert (result.source_rows == 20).all()
    assert (result.target_rows == 20).all()
    assert result.cutoff_definition.str.contains("pooled-global").all()


def test_causal_layer_is_april_only_and_does_not_claim_q25_mapping() -> None:
    result = MODULE.layer_rows(_frame(), "causal_mapped_raw_execution_score")
    assert result.candidate_month.unique().tolist() == ["2025-04"]
    assert "raw_score" in MODULE.LAYERS["causal_mapped_raw_execution_score"]["score_semantics"]
