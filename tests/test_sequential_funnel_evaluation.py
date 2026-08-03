from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.sequential_funnel_evaluation import SequentialFunnelEvaluationError, evaluate_funnel_trials, validate_nested_oof_provenance


def _manifest() -> dict:
    return {"trials": [
        {"trial_id": "T0", "stage": "target_screen", "target_family": "control", "development_only": True, "description": "control", "inference_features": ["atr"]},
        {"trial_id": "T2", "stage": "target_screen", "target_family": "soft_barrier", "development_only": True, "description": "barrier", "uses_gam": True, "inference_features": ["atr"]},
    ]}


def _frame() -> pd.DataFrame:
    rows = []
    stamp = pd.Timestamp("2025-01-01T00:00:00Z")
    for trial in ("T0", "T2"):
        for i, score in enumerate((.9, .8, .1, .2, .3, .4, .5, .6, .7, .0)):
            decision = stamp + pd.Timedelta(hours=i)
            row = {"candidate_id": f"c{i}", "trial_id": trial, "score": score, "__decision_ts__": decision,
                   "__label_available_at__": decision + pd.Timedelta(hours=12), "strict_prequential_oof": True,
                   "execution_gross_ev_12h": .03 if i < 2 else -.01, "execution_cost_return": .01,
                   "execution_net_ev_12h": .02 if i < 2 else -.02, "side_name": "long" if i < 5 else "short"}
            for layer in ("base", "meta", "gam"):
                row[f"{layer}_prediction_fit_end_ts"] = decision - pd.Timedelta(hours=1)
                row[f"{layer}_prediction_generated_ts"] = decision
                row[f"{layer}_prediction_model_id"] = f"{layer}-1"
                row[f"{layer}_prediction_fold_id"] = "f1"
            rows.append(row)
    return pd.DataFrame(rows)


def test_sequential_evaluator_uses_pooled_global_book_and_reports_trials() -> None:
    tables = evaluate_funnel_trials(_frame(), _manifest())
    tails = tables["base_meta_stack_results"]
    assert set(tails.trial_id) == {"T0", "T2"}
    assert tables["correctness_checks"].passed.all()
    assert tables["sequential_advancement_gates"].trial_id.nunique() == 2


def test_nested_lineage_is_required_for_enabled_layer() -> None:
    frame = _frame().drop(columns=["gam_prediction_fit_end_ts"])
    checks = validate_nested_oof_provenance(frame, _manifest())
    assert not checks.loc[checks.check.eq("nested_oof_gam_lineage"), "passed"].all()
    with pytest.raises(SequentialFunnelEvaluationError, match="provenance"):
        evaluate_funnel_trials(frame, _manifest())


def test_future_teacher_feature_is_rejected() -> None:
    manifest = _manifest(); manifest["trials"][0]["inference_features"] = ["future_mfe"]
    checks = validate_nested_oof_provenance(_frame(), manifest)
    assert not checks.loc[checks.check.eq("future_outputs_excluded_from_inference"), "passed"].all()
