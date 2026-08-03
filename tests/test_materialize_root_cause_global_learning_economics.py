import numpy as np
import pandas as pd

from scripts.materialize_root_cause_global_learning_economics import causal_metric_concordance, compute, exact_global_topk


def _predictions() -> pd.DataFrame:
    rows = []
    for family, seed, offset in (("prior", 1, 0.0), ("causal_capacity_oracle", 1, 10.0), ("production_like_lgbm", 1, 5.0), ("future_feature_oracle", 1, 20.0)):
        for i in range(40):
            rows.append({
                "candidate_id": f"id-{i}", "__ts__": pd.Timestamp("2024-08-01", tz="UTC") + pd.Timedelta(hours=i),
                "side_name": "long" if i % 2 else "short", "gross_h12_bps": float(i), "net_h12_bps": float(i - 2),
                "combined_economic_prediction_bps": float(i + offset), "model_family": family, "seed": seed,
                "split": "later_oos", "evaluation_scope": "outer_heldout",
            })
    return pd.DataFrame(rows)


def test_exact_global_topk_never_expands_ties():
    frame = _predictions().query("model_family == 'prior'").copy()
    frame["combined_economic_prediction_bps"] = 1.0
    selected = exact_global_topk(frame, .10)
    assert len(selected) == 4
    assert selected.candidate_id.tolist() == ["id-0", "id-1", "id-10", "id-11"]


def test_compute_uses_one_global_book_and_named_gaps():
    arms, gaps = compute(_predictions(), bootstrap_reps=20)
    top = arms[arms.top_fraction.eq(.10)]
    assert set(top.selected_rows) == {4}
    assert top.selection_scope.eq("GLOBAL_TOP_K_NOT_PER_TIMESTAMP_OR_SIDE").all()
    assert {"null_to_causal", "production_to_causal", "causal_to_future"} <= set(gaps.comparison)


def test_concordance_excludes_future_oracle_and_uses_global_outcomes():
    arms, _ = compute(_predictions(), bootstrap_reps=5)
    rows = []
    for family in ("prior", "production_like_lgbm", "causal_capacity_oracle", "future_feature_oracle"):
        for side in ("long", "short"):
            rows.append({
                "model_family": family, "seed": 1, "split": "development_oof",
                "evaluation_scope": "outer_heldout", "component": "base_directional", "side": side,
                "base_directional__roc_auc": .5 + .01 * len(family),
                "base_directional__pr_auc": .4, "base_directional__log_loss": .7,
                "base_directional__brier": .25, "base_directional__ece": .1,
                "base_directional__spearman_ic": .01, "base_directional__mae": .4,
                "base_directional__calibration_slope": 1.0,
            })
    result = causal_metric_concordance(pd.DataFrame(rows), arms)
    assert result.arms.eq(3).all()
    assert result.excluded_noncausal_families.str.contains("future_feature_oracle").all()
