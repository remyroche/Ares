import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v4_divergence", ROOT / "scripts" / "diagnose_v4_alpha_execution_ev_divergence.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)


def _rows(lineage: str, grade: str, start: str) -> pd.DataFrame:
    values = []
    for index in range(10):
        values.append({
            "candidate_id": f"{lineage}-{index}", "__ts__": pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=index),
            "__symbol__": f"asset{index % 2}", "side_name": "long" if index < 8 else "short",
            "__first_touch_target_soft__": index / 9, "execution_gross_ev_12h": 0.02 - index / 1000,
            "execution_cost_return": 0.01, "execution_net_ev_12h": 0.01 - index / 1000,
            "alpha_score": float(index), "residual_score": float(9 - index),
            "lineage_id": lineage, "evidence_grade": grade,
        })
    return pd.DataFrame(values)


def test_top10_is_global_not_timestamp_or_side_local():
    rows = _rows("a", "A", "2025-01-01")
    tables, _, _ = MOD.build_tables(rows, cadences=("month",))
    selected = tables.loc[(tables.score_family.eq("alpha")) & (tables.scope.eq("pooled_global_top10")) & (tables.side_name.eq("all"))].iloc[0]
    assert selected.selected_rows == 1
    assert selected.candidate_rows == 10
    # Highest alpha row is short; selection therefore cannot be a long/short quota.
    selected_short = tables.loc[(tables.score_family.eq("alpha")) & (tables.scope.eq("pooled_global_top10")) & (tables.side_name.eq("short"))]
    assert len(selected_short) == 1


def test_lineages_are_never_pooled_and_deciles_keep_score_families_separate():
    rows = pd.concat([_rows("a", "A", "2025-01-01"), _rows("b", "B", "2025-01-01")], ignore_index=True)
    periods, deciles, lineage = MOD.build_tables(rows, cadences=("month",))
    assert set(lineage.lineage_id) == {"a", "b"}
    global_rows = periods.loc[(periods.scope.eq("all_candidates")) & (periods.side_name.eq("all"))]
    assert set(global_rows.candidate_rows) == {10}
    assert set(deciles.score_family) == {"alpha", "residual"}
    assert deciles.groupby(["lineage_id", "score_family"]).size().eq(10).all()


def test_metric_exposes_first_touch_execution_divergence_and_reconciliation():
    rows = _rows("a", "A", "2025-01-01")
    metrics = MOD.metric_row(rows, selected=False)
    assert metrics["alpha_rank_ic"] > 0.99
    assert metrics["alpha_to_net_rank_ic"] < -0.99
    assert metrics["gross_cost_net_reconciles"]
