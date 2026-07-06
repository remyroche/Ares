from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from extreme_price_movements.economic_target_optimizer import (
    EconomicTargetSpec,
    append_economic_target_columns,
    build_economic_target,
    candidate_specs,
    economic_target_column_names,
)
from extreme_price_movements.training import _build_base_regression_target
from scripts.optimize_economic_target import _score_candidate


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "__symbol__": ["BTC", "ETH", "SOL", "XRP"],
            "__side__": [1, -1, 1, -1],
            "__y_ret__": [0.030, 0.012, 0.020, -0.010],
            "__u_policy_net__": [0.025, 0.009, 0.018, -0.015],
            "__sl__": [0.005, 0.020, 0.005, 0.005],
            "__tp__": [0.010, 0.040, 0.010, 0.010],
            "__barrier_pct__": [0.010, 0.040, 0.010, 0.010],
            "__mae_ret__": [0.003, 0.005, 0.020, 0.000],
            "__mfe_ret__": [0.020, 0.050, 0.030, 0.001],
            "__bars_policy__": [4, 6, 6, 16],
            "__bars_to_mfe__": [2, 6, 6, 16],
            "__is_timeout__": [0, 0, 0, 1],
        }
    )


def test_economic_target_enforces_cost_margin_and_sl_feasibility() -> None:
    spec = EconomicTargetSpec(
        name="unit",
        utility_source="y_ret",
        cost=0.010,
        margin=0.005,
        sl_buffer=1.2,
        vol_source="sl",
        temperature=1.0,
    )

    target, summary = build_economic_target(_frame(), spec)

    assert np.allclose(target["__u_econ_net__"], [0.020, 0.002, 0.010, -0.020])
    assert np.allclose(target["__u_econ_vol_norm__"], [4.0, 0.1, 2.0, -4.0])
    assert target["__econ_feasible__"].tolist() == [1, 0, 1, 1]
    assert target["__y_econ_bin__"].tolist() == [1, 0, 1, 0]
    assert summary["cost"] == 0.010
    assert summary["margin"] == 0.005
    assert summary["sl_buffer"] == 1.2


def test_economic_target_path_penalty_can_demote_adverse_rows() -> None:
    spec = EconomicTargetSpec(
        name="penalized",
        utility_source="y_ret",
        cost=0.010,
        margin=0.005,
        sl_buffer=1.2,
        vol_source="sl",
        mae_penalty=1.0,
    )

    target, _summary = build_economic_target(_frame(), spec)

    assert target["__y_econ_bin__"].tolist() == [1, 0, 0, 0]
    assert float(target.loc[2, "__u_econ_adjusted_net__"]) < 0.0


def test_append_economic_target_columns_keeps_side_and_audit_columns() -> None:
    out, summary = append_economic_target_columns(_frame(), EconomicTargetSpec(name="audit"))

    for column in economic_target_column_names():
        assert column in out.columns
    assert out["__side__"].tolist() == [1, -1, 1, -1]
    assert out["__econ_target_name__"].nunique() == 1
    assert summary["rows"] == 4


def test_sideaware_economic_target_demotes_dirty_positive_paths() -> None:
    spec = EconomicTargetSpec(
        name="sideaware_unit",
        utility_source="y_ret",
        cost=0.003,
        margin=0.001,
        sl_buffer=0.1,
        vol_source="barrier",
        temperature=1.0,
    )

    target, summary = build_economic_target(_frame(), spec)

    assert target["__y_econ_sideaware_bin__"].tolist() == [1, 1, 0, 0]
    assert target["__econ_sideaware_clean__"].tolist() == [1, 1, 0, 0]
    assert float(target.loc[0, "__y_econ_sideaware_soft__"]) > float(
        target.loc[2, "__y_econ_sideaware_soft__"]
    )
    assert float(target.loc[2, "__u_econ_sideaware_adjusted_net__"]) < float(
        target.loc[2, "__u_econ_sideaware_net__"]
    )
    assert float(target.loc[2, "__y_econ_sideaware_soft__"]) <= 0.08
    assert float(target.loc[3, "__y_econ_sideaware_soft__"]) <= 0.06
    assert int(target.loc[2, "__econ_sideaware_reason_code__"]) == 3
    assert int(target.loc[3, "__econ_sideaware_reason_code__"]) in {2, 6}
    assert summary["sideaware_hard_rate"] == 0.5
    assert summary["sideaware_long_hard_rate"] == 0.5
    assert summary["sideaware_short_hard_rate"] == 0.5


def test_side_resolution_target_separates_long_timeout_and_short_bad_mae() -> None:
    frame = _frame()
    frame.loc[1, "__mae_ret__"] = 0.030
    frame.loc[1, "__mfe_ret__"] = 0.055
    frame.loc[1, "__bars_policy__"] = 6
    frame.loc[1, "__bars_to_mfe__"] = 6
    spec = EconomicTargetSpec(
        name="side_resolution_unit",
        utility_source="y_ret",
        cost=0.003,
        margin=0.001,
        sl_buffer=0.1,
        vol_source="barrier",
        temperature=1.0,
    )

    target, summary = build_economic_target(frame, spec)

    assert target["__y_econ_side_resolution_bin__"].tolist() == [1, 0, 0, 0]
    assert target["__econ_side_resolution_clean__"].tolist() == [1, 0, 0, 0]
    assert float(target.loc[1, "__y_econ_side_resolution_soft__"]) <= 0.03
    assert float(target.loc[2, "__y_econ_side_resolution_soft__"]) <= 0.08
    assert float(target.loc[3, "__y_econ_side_resolution_soft__"]) <= 0.03
    assert int(target.loc[1, "__econ_side_resolution_reason_code__"]) == 3
    assert int(target.loc[3, "__econ_side_resolution_reason_code__"]) in {2, 7}
    assert target["__y_econ_sideaware_execres_bin__"].tolist() == [1, 0, 0, 0]
    assert target["__econ_sideaware_execres_clean__"].tolist() == [1, 0, 0, 0]
    assert target["__econ_sideaware_execres_dirty_positive__"].tolist() == [0, 1, 1, 0]
    assert int(target.loc[1, "__econ_sideaware_execres_reason_code__"]) == 3
    assert int(target.loc[0, "__econ_sideaware_execres_geometry_bucket__"]) == 1
    assert summary["side_resolution_hard_rate"] == 0.25
    assert summary["side_resolution_long_hard_rate"] == 0.5
    assert summary["side_resolution_short_hard_rate"] == 0.0


def test_candidate_specs_names_are_stable_and_unique() -> None:
    specs = candidate_specs(
        utility_sources=["y_ret"],
        margins=[0.005, 0.010],
        vol_sources=["sl"],
        costs=[0.003, 0.010],
        temperatures=[0.5, 0.75],
        mae_penalties=[0.0],
        timeout_penalties=[0.0],
    )

    names = [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert names[0] == "econ_y_ret_sl_c0030_m0050_mae00_to00_t050"
    assert names[-1] == "econ_y_ret_sl_c0100_m0100_mae00_to00_t075"


def test_strict_optimizer_gate_rejects_negative_net_ic() -> None:
    row = {
        "hard_rate": 0.12,
        "feasible_rate": 0.99,
        "soft_std": 0.20,
        "proxy_top10_mean_net": 0.001,
        "proxy_top10_delta_mean": 0.001,
        "proxy_top10_hit_net": 0.40,
        "proxy_top10_q10_net": -0.03,
        "proxy_top10_ic_net": -0.10,
        "proxy_top10_ic_soft": 0.40,
        "oracle_top10_mean_net": 0.03,
        "feature_top_abs_ic": 0.25,
        "proxy_months": 3,
    }
    args = SimpleNamespace(
        min_hard_rate=0.005,
        max_hard_rate=0.60,
        min_feasible_rate=0.10,
        min_soft_std=0.02,
        min_proxy_ic_soft=0.02,
        min_proxy_delta=0.0,
        require_proxy_positive_net=True,
        min_proxy_mean_net=0.0,
        min_proxy_ic_net=0.0,
        min_proxy_hit_net=0.0,
        min_proxy_q10_net=float("-inf"),
    )

    assert _score_candidate(row, args) == float("-inf")
    row["proxy_top10_ic_net"] = 0.05
    assert math.isfinite(_score_candidate(row, args))


def test_base_regression_target_prefers_optimized_economic_column() -> None:
    frame = _frame()
    frame["__y_econ_pos__"] = [1.2, 0.0, 0.7, 0.0]
    frame["__y_econ_reg__"] = [1.2, -0.4, 0.7, -2.0]
    frame["__u_econ_net__"] = [0.020, -0.002, 0.010, -0.020]

    bundle = _build_base_regression_target(frame, side="short", cfg={})

    assert bundle["target_name"] == "optimized_economic_target:__y_econ_pos__"
    assert np.allclose(bundle["target"], [1.2, 0.0, 0.7, 0.0])
    assert np.allclose(bundle["raw_vol_norm_return"], [1.2, -0.4, 0.7, -2.0])
    assert np.allclose(bundle["side_adjusted_return"], [0.020, -0.002, 0.010, -0.020])
