from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_may_july_failure_diagnosis.py"
SPEC = importlib.util.spec_from_file_location("run_may_july_failure_diagnosis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
JULY_PARTIAL_LABEL = MODULE.JULY_PARTIAL_LABEL
run_diagnosis = MODULE.run_diagnosis


def _source_rows(side: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(dates):
        for symbol in ("AAA/USD:USD", "BBB/USD:USD"):
            base = {
                "__ts__": timestamp,
                "__symbol__": symbol,
                "side_name": side,
                "ev_after_1pct": -0.004 if index == 1 else 0.004,
                "score": 0.3 + index * 0.01,
                "score_base": 0.3 + index * 0.01,
                "archetype_policy_key": f"{side}_archetype",
                "clean_exec": float(index % 2 == 0),
                "dirty_positive": 0.0,
                "full_path_bad_mae_1r": 0.0,
                "timeout": 0.0,
            }
            if side == "long":
                base.update(
                    score_base_residual_ev_rank_train_reference=0.95 if symbol.startswith("AAA") else 0.50,
                    score_base_ev_residual_expert_hier_mapped=0.006,
                    dae_b16_05=float(index),
                    prog_eff_24=float(index + 1),
                )
            else:
                base.update(
                    score_meta_base_soft_label=0.7,
                    calibrated_score=0.8,
                    hit_probability=0.7,
                    historical_rank=0.96 if symbol.startswith("AAA") else 0.40,
                    expected_net_ev_after_1pct_side_archetype=0.007,
                    pct_assets_new_low_7d=float(index),
                    gmm_ood_score=float(index + 1),
                )
            rows.append(base)
    return pd.DataFrame(rows)


def _outcome_rows(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side in ("long", "short"):
        for timestamp in dates:
            for symbol in ("AAA/USD:USD", "BBB/USD:USD"):
                rows.append(
                    {
                        "__ts__": timestamp,
                        "__symbol__": symbol,
                        "side_name": side,
                        "__mfe__": 0.02,
                        "__mae__": -0.01,
                        "__first_touch_bar__": 4.0,
                        "__is_timeout__": 0.0,
                        "__first_touch_stop__": 0.0,
                        "__first_touch_hit__": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_run_may_july_failure_diagnosis_normalizes_both_sides_and_costs_once(tmp_path: Path) -> None:
    dates = pd.to_datetime(
        ["2026-05-01 00:00", "2026-06-01 00:00", "2026-07-10 21:00"], utc=True
    )
    long_path = tmp_path / "long.parquet"
    short_path = tmp_path / "short.parquet"
    outcome_path = tmp_path / "outcomes.parquet"
    eligible_path = tmp_path / "eligible.csv"
    short_manifest = tmp_path / "short_manifest.json"
    output_dir = tmp_path / "out"
    _source_rows("long", dates).to_parquet(long_path, index=False)
    _source_rows("short", dates).to_parquet(short_path, index=False)
    _outcome_rows(dates).to_parquet(outcome_path, index=False)
    pd.DataFrame({"symbol": ["AAA/USD:USD", "BBB/USD:USD"], "p90_spread_bps": [10.0, 20.0]}).to_csv(eligible_path, index=False)
    short_manifest.write_text(json.dumps({"schema": "synthetic_hybrid_manifest"}))

    manifest = run_diagnosis(
        output_dir=output_dir, long_source=long_path, short_source=short_path,
        outcome_source=outcome_path, eligible_symbols=eligible_path, short_manifest=short_manifest,
        feature_store=None,
    )

    daily = pd.read_csv(output_dir / "daily.csv")
    monthly = pd.read_csv(output_dir / "monthly.csv")
    slices = pd.read_csv(output_dir / "model_slices.csv")
    selection = pd.read_csv(output_dir / "selection_support.csv")
    assert set(selection["side"]) == {"long", "short"}
    assert JULY_PARTIAL_LABEL in set(monthly["period_label"])
    assert {"long", "short"} == set(selection["side"])
    # AAA is the only selected symbol per side. Its 10 bps spread gives 0.004 + 0.01 - 0.003 - 0.001.
    assert np.isclose(daily.loc[0, "net_return_sum"], 0.020)
    assert np.allclose(
        slices.loc[slices["period_label"].eq("2026-05"), "net_return_sum"], 0.010
    )
    assert (output_dir / "failure_episodes.csv").exists()
    assert (output_dir / "feature_drift.csv").exists()
    assert (output_dir / "nn_loss_neighbors.csv").exists()
    assert (output_dir / "report.md").exists()
    assert manifest["cost_contract"]["cost_count"] == 1
    assert manifest["row_counts"]["fixed_selected_population"] == 6
