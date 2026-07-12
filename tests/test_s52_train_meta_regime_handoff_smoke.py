from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_train_meta_regime_handoff_smoke import _feature_columns, run_smoke


def test_s52_train_meta_feature_columns_exclude_generated_path_labels() -> None:
    frame = pd.DataFrame(
        {
            "score": [0.1, 0.2],
            "regime_clean_exec_score": [0.3, 0.4],
            "long_bad_path_label": [1.0, 0.0],
            "long_path_clean_exec_label": [0.0, 1.0],
            "exec_margin": [0.01, -0.01],
        }
    )
    numeric, categorical = _feature_columns(frame)
    all_cols = set(numeric + categorical)
    assert "score" in all_cols
    assert "regime_clean_exec_score" in all_cols
    assert "long_bad_path_label" not in all_cols
    assert "long_path_clean_exec_label" not in all_cols
    assert "exec_margin" not in all_cols


def test_s52_train_meta_handoff_smoke_learns_clean_filter(tmp_path: Path) -> None:
    handoff_dir = tmp_path / "handoff"
    out_dir = tmp_path / "out"
    handoff_dir.mkdir()
    handoff_rows = []
    ledger_rows = []
    for month_idx, month in enumerate(("2026-04", "2026-05")):
        for idx in range(160):
            clean = idx % 4 != 0
            ts = f"{month}-{(idx % 20) + 1:02d}T{idx % 24:02d}:00:00Z"
            symbol = f"SYM{idx % 7}"
            side = "long" if idx % 3 == 0 else "short"
            quality = 1.0 if clean else -1.0
            score = float(1.0 - idx / 500.0)
            handoff_rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": symbol,
                    "side_name": side,
                    "month": month,
                    "score": score,
                    "selected_top10": True,
                    "source_semantic_family": "quiet_continuation"
                    if clean
                    else "dirty_shock_avoid",
                    "regime_clean_exec_score": 0.9 if clean else 0.1,
                    "regime_bad_mae_score": 0.1 if clean else 0.9,
                    "gmm_entropy": 0.2 + 0.01 * (idx % 5),
                    "latent_speed": quality,
                    "meta_context_weight_hint": 1.0 if clean else 0.2,
                    "meta_threshold_adjustment_hint": 0.0 if clean else 0.5,
                    "aegmm_expected_distance_bin": "q1" if clean else "q3",
                }
            )
            ledger_rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": symbol,
                    "side_name": side,
                    "month": month,
                    "score": score,
                    "selected_top10": True,
                    "exec_margin": 0.010 + 0.001 * month_idx if clean else -0.006,
                    "ev_after_1pct": 0.010 if clean else -0.006,
                    "first_touch_gross": 0.020 if clean else 0.004,
                    "first_touch_bad_mae_1r": 0.0 if clean else 1.0,
                    "full_path_bad_mae_1r": 0.0 if clean else 1.0,
                    "timeout": 0.0,
                    "mfe_before_mae_1r": 1.0 if clean else 0.0,
                    "mae_before_mfe_1r": 0.0 if clean else 1.0,
                    "clean_exec": 1.0 if clean else 0.0,
                    "dirty_positive": 0.0 if clean else 1.0,
                    "underwater_bars_before_mfe_1r": 2.0 if clean else 12.0,
                }
            )
    pd.DataFrame(handoff_rows).to_parquet(
        handoff_dir / "train_meta_regime_handoff.parquet", index=False
    )
    ledger_path = handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
    pd.DataFrame(ledger_rows).to_parquet(ledger_path, index=False)

    run_smoke(
        handoff_dir=handoff_dir,
        ledger_path=ledger_path,
        out_dir=out_dir,
        frontier="top10",
        seed=11,
        train_scope="selected",
        # This fixture is intentionally below the canonical MDA minimum. The
        # smoke asserts meta filtering behavior, not feature-selection quality.
        fixed_selected_features=[
            "score",
            "regime_clean_exec_score",
            "regime_bad_mae_score",
            "gmm_entropy",
            "latent_speed",
            "meta_context_weight_hint",
            "meta_threshold_adjustment_hint",
            "aegmm_expected_distance_bin_q1",
            "aegmm_expected_distance_bin_q3",
        ],
    )

    summary = pd.read_csv(out_dir / "s52_train_meta_regime_handoff_smoke_summary.csv")
    assert not summary.empty
    meta = summary[summary["selector"].astype(str).str.startswith("meta_")]
    assert float(meta["mean_keep030_exec_margin"].max()) > 0.0
    assert float(meta["mean_keep030_full_path_bad_mae"].min()) < float(
        summary["mean_keep100_full_path_bad_mae"].iloc[0]
    )
    threshold_summary = pd.read_csv(
        out_dir / "s52_train_meta_regime_handoff_threshold_policy_summary.csv"
    )
    assert not threshold_summary.empty
    assert {"policy_id", "budget_frac", "threshold_policy_status"}.issubset(
        threshold_summary.columns
    )
    assert (out_dir / "s52_train_meta_regime_handoff_smoke.md").exists()
