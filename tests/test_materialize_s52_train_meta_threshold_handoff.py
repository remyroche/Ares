from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_s52_train_meta_threshold_handoff import materialize


def test_materialize_s52_meta_threshold_handoff_writes_clean_and_offline_files(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke"
    handoff_dir = tmp_path / "handoff"
    out_dir = tmp_path / "out"
    smoke_dir.mkdir()
    handoff_dir.mkdir()
    rows = []
    handoff = []
    for month in ("2026-05", "2026-06"):
        for idx in range(20):
            ts = f"{month}-{idx + 1:02d}T00:00:00Z"
            clean = idx < 8
            rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": f"SYM{idx % 5}",
                    "side_name": "short" if idx % 2 else "long",
                    "month": month,
                    "source_semantic_family": "quiet_continuation" if clean else "dirty_shock_avoid",
                    "exec_margin": 0.01 if clean else -0.005,
                    "ev_after_1pct": 0.007 if clean else -0.008,
                    "ret_net": 0.014 if clean else -0.004,
                    "u_policy_net": 0.017 if clean else -0.001,
                    "first_touch_gross": 0.02 if clean else 0.004,
                    "first_touch_bad_mae_1r": 0.0 if clean else 1.0,
                    "full_path_bad_mae_1r": 0.0 if clean else 1.0,
                    "timeout": 0.0,
                    "mfe_before_mae_1r": 1.0 if clean else 0.0,
                    "mae_before_mfe_1r": 0.0 if clean else 1.0,
                    "clean_exec": 1.0 if clean else 0.0,
                    "dirty_positive": 0.0 if clean else 1.0,
                    "underwater_bars_before_mfe_1r": 2.0 if clean else 10.0,
                    "score_base": 1.0 - idx / 100.0,
                    "score_meta_clean_exec": 0.9 if clean else 0.2,
                    "score_meta_positive_margin": 0.85 if clean else 0.3,
                    "score_meta_clean_minus_risk": 0.8 if clean else 0.1,
                }
            )
            handoff.append(
                {
                    "__ts__": ts,
                    "__symbol__": f"SYM{idx % 5}",
                    "side_name": "short" if idx % 2 else "long",
                    "score": 1.0 - idx / 100.0,
                    "source_tag": "demo",
                    "source_family": "demo",
                    "source_semantic_family": "quiet_continuation" if clean else "dirty_shock_avoid",
                    "gmm_cluster_id": idx % 3,
                    "gmm_entropy": 0.2,
                    "meta_context_weight_hint": 1.0,
                    "meta_threshold_adjustment_hint": 0.0,
                }
            )
    pd.DataFrame(rows).to_parquet(smoke_dir / "s52_train_meta_regime_handoff_smoke_predictions.parquet", index=False)
    pd.DataFrame([{"selector": "meta_clean_minus_risk", "policy_id": "clean_ge_0.65"}]).to_csv(
        smoke_dir / "s52_train_meta_regime_handoff_threshold_policy_summary.csv",
        index=False,
    )
    pd.DataFrame(handoff).to_parquet(handoff_dir / "train_meta_regime_handoff.parquet", index=False)

    manifest = materialize(
        smoke_dir=smoke_dir,
        handoff_dir=handoff_dir,
        out_dir=out_dir,
        selector="meta_clean_minus_risk",
        policy_id="clean_ge_0.65",
        budget_frac=0.25,
        max_side_share=0.75,
    )

    clean = pd.read_parquet(out_dir / "s52_meta_threshold_guarded_candidates.parquet")
    offline = pd.read_parquet(out_dir / "s52_meta_threshold_guarded_offline_eval_candidates.parquet")
    summary = pd.read_csv(out_dir / "s52_meta_threshold_guarded_summary.csv")
    month_summary = pd.read_csv(out_dir / "s52_meta_threshold_guarded_month_summary.csv")
    assert not clean.empty
    assert len(clean) == len(offline)
    assert "exec_margin" not in clean.columns
    assert "exec_margin" in offline.columns
    assert "mean_ret_net" in summary.columns
    assert "mean_u_policy_net" in summary.columns
    assert "ret_net" in month_summary.columns
    assert "u_policy_net" in month_summary.columns
    assert "max_side_share_cap" in clean.columns
    assert float(clean["max_side_share_cap"].max()) == 0.75
    assert manifest["leakage_audit"]["clean_handoff_has_no_realized_outcomes"] is True
    assert int(manifest["leakage_audit"]["duplicate_decision_key_rows"]) == 0
