from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_promoted_cross_asset_topk_selector_objective import run_topk_selector_objective  # noqa: E402


def _write_predictions(path: Path, rows: list[dict], selector: str) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path / "s52_train_meta_regime_handoff_smoke_predictions.parquet", index=False)
    (path / "manifest.json").write_text(json.dumps({"best_selector": {"selector": selector}}))


def _rows(*, promoted_good: bool) -> list[dict]:
    rows: list[dict] = []
    for month in ("2026-05", "2026-06"):
        for idx in range(100):
            clean = idx < 35
            side = "long" if idx % 3 == 0 else "short"
            good_score = float(100 - idx) if clean else float(idx)
            bad_score = float(100 - idx) if not clean else float(idx)
            promoted_score = good_score if promoted_good else bad_score
            rows.append(
                {
                    "__ts__": f"{month}-{(idx % 25) + 1:02d}T{idx % 24:02d}:00:00Z",
                    "__symbol__": f"SYM{idx % 12}",
                    "side_name": side,
                    "month": month,
                    "source_semantic_family": "family_a" if idx % 2 == 0 else "family_b",
                    "score_base": bad_score,
                    "score_meta_clean_exec": promoted_score,
                    "score_meta_positive_margin": good_score,
                    "score_meta_exec_margin": good_score / 100.0,
                    "score_meta_clean_minus_risk": good_score,
                    "score_meta_exec_margin_risk_blend": good_score,
                    "score_meta_context_hint_blend": promoted_score,
                    "score_meta_long_aware_clean_minus_risk": good_score,
                    "score_meta_bad_path": bad_score / 100.0,
                    "score_meta_timeout": 0.1 if clean else 0.8,
                    "score_meta_long_clean_exec": good_score if side == "long" else None,
                    "score_meta_long_bad_path": (bad_score / 100.0) if side == "long" else None,
                    "exec_margin": 0.012 if clean else -0.006,
                    "ev_after_1pct": 0.009 if clean else -0.011,
                    "ret_net": 0.011 if clean else -0.010,
                    "u_policy_net": 0.010 if clean else -0.009,
                    "first_touch_gross": 0.019 if clean else -0.004,
                    "clean_exec": 1.0 if clean else 0.0,
                    "dirty_positive": 0.0 if clean else 1.0,
                    "first_touch_bad_mae_1r": 0.0 if clean else 1.0,
                    "full_path_bad_mae_1r": 0.0 if clean else 1.0,
                    "timeout": 0.0 if clean else 1.0,
                    "mfe_before_mae_1r": 1.0 if clean else 0.0,
                    "mae_before_mfe_1r": 0.0 if clean else 1.0,
                    "underwater_bars_before_mfe_1r": 2.0 if clean else 15.0,
                }
            )
    return rows


def test_topk_selector_objective_uses_prior_month_and_writes_outputs(tmp_path: Path) -> None:
    promoted_dir = tmp_path / "promoted"
    baseline_dir = tmp_path / "baseline"
    _write_predictions(promoted_dir, _rows(promoted_good=True), "meta_context_hint_blend")
    _write_predictions(baseline_dir, _rows(promoted_good=False), "base_score")

    manifest = run_topk_selector_objective(
        promoted_predictions_path=promoted_dir / "s52_train_meta_regime_handoff_smoke_predictions.parquet",
        baseline_predictions_path=baseline_dir / "s52_train_meta_regime_handoff_smoke_predictions.parquet",
        out_dir=tmp_path / "out",
    )

    assert manifest["evaluated_months"] == ["2026-06"]
    assert manifest["first_month_treatment"] == "skipped_as_selector_training_history"
    selections = pd.read_csv(manifest["outputs"]["selections"])
    assert selections["history_months"].iloc[0] == "2026-05"
    assert int(selections["candidate_count"].iloc[0]) > 2
    predictions = pd.read_parquet(manifest["outputs"]["predictions"])
    assert predictions["month"].astype(str).unique().tolist() == ["2026-06"]
    assert "topk_objective_selected_score_col" in predictions.columns
    assert Path(manifest["outputs"]["markdown"]).exists()
    summary = pd.read_csv(manifest["outputs"]["summary"])
    learned = summary[summary["selector"].eq("learned_topk_ev_path_objective")].iloc[0]
    baseline = summary[summary["selector"].astype(str).str.startswith("baseline_fixed")].iloc[0]
    assert float(learned["mean_keep010_ev_after_1pct"]) > float(baseline["mean_keep010_ev_after_1pct"])
    assert float(learned["mean_keep010_full_path_bad_mae"]) < float(baseline["mean_keep010_full_path_bad_mae"])
