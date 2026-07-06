from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_promoted_cross_asset_cell_reliability_overlay import run_overlay  # noqa: E402


def _write_smoke(path: Path, rows: list[dict], selector: str) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path / "s52_train_meta_regime_handoff_smoke_predictions.parquet", index=False)
    (path / "manifest.json").write_text(json.dumps({"best_selector": {"selector": selector}}))


def _make_rows(*, promoted: bool) -> list[dict]:
    rows: list[dict] = []
    for month in ("2026-05", "2026-06"):
        for family in ("good_cell", "bad_cell"):
            for idx in range(50):
                clean = idx < 25
                dirty = not clean
                if family == "good_cell":
                    base_score = float(100 - idx) if dirty else float(idx)
                    promoted_score = float(100 - idx) if clean else float(idx)
                else:
                    base_score = float(100 - idx) if clean else float(idx)
                    promoted_score = float(100 - idx) if dirty else float(idx)
                score = promoted_score if promoted else base_score
                rows.append(
                    {
                        "__ts__": f"{month}-{(idx % 25) + 1:02d}T{idx % 24:02d}:00:00Z",
                        "__symbol__": f"{family.upper()}_SYM{idx % 10}",
                        "side_name": "short",
                        "month": month,
                        "source_semantic_family": family,
                        "score_base": base_score,
                        "score_meta_clean_exec": score,
                        "exec_margin": 0.010 if clean else -0.006,
                        "ev_after_1pct": 0.007 if clean else -0.009,
                        "ret_net": 0.014 if clean else -0.002,
                        "u_policy_net": 0.017 if clean else 0.001,
                        "clean_exec": 1.0 if clean else 0.0,
                        "dirty_positive": 0.0 if clean else 1.0,
                        "full_path_bad_mae_1r": 0.0 if clean else 1.0,
                        "timeout": 0.0,
                        "mfe_before_mae_1r": 1.0 if clean else 0.0,
                        "mae_before_mfe_1r": 0.0 if clean else 1.0,
                        "underwater_bars_before_mfe_1r": 2.0 if clean else 12.0,
                    }
                )
    return rows


def test_cell_reliability_overlay_uses_prior_month_policy(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    promoted = tmp_path / "promoted"
    _write_smoke(baseline, _make_rows(promoted=False), "base_score")
    _write_smoke(promoted, _make_rows(promoted=True), "meta_clean_exec")

    manifest = run_overlay(
        baseline_smoke_dir=baseline,
        promoted_smoke_dir=promoted,
        out_dir=tmp_path / "out",
        min_valid_rows=30,
        min_months=1,
        min_clean_rows=5,
        min_positive_rows=5,
    )

    policy = pd.read_csv(manifest["outputs"]["cell_policies"])
    june = policy[policy["test_month"].astype(str).eq("2026-06")].iloc[0]
    assert int(june["promoted_cells"]) == 1
    assert int(june["baseline_cells"]) == 1
    preds = pd.read_parquet(manifest["outputs"]["predictions"])
    june_preds = preds[preds["month"].astype(str).eq("2026-06")]
    good = june_preds[june_preds["source_semantic_family"].eq("good_cell")]
    bad = june_preds[june_preds["source_semantic_family"].eq("bad_cell")]
    assert good["cell_reliability_uses_promoted"].eq(1.0).all()
    assert bad["cell_reliability_uses_promoted"].eq(0.0).all()
    summary = pd.read_csv(manifest["outputs"]["summary"])
    assert float(summary.loc[0, "mean_keep010_exec_margin"]) > 0.0
