from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_direct_cross_asset_train_meta_feature_set import run


def test_materialize_direct_context_train_meta_feature_set(tmp_path: Path) -> None:
    handoff = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=6, freq="h"),
            "__symbol__": ["A/USD:USD"] * 6,
            "month": ["2026-01"] * 3 + ["2026-02"] * 3,
            "side_name": ["long", "short"] * 3,
            "source_archetype": ["mean_reversion", "breakout"] * 3,
            "strategy_id": ["s"] * 6,
            "normalized_rank_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "ctx_pct_assets_up_1h": [1, 2, 3, 4, 5, 6],
            "xctx_ev_score_oof": [None, None, 0.1, 0.2, 0.3, 0.4],
            "exec_ev_after_1pct_cost": [0.01, -0.01, 0.02, 0.03, -0.02, 0.01],
            "full_sl": [0, 1, 0, 0, 1, 0],
            "timeout": [0, 0, 1, 0, 0, 1],
            "clean_exec_proxy": [1, 0, 0, 1, 0, 0],
        }
    )
    family_dir = tmp_path / "family"
    family_dir.mkdir()
    family_manifest = {
        "family_contract": {
            "f00_score_only": {"features": ["normalized_rank_score"]},
            "f01_raw_breadth": {"features": ["normalized_rank_score", "ctx_pct_assets_up_1h"]},
            "f11_xctx_scores": {"features": ["normalized_rank_score", "xctx_ev_score_oof"]},
        }
    }
    (family_dir / "manifest.json").write_text(json.dumps(family_manifest), encoding="utf-8")
    pd.DataFrame(
        {
            "variant": ["f01_raw_breadth", "f11_xctx_scores"],
            "month": ["2026-01", "2026-02"],
            "side_name": ["long", "short"],
            "source_archetype": ["mean_reversion", "breakout"],
            "top_frac": [0.1, 0.1],
            "delta_mean_ev_after_1pct": [0.01, 0.02],
            "delta_precision_positive_ev": [0.1, 0.2],
            "delta_full_sl_rate": [-0.1, -0.2],
            "delta_timeout_rate": [0.0, 0.1],
        }
    ).to_csv(family_dir / "direct_cross_asset_family_ablation_accepted_cells.csv", index=False)
    handoff_path = tmp_path / "handoff.parquet"
    handoff.to_parquet(handoff_path, index=False)

    manifest = run(
        handoff_path=handoff_path,
        family_ablation_dir=family_dir,
        output_dir=tmp_path / "out",
        min_months=1,
        min_cells=1,
        min_mean_delta_ev=0.0,
        include_cell_context=True,
    )

    out = pd.read_parquet(manifest["outputs"]["handoff"])
    assert "normalized_rank_score" in out.columns
    assert "ctx_pct_assets_up_1h" in out.columns
    assert "xctx_ev_score_oof" in out.columns
    assert not any(col.endswith("_accepted_for_cell") for col in out.columns)

    feature_manifest = json.loads(Path(manifest["outputs"]["feature_manifest"]).read_text())
    assert feature_manifest["no_leakage_contract"]["accepted_cells"].startswith("audit metadata")
    assert feature_manifest["no_leakage_contract"]["stability_features"] == "excluded"
