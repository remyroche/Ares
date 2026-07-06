from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_promoted_cross_asset_meta_handoff import main  # noqa: E402


def test_materialize_promoted_cross_asset_meta_handoff_keeps_only_promoted_oof_columns(tmp_path: Path) -> None:
    handoff_dir = tmp_path / "handoff"
    out_dir = tmp_path / "out"
    handoff_dir.mkdir()
    rows = []
    reps = []
    ledger = []
    for idx in range(6):
        ts = pd.Timestamp("2026-05-01") + pd.Timedelta(hours=idx)
        key = {"__ts__": ts, "__symbol__": f"SYM{idx % 2}", "side_name": "short" if idx % 2 else "long"}
        rows.append({**key, "month": "2026-05", "score": float(idx), "base_feature": float(idx + 1)})
        ledger.append({**key, "month": "2026-05", "selected_top10": True, "exec_margin": 0.01})
        if idx < 4:
            reps.append(
                {
                    **key,
                    "cross_lgbm_exec_margin_score": float(idx) / 10.0,
                    "cross_lgbm_bad_mae_score": 0.2,
                    "cross_lgbm_timeout_score": 0.1,
                    "cross_lgbm_dirty_positive_score": 0.3,
                    "cross_lgbm_clean_risk_composite": 0.4,
                    "market_z_0": 99.0,
                    "exec_margin": -999.0,
                    "full_path_bad_mae_1r": 1.0,
                }
            )
    pd.DataFrame(rows).to_parquet(handoff_dir / "train_meta_regime_handoff.parquet", index=False)
    pd.DataFrame(ledger).to_parquet(handoff_dir / "s52_trailing_regime_scored_ledger.parquet", index=False)
    (handoff_dir / "train_meta_regime_handoff_contract.json").write_text(json.dumps({"source": "test"}))
    rep_path = tmp_path / "representations.parquet"
    pd.DataFrame(reps).to_parquet(rep_path, index=False)
    promotion = {
        "status": "candidate_features_available",
        "promote_to_deeper_meta_eval": [
            {
                "variant": "m1b_cross_lgbm_risk_only_meta",
                "feature_columns": [
                    "cross_lgbm_bad_mae_score",
                    "cross_lgbm_timeout_score",
                    "cross_lgbm_dirty_positive_score",
                    "cross_lgbm_clean_risk_composite",
                    "exec_margin",
                ],
            }
        ],
        "shadow_only": [],
    }
    promotion_path = tmp_path / "promotion.json"
    promotion_path.write_text(json.dumps(promotion))

    rc = main(
        [
            "--handoff-dir",
            str(handoff_dir),
            "--representation-predictions",
            str(rep_path),
            "--promotion-json",
            str(promotion_path),
            "--out-dir",
            str(out_dir),
            "--preferred-variant",
            "m1b_cross_lgbm_risk_only_meta",
        ]
    )
    assert rc == 0
    out = pd.read_parquet(out_dir / "train_meta_regime_handoff.parquet")
    assert len(out) == 6
    assert "base_feature" in out.columns
    assert "cross_lgbm_bad_mae_score" in out.columns
    assert "cross_lgbm_timeout_score" in out.columns
    assert "cross_lgbm_dirty_positive_score" in out.columns
    assert "cross_lgbm_clean_risk_composite" in out.columns
    assert "cross_lgbm_exec_margin_score" not in out.columns
    assert "market_z_0" not in out.columns
    assert "exec_margin" not in out.columns
    promoted_cols = [
        "cross_lgbm_bad_mae_score",
        "cross_lgbm_timeout_score",
        "cross_lgbm_dirty_positive_score",
        "cross_lgbm_clean_risk_composite",
    ]
    assert int(out[promoted_cols].notna().all(axis=1).sum()) == 4
    assert out.loc[out[promoted_cols].isna().all(axis=1), promoted_cols].shape[0] == 2
    contract = json.loads((out_dir / "train_meta_regime_handoff_contract.json").read_text())
    block = contract["promoted_cross_asset_representation"]
    assert block["no_in_sample_backfill"] is True
    assert block["rows_with_all_promoted_columns"] == 4
    assert block["promoted_columns"] == promoted_cols
    assert "Outcome/path columns" in block["leakage_contract"]
    assert (out_dir / "s52_trailing_regime_scored_ledger.parquet").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["promoted_variants"] == ["m1b_cross_lgbm_risk_only_meta"]
    assert np.isclose(manifest["rows_with_all_promoted_columns"], 4)


def test_materialize_promoted_cross_asset_meta_handoff_adds_preentry_cell_interactions(tmp_path: Path) -> None:
    handoff_dir = tmp_path / "handoff"
    out_dir = tmp_path / "out"
    handoff_dir.mkdir()
    rows = []
    reps = []
    ledger = []
    for idx in range(8):
        ts = pd.Timestamp("2026-05-01") + pd.Timedelta(hours=idx)
        family = "quiet_continuation" if idx < 6 else "rare"
        key = {"__ts__": ts, "__symbol__": f"SYM{idx % 3}", "side_name": "short"}
        rows.append(
            {
                **key,
                "month": "2026-05",
                "source_semantic_family": family,
                "base_feature": float(idx),
            }
        )
        ledger.append({**key, "month": "2026-05", "selected_top10": True, "exec_margin": 0.01})
        reps.append(
            {
                **key,
                "cross_lgbm_bad_mae_score": float(idx),
                "cross_lgbm_timeout_score": float(idx + 10),
                "cross_lgbm_dirty_positive_score": float(idx + 20),
                "cross_lgbm_clean_risk_composite": float(idx + 30),
            }
        )
    pd.DataFrame(rows).to_parquet(handoff_dir / "train_meta_regime_handoff.parquet", index=False)
    pd.DataFrame(ledger).to_parquet(handoff_dir / "s52_trailing_regime_scored_ledger.parquet", index=False)
    (handoff_dir / "train_meta_regime_handoff_contract.json").write_text(json.dumps({"source": "test"}))
    rep_path = tmp_path / "representations.parquet"
    pd.DataFrame(reps).to_parquet(rep_path, index=False)
    promotion_path = tmp_path / "promotion.json"
    promotion_path.write_text(
        json.dumps(
            {
                "status": "candidate_features_available",
                "promote_to_deeper_meta_eval": [
                    {
                        "variant": "m1b_cross_lgbm_risk_only_meta",
                        "feature_columns": [
                            "cross_lgbm_bad_mae_score",
                            "cross_lgbm_timeout_score",
                            "cross_lgbm_dirty_positive_score",
                            "cross_lgbm_clean_risk_composite",
                        ],
                    }
                ],
            }
        )
    )

    rc = main(
        [
            "--handoff-dir",
            str(handoff_dir),
            "--representation-predictions",
            str(rep_path),
            "--promotion-json",
            str(promotion_path),
            "--out-dir",
            str(out_dir),
            "--preferred-variant",
            "m1b_cross_lgbm_risk_only_meta",
            "--add-cell-interactions",
            "--min-cell-interaction-rows",
            "5",
        ]
    )

    assert rc == 0
    out = pd.read_parquet(out_dir / "train_meta_regime_handoff.parquet")
    interaction_cols = [col for col in out.columns if "__sxsf__" in col]
    assert len(interaction_cols) == 4
    quiet = out["source_semantic_family"].eq("quiet_continuation")
    bad_col = [col for col in interaction_cols if col.startswith("cross_lgbm_bad_mae_score")][0]
    assert out.loc[quiet, bad_col].tolist() == out.loc[quiet, "cross_lgbm_bad_mae_score"].astype("float32").tolist()
    assert out.loc[~quiet, bad_col].eq(0.0).all()
    contract = json.loads((out_dir / "train_meta_regime_handoff_contract.json").read_text())
    block = contract["promoted_cross_asset_representation"]["cell_interaction_features"]
    assert block["enabled"] is True
    assert block["interaction_column_count"] == 4
    assert "no outcomes" in block["leakage_contract"]
