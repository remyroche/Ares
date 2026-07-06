from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_direct_context_interaction_train_meta_feature_set import run


def test_materialize_direct_context_interaction_train_meta_feature_set(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=8, freq="h"),
            "__symbol__": ["A/USD:USD", "B/USD:USD"] * 4,
            "month": ["2026-01"] * 4 + ["2026-02"] * 4,
            "side_name": ["long", "short"] * 4,
            "source_archetype": ["long_bars", "short_bollinger"] * 4,
            "strategy_id": ["s"] * 8,
            "normalized_rank_score": [0.1, 0.2, 0.4, 0.3, 0.5, 0.1, 0.8, 0.6],
            "xctx_ev_score_oof": [0.3, -0.2, 0.1, 0.5, -0.4, 0.6, 0.7, -0.1],
            "xctx_blend_score": [0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.6, -0.3],
            "xctx_cluster_entropy": [0.5, 0.7, 0.2, 0.8, 0.9, 0.1, 0.3, 0.4],
            "oofctx_dae_reconstruction_error": [0.05, 0.2, 0.08, 0.3, 0.4, 0.1, 0.06, 0.25],
            "exec_ev_after_1pct_cost": [0.01, -0.01, 0.03, 0.02, -0.02, 0.04, 0.05, -0.01],
            "full_sl": [0, 1, 0, 0, 1, 0, 0, 1],
            "timeout": [0, 0, 1, 0, 0, 1, 0, 0],
            "clean_exec_proxy": [1, 0, 0, 1, 0, 0, 1, 0],
        }
    )
    handoff_path = tmp_path / "handoff.parquet"
    feature_manifest_path = tmp_path / "feature_manifest.json"
    interaction_dir = tmp_path / "interaction"
    output_dir = tmp_path / "out"
    interaction_dir.mkdir()
    frame.to_parquet(handoff_path, index=False)
    feature_manifest_path.write_text(
        json.dumps(
            {
                "feature_columns": [
                    "normalized_rank_score",
                    "xctx_ev_score_oof",
                    "xctx_blend_score",
                    "xctx_cluster_entropy",
                    "oofctx_dae_reconstruction_error",
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "variant": ["i2_xctx_cell_interactions"],
            "month": ["2026-02"],
            "selector": ["s12_ev_clean_strong_risk"],
            "top_frac": [0.10],
            "precision_positive_ev": [0.55],
            "ev_weighted_precision": [0.7],
            "mean_ev_after_1pct": [0.02],
            "full_sl_rate": [0.25],
            "timeout_rate": [0.2],
            "clean_exec_proxy_rate": [0.4],
        }
    ).to_csv(interaction_dir / "interaction_train_meta_aggregate.csv", index=False)
    pd.DataFrame(
        {
            "variant": ["i2_xctx_cell_interactions"],
            "month": ["2026-02"],
            "selector": ["s12_ev_clean_strong_risk"],
            "top_frac": [0.10],
            "delta_mean_ev_after_1pct": [0.01],
            "delta_precision_positive_ev": [0.05],
            "delta_full_sl_rate": [-0.02],
            "delta_timeout_rate": [-0.03],
            "delta_clean_exec_proxy_rate": [0.04],
        }
    ).to_csv(interaction_dir / "interaction_train_meta_aggregate_delta.csv", index=False)
    pd.DataFrame(
        {
            "variant": ["i2_xctx_cell_interactions"],
            "selector": ["s12_ev_clean_strong_risk"],
            "cells": [2],
            "better_ev": [1],
            "better_precision": [2],
            "lower_full_sl": [1],
            "lower_timeout": [1],
            "mean_delta_ev": [0.003],
            "mean_delta_precision": [0.02],
            "mean_delta_full_sl": [-0.01],
            "mean_delta_timeout": [-0.02],
            "mean_delta_clean": [0.03],
        }
    ).to_csv(interaction_dir / "interaction_train_meta_cell_delta_summary.csv", index=False)

    manifest = run(
        handoff_path=handoff_path,
        feature_manifest_path=feature_manifest_path,
        interaction_smoke_dir=interaction_dir,
        output_dir=output_dir,
        include_risk_interactions=False,
    )

    out = pd.read_parquet(manifest["outputs"]["handoff"])
    assert "int_side_long" in out.columns
    assert "int_cell_long_long_bars" in out.columns
    assert "intx_xctx_ev_score_oof__int_cell_long_long_bars" in out.columns
    assert "delta_mean_ev_after_1pct" not in out.columns
    assert not any(col.endswith("_accepted_for_cell") for col in out.columns)

    feature_manifest = json.loads(Path(manifest["outputs"]["feature_manifest"]).read_text())
    assert "intx_xctx_ev_score_oof__int_cell_long_long_bars" in feature_manifest["feature_columns"]
    assert feature_manifest["no_leakage_contract"]["interaction_evidence"].startswith("report/manifest")
    assert feature_manifest["feature_group_counts"]["risk_cross_interactions"] == 0
