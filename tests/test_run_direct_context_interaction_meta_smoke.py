from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_direct_context_interaction_meta_smoke import run


def _handoff() -> pd.DataFrame:
    rng = np.random.default_rng(73)
    rows = []
    for month in pd.period_range("2026-01", "2026-04", freq="M"):
        for side in ("long", "short"):
            for archetype in ("long_bars", "short_bollinger"):
                for i in range(170):
                    side_sign = 1.0 if side == "long" else -1.0
                    xctx = rng.normal()
                    risk = rng.normal()
                    interaction_edge = 0.012 * xctx * side_sign
                    ev = 0.01 + interaction_edge - 0.008 * max(risk, 0) + rng.normal(scale=0.004)
                    full_sl = int(risk + 0.4 * side_sign * xctx > 1.0)
                    timeout = int(xctx < -1.1)
                    rows.append(
                        {
                            "__ts__": month.to_timestamp() + pd.Timedelta(minutes=i),
                            "__symbol__": f"SYM{i % 5}/USD:USD",
                            "month": str(month),
                            "side_name": side,
                            "source_archetype": archetype,
                            "normalized_rank_score": xctx + rng.normal(scale=0.1),
                            "xctx_ev_score_oof": xctx,
                            "xctx_blend_score": xctx + rng.normal(scale=0.2),
                            "xctx_cluster_entropy": abs(risk),
                            "oofctx_dae_reconstruction_error": max(risk, 0),
                            "exec_ev_after_1pct_cost": ev,
                            "full_sl": full_sl,
                            "timeout": timeout,
                            "clean_exec_proxy": int(ev > 0 and full_sl == 0 and timeout == 0),
                        }
                    )
    return pd.DataFrame(rows)


def test_direct_context_interaction_meta_smoke_outputs(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.parquet"
    manifest_path = tmp_path / "features.json"
    output_dir = tmp_path / "out"
    _handoff().to_parquet(handoff_path, index=False)
    manifest_path.write_text(
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

    manifest = run(
        handoff_path=handoff_path,
        feature_manifest_path=manifest_path,
        output_dir=output_dir,
        variants=("i0_direct_context", "i2_xctx_cell_interactions"),
        max_fit_rows=3_000,
        min_group_rows=50,
        seed=7,
    )

    assert manifest["rows"] == len(_handoff())
    assert manifest["variants"] == ["i0_direct_context", "i2_xctx_cell_interactions"]
    metadata = {row["variant"]: row for row in manifest["variant_metadata"]}
    assert metadata["i2_xctx_cell_interactions"]["interaction_feature_count"] > 0

    aggregate = pd.read_csv(manifest["outputs"]["aggregate"])
    assert {"variant", "selector", "precision_positive_ev", "full_sl_rate", "timeout_rate"}.issubset(
        aggregate.columns
    )
    assert set(aggregate["variant"]) == {"i0_direct_context", "i2_xctx_cell_interactions"}
    assert "s12_ev_clean_strong_risk" in set(aggregate["selector"])

    deltas = pd.read_csv(manifest["outputs"]["aggregate_delta"])
    assert not deltas.empty
    assert "delta_mean_ev_after_1pct" in deltas.columns

    cell_summary = pd.read_csv(manifest["outputs"]["cell_delta_summary"])
    assert {"variant", "selector", "better_ev", "lower_full_sl"}.issubset(cell_summary.columns)

    worst_cells = pd.read_csv(manifest["outputs"]["worst_cell_tradeoffs"])
    assert {"variant", "selector", "delta_full_sl_rate"}.issubset(worst_cells.columns)

    variant_metadata_path = Path(manifest["outputs"]["variant_metadata"])
    assert variant_metadata_path.exists()
