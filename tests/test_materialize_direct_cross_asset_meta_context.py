from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_direct_cross_asset_meta_context import _join_feature_store_asof, _normalize_ledger, run


def _synthetic_ledger() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    months = pd.period_range("2026-01", "2026-04", freq="M")
    for month_i, month in enumerate(months):
        for side in ("long", "short"):
            for source in ("mean_reversion", "breakout"):
                for i in range(160):
                    edge_feature = rng.normal()
                    drift_feature = rng.normal()
                    score = edge_feature + 0.2 * rng.normal()
                    signed_edge = 0.012 * edge_feature + (0.003 if source == "mean_reversion" else -0.001)
                    if side == "short":
                        signed_edge += 0.002
                    net = signed_edge + 0.02 + 0.01 * rng.normal()
                    if edge_feature < -1.0:
                        exit_reason = "full_sl"
                    elif drift_feature > 1.3:
                        exit_reason = "timeout"
                    elif net > 0.025:
                        exit_reason = "hard_tp"
                    else:
                        exit_reason = "trailing"
                    rows.append(
                        {
                            "timestamp": month.to_timestamp(how="start") + pd.Timedelta(hours=i),
                            "symbol": f"SYM{i % 5}/USD:USD",
                            "side": side,
                            "head": source,
                            "strategy_id": f"{side}_{source}",
                            "normalized_rank_score": score,
                            "oof_prob_uncertainty": abs(drift_feature),
                            "meta_lgbm_entropy": abs(drift_feature) / 2.0,
                            "base_lgbm_regime_centroid_similarity_train": edge_feature,
                            "feature_drift_psi_core": drift_feature,
                            "generated_score_entropy": abs(edge_feature - drift_feature),
                            "net_return": net,
                            "simple_policy_exit_reason": exit_reason,
                            "month_i": month_i,
                        }
                    )
    return pd.DataFrame(rows)


def test_materialize_direct_cross_asset_meta_context_outputs(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.parquet"
    out_dir = tmp_path / "out"
    _synthetic_ledger().to_parquet(ledger_path, index=False)

    manifest = run(
        ledger_path=ledger_path,
        output_dir=out_dir,
        feature_dir=None,
        base_oof_dir=None,
        base_oof_horizon="H5",
        max_asof_staleness_minutes=90,
        max_context_features=12,
        n_components=3,
        n_clusters=3,
        max_fit_rows=1_000,
        min_group_rows=20,
    )

    assert manifest["stability_features"] == "excluded_by_user_request"
    assert manifest["context_feature_count"] >= 4
    assert manifest["latent_feature_count"] == 6
    assert Path(manifest["outputs"]["handoff"]).exists()
    assert Path(manifest["outputs"]["topk_metrics"]).exists()
    assert Path(manifest["outputs"]["deltas"]).exists()

    handoff = pd.read_parquet(manifest["outputs"]["handoff"])
    first_month = sorted(handoff["month"].unique())[0]
    later = handoff["month"] != first_month
    assert handoff.loc[handoff["month"].eq(first_month), "xctx_ev_score_oof"].isna().all()
    assert handoff.loc[later, "xctx_ev_score_oof"].notna().any()
    assert {"xctx_latent_0", "xctx_cluster_entropy", "exec_ev_after_1pct_cost"}.issubset(handoff.columns)

    metrics = pd.read_csv(manifest["outputs"]["topk_metrics"])
    assert {"precision_positive_ev", "ev_weighted_precision", "full_sl_rate", "timeout_rate"}.issubset(
        metrics.columns
    )
    assert set(metrics["score_col"]).issuperset({"xctx_baseline_score", "xctx_ev_score_oof", "xctx_blend_score"})

    features = json.loads(Path(manifest["outputs"]["feature_columns"]).read_text())
    assert all("stability" not in col.lower() for col in features["context_features"])


def test_feature_store_asof_join_preserves_unmatched_symbols(tmp_path: Path) -> None:
    ledger = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 01:00"], utc=True),
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"],
            "side": ["long", "short"],
            "net_return": [0.02, -0.01],
            "simple_policy_exit_reason": ["hard_tp", "full_sl"],
        }
    )
    ledger = _normalize_ledger(ledger)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01 00:00"], utc=True),
            "btc_ret_24h_pct": [0.5],
        }
    ).to_parquet(feature_dir / "symbol=AAA_USD:USD.parquet", index=False)

    joined, contract = _join_feature_store_asof(ledger, feature_dir, max_staleness_minutes=90)

    assert len(joined) == len(ledger)
    assert contract["status"] == "joined_asof"
    assert "ctx_btc_ret_24h_pct" in joined.columns
    assert joined.loc[joined["__symbol__"].eq("AAA/USD:USD"), "ctx_btc_ret_24h_pct"].notna().all()
    assert joined.loc[joined["__symbol__"].eq("BBB/USD:USD"), "ctx_btc_ret_24h_pct"].isna().all()
