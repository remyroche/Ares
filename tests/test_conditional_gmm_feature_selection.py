from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_conditional_gmm_feature_selection import (
    ConditionalFeatureSelectionConfig,
    build_side_aware_targets,
    classify_feature_family,
    infer_lookback_hours,
    run_conditional_feature_selection,
    run_conditional_selection_on_frame,
)


def _synthetic_side_aware_frame(n_per_bucket: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    buckets = np.repeat(np.arange(4), n_per_bucket)
    n = len(buckets)
    ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(buckets * 31, unit="D")
    ts = ts + pd.to_timedelta(np.tile(np.arange(n_per_bucket), 4), unit="h")
    side = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8)
    trend_6h = rng.normal(size=n)
    trend_48h = trend_6h + rng.normal(scale=0.05, size=n)
    funding_24h = rng.normal(size=n)
    volume_48h = rng.normal(size=n)
    vol_24h = rng.normal(size=n)
    bucket_sign = np.where(buckets < 2, 1.0, -1.0)
    utility = (
        0.9 * trend_6h * bucket_sign
        + 0.8 * funding_24h * side
        - 0.6 * np.maximum(vol_24h, 0.0)
        + rng.normal(scale=0.10, size=n)
    )
    mae = np.maximum(0.001, 0.01 + 0.008 * np.maximum(vol_24h, 0.0))
    barrier = np.full(n, 0.012, dtype=np.float32)
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": np.where(np.arange(n) % 3 == 0, "BTC", "ETH"),
            "side": side,
            "trend_6h": trend_6h.astype(np.float32),
            "trend_48h": trend_48h.astype(np.float32),
            "funding_24h": funding_24h.astype(np.float32),
            "volume_48h": volume_48h.astype(np.float32),
            "volatility_24h": vol_24h.astype(np.float32),
            "future_return_6h": utility.astype(np.float32),
            "__y_ret__": utility.astype(np.float32),
            "__u_policy_net__": utility.astype(np.float32),
            "__r_policy_net__": (utility - 0.2 * mae).astype(np.float32),
            "__mae_ret__": mae.astype(np.float32),
            "__mfe_ret__": np.abs(utility).astype(np.float32),
            "__barrier_pct__": barrier,
            "__is_timeout__": (volume_48h > 1.3).astype(np.float32),
            "__y_outcome__": (utility > 0).astype(np.float32) * 2.0,
        }
    )


def _small_config() -> ConditionalFeatureSelectionConfig:
    return ConditionalFeatureSelectionConfig(
        min_feature_finite_frac=0.90,
        min_bucket_rows=20,
        min_side_rows=20,
        shrinkage_k=10.0,
        ic_threshold=0.01,
        max_input_features=20,
        max_selected_pairs=24,
        max_selected_features=8,
        min_unique_values=3,
    )


def test_side_aware_targets_prefer_optimized_economic_utility_columns() -> None:
    frame = _synthetic_side_aware_frame(n_per_bucket=20)
    frame["__u_policy_net__"] = -10.0
    frame["__r_policy_net__"] = -20.0
    frame["__u_econ_net__"] = np.linspace(-0.02, 0.04, len(frame), dtype=np.float32)
    frame["__u_econ_adjusted_net__"] = (
        frame["__u_econ_net__"].to_numpy(dtype=np.float32) - 0.003
    )

    targets, report = build_side_aware_targets(frame)

    assert np.allclose(targets["utility"], frame["__u_econ_net__"])
    assert np.allclose(targets["risk_adjusted_utility"], frame["__u_econ_adjusted_net__"])
    assert report["target_columns"]["utility"]["finite_frac"] == 1.0


def test_feature_family_and_lookback_inference_covers_existing_feature_name_patterns() -> None:
    assert classify_feature_family("adx_14") == "momentum_trend"
    assert classify_feature_family("dn_vol_6") == "volume"
    assert classify_feature_family("path_entropy_12") == "entropy"
    assert classify_feature_family("asset_minus_mkt_oi_1d_peer_resid") == "cross_asset"
    assert classify_feature_family("market_breadth_1h") == "market"
    assert classify_feature_family("xs_dispersion__funding_per_hour") == "cross_sectional"
    assert infer_lookback_hours("dn_vol_6") == 6.0
    assert infer_lookback_hours("adx_di_minus_14") == 14.0
    assert infer_lookback_hours("q_iqr__ret48h_bench_resid") == 48.0
    assert infer_lookback_hours("dist_ema20_atr") == 20.0
    assert infer_lookback_hours("dist_ma100_atr") == 100.0
    assert infer_lookback_hours("dist_prior_day_high") == 24.0


def test_conditional_selection_keeps_weak_global_conditional_and_side_asymmetric_pairs() -> None:
    frame = _synthetic_side_aware_frame()
    result = run_conditional_selection_on_frame(
        frame,
        candidate_features=[
            "trend_6h",
            "trend_48h",
            "funding_24h",
            "volume_48h",
            "volatility_24h",
            "future_return_6h",
            "side_adjusted_trend_6h",
        ],
        config=_small_config(),
        bucket_mode="month",
    )

    validity = result.feature_validity.set_index("feature")
    assert validity.loc["future_return_6h", "status"] == "fail"
    assert validity.loc["side_adjusted_trend_6h", "status"] == "fail"
    assert validity.loc["trend_48h", "status"] == "pass"
    assert float(validity.loc["trend_48h", "horizon_relevance"]) > 0.0

    selected_pairs = result.selected_pairs
    assert selected_pairs["side_coverage_ok"].astype(bool).all()
    assert "trend_6h" in set(selected_pairs["feature"])
    assert bool(selected_pairs[selected_pairs["feature"].eq("trend_6h")]["is_conditional"].any())
    assert "funding_24h" in set(selected_pairs["feature"])
    assert bool(selected_pairs[selected_pairs["feature"].eq("funding_24h")]["is_side_asymmetric"].any())
    assert "side_adjusted_trend_6h" not in set(result.selected_features["feature"])


def test_conditional_selection_rejects_extreme_outlier_features() -> None:
    frame = _synthetic_side_aware_frame()
    frame["pathological_robust_z"] = np.linspace(0.0, 1.0e12, len(frame), dtype=np.float64)

    result = run_conditional_selection_on_frame(
        frame,
        candidate_features=["trend_6h", "pathological_robust_z"],
        config=_small_config(),
        bucket_mode="month",
    )

    validity = result.feature_validity.set_index("feature")
    assert validity.loc["pathological_robust_z", "status"] == "fail"
    assert validity.loc["pathological_robust_z", "reject_reason"] == "extreme_abs_value"


def test_conditional_selection_cli_writes_feature_and_signature_outputs(tmp_path: Path) -> None:
    frame = _synthetic_side_aware_frame()
    labels_path = tmp_path / "labels.parquet"
    output_dir = tmp_path / "selection"
    target_summary_path = tmp_path / "target_summary.json"
    frame.to_parquet(labels_path, index=False)
    target_summary_path.write_text(
        """
        {
          "selected_spec": {"name": "unit"},
          "selected": {
            "objective": 1.0,
            "proxy_top10_mean_net": 0.001,
            "proxy_top10_delta_mean": 0.001,
            "proxy_top10_hit_net": 0.41,
            "proxy_top10_q10_net": -0.01,
            "proxy_top10_ic_net": 0.04,
            "proxy_top10_ic_soft": 0.10,
            "oracle_top10_mean_net": 0.03,
            "hard_rate": 0.2,
            "feasible_rate": 0.9
          },
          "selection_gates": {
            "require_proxy_positive_net": true,
            "min_proxy_mean_net": 0.0,
            "min_proxy_ic_net": 0.0,
            "min_proxy_hit_net": 0.0,
            "min_proxy_q10_net": -1.0
          }
        }
        """,
        encoding="utf-8",
    )

    manifest = run_conditional_feature_selection(
        labels_path=labels_path,
        output_dir=output_dir,
        target_optimization_summary_json=target_summary_path,
        config=_small_config(),
        bucket_mode="month",
    )

    assert Path(manifest["outputs"]["selected_features"]).exists()
    training_feature_list = Path(manifest["outputs"]["training_feature_list"])
    assert training_feature_list.exists()
    training_features = pd.read_csv(training_feature_list)
    assert {
        "feature",
        "selected_feature_position",
        "selected_feature_count",
        "used_by_model",
        "source",
    }.issubset(training_features.columns)
    assert training_features["used_by_model"].astype(str).str.lower().eq("true").all()
    assert Path(manifest["outputs"]["signature_columns"]).exists()
    assert manifest["counts"]["training_feature_list"] == manifest["counts"]["selected_features"]
    assert manifest["target_readiness"]["promotion_status"] == "experimental"
    assert "proxy_top10_q10_net_negative" in manifest["target_readiness"]["weak_reasons"]
    assert manifest["existing_features_only"] is True
    assert manifest["creates_new_live_features"] is False


def test_conditional_selection_inferred_feature_store_rejects_nonnumeric_columns(tmp_path: Path) -> None:
    frame = _synthetic_side_aware_frame(n_per_bucket=20)
    labels_path = tmp_path / "labels.parquet"
    output_dir = tmp_path / "selection"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    frame.to_parquet(labels_path, index=False)

    for symbol, group in frame.groupby("__symbol__"):
        feature_frame = pd.DataFrame(
            {
                "__symbol__": [symbol] * len(group),
                "trend_6h": group["trend_6h"].to_numpy(dtype=np.float32),
                "nonnumeric_symbol": [symbol] * len(group),
            },
            index=pd.to_datetime(group["__ts__"], utc=True),
        )
        feature_frame.to_parquet(feature_dir / f"symbol={symbol}.parquet")

    manifest = run_conditional_feature_selection(
        labels_path=labels_path,
        output_dir=output_dir,
        feature_dir=feature_dir,
        infer_feature_store_schema=True,
        feature_store_schema_files=2,
        config=_small_config(),
        bucket_mode="month",
    )

    validity = pd.read_csv(manifest["outputs"]["feature_validity"]).set_index("feature")
    assert manifest["symbols"] == 2
    assert validity.loc["__symbol__", "status"] == "fail"
    assert validity.loc["trend_6h", "status"] == "pass"
    assert validity.loc["nonnumeric_symbol", "status"] == "fail"


def test_conditional_selection_merges_explicit_and_inferred_feature_store_schema(tmp_path: Path) -> None:
    frame = _synthetic_side_aware_frame(n_per_bucket=20)
    labels_path = tmp_path / "labels.parquet"
    output_dir = tmp_path / "selection"
    feature_dir = tmp_path / "features"
    feature_list_csv = tmp_path / "feature_list.csv"
    feature_dir.mkdir()
    frame.to_parquet(labels_path, index=False)
    pd.DataFrame({"feature": ["trend_6h"]}).to_csv(feature_list_csv, index=False)

    for symbol, group in frame.groupby("__symbol__"):
        feature_frame = pd.DataFrame(
            {
                "path_entropy_12": np.linspace(0.0, 1.0, len(group), dtype=np.float32),
            },
            index=pd.to_datetime(group["__ts__"], utc=True),
        )
        feature_frame.to_parquet(feature_dir / f"symbol={symbol}.parquet")

    manifest = run_conditional_feature_selection(
        labels_path=labels_path,
        output_dir=output_dir,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        infer_feature_store_schema=True,
        feature_store_schema_files=2,
        config=_small_config(),
        bucket_mode="month",
    )

    validity = pd.read_csv(manifest["outputs"]["feature_validity"]).set_index("feature")
    assert validity.loc["trend_6h", "status"] == "pass"
    assert validity.loc["path_entropy_12", "status"] == "pass"
    assert validity.loc["path_entropy_12", "family"] == "entropy"
    assert manifest["counts"]["candidate_features"] >= 2
