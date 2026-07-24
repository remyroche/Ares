#!/usr/bin/env python3
"""Read-only May-July 2026 OOS failure diagnosis.

This report intentionally consumes fixed OOS artifacts and realized paths only.
It neither fits nor tunes predictive models, calibrators, or admission thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa

from extreme_price_movements.diagnostics.concentration_market import (
    build_concentration_market_diagnostics,
)
from extreme_price_movements.diagnostics.distribution_drift import (
    may_june_july_feature_drift,
    nearest_neighbor_losing_trade_diagnostic,
)
from extreme_price_movements.diagnostics.failure_analysis import (
    FailureAnalysisConfig,
    analyze_failure_diagnostics,
)
from extreme_price_movements.diagnostics.performance_calibration import (
    calibration_diagnostics,
    daily_performance_decomposition,
    meta_score_tail_diagnostics,
    monthly_performance_comparison,
)
from extreme_price_movements.static_feature_store import read_static_features


RANGE_START = pd.Timestamp("2026-05-01 00:00:00", tz="UTC")
RANGE_END = pd.Timestamp("2026-07-10 21:00:00", tz="UTC")
JULY_PARTIAL_LABEL = "2026-07 (partial through 2026-07-10 21:00 UTC)"

DEFAULT_LONG_SOURCE = ROOT / (
    "data_perp/reports/"
    "s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_"
    "residual_only_hpo150_wf30_v1/oos_predictions.parquet"
)
DEFAULT_SHORT_SOURCE = ROOT / (
    "data_perp/reports/may_july_failure_diagnosis_20260722_v1/"
    "short_v9_oos_postprocessed/postprocessed_oos_predictions.parquet"
)
DEFAULT_OUTCOME_SOURCE = ROOT / (
    "data_perp/reports/"
    "s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/"
    "best_oos_scored_ledger.parquet"
)
DEFAULT_ELIGIBLE_SYMBOLS = ROOT / (
    "data_perp/reports/"
    "s59_h5_signalclose_causal_stagec_packb_sliding365_meta_hpo150_wf30_"
    "20260721_v1/best_full_oos/p90spread_fee15bps_eligible170/eligible_symbols.csv"
)
DEFAULT_SHORT_MANIFEST = ROOT / (
    "data_perp/artifacts/hybrid_short_v9only_replay_bundle_20260722_v1/manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "data_perp/reports/may_july_failure_diagnosis_20260722_v1/read_only_diagnosis"
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"

KEYS = ["timestamp", "symbol", "side"]
OUTCOME_COLUMNS = [
    "__ts__",
    "__symbol__",
    "side_name",
    "__first_touch_full_path_mfe_norm__",
    "__first_touch_full_path_mae_norm__",
    "__first_touch_bar__",
    "__is_timeout__",
    "__first_touch_stop__",
    "__first_touch_hit__",
    "gmm_posterior_max",
    "gmm_posterior_margin",
    "gmm_ood_score",
    "gmm_entropy",
    "min_mahalanobis",
    "expected_mahalanobis",
    "dae_reconstruction_error_zscore",
    "cluster_speed",
    "cluster_acceleration",
    "latent_speed",
    "latent_acceleration",
    "path_entropy_12",
    "directional_entropy_20",
    *(f"dae_b16_{index:02d}" for index in range(16)),
]
SIDE_FEATURES = {
    "long": (
        "dae_b16_05",
        "prog_eff_24",
        "volume_z_24",
        "mkt_price_up_oi_up_1h",
        "pct_assets_price_down_oi_up_1h",
        "mkt_ret_per_oi_change_1h",
    ),
    "short": (
        "pct_assets_new_low_7d",
        "gmm_ood_score",
        "prog_eff_24",
        "ret120h",
        "rvol_z",
        "mkt_oi_chg_z_24h",
    ),
}
NN_REFERENCE_LIMIT = 4_000
NN_COMPARISON_LIMIT = 1_000
BOOTSTRAP_REPLICATES = 10_000
MARKET_FEATURES = (
    "ret1h",
    "atr_pct_base",
    "rv_24h",
    "market_dispersion_24h",
    "avg_pair_corr_24h",
    "market_pc1_variance_share_12h",
    "efficiency_ratio_20",
    "wick_ratio_4h_max",
    "volume_zscore_48h",
    "btc_oi_dominance_z_ratio",
    "ret24h",
    "eth_ret_24h",
    "market_breadth_1h",
    "mkt_funding_mean",
    "volatility_of_volatility_48",
    "trend_r2_24",
    "binned_return_entropy_24",
)


def _available_columns(path: Path) -> set[str]:
    return set(ds.dataset(path, format="parquet").schema.names)


def _read_projected(path: Path, wanted: Iterable[str]) -> pd.DataFrame:
    available = _available_columns(path)
    columns = [name for name in dict.fromkeys(wanted) if name in available]
    if not columns:
        raise ValueError(f"No requested columns found in {path}")
    return pd.read_parquet(path, columns=columns)


def _read_outcomes(path: Path) -> pd.DataFrame:
    available = _available_columns(path)
    missing = [column for column in OUTCOME_COLUMNS[:3] if column not in available]
    if missing:
        raise KeyError(f"Outcome source is missing join keys: {missing}")
    columns = [column for column in OUTCOME_COLUMNS if column in available]
    timestamp_filter = (
        (ds.field("__ts__") >= pa.scalar(RANGE_START.to_pydatetime()))
        & (ds.field("__ts__") <= pa.scalar(RANGE_END.to_pydatetime()))
    )
    table = ds.dataset(path, format="parquet").to_table(columns=columns, filter=timestamp_filter)
    frame = table.to_pandas()
    frame = frame.rename(
        columns={
            "__ts__": "timestamp",
            "__symbol__": "symbol",
            "side_name": "side",
            "__first_touch_full_path_mfe_norm__": "mfe",
            "__first_touch_full_path_mae_norm__": "mae",
            "__first_touch_bar__": "holding_time",
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["side"] = frame["side"].astype(str).str.lower()
    for column in ("mfe", "mae", "holding_time"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    timeout = pd.to_numeric(frame.get("__is_timeout__"), errors="coerce").fillna(0).gt(0)
    stop = pd.to_numeric(frame.get("__first_touch_stop__"), errors="coerce").fillna(0).gt(0)
    hit = pd.to_numeric(frame.get("__first_touch_hit__"), errors="coerce").fillna(0).gt(0)
    frame["exit_reason"] = np.select(
        [timeout, stop, hit], ["timeout", "stop", "target"], default="unresolved"
    )
    if frame.duplicated(KEYS).any():
        count = int(frame.duplicated(KEYS, keep=False).sum())
        raise ValueError(f"Outcome source has {count} duplicate UTC timestamp/symbol/side keys")
    state_columns = [
        column
        for column in frame.columns
        if column.startswith("gmm_")
        or column.startswith("dae_")
        or column
        in {
            "min_mahalanobis",
            "expected_mahalanobis",
            "cluster_speed",
            "cluster_acceleration",
            "latent_speed",
            "latent_acceleration",
            "path_entropy_12",
            "directional_entropy_20",
        }
    ]
    for column in state_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return frame[KEYS + ["mfe", "mae", "holding_time", "exit_reason", *state_columns]]


def _source_columns(side: str) -> list[str]:
    common = [
        "__ts__", "__symbol__", "side_name", "ev_after_1pct", "score", "score_base",
        "archetype_policy_key", "clean_exec", "dirty_positive", "full_path_bad_mae_1r", "timeout",
    ]
    if side == "long":
        specific = [
            "score_base_residual_ev_rank_train_reference",
            "score_base_ev_residual_expert_hier_mapped",
            "score_base_ev_mapped",
        ]
    else:
        specific = [
            "score_meta_base_soft_label", "calibrated_score", "hit_probability", "historical_rank",
            "expected_net_ev_after_1pct_side_archetype", "expected_net_ev_after_1pct",
        ]
    return common + specific + list(SIDE_FEATURES[side])


def _first_present(frame: pd.DataFrame, names: Iterable[str], label: str) -> str:
    for name in names:
        if name in frame and pd.to_numeric(frame[name], errors="coerce").notna().any():
            return name
    raise KeyError(f"{label} is missing; tried {list(names)}")


def _normalize_source(frame: pd.DataFrame, side: str) -> tuple[pd.DataFrame, dict[str, str]]:
    required = {"__ts__", "__symbol__", "side_name", "ev_after_1pct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{side} source missing required columns: {missing}")
    source = frame.copy()
    source["timestamp"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source["symbol"] = source["__symbol__"].astype(str)
    source["side"] = source["side_name"].astype(str).str.lower()
    source = source.loc[
        source["timestamp"].between(RANGE_START, RANGE_END) & source["side"].eq(side)
    ].copy()
    if side == "long":
        rank_col = _first_present(source, ["score_base_residual_ev_rank_train_reference"], "long rank")
        probability_col = _first_present(source, ["score", "score_base"], "long probability")
        meta_col = probability_col
        expected_col = _first_present(
            source,
            ["score_base_ev_residual_expert_hier_mapped", "score_base_ev_mapped"],
            "long mapped expected EV",
        )
        model = "long_base_sharedstore_residual_only"
        horizon = "h5"
    else:
        rank_col = _first_present(source, ["historical_rank"], "short historical rank")
        probability_col = _first_present(
            source, ["hit_probability", "score_meta_base_soft_label", "calibrated_score"],
            "short hit probability",
        )
        meta_col = _first_present(
            source, ["calibrated_score", "score_meta_base_soft_label", "hit_probability"],
            "short meta score",
        )
        expected_col = _first_present(
            source,
            ["expected_net_ev_after_1pct_side_archetype", "expected_net_ev_after_1pct"],
            "short mapped expected EV",
        )
        model = "hybrid_short_v9only_replay_bundle_20260722_v1"
        horizon = "h5"
    out = pd.DataFrame(index=source.index)
    out["timestamp"] = source["timestamp"]
    out["symbol"] = source["symbol"]
    out["side"] = side
    out["archetype"] = source.get("archetype_policy_key", "__missing__").astype(str)
    out["setup"] = out["archetype"]
    out["base_model"] = model
    out["meta_model"] = model if side == "short" else "residual_ev_rank_train_reference"
    out["horizon"] = horizon
    out["predicted_probability"] = pd.to_numeric(source[probability_col], errors="coerce")
    out["base_probability"] = pd.to_numeric(source["score_base"], errors="coerce")
    out["meta_score"] = pd.to_numeric(source[meta_col], errors="coerce")
    out["rank_score"] = pd.to_numeric(source[rank_col], errors="coerce")
    out["expected_pnl"] = pd.to_numeric(source[expected_col], errors="coerce")
    out["ev_after_1pct"] = pd.to_numeric(source["ev_after_1pct"], errors="coerce")
    out["calibration_target"] = pd.to_numeric(source.get("clean_exec"), errors="coerce")
    out["dirty_positive"] = pd.to_numeric(source.get("dirty_positive"), errors="coerce")
    out["full_path_bad_mae"] = pd.to_numeric(source.get("full_path_bad_mae_1r"), errors="coerce")
    out["timeout"] = pd.to_numeric(source.get("timeout"), errors="coerce")
    for feature in SIDE_FEATURES[side]:
        if feature in source:
            out[f"feature__{feature}"] = pd.to_numeric(source[feature], errors="coerce")
    mapping = {
        "rank_score": rank_col,
        "predicted_probability": probability_col,
        "meta_score": meta_col,
        "expected_pnl": expected_col,
    }
    return out, mapping


def _period_label(timestamps: pd.Series) -> pd.Series:
    month = pd.to_datetime(timestamps, utc=True).dt.strftime("%Y-%m")
    return month.where(month.ne("2026-07"), JULY_PARTIAL_LABEL)


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = frame[column].astype("float32")
    return frame


def _load_market_context(
    feature_store: Path | None,
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read market context through the canonical shared static endpoint."""
    if feature_store is None or not Path(feature_store).exists():
        return pd.DataFrame(), pd.DataFrame(), {"status": "not_requested"}
    store = Path(feature_store)
    store_ts = pd.to_datetime(store.name, format="%Y%m%d_%H%M%S", utc=True)
    loaded = read_static_features(
        feature_store_ts=store_ts,
        data_root=store.parents[1],
        feature_keys=MARKET_FEATURES,
        symbols=symbols,
        start_ts=RANGE_START,
        end_ts=RANGE_END + pd.Timedelta(hours=1),
        output_layout="panels",
    )
    if not loaded:
        return pd.DataFrame(), pd.DataFrame(), {"status": "empty"}
    available = sorted(set(MARKET_FEATURES).intersection(loaded.keys()))
    first = loaded[available[0]]
    timestamps = pd.DatetimeIndex(first.index)
    timestamps = timestamps[(timestamps >= RANGE_START) & (timestamps <= RANGE_END)]
    present_symbols = [symbol for symbol in symbols if symbol in first.columns]
    state = pd.DataFrame({"timestamp": timestamps})
    aliases = {
        "atr_pct_base": "atr",
        "rv_24h": "rv",
        "market_dispersion_24h": "dispersion",
        "avg_pair_corr_24h": "correlation",
        "market_pc1_variance_share_12h": "pc1_share",
        "efficiency_ratio_20": "trend_efficiency",
        "wick_ratio_4h_max": "wick",
        "volume_zscore_48h": "volume_z",
        "btc_oi_dominance_z_ratio": "btc_dominance",
        "eth_ret_24h": "eth_return",
        "market_breadth_1h": "market_breadth",
        "mkt_funding_mean": "funding_proxy",
        "volatility_of_volatility_48": "volatility_of_volatility",
        "trend_r2_24": "trend",
        "binned_return_entropy_24": "entropy",
    }
    for source, destination in aliases.items():
        if source not in loaded:
            continue
        panel = loaded[source].reindex(index=timestamps, columns=present_symbols)
        state[destination] = panel.median(axis=1, skipna=True).to_numpy(dtype=np.float32)
    if "ret24h" in loaded:
        return_panel = loaded["ret24h"].reindex(index=timestamps)
        btc = next((name for name in return_panel.columns if str(name).startswith("BTC/")), None)
        if btc is not None:
            state["btc_return"] = pd.to_numeric(return_panel[btc], errors="coerce").to_numpy(dtype=np.float32)
    market_returns = pd.DataFrame()
    if "ret1h" in loaded:
        panel = loaded["ret1h"].reindex(index=timestamps, columns=present_symbols)
        market_returns = (
            panel.rename_axis("timestamp")
            .stack(future_stack=True)
            .rename("return")
            .rename_axis(index=["timestamp", "symbol"])
            .reset_index()
        )
        market_returns["return"] = pd.to_numeric(market_returns["return"], errors="coerce").astype("float32")
    return market_returns, state, {
        "status": "loaded",
        "endpoint": "read_static_features",
        "feature_store": str(store),
        "requested_features": list(MARKET_FEATURES),
        "available_features": available,
        "symbols_requested": len(symbols),
        "symbols_present": len(present_symbols),
    }


def _stable_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    positions = np.linspace(0, len(frame) - 1, limit, dtype=int)
    return frame.sort_values(KEYS, kind="stable").iloc[positions]


def _tail_tables(population: pd.DataFrame) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    scopes = [("all_periods", population)]
    scopes.extend((str(period), group) for period, group in population.groupby("period_label", sort=True))
    for period, period_group in scopes:
        for side, group in period_group.groupby("side", sort=True):
            for stage, score_column in (
                ("base", "base_probability"),
                ("meta_or_residual", "rank_score"),
            ):
                table = meta_score_tail_diagnostics(
                    group,
                    score_col=score_column,
                    ev_col="net_return",
                    net_return_col="net_return",
                    mfe_col="mfe",
                    mae_col="mae",
                    target_col="calibration_target",
                )
                table.insert(0, "period_label", period)
                table.insert(1, "side", side)
                table.insert(2, "stage", stage)
                tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _cumulative_tail_tables(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("all_periods", population)]
    scopes.extend((str(period), group) for period, group in population.groupby("period_label", sort=True))
    for period, period_group in scopes:
        for side, group in period_group.groupby("side", sort=True):
            for stage, score_column in (("base", "base_probability"), ("meta_or_residual", "rank_score")):
                ordered = group.loc[group[score_column].notna()].sort_values(
                    [score_column, "timestamp", "symbol"],
                    ascending=[False, True, True],
                    kind="stable",
                )
                for percentage in (1, 2, 5, 10):
                    count = max(1, int(math.ceil(len(ordered) * percentage / 100.0)))
                    tail = ordered.iloc[:count]
                    wins = tail.loc[tail["net_return"].gt(0), "net_return"]
                    losses = tail.loc[tail["net_return"].lt(0), "net_return"]
                    rows.append(
                        {
                            "period_label": period,
                            "side": side,
                            "stage": stage,
                            "top_pct": percentage,
                            "candidate_rows": len(ordered),
                            "trade_count": len(tail),
                            "net_ev_mean": tail["net_return"].mean(),
                            "net_ev_sum": tail["net_return"].sum(),
                            "win_rate": tail["net_return"].gt(0).mean(),
                            "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() else np.nan,
                            "clean_precision": tail["calibration_target"].ge(0.5).mean(),
                            "mfe_mean": tail["mfe"].mean(),
                            "mae_mean": tail["mae"].mean(),
                        }
                    )
    return pd.DataFrame(rows)


def _calibration_by_side(population: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics: list[pd.DataFrame] = []
    reliability: list[pd.DataFrame] = []
    for side, group in population.groupby("side", sort=True):
        for stage, score_column in (
            ("base", "base_probability"),
            ("meta_probability", "predicted_probability"),
            ("final_rank_as_score", "rank_score"),
        ):
            result = calibration_diagnostics(
                group.assign(target=group["calibration_target"], diagnostic_score=group[score_column]),
                target_col="target", score_col="diagnostic_score",
                group_cols=["period_label", "archetype"],
            )
            for table, destination in ((result["metrics"], metrics), (result["reliability"], reliability)):
                table = table.copy()
                table.insert(1, "side", side)
                table.insert(2, "stage", stage)
                destination.append(table)
    return {
        "metrics": pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(),
        "reliability": pd.concat(reliability, ignore_index=True) if reliability else pd.DataFrame(),
    }


def _concentration_by_side(
    selected: pd.DataFrame,
    *,
    market_returns: pd.DataFrame | None = None,
    market_state: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    daily: list[pd.DataFrame] = []
    breaks: list[pd.DataFrame] = []
    months: list[pd.DataFrame] = []
    for side, group in selected.groupby("side", sort=True):
        candidate_state_features = [
            column for column in group.columns
            if column.startswith("feature__") or column.startswith("state__")
        ]
        state_features = [
            column for column in candidate_state_features
            if pd.to_numeric(group[column], errors="coerce").notna().mean() >= 0.90
            and pd.to_numeric(group[column], errors="coerce").nunique(dropna=True) > 1
        ][:24]
        embeddings = [
            column for column in state_features if column.startswith("state__dae_b16_")
        ]
        result = build_concentration_market_diagnostics(
            group,
            market_returns=market_returns,
            market_state=market_state,
            model_col="base_model",
            prediction_columns=("predicted_probability", "rank_score"),
            feature_columns=state_features,
            embedding_columns=embeddings,
        )
        for table, destination in (
            (result.daily, daily),
            (result.structural_breaks, breaks),
            (result.monthly_comparisons, months),
        ):
            table = table.copy()
            table.insert(1, "side", side)
            destination.append(table)
    return {
        "daily": pd.concat(daily, ignore_index=True) if daily else pd.DataFrame(),
        "structural_breaks": pd.concat(breaks, ignore_index=True) if breaks else pd.DataFrame(),
        "monthly_comparisons": pd.concat(months, ignore_index=True) if months else pd.DataFrame(),
    }


def _selection_support(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for side, group in population.groupby("side", sort=True):
        ranks = pd.to_numeric(group["rank_score"], errors="coerce")
        fixed = ranks.ge(0.90)
        valid = ranks.notna()
        count = int(valid.sum())
        exact_count = int(math.ceil(count * 0.10)) if count else 0
        exact_index = ranks.loc[valid].sort_values(ascending=False, kind="mergesort").index[:exact_count]
        exact = pd.Series(False, index=group.index)
        exact.loc[exact_index] = True
        rows.append(
            {
                "side": side,
                "candidate_support": count,
                "fixed_rank_ge_090_support": int(fixed.sum()),
                "fixed_rank_ge_090_share": float(fixed.mean()) if len(group) else np.nan,
                "exact_global_per_side_top10_support": exact_count,
                "exact_global_per_side_top10_overlap_with_fixed": int((fixed & exact).sum()),
                "exact_global_per_side_top10_only": int((exact & ~fixed).sum()),
                "fixed_only": int((fixed & ~exact).sum()),
            }
        )
    return pd.DataFrame(rows)


def _model_slices(selected: pd.DataFrame) -> pd.DataFrame:
    columns = ["period_label", "side", "archetype", "base_model", "meta_model"]
    return (
        selected.groupby(columns, observed=True, dropna=False)
        .agg(
            support=("net_return", "size"),
            net_return_sum=("net_return", "sum"),
            net_return_mean=("net_return", "mean"),
            expected_pnl_sum=("expected_pnl", "sum"),
            residual_sum=("residual", "sum"),
            win_rate=("net_return", lambda value: float(value.gt(0).mean())),
            mfe_mean=("mfe", "mean"),
            mae_mean=("mae", "mean"),
            holding_time_mean=("holding_time", "mean"),
            clean_exec_rate=("calibration_target", "mean"),
            dirty_positive_rate=("dirty_positive", "mean"),
            full_path_bad_mae_rate=("full_path_bad_mae", "mean"),
            timeout_rate=("timeout", "mean"),
        )
        .reset_index()
    )


def _weekly_archetype_metrics(selected: pd.DataFrame) -> pd.DataFrame:
    work = selected.assign(
        week_start=selected["timestamp"].dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    )
    return (
        work.groupby(["week_start", "side", "archetype"], observed=True, dropna=False)
        .agg(
            trades=("net_return", "size"),
            net_ev_mean=("net_return", "mean"),
            net_ev_sum=("net_return", "sum"),
            gross_ev_mean=("gross_return", "mean"),
            win_rate=("net_return", lambda value: float(value.gt(0).mean())),
            expected_ev_mean=("expected_pnl", "mean"),
            residual_mean=("residual", "mean"),
            mfe_mean=("mfe", "mean"),
            mae_mean=("mae", "mean"),
            holding_time_mean=("holding_time", "mean"),
        )
        .reset_index()
    )


def _monthly_block_bootstrap(selected: pd.DataFrame) -> pd.DataFrame:
    """Moving-block bootstrap of monthly EV/trade using UTC days as the unit."""
    work = selected.assign(month=selected["timestamp"].dt.strftime("%Y-%m"), day=selected["timestamp"].dt.floor("D"))
    scopes = [("__all__", work)]
    scopes.extend((str(side), group) for side, group in work.groupby("side", sort=True))
    rows: list[dict[str, Any]] = []
    for side, side_group in scopes:
        for month, group in side_group.groupby("month", sort=True):
            daily = group.groupby("day", sort=True)["net_return"].agg(["sum", "count"])
            n_days = len(daily)
            if n_days == 0:
                continue
            block = min(7, n_days)
            if n_days < 14:
                rows.append(
                    {
                        "month": month,
                        "side": side,
                        "days": n_days,
                        "trades": int(len(group)),
                        "net_ev_mean": float(group["net_return"].mean()),
                        "bootstrap_p025": np.nan,
                        "bootstrap_p50": np.nan,
                        "bootstrap_p975": np.nan,
                        "bootstrap_probability_ev_le_zero": np.nan,
                        "bootstrap_replicates": 0,
                        "block_days": block,
                        "status": "insufficient_days_for_7d_block_bootstrap",
                    }
                )
                continue
            block_count = int(math.ceil(n_days / block))
            rng = np.random.default_rng(20260722 + sum(ord(char) for char in f"{side}:{month}"))
            starts = rng.integers(0, n_days, size=(BOOTSTRAP_REPLICATES, block_count))
            offsets = np.arange(block, dtype=np.int64)
            indices = (starts[..., None] + offsets) % n_days
            indices = indices.reshape(BOOTSTRAP_REPLICATES, -1)[:, :n_days]
            sums = daily["sum"].to_numpy(dtype=np.float64)[indices].sum(axis=1)
            counts = daily["count"].to_numpy(dtype=np.float64)[indices].sum(axis=1)
            samples = sums / np.maximum(counts, 1.0)
            rows.append(
                {
                    "month": month,
                    "side": side,
                    "days": n_days,
                    "trades": int(len(group)),
                    "net_ev_mean": float(group["net_return"].mean()),
                    "bootstrap_p025": float(np.quantile(samples, 0.025)),
                    "bootstrap_p50": float(np.quantile(samples, 0.50)),
                    "bootstrap_p975": float(np.quantile(samples, 0.975)),
                    "bootstrap_probability_ev_le_zero": float(np.mean(samples <= 0.0)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "block_days": block,
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def _performance_with_sides(selected: pd.DataFrame, *, monthly: bool) -> pd.DataFrame:
    function = monthly_performance_comparison if monthly else daily_performance_decomposition
    global_rows = function(selected, holding_col="holding_time")
    global_rows.insert(1, "side", "__all__")
    side_rows: list[pd.DataFrame] = [global_rows]
    for side, group in selected.groupby("side", sort=True):
        table = function(group, holding_col="holding_time")
        table.insert(1, "side", side)
        side_rows.append(table)
    return pd.concat(side_rows, ignore_index=True)


def _failure_config(market_state: pd.DataFrame) -> FailureAnalysisConfig:
    return FailureAnalysisConfig(
        expected_pnl_col="expected_pnl",
        realized_pnl_col="realized_pnl",
        bankroll_pnl_col="bankroll_pnl",
        base_model_col="base_model",
        side_col="side",
        setup_col="setup",
        horizon_col="horizon",
        symbol_col="symbol",
        residual_z_window_days=20,
        residual_z_min_periods=5,
        residual_percentile_min_days=10,
        percentile_reference="causal",
        sequence_group_cols=("side",),
        market_state_cols=tuple(column for column in market_state.columns if column != "timestamp"),
    )


def _side_failure_tables(
    selected: pd.DataFrame,
    market_state: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, list[pd.DataFrame]] = {
        "daily": [], "episodes": [], "pre_vs_during": [], "similarity": []
    }
    for side, group in selected.groupby("side", sort=True):
        work = group
        if not market_state.empty:
            work = group.merge(market_state, on="timestamp", how="left", validate="many_to_one")
        result = analyze_failure_diagnostics(
            work.assign(realized_pnl=work["net_return"]),
            _failure_config(market_state),
        )
        for key, table in (
            ("daily", result.daily),
            ("episodes", result.episodes),
            ("pre_vs_during", result.episode_comparisons),
            ("similarity", result.episode_similarity),
        ):
            table = table.copy()
            table.insert(0, "analysis_side", side)
            outputs[key].append(table)
    return {
        key: pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        for key, tables in outputs.items()
    }


def _drift_tables(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    distribution: list[pd.DataFrame] = []
    features: list[pd.DataFrame] = []
    common_numeric = ("rank_score", "predicted_probability", "expected_pnl")
    for side, group in selected.groupby("side", sort=True):
        try:
            distribution.append(
                may_june_july_feature_drift(
                    group, timestamp_column="timestamp", numeric_features=common_numeric,
                    categorical_features=("archetype",), year=2026, include_worst_day=True,
                    outcome_column="net_return",
                ).assign(side=side)
            )
            feature_columns = [column for column in group if column.startswith("feature__")]
            features.append(
                may_june_july_feature_drift(
                    group, timestamp_column="timestamp", numeric_features=feature_columns,
                    year=2026, include_worst_day=True, outcome_column="net_return",
                ).assign(side=side)
            )
        except ValueError as exc:
            raise ValueError(f"{side} drift requires May, June, and July rows: {exc}") from exc
    return (
        pd.concat(distribution, ignore_index=True) if distribution else pd.DataFrame(),
        pd.concat(features, ignore_index=True) if features else pd.DataFrame(),
    )


def _nn_diagnostics(selected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    details: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    for side, group in selected.groupby("side", sort=True):
        feature_columns = [
            column for column in group.columns
            if column.startswith("feature__") or column.startswith("state__")
        ]
        feature_columns = feature_columns[:32] + ["rank_score", "expected_pnl"]
        feature_columns = list(dict.fromkeys(feature_columns))
        reference = _stable_sample(
            group.loc[group["timestamp"].lt(pd.Timestamp("2026-07-01", tz="UTC"))],
            NN_REFERENCE_LIMIT,
        )
        comparison = _stable_sample(
            group.loc[
                group["timestamp"].ge(pd.Timestamp("2026-07-01", tz="UTC"))
                & group["net_return"].lt(0)
            ],
            NN_COMPARISON_LIMIT,
        )
        if comparison.empty:
            summaries[side] = {"n_losing_queries": 0, "note": "no selected losing trades"}
            continue
        medians = reference[feature_columns].apply(pd.to_numeric, errors="coerce").median()
        reference_features = reference[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(medians)
        comparison_features = comparison[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(medians)
        finite_columns = np.isfinite(reference_features.to_numpy(dtype=float)).all(axis=0)
        reference_features = reference_features.loc[:, finite_columns]
        comparison_features = comparison_features.loc[:, finite_columns]
        if reference_features.shape[1] == 0:
            summaries[side] = {"n_losing_queries": len(comparison), "note": "no finite pre-entry features"}
            continue
        diagnostic = nearest_neighbor_losing_trade_diagnostic(
            comparison_features, reference_features,
            reference_is_loss=reference["net_return"].lt(0),
            comparison_is_loss=np.ones(len(comparison), dtype=bool),
            reference_month=reference["period_label"], comparison_month=comparison["period_label"],
            reference_timestamps=reference["timestamp"], comparison_timestamps=comparison["timestamp"],
            reference_ids=reference["symbol"], comparison_ids=comparison["symbol"], k=20,
        )
        detail = diagnostic.neighbors.copy()
        detail.insert(0, "side", side)
        details.append(detail)
        summaries[side] = dict(diagnostic.summary)
        summaries[side]["feature_count"] = int(reference_features.shape[1])
        summaries[side]["reference_scope"] = "May-June selected rows"
        summaries[side]["comparison_scope"] = "July selected losing rows"
    return pd.concat(details, ignore_index=True) if details else pd.DataFrame(), summaries


def _write_csv(output_dir: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(output_dir / f"{name}.csv", index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


def run_diagnosis(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    long_source: Path = DEFAULT_LONG_SOURCE,
    short_source: Path = DEFAULT_SHORT_SOURCE,
    outcome_source: Path = DEFAULT_OUTCOME_SOURCE,
    eligible_symbols: Path = DEFAULT_ELIGIBLE_SYMBOLS,
    short_manifest: Path = DEFAULT_SHORT_MANIFEST,
    feature_store: Path | None = DEFAULT_FEATURE_STORE,
) -> dict[str, Any]:
    """Build read-only diagnosis outputs and return its compact manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = pd.read_csv(eligible_symbols, usecols=["symbol", "p90_spread_bps"])
    eligible["symbol"] = eligible["symbol"].astype(str)
    eligible["p90_spread_bps"] = pd.to_numeric(eligible["p90_spread_bps"], errors="coerce")
    if eligible["symbol"].duplicated().any() or eligible["p90_spread_bps"].isna().any():
        raise ValueError("Eligible symbol/spread map must have one finite p90 spread per symbol")

    long_raw = _read_projected(Path(long_source), _source_columns("long"))
    short_raw = _read_projected(Path(short_source), _source_columns("short"))
    long_rows, long_mapping = _normalize_source(long_raw, "long")
    short_rows, short_mapping = _normalize_source(short_raw, "short")
    source_rows = pd.concat([long_rows, short_rows], ignore_index=True, sort=False)
    before_eligible = len(source_rows)
    source_rows = source_rows.merge(eligible, on="symbol", how="inner", validate="many_to_one")
    outcomes = _read_outcomes(Path(outcome_source))
    joined = source_rows.merge(outcomes, on=KEYS, how="left", validate="one_to_one", indicator=True)
    unmatched_outcomes = int(joined["_merge"].ne("both").sum())
    joined = joined.drop(columns="_merge")
    joined["gross_return"] = joined["ev_after_1pct"] + np.float32(0.01)
    joined["net_return"] = joined["gross_return"] - np.float32(0.003) - joined["p90_spread_bps"] / np.float32(10_000)
    joined["bankroll_pnl"] = joined["net_return"]
    joined["residual"] = joined["net_return"] - joined["expected_pnl"]
    joined["period_label"] = _period_label(joined["timestamp"])
    joined = joined.rename(
        columns={
            column: f"state__{column}"
            for column in joined.columns
            if column.startswith("gmm_")
            or column.startswith("dae_")
            or column
            in {
                "min_mahalanobis", "expected_mahalanobis", "cluster_speed",
                "cluster_acceleration", "latent_speed", "latent_acceleration",
                "path_entropy_12", "directional_entropy_20",
            }
        }
    )
    joined = _downcast(joined)
    population = joined.loc[joined["net_return"].notna() & joined["rank_score"].notna()].copy()
    selected = population.loc[population["rank_score"].ge(0.90)].copy()
    if set(selected["side"]) != {"long", "short"}:
        raise ValueError("Fixed rank_score >= 0.90 population must include both long and short rows")

    daily = _performance_with_sides(selected, monthly=False)
    monthly = _performance_with_sides(selected, monthly=True)
    monthly["period_label"] = monthly["month"].astype(str).replace({"2026-07": JULY_PARTIAL_LABEL})
    daily["period_label"] = _period_label(daily["date"])
    daily["economic_bad_day"] = daily["net_return_sum"].lt(0) & daily["side"].eq("__all__")
    daily["economic_bad_day_rule"] = "net_return_sum < 0"

    market_returns, market_state, market_manifest = _load_market_context(
        feature_store,
        sorted(eligible["symbol"].unique().tolist()),
    )
    failure_selected = selected
    if not market_state.empty:
        failure_selected = selected.merge(market_state, on="timestamp", how="left", validate="many_to_one")

    calibration = _calibration_by_side(population)
    tails = _tail_tables(population)
    cumulative_tails = _cumulative_tail_tables(population)
    distribution_drift, feature_drift = _drift_tables(selected)
    concentration = _concentration_by_side(
        selected,
        market_returns=market_returns,
        market_state=market_state,
    )
    failure = analyze_failure_diagnostics(
        failure_selected.assign(realized_pnl=failure_selected["net_return"]),
        _failure_config(market_state),
    )
    side_failures = _side_failure_tables(selected, market_state)
    failure_daily = failure.daily.merge(
        daily.loc[daily["side"].eq("__all__"), ["date", "economic_bad_day", "net_return_sum"]],
        left_on="day", right_on="date", how="left"
    ).drop(columns="date")
    failure_daily["period_label"] = _period_label(failure_daily["day"])
    failure_daily["statistical_failure_episode"] = failure_daily["failure_day"].astype(bool)
    model_slices = _model_slices(selected)
    weekly_archetype = _weekly_archetype_metrics(selected)
    monthly_bootstrap = _monthly_block_bootstrap(selected)
    selection_support = _selection_support(population)
    nn_neighbors, nn_summary = _nn_diagnostics(selected)

    selected.to_parquet(output_dir / "normalized_selected_ledger.parquet", index=False)
    population_core_columns = [
        "timestamp", "symbol", "side", "archetype", "setup", "base_model", "meta_model",
        "horizon", "base_probability", "predicted_probability", "meta_score", "rank_score",
        "expected_pnl", "gross_return", "net_return", "calibration_target", "period_label",
        "dirty_positive", "full_path_bad_mae", "timeout",
    ]
    population.loc[:, population_core_columns].to_parquet(
        output_dir / "normalized_population_core.parquet", index=False
    )

    tables = {
        "daily": daily,
        "monthly": monthly,
        "calibration_metrics": calibration["metrics"],
        "calibration_reliability": calibration["reliability"],
        "disjoint_tails": tails,
        "cumulative_tails": cumulative_tails,
        "distribution_drift": distribution_drift,
        "feature_drift": feature_drift,
        "concentration_daily": concentration["daily"],
        "concentration_structural_breaks": concentration["structural_breaks"],
        "concentration_monthly": concentration["monthly_comparisons"],
        "failure_daily": failure_daily,
        "failure_episodes": failure.episodes,
        "failure_daily_by_side": side_failures["daily"],
        "failure_episodes_by_side": side_failures["episodes"],
        "pre_vs_during_by_side": side_failures["pre_vs_during"],
        "episode_similarity_by_side": side_failures["similarity"],
        "pre_vs_during": failure.episode_comparisons,
        "counterfactuals": failure.counterfactual_removals,
        "sequence": failure.sequence_metrics,
        "episode_similarity": failure.episode_similarity,
        "model_slices": model_slices,
        "weekly_archetype": weekly_archetype,
        "monthly_block_bootstrap": monthly_bootstrap,
        "selection_support": selection_support,
        "nn_loss_neighbors": nn_neighbors,
        "market_state_hourly": market_state,
    }
    for name, table in tables.items():
        _write_csv(output_dir, name, table)

    short_manifest_payload = json.loads(Path(short_manifest).read_text())
    manifest = {
        "schema": "may_july_failure_diagnosis_v1",
        "read_only": True,
        "predictive_model_fitting_or_tuning": False,
        "resolved_range_utc": {"start": RANGE_START, "end_inclusive": RANGE_END},
        "july_label": JULY_PARTIAL_LABEL,
        "sources": {
            "long": str(long_source), "short": str(short_source), "outcomes": str(outcome_source),
            "eligible_symbols": str(eligible_symbols), "short_hybrid_manifest": str(short_manifest),
        },
        "short_hybrid_manifest": short_manifest_payload,
        "side_mappings": {"long": long_mapping, "short": short_mapping},
        "cost_contract": {
            "gross_return": "ev_after_1pct + 0.01",
            "net_return": "gross_return - 0.003 - p90_spread_bps / 10000",
            "bankroll_pnl": "unit-notional proxy equal to net_return",
            "cost_count": 1,
        },
        "selection_contract": {
            "fixed_population": "rank_score >= 0.90",
            "reranked_by_month_or_day": False,
            "global_per_side_top10": "reported as a support diagnostic only",
        },
        "row_counts": {
            "source_before_eligible_filter": before_eligible,
            "eligible_source_rows": len(source_rows), "unmatched_outcome_rows": unmatched_outcomes,
            "resolved_ranked_population": len(population), "fixed_selected_population": len(selected),
        },
        "failure_contract": {
            "economic_bad_day": "daily selected net_return sum < 0",
            "statistical_failure_episode": failure.manifest["failure_day_rule"],
        },
        "nn_loss_diagnostics": nn_summary,
        "market_context": market_manifest,
        "path_metric_contract": {
            "mfe": "first-touch full-path MFE normalized by the row barrier",
            "mae": "first-touch full-path MAE normalized by the row barrier",
            "holding_time": "first-touch hourly bar count",
        },
        "evidence_limits": [
            "July is partial through 2026-07-10 21:00 UTC.",
            "Bankroll PnL is a unit-notional proxy because no comparable combined-side portfolio replay exists.",
            "The fixed rank>=0.90 population is a frozen score cut, not a reconstruction of every downstream portfolio admission.",
        ],
        "outputs": sorted(f"{name}.csv" for name in tables),
    }
    manifest["outputs"].extend(
        ["normalized_population_core.parquet", "normalized_selected_ledger.parquet"]
    )
    (output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    report = [
        "# May-July 2026 Failure Diagnosis",
        "",
        "Read-only retrospective diagnosis. No predictive model fitting, tuning, calibrator fitting, or threshold selection is performed.",
        "",
        f"Resolved UTC range: {RANGE_START.isoformat()} through {RANGE_END.isoformat()} inclusive.",
        f"July label: {JULY_PARTIAL_LABEL}.",
        "",
        "Fixed selected population: `rank_score >= 0.90`; no month/day reranking. Full-population tail diagnostics are computed separately.",
        "Economic bad day: selected daily `net_return` sum < 0. Statistical failure episodes are residual diagnostics and remain separate.",
        "Bankroll PnL is explicitly a unit-notional proxy; this is not a combined-side portfolio replay.",
        "",
        f"Resolved ranked rows: {len(population)}; fixed selected rows: {len(selected)}; unmatched outcome rows: {unmatched_outcomes}.",
        "",
        "Outputs: " + ", ".join(f"`{name}.csv`" for name in sorted(tables)),
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    return _json_safe(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--long-source", type=Path, default=DEFAULT_LONG_SOURCE)
    parser.add_argument("--short-source", type=Path, default=DEFAULT_SHORT_SOURCE)
    parser.add_argument("--outcome-source", type=Path, default=DEFAULT_OUTCOME_SOURCE)
    parser.add_argument("--eligible-symbols", type=Path, default=DEFAULT_ELIGIBLE_SYMBOLS)
    parser.add_argument("--short-manifest", type=Path, default=DEFAULT_SHORT_MANIFEST)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--skip-market-context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_diagnosis(
        output_dir=args.output_dir, long_source=args.long_source, short_source=args.short_source,
        outcome_source=args.outcome_source, eligible_symbols=args.eligible_symbols,
        short_manifest=args.short_manifest,
        feature_store=None if args.skip_market_context else args.feature_store,
    )
    print(json.dumps(manifest["row_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
