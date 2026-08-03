#!/usr/bin/env python3
"""Materialise a chronological Aug-2025--Jul-2026 Pack-B two-layer ledger.

This is deliberately a reporting/research contract, not a retrospective final
refit: each calendar month is scored by a side-local base model fitted only on
earlier resolved rows; the residual/meta model is trained only on base scores
that were themselves generated out of sample inside that earlier history.

The simulated admission rule is causal and side-local.  A trade is admitted
only when a robust 21-day map of that side's *earlier OOS* scores estimates at
least +0.5% realised policy net return.  The score map uses daily side-relative
score percentiles so refitted monthly model scales are not compared directly.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression


SIDES = ("long", "short")
SOURCE_DEFAULT = Path("data_perp/artifacts/20260720_s59_h5_fullthroughjul10_candleclose_trailing_cost100bps_labels/labels")
FEATURE_STORE_DEFAULT = Path("data_perp/features/20260711_070000")
FULL_UNIVERSE_FEATURE_ROOT = Path("data_perp/artifacts/full_universe_base_features40_20260802_v2")
OUTPUT_DEFAULT = Path("data_perp/artifacts/packb_yearly_side_local_oos_20260802_v1")
BASE_TARGET = "__first_touch_target_soft__"
NET_TARGET = "__r_policy_net__"
DECISION = "__ts__"
LABEL_LAG = pd.Timedelta(hours=24)  # policy paths can use up to 24 bars
BASE_LOOKBACK_DAYS = 180
INNER_OOF_DAYS = 90
BASE_TOP_FRACTION = .40
MAP_DAYS = 21
MAP_THRESHOLD = .005  # +0.5% net return
MIN_MAP_ROWS = 500
MAP_BINS = 20

# These are pre-existing point-in-time Pack-B/source-regime fields.  They are
# intentionally distinct per layer and side; no outcome/path/target field is
# allowed into either matrix.
BASE_FEATURES = {
    "long": [
        "G_VOL", "__regime_vol_12h__", "__regime_trend_12h__", "__meta_raw__chop_score", "ret1h_G_VOL_1",
        "__regime_source_trend_path_score__", "__regime_source_execution_quality_score__",
        "__regime_source_oi_agreement_score__", "__regime_source_location_quality_score__",
        "__regime_source_pullback_retest_score__", "__regime_source_compression_score__",
        "__regime_source_volume_confirmation_score__", "__regime_source_barrier_pressure_score__",
        "__regime_source_quiet_continuation_score__", "__regime_source_base_positive_source_score__",
        "__regime_source_run_entry_score__", "__regime_source_late_run_continuation_score__",
        "__regime_source_barrier_relief_score__", "__regime_source_clean_execution_context_score__",
        "__regime_source_calm_positive_source_score__", "__regime_source_clean_run_entry_score__",
        "__regime_source_compression_capture_candidate_score__", "__regime_source_risk_adjusted_capture_candidate_score__",
        "__regime_source_clean_economic_capture_candidate_score__", "__regime_source_trend_following_score__",
        "__regime_source_vol_compression_score__", "__regime_source_breakout_impulse_score__",
    ],
    "short": [
        "G_VOL", "__regime_vol_48h__", "__regime_trend_48h__", "__meta_raw__volatility_zscore", "ret1h_G_VOL_0",
        "__regime_source_shock_impulse_score__", "__regime_source_execution_risk_score__",
        "__regime_source_oi_agreement_score__", "__regime_source_location_quality_score__",
        "__regime_source_barrier_pressure_score__", "__regime_source_dirty_shock_avoid_score__",
        "__regime_source_retest_reversal_score__", "__regime_source_not_dirty_shock_score__",
        "__regime_source_loud_clean_source_score__", "__regime_source_misleading_location_risk_score__",
        "__regime_source_mean_reversion_score__", "__regime_source_breakout_impulse_score__",
        "__regime_source_dirty_avoid_score__", "__regime_source_execution_quality_score__",
        "__regime_source_volume_confirmation_score__", "__regime_source_quiet_continuation_score__",
        "__regime_source_barrier_relief_score__", "__regime_source_loud_clean_execution_score__",
        "__regime_source_risk_adjusted_capture_candidate_score__", "__regime_source_clean_economic_capture_candidate_score__",
    ],
}
META_FEATURES = {
    "long": [
        "G_VOL", "__regime_vol_48h__", "__regime_volume_12h__", "__regime_volume_48h__", "__regime_trend_48h__",
        "__meta_raw__volatility_zscore", "ret1h_G_VOL_0", "__regime_source_shock_impulse_score__",
        "__regime_source_execution_risk_score__", "__regime_source_dirty_shock_avoid_score__",
        "__regime_source_retest_reversal_score__", "__regime_source_not_dirty_shock_score__",
        "__regime_source_loud_clean_source_score__", "__regime_source_misleading_location_risk_score__",
        "__regime_source_mean_reversion_score__", "__regime_source_dirty_avoid_score__",
    ],
    "short": [
        "G_VOL", "__regime_vol_12h__", "__regime_volume_12h__", "__regime_volume_48h__", "__regime_trend_12h__",
        "__meta_raw__chop_score", "ret1h_G_VOL_1", "__regime_source_trend_path_score__",
        "__regime_source_execution_quality_score__", "__regime_source_pullback_retest_score__",
        "__regime_source_compression_score__", "__regime_source_base_positive_source_score__",
        "__regime_source_run_entry_score__", "__regime_source_late_run_continuation_score__",
        "__regime_source_clean_execution_context_score__", "__regime_source_clean_run_entry_score__",
    ],
}

# Context belongs to the residual/EV layer, not the base alpha model.  These
# are pre-existing config-owned market/cross-sectional fields from the
# authoritative point-in-time store.  They intentionally differ by side and
# are not copied from the base's asset-local 40-field selection.
STORE_META_FEATURES = {
    "long": [
        "mkt_oi_breadth_rising_24h", "mkt_oi_chg_z_24h", "mkt_oi_dispersion_24h",
        "mkt_ret_eq_24h", "mkt_ret_eq_4h", "xasset_mkt_spread_bps",
        "mkt_oi_drawdown_from_24h_peak",
        "mkt_price_up_oi_up_4h", "pct_assets_up_4h", "mkt_funding_mean_z_30d",
        "funding_crowding_x_vol_expansion", "breadth_recovery_from_6h_min",
        "xs_dispersion__xasset_ob_liquidity_peer_resid",
        "xs_dispersion__trend_pct_mkt_resid", "q_upper_tail__xasset_mkt_spread_bps",
    ],
    "short": [
        "mkt_oi_breadth_rising_24h", "mkt_oi_chg_z_24h", "mkt_oi_dispersion_24h",
        "mkt_ret_eq_24h", "mkt_ret_eq_4h", "xasset_mkt_spread_bps",
        "mkt_oi_drawdown_from_7d_peak",
        "mkt_price_down_oi_down_4h", "pct_assets_price_down_oi_down_4h",
        "mkt_funding_dispersion_z_30d", "positive_funding_x_price_down",
        "funding_flip_x_oi_flush", "xs_dispersion__ob_spread_bps_z_24h",
        "xs_dispersion__vol_z_peer_resid", "q_lower_tail__ob_spread_bps_z_24h",
    ],
}


def _store_base_features() -> dict[str, list[str]]:
    """Load the already-selected, per-side 40-field causal base contracts."""
    result: dict[str, list[str]] = {}
    for side in SIDES:
        path = FULL_UNIVERSE_FEATURE_ROOT / side / "target_family_manifest.json"
        payload = json.loads(path.read_text())
        key = f"T2_soft_barrier|tp3_sl2|{side}"
        fields = [
            field for field in payload["feature_contract"][key]
            if field not in {"ret24h_bench_resid", "ret4h_bench_resid"}
        ]
        if len(fields) != 39 or len(set(fields)) != len(fields):
            raise ValueError(f"invalid full-universe {side} base feature contract")
        result[side] = fields
    return result


@dataclass(frozen=True)
class RunSpec:
    scored_start: pd.Timestamp
    end: pd.Timestamp
    max_train_rows: int


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.normalize().replace(day=1), end.normalize().replace(day=1), freq="MS", tz="UTC"))


def _safe_matrix(frame: pd.DataFrame, fields: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = values.median() if medians is None else medians
    return values.fillna(fill).fillna(0.).astype(np.float32), fill


def _sample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    # Stable hash-like stride, not a target-dependent sample.
    ordered = frame.sort_values([DECISION, "candidate_id"], kind="mergesort")
    positions = np.linspace(0, len(ordered) - 1, maximum, dtype=int)
    return ordered.iloc[positions]


def _fit(train: pd.DataFrame, fields: list[str], target: str, maximum: int, binary: bool = False) -> tuple[LGBMRegressor | LGBMClassifier, pd.Series]:
    chosen = _sample(train, maximum)
    x, medians = _safe_matrix(chosen, fields)
    y = pd.to_numeric(chosen[target], errors="raise").astype(float)
    common = dict(
        n_estimators=350, learning_rate=.045,
        num_leaves=31, min_child_samples=250, subsample=.8, colsample_bytree=.8,
        reg_lambda=4., random_state=20260802, n_jobs=-1, verbosity=-1,
    )
    if binary:
        if not y.isin([0., 1.]).all():
            raise ValueError(f"binary base objective requires a 0/1 target, got {target}")
        model: LGBMRegressor | LGBMClassifier = LGBMClassifier(objective="binary", **common)
    else:
        model = LGBMRegressor(objective="regression_l2", **common)
    model.fit(x, y)
    return model, medians


def _predict(model: LGBMRegressor | LGBMClassifier, frame: pd.DataFrame, fields: list[str], medians: pd.Series) -> np.ndarray:
    x, _ = _safe_matrix(frame, fields, medians)
    if isinstance(model, LGBMClassifier):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)
    return np.asarray(model.predict(x), dtype=np.float32)


def _base_context(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["base_score_raw"] = score
    ordered = out.sort_values([DECISION, "side_name", "base_score_raw", "candidate_id"], ascending=[True, True, False, True], kind="mergesort")
    rank = ordered.groupby([DECISION, "side_name"], sort=False).cumcount() + 1
    count = ordered.groupby([DECISION, "side_name"], sort=False)["candidate_id"].transform("size")
    ordered["base_rank_pct_by_timestamp_side"] = 1. - (rank - 1.) / np.maximum(count - 1., 1.)
    selected = ordered[ordered.base_rank_pct_by_timestamp_side.ge(1. - BASE_TOP_FRACTION)].copy()
    cutoff = selected.groupby([DECISION, "side_name"], sort=False).base_score_raw.min().rename("base_cutoff")
    stats = ordered.groupby([DECISION, "side_name"], sort=False).base_score_raw.agg(["mean", "std"])
    selected = selected.join(cutoff, on=[DECISION, "side_name"]).join(stats, on=[DECISION, "side_name"])
    selected["base_margin_to_cutoff"] = selected.base_score_raw - selected.base_cutoff
    selected["base_margin_to_cutoff_z"] = selected.base_margin_to_cutoff.div(selected["std"].where(selected["std"].gt(1e-12))).fillna(0.)
    selected["base_score_z_timestamp_side"] = selected.base_score_raw.sub(selected["mean"]).div(selected["std"].where(selected["std"].gt(1e-12))).fillna(0.)
    # The score percentile is derived only from prior base predictions inside
    # a meta training fold, never from the realised policy return.
    selected["base_score_rank_pct_train_prior"] = selected.base_rank_pct_by_timestamp_side
    return selected.drop(columns=["base_cutoff", "mean", "std"])


def _meta_fields(side: str, meta_features: dict[str, list[str]]) -> list[str]:
    return ["base_score_raw", "base_rank_pct_by_timestamp_side", "base_score_rank_pct_train_prior", "base_margin_to_cutoff", "base_margin_to_cutoff_z", "base_score_z_timestamp_side", *meta_features[side]]


def _score_month(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, maximum: int, lookback_days: int, inner_oof_days: int, inner_base_warmup_days: int, base_target: str, base_binary: bool, base_features: dict[str, list[str]], meta_features: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, object]]:
    history_start = start - pd.Timedelta(days=lookback_days)
    history = data[
        (data[DECISION].ge(history_start))
        & (data[DECISION].lt(start))
        & (data["__label_available_at__"].lt(start))
    ].copy()
    evaluation = data[(data[DECISION].ge(start)) & (data[DECISION].lt(end))].copy()
    if evaluation.empty:
        return evaluation, {"month": str(start), "status": "no evaluation rows"}
    outputs: list[pd.DataFrame] = []
    audit: dict[str, object] = {"month": str(start), "history_start": str(history_start), "side": {}}
    for side in SIDES:
        train = history[history.side_name.eq(side)].copy()
        test = evaluation[evaluation.side_name.eq(side)].copy()
        if min(len(train), len(test)) < 1000:
            raise ValueError(f"{start:%Y-%m} {side}: insufficient rows train={len(train)} test={len(test)}")
        inner_start = max(history_start + pd.Timedelta(days=inner_base_warmup_days), start - pd.Timedelta(days=inner_oof_days))
        early = train[train[DECISION].lt(inner_start)].copy()
        inner = train[train[DECISION].ge(inner_start)].copy()
        if min(len(early), len(inner)) < 1000:
            raise ValueError(f"{start:%Y-%m} {side}: insufficient inner OOF support")
        base_inner, med_inner = _fit(early, base_features[side], base_target, maximum, base_binary)
        inner_context = _base_context(inner, _predict(base_inner, inner, base_features[side], med_inner))
        meta_fields = _meta_fields(side, meta_features)
        meta, meta_medians = _fit(inner_context, meta_fields, NET_TARGET, maximum)
        base, base_medians = _fit(train, base_features[side], base_target, maximum, base_binary)
        test_context = _base_context(test, _predict(base, test, base_features[side], base_medians))
        test_context["meta_expected_net_return"] = _predict(meta, test_context, meta_fields, meta_medians)
        test_context["base_fit_resolved_before"] = start
        test_context["meta_fit_resolved_before"] = inner_start
        output_fields = list(dict.fromkeys([
            "candidate_id", DECISION, "__symbol__", "side_name", base_target, NET_TARGET,
            "__label_available_at__", "base_score_raw", "base_rank_pct_by_timestamp_side",
            "base_margin_to_cutoff", "base_margin_to_cutoff_z", "base_score_z_timestamp_side",
            "meta_expected_net_return", "base_fit_resolved_before", "meta_fit_resolved_before",
        ]))
        outputs.append(test_context.loc[:, output_fields])
        audit["side"][side] = {"base_train_rows": int(len(train)), "meta_oof_rows": int(len(inner_context)), "evaluation_rows": int(len(test_context)), "base_target": base_target, "base_objective": "binary_logloss" if base_binary else "regression_l2", "base_features": base_features[side], "meta_features": meta_fields}
    return pd.concat(outputs, ignore_index=True), audit


def _score_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ordered = out.sort_values(["side_name", DECISION, "meta_expected_net_return", "candidate_id"], ascending=[True, True, False, True], kind="mergesort")
    rank = ordered.groupby(["side_name", DECISION], sort=False).cumcount() + 1
    count = ordered.groupby(["side_name", DECISION], sort=False).candidate_id.transform("size")
    ordered["meta_score_percentile_side_day"] = 1. - (rank - 1.) / np.maximum(count - 1., 1.)
    return ordered.sort_index()


def _robust_map(reference: pd.DataFrame, current: pd.DataFrame) -> np.ndarray:
    # Bin first and use a lightly winsorised *mean* in every bin.  EV is an
    # expectation, so a median would incorrectly turn asymmetric winners into
    # the typical stop/timeout outcome.  Winsorisation prevents one path outlier
    # from setting admission while retaining economically relevant upside.
    work = reference.copy()
    work["bin"] = np.minimum((work.meta_score_percentile_side_day * MAP_BINS).astype(int), MAP_BINS - 1)
    def robust_mean(values: pd.Series) -> float:
        ordered = np.sort(values.to_numpy(float))
        trim = int(np.floor(len(ordered) * .05))
        trimmed = ordered[trim: len(ordered) - trim] if len(ordered) - 2 * trim >= 1 else ordered
        return float(trimmed.mean())
    bins = work.groupby("bin", sort=True).agg(x=("meta_score_percentile_side_day", "median"), y=(NET_TARGET, robust_mean), n=(NET_TARGET, "size")).reset_index()
    if len(bins) < 4:
        raise ValueError("fewer than four robust map bins")
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(bins.x, bins.y, sample_weight=bins.n)
    return iso.predict(current.meta_score_percentile_side_day)


def _apply_admission(scored: pd.DataFrame) -> pd.DataFrame:
    out = _score_percentile(scored)
    out["mapped_21d_ev_net_return"] = np.nan
    out["admitted_21d_ev_ge_0p5pct"] = False
    for day in sorted(out[DECISION].dt.normalize().unique()):
        asof = pd.Timestamp(day)
        if asof.tzinfo is None:
            asof = asof.tz_localize("UTC")
        for side in SIDES:
            current = out[(out[DECISION].dt.normalize().eq(asof)) & (out.side_name.eq(side))]
            ref = out[(out[DECISION].lt(asof)) & (out["__label_available_at__"].lt(asof)) & (out["__label_available_at__"].ge(asof - pd.Timedelta(days=MAP_DAYS))) & (out.side_name.eq(side))]
            if len(ref) < MIN_MAP_ROWS:
                continue
            try:
                mapped = _robust_map(ref, current)
            except ValueError:
                continue
            out.loc[current.index, "mapped_21d_ev_net_return"] = mapped
            out.loc[current.index, "admitted_21d_ev_ge_0p5pct"] = mapped >= MAP_THRESHOLD
    return out


def _period_metrics(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    work = frame.copy()
    work["period_start"] = work[DECISION].dt.to_period(period).dt.start_time.dt.tz_localize("UTC")
    rows: list[dict[str, object]] = []
    for start, group in work.groupby("period_start", sort=True):
        for side in ("global", *SIDES):
            part = group if side == "global" else group[group.side_name.eq(side)]
            selected = part[part.admitted_21d_ev_ge_0p5pct]
            rows.append({"period_start": start, "side": side, "candidate_rows": int(len(part)), "admitted_trades": int(len(selected)), "admission_rate": float(len(selected) / len(part)) if len(part) else np.nan, "mapped_ev_bps": float(selected.mapped_21d_ev_net_return.mean() * 1e4) if len(selected) else np.nan, "realised_net_bps": float(selected[NET_TARGET].mean() * 1e4) if len(selected) else np.nan, "realised_net_sum_bps": float(selected[NET_TARGET].sum() * 1e4) if len(selected) else 0., "raw_score_ic": float(part[["meta_expected_net_return", NET_TARGET]].corr(method="spearman").iloc[0, 1]) if len(part) > 2 else np.nan})
    return pd.DataFrame(rows)


def _join_store_features(frame: pd.DataFrame, store: Path, fields: list[str], begin: pd.Timestamp, finish: pd.Timestamp) -> pd.DataFrame:
    """Exact point-in-time symbol/timestamp join from the historical store.

    Labels retain their original identity and outcome columns.  Store fields
    are read only for the label window and joined on the feature-close time;
    neither forward fill nor a future as-of lookup is permitted.
    """
    out = frame.copy()
    out["__store_symbol__"] = out["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    missing = sorted(set(out["__store_symbol__"].unique()) - {p.name.removeprefix("symbol=").removesuffix(".parquet") for p in store.glob("symbol=*.parquet")})
    if missing:
        raise ValueError(f"feature store misses label symbols: {missing[:10]}")
    for symbol, index in out.groupby("__store_symbol__", sort=False).groups.items():
        path = store / f"symbol={symbol}.parquet"
        # pandas exposes a parquet index as ``ts`` when restoring the frame.
        available = set(pq.ParquetFile(path).schema_arrow.names)
        present = [field for field in fields if field in available]
        timestamp_field = "ts" if "ts" in available else "__index_level_0__"
        source = pd.read_parquet(path, columns=present, filters=[(timestamp_field, ">=", begin), (timestamp_field, "<", finish)])
        source.index = pd.to_datetime(source.index, utc=True)
        aligned = source.reindex(pd.DatetimeIndex(out.loc[index, DECISION]))
        out.loc[index, present] = aligned.to_numpy(np.float32, copy=False)
    out = out.drop(columns="__store_symbol__")
    values = out.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    coverage = 1. - values.isna().mean()
    constant = values.nunique(dropna=True).le(1)
    bad = coverage[coverage.lt(.90)].index.tolist() + constant[constant].index.tolist()
    if bad:
        raise ValueError(f"store feature coverage/variance gate failed: {sorted(set(bad))}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-dir", type=Path, default=SOURCE_DEFAULT)
    p.add_argument("--feature-store", type=Path, default=None, help="authoritative historical point-in-time store; enables the validated full-universe feature contract")
    p.add_argument("--out", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--scored-start", default="2025-08-01")
    p.add_argument("--end", default="2026-07-11")
    p.add_argument("--max-train-rows", type=int, default=250000)
    p.add_argument("--base-lookback-days", type=int, default=BASE_LOOKBACK_DAYS)
    p.add_argument("--base-target", default=BASE_TARGET, help="causal label field used by the base model")
    p.add_argument("--base-binary", action="store_true", help="fit the base as a binary log-loss classifier; requires a 0/1 target")
    p.add_argument("--inner-oof-days", type=int, default=INNER_OOF_DAYS)
    p.add_argument("--inner-base-warmup-days", type=int, default=60, help="minimum earlier resolved history for the base model that creates meta-training OOF scores")
    p.add_argument("--skip-admission", action="store_true", help="write one or more raw chronological OOS months; combine them later before fitting the 21-day map")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    spec = RunSpec(pd.Timestamp(a.scored_start, tz="UTC"), pd.Timestamp(a.end, tz="UTC"), a.max_train_rows)
    paths = sorted(a.labels_dir.glob("train_global_*_5_*.parquet"))
    if len(paths) != 38:
        raise ValueError(f"expected 38 side-month label shards, found {len(paths)}")
    use_store = a.feature_store is not None
    base_features = _store_base_features() if use_store else BASE_FEATURES
    meta_features = STORE_META_FEATURES if use_store else META_FEATURES
    model_fields = set().union(*base_features.values(), *meta_features.values())
    needed = {a.base_target, NET_TARGET} | (set() if use_store else model_fields)
    read_columns = list(dict.fromkeys([
        "candidate_id", DECISION, "__symbol__", "side_name", a.base_target,
        NET_TARGET, *sorted(needed - {a.base_target, NET_TARGET}),
    ]))
    # The source holds 38 large monthly shards.  Read only the seven-or-fewer
    # months required by a given fit/score fold, and only its frozen causal
    # fields; this keeps the full-year materialisation bounded in memory.
    source_by_month: dict[pd.Period, list[Path]] = {}
    for path in paths:
        token = path.stem.rsplit("_", 2)[-2:]
        month = pd.Period(f"{token[0]}-{token[1]}", freq="M")
        source_by_month.setdefault(month, []).append(path)
    if any(len(value) != 2 for value in source_by_month.values()):
        raise ValueError("each source month must contain both sides")

    def load_window(begin: pd.Timestamp, finish: pd.Timestamp) -> pd.DataFrame:
        months = pd.period_range(begin.to_period("M"), (finish - pd.Timedelta(nanoseconds=1)).to_period("M"), freq="M")
        selected = [path for month in months for path in source_by_month.get(month, [])]
        if not selected:
            raise ValueError(f"no source shards for {begin}--{finish}")
        frame = pd.concat([pd.read_parquet(path, columns=read_columns) for path in selected], ignore_index=True)
        missing = needed - set(frame.columns)
        if missing:
            raise KeyError(f"source lacks feature/target fields: {sorted(missing)}")
        frame[DECISION] = pd.to_datetime(frame[DECISION], utc=True)
        frame["side_name"] = frame.side_name.astype(str).str.lower()
        frame["__label_available_at__"] = frame[DECISION] + LABEL_LAG
        frame = frame[frame[DECISION].ge(begin) & frame[DECISION].lt(finish)].copy()
        coverage = 1. - frame.loc[:, sorted(needed)].replace([np.inf, -np.inf], np.nan).isna().mean()
        if (coverage < .90).any():
            raise ValueError(f"causal feature coverage below 90% in {begin:%Y-%m}: {coverage[coverage < .90].to_dict()}")
        if set(frame.side_name.unique()) != set(SIDES) or frame.candidate_id.duplicated().any():
            raise ValueError("source must contain unique rows for exactly long and short")
        if use_store:
            frame = _join_store_features(frame, a.feature_store, sorted(model_fields), begin, finish)
        return frame
    starts = _month_starts(
        spec.scored_start if a.skip_admission else spec.scored_start - pd.offsets.MonthBegin(1),
        spec.end,
    )
    outputs, audits = [], []
    for start in starts:
        end = min(start + pd.offsets.MonthBegin(1), spec.end)
        if end <= start:
            continue
        part, audit = _score_month(load_window(start - pd.Timedelta(days=a.base_lookback_days), end), start, end, spec.max_train_rows, a.base_lookback_days, a.inner_oof_days, a.inner_base_warmup_days, a.base_target, a.base_binary, base_features, meta_features)
        outputs.append(part); audits.append(audit)
        print(json.dumps({"month": str(start), "rows": len(part)}), flush=True)
    scored = pd.concat(outputs, ignore_index=True)
    scored = scored[scored[DECISION].lt(spec.end)].copy()
    a.out.mkdir(parents=True)
    if a.skip_admission:
        scored.to_parquet(a.out / "raw_oos_predictions.parquet", index=False)
        manifest = {"schema": "packb_yearly_side_local_oos_v1", "status": "raw_oos_months_materialized", "scored_window": {"start": str(spec.scored_start), "end_exclusive": str(spec.end)}, "feature_store": str(a.feature_store) if use_store else None, "base": {"target": a.base_target, "objective": "binary_logloss" if a.base_binary else "regression_l2", "lookback_days": a.base_lookback_days, "top_fraction_timestamp_side": BASE_TOP_FRACTION, "features": base_features}, "meta": {"target": NET_TARGET, "strict_base_oof_training_days": a.inner_oof_days, "inner_base_warmup_days": a.inner_base_warmup_days, "features": meta_features}, "label_availability_lag_hours": 24, "row_counts": {"raw_oos": int(len(scored))}, "month_audit": audits}
    else:
        admitted = _apply_admission(scored)
        reporting = admitted[admitted[DECISION].ge(spec.scored_start)].copy()
        reporting.to_parquet(a.out / "oos_predictions.parquet", index=False)
        _period_metrics(reporting, "W-SUN").to_parquet(a.out / "weekly_metrics.parquet", index=False)
        _period_metrics(reporting, "M").to_parquet(a.out / "monthly_metrics.parquet", index=False)
        manifest = {"schema": "packb_yearly_side_local_oos_v1", "status": "materialized", "scored_window": {"start": str(spec.scored_start), "end_exclusive": str(spec.end)}, "warmup_oos_month": str(starts[0]), "feature_store": str(a.feature_store) if use_store else None, "base": {"target": a.base_target, "objective": "binary_logloss" if a.base_binary else "regression_l2", "lookback_days": a.base_lookback_days, "top_fraction_timestamp_side": BASE_TOP_FRACTION, "features": base_features}, "meta": {"target": NET_TARGET, "strict_base_oof_training_days": a.inner_oof_days, "inner_base_warmup_days": a.inner_base_warmup_days, "features": meta_features}, "admission": {"side_local": True, "map_days": MAP_DAYS, "robust_bin_median_isotonic": True, "net_ev_threshold": MAP_THRESHOLD, "minimum_reference_rows": MIN_MAP_ROWS}, "label_availability_lag_hours": 24, "row_counts": {"scored_warmup_and_reporting": int(len(admitted)), "reporting": int(len(reporting)), "admitted": int(reporting.admitted_21d_ev_ge_0p5pct.sum())}, "month_audit": audits}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
