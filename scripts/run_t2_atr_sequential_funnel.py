#!/usr/bin/env python3
"""Run the user-specified sequential ATR-normalised T2 funnel.

Order is intentionally fixed and non-factorial:
  1. screen TP {2,3} x SL {1,2} on pre-final-OOS base predictions;
  2. tune only timeout-label softness for that selected geometry;
  3. select a 3/7/14-day residual-training window on a pre-final-OOS month;
  4. open the final meta-OOS only once with the frozen base+residual stack.

All barriers are units of each candidate's entry ATR.  Ordered one-minute H12
paths establish first-touch order; same-minute dual hits are adverse first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.config import (
    T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS,
    T2_FUNNEL_META_CONTEXT_FEATURE_KEYS,
)
from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.t2_atr_funnel import (
    GEOMETRIES,
    BarrierGeometry,
    T2FunnelError,
    materialize_geometry_events_bulk,
    soft_event_targets,
    top_book_metrics,
)


SCHEMA = "t2_atr_sequential_funnel_v1"
CAPACITY = {
    "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15,
    "min_child_samples": 200, "subsample": 0.80, "colsample_bytree": 0.80,
    "reg_lambda": 5.0, "random_state": 20260801, "n_jobs": 1, "verbosity": -1,
}
TEMPERATURES = (0.10, 0.25, 0.50)
RESIDUAL_DAYS = (3, 7, 14)
# This is a compact, causal-only meta subset.  Missing names are discarded
# against the frozen ledger; the manifest records the exact admitted result.
META_CANDIDATES = (
    "ret4h_bench_resid", "ret24h_bench_resid", "ret4h_peer_resid", "ret24h_peer_resid",
    "trend_pct_mkt_resid", "atr_expansion_ts_resid", "rv_24h_peer_resid",
    "rvol_z_peer_resid", "amihud_z_peer_resid", "liquidity_ratio_peer_resid",
    "coherence_24_ts_resid", "fund_abs_z_mkt_resid", "xasset_funding_peer_resid",
    "funding_1d_chg_ts_resid", "asset_minus_mkt_oi_7d_ts_resid",
    "asset_minus_mkt_oi_1d_peer_resid", "asset_minus_mkt_oi_7d_peer_resid",
    "volume_price_corr_ts_resid", "path_efficiency_24_ts_resid", "atr_percentile",
    "mkt_rv_ratio_1h_24h", "mkt_range_expansion_1h", "market_breadth_1h",
    "market_breadth_24h", "market_dispersion_4h", "mkt_funding_dispersion",
    "mkt_oi_chg_4h", "mkt_oi_flush_z_30d", "mkt_ret_4h", "mkt_return_accel_1h",
    "mkt_regime_change__flush_recovery__delta_1h",
    "mkt_regime_change__oi_contraction__delta_1h",
    "mkt_regime_change__negative_breadth__delta_1h",
    "ob_spread_z_24h", "ob_depth_z_10bps", "amihud_z",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _huber(x: np.ndarray, y: np.ndarray, test: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    model = lgb.LGBMRegressor(objective="huber", alpha=0.90, **CAPACITY)
    model.fit(x, y, sample_weight=sample_weight)
    return model.predict(test)


def _conditional_mean(weights: np.ndarray, values: np.ndarray) -> float:
    total = float(weights.sum())
    return float(np.dot(weights, values) / total) if total > 1e-9 else float(values.mean())


def _add_causal_context(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    side = work["side_name"].astype(str).str.lower()
    if not side.isin(["long", "short"]).all():
        raise T2FunnelError("only long and short candidates are admissible")
    work["side_is_long"] = side.eq("long").astype(np.float32)
    # ``execution_cost_return`` belongs to the realised target ledger.  The
    # historical source may have constructed it from a causal cost schedule,
    # but that provenance is not materialised on this T2 ledger.  Never pass
    # the target column through under a causal-looking alias.  A separately
    # materialised, entry-time cost estimate may populate this field in a
    # future contract after its availability has been audited.
    work["causal_entry_cost_bps"] = np.float32(0.0)
    return work


def _resolved_before(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Return only labels strictly available before the first test decision.

    The feature timestamp is one completed hourly bar before the executable
    decision.  Training on a row merely because its *feature* timestamp is
    earlier would still leak its H12 outcome across a fold boundary.  This
    helper makes the 12-hour availability purge explicit and is used for every
    base and residual fit below.
    """
    missing = {"__decision_ts__", "__label_available_at__"}.difference(train.columns) | {"__decision_ts__"}.difference(test.columns)
    if missing:
        raise T2FunnelError(f"strict timing purge requires {sorted(missing)}")
    first_test_decision = pd.to_datetime(test["__decision_ts__"], utc=True, errors="raise").min()
    available = pd.to_datetime(train["__label_available_at__"], utc=True, errors="raise")
    result = train.loc[available.lt(first_test_decision)].copy()
    if result.empty:
        raise T2FunnelError("no resolved training labels remain after the strict H12 purge")
    if not pd.to_datetime(result["__label_available_at__"], utc=True).lt(first_test_decision).all():
        raise T2FunnelError("strict label-availability purge failed")
    return result


def _read_paths(candidate_ids: set[str], path_files: list[Path]) -> pd.DataFrame:
    """Read only row groups containing required candidates from frozen paths."""
    wanted = set(map(str, candidate_ids))
    parts: list[pd.DataFrame] = []
    columns = ["candidate_id", "side_name", "execution_future_path", "atr_1h", "decision_price"]
    for path in path_files:
        source = pq.ParquetFile(path)
        for index in range(source.num_row_groups):
            ids = source.read_row_group(index, columns=["candidate_id"]).column("candidate_id").to_pylist()
            if not wanted.intersection(map(str, ids)):
                continue
            part = source.read_row_group(index, columns=columns).to_pandas()
            part = part.loc[part.candidate_id.astype(str).isin(wanted)]
            if not part.empty:
                parts.append(part)
    if not parts:
        raise T2FunnelError("no required candidates found in frozen H12 paths")
    result = pd.concat(parts, ignore_index=True)
    if result.candidate_id.duplicated().any():
        raise T2FunnelError("a candidate occurs in more than one frozen path file")
    missing = wanted - set(result.candidate_id.astype(str))
    if missing:
        raise T2FunnelError(f"frozen paths missing {len(missing)} candidate IDs")
    return result


def _base_matrix(frame: pd.DataFrame, raw_features: list[str]) -> np.ndarray:
    return np.column_stack((
        frame.loc[:, raw_features].to_numpy(np.float32),
        frame.loc[:, list(T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS)].to_numpy(np.float32),
    ))


def _fit_base(train: pd.DataFrame, test: pd.DataFrame, raw_features: list[str], sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x, z = _base_matrix(train, raw_features), _base_matrix(test, raw_features)
    labels = train.loc[:, ["t2_upper_soft", "t2_lower_soft", "t2_timeout_soft"]].to_numpy(float)
    probabilities = np.column_stack([np.maximum(_huber(x, labels[:, col], z, sample_weight), 0.0) for col in range(3)])
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-8)
    net = train.execution_net_ev_12h.to_numpy(float)
    means = np.asarray([_conditional_mean(labels[:, col], net) for col in range(3)]) * 10_000.0
    return probabilities @ means, probabilities


def _with_labels(base: pd.DataFrame, events: pd.DataFrame, geometry: BarrierGeometry, temperature: float) -> pd.DataFrame:
    merged = base.merge(events, on="candidate_id", how="left", validate="one_to_one")
    if merged.geometry.isna().any():
        raise T2FunnelError("geometry labels do not cover every candidate")
    soft = soft_event_targets(merged, geometry, temperature_atr=temperature)
    merged[["t2_upper_soft", "t2_lower_soft", "t2_timeout_soft"]] = soft
    return merged


def _score_frame(frame: pd.DataFrame, score: np.ndarray, variant: str, phase: str, geometry: str, temperature: float) -> pd.DataFrame:
    result = frame.loc[:, ["candidate_id", "__ts__", "side_name", "__symbol__", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"]].copy()
    result["score_bps"] = score
    result["variant"] = variant
    result["phase"] = phase
    result["geometry"] = geometry
    result["temperature_atr"] = temperature
    return result


def _phase_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    result = top_book_metrics(scored, score_column="score_bps")
    for column in ("variant", "phase", "geometry", "temperature_atr"):
        result[column] = scored[column].iloc[0]
    return result


def _pick(metrics: pd.DataFrame, keys: list[str]) -> pd.Series:
    top10 = metrics.loc[metrics.top_fraction.eq(0.10)].copy()
    return top10.sort_values(["gross_bps_per_trade", "net_bps_per_trade", *keys], ascending=[False, False, *([True] * len(keys))], kind="mergesort").iloc[0]


def _meta_matrix(frame: pd.DataFrame, meta_raw: list[str]) -> np.ndarray:
    return np.column_stack((
        frame.loc[:, meta_raw].to_numpy(np.float32),
        frame.loc[:, list(T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS)].to_numpy(np.float32),
        frame.loc[:, ["base_expected_net_bps", "base_p_upper", "base_p_lower", "base_p_timeout", "base_probability_width"]].to_numpy(np.float32),
    ))


def _residual_fit(train: pd.DataFrame, test: pd.DataFrame, meta_raw: list[str]) -> np.ndarray:
    target = train.execution_net_ev_12h.to_numpy(float) * 10_000.0 - train.base_expected_net_bps.to_numpy(float)
    return _huber(_meta_matrix(train, meta_raw), target, _meta_matrix(test, meta_raw))


def _bootstrap_delta(base: pd.DataFrame, stacked: pd.DataFrame, *, draws: int = 1000) -> dict[str, float]:
    """Paired UTC-day bootstrap after each arm's independent global top-10."""
    def selected(frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort")
        return ordered.head(int(np.ceil(len(ordered) * 0.10))).assign(day=lambda x: pd.to_datetime(x["__ts__"], utc=True).dt.date)
    a, b = selected(base), selected(stacked)
    days = sorted(set(a.day) | set(b.day))
    a_day = a.groupby("day").execution_gross_ev_12h.mean().reindex(days, fill_value=0.0).to_numpy() * 10_000.0
    b_day = b.groupby("day").execution_gross_ev_12h.mean().reindex(days, fill_value=0.0).to_numpy() * 10_000.0
    rng = np.random.default_rng(20260801)
    samples = np.asarray([(b_day[rng.integers(0, len(days), len(days))] - a_day[rng.integers(0, len(days), len(days))]).mean() for _ in range(draws)])
    # The rows above must share draw indices to be paired.  Recompute once
    # correctly rather than rely on independent daily resamples.
    samples = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, len(days), len(days))
        samples[i] = float((b_day[idx] - a_day[idx]).mean())
    return {"delta_gross_bps_daily_mean": float((b_day - a_day).mean()), "ci90_low": float(np.quantile(samples, .05)), "ci90_high": float(np.quantile(samples, .95)), "p_delta_gt_zero": float((samples > 0.0).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-column", default="oof_fold")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output}")
    payload = json.loads(args.features_json.read_text())
    raw_features = list(validate_feature_columns(payload["raw_feature_columns"]))
    required = {"candidate_id", "__ts__", "__decision_ts__", "__label_available_at__", "__symbol__", "side_name", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", args.fold_column}
    ledger = pd.read_parquet(args.ledger, columns=sorted(required | set(raw_features)))
    ledger = _add_causal_context(ledger)
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="raise")
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True, errors="raise")
    ledger["__label_available_at__"] = pd.to_datetime(ledger["__label_available_at__"], utc=True, errors="raise")
    if ledger.candidate_id.duplicated().any():
        raise T2FunnelError("candidate IDs must be unique")
    if not np.allclose(ledger.execution_gross_ev_12h - ledger.execution_cost_return, ledger.execution_net_ev_12h, atol=1e-10, rtol=0.0):
        raise T2FunnelError("frozen H12 net/cost accounting is inconsistent")
    if not ledger["__decision_ts__"].eq(ledger["__ts__"] + pd.Timedelta(hours=1)).all():
        raise T2FunnelError("execution must occur one completed bar after the feature cutoff")
    if not ledger["__label_available_at__"].eq(ledger["__decision_ts__"] + pd.Timedelta(hours=12)).all():
        raise T2FunnelError("H12 labels must be unavailable until the complete execution horizon")
    paths = _read_paths(set(ledger.candidate_id.astype(str)), list(args.paths))
    check = ledger[["candidate_id", "side_name"]].merge(paths[["candidate_id", "side_name"]], on="candidate_id", validate="one_to_one", suffixes=("_ledger", "_path"))
    if not check.side_name_ledger.astype(str).str.lower().eq(check.side_name_path.astype(str).str.lower()).all():
        raise T2FunnelError("path side differs from frozen candidate side")
    stage = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        events_by_geometry = materialize_geometry_events_bulk(paths, GEOMETRIES)
        for geometry in GEOMETRIES:
            events = events_by_geometry[geometry.name]
            events.to_parquet(stage / f"geometry_events_{geometry.name}.parquet", index=False, compression="zstd")
        base_train = ledger.loc[ledger[args.fold_column].eq("base_train")].copy()
        development = ledger.loc[ledger[args.fold_column].eq("meta_train")].copy()
        final_oos = ledger.loc[ledger[args.fold_column].eq("meta_oos")].copy()
        if min(map(len, (base_train, development, final_oos))) == 0:
            raise T2FunnelError("requires base_train, meta_train, and meta_oos frozen folds")
        all_metrics: list[pd.DataFrame] = []
        # 1) Geometry selection: final OOS is untouched.
        for geometry in GEOMETRIES:
            labelled_train = _with_labels(base_train, events_by_geometry[geometry.name], geometry, 0.25)
            labelled_dev = _with_labels(development, events_by_geometry[geometry.name], geometry, 0.25)
            score, probs = _fit_base(_resolved_before(labelled_train, labelled_dev), labelled_dev, raw_features)
            scored = _score_frame(labelled_dev, score, "base_only", "geometry_development", geometry.name, 0.25)
            all_metrics.append(_phase_metrics(scored))
        geometry_metrics = pd.concat(all_metrics, ignore_index=True)
        selected_geometry = str(_pick(geometry_metrics, ["geometry"])["geometry"])
        geometry = next(g for g in GEOMETRIES if g.name == selected_geometry)
        # 2) Softness selection on the already-selected geometry, also pre-OOS.
        softness_scores: dict[float, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
        soft_metrics: list[pd.DataFrame] = []
        for temperature in TEMPERATURES:
            labelled_train = _with_labels(base_train, events_by_geometry[geometry.name], geometry, temperature)
            labelled_dev = _with_labels(development, events_by_geometry[geometry.name], geometry, temperature)
            score, probs = _fit_base(_resolved_before(labelled_train, labelled_dev), labelled_dev, raw_features)
            softness_scores[temperature] = (labelled_dev, score, probs)
            scored = _score_frame(labelled_dev, score, "base_only", "softness_development", geometry.name, temperature)
            soft_metrics.append(_phase_metrics(scored))
        softness_metrics = pd.concat(soft_metrics, ignore_index=True)
        selected_temperature = float(_pick(softness_metrics, ["temperature_atr"])["temperature_atr"])
        # Freeze selected base, generate stopped-gradient base outputs for all
        # later rows once.  Base fit ends before every meta row.
        labelled_train = _with_labels(base_train, events_by_geometry[geometry.name], geometry, selected_temperature)
        labelled_dev = _with_labels(development, events_by_geometry[geometry.name], geometry, selected_temperature)
        labelled_oos = _with_labels(final_oos, events_by_geometry[geometry.name], geometry, selected_temperature)
        dev_score, dev_probs = _fit_base(_resolved_before(labelled_train, labelled_dev), labelled_dev, raw_features)
        oos_score, oos_probs = _fit_base(_resolved_before(labelled_train, labelled_oos), labelled_oos, raw_features)
        for frame, score, prob in ((labelled_dev, dev_score, dev_probs), (labelled_oos, oos_score, oos_probs)):
            frame["base_expected_net_bps"] = score
            frame[["base_p_upper", "base_p_lower", "base_p_timeout"]] = prob
            frame["base_probability_width"] = prob.max(axis=1) - prob.min(axis=1)
        meta_raw = [name for name in META_CANDIDATES if name in raw_features]
        if len(meta_raw) < 20:
            raise T2FunnelError("frozen causal matrix lacks enough predeclared meta features")
        # 3) Short residual-window selection.  July is a pre-final-OOS holdout;
        # train only on the immediately preceding N resolved calendar days.
        dev_cutoff = labelled_dev["__ts__"].max().normalize() - pd.Timedelta(days=30)
        residual_train_pool = labelled_dev.loc[labelled_dev["__ts__"].lt(dev_cutoff)]
        residual_eval = labelled_dev.loc[labelled_dev["__ts__"].ge(dev_cutoff)]
        residual_metrics: list[pd.DataFrame] = []
        for days in RESIDUAL_DAYS:
            start = dev_cutoff - pd.Timedelta(days=days)
            train = residual_train_pool.loc[residual_train_pool["__ts__"].ge(start)]
            correction = _residual_fit(_resolved_before(train, residual_eval), residual_eval, meta_raw)
            base_book = _score_frame(residual_eval, residual_eval.base_expected_net_bps.to_numpy(), "base_only", "residual_development", geometry.name, selected_temperature)
            stack_book = _score_frame(residual_eval, residual_eval.base_expected_net_bps.to_numpy() + correction, "base_plus_meta", "residual_development", geometry.name, selected_temperature)
            for book in (base_book, stack_book):
                metric = _phase_metrics(book)
                metric["residual_train_days"] = days
                residual_metrics.append(metric)
        residual_metrics_frame = pd.concat(residual_metrics, ignore_index=True)
        selected_days = int(_pick(residual_metrics_frame.loc[residual_metrics_frame.variant.eq("base_plus_meta")], ["residual_train_days"])["residual_train_days"])
        # 4) Final OOS is opened once, using last N resolved development days.
        oos_start = labelled_oos["__ts__"].min()
        train_start = oos_start - pd.Timedelta(days=selected_days)
        final_meta_train = labelled_dev.loc[labelled_dev["__ts__"].ge(train_start) & labelled_dev["__ts__"].lt(oos_start)]
        final_correction = _residual_fit(_resolved_before(final_meta_train, labelled_oos), labelled_oos, meta_raw)
        base_final = _score_frame(labelled_oos, labelled_oos.base_expected_net_bps.to_numpy(), "base_only", "final_oos", geometry.name, selected_temperature)
        stack_final = _score_frame(labelled_oos, labelled_oos.base_expected_net_bps.to_numpy() + final_correction, "base_plus_meta", "final_oos", geometry.name, selected_temperature)
        final_predictions = pd.concat((base_final, stack_final), ignore_index=True)
        final_results = pd.concat((_phase_metrics(base_final), _phase_metrics(stack_final)), ignore_index=True)
        attribution = []
        for variant, book in final_predictions.groupby("variant", observed=True):
            for group_name, column in (("side", "side_name"), ("month", "__ts__")):
                work = book.copy()
                work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.to_period("M").astype(str)
                subset = top_book_metrics(work, score_column="score_bps", group_columns=[column if group_name == "side" else "month"])
                subset["variant"] = variant
                subset["attribution"] = group_name
                attribution.append(subset)
        bootstrap = _bootstrap_delta(base_final, stack_final)
        geometry_metrics.to_parquet(stage / "geometry_screen.parquet", index=False, compression="zstd")
        softness_metrics.to_parquet(stage / "softness_ablation.parquet", index=False, compression="zstd")
        residual_metrics_frame.to_parquet(stage / "residual_window_ablation.parquet", index=False, compression="zstd")
        final_predictions.to_parquet(stage / "base_meta_stack_predictions.parquet", index=False, compression="zstd")
        final_results.to_parquet(stage / "base_meta_stack_results.parquet", index=False, compression="zstd")
        pd.concat(attribution, ignore_index=True).to_parquet(stage / "base_meta_attribution.parquet", index=False, compression="zstd")
        (stage / "base_meta_bootstrap.json").write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n")
        manifest = {
            "schema": SCHEMA,
            "status": "COMPLETED_FINAL_OOS_OPENED_ONCE",
            "target": "T2 soft three-state triple barrier, entry-ATR-normalised, H12",
            "fixed_grid": [{"geometry": g.name, "tp_atr": g.tp_atr, "sl_atr": g.sl_atr} for g in GEOMETRIES],
            "selected_geometry_pre_final_oos": selected_geometry,
            "softness_grid_atr": list(TEMPERATURES),
            "selected_temperature_pre_final_oos": selected_temperature,
            "residual_days_grid": list(RESIDUAL_DAYS),
            "selected_residual_days_pre_final_oos": selected_days,
            "base_features": {"raw_admitted_causal_count": len(raw_features), "derived_causal_context": list(T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS)},
            "meta_features": {"raw_causal_count": len(meta_raw), "raw_causal": meta_raw, "stopped_gradient_context": list(T2_FUNNEL_META_CONTEXT_FEATURE_KEYS)},
            "lineage": {"base_train": "base_train only", "geometry_and_softness_selection": "meta_train only", "residual_window_selection": "pre-final-OOS final 30 calendar days of meta_train", "final_meta_train": f"last {selected_days} calendar days of meta_train", "final_evaluation": "meta_oos opened once"},
            "path_contract": "720 exact one-minute bars; side-normalised; same-minute upper/lower conflict resolves lower/adverse first",
            "selection": "independent pooled-global common-bps ranking; no side/timestamp/asset quotas or portfolio constraints",
            "inputs": {str(path): _sha(path) for path in [args.ledger, args.features_json, *args.paths]},
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output)
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
