#!/usr/bin/env python3
"""Strict weekly-OOF raw-market-state transition/persistence head ablation.

The raw state at ``t+h`` is a *label* only.  Each week and side fits its
imputer, scaler, KMeans state definition, and transition classifier on rows
whose label is resolved before that week's first decision.  Evaluation rows
are never used in any fitted object.  The returned persistence probability is
exactly ``1 - P(state changes)``; it is not an independently trained duplicate
head.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_market_state import (  # noqa: E402
    MARKET_STATE_COLUMNS,
    MARKET_STATE_FAMILIES,
    MARKET_STATE_SCHEMA_VERSION,
    UNAVAILABLE_HISTORICAL_FAMILIES,
    attach_decision_time_market_state,
)


HORIZONS = (1, 3, 6, 12)
GEOMETRY_COLUMNS = (
    "existing_alpha_ev",
    "alpha_prediction_uncertainty",
    "alpha_leaf_support",
    "pred_peak_MFE_12h_ATR",
    "catboost_entropy",
    "base_oof_score",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "oof_clean_favorable_probability",
    "catboost_p_0",
    "catboost_p_1",
    "catboost_p_2",
    "catboost_p_3",
    "catboost_p_4",
    "catboost_p_5",
    "catboost_p_6",
)


@dataclass
class StateBundle:
    imputer: SimpleImputer
    scaler: RobustScaler
    cluster: MiniBatchKMeans


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _finite_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def _fit_state_bundle(x: pd.DataFrame, *, random_state: int, clusters: int) -> StateBundle:
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    raw = imputer.fit_transform(x)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaled = scaler.fit_transform(raw)
    k = max(2, min(int(clusters), len(scaled) // 250, len(scaled)))
    if k < 2:
        raise ValueError("insufficient resolved rows for a raw-state definition")
    cluster = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(random_state),
        batch_size=min(4096, len(scaled)),
        n_init=5,
        max_iter=200,
    )
    cluster.fit(scaled)
    return StateBundle(imputer=imputer, scaler=scaler, cluster=cluster)


def _state_geometry(bundle: StateBundle, x: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    scaled = bundle.scaler.transform(bundle.imputer.transform(x))
    distances = np.asarray(bundle.cluster.transform(scaled), dtype=float)
    ordered = np.partition(distances, kth=1, axis=1)[:, :2]
    nearest = ordered[:, 0]
    margin = ordered[:, 1] - ordered[:, 0]
    # A scale-free uncertainty coordinate: soft assignments based on each row's
    # nearest-centroid distance, not a persisted cluster ID.
    logits = -(distances - nearest[:, None])
    exp_logits = np.exp(np.clip(logits, -50.0, 0.0))
    probs = exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    state = bundle.cluster.predict(scaled).astype(np.int16)
    geometry = pd.DataFrame(
        {
            "state_nearest_distance": nearest.astype("float32"),
            "state_top2_margin": margin.astype("float32"),
            "state_assignment_entropy": entropy.astype("float32"),
        },
        index=x.index,
    )
    return state, geometry


def _prepare_lookup_states(
    base: pd.DataFrame,
    *,
    feature_store_root: Path,
    horizons: tuple[int, ...],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach only causal current/future raw rows; future rows become labels."""

    current = base.copy()
    current["raw_state_lookup_row_id"] = np.arange(len(current), dtype=np.int64)
    attached_frames: dict[int, pd.DataFrame] = {}
    coverages: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for horizon in (0, *horizons):
        lookup = current[["raw_state_lookup_row_id", "__symbol__", "execution_decision_utc"]].copy()
        lookup["execution_decision_utc"] = (
            _utc(lookup["execution_decision_utc"]) + pd.Timedelta(hours=int(horizon))
        )
        result = attach_decision_time_market_state(
            lookup,
            feature_store_root=feature_store_root,
            decision_time_col="execution_decision_utc",
            symbol_col="__symbol__",
            # Completed-bar source at s is observable at s+1h.
            completed_bar_delay=pd.Timedelta("1h"),
            max_staleness=pd.Timedelta("90min"),
        )
        selected = result.frame[["raw_state_lookup_row_id", "mkt_state_source_utc", *MARKET_STATE_COLUMNS]].copy()
        rename = {
            "mkt_state_source_utc": f"raw_state_source_utc_h{horizon}",
            **{column: f"{column}__h{horizon}" for column in MARKET_STATE_COLUMNS},
        }
        attached_frames[horizon] = selected.rename(columns=rename)
        coverage = result.coverage.copy()
        coverage["horizon_hours"] = horizon
        coverages.append(coverage)
        audit = result.source_audit.copy()
        audit["horizon_hours"] = horizon
        audits.append(audit)
    enriched = current
    for horizon in (0, *horizons):
        enriched = enriched.merge(
            attached_frames[horizon], on="raw_state_lookup_row_id", how="left", validate="one_to_one"
        )
    enriched = enriched.drop(columns="raw_state_lookup_row_id")
    return enriched, pd.concat(coverages, ignore_index=True), pd.concat(audits, ignore_index=True)


def _metric_row(
    rows: pd.DataFrame,
    *,
    prediction_column: str,
    target_column: str,
    label: str,
    side: str,
    horizon: int,
    week_start: pd.Timestamp,
) -> dict[str, Any]:
    target = pd.to_numeric(rows[target_column], errors="coerce")
    pred = pd.to_numeric(rows[prediction_column], errors="coerce").clip(1e-6, 1 - 1e-6)
    valid = target.notna() & pred.notna()
    target = target.loc[valid].astype(int)
    pred = pred.loc[valid]
    out: dict[str, Any] = {
        "feature_set": label,
        "side_name": side,
        "horizon_hours": horizon,
        "week_start": week_start,
        "rows": int(len(target)),
        "transition_rate": float(target.mean()) if len(target) else np.nan,
    }
    if len(target) < 2 or target.nunique() < 2:
        return out
    out.update(
        {
            "roc_auc": float(roc_auc_score(target, pred)),
            "average_precision": float(average_precision_score(target, pred)),
            "brier": float(brier_score_loss(target, pred)),
            "log_loss": float(log_loss(target, pred, labels=[0, 1])),
            "prediction_mean": float(pred.mean()),
        }
    )
    top_n = max(1, int(np.ceil(0.10 * len(pred))))
    ordered = rows.loc[valid].assign(__p__=pred).sort_values("__p__", ascending=False)
    top = ordered.head(top_n)
    bottom = ordered.tail(top_n)
    out["top10_transition_rate"] = float(pd.to_numeric(top[target_column], errors="coerce").mean())
    out["bottom10_transition_rate"] = float(pd.to_numeric(bottom[target_column], errors="coerce").mean())
    if "execution_net_ev_12h" in ordered:
        out["top10_net_ev"] = float(pd.to_numeric(top["execution_net_ev_12h"], errors="coerce").mean())
        out["bottom10_net_ev"] = float(pd.to_numeric(bottom["execution_net_ev_12h"], errors="coerce").mean())
        out["prediction_net_ev_spearman"] = float(
            ordered["__p__"].corr(
                pd.to_numeric(ordered["execution_net_ev_12h"], errors="coerce"), method="spearman"
            )
        )
    return out


def _fit_predict_week(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    raw_columns: list[str],
    feature_set: str,
    random_state: int,
    state_bundle: StateBundle,
    transition_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Fit one side/horizon model; state transform is fitted on resolved train only."""

    raw_train = _finite_frame(train, raw_columns)
    raw_eval = _finite_frame(evaluation, raw_columns)
    bundle = state_bundle
    _, train_geometry = _state_geometry(bundle, raw_train)
    _, eval_geometry = _state_geometry(bundle, raw_eval)
    target = np.asarray(transition_target, dtype=np.int8)
    if target.min() == target.max():
        probability = np.full(len(evaluation), float(target.mean()), dtype=np.float32)
        return probability, 1.0 - probability, int(bundle.cluster.n_clusters), []

    if feature_set == "market_state_only":
        x_train = pd.concat([raw_train, train_geometry], axis=1)
        x_eval = pd.concat([raw_eval, eval_geometry], axis=1)
    elif feature_set == "existing_geometry_only":
        columns = [column for column in GEOMETRY_COLUMNS if column in train]
        x_train = _finite_frame(train, columns)
        x_eval = _finite_frame(evaluation, columns)
    elif feature_set == "combined":
        columns = [column for column in GEOMETRY_COLUMNS if column in train]
        x_train = pd.concat([raw_train, train_geometry, _finite_frame(train, columns)], axis=1)
        x_eval = pd.concat([raw_eval, eval_geometry, _finite_frame(evaluation, columns)], axis=1)
    else:
        raise ValueError(f"unknown feature set {feature_set!r}")
    # HGB handles NaN, but explicit train medians make the input contract
    # reproducible and prevent an evaluation-only missingness distribution from
    # affecting fit behavior.
    medians = x_train.median(axis=0, skipna=True).fillna(0.0)
    x_train = x_train.fillna(medians).replace([np.inf, -np.inf], 0.0)
    x_eval = x_eval.fillna(medians).replace([np.inf, -np.inf], 0.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        # Fixed small head: this is a supporting-label ablation, not a hidden
        # HPO search.  The cap also keeps every weekly side/horizon fit cheap.
        max_iter=24,
        max_leaf_nodes=8,
        max_bins=64,
        min_samples_leaf=80,
        l2_regularization=2.0,
        random_state=random_state,
    )
    model.fit(x_train, target)
    probability = model.predict_proba(x_eval)[:, 1].astype(np.float32)
    return probability, 1.0 - probability, int(bundle.cluster.n_clusters), list(x_train.columns)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    requested_base_columns = [
        "candidate_id", "__symbol__", "side_name", "execution_decision_utc", "execution_net_ev_12h", *GEOMETRY_COLUMNS
    ]
    input_schema = set(pq.read_schema(args.input).names)
    # The input has many high-cardinality archetype columns that the transition
    # ablation never consumes.  Do not decompress them into the model process.
    base = pd.read_parquet(args.input, columns=[column for column in requested_base_columns if column in input_schema])
    required = {"__symbol__", "side_name", "execution_decision_utc", "execution_net_ev_12h"}
    missing = sorted(required.difference(base.columns))
    if missing:
        raise ValueError(f"input lacks required columns: {missing}")
    base["execution_decision_utc"] = _utc(base["execution_decision_utc"])
    base = base.loc[base["execution_decision_utc"].notna()].copy()
    base = base.sort_values("execution_decision_utc").reset_index(drop=True)
    horizons = tuple(int(value) for value in args.horizons)
    enriched_path = out / "raw_market_state_transition_rows.parquet"
    coverage_path = out / "raw_market_state_feature_coverage.csv"
    source_audit_path = out / "raw_market_state_source_audit.csv"
    required_enriched_columns = [
        "candidate_id", "__symbol__", "side_name", "execution_decision_utc", "execution_net_ev_12h", *GEOMETRY_COLUMNS,
        *[f"{column}__h{horizon}" for horizon in (0, *horizons) for column in MARKET_STATE_COLUMNS],
    ]
    cache_root = Path(args.market_state_cache_dir) if args.market_state_cache_dir else out
    cache_enriched_path = cache_root / "raw_market_state_transition_rows.parquet"
    cache_coverage_path = cache_root / "raw_market_state_feature_coverage.csv"
    cache_source_audit_path = cache_root / "raw_market_state_source_audit.csv"
    if cache_enriched_path.exists() and not args.refresh_market_state:
        enriched_schema = set(pq.read_schema(cache_enriched_path).names)
        enriched = pd.read_parquet(
            cache_enriched_path,
            columns=[column for column in required_enriched_columns if column in enriched_schema],
        )
        coverage = pd.read_csv(cache_coverage_path)
        source_audit = pd.read_csv(cache_source_audit_path)
    else:
        enriched, coverage, source_audit = _prepare_lookup_states(
            base,
            feature_store_root=Path(args.feature_store_root),
            horizons=horizons,
            output_dir=out,
        )
        enriched.to_parquet(enriched_path, index=False)
        coverage.to_csv(coverage_path, index=False)
        source_audit.to_csv(source_audit_path, index=False)

    raw_now = [f"{column}__h0" for column in MARKET_STATE_COLUMNS]
    weeks = pd.date_range(
        pd.Timestamp(args.first_eval_week, tz="UTC"),
        _utc(enriched["execution_decision_utc"]).max().ceil("D"),
        freq="7D",
    )
    if args.last_eval_week:
        last_week = pd.Timestamp(args.last_eval_week, tz="UTC")
        weeks = weeks[weeks <= last_week]
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    feature_contracts: list[dict[str, Any]] = []
    for week_start in weeks:
        week_end = min(week_start + pd.Timedelta(days=7), _utc(enriched["execution_decision_utc"]).max() + pd.Timedelta("1ns"))
        eval_mask = (
            _utc(enriched["execution_decision_utc"]).ge(week_start)
            & _utc(enriched["execution_decision_utc"]).lt(week_end)
        )
        evaluation_base = enriched.loc[eval_mask]
        if evaluation_base.empty:
            continue
        print(f"transition-head week={week_start.date()} eval_rows={len(evaluation_base)}", flush=True)
        for horizon in horizons:
            future_raw = [f"{column}__h{horizon}" for column in MARKET_STATE_COLUMNS]
            future_map = dict(zip(future_raw, [f"{column}__future" for column in raw_now], strict=True))
            resolution = _utc(enriched["execution_decision_utc"]) + pd.Timedelta(hours=horizon)
            # Only fully resolved raw-state labels are allowed into the train
            # state model and classifier.  This is a real embargo for this
            # auxiliary target even though no EV outcome is used here.
            horizon_columns = [
                "candidate_id", "__symbol__", "side_name", "execution_decision_utc", "execution_net_ev_12h",
                *[column for column in GEOMETRY_COLUMNS if column in enriched], *raw_now, *future_raw,
            ]
            train_base = enriched.loc[resolution.lt(week_start), horizon_columns]
            if len(train_base) < args.min_train_rows:
                continue
            train_base = train_base.rename(columns=future_map)
            evaluation = evaluation_base.loc[:, horizon_columns].copy()
            evaluation["transition_label_resolution_utc"] = (
                _utc(evaluation["execution_decision_utc"]) + pd.Timedelta(hours=horizon)
            )
            for side in ("long", "short"):
                train = train_base.loc[train_base["side_name"].eq(side)].copy()
                evaluation_side = evaluation.loc[evaluation["side_name"].eq(side)].copy()
                if len(train) < args.min_train_rows or evaluation_side.empty:
                    continue
                # A raw input must be materially observed in the *training*
                # slice at both the decision and its future label point.  This
                # prevents an offline-only field (notably historical L2
                # proxies) from being silently median-imputed into the model.
                raw_columns = [
                    current_column
                    for current_column in raw_now
                    if _finite_frame(train, [current_column]).notna().mean().iloc[0] >= float(args.min_train_feature_coverage)
                    and _finite_frame(train, [f"{current_column}__future"]).notna().mean().iloc[0] >= float(args.min_train_feature_coverage)
                ]
                if len(raw_columns) < 2:
                    continue
                future_columns = [f"{column}__future" for column in raw_columns]
                # A label exists only if the selected frozen raw-state input
                # is observable at its resolution time.
                train = train.loc[_finite_frame(train, future_columns).notna().all(axis=1)].copy()
                if len(train) < args.min_train_rows:
                    continue
                print(f"  side={side} h={horizon} resolved_train={len(train)} raw={len(raw_columns)}", flush=True)
                # This frozen bundle is the only state definition used for the
                # week/side/horizon.  It sees only resolved training rows.
                label_bundle = _fit_state_bundle(
                    _finite_frame(train, raw_columns),
                    random_state=int(args.random_state) + int(horizon),
                    clusters=4,
                )
                current_state, _ = _state_geometry(label_bundle, _finite_frame(train, raw_columns))
                future_state, _ = _state_geometry(
                    label_bundle,
                    _finite_frame(train, future_columns).set_axis(raw_columns, axis=1),
                )
                transition_target = (current_state != future_state).astype(np.int8)
                train_rate = float(transition_target.mean())
                eval_current, _ = _state_geometry(label_bundle, _finite_frame(evaluation_side, raw_columns))
                eval_future, _ = _state_geometry(
                    label_bundle,
                    _finite_frame(evaluation_side, [f"{column.replace('__h0', '')}__h{horizon}" for column in raw_columns]).set_axis(raw_columns, axis=1),
                )
                eval_future_columns = [f"{column.replace('__h0', '')}__h{horizon}" for column in raw_columns]
                future_ok = _finite_frame(evaluation_side, eval_future_columns).notna().all(axis=1).to_numpy()
                evaluation_label = np.where(
                    future_ok, (eval_current != eval_future).astype(float), np.nan
                )
                for feature_set in ("existing_geometry_only", "market_state_only", "combined"):
                    print(f"    fit={feature_set}", flush=True)
                    probability, persistence, k, used_columns = _fit_predict_week(
                        train,
                        evaluation_side,
                        raw_columns=raw_columns,
                        feature_set=feature_set,
                        random_state=int(args.random_state) + int(horizon),
                        state_bundle=label_bundle,
                        transition_target=transition_target,
                    )
                    prediction = evaluation_side[[column for column in ("candidate_id", "__symbol__", "side_name", "execution_decision_utc", "execution_net_ev_12h") if column in evaluation_side]].copy()
                    prediction["feature_set"] = feature_set
                    prediction["horizon_hours"] = horizon
                    prediction["week_start"] = week_start
                    prediction["transition_label_resolution_utc"] = evaluation_side["transition_label_resolution_utc"].to_numpy()
                    prediction["raw_state_transition_label"] = evaluation_label
                    prediction["raw_state_persistence_label"] = 1.0 - evaluation_label
                    prediction["oof_transition_probability"] = probability
                    prediction["oof_persistence_probability"] = persistence
                    prediction["state_cluster_count"] = k
                    prediction["training_rows_resolved"] = len(train)
                    prediction["training_transition_rate"] = train_rate
                    predictions.append(prediction)
                    metrics.append(_metric_row(prediction, prediction_column="oof_transition_probability", target_column="raw_state_transition_label", label=feature_set, side=side, horizon=horizon, week_start=week_start))
                    feature_contracts.append({
                        "week_start": week_start,
                        "side_name": side,
                        "horizon_hours": horizon,
                        "feature_set": feature_set,
                        "state_cluster_count": k,
                        "training_rows_resolved": len(train),
                        "input_columns": used_columns,
                        "selected_raw_market_state_columns": raw_columns,
                    })
    if not predictions:
        raise RuntimeError("no weekly OOF transition-head predictions were generated")
    pred_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics)
    aggregate_rows = []
    for (feature_set, side, horizon), group in pred_frame.groupby(["feature_set", "side_name", "horizon_hours"], sort=True):
        aggregate_rows.append(_metric_row(group, prediction_column="oof_transition_probability", target_column="raw_state_transition_label", label=str(feature_set), side=str(side), horizon=int(horizon), week_start=pd.NaT))
    aggregate = pd.DataFrame(aggregate_rows)
    pred_frame.to_parquet(out / "strict_weekly_oof_transition_predictions.parquet", index=False)
    metrics_frame.to_csv(out / "strict_weekly_oof_transition_metrics.csv", index=False)
    aggregate.to_csv(out / "strict_weekly_oof_transition_aggregate_metrics.csv", index=False)
    _write_json(out / "transition_head_feature_contracts.json", {"rows": feature_contracts})
    manifest = {
        "schema": "execution_ev_raw_market_state_transition_head_v1",
        "market_state_schema": MARKET_STATE_SCHEMA_VERSION,
        "input": str(args.input),
        "feature_store_root": str(args.feature_store_root),
        "horizons_hours": list(horizons),
        "first_eval_week": str(args.first_eval_week),
        "raw_state_families": {key: list(value) for key, value in MARKET_STATE_FAMILIES.items()},
        "unavailable_historical_families": UNAVAILABLE_HISTORICAL_FAMILIES,
        "source_timing": "source ts is hourly bar open; adapter joins source_ts <= execution_decision_utc - 1h, then enforces <=90min source staleness",
        "label_rule": "transition at h is frozen weekly raw-state KMeans(current) != KMeans(raw state observable at decision+h); persistence=1-transition; label resolves at decision+h",
        "oof_rule": "per side/week/horizon: all fitted transforms and classifiers use only rows whose transition label resolved before week start; evaluations are later week rows",
        "feature_sets": ["existing_geometry_only", "market_state_only", "combined"],
        "min_train_feature_coverage": float(args.min_train_feature_coverage),
        "no_calendar_regime_weights": True,
        "outputs": {
            "enriched_rows": str(cache_enriched_path),
            "coverage": str(cache_coverage_path),
            "source_audit": str(cache_source_audit_path),
            "predictions": str(out / "strict_weekly_oof_transition_predictions.parquet"),
            "weekly_metrics": str(out / "strict_weekly_oof_transition_metrics.csv"),
            "aggregate_metrics": str(out / "strict_weekly_oof_transition_aggregate_metrics.csv"),
        },
    }
    _write_json(out / "summary.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature-store-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-eval-week", default="2026-06-01")
    parser.add_argument("--last-eval-week", default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--min-train-rows", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=20260726)
    parser.add_argument("--min-train-feature-coverage", type=float, default=0.95)
    parser.add_argument("--refresh-market-state", action="store_true")
    parser.add_argument(
        "--market-state-cache-dir", type=Path, default=None,
        help="Read an immutable raw-state attachment from this prior artifact.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
