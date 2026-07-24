#!/usr/bin/env python3
"""Discover local residual archetypes and broad-market states chronologically.

This runner does not tune or apply a score nudge. It measures whether frozen
pre-entry features can recognize large signed residual states OOS. The optional
8-day reachable-EV policy is an assessment overlay only and is never included
in recognizer inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.market_residual_archetypes import (  # noqa: E402
    PerArchetypeMarketAdverseConfig,
    PerArchetypeMarketAdverseRecognizer,
    adverse_episode_ranking_metrics,
)
from extreme_price_movements.data_store import read_symbol_features  # noqa: E402
from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
    inference_feature_columns,
    strip_outcomes_for_oos,
)

DEFAULT_DATA = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710/candidate_shards"
)
DEFAULT_LEDGER = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710/"
    "frozen_champion_single_source_ledger.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/reports/meta_residual_archetype_discovery_early2025_july_20260712_v1"
)
DEFAULT_THRESHOLD_POLICY = Path(
    "data_perp/artifacts/s59_s52_frozen_native_shadow_20260709/"
    "policy_params/threshold_basis_policy.json"
)
DEFAULT_FEATURE_ROOT = Path("data_perp/features/20260711_070000")
DEFAULT_FOLDS = (
    ("2025_h2_rare_states", "2025-07-01", "2025-10-01"),
    ("2025_q4_year_end", "2025-10-01", "2026-01-01"),
    ("2026_q1_transition", "2026-01-01", "2026-04-01"),
    ("2026_apr_may", "2026-04-01", "2026-06-01"),
    ("2026_june", "2026-06-01", "2026-07-01"),
    ("2026_july_available", "2026-07-01", "2026-08-01"),
)
POST_META_OUTPUT_COLUMNS = {
    "hit_probability",
    "score_regime_calibrated",
    "score_regime_current",
    "score_regime_alternative",
    "score_current_reference",
    "score_meta_uncalibrated",
    "score_residual_overlay",
    "score_shock_adjusted",
    "production_adjusted_rank",
    "production_policy_rank",
    "production_threshold_rank",
}


def _residual_candidate_features(frame: pd.DataFrame, score_col: str) -> list[str]:
    """Return features observable before the alternative meta prediction."""

    excluded = POST_META_OUTPUT_COLUMNS | {str(score_col)}
    return [
        name
        for name in inference_feature_columns(frame, frame.columns)
        if name not in excluded
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _resolve_score(frame: pd.DataFrame, requested: str | None) -> str:
    candidates = [requested] if requested else []
    candidates += ["score_regime_calibrated", "score_meta_base_soft_label", "score"]
    for name in candidates:
        if (
            name
            and name in frame.columns
            and pd.to_numeric(frame[name], errors="coerce").notna().any()
        ):
            return str(name)
    raise ValueError("No usable frozen meta score was found")


def _symbol_feature_path(root: Path, symbol: str) -> Path:
    safe = str(symbol).replace("/", "_")
    direct = root / f"symbol={safe}.parquet"
    if direct.exists():
        return direct
    matches = list(root.glob(f"symbol={safe}*.parquet"))
    return matches[0] if matches else direct


def _append_feature_hydrated_ledger_rows(
    data: pd.DataFrame,
    ledger_path: Path,
    feature_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append ledger rows absent from shards and hydrate observable features."""

    if "row_id" not in data.columns or not ledger_path.exists():
        return data, {"status": "unavailable", "appended_rows": 0}
    ledger_available = set(pq.ParquetFile(ledger_path).schema_arrow.names)
    identity_and_outcomes = {
        "row_id",
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "source_tag",
        "score",
        "score_regime_calibrated",
        "hit_probability",
        "ev_after_1pct",
        "exec_margin",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "outcomes_available",
    }
    columns = sorted(identity_and_outcomes.intersection(ledger_available))
    ledger = pd.read_parquet(ledger_path, columns=columns)
    ledger = ledger.loc[~ledger["row_id"].isin(data["row_id"])].copy()
    if ledger.empty:
        return data, {"status": "no_missing_rows", "appended_rows": 0}
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="coerce")
    if "outcomes_available" in ledger.columns:
        explicit = ledger["outcomes_available"].eq(True)  # noqa: E712
        materialized = pd.to_numeric(
            ledger.get("ev_after_1pct"), errors="coerce"
        ).notna() & pd.to_numeric(ledger.get("clean_exec"), errors="coerce").notna()
        ledger = ledger.loc[explicit | materialized]
    ledger = ledger.dropna(subset=["__ts__", "__symbol__"])

    observable = [
        name
        for name in inference_feature_columns(data, data.columns)
        if name not in POST_META_OUTPUT_COLUMNS
    ]
    values = np.full((len(ledger), len(observable)), np.nan, dtype=np.float32)
    matched = np.zeros(len(ledger), dtype=bool)
    missing_symbols: list[str] = []
    for symbol, idx in ledger.groupby("__symbol__", observed=True, sort=False).groups.items():
        positions = ledger.index.get_indexer(idx)
        path = _symbol_feature_path(feature_root, str(symbol))
        if not path.exists():
            missing_symbols.append(str(symbol))
            continue
        timestamps = ledger.loc[idx, "__ts__"]
        features = read_symbol_features(
            str(path),
            columns=observable,
            start_ts=timestamps.min(),
            end_ts=timestamps.max(),
        )
        if features.empty:
            continue
        features = features.copy(deep=False)
        features.index = pd.to_datetime(features.index, utc=True, errors="coerce")
        features = features.loc[~features.index.duplicated(keep="last")]
        aligned = features.reindex(timestamps.to_numpy())
        available = [name for name in observable if name in aligned.columns]
        if not available:
            continue
        target_columns = np.asarray([observable.index(name) for name in available])
        hydrated = aligned[available].to_numpy(dtype=np.float32, copy=False)
        values[np.ix_(positions, target_columns)] = hydrated
        matched[positions] = np.isfinite(hydrated).any(axis=1)
    missing_observable = [name for name in observable if name not in ledger.columns]
    if missing_observable:
        positions = [observable.index(name) for name in missing_observable]
        hydrated_frame = pd.DataFrame(
            values[:, positions], columns=missing_observable, index=ledger.index
        )
        ledger = pd.concat([ledger, hydrated_frame], axis=1, copy=False)
    ledger = ledger.reindex(columns=data.columns)
    output = pd.concat([data, ledger], ignore_index=True, sort=False, copy=False)
    return output, {
        "status": "appended",
        "appended_rows": int(len(ledger)),
        "rows_with_observable_features": int(matched.sum()),
        "observable_feature_count": int(len(observable)),
        "missing_symbol_count": int(len(missing_symbols)),
        "missing_symbols": missing_symbols,
        "feature_root": str(feature_root),
    }


def _refresh_ledger_scores(data: pd.DataFrame, ledger_path: Path) -> pd.DataFrame:
    """Refresh frozen scores by indexed assignment without a full-frame merge."""

    if "row_id" not in data.columns or not ledger_path.exists():
        return data
    available = set(pq.ParquetFile(ledger_path).schema_arrow.names)
    sources = [
        name
        for name in ("score", "score_regime_calibrated", "hit_probability")
        if name in available
    ]
    if not sources:
        return data
    ledger = pd.read_parquet(ledger_path, columns=["row_id", *sources])
    ledger = ledger.drop_duplicates("row_id", keep="last").set_index("row_id")
    row_ids = data["row_id"]
    for source in sources:
        target = "score_meta_uncalibrated" if source == "score" else source
        mapped = row_ids.map(ledger[source])
        if target in data.columns:
            existing = data[target]
            data[target] = mapped.where(mapped.notna(), existing)
        else:
            data[target] = mapped
    return data


def _lag1(values: pd.Series) -> float:
    data = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(data) < 4 or np.std(data[:-1]) <= 1e-12 or np.std(data[1:]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(data[:-1], data[1:])[0, 1])


def _safe_ap(target: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(target, errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    p = pd.to_numeric(score, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    return float(average_precision_score(y, p)) if np.unique(y).size > 1 else np.nan


def _metrics(predictions: pd.DataFrame, fold: str) -> pd.DataFrame:
    adverse_columns = [
        "meta_resid_arch_prob__base_dirty_high_confidence",
        "meta_resid_arch_prob__base_slow_timeout_positive",
        "meta_resid_arch_prob__base_bad_mae_false_positive",
        "meta_resid_arch_prob__base_high_variance_uncertain",
    ]
    favorable_columns = [
        "meta_resid_arch_prob__base_clean_high_confidence",
        "meta_resid_arch_prob__base_missed_clean_opportunity",
    ]
    work = predictions.copy(deep=False)
    work["predicted_adverse"] = (
        work.reindex(columns=adverse_columns).fillna(0.0).sum(axis=1)
    )
    work["predicted_favorable"] = (
        work.reindex(columns=favorable_columns).fillna(0.0).sum(axis=1)
    )
    work["day"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("D")
    rows: list[dict[str, Any]] = []
    for (side, arch), part in work.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        daily = part.groupby("day", observed=True)[
            "market_adjusted_hit_surprise"
        ].mean()
        top_adverse = part["predicted_adverse"].ge(
            part["predicted_adverse"].quantile(0.90)
        )
        top_favorable = part["predicted_favorable"].ge(
            part["predicted_favorable"].quantile(0.90)
        )
        rows.append(
            {
                "fold": fold,
                "side_name": str(side),
                "archetype_policy_key": str(arch),
                "rows": int(len(part)),
                "days": int(part["day"].nunique()),
                "mean_ev": float(
                    pd.to_numeric(part["ev_after_1pct"], errors="coerce").mean()
                ),
                "mean_market_adjusted_surprise": float(
                    part["market_adjusted_hit_surprise"].mean()
                ),
                "daily_surprise_lag1": _lag1(daily),
                "negative_event_rate": float(part["negative_autocorr_label"].mean()),
                "positive_event_rate": float(part["positive_autocorr_label"].mean()),
                "negative_event_ap": _safe_ap(
                    part["negative_autocorr_label"], part["predicted_adverse"]
                ),
                "positive_event_ap": _safe_ap(
                    part["positive_autocorr_label"], part["predicted_favorable"]
                ),
                "adverse_top10_event_precision": float(
                    part.loc[top_adverse, "negative_autocorr_label"].mean()
                ),
                "adverse_top10_mean_surprise": float(
                    part.loc[top_adverse, "market_adjusted_hit_surprise"].mean()
                ),
                "favorable_top10_event_precision": float(
                    part.loc[top_favorable, "positive_autocorr_label"].mean()
                ),
                "favorable_top10_mean_surprise": float(
                    part.loc[top_favorable, "market_adjusted_hit_surprise"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _folds_for_data(
    frame: pd.DataFrame,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    minimum = pd.to_datetime(frame["__ts__"], utc=True).min()
    maximum = pd.to_datetime(frame["__ts__"], utc=True).max() + pd.Timedelta(
        nanoseconds=1
    )
    folds: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for name, start, end in DEFAULT_FOLDS:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = min(pd.Timestamp(end, tz="UTC"), maximum)
        if start_ts > minimum and end_ts > start_ts:
            folds.append((name, start_ts, end_ts))
    return folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score-column", default=None)
    parser.add_argument("--assessment-score-column", default="score_regime_calibrated")
    parser.add_argument("--threshold-policy", type=Path, default=DEFAULT_THRESHOLD_POLICY)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument(
        "--append-ledger-only-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append rows present only in the frozen ledger and hydrate their "
            "observable inputs from the point-in-time feature store."
        ),
    )
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument(
        "--end", default=None, help="Exclusive; defaults to all available rows"
    )
    parser.add_argument("--min-local-rows", type=int, default=1_200)
    parser.add_argument("--min-valid-days", type=int, default=7)
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument(
        "--label-mode",
        choices=("economic_semantic", "gmm"),
        default="economic_semantic",
    )
    parser.add_argument("--skip-final-fit", action="store_true")
    parser.add_argument("--skip-market-layer", action="store_true")
    parser.add_argument(
        "--disable-residual-aegmm",
        action="store_true",
        help="Ablate the local AE/GMM block while keeping identical semantic labels.",
    )
    parser.add_argument("--skip-8d-assessment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.data)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else None
    time_mask = data["__ts__"].ge(start_ts)
    if end_ts is not None:
        time_mask &= data["__ts__"].lt(end_ts)
    if not bool(time_mask.all()):
        data = data.loc[time_mask].reset_index(drop=True)
    appended_ledger_rows: dict[str, Any] = {
        "status": "disabled",
        "appended_rows": 0,
    }
    if args.append_ledger_only_rows:
        data, appended_ledger_rows = _append_feature_hydrated_ledger_rows(
            data, args.ledger, args.feature_root
        )
    data = _refresh_ledger_scores(data, args.ledger)
    score_col = _resolve_score(data, args.score_column)
    data = (
        data.loc[pd.to_numeric(data[score_col], errors="coerce").notna()]
        .sort_values(["__ts__", "__symbol__", "side_name"], kind="stable")
        .reset_index(drop=True)
    )
    policy_assessment: pd.DataFrame | None = None
    policy: dict[str, Any] | None = None
    candidates = _residual_candidate_features(data, score_col)
    folds = _folds_for_data(data)
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    episode_metric_frames: list[pd.DataFrame] = []
    fold_manifests: list[dict[str, Any]] = []
    frozen_local_ae_gmm: dict[
        tuple[str, str], tuple[dict[str, Any], list[str], list[str]]
    ] = {}
    for fold_index, (fold, valid_start, valid_end) in enumerate(folds):
        train = data.loc[data["__ts__"].lt(valid_start)]
        valid = data.loc[data["__ts__"].ge(valid_start) & data["__ts__"].lt(valid_end)]
        valid_days = int(valid["__ts__"].dt.floor("D").nunique())
        if (
            len(train) < 5_000
            or len(valid) < 100
            or valid_days < int(args.min_valid_days)
        ):
            continue
        config = ResidualArchetypeConfig(
            score_col=score_col,
            rank_scope="global",
            min_local_rows=int(args.min_local_rows),
            max_recognizer_features=int(args.max_features),
            use_residual_ae_gmm=not bool(args.disable_residual_aegmm),
            allow_side_fallback=False,
            label_mode=str(args.label_mode),
            random_state=20260712 + fold_index * 101,
        )
        local = ResidualArchetypeRecognizer(config, candidates)
        local.frozen_ae_gmm_by_local = dict(frozen_local_ae_gmm)
        local.fit(train)
        for key, model in local.local_models.items():
            if key not in frozen_local_ae_gmm and model.ae_gmm_state:
                frozen_local_ae_gmm[key] = (
                    model.ae_gmm_state,
                    list(model.ae_gmm_input_features),
                    list(model.ae_gmm_output_features),
                )
        safe_valid = strip_outcomes_for_oos(valid)
        generated = local.transform_oos(safe_valid)
        market_manifest: dict[str, Any] | None = None
        market_targets = pd.DataFrame(index=valid.index)
        if not args.skip_market_layer:
            try:
                market = PerArchetypeMarketAdverseRecognizer(
                    PerArchetypeMarketAdverseConfig(
                        score_col=score_col,
                        min_archetype_rows=int(args.min_local_rows),
                        max_features=min(int(args.max_features), 64),
                        random_state=20260712 + fold_index * 101,
                    ),
                    candidates,
                ).fit(train)
                market_generated = market.transform_oos(safe_valid)
                generated = pd.concat([generated, market_generated], axis=1, copy=False)
                market_targets = market.prepare_evaluation_targets(valid)
                market_manifest = market.manifest()
            except ValueError as exc:
                market_manifest = {
                    "status": "unavailable_insufficient_or_degenerate_support",
                    "reason": str(exc),
                }
        targets = local.prepare_evaluation_targets(valid)
        keep = [
            name
            for name in (
                "__ts__",
                "__symbol__",
                "side_name",
                "archetype_policy_key",
                "ev_after_1pct",
                "clean_exec",
                "dirty_positive",
                "first_touch_bad_mae_1r",
                "full_path_bad_mae_1r",
                "timeout",
                "reference_rank_pct",
                "reference_ev_equivalent_selected",
                "market_adjusted_hit_surprise",
                "negative_autocorr_label",
                "positive_autocorr_label",
            )
            if name in targets.columns
        ]
        predictions = pd.concat(
            [
                targets[keep].reset_index(drop=True),
                generated.reset_index(drop=True),
                market_targets.reset_index(drop=True),
            ],
            axis=1,
        )
        predictions["fold"] = fold
        prediction_frames.append(predictions)
        metric_frames.append(_metrics(predictions, fold))
        episode_metrics = adverse_episode_ranking_metrics(predictions)
        if not episode_metrics.empty:
            episode_metrics["fold"] = fold
            episode_metric_frames.append(episode_metrics)
        fold_manifests.append(
            {
                "fold": fold,
                "train_start": str(train["__ts__"].min()),
                "train_end": str(train["__ts__"].max()),
                "valid_start": str(valid_start),
                "valid_end_exclusive": str(valid_end),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "valid_days": valid_days,
                "local": local.manifest(),
                "frozen_ae_gmm_key_count": int(len(frozen_local_ae_gmm)),
                "market": market_manifest,
            }
        )
        print(
            json.dumps(
                {
                    "event": "residual_fold_complete",
                    "fold": fold,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                }
            ),
            flush=True,
        )

    predictions_all = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    metrics_all = (
        pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    )
    predictions_all.to_parquet(
        output / "oos_residual_state_predictions.parquet",
        index=False,
        compression="zstd",
    )
    metrics_all.to_csv(output / "oos_metrics_side_archetype.csv", index=False)
    episode_metrics_all = (
        pd.concat(episode_metric_frames, ignore_index=True)
        if episode_metric_frames
        else pd.DataFrame()
    )
    episode_metrics_all.to_csv(
        output / "oos_market_adverse_episode_metrics.csv", index=False
    )

    assessment_manifest: dict[str, Any] | None = None
    if not args.skip_8d_assessment:
        from scripts.score_frozen_champion_full_history import (
            _apply_causal_reachable_ev_policy,
        )

        policy = (
            json.loads(args.threshold_policy.read_text(encoding="utf-8"))
            if args.threshold_policy.exists()
            else {
                "policy_id": "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1",
                "window_days": 8,
                "min_reference_rows": 40,
                "arch_min_reference_rows": 10,
                "top_band_floor": 0.90,
            }
        )
        policy_score_col = (
            str(args.assessment_score_column)
            if args.assessment_score_column in data.columns
            else score_col
        )
        policy_assessment = _apply_causal_reachable_ev_policy(
            data,
            policy=policy,
            score_col=policy_score_col,
            preserve_materialized_policy=False,
        )
        assessment = policy_assessment
        columns = [
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "threshold_basis_selected",
            "threshold_basis_rank_score",
            "threshold_basis_dynamic_ev_target",
            "threshold_basis_dynamic_score_threshold",
        ]
        assessment[columns].to_parquet(
            output / "assessment_8d_hr_off_regimecal_v1.parquet",
            index=False,
            compression="zstd",
        )
        assessment_manifest = {
            "score_column": str(args.assessment_score_column),
            "rows": int(len(assessment)),
            "selected_rows": int(assessment["threshold_basis_selected"].sum()),
            "used_as_model_feature": False,
            "policy_path": str(args.threshold_policy),
            "policy_id": str(policy.get("policy_id", "")),
        }

    final_manifest: dict[str, Any] | None = None
    if not args.skip_final_fit:
        final_config = ResidualArchetypeConfig(
            score_col=score_col,
            rank_scope="global",
            min_local_rows=int(args.min_local_rows),
            max_recognizer_features=int(args.max_features),
            use_residual_ae_gmm=not bool(args.disable_residual_aegmm),
            allow_side_fallback=False,
            label_mode=str(args.label_mode),
            final_refit_all_rows=True,
            random_state=20260712,
        )
        final_local = ResidualArchetypeRecognizer(final_config, candidates).fit(data)
        joblib.dump(
            final_local, output / "final_local_residual_recognizer.joblib", compress=3
        )
        final_local.catalog_.to_csv(
            output / "final_local_residual_catalog.csv", index=False
        )
        final_manifest = {"local": final_local.manifest()}
        if not args.skip_market_layer:
            try:
                final_market = PerArchetypeMarketAdverseRecognizer(
                    PerArchetypeMarketAdverseConfig(score_col=score_col), candidates
                ).fit(data)
                joblib.dump(
                    final_market,
                    output / "final_market_residual_recognizer.joblib",
                    compress=3,
                )
                final_manifest["market"] = final_market.manifest()
            except ValueError as exc:
                final_manifest["market"] = {
                    "status": "unavailable_insufficient_or_degenerate_support",
                    "reason": str(exc),
                }

    manifest = {
        "schema": "meta_residual_archetype_discovery_v1",
        "data": str(args.data.resolve()),
        "ledger": str(args.ledger.resolve()) if args.ledger.exists() else None,
        "available_start": str(data["__ts__"].min()),
        "available_end": str(data["__ts__"].max()),
        "score_column": score_col,
        "candidate_feature_count": int(len(candidates)),
        "label_mode": str(args.label_mode),
        "residual_ae_gmm_enabled": not bool(args.disable_residual_aegmm),
        "oos_ae_gmm_contract": (
            "fit once at first supported side x archetype training fold, then frozen"
        ),
        "candidate_features": candidates,
        "excluded_post_meta_outputs": sorted(POST_META_OUTPUT_COLUMNS | {score_col}),
        "appended_ledger_rows": appended_ledger_rows,
        "folds": fold_manifests,
        "assessment_8d_no_regime": assessment_manifest,
        "final_fit": final_manifest,
        "contracts": {
            "primary_target": "raw frozen-score signed surprise, adjusted by same-timestamp market surprise",
            "local_scope": "side x archetype; global top20 population and train-derived local EV-equivalent top10 threshold",
            "market_scope": (
                "per-archetype adverse episode discovery from market-wide features; "
                "continuous semantic GMM posteriors plus supervised risk probabilities"
            ),
            "recent_performance": (
                "saved 8-day reachable-EV regime-calibrated policy is assessment-only"
            ),
            "hmm": "disabled",
            "final_fit_status": (
                "post-validation all-row refit through available_end; not used as OOS evidence"
            ),
        },
    }
    _write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
