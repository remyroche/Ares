#!/usr/bin/env python3
"""Legacy diagnostic A-F correction ablation for global residual states.

This script does not implement the champion promotion contract because it fits
monthly Huber EV correction models.  Use
``run_global_residual_champion_enhancement.py`` for fixed-cutoff, soft-label,
greedy revisions of the native meta champion.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# Frozen AE inference is small and more stable on one Torch thread; LightGBM
# correction fits are capped at two workers below.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.global_residual_latent_state import (  # noqa: E402
    add_temporal_state_features,
    prepare_archetype_state_partition,
)
from scripts.score_compare_meta_residual_july_oos import (  # noqa: E402
    _append_store_features,
)

DISCOVERY_ROOT = ROOT / "data_perp/reports/global_residual_state_discovery_20260711_v1"
SOURCE_ROOT = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710"
)
DEFAULT_STATES = (
    DISCOVERY_ROOT / "global_side_latent_states/side_timestamp_market_states.parquet"
)
DEFAULT_STATE_MODELS = DISCOVERY_ROOT / "global_side_latent_states/states"
DEFAULT_PARTITION_MANIFEST = (
    DISCOVERY_ROOT / "global_side_latent_states/archetype_partition_manifest.json"
)
DEFAULT_OUTPUT = DISCOVERY_ROOT / "model_ablation"
DEFAULT_FEATURE_ROOT = ROOT / "data_perp/features/20260710_170000"
DEFAULT_CANDIDATES = SOURCE_ROOT / "candidate_shards"
DEFAULT_LEDGER = SOURCE_ROOT / "frozen_champion_single_source_ledger.parquet"
CHAMPION_SCORE_COLUMN = "score_regime_calibrated"

EXISTING_STATE_HINTS = (
    "gmm_prob_",
    "gmm_cluster_posterior_",
    "gmm_mahal",
    "gmm_dist_center",
    "mahalanobis",
    "reconstruction",
    "cluster_speed",
    "cluster_acceleration",
    "state_spectral_",
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    outcome_columns = [
        CHAMPION_SCORE_COLUMN,
        "selected_for_monitor",
        "hit_probability",
        "ev_after_1pct",
        "clean_exec",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    ledger_columns = [
        "row_id",
        "__ts__",
        *outcome_columns,
    ]
    print(
        json.dumps({"event": "ablation_load", "stage": "ledger_read_start"}), flush=True
    )
    ledger = pd.read_parquet(args.ledger, columns=ledger_columns)
    print(
        json.dumps(
            {
                "event": "ablation_load",
                "stage": "ledger_read_complete",
                "rows": len(ledger),
            }
        ),
        flush=True,
    )
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="coerce")
    ledger["_row_hash"] = pd.util.hash_pandas_object(
        ledger["row_id"], index=False
    ).to_numpy(dtype=np.uint64, copy=False)
    ledger = ledger.drop(columns="row_id")
    for name in ledger.select_dtypes(include=["float64"]).columns:
        ledger[name] = pd.to_numeric(ledger[name], downcast="float")
    ledger_by_id = ledger.drop_duplicates("_row_hash").set_index("_row_hash")
    print(
        json.dumps(
            {
                "event": "ablation_load",
                "stage": "ledger_index_complete",
                "rows": len(ledger_by_id),
            }
        ),
        flush=True,
    )
    first = sorted(Path(args.candidate_root).glob("candidates_*.parquet"))[0]
    schema = pq.ParquetFile(first).schema_arrow.names
    existing = [
        name
        for name in schema
        if any(hint in name.lower() for hint in EXISTING_STATE_HINTS)
    ]
    keep = list(
        dict.fromkeys(
            [
                "row_id",
                "__ts__",
                "__symbol__",
                "side_name",
                "archetype_policy_key",
                *existing,
            ]
        )
    )
    parts: list[pd.DataFrame] = []
    candidate_paths = sorted(Path(args.candidate_root).glob("candidates_*.parquet"))
    for path_index, path in enumerate(candidate_paths, start=1):
        part = pd.read_parquet(path, columns=keep)
        part["_row_hash"] = pd.util.hash_pandas_object(
            part["row_id"], index=False
        ).to_numpy(dtype=np.uint64, copy=False)
        part = part.drop(columns="row_id")
        overlay = ledger_by_id.reindex(part["_row_hash"].to_numpy()).reset_index(
            drop=True
        )
        for name in outcome_columns:
            part[name] = overlay[name].to_numpy()
        parts.append(part)
        print(
            json.dumps(
                {
                    "event": "ablation_load",
                    "stage": "candidate_shard_complete",
                    "shard": path_index,
                    "shards": len(candidate_paths),
                    "rows": len(part),
                }
            ),
            flush=True,
        )
    historical = pd.concat(parts, ignore_index=True, copy=False)
    print(
        json.dumps(
            {
                "event": "ablation_load",
                "stage": "historical_concat_complete",
                "rows": len(historical),
            }
        ),
        flush=True,
    )
    del parts
    for name in ("__symbol__", "side_name", "archetype_policy_key"):
        historical[name] = historical[name].astype("category")
    july_columns = [
        "row_id",
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        *outcome_columns,
    ]
    july = pd.read_parquet(
        args.ledger,
        columns=july_columns,
        filters=[("__ts__", ">=", pd.Timestamp("2026-07-01", tz="UTC"))],
    )
    print(
        json.dumps(
            {"event": "ablation_load", "stage": "july_read_complete", "rows": len(july)}
        ),
        flush=True,
    )
    july["_row_hash"] = pd.util.hash_pandas_object(
        july["row_id"], index=False
    ).to_numpy(dtype=np.uint64, copy=False)
    july = july.drop(columns="row_id")
    for name in existing:
        if name not in july:
            july[name] = np.nan
    frame = pd.concat(
        [historical, july.reindex(columns=historical.columns)],
        ignore_index=True,
        sort=False,
    )
    del historical, july, ledger, ledger_by_id
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    for name in ("__symbol__", "side_name", "archetype_policy_key"):
        frame[name] = frame[name].astype("category")

    for name in frame.select_dtypes(include=["float64"]).columns:
        frame[name] = pd.to_numeric(frame[name], downcast="float")
    print(
        json.dumps({"event": "ablation_load", "stage": "complete", "rows": len(frame)}),
        flush=True,
    )
    gc.collect()
    return frame


def _state_features_for_fold(
    states: pd.DataFrame,
    model_root: Path,
    month: str,
    partitions: list[dict[str, str]],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
    for partition in partitions:
        token = partition["token"]
        side = partition["side_name"]
        local = prepare_archetype_state_partition(
            states,
            side=side,
            archetype=partition["archetype_policy_key"],
        )
        local = local.loc[local["__ts__"].lt(end)].copy()
        bundle_path = model_root / f"global_residual_state_{token}_{month}.joblib"
        if not bundle_path.exists():
            continue
        print(
            json.dumps(
                {
                    "event": "state_feature_fold",
                    "stage": "bundle_load_start",
                    "month": month,
                    "state_partition_token": token,
                }
            ),
            flush=True,
        )
        bundle = joblib.load(bundle_path)
        print(
            json.dumps(
                {
                    "event": "state_feature_fold",
                    "stage": "bundle_load_complete",
                    "month": month,
                    "state_partition_token": token,
                }
            ),
            flush=True,
        )
        latent = bundle["ae"].transform(local)
        print(
            json.dumps(
                {
                    "event": "state_feature_fold",
                    "stage": "ae_transform_complete",
                    "month": month,
                    "side": side,
                }
            ),
            flush=True,
        )
        static = bundle["gmm"].transform(latent)
        print(
            json.dumps(
                {
                    "event": "state_feature_fold",
                    "stage": "gmm_transform_complete",
                    "month": month,
                    "side": side,
                }
            ),
            flush=True,
        )
        temporal = add_temporal_state_features(static, local["__ts__"])
        print(
            json.dumps(
                {
                    "event": "state_feature_fold",
                    "stage": "temporal_complete",
                    "month": month,
                    "side": side,
                }
            ),
            flush=True,
        )
        generated = pd.concat(
            [
                local[["__ts__", "side_name"]].reset_index(drop=True),
                latent.reset_index(drop=True),
                temporal.reset_index(drop=True),
            ],
            axis=1,
        )
        generated["archetype_policy_key"] = partition["archetype_policy_key"]
        generated["state_partition_token"] = token
        outputs.append(generated)
        del bundle, latent, static, temporal, generated, local
        gc.collect()
    return pd.concat(outputs, ignore_index=True, sort=False)


def _feature_arms(frame: pd.DataFrame) -> dict[str, list[str]]:
    baseline = [CHAMPION_SCORE_COLUMN]
    existing = [
        name
        for name in frame
        if not name.startswith("global_state_")
        and any(hint in name.lower() for hint in EXISTING_STATE_HINTS)
    ]
    lifecycle = [
        name
        for name in CFG.get("CRASH_LIFECYCLE_NEW_FEATURE_KEYS", [])
        if name in frame
    ]
    global_static = [
        name
        for name in frame
        if name.startswith("global_state_")
        and not any(
            token in name
            for token in ("_delta_", "_acceleration", "speed", "dwell", "transition")
        )
        and "expected_negative_ev" not in name
        and "pred_negative_ev" not in name
    ]
    temporal = [
        name
        for name in frame
        if (
            name.startswith("global_state_")
            and any(
                token in name
                for token in (
                    "_delta_",
                    "_acceleration",
                    "speed",
                    "dwell",
                    "transition",
                )
            )
        )
    ]
    negative_ev = [
        name
        for name in frame
        if name.startswith("global_state_") and "negative_ev" in name
    ]
    return {
        "A_current_champion": baseline,
        "B_existing_aegmm_state": list(dict.fromkeys(baseline + existing)),
        "C_new_lifecycle_features": list(
            dict.fromkeys(baseline + existing + lifecycle)
        ),
        "D_global_residual_aegmm": list(
            dict.fromkeys(baseline + existing + lifecycle + global_static)
        ),
        "E_temporal_transition_features": list(
            dict.fromkeys(baseline + existing + lifecycle + global_static + temporal)
        ),
        "F_expected_negative_ev_head": list(
            dict.fromkeys(
                baseline + existing + lifecycle + global_static + temporal + negative_ev
            )
        ),
    }


def _fit_predict_arm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    def build_model(n_estimators: int = 240) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            objective="huber",
            n_estimators=int(n_estimators),
            learning_rate=0.035,
            num_leaves=31,
            max_depth=5,
            min_child_samples=80,
            max_bin=63,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.10,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=2,
            force_col_wise=True,
            verbosity=-1,
        )

    def rank(values: pd.Series | np.ndarray) -> np.ndarray:
        return (
            pd.Series(np.asarray(values, dtype=np.float64))
            .rank(method="average", pct=True, na_option="bottom")
            .to_numpy(dtype=np.float32)
        )

    target = pd.to_numeric(train["ev_after_1pct"], errors="coerce")
    valid_target = target.notna()
    usable_train = train.loc[valid_target]
    usable_target = target.loc[valid_target]
    if len(usable_train) < 500:
        return rank(valid[CHAMPION_SCORE_COLUMN]), {
            "blend_alpha": 0.0,
            "calibration_objective": np.nan,
            "calibration_rows": 0.0,
        }

    # Choose correction strength on a trailing, train-only validation block.
    # Alpha zero preserves the current champion and prevents a weak correction
    # model from replacing a working rank ordering.
    calibration_start = usable_train["__ts__"].max() - pd.Timedelta(days=60)
    calibration_mask = usable_train["__ts__"].ge(calibration_start).to_numpy()
    if calibration_mask.sum() < 2_000 or (~calibration_mask).sum() < 5_000:
        split = int(len(usable_train) * 0.85)
        calibration_mask = np.arange(len(usable_train)) >= split
    fit_frame = usable_train.loc[~calibration_mask]
    fit_target = usable_target.loc[fit_frame.index]
    calibration = usable_train.loc[calibration_mask]
    calibration_target = usable_target.loc[calibration.index]
    calibration_model = build_model()
    calibration_model.fit(
        fit_frame[features],
        fit_target,
        eval_set=[(calibration[features], calibration_target)],
        callbacks=[lgb.early_stopping(20, first_metric_only=True, verbose=False)],
    )
    model_rank = rank(calibration_model.predict(calibration[features]))
    base_rank = rank(calibration[CHAMPION_SCORE_COLUMN])
    week = calibration["__ts__"].dt.floor("D") - pd.to_timedelta(
        calibration["__ts__"].dt.weekday, unit="D"
    )
    best_alpha = 0.0
    best_objective = -np.inf
    for alpha in np.linspace(0.0, 1.0, 9):
        blended = (1.0 - alpha) * base_rank + alpha * model_rank
        order = np.argsort(-blended, kind="stable")
        n10 = max(1, int(np.ceil(len(order) * 0.10)))
        n20 = max(1, int(np.ceil(len(order) * 0.20)))
        ev10 = float(calibration_target.iloc[order[:n10]].mean())
        ev20 = float(calibration_target.iloc[order[:n20]].mean())
        selected_week = (
            pd.DataFrame(
                {
                    "week": week.iloc[order[:n10]].to_numpy(),
                    "ev": calibration_target.iloc[order[:n10]].to_numpy(),
                }
            )
            .groupby("week", observed=True)["ev"]
            .mean()
        )
        worst_week = float(selected_week.min()) if not selected_week.empty else 0.0
        objective = 0.60 * ev10 + 0.30 * ev20 + 0.10 * worst_week
        if objective > best_objective:
            best_objective = objective
            best_alpha = float(alpha)
    best_iteration = int(calibration_model.best_iteration_ or 240)
    del calibration_model

    full_iterations = max(40, min(240, int(np.ceil(best_iteration * 1.10))))
    model = build_model(full_iterations)
    model.fit(usable_train[features], usable_target)
    prediction_rank = rank(model.predict(valid[features]))
    base_valid_rank = rank(valid[CHAMPION_SCORE_COLUMN])
    blended = (1.0 - best_alpha) * base_valid_rank + best_alpha * prediction_rank
    return blended.astype(np.float32), {
        "blend_alpha": best_alpha,
        "calibration_objective": float(best_objective),
        "calibration_rows": float(calibration_mask.sum()),
        "best_iteration": float(best_iteration),
        "full_iterations": float(full_iterations),
    }


def _selection_metrics(frame: pd.DataFrame, score_col: str, arm: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = frame.copy()
    frame["week_start"] = frame["__ts__"].dt.floor("D") - pd.to_timedelta(
        frame["__ts__"].dt.weekday, unit="D"
    )
    for label, fraction in (("top10", 0.10), ("top20", 0.20)):
        score = pd.to_numeric(frame[score_col], errors="coerce")
        budget = max(1, int(np.ceil(len(frame) * fraction)))
        selected = frame.loc[score.nlargest(budget, keep="first").index]
        week = selected.groupby("week_start", observed=True)["ev_after_1pct"].mean()
        day = selected.groupby(selected["__ts__"].dt.floor("D"), observed=True)[
            "ev_after_1pct"
        ].mean()
        day_key = selected["__ts__"].dt.floor("D")
        daily_ev = (
            pd.to_numeric(selected["ev_after_1pct"], errors="coerce")
            .groupby(day_key, observed=True)
            .mean()
        )
        surprise = pd.to_numeric(
            selected["clean_exec"], errors="coerce"
        ) - pd.to_numeric(selected["hit_probability"], errors="coerce")
        daily_surprise = surprise.groupby(day_key, observed=True).mean()
        ev_ac1 = daily_ev.autocorr(lag=1) if len(daily_ev) >= 5 else np.nan
        signed_ac1 = (
            daily_surprise.autocorr(lag=1) if len(daily_surprise) >= 5 else np.nan
        )
        negative_ac1 = (
            daily_surprise.clip(upper=0.0).autocorr(lag=1)
            if len(daily_surprise) >= 5
            else np.nan
        )
        positive_ac1 = (
            daily_surprise.clip(lower=0.0).autocorr(lag=1)
            if len(daily_surprise) >= 5
            else np.nan
        )
        rows.append(
            {
                "arm": arm,
                "selector": label,
                "selected_rows": len(selected),
                "mean_ev_after_1pct": float(
                    pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean()
                ),
                "sum_ev_after_1pct": float(
                    pd.to_numeric(selected["ev_after_1pct"], errors="coerce").sum()
                ),
                "positive_ev_rate": float(
                    pd.to_numeric(selected["ev_after_1pct"], errors="coerce")
                    .gt(0)
                    .mean()
                ),
                "clean_precision": float(
                    pd.to_numeric(selected["clean_exec"], errors="coerce").mean()
                ),
                "bad_mae_rate": float(
                    pd.to_numeric(
                        selected["full_path_bad_mae_1r"], errors="coerce"
                    ).mean()
                ),
                "timeout_rate": float(
                    pd.to_numeric(selected["timeout"], errors="coerce").mean()
                ),
                "worst_week_ev": float(week.min()),
                "worst_day_ev": float(day.min()),
                "positive_week_fraction": float(week.gt(0).mean()),
                "daily_ev_ac1": float(ev_ac1) if np.isfinite(ev_ac1) else np.nan,
                "signed_surprise_ac1": float(signed_ac1)
                if np.isfinite(signed_ac1)
                else np.nan,
                "negative_surprise_ac1": float(negative_ac1)
                if np.isfinite(negative_ac1)
                else np.nan,
                "positive_surprise_ac1": float(positive_ac1)
                if np.isfinite(positive_ac1)
                else np.nan,
            }
        )
    baseline_count = int(frame["selected_for_monitor"].fillna(False).sum())
    matched = (
        frame.nlargest(max(baseline_count, 1), score_col)
        if baseline_count
        else frame.iloc[0:0]
    )
    matched_day = matched["__ts__"].dt.floor("D")
    matched_daily_ev = (
        pd.to_numeric(matched["ev_after_1pct"], errors="coerce")
        .groupby(matched_day, observed=True)
        .mean()
    )
    matched_surprise = pd.to_numeric(
        matched["clean_exec"], errors="coerce"
    ) - pd.to_numeric(matched["hit_probability"], errors="coerce")
    matched_daily_surprise = matched_surprise.groupby(matched_day, observed=True).mean()
    rows.append(
        {
            "arm": arm,
            "selector": "matched_current_activity",
            "selected_rows": len(matched),
            "mean_ev_after_1pct": float(
                pd.to_numeric(matched["ev_after_1pct"], errors="coerce").mean()
            ),
            "sum_ev_after_1pct": float(
                pd.to_numeric(matched["ev_after_1pct"], errors="coerce").sum()
            ),
            "positive_ev_rate": float(
                pd.to_numeric(matched["ev_after_1pct"], errors="coerce").gt(0).mean()
            ),
            "clean_precision": float(
                pd.to_numeric(matched["clean_exec"], errors="coerce").mean()
            ),
            "bad_mae_rate": float(
                pd.to_numeric(matched["full_path_bad_mae_1r"], errors="coerce").mean()
            ),
            "timeout_rate": float(
                pd.to_numeric(matched["timeout"], errors="coerce").mean()
            ),
            "daily_ev_ac1": float(matched_daily_ev.autocorr(lag=1))
            if len(matched_daily_ev) >= 5
            else np.nan,
            "signed_surprise_ac1": float(matched_daily_surprise.autocorr(lag=1))
            if len(matched_daily_surprise) >= 5
            else np.nan,
            "negative_surprise_ac1": float(
                matched_daily_surprise.clip(upper=0.0).autocorr(lag=1)
            )
            if len(matched_daily_surprise) >= 5
            else np.nan,
            "positive_surprise_ac1": float(
                matched_daily_surprise.clip(lower=0.0).autocorr(lag=1)
            )
            if len(matched_daily_surprise) >= 5
            else np.nan,
        }
    )
    return pd.DataFrame(rows)


def _breakdown_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["oos_month"] = frame["__ts__"].dt.strftime("%Y-%m")
    frame["week_start"] = frame["__ts__"].dt.floor("D") - pd.to_timedelta(
        frame["__ts__"].dt.weekday, unit="D"
    )
    rows: list[dict[str, Any]] = []
    score_columns = [name for name in frame if name.startswith("score__")]
    for score_col in score_columns:
        arm = score_col.removeprefix("score__")
        for month, month_frame in frame.groupby("oos_month", observed=True):
            selectors: dict[str, pd.Series] = {}
            score = pd.to_numeric(month_frame[score_col], errors="coerce")
            for label, fraction in (("top10", 0.10), ("top20", 0.20)):
                mask = pd.Series(False, index=month_frame.index)
                budget = max(1, int(np.ceil(len(month_frame) * fraction)))
                mask.loc[score.nlargest(budget, keep="first").index] = True
                selectors[label] = mask
            baseline_count = int(
                month_frame["selected_for_monitor"].fillna(False).sum()
            )
            matched = pd.Series(False, index=month_frame.index)
            if baseline_count:
                matched.loc[score.nlargest(baseline_count).index] = True
            selectors["matched_current_activity"] = matched
            for selector, mask in selectors.items():
                selected = month_frame.loc[mask]
                scopes = (
                    ("overall", [], [((), selected)]),
                    (
                        "side",
                        ["side_name"],
                        selected.groupby("side_name", observed=True),
                    ),
                    (
                        "archetype",
                        ["archetype_policy_key"],
                        selected.groupby("archetype_policy_key", observed=True),
                    ),
                    (
                        "week",
                        ["week_start"],
                        selected.groupby("week_start", observed=True),
                    ),
                    (
                        "month_side_archetype",
                        ["side_name", "archetype_policy_key"],
                        selected.groupby(
                            ["side_name", "archetype_policy_key"], observed=True
                        ),
                    ),
                    (
                        "week_side_archetype",
                        ["week_start", "side_name", "archetype_policy_key"],
                        selected.groupby(
                            ["week_start", "side_name", "archetype_policy_key"],
                            observed=True,
                        ),
                    ),
                )
                for scope, names, groups in scopes:
                    for key, local in groups:
                        key_values = key if isinstance(key, tuple) else (key,)
                        row = {
                            "arm": arm,
                            "selector": selector,
                            "oos_month": month,
                            "scope": scope,
                            "selected_rows": len(local),
                            "mean_ev_after_1pct": float(
                                pd.to_numeric(
                                    local["ev_after_1pct"], errors="coerce"
                                ).mean()
                            ),
                            "sum_ev_after_1pct": float(
                                pd.to_numeric(
                                    local["ev_after_1pct"], errors="coerce"
                                ).sum()
                            ),
                            "positive_ev_rate": float(
                                pd.to_numeric(local["ev_after_1pct"], errors="coerce")
                                .gt(0)
                                .mean()
                            ),
                            "clean_precision": float(
                                pd.to_numeric(
                                    local["clean_exec"], errors="coerce"
                                ).mean()
                            ),
                            "bad_mae_rate": float(
                                pd.to_numeric(
                                    local["full_path_bad_mae_1r"], errors="coerce"
                                ).mean()
                            ),
                            "timeout_rate": float(
                                pd.to_numeric(local["timeout"], errors="coerce").mean()
                            ),
                        }
                        row.update(
                            {
                                name: value
                                for name, value in zip(names, key_values, strict=True)
                            }
                        )
                        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--state-model-root", type=Path, default=DEFAULT_STATE_MODELS)
    parser.add_argument(
        "--partition-manifest", type=Path, default=DEFAULT_PARTITION_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-months", default="2026-04,2026-05,2026-06,2026-07")
    parser.add_argument(
        "--arms",
        default=(
            "B_existing_aegmm_state,C_new_lifecycle_features,"
            "D_global_residual_aegmm,E_temporal_transition_features,"
            "F_expected_negative_ev_head"
        ),
        help="Comma-separated challenger arms; the current champion is always emitted.",
    )
    parser.add_argument(
        "--purge-hours",
        type=float,
        default=12.0,
        help="Exclude train rows this many hours before each OOS month.",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    eval_months = [
        value.strip() for value in args.eval_months.split(",") if value.strip()
    ]
    requested_arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    valid_arms = {
        "B_existing_aegmm_state",
        "C_new_lifecycle_features",
        "D_global_residual_aegmm",
        "E_temporal_transition_features",
        "F_expected_negative_ev_head",
    }
    unknown_arms = sorted(set(requested_arms).difference(valid_arms))
    if unknown_arms:
        raise ValueError(f"Unknown ablation arms: {unknown_arms}")
    print(
        json.dumps({"event": "ablation_load", "stage": "states_read_start"}), flush=True
    )
    states = pd.read_parquet(args.states)
    print(
        json.dumps(
            {
                "event": "ablation_load",
                "stage": "states_read_complete",
                "rows": len(states),
            }
        ),
        flush=True,
    )
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True)
    print(
        json.dumps({"event": "ablation_load", "stage": "states_timestamp_complete"}),
        flush=True,
    )
    partition_manifest = json.loads(Path(args.partition_manifest).read_text())
    partitions = list(partition_manifest.get("partitions", []))
    if not partitions:
        raise ValueError("No archetype state partitions are available")
    state_features_by_month: dict[str, pd.DataFrame] = {}
    for month in eval_months:
        print(
            json.dumps(
                {
                    "event": "model_ablation_fold",
                    "stage": "state_transform_start",
                    "month": month,
                }
            ),
            flush=True,
        )
        state_features_by_month[month] = _state_features_for_fold(
            states, Path(args.state_model_root), month, partitions
        )
        print(
            json.dumps(
                {
                    "event": "model_ablation_fold",
                    "stage": "state_transform_complete",
                    "month": month,
                    "rows": len(state_features_by_month[month]),
                }
            ),
            flush=True,
        )
    del states
    gc.collect()
    candidates = _load_candidates(args)
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[pd.DataFrame] = []
    feature_manifests: dict[str, Any] = {}
    blend_manifests: dict[str, Any] = {}
    for month in eval_months:
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train_cutoff = start - pd.Timedelta(hours=float(args.purge_hours))
        state_features = state_features_by_month[month]
        month_mask = candidates["__ts__"].ge(start) & candidates["__ts__"].lt(end)
        valid = candidates.loc[month_mask].copy()
        print(
            json.dumps(
                {
                    "event": "model_ablation_fold",
                    "stage": "valid_slice_complete",
                    "month": month,
                    "rows": len(valid),
                }
            ),
            flush=True,
        )
        champion_coverage = float(
            pd.to_numeric(valid[CHAMPION_SCORE_COLUMN], errors="coerce").notna().mean()
        )
        if champion_coverage < 0.95:
            raise ValueError(
                f"Champion score coverage is {champion_coverage:.2%} for {month}; "
                "refusing to turn missing scores into an arbitrary top-k baseline"
            )
        valid["score__A_current_champion"] = pd.to_numeric(
            valid[CHAMPION_SCORE_COLUMN], errors="coerce"
        ).astype(np.float32)
        lifecycle = list(CFG.get("CRASH_LIFECYCLE_NEW_FEATURE_KEYS", []))
        month_arms: dict[str, list[str]] = {
            "A_current_champion": [CHAMPION_SCORE_COLUMN]
        }
        month_blends: dict[str, Any] = {}
        month_features_by_archetype: dict[str, dict[str, list[str]]] = {}

        # Candidate-level lifecycle features are the wide part of this ablation.
        # Side slices only bound peak memory while loading them. Every model,
        # calibration split, and blend is fitted independently per archetype.
        for side in ("long", "short"):
            side_source = candidates.loc[
                candidates["side_name"].eq(side) & candidates["__ts__"].lt(end)
            ].copy()
            print(
                json.dumps(
                    {
                        "event": "model_ablation_fold",
                        "stage": "side_slice_complete",
                        "month": month,
                        "side": side,
                        "rows": len(side_source),
                    }
                ),
                flush=True,
            )
            side_source, _ = _append_store_features(
                side_source,
                Path(args.feature_root),
                lifecycle,
            )
            side_source = side_source.merge(
                state_features[state_features["side_name"].eq(side)],
                on=["__ts__", "side_name", "archetype_policy_key"],
                how="left",
                validate="many_to_one",
            )
            for name in side_source.select_dtypes(include=["float64"]).columns:
                side_source[name] = pd.to_numeric(side_source[name], downcast="float")
            side_archetypes = sorted(
                side_source["archetype_policy_key"].dropna().astype(str).unique()
            )
            for archetype_index, archetype in enumerate(side_archetypes):
                local_mask = (
                    side_source["archetype_policy_key"].astype(str).eq(archetype)
                )
                local_source = side_source.loc[local_mask]
                local_valid_mask = local_source["__ts__"].ge(start).to_numpy()
                valid_mask = valid["side_name"].eq(side).to_numpy()
                valid_mask &= (
                    valid["archetype_policy_key"].astype(str).eq(archetype).to_numpy()
                )
                valid_positions = np.flatnonzero(valid_mask)
                if not np.array_equal(
                    local_source.loc[local_valid_mask, "_row_hash"].to_numpy(),
                    valid.iloc[valid_positions]["_row_hash"].to_numpy(),
                ):
                    raise RuntimeError(
                        f"Candidate order mismatch for {month} {side} {archetype}"
                    )
                if not np.any(local_valid_mask):
                    continue

                train_local = local_source.loc[local_source["__ts__"].lt(train_cutoff)]
                valid_local = local_source.loc[local_valid_mask]
                local_arms = _feature_arms(local_source)
                token = next(
                    (
                        str(partition["token"])
                        for partition in partitions
                        if partition["side_name"] == side
                        and partition["archetype_policy_key"] == archetype
                    ),
                    f"{side}_{archetype}",
                )
                month_features_by_archetype[token] = {}
                for arm in requested_arms:
                    usable = [
                        name
                        for name in local_arms[arm]
                        if name in local_source
                        and pd.to_numeric(local_source[name], errors="coerce")
                        .notna()
                        .any()
                    ]
                    month_features_by_archetype[token][arm] = usable
                    month_arms[arm] = sorted(set(month_arms.get(arm, ())).union(usable))
                    arm_predictions, blend_report = _fit_predict_arm(
                        train_local,
                        valid_local,
                        usable,
                        seed=(
                            20260711
                            + len(arm) * 17
                            + archetype_index * 101
                            + (0 if side == "long" else 10_000)
                        ),
                    )
                    valid.loc[valid.index[valid_positions], f"score__{arm}"] = (
                        arm_predictions
                    )
                    month_blends[f"{token}__{arm}"] = blend_report
                print(
                    json.dumps(
                        {
                            "event": "model_ablation_archetype_complete",
                            "month": month,
                            "side": side,
                            "archetype_policy_key": archetype,
                            "train_rows": len(train_local),
                            "valid_rows": len(valid_local),
                        }
                    ),
                    flush=True,
                )
                del local_source, train_local, valid_local
            del side_source
            gc.collect()

        feature_manifests[month] = {
            "fit_partition": "archetype_policy_key",
            "arms_union": month_arms,
            "by_archetype": month_features_by_archetype,
        }
        blend_manifests[month] = month_blends
        for arm in month_arms:
            score_col = f"score__{arm}"
            valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce").astype(
                np.float32
            )
            metric = _selection_metrics(valid, score_col, arm)
            metric["oos_month"] = month
            metric_rows.append(metric)
        prediction_columns = [name for name in valid if name.startswith("score__")]
        all_predictions.append(
            valid[
                [
                    "_row_hash",
                    "__ts__",
                    "__symbol__",
                    "side_name",
                    "archetype_policy_key",
                    "ev_after_1pct",
                    "clean_exec",
                    "full_path_bad_mae_1r",
                    "timeout",
                    "selected_for_monitor",
                    *prediction_columns,
                ]
            ]
        )
        print(
            json.dumps(
                {
                    "event": "model_ablation_month_complete",
                    "month": month,
                    "train_rows": int(candidates["__ts__"].lt(start).sum()),
                    "valid_rows": len(valid),
                }
            ),
            flush=True,
        )
        del valid, state_features
        gc.collect()
    predictions = pd.concat(all_predictions, ignore_index=True, sort=False)
    predictions = predictions.rename(columns={"_row_hash": "row_id_hash"})
    metrics = pd.concat(metric_rows, ignore_index=True, sort=False)
    breakdowns = _breakdown_metrics(predictions)
    predictions.to_parquet(
        output / "oos_predictions.parquet", index=False, compression="zstd"
    )
    metrics.to_csv(output / "metrics_by_month.csv", index=False)
    breakdowns.to_parquet(
        output / "metrics_breakdowns.parquet", index=False, compression="zstd"
    )
    _write_json(
        output / "manifest.json",
        {
            "schema": "global_residual_state_model_ablation_v1",
            "arms": feature_manifests,
            "train_only_blend_selection": blend_manifests,
            "cost_contract": "ev_after_1pct already subtracts 1% round-trip cost exactly once",
            "base_contract": "The fixed champion base/meta prediction is identical for all arms.",
            "champion_score_column": CHAMPION_SCORE_COLUMN,
            "fit_contract": (
                "Each archetype-specific correction model trains only on that archetype's rows "
                "before its purged OOS boundary. Side slicing is used only to bound "
                "feature-loading memory."
            ),
            "purge_hours": float(args.purge_hours),
            "evidence_contract": (
                "April-June are correction-layer OOS over fixed-model retrospective backcasts; "
                "July is the genuine frozen post-fit OOS acceptance period."
            ),
            "activity_contract": "Top-k and matched-current-activity selectors prevent abstention-only gains.",
            "row_identity_contract": (
                "row_id_hash is pandas' deterministic uint64 hash of the source row_id; "
                "the original high-cardinality string is excluded to keep the full-history ablation memory-bounded."
            ),
        },
    )


if __name__ == "__main__":
    main()
