#!/usr/bin/env python3
"""Strict-OOS causal support/resistance heads and downstream-ready snapshots.

This is an offline research-only consumer of ``causal_sr_engine`` artifacts.
For every held 2026 month it fits each head solely on interaction labels whose
8-hour outcomes resolved before that month's boundary, scores the held
interaction and entry/continuation snapshot populations, and records all fold
identities.  It intentionally does not import or mutate live trading code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.causal_profile_geometry import (
    ANCHORED_VWAP_FEATURES,
    PROFILE_FEATURES,
    VOLATILITY_PARTICIPATION_FEATURES,
)

SOURCE = ROOT / "data_perp/artifacts/causal_sr_engine_2025_train_2026_score_20260830_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_heads_oof_20260830_v1"
SEED = 1729
MAX_TRAIN_ROWS = 250_000

# All values are available in the pre-touch snapshot.  Deliberately exclude
# touch-bar diagnostics and every realised reaction/penetration outcome.
PRIOR_FEATURES = (
    "level_type_support", "level_type_resistance", "highest_timeframe",
    "independent_confluence_count", "raw_candidate_count", "zone_width_atr",
    "level_age_hours", "time_since_last_touch_hours", "pivot_prominence_atr",
    "original_swing_amplitude_atr", "qualified_touch_count",
    "historical_support", "historical_ESS", "strength_uncertainty_proxy",
    "last_reaction_strength", "mean_reaction_strength", "median_reaction_strength",
    "shrunk_historical_strength", "median_reaction_MFE_atr",
    "median_penetration_MAE_atr", "accepted_break_rate",
    "reaction_strength_slope", "rejection_magnitude_slope",
    "penetration_depth_slope", "reaction_speed_slope",
    "median_time_between_touches_hours", "last_time_between_touches_hours",
    "touch_spacing_slope", "role_reversal_count", "source_swing_1h",
    "source_swing_4h", "source_rolling_extreme", "source_prior_day",
    "source_prior_week", "source_vwap", "source_range_boundary",
    "source_role_reversal",
)
CONDITIONAL_FEATURES = PRIOR_FEATURES + (
    "distance_to_zone_atr", "approach_return_atr", "approach_velocity_atr",
    "approach_acceleration_atr", "approach_path_efficiency",
    "approach_directional_consistency", "approach_impulse_size_atr",
    "approach_pullback_depth_atr", "approach_sign_flip_rate",
    "largest_15m_bar_share", "range_compression_1h",
    "fraction_closes_near_zone", "relative_volume", "volume_acceleration",
)

# These are already part of the retained C1 interaction contract.  The
# explicit removal arm below measures their marginal source/downstream value;
# it avoids presenting a duplicated path/volume feature set as a new finding.
INTERACTION_PATH_FEATURES = tuple(field for field in CONDITIONAL_FEATURES if field not in PRIOR_FEATURES)
CONDITIONAL_FEATURE_CONTRACTS: dict[str, tuple[str, ...]] = {
    "full": CONDITIONAL_FEATURES,
    "without_interaction_path": PRIOR_FEATURES,
}
PROFILE_CONTEXT_AVAILABLE = "profile_context_available"

# Keep the market-profile decomposition at the *source-head context* layer.
# These are not candidate filters and never change the S/R ontology: each
# group is a separately fitted, causal context challenger with the same 2025
# training population and held 2026 months as the retained C1 source heads.
PROFILE_CONTEXT_GROUPS: dict[str, tuple[str, ...]] = {
    "all": PROFILE_FEATURES,
    "levels": (
        "profile_poc_distance_atr",
        "profile_vah_distance_atr",
        "profile_val_distance_atr",
        "profile_hvn_distance_atr",
        "profile_lvn_distance_atr",
        "profile_inside_value_area",
        "profile_value_area_width_atr",
    ),
    "balance": (
        "profile_time_balance_strength",
        "profile_time_balance_distance_atr",
    ),
    "oi_at_price": (
        "profile_oi_at_price_z",
        "profile_oi_positioning_imbalance",
        "profile_oi_support_build",
        "profile_oi_resistance_build",
    ),
    "channels": (
        "bb_zscore",
        "bb_width_atr",
        "bb_percent_b",
        "kc_zscore",
        "kc_width_atr",
        "donchian_position",
        "donchian_width_atr",
        "donchian_upper_distance_atr",
        "donchian_lower_distance_atr",
    ),
}
# The retained profile challenger is levels/value-area.  These three blocks
# test one channel family *on top of that exact challenger*, not as a fresh
# standalone replacement.  This is the relevant downstream question after the
# combined channel block failed its MC1 portfolio gate.
PROFILE_CONTEXT_GROUPS.update({
    "levels_plus_bollinger": (
        *PROFILE_CONTEXT_GROUPS["levels"],
        "bb_zscore",
        "bb_width_atr",
        "bb_percent_b",
    ),
    "levels_plus_keltner": (
        *PROFILE_CONTEXT_GROUPS["levels"],
        "kc_zscore",
        "kc_width_atr",
    ),
    "levels_plus_donchian": (
        *PROFILE_CONTEXT_GROUPS["levels"],
        "donchian_position",
        "donchian_width_atr",
        "donchian_upper_distance_atr",
        "donchian_lower_distance_atr",
    ),
    # New, small level-conditioned contexts.  They do not alter the prior
    # broad-profile control: each is tested only on top of retained levels.
    "levels_plus_volatility_participation": (
        *PROFILE_CONTEXT_GROUPS["levels"], *VOLATILITY_PARTICIPATION_FEATURES,
    ),
    "levels_plus_anchored_vwap": (
        *PROFILE_CONTEXT_GROUPS["levels"], *ANCHORED_VWAP_FEATURES,
    ),
    "levels_plus_volatility_participation_anchored_vwap": (
        *PROFILE_CONTEXT_GROUPS["levels"], *VOLATILITY_PARTICIPATION_FEATURES,
        *ANCHORED_VWAP_FEATURES,
    ),
})

# Backward semantic ablations of the retained levels/value-area challenger.
# VAH/VAL describe price levels, while inside-value-area/width describe the
# value-area geometry; keep those separable so a retained reduction has an
# unambiguous inference contract.
_LEVEL_FEATURES = PROFILE_CONTEXT_GROUPS["levels"]
PROFILE_CONTEXT_GROUPS.update({
    "levels_without_poc": tuple(
        field for field in _LEVEL_FEATURES if field != "profile_poc_distance_atr"
    ),
    "levels_without_vah_val": tuple(
        field for field in _LEVEL_FEATURES
        if field not in {"profile_vah_distance_atr", "profile_val_distance_atr"}
    ),
    "levels_without_hvn_lvn": tuple(
        field for field in _LEVEL_FEATURES
        if field not in {"profile_hvn_distance_atr", "profile_lvn_distance_atr"}
    ),
    "levels_without_value_area_geometry": tuple(
        field for field in _LEVEL_FEATURES
        if field not in {"profile_inside_value_area", "profile_value_area_width_atr"}
    ),
})


def _merge_profile_context(
    frame: pd.DataFrame,
    states: pd.DataFrame,
    *,
    timestamp: str,
    fields: tuple[str, ...],
) -> pd.DataFrame:
    """Attach the latest completed causal profile state without future fill."""
    left = frame.copy().reset_index(names="__profile_merge_order")
    left[timestamp] = pd.to_datetime(left[timestamp], utc=True, errors="raise")
    right = states.loc[:, ["__symbol__", "state_ts", *fields]].copy()
    right["state_ts"] = pd.to_datetime(right.state_ts, utc=True, errors="raise")
    if right.duplicated(["__symbol__", "state_ts"]).any():
        raise AssertionError("profile state source duplicates symbol/time identity")
    pieces: list[pd.DataFrame] = []
    for symbol, part in left.groupby("__symbol__", sort=False):
        profile = right.loc[right["__symbol__"].eq(symbol)].sort_values("state_ts", kind="stable")
        current = part.sort_values(timestamp, kind="stable")
        if profile.empty:
            for field in fields:
                current[field] = np.nan
            current["state_ts"] = pd.NaT
        else:
            current = pd.merge_asof(
                current, profile.drop(columns="__symbol__"), left_on=timestamp, right_on="state_ts",
                direction="backward", allow_exact_matches=True, tolerance=pd.Timedelta(hours=1),
            )
        pieces.append(current)
    merged = pd.concat(pieces, ignore_index=True).sort_values("__profile_merge_order", kind="stable").drop(columns="__profile_merge_order")
    if len(merged) != len(frame):
        raise AssertionError("profile state merge changed source-row identity")
    future = pd.to_datetime(merged.get("state_ts"), utc=True, errors="coerce").gt(merged[timestamp])
    if future.fillna(False).any():
        raise AssertionError("profile state merge attached a future state")
    # Match the original all-field screen's 16/22 (72.7%) presence rule while
    # scaling it correctly for smaller blocks.  This flag is only a model
    # input: individual missing values remain explicit and no row is dropped.
    required = max(1, int(np.ceil(.70 * len(fields))))
    merged[PROFILE_CONTEXT_AVAILABLE] = merged.loc[:, list(fields)].notna().sum(axis=1).ge(required).astype("int8")
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.loc[:, columns].copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _bounded_train(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic chronological thinning only; never samples held data."""
    if len(frame) <= MAX_TRAIN_ROWS:
        return frame
    ordered = frame.sort_values("event_ts", kind="stable")
    indices = np.linspace(0, len(ordered) - 1, MAX_TRAIN_ROWS, dtype=np.int64)
    return ordered.iloc[indices].copy()


def _regressor(*, quantile: bool = False) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile" if quantile else "regression_l1",
        alpha=.50,
        n_estimators=320 if quantile else 280,
        learning_rate=.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=160,
        subsample=.80,
        colsample_bytree=.85,
        reg_lambda=12.0,
        random_state=SEED,
        n_jobs=2,
        verbosity=-1,
    )


def _classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=160,
        subsample=.80,
        colsample_bytree=.85,
        reg_lambda=12.0,
        random_state=SEED,
        n_jobs=2,
        verbosity=-1,
    )


def _fit_models(train: pd.DataFrame, features: tuple[str, ...]) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor, lgb.LGBMClassifier, lgb.LGBMRegressor]:
    train = _bounded_train(train)
    x_train = _safe_numeric(train, features)
    prior = _regressor()
    conditional = _regressor()
    breaking = _classifier()
    magnitude = _regressor(quantile=True)
    weights = 1.0 + np.clip(pd.to_numeric(train.historical_ESS, errors="coerce").fillna(0.0).to_numpy(float) / 32.0, 0.0, 1.0)
    y_strength = pd.to_numeric(train.y_reaction_strength, errors="raise").to_numpy(float)
    y_break = pd.to_numeric(train.y_accepted_break, errors="raise").to_numpy(int)
    y_magnitude = pd.to_numeric(train.reaction_MFE_atr, errors="raise").to_numpy(float)
    prior.fit(x_train.loc[:, PRIOR_FEATURES], y_strength, sample_weight=weights)
    conditional.fit(x_train, y_strength, sample_weight=weights)
    breaking.fit(x_train, y_break, sample_weight=weights)
    magnitude.fit(x_train, y_magnitude, sample_weight=weights)
    return prior, conditional, breaking, magnitude


def _predict_models(models: tuple[lgb.LGBMRegressor, lgb.LGBMRegressor, lgb.LGBMClassifier, lgb.LGBMRegressor], score: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prior, conditional, breaking, magnitude = models
    x_score = _safe_numeric(score, features)
    return (
        prior.predict(x_score.loc[:, PRIOR_FEATURES]),
        conditional.predict(x_score),
        breaking.predict_proba(x_score)[:, 1],
        magnitude.predict(x_score),
    )


def _fit_predict(train: pd.DataFrame, score: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _predict_models(_fit_models(train, features), score, features)


def _zone_snapshot_rows(snapshots: pd.DataFrame, *, context_features: tuple[str, ...] = ()) -> pd.DataFrame:
    """Convert support/resistance snapshot columns to one model row per zone."""
    base = ["__symbol__", "snapshot_ts", "target_kind", "target_id", "candidate_id"]
    optional = [column for column in ("state_bar_15m",) if column in snapshots]
    frames: list[pd.DataFrame] = []
    for side in ("support", "resistance"):
        present = snapshots.loc[snapshots.get(f"{side}_available", False).fillna(False)].copy()
        if present.empty:
            continue
        result = present.loc[:, [*base, *optional]].copy()
        result["zone_side"] = side
        result["zone_id"] = present[f"{side}_zone_id"].astype(str)
        result["zone_distance_atr"] = pd.to_numeric(present[f"{side}_distance_atr"], errors="coerce")
        for feature in CONDITIONAL_FEATURES:
            source = f"{side}__{feature}"
            result[feature] = pd.to_numeric(present[source], errors="coerce") if source in present else np.nan
        for feature in context_features:
            result[feature] = pd.to_numeric(present[feature], errors="coerce") if feature in present else np.nan
        frames.append(result)
    if not frames:
        return pd.DataFrame(columns=[*base, *optional, "zone_side", "zone_id", "zone_distance_atr", *CONDITIONAL_FEATURES, *context_features])
    return pd.concat(frames, ignore_index=True)


def _fold_metrics(events: pd.DataFrame, held: pd.Timestamp) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    if events.empty:
        return result
    for column, target_column in (
        ("sr_prior_strength", "y_reaction_strength"),
        ("sr_conditional_strength", "y_reaction_strength"),
        # This head is fit to reaction_MFE_atr.  Comparing it to generic
        # reaction strength was a diagnostic-only metric mismatch, not a
        # training or OOF prediction error.
        ("sr_reaction_magnitude_q50", "reaction_MFE_atr"),
    ):
        target_strength = pd.to_numeric(events[target_column], errors="raise").to_numpy(float)
        pred = pd.to_numeric(events[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(pred) & np.isfinite(target_strength)
        result.append({
            "held_month": held.strftime("%Y-%m"), "head": column, "rows": int(valid.sum()),
            "mae": float(mean_absolute_error(target_strength[valid], pred[valid])) if valid.any() else np.nan,
            "spearman": float(pd.Series(target_strength[valid]).corr(pd.Series(pred[valid]), method="spearman")) if valid.sum() > 2 else np.nan,
        })
    pred = pd.to_numeric(events.sr_accepted_break_probability, errors="coerce").to_numpy(float)
    target = pd.to_numeric(events.y_accepted_break, errors="raise").to_numpy(int)
    valid = np.isfinite(pred)
    result.append({
        "held_month": held.strftime("%Y-%m"), "head": "sr_accepted_break_probability", "rows": int(valid.sum()),
        "auc": float(roc_auc_score(target[valid], pred[valid])) if valid.sum() > 2 and len(np.unique(target[valid])) == 2 else np.nan,
        "brier": float(brier_score_loss(target[valid], pred[valid])) if valid.any() else np.nan,
        "spearman": float(pd.Series(target[valid]).corr(pd.Series(pred[valid]), method="spearman")) if valid.sum() > 2 else np.nan,
    })
    return result


def _wide_snapshot_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["__symbol__", "snapshot_ts", "target_kind", "target_id", "candidate_id"]
    if "state_bar_15m" in rows:
        # Entry snapshots intentionally have no in-trade state-bar.  Pandas
        # pivot_table drops an index group containing NaN, which used to
        # silently erase every entry row.  A private, collision-free key
        # keeps the two snapshot populations separate without imputing a
        # model feature or changing the exported entry contract.
        rows = rows.copy()
        rows["__pivot_state_bar_15m"] = pd.to_numeric(rows["state_bar_15m"], errors="coerce").fillna(-1).astype("int16")
        keys.append("__pivot_state_bar_15m")
    prediction_cols = ["sr_prior_strength", "sr_conditional_strength", "sr_accepted_break_probability", "sr_reaction_magnitude_q50", "zone_distance_atr"]
    wide = rows.pivot_table(index=keys, columns="zone_side", values=prediction_cols, aggfunc="first").reset_index()
    wide.columns = ["_".join(part for part in column if part) if isinstance(column, tuple) else column for column in wide.columns]
    if "__pivot_state_bar_15m" in wide:
        wide["state_bar_15m"] = wide["__pivot_state_bar_15m"].where(wide["__pivot_state_bar_15m"].ge(0), np.nan)
        wide = wide.drop(columns="__pivot_state_bar_15m")
    rename = {}
    for head in prediction_cols:
        for side in ("support", "resistance"):
            source = f"{head}_{side}"
            if source in wide:
                rename[source] = f"sr_{side}_{head.removeprefix('sr_')}"
    wide = wide.rename(columns=rename)
    # Directional long-side summaries. Support rejection and resistance break
    # are favourable; their counterparts are headroom/risk diagnostics.
    s_hold = wide.get("sr_support_conditional_strength", pd.Series(np.nan, index=wide.index))
    r_break = wide.get("sr_resistance_accepted_break_probability", pd.Series(np.nan, index=wide.index))
    s_break = wide.get("sr_support_accepted_break_probability", pd.Series(np.nan, index=wide.index))
    r_react = wide.get("sr_resistance_conditional_strength", pd.Series(np.nan, index=wide.index))
    wide["sr_long_support_hold_strength"] = s_hold
    wide["sr_long_resistance_break_probability"] = r_break
    wide["sr_long_downside_break_probability"] = s_break
    wide["sr_long_resistance_rejection_strength"] = r_react
    wide["sr_long_structure_balance"] = (s_hold + r_break) - (s_break + r_react)
    wide["sr_long_support_distance_atr"] = wide.get("sr_support_zone_distance_atr", np.nan)
    wide["sr_long_resistance_distance_atr"] = wide.get("sr_resistance_zone_distance_atr", np.nan)
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; defaults Feb--Aug 2026")
    parser.add_argument(
        "--profile-state", type=Path,
        help="optional causal-profile materialisation root; states are merged backward/as-of into S/R source rows",
    )
    parser.add_argument(
        "--profile-context-group", choices=tuple(PROFILE_CONTEXT_GROUPS), default="all",
        help="causal profile feature family when --profile-state is present (default: all)",
    )
    parser.add_argument(
        "--conditional-feature-contract", choices=tuple(CONDITIONAL_FEATURE_CONTRACTS), default="full",
        help="retained S/R interaction-feature contract; use the removal arm only as a predeclared diagnostic",
    )
    parser.add_argument(
        "--frozen-train-end",
        help="optional UTC boundary.  When supplied, every held source head is fit only on labels resolved before it.",
    )
    parser.add_argument(
        "--reexport-from", type=Path,
        help="reuse immutable previously scored OOF event/zone predictions and only re-run the snapshot pivot/export",
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    if args.reexport_from is not None:
        previous = args.reexport_from.resolve()
        output.mkdir(parents=True, exist_ok=False)
        events = pd.read_parquet(previous / "interaction_head_oof_predictions.parquet")
        zone_predictions = pd.read_parquet(previous / "zone_snapshot_head_oof_predictions.parquet")
        wide = _wide_snapshot_predictions(zone_predictions)
        wide.loc[wide.target_kind.eq("entry")].to_parquet(output / "entry_sr_oof_features.parquet", index=False, compression="zstd")
        wide.loc[wide.target_kind.eq("continuation")].to_parquet(output / "continuation_sr_oof_features.parquet", index=False, compression="zstd")
        events.to_parquet(output / "interaction_head_oof_predictions.parquet", index=False, compression="zstd")
        zone_predictions.to_parquet(output / "zone_snapshot_head_oof_predictions.parquet", index=False, compression="zstd")
        wide.to_parquet(output / "snapshot_head_oof_features.parquet", index=False, compression="zstd")
        pd.read_parquet(previous / "head_metrics_by_month.parquet").to_parquet(output / "head_metrics_by_month.parquet", index=False)
        pd.read_parquet(previous / "fold_trace.parquet").to_parquet(output / "fold_trace.parquet", index=False)
        previous_manifest = json.loads((previous / "run_manifest.json").read_text(encoding="utf-8"))
        previous_manifest.update({
            "schema": "causal-sr-heads-oof-v1-reexport",
            "reexport_from": str(previous),
            "reexport_from_manifest_sha256": _sha256(previous / "run_manifest.json"),
            "reexport_reason": "corrected nullable entry state-bar pivot key; previously fitted OOF predictions are reused unchanged",
        })
        (output / "run_manifest.json").write_text(json.dumps(previous_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        return
    interactions = pd.read_parquet(source / "interaction_events.parquet")
    snapshots = pd.read_parquet(source / "sr_snapshots.parquet")
    interactions["event_ts"] = pd.to_datetime(interactions.event_ts, utc=True, errors="raise")
    interactions["label_available_ts"] = pd.to_datetime(interactions.label_available_ts, utc=True, errors="raise")
    snapshots["snapshot_ts"] = pd.to_datetime(snapshots.snapshot_ts, utc=True, errors="raise")
    profile_state_path: Path | None = None
    conditional_feature_contract = str(args.conditional_feature_contract)
    conditional_features = CONDITIONAL_FEATURE_CONTRACTS[conditional_feature_contract]
    context_features: tuple[str, ...] = ()
    profile_context_group: str | None = None
    profile_fields: tuple[str, ...] = ()
    if args.profile_state is not None:
        profile_context_group = str(args.profile_context_group)
        profile_fields = PROFILE_CONTEXT_GROUPS[profile_context_group]
        profile_state_path = args.profile_state.resolve() / "profile_hourly_states.parquet"
        if not profile_state_path.is_file():
            raise FileNotFoundError("--profile-state lacks profile_hourly_states.parquet")
        states = pd.read_parquet(profile_state_path)
        required_state = {"__symbol__", "state_ts", *profile_fields}
        missing_state = sorted(required_state.difference(states.columns))
        if missing_state:
            raise AssertionError(f"profile state contract missing {missing_state}")
        interactions = _merge_profile_context(interactions, states, timestamp="event_ts", fields=profile_fields)
        snapshots = _merge_profile_context(snapshots, states, timestamp="snapshot_ts", fields=profile_fields)
        context_features = (*profile_fields, PROFILE_CONTEXT_AVAILABLE)
    feature_columns = (*conditional_features, *context_features)
    missing = set(feature_columns).difference(interactions.columns)
    if missing:
        raise AssertionError(f"interaction contract missing {sorted(missing)}")
    rows = _zone_snapshot_rows(snapshots, context_features=context_features)
    held_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-02-01", "2026-08-01", freq="MS", tz="UTC"))
    output.mkdir(parents=True, exist_ok=False)
    event_frames: list[pd.DataFrame] = []
    snapshot_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    fold_trace: list[dict[str, Any]] = []
    frozen_train: pd.DataFrame | None = None
    frozen_models: tuple[lgb.LGBMRegressor, lgb.LGBMRegressor, lgb.LGBMClassifier, lgb.LGBMRegressor] | None = None
    if args.frozen_train_end:
        frozen_train_end = pd.Timestamp(args.frozen_train_end)
        frozen_train = interactions.loc[
            interactions.label_available_ts.lt(frozen_train_end) & interactions.event_ts.lt(frozen_train_end)
        ].copy()
        if len(frozen_train) < 2_000:
            raise RuntimeError(f"insufficient frozen S/R support before {frozen_train_end}: {len(frozen_train)}")
        frozen_models = _fit_models(frozen_train, feature_columns)
    for held in held_months:
        end = held + pd.offsets.MonthBegin(1)
        if args.frozen_train_end:
            if frozen_train is None or frozen_models is None:
                raise AssertionError("frozen source models were not initialised")
            train = frozen_train
        else:
            train = interactions.loc[interactions.label_available_ts.lt(held) & interactions.event_ts.lt(held)].copy()
        event_test = interactions.loc[interactions.event_ts.ge(held) & interactions.event_ts.lt(end) & interactions.label_available_ts.lt(end)].copy()
        snapshot_test = rows.loc[rows.snapshot_ts.ge(held) & rows.snapshot_ts.lt(end)].copy()
        if len(train) < 2_000 or event_test.empty or snapshot_test.empty:
            raise RuntimeError(f"insufficient strictly prior S/R support for {held:%Y-%m}: train={len(train)}, events={len(event_test)}, snapshots={len(snapshot_test)}")
        if frozen_models is None:
            e_pred = _fit_predict(train, event_test, feature_columns)
            s_pred = _fit_predict(train, snapshot_test, feature_columns)
        else:
            e_pred = _predict_models(frozen_models, event_test, feature_columns)
            s_pred = _predict_models(frozen_models, snapshot_test, feature_columns)
        for frame, predicted in ((event_test, e_pred), (snapshot_test, s_pred)):
            frame["sr_prior_strength"] = predicted[0]
            frame["sr_conditional_strength"] = predicted[1]
            frame["sr_accepted_break_probability"] = predicted[2]
            frame["sr_reaction_magnitude_q50"] = predicted[3]
            frame["held_month"] = held.strftime("%Y-%m")
        metrics.extend(_fold_metrics(event_test, held))
        event_frames.append(event_test)
        snapshot_frames.append(snapshot_test)
        fold_trace.append({"held_month": held.strftime("%Y-%m"), "train_rows": len(train), "event_test_rows": len(event_test), "snapshot_test_rows": len(snapshot_test), "train_label_max": str(train.label_available_ts.max())})
    events = pd.concat(event_frames, ignore_index=True)
    zone_predictions = pd.concat(snapshot_frames, ignore_index=True)
    wide = _wide_snapshot_predictions(zone_predictions)
    wide.loc[wide.target_kind.eq("entry")].to_parquet(output / "entry_sr_oof_features.parquet", index=False, compression="zstd")
    wide.loc[wide.target_kind.eq("continuation")].to_parquet(output / "continuation_sr_oof_features.parquet", index=False, compression="zstd")
    events.to_parquet(output / "interaction_head_oof_predictions.parquet", index=False, compression="zstd")
    zone_predictions.to_parquet(output / "zone_snapshot_head_oof_predictions.parquet", index=False, compression="zstd")
    wide.to_parquet(output / "snapshot_head_oof_features.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(output / "head_metrics_by_month.parquet", index=False)
    pd.DataFrame(fold_trace).to_parquet(output / "fold_trace.parquet", index=False)
    manifest = {
        "schema": "causal-sr-heads-oof-v1",
        "scope": "offline causal S/R research only; no live-model or execution mutation",
        "source": str(source), "source_manifest_sha256": _sha256(source / "run_manifest.json"),
        "folds": fold_trace,
        "heads": {
            "prior_strength": {"target": "y_reaction_strength", "features": list(PRIOR_FEATURES), "model": "LGBM L1 depth3 leaves7"},
            "conditional_strength": {"target": "y_reaction_strength", "features": list(feature_columns), "model": "LGBM L1 depth3 leaves7"},
            "accepted_break": {"target": "y_accepted_break", "features": list(feature_columns), "model": "LGBM binary depth3 leaves7"},
            "reaction_magnitude_q50": {"target": "reaction_MFE_atr", "features": list(feature_columns), "model": "LGBM median-quantile depth3 leaves7"},
        },
        "causality": "each held month trains only on interaction labels whose label_available_ts precedes its declared train boundary; snapshots are generated before any 8h future path; optional profile context is merged only from same-or-earlier completed states",
        "profile_context": None if profile_state_path is None else {
            "state_path": str(profile_state_path),
            "group": profile_context_group,
            "features": list(profile_fields),
            "availability": PROFILE_CONTEXT_AVAILABLE,
            "availability_min_fields": max(1, int(np.ceil(.70 * len(profile_fields)))),
        },
        "frozen_train_end": args.frozen_train_end,
        "conditional_feature_contract": {
            "name": conditional_feature_contract,
            "features": list(conditional_features),
            "removed_from_full": list(INTERACTION_PATH_FEATURES) if conditional_feature_contract == "without_interaction_path" else [],
        },
        "seed": SEED,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
