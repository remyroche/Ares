#!/usr/bin/env python3
"""Build causal hourly soft states and attach them independently to candidates.

This adapter is deliberately separate from the candidate-keyed materializer.
It first creates compact hourly OOF regime states from causal multiview inputs,
then creates a different supervised transition morphology/phase layer using
only labels resolved before each fold.  Both hourly timelines are backward
as-of joined to candidates with independent provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import (  # noqa: E402
    IDENTITY_COLUMNS,
    PROVENANCE_COLUMNS,
    STATE_OOD_COLUMN,
    STATE_PROBABILITY_PREFIX,
    TRANSITION_OOD_COLUMN,
    TRANSITION_PROBABILITY_PREFIX,
    TRANSITION_PROVENANCE_COLUMNS,
    RegimeOOFStackError,
    asof_join_regime_timeline,
    assert_outcome_free,
    derive_soft_state_fields,
    validate_regime_output_frame,
    validate_transition_output_frame,
    validate_candidate_identity,
)


SCHEMA = "candidate_oof_regime_transition_adapter_v1"
PHASES = (
    "stable",
    "approach",
    "immediate_lead",
    "transition",
    "acceleration",
    "early_destination",
    "settled_destination",
)
TARGET_COLUMNS = {"target__phase", "target__transition_active", "target__available_utc"}
IDENTIFIER_COLUMNS = {"source_utc", "execution_decision_utc", "segment_id", "calendar_segment_id", "source_segment_id"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise RegimeOOFStackError(f"{name} has invalid timestamps")
    return parsed


def _safe_hourly_features(frame: pd.DataFrame, *, max_features: int, train_mask: np.ndarray) -> list[str]:
    candidates = [
        column
        for column in frame.columns
        if column not in TARGET_COLUMNS
        and column not in IDENTIFIER_COLUMNS
        and not str(column).startswith(("target__", "source_artifact", "state_context__"))
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    safe = frame.loc[:, ["source_utc", *candidates]].copy()
    assert_outcome_free(safe.drop(columns="source_utc"))
    train = safe.loc[train_mask, candidates].apply(pd.to_numeric, errors="coerce")
    coverage, variance = train.notna().mean(), train.var(skipna=True)
    usable = [
        column
        for column in candidates
        if coverage.get(column, 0.0) >= 0.80 and np.isfinite(variance.get(column, np.nan)) and variance.get(column, 0.0) > 1e-12
    ]
    output = sorted(usable, key=lambda column: (-float(variance[column]), str(column)))[: int(max_features)]
    if len(output) < 2:
        raise RegimeOOFStackError("fewer than two causal hourly multiview features remain")
    return output


def _hour_blocks(source: pd.Series, *, start: pd.Timestamp, frequency: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    values = _utc(source, "source_utc")
    if frequency == "month":
        starts = sorted(pd.Timestamp(period.start_time, tz="UTC") for period in values.dt.to_period("M").unique())
        return [(item, item + pd.offsets.MonthBegin(1)) for item in starts if item + pd.offsets.MonthBegin(1) > start]
    if frequency == "week":
        naive = values.dt.tz_convert("UTC").dt.tz_localize(None)
        starts = sorted(pd.Timestamp(item, tz="UTC") for item in naive.dt.to_period("W-SUN").dt.start_time.unique())
        return [(item, item + pd.Timedelta(days=7)) for item in starts if item + pd.Timedelta(days=7) > start]
    raise RegimeOOFStackError("frequency must be 'week' or 'month'")


def _fit_regime_timeline(
    hourly: pd.DataFrame,
    *,
    start: pd.Timestamp,
    frequency: str,
    purge: pd.Timedelta,
    n_components: int,
    max_features: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for number, (fold_start, fold_end) in enumerate(_hour_blocks(hourly["source_utc"], start=start, frequency=frequency), start=1):
        evaluation = hourly["source_utc"].ge(max(fold_start, start)).to_numpy() & hourly["source_utc"].lt(fold_end).to_numpy()
        train = hourly["source_utc"].lt(fold_start - purge).to_numpy()
        if not evaluation.any():
            continue
        if int(train.sum()) < max(32, int(n_components) * 8):
            raise RegimeOOFStackError(f"regime fold {fold_start.isoformat()} lacks pre-block support")
        features = _safe_hourly_features(hourly, max_features=max_features, train_mask=train)
        imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
        train_x = scaler.fit_transform(imputer.fit_transform(hourly.loc[train, features]))
        eval_x = scaler.transform(imputer.transform(hourly.loc[evaluation, features]))
        model = GaussianMixture(n_components=int(n_components), covariance_type="diag", reg_covar=1e-5, random_state=seed + number).fit(train_x)
        order = np.argsort(model.means_[:, 0], kind="stable")
        probability = model.predict_proba(eval_x)[:, order]
        local = pd.DataFrame({"regime_source_utc": hourly.loc[evaluation, "source_utc"].to_numpy()}, index=hourly.index[evaluation])
        local["regime_fold_id"] = f"{frequency}_{number:03d}_{fold_start.strftime('%Y%m%d')}"
        local["regime_train_end_utc"] = hourly.loc[train, "source_utc"].max()
        local["regime_available_utc"] = local["regime_source_utc"]
        for component in range(int(n_components)):
            local[f"{STATE_PROBABILITY_PREFIX}{component}"] = probability[:, component]
        local[STATE_OOD_COLUMN] = np.maximum(0.0, -model.score_samples(eval_x))
        local = derive_soft_state_fields(local)
        outputs.append(local)
        folds.append({"fold_id": local["regime_fold_id"].iloc[0], "start": fold_start.isoformat(), "end": fold_end.isoformat(), "train_rows": int(train.sum()), "features": features})
    if not outputs:
        raise RegimeOOFStackError("no hourly regime OOF rows materialized")
    return pd.concat(outputs, ignore_index=True).sort_values("regime_source_utc", kind="stable"), folds


def _fit_transition_timeline(
    hourly: pd.DataFrame,
    *,
    start: pd.Timestamp,
    frequency: str,
    purge: pd.Timedelta,
    max_features: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    _required = {"target__phase", "target__transition_active", "target__available_utc"}
    if not _required.issubset(hourly.columns):
        raise RegimeOOFStackError("transition catalogue lacks phase, active, or label-availability fields")
    raw_available = pd.to_datetime(hourly["target__available_utc"], utc=True, errors="coerce")
    # Stable/no-event catalogue rows often have no explicit event-resolution
    # timestamp.  Use the declared maximum forward transition horizon as a
    # conservative floor rather than letting those labels train immediately.
    availability_floor = hourly["source_utc"] + pd.Timedelta(hours=12)
    available = raw_available.where(raw_available.notna(), availability_floor)
    available = pd.Series(
        np.maximum(raw_available.fillna(availability_floor).to_numpy(dtype="datetime64[ns]"), availability_floor.to_numpy(dtype="datetime64[ns]")),
        index=hourly.index,
    )
    available = pd.to_datetime(available, utc=True, errors="raise")
    outputs: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for number, (fold_start, fold_end) in enumerate(_hour_blocks(hourly["source_utc"], start=start, frequency=frequency), start=1):
        evaluation = hourly["source_utc"].ge(max(fold_start, start)).to_numpy() & hourly["source_utc"].lt(fold_end).to_numpy()
        train = hourly["source_utc"].lt(fold_start - purge).to_numpy() & available.lt(fold_start).to_numpy()
        if not evaluation.any():
            continue
        if int(train.sum()) < 32:
            raise RegimeOOFStackError(f"transition fold {fold_start.isoformat()} lacks label-resolved pre-block support")
        features = _safe_hourly_features(hourly, max_features=max_features, train_mask=train)
        imputer = SimpleImputer(strategy="median")
        train_x = imputer.fit_transform(hourly.loc[train, features])
        eval_x = imputer.transform(hourly.loc[evaluation, features])
        phase = hourly.loc[train, "target__phase"].astype(str).where(hourly.loc[train, "target__phase"].astype(str).isin(PHASES), "stable")
        # Deliberately compact: this is a probability context head, not the
        # trading alpha model, and it must be refit every calendar block.
        phase_model = HistGradientBoostingClassifier(max_iter=48, max_leaf_nodes=7, learning_rate=0.08, l2_regularization=1.0, random_state=seed + number)
        phase_model.fit(train_x, phase)
        phase_probability = np.zeros((int(evaluation.sum()), len(PHASES)), dtype=np.float32)
        prediction = phase_model.predict_proba(eval_x)
        for position, label in enumerate(phase_model.classes_.astype(str)):
            phase_probability[:, PHASES.index(label)] = prediction[:, position]
        active = pd.to_numeric(hourly.loc[train, "target__transition_active"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        if active.nunique() < 2:
            active_probability = np.full(int(evaluation.sum()), float(active.iloc[0]) if len(active) else 0.0, dtype=np.float32)
        else:
            active_model = HistGradientBoostingClassifier(max_iter=48, max_leaf_nodes=7, learning_rate=0.08, l2_regularization=1.0, random_state=seed + 10_000 + number)
            active_model.fit(train_x, active)
            active_probability = active_model.predict_proba(eval_x)[:, list(active_model.classes_).index(1)].astype(np.float32)
        local = pd.DataFrame({"transition_source_utc": hourly.loc[evaluation, "source_utc"].to_numpy()}, index=hourly.index[evaluation])
        local["transition_fold_id"] = f"{frequency}_{number:03d}_{fold_start.strftime('%Y%m%d')}"
        local["transition_train_end_utc"] = hourly.loc[train, "source_utc"].max()
        local["transition_available_utc"] = local["transition_source_utc"]
        for position, label in enumerate(PHASES):
            local[f"{TRANSITION_PROBABILITY_PREFIX}{label}"] = phase_probability[:, position]
        local["transition_active_probability"] = active_probability
        local[TRANSITION_OOD_COLUMN] = 0.0
        local = derive_soft_state_fields(local, probability_prefix=TRANSITION_PROBABILITY_PREFIX)
        outputs.append(local)
        folds.append({"fold_id": local["transition_fold_id"].iloc[0], "start": fold_start.isoformat(), "end": fold_end.isoformat(), "train_rows": int(train.sum()), "features": features, "label_resolution_rule": "target__available_utc < fold start"})
    if not outputs:
        raise RegimeOOFStackError("no hourly transition OOF rows materialized")
    return pd.concat(outputs, ignore_index=True).sort_values("transition_source_utc", kind="stable"), folds


def materialize_adapter(
    *,
    candidates_path: Path,
    hourly_regime_path: Path,
    hourly_transition_path: Path,
    output_dir: Path,
    evaluation_start: str,
    evaluation_end: str | None = None,
    frequency: str = "month",
    purge_hours: int = 12,
    n_components: int = 5,
    max_features: int = 32,
    max_lag_hours: int = 2,
    seed: int = 52,
) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        raise RegimeOOFStackError(f"refusing to overwrite existing output: {destination}")
    raw_candidates = pd.read_parquet(candidates_path)
    candidates = validate_candidate_identity(raw_candidates).loc[:, list(IDENTITY_COLUMNS)].copy()
    start = pd.to_datetime(evaluation_start, utc=True, errors="raise")
    candidates = candidates.loc[candidates["__ts__"].ge(start)].copy()
    end = pd.to_datetime(evaluation_end, utc=True, errors="raise") if evaluation_end else None
    if end is not None:
        candidates = candidates.loc[candidates["__ts__"].lt(end)].copy()
    if candidates.empty:
        raise RegimeOOFStackError("no candidates at or after evaluation_start")
    regime_hourly = pd.read_parquet(hourly_regime_path)
    transition_hourly = pd.read_parquet(hourly_transition_path)
    for name, frame in (("regime", regime_hourly), ("transition", transition_hourly)):
        if "source_utc" not in frame:
            raise RegimeOOFStackError(f"{name} hourly catalogue lacks source_utc")
        frame["source_utc"] = _utc(frame["source_utc"], f"{name}.source_utc")
    # Future hourly rows cannot contribute to any requested candidate as-of
    # state.  Trimming them here is both a causal guard and keeps a historical
    # candidate reconstruction from needlessly fitting 2025--26 folds.
    latest_candidate = candidates["__ts__"].max()
    regime_hourly = regime_hourly.loc[regime_hourly["source_utc"].le(latest_candidate)].copy()
    transition_hourly = transition_hourly.loc[transition_hourly["source_utc"].le(latest_candidate)].copy()
    regime_timeline, regime_folds = _fit_regime_timeline(regime_hourly, start=start, frequency=frequency, purge=pd.Timedelta(hours=purge_hours), n_components=n_components, max_features=max_features, seed=seed)
    transition_timeline, transition_folds = _fit_transition_timeline(transition_hourly, start=start, frequency=frequency, purge=pd.Timedelta(hours=purge_hours), max_features=max_features, seed=seed)
    joined = asof_join_regime_timeline(candidates, regime_timeline, by=(), timeline_timestamp_col="regime_source_utc", max_lag=pd.Timedelta(hours=max_lag_hours), provenance_columns=PROVENANCE_COLUMNS)
    joined = asof_join_regime_timeline(joined, transition_timeline, by=(), timeline_timestamp_col="transition_source_utc", max_lag=pd.Timedelta(hours=max_lag_hours), provenance_columns=TRANSITION_PROVENANCE_COLUMNS)
    validate_regime_output_frame(joined)
    validate_transition_output_frame(joined)
    destination.mkdir(parents=True)
    regime_timeline.to_parquet(destination / "hourly_regime_oof.parquet", index=False, compression="zstd")
    transition_timeline.to_parquet(destination / "hourly_transition_oof.parquet", index=False, compression="zstd")
    joined.to_parquet(destination / "candidate_oof_regime_transition.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "MATERIALIZED_CAUSAL_HOURLY_REGIME_AND_TRANSITION",
        "inputs": {"candidates": {"path": str(Path(candidates_path).resolve()), "sha256": _sha256(Path(candidates_path))}, "hourly_regime": {"path": str(Path(hourly_regime_path).resolve()), "sha256": _sha256(Path(hourly_regime_path))}, "hourly_transition": {"path": str(Path(hourly_transition_path).resolve()), "sha256": _sha256(Path(hourly_transition_path))}},
        "contract": {"asof": "backward only", "max_lag_hours": int(max_lag_hours), "regime_train": "pre-block only", "transition_train": "pre-block and target availability strictly before fold start", "transition_layer": "phase simplex plus independent active probability; never substituted from regime layer"},
        "coverage": {"candidate_rows": int(len(candidates)), "joined_rows": int(len(joined)), "exact_candidate_coverage": True, "evaluation_start_utc": start.isoformat(), "evaluation_end_exclusive_utc": end.isoformat() if end is not None else None},
        "folds": {"regime": regime_folds, "transition": transition_folds},
        "outputs": {name: _sha256(destination / name) for name in ("hourly_regime_oof.parquet", "hourly_transition_oof.parquet", "candidate_oof_regime_transition.parquet")},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--hourly-regime", type=Path, required=True)
    parser.add_argument("--hourly-transition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", help="Optional exclusive UTC end for a bounded historical materialization")
    parser.add_argument("--frequency", choices=("week", "month"), default="month")
    parser.add_argument("--purge-hours", type=int, default=12)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--max-lag-hours", type=int, default=2)
    parser.add_argument("--seed", type=int, default=52)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = materialize_adapter(candidates_path=args.candidates, hourly_regime_path=args.hourly_regime, hourly_transition_path=args.hourly_transition, output_dir=args.output_dir, evaluation_start=args.evaluation_start, evaluation_end=args.evaluation_end, frequency=args.frequency, purge_hours=args.purge_hours, n_components=args.n_components, max_features=args.max_features, max_lag_hours=args.max_lag_hours, seed=args.seed)
    print(json.dumps({"status": "ok", "output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
