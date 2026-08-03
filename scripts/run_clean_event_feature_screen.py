#!/usr/bin/env python3
"""Nested side-local feature screen for meaningful favorable-before-adverse MFE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    fit_train_only_isotonic_ev_mapping,
)
from scripts.run_capture_specific_feature_screen import (  # noqa: E402
    PREFIX,
    _numeric,
    feature_family,
)
from scripts.run_execution_ev_mixed_period_remedies import (  # noqa: E402
    ARCHETYPE_COLUMN,
    DECISION_COLUMN,
    DEFAULT_WINDOWS,
    IDENTITY_COLUMNS,
    RESOLUTION_COLUMN,
    SIDE_COLUMN,
    TARGET_COLUMN,
    _temporal_oof_blocks,
    build_forward_split,
)
from scripts.run_exact_policy_capture_hurdle_ablation import (  # noqa: E402
    _fit_or_constant_regressor,
    _predict_regressor,
)
from scripts.run_exact_policy_capture_support_ablation import (  # noqa: E402
    apply_recent_mapping_scope,
)

SCHEMA = "meaningful_mfe_clean_event_feature_screen_v2"
SIDES = ("long", "short")
ARMS = ("core_context", "all_256", "top_64", "top_128")
EVENT_TARGET = "target_clean_order_soft"
FAVORABLE = "favorable_first"
ADVERSE = "adverse_first"
TIMEOUT = "timeout"
SOFT_LABEL = "soft_label"
GRID_NAME = "h12_u1p5atr"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def add_clean_target(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    favorable = output[FAVORABLE].astype(bool).to_numpy()
    adverse = output[ADVERSE].astype(bool).to_numpy()
    timeout = (
        output[TIMEOUT].astype(bool).to_numpy()
        if TIMEOUT in output
        else ~(favorable | adverse)
    )
    outcome_count = (
        favorable.astype(np.int8)
        + adverse.astype(np.int8)
        + timeout.astype(np.int8)
    )
    if (outcome_count != 1).any():
        raise ValueError("meaningful-MFE outcomes must be mutually exclusive")
    if SOFT_LABEL in output:
        soft = pd.to_numeric(output[SOFT_LABEL], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(soft).all() or ((soft < 0.0) | (soft > 1.0)).any():
            raise ValueError("soft meaningful-MFE labels must lie in [0, 1]")
    else:
        soft = np.where(favorable, 1.0, np.where(adverse, 0.0, 0.5))
    output[EVENT_TARGET] = soft.astype(np.float32)
    # The hard event question is favorable-first versus every other valid
    # outcome; timeout is a valid negative, not a missing label.
    output["clean_order_resolved"] = True
    return output


def _segment_statistics(
    values: np.ndarray,
    event: np.ndarray,
    net: np.ndarray,
) -> tuple[float, float, float]:
    finite = np.isfinite(values) & np.isfinite(event) & np.isfinite(net)
    if finite.sum() < 200 or np.unique(values[finite]).size < 2:
        return np.nan, np.nan, np.nan
    ic = pd.Series(values[finite]).corr(
        pd.Series(event[finite]), method="spearman"
    )
    if not np.isfinite(ic) or abs(ic) < 1e-12:
        return float(ic), np.nan, np.nan
    positions = np.flatnonzero(finite)
    count = max(1, int(np.ceil(0.10 * len(positions))))
    selected = positions[
        np.argsort(-np.sign(ic) * values[positions], kind="mergesort")[:count]
    ]
    return (
        float(ic),
        float(event[selected].mean() - event[finite].mean()),
        float((net[selected].mean() - net[finite].mean()) * 10_000.0),
    )


def select_clean_features(
    fit: pd.DataFrame,
    candidate_columns: Sequence[str],
    *,
    max_features: int,
    minimum_coverage: float,
    maximum_per_family: int,
    correlation_cap: float,
) -> tuple[list[str], dict[str, Any]]:
    decision = pd.to_datetime(fit[DECISION_COLUMN], utc=True, errors="raise")
    unique_days = pd.Index(decision.dt.floor("D").unique()).sort_values()
    if len(unique_days) < 4:
        raise ValueError("clean feature selection requires at least four days")
    boundary = unique_days[len(unique_days) // 2]
    masks = (
        decision.dt.floor("D").lt(boundary).to_numpy(),
        decision.dt.floor("D").ge(boundary).to_numpy(),
    )
    event = fit[EVENT_TARGET].to_numpy(dtype=float)
    net = fit[TARGET_COLUMN].to_numpy(dtype=float)
    matrix = _numeric(fit, candidate_columns)
    records = []
    for column in candidate_columns:
        values = matrix[column].to_numpy(dtype=float)
        coverage = float(np.isfinite(values).mean())
        variance = float(np.nanvar(values))
        segments = [
            _segment_statistics(values[mask], event[mask], net[mask])
            for mask in masks
        ]
        ics = [item[0] for item in segments]
        event_lifts = [item[1] for item in segments]
        net_lifts = [item[2] for item in segments]
        stable_sign = (
            all(np.isfinite(item) for item in ics) and ics[0] * ics[1] > 0.0
        )
        robust_ic = min(abs(item) for item in ics) if stable_sign else 0.0
        robust_event_lift = (
            min(event_lifts)
            if all(np.isfinite(item) for item in event_lifts)
            else -np.inf
        )
        robust_net_lift = (
            min(net_lifts)
            if all(np.isfinite(item) for item in net_lifts)
            else -np.inf
        )
        eligible = (
            coverage >= float(minimum_coverage)
            and np.isfinite(variance)
            and variance > 1e-10
            and stable_sign
            and robust_event_lift > 0.0
        )
        records.append(
            {
                "feature": column,
                "family": feature_family(column),
                "coverage": coverage,
                "variance": variance,
                "early_event_ic": ics[0],
                "late_event_ic": ics[1],
                "early_event_lift": event_lifts[0],
                "late_event_lift": event_lifts[1],
                "early_net_lift_bps": net_lifts[0],
                "late_net_lift_bps": net_lifts[1],
                "robust_event_ic": robust_ic,
                "robust_event_lift": robust_event_lift,
                "robust_net_lift_bps": robust_net_lift,
                "selection_score": (
                    100.0 * robust_event_lift
                    + 50.0 * robust_ic
                    if eligible
                    else -np.inf
                ),
                "eligible": bool(eligible),
            }
        )
    ranking = pd.DataFrame(records).sort_values(
        ["eligible", "selection_score", "feature"],
        ascending=[False, False, True],
        kind="stable",
    )
    filled = matrix.copy()
    for column in candidate_columns:
        filled[column] = filled[column].fillna(float(filled[column].median()))
    selected: list[str] = []
    family_counts: dict[str, int] = {}
    for record in ranking.loc[ranking["eligible"]].itertuples(index=False):
        if len(selected) >= int(max_features):
            break
        column = str(record.feature)
        family = str(record.family)
        if family_counts.get(family, 0) >= int(maximum_per_family):
            continue
        if any(
            np.isfinite(correlation := filled[column].corr(filled[existing]))
            and abs(correlation) >= float(correlation_cap)
            for existing in selected
        ):
            continue
        selected.append(column)
        family_counts[family] = family_counts.get(family, 0) + 1
    return selected, {
        "status": "selected" if selected else "no_stable_clean_event_features",
        "fit_rows": int(len(fit)),
        "fit_start_utc": decision.min(),
        "fit_end_utc": decision.max(),
        "segment_boundary_utc": pd.Timestamp(boundary),
        "candidate_features": int(len(candidate_columns)),
        "eligible_features": int(ranking["eligible"].sum()),
        "selected_features": selected,
        "selected_count": int(len(selected)),
        "family_counts": family_counts,
        "ranking": ranking.to_dict(orient="records"),
    }


def _fit_clean(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    columns: Sequence[str],
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    model, constant = _fit_or_constant_regressor(
        _numeric(fit, columns),
        fit[EVENT_TARGET].to_numpy(dtype=float),
        iterations=iterations,
        seed=seed,
        n_jobs=n_jobs,
    )
    return np.clip(
        _predict_regressor(model, constant, _numeric(score, columns)),
        0.0,
        1.0,
    )


def _metric_row(
    evaluation: pd.DataFrame,
    score: np.ndarray,
    *,
    window: str,
    arm: str,
    stage: str,
    scope: str,
) -> dict[str, Any]:
    mask = (
        np.ones(len(evaluation), dtype=bool)
        if scope == "pooled_global"
        else evaluation[SIDE_COLUMN]
        .astype(str)
        .eq(scope.removeprefix("side_"))
        .to_numpy()
    )
    sample = evaluation.loc[mask].reset_index(drop=True)
    prediction = np.asarray(score, dtype=float)[mask]
    resolved = sample["clean_order_resolved"].to_numpy(dtype=bool)
    favorable = sample[FAVORABLE].to_numpy(dtype=np.int8)
    auc = (
        float(roc_auc_score(favorable[resolved], prediction[resolved]))
        if resolved.sum() and np.unique(favorable[resolved]).size == 2
        else np.nan
    )
    count = max(1, int(np.ceil(0.10 * len(sample))))
    selected_pos = np.argsort(-prediction, kind="mergesort")[:count]
    selected = sample.iloc[selected_pos]
    selected_decision = pd.to_datetime(
        selected[DECISION_COLUMN], utc=True, errors="raise"
    )
    latest_start = pd.to_datetime(
        sample[DECISION_COLUMN], utc=True, errors="raise"
    ).max() - pd.Timedelta(days=7)
    latest = selected.loc[selected_decision.ge(latest_start)]
    return {
        "window": window,
        "arm": arm,
        "stage": stage,
        "scope": scope,
        "rows": int(len(sample)),
        "resolved_rows": int(resolved.sum()),
        "resolved_auc": auc,
        "event_spearman": float(
            pd.Series(prediction).corr(
                pd.Series(sample[EVENT_TARGET].to_numpy(dtype=float)),
                method="spearman",
            )
        ),
        "top10_rows": int(count),
        "top10_net_bps": float(selected[TARGET_COLUMN].mean() * 10_000.0),
        "top10_clean_soft_mean": float(selected[EVENT_TARGET].mean()),
        "top10_favorable_rate": float(selected[FAVORABLE].mean()),
        "top10_adverse_rate": float(selected[ADVERSE].mean()),
        "top10_timeout_rate": float(selected[TIMEOUT].mean()),
        "latest_7d_candidate_rows": int(
            pd.to_datetime(
                sample[DECISION_COLUMN], utc=True, errors="raise"
            ).ge(latest_start).sum()
        ),
        "latest_7d_selected_rows": int(len(latest)),
        "latest_7d_net_bps": (
            float(latest[TARGET_COLUMN].mean() * 10_000.0)
            if len(latest)
            else np.nan
        ),
    }


def evaluate_promotion_gate(
    metrics: pd.DataFrame,
    *,
    minimum_latest_selected_rows: int,
) -> dict[str, Any]:
    primary = metrics.loc[
        metrics["stage"].eq("causal_global_recent_mapping")
        & metrics["scope"].eq("pooled_global")
    ].copy()
    windows = list(dict.fromkeys(primary["window"].astype(str)))
    baseline = primary.loc[primary["arm"].eq("all_256")].set_index("window")
    if set(windows) != set(baseline.index):
        raise ValueError("promotion gate is missing an all_256 baseline window")
    challengers: dict[str, Any] = {}
    for arm in ("top_64", "top_128"):
        arm_rows = primary.loc[primary["arm"].eq(arm)].set_index("window")
        checks = []
        for window in windows:
            candidate = arm_rows.loc[window]
            reference = baseline.loc[window]
            delta = float(candidate["top10_net_bps"] - reference["top10_net_bps"])
            checks.append(
                {
                    "window": window,
                    "top10_net_bps": float(candidate["top10_net_bps"]),
                    "all_256_top10_net_bps": float(reference["top10_net_bps"]),
                    "delta_vs_all_256_bps": delta,
                    "latest_7d_candidate_rows": int(
                        candidate["latest_7d_candidate_rows"]
                    ),
                    "latest_7d_selected_rows": int(
                        candidate["latest_7d_selected_rows"]
                    ),
                    "positive_exact_net": bool(candidate["top10_net_bps"] > 0.0),
                    "improves_all_256": bool(delta > 0.0),
                    "adequate_latest_coverage": bool(
                        candidate["latest_7d_selected_rows"]
                        >= int(minimum_latest_selected_rows)
                    ),
                }
            )
        eligible = all(
            row["positive_exact_net"]
            and row["improves_all_256"]
            and row["adequate_latest_coverage"]
            for row in checks
        )
        challengers[arm] = {
            "eligible_for_mda_hpo": bool(eligible),
            "checks": checks,
        }
    eligible_arms = [
        arm
        for arm, payload in challengers.items()
        if payload["eligible_for_mda_hpo"]
    ]
    return {
        "primary_stage": "causal_global_recent_mapping",
        "primary_scope": "pooled_global",
        "baseline_arm": "all_256",
        "minimum_latest_selected_rows": int(minimum_latest_selected_rows),
        "challengers": challengers,
        "eligible_arms": eligible_arms,
        "launch_mda_hpo": bool(eligible_arms),
    }


def _load_frame(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_parquet(args.input)
    labels = pd.read_parquet(args.clean_labels)
    required_labels = [
        *IDENTITY_COLUMNS,
        "grid_name",
        "label_valid",
        "label_resolution_utc",
        FAVORABLE,
        ADVERSE,
        TIMEOUT,
        SOFT_LABEL,
    ]
    missing_labels = sorted(set(required_labels) - set(labels))
    if missing_labels:
        raise ValueError(
            "meaningful-MFE label grid is missing columns: "
            + ", ".join(missing_labels)
        )
    labels = labels.loc[
        labels["grid_name"].astype(str).eq(args.grid_name)
        & labels["label_valid"].astype(bool),
        required_labels,
    ].copy()
    for name, source in (("input", frame), ("clean labels", labels)):
        if source.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
            raise ValueError(f"{name} contains duplicate exact identities")
    joined = frame.merge(
        labels,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("meaningful-MFE labels do not cover every feature-universe row")
    joined[RESOLUTION_COLUMN] = pd.to_datetime(
        joined["label_resolution_utc"], utc=True, errors="raise"
    )
    return add_clean_target(joined.drop(columns="_merge"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame = _load_frame(args)
    universe = json.loads(args.feature_universe_manifest.read_text())
    # This 256-field roster is frozen by the outcome-free loader contract.
    # Coverage eligibility for the selected arms is recomputed inside each fit.
    candidate_columns = list(universe["candidate_feature_columns"])
    core_manifest = json.loads(args.core_feature_manifest.read_text())
    core_columns = list(core_manifest["feature_columns"])
    for column in core_columns:
        prefix = "catboost_archetype__"
        if column.startswith(prefix) and column not in frame:
            frame[column] = (
                frame[ARCHETYPE_COLUMN]
                .astype(str)
                .eq(column[len(prefix) :])
                .astype(np.float32)
            )
    required = [
        *IDENTITY_COLUMNS,
        DECISION_COLUMN,
        RESOLUTION_COLUMN,
        TARGET_COLUMN,
        SIDE_COLUMN,
        EVENT_TARGET,
        *core_columns,
        *candidate_columns,
    ]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError("clean feature screen missing columns: " + ", ".join(missing))
    frame = frame.sort_values(
        [DECISION_COLUMN, "candidate_id"], kind="stable"
    ).reset_index(drop=True)

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for window_index, window in enumerate(DEFAULT_WINDOWS):
        train_pos, evaluation_pos, split = build_forward_split(
            frame, window, purge_hours=args.purge_hours
        )
        train = frame.iloc[train_pos].copy().reset_index(drop=True)
        evaluation = frame.iloc[evaluation_pos].copy().reset_index(drop=True)
        train_scores = {arm: np.full(len(train), np.nan) for arm in ARMS}
        raw_eval_scores = {arm: np.full(len(evaluation), np.nan) for arm in ARMS}
        mapped_eval_scores = {arm: np.full(len(evaluation), np.nan) for arm in ARMS}
        window_report: dict[str, Any] = {"split": split, "sides": {}}
        for side_index, side in enumerate(SIDES):
            train_side = train.loc[train[SIDE_COLUMN].astype(str).eq(side)].copy()
            train_side["__global_position__"] = train_side.index
            train_side = train_side.reset_index(drop=True)
            eval_side = evaluation.loc[
                evaluation[SIDE_COLUMN].astype(str).eq(side)
            ].copy()
            eval_side["__global_position__"] = eval_side.index
            eval_side = eval_side.reset_index(drop=True)
            oof = {arm: np.full(len(train_side), np.nan) for arm in ARMS}
            fold_reports = []
            for fold, (fit_pos, valid_pos) in enumerate(
                _temporal_oof_blocks(train_side, min_train_rows=2_000), start=1
            ):
                fit = train_side.iloc[fit_pos]
                valid = train_side.iloc[valid_pos]
                selected, selection = select_clean_features(
                    fit,
                    candidate_columns,
                    max_features=max(128, args.max_selected_features),
                    minimum_coverage=args.minimum_train_coverage,
                    maximum_per_family=args.maximum_per_family,
                    correlation_cap=args.correlation_cap,
                )
                for arm, columns, offset in (
                    ("core_context", core_columns, 0),
                    ("all_256", candidate_columns, 1),
                    ("top_64", selected[:64], 2),
                    ("top_128", selected[:128], 3),
                ):
                    if not columns:
                        continue
                    oof[arm][valid_pos] = _fit_clean(
                        fit,
                        valid,
                        columns,
                        iterations=args.n_estimators,
                        seed=args.random_state
                        + 100_000 * window_index
                        + 10_000 * side_index
                        + 100 * fold
                        + offset,
                        n_jobs=args.n_jobs,
                    )
                for record in selection["ranking"]:
                    selection_rows.append(
                        {
                            "window": window.name,
                            "side_name": side,
                            "selection_stage": f"oof_fold_{fold}",
                            **record,
                        }
                    )
                fold_reports.append(
                    {
                        "fold": fold,
                        "fit_rows": int(len(fit)),
                        "validation_rows": int(len(valid)),
                        "selection": {
                            key: value
                            for key, value in selection.items()
                            if key != "ranking"
                        },
                    }
                )
            selected_final, final_selection = select_clean_features(
                train_side,
                candidate_columns,
                max_features=max(128, args.max_selected_features),
                minimum_coverage=args.minimum_train_coverage,
                maximum_per_family=args.maximum_per_family,
                correlation_cap=args.correlation_cap,
            )
            raw_eval = {}
            for arm, columns, offset in (
                ("core_context", core_columns, 0),
                ("all_256", candidate_columns, 1),
                ("top_64", selected_final[:64], 2),
                ("top_128", selected_final[:128], 3),
            ):
                if not columns:
                    continue
                raw_eval[arm] = _fit_clean(
                    train_side,
                    eval_side,
                    columns,
                    iterations=args.n_estimators,
                    seed=args.random_state
                    + 100_000 * window_index
                    + 10_000 * side_index
                    + 9_000
                    + offset,
                    n_jobs=args.n_jobs,
                )
            for record in final_selection["ranking"]:
                selection_rows.append(
                    {
                        "window": window.name,
                        "side_name": side,
                        "selection_stage": "final_train",
                        **record,
                    }
                )
            global_train = train_side["__global_position__"].to_numpy(dtype=int)
            global_eval = eval_side["__global_position__"].to_numpy(dtype=int)
            for arm in ARMS:
                if arm not in raw_eval or not np.isfinite(oof[arm]).any():
                    continue
                mapper = fit_train_only_isotonic_ev_mapping(
                    oof[arm],
                    train_side[TARGET_COLUMN].to_numpy(dtype=float),
                    min_rows=24,
                )
                finite = np.isfinite(oof[arm])
                mapped_oof = np.full(len(train_side), np.nan)
                mapped_oof[finite] = mapper.predict(oof[arm][finite])
                train_scores[arm][global_train] = mapped_oof
                raw_eval_scores[arm][global_eval] = raw_eval[arm]
                mapped_eval_scores[arm][global_eval] = mapper.predict(raw_eval[arm])
            window_report["sides"][side] = {
                "train_rows": int(len(train_side)),
                "evaluation_rows": int(len(eval_side)),
                "folds": fold_reports,
                "final_selection": {
                    key: value
                    for key, value in final_selection.items()
                    if key != "ranking"
                },
            }
        for arm in ARMS:
            if not np.isfinite(raw_eval_scores[arm]).all():
                raise RuntimeError(
                    f"{window.name}/{arm} did not score every evaluation row"
                )
            for scope in ("pooled_global", "side_long", "side_short"):
                metrics.append(
                    _metric_row(
                        evaluation,
                        raw_eval_scores[arm],
                        window=window.name,
                        arm=arm,
                        stage="raw_clean_score",
                        scope=scope,
                    )
                )
                metrics.append(
                    _metric_row(
                        evaluation,
                        mapped_eval_scores[arm],
                        window=window.name,
                        arm=arm,
                        stage="pre_recent_mapping",
                        scope=scope,
                    )
                )
            recent, mapping_report = apply_recent_mapping_scope(
                train,
                evaluation,
                train_scores[arm],
                mapped_eval_scores[arm],
                scope="global",
            )
            for scope in ("pooled_global", "side_long", "side_short"):
                metrics.append(
                    _metric_row(
                        evaluation,
                        recent,
                        window=window.name,
                        arm=arm,
                        stage="causal_global_recent_mapping",
                        scope=scope,
                    )
                )
            part = evaluation.loc[:, list(IDENTITY_COLUMNS)].copy()
            part["window"] = window.name
            part["arm"] = arm
            part["raw_clean_score"] = raw_eval_scores[arm]
            part["pre_recent_ev_score"] = mapped_eval_scores[arm]
            part["causal_global_recent_ev_score"] = recent
            predictions.append(part)
            window_report.setdefault("mapping", {})[arm] = mapping_report
        reports[window.name] = window_report

    metric_frame = pd.DataFrame(metrics)
    promotion_gate = evaluate_promotion_gate(
        metric_frame,
        minimum_latest_selected_rows=args.minimum_latest_selected_rows,
    )
    args.output_dir.mkdir(parents=True)
    paths = {
        "metrics": args.output_dir / "clean_feature_screen_metrics.csv",
        "predictions": args.output_dir / "clean_feature_screen_predictions.parquet",
        "selection": args.output_dir / "clean_feature_selection.csv",
    }
    metric_frame.to_csv(paths["metrics"], index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        paths["predictions"], index=False, compression="zstd"
    )
    pd.DataFrame(selection_rows).to_csv(paths["selection"], index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_research_oos_not_promotion_evidence",
        "contract": {
            "target": (
                "fixed h12_u1p5atr meaningful-MFE label: favorable barrier "
                "max(1.5 ATR, 1.5%) before 1.0 ATR adverse; same-hour conflict "
                "is adverse; train on the canonical soft label"
            ),
            "selection": (
                "inside every side-local temporal fit; stable signed hard-event "
                "IC and clean-event top-decile lift in both fit halves; no "
                "evaluation outcomes and no label-geometry search"
            ),
            "models": "fixed CatBoost soft clean-order regressors; no HPO",
            "mapping": (
                "side-local train-OOF isotonic exact-net map followed by causal "
                "global 21-day recent-EV correction"
            ),
            "ranking": "one pooled global top 10%; no timestamp or side quotas",
            "promotion_gate": (
                "top64/top128 must each be assessed against all256; MDA/HPO is "
                "eligible only for an arm with positive exact net, positive "
                "delta, and adequate latest coverage in both June and later July"
            ),
        },
        "inputs": {
            "data": {"path": str(args.input), "sha256": _sha256(args.input)},
            "clean_labels": {
                "path": str(args.clean_labels),
                "sha256": _sha256(args.clean_labels),
            },
            "feature_universe_manifest": {
                "path": str(args.feature_universe_manifest),
                "sha256": _sha256(args.feature_universe_manifest),
            },
            "core_feature_manifest": {
                "path": str(args.core_feature_manifest),
                "sha256": _sha256(args.core_feature_manifest),
            },
        },
        "selector_parameters": {
            "candidate_features": int(len(candidate_columns)),
            "max_selected_features": int(args.max_selected_features),
            "minimum_train_coverage": float(args.minimum_train_coverage),
            "maximum_per_family": int(args.maximum_per_family),
            "correlation_cap": float(args.correlation_cap),
        },
        "model": {
            "iterations": int(args.n_estimators),
            "n_jobs": int(args.n_jobs),
            "random_state": int(args.random_state),
        },
        "promotion_gate": promotion_gate,
        "windows": reports,
        "outputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet"
        ),
    )
    parser.add_argument(
        "--clean-labels",
        type=Path,
        default=Path(
            "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/meaningful_mfe_label_grid.parquet"
        ),
    )
    parser.add_argument("--grid-name", default=GRID_NAME, choices=[GRID_NAME])
    parser.add_argument(
        "--feature-universe-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/feature_universe_manifest.json"
        ),
    )
    parser.add_argument(
        "--core-feature-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_context_clean_regime_diagnosis_forward_july19_20260726_v1/regime_diagnosis_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260727)
    parser.add_argument("--max-selected-features", type=int, default=128)
    parser.add_argument("--minimum-train-coverage", type=float, default=0.99)
    parser.add_argument("--maximum-per-family", type=int, default=16)
    parser.add_argument("--correlation-cap", type=float, default=0.95)
    parser.add_argument("--minimum-latest-selected-rows", type=int, default=25)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
