#!/usr/bin/env python3
"""Run the leakage-safe 12-hour execution-EV meta-head on a joined handoff.

This entry point intentionally defaults to a bounded smoke run.  It trains the
direct and residual side-aware ablations implemented in
``extreme_price_movements.execution_ev_meta`` and writes only OOF evaluation
evidence plus a final-fit scoring bundle.  It does not select or replay a
policy.

The provenance JSON is an explicit contract, for example::

  {
    "schema": "execution_ev_joined_handoff_v2",
    "handoff": {
      "join_mode": "exact_inner_one_to_one",
      "join_keys": ["__ts__", "__symbol__", "side_name"],
      "source_artifacts": {"alpha": "...", "execution_labels": "..."}
    },
    "features": {
      "existing_alpha_ev": {
        "family": "alpha_score", "source": "frozen alpha EV",
        "pre_entry": true, "oof_or_frozen": true,
        "available_at_col": "alpha_available_at", "model_input": true
      },
      "catboost_archetype": {
        "family": "predicted_path_archetype", "source": "frozen CatBoost path classifier",
        "pre_entry": true, "oof_or_frozen": true,
        "available_at_col": "catboost_available_at", "model_input": false
      }
    }
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import (  # noqa: E402
    ExecutionEVTrainerConfig,
    FeatureProvenance,
    save_execution_ev_bundle,
    train_execution_ev_meta,
    validate_execution_ev_training_contract,
    write_execution_ev_report,
)
from extreme_price_movements.execution_timing_risk_meta import (  # noqa: E402
    ExecutionTimingRiskTargetSpec,
    TimingRiskTrainerConfig,
    save_execution_timing_risk_bundle,
    train_execution_timing_risk_meta,
    write_execution_timing_risk_report,
)

HANDOFF_SCHEMA = "execution_ev_joined_handoff_v2"
DEFAULT_ID_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")
REQUIRED_TARGET_COLUMNS = ("execution_net_ev_12h", "existing_alpha_ev")
REQUIRED_EXECUTION_AUDIT_COLUMNS = (
    "execution_decision_utc",
    "execution_label_end_utc",
    "execution_gross_ev_12h",
    "execution_cost_return",
    "existing_alpha_ev_source_basis",
    "alpha_source_cost_return",
)
REQUIRED_FAMILIES = (
    "peak_mfe",
    "catboost_probabilities",
    "catboost_entropy",
    "prediction_uncertainty",
    "leaf_support",
    "alpha_score",
    "base_archetype_labels",
)


def _parse_columns(value: str) -> list[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns:
        raise argparse.ArgumentTypeError("at least one column is required")
    if len(set(columns)) != len(columns):
        raise argparse.ArgumentTypeError("column names must be unique")
    return columns


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - pandas parquet dependency
        raise RuntimeError(
            "pyarrow is required to preflight the smoke row cap"
        ) from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


def _load_provenance(path: Path) -> tuple[dict[str, FeatureProvenance], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid provenance JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("provenance JSON must be an object")
    if payload.get("schema") != HANDOFF_SCHEMA:
        raise ValueError(f"provenance schema must be {HANDOFF_SCHEMA!r}")
    handoff = payload.get("handoff")
    features = payload.get("features")
    if not isinstance(handoff, dict) or not isinstance(features, dict) or not features:
        raise ValueError("provenance requires non-empty handoff and features objects")
    if handoff.get("join_mode") != "exact_inner_one_to_one":
        raise ValueError("handoff.join_mode must be exact_inner_one_to_one")
    if not isinstance(handoff.get("join_keys"), list) or not handoff["join_keys"]:
        raise ValueError("handoff.join_keys must be a non-empty list")
    sources = handoff.get("source_artifacts")
    if not isinstance(sources, (dict, list)) or not sources:
        raise ValueError("handoff.source_artifacts must identify the joined sources")

    parsed: dict[str, FeatureProvenance] = {}
    for column, raw in features.items():
        if not isinstance(column, str) or not isinstance(raw, dict):
            raise ValueError("provenance.features must map column names to objects")
        if not isinstance(raw.get("family"), str) or not raw["family"].strip():
            raise ValueError(f"feature {column!r} requires a non-empty family")
        if not isinstance(raw.get("source"), str) or not raw["source"].strip():
            raise ValueError(f"feature {column!r} requires a non-empty source")
        if not isinstance(raw.get("pre_entry"), bool) or not isinstance(
            raw.get("oof_or_frozen"), bool
        ):
            raise ValueError(
                f"feature {column!r} must declare boolean pre_entry and oof_or_frozen"
            )
        if "model_input" in raw and not isinstance(raw["model_input"], bool):
            raise ValueError(
                f"feature {column!r} model_input must be boolean when supplied"
            )
        if not raw.get("available_at_col"):
            raise ValueError(
                f"strict provenance requires available_at_col for feature {column!r}"
            )
        try:
            parsed[column] = FeatureProvenance(
                family=str(raw["family"]),
                source=str(raw["source"]),
                pre_entry=bool(raw["pre_entry"]),
                available_at_col=str(raw["available_at_col"]),
                oof_or_frozen=bool(raw["oof_or_frozen"]),
                model_input=bool(raw.get("model_input", True)),
                class_order=(
                    tuple(str(value) for value in raw["class_order"])
                    if raw.get("class_order") is not None
                    else None
                ),
                class_order_sha256=(
                    str(raw["class_order_sha256"])
                    if raw.get("class_order_sha256") is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise ValueError(
                f"feature {column!r} is missing provenance field {exc.args[0]!r}"
            ) from exc
    return parsed, payload


def _utc(values: pd.Series, *, column: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"{column!r} contains null or invalid timestamps")
    return converted


def _validate_handoff(
    frame: pd.DataFrame,
    *,
    provenance: Mapping[str, FeatureProvenance],
    provenance_payload: Mapping[str, Any],
    id_columns: Sequence[str],
    timestamp_col: str,
    side_col: str,
    archetype_col: str,
    label_end_time_col: str | None,
    max_span_days: float,
) -> pd.DataFrame:
    if side_col not in id_columns:
        raise ValueError("joined handoff identity must include the side column")
    required = list(
        dict.fromkeys(
            [
                *id_columns,
                timestamp_col,
                side_col,
                archetype_col,
                *REQUIRED_TARGET_COLUMNS,
                *REQUIRED_EXECUTION_AUDIT_COLUMNS,
                *provenance.keys(),
                *[
                    spec.available_at_col
                    for spec in provenance.values()
                    if spec.available_at_col
                ],
            ]
        )
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "joined handoff is missing required columns: " + ", ".join(missing)
        )
    declared_keys = list(provenance_payload["handoff"]["join_keys"])
    if list(id_columns) != declared_keys:
        raise ValueError(
            "--id-cols must exactly match provenance handoff.join_keys; "
            f"got {list(id_columns)!r}, declared {declared_keys!r}"
        )
    work = frame.copy()
    work[timestamp_col] = _utc(work[timestamp_col], column=timestamp_col)
    signal_timestamp_col = declared_keys[0]
    if signal_timestamp_col != timestamp_col:
        work[signal_timestamp_col] = _utc(
            work[signal_timestamp_col], column=signal_timestamp_col
        )
    if work[timestamp_col].dt.tz is None:  # defensive; utc=True always sets it
        raise ValueError("decision timestamps must normalize to UTC")
    # Identity is defined on canonical UTC timestamps. Checking before this
    # conversion could miss the same instant encoded with different offsets.
    if work.loc[:, list(id_columns)].isna().any().any():
        raise ValueError("joined handoff has null row-identity values")
    if work.duplicated(list(id_columns)).any():
        raise ValueError("joined handoff violates exact one-to-one identity uniqueness")
    if max_span_days > 0:
        span_days = (
            work[timestamp_col].max() - work[timestamp_col].min()
        ).total_seconds() / 86_400.0
        if span_days > max_span_days:
            raise ValueError(
                f"smoke date cap exceeded: {span_days:.3f} days > {max_span_days:.3f}; "
                "use a smaller joined handoff"
            )
    decision = _utc(work["execution_decision_utc"], column="execution_decision_utc")
    if not (decision == work[signal_timestamp_col] + pd.Timedelta(hours=1)).all():
        raise ValueError(
            "execution decision timestamps must equal signal timestamp + one hour"
        )
    work["execution_decision_utc"] = decision
    if label_end_time_col:
        if label_end_time_col not in work.columns:
            raise ValueError(f"label-end column {label_end_time_col!r} is missing")
        work[label_end_time_col] = _utc(
            work[label_end_time_col], column=label_end_time_col
        )
        if not (work[label_end_time_col] == decision + pd.Timedelta(hours=12)).all():
            raise ValueError(
                "execution label-end timestamps must equal decision timestamp + 12 hours"
            )

    for column in REQUIRED_TARGET_COLUMNS:
        values = pd.to_numeric(work[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"joined handoff has non-finite required target/baseline {column!r}"
            )
        work[column] = values.astype("float64")
    for column in (
        "execution_gross_ev_12h",
        "execution_cost_return",
        "existing_alpha_ev_source_basis",
        "alpha_source_cost_return",
    ):
        values = pd.to_numeric(work[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"joined handoff has non-finite execution accounting column {column!r}"
            )
        work[column] = values.astype("float64")
    if (
        float(
            (
                work["execution_gross_ev_12h"]
                - work["execution_cost_return"]
                - work["execution_net_ev_12h"]
            )
            .abs()
            .max()
        )
        > 1e-6
    ):
        raise ValueError("execution gross-cost-net accounting identity is inconsistent")
    if (
        float(
            (
                work["existing_alpha_ev_source_basis"]
                + work["alpha_source_cost_return"]
                - work["execution_cost_return"]
                - work["existing_alpha_ev"]
            )
            .abs()
            .max()
        )
        > 1e-6
    ):
        raise ValueError("alpha EV cost-basis reconciliation is inconsistent")
    if set(work[side_col].astype(str).str.lower()) - {"long", "short"}:
        raise ValueError(f"{side_col!r} must contain only canonical long/short values")
    if work[archetype_col].isna().any():
        raise ValueError(f"{archetype_col!r} must be explicit for every joined row")

    family_columns: dict[str, list[str]] = {family: [] for family in REQUIRED_FAMILIES}
    for column, spec in provenance.items():
        if spec.model_input and spec.family in family_columns:
            family_columns[spec.family].append(column)
        if spec.model_input:
            values = pd.to_numeric(work[column], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"joined handoff has non-finite feature {column!r}")
            work[column] = values.astype("float64")
        elif work[column].isna().any():
            raise ValueError(f"joined handoff has null calibration context {column!r}")
        available = _utc(work[spec.available_at_col], column=spec.available_at_col)
        if (available > work[timestamp_col]).any():
            raise ValueError(
                f"feature {column!r} was available after the decision timestamp"
            )
    missing_families = [
        family for family, columns in family_columns.items() if not columns
    ]
    if missing_families:
        raise ValueError(
            "provenance is missing feature families: " + ", ".join(missing_families)
        )
    if len(family_columns["catboost_probabilities"]) < 2:
        raise ValueError(
            "joined handoff requires the complete CatBoost probability vector (at least two columns)"
        )
    if "existing_alpha_ev" not in family_columns["alpha_score"]:
        raise ValueError(
            "existing_alpha_ev must be the declared frozen alpha_score baseline feature"
        )

    declared_rows = provenance_payload["handoff"].get("row_count")
    if declared_rows is not None and int(declared_rows) != len(work):
        raise ValueError(
            f"handoff.row_count={declared_rows} does not match joined rows={len(work)}"
        )
    work = work.sort_values(
        [timestamp_col, *id_columns[1:]], kind="stable"
    ).reset_index(drop=True)
    # Reuse the trainer's complete semantic contract during dry runs too.  In
    # particular, this rejects outcome-like feature names and non-frozen inputs.
    validate_execution_ev_training_contract(
        work,
        provenance,
        decision_time_col=timestamp_col,
        predicted_path_archetype_col=archetype_col,
    )
    return work


def _oof_ledger(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    timestamp_col: str,
    oof_provenance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    audit_columns = [
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_exit_reason",
        "execution_exit_hour",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "existing_alpha_ev_source_basis",
        "alpha_source_cost_return",
        "existing_alpha_ev",
        "side_name",
        "catboost_archetype",
    ]
    upstream_oof = [
        column
        for column in frame.columns
        if column.endswith("_oof_fold") or column.endswith("_available_at")
    ]
    keep = list(
        dict.fromkeys(
            [
                *id_columns,
                timestamp_col,
                *[column for column in audit_columns if column in frame.columns],
                *upstream_oof,
            ]
        )
    )
    output = frame.loc[:, keep].copy()
    for column in predictions.columns:
        output[column] = predictions[column].to_numpy()
        output[f"{column}__is_oof"] = predictions[column].notna().to_numpy()
    if oof_provenance is not None:
        if not oof_provenance.index.equals(frame.index):
            raise ValueError(
                "OOF provenance index does not match the exact joined handoff"
            )
        for column in oof_provenance.columns:
            if column in output.columns:
                raise ValueError(
                    f"OOF provenance column conflicts with ledger column {column!r}"
                )
            output[column] = oof_provenance[column].to_numpy()
    return output


def _winner_table(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    timestamp_col: str,
) -> pd.DataFrame:
    actual = pd.to_numeric(frame["execution_net_ev_12h"], errors="coerce").to_numpy(
        dtype=float
    )
    weeks = frame[timestamp_col].dt.floor("D") - pd.to_timedelta(
        frame[timestamp_col].dt.weekday, unit="D"
    )
    rows: list[dict[str, Any]] = []
    for column in predictions.columns:
        predicted = pd.to_numeric(predictions[column], errors="coerce").to_numpy(
            dtype=float
        )
        valid = np.isfinite(actual) & np.isfinite(predicted)
        if not valid.any():
            continue
        y, score = actual[valid], predicted[valid]
        tail_count = max(1, int(np.ceil(len(y) * 0.10)))
        tail = np.argsort(score, kind="stable")[-tail_count:]
        residual = score - y
        weekly_tail: list[float] = []
        for _, positions in pd.Series(np.flatnonzero(valid)).groupby(
            weeks.iloc[np.flatnonzero(valid)].to_numpy()
        ):
            local = positions.to_numpy(dtype=int)
            local_tail = local[
                np.argsort(predicted[local], kind="stable")[
                    -max(1, int(np.ceil(len(local) * 0.10))) :
                ]
            ]
            weekly_tail.append(float(np.mean(actual[local_tail])))
        rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        rank_score = pd.Series(score).rank(method="average").to_numpy(dtype=float)
        ic = (
            float(np.corrcoef(rank_y, rank_score)[0, 1])
            if len(y) > 1 and np.std(rank_y) and np.std(rank_score)
            else float("nan")
        )
        mode, arm = column.split("__", maxsplit=1)
        rows.append(
            {
                "prediction": column,
                "mode": mode,
                "arm": arm,
                "oof_rows": int(len(y)),
                "top10_rows": int(tail_count),
                "top10_mean_net_ev": float(np.mean(y[tail])),
                "top10_sum_net_ev": float(np.sum(y[tail])),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "spearman": ic,
                "weeks": int(len(weekly_tail)),
                "weekly_top10_mean_net_ev": float(np.mean(weekly_tail)),
                "weekly_top10_std_net_ev": float(np.std(weekly_tail)),
                "worst_week_top10_mean_net_ev": float(np.min(weekly_tail)),
                "positive_week_fraction": float(np.mean(np.asarray(weekly_tail) > 0.0)),
            }
        )
    if not rows:
        raise ValueError("trainer emitted no finite OOF predictions")
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "top10_mean_net_ev",
                "positive_week_fraction",
                "worst_week_top10_mean_net_ev",
                "mae",
            ],
            ascending=[False, False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Exact joined handoff parquet."
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        required=True,
        help="Strict joined-handoff provenance JSON.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--id-cols",
        type=_parse_columns,
        default=None,
        help=(
            "Exact handoff identity columns. By default, use the join_keys "
            "declared by the provenance artifact."
        ),
    )
    parser.add_argument("--timestamp-col")
    parser.add_argument("--side-col", default="side_name")
    parser.add_argument("--archetype-col", default="catboost_archetype")
    parser.add_argument("--label-end-time-col", default="execution_label_end_utc")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Hard row cap. Defaults to 5,000 in smoke mode and 1,000,000 in production.",
    )
    parser.add_argument(
        "--max-span-days",
        type=float,
        default=None,
        help="Hard date-span cap. Defaults to 31 days in smoke mode and 120 in production.",
    )
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--min-train-rows", type=int)
    parser.add_argument("--hpo-trials", type=int)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--early-stopping-rounds", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument(
        "--production",
        action="store_true",
        help=(
            "Use the full OOF calendar, execution decision timestamp, three outer "
            "folds, side-local HPO, and production-sized LightGBM defaults."
        ),
    )
    parser.add_argument(
        "--enable-timing-risk-head",
        action="store_true",
        help=(
            "Explicitly opt into the companion timing-risk model. It is deferred "
            "by default until the execution-EV winner is stable."
        ),
    )
    parser.add_argument(
        "--disable-timing-risk-head",
        action="store_true",
        help="Skip the side-local 12h timing plus loss-risk companion ablation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and persist the input manifest without fitting.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Path]:
    production_mode = bool(getattr(args, "production", False))

    def resolved(value: Any, *, smoke: Any, production: Any) -> Any:
        return (
            value if value is not None else (production if production_mode else smoke)
        )

    args.timestamp_col = resolved(
        getattr(args, "timestamp_col", None),
        smoke="__ts__",
        production="execution_decision_utc",
    )
    args.max_rows = resolved(args.max_rows, smoke=5_000, production=1_000_000)
    args.max_span_days = resolved(args.max_span_days, smoke=31.0, production=120.0)
    args.n_splits = resolved(args.n_splits, smoke=2, production=3)
    args.min_train_rows = resolved(args.min_train_rows, smoke=50, production=5_000)
    args.hpo_trials = resolved(args.hpo_trials, smoke=0, production=40)
    args.n_estimators = resolved(args.n_estimators, smoke=150, production=1_500)
    args.early_stopping_rounds = resolved(
        args.early_stopping_rounds, smoke=30, production=100
    )
    args.n_jobs = resolved(args.n_jobs, smoke=1, production=3)
    if args.max_rows < 1 or args.max_span_days < 0:
        raise ValueError(
            "max-rows must be positive and max-span-days must be non-negative"
        )
    if args.n_splits < 1 or args.min_train_rows < 4 or args.n_estimators < 1:
        raise ValueError("invalid trainer size arguments")
    if not args.input.is_file() or args.input.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("--input must be an existing parquet handoff")
    if not args.provenance_json.is_file():
        raise ValueError("--provenance-json must be an existing file")
    preflight_rows = _parquet_row_count(args.input)
    if preflight_rows > args.max_rows:
        raise ValueError(
            f"smoke row cap exceeded before loading: {preflight_rows} rows > {args.max_rows}; "
            "prepare a bounded handoff"
        )
    provenance, provenance_payload = _load_provenance(args.provenance_json)
    id_columns = (
        list(args.id_cols)
        if args.id_cols is not None
        else list(map(str, provenance_payload["handoff"]["join_keys"]))
    )
    frame = pd.read_parquet(args.input)
    if len(frame) != preflight_rows:
        raise ValueError("parquet metadata row count changed while loading the handoff")
    frame = _validate_handoff(
        frame,
        provenance=provenance,
        provenance_payload=provenance_payload,
        id_columns=id_columns,
        timestamp_col=args.timestamp_col,
        side_col=args.side_col,
        archetype_col=args.archetype_col,
        label_end_time_col=args.label_end_time_col,
        max_span_days=args.max_span_days,
    )
    timing_risk_enabled = bool(
        getattr(args, "enable_timing_risk_head", False)
    ) and not bool(getattr(args, "disable_timing_risk_head", False))
    if timing_risk_enabled:
        timing_columns = (
            "execution_exit_hour",
            "execution_exit_reason",
        )
        missing_timing = [column for column in timing_columns if column not in frame]
        if missing_timing:
            raise ValueError(
                "timing/risk companion head requires execution label fields: "
                + ", ".join(missing_timing)
            )
    config = ExecutionEVTrainerConfig(
        n_splits=args.n_splits,
        min_train_rows=args.min_train_rows,
        purge_hours=12.0,
        embargo_hours=12.0,
        hpo_trials=args.hpo_trials,
        early_stopping_rounds=args.early_stopping_rounds,
        n_estimators=args.n_estimators,
        side_col=args.side_col,
        catboost_archetype_col=args.archetype_col,
        decision_time_col=args.timestamp_col,
        label_end_time_col=args.label_end_time_col,
        run_ablations=not args.no_ablations,
        n_jobs=args.n_jobs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema": "execution_ev_meta_runner_v1",
        "run_mode": "production" if production_mode else "smoke",
        "input": {
            "path": str(args.input),
            "sha256": _sha256(args.input),
            "rows": int(len(frame)),
        },
        "provenance": {
            "path": str(args.provenance_json),
            "sha256": _sha256(args.provenance_json),
            "payload": provenance_payload,
        },
        "timestamp_range_utc": {
            "start": frame[args.timestamp_col].min(),
            "end": frame[args.timestamp_col].max(),
        },
        "identity_columns": id_columns,
        "trainer_config": asdict(config),
        "timing_risk_head_enabled": timing_risk_enabled,
        "resource_caps": {
            "max_rows": args.max_rows,
            "max_span_days": args.max_span_days,
        },
        "leakage_contract": "exact one-to-one handoff; finite pre-entry OOF/frozen features; feature availability <= decision time; 12h purge and embargo; OOF-only model comparison",
    }
    manifest_path = _write_json(args.output_dir / "manifest.json", manifest)
    if args.dry_run:
        return {"manifest": manifest_path}

    bundle = train_execution_ev_meta(frame, provenance, config=config)
    bundle_path = save_execution_ev_bundle(
        bundle, args.output_dir / "execution_ev_bundle.joblib"
    )
    report_paths = write_execution_ev_report(bundle, args.output_dir)
    ledger = _oof_ledger(
        frame,
        bundle.oof_predictions,
        id_columns=id_columns,
        timestamp_col=args.timestamp_col,
        oof_provenance=getattr(bundle, "oof_provenance", None),
    )
    oof_path = args.output_dir / "joined_execution_ev_oof.parquet"
    ledger.to_parquet(oof_path, index=False, compression="zstd")
    leaderboard = _winner_table(
        frame, bundle.oof_predictions, timestamp_col=args.timestamp_col
    )
    leaderboard_path = args.output_dir / "oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    model_mode_leaderboard = leaderboard.loc[
        leaderboard["arm"].eq("all_features")
    ].reset_index(drop=True)
    if model_mode_leaderboard.empty:
        raise ValueError(
            "execution-EV trainer emitted no all_features arm for direct/residual selection"
        )
    winner = model_mode_leaderboard.iloc[0].to_dict()
    diagnostic_winner = leaderboard.iloc[0].to_dict()
    winner_payload = {
        "winner": winner,
        "selection_scope": "direct_vs_residual_all_features_only",
        "selection_rule": "among all_features direct/residual arms: highest aggregate OOF top10_mean_net_ev; ties use positive_week_fraction, worst_week_top10_mean_net_ev, then lower MAE",
        "best_diagnostic_arm": diagnostic_winner,
        "ablation_contract": "leave-one-family-out and reduced-input arms are diagnostics; they cannot be promoted as the direct-versus-residual winner",
        "status": "evaluation_only_not_policy_selection",
        "regression_and_stability_diagnostics": winner,
    }
    winner_path = _write_json(args.output_dir / "winner.json", winner_payload)
    timing_paths: dict[str, Path] = {}
    timing_bundle = None
    if timing_risk_enabled:
        timing_config = TimingRiskTrainerConfig(
            n_splits=args.n_splits,
            min_train_rows=args.min_train_rows,
            purge_hours=12.0,
            embargo_hours=12.0,
            early_stopping_rounds=args.early_stopping_rounds,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            side_col=args.side_col,
            catboost_archetype_col=args.archetype_col,
            decision_time_col=args.timestamp_col,
            label_end_time_col=args.label_end_time_col,
        )
        timing_bundle = train_execution_timing_risk_meta(
            frame,
            provenance,
            config=timing_config,
            target_spec=ExecutionTimingRiskTargetSpec(),
        )
        timing_bundle_path = save_execution_timing_risk_bundle(
            timing_bundle,
            args.output_dir / "execution_timing_risk_bundle.joblib",
        )
        timing_report_paths = write_execution_timing_risk_report(
            timing_bundle, args.output_dir / "timing_risk"
        )
        timing_paths = {
            "timing_risk_bundle": timing_bundle_path,
            **{
                f"timing_risk_{key}": value
                for key, value in timing_report_paths.items()
            },
        }
    manifest.update(
        {
            "status": "completed",
            "bundle": bundle_path.name,
            "oof_ledger": oof_path.name,
            "module_report_paths": {
                key: value.name for key, value in report_paths.items()
            },
            "leaderboard": leaderboard_path.name,
            "winner": winner,
            "oof_contract": bundle.report["oof_contract"],
            "folds": bundle.report["folds"],
            "timing_risk": (
                {
                    "status": "completed",
                    "bundle": timing_paths["timing_risk_bundle"].name,
                    "target_contract": timing_bundle.report["target_contract"],
                    "oof_contract": timing_bundle.report["oof_contract"],
                }
                if timing_bundle is not None
                else {"status": "disabled"}
            ),
        }
    )
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "bundle": bundle_path,
        "oof": oof_path,
        "leaderboard": leaderboard_path,
        "winner": winner_path,
        **timing_paths,
        **report_paths,
    }


def main() -> None:
    args = _parser().parse_args()
    try:
        paths = run(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"execution-EV meta runner failed: {exc}") from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
