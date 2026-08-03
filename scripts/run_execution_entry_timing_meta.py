#!/usr/bin/env python3
"""Train the leakage-safe post-execution-EV entry timing/action-value head."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_entry_timing_meta import (  # noqa: E402
    EntryAction,
    EntryTimingFeatureProvenance,
    EntryTimingTargetSpec,
    EntryTimingTrainerConfig,
    _atomic_json,
    _fingerprint_columns,
    build_counterfactual_entry_action_labels,
    exact_entry_timing_fingerprint,
    train_execution_entry_timing_meta,
    validate_entry_timing_feature_contract,
    write_execution_entry_timing_artifacts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_manifest_hash(payload: Mapping[str, Any]) -> str:
    canonical = {
        str(key): value
        for key, value in payload.items()
        if key != "prediction_role_manifest_sha256"
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_execution_ev_target_manifest(
    path: Path, *, horizon_hours: float
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("--execution-ev-target-manifest must be an existing file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("--execution-ev-target-manifest must contain a JSON object") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("--execution-ev-target-manifest must contain a JSON object")
    signed = payload.get("prediction_role_manifest_sha256")
    if not isinstance(signed, str) or not signed:
        raise ValueError("execution-EV target manifest is missing signed manifest identity")
    if not hmac.compare_digest(signed, _canonical_manifest_hash(payload)):
        raise ValueError("execution-EV target manifest signed identity does not verify")
    if payload.get("schema") != "execution_ev_12h_hourly_policy_labels_v2":
        raise ValueError("execution-EV target manifest has an unsupported schema")
    if payload.get("prediction_role") != "execution_ev_12h_labels":
        raise ValueError("execution-EV target manifest has the wrong prediction role")
    timing = payload.get("timing")
    policy = payload.get("policy")
    if not isinstance(timing, Mapping) or not isinstance(policy, Mapping):
        raise ValueError("execution-EV target manifest is missing timing or policy identity")
    if timing.get("first_path_timestamp") != "__decision_ts__":
        raise ValueError(
            "execution-EV target manifest must define __decision_ts__ as its first path timestamp"
        )
    if timing.get("signal_timestamp") != "__ts__":
        raise ValueError("execution-EV target manifest must retain '__ts__' signal timing")
    try:
        decision_delay_hours = float(timing.get("decision_delay_hours"))
    except (TypeError, ValueError) as exc:
        raise ValueError("execution-EV target manifest has an invalid decision_delay_hours") from exc
    if decision_delay_hours < 0.0:
        raise ValueError("execution-EV target manifest has a negative decision_delay_hours")
    try:
        manifest_horizon = float(timing.get("horizon_hours"))
    except (TypeError, ValueError) as exc:
        raise ValueError("execution-EV target manifest has an invalid horizon_hours") from exc
    if manifest_horizon != float(horizon_hours):
        raise ValueError(
            "execution-EV target manifest horizon does not match --horizon-hours"
        )
    expected_end = f"__decision_ts__ + {int(manifest_horizon)}h"
    if timing.get("label_end") != expected_end:
        raise ValueError("execution-EV target manifest has an incompatible terminal timing identity")
    long_geometry = policy.get("long_geometry")
    short_geometry = policy.get("short_geometry")
    policy_sha256 = policy.get("sha256")
    if not isinstance(long_geometry, Mapping) or not isinstance(short_geometry, Mapping):
        raise ValueError("execution-EV target manifest is missing long/short policy geometry")
    if not isinstance(policy_sha256, str) or not policy_sha256:
        raise ValueError("execution-EV target manifest is missing policy manifest hash")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "signed_manifest_sha256": signed,
        "schema": payload.get("schema"),
        "horizon_hours": manifest_horizon,
        "decision_delay_hours": decision_delay_hours,
        "policy_manifest_sha256": policy_sha256,
        "long_geometry": dict(long_geometry),
        "short_geometry": dict(short_geometry),
    }


def _none(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def _ints(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("wait minutes must be positive")
    return parsed


def _floats(value: str) -> tuple[float, ...]:
    parsed = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    if any(item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("adverse offsets must be positive")
    return parsed


def _load_provenance(path: Path) -> tuple[dict[str, EntryTimingFeatureProvenance], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("--provenance-json must contain a JSON object") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("--provenance-json must contain a JSON object")
    raw = payload.get("features", payload.get("provenance", payload.get("feature_provenance")))
    if not isinstance(raw, Mapping):
        raise ValueError("provenance JSON must contain features/provenance/feature_provenance mapping")
    result: dict[str, EntryTimingFeatureProvenance] = {}
    for name, record in raw.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise ValueError("every entry timing feature provenance record must be an object")
        result[name] = EntryTimingFeatureProvenance(**dict(record))
    return result, dict(payload)


def _action_grid(waits: tuple[int, ...], offsets: tuple[float, ...]) -> tuple[EntryAction, ...]:
    actions = [EntryAction("enter_now")]
    for wait in waits:
        actions.append(EntryAction("wait_market", wait_minutes=wait))
        for offset in offsets:
            actions.append(EntryAction("adverse_limit", wait_minutes=wait, adverse_offset_atr=offset))
    return tuple(actions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Joined OOF/frozen handoff with train-only 1m path column.")
    parser.add_argument("--provenance-json", type=Path, required=True, help="Feature provenance with OOF-fold or frozen-bundle identifiers.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-ev-target-manifest", type=Path, required=True)
    parser.add_argument("--timestamp-col", default="__decision_ts__")
    parser.add_argument("--side-col", default="side_name")
    parser.add_argument("--archetype-col", default="catboost_archetype")
    parser.add_argument("--label-end-time-col", default="execution_label_end_utc")
    parser.add_argument("--path-col", default="execution_future_path")
    parser.add_argument("--atr-col", default="atr_1h")
    parser.add_argument("--decision-price-col", default=None)
    parser.add_argument("--cost-return-col", default=None)
    parser.add_argument("--fee-return-col", default=None)
    parser.add_argument("--entry-spread-bps-col", default=None)
    parser.add_argument("--exit-spread-bps-col", default=None)
    parser.add_argument(
        "--allow-action-invariant-all-in-cost",
        action="store_true",
        help="Explicitly allow an all-in cost only when it is invariant across every action.",
    )
    parser.add_argument("--horizon-hours", type=float, default=12.0)
    parser.add_argument("--meaningful-mfe-atr", type=float, default=1.5)
    parser.add_argument("--meaningful-mfe-return-floor", type=float, default=0.015)
    parser.add_argument("--adverse-mae-atr", type=float, default=0.25)
    parser.add_argument("--wait-minutes", type=_ints, default=(5, 10, 20))
    parser.add_argument("--adverse-offset-atr", type=_floats, default=(0.25, 0.50))
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--embargo-hours", type=float, default=12.0)
    parser.add_argument("--n-estimators", type=int, default=320)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--hpo-trials", type=int, default=8)
    parser.add_argument("--decision-hpo-trials", type=int, default=8)
    parser.add_argument(
        "--counterfactual-labels",
        type=Path,
        default=None,
        help=(
            "Reuse a previously materialized train-only action ledger. Its "
            "action IDs, row positions, target fingerprint and source input "
            "must match this run."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate the feature/path contract and write only the manifest.")
    return parser


def run(args: argparse.Namespace) -> dict[str, Path]:
    if not args.input.is_file() or args.input.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("--input must be an existing parquet handoff")
    if not args.provenance_json.is_file():
        raise ValueError("--provenance-json must be an existing file")
    if args.output_dir.exists():
        raise ValueError("--output-dir must not already exist")
    if args.timestamp_col != "__decision_ts__":
        raise ValueError("--timestamp-col must be upstream execution-EV '__decision_ts__'")
    if _none(args.label_end_time_col) != "execution_label_end_utc":
        raise ValueError("--label-end-time-col must be 'execution_label_end_utc'")
    target_manifest = _load_execution_ev_target_manifest(
        args.execution_ev_target_manifest, horizon_hours=args.horizon_hours
    )
    frame = pd.read_parquet(args.input)
    if not {"__ts__", "__decision_ts__", "execution_label_end_utc"}.issubset(frame.columns):
        raise ValueError(
            "entry timing input must retain upstream __ts__, __decision_ts__, and "
            "execution_label_end_utc"
        )
    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    decisions = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="coerce")
    label_end = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="coerce")
    if signal.isna().any() or decisions.isna().any() or label_end.isna().any():
        raise ValueError("entry timing input has invalid upstream execution-EV timing timestamps")
    if not (
        decisions == signal + pd.Timedelta(hours=target_manifest["decision_delay_hours"])
    ).all():
        raise ValueError(
            "entry timing input __decision_ts__ does not match the execution-EV target "
            "signal-to-decision timing"
        )
    if not (label_end == decisions + pd.Timedelta(hours=args.horizon_hours)).all():
        raise ValueError(
            "entry timing input execution_label_end_utc does not match the execution-EV target horizon"
        )
    provenance, provenance_payload = _load_provenance(args.provenance_json)
    config = EntryTimingTrainerConfig(
        n_splits=args.n_splits,
        min_train_rows=args.min_train_rows,
        purge_hours=args.purge_hours,
        embargo_hours=args.embargo_hours,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        hpo_trials=args.hpo_trials,
        decision_hpo_trials=args.decision_hpo_trials,
        n_jobs=args.n_jobs,
        decision_time_col=args.timestamp_col,
        side_col=args.side_col,
        archetype_col=args.archetype_col,
        label_end_time_col=_none(args.label_end_time_col),
        action_grid=_action_grid(tuple(args.wait_minutes), tuple(args.adverse_offset_atr)),
    )
    target_spec = EntryTimingTargetSpec(
        path_col=args.path_col,
        atr_col=args.atr_col,
        decision_price_col=_none(args.decision_price_col),
        cost_return_col=_none(args.cost_return_col),
        fee_return_col=_none(args.fee_return_col),
        entry_spread_bps_col=_none(args.entry_spread_bps_col),
        exit_spread_bps_col=_none(args.exit_spread_bps_col),
        allow_action_invariant_all_in_cost=bool(args.allow_action_invariant_all_in_cost),
        horizon_hours=args.horizon_hours,
        meaningful_mfe_atr=getattr(args, "meaningful_mfe_atr", 1.5),
        meaningful_mfe_return_floor=getattr(args, "meaningful_mfe_return_floor", 0.015),
        adverse_mae_atr=getattr(args, "adverse_mae_atr", 0.25),
        long_policy_geometry=target_manifest["long_geometry"],
        short_policy_geometry=target_manifest["short_geometry"],
        execution_ev_target_manifest_path=target_manifest["path"],
        execution_ev_target_manifest_sha256=target_manifest["sha256"],
        execution_ev_target_signed_manifest_sha256=target_manifest["signed_manifest_sha256"],
        execution_ev_target_schema=str(target_manifest["schema"]),
        execution_ev_policy_manifest_sha256=target_manifest["policy_manifest_sha256"],
    )
    feature_names, execution_ev_feature = validate_entry_timing_feature_contract(frame, provenance, config=config)
    fingerprint_columns = _fingerprint_columns(config, target_spec, provenance)
    input_fingerprint = exact_entry_timing_fingerprint(frame, fingerprint_columns)
    # This validates timestamp causality and exact fee/spread accounting before
    # model fitting.  The returned labels remain train-only and are never
    # forwarded to the scoring API.
    cached_labels = getattr(args, "counterfactual_labels", None)
    if cached_labels is not None:
        if not cached_labels.is_file():
            raise ValueError("--counterfactual-labels must be an existing parquet file")
        cache_manifest_path = cached_labels.parent / "manifest.json"
        if not cache_manifest_path.is_file():
            raise ValueError("cached counterfactual labels require their runner manifest")
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest.get("input_fingerprint") != input_fingerprint:
            raise ValueError("cached counterfactual labels input fingerprint does not match")
        cache_record = cache_manifest.get("counterfactual_labels", {})
        if (
            not isinstance(cache_record, Mapping)
            or cache_record.get("sha256") != _sha256(cached_labels)
        ):
            raise ValueError("cached counterfactual label hash does not verify")
        labels = pd.read_parquet(cached_labels)
        expected_actions = {action.action_id for action in config.action_grid}
        if set(labels.get("action_id", pd.Series(dtype=str)).astype(str)) != expected_actions:
            raise ValueError("cached counterfactual labels do not match the action grid")
        if len(labels) != len(frame) * len(config.action_grid):
            raise ValueError("cached counterfactual labels do not cover every row/action")
        positions = pd.to_numeric(labels.get("base_position"), errors="coerce")
        if (
            positions.isna().any()
            or set(positions.astype(int)) != set(range(len(frame)))
        ):
            raise ValueError("cached counterfactual labels have invalid base positions")
    else:
        labels = build_counterfactual_entry_action_labels(
            frame,
            target_spec=target_spec,
            action_grid=config.action_grid,
            decision_time_col=config.decision_time_col,
            side_col=config.side_col,
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if cached_labels is None:
        labels_path = args.output_dir / "counterfactual_action_labels.parquet"
        labels.to_parquet(labels_path, index=False, compression="zstd")
    else:
        labels_path = cached_labels
    manifest = {
        "schema": "execution_entry_timing_runner_v2",
        "input": {"path": str(args.input), "sha256": _sha256(args.input), "rows": int(len(frame))},
        "provenance": {"path": str(args.provenance_json), "sha256": _sha256(args.provenance_json), "payload": provenance_payload},
        "execution_ev_target": target_manifest,
        "config": asdict(config),
        "target_spec": asdict(target_spec),
        "feature_names": feature_names,
        "protected_execution_ev_feature": execution_ev_feature,
        "input_fingerprint": input_fingerprint,
        "fingerprint_columns": fingerprint_columns,
        "counterfactual_rows": int(len(labels)),
        "counterfactual_labels": {
            "path": labels_path.name,
            "source_path": str(labels_path.resolve()),
            "sha256": _sha256(labels_path),
            "role": "train_only_never_scoring_input",
        },
        "leakage_contract": "exact fixed-horizon 1m paths and all realized labels are train-only; action labels use the signed execution-EV long/short geometry reanchored at each fill; predictive upstream inputs carry row-level source fold/cutoff OOF evidence; scorer accepts only pre-entry frozen inputs; chronological purge/embargo and train-OOF isotonic calibration apply",
    }
    manifest_path = args.output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    if args.dry_run:
        return {"manifest": manifest_path, "counterfactual_labels": labels_path}
    bundle = train_execution_entry_timing_meta(
        frame,
        provenance,
        config=config,
        target_spec=target_spec,
        counterfactual_labels=labels,
    )
    paths = write_execution_entry_timing_artifacts(bundle, args.output_dir)
    manifest["status"] = "completed"
    manifest["bundle_fingerprint"] = bundle.bundle_fingerprint
    manifest["input_fingerprint"] = bundle.input_fingerprint
    manifest["output"] = {name: path.name for name, path in paths.items()}
    _atomic_json(manifest_path, manifest)
    return {"manifest": manifest_path, **paths}


def main() -> None:
    args = _parser().parse_args()
    try:
        paths = run(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"execution entry timing runner failed: {exc}") from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
