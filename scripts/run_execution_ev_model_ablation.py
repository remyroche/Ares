#!/usr/bin/env python3
"""Run bounded downstream execution-EV model and MDA ablations on a handoff."""

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

from extreme_price_movements.execution_ev_meta import FeatureProvenance  # noqa: E402
from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    ALGORITHM_NAMES,
    ExecutionEVModelAblationConfig,
    save_execution_ev_model_ablation_bundle,
    train_execution_ev_model_ablation,
    validate_execution_ev_model_ablation_contract,
    write_execution_ev_model_ablation_report,
)

HANDOFF_SCHEMA = "execution_ev_joined_handoff_v2"
DEFAULT_ID_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")


def _parse_columns(value: str) -> list[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns or len(columns) != len(set(columns)):
        raise argparse.ArgumentTypeError("columns must be a non-empty unique list")
    return columns


def _parse_algorithms(value: str) -> tuple[str, ...]:
    names = tuple(_parse_columns(value))
    unknown = sorted(set(names) - set(ALGORITHM_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError("unknown algorithms: " + ", ".join(unknown))
    return names


def _parse_target_modes(value: str) -> tuple[str, ...]:
    modes = tuple(_parse_columns(value))
    if not set(modes) <= {"direct", "residual"}:
        raise argparse.ArgumentTypeError(
            "--target-modes must contain direct and/or residual"
        )
    return modes


def _parse_feature_arms(value: str) -> tuple[str, ...]:
    arms = tuple(_parse_columns(value))
    if not set(arms) <= {"all_features", "mda_1se"}:
        raise argparse.ArgumentTypeError(
            "--feature-arms supports only all_features,mda_1se"
        )
    return arms


def _utc(values: pd.Series, *, column: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"{column!r} contains null or invalid timestamps")
    return converted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("pyarrow is required to preflight the row cap") from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


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


def _load_provenance(path: Path) -> tuple[dict[str, FeatureProvenance], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid provenance JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != HANDOFF_SCHEMA:
        raise ValueError(f"provenance schema must be {HANDOFF_SCHEMA!r}")
    handoff = payload.get("handoff")
    features = payload.get("features")
    if not isinstance(handoff, dict) or not isinstance(features, dict) or not features:
        raise ValueError("provenance requires non-empty handoff and features objects")
    if handoff.get("join_mode") != "exact_inner_one_to_one":
        raise ValueError("handoff.join_mode must be exact_inner_one_to_one")
    if not isinstance(handoff.get("join_keys"), list) or not handoff["join_keys"]:
        raise ValueError("handoff.join_keys must be a non-empty list")
    if not handoff.get("source_artifacts"):
        raise ValueError("handoff.source_artifacts must identify the joined sources")
    parsed: dict[str, FeatureProvenance] = {}
    for column, raw in features.items():
        if not isinstance(column, str) or not isinstance(raw, dict):
            raise ValueError("provenance.features must map names to objects")
        required = ("family", "source", "pre_entry", "oof_or_frozen", "available_at_col")
        if any(key not in raw for key in required):
            raise ValueError(f"feature {column!r} has incomplete strict provenance")
        if not isinstance(raw["pre_entry"], bool) or not isinstance(raw["oof_or_frozen"], bool):
            raise ValueError(f"feature {column!r} must declare boolean provenance flags")
        parsed[column] = FeatureProvenance(
            family=str(raw["family"]),
            source=str(raw["source"]),
            pre_entry=bool(raw["pre_entry"]),
            oof_or_frozen=bool(raw["oof_or_frozen"]),
            available_at_col=str(raw["available_at_col"]),
            model_input=bool(raw.get("model_input", True)),
            class_order=(
                tuple(map(str, raw["class_order"]))
                if raw.get("class_order") is not None
                else None
            ),
            class_order_sha256=(
                str(raw["class_order_sha256"])
                if raw.get("class_order_sha256") is not None
                else None
            ),
        )
    return parsed, payload


def _validate_handoff(
    frame: pd.DataFrame,
    *,
    provenance: Mapping[str, FeatureProvenance],
    payload: Mapping[str, Any],
    id_columns: Sequence[str],
    config: ExecutionEVModelAblationConfig,
    max_span_days: float,
) -> pd.DataFrame:
    if list(id_columns) != list(payload["handoff"]["join_keys"]):
        raise ValueError("--id-cols must exactly match provenance handoff.join_keys")
    required = list(
        dict.fromkeys(
            [
                *id_columns,
                config.decision_time_col,
                config.side_col,
                config.catboost_archetype_col,
                config.target_spec.net_ev_col,
                config.target_spec.alpha_ev_col,
                config.gross_ev_col,
                *provenance.keys(),
                *[
                    spec.available_at_col
                    for spec in provenance.values()
                    if spec.available_at_col
                ],
            ]
        )
    )
    if config.label_end_time_col:
        required.append(config.label_end_time_col)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("joined handoff is missing required columns: " + ", ".join(missing))
    work = frame.copy()
    work[config.decision_time_col] = _utc(
        work[config.decision_time_col], column=config.decision_time_col
    )
    if work.loc[:, list(id_columns)].isna().any().any():
        raise ValueError("joined handoff has null row identity")
    if work.duplicated(list(id_columns)).any():
        raise ValueError("joined handoff violates exact one-to-one identity uniqueness")
    if max_span_days > 0:
        span = (
            work[config.decision_time_col].max() - work[config.decision_time_col].min()
        ).total_seconds() / 86_400.0
        if span > max_span_days:
            raise ValueError(
                f"smoke date cap exceeded: {span:.3f} days > {max_span_days:.3f} days"
            )
    if config.label_end_time_col:
        work[config.label_end_time_col] = _utc(
            work[config.label_end_time_col], column=config.label_end_time_col
        )
    for column in (config.target_spec.net_ev_col, config.gross_ev_col):
        values = pd.to_numeric(work[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"joined handoff has non-finite target {column!r}")
        work[column] = values.astype("float64")
    work = work.sort_values(list(id_columns), kind="stable").reset_index(drop=True)
    validate_execution_ev_model_ablation_contract(
        work,
        provenance,
        decision_time_col=config.decision_time_col,
        side_col=config.side_col,
        catboost_archetype_col=config.catboost_archetype_col,
        additional_input_families=config.additional_input_families,
    )
    return work


def _oof_ledger(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    provenance: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    config: ExecutionEVModelAblationConfig,
) -> pd.DataFrame:
    keep = list(
        dict.fromkeys(
            [
                *id_columns,
                config.decision_time_col,
                config.target_spec.net_ev_col,
                config.gross_ev_col,
                config.side_col,
                config.catboost_archetype_col,
            ]
        )
    )
    output = frame.loc[:, keep].copy()
    for column in predictions.columns:
        output[column] = predictions[column].to_numpy()
        output[f"{column}__is_oof"] = predictions[column].notna().to_numpy()
    for column in provenance.columns:
        output[column] = provenance[column].to_numpy()
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-cols", type=_parse_columns, default=None)
    parser.add_argument("--timestamp-col", default="__ts__")
    parser.add_argument("--side-col", default="side_name")
    parser.add_argument("--archetype-col", default="catboost_archetype")
    parser.add_argument("--label-end-time-col", default="execution_label_end_utc")
    parser.add_argument("--gross-ev-col", default="execution_gross_ev_12h")
    parser.add_argument("--algorithms", type=_parse_algorithms, default=ALGORITHM_NAMES)
    parser.add_argument(
        "--additional-input-families",
        type=_parse_columns,
        default=[],
        help="Explicit provenance families to add to the canonical frozen inputs.",
    )
    parser.add_argument(
        "--target-modes", type=_parse_target_modes, default=("direct", "residual")
    )
    parser.add_argument(
        "--feature-arms",
        type=_parse_feature_arms,
        default=("all_features", "mda_1se"),
    )
    parser.add_argument("--max-rows", type=int, default=5_000)
    parser.add_argument("--max-span-days", type=float, default=31.0)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--min-fit-rows", type=int, default=32)
    parser.add_argument("--hpo-trials", type=int, default=12)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--mda-min-features", type=int, default=8)
    parser.add_argument("--mda-max-steps", type=int, default=24)
    parser.add_argument("--mda-repeats", type=int, default=1)
    parser.add_argument("--isotonic-min-rows", type=int, default=24)
    parser.add_argument(
        "--recent-ev-correction-routes",
        type=_parse_columns,
        default=["catboost_predicted_archetype", "gmm_archetype"],
    )
    parser.add_argument("--gmm-archetype-col", default="gmm_cluster_id")
    parser.add_argument("--disable-recent-ev-correction", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.max_rows < 1:
        raise ValueError("max_rows must be positive")
    if args.hpo_trials < 0:
        raise ValueError("hpo_trials must be non-negative")
    if _parquet_row_count(args.input) > args.max_rows:
        raise ValueError("input exceeds --max-rows before loading; use a bounded joined handoff")
    provenance, payload = _load_provenance(args.provenance_json)
    id_columns = tuple(args.id_cols or payload["handoff"]["join_keys"])
    config = ExecutionEVModelAblationConfig(
        n_splits=args.n_splits,
        min_train_rows=args.min_train_rows,
        min_fit_rows=args.min_fit_rows,
        hpo_trials=args.hpo_trials,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        side_col=args.side_col,
        decision_time_col=args.timestamp_col,
        label_end_time_col=args.label_end_time_col or None,
        catboost_archetype_col=args.archetype_col,
        gross_ev_col=args.gross_ev_col,
        mda_min_features=args.mda_min_features,
        mda_max_steps=args.mda_max_steps,
        mda_repeats=args.mda_repeats,
        isotonic_min_rows=args.isotonic_min_rows,
        recent_ev_correction_enabled=not args.disable_recent_ev_correction,
        recent_ev_correction_routes=tuple(args.recent_ev_correction_routes),
        gmm_archetype_col=args.gmm_archetype_col,
        algorithms=tuple(args.algorithms),
        target_modes=tuple(args.target_modes),
        additional_input_families=tuple(args.additional_input_families),
        feature_arms=tuple(args.feature_arms),
    )
    frame = _validate_handoff(
        pd.read_parquet(args.input),
        provenance=provenance,
        payload=payload,
        id_columns=id_columns,
        config=config,
        max_span_days=args.max_span_days,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "execution_ev_model_ablation_runner_v1",
        "input": {"path": str(args.input), "sha256": _sha256(args.input), "rows": len(frame)},
        "provenance": {
            "path": str(args.provenance_json),
            "sha256": _sha256(args.provenance_json),
            "payload": payload,
        },
        "identity_columns": id_columns,
        "trainer_config": asdict(config),
        "smoke_caps": {"max_rows": args.max_rows, "max_span_days": args.max_span_days},
        "leakage_contract": (
            "strict pre-entry OOF/frozen provenance; side-local purged OOF; "
            "HPO/MDA/isotonic fit only on outer training data; recent-EV "
            "correction uses daily resolved-before-snapshot OOF outcomes only"
        ),
    }
    manifest_path = _write_json(args.output_dir / "manifest.json", manifest)
    if args.dry_run:
        return {"manifest": manifest_path}
    bundle = train_execution_ev_model_ablation(frame, provenance, config=config)
    bundle_path = save_execution_ev_model_ablation_bundle(
        bundle, args.output_dir / "execution_ev_model_ablation_bundle.joblib"
    )
    paths = write_execution_ev_model_ablation_report(bundle, args.output_dir)
    ledger = _oof_ledger(
        frame,
        bundle.oof_predictions,
        bundle.oof_provenance,
        id_columns=id_columns,
        config=config,
    )
    ledger_path = args.output_dir / "joined_execution_ev_model_ablation_oof.parquet"
    ledger.to_parquet(ledger_path, index=False, compression="zstd")
    leaderboard = pd.DataFrame(bundle.report["leaderboard"])
    eligible = leaderboard.loc[leaderboard["feature_arm"].isin(["all_features", "mda_1se"])]
    winner = eligible.iloc[0].to_dict() if not eligible.empty else {}
    winner_path = _write_json(
        args.output_dir / "winner.json",
        {
            "winner": winner,
            "selection_scope": "OOF model ablation only; not a policy promotion",
            "selection_rule": "highest ranking objective (top-k gross/net EV plus rank correlation); calibration is diagnostic only",
        },
    )
    return {
        "manifest": manifest_path,
        "bundle": bundle_path,
        "oof": ledger_path,
        "winner": winner_path,
        **paths,
    }


def main() -> None:
    paths = run(_parser().parse_args())
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
