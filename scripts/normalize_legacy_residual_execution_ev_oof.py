#!/usr/bin/env python3
"""Normalize a provable legacy residual-OOF artifact for execution-EV use.

The legacy residual artifact predates the row-level execution-EV OOF contract.
It can be adopted only when all of the following independent facts agree:

* UTC timestamp/symbol/side rows exact-join the declared handoff;
* canonical 1h candidate IDs are reproducible on both sources;
* manifest fold intervals partition the OOF rows with their recorded counts;
* the documented signed legacy label-resolution offset reproduces every fold's
  recorded purged training-row count and keeps the latest training label before
  the validation boundary.

No prediction is recomputed, reranked, or calibrated. Any missing or divergent
evidence fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import (
    deterministic_candidate_ids,  # noqa: E402
)

SCHEMA = "legacy_residual_execution_ev_oof_normalizer_v1"
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side_name")
CANDIDATE_ID_COLUMN = "candidate_id"
DEFAULT_CANDIDATE_COLUMNS = (
    "archetype_label_family",
    "archetype_policy_key",
)
HANDOFF_JOIN_LABEL_END_COLUMN = "__legacy_handoff_label_end_ts__"
LEGACY_ALPHA_COST_RETURN = 0.01
LEGACY_ALPHA_TARGET_MODES = frozenset(
    {"residual_net_ev_after_1pct", "ev_after_1pct"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise argparse.ArgumentTypeError("at least one candidate column is required")
    if len(columns) != len(set(columns)):
        raise argparse.ArgumentTypeError("candidate columns must be unique")
    return columns


def _utc(values: pd.Series, *, source: str, column: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{source}: {column!r} contains null or invalid timestamps")
    return result


def _strings(values: pd.Series, *, source: str, column: str) -> pd.Series:
    result = values.astype("string").str.strip()
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{source}: {column!r} contains null or blank identity values")
    return result.astype(str)


def _sides(values: pd.Series, *, source: str, column: str) -> pd.Series:
    result = _strings(values, source=source, column=column).str.lower()
    invalid = sorted(set(result).difference({"long", "short"}))
    if invalid:
        raise ValueError(
            f"{source}: {column!r} must contain canonical long/short sides; "
            f"invalid={invalid[:5]!r}"
        )
    return result


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, source: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {', '.join(missing)}")


def _assert_unique(frame: pd.DataFrame, *, source: str) -> None:
    duplicates = int(frame.duplicated(list(IDENTITY_COLUMNS), keep=False).sum())
    if duplicates:
        raise ValueError(
            f"{source}: duplicate UTC timestamp/symbol/side identities: {duplicates}"
        )


def _normalise_identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    _require_columns(frame, IDENTITY_COLUMNS, source=source)
    result = frame.copy()
    result["__ts__"] = _utc(result["__ts__"], source=source, column="__ts__")
    result["__symbol__"] = _strings(
        result["__symbol__"], source=source, column="__symbol__"
    )
    result["side_name"] = _sides(
        result["side_name"], source=source, column="side_name"
    )
    _assert_unique(result, source=source)
    return result


def _add_or_verify_candidate_ids(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    result = frame.copy()
    canonical = deterministic_candidate_ids(result, timeframe="1h")
    if CANDIDATE_ID_COLUMN in result:
        existing = _strings(
            result[CANDIDATE_ID_COLUMN], source=source, column=CANDIDATE_ID_COLUMN
        )
        if not existing.eq(canonical).all():
            raise ValueError(
                f"{source}: existing candidate_id values do not match canonical "
                "UTC/symbol/1h/side identities"
            )
    result[CANDIDATE_ID_COLUMN] = canonical.to_numpy(dtype=object)
    if result[CANDIDATE_ID_COLUMN].duplicated().any():
        raise ValueError(f"{source}: canonical 1h candidate IDs are not unique")
    return result


def _parse_timestamp(value: Any, *, source: str, field: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"{source}: invalid {field!r}")
    return pd.Timestamp(timestamp)


def _folds(manifest_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"residual_manifest: cannot read {manifest_path}") from exc
    rows = payload.get("folds")
    if not isinstance(rows, list) or not rows:
        raise ValueError("residual_manifest: non-empty 'folds' list is required")
    folds: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"residual_manifest: fold {index} is not an object")
        required = (
            "train_end_exclusive",
            "test_start",
            "test_end_exclusive",
            "train_rows",
            "test_rows",
        )
        missing = sorted(set(required).difference(row))
        if missing:
            raise ValueError(
                f"residual_manifest: fold {index} missing required fields: {missing}"
            )
        train_cutoff = _parse_timestamp(
            row["train_end_exclusive"],
            source=f"residual_manifest fold {index}",
            field="train_end_exclusive",
        )
        test_start = _parse_timestamp(
            row["test_start"],
            source=f"residual_manifest fold {index}",
            field="test_start",
        )
        test_end = _parse_timestamp(
            row["test_end_exclusive"],
            source=f"residual_manifest fold {index}",
            field="test_end_exclusive",
        )
        if train_cutoff != test_start:
            raise ValueError(
                f"residual_manifest: fold {index} train_end_exclusive must equal test_start"
            )
        if test_end <= test_start:
            raise ValueError(f"residual_manifest: fold {index} has non-positive test span")
        try:
            train_rows = int(row["train_rows"])
            test_rows = int(row["test_rows"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"residual_manifest: fold {index} has invalid row counts"
            ) from exc
        if train_rows < 1 or test_rows < 1:
            raise ValueError(f"residual_manifest: fold {index} row counts must be positive")
        folds.append(
            {
                # materialize_execution_ev_alpha_oof uses this same default when
                # a legacy source manifest has no row-level fold identifiers.
                "oof_fold": str(index),
                "train_cutoff": train_cutoff,
                "test_start": test_start,
                "test_end_exclusive": test_end,
                "train_rows": train_rows,
                "test_rows": test_rows,
            }
        )
    folds.sort(key=lambda fold: fold["test_start"])
    for previous, current in zip(folds, folds[1:]):
        if current["test_start"] < previous["test_end_exclusive"]:
            raise ValueError("residual_manifest: overlapping OOF fold intervals")
    return folds


def _legacy_alpha_cost_basis(manifest_path: Path) -> dict[str, Any]:
    """Prove the legacy residual score already carries one 1% net cost.

    The normalizer may not infer this from a score-column name.  It accepts
    only an explicit residual-net target mode or an explicit residual target
    expression rooted in ``ev_after_1pct``.  If both forms are present they
    must agree with the same 1% net-return semantics.
    """

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"residual_manifest: cannot read {manifest_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("residual_manifest: expected a JSON object for cost proof")

    evidence: list[dict[str, str]] = []
    declared_modes: list[tuple[str, str]] = []
    if "target_mode" in payload:
        declared_modes.append(("target_mode", str(payload["target_mode"])))
    params = payload.get("params")
    if isinstance(params, Mapping) and "target_mode" in params:
        declared_modes.append(("params.target_mode", str(params["target_mode"])))
    for field, value in declared_modes:
        normalized = value.strip().lower()
        if normalized not in LEGACY_ALPHA_TARGET_MODES:
            raise ValueError(
                "residual_manifest: declared target evidence is inconsistent with "
                f"a 1% residual-net EV target: {field}={value!r}"
            )
        evidence.append({"field": field, "value": value})

    if "residual_expert_target" in payload:
        value = payload["residual_expert_target"]
        if not isinstance(value, str):
            raise ValueError("residual_manifest: residual_expert_target must be a string")
        normalized = value.strip().lower()
        if "ev_after_1pct" not in normalized or not any(
            token in normalized for token in ("residual", "-", "minus")
        ):
            raise ValueError(
                "residual_manifest: residual_expert_target is not an explicit "
                "residual ev_after_1pct target"
            )
        evidence.append({"field": "residual_expert_target", "value": value})

    if not evidence:
        raise ValueError(
            "residual_manifest: explicit 1% residual-net target evidence is required; "
            "expected target_mode=residual_net_ev_after_1pct/ev_after_1pct or "
            "residual_expert_target rooted in ev_after_1pct"
        )
    return {
        "deducted_cost_return": LEGACY_ALPHA_COST_RETURN,
        "cost_unit": "return",
        "target_semantics": "residual_net_ev_after_1pct",
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
        },
        "source_manifest_evidence": evidence,
        "verification": (
            "source manifest explicitly declares a residual target in ev_after_1pct "
            "units; that target contains exactly one 1% round-trip cost"
        ),
    }


def _load_handoff(args: argparse.Namespace) -> pd.DataFrame:
    source = "candidate_handoff"
    schema = set(pq.read_schema(args.candidate_handoff).names)
    required = {
        *IDENTITY_COLUMNS,
        args.handoff_label_end_col,
        *args.candidate_columns,
    }
    missing = sorted(required.difference(schema))
    if missing:
        raise ValueError(f"{source}: missing required columns: {', '.join(missing)}")
    selected = [*IDENTITY_COLUMNS, args.handoff_label_end_col, *args.candidate_columns]
    if CANDIDATE_ID_COLUMN in schema:
        selected.append(CANDIDATE_ID_COLUMN)
    selected = list(dict.fromkeys(selected))
    raw = pd.read_parquet(args.candidate_handoff, columns=selected)
    if raw.empty:
        raise ValueError(f"{source}: source parquet is empty")
    out = _normalise_identity(raw, source=source)
    out[args.handoff_label_end_col] = _utc(
        raw[args.handoff_label_end_col],
        source=source,
        column=args.handoff_label_end_col,
    ).to_numpy()
    out = _add_or_verify_candidate_ids(out, source=source)
    return out


def _load_residual(args: argparse.Namespace) -> pd.DataFrame:
    source = "residual_oof"
    schema = set(pq.read_schema(args.residual_oof).names)
    required = {
        *IDENTITY_COLUMNS,
        args.residual_label_end_col,
        args.residual_ev_col,
        args.base_ev_col,
    }
    missing = sorted(required.difference(schema))
    if missing:
        raise ValueError(f"{source}: missing required columns: {', '.join(missing)}")
    selected = [
        *IDENTITY_COLUMNS,
        args.residual_label_end_col,
        args.residual_ev_col,
        args.base_ev_col,
    ]
    if CANDIDATE_ID_COLUMN in schema:
        selected.append(CANDIDATE_ID_COLUMN)
    selected = list(dict.fromkeys(selected))
    raw = pd.read_parquet(args.residual_oof, columns=selected)
    if raw.empty:
        raise ValueError(f"{source}: source parquet is empty")
    out = _normalise_identity(raw, source=source)
    out[args.residual_label_end_col] = _utc(
        raw[args.residual_label_end_col],
        source=source,
        column=args.residual_label_end_col,
    ).to_numpy()
    for column in (args.residual_ev_col, args.base_ev_col):
        values = raw[column].to_numpy(copy=True)
        if not np.isfinite(pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)).all():
            raise ValueError(f"{source}: {column!r} contains missing or non-finite values")
        out[column] = values
    out = _add_or_verify_candidate_ids(out, source=source)
    return out


def _require_exact_handoff_join(
    residual: pd.DataFrame,
    handoff: pd.DataFrame,
    *,
    handoff_label_end_col: str,
) -> pd.DataFrame:
    lookup = handoff.loc[
        :, [*IDENTITY_COLUMNS, handoff_label_end_col, CANDIDATE_ID_COLUMN]
    ].rename(columns={handoff_label_end_col: HANDOFF_JOIN_LABEL_END_COLUMN})
    joined = residual.merge(
        lookup,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        suffixes=("", "_handoff"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = int(joined["_merge"].ne("both").sum())
        raise ValueError(
            f"candidate_handoff: {missing} residual OOF rows lack an exact UTC identity match"
        )
    joined = joined.drop(columns="_merge")
    handoff_candidate_id = f"{CANDIDATE_ID_COLUMN}_handoff"
    if not joined[CANDIDATE_ID_COLUMN].eq(joined[handoff_candidate_id]).all():
        raise ValueError("canonical 1h candidate IDs disagree after the exact identity join")
    return joined.drop(columns=handoff_candidate_id)


def _assign_fold_provenance(
    joined: pd.DataFrame,
    handoff: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
    *,
    residual_label_end_col: str,
    handoff_label_end_col: str,
    legacy_offset: pd.Timedelta,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = joined.copy()
    result["oof_fold"] = pd.Series(pd.NA, index=result.index, dtype="string")
    for column in (
        "validation_start",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "train_max_decision_ts",
        "train_fit_cutoff_exclusive",
    ):
        result[column] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")

    expected_offset = pd.Series(legacy_offset, index=result.index)
    observed_offset = result[residual_label_end_col].sub(
        result[HANDOFF_JOIN_LABEL_END_COLUMN]
    )
    if not observed_offset.eq(expected_offset).all():
        counts = observed_offset.value_counts(dropna=False).head(5).to_dict()
        raise ValueError(
            "residual_oof: label-resolution offset from the handoff is not the "
            f"required signed {legacy_offset}; observed={counts!r}"
        )

    fold_audit: list[dict[str, Any]] = []
    assigned = pd.Series(False, index=result.index)
    for fold in folds:
        test_mask = result["__ts__"].ge(fold["test_start"]) & result["__ts__"].lt(
            fold["test_end_exclusive"]
        )
        if (assigned & test_mask).any():
            raise ValueError("residual_manifest: OOF rows map to multiple folds")
        assigned |= test_mask
        test_rows = int(test_mask.sum())
        if test_rows != int(fold["test_rows"]):
            raise ValueError(
                f"residual_manifest: fold {fold['oof_fold']!r} test rows={test_rows} "
                f"do not match recorded {fold['test_rows']}"
            )

        train_mask = handoff["__ts__"].lt(fold["train_cutoff"]) & handoff[
            handoff_label_end_col
        ].add(legacy_offset).lt(fold["train_cutoff"])
        train_rows = int(train_mask.sum())
        if train_rows != int(fold["train_rows"]):
            raise ValueError(
                f"residual_manifest: fold {fold['oof_fold']!r} reconstructed train "
                f"rows={train_rows} do not match recorded {fold['train_rows']}"
            )
        latest_label = handoff.loc[train_mask, handoff_label_end_col].add(legacy_offset).max()
        latest_decision = handoff.loc[train_mask, "__ts__"].max()
        if pd.isna(latest_label) or pd.isna(latest_decision):
            raise ValueError(f"residual_manifest: fold {fold['oof_fold']!r} has empty train")
        if latest_label >= fold["test_start"]:
            raise ValueError(
                f"residual_manifest: fold {fold['oof_fold']!r} training labels do not "
                "resolve before validation"
            )
        if latest_decision >= fold["test_start"]:
            raise ValueError(
                f"residual_manifest: fold {fold['oof_fold']!r} train decisions overlap validation"
            )

        # The downstream legacy contract requires a cutoff after all training
        # labels resolve. Retain the actual maximum decision separately.
        result.loc[test_mask, "oof_fold"] = str(fold["oof_fold"])
        result.loc[test_mask, "validation_start"] = fold["test_start"]
        result.loc[test_mask, "train_decision_cutoff"] = latest_label
        result.loc[test_mask, "label_resolution_available_at"] = latest_label
        result.loc[test_mask, "train_max_decision_ts"] = latest_decision
        result.loc[test_mask, "train_fit_cutoff_exclusive"] = fold["train_cutoff"]
        fold_audit.append(
            {
                "oof_fold": str(fold["oof_fold"]),
                "test_start": fold["test_start"],
                "test_end_exclusive": fold["test_end_exclusive"],
                "recorded_test_rows": int(fold["test_rows"]),
                "verified_test_rows": test_rows,
                "recorded_train_rows": int(fold["train_rows"]),
                "verified_train_rows": train_rows,
                "train_fit_cutoff_exclusive": fold["train_cutoff"],
                "max_train_decision_ts": latest_decision,
                "max_train_label_end_from_handoff": handoff.loc[
                    train_mask, handoff_label_end_col
                ].max(),
                "max_train_label_resolution_available_at": latest_label,
                "label_resolution_before_validation": True,
            }
        )
    if not assigned.all():
        raise ValueError(
            f"residual_manifest: {int((~assigned).sum())} residual OOF rows fall outside declared folds"
        )
    return result, fold_audit


def _assert_output_paths_unused(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError("refusing to overwrite existing outputs: " + ", ".join(existing))


def run(args: argparse.Namespace) -> dict[str, Path]:
    output_manifest = args.output_manifest or args.output_oof.with_suffix(
        ".provenance.json"
    )
    _assert_output_paths_unused(
        [args.output_oof, args.output_candidate_handoff, output_manifest]
    )
    if args.output_oof == args.output_candidate_handoff:
        raise ValueError("OOF and candidate-handoff outputs must differ")
    if not np.isfinite(float(args.legacy_label_end_offset_hours)):
        raise ValueError("legacy label-resolution offset must be finite")

    folds = _folds(args.residual_manifest)
    alpha_cost_basis = _legacy_alpha_cost_basis(args.residual_manifest)
    handoff = _load_handoff(args)
    residual = _load_residual(args)
    joined = _require_exact_handoff_join(
        residual,
        handoff,
        handoff_label_end_col=args.handoff_label_end_col,
    )
    normalized_oof, fold_audit = _assign_fold_provenance(
        joined,
        handoff,
        folds,
        residual_label_end_col=args.residual_label_end_col,
        handoff_label_end_col=args.handoff_label_end_col,
        legacy_offset=pd.Timedelta(hours=float(args.legacy_label_end_offset_hours)),
    )
    normalized_oof["available_at"] = normalized_oof["__ts__"].to_numpy()
    if not (
        normalized_oof["label_resolution_available_at"]
        <= normalized_oof["train_decision_cutoff"]
    ).all():
        raise ValueError("normalized OOF label-resolution cutoff is inconsistent")
    if not (
        normalized_oof["train_decision_cutoff"] < normalized_oof["validation_start"]
    ).all():
        raise ValueError("normalized OOF training cutoff overlaps validation")

    oof_columns = [
        *IDENTITY_COLUMNS,
        CANDIDATE_ID_COLUMN,
        args.residual_label_end_col,
        args.residual_ev_col,
        args.base_ev_col,
        "oof_fold",
        "validation_start",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "train_max_decision_ts",
        "train_fit_cutoff_exclusive",
        "available_at",
    ]
    oof_columns = list(dict.fromkeys(oof_columns))
    normalized_oof = normalized_oof.loc[:, oof_columns].sort_values(
        [*IDENTITY_COLUMNS, CANDIDATE_ID_COLUMN], kind="stable"
    ).reset_index(drop=True)
    normalized_handoff = handoff.loc[
        :, [*IDENTITY_COLUMNS, CANDIDATE_ID_COLUMN, *args.candidate_columns]
    ].sort_values([*IDENTITY_COLUMNS, CANDIDATE_ID_COLUMN], kind="stable").reset_index(
        drop=True
    )

    args.output_oof.parent.mkdir(parents=True, exist_ok=True)
    args.output_candidate_handoff.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    normalized_oof.to_parquet(args.output_oof, index=False, compression="zstd")
    normalized_handoff.to_parquet(
        args.output_candidate_handoff, index=False, compression="zstd"
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "rows": {
            "normalized_oof": int(len(normalized_oof)),
            "normalized_candidate_handoff": int(len(normalized_handoff)),
        },
        "identity_contract": {
            "keys": list(IDENTITY_COLUMNS),
            "timestamp_contract": "timezone-aware UTC exact equality",
            "candidate_id": "symbol|UTC ISO Z timestamp|1h|canonical side",
            "candidate_id_source": "extreme_price_movements.side_aware.candidate_id_series",
            "source_candidate_ids": {
                "residual_oof": CANDIDATE_ID_COLUMN
                in pq.read_schema(args.residual_oof).names,
                "candidate_handoff": CANDIDATE_ID_COLUMN
                in pq.read_schema(args.candidate_handoff).names,
            },
        },
        "legacy_label_resolution_contract": {
            "source": "candidate_handoff.__label_path_end_ts__",
            "residual_oof_column": args.residual_label_end_col,
            "handoff_column": args.handoff_label_end_col,
            "signed_offset_hours": float(args.legacy_label_end_offset_hours),
            "verification": "every exact-joined residual OOF row has residual_label_end = handoff_label_end + signed_offset",
            "documented_reason": "pre-fix DuckDB TIMESTAMPTZ-to-TIMESTAMP host-timezone cast",
        },
        "fold_provenance": fold_audit,
        "compatibility_columns": {
            "oof_fold": "legacy manifest fold index; matches materialize_execution_ev_alpha_oof default fold IDs",
            "validation_start": "manifest test_start",
            "train_decision_cutoff": "maximum reconstructed training-label resolution time; retained separately from train_max_decision_ts",
            "label_resolution_available_at": "maximum reconstructed training-label resolution time",
            "train_max_decision_ts": "actual maximum decision timestamp in the reconstructed training set",
            "train_fit_cutoff_exclusive": "manifest train_end_exclusive used by the residual fit",
            "available_at": "OOF prediction emitted at its decision timestamp",
        },
        "prediction_contract": {
            "preserved_without_recompute": [args.residual_ev_col, args.base_ev_col],
            "residual_ev_column": args.residual_ev_col,
            "base_ev_column": args.base_ev_col,
        },
        "alpha_cost_basis": alpha_cost_basis,
        "candidate_handoff_columns": list(normalized_handoff.columns),
        "source_artifacts": {
            "residual_oof": {
                "path": str(args.residual_oof.resolve()),
                "sha256": _sha256(args.residual_oof),
            },
            "candidate_handoff": {
                "path": str(args.candidate_handoff.resolve()),
                "sha256": _sha256(args.candidate_handoff),
            },
            "residual_manifest": {
                "path": str(args.residual_manifest.resolve()),
                "sha256": _sha256(args.residual_manifest),
            },
        },
        "outputs": {
            "normalized_oof": {
                "path": str(args.output_oof.resolve()),
                "sha256": _sha256(args.output_oof),
            },
            "normalized_candidate_handoff": {
                "path": str(args.output_candidate_handoff.resolve()),
                "sha256": _sha256(args.output_candidate_handoff),
            },
        },
    }
    manifest["provenance_manifest_sha256"] = _canonical_json_hash(manifest)
    _write_json(output_manifest, manifest)
    return {
        "oof": args.output_oof,
        "candidate_handoff": args.output_candidate_handoff,
        "manifest": output_manifest,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-oof", type=Path, required=True)
    parser.add_argument("--candidate-handoff", type=Path, required=True)
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--output-oof", type=Path, required=True)
    parser.add_argument("--output-candidate-handoff", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument(
        "--residual-ev-col", default="score_base_ev_residual_expert_hier_mapped"
    )
    parser.add_argument("--base-ev-col", default="score_base_ev_mapped")
    parser.add_argument("--residual-label-end-col", default="__label_path_end_ts__")
    parser.add_argument("--handoff-label-end-col", default="__label_path_end_ts__")
    parser.add_argument(
        "--legacy-label-end-offset-hours",
        type=float,
        default=2.0,
        help="Signed residual-OFF minus handoff label-resolution offset in hours.",
    )
    parser.add_argument(
        "--candidate-columns",
        type=_parse_columns,
        default=list(DEFAULT_CANDIDATE_COLUMNS),
        help=(
            "Comma-separated handoff columns retained for the downstream materializer; "
            "include its chosen pre-entry leaf-bin columns."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        outputs = run(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"legacy residual OOF normalization failed: {exc}") from exc
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
