#!/usr/bin/env python3
"""Materialize a strict execution-EV alpha OOF adapter without outcomes.

The canonical residual-meta OOF artifact supplies the alpha prediction and all
row-level OOF timing evidence.  It is joined to the candidate handoff exactly
on ``__ts__``, ``__symbol__``, ``side_name``, and ``candidate_id``.  The
residual manifest is used only to validate the supplied OOF fold evidence and
to bound causal leaf support; this adapter never infers a fold or training
cutoff from manifest boundaries.

Example::

  python scripts/materialize_execution_ev_alpha_oof.py \
    --residual-oof residual/oos_predictions.parquet \
    --candidate-handoff candidate_handoff.parquet \
    --residual-manifest residual/manifest.json \
    --leaf-bin-cols base_leaf_bin,meta_leaf_bin \
    --output execution_ev_alpha_oof.parquet
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_auxiliary_lgbm import (  # noqa: E402
    fit_base_archetype_label_feature_contract,
    transform_base_archetype_label_features,
)

SCHEMA = "execution_ev_alpha_oof_v2"
KEYS = ("__ts__", "__symbol__", "side_name")
CANDIDATE_ID_COLUMN = "candidate_id"
JOIN_KEYS = (*KEYS, CANDIDATE_ID_COLUMN)
FORBIDDEN_COLUMN_TOKENS = (
    "realized",
    "future_",
    "label",
    "target",
    "outcome",
    "y_exec",
    "ev_after",
    "ret_net",
    "actual_",
)
FORBIDDEN_BASE_ARCHETYPE_SOURCE_TOKENS = (
    "catboost",
    "path",
    "realized",
    "future_",
    "target",
    "outcome",
    "y_exec",
    "ev_after",
    "ret_net",
    "actual_",
)
DEFAULT_BASE_ARCHETYPE_SOURCE_COLUMNS = (
    "archetype_label_family",
    "archetype_policy_key",
)
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


def _alpha_cost_basis(manifest_path: Path) -> dict[str, Any]:
    """Prove the residual alpha prediction already includes one 1% cost."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"residual_manifest: cannot read cost evidence from {manifest_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("residual_manifest: expected a JSON object for cost proof")

    evidence: list[dict[str, str]] = []
    declared_modes: list[tuple[str, str]] = []
    if "target_mode" in payload:
        declared_modes.append(("target_mode", str(payload["target_mode"])))
    params = payload.get("params")
    if isinstance(params, Mapping) and "target_mode" in params:
        declared_modes.append(("params.target_mode", str(params["target_mode"])))
    feature_selection = payload.get("feature_selection_report")
    if isinstance(feature_selection, Mapping):
        target = feature_selection.get("target")
        if target is not None:
            declared_modes.append(("feature_selection_report.target", str(target)))
    for field, value in declared_modes:
        normalized = value.strip().lower()
        if normalized not in LEGACY_ALPHA_TARGET_MODES:
            raise ValueError(
                "residual_manifest: declared target evidence is inconsistent with "
                f"a 1% residual-net EV target: {field}={value!r}"
            )
        evidence.append({"field": field, "value": value})

    residual_target = payload.get("residual_expert_target")
    if residual_target is not None:
        if not isinstance(residual_target, str):
            raise ValueError(
                "residual_manifest: residual_expert_target must be a string"
            )
        normalized = residual_target.strip().lower()
        if "ev_after_1pct" not in normalized or not any(
            token in normalized for token in ("residual", "-", "minus")
        ):
            raise ValueError(
                "residual_manifest: residual_expert_target is not an explicit "
                "residual ev_after_1pct target"
            )
        evidence.append(
            {"field": "residual_expert_target", "value": residual_target}
        )
    if not evidence:
        raise ValueError(
            "residual_manifest: explicit 1% residual-net target evidence is required"
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
            "source manifest explicitly declares a residual target in "
            "ev_after_1pct units; that target contains exactly one 1% "
            "round-trip cost"
        ),
    }


def _canonical_json_hash(
    payload: Mapping[str, Any], *, excluded: Sequence[str] = ()
) -> str:
    excluded_keys = set(excluded)
    canonical = {
        str(key): _json_safe(value)
        for key, value in payload.items()
        if key not in excluded_keys
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _parse_columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise argparse.ArgumentTypeError(
            "at least one pre-entry leaf-bin column is required"
        )
    if len(columns) != len(set(columns)):
        raise argparse.ArgumentTypeError("pre-entry leaf-bin columns must be unique")
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
    if not result.isin(("long", "short")).all():
        invalid = sorted(set(result) - {"long", "short"})
        raise ValueError(
            f"{source}: {column!r} must contain canonical long/short sides; "
            f"invalid={invalid[:5]!r}"
        )
    return result


def _assert_unique(
    frame: pd.DataFrame,
    *,
    source: str,
    keys: Sequence[str] = JOIN_KEYS,
) -> None:
    duplicate_rows = int(frame.duplicated(list(keys), keep=False).sum())
    if duplicate_rows:
        raise ValueError(
            f"{source}: duplicate rows violate exact identity on {list(keys)!r}; "
            f"duplicate_rows={duplicate_rows}"
        )


def _required_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, source: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {', '.join(missing)}")


def _require_pre_entry_column(column: str) -> None:
    name = column.lower()
    if any(token in name for token in FORBIDDEN_COLUMN_TOKENS):
        raise ValueError(
            f"pre-entry leaf-bin column {column!r} appears outcome-derived; "
            "only pre-entry candidate-handoff columns are allowed"
        )


def _require_base_archetype_source_column(column: str) -> None:
    name = column.lower()
    if any(token in name for token in FORBIDDEN_BASE_ARCHETYPE_SOURCE_TOKENS):
        raise ValueError(
            f"base archetype source column {column!r} is not an allowed pre-entry "
            "base archetype identity; CatBoost and path outcome labels are forbidden"
        )


def _validate_base_archetype_sources(
    frame: pd.DataFrame, columns: Sequence[str]
) -> None:
    for column in columns:
        _strings(frame[column], source="candidate_handoff", column=column)


def _validate_leaf_bins(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = frame[column]
        if values.isna().any():
            raise ValueError(
                f"candidate_handoff: leaf-bin column {column!r} contains missing rows"
            )
        if pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise ValueError(
                    f"candidate_handoff: leaf-bin column {column!r} contains non-finite rows"
                )
        elif values.astype("string").str.strip().eq("").any():
            raise ValueError(
                f"candidate_handoff: leaf-bin column {column!r} contains blank rows"
            )


def _normalise_identity(
    frame: pd.DataFrame,
    *,
    source: str,
    timestamp_col: str,
    symbol_col: str,
    side_col: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": _utc(frame[timestamp_col], source=source, column=timestamp_col),
            "__symbol__": _strings(frame[symbol_col], source=source, column=symbol_col),
            "side_name": _sides(frame[side_col], source=source, column=side_col),
        }
    )


def _load_oof(args: argparse.Namespace) -> pd.DataFrame:
    source = "residual_oof"
    raw = pd.read_parquet(args.residual_oof)
    if raw.empty:
        raise ValueError(f"{source}: source parquet is empty")
    _required_columns(
        raw,
        [
            args.oof_timestamp_col,
            args.oof_symbol_col,
            args.oof_side_col,
            args.oof_candidate_id_col,
            args.oof_fold_col,
            args.oof_validation_start_col,
            args.oof_train_decision_cutoff_col,
            args.oof_label_resolution_available_at_col,
            args.oof_available_at_col,
            args.residual_ev_col,
            args.base_ev_col,
        ],
        source=source,
    )
    out = _normalise_identity(
        raw,
        source=source,
        timestamp_col=args.oof_timestamp_col,
        symbol_col=args.oof_symbol_col,
        side_col=args.oof_side_col,
    )
    out[CANDIDATE_ID_COLUMN] = _strings(
        raw[args.oof_candidate_id_col], source=source, column=args.oof_candidate_id_col
    ).to_numpy()
    out["oof_fold"] = _strings(
        raw[args.oof_fold_col], source=source, column=args.oof_fold_col
    ).to_numpy()
    for input_column, output_column in (
        (args.oof_validation_start_col, "validation_start"),
        (args.oof_train_decision_cutoff_col, "train_decision_cutoff"),
        (args.oof_label_resolution_available_at_col, "label_resolution_available_at"),
        (args.oof_available_at_col, "available_at"),
    ):
        out[output_column] = _utc(
            raw[input_column], source=source, column=input_column
        ).to_numpy()
    if (out["available_at"] > out["__ts__"]).any():
        raise ValueError(
            f"{source}: feature availability is after the decision timestamp"
        )
    if not (out["train_decision_cutoff"] < out["validation_start"]).all():
        raise ValueError(
            f"{source}: train decision cutoff must be strictly before validation start"
        )
    if not (out["validation_start"] <= out["__ts__"]).all():
        raise ValueError(
            f"{source}: validation start is after an OOF decision timestamp"
        )
    if not (out["train_decision_cutoff"] < out["__ts__"]).all():
        raise ValueError(
            f"{source}: train decision cutoff must be strictly before decision"
        )
    if not (out["label_resolution_available_at"] <= out["train_decision_cutoff"]).all():
        raise ValueError(
            f"{source}: training labels must resolve before train decision cutoff availability"
        )
    for input_column, output_column in (
        (args.residual_ev_col, "existing_alpha_ev"),
        (args.base_ev_col, "base_alpha_ev"),
    ):
        values = pd.to_numeric(raw[input_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"{source}: {input_column!r} contains missing or non-finite rows"
            )
        out[output_column] = values
    _assert_unique(out, source=source)
    return out


def _load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    source = "candidate_handoff"
    raw = pd.read_parquet(args.candidate_handoff)
    if raw.empty:
        raise ValueError(f"{source}: source parquet is empty")
    _required_columns(
        raw,
        [
            args.candidate_timestamp_col,
            args.candidate_symbol_col,
            args.candidate_side_col,
            args.candidate_id_col,
            *args.leaf_bin_cols,
            *args.base_archetype_source_cols,
            *(
                [args.candidate_available_at_col]
                if args.candidate_available_at_col
                else []
            ),
        ],
        source=source,
    )
    out = _normalise_identity(
        raw,
        source=source,
        timestamp_col=args.candidate_timestamp_col,
        symbol_col=args.candidate_symbol_col,
        side_col=args.candidate_side_col,
    )
    out[CANDIDATE_ID_COLUMN] = _strings(
        raw[args.candidate_id_col], source=source, column=args.candidate_id_col
    ).to_numpy()
    for column in args.leaf_bin_cols:
        out[column] = raw[column].to_numpy()
    _validate_leaf_bins(out, args.leaf_bin_cols)
    for column in args.base_archetype_source_cols:
        out[column] = _strings(raw[column], source=source, column=column).to_numpy()
    _validate_base_archetype_sources(out, args.base_archetype_source_cols)
    if args.candidate_available_at_col:
        available_at = _utc(
            raw[args.candidate_available_at_col],
            source=source,
            column=args.candidate_available_at_col,
        )
        if (available_at > out["__ts__"]).any():
            raise ValueError(
                "candidate_handoff: pre-entry leaf-bin availability is after its decision timestamp"
            )
    _assert_unique(out, source=source)
    return out


def _parse_timestamp(value: Any, *, fold: int, name: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"residual_manifest: fold {fold} has invalid {name!r}")
    return pd.Timestamp(timestamp)


def _folds(manifest_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"residual_manifest: cannot read valid JSON from {manifest_path}"
        ) from exc
    rows = payload.get("folds")
    if not isinstance(rows, list) or not rows:
        raise ValueError("residual_manifest: non-empty 'folds' list is required")
    folds: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"residual_manifest: fold {index} must be an object")
        if "test_start" not in row:
            raise ValueError(f"residual_manifest: fold {index} is missing 'test_start'")
        end_column = "test_end_exclusive" if "test_end_exclusive" in row else "test_end"
        if end_column not in row:
            raise ValueError(
                f"residual_manifest: fold {index} needs 'test_end_exclusive' or 'test_end'"
            )
        start = _parse_timestamp(row["test_start"], fold=index, name="test_start")
        end = _parse_timestamp(row[end_column], fold=index, name=end_column)
        if end <= start:
            raise ValueError(
                f"residual_manifest: fold {index} has non-positive test interval"
            )
        fold_id = str(row.get("fold_id", index))
        if fold_id in identifiers:
            raise ValueError(f"residual_manifest: duplicate fold_id {fold_id!r}")
        identifiers.add(fold_id)
        folds.append(
            {
                "oof_fold": fold_id,
                "test_start": start,
                "test_end_exclusive": end,
                "end_source": end_column,
            }
        )
    folds.sort(key=lambda row: row["test_start"])
    for previous, current in zip(folds, folds[1:]):
        if current["test_start"] < previous["test_end_exclusive"]:
            raise ValueError("residual_manifest: test fold boundaries overlap")
    return folds


def _validate_oof_fold_provenance(
    oof: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
) -> None:
    by_id = {str(fold["oof_fold"]): fold for fold in folds}
    unknown = sorted(set(oof["oof_fold"]).difference(by_id))
    if unknown:
        raise ValueError(
            "residual_oof: row-level OOF fold IDs are not declared by residual_manifest: "
            f"{unknown[:5]!r}"
        )
    for fold_id, rows in oof.groupby("oof_fold", sort=False):
        fold = by_id[str(fold_id)]
        if not rows["validation_start"].eq(fold["test_start"]).all():
            raise ValueError(
                f"residual_oof: row-level validation_start does not match manifest test_start "
                f"for OOF fold {fold_id!r}"
            )
        if not (
            rows["__ts__"].ge(fold["test_start"])
            & rows["__ts__"].lt(fold["test_end_exclusive"])
        ).all():
            raise ValueError(
                f"residual_oof: row-level OOF fold {fold_id!r} conflicts with manifest boundaries"
            )


def _leaf_support(
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
    leaf_bin_cols: Sequence[str],
) -> pd.DataFrame:
    result = scored.copy()
    result["alpha_leaf_tuple_support_log1p"] = np.float32(0.0)
    result["alpha_leaf_individual_support_log1p_min"] = np.float32(0.0)
    result["alpha_leaf_support"] = np.float32(0.0)
    for fold in folds:
        test_mask = result["oof_fold"].eq(fold["oof_fold"])
        if not test_mask.any():
            continue
        context = candidates.loc[
            candidates["__ts__"].lt(fold["test_start"]), list(leaf_bin_cols)
        ]
        test = result.loc[test_mask, list(leaf_bin_cols)]
        if context.empty:
            tuple_counts = np.zeros(len(test), dtype=np.int64)
            individual_min = np.zeros(len(test), dtype=np.int64)
        else:
            tuple_count_frame = (
                context.groupby(list(leaf_bin_cols), sort=False, dropna=False)
                .size()
                .rename("__tuple_support__")
                .reset_index()
            )
            tuple_counts = (
                test.merge(
                    tuple_count_frame, on=list(leaf_bin_cols), how="left", sort=False
                )["__tuple_support__"]
                .fillna(0)
                .to_numpy(dtype=np.int64)
            )
            individual = []
            for column in leaf_bin_cols:
                counts = (
                    context.groupby(column, sort=False, dropna=False)
                    .size()
                    .rename("__support__")
                    .reset_index()
                )
                individual.append(
                    test.merge(counts, on=column, how="left", sort=False)["__support__"]
                    .fillna(0)
                    .to_numpy(dtype=np.int64)
                )
            individual_min = np.minimum.reduce(individual)
        conservative = np.minimum(tuple_counts, individual_min)
        result.loc[test_mask, "alpha_leaf_tuple_support_log1p"] = np.log1p(
            tuple_counts
        ).astype(np.float32)
        result.loc[test_mask, "alpha_leaf_individual_support_log1p_min"] = np.log1p(
            individual_min
        ).astype(np.float32)
        result.loc[test_mask, "alpha_leaf_support"] = np.log1p(conservative).astype(
            np.float32
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output.exists():
        raise ValueError(
            f"refusing to overwrite existing output parquet: {args.output}"
        )
    manifest_output = args.output_manifest or args.output.with_suffix(".manifest.json")
    if manifest_output.exists():
        raise ValueError(
            f"refusing to overwrite existing output manifest: {manifest_output}"
        )
    for column in args.leaf_bin_cols:
        _require_pre_entry_column(column)
    for column in args.base_archetype_source_cols:
        _require_base_archetype_source_column(column)
    if args.base_archetype_canonical_source not in args.base_archetype_source_cols:
        raise ValueError(
            "base archetype canonical source must be included in base archetype source columns"
        )

    oof = _load_oof(args)
    candidates = _load_candidates(args)
    base_archetype_label_feature_contract = fit_base_archetype_label_feature_contract(
        candidates,
        source_columns=args.base_archetype_source_cols,
        canonical_source=args.base_archetype_canonical_source,
    )
    joined = oof.merge(
        candidates,
        on=list(JOIN_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = int(joined["_merge"].ne("both").sum())
        raise ValueError(
            f"candidate_handoff: {missing} residual OOF rows have no exact candidate identity match"
        )
    joined = joined.drop(columns="_merge")
    base_archetype_label_features = transform_base_archetype_label_features(
        joined,
        base_archetype_label_feature_contract,
    )
    if base_archetype_label_features.dtypes.ne(np.dtype(np.float32)).any():
        raise ValueError("base archetype one-hot transform must emit float32 features")
    joined = pd.concat([joined, base_archetype_label_features], axis=1)
    joined = joined.drop(columns=list(args.base_archetype_source_cols))
    folds = _folds(args.residual_manifest)
    _validate_oof_fold_provenance(joined, folds)
    joined["alpha_prediction_uncertainty"] = np.abs(
        joined["existing_alpha_ev"].to_numpy(dtype=float)
        - joined["base_alpha_ev"].to_numpy(dtype=float)
    )
    joined = _leaf_support(candidates, joined, folds, args.leaf_bin_cols)
    if not np.isfinite(
        joined[
            [
                "existing_alpha_ev",
                "base_alpha_ev",
                "alpha_prediction_uncertainty",
                "alpha_leaf_tuple_support_log1p",
                "alpha_leaf_individual_support_log1p_min",
                "alpha_leaf_support",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "materialized alpha adapter contains non-finite required values"
        )
    _assert_unique(joined, source="materialized alpha adapter")
    joined = joined.sort_values(list(JOIN_KEYS), kind="stable").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "rows": int(len(joined)),
        "columns": list(joined.columns),
        "alpha_cost_basis": _alpha_cost_basis(args.residual_manifest),
        "join": {
            "mode": "exact_inner_one_to_one",
            "keys": list(JOIN_KEYS),
            "timestamp_contract": "normalized timezone-aware UTC exact equality",
        },
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
        "definitions": {
            "existing_alpha_ev": (
                f"canonical residual OOF prediction column {args.residual_ev_col!r}; "
                "pre-entry alpha EV OOF prediction, never a supervised target"
            ),
            "base_alpha_ev": f"base mapped EV column {args.base_ev_col!r}; retained only for audit",
            "alpha_prediction_uncertainty": "abs(existing_alpha_ev - base_alpha_ev)",
            "candidate_id": (
                f"canonical residual OOF identity column {args.oof_candidate_id_col!r}; "
                "validated by exact candidate-handoff identity join"
            ),
            "available_at": (
                f"canonical residual OOF availability column {args.oof_available_at_col!r}; "
                "required to be no later than the decision timestamp"
            ),
            "oof_fold": (
                f"canonical residual OOF fold column {args.oof_fold_col!r}; validated against "
                "the residual manifest and never inferred by this adapter"
            ),
            "validation_start": (
                f"canonical residual OOF validation-start column {args.oof_validation_start_col!r}; "
                "validated against the declared OOF fold"
            ),
            "train_decision_cutoff": (
                f"canonical residual OOF cutoff column {args.oof_train_decision_cutoff_col!r}; "
                "retained without inference"
            ),
            "label_resolution_available_at": (
                "canonical residual OOF training-label resolution availability column "
                f"{args.oof_label_resolution_available_at_col!r}; retained without inference"
            ),
            "alpha_leaf_support": (
                "log1p(min(tuple support, minimum individual leaf-bin support)); each support "
                "count uses only candidate-handoff rows with __ts__ strictly before that fold test_start"
            ),
            "alpha_leaf_tuple_support_log1p": "log1p(pre-fold exact joint leaf-bin tuple count)",
            "alpha_leaf_individual_support_log1p_min": "log1p(minimum pre-fold individual leaf-bin count)",
            "leaf_bin_columns": list(args.leaf_bin_cols),
            "base_archetype_label_features": (
                "float32 one-hot base archetype identities encoded with the shared "
                "path_auxiliary_lgbm contract"
            ),
            "base_archetype_label_feature_contract": base_archetype_label_feature_contract,
            "outcome_contract": (
                "no outcome, realized-return, target, CatBoost, or path label columns are used or "
                "emitted; approved pre-entry base archetype identities are encoded as one-hots"
            ),
        },
        "folds": [
            {
                "oof_fold": fold["oof_fold"],
                "test_start": fold["test_start"],
                "test_end_exclusive": fold["test_end_exclusive"],
                "test_end_source": fold["end_source"],
                "oof_rows": int(joined["oof_fold"].eq(fold["oof_fold"]).sum()),
                "context_rows": int(candidates["__ts__"].lt(fold["test_start"]).sum()),
            }
            for fold in folds
        ],
        "output_sha256": _sha256(args.output),
        "source_artifact_sha256": _sha256(args.output),
        "prediction_role": "alpha_ev_oof",
        "prediction_columns": {
            "existing_alpha_ev": {
                "role": "pre_entry_alpha_ev_oof_prediction",
                "target": False,
            }
        },
    }
    manifest["prediction_role_manifest_sha256"] = _canonical_json_hash(manifest)
    _write_json(manifest_output, manifest)
    return {"output": args.output, "manifest": manifest_output}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-oof", type=Path, required=True)
    parser.add_argument("--candidate-handoff", type=Path, required=True)
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument(
        "--residual-ev-col", default="score_base_ev_residual_expert_hier_mapped"
    )
    parser.add_argument("--base-ev-col", default="score_base_ev_mapped")
    parser.add_argument("--leaf-bin-cols", type=_parse_columns, required=True)
    parser.add_argument(
        "--base-archetype-source-cols",
        type=_parse_columns,
        default=list(DEFAULT_BASE_ARCHETYPE_SOURCE_COLUMNS),
        help="Comma-separated pre-entry base archetype identity columns from candidate handoff.",
    )
    parser.add_argument(
        "--base-archetype-canonical-source",
        default="archetype_label_family",
        help="Source column whose one-hots define the canonical base archetype identity.",
    )
    parser.add_argument("--oof-timestamp-col", default="__ts__")
    parser.add_argument("--oof-symbol-col", default="__symbol__")
    parser.add_argument("--oof-side-col", default="side_name")
    parser.add_argument("--oof-candidate-id-col", default=CANDIDATE_ID_COLUMN)
    parser.add_argument("--oof-fold-col", default="oof_fold")
    parser.add_argument("--oof-validation-start-col", default="validation_start")
    parser.add_argument(
        "--oof-train-decision-cutoff-col", default="train_decision_cutoff"
    )
    parser.add_argument(
        "--oof-label-resolution-available-at-col",
        default="label_resolution_available_at",
    )
    parser.add_argument("--oof-available-at-col", default="available_at")
    parser.add_argument("--candidate-timestamp-col", default="__ts__")
    parser.add_argument("--candidate-symbol-col", default="__symbol__")
    parser.add_argument("--candidate-side-col", default="side_name")
    parser.add_argument("--candidate-id-col", default=CANDIDATE_ID_COLUMN)
    parser.add_argument(
        "--candidate-available-at-col",
        default=None,
        help="Optional handoff availability timestamp; must be no later than its __ts__.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        paths = run(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"execution-EV alpha OOF materialization failed: {exc}"
        ) from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
