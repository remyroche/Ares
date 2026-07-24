#!/usr/bin/env python3
"""Evaluate and fit EV-mapped side residual experts.

The canonical architecture uses the base score directly. Independent long and
short experts predict only the residual net EV left after a train-only
hierarchical side/archetype EV map. Their outputs therefore remain comparable
in the global auction. The historical shared-meta backbone remains available
as an explicit ablation.

Feature selection and correction-strength tuning are performed once on the
March 2026 calibration window.  April and later months use expanding training
windows with the selected contracts and strengths frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import lgbm_pipeline as staged_lgbm  # noqa: E402
from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.data_store import _feature_schema_names  # noqa: E402
from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    AE_GMM_FEATURE_COLUMNS,
)
from extreme_price_movements.supervised_market_state_calibration import (  # noqa: E402
    expected_ev_rank,
    fit_hierarchical_ev_calibrator,
    predict_hierarchical_ev,
)
from extreme_price_movements.training_utils import get_meta_feature_keys  # noqa: E402
from scripts.report_s52_trailing_regime_meta_handoff import (  # noqa: E402
    _read_feature_store_symbol_context_batch,
)

KEYS = ("__ts__", "__symbol__", "side_name")
ANCHORS = (
    "score",
    "base_score_rank_pct_train_prior",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
)
AEGMM_CANDIDATES = tuple(AE_GMM_FEATURE_COLUMNS)
FORBIDDEN_CONTEXT_PREFIXES = ("rel_rankband_", "rel_marginband_")
META_OBSERVABLE_CONTEXT_PREFIXES = (
    "base_reliability_",
    "meta_long_aegmm_",
    "meta_short_aegmm_",
    "meta_market_aegmm_",
)
# This is deliberately narrower than AE_GMM_FEATURE_COLUMNS: the latter are
# incumbent base-state inputs.  The prefixes below identify only the new
# causal base-reliability and meta-local AE/GMM block, allowing a fair feature
# ablation without removing the production V9 context contract.
NEW_META_STATE_CONTEXT_PREFIXES = META_OBSERVABLE_CONTEXT_PREFIXES
META_AEGMM_CONTEXT_PREFIXES = (
    "meta_long_aegmm_",
    "meta_short_aegmm_",
    "meta_market_aegmm_",
)
BASE_RELIABILITY_CONTEXT_PREFIXES = ("base_reliability_",)
DEFAULT_GLOBAL_PREDICTIONS = Path(
    "data_perp/reports/meta_v9_recovery_20260713/"
    "anchored_oldparams_fullhistory_oos_v1/"
    "s52_train_meta_regime_handoff_smoke_predictions.parquet"
)
DEFAULT_HANDOFF = Path(
    "data_perp/reports/"
    "s59_h5_fullthroughjul10_base_configfull_freshmda_fixedparams_wf30_20260713/"
    "meta_handoff_top30_allsafe_aegmmfull_fullcoverage_20260714/"
    "train_meta_regime_handoff.parquet"
)
DEFAULT_SCORED_LEDGER = DEFAULT_HANDOFF.with_name(
    "s52_trailing_regime_scored_ledger.parquet"
)
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260711_070000")
DEFAULT_CONTRACT = Path(
    "extreme_price_movements/config/"
    "meta_v9_anchor_oldparams_residual_backbone_v1.json"
)
DEFAULT_OUT = Path(
    "data_perp/reports/meta_v9_recovery_20260713/"
    "ev_mapped_side_base_residual_expert_staged_mda_hpo_ablation_v1"
)
DEFAULT_CONTROL = Path(
    "data_perp/reports/meta_v9_recovery_20260713/"
    "ev_mapped_side_base_residual_expert_canonical_v1"
)
HISTORICAL_V9_REFERENCE = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v6_15mchart_base_frozenfs_fixedparams_"
    "may_july_combined_20260708/"
    "train_meta_frozenfs_fixedparams_train_may_june_score_july_"
    "20260709_savedmodels"
)
HISTORICAL_V9_CODE_REVISION = "f27913380f"
OOS_FOLD_BUNDLE_SCHEMA = "side_base_residual_expert_oos_fold_bundle_v1"
SELECTION_HPO_CONTRACT_SCHEMA = (
    "side_base_residual_expert_selection_hpo_contract_v1"
)
SELECTION_HPO_MANIFEST_FILENAME = "staged_selection_hpo_manifest.json"
SELECTION_HPO_COMPLETED_STATUS = "selection_hpo_complete"
SELECTION_HPO_ALGORITHM = "staged_side_local_mda_and_hpo_v1"


def _quoted(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_iso(value: str | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _file_input_identity(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Selection/HPO input file does not exist: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _feature_store_input_identity(
    feature_dir: Path,
    feature_store_schema: Iterable[str],
) -> dict[str, Any]:
    """Return a cheap, deterministic identity for logical-store selection inputs."""
    feature_dir = Path(feature_dir)
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Feature store directory does not exist: {feature_dir}")
    inventory: list[dict[str, Any]] = []
    for path in sorted(feature_dir.rglob("*")):
        if not path.is_file() or path.name.endswith((".lock", ".wal", ".shm")):
            continue
        stat = path.stat()
        inventory.append(
            {
                "path": str(path.relative_to(feature_dir)),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return {
        "path": str(feature_dir.resolve()),
        "logical_schema_sha256": _stable_json_sha256(sorted(map(str, feature_store_schema))),
        "inventory_sha256": _stable_json_sha256(inventory),
        "file_count": len(inventory),
    }


def _selection_hpo_contract(
    *,
    source_mode: str,
    handoff: Path,
    scored_ledger: Path,
    global_predictions: Path,
    feature_dir: Path,
    feature_store_schema: Iterable[str],
    backbone_contract: Path,
    candidate_context: Iterable[str],
    backbone_score: str,
    backbone_score_col: str,
    calibration_start: pd.Timestamp,
    calibration_end: pd.Timestamp,
    eval_start: str,
    eval_end: str,
    selection_mode: str,
    context_contract_only: bool,
    excluded_context_prefixes: Iterable[str],
    selection_max_rows: int,
    hpo_max_rows: int,
    hpo_trials: int,
    hpo_patience: int,
    seed: int,
    fixed_hpo_params_manifest: Path | None,
) -> dict[str, Any]:
    """Build the immutable input contract for side-local selection and HPO."""
    source_inputs: dict[str, Any] = {"handoff": _file_input_identity(handoff)}
    if source_mode == "current_handoff":
        source_inputs["scored_ledger"] = _file_input_identity(scored_ledger)
    elif source_mode == "legacy_intersection":
        source_inputs["global_predictions"] = _file_input_identity(global_predictions)
    else:
        raise ValueError(f"Unsupported selection source mode: {source_mode}")
    inputs = {
        "algorithm": SELECTION_HPO_ALGORITHM,
        "source_mode": str(source_mode),
        "source_inputs": source_inputs,
        "feature_store": _feature_store_input_identity(
            feature_dir, feature_store_schema
        ),
        "backbone_contract": _file_input_identity(backbone_contract),
        "candidate_context": list(dict.fromkeys(map(str, candidate_context))),
        "target_and_backbone": {
            "target": "residual_net_ev_after_1pct",
            "backbone_score": str(backbone_score),
            "backbone_score_col": str(backbone_score_col),
            "protected_anchors": list(ANCHORS),
            "archetype_contract": "side_name__archetype_policy_key",
        },
        "selection_hpo_settings": {
            "selection_mode": str(selection_mode),
            "context_contract_only": bool(context_contract_only),
            "excluded_context_prefixes": list(map(str, excluded_context_prefixes)),
            "selection_max_rows": int(selection_max_rows),
            "hpo_max_rows": int(hpo_max_rows),
            "hpo_trials": int(hpo_trials),
            "hpo_patience": int(hpo_patience),
            "seed": int(seed),
            "fixed_hpo_params_manifest": (
                _file_input_identity(fixed_hpo_params_manifest)
                if fixed_hpo_params_manifest is not None
                else None
            ),
        },
        "dates": {
            "calibration_start": calibration_start.isoformat(),
            "calibration_end": calibration_end.isoformat(),
            "eval_start": _utc_iso(eval_start),
            "eval_end": _utc_iso(eval_end),
        },
    }
    return {
        "schema": SELECTION_HPO_CONTRACT_SCHEMA,
        "fingerprint": _stable_json_sha256(inputs),
        "fingerprint_inputs": inputs,
    }


def _read_completed_selection_hpo_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Selection/HPO manifest is not an object: {path}")
    required = {
        "schema",
        "status",
        "fingerprint",
        "fingerprint_inputs",
        "selected_features",
        "hpo_params",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "Selection/HPO manifest lacks the strict reusable contract fields: "
            + ", ".join(missing)
        )
    if payload["schema"] != SELECTION_HPO_CONTRACT_SCHEMA:
        raise ValueError(f"Selection/HPO manifest has unsupported schema: {path}")
    if payload["status"] != SELECTION_HPO_COMPLETED_STATUS:
        raise ValueError(f"Selection/HPO manifest is not complete: {path}")
    if not isinstance(payload["fingerprint"], str) or not payload["fingerprint"]:
        raise ValueError(f"Selection/HPO manifest lacks a strict fingerprint: {path}")
    if not isinstance(payload["fingerprint_inputs"], dict):
        raise ValueError(f"Selection/HPO manifest has invalid fingerprint inputs: {path}")
    if _stable_json_sha256(payload["fingerprint_inputs"]) != payload["fingerprint"]:
        raise ValueError(
            f"Selection/HPO manifest fingerprint does not match its inputs: {path}"
        )
    for side in ("long", "short"):
        features = dict(payload["selected_features"]).get(side)
        params = dict(payload["hpo_params"]).get(side)
        if not isinstance(features, list) or not all(
            isinstance(feature, str) for feature in features
        ):
            raise ValueError(f"Selection/HPO manifest has invalid {side} features: {path}")
        if not isinstance(params, dict) or not params:
            raise ValueError(f"Selection/HPO manifest has invalid {side} HPO params: {path}")
    return payload


def _find_reusable_selection_hpo_manifest(
    expected: dict[str, Any],
    *,
    output_dir: Path,
    manifest_path: Path | None,
    force: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Reuse only an exact, completed, fingerprinted staged-selection contract."""
    expected_fingerprint = str(expected["fingerprint"])
    if force:
        return None, {"mode": "forced_rerun", "reused": False}
    if manifest_path is not None:
        path = Path(manifest_path)
        if path.is_dir():
            path = path / SELECTION_HPO_MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Selection/HPO manifest does not exist: {path}")
        contract = _read_completed_selection_hpo_manifest(path)
        if (
            contract["fingerprint"] != expected_fingerprint
            or contract["fingerprint_inputs"] != expected["fingerprint_inputs"]
        ):
            raise ValueError(
                "Explicit selection/HPO manifest fingerprint does not match the "
                "current source, feature-store, candidate, target, date, and HPO contract"
            )
        return contract, {
            "mode": "explicit_manifest",
            "reused": True,
            "path": str(path),
        }

    root = Path(output_dir).parent
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    mismatched_candidates = 0
    legacy_or_invalid_candidates = 0
    if root.is_dir():
        for path in root.glob(f"*/{SELECTION_HPO_MANIFEST_FILENAME}"):
            if path.parent.resolve() == Path(output_dir).resolve():
                continue
            try:
                contract = _read_completed_selection_hpo_manifest(path)
            except (OSError, ValueError, json.JSONDecodeError):
                legacy_or_invalid_candidates += 1
                continue
            if (
                contract["fingerprint"] == expected_fingerprint
                and contract["fingerprint_inputs"] == expected["fingerprint_inputs"]
            ):
                candidates.append((path.stat().st_mtime, path, contract))
            else:
                mismatched_candidates += 1
    provenance = {
        "mode": "automatic_sibling_reports",
        "reused": bool(candidates),
        "registry_root": str(root),
        "candidates": len(candidates),
        "mismatched_candidates": mismatched_candidates,
        "legacy_or_invalid_candidates": legacy_or_invalid_candidates,
    }
    if not candidates:
        return None, provenance
    _, path, contract = max(candidates, key=lambda item: (item[0], str(item[1])))
    provenance["path"] = str(path)
    return contract, provenance


def _write_selection_hpo_manifest(out_dir: Path, report: dict[str, Any]) -> Path:
    path = Path(out_dir) / SELECTION_HPO_MANIFEST_FILENAME
    path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _contract_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("selected_feature_union") or []
    return [str(value) for value in values]


def _context_candidates(contract_features: Iterable[str], available: set[str]) -> list[str]:
    values: list[str] = []
    for feature in (*contract_features, *AEGMM_CANDIDATES):
        feature = str(feature)
        if feature in ANCHORS or feature not in available:
            continue
        if feature.startswith(FORBIDDEN_CONTEXT_PREFIXES):
            continue
        values.append(feature)
    return list(dict.fromkeys(values))


def _logical_feature_store_schema(feature_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(Path(feature_dir).glob("symbol=*.parquet")):
        names.update(_feature_schema_names(str(path)))
    return names


def _augment_from_feature_store(
    frame: pd.DataFrame,
    feature_dir: Path,
    requested_features: Iterable[str],
) -> pd.DataFrame:
    """Join only requested point-in-time columns from the logical store."""
    requested = [
        str(feature)
        for feature in dict.fromkeys(requested_features)
        if str(feature) not in frame.columns
    ]
    if not requested:
        return frame
    key_frame = frame.loc[:, list(KEYS)].copy()
    key_frame["__ts__"] = pd.to_datetime(
        key_frame["__ts__"], utc=True, errors="coerce"
    )
    key_frame["__symbol__"] = key_frame["__symbol__"].astype(str)
    symbols = key_frame["__symbol__"].drop_duplicates().astype(str).tolist()
    parts: list[pd.DataFrame] = []
    fallback_symbols = 0
    try:
        feature_store_ts = pd.to_datetime(
            feature_dir.name, format="%Y%m%d_%H%M%S", utc=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid static feature-store directory: {feature_dir}") from exc
    data_root = feature_dir.parent.parent
    # Read the canonical logical store in bounded multi-symbol batches.  The
    # former implementation invoked the same endpoint separately for every
    # symbol and feature subset, which made a 15-arm ablation spend most of
    # its time reopening tiny Parquet projections.  The logical API applies
    # the same static/delta coalescing contract as inference; batching changes
    # only I/O granularity, not feature values.
    wanted_by_symbol = {
        symbol: key_frame.loc[
            key_frame["__symbol__"].eq(symbol), ["__ts__", "__symbol__"]
        ].drop_duplicates()
        for symbol in symbols
    }
    # A chronological beginning/middle/end selection sample spans most of the
    # historical range.  Keep the feature-store working set bounded before the
    # requested timestamps are joined back, otherwise a modest 45k-row sample
    # can transiently materialize every history row for 32 symbols.
    batch_size = max(1, int(os.getenv("EPM_META_FEATURE_BATCH_SIZE", "8")))
    for offset in range(0, len(symbols), batch_size):
        batch_symbols = symbols[offset : offset + batch_size]
        batch_wanted = [wanted_by_symbol[symbol] for symbol in batch_symbols]
        start_ts = min(part["__ts__"].min() for part in batch_wanted)
        end_ts = max(part["__ts__"].max() for part in batch_wanted)
        loaded = _read_feature_store_symbol_context_batch(
            feature_store_ts=feature_store_ts,
            data_root=data_root,
            symbols=batch_symbols,
            columns=requested,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        for symbol in batch_symbols:
            wanted = wanted_by_symbol[symbol]
            part = loaded.get(symbol, pd.DataFrame())
            if part.empty:
                continue
            part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
            part["__symbol__"] = part["__symbol__"].astype(str)
            part = part.merge(
                wanted,
                on=["__ts__", "__symbol__"],
                how="inner",
                validate="one_to_one",
                copy=False,
            )
            if not part.empty:
                parts.append(part)
    if not parts:
        return frame
    context = pd.concat(parts, ignore_index=True, copy=False)
    left = frame.copy(deep=False)
    left["__ts__"] = pd.to_datetime(left["__ts__"], utc=True, errors="coerce")
    left["__symbol__"] = left["__symbol__"].astype(str)
    merged = left.merge(
        context,
        on=["__ts__", "__symbol__"],
        how="left",
        validate="many_to_one",
        copy=False,
    )
    merged["__ts__"] = pd.to_datetime(merged["__ts__"], utc=True, errors="coerce")
    return merged

def _load_joined_with_feature_store(
    global_predictions: Path,
    handoff: Path,
    feature_dir: Path,
    context_features: list[str],
    handoff_schema: set[str],
    **kwargs: Any,
) -> pd.DataFrame:
    embedded = [feature for feature in context_features if feature in handoff_schema]
    frame = _load_joined(
        global_predictions,
        handoff,
        embedded,
        **kwargs,
    )
    return _augment_from_feature_store(frame, feature_dir, context_features)


def _empirical_percentile(
    values: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Map scores through a frozen train-only empirical distribution."""
    output = np.full(len(values), 0.5, dtype=np.float32)
    finite = np.isfinite(values)
    reference = np.sort(reference[np.isfinite(reference)].astype(np.float32, copy=False))
    if not len(reference):
        return output
    left = np.searchsorted(reference, values[finite], side="left")
    right = np.searchsorted(reference, values[finite], side="right")
    output[finite] = ((left + right) / (2.0 * len(reference))).astype(np.float32)
    return output


def _add_frozen_base_train_prior_rank(
    frame: pd.DataFrame,
    *,
    fit_end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Build the missing base-rank anchor from the authorized train period."""
    score = pd.to_numeric(frame["score"], errors="coerce").to_numpy(np.float32)
    fit_mask = frame["__ts__"].lt(fit_end_exclusive).to_numpy()
    frame["base_score_rank_pct_train_prior"] = _empirical_percentile(
        score,
        score[fit_mask],
    )
    return frame


def _load_current_handoff(
    handoff: Path,
    scored_ledger: Path,
    context_features: list[str],
    *,
    end_exclusive: pd.Timestamp | None = None,
    sample_max_rows: int | None = None,
) -> pd.DataFrame:
    """Load the current top-30 universe without intersecting an older ledger."""
    h = str(handoff.resolve()).replace("'", "''")
    s = str(scored_ledger.resolve()).replace("'", "''")
    handoff_schema = set(pq.read_schema(handoff).names)
    embedded = [feature for feature in context_features if feature in handoff_schema]
    context_select = ",\n            ".join(
        f"CAST(h.{_quoted(feature)} AS FLOAT) AS {_quoted(feature)}"
        for feature in embedded
    )
    if context_select:
        context_select = ",\n            " + context_select
    where_clause = ""
    if end_exclusive is not None:
        cutoff = end_exclusive.tz_convert("UTC")
        where_clause = (
            "WHERE h.__ts__ < "
            f"TIMESTAMPTZ '{cutoff:%Y-%m-%d %H:%M:%S}+00'"
        )
    joined_query = f"""
        SELECT
            h.__ts__, h.__symbol__, lower(h.side_name) AS side_name,
            h.__label_path_end_ts__ AS __label_path_end_ts__,
            CAST(s.ev_after_1pct AS FLOAT) AS ev_after_1pct,
            CAST(s.clean_exec AS FLOAT) AS clean_exec,
            CAST(s.dirty_positive AS FLOAT) AS dirty_positive,
            CAST(s.full_path_bad_mae_1r AS FLOAT) AS full_path_bad_mae_1r,
            CAST(s.timeout AS FLOAT) AS timeout,
            CAST(h.score AS FLOAT) AS score_base,
            CAST(h.score AS FLOAT) AS score,
            CAST(h.base_margin_to_cutoff AS FLOAT) AS base_margin_to_cutoff,
            CAST(h.base_margin_to_cutoff_z AS FLOAT) AS base_margin_to_cutoff_z,
            CAST(h.base_signal_zscore_within_archetype AS FLOAT)
                AS base_signal_zscore_within_archetype,
            coalesce(
                nullif(h.archetype_policy_key, ''),
                nullif(h.archetype_label_family, ''),
                'unknown'
            ) AS archetype_policy_key
            {context_select}
        FROM read_parquet('{h}') h
        INNER JOIN read_parquet('{s}') s
          ON h.__ts__ = s.__ts__
         AND h.__symbol__ = s.__symbol__
         AND lower(h.side_name) = lower(s.side_name)
        {where_clause}
    """
    if sample_max_rows is not None and int(sample_max_rows) > 0:
        per_bucket = max(1, int(math.ceil(int(sample_max_rows) / 3.0)))
        query = f"""
            WITH joined AS ({joined_query}), periodized AS (
                SELECT *, ntile(3) OVER (
                    ORDER BY __ts__, __symbol__, side_name
                ) AS __sample_period
                FROM joined
            ), ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY __sample_period
                    ORDER BY __ts__, __symbol__, side_name
                ) AS __period_row,
                count(*) OVER (PARTITION BY __sample_period) AS __period_rows
                FROM periodized
            )
            SELECT * EXCLUDE (__sample_period, __period_row, __period_rows)
            FROM ranked
            WHERE __period_rows <= {per_bucket}
               OR floor((__period_row - 1) * {per_bucket} / __period_rows)
                  < floor(__period_row * {per_bucket} / __period_rows)
            ORDER BY __ts__, __symbol__, side_name
        """
    else:
        query = joined_query + " ORDER BY h.__ts__, h.__symbol__, h.side_name"
    frame = duckdb.sql(query).df()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["__label_path_end_ts__"] = pd.to_datetime(
        frame["__label_path_end_ts__"], utc=True, errors="coerce"
    )
    if frame["__label_path_end_ts__"].isna().any():
        raise ValueError(
            "Current meta handoff contains missing __label_path_end_ts__ rows; "
            "residual-expert training cannot purge unresolved forward paths."
        )
    frame["side_name"] = frame["side_name"].astype("category")
    frame["archetype_policy_key"] = frame["archetype_policy_key"].astype("category")
    return frame


def _resolved_train_before(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    max_train_days: int | None = None,
) -> pd.DataFrame:
    """Return signal rows whose complete label path resolved before cutoff."""

    if "__label_path_end_ts__" not in frame.columns:
        raise ValueError(
            "Residual-expert training requires __label_path_end_ts__ for causal purging"
        )
    resolution = pd.to_datetime(
        frame["__label_path_end_ts__"], utc=True, errors="coerce"
    )
    if resolution.isna().any():
        raise ValueError(
            "Residual-expert training found non-finite __label_path_end_ts__ rows"
        )
    mask = frame["__ts__"].lt(cutoff) & resolution.lt(cutoff)
    if max_train_days is not None and int(max_train_days) > 0:
        start = cutoff - pd.Timedelta(days=int(max_train_days))
        mask &= frame["__ts__"].ge(start)
    return frame.loc[mask]


def _load_current_handoff_with_feature_store(
    handoff: Path,
    scored_ledger: Path,
    feature_dir: Path,
    context_features: list[str],
    *,
    rank_fit_end_exclusive: pd.Timestamp,
    **kwargs: Any,
) -> pd.DataFrame:
    frame = _load_current_handoff(
        handoff,
        scored_ledger,
        context_features,
        **kwargs,
    )
    frame = _augment_from_feature_store(frame, feature_dir, context_features)
    return _add_frozen_base_train_prior_rank(
        frame,
        fit_end_exclusive=rank_fit_end_exclusive,
    )


def _full_meta_context_candidates(
    contract_features: Iterable[str],
    available: set[str],
) -> list[str]:
    configured = get_meta_feature_keys("clf", CFG)
    candidates = _context_candidates(
        [*configured, *contract_features],
        available,
    )
    # Reliability and local meta AE/GMM outputs are generated entirely from
    # pre-entry values plus already-resolved base OOS outcomes.  They are not
    # config.py raw features, so explicitly admit the namespaced contracts.
    candidates.extend(
        feature
        for feature in sorted(available)
        if feature.startswith(META_OBSERVABLE_CONTEXT_PREFIXES)
    )
    return list(dict.fromkeys(candidates))


def _load_joined(
    predictions: Path,
    handoff: Path,
    context_features: list[str],
    *,
    end_exclusive: pd.Timestamp | None = None,
    sample_max_rows: int | None = None,
) -> pd.DataFrame:
    p = str(predictions.resolve()).replace("'", "''")
    h = str(handoff.resolve()).replace("'", "''")
    handoff_select = ",\n            ".join(
        f"CAST(h.{_quoted(feature)} AS FLOAT) AS {_quoted(feature)}"
        for feature in context_features
    )
    if handoff_select:
        handoff_select = ",\n            " + handoff_select
    where_clause = ""
    if end_exclusive is not None:
        cutoff = end_exclusive.tz_convert("UTC")
        where_clause = (
            "WHERE p.__ts__ < "
            f"TIMESTAMPTZ '{cutoff:%Y-%m-%d %H:%M:%S}+00'"
        )
    joined_query = f"""
        SELECT
            p.__ts__, p.__symbol__, lower(p.side_name) AS side_name,
            CAST(p.ev_after_1pct AS FLOAT) AS ev_after_1pct,
            CAST(p.clean_exec AS FLOAT) AS clean_exec,
            CAST(p.dirty_positive AS FLOAT) AS dirty_positive,
            CAST(p.full_path_bad_mae_1r AS FLOAT) AS full_path_bad_mae_1r,
            CAST(p.timeout AS FLOAT) AS timeout,
            CAST(p.score_base AS FLOAT) AS score_base,
            CAST(p.score_meta_base_soft_label AS FLOAT) AS score_meta,
            CAST(h.score AS FLOAT) AS score,
            CAST(p.base_score_rank_pct_train_prior AS FLOAT)
                AS base_score_rank_pct_train_prior,
            CAST(p.base_margin_to_cutoff AS FLOAT) AS base_margin_to_cutoff,
            CAST(p.base_margin_to_cutoff_z AS FLOAT) AS base_margin_to_cutoff_z,
            CAST(p.base_signal_zscore_within_archetype AS FLOAT)
                AS base_signal_zscore_within_archetype,
            coalesce(
                nullif(p.archetype_policy_key, ''),
                nullif(p.archetype_label_family, ''),
                'unknown'
            ) AS archetype_policy_key
            {handoff_select}
        FROM read_parquet('{p}') p
        INNER JOIN read_parquet('{h}') h
          ON p.__ts__ = h.__ts__
         AND p.__symbol__ = h.__symbol__
         AND lower(p.side_name) = lower(h.side_name)
        {where_clause}
    """
    if sample_max_rows is not None and int(sample_max_rows) > 0:
        per_bucket = max(1, int(math.ceil(int(sample_max_rows) / 3.0)))
        query = f"""
            WITH joined AS (
                {joined_query}
            ), periodized AS (
                SELECT *, ntile(3) OVER (
                    ORDER BY __ts__, __symbol__, side_name
                ) AS __sample_period
                FROM joined
            ), ranked AS (
                SELECT *,
                    row_number() OVER (
                        PARTITION BY __sample_period
                        ORDER BY __ts__, __symbol__, side_name
                    ) AS __period_row,
                    count(*) OVER (PARTITION BY __sample_period) AS __period_rows
                FROM periodized
            )
            SELECT * EXCLUDE (__sample_period, __period_row, __period_rows)
            FROM ranked
            WHERE __period_rows <= {per_bucket}
               OR floor((__period_row - 1) * {per_bucket} / __period_rows)
                  < floor(__period_row * {per_bucket} / __period_rows)
            ORDER BY __ts__, __symbol__, side_name
        """
    else:
        query = joined_query + " ORDER BY p.__ts__, p.__symbol__, p.side_name"
    frame = duckdb.sql(query).df()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["side_name"] = frame["side_name"].astype("category")
    frame["archetype_policy_key"] = frame["archetype_policy_key"].astype("category")
    for feature in (*ANCHORS, *context_features):
        if feature in frame:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(
                np.float32, copy=False
            )
    return frame


def _active_feature_audit(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in frame.columns:
            rows.append(
                {
                    "feature": feature,
                    "family": "aegmm" if feature in AEGMM_CANDIDATES else "context",
                    "finite_rate": 0.0,
                    "unique_values": 0,
                    "std": math.nan,
                    "active": False,
                }
            )
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=np.float64, copy=False))
        unique = int(values.loc[finite].nunique(dropna=True)) if finite.any() else 0
        rows.append(
            {
                "feature": feature,
                "family": "aegmm" if feature in AEGMM_CANDIDATES else "context",
                "finite_rate": float(finite.mean()),
                "unique_values": unique,
                "std": float(values.loc[finite].std()) if finite.any() else math.nan,
                # Joint availability is enforced once across the whole basket
                # by lgbm_pipeline immediately before staged selection. This
                # audit only removes empty/constant inputs.
                "active": bool(finite.any() and unique >= 8),
            }
        )
    return pd.DataFrame(rows)


def _model_params(
    seed: int,
    tuned_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    # Deliberately regularized: the expert corrects a strong global backbone.
    params: dict[str, Any] = {
        "objective": "regression_l2",
        "learning_rate": 0.025,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 500,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "feature_fraction": 0.70,
        "lambda_l1": 2.0,
        "lambda_l2": 8.0,
        "seed": int(seed),
        "num_threads": max(1, min(8, int(os.cpu_count() or 1))),
        "verbosity": -1,
    }
    rounds = 140
    if tuned_params:
        tuned = dict(tuned_params)
        rounds = max(20, int(tuned.pop("n_estimators", rounds) or rounds))
        aliases = {
            "min_child_samples": "min_data_in_leaf",
            "min_split_gain": "min_gain_to_split",
            "reg_alpha": "lambda_l1",
            "reg_lambda": "lambda_l2",
            "subsample": "bagging_fraction",
            "subsample_freq": "bagging_freq",
            "colsample_bytree": "feature_fraction",
        }
        ignored = {
            "boosting_type",
            "class_weight",
            "importance_type",
            "n_jobs",
            "random_state",
            "silent",
        }
        for key, value in tuned.items():
            if key in ignored or value is None:
                continue
            params[aliases.get(key, key)] = value
    params["seed"] = int(seed)
    params["num_threads"] = max(1, min(8, int(os.cpu_count() or 1)))
    params["verbosity"] = -1
    return params, rounds


def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame.loc[:, features].to_numpy(dtype=np.float32, copy=False)


def _fit_ev_map(frame: pd.DataFrame, score_col: str):
    return fit_hierarchical_ev_calibrator(
        frame,
        pd.to_numeric(frame[score_col], errors="coerce").to_numpy(np.float32),
        pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(np.float32),
        shrink_rows=4_000.0,
        min_local_rows=1_000,
        local_weight_cap=0.65,
        tail_weight_top10=5.0,
        tail_weight_top20=2.0,
        tail_weight_by_score_quantile=True,
        rank_blend=1.0,
    )


def _fit_side_models(
    train: pd.DataFrame,
    features_by_side: dict[str, list[str]],
    *,
    backbone_score_col: str,
    seed: int,
    params_by_side: dict[str, dict[str, Any]] | None = None,
    sample_weight_half_life_months: float | None = None,
    min_leaf_scaling_alpha: float = 0.0,
    hpo_reference_rows: int = 45_000,
) -> tuple[Any, dict[str, lgb.Booster]]:
    ev_map = _fit_ev_map(train, backbone_score_col)
    raw = pd.to_numeric(train[backbone_score_col], errors="coerce").to_numpy(
        np.float32
    )
    expected = predict_hierarchical_ev(ev_map, train, raw)
    residual = pd.to_numeric(train["ev_after_1pct"], errors="coerce").to_numpy(
        np.float32
    ) - expected
    models: dict[str, lgb.Booster] = {}
    sides = train["side_name"].astype(str).to_numpy()
    for offset, side in enumerate(("long", "short")):
        mask = (sides == side) & np.isfinite(residual)
        features = features_by_side.get(side) or []
        if int(mask.sum()) < 5_000 or not features:
            continue
        # Tail rows matter more but lower-ranked candidates anchor the residual.
        # Use train-only score quantiles because base and meta scores do not
        # share an absolute numerical scale.
        q80, q90 = np.quantile(raw[mask], [0.80, 0.90])
        weight = np.where(
            raw[mask] >= q90,
            3.0,
            np.where(raw[mask] >= q80, 1.5, 0.5),
        ).astype(np.float32)
        if sample_weight_half_life_months is not None and sample_weight_half_life_months > 0:
            half_life_days = float(sample_weight_half_life_months) * 30.4375
            timestamps = pd.to_datetime(train.loc[mask, "__ts__"], utc=True, errors="coerce")
            latest = timestamps.max()
            age_days = (latest - timestamps).dt.total_seconds().to_numpy(np.float64) / 86_400.0
            decay = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / half_life_days)
            # Preserve LightGBM's effective aggregate weight while changing only
            # the relative influence of older observations.
            decay /= max(float(np.nanmean(decay)), 1e-12)
            weight *= decay.astype(np.float32)
        dataset = lgb.Dataset(
            _matrix(train.loc[mask], features),
            label=residual[mask],
            weight=weight,
            feature_name=features,
            free_raw_data=True,
        )
        params, rounds = _model_params(
            seed + offset,
            (params_by_side or {}).get(side),
        )
        base_leaf = max(1, int(params.get("min_data_in_leaf", 1)))
        row_ratio = float(mask.sum()) / max(1, int(hpo_reference_rows))
        leaf_multiplier = max(1.0, row_ratio ** max(0.0, float(min_leaf_scaling_alpha)))
        params["min_data_in_leaf"] = max(1, int(math.ceil(base_leaf * leaf_multiplier)))
        params["_ares_leaf_scaling"] = {
            "base_min_data_in_leaf": base_leaf,
            "effective_min_data_in_leaf": int(params["min_data_in_leaf"]),
            "training_rows": int(mask.sum()),
            "hpo_reference_rows": int(hpo_reference_rows),
            "alpha": float(min_leaf_scaling_alpha),
        }
        # Private provenance must not reach LightGBM's parameter parser.
        fit_params = {key: value for key, value in params.items() if not key.startswith("_ares_")}
        model = lgb.train(
            fit_params,
            dataset,
            num_boost_round=rounds,
        )
        # LightGBM 4 no longer exposes Booster.set_attr(). Keep the runtime
        # provenance on the Python bundle; the run manifest records the global
        # ablation contract as well.
        model._ares_leaf_scaling = params["_ares_leaf_scaling"]
        model._ares_sample_weight_half_life_months = float(
            sample_weight_half_life_months or 0.0
        )
        print(
            f"[fit] side={side} rows={int(mask.sum()):,} "
            f"features={len(features)} rounds={rounds} "
            f"leaf={params['min_data_in_leaf']} half_life_m={sample_weight_half_life_months or 0.0}",
            flush=True,
        )
        models[side] = model
    return ev_map, models


def _predict_side_residuals(
    frame: pd.DataFrame,
    models: dict[str, lgb.Booster],
    features_by_side: dict[str, list[str]],
) -> np.ndarray:
    output = np.zeros(len(frame), dtype=np.float32)
    sides = frame["side_name"].astype(str).to_numpy()
    for side, model in models.items():
        mask = sides == side
        if mask.any():
            output[mask] = model.predict(
                _matrix(frame.loc[mask], features_by_side[side])
            ).astype(np.float32)
    return output


def _fit_corrected_ev_map(
    train: pd.DataFrame,
    baseline_map: Any,
    models: dict[str, lgb.Booster],
    features_by_side: dict[str, list[str]],
    alpha_by_side: dict[str, float],
    *,
    backbone_score_col: str,
):
    """Calibrate the residual-corrected score into a common net-EV unit."""
    raw = pd.to_numeric(train[backbone_score_col], errors="coerce").to_numpy(
        np.float32
    )
    baseline_ev = predict_hierarchical_ev(baseline_map, train, raw)
    residual = _predict_side_residuals(train, models, features_by_side)
    alpha = (
        train["side_name"]
        .astype(str)
        .map(alpha_by_side)
        .fillna(0.0)
        .to_numpy(np.float32)
    )
    corrected = baseline_ev + alpha * residual
    return fit_hierarchical_ev_calibrator(
        train,
        corrected,
        pd.to_numeric(train["ev_after_1pct"], errors="coerce").to_numpy(np.float32),
        shrink_rows=4_000.0,
        min_local_rows=1_000,
        local_weight_cap=0.65,
        tail_weight_top10=5.0,
        tail_weight_top20=2.0,
        tail_weight_by_score_quantile=True,
        rank_blend=1.0,
    )


def _select_features_once(
    train: pd.DataFrame,
    active_context: list[str],
    *,
    backbone_score_col: str,
    seed: int,
    max_context_features: int,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    initial = list(dict.fromkeys([*ANCHORS, *active_context]))
    initial = [feature for feature in initial if feature in train]
    features_by_side = {"long": initial, "short": initial}
    _, models = _fit_side_models(
        train,
        features_by_side,
        backbone_score_col=backbone_score_col,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    selected: dict[str, list[str]] = {}
    for side in ("long", "short"):
        model = models.get(side)
        if model is None:
            selected[side] = list(ANCHORS)
            continue
        gain = np.asarray(model.feature_importance("gain"), dtype=np.float64)
        split = np.asarray(model.feature_importance("split"), dtype=np.int64)
        names = list(model.feature_name())
        order = np.argsort(-gain, kind="stable")
        context = [
            names[index]
            for index in order
            if names[index] not in ANCHORS and gain[index] > 0
        ][: int(max_context_features)]
        selected[side] = list(dict.fromkeys([*ANCHORS, *context]))
        total_gain = float(gain.sum())
        for name, feature_gain, feature_split in zip(names, gain, split):
            rows.append(
                {
                    "side": side,
                    "feature": name,
                    "family": (
                        "anchor" if name in ANCHORS else
                        "aegmm" if name in AEGMM_CANDIDATES else "fresh_context"
                    ),
                    "gain": float(feature_gain),
                    "gain_share": float(feature_gain / total_gain) if total_gain > 0 else 0.0,
                    "split": int(feature_split),
                    "selected": name in selected[side],
                }
            )
    return selected, pd.DataFrame(rows)


def _archetype_balanced_tail_weights(
    frame: pd.DataFrame,
    raw_score: np.ndarray,
) -> np.ndarray:
    """Balance archetype support without discarding the economic score tail."""
    tail = _economic_tail_weights(raw_score).astype(np.float64)
    labels = frame["archetype_policy_key"].astype(str)
    counts = labels.map(labels.value_counts(dropna=False)).to_numpy(np.float64)
    reference = float(np.nanmedian(counts[counts > 0])) if np.any(counts > 0) else 1.0
    balance = np.sqrt(reference / np.maximum(counts, 1.0))
    balance = np.clip(balance, 0.35, 3.0)
    weights = tail * balance
    weights /= max(float(np.mean(weights)), 1e-12)
    return weights.astype(np.float32)


def _economic_tail_weights(raw_score: np.ndarray) -> np.ndarray:
    """Top-tail economics without side/archetype support reweighting."""
    q80, q90 = np.quantile(raw_score[np.isfinite(raw_score)], [0.80, 0.90])
    tail = np.where(raw_score >= q90, 3.0, np.where(raw_score >= q80, 1.5, 0.5))
    tail /= max(float(np.mean(tail)), 1e-12)
    return tail.astype(np.float32)


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _stable_json_sha256(value: Any) -> str:
    """Hash JSON-compatible provenance with a canonical representation."""

    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_text_sha256(model: lgb.Booster) -> str:
    return hashlib.sha256(model.model_to_string().encode("utf-8")).hexdigest()


def _fold_frame_boundary(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("OOS fold bundle cannot serialize non-UTC signal timestamps")
    boundary: dict[str, Any] = {
        "rows": int(len(frame)),
        "signal_timestamp_min": timestamps.min().isoformat(),
        "signal_timestamp_max": timestamps.max().isoformat(),
    }
    if "__label_path_end_ts__" in frame:
        label_end = pd.to_datetime(
            frame["__label_path_end_ts__"], utc=True, errors="coerce"
        )
        if label_end.isna().any():
            raise ValueError("OOS fold bundle cannot serialize non-UTC label resolution timestamps")
        boundary["label_path_end_timestamp_min"] = label_end.min().isoformat()
        boundary["label_path_end_timestamp_max"] = label_end.max().isoformat()
    return boundary


def _persist_oos_fold_bundle(
    *,
    out_dir: Path,
    fold_id: str,
    oos_fit_mode: str,
    backbone_score: str,
    backbone_score_col: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_ev_map: Any,
    residual_models: dict[str, lgb.Booster],
    corrected_ev_map: Any,
    alpha_by_side: dict[str, float],
    features_by_side: dict[str, list[str]],
    params_by_side: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Persist the exact fitted state used for one reported OOS fold."""

    fold_dir = Path(out_dir) / "fold_models" / str(fold_id)
    fold_dir.mkdir(parents=True, exist_ok=True)
    feature_contract = {
        side: [str(feature) for feature in features_by_side.get(side, [])]
        for side in ("long", "short")
    }
    configured_params = {
        side: _jsonable(dict(params_by_side.get(side, {})))
        for side in ("long", "short")
    }
    effective_params = {
        side: _jsonable(dict(model.params))
        for side, model in sorted(residual_models.items())
    }
    component_hashes = {
        "feature_contract_sha256": _stable_json_sha256(feature_contract),
        "alpha_by_side_sha256": _stable_json_sha256(
            {side: float(alpha_by_side.get(side, 0.0)) for side in ("long", "short")}
        ),
        "configured_model_params_by_side_sha256": _stable_json_sha256(configured_params),
        "residual_model_text_sha256": {
            side: _model_text_sha256(model)
            for side, model in sorted(residual_models.items())
        },
    }
    payload = {
        "schema": OOS_FOLD_BUNDLE_SCHEMA,
        "fold_id": str(fold_id),
        "oos_fit_mode": str(oos_fit_mode),
        "backbone_score": str(backbone_score),
        "backbone_score_col": str(backbone_score_col),
        "train_boundary": _fold_frame_boundary(train),
        "test_boundary": _fold_frame_boundary(test),
        "feature_contract": feature_contract,
        "alpha_by_side": {
            side: float(alpha_by_side.get(side, 0.0)) for side in ("long", "short")
        },
        "configured_model_params_by_side": configured_params,
        "effective_model_params_by_side": effective_params,
        "baseline_ev_map": baseline_ev_map,
        "residual_models": residual_models,
        "corrected_ev_map": corrected_ev_map,
        "component_hashes": component_hashes,
        "final_refit_included": False,
    }
    bundle_path = fold_dir / "bundle.joblib"
    joblib.dump(payload, bundle_path, compress=3)

    hashes = {
        "bundle_sha256": _sha256(bundle_path),
        **component_hashes,
    }
    manifest = {
        "schema": OOS_FOLD_BUNDLE_SCHEMA,
        "fold_id": str(fold_id),
        "bundle_path": str(bundle_path),
        "oos_fit_mode": str(oos_fit_mode),
        "backbone_score": str(backbone_score),
        "backbone_score_col": str(backbone_score_col),
        "train_boundary": payload["train_boundary"],
        "test_boundary": payload["test_boundary"],
        "feature_contract": feature_contract,
        "alpha_by_side": payload["alpha_by_side"],
        "configured_model_params_by_side": configured_params,
        "effective_model_params_by_side": effective_params,
        "residual_model_sides": sorted(residual_models),
        "component_hashes": component_hashes,
        "final_refit_included": False,
        "hashes": hashes,
    }
    manifest_path = fold_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "bundle_path": str(bundle_path),
        "manifest_path": str(manifest_path),
        "bundle_sha256": hashes["bundle_sha256"],
    }


def _load_fixed_hpo_params_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load a frozen long/short parameter contract without reusing features."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("hpo_params") or payload.get("model_params_by_side") or {}
    params = {
        str(side): dict(values)
        for side, values in dict(raw).items()
        if isinstance(values, dict)
    }
    for side in ("long", "short"):
        if not params.get(side):
            raise ValueError(
                f"Fixed parameter manifest must contain non-empty {side} params: {path}"
            )
    return {side: params[side] for side in ("long", "short")}


def _staged_select_and_hpo_once(
    train: pd.DataFrame,
    active_context: list[str],
    *,
    selection_hpo_contract: dict[str, Any],
    backbone_score_col: str,
    seed: int,
    out_dir: Path,
    selection_max_rows: int,
    hpo_max_rows: int,
    hpo_trials: int,
    hpo_patience: int,
    fixed_hpo_params: dict[str, dict[str, Any]] | None = None,
    fixed_hpo_params_source: str | None = None,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    pd.DataFrame,
    dict[str, Any],
]:
    """Run the canonical staged selector and side-local HPO exactly once."""
    initial = [
        feature
        for feature in dict.fromkeys([*ANCHORS, *active_context])
        if feature in train
    ]
    baseline_map = _fit_ev_map(train, backbone_score_col)
    raw = pd.to_numeric(train[backbone_score_col], errors="coerce").to_numpy(
        np.float32
    )
    expected = predict_hierarchical_ev(baseline_map, train, raw)
    realized = pd.to_numeric(train["ev_after_1pct"], errors="coerce").to_numpy(
        np.float32
    )
    residual = (realized - expected).astype(np.float32)
    valid = np.isfinite(residual) & np.isfinite(raw)
    work = train.loc[valid].reset_index(drop=True)
    X = work.loc[:, initial].astype(np.float32, copy=False)
    y = residual[valid]
    raw_valid = raw[valid]
    weights = _archetype_balanced_tail_weights(work, raw_valid)
    hard = (y > 0.0).astype(np.float32)
    timestamps = work["__ts__"].to_numpy()
    assets = work["__symbol__"].astype(str).to_numpy()
    sides = work["side_name"].astype(str).to_numpy()
    archetypes = work["archetype_policy_key"].astype(str).to_numpy()
    side_archetypes = np.char.add(np.char.add(sides, "__"), archetypes)

    # These are process-local controls for the canonical pipeline. Both stages
    # use chronological beginning/middle/end sampling and no feature-count cap.
    staged_lgbm.LGBM_TIME_SPREAD_HPO_SELECTION = True
    staged_lgbm.LGBM_FEATURE_SELECTION_FORCE_RECENT_ROWS = False
    staged_lgbm.LGBM_RACE_MAX_ROWS = int(selection_max_rows)
    staged_lgbm.LGBM_HPO_MAX_ROWS = int(hpo_max_rows)
    staged_lgbm.LGBM_HPO_ROW_SUBSAMPLE_FRAC = 1.0
    staged_lgbm.LGBM_MIN_FEATURES = len(ANCHORS)
    staged_lgbm.LGBM_SELECTED_FEATURES_MIN = len(ANCHORS)
    staged_lgbm.LGBM_SELECTED_FEATURES_MAX = 0
    staged_lgbm.LGBM_OBJECTIVE = "balanced_topk"
    staged_lgbm.LGBM_ARCHETYPE_FEATURE_SELECTION = True

    selection_dir = out_dir / "staged_feature_selection"
    cfg = {
        "protected_features": list(ANCHORS),
        "force_include_features": list(ANCHORS),
        "lgbm_joint_complete_case_filter_enabled": True,
        "lgbm_feature_min_coverage": 0.90,
        "lgbm_feature_coverage_scope": "all_post_warmup",
        "lgbm_feature_coverage_warmup_days": 30,
        "lgbm_feature_coverage_allow_model_derived_exemptions": False,
        "lgbm_time_feature_selector_bypass_enabled": False,
        "mda_config": {
            "enabled": True,
            "objective": "topk_opportunity_precision",
            "topk_fracs": [0.10, 0.20, 0.30],
            "topk_frac_weights": [0.50, 0.30, 0.20],
            "archetype_conditioned_enabled": True,
            "archetype_global_weight": 0.25,
            "archetype_macro_weight": 0.60,
            "archetype_worst_weight": 0.15,
            "side_tail_across_archetypes_unweighted": True,
            "force_include_features": list(ANCHORS),
            "protected_features": list(ANCHORS),
            "report_dir": str(selection_dir),
            "write_mda_report": True,
        },
    }
    label_context = {
        "feature_selection_archetype": side_archetypes,
        "archetype": side_archetypes,
        "side_name": sides,
        "side": sides,
        "side_mda_sample_weight": _economic_tail_weights(raw_valid),
        "clean_exec": pd.to_numeric(work["clean_exec"], errors="coerce").to_numpy(
            np.float32
        ),
        "dirty_positive": pd.to_numeric(
            work["dirty_positive"], errors="coerce"
        ).to_numpy(np.float32),
        "bad_mae_1r": pd.to_numeric(
            work["full_path_bad_mae_1r"], errors="coerce"
        ).to_numpy(np.float32),
        "timeout": pd.to_numeric(work["timeout"], errors="coerce").to_numpy(
            np.float32
        ),
        "y_ret": realized[valid],
        "net_utility": realized[valid],
        "y_bin": hard,
    }
    candidate = staged_lgbm.train_lgbm_stability_candidate(
        X,
        y,
        sample_weight=weights,
        random_state=int(seed),
        mode="regressor",
        timestamps=timestamps,
        assets=assets,
        returns=y,
        hard_labels=hard,
        hpo_objective_mode="train_meta",
        reference_artifact_dir=selection_dir,
        cfg=cfg,
        label_context=label_context,
    )
    if candidate is None:
        raise RuntimeError("staged feature selection did not return a candidate")
    candidate_metrics = dict(candidate.get("metrics") or {})
    selected_by_side = candidate_metrics.get(
        "per_side_feature_selection_selected_features", {}
    )
    if not bool(candidate_metrics.get("per_side_feature_selection_enabled", False)):
        raise RuntimeError(
            "Staged meta selection did not enable the required side-local selector."
        )
    selected: dict[str, list[str]] = {}
    for side in ("long", "short"):
        values = [str(v) for v in selected_by_side.get(side, [])]
        if not values:
            raise RuntimeError(
                f"Staged meta selection produced no side-local contract for {side}."
            )
        selected[side] = list(
            dict.fromkeys([*ANCHORS, *[value for value in values if value in initial]])
        )
        missing = [anchor for anchor in ANCHORS if anchor not in selected[side]]
        if missing:
            raise AssertionError(f"{side} selector dropped protected anchors: {missing}")

    hpo_params: dict[str, dict[str, Any]] = {}
    hpo_reports: dict[str, Any] = {}
    realized_valid = realized[valid]
    positive_ev = (realized_valid > 0.0).astype(np.float32)
    for offset, side in enumerate(("long", "short"), start=1):
        mask = sides == side
        if int(mask.sum()) < 5_000:
            raise RuntimeError(f"insufficient {side} rows for side-local HPO")
        if fixed_hpo_params is not None:
            params = dict(fixed_hpo_params[side])
            report = {
                "hpo_skipped": True,
                "hpo_completed_trials": 0,
                "hpo_best_value": None,
                "parameter_contract": "frozen_side_params",
                "parameter_source": str(fixed_hpo_params_source or ""),
            }
        else:
            params, report = staged_lgbm._run_lgbm_hpo(
                X.loc[mask].reset_index(drop=True),
                y[mask],
                weights[mask],
                selected[side],
                classifier=False,
                groups=archetypes[mask],
                timestamps=timestamps[mask],
                returns=realized_valid[mask],
                metric_y=positive_ev[mask],
                random_state=int(seed) + offset * 2003,
                max_trials=int(hpo_trials),
                patience=int(hpo_patience),
                objective_mode="train_meta",
                cfg=cfg,
            )
        hpo_params[side] = dict(params)
        hpo_reports[side] = dict(report)
        print(
            f"[params] side={side} features={len(selected[side])} "
            f"trials={int(report.get('hpo_completed_trials', 0) or 0)} "
            f"source={report.get('parameter_contract', 'hpo')}",
            flush=True,
        )

    feature_stats = candidate.get("feature_stats")
    if not isinstance(feature_stats, pd.DataFrame):
        feature_stats = pd.DataFrame()
    report = {
        **selection_hpo_contract,
        "status": SELECTION_HPO_COMPLETED_STATUS,
        "selection_rows_available": int(len(work)),
        "selection_start": work["__ts__"].min().isoformat(),
        "selection_end": work["__ts__"].max().isoformat(),
        "selection_max_rows": int(selection_max_rows),
        "hpo_max_rows": int(hpo_max_rows),
        "hpo_trials_requested_per_side": (
            0 if fixed_hpo_params is not None else int(hpo_trials)
        ),
        "parameter_contract": (
            "frozen_side_params" if fixed_hpo_params is not None else "side_local_hpo"
        ),
        "parameter_source": str(fixed_hpo_params_source or ""),
        "sample_policy": "chronological_begin_middle_end",
        "target": "residual_net_ev_after_1pct",
        "hpo_objective": (
            "balanced_topk_actual_ev_precision_and_archetype_stability"
        ),
        "archetype_contract": "side_name__archetype_policy_key",
        "protected_anchors": list(ANCHORS),
        "selected_features": selected,
        "hpo_params": hpo_params,
        "candidate_metrics": candidate_metrics,
        "hpo_reports": hpo_reports,
    }
    _write_selection_hpo_manifest(out_dir, report)
    return selected, hpo_params, feature_stats, report


def _top_fraction_mask(score: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    values = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=-np.inf)
    count = max(1, int(math.ceil(len(values) * float(fraction))))
    chosen = np.argpartition(values, len(values) - count)[-count:]
    mask = np.zeros(len(values), dtype=bool)
    mask[chosen] = True
    return mask


def _score_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    selected = _top_fraction_mask(score)
    ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(np.float64)
    if "week_start" in frame:
        week = frame["week_start"]
    else:
        ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        week = ts.dt.normalize() - pd.to_timedelta(ts.dt.weekday, unit="D")
    if "calendar_month" in frame:
        month = frame["calendar_month"]
    else:
        month = pd.to_datetime(
            frame["__ts__"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m")
    selected_ev = ev[selected]
    selected_sides = frame.loc[selected, "side_name"].astype(str)
    weekly = pd.Series(selected_ev).groupby(week.loc[selected].reset_index(drop=True)).mean()
    monthly = pd.Series(selected_ev).groupby(month.loc[selected].reset_index(drop=True)).mean()
    return {
        "candidate_rows": float(len(frame)),
        "selected_rows": float(selected.sum()),
        "mean_ev_after_1pct": float(np.nanmean(selected_ev)),
        "worst_week_ev_after_1pct": float(weekly.min()),
        "worst_month_ev_after_1pct": float(monthly.min()),
        "clean_exec_precision": float(
            pd.to_numeric(frame.loc[selected, "clean_exec"], errors="coerce").mean()
        ),
        "dirty_positive_rate": float(
            pd.to_numeric(frame.loc[selected, "dirty_positive"], errors="coerce").mean()
        ),
        "full_path_bad_mae_rate": float(
            pd.to_numeric(frame.loc[selected, "full_path_bad_mae_1r"], errors="coerce").mean()
        ),
        "timeout_rate": float(
            pd.to_numeric(frame.loc[selected, "timeout"], errors="coerce").mean()
        ),
        "long_share": float(selected_sides.eq("long").mean()),
        "short_share": float(selected_sides.eq("short").mean()),
    }


def _tune_alpha(
    validation: pd.DataFrame,
    baseline_ev: np.ndarray,
    residual: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    if "week_start" not in validation or "calendar_month" not in validation:
        validation = validation.copy(deep=False)
        timestamps = pd.to_datetime(validation["__ts__"], utc=True, errors="coerce")
        day = timestamps.dt.normalize()
        validation["week_start"] = day - pd.to_timedelta(day.dt.weekday, unit="D")
        validation["calendar_month"] = timestamps.dt.strftime("%Y-%m")
    sides = validation["side_name"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    baseline = _score_metrics(validation, baseline_ev)
    for alpha_long in np.arange(0.0, 1.01, 0.20):
        for alpha_short in np.arange(0.0, 1.01, 0.20):
            alpha = np.where(sides == "long", alpha_long, alpha_short)
            score = baseline_ev + alpha.astype(np.float32) * residual
            metric = _score_metrics(validation, score)
            mean_gain = metric["mean_ev_after_1pct"] - baseline["mean_ev_after_1pct"]
            allowance = max(0.0, mean_gain / 5.0)
            admissible = (
                metric["worst_week_ev_after_1pct"]
                >= baseline["worst_week_ev_after_1pct"] - allowance
                and metric["worst_month_ev_after_1pct"]
                >= baseline["worst_month_ev_after_1pct"] - allowance
            )
            rows.append(
                {
                    "alpha_long": float(alpha_long),
                    "alpha_short": float(alpha_short),
                    "admissible": bool(admissible),
                    **metric,
                }
            )
    search = pd.DataFrame(rows)
    eligible = search.loc[search["admissible"]]
    winner = (eligible if not eligible.empty else search).sort_values(
        ["mean_ev_after_1pct", "worst_week_ev_after_1pct"],
        ascending=False,
        kind="stable",
    ).iloc[0]
    return {
        "long": float(winner["alpha_long"]),
        "short": float(winner["alpha_short"]),
    }, search


def _breakdown(frame: pd.DataFrame, score_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("month", ["calendar_month"]),
        ("week", ["week_start"]),
        ("side", ["side_name"]),
        ("side_archetype", ["side_name", "archetype_policy_key"]),
        ("month_side", ["calendar_month", "side_name"]),
    ]
    for scope, group_cols in scopes:
        grouped = [((), frame)] if not group_cols else frame.groupby(group_cols, observed=True, sort=True)
        for group_key, group in grouped:
            keys = group_key if isinstance(group_key, tuple) else (group_key,)
            labels = dict(zip(group_cols, keys))
            for score_col in score_columns:
                metric = _score_metrics(
                    group, pd.to_numeric(group[score_col], errors="coerce").to_numpy(np.float32)
                )
                rows.append({"scope": scope, "model": score_col, **labels, **metric})
    return pd.DataFrame(rows)


def _ev_comparability(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    work = frame[["side_name", "archetype_policy_key", "ev_after_1pct", score_col]].copy()
    work["ev_bin"] = pd.qcut(work[score_col], 20, labels=False, duplicates="drop")
    return (
        work.groupby(["ev_bin", "side_name", "archetype_policy_key"], observed=True)
        .agg(rows=("ev_after_1pct", "size"), predicted_ev=(score_col, "mean"), realized_ev=("ev_after_1pct", "mean"))
        .reset_index()
    )


def _compare_control_metrics(
    metrics: pd.DataFrame,
    *,
    score_col: str,
    control_dir: Path,
    current_oos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    control_oos_path = control_dir / "oos_predictions.parquet"
    if current_oos is not None and control_oos_path.exists():
        keys = ["__ts__", "__symbol__", "side_name"]
        outcomes = [
            "ev_after_1pct",
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "calendar_month",
            "week_start",
            "archetype_policy_key",
        ]
        current_frame = current_oos.loc[:, [*keys, *outcomes, score_col]].copy()
        reference_score = pd.read_parquet(
            control_oos_path,
            columns=[*keys, score_col],
        ).rename(columns={score_col: "__control_score"})
        for frame in (current_frame, reference_score):
            frame["__ts__"] = pd.to_datetime(
                frame["__ts__"], utc=True, errors="coerce"
            )
            frame["__symbol__"] = frame["__symbol__"].astype(str)
            frame["side_name"] = frame["side_name"].astype(str).str.lower()
        if current_frame.duplicated(keys).any() or reference_score.duplicated(keys).any():
            raise ValueError("Control comparison requires unique timestamp/symbol/side rows.")
        exact = current_frame.merge(
            reference_score,
            on=keys,
            how="left",
            validate="one_to_one",
            copy=False,
        )
        missing = int(exact["__control_score"].isna().sum())
        current_rows = int(len(exact))
        exact = exact.loc[exact["__control_score"].notna()].copy()
        if exact.empty:
            return pd.DataFrame()
        current = _breakdown(exact, [score_col])
        control = _breakdown(exact, ["__control_score"])
        control["model"] = score_col
        comparison_basis = "exact_timestamp_symbol_side_overlap"
    else:
        control_path = control_dir / "metrics.csv"
        if not control_path.exists():
            return pd.DataFrame()
        control = pd.read_csv(control_path)
        comparison_basis = "aggregate_metrics_fallback"
    if comparison_basis != "exact_timestamp_symbol_side_overlap":
        current = metrics.loc[metrics["model"].eq(score_col)].copy()
    reference = control.loc[control["model"].eq(score_col)].copy()
    keys = [
        column
        for column in (
            "scope",
            "calendar_month",
            "week_start",
            "side_name",
            "archetype_policy_key",
        )
        if column in current.columns and column in reference.columns
    ]
    for frame in (current, reference):
        if "week_start" in keys:
            frame["week_start"] = (
                pd.to_datetime(frame["week_start"], utc=True, errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("__all__")
            )
        for column in keys:
            if column == "week_start":
                continue
            frame[column] = frame[column].astype("string").fillna("__all__")
    value_cols = [
        column
        for column in (
            "selected_rows",
            "mean_ev_after_1pct",
            "worst_week_ev_after_1pct",
            "worst_month_ev_after_1pct",
            "clean_exec_precision",
            "dirty_positive_rate",
            "full_path_bad_mae_rate",
            "timeout_rate",
            "long_share",
            "short_share",
        )
        if column in current.columns and column in reference.columns
    ]
    joined = current[[*keys, *value_cols]].merge(
        reference[[*keys, *value_cols]],
        on=keys,
        how="inner",
        suffixes=("_ablation", "_control"),
        validate="one_to_one",
    )
    for column in value_cols:
        joined[f"delta_{column}"] = (
            pd.to_numeric(joined[f"{column}_ablation"], errors="coerce")
            - pd.to_numeric(joined[f"{column}_control"], errors="coerce")
        )
    joined["comparison_basis"] = comparison_basis
    if comparison_basis == "exact_timestamp_symbol_side_overlap":
        joined["comparison_current_rows"] = current_rows
        joined["comparison_overlap_rows"] = int(len(exact))
        joined["comparison_missing_control_rows"] = missing
        joined["comparison_overlap_fraction"] = float(len(exact) / max(current_rows, 1))
    return joined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-predictions", type=Path, default=DEFAULT_GLOBAL_PREDICTIONS)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument(
        "--scored-ledger",
        type=Path,
        default=DEFAULT_SCORED_LEDGER,
        help="Outcome ledger matching --handoff for current_handoff source mode.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("current_handoff", "legacy_intersection"),
        default="current_handoff",
        help=(
            "current_handoff preserves the complete current top-30 universe; "
            "legacy_intersection is retained only for exact historical ablations."
        ),
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--calibration-month", default="2026-03")
    parser.add_argument("--eval-start", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-11")
    parser.add_argument(
        "--max-train-days",
        type=int,
        default=365,
        help="Maximum causal train history per fit; 0 disables the rolling cap.",
    )
    parser.add_argument(
        "--sample-weight-half-life-months",
        type=float,
        default=0.0,
        help="Exponential observation-weight half-life in months; 0 keeps the baseline weights.",
    )
    parser.add_argument(
        "--min-leaf-scaling-alpha",
        type=float,
        default=0.0,
        help=(
            "Scale min_data_in_leaf by max(1, (side_train_rows / "
            "hpo_reference_rows) ** alpha)."
        ),
    )
    parser.add_argument(
        "--hpo-reference-rows",
        type=int,
        default=45_000,
        help="HPO sample size used as the reference for min-leaf scaling.",
    )
    parser.add_argument(
        "--skip-final-refit",
        action="store_true",
        help="Skip the non-OOS final fit for ablation matrix runs.",
    )
    parser.add_argument(
        "--oos-fit-mode",
        choices=("frozen_pre_eval", "expanding_monthly"),
        default="frozen_pre_eval",
        help=(
            "frozen_pre_eval fits one model on all rows before eval-start and "
            "reuses it for every OOS month. expanding_monthly is retained only "
            "as an explicit diagnostic."
        ),
    )
    parser.add_argument("--max-context-features", type=int, default=32)
    parser.add_argument(
        "--context-contract-only",
        action="store_true",
        help=(
            "Use the frozen raw V9 contract plus embedded AE/GMM/reliability "
            "context instead of eagerly loading every config.py meta field. "
            "This is a memory-bounded training mode; staged MDA/HPO remains active."
        ),
    )
    parser.add_argument(
        "--exclude-new-meta-state-context",
        action="store_true",
        help=(
            "Exclude only the new base_reliability_* and meta_*_aegmm_* "
            "features. Intended for controlled comparisons against the same "
            "V9 residual target, corrected-path handoff, split, and HPO "
            "parameter contract."
        ),
    )
    parser.add_argument(
        "--exclude-meta-aegmm-context",
        action="store_true",
        help="Exclude only the new meta_long_aegmm_* and meta_short_aegmm_* block.",
    )
    parser.add_argument(
        "--exclude-base-reliability-context",
        action="store_true",
        help="Exclude only the causal base_reliability_* context block.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("staged_mda", "gain_cap"),
        default="staged_mda",
    )
    parser.add_argument("--feature-selection-max-rows", type=int, default=45_000)
    parser.add_argument("--hpo-max-rows", type=int, default=45_000)
    parser.add_argument("--hpo-trials", type=int, default=150)
    parser.add_argument("--hpo-patience", type=int, default=40)
    parser.add_argument(
        "--force-selection-hpo",
        action="store_true",
        help=(
            "Rerun staged side-local selection and HPO instead of reusing an exact "
            "completed sibling or explicit selection contract."
        ),
    )
    parser.add_argument(
        "--reuse-selection-manifest",
        type=Path,
        default=None,
        help=(
            "Reuse a completed staged_selection_hpo_manifest.json and skip "
            "feature selection/HPO. The manifest feature and parameter "
            "contracts are validated against the current stores before fit."
        ),
    )
    parser.add_argument(
        "--reuse-hpo-params-manifest",
        type=Path,
        default=None,
        help=(
            "Reuse only frozen long/short model parameters while rerunning "
            "candidate-aware feature selection. Unlike --reuse-selection-manifest, "
            "this does not freeze the selected feature names."
        ),
    )
    parser.add_argument(
        "--backbone-score",
        choices=("meta", "base"),
        default="base",
        help=(
            "Score whose train-only side/archetype EV residual is learned. "
            "The base arm does not use the shared meta score as an input."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.max_train_days < 0:
        parser.error("--max-train-days must be non-negative")
    if args.sample_weight_half_life_months < 0:
        parser.error("--sample-weight-half-life-months must be non-negative")
    if not 0.0 <= args.min_leaf_scaling_alpha <= 1.0:
        parser.error("--min-leaf-scaling-alpha must be in [0, 1]")
    if args.hpo_reference_rows < 1:
        parser.error("--hpo-reference-rows must be positive")
    if (
        args.reuse_selection_manifest is not None
        and args.reuse_hpo_params_manifest is not None
    ):
        parser.error(
            "--reuse-selection-manifest and --reuse-hpo-params-manifest are mutually exclusive"
        )
    if args.force_selection_hpo and args.reuse_selection_manifest is not None:
        parser.error(
            "--force-selection-hpo and --reuse-selection-manifest are mutually exclusive"
        )
    if args.force_selection_hpo and args.reuse_hpo_params_manifest is not None:
        parser.error(
            "--force-selection-hpo and --reuse-hpo-params-manifest are mutually exclusive"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fixed_hpo_params = (
        _load_fixed_hpo_params_manifest(args.reuse_hpo_params_manifest)
        if args.reuse_hpo_params_manifest is not None
        else None
    )

    handoff_schema = set(pq.read_schema(args.handoff).names)
    feature_store_schema = _logical_feature_store_schema(args.feature_dir)
    available = handoff_schema | feature_store_schema
    contract_features = _contract_features(args.contract)
    backbone_score_col = "score_meta" if args.backbone_score == "meta" else "score_base"
    score_prefix = "score_meta" if args.backbone_score == "meta" else "score_base"
    calibration_start = pd.Timestamp(f"{args.calibration_month}-01", tz="UTC")
    calibration_end = calibration_start + pd.offsets.MonthBegin(1)

    def load_source(
        context: list[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        if args.source_mode == "current_handoff":
            return _load_current_handoff_with_feature_store(
                args.handoff,
                args.scored_ledger,
                args.feature_dir,
                context,
                rank_fit_end_exclusive=calibration_start,
                **kwargs,
            )
        return _load_joined_with_feature_store(
            args.global_predictions,
            args.handoff,
            args.feature_dir,
            context,
            handoff_schema,
            **kwargs,
        )

    def excluded_new_meta_context_prefixes() -> tuple[str, ...]:
        if args.exclude_new_meta_state_context:
            return NEW_META_STATE_CONTEXT_PREFIXES
        prefixes: list[str] = []
        if args.exclude_meta_aegmm_context:
            prefixes.extend(META_AEGMM_CONTEXT_PREFIXES)
        if args.exclude_base_reliability_context:
            prefixes.extend(BASE_RELIABILITY_CONTEXT_PREFIXES)
        return tuple(prefixes)

    def filter_new_meta_state_context(features: list[str]) -> list[str]:
        prefixes = excluded_new_meta_context_prefixes()
        if not prefixes:
            return features
        return [
            feature
            for feature in features
            if not feature.startswith(prefixes)
        ]

    selection_report: dict[str, Any]
    if args.selection_mode == "staged_mda":
        candidate_context = (
            _context_candidates(contract_features, available)
            if args.context_contract_only
            else _full_meta_context_candidates(contract_features, available)
        )
        if args.context_contract_only:
            candidate_context.extend(
                feature
                for feature in sorted(available)
                if feature.startswith(META_OBSERVABLE_CONTEXT_PREFIXES)
            )
            candidate_context = list(dict.fromkeys(candidate_context))
        candidate_context = filter_new_meta_state_context(candidate_context)
        selection_hpo_contract = _selection_hpo_contract(
            source_mode=str(args.source_mode),
            handoff=args.handoff,
            scored_ledger=args.scored_ledger,
            global_predictions=args.global_predictions,
            feature_dir=args.feature_dir,
            feature_store_schema=feature_store_schema,
            backbone_contract=args.contract,
            candidate_context=candidate_context,
            backbone_score=str(args.backbone_score),
            backbone_score_col=backbone_score_col,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            selection_mode=str(args.selection_mode),
            context_contract_only=bool(args.context_contract_only),
            excluded_context_prefixes=excluded_new_meta_context_prefixes(),
            selection_max_rows=int(args.feature_selection_max_rows),
            hpo_max_rows=int(args.hpo_max_rows),
            hpo_trials=int(args.hpo_trials),
            hpo_patience=int(args.hpo_patience),
            seed=int(args.seed),
            fixed_hpo_params_manifest=args.reuse_hpo_params_manifest,
        )
        reusable_manifest, reuse_provenance = _find_reusable_selection_hpo_manifest(
            selection_hpo_contract,
            output_dir=args.out_dir,
            manifest_path=args.reuse_selection_manifest,
            force=bool(args.force_selection_hpo),
        )
        if reusable_manifest is not None:
            selection_report = dict(reusable_manifest)
            features_by_side = {
                str(side): [str(feature) for feature in features]
                for side, features in dict(
                    selection_report.get("selected_features") or {}
                ).items()
            }
            params_by_side = {
                str(side): dict(params)
                for side, params in dict(
                    selection_report.get("hpo_params") or {}
                ).items()
            }
            for side in ("long", "short"):
                if side not in features_by_side or side not in params_by_side:
                    raise ValueError(
                        "Selection manifest must contain selected features and "
                        f"HPO params for {side}."
                    )
                missing_anchors = [
                    feature
                    for feature in ANCHORS
                    if feature not in features_by_side[side]
                ]
                if missing_anchors:
                    raise ValueError(
                        f"Selection manifest {side} contract is missing protected "
                        f"anchors: {missing_anchors}"
                    )
            selection_report["reuse_provenance"] = reuse_provenance
            _write_selection_hpo_manifest(args.out_dir, selection_report)
            importance = pd.DataFrame(
                [
                    {
                        "side": side,
                        "feature": feature,
                        "selected": True,
                        "source": "reused_staged_selection_hpo_manifest",
                    }
                    for side, features in features_by_side.items()
                    for feature in features
                ]
            )
            print(
                "[selection-resume] "
                f"manifest={reuse_provenance.get('path')} "
                + " ".join(
                    f"{side}_features={len(features)}"
                    for side, features in features_by_side.items()
                ),
                flush=True,
            )
        else:
            selection_train = load_source(
                candidate_context,
                end_exclusive=calibration_start,
                sample_max_rows=int(args.feature_selection_max_rows),
            )
            print(
                f"[load-selection] rows={len(selection_train):,} "
                f"configured_context_candidates={len(candidate_context)}",
                flush=True,
            )
            audit = _active_feature_audit(selection_train, candidate_context)
            active_context = audit.loc[
                audit["active"], "feature"
            ].astype(str).tolist()
            (
                features_by_side,
                params_by_side,
                importance,
                selection_report,
            ) = _staged_select_and_hpo_once(
                selection_train,
                active_context,
                selection_hpo_contract=selection_hpo_contract,
                backbone_score_col=backbone_score_col,
                seed=int(args.seed),
                out_dir=args.out_dir,
                selection_max_rows=int(args.feature_selection_max_rows),
                hpo_max_rows=int(args.hpo_max_rows),
                hpo_trials=int(args.hpo_trials),
                hpo_patience=int(args.hpo_patience),
                fixed_hpo_params=fixed_hpo_params,
                fixed_hpo_params_source=(
                    str(args.reuse_hpo_params_manifest)
                    if args.reuse_hpo_params_manifest is not None
                    else None
                ),
            )
            selection_report["reuse_provenance"] = reuse_provenance
            _write_selection_hpo_manifest(args.out_dir, selection_report)
        selected_context = filter_new_meta_state_context([
            feature
            for feature in dict.fromkeys(
                features_by_side.get("long", []) + features_by_side.get("short", [])
            )
            if feature not in ANCHORS
        ])
        if "selection_train" in locals():
            del selection_train
        frame = load_source(
            selected_context,
        )
        frame["__ts__"] = pd.to_datetime(
            frame["__ts__"], utc=True, errors="coerce"
        )
        audit = _active_feature_audit(frame, selected_context)
        inactive_selected = audit.loc[~audit["active"], "feature"].astype(str).tolist()
        if inactive_selected:
            raise ValueError(
                "Replayed feature contract contains unavailable or constant "
                f"features: {inactive_selected[:20]}"
            )
        selection_train = _resolved_train_before(
            frame,
            calibration_start,
            max_train_days=int(args.max_train_days) or None,
        )
    else:
        candidate_context = filter_new_meta_state_context(
            _context_candidates(contract_features, available)
        )
        frame = load_source(
            candidate_context,
        )
        audit = _active_feature_audit(frame, candidate_context)
        active_context = audit.loc[audit["active"], "feature"].astype(str).tolist()
        selection_train = _resolved_train_before(
            frame,
            calibration_start,
            max_train_days=int(args.max_train_days) or None,
        )
        features_by_side, importance = _select_features_once(
            selection_train,
            active_context,
            backbone_score_col=backbone_score_col,
            seed=int(args.seed),
            max_context_features=int(args.max_context_features),
        )
        params_by_side = {}
        selection_report = {
            "sample_policy": "legacy_gain_cap",
            "max_context_features": int(args.max_context_features),
        }
    print(
        f"[load-model] rows={len(frame):,} selected_context="
        f"{len(set(features_by_side.get('long', []) + features_by_side.get('short', [])) - set(ANCHORS))}",
        flush=True,
    )
    calibration = frame.loc[
        frame["__ts__"].ge(calibration_start) & frame["__ts__"].lt(calibration_end)
    ]
    print(
        "[selection] "
        + " ".join(
            f"{side}_features={len(features)}"
            for side, features in features_by_side.items()
        ),
        flush=True,
    )
    tune_map, tune_models = _fit_side_models(
        selection_train,
        features_by_side,
        backbone_score_col=backbone_score_col,
        seed=int(args.seed) + 100,
        params_by_side=params_by_side,
        sample_weight_half_life_months=float(args.sample_weight_half_life_months) or None,
        min_leaf_scaling_alpha=float(args.min_leaf_scaling_alpha),
        hpo_reference_rows=int(args.hpo_reference_rows),
    )
    calibration_raw = pd.to_numeric(
        calibration[backbone_score_col], errors="coerce"
    ).to_numpy(np.float32)
    calibration_ev = predict_hierarchical_ev(tune_map, calibration, calibration_raw)
    calibration_residual = _predict_side_residuals(calibration, tune_models, features_by_side)
    alpha_by_side, alpha_search = _tune_alpha(calibration, calibration_ev, calibration_residual)
    print(f"[alpha] {alpha_by_side}", flush=True)

    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    fold_frames: list[pd.DataFrame] = []
    fold_manifest: list[dict[str, Any]] = []
    frozen_oos_state: tuple[Any, dict[str, lgb.Booster], Any] | None = None
    if args.oos_fit_mode == "frozen_pre_eval":
        frozen_train = _resolved_train_before(
            frame,
            eval_start,
            max_train_days=int(args.max_train_days) or None,
        )
        frozen_ev_map, frozen_models = _fit_side_models(
            frozen_train,
            features_by_side,
            backbone_score_col=backbone_score_col,
            seed=int(args.seed) + 1000,
            params_by_side=params_by_side,
            sample_weight_half_life_months=float(args.sample_weight_half_life_months) or None,
            min_leaf_scaling_alpha=float(args.min_leaf_scaling_alpha),
            hpo_reference_rows=int(args.hpo_reference_rows),
        )
        frozen_corrected_ev_map = _fit_corrected_ev_map(
            frozen_train,
            frozen_ev_map,
            frozen_models,
            features_by_side,
            alpha_by_side,
            backbone_score_col=backbone_score_col,
        )
        frozen_oos_state = (
            frozen_ev_map,
            frozen_models,
            frozen_corrected_ev_map,
        )
        print(
            f"[oos-fit] mode=frozen_pre_eval train_rows={len(frozen_train):,} "
            f"train_end_exclusive={eval_start.isoformat()}",
            flush=True,
        )
    current = pd.Timestamp(eval_start.year, eval_start.month, 1, tz="UTC")
    while current < eval_end:
        end = min(current + pd.offsets.MonthBegin(1), eval_end)
        train_cutoff = (
            eval_start if args.oos_fit_mode == "frozen_pre_eval" else current
        )
        train = _resolved_train_before(
            frame,
            train_cutoff,
            max_train_days=int(args.max_train_days) or None,
        )
        test = frame.loc[frame["__ts__"].ge(current) & frame["__ts__"].lt(end)].copy()
        if test.empty:
            current = end
            continue
        print(
            f"[fold] test={current:%Y-%m} train_rows={len(train):,} "
            f"test_rows={len(test):,}",
            flush=True,
        )
        if frozen_oos_state is None:
            ev_map, models = _fit_side_models(
                train,
                features_by_side,
                backbone_score_col=backbone_score_col,
                seed=int(args.seed) + len(fold_frames) + 1000,
                params_by_side=params_by_side,
                sample_weight_half_life_months=float(args.sample_weight_half_life_months) or None,
                min_leaf_scaling_alpha=float(args.min_leaf_scaling_alpha),
                hpo_reference_rows=int(args.hpo_reference_rows),
            )
            corrected_ev_map = _fit_corrected_ev_map(
                train,
                ev_map,
                models,
                features_by_side,
                alpha_by_side,
                backbone_score_col=backbone_score_col,
            )
        else:
            ev_map, models, corrected_ev_map = frozen_oos_state
        raw = pd.to_numeric(test[backbone_score_col], errors="coerce").to_numpy(
            np.float32
        )
        expected = predict_hierarchical_ev(ev_map, test, raw)
        residual = _predict_side_residuals(test, models, features_by_side)
        alpha = test["side_name"].astype(str).map(alpha_by_side).fillna(0.0).to_numpy(np.float32)
        corrected_ev = expected + alpha * residual
        corrected_ev_mapped = predict_hierarchical_ev(
            corrected_ev_map, test, corrected_ev
        )
        test["score_base_rank"] = pd.to_numeric(test["score_base"], errors="coerce").astype(np.float32)
        if "score_meta" in test:
            test["score_meta_raw"] = pd.to_numeric(
                test["score_meta"], errors="coerce"
            ).astype(np.float32)
        test[f"{score_prefix}_ev_mapped"] = expected.astype(np.float32)
        test[f"{score_prefix}_ev_residual_expert"] = corrected_ev.astype(np.float32)
        test[f"{score_prefix}_ev_residual_expert_hier_mapped"] = (
            corrected_ev_mapped.astype(np.float32)
        )
        test["meta_residual_expert_delta_ev"] = (alpha * residual).astype(np.float32)
        test[f"{score_prefix}_ev_rank_train_reference"] = expected_ev_rank(
            ev_map, expected, raw
        )
        test[f"{score_prefix}_residual_ev_rank_train_reference"] = expected_ev_rank(
            corrected_ev_map, corrected_ev_mapped, corrected_ev
        )
        test["calendar_month"] = test["__ts__"].dt.strftime("%Y-%m")
        day = test["__ts__"].dt.normalize()
        test["week_start"] = day - pd.to_timedelta(day.dt.weekday, unit="D")
        fold_id = f"{current:%Y-%m-%d}_{end:%Y-%m-%d}"
        fold_bundle = _persist_oos_fold_bundle(
            out_dir=args.out_dir,
            fold_id=fold_id,
            oos_fit_mode=str(args.oos_fit_mode),
            backbone_score=str(args.backbone_score),
            backbone_score_col=backbone_score_col,
            train=train,
            test=test,
            baseline_ev_map=ev_map,
            residual_models=models,
            corrected_ev_map=corrected_ev_map,
            alpha_by_side=alpha_by_side,
            features_by_side=features_by_side,
            params_by_side=params_by_side,
        )
        fold_frames.append(test)
        fold_manifest.append(
            {
                "fold_id": fold_id,
                "train_end_exclusive": current.isoformat(),
                "test_start": current.isoformat(),
                "test_end_exclusive": end.isoformat(),
                "train_rows": int(len(train)),
                "oos_fit_mode": str(args.oos_fit_mode),
                "test_rows": int(len(test)),
                "long_model": "long" in models,
                "short_model": "short" in models,
                "oos_fold_bundle": fold_bundle,
            }
        )
        current = end

    scored = pd.concat(fold_frames, ignore_index=True)
    score_columns = list(dict.fromkeys([
        "score_base_rank",
        *(["score_meta_raw"] if "score_meta_raw" in scored else []),
        f"{score_prefix}_ev_mapped",
        f"{score_prefix}_ev_residual_expert",
        f"{score_prefix}_ev_residual_expert_hier_mapped",
    ]))
    metrics = _breakdown(scored, score_columns)
    overall = metrics.loc[metrics["scope"].eq("overall")].set_index("model")
    identity = [
        column
        for column in (
            "scope", "calendar_month", "week_start", "side_name", "archetype_policy_key"
        )
        if column in metrics
    ]
    base_metric = metrics.loc[
        metrics["model"].eq("score_base_rank"),
        [*identity, "mean_ev_after_1pct"],
    ].rename(columns={"mean_ev_after_1pct": "base_mean_ev_after_1pct"})
    metrics = metrics.merge(base_metric, on=identity, how="left", validate="many_to_one")
    metrics["delta_ev_vs_base"] = (
        metrics["mean_ev_after_1pct"] - metrics["base_mean_ev_after_1pct"]
    )

    final_score_col = f"{score_prefix}_ev_residual_expert_hier_mapped"
    control_comparison = _compare_control_metrics(
        metrics,
        score_col=final_score_col,
        control_dir=args.control_dir,
        current_oos=scored,
    )

    scored.to_parquet(args.out_dir / "oos_predictions.parquet", index=False, compression="zstd")
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    audit.to_csv(args.out_dir / "feature_availability.csv", index=False)
    importance.to_csv(args.out_dir / "feature_selection_importance.csv", index=False)
    alpha_search.to_csv(args.out_dir / "alpha_search_march.csv", index=False)
    if not control_comparison.empty:
        control_comparison.to_csv(
            args.out_dir / "ablation_vs_gain_cap_control.csv", index=False
        )
    _ev_comparability(
        scored, f"{score_prefix}_ev_residual_expert_hier_mapped"
    ).to_csv(
        args.out_dir / "ev_comparability_by_side_archetype.csv", index=False
    )
    feature_contract = {
        "long": features_by_side.get("long", []),
        "short": features_by_side.get("short", []),
    }

    # Final refit is deliberately separate from OOS evaluation. Feature
    # selection remains frozen before March and correction strength remains
    # frozen on March; only model coefficients and train-derived EV curves see
    # all currently resolved labels. Matrix ablations can skip it safely.
    final_model_path: Path | None = None
    final_train = pd.DataFrame()
    handoff_manifest_path = args.handoff.parent / "manifest.json"
    handoff_manifest = (
        json.loads(handoff_manifest_path.read_text(encoding="utf-8"))
        if handoff_manifest_path.is_file()
        else {}
    )
    frozen_ae_gmm = handoff_manifest.get("frozen_ae_gmm_context_contract") or {}
    model_provenance = {
        "global_predictions": str(args.global_predictions),
        "handoff": str(args.handoff),
        "scored_ledger": str(args.scored_ledger),
        "handoff_manifest": str(handoff_manifest_path),
        "base_source": handoff_manifest.get("base_source"),
        "feature_dir": handoff_manifest.get("feature_dir"),
        "frozen_ae_gmm_state_path": frozen_ae_gmm.get("state_path"),
        "frozen_ae_gmm_state_hash": frozen_ae_gmm.get("cycle_state_hash"),
        "frozen_ae_gmm_output_count": frozen_ae_gmm.get("generated_features"),
    }
    if not args.skip_final_refit:
        final_ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(
            np.float32
        )
        final_train = frame.loc[np.isfinite(final_ev)]
        if int(args.max_train_days) > 0 and not final_train.empty:
            final_cutoff = pd.to_datetime(final_train["__ts__"], utc=True).max() + pd.Timedelta(seconds=1)
            final_train = _resolved_train_before(
                final_train,
                final_cutoff,
                max_train_days=int(args.max_train_days),
            )
        final_ev_map, final_models = _fit_side_models(
            final_train,
            features_by_side,
            backbone_score_col=backbone_score_col,
            seed=int(args.seed) + 10_000,
            params_by_side=params_by_side,
            sample_weight_half_life_months=float(args.sample_weight_half_life_months) or None,
            min_leaf_scaling_alpha=float(args.min_leaf_scaling_alpha),
            hpo_reference_rows=int(args.hpo_reference_rows),
        )
        final_corrected_ev_map = _fit_corrected_ev_map(
            final_train,
            final_ev_map,
            final_models,
            features_by_side,
            alpha_by_side,
            backbone_score_col=backbone_score_col,
        )
        final_model_path = args.out_dir / "final_side_residual_expert.joblib"
        final_payload = {
            "schema": "side_base_residual_expert_inference_v2",
            "backbone_score": str(args.backbone_score),
            "backbone_score_col": backbone_score_col,
            "feature_contract": feature_contract,
            "alpha_by_side": alpha_by_side,
            "baseline_ev_map": final_ev_map,
            "residual_models": final_models,
            "corrected_ev_map": final_corrected_ev_map,
            "model_params_by_side": params_by_side,
            "round_trip_cost": 0.01,
            "feature_selection_fit_end_exclusive": calibration_start.isoformat(),
            "alpha_calibration_month": str(args.calibration_month),
            "final_train_rows": int(len(final_train)),
            "final_train_start": final_train["__ts__"].min().isoformat(),
            "final_train_last_label_timestamp": final_train["__ts__"].max().isoformat(),
            "provenance": model_provenance,
        }
        joblib.dump(final_payload, final_model_path, compress=3)
        for side, model in final_models.items():
            model.save_model(str(args.out_dir / f"final_{side}_residual_expert.txt"))

    manifest = {
        "generated_by": Path(__file__).name,
        "architecture": (
            f"{args.backbone_score}_backbone_plus_ev_mapped_side_residual_experts_v1"
        ),
        "backbone_score": str(args.backbone_score),
        "canonical": False,
        "ablation_control": str(args.control_dir),
        "historical_v9_reference": str(HISTORICAL_V9_REFERENCE),
        "historical_v9_code_revision": HISTORICAL_V9_CODE_REVISION,
        "global_predictions": str(args.global_predictions),
        "handoff": str(args.handoff),
        "scored_ledger": str(args.scored_ledger),
        "source_mode": str(args.source_mode),
        "context_contract_only": bool(args.context_contract_only),
        "exclude_new_meta_state_context": bool(args.exclude_new_meta_state_context),
        "exclude_meta_aegmm_context": bool(args.exclude_meta_aegmm_context),
        "exclude_base_reliability_context": bool(args.exclude_base_reliability_context),
        "excluded_new_meta_state_context_prefixes": list(
            excluded_new_meta_context_prefixes()
        ),
        "backbone_contract": str(args.contract),
        "backbone_direct_anchors_present": {
            feature: feature in contract_features for feature in ANCHORS
        },
        "residual_expert_target": (
            "ev_after_1pct - train_only_hierarchical_expected_ev"
            f"({args.backbone_score}_score, side, archetype)"
        ),
        "residual_expert_excluded_families": list(FORBIDDEN_CONTEXT_PREFIXES),
        "feature_selection_fit_end_exclusive": calibration_start.isoformat(),
        "alpha_calibration_month": str(args.calibration_month),
        "feature_selection_scope": str(args.selection_mode),
        "feature_selection_report": selection_report,
        "feature_contract": feature_contract,
        "alpha_by_side": alpha_by_side,
        "model_params_by_side": params_by_side,
        "oos_fit_mode": str(args.oos_fit_mode),
        "training_window": {
            "max_train_days": int(args.max_train_days),
            "sample_weight_half_life_months": float(args.sample_weight_half_life_months),
            "min_leaf_scaling_alpha": float(args.min_leaf_scaling_alpha),
            "hpo_reference_rows": int(args.hpo_reference_rows),
        },
        "folds": fold_manifest,
        "leakage_contract": {
            "base_model_retrained": False,
            "global_meta_backbone_retrained": False,
            "feature_selection_and_alpha_tuning": "rows before April only; March calibration",
            "expanding_residual_expert_fit": bool(
                args.oos_fit_mode == "expanding_monthly"
            ),
            "single_frozen_pre_eval_fit": bool(
                args.oos_fit_mode == "frozen_pre_eval"
            ),
            "oos_outcomes_used_as_features": False,
            "ev_maps_fit_on_test_rows": False,
            "shared_meta_score_used_as_residual_expert_input": bool(
                args.backbone_score == "meta"
            ),
            "final_refit_used_for_oos_metrics": False,
            "forward_label_purge": (
                "__ts__ < train_cutoff and __label_path_end_ts__ < train_cutoff"
            ),
        },
        "final_refit": {
            "path": str(final_model_path) if final_model_path is not None else None,
            "sha256": _sha256(final_model_path) if final_model_path is not None else None,
            "train_rows": int(len(final_train)),
            "train_start": final_train["__ts__"].min().isoformat() if not final_train.empty else None,
            "last_label_timestamp": final_train["__ts__"].max().isoformat() if not final_train.empty else None,
            "feature_selection_frozen_at": calibration_start.isoformat(),
            "alpha_frozen_from_month": str(args.calibration_month),
            "long_model": bool(final_model_path is not None),
            "short_model": bool(final_model_path is not None),
            "used_for_reported_oos_predictions": False,
        },
        "model_provenance": model_provenance,
        "score_semantics": {
            f"{score_prefix}_ev_residual_expert": (
                "backbone expected net EV plus side residual EV"
            ),
            f"{score_prefix}_ev_residual_expert_hier_mapped": (
                "train-only hierarchical monotonic expected net EV after 1pct cost"
            ),
            f"{score_prefix}_residual_ev_rank_train_reference": (
                "frozen train-derived percentile of the common EV unit"
            ),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "alpha_by_side": alpha_by_side, "overall": overall["mean_ev_after_1pct"].to_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
