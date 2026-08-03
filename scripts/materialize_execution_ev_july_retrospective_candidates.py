#!/usr/bin/env python3
"""Materialize a fail-closed retrospective July execution-EV candidate surface.

This is deliberately a *pre-score* adapter.  It reads a complete canonical
static feature store at exact signal timestamps, duplicates the point-in-time
universe into long and short candidate streams, and validates every frozen
Pack-B raw feature contract before writing a single candidate.  It does not
load labels, outcomes, or a recent-EV map and never claims forward/OOS status.

The July 27 forward-confirmation source lock is intentionally not read or
modified: July 20--23 is a retrospective reconstruction unless a separately
frozen earlier source lock proves otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import deterministic_candidate_ids


SCHEMA = "execution_ev_july_retrospective_candidate_surface_v1"
SIDES = ("long", "short")
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
DEFAULT_SPEC = ROOT / "configs/execution_ev_forward_confirmation_candidate_20260728_v1.json"
DEFAULT_ROLE_ROOT = ROOT / "data_perp/artifacts/packb_path_auxiliary_role_bundles_20260725_v1_31_8"
DEFAULT_CATBOOST_ROOT = ROOT / "data_perp/reports/catboost_path_archetype_packb31_8_structural_balance_20260725_v1"
DEFAULT_HEAD_CONTRACT = ROOT / "data_perp/artifacts/execution_ev_forward_final_heads_20260728_v1/feature_contract.json"
DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"

# These values are created by score_packb_final_refits_forward.py before the
# residual/support heads see the selected top-40% stream.  They are not raw
# static features and must not be manufactured by this adapter.
DERIVED_AFTER_BASE = {
    "prediction",
    "base_prediction",
    "base_oof_score",
    "score",
    "base_candidate_rank_timestamp_side",
    "base_candidate_rank_pct_timestamp_side",
    "base_rank_timestamp_side",
    "base_rank_pct_timestamp_side",
    "base_rank_decile",
    "base_cutoff_score",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
    "base_score_z_timestamp_side",
    "group_score_mean",
    "group_score_std",
    "decile_score_mean",
    "decile_score_std",
    "archetype_label_family",
    "archetype_policy_key",
    "base_alpha_ev",
    "residual_delta_ev",
    "existing_alpha_ev",
    "alpha_prediction_uncertainty",
    "alpha_leaf_support",
    "oof_clean_favorable_probability",
    "pred_peak_MFE_12h_ATR",
    "catboost_entropy",
    "catboost_archetype",
    *(f"catboost_p_{index}" for index in range(7)),
}
# These columns are materialised after Pack-B scoring by the frozen,
# side-local AE/GMM state.  They are required by some downstream frozen head
# contracts, but they cannot be requested from the canonical static store.
# Keep this deliberately explicit: a newly introduced learned representation
# must be added here and supplied by the downstream representation adapter.
FROZEN_SIDE_LOCAL_REPRESENTATION = {
    "dae_b16_00",
    "dae_b16_02",
    "dae_b16_04",
    "dae_b16_08",
    "dae_b16_14",
    "expected_mahalanobis",
    "gmm_cluster_posterior_4",
    "gmm_dist_center_4",
    "gmm_dist_center_9",
    "gmm_ood_score",
    "gmm_representation_available",
}
DERIVED_PREFIXES = ("base_archetype_label__", "catboost_archetype__")
RUNTIME_COLUMNS = {"side", "side_name"}


class CandidateSurfacePreflightError(RuntimeError):
    """Raised after a persisted report identifies an incomplete source surface."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _utc(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required frozen AE/GMM evidence is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_frozen_ae_contracts(
    ae_root: Path = DEFAULT_AE_ROOT,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Validate and hash-bind the frozen outcome-free side-local AE/GMM inputs."""

    summary_path = ae_root / "summary.json"
    summary = _read_json(summary_path)
    if (
        summary.get("schema") != "packb_pre_march_side_ae_runner_v1"
        or summary.get("status") != "FROZEN_LONG_AND_SHORT_AE_GMM"
    ):
        raise ValueError("frozen AE/GMM summary has an unexpected schema or status")

    features_by_side: dict[str, list[str]] = {}
    evidence_by_side: dict[str, Any] = {}
    for side in SIDES:
        side_summary = summary.get("sides", {}).get(side)
        if not isinstance(side_summary, Mapping):
            raise ValueError(f"frozen AE/GMM summary is missing {side}")
        state_summary = side_summary.get("ae_gmm")
        if not isinstance(state_summary, Mapping):
            raise ValueError(f"frozen AE/GMM summary is missing {side} state evidence")

        contract_path = ae_root / side / "loader_evidence" / "frozen_feature_contract.json"
        loader_path = ae_root / side / "loader_evidence" / "loader_evidence.json"
        state_dir = ae_root / side / "ae_gmm"
        state_path = state_dir / "ae_gmm_state.pkl"
        metadata_path = state_dir / "ae_gmm_state_metadata.json"
        stage_manifest_path = state_dir / "side_stage_manifest.json"
        stage_config_path = state_dir / "stage_config.json"
        contract = _read_json(contract_path)
        loader = _read_json(loader_path)
        metadata = _read_json(metadata_path)
        stage_manifest = _read_json(stage_manifest_path)
        stage_config = _read_json(stage_config_path)

        columns = [str(value) for value in contract.get("feature_columns", [])]
        if (
            contract.get("schema") != "packb_static_point_feature_loader_v1"
            or len(columns) != 256
            or len(set(columns)) != 256
            or int(contract.get("max_feature_columns", -1)) != 256
        ):
            raise ValueError(f"{side} frozen AE/GMM raw feature contract is not the exact 256-column contract")
        overlap = sorted(set(columns).intersection(FROZEN_SIDE_LOCAL_REPRESENTATION))
        if overlap:
            raise ValueError(f"{side} frozen AE/GMM raw contract contains learned outputs: {overlap}")

        feature_contract_sha = str(contract.get("feature_contract_sha256", ""))
        if not feature_contract_sha or any(
            str(value) != feature_contract_sha
            for value in (
                loader.get("feature_contract_sha256"),
                metadata.get("feature_contract_sha256"),
                stage_config.get("feature_contract_sha256"),
                side_summary.get("feature_contract_sha256"),
                state_summary.get("feature_contract_sha256"),
            )
        ):
            raise ValueError(f"{side} frozen AE/GMM feature-contract hash lineage mismatch")
        if (
            stage_config.get("schema") != "packb_side_local_ae_stage_v1"
            or stage_config.get("side") != side
            or stage_config.get("outcome_free") is not True
            or list(stage_config.get("economic_targets", []))
        ):
            raise ValueError(f"{side} AE/GMM state is not explicitly outcome-free")
        if (
            stage_manifest.get("schema") != "packb_side_stage_manifest_v1"
            or stage_manifest.get("side") != side
            or stage_manifest.get("stage") != "ae_gmm"
            or stage_manifest.get("artifact", {}).get("kind") != "ae_gmm_state"
            or stage_manifest.get("artifact", {}).get("scope") != side
        ):
            raise ValueError(f"{side} AE/GMM side-state manifest is invalid")
        if (
            metadata.get("schema") != "packb_side_local_ae_stage_v1"
            or metadata.get("side") != side
        ):
            raise ValueError(f"{side} AE/GMM state metadata is invalid")

        state_sha = _sha256(state_path)
        config_sha = _sha256(stage_config_path)
        if any(
            str(value) != state_sha
            for value in (
                stage_manifest.get("artifact", {}).get("sha256"),
                metadata.get("state_sha256"),
                state_summary.get("state_sha256"),
            )
        ):
            raise ValueError(f"{side} frozen AE/GMM state hash mismatch")
        if any(
            str(value) != config_sha
            for value in (
                stage_manifest.get("stage_config", {}).get("sha256"),
                metadata.get("stage_config_sha256"),
                state_summary.get("stage_config_sha256"),
            )
        ):
            raise ValueError(f"{side} frozen AE/GMM outcome-free config hash mismatch")

        features_by_side[side] = columns
        evidence_by_side[side] = {
            "raw_feature_count": len(columns),
            "feature_contract_sha256": feature_contract_sha,
            "feature_contract_file": {
                "path": str(contract_path),
                "sha256": _sha256(contract_path),
            },
            "loader_evidence_file": {
                "path": str(loader_path),
                "sha256": _sha256(loader_path),
            },
            "state": {"path": str(state_path), "sha256": state_sha, "scope": side},
            "state_metadata_file": {
                "path": str(metadata_path),
                "sha256": _sha256(metadata_path),
            },
            "side_stage_manifest_file": {
                "path": str(stage_manifest_path),
                "sha256": _sha256(stage_manifest_path),
            },
            "outcome_free_stage_config": {
                "path": str(stage_config_path),
                "sha256": config_sha,
                "outcome_free": True,
                "economic_targets": [],
            },
        }
    union = sorted(set(features_by_side["long"]).union(features_by_side["short"]))
    if len(union) != 263:
        raise ValueError(f"frozen side-local AE/GMM raw feature union must contain 263 columns, got {len(union)}")
    return features_by_side, {
        "root": str(ae_root),
        "summary_file": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        "raw_feature_count_by_side": {side: len(features_by_side[side]) for side in SIDES},
        "raw_feature_union_count": len(union),
        "outcomes_used": False,
        "sides": evidence_by_side,
    }


def _role_record(spec: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    records = [record for record in spec["models"] if record.get("role") == role]
    if len(records) != 1:
        raise ValueError(f"expected exactly one frozen model record for {role}")
    record = records[0]
    model_path = _resolve(str(record["path"]))
    if not model_path.is_file() or _sha256(model_path) != str(record["sha256"]):
        raise ValueError(f"frozen model hash mismatch for {role}")
    return record


def _timing_features(state: Mapping[str, Any]) -> list[str]:
    by_horizon = state.get("selected_features_by_horizon")
    if by_horizon is None:
        return list(state["selected_features"])
    return list(
        dict.fromkeys(
            feature
            for horizon in sorted(by_horizon, key=lambda value: int(value))
            for feature in by_horizon[horizon]
        )
    )


def _static_columns(features: Sequence[str]) -> list[str]:
    return sorted(
        {
            str(feature)
            for feature in features
            if str(feature) not in DERIVED_AFTER_BASE
            and str(feature) not in FROZEN_SIDE_LOCAL_REPRESENTATION
            and str(feature) not in RUNTIME_COLUMNS
            and not str(feature).startswith(DERIVED_PREFIXES)
        }
    )


def collect_frozen_contracts(
    *,
    spec_path: Path = DEFAULT_SPEC,
    role_root: Path = DEFAULT_ROLE_ROOT,
    catboost_root: Path = DEFAULT_CATBOOST_ROOT,
    head_contract_path: Path = DEFAULT_HEAD_CONTRACT,
    ae_features_by_side: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, list[str]]]:
    """Load and hash-check the frozen side-local model input contracts."""

    from lightgbm import Booster

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    contracts: dict[str, dict[str, list[str]]] = {}
    for side in SIDES:
        base = Booster(model_file=str(_resolve(_role_record(spec, f"base_{side}")["path"])))
        residual = Booster(model_file=str(_resolve(_role_record(spec, f"residual_{side}")["path"])))
        _role_record(spec, f"clean_favorable_event_{side}")
        _role_record(spec, f"peak_mfe_{side}")
        _role_record(spec, f"path_catboost_{side}")
        timing = joblib.load(role_root / "shared" / "meaningful_mfe_event" / side / "timing_cdf_family.joblib")
        peak = joblib.load(role_root / "roles" / "peak_mfe_12h_atr__conditional_mean" / side / "role_bundle.joblib")
        classifier = joblib.load(catboost_root / f"side={side}" / "path_archetype_classifier.joblib")
        contracts[side] = {
            "base": list(base.feature_name()),
            "residual": list(residual.feature_name()),
            "clean_favorable_event": _timing_features(timing["side_models"][side]),
            "peak_mfe_conditional": list(peak["selected_features"]),
            "path_catboost": list(classifier.feature_columns),
            "frozen_ae_raw": list(ae_features_by_side[side]),
        }

    # The final direct/capture heads intentionally consume score-stage outputs,
    # not static rows.  Read and validate their side contracts here so the
    # manifest makes that boundary explicit instead of silently ignoring them.
    head = json.loads(head_contract_path.read_text(encoding="utf-8"))
    for side in SIDES:
        contracts[side]["final_head_preentry"] = list(head["feature_columns_by_side"][side])
    return contracts


def static_requirements(contracts: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, list[str]]:
    """Return the raw static requirements for each side without score outputs."""

    output: dict[str, list[str]] = {}
    for side in SIDES:
        if side not in contracts:
            raise ValueError(f"contracts missing {side}")
        raw: list[str] = []
        for role, fields in contracts[side].items():
            if role == "final_head_preentry":
                continue
            raw.extend(map(str, fields))
        output[side] = _static_columns(raw)
    return output


def _symbol_from_path(path: Path) -> str:
    token = path.stem.removeprefix("symbol=")
    return token.replace("_", "/", 1)


def _load_universe(features_dir: Path, universe_path: Path | None) -> list[tuple[str, Path]]:
    paths = sorted(features_dir.glob("symbol=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no canonical symbol=*.parquet files under {features_dir}")
    by_symbol = {_symbol_from_path(path): path for path in paths}
    if universe_path is None:
        return sorted(by_symbol.items())
    raw = pd.read_csv(universe_path)
    column = next((name for name in ("__symbol__", "symbol") if name in raw), None)
    if column is None:
        # The frozen inference-universe configs are also allowed to be simple
        # newline-delimited symbol lists.  Keep CSV parsing as the primary
        # contract, but do not silently discard the first symbol as a header.
        symbols = [
            line.strip()
            for line in universe_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not symbols:
            raise ValueError("universe file needs __symbol__/symbol CSV or a non-empty line-delimited symbol list")
    else:
        symbols = [str(value).strip() for value in raw[column].dropna()]
    if len(symbols) != len(set(symbols)):
        raise ValueError("universe file has duplicate symbols")
    missing = sorted(set(symbols).difference(by_symbol))
    if missing:
        raise ValueError("universe symbols absent from static surface: " + ", ".join(missing[:10]))
    return [(symbol, by_symbol[symbol]) for symbol in sorted(symbols)]


def _read_symbol_rows(
    path: Path,
    *,
    symbol: str,
    columns: Sequence[str],
    timestamps: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[str], int]:
    schema = set(pq.ParquetFile(path).schema.names)
    available = [column for column in columns if column in schema]
    missing_columns = sorted(set(columns).difference(schema))
    read_columns = [*available]
    if "__symbol__" in schema:
        read_columns.append("__symbol__")
    if "ts" in schema:
        read_columns.append("ts")
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(read_columns)))
    if "ts" in frame:
        frame["__ts__"] = pd.to_datetime(frame.pop("ts"), utc=True, errors="coerce")
    else:
        frame["__ts__"] = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[frame["__ts__"].notna()].copy()
    if frame["__ts__"].duplicated().any():
        raise CandidateSurfacePreflightError(f"duplicate static timestamps for {symbol}: {path}")
    if "__symbol__" in frame:
        source_symbols = frame["__symbol__"].dropna().astype(str).unique()
        if len(source_symbols) > 1 or (len(source_symbols) == 1 and source_symbols[0] != symbol):
            raise CandidateSurfacePreflightError(f"static symbol identity mismatch for {symbol}: {path}")
    frame["__symbol__"] = symbol
    frame["__source_row_present__"] = True
    frame = frame.set_index("__ts__").reindex(timestamps).reset_index()
    frame = frame.rename(columns={"index": "__ts__"})
    frame["__symbol__"] = symbol
    return frame.loc[:, ["__ts__", "__symbol__", "__source_row_present__", *available]], missing_columns, len(schema)


def _coverage_tables(
    raw_rows: pd.DataFrame,
    *,
    timestamps: pd.DatetimeIndex,
    symbols: Sequence[str],
    required_by_side: Mapping[str, Sequence[str]],
    missing_columns_by_symbol: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    union = sorted(set(required_by_side["long"]).union(required_by_side["short"]))
    details: list[dict[str, Any]] = []
    hourly: list[dict[str, Any]] = []
    for timestamp in timestamps:
        local = raw_rows.loc[raw_rows["__ts__"].eq(timestamp)].set_index("__symbol__")
        complete_symbols = 0
        incomplete_reasons: dict[str, list[str]] = {}
        for symbol in symbols:
            reasons = [f"missing_column:{column}" for column in missing_columns_by_symbol[symbol]]
            if symbol not in local.index or not pd.Series(
                [local.loc[symbol, "__source_row_present__"]]
            ).eq(True).all():
                reasons.append("missing_timestamp")
            else:
                row = local.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    reasons.append("duplicate_source_row")
                else:
                    for column in union:
                        if column not in row.index:
                            reasons.append(f"missing_column:{column}")
                            continue
                        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                        if not np.isfinite(value):
                            reasons.append(f"nonfinite:{column}")
            if reasons:
                incomplete_reasons[symbol] = sorted(set(reasons))
                details.append({"__ts__": timestamp, "__symbol__": symbol, "reasons": incomplete_reasons[symbol]})
            else:
                complete_symbols += 1
        complete = complete_symbols == len(symbols) and bool(symbols)
        hourly.append(
            {
                "signal_utc": timestamp,
                "execution_decision_utc": timestamp + pd.Timedelta(hours=1),
                "source_symbol_rows": int(
                    local["__source_row_present__"].eq(True).sum()
                ),
                "expected_symbols": int(len(symbols)),
                "complete_symbols": int(complete_symbols),
                "candidate_rows_if_complete": int(2 * complete_symbols),
                "both_sides_complete": bool(complete_symbols > 0),
                "all_required_point_in_time_features_complete": bool(complete),
                "complete": bool(complete),
                "incomplete_symbol_count": int(len(incomplete_reasons)),
            }
        )
    hourly_frame = pd.DataFrame(hourly)
    daily = (
        hourly_frame.assign(utc_date=hourly_frame["execution_decision_utc"].dt.floor("D"))
        .groupby("utc_date", as_index=False)
        .agg(
            hours=("signal_utc", "size"),
            complete_hours=("complete", "sum"),
            expected_candidate_rows=("candidate_rows_if_complete", "sum"),
            both_sides_complete=("both_sides_complete", "all"),
            all_required_point_in_time_features_complete=("all_required_point_in_time_features_complete", "all"),
        )
    )
    daily["complete"] = daily["all_required_point_in_time_features_complete"].astype(bool)
    return hourly_frame, daily, details


def materialize(
    *,
    features_dir: Path,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    output_dir: Path,
    contracts: Mapping[str, Mapping[str, Sequence[str]]],
    universe_path: Path | None = None,
    frozen_ae_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize candidates or persist a fail-closed preflight report."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    start = _utc(start, name="start").floor("h")
    end_exclusive = _utc(end_exclusive, name="end_exclusive").floor("h")
    if end_exclusive <= start:
        raise ValueError("end_exclusive must be after start")
    timestamps = pd.date_range(start, end_exclusive, freq="1h", inclusive="left", tz="UTC")
    requirements = static_requirements(contracts)
    if any("frozen_ae_raw" in contracts.get(side, {}) for side in SIDES) and frozen_ae_evidence is None:
        raise ValueError("frozen_ae_evidence is required when frozen_ae_raw contracts are active")
    required = sorted(set(requirements["long"]).union(requirements["short"]))
    universe = _load_universe(features_dir, universe_path)
    symbols = [symbol for symbol, _ in universe]

    rows: list[pd.DataFrame] = []
    missing_columns: dict[str, list[str]] = {}
    source_files: list[dict[str, Any]] = []
    for symbol, path in universe:
        source, missing, schema_columns = _read_symbol_rows(
            path, symbol=symbol, columns=required, timestamps=timestamps
        )
        rows.append(source)
        missing_columns[symbol] = missing
        source_files.append({"symbol": symbol, "path": str(path), "sha256": _sha256(path), "schema_columns": schema_columns})
    raw_rows = pd.concat(rows, ignore_index=True, copy=False)
    hourly, daily, incomplete = _coverage_tables(
        raw_rows,
        timestamps=timestamps,
        symbols=symbols,
        required_by_side=requirements,
        missing_columns_by_symbol=missing_columns,
    )

    output_dir.mkdir(parents=True)
    hourly_path = output_dir / "hourly_raw_coverage.csv"
    daily_path = output_dir / "daily_raw_coverage.csv"
    incomplete_path = output_dir / "incomplete_source_rows.json"
    hourly.to_csv(hourly_path, index=False)
    daily.to_csv(daily_path, index=False)
    _write_json(incomplete_path, {"rows": incomplete})
    source_manifest = {
        "schema": SCHEMA,
        "status": "retrospective_non_promotable_preflight",
        "retrospective_reason": "July 20--23 source lock is not the later frozen July-27 confirmation lock",
        "outcomes_used": False,
        "feature_availability": "canonical static feature timestamp is the point-in-time availability timestamp",
        "window": {"start_inclusive_utc": start, "end_exclusive_utc": end_exclusive},
        "universe": {"source": str(universe_path) if universe_path else "all_symbol_parquet_files", "symbols": symbols, "count": len(symbols)},
        "contracts": {"raw_static_columns_by_side": requirements, "roles_by_side": contracts},
        "frozen_side_local_ae_gmm": frozen_ae_evidence,
        "source_files": source_files,
        "coverage": {
            "hourly": {"path": hourly_path, "sha256": _sha256(hourly_path)},
            "daily": {"path": daily_path, "sha256": _sha256(daily_path)},
            "incomplete_rows": {"path": incomplete_path, "sha256": _sha256(incomplete_path), "rows": len(incomplete)},
            "complete_hours": int(hourly["complete"].sum()),
            "requested_hours": int(len(hourly)),
        },
    }
    if not bool(hourly["complete"].all()):
        source_manifest["status"] = "blocked_incomplete_point_in_time_static_surface"
        source_manifest["candidates_written"] = False
        manifest_path = output_dir / "source_manifest.json"
        _write_json(manifest_path, source_manifest)
        raise CandidateSurfacePreflightError(
            "static feature surface is incomplete; see " + str(manifest_path)
        )

    raw = raw_rows.loc[:, ["__ts__", "__symbol__", *required]].copy()
    for column in required:
        raw[column] = pd.to_numeric(raw[column], errors="raise").astype(np.float32)
    candidates: list[pd.DataFrame] = []
    for side, sign in (("long", 1.0), ("short", -1.0)):
        # Retain the complete validated union on both streams.  A scorer only
        # selects its side-local contract, but carrying a partial union would
        # introduce artificial NaNs and hide an incomplete source surface.
        local = raw.copy()
        local["side_name"] = side
        local["side"] = np.float32(sign)
        local["execution_decision_utc"] = local["__ts__"] + pd.Timedelta(hours=1)
        local["feature_available_at"] = local["__ts__"]
        local["candidate_id"] = deterministic_candidate_ids(local, timeframe="1h")
        candidates.append(local)
    candidate_frame = pd.concat(candidates, ignore_index=True, copy=False).sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="mergesort"
    ).reset_index(drop=True)
    if candidate_frame.duplicated(list(IDENTITY)).any() or candidate_frame["candidate_id"].duplicated().any():
        raise AssertionError("candidate identity must be unique")
    if (candidate_frame["feature_available_at"] > candidate_frame["execution_decision_utc"]).any():
        raise AssertionError("point-in-time feature availability occurs after decision")
    candidates_path = output_dir / "candidate_features.parquet"
    candidate_frame.to_parquet(candidates_path, index=False, compression="zstd")
    source_manifest["status"] = "materialized_retrospective_non_promotable"
    source_manifest["candidates_written"] = True
    source_manifest["output"] = {"path": candidates_path, "sha256": _sha256(candidates_path), "rows": int(len(candidate_frame)), "columns": int(len(candidate_frame.columns))}
    manifest_path = output_dir / "source_manifest.json"
    _write_json(manifest_path, source_manifest)
    return {"candidates": candidates_path, "source_manifest": manifest_path, "hourly_coverage": hourly_path, "daily_coverage": daily_path}


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True, help="Canonical data_perp/features/YYYYMMDD_HHMMSS surface.")
    parser.add_argument("--start", default="2026-07-20T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-07-24T00:00:00Z")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--universe-file", type=Path, help="Optional CSV with deterministic __symbol__ or symbol universe.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--role-root", type=Path, default=DEFAULT_ROLE_ROOT)
    parser.add_argument("--catboost-root", type=Path, default=DEFAULT_CATBOOST_ROOT)
    parser.add_argument("--head-contract", type=Path, default=DEFAULT_HEAD_CONTRACT)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    ae_features_by_side, frozen_ae_evidence = load_frozen_ae_contracts(args.ae_root)
    contracts = collect_frozen_contracts(
        spec_path=args.spec,
        role_root=args.role_root,
        catboost_root=args.catboost_root,
        head_contract_path=args.head_contract,
        ae_features_by_side=ae_features_by_side,
    )
    result = materialize(
        features_dir=args.features_dir,
        start=pd.Timestamp(args.start),
        end_exclusive=pd.Timestamp(args.end_exclusive),
        output_dir=args.output_dir,
        contracts=contracts,
        universe_path=args.universe_file,
        frozen_ae_evidence=frozen_ae_evidence,
    )
    print(json.dumps(_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
