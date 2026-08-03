"""Immutable Stage-II winner publication and locked OOS evidence contract.

The in-memory Stage-II funnel is intentionally a *development* selector.  This
module is the separate release boundary: it publishes the selected development
winner atomically, binds it to the exact Stage-I base artifact, and accepts a
later already-scored OOS ledger only after exhaustive identity, timing and
strict-OOF lineage checks.  It neither fits a model nor reads source data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "stage_ii_locked_oos_v1"
_DIRECT_R3_SEMANTICS = "same_side_direct_strict_oof_probabilities_without_conversion"
_META_SEMANTICS = "raw_predicted_residual_bps"
_TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


class StageIIProductionError(ValueError):
    """A release/OOS contract was incomplete, mutable, or non-causal."""


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise StageIIProductionError(f"{name} must be a valid timestamp")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _sha(value: str, *, name: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", value) or len(set(value)) == 1:
        raise StageIIProductionError(f"{name} must be a non-placeholder SHA-256")
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _digest_bytes(path: Path) -> str:
    hash_ = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hash_.update(block)
    return hash_.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _identity_digest(frame: pd.DataFrame, *, columns: Sequence[str]) -> str:
    if any(column not in frame for column in columns):
        raise StageIIProductionError("identity digest is missing a required identity column")
    text = frame.loc[:, list(columns)].astype("string").fillna("<NA>")
    records = ("|".join(row) for row in text.sort_values(list(columns), kind="stable").to_numpy(str))
    hash_ = sha256()
    for record in records:
        hash_.update(record.encode("utf-8"))
        hash_.update(b"\n")
    return hash_.hexdigest()


def _content_digest(frame: pd.DataFrame) -> str:
    """Stable digest of observed OOS content, not merely row identity."""
    columns = sorted(map(str, frame.columns))
    # The validated ledger is already in canonical decision/candidate order.
    # Hashing pandas' stable row fingerprints avoids materialising a many-GB
    # all-string CSV for the production OOS population.
    values = pd.util.hash_pandas_object(
        frame.loc[:, columns], index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    return sha256(values.tobytes()).hexdigest()


def _feature_contract_hash(manifest: "StageIIWinnerManifest") -> str:
    return sha256(json.dumps({"meta": manifest.ordered_meta_features, "archetype": manifest.ordered_archetype_features}, separators=(",", ":")).encode("utf-8")).hexdigest()


def _fold_lineage_hash(folds: Sequence["StageIIFoldLineage"]) -> str:
    return sha256(json.dumps([asdict(value) for value in folds], sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _bool(series: pd.Series, *, name: str) -> np.ndarray:
    output: list[bool] = []
    for value in series.to_numpy(dtype=object):
        if isinstance(value, (bool, np.bool_)):
            output.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            output.append(bool(value))
        else:
            raise StageIIProductionError(f"{name} must contain only explicit booleans/0/1")
    return np.asarray(output, dtype=bool)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise StageIIProductionError(f"OOS ledger lacks {column!r}")
    value = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(value).all():
        raise StageIIProductionError(f"{column!r} must be finite")
    return value


@dataclass(frozen=True)
class StageIIWindowContract:
    """Non-overlapping decision-time intervals for history, dev and locked OOS."""

    history_start: Any
    history_end: Any
    development_start: Any
    development_end: Any
    locked_evaluation_start: Any
    locked_evaluation_end: Any

    def validate(self) -> None:
        values = [_timestamp(getattr(self, name), name=name) for name in (
            "history_start", "history_end", "development_start", "development_end",
            "locked_evaluation_start", "locked_evaluation_end",
        )]
        if not (
            values[0] < values[1] <= values[2] < values[3] <= values[4] < values[5]
        ):
            raise StageIIProductionError(
                "history, development and locked-evaluation windows must be strictly ordered and non-overlapping"
            )

    def contains(self, value: pd.Series, *, interval: str) -> np.ndarray:
        self.validate()
        start = _timestamp(getattr(self, f"{interval}_start"), name=f"{interval}_start")
        end = _timestamp(getattr(self, f"{interval}_end"), name=f"{interval}_end")
        return value.ge(start).to_numpy() & value.lt(end).to_numpy()


@dataclass(frozen=True)
class StageIIFoldLineage:
    fold_id: int
    train_max_label_available_ts: Any
    validation_start_ts: Any
    validation_end_ts: Any | None = None

    def validate(self) -> None:
        if int(self.fold_id) < 0:
            raise StageIIProductionError("fold_id must be non-negative")
        train_max = _timestamp(self.train_max_label_available_ts, name="train_max_label_available_ts")
        start = _timestamp(self.validation_start_ts, name="validation_start_ts")
        if not train_max < start:
            raise StageIIProductionError("fold lineage must use only prior-resolved labels")
        if self.validation_end_ts is not None and not start < _timestamp(self.validation_end_ts, name="validation_end_ts"):
            raise StageIIProductionError("fold validation end must follow its validation start")


@dataclass(frozen=True)
class StageIIWinnerManifest:
    run_id: str
    dataset_id: str
    dataset_sha256: str
    label_manifest_id: str
    label_manifest_sha256: str
    universe_id: str
    universe_sha256: str
    code_revision: str
    stage_i_base_winner_artifact_id: str
    stage_i_base_winner_artifact_sha256: str
    stage_i_base_oof_ledger_sha256: str
    selected_discovery_candidate_id: str
    selected_control_arm: str
    selected_config: Mapping[str, Any]
    ordered_meta_features: tuple[str, ...]
    ordered_archetype_features: tuple[str, ...]
    development_identity_sha256: str
    window: StageIIWindowContract
    schema: str = SCHEMA

    def validate(self) -> None:
        for name in (
            "run_id", "dataset_id", "label_manifest_id", "universe_id", "code_revision",
            "stage_i_base_winner_artifact_id", "selected_discovery_candidate_id", "selected_control_arm",
        ):
            if not str(getattr(self, name)).strip():
                raise StageIIProductionError(f"{name} must be non-empty")
        for name in (
            "dataset_sha256", "label_manifest_sha256", "universe_sha256",
            "stage_i_base_winner_artifact_sha256", "stage_i_base_oof_ledger_sha256",
            "development_identity_sha256",
        ):
            _sha(getattr(self, name), name=name)
        revision = str(self.code_revision)
        if not (re.fullmatch(r"[0-9a-f]{7,}", revision) or re.fullmatch(r"sha256:[0-9a-f]{64}", revision)):
            raise StageIIProductionError("code_revision must be git-like >=7 hex or sha256:<64 hex>")
        if self.schema != SCHEMA:
            raise StageIIProductionError("unsupported Stage-II release schema")
        if not isinstance(self.selected_config, Mapping) or not self.selected_config:
            raise StageIIProductionError("selected_config must be a non-empty frozen mapping")
        for name in ("ordered_meta_features", "ordered_archetype_features"):
            values = tuple(map(str, getattr(self, name)))
            if not values or any(not value.strip() for value in values) or len(set(values)) != len(values):
                raise StageIIProductionError(f"{name} must be a unique non-empty ordered feature list")
        self.window.validate()


def publish_stage_ii_winner_bundle(
    output_directory: str | Path,
    *,
    manifest: StageIIWinnerManifest,
    development_identity: pd.DataFrame,
    development_metrics: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    control_metrics: pd.DataFrame,
) -> Path:
    """Atomically publish the frozen development winner and compact evidence."""
    manifest.validate()
    identity_columns = (
        "candidate_id", "symbol", "signal_close_ts", "decision_ts", "label_available_ts", "side_name",
    )
    if _identity_digest(development_identity, columns=identity_columns) != manifest.development_identity_sha256:
        raise StageIIProductionError("development identity digest does not match the frozen winner manifest")
    dev = development_identity.loc[:, identity_columns].copy()
    for name in ("signal_close_ts", "decision_ts", "label_available_ts"):
        dev[name] = pd.to_datetime(dev[name], utc=True, errors="coerce")
    if dev.isna().any().any() or any(dev[name].astype("string").str.strip().eq("").any() for name in ("candidate_id", "symbol", "side_name")) or dev.duplicated(["candidate_id", "symbol", "decision_ts", "side_name"]).any():
        raise StageIIProductionError("development winner identity is incomplete or duplicated")
    if not np.allclose((dev.decision_ts - dev.signal_close_ts).dt.total_seconds() / 3600, 1.0) or not np.allclose((dev.label_available_ts - dev.signal_close_ts).dt.total_seconds() / 3600, 13.0):
        raise StageIIProductionError("development winner timing must use close +1h entry and +13h labels")
    if not manifest.window.contains(dev.decision_ts, interval="development").all():
        raise StageIIProductionError("development identity extends outside the frozen development window")
    if not all(isinstance(value, pd.DataFrame) for value in (development_metrics, candidate_audit, control_metrics)):
        raise StageIIProductionError("winner publication requires DataFrame development evidence")
    output = Path(output_directory).resolve()
    if output.exists():
        raise StageIIProductionError(f"immutable winner output already exists: {output}")
    if not output.parent.exists():
        raise StageIIProductionError("winner output parent does not exist")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        development_metrics.to_parquet(temporary / "development_metrics.parquet", index=False, compression="zstd")
        candidate_audit.to_parquet(temporary / "candidate_audit.parquet", index=False, compression="zstd")
        control_metrics.to_parquet(temporary / "control_metrics.parquet", index=False, compression="zstd")
        dev.to_parquet(temporary / "development_identity.parquet", index=False, compression="zstd")
        _write_json(temporary / "winner_manifest.json", asdict(manifest))
        _write_json(temporary / "feature_contract.json", {
            "ordered_meta_features": manifest.ordered_meta_features,
            "ordered_archetype_features": manifest.ordered_archetype_features,
            "feature_contract_sha256": _feature_contract_hash(manifest),
        })
        files = sorted(path for path in temporary.iterdir() if path.is_file())
        _write_json(temporary / "checksums.json", {path.name: _digest_bytes(path) for path in files})
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def load_stage_ii_winner_bundle(path: str | Path) -> StageIIWinnerManifest:
    """Load only a complete, checksummed immutable winner bundle."""
    root = Path(path).resolve()
    required = ("winner_manifest.json", "checksums.json", "feature_contract.json")
    if not root.is_dir() or any(not (root / name).is_file() for name in required):
        raise StageIIProductionError("winner bundle is incomplete")
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    if not isinstance(checksums, Mapping):
        raise StageIIProductionError("winner checksums are malformed")
    for name, expected in checksums.items():
        file = root / str(name)
        if not file.is_file() or _digest_bytes(file) != expected:
            raise StageIIProductionError("winner bundle checksum mismatch")
    raw = json.loads((root / "winner_manifest.json").read_text(encoding="utf-8"))
    raw["window"] = StageIIWindowContract(**raw["window"])
    raw["ordered_meta_features"] = tuple(raw["ordered_meta_features"])
    raw["ordered_archetype_features"] = tuple(raw["ordered_archetype_features"])
    manifest = StageIIWinnerManifest(**raw)
    manifest.validate()
    return manifest


@dataclass(frozen=True)
class StageIILockedOOSColumns:
    candidate_id: str = "candidate_id"
    symbol: str = "symbol"
    signal_close_ts: str = "signal_close_ts"
    decision_ts: str = "decision_ts"
    label_available_ts: str = "label_available_ts"
    side: str = "side_name"
    exact_gross_bps: str = "exact_gross_bps"
    exact_net_bps: str = "exact_net_bps"
    total_cost_bps: str = "total_cost_bps"
    base_expected_bps: str = "prequential_base_expected_net_bps"
    base_r3_adverse: str = "r3_p_adverse"
    base_r3_weak: str = "r3_p_weak"
    base_r3_clear: str = "r3_p_clear"
    base_r3_contrast: str = "r3_raw_clear_minus_adverse"
    base_strict_oof: str = "base_is_strict_oof"
    base_source_side: str = "base_source_side"
    base_semantics: str = "base_score_semantics"
    base_fold_id: str = "base_oof_fold_id"
    base_train_max_available: str = "base_train_max_label_available_ts"
    base_map_prequential: str = "base_map_is_prequential"
    base_map_source_side: str = "base_map_source_side"
    base_map_max_available: str = "base_map_max_label_available_ts"
    meta_residual_bps: str = "meta_raw_predicted_residual_bps"
    meta_reconstructed_bps: str = "meta_reconstructed_expected_net_bps"
    meta_strict_oof: str = "meta_is_strict_oof"
    meta_source_side: str = "meta_source_side"
    meta_semantics: str = "meta_score_semantics"
    meta_fold_id: str = "meta_oof_fold_id"
    meta_train_max_available: str = "meta_train_max_label_available_ts"


@dataclass(frozen=True)
class StageIILockedOOSRequest:
    winner_bundle: str | Path
    ledger: pd.DataFrame
    base_folds: tuple[StageIIFoldLineage, ...]
    meta_folds: tuple[StageIIFoldLineage, ...]
    stage_i_base_winner_artifact_sha256: str = ""
    stage_i_base_oof_ledger_sha256: str = ""
    # Full identity joins prove both scored layers belong to this candidate.
    base_identity_columns: tuple[str, str, str, str] = (
        "base_candidate_id", "base_symbol", "base_decision_ts", "base_side_name",
    )
    meta_identity_columns: tuple[str, str, str, str] = (
        "meta_candidate_id", "meta_symbol", "meta_decision_ts", "meta_side_name",
    )
    columns: StageIILockedOOSColumns = StageIILockedOOSColumns()
    # Each layer receives its own causal side-local map.  This prevents an
    # accidental meta map from being used to make the base comparison look good.
    base_admission_score: str = "base_causal_21d_side_expected_net_bps"
    base_admission_flag: str = "base_causal_21d_side_admitted_ge_50bps"
    meta_admission_score: str = "meta_causal_21d_side_expected_net_bps"
    meta_admission_flag: str = "meta_causal_21d_side_admitted_ge_50bps"
    base_admission_source_side: str = "base_causal_21d_admission_source_side"
    base_admission_prequential_flag: str = "base_causal_21d_admission_is_prequential"
    base_admission_train_max_available: str = "base_causal_21d_admission_max_label_available_ts"
    base_admission_window_days: str = "base_causal_21d_admission_window_days"
    meta_admission_source_side: str = "meta_causal_21d_admission_source_side"
    meta_admission_prequential_flag: str = "meta_causal_21d_admission_is_prequential"
    meta_admission_train_max_available: str = "meta_causal_21d_admission_max_label_available_ts"
    meta_admission_window_days: str = "meta_causal_21d_admission_window_days"


def _fold_map(folds: Sequence[StageIIFoldLineage], *, layer: str) -> dict[int, StageIIFoldLineage]:
    if not folds:
        raise StageIIProductionError(f"{layer} strict OOS requires fold lineage")
    result: dict[int, StageIIFoldLineage] = {}
    for fold in folds:
        fold.validate()
        if int(fold.fold_id) in result:
            raise StageIIProductionError(f"{layer} fold ids must be unique")
        result[int(fold.fold_id)] = fold
    return result


def _validate_layer_folds(
    work: pd.DataFrame, *, layer: str, fold_column: str, max_available_column: str,
    source_side_column: str, strict_column: str, semantics_column: str,
    expected_semantics: str, folds: Mapping[int, StageIIFoldLineage], columns: StageIILockedOOSColumns,
) -> None:
    if not _bool(work[strict_column], name=strict_column).all():
        raise StageIIProductionError(f"every {layer} OOS row must be strict OOF")
    if not work[source_side_column].astype(str).str.lower().eq(work[columns.side]).all():
        raise StageIIProductionError(f"{layer} source side does not match candidate side")
    if not work[semantics_column].astype(str).eq(expected_semantics).all():
        raise StageIIProductionError(f"{layer} score semantics are not the frozen direct contract")
    ids = pd.to_numeric(work[fold_column], errors="coerce").to_numpy(float)
    if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all() or not set(ids.astype(int)).issubset(folds):
        raise StageIIProductionError(f"{layer} OOS rows reference an unknown fold")
    row_max = pd.to_datetime(work[max_available_column], utc=True, errors="coerce")
    if row_max.isna().any():
        raise StageIIProductionError(f"{layer} train-label cutoff is invalid")
    decision = work[columns.decision_ts]
    for fold_id in np.unique(ids.astype(int)):
        fold = folds[int(fold_id)]
        start = _timestamp(fold.validation_start_ts, name="validation_start_ts")
        end = None if fold.validation_end_ts is None else _timestamp(fold.validation_end_ts, name="validation_end_ts")
        mask = ids.astype(int) == fold_id
        if not decision.loc[mask].ge(start).all() or (end is not None and not decision.loc[mask].lt(end).all()):
            raise StageIIProductionError(f"{layer} row assignment falls outside its validation fold")
        train_max = _timestamp(fold.train_max_label_available_ts, name="train_max_label_available_ts")
        if not row_max.loc[mask].eq(train_max).all() or not row_max.loc[mask].lt(decision.loc[mask]).all():
            raise StageIIProductionError(f"{layer} row cutoff is not its strict prior-resolved fold cutoff")


def _validate_identity_join(work: pd.DataFrame, request: StageIILockedOOSRequest) -> None:
    columns = request.columns
    native = (columns.candidate_id, columns.symbol, columns.decision_ts, columns.side)
    for joined in (request.base_identity_columns, request.meta_identity_columns):
        if any(name not in work for name in joined):
            raise StageIIProductionError("OOS ledger lacks full base/meta identity join fields")
        if not work[joined[0]].astype(str).eq(work[native[0]].astype(str)).all() or not work[joined[1]].astype(str).eq(work[native[1]].astype(str)).all() or not pd.to_datetime(work[joined[2]], utc=True, errors="coerce").eq(work[native[2]]).all() or not work[joined[3]].astype(str).str.lower().eq(work[native[3]]).all():
            raise StageIIProductionError("base/meta full identity join differs from the locked candidate row")


def validate_stage_ii_locked_oos(request: StageIILockedOOSRequest) -> tuple[StageIIWinnerManifest, pd.DataFrame]:
    """Validate an already-scored evaluation ledger without selection/refitting."""
    manifest = load_stage_ii_winner_bundle(request.winner_bundle)
    if request.stage_i_base_winner_artifact_sha256 != manifest.stage_i_base_winner_artifact_sha256 or request.stage_i_base_oof_ledger_sha256 != manifest.stage_i_base_oof_ledger_sha256:
        raise StageIIProductionError("locked OOS request is not bound to the frozen Stage-I base artifact/ledger")
    if not isinstance(request.ledger, pd.DataFrame) or request.ledger.empty:
        raise StageIIProductionError("locked OOS ledger must be a non-empty DataFrame")
    c = request.columns
    required = {
        c.candidate_id, c.symbol, c.signal_close_ts, c.decision_ts, c.label_available_ts, c.side,
        c.exact_gross_bps, c.exact_net_bps, c.total_cost_bps, c.base_expected_bps,
        c.base_r3_adverse, c.base_r3_weak, c.base_r3_clear, c.base_r3_contrast,
        c.base_strict_oof, c.base_source_side, c.base_semantics, c.base_fold_id, c.base_train_max_available,
        c.base_map_prequential, c.base_map_source_side, c.base_map_max_available,
        c.meta_residual_bps, c.meta_reconstructed_bps, c.meta_strict_oof, c.meta_source_side,
        c.meta_semantics, c.meta_fold_id, c.meta_train_max_available,
        request.base_admission_score, request.base_admission_flag, request.meta_admission_score,
        request.meta_admission_flag, request.base_admission_source_side, request.base_admission_prequential_flag,
        request.base_admission_train_max_available, request.base_admission_window_days,
        request.meta_admission_source_side, request.meta_admission_prequential_flag,
        request.meta_admission_train_max_available, request.meta_admission_window_days,
        *manifest.ordered_meta_features, *manifest.ordered_archetype_features,
        *request.base_identity_columns, *request.meta_identity_columns,
    }
    missing = sorted(required.difference(request.ledger.columns))
    if missing:
        raise StageIIProductionError(f"locked OOS ledger lacks fields: {missing[:12]}")
    work = request.ledger.copy()
    for name in (c.signal_close_ts, c.decision_ts, c.label_available_ts):
        work[name] = pd.to_datetime(work[name], utc=True, errors="coerce")
        if work[name].isna().any():
            raise StageIIProductionError(f"{name} must be valid UTC")
    if not np.allclose((work[c.decision_ts] - work[c.signal_close_ts]).dt.total_seconds() / 3600, 1.0):
        raise StageIIProductionError("decision must occur exactly one hour after signal close")
    if not np.allclose((work[c.label_available_ts] - work[c.signal_close_ts]).dt.total_seconds() / 3600, 13.0):
        raise StageIIProductionError("labels must be available exactly signal close +13h")
    if not manifest.window.contains(work[c.decision_ts], interval="locked_evaluation").all():
        raise StageIIProductionError("locked OOS ledger contains a development/history/future evaluation decision")
    work[c.side] = work[c.side].astype(str).str.lower()
    identity = pd.DataFrame({"candidate": work[c.candidate_id].astype("string"), "symbol": work[c.symbol].astype("string"), "decision": work[c.decision_ts], "side": work[c.side]})
    if identity.isna().any().any() or identity[["candidate", "symbol"]].apply(lambda value: value.str.strip().eq("")).any().any() or identity.duplicated().any() or not work[c.side].isin(("long", "short")).all():
        raise StageIIProductionError("locked OOS identity must be unique and use canonical sides")
    _validate_identity_join(work, request)
    gross, net, cost = _numeric(work, c.exact_gross_bps), _numeric(work, c.exact_net_bps), _numeric(work, c.total_cost_bps)
    if not np.allclose(cost, 100.0) or not np.allclose(gross - cost, net, rtol=0.0, atol=1e-4):
        raise StageIIProductionError("locked OOS must use gross - 100bps cost = exact net exactly once")
    simplex = np.column_stack([_numeric(work, c.base_r3_adverse), _numeric(work, c.base_r3_weak), _numeric(work, c.base_r3_clear)])
    if (simplex < 0).any() or not np.allclose(simplex.sum(axis=1), 1.0, atol=1e-6):
        raise StageIIProductionError("direct R3 base output must be a finite probability simplex")
    if not np.allclose(_numeric(work, c.base_r3_contrast), simplex[:, 2] - simplex[:, 0], atol=1e-6):
        raise StageIIProductionError("R3 raw contrast must equal P(clear)-P(adverse)")
    _numeric(work, c.base_expected_bps)
    if not _bool(work[c.base_map_prequential], name=c.base_map_prequential).all() or not work[c.base_map_source_side].astype(str).str.lower().eq(work[c.side]).all():
        raise StageIIProductionError("base expected-net map must be same-side and explicitly prequential")
    base_map_cutoff = pd.to_datetime(work[c.base_map_max_available], utc=True, errors="coerce")
    if base_map_cutoff.isna().any() or not base_map_cutoff.lt(work[c.decision_ts]).all():
        raise StageIIProductionError("base expected-net map uses current/future resolved labels")
    residual, reconstructed = _numeric(work, c.meta_residual_bps), _numeric(work, c.meta_reconstructed_bps)
    if not np.allclose(reconstructed, _numeric(work, c.base_expected_bps) + residual, atol=1e-5):
        raise StageIIProductionError("meta common-bps reconstruction must be base expected bps + raw residual")
    base_folds, meta_folds = _fold_map(request.base_folds, layer="base"), _fold_map(request.meta_folds, layer="meta")
    _validate_layer_folds(work, layer="base", fold_column=c.base_fold_id, max_available_column=c.base_train_max_available, source_side_column=c.base_source_side, strict_column=c.base_strict_oof, semantics_column=c.base_semantics, expected_semantics=_DIRECT_R3_SEMANTICS, folds=base_folds, columns=c)
    _validate_layer_folds(work, layer="meta", fold_column=c.meta_fold_id, max_available_column=c.meta_train_max_available, source_side_column=c.meta_source_side, strict_column=c.meta_strict_oof, semantics_column=c.meta_semantics, expected_semantics=_META_SEMANTICS, folds=meta_folds, columns=c)
    archetypes = work.loc[:, list(manifest.ordered_archetype_features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(archetypes).all():
        raise StageIIProductionError("locked OOS archetype soft features must be finite")
    membership_columns = [name for name in manifest.ordered_archetype_features if name.startswith("meta_conversion_arch_prob__")]
    if len(membership_columns) < 2 or not np.allclose(work.loc[:, membership_columns].to_numpy(float).sum(axis=1), 1.0, atol=1e-6):
        raise StageIIProductionError("locked OOS archetype memberships must remain a soft probability simplex")
    meta_features = work.loc[:, list(manifest.ordered_meta_features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(meta_features).all():
        raise StageIIProductionError("locked OOS selected meta features must be finite and exact-contract complete")
    for layer, score, flag, source, prequential, cutoff, days in (
        ("base", request.base_admission_score, request.base_admission_flag, request.base_admission_source_side, request.base_admission_prequential_flag, request.base_admission_train_max_available, request.base_admission_window_days),
        ("meta", request.meta_admission_score, request.meta_admission_flag, request.meta_admission_source_side, request.meta_admission_prequential_flag, request.meta_admission_train_max_available, request.meta_admission_window_days),
    ):
        if not work[source].astype(str).str.lower().eq(work[c.side]).all() or not _bool(work[prequential], name=prequential).all() or not pd.to_numeric(work[days], errors="coerce").eq(21).all():
            raise StageIIProductionError(f"{layer} 21-day admission lineage is not same-side/prequential/canonical")
        admission_cutoff = pd.to_datetime(work[cutoff], utc=True, errors="coerce")
        if admission_cutoff.isna().any() or not admission_cutoff.lt(work[c.decision_ts]).all():
            raise StageIIProductionError(f"{layer} 21-day admission map uses current/future labels")
        values = pd.to_numeric(work[score], errors="coerce")
        admitted = _bool(work[flag], name=flag)
        if admitted[values.isna().to_numpy()].any():
            raise StageIIProductionError("unmapped admission rows may not be admitted")
        if not np.array_equal(admitted, values.ge(50.0).fillna(False).to_numpy(bool)):
            raise StageIIProductionError(f"{layer} admission flag must exactly equal finite mapped score >= 50bps")
    return manifest, work.sort_values([c.decision_ts, c.candidate_id], kind="stable").reset_index(drop=True)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(pd.Series(left).corr(pd.Series(right), method="spearman")) if len(left) >= 2 and np.std(left) > 0 and np.std(right) > 0 else float("nan")


def _lag1(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:-1], values[1:])[0, 1]) if len(values) >= 3 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0 else float("nan")


def _digest_selected(frame: pd.DataFrame, columns: StageIILockedOOSColumns) -> str:
    return _identity_digest(frame, columns=(columns.candidate_id, columns.symbol, columns.decision_ts, columns.side))


def build_stage_ii_locked_oos_report(request: StageIILockedOOSRequest) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detailed base/meta, raw/admitted pooled-global report from one OOS ledger."""
    _, work = validate_stage_ii_locked_oos(request)
    c = request.columns
    summaries: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    layers = {
        "base": (c.base_expected_bps, request.base_admission_score, request.base_admission_flag),
        "meta": (c.meta_reconstructed_bps, request.meta_admission_score, request.meta_admission_flag),
    }
    for layer, (raw_score, admission_score, admission_flag) in layers.items():
        populations = (
            ("without_21d_admission", work, raw_score),
            ("with_21d_side_local_admission", work.loc[_bool(work[admission_flag], name=admission_flag) & pd.to_numeric(work[admission_score], errors="coerce").notna()], admission_score),
        )
        for scope, population, score in populations:
            ordered = population.sort_values(
                [score, c.side, c.candidate_id, c.symbol, c.decision_ts],
                ascending=[False, True, True, True, True], kind="stable",
            )
            population_score = population[score].to_numpy(float)
            population_net = population[c.exact_net_bps].to_numpy(float)
            summaries.append({
                "record_type": "candidate_population", "layer": layer, "admission_scope": scope,
                "top_fraction": np.nan, "ranking_basis": "no_tail_selection_candidate_population_diagnostic",
                "original_population_rows": len(work), "eligible_rows": len(population), "selected_rows": 0,
                "candidate_score_net_spearman": _spearman(population_score, population_net),
                "candidate_calibration_slope": float(np.polyfit(population_score, population_net, 1)[0]) if len(population) >= 2 and np.std(population_score) > 0 else np.nan,
                "candidate_calibration_intercept": float(np.polyfit(population_score, population_net, 1)[1]) if len(population) >= 2 and np.std(population_score) > 0 else np.nan,
                "feature_coverage": 1.0,
            })
            for fraction in _TOP_FRACTIONS:
                k = max(1, int(np.ceil(fraction * len(work))))
                selected = ordered.head(min(k, len(ordered))).copy()
                net = selected[c.exact_net_bps].to_numpy(float)
                gross = selected[c.exact_gross_bps].to_numpy(float)
                score_values = selected[score].to_numpy(float)
                time_selected = selected.sort_values([c.decision_ts, c.candidate_id], kind="stable")
                residual = time_selected[c.exact_net_bps].to_numpy(float) - time_selected[score].to_numpy(float)
                selected["__month"] = selected[c.decision_ts].dt.strftime("%Y-%m")
                selected["__week"] = selected[c.decision_ts].dt.strftime("%G-W%V")
                selected["__day"] = selected[c.decision_ts].dt.strftime("%Y-%m-%d")
                day_counts = selected.groupby("__day", observed=True).size().to_numpy(float)
                week_counts = selected.groupby("__week", observed=True).size().to_numpy(float)
                common = {"layer": layer, "admission_scope": scope, "top_fraction": fraction, "ranking_basis": "one_pooled_global_common_bps_book_no_local_rerank", "original_population_rows": len(work), "eligible_rows": len(population), "selected_rows": len(selected), "selected_identity_sha256": _digest_selected(selected, c) if len(selected) else None}
                summaries.append({"record_type": "selected_tail", **common, "gross_bps_per_trade": float(gross.mean()) if len(gross) else np.nan, "net_bps_per_trade": float(net.mean()) if len(net) else np.nan, "cost_bps_per_trade": float((gross-net).mean()) if len(net) else np.nan, "gross_bps_sum": float(gross.sum()), "net_bps_sum": float(net.sum()), "score_net_spearman": _spearman(score_values, net), "calibration_slope": float(np.polyfit(score_values, net, 1)[0]) if len(selected) >= 2 and np.std(score_values) > 0 else np.nan, "calibration_intercept": float(np.polyfit(score_values, net, 1)[1]) if len(selected) >= 2 and np.std(score_values) > 0 else np.nan, "signed_residual_mean_bps": float(residual.mean()) if len(net) else np.nan, "signed_residual_lag1_autocorrelation": _lag1(residual), "feature_coverage": 1.0, "max_day_share": float(day_counts.max()/len(selected)) if len(selected) else np.nan, "day_hhi": float(np.square(day_counts / len(selected)).sum()) if len(selected) else np.nan, "max_week_share": float(week_counts.max()/len(selected)) if len(selected) else np.nan, "week_hhi": float(np.square(week_counts / len(selected)).sum()) if len(selected) else np.nan, "max_symbol_share": float(selected[c.symbol].value_counts(normalize=True).max()) if len(selected) else np.nan})
                for dimension, columns in (("month", ["__month"]), ("week", ["__week"]), ("side", [c.side]), ("month_side", ["__month", c.side]), ("week_side", ["__week", c.side])):
                    for keys, group in selected.groupby(columns, sort=True, observed=True):
                        values = keys if isinstance(keys, tuple) else (keys,)
                        record = {**common, "scope": dimension, "month": "__all__", "week": "__all__", "side": "__all__", "selected_rows": len(group), "gross_bps_per_trade": float(group[c.exact_gross_bps].mean()), "net_bps_per_trade": float(group[c.exact_net_bps].mean()), "gross_bps_sum": float(group[c.exact_gross_bps].sum()), "net_bps_sum": float(group[c.exact_net_bps].sum()), "selected_identity_sha256": _digest_selected(group, c)}
                        for name, value in zip(columns, values, strict=True):
                            record[{"__month": "month", "__week": "week", c.side: "side"}[name]] = str(value)
                        contributions.append(record)
    summary = pd.DataFrame(summaries)
    contribution = pd.DataFrame(contributions)
    if not summary.empty:
        for scope in ("month", "week", "month_side", "week_side"):
            worst = contribution.loc[contribution.scope.eq(scope)].groupby(["layer", "admission_scope", "top_fraction"], as_index=False).net_bps_per_trade.min().rename(columns={"net_bps_per_trade": f"worst_{scope}_net_bps_per_trade"})
            summary = summary.merge(worst, on=["layer", "admission_scope", "top_fraction"], how="left", validate="many_to_one")
    return summary, contribution


@dataclass(frozen=True)
class StageIILockedOOSScoringRequest:
    """One-shot scorer contract; it cannot expose a selection/HPO operation."""

    winner_bundle: str | Path
    history: pd.DataFrame
    development: pd.DataFrame
    evaluation_identity: pd.DataFrame
    base_folds: tuple[StageIIFoldLineage, ...]
    meta_folds: tuple[StageIIFoldLineage, ...]
    stage_i_base_winner_artifact_sha256: str
    stage_i_base_oof_ledger_sha256: str
    columns: StageIILockedOOSColumns = StageIILockedOOSColumns()


@dataclass(frozen=True)
class StageIILockedOOSScoringResult:
    ledger: pd.DataFrame
    provenance: Mapping[str, Any]


LockedOOSScorer = Any


def _validate_fit_population(frame: pd.DataFrame, *, name: str, interval: str, label_cutoff: pd.Timestamp, columns: StageIILockedOOSColumns, window: StageIIWindowContract) -> None:
    required = {columns.decision_ts, columns.label_available_ts}
    if not isinstance(frame, pd.DataFrame) or frame.empty or not required.issubset(frame.columns):
        raise StageIIProductionError(f"{name} fit population requires decision and label-availability rows")
    decision = pd.to_datetime(frame[columns.decision_ts], utc=True, errors="coerce")
    available = pd.to_datetime(frame[columns.label_available_ts], utc=True, errors="coerce")
    if decision.isna().any() or available.isna().any() or not window.contains(decision, interval=interval).all():
        raise StageIIProductionError(f"{name} fit population is outside its frozen decision window")
    if not available.lt(label_cutoff).all():
        raise StageIIProductionError(f"{name} labels are unresolved at the next-stage/evaluation cutoff")


def run_stage_ii_locked_oos_scoring(
    request: StageIILockedOOSScoringRequest,
    *,
    scorer: LockedOOSScorer,
) -> tuple[StageIIWinnerManifest, pd.DataFrame]:
    """Call a frozen scorer exactly once, then validate its locked OOS output.

    The scorer receives the immutable manifest and fit/evaluation frames but
    no API for candidate selection or HPO.  Returned provenance must bind the
    result to the same winner, feature contract, model and labels before the
    normal locked-ledger validator accepts it.
    """
    manifest = load_stage_ii_winner_bundle(request.winner_bundle)
    c = request.columns
    if request.stage_i_base_winner_artifact_sha256 != manifest.stage_i_base_winner_artifact_sha256 or request.stage_i_base_oof_ledger_sha256 != manifest.stage_i_base_oof_ledger_sha256:
        raise StageIIProductionError("scoring request is not bound to the frozen Stage-I base artifact/ledger")
    _validate_fit_population(request.history, name="history", interval="history", label_cutoff=_timestamp(manifest.window.development_start, name="development_start"), columns=c, window=manifest.window)
    _validate_fit_population(request.development, name="development", interval="development", label_cutoff=_timestamp(manifest.window.locked_evaluation_start, name="locked_evaluation_start"), columns=c, window=manifest.window)
    identity_required = {c.candidate_id, c.symbol, c.signal_close_ts, c.decision_ts, c.side}
    if not isinstance(request.evaluation_identity, pd.DataFrame) or request.evaluation_identity.empty or not identity_required.issubset(request.evaluation_identity.columns):
        raise StageIIProductionError("locked evaluation identity is incomplete")
    eval_decision = pd.to_datetime(request.evaluation_identity[c.decision_ts], utc=True, errors="coerce")
    if eval_decision.isna().any() or not manifest.window.contains(eval_decision, interval="locked_evaluation").all():
        raise StageIIProductionError("locked scorer received an evaluation identity outside the locked window")
    # Exactly one invocation: no retry, selection, comparison or fallback.
    result = scorer({"winner_manifest": manifest, "history": request.history.copy(), "development": request.development.copy(), "evaluation_identity": request.evaluation_identity.copy(), "reselection_forbidden": True, "hpo_forbidden": True})
    if not isinstance(result, StageIILockedOOSScoringResult) or not isinstance(result.provenance, Mapping):
        raise StageIIProductionError("locked scorer must return a ledger and provenance")
    provenance = result.provenance
    expected_manifest_hash = _digest_bytes(Path(request.winner_bundle) / "winner_manifest.json")
    checks = {
        "winner_manifest_sha256": expected_manifest_hash,
        "feature_contract_sha256": _feature_contract_hash(manifest),
        "label_manifest_sha256": manifest.label_manifest_sha256,
        "stage_i_base_winner_artifact_sha256": manifest.stage_i_base_winner_artifact_sha256,
        "stage_i_base_oof_ledger_sha256": manifest.stage_i_base_oof_ledger_sha256,
        "base_fold_lineage_sha256": _fold_lineage_hash(request.base_folds),
        "meta_fold_lineage_sha256": _fold_lineage_hash(request.meta_folds),
    }
    if any(str(provenance.get(name, "")) != expected for name, expected in checks.items()):
        raise StageIIProductionError("locked scorer provenance is not bound to the frozen winner/feature/label/base contract")
    _sha(provenance.get("model_sha256", ""), name="scorer model_sha256")
    if provenance.get("reselection_forbidden") is not True or provenance.get("hpo_forbidden") is not True or provenance.get("selected_discovery_candidate_id") != manifest.selected_discovery_candidate_id or provenance.get("selected_control_arm") != manifest.selected_control_arm:
        raise StageIIProductionError("locked scorer attempted or failed to forbid reselection/HPO")
    scored = StageIILockedOOSRequest(request.winner_bundle, result.ledger, request.base_folds, request.meta_folds, request.stage_i_base_winner_artifact_sha256, request.stage_i_base_oof_ledger_sha256, columns=c)
    validated_manifest, ledger = validate_stage_ii_locked_oos(scored)
    identity_columns = (c.candidate_id, c.symbol, c.signal_close_ts, c.decision_ts, c.side)
    if _identity_digest(ledger, columns=identity_columns) != _identity_digest(request.evaluation_identity, columns=identity_columns):
        raise StageIIProductionError("locked scorer ledger does not exactly match the supplied evaluation identity")
    return validated_manifest, ledger


def run_and_publish_stage_ii_locked_oos_scoring(
    output_directory: str | Path,
    request: StageIILockedOOSScoringRequest,
    *,
    scorer: LockedOOSScorer,
) -> Path:
    """One-shot frozen scoring followed by immutable OOS publication."""
    captured: dict[str, Any] = {}

    def capture(context: Mapping[str, Any]) -> StageIILockedOOSScoringResult:
        result = scorer(context)
        if isinstance(result, StageIILockedOOSScoringResult):
            captured.update(dict(result.provenance))
        return result

    _, ledger = run_stage_ii_locked_oos_scoring(request, scorer=capture)
    return publish_stage_ii_locked_oos_bundle(
        output_directory,
        request=StageIILockedOOSRequest(
            request.winner_bundle, ledger, request.base_folds, request.meta_folds,
            request.stage_i_base_winner_artifact_sha256, request.stage_i_base_oof_ledger_sha256,
            columns=request.columns,
        ),
        scorer_model_sha256=str(captured.get("model_sha256", "")),
    )


def publish_stage_ii_locked_oos_bundle(
    output_directory: str | Path, *, request: StageIILockedOOSRequest,
    scorer_model_sha256: str,
) -> Path:
    """Validate then atomically publish the one locked OOS ledger and reports."""
    scorer_model_sha256 = _sha(scorer_model_sha256, name="scorer_model_sha256")
    manifest, ledger = validate_stage_ii_locked_oos(request)
    summary, contribution = build_stage_ii_locked_oos_report(request)
    output = Path(output_directory).resolve()
    if output.exists() or not output.parent.exists():
        raise StageIIProductionError("locked OOS output must have a new path under an existing parent")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        ledger.to_parquet(temporary / "locked_oos_ledger.parquet", index=False, compression="zstd")
        summary.to_parquet(temporary / "oos_tail_summary.parquet", index=False, compression="zstd")
        contribution.to_parquet(temporary / "oos_selected_contributions.parquet", index=False, compression="zstd")
        _write_json(temporary / "run_manifest.json", {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(), "winner_manifest_sha256": _digest_bytes(Path(request.winner_bundle) / "winner_manifest.json"), "locked_evaluation_window": asdict(manifest.window), "oos_identity_sha256": _identity_digest(ledger, columns=(request.columns.candidate_id, request.columns.symbol, request.columns.signal_close_ts, request.columns.decision_ts, request.columns.side)), "oos_content_sha256": _content_digest(ledger), "frozen_feature_contract_sha256": _feature_contract_hash(manifest), "scorer_model_sha256": scorer_model_sha256, "stage_i_base_winner_artifact_sha256": request.stage_i_base_winner_artifact_sha256, "stage_i_base_oof_ledger_sha256": request.stage_i_base_oof_ledger_sha256, "selection_forbidden": True, "reselection_forbidden": True})
        files = sorted(path for path in temporary.iterdir() if path.is_file())
        _write_json(temporary / "checksums.json", {path.name: _digest_bytes(path) for path in files})
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


__all__ = [
    "SCHEMA", "StageIIProductionError", "StageIIFoldLineage", "StageIILockedOOSColumns",
    "StageIILockedOOSRequest", "StageIILockedOOSScoringRequest", "StageIILockedOOSScoringResult",
    "StageIIWindowContract", "StageIIWinnerManifest",
    "build_stage_ii_locked_oos_report", "load_stage_ii_winner_bundle",
    "publish_stage_ii_locked_oos_bundle", "publish_stage_ii_winner_bundle",
    "run_and_publish_stage_ii_locked_oos_scoring", "run_stage_ii_locked_oos_scoring",
    "validate_stage_ii_locked_oos",
]
