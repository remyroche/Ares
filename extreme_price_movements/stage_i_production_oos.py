"""Frozen-winner, strict-OOF production orchestration for Stage I.

This module is intentionally a *boundary* around :mod:`stage_i_strict_oof`.
It does not discover features, tune models, materialise a feature panel, or
rewrite a winning selector result.  Its job is to make a later 2024--26 OOS
generation reproducible and auditable:

``frozen four-cell winner -> side plans -> strict R3/map/residual OOF ->
 pooled-global reports -> atomic immutable artifact``.

The full-period feature-selection reused-backward exception is represented
explicitly in the winner bundle.  It is never described as an untouched
feature-selection result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from .stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
    StageIHeadContract,
)
from .stage_i_strict_oof import (
    SCHEMA as STRICT_OOF_SCHEMA,
    StageIStrictOOFPlan,
    StageIStrictOOFResult,
    generate_stage_i_strict_oof,
    write_stage_i_strict_oof_artifact,
)
from .prequential_r3_value_map import PrequentialR3ValueMapConfig


SCHEMA = "stage_i_production_winner_oos_v1"
_SIDES = ("long", "short")
_TAILS = (0.01, 0.05, 0.10, 0.20)


class StageIProductionOOSError(ValueError):
    """Raised when a proposed OOS run is not a frozen Stage-I contract."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _sha256(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _verify_hash(value: Mapping[str, Any], expected: str, *, label: str) -> None:
    if len(str(expected)) != 64 or _sha256(dict(value)) != str(expected):
        raise StageIProductionOOSError(f"{label} SHA256 does not match its immutable manifest")


def _require_runtime_semantics(params: Mapping[str, Any], *, layer: str, label: str) -> None:
    objective = str(params.get("objective", "")).lower()
    if layer == "base":
        if objective != "multiclass" or int(params.get("num_class", -1)) != 3:
            raise StageIProductionOOSError(f"{label} must freeze objective=multiclass and num_class=3")
    elif objective != "huber":
        raise StageIProductionOOSError(f"{label} must freeze objective=huber")


def _require_immutable_revision(value: str) -> None:
    # A full git SHA, an abbreviated (>=7) SHA, or an explicit immutable code
    # state hash.  Human labels such as "latest" are not replayable lineage.
    valid = bool(re.fullmatch(r"[0-9a-fA-F]{7,64}", value)) or bool(re.fullmatch(r"sha256:[0-9a-fA-F]{64}", value))
    if not valid:
        raise StageIProductionOOSError("code_revision must be a git-like immutable revision or sha256:<64-hex> code-state hash")


def _ordered_features(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    fields = tuple(str(value) for value in values)
    if not fields or any(not value.strip() for value in fields) or len(set(fields)) != len(fields):
        raise StageIProductionOOSError(f"{label} must be a non-empty exact ordered feature list without duplicates")
    return fields


@dataclass(frozen=True)
class StageIWinnerCell:
    """One hash-bound selected list and frozen model contract."""

    contract: StageIHeadContract
    selected_feature_names: tuple[str, ...]
    lgbm_params: Mapping[str, Any]
    selector_manifest: Mapping[str, Any]
    selector_manifest_sha256: str
    source_manifest: Mapping[str, Any]
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        features = _ordered_features(self.selected_feature_names, label=self.contract.artifact_key)
        object.__setattr__(self, "selected_feature_names", features)
        if not isinstance(self.lgbm_params, Mapping) or not dict(self.lgbm_params):
            raise StageIProductionOOSError(f"{self.contract.artifact_key} requires frozen non-empty LGBM parameters")
        _require_runtime_semantics(self.lgbm_params, layer=self.contract.layer, label=self.contract.artifact_key)
        _verify_hash(self.selector_manifest, self.selector_manifest_sha256, label=f"{self.contract.artifact_key} selector manifest")
        _verify_hash(self.source_manifest, self.source_manifest_sha256, label=f"{self.contract.artifact_key} source manifest")
        # Different historical selector schemas name these fields differently.
        # Every exposed feature contract must agree exactly.  HPO manifests may
        # omit the layer-fixed objective fields; normalise only those immutable
        # runtime semantics before comparing the final fit parameters.
        found_feature_manifest = False
        for key in ("selected_feature_names", "stage_i_selected_feature_contract", "selected_feature_contract"):
            manifest_features = self.selector_manifest.get(key)
            if manifest_features is not None:
                found_feature_manifest = True
            if manifest_features is not None and tuple(map(str, manifest_features)) != features:
                raise StageIProductionOOSError(
                    f"{self.contract.artifact_key} selector manifest {key} disagrees with its frozen ordered feature list"
                )
        found_params_manifest = False
        for key in ("lgbm_params", "frozen_lgbm_params", "params", "best_params"):
            manifest_params = self.selector_manifest.get(key)
            if manifest_params is not None:
                found_params_manifest = True
                normalised = dict(manifest_params)
                if self.contract.layer == "base":
                    normalised.setdefault("objective", "multiclass")
                    normalised.setdefault("num_class", 3)
                else:
                    normalised.setdefault("objective", "huber")
                if _canonical_bytes(normalised) != _canonical_bytes(dict(self.lgbm_params)):
                    raise StageIProductionOOSError(
                        f"{self.contract.artifact_key} selector manifest {key} disagrees with frozen runtime parameters"
                    )
        if not found_feature_manifest or not found_params_manifest:
            raise StageIProductionOOSError(
                f"{self.contract.artifact_key} selector manifest must bind one recognised exact feature-list and parameter field"
            )
        if self.contract.layer == "meta":
            missing = [name for name in STAGE_I_META_BASE_OOF_HANDOFF_FEATURES if name not in features]
            if missing:
                raise StageIProductionOOSError(
                    f"{self.contract.artifact_key} omits required direct same-side R3 OOF handoffs: {missing}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": asdict(self.contract),
            "selected_feature_names": list(self.selected_feature_names),
            "lgbm_params": dict(self.lgbm_params),
            "selector_manifest": dict(self.selector_manifest),
            "selector_manifest_sha256": self.selector_manifest_sha256,
            "source_manifest": dict(self.source_manifest),
            "source_manifest_sha256": self.source_manifest_sha256,
        }


@dataclass(frozen=True)
class StageIFeatureSelectionReuseException:
    """The approved full-period selection reused backward exception.

    The representation is deliberately required instead of silently calling a
    historical list a pre-2024 untouched selector outcome.
    """

    approved: bool
    selection_reference_start_utc: str
    selection_reference_end_utc: str
    rationale: str
    disposition: str = "approved_full_period_feature_selection_reused_backward_exception"

    def __post_init__(self) -> None:
        if self.approved is not True:
            raise StageIProductionOOSError("the reuse-backward exception must be explicitly user-approved")
        if self.disposition != "approved_full_period_feature_selection_reused_backward_exception":
            raise StageIProductionOOSError("feature-selection lineage cannot claim untouched historical selection")
        start = pd.Timestamp(self.selection_reference_start_utc)
        end = pd.Timestamp(self.selection_reference_end_utc)
        if pd.isna(start) or pd.isna(end) or not start < end or not str(self.rationale).strip():
            raise StageIProductionOOSError("reuse-backward exception needs an ordered reference window and rationale")


@dataclass(frozen=True)
class StageIOOSCalendar:
    """Calendar and non-negotiable execution/label timing conventions."""

    evaluation_start_utc: str
    evaluation_end_utc: str
    signal_to_decision_hours: float = 1.0
    signal_to_label_available_hours: float = 13.0
    target_horizon_hours: float = 12.0

    def __post_init__(self) -> None:
        start, end = pd.Timestamp(self.evaluation_start_utc), pd.Timestamp(self.evaluation_end_utc)
        if pd.isna(start) or pd.isna(end) or not start < end:
            raise StageIProductionOOSError("OOS calendar needs an ordered UTC evaluation window")
        start_2024 = pd.Timestamp("2024-01-01T00:00:00Z")
        start_2026 = pd.Timestamp("2026-01-01T00:00:00Z")
        start_utc = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end_utc = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if start_utc > start_2024 or end_utc < start_2026:
            raise StageIProductionOOSError("Stage-I production OOS calendar must cover the 2024--2026 evaluation period")
        if (float(self.signal_to_decision_hours), float(self.signal_to_label_available_hours), float(self.target_horizon_hours)) != (1.0, 13.0, 12.0):
            raise StageIProductionOOSError("Stage I requires signal-close -> +1h decision/entry -> +13h exact-H12 availability")


@dataclass(frozen=True)
class StageIProductionWinnerBundle:
    """The only allowed input to a production Stage-I OOS generation."""

    cells: tuple[StageIWinnerCell, ...]
    code_revision: str
    calendar: StageIOOSCalendar
    feature_selection_exception: StageIFeatureSelectionReuseException
    run_id: str = "stage_i_production_oos"
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or not str(self.code_revision).strip() or not str(self.run_id).strip():
            raise StageIProductionOOSError("winner bundle requires schema, code revision and run id")
        _require_immutable_revision(self.code_revision)
        expected = set(STAGE_I_ACTIVE_CONTRACTS)
        observed = [cell.contract for cell in self.cells]
        if set(observed) != expected or len(observed) != len(expected):
            raise StageIProductionOOSError("winner bundle must bind exactly the authorised four Stage-I cells")

    def cell(self, *, layer: str, side: str) -> StageIWinnerCell:
        found = [cell for cell in self.cells if cell.contract.layer == layer and cell.contract.side == side]
        if len(found) != 1:
            raise StageIProductionOOSError(f"winner bundle lacks unique {layer}/{side} cell")
        return found[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "code_revision": self.code_revision,
            "calendar": asdict(self.calendar),
            "feature_selection_exception": asdict(self.feature_selection_exception),
            "cells": [cell.to_dict() for cell in self.cells],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageIProductionWinnerBundle":
        cells = []
        for item in raw.get("cells", []):
            contract = StageIHeadContract(**dict(item["contract"]))
            cells.append(StageIWinnerCell(
                contract=contract,
                selected_feature_names=tuple(item["selected_feature_names"]),
                lgbm_params=dict(item["lgbm_params"]),
                selector_manifest=dict(item["selector_manifest"]),
                selector_manifest_sha256=str(item["selector_manifest_sha256"]),
                source_manifest=dict(item["source_manifest"]),
                source_manifest_sha256=str(item["source_manifest_sha256"]),
            ))
        return cls(
            cells=tuple(cells), code_revision=str(raw["code_revision"]),
            calendar=StageIOOSCalendar(**dict(raw["calendar"])),
            feature_selection_exception=StageIFeatureSelectionReuseException(**dict(raw["feature_selection_exception"])),
            run_id=str(raw.get("run_id", "stage_i_production_oos")), schema=str(raw.get("schema", SCHEMA)),
        )


@dataclass(frozen=True)
class StageISideProductionInput:
    """Narrow, already-loaded side panel passed to the strict generator."""

    side: str
    candidate_ids: Sequence[Any]
    symbols: Sequence[Any]
    signal_close_timestamps: Sequence[Any]
    decision_timestamps: Sequence[Any]
    label_available_timestamps: Sequence[Any]
    frame: pd.DataFrame
    r3_target: Sequence[int]
    exact_net_bps: Sequence[float]
    exact_gross_bps: Sequence[float]
    panel_manifest: Mapping[str, Any]
    panel_manifest_sha256: str
    sample_weight: Sequence[float] | None = None
    n_validation_folds: int = 4
    min_train_rows: int = 500
    materialized_panel_manifest_sha256: str | None = None
    materialized_panel_content_sha256: str | None = None
    selected_feature_readiness: Mapping[str, str] | None = None


def _utc(values: Sequence[Any], *, label: str, n: int) -> pd.Series:
    output = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if len(output) != n or output.isna().any():
        raise StageIProductionOOSError(f"{label} must be aligned finite UTC timestamps")
    return output


def _validate_side_input(
    source: StageISideProductionInput,
    *, calendar: StageIOOSCalendar,
) -> pd.DataFrame:
    side = str(source.side).lower()
    if side not in _SIDES:
        raise StageIProductionOOSError("production OOS input must be side=long or side=short")
    n = len(source.frame)
    if n < 2:
        raise StageIProductionOOSError(f"{side} input needs at least two rows")
    ids = np.asarray(source.candidate_ids, dtype=object).reshape(-1)
    symbols = np.asarray(source.symbols, dtype=object).reshape(-1)
    if len(ids) != n or len(symbols) != n or pd.isna(ids).any() or pd.isna(symbols).any() or len(pd.unique(ids)) != n:
        raise StageIProductionOOSError(f"{side} needs non-null side-unique candidate ids and symbols")
    if any(not str(value).strip() for value in symbols):
        raise StageIProductionOOSError(f"{side} needs non-empty canonical symbols")
    signal = _utc(source.signal_close_timestamps, label="signal_close_timestamps", n=n)
    decision = _utc(source.decision_timestamps, label="decision_timestamps", n=n)
    available = _utc(source.label_available_timestamps, label="label_available_timestamps", n=n)
    if not decision.eq(signal + pd.Timedelta(hours=calendar.signal_to_decision_hours)).all():
        raise StageIProductionOOSError("production Stage-I decision timestamps must be exactly signal-close +1h")
    if not available.eq(signal + pd.Timedelta(hours=calendar.signal_to_label_available_hours)).all():
        raise StageIProductionOOSError("production Stage-I labels must be exactly signal-close +13h")
    start = pd.Timestamp(calendar.evaluation_start_utc, tz="UTC") if pd.Timestamp(calendar.evaluation_start_utc).tzinfo is None else pd.Timestamp(calendar.evaluation_start_utc).tz_convert("UTC")
    end = pd.Timestamp(calendar.evaluation_end_utc, tz="UTC") if pd.Timestamp(calendar.evaluation_end_utc).tzinfo is None else pd.Timestamp(calendar.evaluation_end_utc).tz_convert("UTC")
    # Earlier rows are valid, and necessary, prior-resolved training history.
    # The calendar gates only what is persisted/evaluated after strict OOF is
    # generated across that complete causal history.  Future rows are never a
    # permissible source nor an evaluation row.
    if (signal > end).any() or not signal.between(start, end, inclusive="both").any():
        raise StageIProductionOOSError(f"{side} input must contain evaluation rows and no post-calendar rows")
    _verify_hash(source.panel_manifest, source.panel_manifest_sha256, label=f"{side} panel/input manifest")
    for label, value in (
        ("materialized_panel_manifest_sha256", source.materialized_panel_manifest_sha256),
        ("materialized_panel_content_sha256", source.materialized_panel_content_sha256),
    ):
        if value is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(value)):
            raise StageIProductionOOSError(f"{side} {label} must be an immutable SHA256")
    if source.selected_feature_readiness is not None:
        if not isinstance(source.selected_feature_readiness, Mapping):
            raise StageIProductionOOSError(f"{side} selected_feature_readiness must be a mapping")
        evaluation_start = pd.Timestamp(calendar.evaluation_start_utc)
        evaluation_start = (
            evaluation_start.tz_localize("UTC")
            if evaluation_start.tzinfo is None else evaluation_start.tz_convert("UTC")
        )
        for feature, raw_boundary in source.selected_feature_readiness.items():
            boundary = pd.to_datetime(raw_boundary, utc=True, errors="coerce")
            if pd.isna(boundary) or boundary > evaluation_start:
                raise StageIProductionOOSError(
                    f"{side}/{feature} readiness is invalid or later than required evaluation start"
                )
    identity = pd.DataFrame({
        "candidate_id": ids, "symbol": symbols.astype(str), "signal_close_ts": signal,
        "decision_ts": decision, "side_name": side,
        "source_label_available_ts": available,
        "source_exact_gross_bps": np.asarray(source.exact_gross_bps, dtype=np.float32).reshape(-1),
        "source_exact_net_bps": np.asarray(source.exact_net_bps, dtype=np.float32).reshape(-1),
    })
    if identity.duplicated(["candidate_id", "symbol", "signal_close_ts", "decision_ts", "side_name"]).any():
        raise StageIProductionOOSError("production OOS has duplicate full candidate/symbol/signal-close/decision/side identities")
    if len(identity) != n or not np.isfinite(identity[["source_exact_gross_bps", "source_exact_net_bps"]].to_numpy(dtype=np.float32)).all():
        raise StageIProductionOOSError(f"{side} source labels must be finite and aligned")
    return identity


def build_stage_i_production_plans(
    bundle: StageIProductionWinnerBundle,
    inputs: Sequence[StageISideProductionInput],
) -> tuple[list[StageIStrictOOFPlan], dict[str, pd.DataFrame]]:
    """Validate identity/timing and build one selected-feature plan per side."""
    by_side = {str(source.side).lower(): source for source in inputs}
    if set(by_side) != set(_SIDES) or len(inputs) != 2:
        raise StageIProductionOOSError("production OOS requires exactly one long and one short input")
    plans: list[StageIStrictOOFPlan] = []
    identity: dict[str, pd.DataFrame] = {}
    for side in _SIDES:
        source = by_side[side]
        identity[side] = _validate_side_input(source, calendar=bundle.calendar)
        base, meta = bundle.cell(layer="base", side=side), bundle.cell(layer="meta", side=side)
        if base.source_manifest_sha256 != source.panel_manifest_sha256 or meta.source_manifest_sha256 != source.panel_manifest_sha256:
            raise StageIProductionOOSError(
                f"{side} winner source manifests must reference the exact hash-bound production panel manifest"
            )
        selected_raw = tuple(dict.fromkeys(
            feature
            for cell in (base, meta)
            for feature in cell.selected_feature_names
            if feature not in STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
        ))
        if source.selected_feature_readiness is not None:
            readiness = {str(key): str(value) for key, value in source.selected_feature_readiness.items()}
            if set(readiness) != set(selected_raw):
                raise StageIProductionOOSError(
                    f"{side} readiness contract must cover exactly every selected raw field"
                )
            signal = pd.to_datetime(source.signal_close_timestamps, utc=True, errors="raise")
            for feature in selected_raw:
                boundary = pd.Timestamp(readiness[feature])
                values = pd.to_numeric(source.frame[feature], errors="coerce").to_numpy(float)
                if np.isfinite(values[signal < boundary]).any():
                    raise StageIProductionOOSError(
                        f"{side}/{feature} carries a finite value before its readiness boundary"
                    )
        plans.append(StageIStrictOOFPlan(
            side=side, candidate_ids=source.candidate_ids, frame=source.frame,
            r3_target=source.r3_target, exact_net_bps=source.exact_net_bps,
            exact_gross_bps=source.exact_gross_bps,
            decision_timestamps=source.decision_timestamps,
            label_available_timestamps=source.label_available_timestamps,
            base_feature_names=base.selected_feature_names,
            meta_feature_names=meta.selected_feature_names,
            base_params=base.lgbm_params, residual_params=meta.lgbm_params,
            sample_weight=source.sample_weight, n_validation_folds=source.n_validation_folds,
            min_train_rows=source.min_train_rows,
        ))
    return plans, identity


def _selected_input_content_sha256(
    bundle: StageIProductionWinnerBundle,
    source: StageISideProductionInput,
    identity: pd.DataFrame,
) -> str:
    """Bind the exact identity, labels and selected raw values used by a run."""
    side = str(source.side).lower()
    selected_raw = tuple(dict.fromkeys(
        feature
        for layer in ("base", "meta")
        for feature in bundle.cell(layer=layer, side=side).selected_feature_names
        if feature not in STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
    ))
    missing = sorted(set(selected_raw) - set(source.frame.columns))
    if missing:
        raise StageIProductionOOSError(f"{side} selected input digest lacks raw features: {missing[:12]}")
    r3 = np.asarray(source.r3_target, dtype=np.int8).reshape(-1)
    weight = (
        np.ones(len(identity), dtype=np.float32)
        if source.sample_weight is None
        else np.asarray(source.sample_weight, dtype=np.float32).reshape(-1)
    )
    if len(r3) != len(identity) or len(weight) != len(identity):
        raise StageIProductionOOSError(f"{side} selected input digest vectors are not aligned")
    controls = {
        "n_validation_folds": int(source.n_validation_folds),
        "min_train_rows": int(source.min_train_rows),
        "value_map": asdict(PrequentialR3ValueMapConfig(side=side)),
        "materialized_panel_manifest_sha256": source.materialized_panel_manifest_sha256,
        "materialized_panel_content_sha256": source.materialized_panel_content_sha256,
        "selected_feature_readiness": dict(source.selected_feature_readiness or {}),
    }
    digest = sha256()
    digest.update(_canonical_bytes({
        "schema": "stage_i_selected_input_column_stream_v1",
        "identity_columns": list(identity.columns),
        "selected_raw_features": list(selected_raw),
        "rows": int(len(identity)),
    }))
    # Bound values a column at a time.  This avoids constructing a second
    # all-row wide DataFrame immediately before the strict generator makes its
    # own selected-frame copy.
    for name, values in (
        *[(str(column), identity[column]) for column in identity.columns],
        ("r3_target", pd.Series(r3, copy=False)),
        ("sample_weight", pd.Series(weight, copy=False)),
        *[(f"feature::{feature}", source.frame[feature]) for feature in selected_raw],
    ):
        digest.update(_canonical_bytes({"column": name, "dtype": str(values.dtype)}))
        for start in range(0, len(values), 100_000):
            hashed = pd.util.hash_pandas_object(
                values.iloc[start:start + 100_000], index=False, categorize=True
            ).to_numpy(dtype=np.uint64)
            digest.update(hashed.tobytes())
    digest.update(_canonical_bytes(controls))
    return digest.hexdigest()


def _decorate_result(result: StageIStrictOOFResult, identity: pd.DataFrame) -> StageIStrictOOFResult:
    prediction = result.predictions.copy()
    expected_ids = set(identity["candidate_id"].tolist())
    observed_ids = set(prediction["candidate_id"].tolist())
    if len(prediction) != len(identity) or observed_ids != expected_ids:
        raise StageIProductionOOSError("strict OOF output must preserve the complete frozen candidate population")
    expected_side = str(identity["side_name"].iloc[0])
    expected_keys = prediction["candidate_id"].map(lambda value: f"{expected_side}::{value}")
    if "candidate_key" not in prediction or not prediction["candidate_key"].astype(str).equals(expected_keys.astype(str)):
        raise StageIProductionOOSError("strict OOF candidate_key must equal side::candidate_id exactly")
    if prediction["candidate_id"].duplicated().any() or identity["candidate_id"].duplicated().any():
        raise StageIProductionOOSError("strict result or source identity has duplicate candidate ids")
    expected = identity.set_index("candidate_id").reindex(prediction["candidate_id"].to_numpy())
    if expected.isna().any().any():
        raise StageIProductionOOSError("strict OOF output contains candidates absent from the frozen input identity")
    output_available = pd.to_datetime(prediction["label_available_ts"], utc=True, errors="coerce")
    source_available = pd.to_datetime(expected["source_label_available_ts"], utc=True, errors="coerce")
    if not output_available.reset_index(drop=True).equals(source_available.reset_index(drop=True)):
        raise StageIProductionOOSError("strict OOF label_available_ts differs from the hash-bound source label ledger")
    for output_name, source_name in (("exact_gross_bps", "source_exact_gross_bps"), ("exact_net_bps", "source_exact_net_bps")):
        actual = pd.to_numeric(prediction[output_name], errors="coerce").to_numpy(dtype=np.float32)
        source = pd.to_numeric(expected[source_name], errors="coerce").to_numpy(dtype=np.float32)
        if not np.array_equal(actual, source):
            raise StageIProductionOOSError(f"strict OOF {output_name} differs from the hash-bound source label ledger")
    merged = prediction.merge(
        identity.loc[:, ["candidate_id", "side_name", "decision_ts", "symbol", "signal_close_ts"]],
        on=["candidate_id", "side_name", "decision_ts"], how="left", validate="one_to_one", sort=False,
    )
    if len(merged) != len(prediction) or merged[["symbol", "signal_close_ts"]].isna().any().any():
        raise StageIProductionOOSError("strict OOF output did not preserve full immutable source identity")
    # Reorder the externally merged result back to the generator's row order.
    merged.index = prediction.index
    return StageIStrictOOFResult(
        side=result.side, predictions=merged, fold_provenance=result.fold_provenance,
        value_map_provenance=result.value_map_provenance, plan_summary=result.plan_summary,
    )


def _explicit_boolean(values: pd.Series, *, label: str) -> np.ndarray:
    output: list[bool] = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)):
            output.append(bool(value))
        elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) and float(value) in {0.0, 1.0}:
            output.append(bool(int(value)))
        else:
            raise StageIProductionOOSError(f"{label} must be explicit boolean/0/1 provenance, not a truthy value")
    return np.asarray(output, dtype=bool)


def validate_stage_i_strict_prediction_flags(predictions: pd.DataFrame) -> pd.DataFrame:
    """Require explicit base/meta OOF availability; never infer it from score."""
    required = {
        "base_strict_oof_available", "strict_oof_available", "r3_p_adverse",
        "r3_p_weak", "r3_p_clear", "r3_opportunity_score",
        "prequential_base_expected_net_bps", "residual_oof_bps",
        "reconstructed_expected_net_bps",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise StageIProductionOOSError(f"strict production reporting needs explicit availability fields: {missing}")
    out = predictions.copy()
    base = _explicit_boolean(out["base_strict_oof_available"], label="base_strict_oof_available")
    meta = _explicit_boolean(out["strict_oof_available"], label="strict_oof_available")
    base_score_fields = ["r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score", "prequential_base_expected_net_bps"]
    meta_score_fields = ["residual_oof_bps", "reconstructed_expected_net_bps"]
    if ((~base)[:, None] & np.isfinite(out.loc[:, base_score_fields].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))).any():
        raise StageIProductionOOSError("base non-OOF rows cannot carry finite R3/base scores")
    if ((~meta)[:, None] & np.isfinite(out.loc[:, meta_score_fields].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))).any():
        raise StageIProductionOOSError("meta non-OOF rows cannot carry finite residual/reconstructed scores")
    if (meta & ~base).any():
        raise StageIProductionOOSError("meta strict OOF requires an available strict same-side base OOF handoff")
    out["base_strict_oof_available"] = base
    out["strict_oof_available"] = meta
    return out


def _rank_ic(score: pd.Series, target: pd.Series) -> float:
    clean = pd.DataFrame({"score": pd.to_numeric(score, errors="coerce"), "target": pd.to_numeric(target, errors="coerce")}).dropna()
    return float(clean["score"].corr(clean["target"], method="spearman")) if len(clean) >= 3 else np.nan


def _calibration(score: pd.Series, target: pd.Series) -> tuple[float, float]:
    clean = pd.DataFrame({"score": pd.to_numeric(score, errors="coerce"), "target": pd.to_numeric(target, errors="coerce")}).dropna()
    if len(clean) < 3 or clean["score"].nunique() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(clean["score"].to_numpy(), clean["target"].to_numpy(), 1)
    return float(slope), float(intercept)


def _concentration(selected: pd.DataFrame) -> dict[str, float]:
    if selected.empty:
        return {"daily_hhi": np.nan, "weekly_hhi": np.nan, "symbol_hhi": np.nan, "largest_day_share": np.nan, "largest_week_share": np.nan, "largest_symbol_share": np.nan}
    values: dict[str, float] = {}
    for name, series in {
        "daily": selected["decision_ts"].dt.strftime("%Y-%m-%d"),
        "weekly": selected["decision_ts"].dt.strftime("%G-W%V"),
        "symbol": selected["symbol"].astype(str),
    }.items():
        shares = series.value_counts(normalize=True, sort=False).to_numpy(dtype=float)
        values[f"{name}_hhi"] = float(np.square(shares).sum())
        values[f"largest_{'day' if name == 'daily' else 'week' if name == 'weekly' else 'symbol'}_share"] = float(shares.max())
    return values


def _attribution_rows(selected: pd.DataFrame, common: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    dimensions = {
        "month": ["month"], "week": ["week"], "side": ["side_name"],
        "month_side": ["month", "side_name"], "week_side": ["week", "side_name"],
    }
    for scope, columns in dimensions.items():
        for key, group in selected.groupby(columns[0] if len(columns) == 1 else columns, observed=True, sort=True):
            values = (key,) if not isinstance(key, tuple) else key
            record = dict(common)
            record.update({"row_type": "selected_contribution", "scope": scope, "period_key": "|".join(map(str, values)), "selected_rows": int(len(group)), "realised_net_bps_per_trade": float(group["exact_net_bps"].mean()), "realised_gross_bps_per_trade": float(group["exact_gross_bps"].mean()), "realised_net_total_bps": float(group["exact_net_bps"].sum())})
            result.append(record)
    return result


def _selection_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    layer: str,
    admission_mode: str,
    requested_population_rows: int | None = None,
) -> pd.DataFrame:
    work = frame.loc[np.isfinite(pd.to_numeric(frame[score_column], errors="coerce"))].copy()
    if work.empty:
        requested_population_rows = int(requested_population_rows or 0)
        # A side-local 21-day map can legitimately admit nobody during early
        # support burn-in.  Do not make that result disappear: it is material
        # evidence about admission coverage and must remain comparable with
        # the raw view at every predeclared global tail.
        quality = {
            "layer": layer, "admission_mode": admission_mode, "row_type": "quality",
            "scope": "pooled_global", "period_key": "__all__",
            "candidate_rows": requested_population_rows, "eligible_rows": 0,
            "requested_population_rows": requested_population_rows, "selected_rows": 0,
            "rank_ic_net": np.nan, "calibration_slope": np.nan,
            "calibration_intercept_bps": np.nan,
        }
        rows: list[dict[str, Any]] = [quality]
        for fraction in _TAILS:
            requested_k = max(1, int(np.ceil(float(fraction) * requested_population_rows))) if requested_population_rows else 0
            rows.append({
                "layer": layer, "admission_mode": admission_mode,
                "top_fraction": float(fraction), "candidate_rows": requested_population_rows,
                "eligible_rows": 0, "requested_selected_rows": requested_k,
                "selected_rows": 0, "selection": "pooled_global_once_no_timestamp_or_side_rerank",
                "row_type": "pooled_global", "scope": "pooled_global",
                "period_key": "__all__", "realised_net_bps_per_trade": np.nan,
                "realised_gross_bps_per_trade": np.nan, "realised_net_total_bps": np.nan,
                **_concentration(work),
            })
        return pd.DataFrame(rows)
    work["month"] = work["decision_ts"].dt.strftime("%Y-%m")
    work["week"] = work["decision_ts"].dt.strftime("%G-W%V")
    ordered = work.sort_values([score_column, "candidate_key"], ascending=[False, True], kind="stable")
    slope, intercept = _calibration(work[score_column], work["exact_net_bps"])
    requested_population_rows = int(requested_population_rows if requested_population_rows is not None else len(ordered))
    if requested_population_rows < len(ordered):
        raise StageIProductionOOSError("requested tail population cannot be smaller than the eligible population")
    common_quality = {"layer": layer, "admission_mode": admission_mode, "row_type": "quality", "scope": "pooled_global", "period_key": "__all__", "candidate_rows": int(len(work)), "eligible_rows": int(len(work)), "requested_population_rows": requested_population_rows, "selected_rows": int(len(work)), "rank_ic_net": _rank_ic(work[score_column], work["exact_net_bps"]), "calibration_slope": slope, "calibration_intercept_bps": intercept}
    rows: list[dict[str, Any]] = [common_quality]
    # These are candidate-population diagnostics, not local tail selections.
    # Tail selection below remains a single pooled-global order.
    quality_dimensions = {
        "month": ["month"], "week": ["week"], "side": ["side_name"],
        "month_side": ["month", "side_name"], "week_side": ["week", "side_name"],
    }
    for scope, columns in quality_dimensions.items():
        for key, group in work.groupby(columns[0] if len(columns) == 1 else columns, observed=True, sort=True):
            values = (key,) if not isinstance(key, tuple) else key
            group_slope, group_intercept = _calibration(group[score_column], group["exact_net_bps"])
            rows.append({
                "layer": layer, "admission_mode": admission_mode, "row_type": "quality",
                "scope": scope, "period_key": "|".join(map(str, values)),
                "candidate_rows": int(len(group)), "eligible_rows": int(len(group)),
                "requested_population_rows": requested_population_rows, "selected_rows": int(len(group)),
                "rank_ic_net": _rank_ic(group[score_column], group["exact_net_bps"]),
                "calibration_slope": group_slope, "calibration_intercept_bps": group_intercept,
            })
    for fraction in _TAILS:
        requested_k = max(1, int(np.ceil(float(fraction) * requested_population_rows)))
        # Admission may leave fewer candidates than requested; in that case
        # take all eligible rows while preserving the requested-k denominator
        # visibly in the report.  Never re-rank separately by time or side.
        selected = ordered.head(min(requested_k, len(ordered))).copy()
        common = {"layer": layer, "admission_mode": admission_mode, "top_fraction": float(fraction), "candidate_rows": requested_population_rows, "eligible_rows": int(len(ordered)), "requested_selected_rows": requested_k, "selected_rows": int(len(selected)), "selection": "pooled_global_once_no_timestamp_or_side_rerank"}
        concentration = _concentration(selected)
        rows.append({**common, "row_type": "pooled_global", "scope": "pooled_global", "period_key": "__all__", "realised_net_bps_per_trade": float(selected["exact_net_bps"].mean()), "realised_gross_bps_per_trade": float(selected["exact_gross_bps"].mean()), "realised_net_total_bps": float(selected["exact_net_bps"].sum()), **concentration})
        attributes = _attribution_rows(selected, common)
        rows.extend(attributes)
        for scope in ("month", "week", "month_side", "week_side"):
            choices = [row for row in attributes if row["scope"] == scope]
            if choices:
                worst = min(choices, key=lambda row: (row["realised_net_bps_per_trade"], row["period_key"]))
                rows.append({**worst, "row_type": "worst_period", "scope": f"worst_{scope}"})
    return pd.DataFrame(rows)


def build_stage_i_production_metrics(
    predictions: pd.DataFrame,
    *,
    admission_spec: Causal21dAdmissionSpec,
    calendar: StageIOOSCalendar | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Create base/meta and raw/admitted reports without local re-ranking."""
    required = {"candidate_key", "candidate_id", "symbol", "side_name", "signal_close_ts", "decision_ts", "label_available_ts", "exact_net_bps", "exact_gross_bps", "prequential_base_expected_net_bps", "reconstructed_expected_net_bps"}
    missing = sorted(required - set(predictions.columns))
    if missing or predictions["candidate_key"].duplicated().any():
        raise StageIProductionOOSError(f"cannot report Stage-I OOS metrics; missing/non-unique fields: {missing}")
    predictions = validate_stage_i_strict_prediction_flags(predictions)
    rows: list[pd.DataFrame] = []
    audits: dict[str, pd.DataFrame] = {}
    for layer, score, availability in (
        ("base", "prequential_base_expected_net_bps", "base_strict_oof_available"),
        ("meta_residual", "reconstructed_expected_net_bps", "strict_oof_available"),
    ):
        full_source = predictions.loc[predictions[availability].astype(bool)].copy()
        source = full_source if calendar is None else _evaluation_window(full_source, calendar)
        raw = _selection_metrics(
            source, score_column=score, layer=layer,
            admission_mode="without_21d_admission", requested_population_rows=len(source),
        )
        if not raw.empty:
            rows.append(raw)
        admission_input = full_source.rename(columns={score: "__score__", "exact_net_bps": "net_bps"})
        admitted, audit = apply_causal_21d_side_admission(
            admission_input, score_column="__score__", net_column="net_bps",
            decision_column="decision_ts", label_available_column="label_available_ts",
            identity_column="candidate_key", spec=admission_spec,
        )
        admitted = admitted.rename(columns={"net_bps": "exact_net_bps", "causal_21d_side_expected_net_bps": "__mapped__"})
        if calendar is not None:
            admitted = _evaluation_window(admitted, calendar)
            start, end = _calendar_bounds(calendar)
            audit_snapshot = pd.to_datetime(audit["snapshot_utc"], utc=True, errors="coerce")
            audit = audit.loc[audit_snapshot.between(start.normalize(), end.normalize(), inclusive="both")].copy()
        audit["used_pre_evaluation_reference_history"] = calendar is not None
        accepted = admitted.loc[admitted["causal_21d_side_admitted_ge_50bps"].astype(bool)].copy()
        mapped = _selection_metrics(
            accepted, score_column="__mapped__", layer=layer,
            admission_mode="with_side_local_causal_21d_admission",
            requested_population_rows=len(source),
        )
        if not mapped.empty:
            rows.append(mapped)
        audits[layer] = audit
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), audits


def _calendar_bounds(calendar: StageIOOSCalendar) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(calendar.evaluation_start_utc)
    end = pd.Timestamp(calendar.evaluation_end_utc)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    return start, end


def _evaluation_window(predictions: pd.DataFrame, calendar: StageIOOSCalendar) -> pd.DataFrame:
    start, end = _calendar_bounds(calendar)
    signal = pd.to_datetime(predictions["signal_close_ts"], utc=True, errors="coerce")
    if signal.isna().any():
        raise StageIProductionOOSError("strict production output lacks finite signal-close timestamps")
    output = predictions.loc[signal.between(start, end, inclusive="both")].copy()
    if output.empty:
        raise StageIProductionOOSError("strict OOF generated no rows in the frozen evaluation calendar")
    return output


def _audit_vector(
    *, side: str, layer: str, scope: str, feature: str, values: pd.Series,
    generated_handoff: bool, coverage_basis: str, burnin_exempt: bool,
    first_ready_timestamp_utc: pd.Timestamp | None = None,
    pre_readiness_rows: int = 0,
    pre_readiness_finite_rows: int = 0,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    coverage = float(finite.mean()) if len(finite) else 0.0
    nonconstant = bool(finite.any() and np.unique(numeric[finite]).size > 1)
    return {
        "side_name": side, "layer": layer, "scope": scope,
        "feature_name": feature, "generated_same_side_r3_handoff": generated_handoff,
        "coverage_basis": coverage_basis, "burnin_exempt": burnin_exempt,
        "first_ready_timestamp_utc": first_ready_timestamp_utc,
        "pre_readiness_rows": int(pre_readiness_rows),
        "pre_readiness_finite_rows": int(pre_readiness_finite_rows),
        "rows": int(len(values)), "finite_rows": int(finite.sum()),
        "finite_coverage": coverage, "nonconstant": nonconstant,
        "status": "pass" if coverage >= .90 and nonconstant else "fail",
    }


def build_stage_i_selected_feature_audit(
    bundle: StageIProductionWinnerBundle,
    inputs: Sequence[StageISideProductionInput],
    full_predictions: pd.DataFrame,
    evaluation_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Audit exactly the selected raw and generated handoff features.

    Raw selected fields must be usable on the whole supplied historical panel.
    The five generated base handoffs are evaluated on the base OOF subset for
    full-history coverage, because their unfilled strict burn-in is deliberate;
    all selected fields must still pass on their relevant strict evaluation
    rows.
    """
    full_predictions = validate_stage_i_strict_prediction_flags(full_predictions)
    evaluation_predictions = validate_stage_i_strict_prediction_flags(evaluation_predictions)
    by_side = {str(item.side).lower(): item for item in inputs}
    records: list[dict[str, Any]] = []
    for side in _SIDES:
        source = by_side[side]
        source_frame = source.frame
        prediction = full_predictions.loc[full_predictions["side_name"].eq(side)].set_index("candidate_id")
        evaluation = evaluation_predictions.loc[evaluation_predictions["side_name"].eq(side)].set_index("candidate_id")
        source_ids = pd.Index(np.asarray(source.candidate_ids, dtype=object))
        source_signal = pd.to_datetime(
            source.signal_close_timestamps, utc=True, errors="raise"
        )
        source_position = pd.Series(
            np.arange(len(source_ids), dtype=np.int64), index=source_ids
        )
        prediction = prediction.reindex(source_ids)
        if prediction.index.isna().any() or len(prediction) != len(source_frame):
            raise StageIProductionOOSError(f"{side} feature audit cannot align full prediction lineage to input rows")
        for layer, availability in (("base", "base_strict_oof_available"), ("meta", "strict_oof_available")):
            cell = bundle.cell(layer=layer, side=side)
            strict_eval = evaluation.loc[evaluation[availability].astype(bool)]
            if strict_eval.empty:
                raise StageIProductionOOSError(f"{side}/{layer} has no strict OOF rows in the evaluation window for feature audit")
            for feature in cell.selected_feature_names:
                generated = feature in STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
                if generated:
                    full_values = prediction.loc[prediction["base_strict_oof_available"].astype(bool), feature]
                    evaluation_values = strict_eval[feature]
                    records.append(_audit_vector(
                        side=side, layer=layer, scope="full_input", feature=feature,
                        values=full_values, generated_handoff=True,
                        coverage_basis="base_strict_oof_available_only", burnin_exempt=True,
                    ))
                else:
                    if feature not in source_frame.columns:
                        raise StageIProductionOOSError(f"{side}/{layer} selected raw feature missing from input panel: {feature}")
                    boundary = (
                        pd.Timestamp(source.selected_feature_readiness[feature])
                        if source.selected_feature_readiness is not None
                        else source_signal.min()
                    )
                    before = source_signal < boundary
                    raw_numeric = pd.to_numeric(
                        source_frame[feature], errors="coerce"
                    ).to_numpy(float)
                    pre_finite = int(np.isfinite(raw_numeric[before]).sum())
                    if pre_finite:
                        raise StageIProductionOOSError(
                            f"{side}/{layer}/{feature} is finite before frozen readiness"
                        )
                    records.append(_audit_vector(
                        side=side, layer=layer, scope="full_input", feature=feature,
                        values=source_frame.loc[~before, feature], generated_handoff=False,
                        coverage_basis="post_feature_readiness_rows", burnin_exempt=bool(before.any()),
                        first_ready_timestamp_utc=boundary,
                        pre_readiness_rows=int(before.sum()),
                        pre_readiness_finite_rows=pre_finite,
                    ))
                    positions = source_position.reindex(strict_eval.index)
                    if positions.isna().any():
                        raise StageIProductionOOSError(
                            f"{side}/{layer} strict evaluation identities are absent from input panel"
                        )
                    evaluation_values = source_frame[feature].iloc[
                        positions.to_numpy(dtype=np.int64)
                    ]
                records.append(_audit_vector(
                    side=side, layer=layer, scope="evaluation_strict_oof", feature=feature,
                    values=evaluation_values, generated_handoff=generated,
                    coverage_basis="layer_strict_oof_available", burnin_exempt=False,
                ))
    audit = pd.DataFrame(records)
    if audit.empty or audit["status"].ne("pass").any():
        failures = audit.loc[audit["status"].ne("pass"), ["side_name", "layer", "scope", "feature_name", "finite_coverage", "nonconstant"]]
        raise StageIProductionOOSError(f"selected-feature coverage/nonconstant audit failed: {failures.to_dict(orient='records')[:12]}")
    return audit


def _checksums(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"}


def _load_completed(root: Path, bundle: StageIProductionWinnerBundle) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    if not root.exists():
        return None
    if not manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite incomplete/non-manifest Stage-I artifact: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("winner_bundle_sha256") != bundle.sha256:
        raise FileExistsError(f"refusing to overwrite an artifact with a different frozen winner bundle: {root}")
    if manifest.get("checksums") != _checksums(root):
        raise StageIProductionOOSError(f"existing Stage-I artifact checksum verification failed: {root}")
    return manifest


StrictGenerator = Callable[[StageIStrictOOFPlan], StageIStrictOOFResult]
StrictWriter = Callable[..., Mapping[str, Any]]


def run_stage_i_production_oos(
    bundle: StageIProductionWinnerBundle,
    inputs: Sequence[StageISideProductionInput],
    output_dir: str | Path,
    *,
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(),
    generate: StrictGenerator = generate_stage_i_strict_oof,
    write_strict: StrictWriter = write_stage_i_strict_oof_artifact,
) -> Mapping[str, Any]:
    """Generate the frozen Stage-I OOS artifact, atomically and restart-safely.

    Passing a matching completed output does not re-run fitting.  Any mismatch,
    partial directory, or checksum drift fails closed rather than blending
    results from distinct winner contracts.
    """
    root = Path(output_dir)
    plans, identity = build_stage_i_production_plans(bundle, inputs)
    input_content_sha256 = {
        str(source.side).lower(): _selected_input_content_sha256(
            bundle, source, identity[str(source.side).lower()]
        )
        for source in inputs
    }
    if existing := _load_completed(root, bundle):
        expected_panels = {
            str(source.side).lower(): str(source.panel_manifest_sha256)
            for source in inputs
        }
        if existing.get("selected_input_content_sha256") != input_content_sha256:
            raise StageIProductionOOSError(
                "existing Stage-I artifact has the same winner but different selected input content"
            )
        if existing.get("input_panel_manifest_sha256") != expected_panels:
            raise StageIProductionOOSError(
                "existing Stage-I artifact has the same winner but different panel source binding"
            )
        expected_materialized = {
            str(source.side).lower(): {
                "manifest_sha256": source.materialized_panel_manifest_sha256,
                "content_sha256": source.materialized_panel_content_sha256,
            }
            for source in inputs
        }
        if existing.get("materialized_selected_panel_sha256") != expected_materialized:
            raise StageIProductionOOSError(
                "existing Stage-I artifact has different selected-panel cache lineage"
            )
        return {**existing, "restart_status": "reused_verified_immutable_artifact"}
    full_results = [_decorate_result(generate(plan), identity[plan.side]) for plan in plans]
    full_predictions = validate_stage_i_strict_prediction_flags(
        pd.concat([result.predictions for result in full_results], ignore_index=True)
    )
    evaluation_predictions = _evaluation_window(full_predictions, bundle.calendar)
    evaluation_results: list[StageIStrictOOFResult] = []
    for result in full_results:
        subset = evaluation_predictions.loc[evaluation_predictions["side_name"].eq(result.side)].copy()
        evaluation_results.append(StageIStrictOOFResult(
            side=result.side, predictions=subset, fold_provenance=result.fold_provenance,
            value_map_provenance=result.value_map_provenance, plan_summary=result.plan_summary,
        ))
    feature_audit = build_stage_i_selected_feature_audit(
        bundle, inputs, full_predictions, evaluation_predictions,
    )
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=parent))
    artifact = temporary_parent / root.name
    try:
        strict_manifest = dict(write_strict(
            evaluation_results, artifact, admission_spec=admission_spec,
            admission_reference_results=full_results,
        ))
        # The strict writer receives evaluation rows only, so none of its
        # pooled metrics/admission artifacts can accidentally include older
        # training history.  Preserve the complete pre-calendar lineage in a
        # separate, clearly non-evaluation ledger.
        full_predictions.to_parquet(artifact / "full_history_raw_oof_predictions.parquet", index=False, compression="zstd")
        evaluation_predictions.to_parquet(artifact / "evaluation_window_raw_oof_predictions.parquet", index=False, compression="zstd")
        evaluation_predictions.loc[evaluation_predictions["base_strict_oof_available"].astype(bool)].to_parquet(
            artifact / "evaluation_window_base_strict_oof_predictions.parquet", index=False, compression="zstd"
        )
        evaluation_predictions.loc[evaluation_predictions["strict_oof_available"].astype(bool)].to_parquet(
            artifact / "evaluation_window_meta_strict_oof_predictions.parquet", index=False, compression="zstd"
        )
        metrics, audits = build_stage_i_production_metrics(
            full_predictions, admission_spec=admission_spec, calendar=bundle.calendar,
        )
        metrics.to_parquet(artifact / "detailed_base_meta_21d_pooled_global_metrics.parquet", index=False, compression="zstd")
        feature_audit.to_parquet(artifact / "selected_feature_coverage_audit.parquet", index=False, compression="zstd")
        for layer, audit in audits.items():
            audit.to_parquet(artifact / f"{layer}_causal_21d_admission_audit.parquet", index=False, compression="zstd")
        (artifact / "winner_bundle.json").write_bytes(_canonical_bytes(bundle.to_dict()) + b"\n")
        manifest = {
            **strict_manifest,
            "schema": SCHEMA,
            "status": "complete",
            "strict_oof_schema": STRICT_OOF_SCHEMA,
            "winner_bundle_sha256": bundle.sha256,
            "winner_bundle": bundle.to_dict(),
            "input_panel_manifest_sha256": {
                str(source.side).lower(): str(source.panel_manifest_sha256) for source in inputs
            },
            "materialized_selected_panel_sha256": {
                str(source.side).lower(): {
                    "manifest_sha256": source.materialized_panel_manifest_sha256,
                    "content_sha256": source.materialized_panel_content_sha256,
                }
                for source in inputs
            },
            "selected_input_content_sha256": input_content_sha256,
            "feature_selection_claim": "approved full-period feature-selection reused backward exception; not untouched historical feature selection",
            "identity": "candidate_id + symbol + signal_close_ts + decision_ts + side_name",
            "timing": "signal close -> +1h decision/entry -> +12h H12 path; label_available_ts = signal close +13h",
            "burn_in": "genuine partial strict-OOF base/residual burn-in remains unavailable; no in-sample fill",
            "reporting": "raw and side-local causal 21-day admission, then one pooled-global ranking with attribution only afterwards",
            "files": [
                "full_history_raw_oof_predictions.parquet",
                "evaluation_window_raw_oof_predictions.parquet",
                "evaluation_window_base_strict_oof_predictions.parquet",
                "evaluation_window_meta_strict_oof_predictions.parquet",
                "selected_feature_coverage_audit.parquet",
                "detailed_base_meta_21d_pooled_global_metrics.parquet", "winner_bundle.json",
                *[f"{layer}_causal_21d_admission_audit.parquet" for layer in audits],
            ],
        }
        manifest["checksums"] = _checksums(artifact)
        (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
        os.replace(artifact, root)
        return manifest
    except Exception:
        shutil.rmtree(temporary_parent, ignore_errors=True)
        raise
    finally:
        if temporary_parent.exists():
            temporary_parent.rmdir()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-I frozen-winner production OOS preflight")
    parser.add_argument("--show-contract", action="store_true", help="show the required four winner cells; does not load data")
    parser.add_argument("--winner-bundle", type=Path, help="validate a JSON winner bundle; does not load data or train")
    args = parser.parse_args(argv)
    if args.show_contract:
        for contract in STAGE_I_ACTIVE_CONTRACTS:
            print(contract.artifact_key)
        return 0
    if args.winner_bundle:
        bundle = StageIProductionWinnerBundle.from_dict(json.loads(args.winner_bundle.read_text()))
        print(json.dumps({"schema": bundle.schema, "winner_bundle_sha256": bundle.sha256, "calendar": asdict(bundle.calendar)}, default=_json_default))
        return 0
    parser.error("provide --show-contract or --winner-bundle; production execution is library-driven")
    return 2


__all__ = [
    "SCHEMA", "StageIProductionOOSError", "StageIWinnerCell",
    "StageIFeatureSelectionReuseException", "StageIOOSCalendar",
    "StageIProductionWinnerBundle", "StageISideProductionInput",
    "build_stage_i_production_plans", "build_stage_i_production_metrics",
    "run_stage_i_production_oos", "main",
]
