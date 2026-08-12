"""One-time, frozen final-OOS replay for the leaf-reasoning funnel.

This is deliberately a *consumer* of a completed development decision.  It
does not know how to train, select features, tune a model, cluster leaves, or
choose a successor generation.  Those choices must already be represented by
a hash-bound :class:`FinalOOSReplayContract` before the November 2024 panel is
even read.

The only permitted operation is deterministic inference:

``raw causal November panel -> frozen side-local base -> frozen value map``
``-> frozen side-local residual meta -> one pooled common-bps ranking``.

The module uses native LightGBM text models by default.  Loading an arbitrary
pickle/joblib object is intentionally unsupported at this boundary.  Test
callers may inject a narrow ``model_loader`` instead.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from .leaf_reasoning_meta_funnel import (
    CLUSTER_THRESHOLD_BY_ARM,
    ClusterTaxonomyContract,
    FrozenMetaModelSpec,
    MetaFunnelError,
    S_ALLOWED,
)
from .tp6_transport_validation import (
    TP6_COST_BPS,
    TP6_LABEL_RESOLUTION_HOURS,
    global_common_bps_topk_metrics,
)


SCHEMA = "leaf_reasoning_final_oos_replay_v1"
FINAL_OOS_START = pd.Timestamp("2024-11-01T00:00:00Z")
FINAL_OOS_END = pd.Timestamp("2024-12-01T00:00:00Z")
DEVELOPMENT_CUTOFF = FINAL_OOS_START
SIDES = ("long", "short")
CLASS_ORDER = ("adverse", "weak", "clear")
DERIVED_BASE_COLUMNS = ("p_adverse", "p_weak", "p_clear", "base_expected_bps")
RESERVED_SCORE_COLUMNS = frozenset((*DERIVED_BASE_COLUMNS, "predicted_residual_bps", "common_bps_score"))
RAW_LEAF_ID_TOKENS = ("leaf_id", "leaf_token", "leaf_assignment", "raw_leaf")
REQUIRED_INPUT_COLUMNS = (
    "candidate_id", "side_name", "decision_ts", "entry_ts", "label_available_ts",
    "feature_available_ts", "causal_state_available_ts", "gross_bps", "net_bps",
)
DEFAULT_TOP_FRACTIONS = (0.01, 0.05, 0.10)


class FinalOOSReplayError(ValueError):
    """Raised when a final replay is not an immutable, pre-November contract."""


class FrozenModel(Protocol):
    def predict(self, data: Any, *args: Any, **kwargs: Any) -> Any: ...


ModelLoader = Callable[[Path, str, str], FrozenModel]


def _utc(value: object, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    if pd.isna(result):
        raise FinalOOSReplayError(f"{name} must be a finite UTC timestamp")
    return result


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return _utc(value, name="json timestamp").isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _ordered_fields(values: Sequence[object], *, label: str) -> tuple[str, ...]:
    fields = tuple(map(str, values))
    if not fields or any(not item.strip() for item in fields) or len(set(fields)) != len(fields):
        raise FinalOOSReplayError(f"{label} must be a non-empty exact ordered feature list without duplicates")
    return fields


def _relative_path(path: object, *, root: Path) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else (root / candidate)


@dataclass(frozen=True)
class FrozenArtifact:
    """Hash-bound pre-November artifact used by final inference only."""

    path: Path
    sha256: str
    fit_end_utc: str
    role: str

    def __post_init__(self) -> None:
        path = Path(self.path).resolve()
        if not path.is_file():
            raise FinalOOSReplayError(f"{self.role} artifact is absent: {path}")
        expected = str(self.sha256).lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise FinalOOSReplayError(f"{self.role} artifact SHA-256 is invalid")
        if _sha256_file(path) != expected:
            raise FinalOOSReplayError(f"{self.role} artifact SHA-256 does not match: {path}")
        fit_end = _utc(self.fit_end_utc, name=f"{self.role}.fit_end_utc")
        if fit_end >= DEVELOPMENT_CUTOFF:
            raise FinalOOSReplayError(
                f"{self.role} artifact must be fit strictly before final November OOS, got {fit_end.isoformat()}"
            )
        if not str(self.role).strip():
            raise FinalOOSReplayError("artifact role must be non-empty")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", expected)
        object.__setattr__(self, "fit_end_utc", fit_end.isoformat())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, root: Path, role: str) -> "FrozenArtifact":
        if not isinstance(raw, Mapping):
            raise FinalOOSReplayError(f"{role} artifact declaration must be an object")
        missing = {"path", "sha256", "fit_end_utc"}.difference(raw)
        if missing:
            raise FinalOOSReplayError(f"{role} artifact declaration is missing {sorted(missing)}")
        return cls(
            path=_relative_path(raw["path"], root=root), sha256=str(raw["sha256"]),
            fit_end_utc=str(raw["fit_end_utc"]), role=role,
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256, "fit_end_utc": self.fit_end_utc, "role": self.role}


def _require_json_artifact(artifact: FrozenArtifact, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(artifact.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalOOSReplayError(f"{role} must be a readable JSON artifact") from exc
    if not isinstance(value, Mapping):
        raise FinalOOSReplayError(f"{role} JSON artifact must be an object")
    # These artifacts are selected on development only.  A producer must say
    # so explicitly rather than relying on the path/name of a directory.
    if value.get("final_november_oos_consumed") is True:
        raise FinalOOSReplayError(f"{role} artifact has already consumed final November OOS")
    end = value.get("development_evaluation_end_utc", value.get("evaluation_end_utc"))
    if end is not None and _utc(end, name=f"{role} evaluation end") > DEVELOPMENT_CUTOFF:
        raise FinalOOSReplayError(f"{role} artifact claims evaluation beyond the final-OOS boundary")
    return dict(value)


@dataclass(frozen=True)
class FrozenSideScoringContract:
    side: str
    base_model: FrozenArtifact
    base_feature_columns: tuple[str, ...]
    base_value_map: FrozenArtifact
    meta_model: FrozenArtifact
    meta_feature_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        side = str(self.side).lower()
        if side not in SIDES:
            raise FinalOOSReplayError("side scoring contract must be long or short")
        base = _ordered_fields(self.base_feature_columns, label=f"{side} base feature contract")
        meta = _ordered_fields(self.meta_feature_columns, label=f"{side} meta feature contract")
        missing = sorted(set(DERIVED_BASE_COLUMNS).difference(meta))
        if missing:
            raise FinalOOSReplayError(
                f"{side} meta contract must consume direct same-side base outputs: {missing}"
            )
        forbidden = sorted(set(meta).intersection({"gross_bps", "net_bps", "label_available_ts", "candidate_id", "side_name", "decision_ts", "entry_ts"}))
        if forbidden:
            raise FinalOOSReplayError(f"{side} meta feature contract includes outcome/identity fields: {forbidden}")
        raw_leaf = sorted(
            field for field in meta
            if any(token in field.lower() for token in RAW_LEAF_ID_TOKENS)
            and field != "base_reasoning__g1_leaf_assignment_count"
        )
        if raw_leaf:
            raise FinalOOSReplayError(
                f"{side} final meta contract may consume only compact G1/G2/G3 reasoning summaries, never raw leaf identifiers: {raw_leaf}"
            )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "base_feature_columns", base)
        object.__setattr__(self, "meta_feature_columns", meta)


@dataclass(frozen=True)
class FinalOOSReplayContract:
    """The complete, already-selected development contract required for OOS."""

    development_selection_artifact: FrozenArtifact
    feature_group_artifact: FrozenArtifact
    taxonomy_artifact: FrozenArtifact
    successor_decision_artifact: FrozenArtifact
    frozen_meta_model_spec_artifact: FrozenArtifact
    selected_arm: str
    successor: str
    selected_meta_features_by_side: Mapping[str, Sequence[str]]
    scoring_by_side: Mapping[str, FrozenSideScoringContract]
    causal_state_artifacts: tuple[FrozenArtifact, ...]
    development_transports: tuple[str, ...]
    source_payload: Mapping[str, Any]
    # Produced by the development-only finalizer when it refits the selected
    # native models through the October cutoff.  It remains optional for
    # backwards compatibility with already-issued test/development contracts,
    # but, when supplied, is part of the contract hash and is validated here
    # rather than being an unaudited sidecar.
    finalization_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        arm = str(self.selected_arm).strip()
        if not arm:
            raise FinalOOSReplayError("final replay needs an explicit development-selected meta arm")
        successor = str(self.successor).upper()
        if successor not in S_ALLOWED:
            raise FinalOOSReplayError(f"final replay successor must be one of {S_ALLOWED}")
        transport_set = set(map(str, self.development_transports))
        expected = {"transport_a_2023q4_to_2024h1", "transport_b_2024h1_to_2024h2_to_date"}
        if transport_set != expected or len(self.development_transports) != len(expected):
            raise FinalOOSReplayError(
                "final replay requires exactly the two declared development transports, never November"
            )
        if set(self.scoring_by_side) != set(SIDES):
            raise FinalOOSReplayError("final replay requires one frozen base/meta scoring contract per side")
        selected = {str(side): _ordered_fields(fields, label=f"selected meta features/{side}") for side, fields in self.selected_meta_features_by_side.items()}
        if set(selected) != set(SIDES):
            raise FinalOOSReplayError("selected meta feature decision must cover exactly long and short")
        for side in SIDES:
            scoring = self.scoring_by_side[side]
            if scoring.side != side:
                raise FinalOOSReplayError(f"scoring contract side mismatch for {side}")
            if tuple(scoring.meta_feature_columns) != selected[side]:
                raise FinalOOSReplayError(
                    f"{side} runtime meta features differ from the frozen development-selected feature group"
                )
        if not self.causal_state_artifacts:
            raise FinalOOSReplayError("final replay requires hash-bound frozen causal state artifacts")
        if self.finalization_provenance is not None:
            provenance = dict(self.finalization_provenance)
            if provenance.get("schema") != "leaf_reasoning_finalizer_v1":
                raise FinalOOSReplayError("finalization provenance has an unknown schema")
            cutoff = provenance.get("development_cutoff_utc")
            if cutoff is None or _utc(cutoff, name="finalization provenance cutoff") != DEVELOPMENT_CUTOFF:
                raise FinalOOSReplayError("finalization provenance must bind exactly the November development cutoff")
            required_provenance = {
                "development_selection_sha256", "base_training_data_sha256",
                "meta_training_data_sha256", "causal_feature_contract_sha256",
            }
            missing_provenance = sorted(required_provenance.difference(provenance))
            if missing_provenance:
                raise FinalOOSReplayError(
                    "finalization provenance lacks hash-bound training lineage: "
                    f"{missing_provenance}"
                )
            for name in required_provenance:
                value = str(provenance[name]).lower()
                if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise FinalOOSReplayError(f"finalization provenance has invalid {name}")
            object.__setattr__(self, "finalization_provenance", provenance)
        object.__setattr__(self, "selected_arm", arm)
        object.__setattr__(self, "successor", successor)
        object.__setattr__(self, "selected_meta_features_by_side", selected)
        object.__setattr__(self, "development_transports", tuple(map(str, self.development_transports)))

    @property
    def sha256(self) -> str:
        # Source paths are deliberate lineage.  This hash protects the exact
        # contract that consumes the one-time final OOS, not a mutable label.
        return sha256(_canonical_json(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, root: Path) -> "FinalOOSReplayContract":
        if not isinstance(raw, Mapping):
            raise FinalOOSReplayError("frozen final-OOS contract must be a JSON object")
        if raw.get("schema") != SCHEMA:
            raise FinalOOSReplayError(f"final-OOS contract schema must be {SCHEMA!r}")
        if raw.get("status") != "DEVELOPMENT_SELECTED_FROZEN_FINAL_OOS_CONTRACT":
            raise FinalOOSReplayError("final-OOS contract is not explicitly frozen from a development selection")
        if raw.get("final_november_oos_consumed") is not False:
            raise FinalOOSReplayError("final-OOS contract must explicitly declare final_november_oos_consumed=false")
        selection = raw.get("development_selection")
        if not isinstance(selection, Mapping):
            raise FinalOOSReplayError("final-OOS contract lacks development_selection")
        required = {
            "selected_arm", "successor", "selected_meta_features_by_side", "development_transports",
            "selection_artifact", "feature_group_artifact", "taxonomy_artifact", "successor_decision_artifact",
            "frozen_meta_model_spec_artifact",
        }
        missing = sorted(required.difference(selection))
        if missing:
            raise FinalOOSReplayError(f"development selection lacks {missing}")
        if selection.get("final_november_oos_consumed") is not False:
            raise FinalOOSReplayError("development selection must explicitly retain untouched November OOS")
        end = selection.get("development_evaluation_end_utc")
        if end is None or _utc(end, name="development selection evaluation end") != DEVELOPMENT_CUTOFF:
            raise FinalOOSReplayError("development selection must end exactly at 2024-11-01T00:00:00Z")
        selection_artifact = FrozenArtifact.from_dict(selection["selection_artifact"], root=root, role="development selection")
        group_artifact = FrozenArtifact.from_dict(selection["feature_group_artifact"], root=root, role="feature group")
        taxonomy_artifact = FrozenArtifact.from_dict(selection["taxonomy_artifact"], root=root, role="cluster taxonomy")
        successor_artifact = FrozenArtifact.from_dict(selection["successor_decision_artifact"], root=root, role="successor decision")
        model_spec_artifact = FrozenArtifact.from_dict(
            selection["frozen_meta_model_spec_artifact"], root=root, role="frozen meta model spec",
        )
        score = raw.get("scoring")
        if not isinstance(score, Mapping):
            raise FinalOOSReplayError("final-OOS contract lacks frozen scoring artifacts")
        side_contracts: dict[str, FrozenSideScoringContract] = {}
        for side in SIDES:
            entry = score.get(side)
            if not isinstance(entry, Mapping):
                raise FinalOOSReplayError(f"final-OOS scoring lacks {side} contract")
            fields = {"base_model", "base_feature_columns", "base_value_map", "meta_model", "meta_feature_columns"}
            missing = sorted(fields.difference(entry))
            if missing:
                raise FinalOOSReplayError(f"{side} scoring contract lacks {missing}")
            side_contracts[side] = FrozenSideScoringContract(
                side=side,
                base_model=FrozenArtifact.from_dict(entry["base_model"], root=root, role=f"{side} base model"),
                base_feature_columns=_ordered_fields(entry["base_feature_columns"], label=f"{side} base feature contract"),
                base_value_map=FrozenArtifact.from_dict(entry["base_value_map"], root=root, role=f"{side} base value map"),
                meta_model=FrozenArtifact.from_dict(entry["meta_model"], root=root, role=f"{side} meta model"),
                meta_feature_columns=_ordered_fields(entry["meta_feature_columns"], label=f"{side} meta feature contract"),
            )
        raw_states = raw.get("causal_state_artifacts")
        if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)) or not raw_states:
            raise FinalOOSReplayError("causal_state_artifacts must be a non-empty list")
        states = tuple(FrozenArtifact.from_dict(item, root=root, role=f"causal state {index}") for index, item in enumerate(raw_states))
        contract = cls(
            development_selection_artifact=selection_artifact,
            feature_group_artifact=group_artifact,
            taxonomy_artifact=taxonomy_artifact,
            successor_decision_artifact=successor_artifact,
            frozen_meta_model_spec_artifact=model_spec_artifact,
            selected_arm=str(selection["selected_arm"]), successor=str(selection["successor"]),
            selected_meta_features_by_side=selection["selected_meta_features_by_side"],
            scoring_by_side=side_contracts, causal_state_artifacts=states,
            development_transports=tuple(map(str, selection["development_transports"])), source_payload=dict(raw),
            finalization_provenance=(
                dict(raw["finalization_provenance"])
                if isinstance(raw.get("finalization_provenance"), Mapping) else None
            ),
        )
        if raw.get("finalization_provenance") is not None and not isinstance(raw.get("finalization_provenance"), Mapping):
            raise FinalOOSReplayError("finalization_provenance must be an object when supplied")
        contract._validate_development_bindings()
        return contract

    @classmethod
    def from_json_path(cls, path: str | Path) -> "FinalOOSReplayContract":
        contract_path = Path(path).resolve()
        try:
            raw = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FinalOOSReplayError(f"cannot read frozen final-OOS contract: {contract_path}") from exc
        return cls.from_dict(raw, root=contract_path.parent)

    def _validate_development_bindings(self) -> None:
        """Require each of the four development-only decisions independently.

        It is not sufficient to pass an arm name.  The feature-group, cluster
        taxonomy, and successor decision must each be present and agree with
        the runtime feature/model contract.  This makes a later final replay
        impossible while development selection is still pending.
        """
        selection = _require_json_artifact(self.development_selection_artifact, role="development selection")
        groups = _require_json_artifact(self.feature_group_artifact, role="feature group")
        taxonomy = _require_json_artifact(self.taxonomy_artifact, role="cluster taxonomy")
        successor = _require_json_artifact(self.successor_decision_artifact, role="successor decision")
        model_spec = _require_json_artifact(self.frozen_meta_model_spec_artifact, role="frozen meta model spec")
        for name, item in (("development selection", selection), ("feature group", groups), ("cluster taxonomy", taxonomy), ("successor decision", successor)):
            if item.get("development_only") is not True:
                raise FinalOOSReplayError(f"{name} artifact must explicitly declare development_only=true")
        if selection.get("selected_arm") != self.selected_arm:
            raise FinalOOSReplayError("development selection artifact does not bind the selected meta arm")
        if str(selection.get("successor", "")).upper() != self.successor:
            raise FinalOOSReplayError("development selection artifact does not bind the selected successor")
        if successor.get("successor") is None or str(successor.get("successor")).upper() != self.successor:
            raise FinalOOSReplayError("successor decision artifact does not bind the selected S generation")
        if not str(successor.get("terminal_decision", "")).strip():
            raise FinalOOSReplayError("successor decision artifact must record its explicit S terminal decision")
        try:
            frozen_spec = FrozenMetaModelSpec(
                family=str(model_spec["family"]), params=dict(model_spec["params"]),
                contract_id=str(model_spec["contract_id"]),
            )
        except (KeyError, TypeError, MetaFunnelError) as exc:
            raise FinalOOSReplayError("frozen meta model spec must bind the selected LightGBM Huber objective") from exc
        if not frozen_spec.contract_id.strip():  # pragma: no cover - constructor already defends this
            raise FinalOOSReplayError("frozen meta model spec has no immutable contract identifier")
        selected = groups.get("selected_meta_features_by_side", groups.get("feature_contract"))
        if not isinstance(selected, Mapping):
            raise FinalOOSReplayError("feature group artifact lacks selected_meta_features_by_side")
        normalized = {str(side): tuple(map(str, values)) for side, values in selected.items() if isinstance(values, Sequence) and not isinstance(values, (str, bytes))}
        if normalized != dict(self.selected_meta_features_by_side):
            raise FinalOOSReplayError("feature group artifact differs from the selected side-local meta feature contract")
        if groups.get("selected_arm") != self.selected_arm:
            raise FinalOOSReplayError("feature group artifact does not bind the selected meta arm")
        # The C contract is supplied even if the selected arm eventually
        # chooses no cluster addition.  Validate its allowed linkage and the
        # fixed C1--C4 threshold grid before final scoring.
        linkage = str(taxonomy.get("linkage", "")).lower()
        supplied_thresholds = taxonomy.get("threshold_by_arm", CLUSTER_THRESHOLD_BY_ARM)
        try:
            thresholds = {str(key): float(value) for key, value in dict(supplied_thresholds).items()}
        except (TypeError, ValueError) as exc:
            raise FinalOOSReplayError("cluster taxonomy threshold_by_arm is malformed") from exc
        if thresholds != dict(CLUSTER_THRESHOLD_BY_ARM):
            raise FinalOOSReplayError("cluster taxonomy must retain the predeclared C1=.60/C2=.70/C3=.80/C4=.90 grid")
        try:
            ClusterTaxonomyContract(
                linkage=linkage,
                cluster_ids_by_arm=taxonomy["cluster_ids_by_arm"],
                threshold_by_arm=thresholds,
                c5_source_arm=str(taxonomy.get("c5_source_arm", "C1")),
                c6_source_arm=str(taxonomy.get("c6_source_arm", "C5")),
                top_decile_coverage_target=float(taxonomy.get("top_decile_coverage_target", .95)),
                top_decile_coverage_by_arm=taxonomy.get("top_decile_coverage_by_arm", {}),
                portable_top_decile_coverage_by_arm=taxonomy.get("portable_top_decile_coverage_by_arm", {}),
                production_soft_cap=int(taxonomy.get("production_soft_cap", 12)),
                exploratory_hard_cap=int(taxonomy.get("exploratory_hard_cap", 20)),
                c6_best_cross_era_score=taxonomy.get("c6_best_cross_era_score"),
                c6_best_cross_era_standard_error=taxonomy.get("c6_best_cross_era_standard_error"),
                c6_compact_cross_era_score=taxonomy.get("c6_compact_cross_era_score"),
            )
        except (KeyError, MetaFunnelError, TypeError, ValueError) as exc:
            raise FinalOOSReplayError("cluster taxonomy lacks a valid frozen C1--C6/coverage/one-SE decision") from exc

    def to_dict(self) -> dict[str, Any]:
        output = {
            "schema": SCHEMA,
            "status": "DEVELOPMENT_SELECTED_FROZEN_FINAL_OOS_CONTRACT",
            "final_november_oos_consumed": False,
            "development_selection": {
                "selected_arm": self.selected_arm, "successor": self.successor,
                "selected_meta_features_by_side": {side: list(values) for side, values in self.selected_meta_features_by_side.items()},
                "development_transports": list(self.development_transports),
                "selection_artifact": self.development_selection_artifact.to_dict(),
                "feature_group_artifact": self.feature_group_artifact.to_dict(),
                "taxonomy_artifact": self.taxonomy_artifact.to_dict(),
                "successor_decision_artifact": self.successor_decision_artifact.to_dict(),
                "frozen_meta_model_spec_artifact": self.frozen_meta_model_spec_artifact.to_dict(),
                "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
                "final_november_oos_consumed": False,
            },
            "scoring": {
                side: {
                    "base_model": source.base_model.to_dict(), "base_feature_columns": list(source.base_feature_columns),
                    "base_value_map": source.base_value_map.to_dict(), "meta_model": source.meta_model.to_dict(),
                    "meta_feature_columns": list(source.meta_feature_columns),
                }
                for side, source in self.scoring_by_side.items()
            },
            "causal_state_artifacts": [artifact.to_dict() for artifact in self.causal_state_artifacts],
        }
        if self.finalization_provenance is not None:
            output["finalization_provenance"] = dict(self.finalization_provenance)
        return output


def _default_model_loader(path: Path, role: str, side: str) -> FrozenModel:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise FinalOOSReplayError("LightGBM is required to score native frozen model text artifacts") from exc
    try:
        return lgb.Booster(model_file=str(path))
    except Exception as exc:  # pragma: no cover - third-party parsing
        raise FinalOOSReplayError(f"could not load {side} frozen {role} LightGBM text model: {path}") from exc


def _numeric_matrix(frame: pd.DataFrame, fields: Sequence[str], *, label: str) -> pd.DataFrame:
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        raise FinalOOSReplayError(f"{label} input panel is missing frozen fields: {missing[:16]}")
    # A native LightGBM model supports its frozen missing-value routing.  We
    # deliberately do not fill or impute here; coverage is audited separately.
    output = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    invalid = np.isinf(output.to_numpy(float))
    if invalid.any():
        raise FinalOOSReplayError(f"{label} model input has infinite values")
    return output.astype(np.float32)


def _validate_model_feature_names(model: FrozenModel, fields: Sequence[str], *, role: str, side: str) -> None:
    count = getattr(model, "num_feature", None)
    if callable(count):
        try:
            declared = int(count())
        except Exception:
            declared = None
        if declared is not None and declared != len(fields):
            raise FinalOOSReplayError(
                f"{side} {role} model expects {declared} features but frozen contract declares {len(fields)}"
            )
    names = getattr(model, "feature_name", None)
    if callable(names):
        try:
            declared_names = tuple(map(str, names()))
        except Exception:
            declared_names = ()
        # Native boosters created from unnamed NumPy data expose Column_0...;
        # this is not an alternative feature contract.  Named model fields,
        # however, must bind exactly to prevent order-only substitution.
        if declared_names and not all(name.startswith("Column_") for name in declared_names):
            if declared_names != tuple(map(str, fields)):
                raise FinalOOSReplayError(f"{side} {role} model feature order differs from the frozen contract")


def _base_probabilities(model: FrozenModel, matrix: pd.DataFrame, *, side: str) -> np.ndarray:
    try:
        output = np.asarray(model.predict(matrix), dtype=np.float64)
    except Exception as exc:  # pragma: no cover - third party
        raise FinalOOSReplayError(f"{side} frozen base model prediction failed") from exc
    if output.ndim != 2 or output.shape != (len(matrix), len(CLASS_ORDER)):
        raise FinalOOSReplayError(
            f"{side} frozen base model must emit [rows, adverse/weak/clear] probabilities"
        )
    if not np.isfinite(output).all() or (output < -1e-8).any() or (output > 1.0 + 1e-8).any():
        raise FinalOOSReplayError(f"{side} frozen base probabilities are not valid")
    if not np.allclose(output.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise FinalOOSReplayError(f"{side} frozen base probabilities do not sum to one")
    return output.astype(np.float32)


def _meta_residual(model: FrozenModel, matrix: pd.DataFrame, *, side: str) -> np.ndarray:
    try:
        output = np.asarray(model.predict(matrix), dtype=np.float64)
    except Exception as exc:  # pragma: no cover - third party
        raise FinalOOSReplayError(f"{side} frozen meta model prediction failed") from exc
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or len(output) != len(matrix) or not np.isfinite(output).all():
        raise FinalOOSReplayError(f"{side} frozen meta residual must be one finite bps value per row")
    return output.astype(np.float32)


def _load_value_map(artifact: FrozenArtifact, *, side: str) -> np.ndarray:
    payload = _require_json_artifact(artifact, role=f"{side} base value map")
    if str(payload.get("side_name", side)).lower() != side:
        raise FinalOOSReplayError(f"{side} base value map is side-mismatched")
    if payload.get("fit_end_utc") is not None and _utc(payload["fit_end_utc"], name=f"{side} value map fit end") >= DEVELOPMENT_CUTOFF:
        raise FinalOOSReplayError(f"{side} base value map was not fit before final OOS")
    raw = payload.get("class_expected_net_bps")
    if isinstance(raw, Mapping):
        if set(map(str, raw)) != set(CLASS_ORDER):
            raise FinalOOSReplayError(f"{side} base value map must name adverse/weak/clear exactly")
        values = np.asarray([raw[name] for name in CLASS_ORDER], dtype=np.float64)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == len(CLASS_ORDER):
        values = np.asarray(raw, dtype=np.float64)
    else:
        raise FinalOOSReplayError(f"{side} base value map lacks class_expected_net_bps in canonical class order")
    if not np.isfinite(values).all():
        raise FinalOOSReplayError(f"{side} base value map values must be finite common net bps")
    return values.astype(np.float32)


def _input_panel(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise FinalOOSReplayError("final-OOS input panel must be a non-empty DataFrame")
    missing = sorted(set(REQUIRED_INPUT_COLUMNS).difference(frame.columns))
    if missing:
        raise FinalOOSReplayError(f"final-OOS input panel is missing {missing}")
    reserved = sorted(set(frame.columns).intersection(RESERVED_SCORE_COLUMNS))
    if reserved:
        raise FinalOOSReplayError(
            "final-OOS input may not smuggle base/meta score fields; they must be recomputed from frozen models: "
            f"{reserved}"
        )
    out = frame.copy()
    out["candidate_id"] = out["candidate_id"].astype("string")
    out["side_name"] = out["side_name"].astype("string").str.lower()
    if out["candidate_id"].isna().any() or out["candidate_id"].str.strip().eq("").any() or not set(out["side_name"]).issubset(SIDES):
        raise FinalOOSReplayError("final-OOS input must have non-empty candidate IDs and canonical sides")
    for column in ("decision_ts", "entry_ts", "label_available_ts", "feature_available_ts", "causal_state_available_ts"):
        out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    if out.loc[:, ["decision_ts", "entry_ts", "label_available_ts", "feature_available_ts", "causal_state_available_ts"]].isna().any().any():
        raise FinalOOSReplayError("final-OOS input contains an invalid UTC timestamp")
    if not out["decision_ts"].ge(FINAL_OOS_START).all() or not out["decision_ts"].lt(FINAL_OOS_END).all():
        raise FinalOOSReplayError("final-OOS replay accepts only decision rows in [2024-11-01, 2024-12-01)")
    if not (out["entry_ts"] == out["decision_ts"] + pd.Timedelta(hours=1)).all():
        raise FinalOOSReplayError("final-OOS replay requires entry at the next hourly open after the candidate bar close")
    horizon = (out["label_available_ts"] - out["decision_ts"]).dt.total_seconds().to_numpy(float) / 3600.0
    if not np.allclose(horizon, TP6_LABEL_RESOLUTION_HOURS, rtol=0.0, atol=1e-6):
        raise FinalOOSReplayError("final-OOS outcome labels must be the H12 next-open path resolving 13h after decision")
    if (out["feature_available_ts"] > out["decision_ts"]).any() or (out["causal_state_available_ts"] > out["decision_ts"]).any():
        raise FinalOOSReplayError("feature/state availability may not be after its final-OOS decision timestamp")
    out["candidate_key"] = out["side_name"].astype(str) + "::" + out["candidate_id"].astype(str)
    if out["candidate_key"].duplicated().any():
        raise FinalOOSReplayError("final-OOS candidate identities must be globally unique after side qualification")
    for column in ("gross_bps", "net_bps"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if not np.isfinite(out.loc[:, ["gross_bps", "net_bps"]].to_numpy(float)).all():
        raise FinalOOSReplayError("final-OOS labels must be finite for post-replay reporting")
    if not np.allclose(out["gross_bps"].to_numpy(float) - out["net_bps"].to_numpy(float), TP6_COST_BPS, rtol=0.0, atol=0.02):
        raise FinalOOSReplayError("final-OOS gross/net labels must charge the fixed 100-bps cost exactly once")
    return out


def _feature_coverage(frame: pd.DataFrame, *, side: str, fields: Sequence[str], layer: str, min_coverage: float) -> pd.DataFrame:
    matrix = _numeric_matrix(frame, fields, label=f"{side} {layer}")
    finite = np.isfinite(matrix.to_numpy(float))
    rows = []
    for index, field in enumerate(fields):
        values = matrix.iloc[:, index].to_numpy(float)
        finite_values = values[finite[:, index]]
        coverage = float(finite[:, index].mean())
        rows.append({
            "side_name": side, "layer": layer, "feature": field, "rows": int(len(frame)),
            "finite_rows": int(finite[:, index].sum()), "finite_coverage": coverage,
            "unique_finite_values": int(np.unique(finite_values).size), "minimum_coverage": float(min_coverage),
            "passes_coverage": bool(coverage >= min_coverage),
        })
    audit = pd.DataFrame(rows)
    failed = audit.loc[~audit["passes_coverage"], "feature"].tolist()
    if failed:
        raise FinalOOSReplayError(f"{side} {layer} features violate the frozen coverage contract: {failed[:12]}")
    return audit


def _score_side(
    frame: pd.DataFrame,
    *,
    contract: FrozenSideScoringContract,
    model_loader: ModelLoader,
    min_feature_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    side = contract.side
    raw = frame.loc[frame["side_name"].eq(side)].copy()
    if raw.empty:
        raise FinalOOSReplayError(f"final-OOS input has no {side} candidates")
    base_audit = _feature_coverage(raw, side=side, fields=contract.base_feature_columns, layer="base", min_coverage=min_feature_coverage)
    base_matrix = _numeric_matrix(raw, contract.base_feature_columns, label=f"{side} base")
    base_model = model_loader(contract.base_model.path, "base", side)
    _validate_model_feature_names(base_model, contract.base_feature_columns, role="base", side=side)
    probabilities = _base_probabilities(base_model, base_matrix, side=side)
    class_values = _load_value_map(contract.base_value_map, side=side)
    raw.loc[:, "p_adverse"] = probabilities[:, 0]
    raw.loc[:, "p_weak"] = probabilities[:, 1]
    raw.loc[:, "p_clear"] = probabilities[:, 2]
    raw.loc[:, "base_expected_bps"] = np.matmul(probabilities, class_values).astype(np.float32)
    meta_audit = _feature_coverage(raw, side=side, fields=contract.meta_feature_columns, layer="meta", min_coverage=min_feature_coverage)
    meta_matrix = _numeric_matrix(raw, contract.meta_feature_columns, label=f"{side} meta")
    meta_model = model_loader(contract.meta_model.path, "meta", side)
    _validate_model_feature_names(meta_model, contract.meta_feature_columns, role="meta", side=side)
    residual = _meta_residual(meta_model, meta_matrix, side=side)
    raw.loc[:, "predicted_residual_bps"] = residual
    raw.loc[:, "common_bps_score"] = raw["base_expected_bps"].to_numpy(np.float32) + residual
    raw.loc[:, "base_model_fit_end_utc"] = contract.base_model.fit_end_utc
    raw.loc[:, "base_value_map_fit_end_utc"] = contract.base_value_map.fit_end_utc
    raw.loc[:, "meta_model_fit_end_utc"] = contract.meta_model.fit_end_utc
    raw.loc[:, "base_feature_contract_sha256"] = sha256(_canonical_json(list(contract.base_feature_columns))).hexdigest()
    raw.loc[:, "meta_feature_contract_sha256"] = sha256(_canonical_json(list(contract.meta_feature_columns))).hexdigest()
    raw.loc[:, "base_model_sha256"] = contract.base_model.sha256
    raw.loc[:, "base_value_map_sha256"] = contract.base_value_map.sha256
    raw.loc[:, "meta_model_sha256"] = contract.meta_model.sha256
    return raw, pd.concat([base_audit, meta_audit], ignore_index=True)


def _selected_books(scored: pd.DataFrame, *, top_fractions: Sequence[float]) -> pd.DataFrame:
    ordered = scored.sort_values(["common_bps_score", "candidate_key"], ascending=[False, True], kind="stable")
    frames = []
    for fraction in top_fractions:
        count = max(1, int(np.ceil(len(ordered) * float(fraction))))
        frames.append(ordered.head(count).assign(top_fraction=float(fraction), global_rank=np.arange(1, count + 1, dtype=np.int32)))
    return pd.concat(frames, ignore_index=True)


def _metrics(scored: pd.DataFrame, *, top_fractions: Sequence[float]) -> pd.DataFrame:
    # Reuse the canonical evaluator.  It receives a side-qualified identity in
    # the candidate_id position so a repeated source ID can never create a
    # different cross-side ranking tie-breaker.
    metric_input = scored.copy()
    metric_input["candidate_id"] = metric_input["candidate_key"].astype(str)
    metric_input = metric_input.rename(columns={"gross_bps": "gross_bps", "net_bps": "net_bps"})
    return global_common_bps_topk_metrics(
        metric_input, score_column="common_bps_score", transport="final_oos_2024-11", top_fractions=top_fractions,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _checksum_tree(root: Path) -> dict[str, str]:
    return {
        item.relative_to(root).as_posix(): _sha256_file(item)
        for item in sorted(root.rglob("*")) if item.is_file() and item.name not in {"run_manifest.json", "checksums.json"}
    }


def _acquire_once(registry: Path, *, contract: FinalOOSReplayContract, output: Path) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": f"{SCHEMA}_consumption_v1", "status": "RUNNING_FINAL_OOS_REPLAY",
        "contract_sha256": contract.sha256, "output_dir": str(output.resolve()),
        "started_utc": datetime.now(timezone.utc).isoformat(), "final_oos": "2024-11",
    }
    try:
        descriptor = os.open(str(registry), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FinalOOSReplayError(
            f"the selected frozen contract has already reserved/consumed its one-time final OOS registry: {registry}"
        ) from exc
    try:
        os.write(descriptor, _canonical_json(payload) + b"\n")
    finally:
        os.close(descriptor)


def _complete_once(registry: Path, *, contract: FinalOOSReplayContract, output: Path) -> None:
    payload = {
        "schema": f"{SCHEMA}_consumption_v1", "status": "FINAL_OOS_REPLAY_COMPLETE",
        "contract_sha256": contract.sha256, "output_dir": str(output.resolve()),
        "completed_utc": datetime.now(timezone.utc).isoformat(), "final_oos": "2024-11",
    }
    temporary = registry.with_name(f".{registry.name}.{uuid4().hex}.tmp")
    _write_json(temporary, payload)
    os.replace(temporary, registry)


@dataclass(frozen=True)
class FinalOOSReplayResult:
    output_dir: Path
    scored_predictions: pd.DataFrame
    metrics: pd.DataFrame
    manifest: Mapping[str, Any]


def run_leaf_reasoning_final_oos_replay(
    contract: FinalOOSReplayContract,
    panel: pd.DataFrame,
    *,
    output_dir: str | Path,
    consumption_registry: str | Path,
    model_loader: ModelLoader | None = None,
    min_feature_coverage: float = 0.99,
    top_fractions: Sequence[float] = DEFAULT_TOP_FRACTIONS,
) -> FinalOOSReplayResult:
    """Score the one untouched November 2024 candidate population exactly once.

    There is intentionally no fitter, feature selector, HPO object, policy
    optimiser, refit toggle, or evaluation-driven admission path in this API.
    A caller must build/freeze those artifacts before invoking this function.
    """
    if not 0.0 < float(min_feature_coverage) <= 1.0:
        raise FinalOOSReplayError("min_feature_coverage must be in (0, 1]")
    fractions = tuple(float(value) for value in top_fractions)
    if not fractions or len(set(fractions)) != len(fractions) or any(not 0.0 < value <= 1.0 for value in fractions):
        raise FinalOOSReplayError("top fractions must be unique values in (0, 1]")
    output = Path(output_dir)
    registry = Path(consumption_registry)
    if output.exists():
        raise FinalOOSReplayError("final-OOS output directory already exists; replay outputs are immutable")
    work = _input_panel(panel)
    _acquire_once(registry, contract=contract, output=output)
    loader = _default_model_loader if model_loader is None else model_loader
    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=staging_parent))
    published = False
    try:
        scored, audits = [], []
        for side in SIDES:
            values, audit = _score_side(
                work, contract=contract.scoring_by_side[side], model_loader=loader,
                min_feature_coverage=float(min_feature_coverage),
            )
            scored.append(values); audits.append(audit)
        prediction = pd.concat(scored, ignore_index=True)
        if prediction["candidate_key"].duplicated().any() or len(prediction) != len(work):
            raise FinalOOSReplayError("frozen final scorer did not preserve the complete cross-side candidate population")
        # State artifacts need not be runtime feature columns themselves, but
        # their fit cutoff/hash is carried on every row for direct audit.
        prediction["causal_state_artifact_sha256_json"] = json.dumps([item.sha256 for item in contract.causal_state_artifacts])
        prediction["causal_state_fit_end_utc_json"] = json.dumps([item.fit_end_utc for item in contract.causal_state_artifacts])
        metrics = _metrics(prediction, top_fractions=fractions)
        selected = _selected_books(prediction, top_fractions=fractions)
        coverage = pd.concat(audits, ignore_index=True)
        provenance = pd.DataFrame([
            {
                "contract_sha256": contract.sha256, "selected_arm": contract.selected_arm,
                "successor": contract.successor, "final_oos_start_utc": FINAL_OOS_START,
                "final_oos_end_utc": FINAL_OOS_END, "development_cutoff_utc": DEVELOPMENT_CUTOFF,
                "global_ranking": "one_pooled_cross_side_common_bps_book_after_frozen_base_plus_frozen_meta",
                "entry_convention": "candidate_bar_close_decision_to_next_hourly_open_entry",
                "label_contract": "H12_from_next_open_resolves_13h_after_decision",
                "cost_contract": "fixed_100_bps_charged_exactly_once_in_realized_gross_minus_net",
                "selection_hpo_refit": "forbidden_in_final_oos_runner",
                "causal_state": "all state artifacts hash-bound and fitted strictly before 2024-11-01; each row state available by decision",
            }
        ])
        prediction.to_parquet(staging / "final_oos_scored_predictions.parquet", index=False, compression="zstd")
        metrics.to_parquet(staging / "final_oos_global_topk_metrics.parquet", index=False, compression="zstd")
        selected.to_parquet(staging / "final_oos_selected_candidates.parquet", index=False, compression="zstd")
        coverage.to_parquet(staging / "final_oos_feature_coverage.parquet", index=False, compression="zstd")
        provenance.to_parquet(staging / "final_oos_provenance.parquet", index=False, compression="zstd")
        _write_json(staging / "frozen_final_oos_contract.json", contract.to_dict())
        manifest = {
            "schema": SCHEMA, "status": "COMPLETE_ONE_TIME_UNTOUCHED_FINAL_OOS_REPLAY",
            "contract_sha256": contract.sha256, "selected_arm": contract.selected_arm, "successor": contract.successor,
            "final_november_oos_consumed": True, "final_oos_window": {"start_utc": FINAL_OOS_START, "end_utc": FINAL_OOS_END},
            "development_selection": "hash-bound feature-group, taxonomy, and S decision supplied before reading final panel",
            "frozen_model_cutoff": "every base/meta/value-map/causal-state artifact fit strictly before 2024-11-01",
            "causal_input_gate": "feature_available_ts <= decision_ts and causal_state_available_ts <= decision_ts for every candidate",
            "entry": "candidate bar close -> next hourly open", "label": "H12 path -> decision +13h label availability",
            "cost": TP6_COST_BPS, "common_bps_mapping": "frozen base class-value map + frozen Huber residual meta model",
            "global_ranking": "one pooled cross-side book after common-bps mapping; no side/timestamp re-ranking",
            "top_fractions": list(fractions), "min_feature_coverage": float(min_feature_coverage),
            "no_final_oos_selection_hpo_or_refit_tuning": True,
            "files": [
                "final_oos_scored_predictions.parquet", "final_oos_global_topk_metrics.parquet",
                "final_oos_selected_candidates.parquet", "final_oos_feature_coverage.parquet",
                "final_oos_provenance.parquet", "frozen_final_oos_contract.json",
            ],
        }
        manifest["checksums"] = _checksum_tree(staging)
        _write_json(staging / "run_manifest.json", manifest)
        _write_json(staging / "checksums.json", manifest["checksums"])
        os.replace(staging, output)
        published = True
        _complete_once(registry, contract=contract, output=output)
        return FinalOOSReplayResult(output, prediction, metrics, manifest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        # An unsuccessful run did not publish results and therefore did not
        # consume the holdout.  Remove only our own lock; an existing registry
        # was rejected before this point and is never touched.
        try:
            if not published and registry.exists():
                payload = json.loads(registry.read_text(encoding="utf-8"))
                if payload.get("status") == "RUNNING_FINAL_OOS_REPLAY" and payload.get("contract_sha256") == contract.sha256:
                    registry.unlink()
        except Exception:
            # A corrupted/ambiguous registry is deliberately left in place so
            # future calls fail closed rather than replay the final holdout.
            pass
        raise


__all__ = [
    "SCHEMA", "FINAL_OOS_START", "FINAL_OOS_END", "DEVELOPMENT_CUTOFF", "FinalOOSReplayError",
    "FrozenArtifact", "FrozenSideScoringContract", "FinalOOSReplayContract", "FinalOOSReplayResult",
    "run_leaf_reasoning_final_oos_replay",
]
