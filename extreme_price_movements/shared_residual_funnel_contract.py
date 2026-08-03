"""Frozen sequential handoff for the shared exact-net residual funnel.

Stage III must consume a named predecessor winner, rather than silently
re-selecting a target, geometry, feature set, or regime expert.  This compact
contract is intentionally independent of calibration/materialisation code so
it can be written by a runner manifest and verified before a later stage is
allowed to start.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "shared_residual_funnel_handoff_v1"
REQUIRED_FIELDS = frozenset(
    {
        "selected_arm",
        "target",
        "reconstruction",
        "feature_list",
        "model_class",
        "geometry",
        "cost",
        "entry",
        "label_availability",
        "ranking",
        "calibration",
    }
)
SHARED_MODEL_CLASS = "shared_exact_net_residual"
EXACT_RESIDUAL_TARGET = (
    "exact_net_bps_minus_frozen_causal_base_expected_net_bps_"
    "minus_prior_resolved_soft_regime_residual_bps"
)
COMMON_BPS_RECONSTRUCTION = (
    "frozen_base_expected_net_bps_plus_prior_resolved_soft_regime_residual_bps_"
    "plus_predicted_candidate_residual_bps"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def artifact_sha256(path: str | Path) -> str:
    """Return a content hash for a frozen predecessor artifact."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"predecessor artifact is missing or not a file: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_shared_common_bps(
    base_expected_net_bps: Sequence[float],
    soft_regime_prior_residual_bps: Sequence[float],
    predicted_candidate_residual_bps: Sequence[float],
) -> np.ndarray:
    """Reconstruct the one shared expert's globally comparable score."""
    base = np.asarray(base_expected_net_bps, dtype=float)
    prior = np.asarray(soft_regime_prior_residual_bps, dtype=float)
    candidate = np.asarray(predicted_candidate_residual_bps, dtype=float)
    if base.ndim != 1 or prior.shape != base.shape or candidate.shape != base.shape:
        raise ValueError("shared score components must be aligned one-dimensional arrays")
    if not (np.isfinite(base).all() and np.isfinite(prior).all() and np.isfinite(candidate).all()):
        raise ValueError("shared score components must be finite common-bps values")
    return base + prior + candidate


def _normalise_predecessors(paths: Sequence[str | Path]) -> tuple[dict[str, str], ...]:
    if not paths:
        raise ValueError("shared residual handoff requires at least one predecessor artifact")
    records = [
        {"path": str(Path(path).resolve()), "sha256": artifact_sha256(path)}
        for path in paths
    ]
    return tuple(sorted(records, key=lambda record: record["path"]))


def _reject_local_routing(value: Any, *, field: str) -> None:
    text = _canonical_json(value).lower()
    if field == "routing" and text.strip('"') == "shared_no_hard_routing":
        return
    banned = ("local", "per_regime", "per-regime", "hard_routing", "hard-routing")
    if any(token in text for token in banned):
        raise ValueError(
            f"{field} violates the shared-residual contract: local/per-regime experts "
            "and hard routing are forbidden"
        )


@dataclass(frozen=True)
class SharedResidualFunnelContract:
    """Immutable Stage-III handoff, including predecessor content hashes."""

    schema: str
    predecessors: tuple[dict[str, str], ...]
    selected_arm: str
    target: str
    reconstruction: str
    feature_list: tuple[str, ...]
    feature_list_sha256: str
    model_class: str
    geometry: Mapping[str, Any]
    cost: Mapping[str, Any]
    entry: Mapping[str, Any]
    label_availability: Mapping[str, Any]
    ranking: Mapping[str, Any]
    calibration: Mapping[str, Any]
    routing: str = "shared_no_hard_routing"

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported handoff schema: {self.schema!r}")
        if not self.predecessors:
            raise ValueError("handoff has no frozen predecessor artifacts")
        if not str(self.selected_arm).strip():
            raise ValueError("handoff selected_arm is required")
        if self.target != EXACT_RESIDUAL_TARGET:
            raise ValueError("handoff target must be the approved regime-centered exact-net residual")
        if self.reconstruction != COMMON_BPS_RECONSTRUCTION:
            raise ValueError("handoff reconstruction must be the approved common-bps score")
        if self.model_class != SHARED_MODEL_CLASS:
            raise ValueError("handoff must use the one shared exact-net residual expert")
        if not self.feature_list:
            raise ValueError("handoff feature_list is empty")
        expected_geometry = {"tp_atr": 6.0, "sl_atr": 4.0, "horizon_hours": 12.0}
        for key, expected in expected_geometry.items():
            if float(self.geometry.get(key, float("nan"))) != expected:
                raise ValueError(f"handoff geometry must freeze {key}={expected}")
        if float(self.cost.get("total_cost_bps", float("nan"))) != 100.0:
            raise ValueError("handoff must freeze total_cost_bps=100")
        if int(self.cost.get("application_count", -1)) != 1:
            raise ValueError("handoff must apply the declared cost exactly once")
        if float(self.entry.get("signal_to_entry_hours", float("nan"))) != 1.0:
            raise ValueError("handoff must enter one hour after the signal close")
        if float(self.label_availability.get("signal_to_available_hours", float("nan"))) != 13.0:
            raise ValueError("handoff must freeze signal-to-label availability at 13 hours")
        if self.label_availability.get("strict_comparison") != "label_available_ts < fit_cutoff":
            raise ValueError("handoff must use strict prior-resolved label availability")
        if self.ranking.get("selection") != "pooled_global_after_common_bps_mapping":
            raise ValueError("handoff must rank pooled-global after common-bps mapping")
        expected_feature_hash = sha256(_canonical_json(list(self.feature_list)).encode()).hexdigest()
        if self.feature_list_sha256 != expected_feature_hash:
            raise ValueError("handoff feature_list hash does not match its declared feature list")
        _reject_local_routing(self.routing, field="routing")
        # Calibration may legitimately be side-local/shrunk; only model
        # experts and routing are forbidden from becoming local or hard-routed.
        for name, value in (("selected_arm", self.selected_arm), ("model_class", self.model_class)):
            _reject_local_routing(value, field=name)

    def verify_predecessors(self) -> None:
        self.validate()
        for record in self.predecessors:
            actual = artifact_sha256(record["path"])
            if actual != record["sha256"]:
                raise ValueError(
                    "frozen predecessor artifact hash mismatch: "
                    f"{record['path']} expected={record['sha256']} actual={actual}"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["predecessors"] = list(payload["predecessors"])
        payload["feature_list"] = list(payload["feature_list"])
        return payload


def freeze_shared_residual_handoff(
    *, predecessor_artifacts: Sequence[str | Path], **manifest: Any
) -> SharedResidualFunnelContract:
    """Freeze the approved shared expert into a serialisable handoff.

    The named fields are deliberately required: callers cannot construct a
    generic manifest that later re-decides economics or routing.
    """
    missing = REQUIRED_FIELDS.difference(manifest)
    if missing:
        raise ValueError(f"handoff manifest is missing required fields: {sorted(missing)}")
    features = tuple(str(value) for value in manifest["feature_list"] if str(value).strip())
    contract = SharedResidualFunnelContract(
        schema=SCHEMA_VERSION,
        predecessors=_normalise_predecessors(predecessor_artifacts),
        selected_arm=str(manifest["selected_arm"]),
        target=str(manifest["target"]),
        reconstruction=str(manifest["reconstruction"]),
        feature_list=features,
        feature_list_sha256=sha256(_canonical_json(list(features)).encode()).hexdigest(),
        model_class=str(manifest["model_class"]),
        geometry=dict(manifest["geometry"]),
        cost=dict(manifest["cost"]),
        entry=dict(manifest["entry"]),
        label_availability=dict(manifest["label_availability"]),
        ranking=dict(manifest["ranking"]),
        calibration=dict(manifest["calibration"]),
        routing=str(manifest.get("routing", "shared_no_hard_routing")),
    )
    contract.validate()
    return contract


def write_shared_residual_handoff(
    contract: SharedResidualFunnelContract, path: str | Path) -> Path:
    contract.verify_predecessors()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_canonical_json(contract.to_dict()) + "\n", encoding="utf-8")
    return destination


def load_shared_residual_handoff(path: str | Path) -> SharedResidualFunnelContract:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"shared residual handoff is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    contract = SharedResidualFunnelContract(
        schema=str(payload.get("schema", "")),
        predecessors=tuple(dict(value) for value in payload.get("predecessors", [])),
        selected_arm=str(payload.get("selected_arm", "")),
        target=str(payload.get("target", "")),
        reconstruction=str(payload.get("reconstruction", "")),
        feature_list=tuple(str(value) for value in payload.get("feature_list", [])),
        feature_list_sha256=str(payload.get("feature_list_sha256", "")),
        model_class=str(payload.get("model_class", "")),
        geometry=dict(payload.get("geometry", {})),
        cost=dict(payload.get("cost", {})),
        entry=dict(payload.get("entry", {})),
        label_availability=dict(payload.get("label_availability", {})),
        ranking=dict(payload.get("ranking", {})),
        calibration=dict(payload.get("calibration", {})),
        routing=str(payload.get("routing", "")),
    )
    contract.verify_predecessors()
    return contract
