"""Universal point-in-time and target-proximity gate for model features.

The gate is intentionally small and dependency-agnostic.  Callers provide a
feature registry whose ``raw_dependencies`` form a directed acyclic graph.  A
feature is admissible only when the complete dependency closure is causal,
live-reproducible, and free of target/future-path/cost contamination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


SCHEMA = "feature_provenance_gate_v1"

# These are namespaces, not a blacklist of ordinary words.  In particular,
# causal features such as ``trend_slope`` remain admissible; ``future_slope``
# and realised path labels do not.
FORBIDDEN_NAME_PATTERNS = (
    re.compile(r"(^|_)(?:target|label|outcome)(?:_|$)", re.I),
    re.compile(r"(^|_)(?:future|suffix|post_entry|postentry)(?:_|$)", re.I),
    re.compile(r"(^|_)(?:realized|realised|hindsight)(?:_|$)", re.I),
    re.compile(r"(^|_)(?:known_row_cost|execution_cost_return)(?:_|$)", re.I),
    re.compile(r"(^|_)(?:estimated_net_if_exit|net_exit_now|net_continue)(?:_|$)", re.I),
    re.compile(r"(^|_)(?:gross_ev|net_ev|execution_gross|execution_net)(?:_|$)", re.I),
    re.compile(r"(^|_)(?:peak_mfe|meaningful_mfe|mae_before|future_slope|mfe_persistence)(?:_|$)", re.I),
    re.compile(r"(^|_)(?:favorable_first|adverse_first|timeout|continue_better)(?:_|$)", re.I),
)


class FeatureProvenanceError(ValueError):
    """Raised when a feature registry or frame violates the causal contract."""


@dataclass(frozen=True)
class FeatureLineageRecord:
    feature_name: str
    raw_dependencies: tuple[str, ...] = field(default_factory=tuple)
    feature_available_ts: str | None = None
    decision_layer: str = "entry"
    target_algebra_overlap: bool = False
    future_path_dependency: bool = False
    cost_dependency: bool = False
    oof_required: bool = False
    oof_verified: bool = False
    live_reproducible: bool = False
    point_in_time_safe: bool = False
    admission_decision: str = "UNASSESSED"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureLineageRecord":
        required = {"feature_name"}
        missing = sorted(required.difference(value))
        if missing:
            raise FeatureProvenanceError(f"feature lineage missing required fields: {missing}")
        dependencies = value.get("raw_dependencies", value.get("dependencies", ()))
        if dependencies is None:
            dependencies = ()
        return cls(
            feature_name=str(value["feature_name"]),
            raw_dependencies=tuple(map(str, dependencies)),
            feature_available_ts=None if value.get("feature_available_ts") is None else str(value["feature_available_ts"]),
            decision_layer=str(value.get("decision_layer", "entry")),
            target_algebra_overlap=bool(value.get("target_algebra_overlap", False)),
            future_path_dependency=bool(value.get("future_path_dependency", False)),
            cost_dependency=bool(value.get("cost_dependency", False)),
            oof_required=bool(value.get("oof_required", False)),
            oof_verified=bool(value.get("oof_verified", False)),
            live_reproducible=bool(value.get("live_reproducible", False)),
            point_in_time_safe=bool(value.get("point_in_time_safe", False)),
            admission_decision=str(value.get("admission_decision", "UNASSESSED")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "feature_name": self.feature_name,
            "raw_dependencies": list(self.raw_dependencies),
            "feature_available_ts": self.feature_available_ts,
            "decision_layer": self.decision_layer,
            "target_algebra_overlap": self.target_algebra_overlap,
            "future_path_dependency": self.future_path_dependency,
            "cost_dependency": self.cost_dependency,
            "oof_required": self.oof_required,
            "oof_verified": self.oof_verified,
            "live_reproducible": self.live_reproducible,
            "point_in_time_safe": self.point_in_time_safe,
            "admission_decision": self.admission_decision,
        }


def forbidden_name_reason(name: str) -> str | None:
    for pattern in FORBIDDEN_NAME_PATTERNS:
        if pattern.search(str(name)):
            return f"forbidden target/future/cost namespace: {name}"
    return None


def validate_feature_columns(feature_columns: Iterable[str]) -> tuple[str, ...]:
    """Cheap name-level gate used before a feature matrix is constructed."""

    columns = tuple(map(str, feature_columns))
    if len(columns) != len(set(columns)):
        raise FeatureProvenanceError("duplicate feature names are not admissible")
    forbidden = [reason for name in columns if (reason := forbidden_name_reason(name))]
    if forbidden:
        raise FeatureProvenanceError("; ".join(forbidden))
    return columns


def _records(value: Mapping[str, Any] | Iterable[Mapping[str, Any] | FeatureLineageRecord]) -> dict[str, FeatureLineageRecord]:
    if isinstance(value, Mapping):
        iterable = value.values()
    else:
        iterable = value
    output: dict[str, FeatureLineageRecord] = {}
    for item in iterable:
        record = item if isinstance(item, FeatureLineageRecord) else FeatureLineageRecord.from_mapping(item)
        if record.feature_name in output:
            raise FeatureProvenanceError(f"duplicate lineage record: {record.feature_name}")
        output[record.feature_name] = record
    return output


def validate_feature_lineage(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any] | FeatureLineageRecord],
    feature_names: Sequence[str] | None = None,
    *,
    require_live: bool = True,
    require_oof: bool = True,
) -> dict[str, Any]:
    """Validate records and their transitive raw-dependency closure."""

    registry = _records(records)
    selected = tuple(registry) if feature_names is None else validate_feature_columns(feature_names)
    errors: dict[str, list[str]] = {}
    visiting: list[str] = []
    memo: dict[str, set[str]] = {}

    def walk(name: str) -> set[str]:
        if name in memo:
            return memo[name]
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise FeatureProvenanceError(f"feature lineage cycle: {cycle}")
        record = registry.get(name)
        if record is None:
            errors.setdefault(name, []).append("missing lineage record")
            return set()
        visiting.append(name)
        reasons: set[str] = set()
        direct_reason = forbidden_name_reason(name)
        if direct_reason:
            reasons.add(direct_reason)
        for dependency in record.raw_dependencies:
            reasons.update(walk(dependency))
        if record.target_algebra_overlap:
            reasons.add("target algebra overlap")
        if record.future_path_dependency:
            reasons.add("future path dependency")
        if record.cost_dependency:
            reasons.add("cost dependency")
        if not record.point_in_time_safe:
            reasons.add("not point-in-time safe")
        if require_live and not record.live_reproducible:
            reasons.add("not live reproducible")
        if require_oof and record.oof_required and not record.oof_verified:
            reasons.add("required OOF lineage is unverified")
        visiting.pop()
        memo[name] = reasons
        if reasons:
            errors.setdefault(name, []).extend(sorted(reasons))
        return reasons

    for name in selected:
        walk(name)
    return {
        "schema": SCHEMA,
        "selected_features": list(selected),
        "feature_count": len(selected),
        "passed": not errors,
        "errors": {name: sorted(set(values)) for name, values in errors.items()},
        "admission_decision": "ADMITTED_CAUSAL" if not errors else "REJECTED_PROVENANCE",
    }


def assert_feature_lineage(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any] | FeatureLineageRecord],
    feature_names: Sequence[str] | None = None,
    *,
    require_live: bool = True,
    require_oof: bool = True,
) -> dict[str, Any]:
    report = validate_feature_lineage(records, feature_names, require_live=require_live, require_oof=require_oof)
    if not report["passed"]:
        raise FeatureProvenanceError(json_report(report))
    return report


def audit_feature_frame(
    frame: pd.DataFrame,
    records: Mapping[str, Any] | Iterable[Mapping[str, Any] | FeatureLineageRecord],
    feature_names: Sequence[str],
    *,
    decision_column: str,
    available_column: str,
    require_live: bool = True,
    require_oof: bool = True,
) -> dict[str, Any]:
    """Add row-level timestamp checks to the registry-level report."""

    report = validate_feature_lineage(records, feature_names, require_live=require_live, require_oof=require_oof)
    missing = sorted({decision_column, available_column}.difference(frame.columns))
    if missing:
        report["passed"] = False
        report.setdefault("errors", {})["__frame__"] = [f"missing timestamp columns: {missing}"]
        report["admission_decision"] = "REJECTED_PROVENANCE"
        return report
    decision = pd.to_datetime(frame[decision_column], utc=True, errors="coerce")
    available = pd.to_datetime(frame[available_column], utc=True, errors="coerce")
    bad = available.isna() | decision.isna() | available.gt(decision)
    report["timestamp_rows_checked"] = int(len(frame))
    report["timestamp_violations"] = int(bad.sum())
    if bool(bad.any()):
        report["passed"] = False
        report.setdefault("errors", {})["__frame__"] = ["feature availability is after decision timestamp or missing"]
        report["admission_decision"] = "REJECTED_PROVENANCE"
    return report


def json_report(report: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(report), sort_keys=True, default=str)
