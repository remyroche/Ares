"""Fail-closed compact F4 contracts built from causal F3 transform families.

F4 is not another name for the full F3 matrix.  It is a small, nested
representation selected inside each outer transport from *inner* grouped MDA:
the portable raw fields are retained and only the best causal transform
families are added.  A representation may be promoted only when the exact
per-side field lists agree across the declared transports.  That last rule is
deliberately conservative: otherwise an apparent F4 result describes a
different model contract in each era and cannot be handed to later stages as a
single portable manifest.

This module is intentionally model-free.  ``feature_portability_mda`` owns the
chronological scoring, while this module owns deterministic feature-group and
manifest lineage.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "stage_a_f4_compact_contract_v1"
F4_COMPACT_PREFIX = "F4_compact_top"
F4_TRANSFORM_GROUPS: tuple[tuple[str, str], ...] = (
    ("rank_w90", "__causal_rank_w90"),
    ("rank_w180", "__causal_rank_w180"),
    ("robust_z_w90", "__causal_robust_z_w90"),
    ("robust_z_w180", "__causal_robust_z_w180"),
    ("delta_p4", "__causal_delta_p4"),
    ("delta_p24", "__causal_delta_p24"),
)
# Predeclared nested counts.  Keeping at most three of six transform families
# makes every F4 arm genuinely more compact than full F3, while avoiding a
# broad factorial over source fields and transform variants.
DEFAULT_F4_GROUP_COUNTS: tuple[int, ...] = (1, 2, 3)


class F4CompactContractError(ValueError):
    """Raised when a compact F4 field contract cannot be proven."""


def _normalise_contract(contract: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if set(contract) != {"long", "short"}:
        raise F4CompactContractError("F4 compact contracts require exactly long and short side fields")
    result: dict[str, tuple[str, ...]] = {}
    for side in ("long", "short"):
        fields = tuple(dict.fromkeys(map(str, contract[side])))
        if not fields or len(fields) != len(contract[side]) or any(not field for field in fields):
            raise F4CompactContractError(f"F4 {side} contract requires unique non-empty fields")
        result[side] = fields
    return result


def f4_transform_groups(
    f3_contract: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Split a complete F3 contract into portable raw + six transform groups.

    The mapping is checked source by source, not merely by suffix presence.
    Thus a caller cannot construct a compact F4 arm from a convenient subset
    of the materialised robust-z/rank/delta fields.
    """
    contract = _normalise_contract(f3_contract)
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    suffixes = tuple(suffix for _, suffix in F4_TRANSFORM_GROUPS)
    for side, fields in contract.items():
        raw = tuple(field for field in fields if not field.endswith(suffixes))
        if not raw:
            raise F4CompactContractError(f"F3 {side} contract has no portable raw source fields")
        if len(raw) != len(set(raw)):
            raise F4CompactContractError(f"F3 {side} raw sources must be unique")
        groups: dict[str, tuple[str, ...]] = {"portable_raw": raw}
        for name, suffix in F4_TRANSFORM_GROUPS:
            actual = tuple(field for field in fields if field.endswith(suffix))
            expected = tuple(f"{source}{suffix}" for source in raw)
            if actual != expected:
                missing = sorted(set(expected).difference(actual))
                unexpected = sorted(set(actual).difference(expected))
                raise F4CompactContractError(
                    f"F3 {side}/{name} must contain exactly one causal transform per raw source "
                    f"(missing={missing[:4]}, unexpected={unexpected[:4]})"
                )
            groups[name] = actual
        expected_all = {field for values in groups.values() for field in values}
        if set(fields) != expected_all:
            unexpected = sorted(set(fields).difference(expected_all))
            raise F4CompactContractError(f"F3 {side} includes fields outside the F4 transform family contract: {unexpected[:8]}")
        output[side] = groups
    return output


def compact_representation_name(group_count: int) -> str:
    if group_count < 1 or group_count >= len(F4_TRANSFORM_GROUPS):
        raise F4CompactContractError("F4 group count must be between one and five to remain compact")
    return f"{F4_COMPACT_PREFIX}{int(group_count):02d}"


def validate_group_counts(group_counts: Sequence[int]) -> tuple[int, ...]:
    counts = tuple(dict.fromkeys(map(int, group_counts)))
    if not counts:
        raise F4CompactContractError("F4 requires at least one predeclared compact group count")
    if tuple(sorted(counts)) != counts:
        raise F4CompactContractError("F4 group counts must be strictly increasing")
    for count in counts:
        compact_representation_name(count)
    return counts


def compact_contracts_for_ranked_groups(
    groups: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    ranked_transform_groups: Sequence[str],
    group_counts: Sequence[int] = DEFAULT_F4_GROUP_COUNTS,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return predeclared nested F4 contracts for one outer transport.

    ``ranked_transform_groups`` must be calculated on predecessor inner-fold
    evidence.  It may contain every transform family exactly once; no outcome
    values are accepted by this pure contract builder.
    """
    counts = validate_group_counts(group_counts)
    expected_groups = {name for name, _ in F4_TRANSFORM_GROUPS}
    ranking = tuple(map(str, ranked_transform_groups))
    if len(ranking) != len(expected_groups) or set(ranking) != expected_groups:
        raise F4CompactContractError("F4 ranking must contain each causal transform family exactly once")
    if set(groups) != {"long", "short"}:
        raise F4CompactContractError("F4 transform groups require long and short entries")
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for count in counts:
        selected = ranking[:count]
        representation = compact_representation_name(count)
        by_side: dict[str, tuple[str, ...]] = {}
        for side in ("long", "short"):
            side_groups = groups[side]
            if set(side_groups) != {"portable_raw", *expected_groups}:
                raise F4CompactContractError(f"F4 {side} transform group map is incomplete")
            fields = tuple(side_groups["portable_raw"])
            for group in selected:
                fields += tuple(side_groups[group])
            by_side[side] = tuple(dict.fromkeys(fields))
        output[representation] = by_side
    return output


def restrict_f4_transform_groups_to_sources(
    groups: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    source_fields_by_side: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Restrict F4 to the exact cross-transport coverage-safe sources.

    The source list is intentionally supplied per side and every selected
    source carries its raw field plus all six causal transform fields.  This
    avoids the unsafe alternative of allowing an incomplete full-F3 matrix to
    decide which individual transform columns happen to be non-null in one
    transport.
    """
    if set(groups) != {"long", "short"} or set(source_fields_by_side) != {"long", "short"}:
        raise F4CompactContractError("F4 source intersection requires exact long/short entries")
    names = ("portable_raw", *(name for name, _ in F4_TRANSFORM_GROUPS))
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for side in ("long", "short"):
        selected = tuple(map(str, source_fields_by_side[side]))
        available = tuple(map(str, groups[side]["portable_raw"]))
        if not selected or len(selected) != len(set(selected)):
            raise F4CompactContractError(f"F4 {side} coverage intersection must be non-empty and unique")
        unavailable = sorted(set(selected).difference(available))
        if unavailable:
            raise F4CompactContractError(f"F4 {side} coverage intersection has unknown source fields: {unavailable[:8]}")
        positions = [available.index(source) for source in selected]
        result[side] = {
            name: tuple(map(str, groups[side][name])) if name == "portable_raw" and selected == available
            else tuple(str(groups[side][name][index]) for index in positions)
            for name in names
        }
        # The comprehension above selects raw sources by the same positional
        # lineage as every transform family.  Recheck it explicitly so a
        # malformed group mapping cannot yield a partial source bundle.
        if result[side]["portable_raw"] != selected:
            raise F4CompactContractError(f"F4 {side} source intersection ordering changed unexpectedly")
    return result


def compact_contract_payload(
    *,
    source_representation: str,
    by_transport: Mapping[str, Mapping[str, Mapping[str, Sequence[str]]]],
    ranking_by_transport: Mapping[str, Sequence[str]],
    group_counts: Sequence[int] = DEFAULT_F4_GROUP_COUNTS,
) -> dict[str, Any]:
    """Build selector input and identify exact cross-transport F4 contracts."""
    counts = validate_group_counts(group_counts)
    transports = tuple(map(str, by_transport))
    if len(transports) < 2 or len(set(transports)) != len(transports):
        raise F4CompactContractError("F4 compact contracts require both unique development transports")
    representations = [compact_representation_name(count) for count in counts]
    by_representation: dict[str, dict[str, Any]] = {}
    for representation in representations:
        per_transport: dict[str, dict[str, list[str]]] = {}
        for transport in transports:
            transport_contracts = by_transport[transport]
            if representation not in transport_contracts:
                raise F4CompactContractError(f"{transport} is missing compact contract {representation}")
            contract = _normalise_contract(transport_contracts[representation])
            per_transport[transport] = {side: list(contract[side]) for side in ("long", "short")}
        first = per_transport[transports[0]]
        stable = all(per_transport[transport] == first for transport in transports[1:])
        by_representation[representation] = {
            "by_transport": per_transport,
            "stable_across_transports": bool(stable),
            "stable_feature_contract": first if stable else None,
        }
    return {
        "schema": SCHEMA,
        "status": "F4_COMPACT_CONTRACTS_MATERIALIZED_DEVELOPMENT_ONLY",
        "source_representation": str(source_representation),
        "transports": list(transports),
        "group_counts": list(counts),
        "transform_groups": [name for name, _ in F4_TRANSFORM_GROUPS],
        "ranked_transform_groups_by_transport": {
            str(transport): list(map(str, ranking_by_transport[transport])) for transport in transports
        },
        "representations": by_representation,
        "final_november_oos_consumed": False,
    }


def selected_compact_feature_manifest(
    *,
    selection: Mapping[str, Any] | None,
    compact_contracts: Mapping[str, Any],
    required_transports: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve the selected F4 representation into one immutable field list.

    A winner whose transport-specific inner MDA rankings chose different field
    lists is explicitly rejected.  Retaining an intersection or union would
    manufacture an unevaluated contract and break the meaning of the F4
    transport result.
    """
    payload = dict(compact_contracts)
    transports = tuple(map(str, payload.get("transports", ())))
    expected = tuple(map(str, required_transports)) or transports
    if len(expected) < 2 or set(transports) != set(expected):
        raise F4CompactContractError("compact contract transports do not match the required development transports")
    base: dict[str, Any] = {
        "schema": "stage_a_f4_selected_portable_feature_manifest_v1",
        "selection_scope": "development_only",
        "final_november_oos_consumed": False,
        "source_representation": payload.get("source_representation"),
        "required_transports": list(expected),
        "selection": dict(selection) if selection is not None else None,
        "meta_control_gate": "PENDING: later L0/F4 compact meta comparison must show no harm before F4 becomes the default meta/base contract",
    }
    if not selection:
        return {**base, "status": "F4_NO_COMPACT_FEATURE_MANIFEST_ADVANCES", "feature_contract": None}
    representation = str(selection.get("representation", ""))
    if not representation.startswith(F4_COMPACT_PREFIX):
        raise F4CompactContractError("only an F4 compact representation can create an F4 portable feature manifest")
    representations = payload.get("representations")
    if not isinstance(representations, Mapping) or representation not in representations:
        raise F4CompactContractError(f"selected F4 representation lacks compact contract lineage: {representation}")
    record = representations[representation]
    if not isinstance(record, Mapping) or not bool(record.get("stable_across_transports", False)):
        return {
            **base,
            "status": "F4_SELECTED_PROCEDURE_HAS_NO_STABLE_COMPACT_CONTRACT",
            "selected_representation": representation,
            "feature_contract": None,
        }
    contract = record.get("stable_feature_contract")
    if not isinstance(contract, Mapping):
        raise F4CompactContractError("stable F4 representation lacks a stable feature contract")
    normalized = _normalise_contract(contract)
    f3_eligible = bool(selection.get("full_f3_control_eligible", False))
    return {
        **base,
        "status": "F4_TRANSPORT_SELECTED_COMPACT_FEATURE_MANIFEST",
        "selected_representation": representation,
        "feature_contract": {side: list(normalized[side]) for side in ("long", "short")},
        "feature_counts": {side: len(normalized[side]) for side in ("long", "short")},
        "coverage_safe_source_intersection": payload.get("coverage_safe_source_intersection"),
        "base_control_verified": True,
        "full_f3_control_verified": True,
        "full_f3_control_eligible": f3_eligible,
        "full_f3_control_status": (
            "ELIGIBLE_NONINFERIORITY_PASSED" if f3_eligible
            else "FULL_F3_DIAGNOSTIC_INELIGIBLE_COVERAGE_NOT_A_PROMOTION_GATE"
        ),
        "base_control_gate": (
            "passed: positive incremental pooled-global top-10 net versus F0, non-negative top-10 net versus full F3, and stable grouped MDA in both development transports"
            if f3_eligible else
            "passed: positive incremental pooled-global top-10 net versus F0 and stable grouped MDA in both development transports; full F3 was diagnostic/ineligible on coverage"
        ),
    }


__all__ = [
    "SCHEMA", "F4_COMPACT_PREFIX", "F4_TRANSFORM_GROUPS", "DEFAULT_F4_GROUP_COUNTS",
    "F4CompactContractError", "compact_representation_name", "compact_contract_payload",
    "compact_contracts_for_ranked_groups", "f4_transform_groups", "selected_compact_feature_manifest",
    "restrict_f4_transform_groups_to_sources", "validate_group_counts",
]
