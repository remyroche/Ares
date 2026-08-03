#!/usr/bin/env python3
"""Build and enforce an append-only opportunity-incident support gate.

The gate separates retrospective model-OOF support, resolved-forward research
support, and genuine incumbent-portfolio prospective support.  It refuses
source hash drift, cross-lineage pooling, frozen-packet rewrites, and
retroactive inserts behind a prior lineage watermark.  It never trains a
classifier or changes admission, exposure, timing, or portfolio policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "configs/prospective_opportunity_incident_gate_20260729_v1.json"
)
CURRENT_LINEAGE = "current_2026_execution_ev"
COMMON30_LINEAGE = "historical_2025_common30_12h_cost100bps_direct_ev_oof"
PACKET_IDENTITY = ("lineage", "opportunity_incident_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _packet_hash(row: pd.Series, original_columns: Iterable[str]) -> str:
    payload = {column: row[column] for column in sorted(original_columns)}
    return _canonical_hash(payload)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("status") != "FAIL_CLOSED_NO_ROUTER_AUTHORIZED":
        raise ValueError("incident gate config is not fail-closed")
    return config


def verify_sources(config: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, specification in config["sources"].items():
        path = (ROOT / specification["path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != specification["sha256"]:
            raise ValueError(f"frozen incident-gate source hash changed: {key}")
        result[key] = path
    transition_registry = json.loads(
        result["frozen_transition_registry"].read_text(encoding="utf-8")
    )
    if (
        transition_registry.get("status")
        != "FROZEN_CONTEXT_ONLY_NO_DIRECT_POLICY_CONTROL"
    ):
        raise ValueError("transition context registry is not frozen")
    for key, specification in transition_registry["sources"].items():
        path = (ROOT / specification["path"]).resolve()
        if not path.exists() or sha256(path) != specification["sha256"]:
            raise ValueError(f"frozen transition source changed: {key}")
    return result


def _require(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks required columns: {missing}")


def classify_current_packet_role(
    packet: pd.Series,
    selected: pd.DataFrame,
    role_contract: dict[str, Any],
) -> tuple[str, int, str]:
    start = pd.Timestamp(packet["incident_start_utc"])
    end = pd.Timestamp(packet["incident_end_utc"])
    local = selected.loc[
        selected["__ts__"].ge(start) & selected["__ts__"].lt(end)
    ]
    roles = sorted(local["failure_first_history_role"].dropna().astype(str).unique())
    joined = "|".join(roles)
    if not len(local):
        return "NO_SELECTED_ROWS", 0, joined
    retrospective = set(role_contract["retrospective_roles"])
    forward = set(role_contract["forward_research_roles"])
    observed = set(roles)
    if observed and observed.issubset(retrospective):
        return "RETROSPECTIVE_OUTER_OOF", int(len(local)), joined
    if observed and observed.issubset(forward):
        return "RESOLVED_FORWARD_RESEARCH", int(len(local)), joined
    return "MIXED_OR_UNKNOWN_PROVENANCE", int(len(local)), joined


def _validate_independence(
    ledger: pd.DataFrame, merge_gap_hours: int
) -> None:
    seen_events: set[tuple[str, str]] = set()
    for lineage, local in ledger.groupby("lineage", sort=True):
        local = local.sort_values("incident_start_utc", kind="stable")
        previous_end: pd.Timestamp | None = None
        for row in local.itertuples(index=False):
            start = pd.Timestamp(row.incident_start_utc)
            end = pd.Timestamp(row.incident_end_utc)
            if previous_end is not None and start <= previous_end + pd.Timedelta(
                hours=int(merge_gap_hours)
            ):
                raise ValueError(
                    f"{lineage} contains unconsolidated overlapping incidents"
                )
            previous_end = end
            for event_id in str(row.source_event_ids).split("|"):
                key = (str(lineage), event_id)
                if key in seen_events:
                    raise ValueError(f"source event appears in multiple packets: {key}")
                seen_events.add(key)


def build_ledger(
    snapshot_packets: pd.DataFrame,
    common30_packets: pd.DataFrame,
    current_selected: pd.DataFrame,
    config: dict[str, Any],
    *,
    asof_utc: pd.Timestamp,
) -> pd.DataFrame:
    required = {
        *PACKET_IDENTITY,
        "source_event_ids",
        "incident_start_utc",
        "incident_end_utc",
        "reference_status",
        "packet_available_utc",
        "packet_frozen",
    }
    _require(snapshot_packets, required, "packet snapshot")
    _require(common30_packets, required, "common30 packets")
    _require(
        current_selected,
        ("__ts__", "failure_first_history_role"),
        "current selected candidates",
    )
    snapshot = snapshot_packets.copy()
    common = common30_packets.copy()
    if not common["lineage"].eq(COMMON30_LINEAGE).all():
        raise ValueError("common30 packets violate the configured lineage")
    overlap = snapshot.loc[:, list(PACKET_IDENTITY)].merge(
        common.loc[:, list(PACKET_IDENTITY)],
        on=list(PACKET_IDENTITY),
        how="inner",
    )
    if len(overlap):
        raise ValueError("packet sources overlap identities")
    original_columns = sorted(set(snapshot.columns) | set(common.columns))
    ledger = pd.concat(
        [snapshot, common], ignore_index=True, sort=False
    )
    for column in (
        "incident_start_utc",
        "incident_end_utc",
        "packet_available_utc",
    ):
        ledger[column] = pd.to_datetime(ledger[column], utc=True, errors="raise")
    current_selected = current_selected.copy()
    current_selected["__ts__"] = pd.to_datetime(
        current_selected["__ts__"], utc=True, errors="raise"
    )
    if ledger.duplicated(list(PACKET_IDENTITY)).any():
        raise ValueError("packet identities must be unique across the registry")
    ledger["evidence_role"] = "RETROSPECTIVE_SEPARATE_LINEAGE"
    ledger["role_selected_rows"] = 0
    ledger["source_history_roles"] = ""
    current_mask = ledger["lineage"].eq(CURRENT_LINEAGE)
    for index, packet in ledger.loc[current_mask].iterrows():
        role, rows, source_roles = classify_current_packet_role(
            packet,
            current_selected,
            config["current_role_contract"],
        )
        ledger.at[index, "evidence_role"] = role
        ledger.at[index, "role_selected_rows"] = rows
        ledger.at[index, "source_history_roles"] = source_roles
    ledger["available_by_asof"] = ledger["packet_available_utc"].le(asof_utc)
    ledger["fully_resolution_frozen"] = (
        ledger["packet_frozen"].fillna(False).astype(bool)
        & ledger["available_by_asof"]
    )
    lineage_specs = config["lineages"]
    ledger["lineage_contract_id"] = ledger["lineage"].map(
        {
            key: value["contract_id"]
            for key, value in lineage_specs.items()
            if value.get("contract_id")
        }
    )
    if ledger["lineage_contract_id"].isna().any():
        unknown = sorted(
            ledger.loc[ledger["lineage_contract_id"].isna(), "lineage"].unique()
        )
        raise ValueError(f"unregistered packet lineages: {unknown}")
    detector_lineages = {
        key
        for key, value in lineage_specs.items()
        if value.get("detector_support_eligible", False)
    }
    portfolio_lineages = {
        key
        for key, value in lineage_specs.items()
        if value.get("portfolio_promotion_eligible", False)
    }
    ledger["detector_support_eligible"] = (
        ledger["lineage"].isin(detector_lineages)
        & ledger["fully_resolution_frozen"]
        & ledger["evidence_role"].isin(
            {"RETROSPECTIVE_OUTER_OOF", "RESOLVED_FORWARD_RESEARCH"}
        )
    )
    ledger["taxonomy_support_eligible"] = (
        ledger["detector_support_eligible"]
        & ledger["reference_status"].eq("AVAILABLE")
    )
    ledger["prospective_forward_support"] = (
        ledger["detector_support_eligible"]
        & ledger["evidence_role"].eq("RESOLVED_FORWARD_RESEARCH")
    )
    ledger["incumbent_portfolio_support_eligible"] = (
        ledger["lineage"].isin(portfolio_lineages)
        & ledger["fully_resolution_frozen"]
    )
    ledger["packet_content_sha256"] = [
        _packet_hash(row, original_columns)
        for _, row in ledger.iterrows()
    ]
    _validate_independence(
        ledger, int(config["independence_contract"]["incident_merge_gap_hours"])
    )
    return ledger.sort_values(
        ["lineage", "incident_start_utc"], kind="stable"
    ).reset_index(drop=True)


def assert_append_only(
    prior: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    required = {*PACKET_IDENTITY, "packet_content_sha256", "incident_start_utc"}
    _require(prior, required, "prior incident ledger")
    _require(current, required, "current incident ledger")
    prior_keyed = prior.set_index(list(PACKET_IDENTITY))
    current_keyed = current.set_index(list(PACKET_IDENTITY))
    missing = prior_keyed.index.difference(current_keyed.index)
    if len(missing):
        raise ValueError("append-only update removed frozen packets")
    observed = current_keyed.loc[prior_keyed.index, "packet_content_sha256"]
    changed = observed.ne(prior_keyed["packet_content_sha256"])
    if changed.any():
        raise ValueError("append-only update rewrote frozen packet content")
    for lineage, old in prior.groupby("lineage", sort=True):
        watermark = pd.to_datetime(old["incident_start_utc"], utc=True).max()
        old_ids = set(old["opportunity_incident_id"].astype(str))
        added = current.loc[
            current["lineage"].eq(lineage)
            & ~current["opportunity_incident_id"].astype(str).isin(old_ids)
        ]
        if len(added) and pd.to_datetime(
            added["incident_start_utc"], utc=True
        ).le(watermark).any():
            raise ValueError("append-only update inserted behind lineage watermark")


def gate_report(
    ledger: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    minimum = int(config["gate"]["minimum_independent_incidents"])
    target = int(config["gate"]["target_independent_incidents"])
    detector = int(ledger["detector_support_eligible"].sum())
    taxonomy = int(ledger["taxonomy_support_eligible"].sum())
    forward = int(ledger["prospective_forward_support"].sum())
    portfolio = int(ledger["incumbent_portfolio_support_eligible"].sum())
    return {
        "candidate_current_model_incidents": detector,
        "taxonomy_usable_current_model_incidents": taxonomy,
        "prospective_forward_research_incidents": forward,
        "promotion_grade_incumbent_portfolio_incidents": portfolio,
        "minimum_required": minimum,
        "target_required": target,
        "remaining_to_minimum_current_model": max(0, minimum - detector),
        "remaining_to_minimum_incumbent_portfolio": max(0, minimum - portfolio),
        "supervised_failure_detector_training_authorized": detector >= minimum,
        "opportunity_state_router_authorized": (
            detector >= minimum and taxonomy >= minimum
        ),
        "incumbent_portfolio_promotion_authorized": portfolio >= minimum,
    }


def _summary(ledger: pd.DataFrame) -> pd.DataFrame:
    return (
        ledger.groupby("lineage", observed=True, sort=True)
        .agg(
            packets=("opportunity_incident_id", "size"),
            frozen=("fully_resolution_frozen", "sum"),
            detector_support=("detector_support_eligible", "sum"),
            taxonomy_support=("taxonomy_support_eligible", "sum"),
            prospective_forward_support=("prospective_forward_support", "sum"),
            incumbent_portfolio_support=(
                "incumbent_portfolio_support_eligible",
                "sum",
            ),
            first_anchor=("incident_start_utc", "min"),
            last_anchor=("incident_start_utc", "max"),
        )
        .reset_index()
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    config = load_config(config_path)
    sources = verify_sources(config)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    asof = pd.Timestamp(args.asof_utc)
    if asof.tzinfo is None:
        asof = asof.tz_localize("UTC")
    else:
        asof = asof.tz_convert("UTC")
    ledger = build_ledger(
        pd.read_parquet(sources["packet_snapshot_packets"]),
        pd.read_parquet(sources["common30_packets"]),
        pd.read_parquet(sources["current_selected_candidates"]),
        config,
        asof_utc=asof,
    )
    parent_registry_sha256 = None
    if args.prior_registry is not None:
        prior_registry_path = Path(args.prior_registry)
        prior_registry = json.loads(
            prior_registry_path.read_text(encoding="utf-8")
        )
        prior_ledger_path = Path(prior_registry["outputs"]["ledger"]["path"])
        if sha256(prior_ledger_path) != prior_registry["outputs"]["ledger"]["sha256"]:
            raise ValueError("prior incident ledger hash changed")
        assert_append_only(pd.read_parquet(prior_ledger_path), ledger)
        parent_registry_sha256 = sha256(prior_registry_path)
    report = gate_report(ledger, config)
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = output / "incident_support_ledger.parquet"
    summary_path = output / "lineage_support_summary.csv"
    ledger.to_parquet(ledger_path, index=False, compression="zstd")
    _summary(ledger).to_csv(summary_path, index=False)
    registry = {
        "schema": "prospective_opportunity_incident_registry_v1",
        "status": (
            "GATE_OPEN"
            if report["incumbent_portfolio_promotion_authorized"]
            else "FAIL_CLOSED_SUPPORT_INSUFFICIENT"
        ),
        "asof_utc": asof,
        "parent_registry_sha256": parent_registry_sha256,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256(config_path),
        },
        "gate": report,
        "lineage_watermarks": {
            str(lineage): pd.to_datetime(local["incident_start_utc"], utc=True).max()
            for lineage, local in ledger.groupby("lineage", sort=True)
        },
        "source_hashes_verified": True,
        "cross_lineage_pooling": False,
        "append_only_contract_enforced": True,
        "forbidden_actions_below_gate": config["forbidden_actions_below_gate"],
        "outputs": {
            "ledger": {
                "path": str(ledger_path.resolve()),
                "sha256": sha256(ledger_path),
            },
            "summary": {
                "path": str(summary_path.resolve()),
                "sha256": sha256(summary_path),
            },
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    registry_path = output / "registry.json"
    registry_path.write_text(
        json.dumps(_safe(registry), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "registry.sha256").write_text(
        sha256(registry_path) + "\n", encoding="utf-8"
    )
    if args.require_authorized and not report[
        "incumbent_portfolio_promotion_authorized"
    ]:
        raise RuntimeError(
            "incident gate is fail-closed: incumbent portfolio support is "
            f"{report['promotion_grade_incumbent_portfolio_incidents']}/"
            f"{report['minimum_required']}"
        )
    return registry


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--asof-utc",
        default=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    result.add_argument("--prior-registry", type=Path)
    result.add_argument("--require-authorized", action="store_true")
    return result


def main() -> None:
    print(
        json.dumps(
            _safe(run(parser().parse_args())),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
