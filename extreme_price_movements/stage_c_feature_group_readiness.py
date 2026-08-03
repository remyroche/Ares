"""Read-only readiness audit for the Stage-C C0--C8 feature arms.

The audit deliberately *does not* fit a model.  It records whether a proposed
Stage-C arm has an already sealed, causal Stage-C v4 result which can be
reused, or whether its source contract still prevents a safe experiment.  It
also rechecks the two controls that make a paired comparison meaningful:
identical held-out candidate rows and strict chronological OOF training.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ARM_TO_GROUP = {
    "C0": "F0_existing_E15_control",
    "C1": "F1_price_continuation_exhaustion",
    "C2": "F2_volume_liquidity_proxies",
    "C3": "F3_volatility_transition",
    "C4": "F4_oi_dynamics",
    "C5": "F5_funding_crowding",
    "C6": "F6_cross_sectional_confirmation",
    "C7": "F7_causal_regime_transition",
    "C8": "F8_predeclared_composites",
}
GROUP_TO_STAGE1_ARM = {group: arm for arm, group in ARM_TO_GROUP.items()}
BLOCKED_GROUPS = {"F4_oi_dynamics", "F5_funding_crowding", "F7_causal_regime_transition"}
REUSABLE_GROUPS = set(ARM_TO_GROUP.values()).difference(BLOCKED_GROUPS)
HORIZON = pd.Timedelta(hours=12)


@dataclass(frozen=True)
class AuditInputs:
    """Sealed inputs required by the read-only readiness audit."""

    feature_panel: Path
    panel_groups: Path
    panel_lineage: Path
    panel_coverage: Path
    stage1_identity: Path
    stage1_stability: Path
    stage1_manifest: Path
    stage1_results: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_inputs(inputs: AuditInputs) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    required = (
        inputs.feature_panel, inputs.panel_groups, inputs.panel_lineage, inputs.panel_coverage,
        inputs.stage1_identity, inputs.stage1_stability, inputs.stage1_manifest, inputs.stage1_results,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage-C readiness input missing: {missing}")
    panel = pd.read_parquet(inputs.feature_panel)
    groups = json.loads(inputs.panel_groups.read_text(encoding="utf-8"))
    lineage = pd.read_parquet(inputs.panel_lineage)
    coverage = pd.read_parquet(inputs.panel_coverage)
    identity = pd.read_parquet(inputs.stage1_identity)
    stability = pd.read_parquet(inputs.stage1_stability)
    manifest = json.loads(inputs.stage1_manifest.read_text(encoding="utf-8"))
    results = pd.read_parquet(inputs.stage1_results)
    return panel, groups, lineage, coverage, identity, stability, manifest, results


def _strict_oof_check(stability: pd.DataFrame) -> bool:
    development = stability.loc[stability["split"].eq("development_oof")].copy()
    if development.empty:
        return False
    fold_start = pd.to_datetime(development["fold_start_utc"], utc=True, errors="coerce")
    train_max = pd.to_datetime(development["train_decision_ts_max"], utc=True, errors="coerce")
    label_max = pd.to_datetime(development["train_label_available_ts_max"], utc=True, errors="coerce")
    embargo = pd.to_numeric(development["purge_embargo_hours"], errors="coerce")
    return bool(
        (embargo.eq(12)).all()
        and train_max.lt(fold_start - HORIZON).all()
        and label_max.lt(fold_start).all()
        and set(development["fold"].astype(str)).issuperset({"2024-04", "2024-05", "2024-06", "2024-07"})
    )


def _identical_rows_by_arm(identity: pd.DataFrame) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for arm, part in identity.groupby("arm", sort=False):
        result[str(arm)] = bool(part["identical_to_c0"].astype(bool).all())
    return result


def _group_features(groups: dict[str, Any], group: str, lineage: pd.DataFrame) -> list[str]:
    if group == "F0_existing_E15_control":
        return sorted(lineage.loc[lineage.feature_group.eq(group), "feature_name"].astype(str).tolist())
    return list(groups.get(group, []))


def _coverage_summary(coverage: pd.DataFrame, features: list[str], group: str) -> pd.DataFrame:
    """Compress the Stage-0 per-field ledger into a per-group audit view."""
    if not features:
        return pd.DataFrame(columns=["arm", "feature_group", "month", "side", "source_symbol", "rows", "feature_count", "mean_missing_rate", "max_missing_rate"])
    fields = coverage.loc[coverage.feature_name.isin(features)].copy()
    if fields.empty:
        return pd.DataFrame(columns=["arm", "feature_group", "month", "side", "source_symbol", "rows", "feature_count", "mean_missing_rate", "max_missing_rate"])
    group_by = ["month", "side", "source_symbol"]
    result = fields.groupby(group_by, as_index=False).agg(
        rows=("rows", "max"),
        feature_count=("feature_name", "nunique"),
        mean_missing_rate=("missing_rate", "mean"),
        max_missing_rate=("missing_rate", "max"),
    )
    result.insert(0, "feature_group", group)
    result.insert(0, "arm", GROUP_TO_STAGE1_ARM[group])
    return result


def _result_coverage(results: pd.DataFrame, arm: str) -> tuple[bool, int]:
    """Confirm that the reusable v4 evidence includes every fixed final month."""
    final = results.loc[
        results.arm.eq(arm) & results.split.eq("final_oos") & results.scope.eq("month")
    ]
    return set(final.month.dropna().astype(str)).issuperset({"2024-08", "2024-09", "2024-10", "2024-11"}), len(final)


def build_readiness(
    *, inputs: AuditInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build C0--C8 readiness records from sealed, already-materialised data."""
    panel, groups, lineage, coverage, identity, stability, manifest, results = _load_inputs(inputs)
    required_panel = {"candidate_id", "decision_ts", "feature_available_ts"}
    if missing := sorted(required_panel.difference(panel.columns)):
        raise ValueError(f"feature panel misses contract fields: {missing}")
    point_in_time = bool(
        pd.to_datetime(panel.feature_available_ts, utc=True, errors="coerce").le(
            pd.to_datetime(panel.decision_ts, utc=True, errors="coerce")
        ).all()
    )
    strict_oof = _strict_oof_check(stability)
    identical = _identical_rows_by_arm(identity)
    blocked_reasons = groups.get("blocked", {})
    readiness_rows: list[dict[str, Any]] = []
    source_rows: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    for arm, group in ARM_TO_GROUP.items():
        features = _group_features(groups, group, lineage)
        is_blocked = group in BLOCKED_GROUPS
        final_coverage, final_rows = _result_coverage(results, arm) if not is_blocked else (False, 0)
        source_reason = blocked_reasons.get(group.replace("_dynamics", "").replace("_crowding", "").replace("_causal_regime_transition", ""))
        # The Stage-0 group manifest keys use F4/F5/F7 while the readable
        # group names above preserve the C0--C8 specification terminology.
        if group == "F4_oi_dynamics":
            source_reason = blocked_reasons.get("F4", source_reason)
        elif group == "F5_funding_crowding":
            source_reason = blocked_reasons.get("F5", source_reason)
        elif group == "F7_causal_regime_transition":
            source_reason = blocked_reasons.get("F7", source_reason)
        if is_blocked:
            status = "SOURCE_BLOCKED"
            action = "Do not fit; require native source observed_ts/available_ts (F4/F5) or candidate-ID strict OOF/prequential lineage (F7)."
        else:
            status = "REUSABLE_SEALED_V4"
            action = "Reuse the sealed v4 paired result; a new fit is neither started nor authorised by this readiness audit."
            source_reason = "sealed causal Stage-0 OHLCV/frozen-control lineage and strict Stage-1 OOF evidence available"
        group_lineage = lineage.loc[lineage.feature_group.eq(group)].copy()
        if not group_lineage.empty:
            group_lineage.insert(0, "arm", arm)
            source_rows.append(group_lineage)
        group_coverage = _coverage_summary(coverage, features, group)
        if not group_coverage.empty:
            coverage_rows.append(group_coverage)
        readiness_rows.append({
            "arm": arm,
            "feature_group": group,
            "availability_status": status,
            "feature_count": len(features),
            # C0 is deliberately inherited from its separate sealed E15
            # control panel, rather than duplicated into the OHLCV sidecar.
            "feature_columns_present": True if group == "F0_existing_E15_control" else (bool(set(features).issubset(panel.columns)) if features else is_blocked),
            "point_in_time_safe": bool(point_in_time and (not is_blocked)),
            "strict_oof_verified": bool(strict_oof and (not is_blocked)),
            "identical_rows_verified": bool(identical.get(arm, False)) if not is_blocked else False,
            "latest_final_month_coverage": bool(final_coverage),
            "final_month_result_rows": int(final_rows),
            "can_run_now": bool(not is_blocked and strict_oof and identical.get(arm, False) and final_coverage),
            "new_model_fit_started": False,
            "source_reason": source_reason,
            "next_action": action,
        })
    checks = {
        "stage0_feature_available_by_decision": point_in_time,
        "stage1_strict_development_oof_h12_purge": strict_oof,
        "stage1_reusable_arms_identical_to_c0": bool(all(identical.get(arm, False) for arm in ("C0", "C1", "C2", "C3", "C6", "C8"))),
        "stage1_manifest_is_stage_c_v4": manifest.get("schema") == "stage_c_conditional_retention_ablation_v4",
        "no_new_model_fit": True,
        "blocked_f4_f5_f7_remain_excluded": bool(all(row["availability_status"] == "SOURCE_BLOCKED" for row in readiness_rows if row["arm"] in {"C4", "C5", "C7"})),
    }
    report = {
        "schema": "stage_c_feature_group_readiness_v1",
        "status": "READINESS_ONLY_NO_NEW_FIT",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "protocol": {
            "target": "retain_h0_given_clear; exact H12 clear-first support",
            "strict_oof_rule": "decision_ts < fold_start - 12h AND label_available_ts < fold_start",
            "paired_row_rule": "C0/Cx identity hashes must match per split and fold",
            "scope": "C0--C8 feature-group availability and reuse only; no policy, threshold, portfolio, action, or target experiment",
        },
    }
    return pd.DataFrame(readiness_rows), pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame(), pd.concat(source_rows, ignore_index=True) if source_rows else pd.DataFrame(), report


def run(*, inputs: AuditInputs, output: Path) -> dict[str, Any]:
    """Write an atomic readiness artifact; fitting is intentionally absent."""
    if output.exists():
        raise FileExistsError(output)
    readiness, coverage, source_lineage, report = build_readiness(inputs=inputs)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        outputs: list[str] = []
        for name, frame in (
            ("stage_c_feature_group_readiness.parquet", readiness),
            ("stage_c_feature_group_coverage.parquet", coverage),
            ("stage_c_feature_group_source_lineage.parquet", source_lineage),
        ):
            frame.to_parquet(stage / name, index=False, compression="zstd")
            outputs.append(name)
        _write_json(stage / "correctness_test_report.json", report)
        outputs.append("correctness_test_report.json")
        lines = [
            "# Stage-C C0--C8 feature-group readiness",
            "",
            "This is a read-only audit. It starts no model fit and makes no policy or execution decision.",
            "",
            "| Arm | Group | Status | Features | Action |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for row in readiness.itertuples(index=False):
            lines.append(f"| {row.arm} | {row.feature_group} | {row.availability_status} | {row.feature_count} | {row.next_action} |")
        (stage / "STAGE_C_FEATURE_AVAILABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        outputs.append("STAGE_C_FEATURE_AVAILABILITY_REPORT.md")
        manifest = {
            "schema": "stage_c_feature_group_readiness_v1",
            "status": "READINESS_ONLY_NO_NEW_FIT",
            "inputs": {str(path): _sha256(path) for path in vars(inputs).values()},
            "rows": {"readiness": len(readiness), "coverage": len(coverage), "source_lineage": len(source_lineage)},
            "correctness": report,
            "outputs": {name: _sha256(stage / name) for name in outputs},
        }
        _write_json(stage / "run_manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
