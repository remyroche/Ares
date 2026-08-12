"""Fail-closed F4 selection of a portable feature representation.

F4 is deliberately a *consumer* of completed F0--F3 experiments.  It does
not fit a model, tune hyperparameters, regenerate an OOF score, or evaluate a
terminal holdout.  The input is a compact, transport-level development ledger
whose rows have already been produced by chronological runs.  This separation
matters: a tail table alone is not evidence of an incremental feature effect.

The selector accepts one pooled-global ranking per representation/transport.
It rejects side-local or timestamp-local top-k tables, missing grouped MDA,
coverage below 99%, and any non-development/November-final row.  Its primary
selection quantity is incremental global top-10 net bps.  Top-5, worst-month
top-10, and rank-IC are recorded as secondary diagnostics only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_portability_f4_compact import F4_COMPACT_PREFIX, selected_compact_feature_manifest


SCHEMA = "stage_a_f4_feature_contract_selector_v1"
DEVELOPMENT_STAGES = frozenset({"development", "development_oof", "development_transport"})
REQUIRED_EVIDENCE_COLUMNS = frozenset(
    {
        "representation",
        "transport",
        "feature_count",
        "coverage",
        "incremental_top10_net_bps",
        "transport_mda_bps",
        "development_stage",
        "chronological_verified",
        "global_ranking_verified",
        "ranking_scope",
        "model_hpo_performed",
    }
)


class FeaturePortabilitySelectionError(ValueError):
    """Raised for a malformed or unsafe F4 input contract."""


@dataclass(frozen=True)
class FeaturePortabilitySelectionPolicy:
    """Fixed F4 admission and compactness policy."""

    min_coverage: float = 0.99
    required_transports: tuple[str, ...] = ()
    one_se_multiplier: float = 1.0
    required_representation_prefix: str | None = None
    require_nonnegative_f3_control_lift: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_coverage <= 1.0:
            raise FeaturePortabilitySelectionError("min_coverage must be in [0, 1]")
        if self.one_se_multiplier < 0.0:
            raise FeaturePortabilitySelectionError("one_se_multiplier must be non-negative")
        if len(set(self.required_transports)) != len(self.required_transports):
            raise FeaturePortabilitySelectionError("required_transports must be unique")
        if self.required_representation_prefix is not None and not str(self.required_representation_prefix):
            raise FeaturePortabilitySelectionError("required_representation_prefix must be non-empty when set")


@dataclass(frozen=True)
class FeaturePortabilitySelection:
    diagnostics: pd.DataFrame
    selected: Mapping[str, Any] | None
    manifest: Mapping[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise FeaturePortabilitySelectionError(f"unsupported table input (use parquet or csv): {path}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FeaturePortabilitySelectionError(f"invalid JSON input: {path}") from exc


def _normalise_bool(value: pd.Series, name: str) -> pd.Series:
    if value.isna().any():
        raise FeaturePortabilitySelectionError(f"{name} may not contain missing values")
    normalized = value.map(
        lambda item: item if isinstance(item, bool) else str(item).strip().lower() in {"true", "1"}
    )
    return normalized.astype(bool)


def _normalise_evidence(evidence: pd.DataFrame, policy: FeaturePortabilitySelectionPolicy) -> pd.DataFrame:
    missing = sorted(REQUIRED_EVIDENCE_COLUMNS.difference(evidence.columns))
    if missing:
        raise FeaturePortabilitySelectionError(f"F4 evidence missing required columns: {missing}")
    if evidence.empty:
        raise FeaturePortabilitySelectionError("F4 evidence is empty")
    out = evidence.copy()
    for name in ("representation", "transport", "development_stage"):
        out[name] = out[name].astype(str)
    if (out["representation"].str.len() == 0).any() or (out["transport"].str.len() == 0).any():
        raise FeaturePortabilitySelectionError("representation and transport names must be non-empty")
    if out.duplicated(["representation", "transport"]).any():
        raise FeaturePortabilitySelectionError("F4 requires exactly one pooled result per representation x transport")
    for name in ("feature_count", "coverage", "incremental_top10_net_bps", "transport_mda_bps"):
        out[name] = pd.to_numeric(out[name], errors="coerce")
    if out[["feature_count", "coverage", "incremental_top10_net_bps"]].isna().any().any():
        raise FeaturePortabilitySelectionError("F4 requires finite feature_count, coverage, and top-10 lift")
    if policy.require_nonnegative_f3_control_lift:
        required_f3 = {"incremental_vs_f3_top10_net_bps", "full_f3_control_eligible"}
        missing_f3 = sorted(required_f3.difference(out.columns))
        if missing_f3:
            raise FeaturePortabilitySelectionError(
                f"F4 compact selection requires conditional full-F3 control evidence: {missing_f3}"
            )
        out["full_f3_control_eligible"] = _normalise_bool(
            out["full_f3_control_eligible"], "full_f3_control_eligible",
        )
        out["incremental_vs_f3_top10_net_bps"] = pd.to_numeric(
            out["incremental_vs_f3_top10_net_bps"], errors="coerce",
        )
        eligible_f3 = out["full_f3_control_eligible"]
        if out.loc[eligible_f3, "incremental_vs_f3_top10_net_bps"].isna().any():
            raise FeaturePortabilitySelectionError("coverage-eligible full-F3 control lift must be finite")
    if (out["feature_count"] < 1).any() or (out["feature_count"] % 1 != 0).any():
        raise FeaturePortabilitySelectionError("feature_count must be a positive integer")
    if (~out["coverage"].between(0.0, 1.0)).any():
        raise FeaturePortabilitySelectionError("coverage must be in [0, 1]")
    out["chronological_verified"] = _normalise_bool(out["chronological_verified"], "chronological_verified")
    out["global_ranking_verified"] = _normalise_bool(out["global_ranking_verified"], "global_ranking_verified")
    if not out["development_stage"].isin(DEVELOPMENT_STAGES).all():
        values = sorted(set(out.loc[~out["development_stage"].isin(DEVELOPMENT_STAGES), "development_stage"]))
        raise FeaturePortabilitySelectionError(f"F4 accepts development rows only, not: {values}")
    if not out["ranking_scope"].astype(str).eq("pooled_global").all():
        raise FeaturePortabilitySelectionError("F4 requires one pooled_global ranking, never a side/timestamp-local top-k")
    out["model_hpo_performed"] = _normalise_bool(out["model_hpo_performed"], "model_hpo_performed")
    if out["model_hpo_performed"].any():
        raise FeaturePortabilitySelectionError("F4 is a selector and must not consume model-HPO evidence")
    # Explicitly prohibit terminal-November leakage even if a producer gives it
    # a misleading development label.
    for name in ("period", "evaluation_start", "evaluation_end", "final_oos_name"):
        if name in out:
            values = out[name].dropna().astype(str).str.lower()
            if values.str.contains(r"2024-11|final.*oos|untouched", regex=True).any():
                raise FeaturePortabilitySelectionError("F4 must never consume final November OOS evidence")
    if policy.required_transports:
        expected = set(policy.required_transports)
        observed = set(out["transport"])
        if observed != expected:
            raise FeaturePortabilitySelectionError(
                f"required transports do not match evidence: expected={sorted(expected)}, observed={sorted(observed)}"
            )
    elif out["transport"].nunique() < 2:
        raise FeaturePortabilitySelectionError("F4 requires evidence from both chronological transports")
    return out


def _lineage_index(lineage: Any) -> dict[str, list[Mapping[str, Any]]]:
    if isinstance(lineage, Mapping):
        records: Any = lineage.get("arms", lineage.get("records", lineage.get("lineage", lineage)))
        if isinstance(records, Mapping):
            records = list(records.values())
    else:
        records = lineage
    if not isinstance(records, list):
        raise FeaturePortabilitySelectionError("lineage must be a JSON list or an arms/records mapping")
    output: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise FeaturePortabilitySelectionError("lineage records must be objects")
        name = record.get("representation", record.get("arm"))
        if name is None:
            raise FeaturePortabilitySelectionError("lineage record lacks representation/arm")
        output.setdefault(str(name), []).append(record)
    return output


def _audit_feature_coverage(audit: pd.DataFrame) -> dict[str, float]:
    required = {"feature", "coverage"}
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise FeaturePortabilitySelectionError(f"audit lacks required columns: {missing}")
    checked = audit.copy()
    if "reference_ready" in checked:
        checked = checked.loc[_normalise_bool(checked["reference_ready"], "reference_ready")]
    checked["coverage"] = pd.to_numeric(checked["coverage"], errors="coerce")
    return checked.groupby("feature", observed=True)["coverage"].min().to_dict()


def validate_lineage_and_audit(
    evidence: pd.DataFrame,
    lineage: Any,
    audit: pd.DataFrame,
    *,
    policy: FeaturePortabilitySelectionPolicy = FeaturePortabilitySelectionPolicy(),
) -> pd.DataFrame:
    """Attach non-negotiable F0--F3 lineage/audit checks to F4 evidence.

    A representation's recorded feature count must agree with each completed
    side/transport lineage record.  Every declared feature must have a
    reference-ready audit coverage record.  Thus a transform whose source was
    audited but whose generated feature was never audited is rejected rather
    than assumed portable.
    """
    out = _normalise_evidence(evidence, policy)
    by_representation = _lineage_index(lineage)
    coverage = _audit_feature_coverage(audit)
    rows: list[dict[str, Any]] = []
    for row in out.to_dict("records"):
        records = [item for item in by_representation.get(row["representation"], []) if str(item.get("run", item.get("transport", row["transport"]))) == row["transport"]]
        reasons: list[str] = []
        if not records:
            reasons.append("missing_transport_lineage")
        else:
            if not all(bool(item.get("oof_materialised", False)) for item in records):
                reasons.append("oof_not_materialised")
            counts = {int(item.get("feature_count", -1)) for item in records}
            # The base contract is side-local; compactness is conservatively
            # measured by the larger side's declared field count.
            if not counts or max(counts) != int(row["feature_count"]):
                reasons.append("feature_count_lineage_mismatch")
            features = [str(feature) for item in records for feature in item.get("features", [])]
            if not features:
                reasons.append("missing_declared_feature_list")
            else:
                absent = sorted(set(features).difference(coverage))
                below = sorted(feature for feature in set(features) if coverage.get(feature, -np.inf) < policy.min_coverage)
                if absent:
                    reasons.append("audit_missing_declared_features")
                if below:
                    reasons.append("audit_coverage_below_99pct")
        row["lineage_audit_verified"] = not reasons
        row["lineage_audit_reasons"] = ";".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def select_feature_portability_contract(
    evidence: pd.DataFrame,
    *,
    policy: FeaturePortabilitySelectionPolicy = FeaturePortabilitySelectionPolicy(),
) -> FeaturePortabilitySelection:
    """Select a compact F4 representation from complete development evidence."""
    out = _normalise_evidence(evidence, policy)
    if "lineage_audit_verified" not in out:
        out["lineage_audit_verified"] = False
        out["lineage_audit_reasons"] = "lineage_and_audit_not_validated"
    out["lineage_audit_verified"] = _normalise_bool(out["lineage_audit_verified"], "lineage_audit_verified")
    diagnostics: list[dict[str, Any]] = []
    for representation, rows in out.groupby("representation", sort=True):
        rows = rows.sort_values("transport", kind="stable")
        reasons: list[str] = []
        if rows["transport"].nunique() < 2:
            reasons.append("missing_both_transports")
        if policy.required_representation_prefix is not None and not str(representation).startswith(policy.required_representation_prefix):
            reasons.append("not_required_compact_f4_representation")
        if not rows["chronological_verified"].all():
            reasons.append("chronology_unproven")
        if not rows["global_ranking_verified"].all():
            reasons.append("not_one_pooled_global_ranking")
        if not rows["lineage_audit_verified"].all():
            reasons.extend(sorted(set(filter(None, rows.loc[~rows["lineage_audit_verified"], "lineage_audit_reasons"].astype(str)))))
        if (rows["coverage"] < policy.min_coverage).any():
            reasons.append("coverage_below_99pct")
        if (rows["incremental_top10_net_bps"] <= 0.0).any():
            reasons.append("non_positive_incremental_top10_in_transport")
        if policy.require_nonnegative_f3_control_lift and (
            rows.loc[rows["full_f3_control_eligible"], "incremental_vs_f3_top10_net_bps"] < 0.0
        ).any():
            reasons.append("harms_full_f3_control_in_transport")
        top10 = rows["incremental_top10_net_bps"].to_numpy(float)
        mda = rows["transport_mda_bps"].to_numpy(float)
        finite_mda = mda[np.isfinite(mda)]
        if len(finite_mda) != len(rows):
            reasons.append("missing_grouped_chronological_transport_mda")
        median_mda = float(np.median(finite_mda)) if len(finite_mda) else float("nan")
        mad_mda = float(np.median(np.abs(finite_mda - median_mda))) if len(finite_mda) else float("nan")
        stable_score = median_mda - 0.5 * mad_mda if len(finite_mda) else float("nan")
        # A zero/negative stable MDA does not erase the primary economic gate,
        # but it is never allowed to win a portability selection.
        if np.isfinite(stable_score) and stable_score <= 0.0:
            reasons.append("non_positive_stable_transport_mda")
        std_error = float(np.std(top10, ddof=1) / np.sqrt(len(top10))) if len(top10) > 1 else float("nan")
        record: dict[str, Any] = {
            "representation": representation,
            "transport_count": int(len(rows)),
            "feature_count": int(rows["feature_count"].max()),
            "top10_incremental_mean_bps": float(np.mean(top10)),
            "top10_incremental_median_bps": float(np.median(top10)),
            "top10_incremental_worst_bps": float(np.min(top10)),
            "top10_incremental_standard_error_bps": std_error,
            "transport_mda_median_bps": median_mda,
            "transport_mda_mad_bps": mad_mda,
            "stable_transport_mda_score_bps": stable_score,
            "coverage_min": float(rows["coverage"].min()),
            "both_transport_positive_incremental_top10": bool((top10 > 0.0).all()),
            "full_f3_control_eligible": bool(rows["full_f3_control_eligible"].all()) if policy.require_nonnegative_f3_control_lift else None,
            "both_transport_nonnegative_full_f3_lift": bool(
                (rows.loc[rows["full_f3_control_eligible"], "incremental_vs_f3_top10_net_bps"] >= 0.0).all()
            ) if policy.require_nonnegative_f3_control_lift else None,
            "admissible": not reasons,
            "rejection_reasons": ";".join(dict.fromkeys(reasons)),
        }
        for source, target in (
            ("incremental_top5_net_bps", "top5_incremental_median_bps"),
            ("incremental_worst_month_top10_net_bps", "worst_month_top10_incremental_worst_bps"),
            ("incremental_rank_ic", "rank_ic_incremental_median"),
        ):
            values = pd.to_numeric(rows[source], errors="coerce").dropna().to_numpy(float) if source in rows else np.array([])
            record[target] = float(np.median(values)) if len(values) else float("nan")
        diagnostics.append(record)
    diagnostic_frame = pd.DataFrame(diagnostics).sort_values("representation", kind="stable").reset_index(drop=True)
    accepted = diagnostic_frame.loc[diagnostic_frame["admissible"]].copy()
    selected: dict[str, Any] | None = None
    if not accepted.empty:
        best = float(accepted["top10_incremental_mean_bps"].max())
        best_row = accepted.loc[accepted["top10_incremental_mean_bps"].eq(best)].sort_values(
            ["top10_incremental_standard_error_bps", "feature_count", "representation"], kind="stable"
        ).iloc[0]
        threshold = best - policy.one_se_multiplier * float(best_row["top10_incremental_standard_error_bps"])
        compact = accepted.loc[accepted["top10_incremental_mean_bps"].ge(threshold)].sort_values(
            ["feature_count", "stable_transport_mda_score_bps", "representation"], ascending=[True, False, True], kind="stable"
        )
        winner = compact.iloc[0]
        selected = {
            "representation": str(winner["representation"]),
            "feature_count": int(winner["feature_count"]),
            "selection_rule": "smallest_feature_contract_within_one_standard_error_of_best_incremental_global_top10_net",
            "best_primary_mean_bps": best,
            "one_se_threshold_bps": threshold,
            "stable_transport_mda_score_bps": float(winner["stable_transport_mda_score_bps"]),
            "full_f3_control_eligible": bool(winner["full_f3_control_eligible"])
            if policy.require_nonnegative_f3_control_lift else False,
        }
    manifest = {
        "schema": SCHEMA,
        "status": "F4_FEATURE_CONTRACT_SELECTED" if selected else "F4_NO_FEATURE_CONTRACT_ADVANCES",
        "selection_scope": "development_only",
        "final_november_oos_consumed": False,
        "global_ranking_required": "one pooled-global ranking after common-bps mapping; never per-side or per-timestamp",
        "model_hpo_performed": False,
        "policy": asdict(policy),
        "selected": selected,
        "representations": int(len(diagnostic_frame)),
    }
    return FeaturePortabilitySelection(diagnostics=diagnostic_frame, selected=selected, manifest=manifest)


def write_feature_portability_selection_artifacts(
    result: FeaturePortabilitySelection,
    output_dir: Path,
    *,
    input_paths: Mapping[str, Path],
    compact_contracts: Mapping[str, Any] | None = None,
) -> Mapping[str, Path]:
    """Atomically persist a new F4 result directory; overwrites are forbidden."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"F4 output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        diagnostic_path = stage / "f4_feature_contract_diagnostics.parquet"
        selected_path = stage / "f4_selected_feature_contract.json"
        manifest_path = stage / "f4_run_manifest.json"
        result.diagnostics.to_parquet(diagnostic_path, index=False, compression="zstd")
        selected_path.write_text(json.dumps(result.selected, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
        outputs = {
            diagnostic_path.name: _sha256(diagnostic_path),
            selected_path.name: _sha256(selected_path),
        }
        selected_compact_path: Path | None = None
        if compact_contracts is not None:
            # This is the canonical linked-spec artifact for the selected F4
            # contract.  It is written only after the feature result itself is
            # selected and refuses to synthesize a union/intersection from
            # transport-specific field lists.
            selected_policy = result.manifest.get("policy", {})
            if (
                selected_policy.get("required_representation_prefix") != F4_COMPACT_PREFIX
                or not bool(selected_policy.get("require_nonnegative_f3_control_lift", False))
            ):
                raise FeaturePortabilitySelectionError(
                    "a promoted F4 compact manifest requires explicit F4-prefix and full-F3-control selection gates"
                )
            try:
                compact_manifest = selected_compact_feature_manifest(
                    selection=result.selected,
                    compact_contracts=compact_contracts,
                    required_transports=result.manifest.get("policy", {}).get("required_transports", ()),
                )
            except ValueError as exc:
                raise FeaturePortabilitySelectionError(f"F4 compact manifest failed closed: {exc}") from exc
            selected_compact_path = stage / "portable_feature_manifest.json"
            compact_manifest["selection_artifact"] = {
                "path": selected_path.name,
                "sha256": _sha256(selected_path),
            }
            selected_compact_path.write_text(
                json.dumps(compact_manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
                encoding="utf-8",
            )
            outputs[selected_compact_path.name] = _sha256(selected_compact_path)
        manifest = dict(result.manifest)
        if compact_contracts is not None:
            manifest["compact_manifest_status"] = compact_manifest["status"]
            if compact_manifest["status"] != "F4_TRANSPORT_SELECTED_COMPACT_FEATURE_MANIFEST":
                # A transport-adaptive procedure can look attractive in the
                # aggregate, but it is not an actual portable feature
                # *contract* when its exact lists differ across eras.  Keep
                # the diagnostic selection JSON, but make the terminal run
                # status unambiguously fail closed.
                manifest["status"] = "F4_NO_COMPACT_FEATURE_CONTRACT_ADVANCES"
        manifest.update(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in sorted(input_paths.items())},
                "outputs": outputs,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    output_paths = {
        "diagnostics": output_dir / "f4_feature_contract_diagnostics.parquet",
        "selected": output_dir / "f4_selected_feature_contract.json",
        "manifest": output_dir / "f4_run_manifest.json",
    }
    if compact_contracts is not None:
        output_paths["portable_feature_manifest"] = output_dir / "portable_feature_manifest.json"
    return output_paths


def completed_f0_f3_evidence(
    results: pd.DataFrame,
    lineage: Any,
    audit: pd.DataFrame,
    *,
    result_manifest: Mapping[str, Any],
    mda: pd.DataFrame | None = None,
    policy: FeaturePortabilitySelectionPolicy = FeaturePortabilitySelectionPolicy(),
) -> pd.DataFrame:
    """Adapt the completed F0--F3 bundle, failing closed without MDA.

    The historical merged table has paired pooled-global tail returns, but no
    grouped chronological feature-permutation values.  It can therefore
    produce a diagnostic F4 result, never an admission.  Supplying a complete
    MDA table (``representation, transport, transport_mda_bps``) enables a
    future genuine F4 selection without changing this adapter.
    """
    if result_manifest.get("base_feature_decision", {}).get("status") != "BASE_FEATURE_WINNER_SELECTED_ON_DEVELOPMENT_ONLY":
        raise FeaturePortabilitySelectionError("completed result manifest does not prove development-only selection")
    required = {"transport", "ranking_basis", "top_fraction", "scope", "arm", "net_bps_per_trade", "feature_contract_99pct_pass", "arm_status"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise FeaturePortabilitySelectionError(f"completed results lack required columns: {missing}")
    if results["transport"].astype(str).str.contains("final_untouched_oos|2024_11", case=False, regex=True).any():
        raise FeaturePortabilitySelectionError("completed results include final November OOS")
    lineage_by_arm = _lineage_index(lineage)
    mda_lookup: dict[tuple[str, str], float] = {}
    if mda is not None:
        needed = {"representation", "transport", "transport_mda_bps"}
        if missing_mda := sorted(needed.difference(mda.columns)):
            raise FeaturePortabilitySelectionError(f"MDA input lacks required columns: {missing_mda}")
        if mda.duplicated(["representation", "transport"]).any():
            raise FeaturePortabilitySelectionError("MDA input must contain one result per representation x transport")
        mda_lookup = {(str(row.representation), str(row.transport)): float(row.transport_mda_bps) for row in mda.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    baseline = "F0_current_frozen"
    base = results.loc[(results["ranking_basis"].eq("global_common_bps_score")) & (results["scope"].eq("global"))]
    for (transport, arm), arm_rows in base.groupby(["transport", "arm"], sort=True):
        if arm == baseline:
            continue
        control = base.loc[(base["transport"].eq(transport)) & (base["arm"].eq(baseline))]
        def value(part: pd.DataFrame, fraction: float) -> float:
            selected = part.loc[np.isclose(pd.to_numeric(part["top_fraction"], errors="coerce"), fraction)]
            return float(selected["net_bps_per_trade"].iloc[0]) if len(selected) == 1 else float("nan")
        records = [item for item in lineage_by_arm.get(str(arm), []) if str(item.get("run")) == str(transport)]
        counts = {int(item.get("feature_count", -1)) for item in records}
        coverage_pass = bool(arm_rows["feature_contract_99pct_pass"].astype(bool).all())
        top10 = value(arm_rows, 0.10) - value(control, 0.10)
        top5 = value(arm_rows, 0.05) - value(control, 0.05)
        monthly_arm = results.loc[(results["transport"].eq(transport)) & (results["arm"].eq(arm)) & (results["scope"].eq("month")) & np.isclose(pd.to_numeric(results["top_fraction"], errors="coerce"), 0.10)]
        monthly_base = results.loc[(results["transport"].eq(transport)) & (results["arm"].eq(baseline)) & (results["scope"].eq("month")) & np.isclose(pd.to_numeric(results["top_fraction"], errors="coerce"), 0.10)]
        month_lift = monthly_arm.merge(monthly_base, on="period", suffixes=("_arm", "_base"))
        rows.append({
            "representation": str(arm), "transport": str(transport),
            "feature_count": max(counts) if counts else -1,
            "coverage": 1.0 if coverage_pass else 0.0,
            "incremental_top10_net_bps": top10,
            "incremental_top5_net_bps": top5,
            "incremental_worst_month_top10_net_bps": float((month_lift["net_bps_per_trade_arm"] - month_lift["net_bps_per_trade_base"]).min()) if not month_lift.empty else float("nan"),
            "incremental_rank_ic": float("nan"),
            "transport_mda_bps": mda_lookup.get((str(arm), str(transport)), float("nan")),
            "development_stage": "development_transport",
            "chronological_verified": True,
            "global_ranking_verified": True,
            "ranking_scope": "pooled_global",
            "model_hpo_performed": False,
        })
    evidence = pd.DataFrame(rows)
    # Leave NaN MDA explicit for a user-facing diagnostic, then selection's
    # strict normalizer will reject it.  This is intentionally not fabricated.
    return validate_lineage_and_audit(evidence, lineage, audit, policy=policy)
