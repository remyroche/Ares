#!/usr/bin/env python3
"""Outcome-blind February--March crowding-support sensitivities.

The canonical absolute-crowding estimand remains the sealed v1 primary and
failed.  This runner tests two *predeclared* alternative estimands only: omit
absolute crowding, or retain it as raw/log continuous context.  Neither is a
support repair, a ranking change, nor a policy action.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1_RUNNER = ROOT / "scripts/run_febmar_nested_overlap_audit.py"
V1_ARTIFACT = ROOT / "data_perp/artifacts/febmar_nested_overlap_audit_20260730_v1"
DEFAULT_PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
DEFAULT_MANIFEST = DEFAULT_PANEL.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febmar_overlap_crowding_sensitivity_20260730_v2"
N_BOOTSTRAP = 100


class SensitivityError(RuntimeError):
    pass


def _load_v1() -> Any:
    spec = importlib.util.spec_from_file_location("febmar_overlap_v1", V1_RUNNER)
    if spec is None or spec.loader is None:
        raise SensitivityError("cannot load sealed v1 helper implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sensitivity_configs() -> tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...]:
    """Fixed ex ante alternatives.  This list must never depend on outcomes."""
    core = ("side_name", "__symbol__", "score_ventile")
    return (
        ("omit_absolute_crowding", (), core, "Conditional on side, asset and source-frozen score ventile only; absolute candidate crowding is deliberately omitted."),
        ("continuous_raw_log_crowding", ("candidate_group_rows", "log1p_candidate_group_rows"), core, "Conditional on side, asset, source-frozen score ventile, and raw/log candidate-group size as continuous variables; it is not the primary absolute-crowding estimand."),
    )


def _structural_rollups(excluded: pd.DataFrame) -> pd.DataFrame:
    """Outcome-free v1 structural support diagnosis at auditable granularities."""
    base = excluded.loc[excluded.excluded_by_common_support].copy()
    groups = {
        "role_side": ["role", "side_name"],
        "role_side_score_band": ["role", "side_name", "score_ventile"],
        "role_side_crowding_bin": ["role", "side_name", "candidate_group_size_bin"],
        "role_side_asset": ["role", "side_name", "__symbol__"],
        "role_side_score_crowding_asset": ["role", "side_name", "score_ventile", "candidate_group_size_bin", "__symbol__"],
    }
    pieces = []
    for name, fields in groups.items():
        local = base.groupby(["covariate_set", *fields], dropna=False, observed=True).rows.sum().rename("excluded_rows").reset_index()
        local.insert(1, "rollup", name)
        pieces.append(local)
    return pd.concat(pieces, ignore_index=True)


def run(*, panel: Path, manifest: Path, v1_artifact: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    required = (panel, manifest, v1_artifact / "manifest.json", v1_artifact / "coverage.parquet", v1_artifact / "excluded_cohorts.parquet")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("canonical input or sealed v1 audit input is absent")
    v1 = _load_v1(); common = v1._load_common()
    # The sealed v1 has no passing set and therefore never executed its 250
    # draws.  This separate sensitivity fixes its own 100-draw day-block
    # interval budget before *any* outcome computation.
    v1.N_BOOTSTRAP = N_BOOTSTRAP
    frame, _, _, _ = common._load_canonical(panel)
    source_all = frame.loc[frame.candidate_month.astype(str).eq("2025-02")].copy()
    target_all = frame.loc[frame.candidate_month.astype(str).eq("2025-03")].copy()
    source, target = common.stable_top(source_all, "base_oof_score"), common.stable_top(target_all, "base_oof_score")
    for local in (source, target):
        local["log1p_candidate_group_rows"] = np.log1p(pd.to_numeric(local.candidate_group_rows, errors="raise"))
    v1_coverage = pd.read_parquet(v1_artifact / "coverage.parquet")
    primary = v1_coverage.loc[v1_coverage.covariate_set.eq("core_score_context")].iloc[0].to_dict()
    if bool(primary["common_support_pass"]):
        raise SensitivityError("v1 primary seal unexpectedly does not record failed absolute-crowding support")
    structure = _structural_rollups(pd.read_parquet(v1_artifact / "excluded_cohorts.parquet"))
    coverage_parts: list[pd.DataFrame] = []; balance_parts: list[pd.DataFrame] = []
    response_parts: list[pd.DataFrame] = []; sensitivity_parts: list[pd.DataFrame] = []
    for name, continuous, categorical, description in _sensitivity_configs():
        supported_source, supported_target, odds, overlap_source, overlap_target, summary, balance = v1.fit_support(source, target, continuous=continuous, categorical=categorical)
        summary.update({"covariate_set": name, "continuous_covariates": list(continuous), "categorical_covariates": list(categorical), "estimand_description": description, "outcome_decomposition_status": "RUN" if summary["common_support_pass"] else "NOT_RUN_FAILED_COMMON_SUPPORT"})
        coverage_parts.append(pd.DataFrame([summary]))
        balance_parts.append(balance.assign(covariate_set=name, common_support_pass=summary["common_support_pass"]))
        if not summary["common_support_pass"]:
            continue
        intervals = v1._day_block_intervals(common, source, target, supported_source, supported_target, odds)
        response = pd.DataFrame(v1._decompose(common, source, target, supported_source, supported_target, odds))
        response["covariate_set"] = name; response["estimand_description"] = description
        response["interval_method"] = f"fixed_support_fixed_weight_day_block_bootstrap_{N_BOOTSTRAP}"
        for row_index, row in response.iterrows():
            for field, (low, high) in intervals[row.metric].items():
                response.loc[row_index, f"{field}_ci95_low"] = low
                response.loc[row_index, f"{field}_ci95_high"] = high
        response_parts.append(response)
        sensitivity_parts.append(v1._overlap_sensitivity(common, supported_source, supported_target, overlap_source, overlap_target).assign(covariate_set=name, estimand_description=description, sensitivity="overlap_weights_not_a_support_repair"))
    coverage = pd.concat(coverage_parts, ignore_index=True); balance = pd.concat(balance_parts, ignore_index=True)
    response = pd.concat(response_parts, ignore_index=True) if response_parts else pd.DataFrame()
    overlap = pd.concat(sensitivity_parts, ignore_index=True) if sensitivity_parts else pd.DataFrame()
    selected = pd.concat((source.assign(role="source_february"), target.assign(role="target_march")), ignore_index=True).loc[:, ["candidate_id", "side_name", "__symbol__", "__ts__", "candidate_month", "base_oof_score", "score_ventile", "candidate_group_rows", "candidate_group_size_bin", "role"]]
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True, exist_ok=False)
    try:
        outputs: dict[str, Any] = {}
        for name, table in (("coverage", coverage), ("balance", balance), ("conditional_response_decomposition", response), ("overlap_weight_sensitivity", overlap), ("structural_excluded_mass_from_v1", structure), ("frozen_selection", selected)):
            target_path = stage / f"{name}.parquet"; table.to_parquet(target_path, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / target_path.name), "rows": int(len(table)), "sha256": sha256(target_path)}
        report = {
            "schema": "febmar_overlap_crowding_sensitivity_v2", "status": "DIAGNOSTIC_SENSITIVITY_ONLY_NO_MAPPING_NO_POLICY_ACTION", "promotion_eligible": False,
            "inputs": {"canonical_panel": {"path": str(panel), "sha256": sha256(panel)}, "canonical_manifest": {"path": str(manifest), "sha256": sha256(manifest)}, "sealed_v1_primary": {"path": str(v1_artifact / "manifest.json"), "sha256": sha256(v1_artifact / "manifest.json"), "primary_covariate_set": "core_score_context", "primary_common_support_pass": False}},
            "primary_status": {"absolute_crowding_estimand": "v1 core_score_context: side + asset + source-frozen score ventile + categorical candidate-group-size bin", "common_support_pass": False, "failure": "target March support coverage below predeclared 50% gate", "v1_primary_coverage": _safe(primary)},
            "sensitivity_contract": "The two candidate context sets are fixed before outcome access. Passing a sensitivity only supports its named conditional estimand; it neither repairs v1 common support nor authorises a ranking, calibration, mapping or policy change.",
            "outcome_contract": "Outcome decomposition, overlap-weight sensitivity and 95% fixed-support/fixed-weight day-block intervals are emitted only after a sensitivity clears the same v1 gates.",
            "structural_diagnosis": "Structural excluded mass is read from the sealed v1 absolute-crowding support audit and reported by role, side, frozen score band, crowding bin and asset. It is an outcome-free diagnostic, not a covariate-set selection rule.",
            "outputs": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        _write_json(stage / "manifest.json", report)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel", type=Path, default=DEFAULT_PANEL); result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--v1-artifact", type=Path, default=V1_ARTIFACT); result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args()
    print(json.dumps(_safe(run(panel=args.panel, manifest=args.manifest, v1_artifact=args.v1_artifact, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
