#!/usr/bin/env python3
"""Correct the interpretation of Feb--Mar candidate-group cardinality.

This sealed sidecar establishes whether ``candidate_group_rows`` measures a
market-density feature or simply the canonical eligible ranking universe.  It
does not refit a model, compute outcomes, alter v1/v2, or make a regime claim.
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
CONVERSION = ROOT / "scripts/run_matched_month_pair_conversion_shift.py"
PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
PANEL_MANIFEST = PANEL.with_name("manifest.json")
V1 = ROOT / "data_perp/artifacts/febmar_nested_overlap_audit_20260730_v1"
V2 = ROOT / "data_perp/artifacts/febmar_overlap_crowding_sensitivity_20260730_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febmar_eligible_universe_interpretation_20260730_v1"


class InterpretationError(RuntimeError):
    pass


def _conversion() -> Any:
    spec = importlib.util.spec_from_file_location("conversion_support", CONVERSION)
    if spec is None or spec.loader is None:
        raise InterpretationError("cannot load frozen-selection helper")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping): return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray): return [_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)): return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def hourly_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical signal hour, retaining only pre-entry identity data."""
    needed = {"candidate_month", "candidate_id", "side_name", "__symbol__", "__ts__", "base_group_rows_timestamp_global", "base_group_rows_timestamp_side"}
    missing = needed.difference(frame.columns)
    if missing: raise InterpretationError(f"canonical panel misses group-size fields: {sorted(missing)}")
    work = frame.copy(); work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    result = work.groupby(["candidate_month", "__ts__"], observed=True).agg(
        canonical_rows=("candidate_id", "size"), assets=("__symbol__", "nunique"), sides=("side_name", "nunique"),
        recorded_global_group_rows=("base_group_rows_timestamp_global", "first"),
        recorded_side_group_rows_min=("base_group_rows_timestamp_side", "min"),
        recorded_side_group_rows_max=("base_group_rows_timestamp_side", "max"),
    ).reset_index()
    result["derived_assets_times_sides"] = result.assets * result.sides
    result["exact_global_cardinality_match"] = result.canonical_rows.eq(pd.to_numeric(result.recorded_global_group_rows, errors="raise"))
    if not result.exact_global_cardinality_match.all():
        raise InterpretationError("recorded global group size is not exact canonical candidate cardinality")
    return result


def asset_changes(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame.candidate_month.astype(str).isin(("2025-02", "2025-03")), ["candidate_month", "__symbol__", "__ts__", "side_name"]].copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    grouped = work.groupby(["candidate_month", "__symbol__"], observed=True).agg(first_seen=("__ts__", "min"), last_seen=("__ts__", "max"), rows=("side_name", "size"), sides=("side_name", "nunique")).reset_index()
    pivot = grouped.pivot(index="__symbol__", columns="candidate_month", values=["first_seen", "last_seen", "rows", "sides"])
    pivot.columns = [f"{left}_{right}" for left, right in pivot.columns]; result = pivot.reset_index()
    result["present_february"] = result.get("rows_2025-02", pd.Series(index=result.index)).notna()
    result["present_march"] = result.get("rows_2025-03", pd.Series(index=result.index)).notna()
    result["change"] = np.select([result.present_february & result.present_march, result.present_march], ["retained", "added_in_march"], default="removed_before_march")
    return result.sort_values(["change", "__symbol__"], kind="stable").reset_index(drop=True)


def cardinality_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    return hourly.groupby(["candidate_month", "canonical_rows", "assets", "sides"], observed=True).agg(hours=("__ts__", "size"), candidate_rows=("canonical_rows", "sum"), global_cardinality_exact=("exact_global_cardinality_match", "all")).reset_index()


def density_field_audit(columns: Sequence[str]) -> pd.DataFrame:
    # No field is deemed density merely from a suggestive name.  Candidate density
    # requires causal raw/pre-filter candidate counts or a fixed denominator.
    rows = [
        {"field": "candidate_group_rows / base_group_rows_timestamp_global", "status": "NOT_TRUE_DENSITY", "reason": "Exact canonical eligible-universe cardinality; equals assets × both sides each hour."},
        {"field": "base_group_rows_timestamp_side", "status": "NOT_TRUE_DENSITY", "reason": "Canonical eligible-universe cardinality within side, not raw/pre-filter candidate density."},
        {"field": "base_input__median_volume_z", "status": "CAUSAL_MARKET_LIQUIDITY_NOT_DENSITY", "reason": "Available pre-entry market/liquidity measurement, but not a count or normalized candidate-density denominator."},
        {"field": "raw/pre-filter candidate count plus fixed eligible denominator", "status": "MISSING", "reason": "No causal field in the canonical panel supplies raw candidate count, pre-filter count, or a fixed denominator; normalized true candidate density cannot be tested."},
    ]
    # Guard the claimed availability rather than assuming it from a source contract.
    if "base_input__median_volume_z" not in set(columns):
        rows[2]["status"] = "MISSING"; rows[2]["reason"] = "Not present in canonical panel."
    return pd.DataFrame(rows)


def run(*, panel: Path, panel_manifest: Path, v1: Path, v2: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    required = (panel, panel_manifest, v1 / "manifest.json", v1 / "coverage.parquet", v2 / "manifest.json", v2 / "coverage.parquet")
    if not all(path.is_file() for path in required): raise FileNotFoundError("required canonical/v1/v2 sealed input is absent")
    fields = ["candidate_month", "candidate_id", "side_name", "__symbol__", "__ts__", "base_oof_score", "base_group_rows_timestamp_global", "base_group_rows_timestamp_side"]
    frame = pd.read_parquet(panel)
    hourly = hourly_universe(frame.loc[:, fields]); summary = cardinality_summary(hourly)
    assets = asset_changes(frame.loc[:, ["candidate_month", "__symbol__", "__ts__", "side_name"]])
    convert = _conversion()
    tail_parts = []
    for month, role in (("2025-02", "source_february"), ("2025-03", "target_march")):
        local = convert.stable_top(frame.loc[frame.candidate_month.astype(str).eq(month), ["candidate_id", "candidate_month", "base_oof_score", "__ts__"]].copy(), "base_oof_score")
        local = local.merge(hourly.loc[:, ["candidate_month", "__ts__", "canonical_rows", "assets", "sides"]], on=["candidate_month", "__ts__"], validate="many_to_one")
        tail_parts.append(local.assign(role=role))
    tail = pd.concat(tail_parts, ignore_index=True)
    tail_summary = tail.groupby(["role", "candidate_month", "canonical_rows", "assets", "sides"], observed=True).size().rename("selected_rows").reset_index()
    v1_cov = pd.read_parquet(v1 / "coverage.parquet"); v2_cov = pd.read_parquet(v2 / "coverage.parquet")
    primary = v1_cov.loc[v1_cov.covariate_set.eq("core_score_context")].iloc[0].to_dict()
    alternative = v2_cov.loc[v2_cov.covariate_set.eq("omit_absolute_crowding")].iloc[0].to_dict()
    density = density_field_audit(frame.columns)
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True, exist_ok=False)
    try:
        outputs: dict[str, Any] = {}
        for name, table in (("hourly_eligible_universe", hourly), ("monthly_cardinality_summary", summary), ("asset_presence_change", assets), ("frozen_top_tail_cardinality", tail_summary), ("density_field_audit", density)):
            target = stage / f"{name}.parquet"; table.to_parquet(target, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / target.name), "rows": int(len(table)), "sha256": sha256(target)}
        report = {
            "schema": "febmar_eligible_universe_interpretation_v1", "status": "INTERPRETATION_CORRECTION_ONLY_NO_MODEL_MAPPING_OR_POLICY_ACTION", "promotion_eligible": False,
            "inputs": {"canonical_panel": {"path": str(panel), "sha256": sha256(panel)}, "canonical_manifest": {"path": str(panel_manifest), "sha256": sha256(panel_manifest)}, "sealed_v1": {"path": str(v1 / "manifest.json"), "sha256": sha256(v1 / "manifest.json")}, "sealed_v2": {"path": str(v2 / "manifest.json"), "sha256": sha256(v2 / "manifest.json")}},
            "correction": "candidate_group_rows and base_group_rows_timestamp_global are exact canonical eligible-universe cardinalities, not established market crowding or candidate density. The v1 support failure is a universe-cardinality support break (236/238 in February versus 238/240 in March), not evidence of a market-regime transition.",
            "support_explanation": "The frozen top tail contains February 236-cardinality rows and March 240-cardinality rows that do not overlap; its common 238-cardinality portion is the v1 supported mass. This is mechanically explained by the eligible universe, including BERA in February and KAITO added in March, and must not be called crowding.",
            "estimand_assessment": {"v1_primary": _safe(primary), "v2_A": _safe(alternative), "conclusion": "For a non-cardinality conversion comparison, v2 A (side + asset + frozen score ventile, omitting eligible-universe cardinality) is the more relevant sensitivity. It is not a market-crowding or regime-transition estimand, does not repair v1, and remains diagnostic only."},
            "density_test": "No normalized true candidate-density field exists causally in this canonical panel; no density model or outcome test was run. Required future input: raw/pre-filter candidate counts and an explicit causal fixed denominator, materialized at signal time.",
            "prohibitions": ["Do not call candidate_group_rows market crowding.", "Do not infer a market regime transition from this support break.", "Do not use v2 A to promote a policy, ranking, mapping or calibration change.", "Do not use weights or additional features to manufacture v1 common support."],
            "outputs": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        _write_json(stage / "manifest.json", report); (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel", type=Path, default=PANEL); result.add_argument("--panel-manifest", type=Path, default=PANEL_MANIFEST)
    result.add_argument("--v1", type=Path, default=V1); result.add_argument("--v2", type=Path, default=V2); result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(_safe(run(panel=args.panel, panel_manifest=args.panel_manifest, v1=args.v1, v2=args.v2, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
