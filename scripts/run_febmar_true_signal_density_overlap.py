#!/usr/bin/env python3
"""Outcome-blind Feb--Mar overlap sensitivity using frozen score-signal density.

This is distinct from eligible-universe cardinality.  A February-only numeric
base-score cutoff and band are frozen before March is read; per-hour counts and
fractions of scores above/near that cutoff form pre-entry signal-density fields.
The diagnostic never treats eligible asset count as market crowding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1_RUNNER = ROOT / "scripts/run_febmar_nested_overlap_audit.py"
PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
PANEL_MANIFEST = PANEL.with_name("manifest.json")
INTERPRETATION = ROOT / "data_perp/artifacts/febmar_eligible_universe_interpretation_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febmar_true_signal_density_overlap_20260730_v1"
TOP_FRACTION = .10
N_BOOTSTRAP = 100


class DensityError(RuntimeError):
    pass


def _load_v1() -> Any:
    spec = importlib.util.spec_from_file_location("febmar_v1_density", V1_RUNNER)
    if spec is None or spec.loader is None: raise DensityError("cannot load v1 helpers")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
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
        temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def freeze_february_score_definition(february: pd.DataFrame) -> dict[str, float]:
    """Freeze numeric cutoff and local (q90--q95) width from February only."""
    score = pd.to_numeric(february.base_oof_score, errors="raise").to_numpy(float)
    if len(score) < 2 or not np.isfinite(score).all(): raise DensityError("February score distribution is unavailable")
    order = np.lexsort((february.candidate_id.astype(str).to_numpy(), -score))
    cutoff = float(score[order[max(0, math.ceil(len(score) * TOP_FRACTION) - 1)]])
    q90, q95 = np.quantile(score, [.90, .95])
    band = float(max(q95 - q90, 1e-12))
    return {"top_fraction": TOP_FRACTION, "numeric_top10_cutoff": cutoff, "q90": float(q90), "q95": float(q95), "near_cutoff_band_width": band, "definition": "abs(score - frozen February top-10 numeric cutoff) <= frozen February (q95 - q90)"}


def materialize_signal_density(frame: pd.DataFrame, frozen: Mapping[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build signal-time fields using fixed February score thresholds only."""
    work = frame.copy(); work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    score = pd.to_numeric(work.base_oof_score, errors="raise")
    cutoff, band = float(frozen["numeric_top10_cutoff"]), float(frozen["near_cutoff_band_width"])
    work["__above_frozen_threshold__"] = score.ge(cutoff)
    work["__near_frozen_cutoff__"] = score.sub(cutoff).abs().le(band)
    grouped = work.groupby("__ts__", observed=True)
    hourly = grouped.agg(
        eligible_asset_count=("__symbol__", "nunique"), eligible_rows=("candidate_id", "size"), eligible_sides=("side_name", "nunique"),
        above_threshold_count=("__above_frozen_threshold__", "sum"), near_cutoff_count=("__near_frozen_cutoff__", "sum"),
    )
    long_above = work.loc[work.side_name.astype(str).str.lower().eq("long")].groupby("__ts__", observed=True).__above_frozen_threshold__.sum()
    short_above = work.loc[work.side_name.astype(str).str.lower().eq("short")].groupby("__ts__", observed=True).__above_frozen_threshold__.sum()
    hourly["above_threshold_long_count"] = long_above.reindex(hourly.index, fill_value=0).astype(int)
    hourly["above_threshold_short_count"] = short_above.reindex(hourly.index, fill_value=0).astype(int)
    hourly["above_threshold_fraction"] = hourly.above_threshold_count / hourly.eligible_rows
    hourly["near_cutoff_fraction"] = hourly.near_cutoff_count / hourly.eligible_rows
    hourly["above_threshold_long_short_imbalance"] = (hourly.above_threshold_long_count - hourly.above_threshold_short_count) / hourly.above_threshold_count.clip(lower=1)
    hourly = hourly.reset_index()
    fields = ["above_threshold_count", "above_threshold_fraction", "above_threshold_long_short_imbalance", "near_cutoff_fraction"]
    joined = work.merge(hourly.loc[:, ["__ts__", "eligible_asset_count", "eligible_rows", *fields]], on="__ts__", validate="many_to_one")
    return joined, hourly


def run(*, panel: Path, panel_manifest: Path, interpretation: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    required = (panel, panel_manifest, interpretation / "manifest.json")
    if not all(path.is_file() for path in required): raise FileNotFoundError("canonical panel or interpretation sidecar is absent")
    v1 = _load_v1(); common = v1._load_common(); v1.N_BOOTSTRAP = N_BOOTSTRAP
    frame, _, _, _ = common._load_canonical(panel)
    feb_all, mar_all = frame.loc[frame.candidate_month.astype(str).eq("2025-02")].copy(), frame.loc[frame.candidate_month.astype(str).eq("2025-03")].copy()
    frozen = freeze_february_score_definition(feb_all)
    enriched, hourly = materialize_signal_density(pd.concat([feb_all, mar_all], ignore_index=True), frozen)
    source = common.stable_top(enriched.loc[enriched.candidate_month.astype(str).eq("2025-02")].copy(), "base_oof_score")
    target = common.stable_top(enriched.loc[enriched.candidate_month.astype(str).eq("2025-03")].copy(), "base_oof_score")
    continuous = ("above_threshold_count", "above_threshold_fraction", "above_threshold_long_short_imbalance", "near_cutoff_fraction")
    categorical = ("side_name", "__symbol__", "score_ventile")
    left, right, odds, overlap_left, overlap_right, coverage, balance = v1.fit_support(source, target, continuous=continuous, categorical=categorical)
    coverage.update({"covariate_set": "frozen_score_signal_density", "continuous_covariates": list(continuous), "categorical_covariates": list(categorical), "outcome_decomposition_status": "RUN" if coverage["common_support_pass"] else "NOT_RUN_FAILED_COMMON_SUPPORT"})
    response = pd.DataFrame(); overlap = pd.DataFrame()
    if coverage["common_support_pass"]:
        intervals = v1._day_block_intervals(common, source, target, left, right, odds)
        response = pd.DataFrame(v1._decompose(common, source, target, left, right, odds))
        response["covariate_set"] = "frozen_score_signal_density"; response["interval_method"] = f"fixed_support_fixed_weight_day_block_bootstrap_{N_BOOTSTRAP}"
        for index, row in response.iterrows():
            for field, (low, high) in intervals[row.metric].items(): response.loc[index, f"{field}_ci95_low"], response.loc[index, f"{field}_ci95_high"] = low, high
        overlap = v1._overlap_sensitivity(common, left, right, overlap_left, overlap_right).assign(covariate_set="frozen_score_signal_density", sensitivity="overlap_weights_not_a_support_repair")
    hour_summary = hourly.assign(candidate_month=pd.to_datetime(hourly.__ts__, utc=True).dt.strftime("%Y-%m")).groupby("candidate_month", observed=True)[["eligible_asset_count", "eligible_rows", "above_threshold_count", "above_threshold_fraction", "above_threshold_long_short_imbalance", "near_cutoff_fraction"]].agg(["mean", "std", "min", "max"]).reset_index()
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True, exist_ok=False)
    try:
        outputs: dict[str, Any] = {}
        for name, table in (("coverage", pd.DataFrame([coverage])), ("balance", balance.assign(covariate_set="frozen_score_signal_density", common_support_pass=coverage["common_support_pass"])), ("conditional_response_decomposition", response), ("overlap_weight_sensitivity", overlap), ("hourly_frozen_signal_density", hourly), ("monthly_signal_density_summary", hour_summary)):
            target_path = stage / f"{name}.parquet"; table.to_parquet(target_path, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / target_path.name), "rows": int(len(table)), "sha256": sha256(target_path)}
        report = {
            "schema": "febmar_true_signal_density_overlap_v1", "status": "DIAGNOSTIC_SENSITIVITY_ONLY_NO_MAPPING_NO_POLICY_ACTION", "promotion_eligible": False,
            "inputs": {"canonical_panel": {"path": str(panel), "sha256": sha256(panel)}, "canonical_manifest": {"path": str(panel_manifest), "sha256": sha256(panel_manifest)}, "interpretation_correction": {"path": str(interpretation / "manifest.json"), "sha256": sha256(interpretation / "manifest.json")}},
            "frozen_definition": frozen,
            "estimand": "Conditional on side, asset, source-frozen monthly score ventile and signal-time frozen-score density fields. Eligible asset count is reported separately and is not a matching covariate or market-crowding proxy.",
            "support_contract": "Propensity/support fit uses only the named causal context fields; no execution outcome enters it. The unchanged v1 coverage/ESS/balance gates apply. Outcome decomposition and intervals are emitted only on a pass.",
            "interpretation_limits": ["This tests score/signal density, not market crowding.", "It does not repair the v1 eligible-universe-cardinality estimand.", "It is not evidence of a market regime transition.", "No ranking, mapping, calibration or policy action is permitted."],
            "outputs": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        _write_json(stage / "manifest.json", report); (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8"); os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--panel", type=Path, default=PANEL); p.add_argument("--panel-manifest", type=Path, default=PANEL_MANIFEST); p.add_argument("--interpretation", type=Path, default=INTERPRETATION); p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv); print(json.dumps(_safe(run(panel=args.panel, panel_manifest=args.panel_manifest, interpretation=args.interpretation, output_dir=args.output_dir)), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
