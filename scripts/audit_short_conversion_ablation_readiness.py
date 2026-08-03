#!/usr/bin/env python3
"""Readiness and side attribution audit for a proposed short conversion ablation.

No ablation is fitted here.  The audit first measures chronological OOF outcome
support, then attributes the already-sealed frozen-score-density conversion
shift by side only after separately passing outcome-blind support gates.
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
DENSITY_RUNNER = ROOT / "scripts/run_febmar_true_signal_density_overlap.py"
PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
PANEL_MANIFEST = PANEL.with_name("manifest.json")
DENSITY_ARTIFACT = ROOT / "data_perp/artifacts/febmar_true_signal_density_overlap_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/short_conversion_ablation_readiness_20260730_v1"
N_BOOTSTRAP = 100
MIN_POSITIVE_ROWS = 1_000
MIN_NONPOSITIVE_ROWS = 1_000
MIN_POSITIVE_DAY_BLOCKS = 20
MIN_NONPOSITIVE_DAY_BLOCKS = 20


class ReadinessError(RuntimeError):
    pass


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ReadinessError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


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


def chronological_support(frame: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    """Counts targets for P(net>0), favourable magnitude and adverse severity."""
    work = frame.loc[frame.side_name.astype(str).str.lower().eq("short")].copy()
    work["__day__"] = pd.to_datetime(work.__ts__, utc=True).dt.date
    work["__positive__"] = pd.to_numeric(work.execution_net_ev_12h, errors="raise").gt(0)
    groups = ["fold_id", "fold_validation_start_utc", "fold_validation_end_utc"]
    rows = []
    for keys, local in work.groupby(groups, observed=True, dropna=False):
        positive = local.__positive__
        pos, neg = int(positive.sum()), int((~positive).sum())
        pos_days, neg_days = int(local.loc[positive, "__day__"].nunique()), int(local.loc[~positive, "__day__"].nunique())
        rows.append({"scope": scope, "fold_id": keys[0], "fold_validation_start_utc": keys[1], "fold_validation_end_utc": keys[2], "rows": int(len(local)), "net_positive_rows_for_probability_and_favourable_magnitude": pos, "net_nonpositive_rows_for_adverse_severity": neg, "positive_day_blocks": pos_days, "nonpositive_day_blocks": neg_days, "support_pass": bool(pos >= MIN_POSITIVE_ROWS and neg >= MIN_NONPOSITIVE_ROWS and pos_days >= MIN_POSITIVE_DAY_BLOCKS and neg_days >= MIN_NONPOSITIVE_DAY_BLOCKS)})
    return pd.DataFrame(rows)


def side_decomposition(v1: Any, common: Any, source: pd.DataFrame, target: pd.DataFrame, *, side: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    left = source.loc[source.side_name.astype(str).str.lower().eq(side)].copy()
    right = target.loc[target.side_name.astype(str).str.lower().eq(side)].copy()
    continuous = ("above_threshold_count", "above_threshold_fraction", "above_threshold_long_short_imbalance", "near_cutoff_fraction")
    categorical = ("__symbol__", "score_ventile")
    supported_left, supported_right, odds, overlap_left, overlap_right, coverage, balance = v1.fit_support(left, right, continuous=continuous, categorical=categorical)
    coverage.update({"side_name": side, "continuous_covariates": list(continuous), "categorical_covariates": list(categorical), "outcome_decomposition_status": "RUN" if coverage["common_support_pass"] else "NOT_RUN_FAILED_COMMON_SUPPORT"})
    if not coverage["common_support_pass"]:
        return coverage, balance.assign(side_name=side, common_support_pass=False), pd.DataFrame(), pd.DataFrame()
    intervals = v1._day_block_intervals(common, left, right, supported_left, supported_right, odds)
    response = pd.DataFrame(v1._decompose(common, left, right, supported_left, supported_right, odds))
    response["side_name"] = side; response["interval_method"] = f"fixed_support_fixed_weight_day_block_bootstrap_{N_BOOTSTRAP}"
    for index, row in response.iterrows():
        for field, (low, high) in intervals[row.metric].items(): response.loc[index, f"{field}_ci95_low"], response.loc[index, f"{field}_ci95_high"] = low, high
    overlap = v1._overlap_sensitivity(common, supported_left, supported_right, overlap_left, overlap_right).assign(side_name=side, sensitivity="overlap_weights_not_a_support_repair")
    return coverage, balance.assign(side_name=side, common_support_pass=True), response, overlap


def run(*, panel: Path, panel_manifest: Path, density_artifact: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    required = (panel, panel_manifest, density_artifact / "manifest.json")
    if not all(path.is_file() for path in required): raise FileNotFoundError("canonical panel or frozen-density artifact absent")
    v1 = _load(V1_RUNNER, "short_readiness_v1"); density = _load(DENSITY_RUNNER, "short_readiness_density"); common = v1._load_common(); v1.N_BOOTSTRAP = N_BOOTSTRAP
    frame, _, _, _ = common._load_canonical(panel)
    february, march = frame.loc[frame.candidate_month.astype(str).eq("2025-02")].copy(), frame.loc[frame.candidate_month.astype(str).eq("2025-03")].copy()
    frozen = density.freeze_february_score_definition(february)
    enriched, _ = density.materialize_signal_density(pd.concat([february, march], ignore_index=True), frozen)
    source = common.stable_top(enriched.loc[enriched.candidate_month.astype(str).eq("2025-02")].copy(), "base_oof_score")
    target = common.stable_top(enriched.loc[enriched.candidate_month.astype(str).eq("2025-03")].copy(), "base_oof_score")
    support = pd.concat((chronological_support(march, scope="march_all_short_oof"), chronological_support(target, scope="march_short_within_frozen_global_top10")), ignore_index=True)
    coverages: list[dict[str, Any]] = []; balances = []; responses = []; overlaps = []
    for side in ("long", "short"):
        coverage, balance, response, overlap = side_decomposition(v1, common, source, target, side=side)
        coverages.append(coverage); balances.append(balance); responses.append(response); overlaps.append(overlap)
    coverage_df = pd.DataFrame(coverages); balance_df = pd.concat(balances, ignore_index=True)
    response_df = pd.concat(responses, ignore_index=True) if any(not item.empty for item in responses) else pd.DataFrame()
    overlap_df = pd.concat(overlaps, ignore_index=True) if any(not item.empty for item in overlaps) else pd.DataFrame()
    recommendation = {
        "outcome_support_minimums_per_chronological_oof_fold": {"positive_net_rows_for_probability_and_favourable_magnitude": MIN_POSITIVE_ROWS, "nonpositive_net_rows_for_adverse_severity": MIN_NONPOSITIVE_ROWS, "positive_day_blocks": MIN_POSITIVE_DAY_BLOCKS, "nonpositive_day_blocks": MIN_NONPOSITIVE_DAY_BLOCKS},
        "predeclared_tail_weight_grid": [1.0, 1.25, 1.5, 2.0],
        "weight_definition": "w = 1 + (multiplier - 1) * I(base score is in the fixed pooled-global top 10% training tail); do not tune threshold, multiplier grid, target definitions or side after outcome inspection.",
        "confirmation_gates": ["Select at most one weight using only a predeclared inner chronological OOF criterion, never the March outer fold.", "Report every grid arm and the no-weight baseline; no winner-by-month or winner-by-component selection.", "Confirm on a never-used chronological outer fold with at least 20 positive and 20 nonpositive day blocks; require the net-EV day-block CI lower bound above zero and no material worsening in adverse severity/stop rate.", "March contains one chronological OOF fold only: it is sufficient for target support readiness but insufficient by itself for ablation/HPO selection or confirmation. Require a separate month/fold."],
    }
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True, exist_ok=False)
    try:
        outputs: dict[str, Any] = {}
        for name, table in (("short_chronological_oof_support", support), ("side_density_support", coverage_df), ("side_density_balance", balance_df), ("side_conditional_response_decomposition", response_df), ("side_overlap_weight_sensitivity", overlap_df)):
            target_path = stage / f"{name}.parquet"; table.to_parquet(target_path, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / target_path.name), "rows": int(len(table)), "sha256": sha256(target_path)}
        report = {"schema": "short_conversion_ablation_readiness_v1", "status": "DIAGNOSTIC_READINESS_ONLY_NO_ABLATION_NO_POLICY_ACTION", "promotion_eligible": False,
                  "inputs": {"canonical_panel": {"path": str(panel), "sha256": sha256(panel)}, "canonical_manifest": {"path": str(panel_manifest), "sha256": sha256(panel_manifest)}, "frozen_density_artifact": {"path": str(density_artifact / "manifest.json"), "sha256": sha256(density_artifact / "manifest.json")}},
                  "readiness_interpretation": "Target-count readiness is assessed before any short conversion fit. Side outcome decomposition is only present for separately passing, outcome-blind side support; it attributes the sealed score-signal-density diagnostic and does not select an ablation.", "recommendation": recommendation,
                  "limits": ["No short-only model or HPO is fitted.", "The single March OOF fold must not be used to choose tail weights and confirm them.", "Score-signal density is not market crowding or a regime-transition label.", "No ranking, mapping, calibration or policy change is authorised."], "outputs": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}}
        _write_json(stage / "manifest.json", report); (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8"); os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--panel", type=Path, default=PANEL); p.add_argument("--panel-manifest", type=Path, default=PANEL_MANIFEST); p.add_argument("--density-artifact", type=Path, default=DENSITY_ARTIFACT); p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv); print(json.dumps(_safe(run(panel=args.panel, panel_manifest=args.panel_manifest, density_artifact=args.density_artifact, output_dir=args.output_dir)), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
