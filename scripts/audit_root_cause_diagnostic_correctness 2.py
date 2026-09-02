#!/usr/bin/env python3
"""Fail-closed correctness audit for the pinned root-cause Stage 0/1 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ART = Path("data_perp/artifacts")
ROOT = Path(__file__).resolve().parents[1]
POINTER = ART / "root_cause_diagnostic_canonical_20260731.json"
DEFAULT_OUTPUT = ART / "root_cause_diagnostic_correctness_20260731_v5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(name: str, passed: bool, detail: str, observed: Any = None) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail, "observed": observed}


def run(pointer_path: Path = POINTER, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    pointer = json.loads(pointer_path.read_text())
    stage0 = Path(pointer["stage0_substrate"])
    stage1 = Path(pointer["stage1_oracle_ladder"])
    stage0_manifest = json.loads((stage0 / "diagnostic_population_manifest.json").read_text())
    stage1_manifest = json.loads((stage1 / "run_manifest.json").read_text())
    ledger = pd.read_parquet(stage0 / "diagnostic_row_ledger.parquet")
    results = pd.read_parquet(stage1 / "oracle_ladder_results.parquet")
    sensitivity = pd.read_parquet(stage1 / "target_sensitivity_results.parquet")

    expected_scores = {"score_base_alpha", "score_residual_alpha", "score_residual_delta_alpha", "score_base_expected_ev", "score_residual_expected_ev"}
    expected_sensitivities = {"entry_delay_0m", "entry_delay_1m", "entry_delay_5m", "entry_delay_10m", "path_resolution_1m", "path_resolution_5m", "path_resolution_15m", "small_barrier_perturbation", "timeout_perturbation", "entry_price_perturbation"}
    stage0_runner = stage0_manifest.get("runner", {})
    stage1_runner = stage1_manifest.get("runner", {})
    checks = [
        _check("pointer_stage0_digest", _sha256(stage0 / "diagnostic_population_manifest.json") == pointer["stage0_manifest_sha256"], "canonical pointer pins the exact Stage 0 manifest"),
        _check("pointer_stage1_digest", _sha256(stage1 / "run_manifest.json") == pointer["stage1_manifest_sha256"], "canonical pointer pins the exact Stage 1 manifest"),
        _check("stage0_runner_digest", bool(stage0_runner.get("path")) and _sha256(ROOT / stage0_runner["path"]) == stage0_runner.get("sha256"), "Stage 0 manifest records the exact current materializer source"),
        _check("stage1_runner_digest", bool(stage1_runner.get("path")) and _sha256(ROOT / stage1_runner["path"]) == stage1_runner.get("sha256"), "Stage 1 manifest records the exact current oracle-runner source"),
        _check("stage1_binds_stage0_ledger", stage1_manifest["ledger_sha256"] == stage0_manifest["outputs_sha256"]["diagnostic_row_ledger.parquet"], "oracle runner uses exact materialised ledger"),
        _check("candidate_identity", not ledger.candidate_id.duplicated().any() and len(ledger) == stage0_manifest["rows"], "one row per candidate and manifest row count match", int(len(ledger))),
        _check("causal_timing", ledger.feature_cutoff_ts.le(ledger.decision_ts).all() and ledger.score_ts.eq(ledger.feature_cutoff_ts).all() and ledger.label_available_ts.eq(ledger.label_end_ts).all(), "feature/score cutoff precedes decision; label is available only at H12"),
        _check("h12_horizon", ledger.label_end_ts.eq(ledger.decision_ts + pd.Timedelta(hours=12)).all(), "every target is exact H12"),
        _check("gross_net_reconciliation", bool(np.allclose(ledger.net_reconciliation_bps.to_numpy(float), 0.0, atol=1e-6, rtol=0.0)), "execution-adjusted gross minus fee equals net", float(np.abs(ledger.net_reconciliation_bps).max())),
        _check("no_future_cost_feature", not any("known_row_cost" in column or "delta_continue_bps" == column for column in ledger.columns), "known realised-cost proxy and sealed legacy delta are absent"),
        _check("oof_score_lineage", ledger.residual_is_oof.astype(bool).all() and expected_scores.issubset(ledger.columns), "all frozen score fields are exact-ID OOF and complete"),
        _check("single_product_family", ledger.contract_family.eq("PF_USD_LINEAR_PERPETUAL").all() and ledger.settlement_currency.eq("USD").all() and ledger.symbol.astype(str).str.fullmatch(r"[A-Z0-9]+/USD:USD").all(), "no inverse or mixed settlement products pooled"),
        _check("global_topk_only", results.selection_scope.eq("GLOBAL_TOP_K").all(), "all oracle selections are global, never timestamp-local"),
        _check("stage1_top_fractions", set(results.top_fraction.unique()) == {0.01, 0.05, 0.10, 0.20}, "all required global top-k fractions are materialised"),
        _check("o4_net_not_fabricated", results.loc[results.oracle.eq("O4_hindsight_permitted_action"), "net_status"].str.startswith("NOT_AVAILABLE").all(), "policy oracle does not fabricate a causal post-action net return"),
        _check("sensitivity_completeness", expected_sensitivities.issubset(set(sensitivity.sensitivity)), "all requested sensitivity cells exist; unavailable cells are explicit"),
    ]
    check_frame = pd.DataFrame(checks)
    status = "PASS" if check_frame.passed.all() else "FAIL"
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    check_frame.to_parquet(staging / "correctness_checks.parquet", index=False)
    report = {"schema": "root_cause_diagnostic_correctness_v1", "status": status, "passed": int(check_frame.passed.sum()), "total": int(len(check_frame)), "pointer": str(pointer_path), "checks": checks}
    (staging / "correctness_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (staging / "manifest.sha256").write_text(_sha256(staging / "correctness_report.json") + "\n")
    staging.rename(output)
    if status != "PASS":
        raise RuntimeError("root-cause diagnostic correctness audit failed; inspect emitted report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer", type=Path, default=POINTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.pointer, args.output), indent=2, default=str))


if __name__ == "__main__":
    main()
