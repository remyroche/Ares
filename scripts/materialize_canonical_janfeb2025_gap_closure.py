#!/usr/bin/env python3
"""Seal only canonical January--February 2025 base/residual gap stages.

The accepted canonical lineage begins with February base OOF.  This runner
therefore materializes February's already-frozen *base-only* top-40 warmup plus
its matching exact 12h deployed-policy economics, after hash and identity
verification.  It never fabricates January, a February residual score, or a
replacement score from a comparator/historical/in-sample artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = ("candidate_id", "side_name", "__symbol__", "__ts__")
SCHEMA = "canonical_janfeb2025_gap_closure_v1"
DEFAULT_READINESS = ROOT / "data_perp/artifacts/canonical_base_residual_gap_readiness_20260730_v2"
DEFAULT_BASE = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1"
DEFAULT_TOP40 = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1"
DEFAULT_POPULATION = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2"
DEFAULT_RESIDUAL = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_janfeb2025_gap_closure_20260730_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ID:
        if column not in output:
            raise ValueError(f"canonical identity column missing: {column}")
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="raise")
    output["side_name"] = output["side_name"].astype(str).str.lower()
    output["candidate_id"] = output["candidate_id"].astype(str)
    if output.duplicated(list(ID)).any() or output["candidate_id"].duplicated().any():
        raise ValueError("canonical candidate identity is not unique")
    return output


def verify_february_compatibility(*, base: pd.DataFrame, top40: pd.DataFrame, warmup: pd.DataFrame, population: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prove February warmup is exact base OOF/top40, not a score substitute."""
    base, top40, population = map(_normalise, (base, top40, population))
    # The readiness warmup intentionally carries a minimal prediction schema.
    # Restore symbol only through the exact accepted top40 identity; never infer
    # it from a candidate-id string or an outside candidate population.
    if "__symbol__" not in warmup:
        minimal = warmup.copy()
        minimal["__ts__"] = pd.to_datetime(minimal["__ts__"], utc=True, errors="raise")
        minimal["side_name"] = minimal["side_name"].astype(str).str.lower()
        minimal["candidate_id"] = minimal["candidate_id"].astype(str)
        warmup = minimal.merge(top40.loc[:, list(ID)], on=["candidate_id", "side_name", "__ts__"], how="inner", validate="one_to_one")
        if len(warmup) != len(minimal):
            raise ValueError("minimal February warmup cannot be exactly enriched from accepted top40 identity")
    warmup = _normalise(warmup)
    feb = lambda x: x.loc[x["__ts__"].dt.strftime("%Y-%m").eq("2025-02")].copy()
    base_feb, top_feb, warm_feb, population_feb = map(feb, (base, top40, warmup, population))
    if any(frame.empty for frame in (base_feb, top_feb, warm_feb, population_feb)):
        raise ValueError("February canonical source is empty")
    if not warm_feb["residual_is_oof"].eq(False).all():
        raise ValueError("warmup sidecar incorrectly claims residual OOF")
    if "base_oof_score" not in base_feb or "base_oof_score" not in top_feb or "base_oof_score" not in warm_feb:
        raise ValueError("February base score is absent")
    expected = top_feb.merge(base_feb.loc[:, [*ID, "base_oof_score"]], on=list(ID), suffixes=("__top", "__base"), how="inner", validate="one_to_one")
    if len(expected) != len(top_feb) or not np.allclose(expected["base_oof_score__top"], expected["base_oof_score__base"], rtol=0, atol=0, equal_nan=False):
        raise ValueError("top40 does not exactly match canonical base OOF scores")
    warm = warm_feb.merge(expected.loc[:, [*ID, "base_oof_score__base"]], on=list(ID), how="inner", validate="one_to_one")
    if len(warm) != len(warm_feb) or len(warm) != len(expected) or not np.allclose(warm["base_oof_score"], warm["base_oof_score__base"], rtol=0, atol=0, equal_nan=False):
        raise ValueError("February warmup is not an exact canonical top40 base-score projection")
    joined = warm_feb.merge(population_feb, on=list(ID), how="inner", validate="one_to_one", suffixes=("__warmup", ""))
    if len(joined) != len(warm_feb):
        raise ValueError("accepted exact-12h canonical economics do not cover every February warmup candidate")
    economics = ("execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "execution_label_available_at_utc")
    if any(column not in joined for column in economics):
        raise ValueError("accepted population lacks exact deployed-policy economics")
    numeric = joined.loc[:, economics[:3]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all() or not np.allclose(numeric["execution_gross_ev_12h"] - numeric["execution_cost_return"], numeric["execution_net_ev_12h"], rtol=0, atol=1e-10):
        raise ValueError("accepted execution economics are incomplete or do not reconcile")
    keep = [*ID, "__decision_ts__", "base_oof_score", "residual_is_oof", "execution_decision_utc", "execution_label_end_utc", "execution_label_available_at_utc", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "execution_exit_reason", "execution_exit_minute"]
    output = joined.loc[:, keep].copy()
    output["score_stage"] = "canonical_base_oof_top40_warmup_only"
    output["residual_stage"] = "NOT_MATERIALIZED_MONTHLY_RESIDUAL_OOF_UNSUPPORTED"
    output["economics_stage"] = "accepted_exact_1m_deployed_policy_12h"
    proof = {"base_feb_rows": int(len(base_feb)), "top40_feb_rows": int(len(top_feb)), "warmup_feb_rows": int(len(warm_feb)), "sealed_rows": int(len(output)), "all_warmup_residual_is_oof_false": True, "economics_reconciliation": "gross-cost=net", "score_is_canonical_base_oof": True, "residual_score_materialized": False}
    return output.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True), proof


def run(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    readiness_manifest = json.loads((args.readiness / "manifest.json").read_text())
    base_manifest = json.loads((args.base / "manifest.json").read_text())
    top_manifest = json.loads((args.top40 / "manifest.json").read_text())
    residual_manifest = json.loads((args.residual / "manifest.json").read_text())
    base_path, top_path = args.base / "oof_predictions.parquet", args.top40 / "population.parquet"
    if str(base_manifest.get("outputs", {}).get("oof_predictions.parquet")) != sha256_file(base_path) or str(top_manifest.get("base_oof_sha256")) != sha256_file(base_path):
        raise ValueError("base OOF hash is not accepted by both canonical manifests")
    warmup_path = args.readiness / "february_2025_base_oof_warmup.parquet"
    sealed, proof = verify_february_compatibility(base=pd.read_parquet(base_path), top40=pd.read_parquet(top_path), warmup=pd.read_parquet(warmup_path), population=pd.read_parquet(args.population / "population.parquet"))
    # The base accepted population gate is authoritative evidence that January
    # is excluded because its canonical path-input join does not exist.
    population_gate = json.loads((args.population / "population_gate.json").read_text())
    january_exclusion = next((value for value in population_gate.get("exclusions", []) if "January 2025" in value), "January canonical exact path-input exclusion evidence absent")
    ledger = pd.DataFrame([
        {"month": "2025-01", "stage": "canonical_base_oof_score", "status": "BLOCKED", "reason": "accepted canonical base OOF artifact starts in 2025-02; no January score with accepted model/feature/OOF provenance", "rows": 0},
        {"month": "2025-01", "stage": "canonical_residual_oof_score", "status": "BLOCKED", "reason": "no canonical January base top40 OOF population and no prior canonical residual support", "rows": 0},
        {"month": "2025-01", "stage": "candidate_local_exact_12h_execution_economics", "status": "BLOCKED", "reason": january_exclusion, "rows": 0},
        {"month": "2025-02", "stage": "canonical_base_oof_top40_warmup", "status": "MATERIALIZED", "reason": "exact identity and score equality to accepted February canonical base OOF/top40; base-only warmup", "rows": int(len(sealed))},
        {"month": "2025-02", "stage": "canonical_residual_oof_score", "status": "BLOCKED", "reason": "aggregate canonical residual manifest explicitly labels February base passthrough warmup; residual_is_oof=false; monthly support begins March", "rows": 0},
        {"month": "2025-02", "stage": "candidate_local_exact_12h_execution_economics", "status": "MATERIALIZED", "reason": "accepted canonical exact-policy base population, 1m deployed-policy 12h labels, exact identity and gross-cost=net reconciliation", "rows": int(len(sealed))},
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}."))
    try:
        sealed_path = temporary / "february_2025_canonical_base_only_exact12h_sidecar.parquet"; sealed.to_parquet(sealed_path, index=False)
        ledger_path = temporary / "janfeb2025_canonical_gap_closure_ledger.csv"; ledger.to_csv(ledger_path, index=False)
        blockers_path = temporary / "blocker_ledger.csv"; ledger.loc[ledger["status"].eq("BLOCKED")].to_csv(blockers_path, index=False)
        outputs = {path.name: sha256_file(path) for path in (sealed_path, ledger_path, blockers_path)}
        manifest: dict[str, Any] = {"schema": SCHEMA, "status": "PARTIAL_CANONICAL_STAGE_CLOSURE_WITH_EXPLICIT_BLOCKERS", "scope": ["2025-01", "2025-02"], "no_substitution": "No comparator, pooled, historical, direct-EV, or in-sample score was used.", "february_proof": proof, "january_blocker": january_exclusion, "residual_contract": {"source_status": residual_manifest.get("status"), "february_residual_is_oof": False, "conclusion": "February sidecar is not a base+residual stack score and must not enter canonical stack policy evaluation."}, "canonical_lineage": {"readiness_manifest_sha256": sha256_file(args.readiness / "manifest.json"), "base_manifest_sha256": sha256_file(args.base / "manifest.json"), "top40_manifest_sha256": sha256_file(args.top40 / "manifest.json"), "population_gate_sha256": sha256_file(args.population / "population_gate.json"), "base_oof_sha256": sha256_file(base_path)}, "policy": "unchanged; no score recalibration, model fit, threshold, or portfolio policy is emitted", "promotion_eligible": False, "outputs_sha256": outputs}
        manifest_path = temporary / "manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (temporary / "manifest.sha256").write_text(f"{sha256_file(manifest_path)}  manifest.json\n")
        os.replace(temporary, args.output); return args.output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--top40", type=Path, default=DEFAULT_TOP40)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(run(parse_args()))
