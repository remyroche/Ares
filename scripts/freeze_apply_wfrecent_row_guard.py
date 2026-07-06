#!/usr/bin/env python3
"""Freeze and apply a wf_recent row-level diagnostic guard.

The leading validated row guard is:

    all__recent_perf_risk__q90__rank70

This utility materializes it as a frozen transform so it can be used in
prospective dual scoring.  Freeze fits percentile references and the guard
threshold on rows before a cutoff.  Apply scores any compatible candidate
ledger and writes a normal parquet with `portfolio_rank_adjustment` updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_apply_wfrecent_smooth_penalty_bundle import (  # noqa: E402
    _coverage_table,
    _refs_from_npz,
    _refs_to_npz,
    _sha256_file,
    _sha256_json,
)
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    VetoRule,
    _apply_risk_scores,
    _fit_percentile_reference,
    _fit_rule_thresholds,
    _fmt_table,
    _head_name,
    _json_safe,
    _rule_mask,
)


DEFAULT_RULE = VetoRule("recent_perf_risk", "all", 0.90, 0.70)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted({"timestamp", "strategy_id", "symbol"} - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out[out["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["head"] = out["strategy_id"].map(_head_name)
    if "portfolio_rank_adjustment" not in out.columns:
        out["portfolio_rank_adjustment"] = np.float32(0.0)
    else:
        out["portfolio_rank_adjustment"] = (
            pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).astype("float32")
        )
    return out


def _rule_from_args(args: argparse.Namespace) -> VetoRule:
    return VetoRule(
        str(args.score_name),
        str(args.scope),
        float(args.risk_quantile),
        float(args.min_rank_pct),
    )


def _guard_audit(scored: pd.DataFrame, rule: VetoRule, threshold: float) -> dict[str, Any]:
    mask = _rule_mask(scored, rule, threshold)
    score = pd.to_numeric(scored[rule.score_name], errors="coerce")
    ranks = pd.to_numeric(scored.get("rank_pct", scored.get("policy_rank_pct")), errors="coerce")
    return {
        "candidate_rows": int(len(scored)),
        "guarded_rows": int(mask.sum()),
        "guarded_share": float(mask.mean()) if len(mask) else 0.0,
        "threshold": float(threshold),
        "score_name": rule.score_name,
        "scope": rule.scope,
        "risk_quantile": float(rule.risk_quantile),
        "min_rank_pct": float(rule.min_rank_pct),
        "score_finite_rate": float(np.isfinite(score.to_numpy(dtype=float, copy=False)).mean()) if len(score) else 0.0,
        "score_p50": float(score.quantile(0.50)) if score.notna().any() else None,
        "score_p90": float(score.quantile(0.90)) if score.notna().any() else None,
        "score_p95": float(score.quantile(0.95)) if score.notna().any() else None,
        "rank_finite_rate": float(np.isfinite(ranks.to_numpy(dtype=float, copy=False)).mean()) if len(ranks) else 0.0,
    }


def _freeze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(args.candidates)
    cutoff = _utc_timestamp(str(args.cutoff))
    train = candidates[candidates["timestamp"].lt(cutoff)].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError(f"No rows before cutoff {cutoff.isoformat()}")
    rule = _rule_from_args(args)
    refs = _fit_percentile_reference(train)
    scored = _apply_risk_scores(train, refs)
    thresholds = _fit_rule_thresholds(scored, [rule])
    threshold = float(thresholds[rule])
    if not np.isfinite(threshold):
        raise ValueError(f"Non-finite threshold for {rule}: {threshold}")

    refs_path = args.output_dir / "row_guard_risk_percentile_refs.npz"
    ref_mapping = _refs_to_npz(refs, refs_path)
    coverage = _coverage_table(train)
    audit = _guard_audit(scored, rule, threshold)
    manifest = {
        "generated_by": "freeze_apply_wfrecent_row_guard.freeze",
        "candidate_source": str(args.candidates),
        "candidate_source_sha256": _sha256_file(args.candidates),
        "cutoff": cutoff.isoformat(),
        "train_rows": int(len(train)),
        "train_start": train["timestamp"].min().isoformat(),
        "train_end": train["timestamp"].max().isoformat(),
        "rule": asdict(rule),
        "threshold": threshold,
        "reference_mapping": ref_mapping,
        "reference_npz": refs_path.name,
        "audit": audit,
    }
    manifest["bundle_hash"] = _sha256_json({k: v for k, v in manifest.items() if k != "bundle_hash"})
    (args.output_dir / "row_guard_bundle_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    coverage.to_csv(args.output_dir / "row_guard_feature_coverage.csv", index=False)
    pd.DataFrame([audit]).to_csv(args.output_dir / "row_guard_freeze_audit.csv", index=False)
    lines = [
        "# wf_recent Row Guard Frozen Bundle",
        "",
        f"Cutoff: `{cutoff.isoformat()}`",
        f"Training rows: `{len(train)}`",
        f"Bundle hash: `{manifest['bundle_hash']}`",
        "",
        "## Rule",
        "",
        pd.DataFrame([{**asdict(rule), "threshold": threshold}]).to_markdown(index=False),
        "",
        "## Freeze Audit",
        "",
        pd.DataFrame([audit]).to_markdown(index=False),
        "",
        "## Feature Coverage",
        "",
        _fmt_table(coverage, ["column", "present", "finite_rate", "missing_count"]),
    ]
    (args.output_dir / "row_guard_bundle_report.md").write_text("\n".join(lines) + "\n")


def _apply(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bundle_dir / "row_guard_bundle_manifest.json").read_text())
    refs = _refs_from_npz(args.bundle_dir / str(manifest["reference_npz"]), dict(manifest["reference_mapping"]))
    rule_payload = dict(manifest["rule"])
    rule = VetoRule(
        str(rule_payload["score_name"]),
        str(rule_payload["scope"]),
        float(rule_payload["risk_quantile"]),
        float(rule_payload["min_rank_pct"]),
    )
    threshold = float(manifest["threshold"])
    candidates = _load_candidates(args.candidates)
    scored = _apply_risk_scores(candidates, refs)
    mask = _rule_mask(scored, rule, threshold)
    base_adjustment = pd.to_numeric(scored["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out = scored.copy()
    out["row_guard_variant"] = f"{rule.scope}__{rule.score_name}__q{int(rule.risk_quantile * 100)}__rank{int(rule.min_rank_pct * 100)}"
    out["row_guard_bundle_hash"] = str(manifest["bundle_hash"])
    out["row_guard_score_name"] = rule.score_name
    out["row_guard_score"] = pd.to_numeric(out[rule.score_name], errors="coerce").astype("float32")
    out["row_guard_threshold"] = np.float32(threshold)
    out["row_guard_triggered"] = mask.astype("bool")
    out["portfolio_rank_adjustment"] = base_adjustment
    out.loc[mask, "portfolio_rank_adjustment"] = np.float32(-1.0)

    label = str(args.label).strip() or "recent_perf_row_guard_q90_rank70"
    out_path = args.output_dir / f"{label}_candidates.parquet"
    out.drop(columns=["head"], errors="ignore").to_parquet(out_path, index=False)
    audit = _guard_audit(scored, rule, threshold)
    audit.update(
        {
            "label": label,
            "output": str(out_path),
            "output_sha256": _sha256_file(out_path),
            "candidate_source": str(args.candidates),
            "candidate_source_sha256": _sha256_file(args.candidates),
            "bundle_dir": str(args.bundle_dir),
            "bundle_hash": manifest["bundle_hash"],
            "candidate_start": candidates["timestamp"].min().isoformat() if len(candidates) else "",
            "candidate_end": candidates["timestamp"].max().isoformat() if len(candidates) else "",
        }
    )
    pd.DataFrame([audit]).to_csv(args.output_dir / "row_guard_apply_audit.csv", index=False)
    apply_manifest = {
        "generated_by": "freeze_apply_wfrecent_row_guard.apply",
        **audit,
    }
    (args.output_dir / "row_guard_apply_manifest.json").write_text(
        json.dumps(_json_safe(apply_manifest), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# wf_recent Row Guard Apply",
        "",
        f"Bundle: `{args.bundle_dir}`",
        f"Bundle hash: `{manifest['bundle_hash']}`",
        f"Output: `{out_path}`",
        "",
        "## Apply Audit",
        "",
        pd.DataFrame([audit]).to_markdown(index=False),
    ]
    (args.output_dir / "row_guard_apply_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--candidates", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--cutoff", default="2026-06-27T13:00:00+00:00")
    freeze.add_argument("--score-name", default=DEFAULT_RULE.score_name)
    freeze.add_argument("--scope", default=DEFAULT_RULE.scope)
    freeze.add_argument("--risk-quantile", type=float, default=DEFAULT_RULE.risk_quantile)
    freeze.add_argument("--min-rank-pct", type=float, default=DEFAULT_RULE.min_rank_pct)

    apply = sub.add_parser("apply")
    apply.add_argument("--bundle-dir", type=Path, required=True)
    apply.add_argument("--candidates", type=Path, required=True)
    apply.add_argument("--output-dir", type=Path, required=True)
    apply.add_argument("--label", default="recent_perf_row_guard_q90_rank70")

    args = parser.parse_args()
    if args.cmd == "freeze":
        _freeze(args)
    elif args.cmd == "apply":
        _apply(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
