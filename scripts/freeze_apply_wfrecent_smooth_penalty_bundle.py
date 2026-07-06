#!/usr/bin/env python3
"""Freeze and apply wf_recent smooth rank-penalty challenger bundles.

The research replays used expanding month-by-month diagnostic references. For
prospective scoring, references must be frozen before the scoring interval.
This script creates that frozen bundle and applies it to candidate ledgers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablate_wfrecent_smooth_rank_penalty import SmoothRule, _fit_threshold, _penalty_values  # noqa: E402
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    RAW_GROUPS,
    _apply_risk_scores,
    _fit_percentile_reference,
    _fmt_table,
    _head_name,
    _json_safe,
)


DEFAULT_RULES = {
    "q85_aggressive": SmoothRule("composite_risk", "long_dist", 0.85, 0.70, 0.05, 1.0),
    "q90_conservative": SmoothRule("composite_risk", "long_dist", 0.90, 0.70, 0.05, 1.0),
}


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True).encode("utf-8")).hexdigest()


def _raw_columns() -> list[str]:
    cols: list[str] = []
    for group in RAW_GROUPS.values():
        for col, _invert in group:
            if col not in cols:
                cols.append(col)
    return cols


def _load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    frame["head"] = frame["strategy_id"].map(_head_name)
    if "portfolio_rank_adjustment" not in frame.columns:
        frame["portfolio_rank_adjustment"] = np.float32(0.0)
    else:
        frame["portfolio_rank_adjustment"] = pd.to_numeric(frame["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).astype("float32")
    return frame


def _rank_series(frame: pd.DataFrame) -> pd.Series:
    if "rank_pct" in frame.columns:
        return pd.to_numeric(frame["rank_pct"], errors="coerce").fillna(0.0)
    if "policy_rank_pct" in frame.columns:
        return pd.to_numeric(frame["policy_rank_pct"], errors="coerce").fillna(0.0)
    return pd.to_numeric(frame.get("normalized_rank_score"), errors="coerce").fillna(0.0)


def _refs_to_npz(refs: dict[str, dict[str, dict[str, np.ndarray]]], path: Path) -> dict[str, str]:
    arrays: dict[str, np.ndarray] = {}
    mapping: dict[str, str] = {}
    for head, cols in refs.items():
        for col, payload in cols.items():
            key = f"ref_{len(arrays):05d}"
            arrays[key] = np.asarray(payload.get("sorted", np.asarray([], dtype=np.float64)), dtype=np.float64)
            mapping[f"{head}::{col}"] = key
    np.savez_compressed(path, **arrays)
    return mapping


def _refs_from_npz(path: Path, mapping: dict[str, str]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    loaded = np.load(path)
    refs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for logical, key in mapping.items():
        head, col = logical.split("::", 1)
        refs.setdefault(head, {})[col] = {"sorted": np.asarray(loaded[key], dtype=np.float64)}
    return refs


def _coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in _raw_columns():
        present = col in frame.columns
        vals = pd.to_numeric(frame[col], errors="coerce") if present else pd.Series(dtype=float)
        finite = vals.replace([np.inf, -np.inf], np.nan).notna() if present else pd.Series(dtype=bool)
        rows.append(
            {
                "column": col,
                "present": bool(present),
                "finite_rate": float(finite.mean()) if present and len(finite) else 0.0,
                "missing_count": int((~finite).sum()) if present and len(finite) else int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def _freeze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(args.candidates)
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    train = candidates[candidates["timestamp"].lt(cutoff)].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError(f"No rows before cutoff {cutoff.isoformat()}")

    coverage = _coverage_table(train)
    refs = _fit_percentile_reference(train)
    scored = _apply_risk_scores(train, refs)
    refs_path = args.output_dir / "risk_percentile_refs.npz"
    ref_mapping = _refs_to_npz(refs, refs_path)

    rules_payload = {}
    thresholds = {}
    for name, rule in DEFAULT_RULES.items():
        threshold = _fit_threshold(scored, rule)
        if not np.isfinite(float(threshold)):
            raise ValueError(f"Non-finite threshold for {name}: {threshold}")
        rules_payload[name] = rule.__dict__
        thresholds[name] = float(threshold)

    manifest = {
        "generated_by": "freeze_apply_wfrecent_smooth_penalty_bundle.freeze",
        "candidate_source": str(args.candidates),
        "candidate_source_sha256": _sha256_file(args.candidates),
        "cutoff": cutoff.isoformat(),
        "train_rows": int(len(train)),
        "train_start": train["timestamp"].min().isoformat(),
        "train_end": train["timestamp"].max().isoformat(),
        "rules": rules_payload,
        "thresholds": thresholds,
        "raw_columns": _raw_columns(),
        "reference_mapping": ref_mapping,
        "reference_npz": refs_path.name,
    }
    manifest["bundle_hash"] = _sha256_json({k: v for k, v in manifest.items() if k != "bundle_hash"})
    (args.output_dir / "smooth_penalty_bundle_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    coverage.to_csv(args.output_dir / "smooth_penalty_bundle_feature_coverage.csv", index=False)
    lines = [
        "# wf_recent Smooth Penalty Frozen Bundle",
        "",
        f"Cutoff: `{cutoff.isoformat()}`",
        f"Training rows: `{len(train)}`",
        f"Bundle hash: `{manifest['bundle_hash']}`",
        "",
        "## Rules",
        "",
        pd.DataFrame(
            [
                {"variant": name, **rule, "threshold": thresholds[name]}
                for name, rule in rules_payload.items()
            ]
        ).to_markdown(index=False),
        "",
        "## Feature Coverage",
        "",
        _fmt_table(coverage, ["column", "present", "finite_rate", "missing_count"]),
    ]
    (args.output_dir / "smooth_penalty_bundle_report.md").write_text("\n".join(lines) + "\n")


def _apply(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.bundle_dir / "smooth_penalty_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    refs = _refs_from_npz(args.bundle_dir / str(manifest["reference_npz"]), dict(manifest["reference_mapping"]))
    candidates = _load_candidates(args.candidates)
    scored = _apply_risk_scores(candidates, refs)
    base_adjustment = pd.to_numeric(scored["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ranks = _rank_series(scored)

    audit_rows = []
    for name, rule_payload in manifest["rules"].items():
        rule = SmoothRule(**rule_payload)
        threshold = float(manifest["thresholds"][name])
        penalty = _penalty_values(scored, rule, threshold)
        out = scored.copy()
        out["smooth_penalty_variant"] = name
        out["smooth_penalty_score_name"] = rule.score_name
        out["smooth_penalty_threshold"] = np.float32(threshold)
        out["smooth_penalty_value"] = penalty.astype("float32")
        out["smooth_penalty_bundle_hash"] = str(manifest["bundle_hash"])
        out["portfolio_rank_adjustment"] = np.clip(base_adjustment + penalty, -1.0, 1.0).astype("float32")
        out_path = args.output_dir / f"{name}_smooth_penalty_candidates.parquet"
        out.drop(columns=["head"], errors="ignore").to_parquet(out_path, index=False)
        mask = penalty < 0.0
        audit_rows.append(
            {
                "variant": name,
                "output": str(out_path),
                "output_sha256": _sha256_file(out_path),
                "candidate_rows": int(len(out)),
                "penalized_rows": int(np.sum(mask)),
                "penalized_share": float(np.mean(mask)) if len(mask) else 0.0,
                "mean_penalty": float(np.mean(penalty[mask])) if np.any(mask) else 0.0,
                "min_penalty": float(np.min(penalty[mask])) if np.any(mask) else 0.0,
                "rank_cutoff_rows": int(ranks.ge(rule.min_rank_pct).sum()),
                "threshold": threshold,
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / "smooth_penalty_apply_audit.csv", index=False)
    apply_manifest = {
        "generated_by": "freeze_apply_wfrecent_smooth_penalty_bundle.apply",
        "bundle_dir": str(args.bundle_dir),
        "bundle_hash": manifest["bundle_hash"],
        "candidate_source": str(args.candidates),
        "candidate_source_sha256": _sha256_file(args.candidates),
        "candidate_rows": int(len(candidates)),
        "candidate_start": candidates["timestamp"].min().isoformat(),
        "candidate_end": candidates["timestamp"].max().isoformat(),
        "outputs": audit_rows,
    }
    (args.output_dir / "smooth_penalty_apply_manifest.json").write_text(json.dumps(_json_safe(apply_manifest), indent=2, sort_keys=True) + "\n")
    lines = [
        "# wf_recent Smooth Penalty Bundle Apply",
        "",
        f"Bundle: `{args.bundle_dir}`",
        f"Bundle hash: `{manifest['bundle_hash']}`",
        f"Candidate rows: `{len(candidates)}`",
        "",
        "## Output Audit",
        "",
        _fmt_table(audit, ["variant", "candidate_rows", "penalized_rows", "penalized_share", "mean_penalty", "min_penalty", "threshold", "output_sha256"]),
    ]
    (args.output_dir / "smooth_penalty_apply_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--candidates", type=Path, default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701/combo_candidates.parquet"))
    freeze.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_freeze_20260701"))
    freeze.add_argument("--cutoff", default="2026-06-27T00:00:00+00:00")
    apply = sub.add_parser("apply")
    apply.add_argument("--bundle-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_freeze_20260701"))
    apply.add_argument("--candidates", type=Path, default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701/combo_candidates.parquet"))
    apply.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_freeze_apply_smoke_20260701"))
    args = parser.parse_args()
    if args.cmd == "freeze":
        _freeze(args)
    elif args.cmd == "apply":
        _apply(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
