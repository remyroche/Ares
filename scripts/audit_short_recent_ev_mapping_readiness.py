#!/usr/bin/env python3
"""Audit causal recent-EV mapping readiness for a future frozen short score.

This creates no map.  It measures only labels that were resolved before each
daily March snapshot, and separates the available canonical-base proxy from the
as-yet-unmaterialized short-conversion score required for an honest evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
PANEL_MANIFEST = PANEL.with_name("manifest.json")
SHORT_READINESS = ROOT / "data_perp/artifacts/short_conversion_ablation_readiness_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/short_recent_ev_mapping_readiness_20260730_v1"
WINDOW_DAYS = 21
GRID = (
    {"name": "light", "minimum_reference_rows": 1000, "minimum_short_rows": 500, "short_shrinkage": 250.0},
    {"name": "standard", "minimum_reference_rows": 2000, "minimum_short_rows": 1000, "short_shrinkage": 500.0},
    {"name": "conservative", "minimum_reference_rows": 5000, "minimum_short_rows": 2000, "short_shrinkage": 1000.0},
)


class MappingReadinessError(RuntimeError):
    pass


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


def daily_reference_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Strictly prior resolved labels for every March UTC-day evaluation batch."""
    required = {"candidate_id", "candidate_month", "side_name", "__ts__", "execution_label_end_utc", "execution_net_ev_12h", "base_oof_score", "fold_id"}
    missing = required.difference(frame.columns)
    if missing: raise MappingReadinessError(f"panel lacks required columns: {sorted(missing)}")
    work = frame.copy(); work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise"); work["execution_label_end_utc"] = pd.to_datetime(work["execution_label_end_utc"], utc=True, errors="raise")
    if work.candidate_id.astype(str).duplicated().any(): raise MappingReadinessError("candidate identities are not unique")
    march = work.loc[work.candidate_month.astype(str).eq("2025-03")].copy()
    rows = []
    for snapshot, batch in march.groupby(march.__ts__.dt.floor("D"), observed=True, sort=True):
        reference = work.loc[work.execution_label_end_utc.lt(snapshot) & work.execution_label_end_utc.ge(snapshot - pd.Timedelta(days=WINDOW_DAYS))].copy()
        reference_short = reference.loc[reference.side_name.astype(str).str.lower().eq("short")]
        overlap = len(set(batch.candidate_id.astype(str)).intersection(reference.candidate_id.astype(str)))
        rows.append({"snapshot_utc": snapshot, "window_start_utc": snapshot - pd.Timedelta(days=WINDOW_DAYS), "evaluation_rows": int(len(batch)), "evaluation_short_rows": int(batch.side_name.astype(str).str.lower().eq("short").sum()), "reference_rows": int(len(reference)), "reference_short_rows": int(len(reference_short)), "reference_label_end_max_utc": reference.execution_label_end_utc.max() if len(reference) else pd.NaT, "strict_label_end_before_snapshot": bool(reference.execution_label_end_utc.lt(snapshot).all()), "evaluation_reference_identity_overlap": int(overlap), "evaluation_score_available_base_proxy": int(pd.to_numeric(batch.base_oof_score, errors="coerce").notna().sum()), "evaluation_label_available": int(pd.to_numeric(batch.execution_net_ev_12h, errors="coerce").notna().sum()), "evaluation_fold_ids": "|".join(sorted(batch.fold_id.astype(str).unique()))})
    return pd.DataFrame(rows)


def grid_feasibility(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config in GRID:
        work = audit.copy()
        work["grid_name"] = config["name"]
        work["minimum_reference_rows"] = config["minimum_reference_rows"]
        work["minimum_short_rows"] = config["minimum_short_rows"]
        work["short_shrinkage"] = config["short_shrinkage"]
        work["pooled_support_pass"] = work.reference_rows.ge(config["minimum_reference_rows"])
        work["short_support_pass"] = work.reference_short_rows.ge(config["minimum_short_rows"])
        work["snapshot_legal"] = work.strict_label_end_before_snapshot & work.evaluation_reference_identity_overlap.eq(0)
        work["snapshot_mapping_ready_proxy"] = work.pooled_support_pass & work.short_support_pass & work.snapshot_legal
        work["short_weight_if_mapped"] = work.reference_short_rows / (work.reference_short_rows + float(config["short_shrinkage"]))
        rows.append(work)
    return pd.concat(rows, ignore_index=True)


def score_contract_readiness(frame: pd.DataFrame) -> pd.DataFrame:
    march = frame.loc[frame.candidate_month.astype(str).eq("2025-03")]
    return pd.DataFrame([
        {"score_contract": "canonical_base_oof_proxy", "status": "AVAILABLE_PROXY_ONLY", "rows": int(len(march)), "finite_score_rows": int(pd.to_numeric(march.base_oof_score, errors="coerce").notna().sum()), "reason": "Canonical base OOF score is present, but it is not the proposed frozen short-conversion score."},
        {"score_contract": "frozen_short_conversion_oof_score", "status": "NOT_MATERIALIZED_BLOCKS_MAPPING_EVALUATION", "rows": int(len(march.loc[march.side_name.astype(str).str.lower().eq("short")])), "finite_score_rows": 0, "reason": "No immutable score column/artifact declares this proposed score, its outer-OOF fold lineage, train cutoff, or candidate-level availability. Mapping may not be implemented/evaluated until supplied."},
    ])


def run(*, panel: Path, panel_manifest: Path, short_readiness: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    required = (panel, panel_manifest, short_readiness / "manifest.json")
    if not all(path.is_file() for path in required): raise FileNotFoundError("canonical panel or short readiness seal is absent")
    frame = pd.read_parquet(panel, columns=["candidate_id", "candidate_month", "side_name", "__ts__", "execution_label_end_utc", "execution_net_ev_12h", "base_oof_score", "fold_id"])
    audit = daily_reference_audit(frame); grid = grid_feasibility(audit); contracts = score_contract_readiness(frame)
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True, exist_ok=False)
    try:
        outputs: dict[str, Any] = {}
        for name, table in (("score_contract_readiness", contracts), ("march_daily_label_reference_audit", audit), ("admissible_grid_feasibility", grid)):
            target = stage / f"{name}.parquet"; table.to_parquet(target, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / target.name), "rows": int(len(table)), "sha256": sha256(target)}
        report = {"schema": "short_recent_ev_mapping_readiness_v1", "status": "READINESS_ONLY_SCORE_NOT_MATERIALIZED_NO_MAPPING", "promotion_eligible": False,
                  "inputs": {"canonical_panel": {"path": str(panel), "sha256": sha256(panel)}, "canonical_manifest": {"path": str(panel_manifest), "sha256": sha256(panel_manifest)}, "short_conversion_readiness": {"path": str(short_readiness / "manifest.json"), "sha256": sha256(short_readiness / "manifest.json")}},
                  "causal_rule": "For UTC-day snapshot S, reference labels must satisfy S-21d <= execution_label_end_utc < S. Evaluation candidates for S are forbidden from the reference set; reference/evaluation identity intersection must be zero.",
                  "score_boundary": "The canonical base OOF score proves label-reference geometry only. The proposed frozen short-conversion OOF score is absent, so no map can be fitted, scored, evaluated or promoted from this audit.",
                  "admissible_grid": list(GRID),
                  "gate_recommendations": ["Materialize a unique candidate-level frozen short score with is_outer_oof=true, fold ID, validation range, training-label cutoff and score availability timestamp.", "Require the candidate's mapping snapshot strictly after all reference label ends and forbid any reference/evaluation identity overlap.", "Use one predeclared grid member only; do not select shrinkage/minimum support using March economics.", "For a later evaluation, score every map-eligible candidate once; never evaluate the map on any row used as its fit reference.", "Treat snapshots failing any support/legal gate as unmapped warm-up, not as global fallback or zero EV."],
                  "outputs": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}}
        _write_json(stage / "manifest.json", report); (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8"); os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--panel", type=Path, default=PANEL); p.add_argument("--panel-manifest", type=Path, default=PANEL_MANIFEST); p.add_argument("--short-readiness", type=Path, default=SHORT_READINESS); p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv); print(json.dumps(_safe(run(panel=args.panel, panel_manifest=args.panel_manifest, short_readiness=args.short_readiness, output_dir=args.output_dir)), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
