#!/usr/bin/env python3
"""Fail closed on adding mapped direct-EV to the May--July four-layer panel.

The audit deliberately uses the exact all-score waterfall identities and the
same direct-q25 score source.  It can measure whether 21-day resolved-label
support would be adequate, but it must not materialise a map until the direct
score is accompanied by per-row decision-time availability/provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WATERFALL = ROOT / "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1"
DEFAULT_DIRECT = ROOT / "data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/mayjul_identical_four_layer_mapping_readiness_20260730_v1"
SCHEMA = "mayjul_identical_four_layer_mapping_readiness_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
WINDOW_DAYS = 21
MIN_REFERENCE_ROWS = 500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _bound(root: Path, output: str) -> tuple[Path, dict[str, Any], Path]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = manifest.get("outputs", {}).get(output, {})
    path = Path(str(item.get("path", "")))
    if not path.is_absolute():
        # Artifact manifests conventionally express paths from repository root,
        # while a few older manifests use their artifact-local filename.
        path = ROOT / path if (ROOT / path).is_file() else root / path
    if not path.is_file() or item.get("sha256") != sha256(path):
        raise ValueError(f"{output} is not hash-bound in {root}")
    return path, manifest, manifest_path


def _normalise(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks identity: {missing}")
    work = frame.copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["side_name"] = work["side_name"].astype(str).str.lower()
    work["__symbol__"] = work["__symbol__"].astype(str).str.replace("_", "/", regex=False)
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    if work.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{name} has duplicate exact IDs")
    return work


def causal_support(frame: pd.DataFrame) -> pd.DataFrame:
    """Audit strictly earlier resolved 21-day support without fitting a map."""

    work = frame.copy()
    rows: list[dict[str, Any]] = []
    for snapshot, group in work.groupby(work["execution_decision_utc"].dt.floor("D"), sort=True):
        lower = snapshot - pd.Timedelta(days=WINDOW_DAYS)
        reference = work.loc[
            work["execution_label_end_utc"].lt(snapshot)
            & work["execution_label_end_utc"].ge(lower)
        ]
        counts = reference.groupby("side_name", observed=True).size()
        rows.append({
            "snapshot_utc": snapshot, "reference_window_start_utc": lower,
            "reference_window_end_utc": snapshot, "reference_rows": int(len(reference)),
            "long_reference_rows": int(counts.get("long", 0)), "short_reference_rows": int(counts.get("short", 0)),
            "reference_label_end_max_utc": reference["execution_label_end_utc"].max() if len(reference) else pd.NaT,
            "strictly_resolved_before_snapshot": bool(reference["execution_label_end_utc"].lt(snapshot).all()),
            "map_support_available": bool(len(reference) >= MIN_REFERENCE_ROWS),
            "current_candidate_rows": int(len(group)),
        })
    return pd.DataFrame(rows)


def audit(waterfall_root: Path, direct_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    waterfall_path, waterfall_manifest, waterfall_manifest_path = _bound(waterfall_root, "allscore_waterfall")
    direct_path, direct_manifest, direct_manifest_path = _bound(direct_root, "historical_oof_winner")
    waterfall = _normalise(pd.read_parquet(waterfall_path), name="waterfall")
    direct = _normalise(pd.read_parquet(direct_path), name="direct OOF winner")
    required = {"execution_decision_utc", "execution_label_end_utc", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "score_direct_q25_challenger_bps"}
    if missing := sorted(required.difference(waterfall.columns)):
        raise ValueError(f"waterfall lacks exact direct/label fields: {missing}")
    if "q25_net_bps" not in direct:
        raise ValueError("direct OOF source lacks q25 direct score")
    for column in ("execution_decision_utc", "execution_label_end_utc"):
        waterfall[column] = pd.to_datetime(waterfall[column], utc=True, errors="raise")
    if not waterfall["execution_decision_utc"].eq(waterfall["__ts__"] + pd.Timedelta(hours=1)).all() or not waterfall["execution_label_end_utc"].eq(waterfall["execution_decision_utc"] + pd.Timedelta(hours=12)).all():
        raise ValueError("waterfall timing is not exact signal+1h/H12")
    joined = waterfall.merge(direct.loc[:, [*IDENTITY, "q25_net_bps"]], on=list(IDENTITY), how="left", validate="one_to_one")
    coverage = joined.assign(
        candidate_month=joined["__ts__"].dt.strftime("%Y-%m"),
        direct_present=joined["q25_net_bps"].notna(),
    ).groupby(["candidate_month", "side_name", "direct_present"], observed=True).size().rename("rows").reset_index()
    exact_score_values = np.isclose(
        joined.loc[joined["q25_net_bps"].notna(), "score_direct_q25_challenger_bps"].to_numpy(float),
        joined.loc[joined["q25_net_bps"].notna(), "q25_net_bps"].to_numpy(float),
        rtol=0.0, atol=0.0,
    ).all()
    support = causal_support(waterfall)
    direct_time_fields = [column for column in direct.columns if "available" in column.lower() or "decision" in column.lower() or "oof" in column.lower() or "fold" in column.lower()]
    has_score_availability = any("available" in column.lower() for column in direct_time_fields)
    has_row_oof_lineage = any("oof" in column.lower() or "fold" in column.lower() for column in direct_time_fields)
    requirements = pd.DataFrame([
        {"requirement": "exact all-score candidate IDs and direct q25 score identity", "available": bool(joined["q25_net_bps"].notna().all() and exact_score_values), "evidence": "all waterfall rows join one-to-one and q25 values are bit-identical", "missing_contract": None},
        {"requirement": "exact H12 economics and strictly earlier resolved 21-day reference labels", "available": bool(support["strictly_resolved_before_snapshot"].all()), "evidence": "waterfall H12 label end is decision+12h; daily reference audit uses label_end < snapshot", "missing_contract": None},
        {"requirement": "score availability at/before decision", "available": bool(has_score_availability), "evidence": ",".join(direct_time_fields) if direct_time_fields else "no score-availability field", "missing_contract": "per-candidate score_available_at <= execution_decision_utc"},
        {"requirement": "per-candidate frozen OOF/model lineage for the direct score", "available": bool(has_row_oof_lineage), "evidence": ",".join(direct_time_fields) if direct_time_fields else "no fold/provenance field", "missing_contract": "model/fold/fit-cutoff provenance bound to each q25 score"},
    ])
    bounds = {
        "waterfall": {"path": str(waterfall_path), "sha256": sha256(waterfall_path), "manifest": str(waterfall_manifest_path), "manifest_sha256": sha256(waterfall_manifest_path), "rows": int(len(waterfall))},
        "direct": {"path": str(direct_path), "sha256": sha256(direct_path), "manifest": str(direct_manifest_path), "manifest_sha256": sha256(direct_manifest_path), "rows": int(len(direct))},
        "exact_identity_rows": int(len(joined)), "exact_direct_score_rows": int(joined["q25_net_bps"].notna().sum()),
        "direct_score_bit_identical_to_waterfall": bool(exact_score_values),
        "allscore_score_column": "score_direct_q25_challenger_bps", "source_score_column": "q25_net_bps",
        "minimum_reference_rows": MIN_REFERENCE_ROWS, "window_days": WINDOW_DAYS,
        "map_support_candidate_rows": int(support.loc[support.map_support_available, "current_candidate_rows"].sum()),
        "map_warmup_candidate_rows": int(support.loc[~support.map_support_available, "current_candidate_rows"].sum()),
    }
    return requirements, coverage, {"bounds": bounds, "support": support}


def run(*, waterfall: Path, direct: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    readiness, coverage, extra = audit(waterfall, direct)
    support = extra.pop("support")
    legal = bool(readiness.available.all())
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        outputs: dict[str, dict[str, Any]] = {}
        for name, table in (("readiness", readiness), ("identity_side_coverage", coverage), ("causal_support_audit", support)):
            path = stage / f"{name}.parquet"
            table.to_parquet(path, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / path.name), "rows": int(len(table)), "sha256": sha256(path)}
        manifest = {
            "schema": SCHEMA,
            "status": "READY_FOR_MATERIALIZATION" if legal else "FAIL_CLOSED_MISSING_DIRECT_SCORE_AVAILABILITY_AND_LINEAGE",
            "promotion_eligible": False, "materialization_legal": legal,
            "reason": "No causal map or four-layer rerun was made: exact direct-score identity and labels are available, but the source lacks per-score availability and row-level frozen OOF/model lineage.",
            "minimal_materialization_request": "Publish a hash-bound successor to historical_oof_winner on the same 127777 candidate IDs that retains q25_net_bps unchanged and adds score_available_at, model/fold/fit-cutoff lineage, and an assertion score_available_at <= execution_decision_utc. Then fit only a causal 21-day map against earlier resolved exact H12 waterfall labels (retain warm-up), with pooled global selection and candidate-id ties; do not refit/re-rank the direct model.",
            "contracts": {"identity": list(IDENTITY), "score": "score_direct_q25_challenger_bps / q25_net_bps", "reference_rule": "execution_label_end_utc < UTC day snapshot and >= snapshot-21d", "minimum_reference_rows": MIN_REFERENCE_ROWS, "selection": "one pooled global book; no side/timestamp/asset quotas", "outcome_guard": "labels are map training/outcome-only and unavailable to candidates until their H12 resolution"},
            "bounds": extra["bounds"], "outputs": outputs,
            "outputs_sha256": {f"{name}.parquet": item["sha256"] for name, item in outputs.items()},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waterfall", type=Path, default=DEFAULT_WATERFALL)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(_safe(run(waterfall=args.waterfall, direct=args.direct, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
