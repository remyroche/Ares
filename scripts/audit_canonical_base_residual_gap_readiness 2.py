#!/usr/bin/env python3
"""Audit the unscored canonical base+residual calendar windows.

This runner never fills a canonical score gap with a historical, pooled, or
comparator model.  It consumes the authoritative Pack-B label inventory and
records the exact readiness of candidate/features/labels, base OOF, residual
OOF, and 12h economic evaluation per missing month.  A verified February 2025
base-only warm-up is re-materialized as a clearly non-residual input slice;
all other missing stages remain explicit blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INVENTORY = ROOT / "docs/pipeline_roadmap/20260724/r3/current_label_inventory_audit.json"
DEFAULT_LABEL_DIR = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
DEFAULT_FEBAPR_OOF = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/oof_predictions.parquet"
DEFAULT_FEBAPR_GATE = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/coverage_economics_gate.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_base_residual_gap_readiness_20260730_v1"
SCHEMA = "canonical_base_residual_gap_readiness_v1"
GAP_MONTHS = tuple(pd.period_range("2025-01", "2025-02", freq="M").astype(str)) + tuple(
    pd.period_range("2025-05", "2025-12", freq="M").astype(str)
) + tuple(pd.period_range("2026-01", "2026-04", freq="M").astype(str))
SIDES = ("long", "short")
INPUT_REQUIRED = {
    "candidate_id", "__ts__", "__decision_ts__", "side_name",
    "__first_touch_target_soft__", "__first_touch_capture_net__", "__first_touch_valid_path__",
}
FEB_BASE_REQUIRED = {
    "candidate_id", "__ts__", "__decision_ts__", "side_name", "base_oof_score", "residual_is_oof",
}
MIN_NUMERIC_FEATURES = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_inventory(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    content = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(content.get("per_file", []))
    required = {"file", "month", "side_from_filename", "rows", "expected_current_rows"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"inventory lacks columns: {missing}")
    rows = rows.loc[rows["month"].isin(GAP_MONTHS)].copy()
    if rows.empty:
        raise ValueError("inventory does not cover any requested score gaps")
    return content, rows


def _numeric_feature_count(path: Path) -> int:
    """Count raw numeric predictors without loading the source matrix."""

    result = 0
    for field in pq.read_schema(path):
        if field.name.startswith("__"):
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_boolean(field.type):
            result += 1
    return result


def _source_rows_for_month(
    inventory: pd.DataFrame,
    *,
    label_dir: Path,
    month: str,
) -> tuple[list[dict[str, Any]], set[tuple[str, pd.Timestamp, str]]]:
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, pd.Timestamp, str]] = set()
    month_rows = inventory.loc[inventory["month"].eq(month)]
    for side in SIDES:
        entry = month_rows.loc[month_rows["side_from_filename"].eq(side)]
        if len(entry) != 1:
            records.append({"month": month, "side_name": side, "stage": "candidate_feature_base_label", "status": "BLOCKED", "reason": "inventory_side_entry_missing_or_ambiguous", "rows": 0})
            continue
        item = entry.iloc[0]
        source = label_dir / str(item["file"])
        if not source.exists():
            records.append({"month": month, "side_name": side, "stage": "candidate_feature_base_label", "status": "BLOCKED", "reason": "authoritative_label_shard_missing", "rows": 0, "source_path": str(source)})
            continue
        available = set(pd.read_parquet(source).columns)
        missing = sorted(INPUT_REQUIRED.difference(available))
        if missing:
            records.append({"month": month, "side_name": side, "stage": "candidate_feature_base_label", "status": "BLOCKED", "reason": "required_canonical_columns_missing:" + ",".join(missing), "rows": 0, "source_path": str(source)})
            continue
        feature_count = _numeric_feature_count(source)
        if feature_count < MIN_NUMERIC_FEATURES:
            records.append({"month": month, "side_name": side, "stage": "candidate_feature_base_label", "status": "BLOCKED", "reason": "insufficient_numeric_causal_feature_columns", "rows": 0, "feature_columns": feature_count, "source_path": str(source)})
            continue
        frame = pd.read_parquet(source, columns=sorted(INPUT_REQUIRED)).copy()
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="coerce")
        valid = (
            frame["candidate_id"].notna()
            & frame["__ts__"].notna()
            & frame["__decision_ts__"].eq(frame["__ts__"] + pd.Timedelta(hours=1))
            & frame["side_name"].astype(str).eq(side)
            & pd.to_numeric(frame["__first_touch_valid_path__"], errors="coerce").eq(1)
        )
        duplicate_ids = int(frame["candidate_id"].duplicated().sum())
        status = "READY" if valid.all() and duplicate_ids == 0 and len(frame) == int(item["expected_current_rows"]) else "BLOCKED"
        reason = "authoritative_candidate_feature_base_label_ready" if status == "READY" else "identity_or_path_validity_or_row_count_mismatch"
        records.append({
            "month": month, "side_name": side, "stage": "candidate_feature_base_label", "status": status, "reason": reason,
            "rows": int(len(frame)), "expected_rows": int(item["expected_current_rows"]), "invalid_rows": int((~valid).sum()),
            "duplicate_candidate_ids": duplicate_ids, "source_path": str(source), "source_file": str(item["file"]),
            "feature_columns": feature_count,
        })
        if status == "READY":
            identities.update(
                (str(candidate_id), timestamp, str(side_name))
                for candidate_id, timestamp, side_name in frame.loc[:, ["candidate_id", "__ts__", "side_name"]].itertuples(index=False, name=None)
            )
    return records, identities


def _february_base_evidence(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    available = set(pd.read_parquet(path).columns)
    missing = FEB_BASE_REQUIRED.difference(available)
    if missing:
        raise ValueError(f"February OOF source lacks {sorted(missing)}")
    frame = pd.read_parquet(path, columns=sorted(FEB_BASE_REQUIRED)).copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="coerce")
    return frame.loc[frame["__ts__"].dt.strftime("%Y-%m").eq("2025-02")].copy()


def build_gap_readiness(
    *,
    inventory_path: Path,
    label_dir: Path,
    febapr_oof_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return stage ledger and a February-only re-materialisable base slice."""

    _, inventory = _load_inventory(inventory_path)
    rows: list[dict[str, Any]] = []
    feb_materialization = pd.DataFrame()
    for month in GAP_MONTHS:
        input_rows, identities = _source_rows_for_month(inventory, label_dir=label_dir, month=month)
        rows.extend(input_rows)
        ready_input_rows = sum(item["rows"] for item in input_rows if item["status"] == "READY")
        # No implicitly comparable OOF score source is accepted.  The February
        # source is the one narrow exception: its manifest explicitly calls it
        # a base passthrough warm-up, so only its base stage can be preserved.
        base_status, base_reason, base_rows = "BLOCKED", "canonical_base_oof_score_not_materialized", 0
        residual_status, residual_reason, residual_rows = "BLOCKED", "canonical_residual_oof_score_not_materialized", 0
        economics_status, economics_reason = "BLOCKED", "candidate_local_exact_12h_execution_economics_not_materialized"
        if month == "2025-02":
            feb = _february_base_evidence(febapr_oof_path)
            if not feb.empty:
                keys = set(
                    (str(candidate_id), timestamp, str(side_name))
                    for candidate_id, timestamp, side_name in feb.loc[:, ["candidate_id", "__ts__", "side_name"]].itertuples(index=False, name=None)
                )
                source_aligned = keys.issubset(identities) and len(keys) == len(feb)
                score_valid = feb["base_oof_score"].notna().all() and feb["__decision_ts__"].eq(feb["__ts__"] + pd.Timedelta(hours=1)).all()
                if source_aligned and score_valid:
                    base_status, base_reason, base_rows = "READY", "accepted_february_base_oof_warmup_only", int(len(feb))
                    feb_materialization = feb.copy()
                else:
                    base_reason = "february_base_oof_identity_or_timestamp_mismatch"
                if not feb["residual_is_oof"].astype(bool).any():
                    residual_reason = "february_is_explicit_base_passthrough_warmup_not_residual_oof"
        rows.extend([
            {"month": month, "side_name": "both", "stage": "canonical_base_oof_score", "status": base_status, "reason": base_reason, "rows": base_rows, "upstream_ready_rows": ready_input_rows},
            {"month": month, "side_name": "both", "stage": "canonical_residual_oof_score", "status": residual_status, "reason": residual_reason, "rows": residual_rows, "upstream_ready_rows": ready_input_rows},
            {"month": month, "side_name": "both", "stage": "candidate_local_exact_12h_execution_economics", "status": economics_status, "reason": economics_reason, "rows": 0, "upstream_ready_rows": ready_input_rows},
        ])
    ledger = pd.DataFrame.from_records(rows)
    return ledger, feb_materialization


def materialize_gap_readiness(
    *,
    inventory_path: Path = DEFAULT_INVENTORY,
    label_dir: Path = DEFAULT_LABEL_DIR,
    febapr_oof_path: Path = DEFAULT_FEBAPR_OOF,
    febapr_gate_path: Path = DEFAULT_FEBAPR_GATE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    inventory_path, label_dir, febapr_oof_path, febapr_gate_path, output_dir = map(Path, (inventory_path, label_dir, febapr_oof_path, febapr_gate_path, output_dir))
    if not febapr_gate_path.exists():
        raise FileNotFoundError(febapr_gate_path)
    inventory, _ = _load_inventory(inventory_path)
    ledger, february = build_gap_readiness(inventory_path=inventory_path, label_dir=label_dir, febapr_oof_path=febapr_oof_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "canonical_gap_stage_ledger.csv"
    summary_path = output_dir / "canonical_gap_month_summary.csv"
    february_path = output_dir / "february_2025_base_oof_warmup.parquet"
    ledger.to_csv(ledger_path, index=False)
    summary = (
        ledger.assign(blocked=ledger["status"].eq("BLOCKED"))
        .groupby("month", sort=True)
        .agg(stages=("stage", "size"), blocked_stages=("blocked", "sum"), ready_rows=("rows", "sum"))
        .reset_index()
    )
    summary["base_residual_scoreable"] = False
    summary.to_csv(summary_path, index=False)
    if not february.empty:
        february.to_parquet(february_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "purpose": "strict no-substitution audit of canonical base+residual historical score gaps",
        "research_only": True,
        "promotion_eligible": False,
        "gap_months": list(GAP_MONTHS),
        "no_lineage_substitution": "historical/comparator/pooled score artifacts are not accepted as canonical replacements",
        "signed_by": "codex_root_canonical_gap_audit",
        "signature_type": "detached_sha256_manifest",
        "source_contract": {
            "inventory_path": str(inventory_path),
            "inventory_sha256": _sha256(inventory_path),
            "inventory_status": inventory.get("status"),
            "label_dir": str(label_dir),
            "february_oof_path": str(febapr_oof_path),
            "february_oof_sha256": _sha256(febapr_oof_path),
            "february_gate_path": str(febapr_gate_path),
            "february_gate_sha256": _sha256(febapr_gate_path),
        },
        "counts": {
            "stage_rows": int(len(ledger)),
            "blocked_stage_rows": int(ledger["status"].eq("BLOCKED").sum()),
            "ready_stage_rows": int(ledger["status"].eq("READY").sum()),
            "february_base_only_rows": int(len(february)),
            "fully_scoreable_base_residual_months": 0,
        },
        "outputs_sha256": {
            ledger_path.name: _sha256(ledger_path),
            summary_path.name: _sha256(summary_path),
            **({february_path.name: _sha256(february_path)} if february_path.exists() else {}),
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    (output_dir / "manifest.sha256").write_text(_sha256(manifest_path) + "  manifest.json\n", encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--febapr-oof", type=Path, default=DEFAULT_FEBAPR_OOF)
    parser.add_argument("--febapr-gate", type=Path, default=DEFAULT_FEBAPR_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(json.dumps(_safe(materialize_gap_readiness(
        inventory_path=args.inventory, label_dir=args.label_dir, febapr_oof_path=args.febapr_oof,
        febapr_gate_path=args.febapr_gate, output_dir=args.output_dir,
    )), indent=2, sort_keys=True))
