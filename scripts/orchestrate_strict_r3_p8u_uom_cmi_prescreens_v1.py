#!/usr/bin/env python3
"""Guarded research-only handoff from full panels to the U/O/M CMI prescreens.

This runner exists because full causal panel materialisation is intentionally
slow and resumable.  It waits for *all* source-repaired December-2024 through
July-2025 monthly panels, verifies exact Base identity coverage and a
target-free schema, then launches only the three pre-frozen family CMI
prescreens.  It never fits MC1, opens an admission/portfolio path, contacts an
exchange, or changes a live bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
MONTHS = ("2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07")
FREEZE = ROOT / "config/strict_r3_p8u_uom_cmi_input_freeze_source_repaired_expandedbase_20260829_v1.json"
SELECTOR = ROOT / "scripts/select_strict_r3_p8u_meta_fullfeatures_v1.py"
CONFIG = ROOT / "config/strict_r3_p8u_meta_target_query_grid_source_repaired_dec24_jul25_20260829_v2_expanded_base_lineage.json"
SCHEMA = "strict_r3_p8u_uom_cmi_orchestration_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _once(path: Path, payload: Any) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _event(root: Path, **payload: Any) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _load_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(FREEZE.read_text())
    if frozen.get("schema") != "strict_r3_p8u_uom_cmi_input_freeze_v1":
        raise AssertionError("unexpected CMI input-freeze schema")
    if _sha(CONFIG) != str(frozen["full_feature_selector_config_sha256"]):
        raise AssertionError("expanded-base selector config hash does not match frozen CMI plan")
    raw = json.loads(CONFIG.read_text())
    expected = {"magnitude", "over", "under"}
    if set(frozen["selected_arms"]) != expected:
        raise AssertionError("CMI plan must contain exactly Under/Over/Magnitude")
    return frozen, raw


def _panel_path(roots: tuple[Path, ...], month: str) -> Path | None:
    matches = [root / f"month={month}" / "causal_feature_universe.parquet" for root in roots]
    existing = [path for path in matches if path.exists()]
    if len(existing) > 1:
        raise AssertionError(f"{month}: multiple full-feature owners")
    return existing[0] if existing else None


def _ready_audit(raw: dict[str, Any]) -> tuple[bool, pd.DataFrame]:
    source = raw["source"]
    roots = tuple(ROOT / str(value) for value in source["full_feature_roots"])
    base_root = ROOT / str(source["base_target_free_root"])
    # Keep forbidden labels in sync with the target/query producer.  Importing
    # this tiny tuple would execute the research module; defining it here keeps
    # the scheduler lightweight and explicit.
    prohibited = {
        "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
        "policy_cost_bps", "policy_outcome_source", "label_source_complete_1m_path",
        "supportive_path_valid", "supportive_label_available_ts", "path_arch_peak_mfe_atr",
        "path_arch_atr_fraction",
    }
    rows: list[dict[str, Any]] = []
    for month in MONTHS:
        panel = _panel_path(roots, month)
        base = base_root / f"month={month}.parquet"
        row: dict[str, Any] = {"month": month, "panel": str(panel) if panel else None, "ready": False}
        if panel is None or not base.exists():
            rows.append(row)
            continue
        names = set(pq.ParquetFile(panel).schema_arrow.names)
        leaked = sorted(prohibited.intersection(names))
        feature_identity = pd.read_parquet(panel, columns=list(IDENTITY))
        base_identity = pd.read_parquet(base, columns=list(IDENTITY))
        for part in (feature_identity, base_identity):
            part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        duplicated_feature = bool(feature_identity.duplicated(list(IDENTITY)).any())
        duplicated_base = bool(base_identity.duplicated(list(IDENTITY)).any())
        matches = base_identity.merge(feature_identity, on=list(IDENTITY), how="left", indicator=True)["_merge"].eq("both")
        row.update({
            "panel_rows": int(len(feature_identity)), "base_rows": int(len(base_identity)),
            "panel_field_count": int(len(names)), "target_free": not leaked,
            "leaked_columns": leaked, "duplicate_panel_identity": duplicated_feature,
            "duplicate_base_identity": duplicated_base, "base_identity_matched": int(matches.sum()),
            "ready": bool(not leaked and not duplicated_feature and not duplicated_base and matches.all()),
        })
        rows.append(row)
    audit = pd.DataFrame(rows)
    return bool(len(audit) == len(MONTHS) and audit.ready.all()), audit


def _run(root: Path, *, poll_seconds: int) -> None:
    frozen, raw = _load_freeze()
    logs = root / "logs"; logs.mkdir()
    _once(root / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline research-only CMI handoff; no MC1/admission/portfolio/live/execution/exchange mutation",
        "freeze": str(FREEZE), "freeze_sha256": _sha(FREEZE),
        "config": str(CONFIG), "config_sha256": _sha(CONFIG),
        "selector": str(SELECTOR), "selector_sha256": _sha(SELECTOR),
        "months": list(MONTHS), "arms": frozen["selected_arms"],
        "top_base_fraction": 0.15, "selector_stage": "prescreen", "mda_requested": False,
        "causality": "wait for exact target-free monthly panels; validate Base identity before any outcome-based CMI evidence; CMI contracts are research-only",
    })
    while True:
        ready, audit = _ready_audit(raw)
        audit.to_parquet(root / "panel_readiness_audit.parquet", index=False, compression="zstd")
        _event(root, event="panel_readiness", ready=ready, ready_months=int(audit.ready.sum()))
        if ready:
            break
        time.sleep(poll_seconds)
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    for family, arm in frozen["selected_arms"].items():
        out = ROOT / f"data_perp/artifacts/strict_r3_p8u_uom_cmi_{family}_full1400_baseexp_top15_source_repaired_20260829_v1"
        if out.exists():
            raise FileExistsError(f"refusing to overwrite immutable CMI root: {out}")
        log = (logs / f"{family}.log").open("w", encoding="utf-8")
        command = [
            sys.executable, str(SELECTOR), "--config", str(CONFIG), "--out", str(out),
            "--stage", "prescreen", "--held-months", "2025-05,2025-06,2025-07",
            "--arm", str(arm), "--top-base-fraction", "0.15", "--screen-block-size", "64",
            "--veto-rows", "60000", "--n-jobs", "1",
        ]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        processes.append((family, process, log))
        _event(root, event="cmi_started", family=family, arm=arm, pid=process.pid, out=str(out))
    failed = False
    for family, process, log in processes:
        code = process.wait(); log.close()
        _event(root, event="cmi_finished", family=family, returncode=code)
        failed = failed or code != 0
    if failed:
        raise SystemExit("at least one immutable CMI prescreen failed; inspect per-family logs")
    _once(root / "completion.json", {
        "all_three_prescreens_completed": True,
        "next_gate": "independently audit CMI receipts before freezing contracts or fitting Meta heads",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    if args.poll_seconds < 15:
        raise ValueError("poll interval must be at least 15 seconds")
    args.out.mkdir(parents=True)
    _run(args.out.resolve(), poll_seconds=int(args.poll_seconds))


if __name__ == "__main__":
    main()
