#!/usr/bin/env python3
"""Finish the fully phase-native Strict-R3 four-decision research replay.

This is an offline, immutable orchestrator.  It is deliberately blocked until
the three independently materialised phase feature streams have completed.
It validates coverage before any model is scored, builds score panels without
outcomes, and attaches the frozen policy labels only at the MC1 stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PHASES = (15, 30, 45)
STREAM_TAG = "v5_nativeledger"
PHASE0_ROOT = ROOT / "data_perp/artifacts/strict_r3_full_stack_phase_h1_mayjul_20260818_v2"
PHASE0_CURRENT = PHASE0_ROOT / "mc1_phase00_current_prequential_v1/predictions_current_v5_mc1_d2.parquet"
PHASE0_BCF = PHASE0_ROOT / "mc1_phase00_bcf_prequential_v1/predictions_bcf_mc1_d2.parquet"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    print(json.dumps({"event": "start", "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _coverage(root: Path) -> Path:
    contract = json.loads((ROOT / "config/strict_r3_canonical_v2_feature_contract.json").read_text())
    fields = list(contract["base_fields_by_side"]["long"])
    rows: list[dict[str, object]] = []
    for phase in PHASES:
        stream = root / f"phase{phase}_streamed_{STREAM_TAG}"
        if not (stream / "checkpoint.json").is_file() or not (root / f"phase{phase}_features_complete").is_file():
            raise RuntimeError(f"phase={phase} feature stream is incomplete")
        pieces = sorted((stream / "replay_feature_shards").glob("block_*/canonical120_features.parquet"))
        if not pieces:
            raise RuntimeError(f"phase={phase} has no retained replay feature shards")
        # Inspect the physical schemas before projecting the frozen base contract.
        # Projected reads alone cannot prove that a feature shard contains no target
        # or realized-outcome fields.
        forbidden_by_piece = {
            str(path): sorted(
                name
                for name in pq.ParquetFile(path).schema.names
                if name.startswith(("policy_", "label_", "outcome_", "target_"))
            )
            for path in pieces
        }
        forbidden_by_piece = {
            path: names for path, names in forbidden_by_piece.items() if names
        }
        if forbidden_by_piece:
            raise RuntimeError(
                f"phase={phase} physical feature shard is not target-free: "
                f"{forbidden_by_piece}"
            )
        frame = pd.concat([pd.read_parquet(path, columns=["__decision_ts__", *fields]) for path in pieces], ignore_index=True)
        ts = pd.to_datetime(frame["__decision_ts__"], utc=True)
        frame = frame.loc[(ts >= pd.Timestamp("2026-05-01", tz="UTC")) & (ts < pd.Timestamp("2026-08-01", tz="UTC"))].copy()
        if frame.empty or not ts.loc[frame.index].dt.minute.eq(phase).all():
            raise RuntimeError(f"phase={phase} feature stream has incorrect held timestamps")
        missing = sorted(set(fields).difference(frame.columns))
        if missing:
            raise RuntimeError(f"phase={phase} misses frozen base fields: {missing}")
        per_row = frame.loc[:, fields].notna().mean(axis=1)
        per_field = frame.loc[:, fields].notna().mean(axis=0)
        row_coverage = float(per_row.ge(0.90).mean())
        field_coverage = float(per_field.ge(0.90).mean())
        if row_coverage < 0.90 or field_coverage < 0.90:
            raise RuntimeError(f"phase={phase} frozen-contract coverage failed rows={row_coverage:.4f} fields={field_coverage:.4f}")
        rows.append({
            "phase_minutes": phase, "rows": int(len(frame)), "feature_count": len(fields),
            "rows_with_90pct_contract": row_coverage, "fields_with_90pct_row_coverage": field_coverage,
            "min_field_coverage": float(per_field.min()), "max_missing_per_row_fraction": float((1.0 - per_row).max()),
            "target_free": True,
        })
    path = root / "phase_native_feature_coverage_audit.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _phase_score_isolation_audit(raw_history: Path, raw_live: Path, out: Path) -> Path:
    """Prove phase-native scoring before the final shared portfolio replay.

    Each outer scorer receipt must carry its own phase, source-stream tag and
    target-free contract.  This receipt deliberately audits the raw score
    ledgers, rather than downstream MC1 or auction output, because those later
    stages legitimately combine chronology after phase-local scoring is done.
    """
    rows: list[dict[str, object]] = []
    for ledger_kind, ledger in (("history", raw_history), ("mayjul", raw_live)):
        for path in sorted(ledger.glob("**/run_manifest.json")):
            manifest = json.loads(path.read_text())
            if "phase_minutes" not in manifest:
                continue
            phase = int(manifest["phase_minutes"])
            if phase not in PHASES:
                raise RuntimeError(f"unexpected phase in raw-score receipt {path}: {phase}")
            if f"phase={phase:02d}" not in str(path):
                raise RuntimeError(f"phase receipt path does not match its declared phase: {path}")
            if manifest.get("phase_stream_tag") != STREAM_TAG:
                raise RuntimeError(f"phase={phase} raw score used an unexpected feature stream")
            if manifest.get("outcome_columns_consumed") not in ([], None):
                raise RuntimeError(f"phase={phase} raw score consumed outcome fields")
            rows.append({
                "ledger_kind": ledger_kind,
                "phase_minutes": phase,
                "family": "current" if "/current/" in str(path) else "bcf",
                "receipt": str(path),
                "receipt_sha256": _sha(path),
                "phase_stream_tag": manifest.get("phase_stream_tag"),
                "historical_native_ledger": bool(manifest.get("historical_native_ledger", False)),
                "outcome_columns_consumed": json.dumps(manifest.get("outcome_columns_consumed", [])),
            })
    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError("no phase-native raw-score receipts available for isolation audit")
    expected = {
        ("history", phase, family)
        for phase in PHASES for family in ("current", "bcf")
    } | {
        ("mayjul", phase, family)
        for phase in PHASES for family in ("current", "bcf")
    }
    observed = {
        (str(row.ledger_kind), int(row.phase_minutes), str(row.family))
        for row in audit.itertuples(index=False)
    }
    missing = sorted(expected.difference(observed))
    if missing:
        raise RuntimeError(f"phase-native raw-score isolation receipt is incomplete: {missing}")
    path = out / "phase_native_score_isolation_audit.parquet"
    audit.sort_values(["ledger_kind", "family", "phase_minutes"], kind="stable").to_parquet(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Immutable chain output directory. Defaults to the original v1 "
            "location; supply a new directory when preserving a failed receipt."
        ),
    )
    args = parser.parse_args()
    root = args.root
    if not (root / "features_complete").is_file():
        raise RuntimeError("all three phase feature streams must complete before this chain starts")
    out = args.out_dir or (root / "native_completion_chain_v1")
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    coverage = _coverage(root)
    raw_history = out / "raw_phase_history"
    raw_live = out / "raw_phase_mayjul"
    _run([
        sys.executable, str(ROOT / "scripts/score_strict_r3_phase_h1_full_stack.py"),
        "--feature-root", str(root), "--candidate-root", str(root), "--out-dir", str(raw_history),
        "--phases", "15,30,45", "--phase-stream-tag", STREAM_TAG, "--historical-native-ledger",
        "--score-end-exclusive", "2026-05-01T00:00:00Z",
    ])
    _run([
        sys.executable, str(ROOT / "scripts/score_strict_r3_phase_h1_full_stack.py"),
        "--feature-root", str(root), "--candidate-root", str(root), "--out-dir", str(raw_live),
        "--phases", "15,30,45", "--phase-stream-tag", STREAM_TAG,
    ])
    phase_score_audit = _phase_score_isolation_audit(raw_history, raw_live, out)
    panel_paths: dict[int, tuple[Path, Path]] = {}
    for phase in PHASES:
        candidate = root / f"candidate_union_phase{phase}/target_free_candidate_population.parquet"
        policy = out / f"policy_phase{phase}"
        _run([
            sys.executable, str(ROOT / "scripts/materialize_strict_r3_phase_h1_policy_labels.py"),
            "--candidates", str(candidate),
            "--start", f"2025-11-01T00:{phase:02d}:00Z", "--end", f"2026-08-01T00:{phase:02d}:00Z",
            "--out-dir", str(policy),
        ])
        panels = out / f"mc1_panels_phase{phase}"
        _run([
            sys.executable, str(ROOT / "scripts/assemble_strict_r3_phase_h1_mc1_panels.py"),
            "--current-native-history-root", str(raw_history), "--bcf-native-history-root", str(raw_history),
            "--current-raw-root", str(raw_live), "--bcf-raw-root", str(raw_live),
            "--phase", str(phase), "--start", "2026-05-01T00:00:00Z", "--end", "2026-08-01T00:00:00Z",
            "--out-dir", str(panels),
        ])
        mc1 = out / f"mc1_phase{phase}_native"
        _run([
            sys.executable, str(ROOT / "scripts/replay_strict_r3_score_family_mc1_canonical_policy.py"),
            "--bcf-scores", str(panels / "bcf_scores_target_free.parquet"),
            "--current-scores", str(panels / "current_scores_target_free.parquet"),
            "--canonical-policy", str(policy / "canonical_policy_contract.parquet"),
            "--start", "2026-05-01T00:00:00Z", "--end", "2026-08-01T00:00:00Z", "--out-dir", str(mc1),
        ])
        panel_paths[phase] = (
            mc1 / "predictions_current_v5_mc1_d2.parquet",
            mc1 / "predictions_bcf_mc1_d2.parquet",
        )
    pooled = out / "pooled_four_phase_native"
    command = [
        sys.executable, str(ROOT / "scripts/replay_strict_r3_phase_h1_pooled_dual_portfolio.py"),
        "--out-dir", str(pooled),
        "--phase", "0", str(PHASE0_CURRENT), str(PHASE0_BCF),
    ]
    for phase in PHASES:
        current, bcf = panel_paths[phase]
        command.extend(["--phase", str(phase), str(current), str(bcf)])
    _run(command)
    report = out / "REPORT.md"
    _run([
        sys.executable, str(ROOT / "scripts/report_strict_r3_phase_h1_native_completion.py"),
        "--chain-root", str(out),
    ])
    manifest = {
        "schema": "strict_r3_phase_h1_native_completion_chain_v1",
        "scope": "offline research only; live artifacts and exchange I/O are prohibited",
        "feature_coverage_audit": {"path": str(coverage), "sha256": _sha(coverage)},
        "phase_native_score_isolation_audit": {"path": str(phase_score_audit), "sha256": _sha(phase_score_audit)},
        "raw_history": str(raw_history), "raw_mayjul": str(raw_live), "pooled_replay": str(pooled),
        "report": str(report),
        "phase0_control": {"current": str(PHASE0_CURRENT), "bcf": str(PHASE0_BCF)},
        "phase_native_offsets": list(PHASES),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
