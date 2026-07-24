#!/usr/bin/env python3
"""Materialize a resume-safe frozen diagnostic failure backcast by month.

This backcast is deliberately not labelled OOS.  It applies one frozen base
model and one frozen AE/GMM representation to historical observable features,
then evaluates the full top-30 candidate stream on cost-aware hourly-close
paths.  The top-10-equivalent population is retained as a separate monitor
flag.  Monthly shards are hard-linked into one candidate directory so the
failure-taxonomy runner can stream the same bytes without a duplicate ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = ROOT / "data_perp/features/20260711_070000"
DEFAULT_BASE_RUN = ROOT / (
    "data_perp/reports/s59_h5_benchmark66_matchedaegmm_refit_wf30_20260716_v1"
)
DEFAULT_META_HANDOFF = DEFAULT_BASE_RUN / "meta_handoff_top30_allsafe_newaegmm_20260717"
DEFAULT_ARTIFACT = ROOT / (
    "data_perp/artifacts/"
    "s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2"
)
DEFAULT_LABEL_POLICY = ROOT / (
    "data_perp/artifacts/"
    "20260713_s59_h5_fullthroughjul10_trailing_cost100bps_labels/labels/"
    "side_archetype_label_manifest.json"
)
DEFAULT_RESIDUAL_REFERENCE = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "july_current_contract_refit/train_reference_scores.parquet"
)
DEFAULT_OUTPUT = ROOT / "data_perp/reports/failure_three_year_backcast_20260719_v1"


@dataclass(frozen=True)
class MonthChunk:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def key(self) -> str:
        return self.start.strftime("%Y%m")


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"Timestamp must carry an explicit timezone: {value}")
    return timestamp.tz_convert("UTC")


def _month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[MonthChunk]:
    chunks: list[MonthChunk] = []
    cursor = start
    while cursor < end:
        next_month = (
            cursor.tz_localize(None).to_period("M") + 1
        ).start_time.tz_localize("UTC")
        boundary = min(end, next_month)
        chunks.append(MonthChunk(cursor, boundary))
        cursor = boundary
    return chunks


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_completed_chunk(
    directory: Path,
    chunk: MonthChunk,
    *,
    min_path_coverage: float,
) -> bool:
    manifest_path = directory / "manifest.json"
    output_path = directory / "frozen_predictions.parquet"
    if not manifest_path.exists() or not output_path.exists():
        return False
    try:
        manifest = _json(manifest_path)
        if manifest.get("start") != chunk.start.isoformat():
            return False
        if manifest.get("end_exclusive") != chunk.end.isoformat():
            return False
        if int(manifest.get("rows", 0)) <= 0:
            return False
        if manifest.get("return_unit") != "decimal_notional_return":
            return False
        if bool(manifest.get("cost_counted_once")) is not True:
            return False
        if (
            manifest.get("outcome_contract_version")
            != "hourly_close_policy_proxy_v2_activation_deadline"
        ):
            return False
        if int(manifest.get("policy_bar_minutes", 0)) != 15:
            return False
        observable = [
            str(name) for name in manifest.get("observable_feature_names", [])
        ]
        if not any(name.startswith("base_attr_") for name in observable):
            return False
        path_stats = manifest.get("path_stats") or {}
        if not path_stats:
            return False
        if any(
            float(stats.get("coverage", 0.0)) < float(min_path_coverage)
            for stats in path_stats.values()
        ):
            return False
        return pq.ParquetFile(output_path).metadata.num_rows == int(manifest["rows"])
    except Exception:
        return False


def _command(args: argparse.Namespace, chunk: MonthChunk, directory: Path) -> list[str]:
    artifact = Path(args.model_artifact_root)
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/backfill_complete_july_meta_predictions.py"),
        "--feature-root",
        str(args.feature_root),
        "--base-reference",
        str(args.base_reference),
        "--model-artifact-root",
        str(artifact),
        "--ae-gmm-state",
        str(args.ae_gmm_state),
        "--meta-handoff-dir",
        str(args.meta_handoff_dir),
        "--residual-bundle",
        str(args.residual_bundle),
        "--residual-train-reference",
        str(args.residual_train_reference),
        "--native-run-id",
        str(args.native_run_id),
        "--policy-manifest",
        str(args.policy_manifest),
        "--source-manifest",
        str(args.source_manifest),
        "--output-dir",
        str(directory),
        "--start",
        chunk.start.isoformat(),
        "--end-exclusive",
        chunk.end.isoformat(),
        "--evidence-scope",
        "frozen_backcast_diagnostic",
        "--base-only-backcast",
        "--base-top-frac",
        str(args.base_top_frac),
        "--backcast-admission-frac",
        str(args.admission_frac),
        "--backcast-outcome-source",
        "hourly_close_proxy",
        "--backcast-proxy-horizon-hours",
        str(args.proxy_horizon_hours),
        "--backcast-policy-bar-minutes",
        str(args.policy_bar_minutes),
        "--backcast-include-observable-features",
    ]


def _run_chunk(args: argparse.Namespace, chunk: MonthChunk) -> dict[str, Any]:
    output = Path(args.output)
    directory = output / "monthly" / chunk.key
    directory.mkdir(parents=True, exist_ok=True)
    if args.resume and _valid_completed_chunk(
        directory, chunk, min_path_coverage=args.min_path_coverage
    ):
        return {
            "key": chunk.key,
            "status": "reused",
            **_json(directory / "manifest.json"),
        }
    command = _command(args, chunk, directory)
    if args.dry_run:
        return {"key": chunk.key, "status": "dry_run", "command": command}
    log_path = directory / "run.log"
    environment = dict(os.environ)
    environment.setdefault("PYTHONPATH", str(ROOT))
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/ares-mpl-cache")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Backcast chunk {chunk.key} failed with {result.returncode}; see {log_path}"
        )
    if not _valid_completed_chunk(
        directory, chunk, min_path_coverage=args.min_path_coverage
    ):
        raise RuntimeError(f"Backcast chunk {chunk.key} failed completion validation")
    return {
        "key": chunk.key,
        "status": "completed",
        **_json(directory / "manifest.json"),
    }


def _link_candidate_shards(output: Path, chunks: list[MonthChunk]) -> Path:
    candidate_root = output / "candidate_shards"
    candidate_root.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        source = output / "monthly" / chunk.key / "frozen_predictions.parquet"
        target = candidate_root / f"candidates_{chunk.key}.parquet"
        if target.exists() and os.path.samefile(source, target):
            continue
        if target.exists():
            target.unlink()
        os.link(source, target)
    return candidate_root


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = _utc(args.start)
    end = _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("--end-exclusive must be later than --start")
    chunks = _month_chunks(start, end)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as pool:
        futures = {pool.submit(_run_chunk, args, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "chunk": result["key"],
                        "status": result["status"],
                        "rows": result.get("rows"),
                    }
                ),
                flush=True,
            )
    results.sort(key=lambda row: row["key"])
    candidate_root = None if args.dry_run else _link_candidate_shards(output, chunks)
    total_rows = sum(int(row.get("rows", 0)) for row in results)
    selected_rows = sum(int(row.get("selected_for_monitor_rows", 0)) for row in results)
    covered_days = sum(int(row.get("days", 0)) for row in results)
    manifest = {
        "schema": "three_year_frozen_failure_backcast_v1",
        "evidence_scope": "frozen_backcast_diagnostic_not_oos",
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "calendar_span_days": int((end - start) / pd.Timedelta(days=1)),
        "covered_days_sum": int(covered_days),
        "month_chunks": int(len(chunks)),
        "rows": int(total_rows),
        "selected_for_monitor_rows": int(selected_rows),
        "base_top_frac": float(args.base_top_frac),
        "admission_frac": float(args.admission_frac),
        "round_trip_cost": 0.01,
        "return_unit": "decimal_notional_return",
        "cost_counted_once": True,
        "outcome_source": "canonical_krakenfutures_hourly_close_path",
        "outcome_contract_version": "hourly_close_policy_proxy_v2_activation_deadline",
        "policy_bar_minutes": int(args.policy_bar_minutes),
        "execution_parity_claim": False,
        "candidate_root": str(candidate_root) if candidate_root is not None else None,
        "chunks": results,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, default=str), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-07-18T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-07-18T00:00:00Z")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument(
        "--base-reference",
        type=Path,
        default=DEFAULT_BASE_RUN / "best_oos_scored_ledger.parquet",
    )
    parser.add_argument("--model-artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--ae-gmm-state",
        type=Path,
        default=DEFAULT_ARTIFACT / "ae_gmm_state/ae_gmm_state.pkl",
    )
    parser.add_argument("--meta-handoff-dir", type=Path, default=DEFAULT_META_HANDOFF)
    parser.add_argument(
        "--residual-bundle",
        type=Path,
        default=DEFAULT_ARTIFACT / "policy_params/v9_tail95_predecessor_bundle.joblib",
    )
    parser.add_argument(
        "--residual-train-reference", type=Path, default=DEFAULT_RESIDUAL_REFERENCE
    )
    parser.add_argument(
        "--native-run-id",
        default="s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2",
    )
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_LABEL_POLICY)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_META_HANDOFF / "manifest.json"
    )
    parser.add_argument("--base-top-frac", type=float, default=0.30)
    parser.add_argument("--admission-frac", type=float, default=0.10)
    parser.add_argument("--proxy-horizon-hours", type=int, default=24)
    parser.add_argument("--policy-bar-minutes", type=int, default=15)
    parser.add_argument("--min-path-coverage", type=float, default=0.90)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
