#!/usr/bin/env python3
"""Build bounded historical inputs for the canonical deployed-policy simulator.

This adapter does not simulate returns.  It reconstructs the three immutable
inputs consumed by ``materialize_execution_ev_policy_labels.py``:

* the complete historical candidate identity stream;
* an explicit side-parent-fallback context; and
* the archived canonical barrier/ATR path-input subset.

The archived raw candidate barrier is used only as a parity witness.  The
canonical path-target barrier is authoritative and must match it bit-for-bit
for every admitted row.  Historical policy archetype strings are deliberately
ignored: the referenced current label artifact must prove that every candidate
resolved to the side-parent fallback.

The resulting replay uses the current frozen per-asset spread baseline.  It is
therefore a *current-spread counterfactual on historical paths*, not factual
historical execution-cost evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reconstruct_janfeb2025_execution_ev_12h_oof import (  # noqa: E402
    IDENTITY,
    requested_months,
    source_paths,
)

SCHEMA = "historical_execution_ev_deployed_policy_inputs_v1"
SIDE_PARENT_SENTINEL = "historical_side_parent_fallback"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = {
        str(key): _safe(value)
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_identity(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    output = frame.copy()
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="coerce")
    if output["__ts__"].isna().any():
        raise ValueError(f"{name} contains invalid __ts__")
    output["__symbol__"] = output["__symbol__"].astype(str)
    output["side_name"] = output["side_name"].astype(str).str.lower()
    output["candidate_id"] = output["candidate_id"].astype(str)
    if not output["side_name"].isin(("long", "short")).all():
        raise ValueError(f"{name} has noncanonical side_name values")
    if output.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError(f"{name} has duplicate exact candidate identities")
    return output


def load_historical_candidates(
    labels_root: Path,
    *,
    start_month: str,
    end_month: str,
) -> tuple[pd.DataFrame, list[Path]]:
    paths = source_paths(
        labels_root, start_month=start_month, end_month=end_month
    )
    columns = [
        *IDENTITY,
        "__decision_ts__",
        "__barrier_pct__",
    ]
    parts = [pd.read_parquet(path, columns=columns) for path in paths]
    rows = _normalize_identity(pd.concat(parts, ignore_index=True), name="candidates")
    rows["__decision_ts__"] = pd.to_datetime(
        rows["__decision_ts__"], utc=True, errors="coerce"
    )
    expected = rows["__ts__"] + pd.Timedelta(hours=1)
    if rows["__decision_ts__"].isna().any() or not rows[
        "__decision_ts__"
    ].equals(expected):
        raise ValueError("historical decision timestamp is not signal + one hour")
    barrier = pd.to_numeric(rows["__barrier_pct__"], errors="coerce")
    if not np.isfinite(barrier).all() or not barrier.gt(0.0).all():
        raise ValueError("historical source barrier is not finite and positive")
    rows["__barrier_pct__"] = barrier.astype(np.float32)
    return rows.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True), paths


def load_archived_path_inputs(
    paths: Sequence[Path],
    *,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    months = requested_months(start_month, end_month)
    start = months[0].start_time.tz_localize("UTC")
    end = (months[-1] + 1).start_time.tz_localize("UTC")
    columns = [
        *IDENTITY,
        "__barrier_pct__",
        "__path_auxiliary_atr_fraction__",
    ]
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        parts.append(frame.loc[(frame["__ts__"] >= start) & (frame["__ts__"] < end)])
    rows = _normalize_identity(pd.concat(parts, ignore_index=True), name="path inputs")
    for column in ("__barrier_pct__", "__path_auxiliary_atr_fraction__"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce").astype(np.float32)
    finite = np.isfinite(
        rows[["__barrier_pct__", "__path_auxiliary_atr_fraction__"]].to_numpy(
            dtype=np.float64
        )
    ).all(axis=1)
    positive = (
        rows[["__barrier_pct__", "__path_auxiliary_atr_fraction__"]]
        .gt(0.0)
        .all(axis=1)
        .to_numpy()
    )
    return rows.loc[finite & positive].sort_values(
        list(IDENTITY), kind="stable"
    ).reset_index(drop=True)


def _reference_contract(reference_manifest: Path, policy_path: Path) -> dict[str, Any]:
    payload = json.loads(reference_manifest.read_text(encoding="utf-8"))
    geometry = payload.get("geometry", {})
    source = payload.get("source", {})
    if (
        float(geometry.get("fallback_rate", -1.0)) != 1.0
        or int(geometry.get("side_archetype_rows", -1)) != 0
    ):
        raise ValueError(
            "reference policy labels do not prove universal side-parent fallback"
        )
    expected_policy = str(source.get("policy_sha256", ""))
    actual_policy = _sha256(policy_path)
    if not expected_policy or expected_policy != actual_policy:
        raise ValueError("policy hash differs from the canonical reference labels")
    return {
        "manifest": str(reference_manifest),
        "manifest_sha256": _sha256(reference_manifest),
        "policy_sha256": actual_policy,
        "fallback_rate": 1.0,
        "side_archetype_rows": 0,
    }


def _spread_contract(
    spread_baseline: Path, symbols: Sequence[str]
) -> dict[str, Any]:
    baseline = pd.read_csv(spread_baseline)
    required = {"symbol", "p90_spread_bps"}
    missing = sorted(required.difference(baseline.columns))
    if missing:
        raise ValueError(f"spread baseline misses columns: {missing}")
    baseline["symbol"] = baseline["symbol"].astype(str)
    spread = pd.to_numeric(baseline["p90_spread_bps"], errors="coerce")
    valid = np.isfinite(spread) & spread.gt(0.0)
    mapped = set(baseline.loc[valid, "symbol"])
    required_symbols = set(map(str, symbols))
    unmapped = sorted(required_symbols.difference(mapped))
    if unmapped:
        raise ValueError(
            "current spread baseline does not cover historical replay symbols: "
            f"{unmapped}"
        )
    return {
        "path": str(spread_baseline),
        "sha256": _sha256(spread_baseline),
        "spread_column": "p90_spread_bps",
        "mapped_symbols": len(required_symbols),
        "unmapped_symbols": [],
        "historical_interpretation": (
            "current frozen per-asset spread counterfactual; not contemporaneous "
            "historical spread evidence"
        ),
    }


def _coverage_rows(merged: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    work = merged.copy()
    work["month"] = work["__ts__"].dt.strftime("%Y-%m")
    for (month, side), rows in work.groupby(["month", "side_name"], sort=True):
        total = int(len(rows))
        admitted = int(rows["__has_path_input__"].sum())
        output.append(
            {
                "month": str(month),
                "side_name": str(side),
                "candidate_rows": total,
                "canonical_path_input_rows": admitted,
                "canonical_path_input_coverage": admitted / max(total, 1),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir
    outputs = {
        "candidates": output_dir / "candidates.parquet",
        "context": output_dir / "context.parquet",
        "path_targets": output_dir / "path_targets.parquet",
        "manifest": output_dir / "manifest.json",
    }
    if output_dir.exists() or any(path.exists() for path in outputs.values()):
        raise ValueError("refusing to overwrite historical policy-input outputs")

    reference = _reference_contract(args.reference_manifest, args.policy_json)
    candidates, source_files = load_historical_candidates(
        args.labels_root,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    archived = load_archived_path_inputs(
        args.path_input_files,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    universe: dict[str, Any] = {
        "mode": "all_source_symbols",
        "allowlist": None,
        "allowlist_sha256": None,
    }
    if args.symbol_allowlist is not None:
        symbols = {
            line.strip()
            for line in args.symbol_allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not symbols:
            raise ValueError("symbol allowlist is empty")
        available = set(candidates["__symbol__"].astype(str))
        unknown = sorted(symbols.difference(available))
        if unknown:
            raise ValueError(f"symbol allowlist contains unknown source symbols: {unknown}")
        candidates = candidates.loc[
            candidates["__symbol__"].astype(str).isin(symbols)
        ].reset_index(drop=True)
        archived = archived.loc[
            archived["__symbol__"].astype(str).isin(symbols)
        ].reset_index(drop=True)
        universe = {
            "mode": "frozen_symbol_allowlist",
            "allowlist": str(args.symbol_allowlist),
            "allowlist_sha256": _sha256(args.symbol_allowlist),
            "symbols": sorted(symbols),
            "symbol_count": len(symbols),
            "interpretation": (
                "diagnostic common-universe ablation; never compare its coverage "
                "or economics to an unrestricted book without an explicit slice"
            ),
        }
    witness = candidates.merge(
        archived,
        on=list(IDENTITY),
        how="left",
        suffixes=("__source", "__canonical"),
        validate="one_to_one",
        indicator=True,
    )
    witness["__has_path_input__"] = witness["_merge"].eq("both")
    admitted = witness.loc[witness["__has_path_input__"]].copy()
    if admitted.empty:
        raise ValueError("no exact historical canonical path inputs join candidates")
    source_barrier = admitted["__barrier_pct____source"].to_numpy(np.float32)
    canonical_barrier = admitted["__barrier_pct____canonical"].to_numpy(np.float32)
    barrier_equal = source_barrier.view(np.uint32) == canonical_barrier.view(np.uint32)
    if not barrier_equal.all():
        raise ValueError(
            "archived canonical barrier differs from its historical source witness "
            f"for {int((~barrier_equal).sum())} admitted rows"
        )
    coverage = _coverage_rows(witness)
    failures = [
        row
        for row in coverage
        if row["canonical_path_input_coverage"] < args.minimum_join_coverage
    ]
    if failures:
        raise ValueError(
            "canonical historical path-input coverage is below the configured "
            f"minimum: {failures}"
        )
    spread = _spread_contract(
        args.spread_baseline, admitted["__symbol__"].unique().tolist()
    )

    candidate_output = candidates.loc[:, list(IDENTITY)].copy()
    context_output = candidate_output.copy()
    context_output["policy_archetype"] = SIDE_PARENT_SENTINEL
    path_output = admitted.loc[
        :,
        [
            *IDENTITY,
            "__barrier_pct____canonical",
            "__path_auxiliary_atr_fraction__",
        ],
    ].rename(columns={"__barrier_pct____canonical": "__barrier_pct__"})

    output_dir.mkdir(parents=True)
    candidate_output.to_parquet(
        outputs["candidates"], index=False, compression="zstd"
    )
    context_output.to_parquet(outputs["context"], index=False, compression="zstd")
    path_output.to_parquet(
        outputs["path_targets"], index=False, compression="zstd"
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "ready_for_current_spread_counterfactual_materialization",
        "period": {
            "start_month": args.start_month,
            "end_month": args.end_month,
        },
        "contract": {
            "economic_interpretation": (
                "current frozen spread counterfactual on historical exact-one-minute "
                "paths; not factual historical execution costs"
            ),
            "simulator": (
                "downstream materialize_execution_ev_policy_labels.py using "
                "extreme_price_movements.simple_policy_optimiser.simulate_and_score"
            ),
            "geometry": "side-parent fallback proven by canonical reference manifest",
            "barrier": "archived canonical path input with bit-exact source witness",
            "atr": (
                "archived canonical raw Wilder ATR14 fraction at signal; retained "
                "for input-lineage parity and eligibility, not used by simulator"
            ),
            "decision": "signal timestamp + one hour",
        },
        "universe": universe,
        "reference": reference,
        "spread": spread,
        "source": {
            "candidate_files": [str(path) for path in source_files],
            "candidate_file_sha256": {
                str(path): _sha256(path) for path in source_files
            },
            "path_input_files": [str(path) for path in args.path_input_files],
            "path_input_file_sha256": {
                str(path): _sha256(path) for path in args.path_input_files
            },
        },
        "parity": {
            "candidate_rows": int(len(candidates)),
            "admitted_rows": int(len(admitted)),
            "barrier_bit_exact_rows": int(barrier_equal.sum()),
            "barrier_mismatch_rows": int((~barrier_equal).sum()),
            "decision_timestamp_exact": True,
            "coverage_by_side_month": coverage,
            "minimum_join_coverage": float(args.minimum_join_coverage),
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(
                    len(candidate_output)
                    if name in {"candidates", "context"}
                    else len(path_output)
                ),
            }
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json(outputs["manifest"], manifest)
    return outputs


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--labels-root",
        type=Path,
        default=ROOT
        / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels",
    )
    result.add_argument(
        "--path-input-files",
        type=Path,
        nargs="+",
        default=[
            ROOT
            / "data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels/train_global_long_3.parquet",
            ROOT
            / "data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels/train_global_short_3.parquet",
        ],
    )
    result.add_argument(
        "--reference-manifest",
        type=Path,
        default=ROOT
        / "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/manifest.json",
    )
    result.add_argument(
        "--policy-json",
        type=Path,
        default=ROOT
        / "data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1/production_staging/best_policy_params.json",
    )
    result.add_argument(
        "--spread-baseline",
        type=Path,
        default=ROOT
        / "data_perp/exchanges/krakenfutures/spread_model/per_asset_spread_baseline_latest.csv",
    )
    result.add_argument("--start-month", default="2025-02")
    result.add_argument("--end-month", default="2025-04")
    result.add_argument("--minimum-join-coverage", type=float, default=0.70)
    result.add_argument(
        "--symbol-allowlist",
        type=Path,
        help="optional frozen diagnostic universe, one canonical symbol per line",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0.0 < args.minimum_join_coverage <= 1.0:
        raise ValueError("--minimum-join-coverage must lie in (0, 1]")
    outputs = run(args)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
