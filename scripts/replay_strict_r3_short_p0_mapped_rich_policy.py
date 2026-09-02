#!/usr/bin/env python3
"""Replay the frozen short P0/F90 mapping arms through the rich policy.

This is deliberately an *evaluation* producer.  It does not fit a mapper,
choose a threshold, inspect a future path before admission, or alter any live
contract.  It joins target-free P0/F90 score identities to already-frozen
strictly-OOS mapper decisions, materialises outcome paths only afterwards, and
then applies the same side-aware rich policy and chronological portfolio
auction used by the long research process.
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

from extreme_price_movements.strict_r3_rich_policy import (  # noqa: E402
    RichPolicyParams,
    causal_portfolio_selection,
    policy_metrics,
)
from scripts.run_strict_r3_rich_policy_hpo import (  # noqa: E402
    _median_atr_fraction,
    _simulate_frame,
    load_score_population,
    materialize_paths,
)


DEFAULT_LEDGERS = (
    ROOT / "data_perp/artifacts/strict_r3_short_p0_f90_prequential_ledger_2024_geometry_20260821_v3",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_f90_prequential_ledger_2025_20260821_v6",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_f90_prequential_ledger_2026_20260821_v1",
)
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_short_p0_rich_policy_hpo_2024select_2025_2026oos_20260821_v1/frozen_challenger.json"
DEFAULT_BARS = ROOT / "15m_ohlcv_perp"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_mapped_rich_policy_oos_2025_2026_20260821_v1"

ARM_SPECS: tuple[dict[str, str], ...] = (
    {
        "name": "raw_cell_day28",
        "path": "data_perp/artifacts/strict_r3_short_p0_baseonly_cell_day28_admission_2025jan_2026jul_20260821_v1/score_and_cell_day_admission_provenance.parquet",
        "timestamp": "__decision_ts__",
        "expected": "causal_21d_side_expected_net_bps",
        "admitted": "causal_21d_side_admitted_ge_50bps",
    },
    {
        "name": "r5_posterior_trust",
        "path": "data_perp/artifacts/strict_r3_short_p0_r5_monthly_oof_2025dec_2026jul_20260821_v1/short_r5_oof_predictions.parquet",
        "timestamp": "__decision_ts__",
        "expected": "trust_corrected_expected_net_bps",
        "admitted": "trust_posterior_admitted_ge_50bps",
    },
    {
        "name": "mc1_d2",
        "path": "data_perp/artifacts/strict_r3_short_p0_baseonly_static_mc1_oof_extended_2025jul_2026jul_20260821_v1/short_current_mc1_oof_predictions.parquet",
        "timestamp": "__decision_ts__",
        "expected": "mc1_d2_expected_net_bps",
        "admitted": "mc1_d2_admitted_ge_50bps",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No accepted resolved trades._"
    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{value:.2f}"
        return str(value).replace("|", "\\|")
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False, name=None))
    return "\n".join(lines)


def _load_arm(spec: dict[str, str], *, threshold: float) -> pd.DataFrame:
    path = ROOT / spec["path"]
    if not path.exists():
        raise FileNotFoundError(path)
    columns = ["candidate_id", spec["timestamp"], spec["expected"], spec["admitted"]]
    work = pd.read_parquet(path, columns=columns).rename(columns={
        spec["timestamp"]: "timestamp",
        spec["expected"]: "mapped_expected_net_bps",
        spec["admitted"]: "mapper_admitted",
    })
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["mapped_expected_net_bps"] = pd.to_numeric(work["mapped_expected_net_bps"], errors="coerce")
    # The boolean is source evidence.  Requiring the declared floor as well
    # makes accidental field corruption fail closed.
    work["admitted"] = (
        work["mapper_admitted"].astype(bool)
        & np.isfinite(work["mapped_expected_net_bps"])
        & work["mapped_expected_net_bps"].ge(float(threshold))
    )
    if work.duplicated(["candidate_id", "timestamp"]).any():
        raise AssertionError(f"{spec['name']} mapper source has duplicate candidate identities")
    return work


def _yearly(frame: pd.DataFrame, selected: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in (2025, 2026):
        mask = frame["timestamp"].dt.year.eq(year).to_numpy() & np.asarray(selected, bool)
        summary, _, _ = policy_metrics(frame, mask)
        rows.append({"year": year, **summary})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite immutable output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    policy_file = Path(args.policy).resolve()
    policy_blob = json.loads(policy_file.read_text())
    params = RichPolicyParams.from_mapping(policy_blob["params"])
    median_atr = float(policy_blob["median_atr_fraction_fitted_on_complete_2024_development"])
    population = load_score_population(args.ledger or list(DEFAULT_LEDGERS), side="short")
    population = population[["candidate_id", "timestamp", "symbol"]]

    all_yearly: list[pd.DataFrame] = []
    all_monthly: list[pd.DataFrame] = []
    all_weekly: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    exit_causes: list[pd.DataFrame] = []
    for spec in ARM_SPECS:
        mapping = _load_arm(spec, threshold=args.threshold_bps)
        joined = mapping.merge(population, on=["candidate_id", "timestamp"], how="left", validate="one_to_one")
        missing_symbol = int(joined["symbol"].isna().sum())
        if missing_symbol:
            raise AssertionError(f"{spec['name']}: {missing_symbol} mapper identities absent from P0/F90 ledger")
        before_path = joined.loc[joined["admitted"].astype(bool)].copy()
        before_path["score"] = before_path["mapped_expected_net_bps"]
        # This is the first post-admission operation that accesses a future
        # path.  It is never used to restore or create an admission.
        resolved, paths, coverage = materialize_paths(before_path, bars_root=Path(args.bars_root).resolve())
        replay = _simulate_frame(
            resolved, paths, params=params, median_atr=median_atr,
            theta_paths=None, side="short",
        )
        selected = causal_portfolio_selection(replay)
        summary, monthly, weekly = policy_metrics(replay, selected)
        accepted = replay.loc[selected].copy()
        accepted["arm"] = spec["name"]
        accepted["selected_by"] = "mapped_expected_net_bps"
        accepted.to_parquet(output / f"accepted_{spec['name']}.parquet", index=False)
        coverage.to_parquet(output / f"path_coverage_{spec['name']}.parquet", index=False)
        annual = _yearly(replay, selected)
        annual["arm"] = spec["name"]
        all_yearly.append(annual)
        monthly["arm"] = spec["name"]
        weekly["arm"] = spec["name"]
        all_monthly.append(monthly)
        all_weekly.append(weekly)
        causes = accepted.groupby("exit_reason", dropna=False).agg(
            trades=("candidate_id", "size"),
            net_bps_per_trade=("net_bps", "mean"),
            total_net_bps=("net_bps", "sum"),
        ).reset_index()
        causes["arm"] = spec["name"]
        exit_causes.append(causes)
        provenance.append({
            "arm": spec["name"], "mapping_rows": int(len(mapping)),
            "mapper_admitted_before_path": int(len(before_path)),
            "complete_policy_paths_after_admission": int(len(resolved)),
            "portfolio_accepted": int(selected.sum()),
            "missing_ledger_identity": missing_symbol,
            "mapping_source": str(ROOT / spec["path"]),
            "mapping_source_sha256": _sha256(ROOT / spec["path"]),
            **summary,
        })

    yearly = pd.concat(all_yearly, ignore_index=True)
    monthly = pd.concat(all_monthly, ignore_index=True)
    weekly = pd.concat(all_weekly, ignore_index=True)
    provenance_frame = pd.DataFrame(provenance)
    causes = pd.concat(exit_causes, ignore_index=True)
    yearly.to_parquet(output / "comparison_yearly.parquet", index=False)
    monthly.to_parquet(output / "comparison_monthly.parquet", index=False)
    weekly.to_parquet(output / "comparison_weekly.parquet", index=False)
    provenance_frame.to_parquet(output / "admission_and_portfolio_provenance.parquet", index=False)
    causes.to_parquet(output / "exit_causes.parquet", index=False)
    correctness = {
        "schema": "strict_r3_short_p0_mapped_rich_policy_replay_v1",
        "status": "passed",
        "side": "short",
        "score_and_admission": "Frozen mapper OOS decisions only; expected mapped EV >= declared threshold before any future path is loaded.",
        "policy": "Frozen 2024-only rich-policy HPO winner; side-aware short simulation; H12 / 48 future 15m bars; cost applied once.",
        "portfolio": "Chronological two-concurrent, two-per-timestamp, one-per-asset auction; priority is mapped expected net bps.",
        "outcome_handling": "Incomplete/missing paths are not used to change admission and are excluded only from resolved-outcome metrics.",
        "arm_provenance": provenance,
    }
    (output / "correctness_report.json").write_text(json.dumps(correctness, indent=2, sort_keys=True) + "\n")
    report = [
        "# Short P0/F90 — Mapped Rich-Policy OOS Replay",
        "",
        "This compares the already-frozen raw Cell-day, R5 posterior-trust, and MC1-d2 mapper decisions. It is not a mapper or policy HPO: all mapping admission is frozen before exact H12 paths are materialised.",
        "",
        f"- Side: `short`; mapped-EV admission floor: `{args.threshold_bps:.1f}` bps.",
        f"- Policy: `{policy_file}`; selected in 2024 only, then frozen.",
        "- Execution: decision-time first 15-minute open; H12 exact 15-minute rich policy; 100 bps cost exactly once.",
        "- Portfolio: two concurrent, max two new per timestamp, one position per asset; mapped expected net bps priority.",
        "",
        "## Admission and aggregate portfolio evidence",
        "",
        _render(provenance_frame),
        "",
        "## Yearly constrained replay",
        "",
        _render(yearly),
        "",
        "## Monthly constrained replay",
        "",
        _render(monthly),
        "",
        "## Exit causes",
        "",
        _render(causes),
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    manifest = {
        "schema": "strict_r3_short_p0_mapped_rich_policy_replay_v1",
        "side": "short", "status": "complete", "threshold_bps": float(args.threshold_bps),
        "policy": str(policy_file), "policy_sha256": _sha256(policy_file),
        "policy_selection": "2024 only (rich policy HPO); replay years 2025 and 2026 never used in policy selection",
        "ledger_roots": [str(path) for path in (args.ledger or list(DEFAULT_LEDGERS))],
        "bars_root": str(Path(args.bars_root).resolve()),
        "arms": list(ARM_SPECS),
        "prohibitions": ["no_live_state", "no_exchange_io", "no_mapper_refit", "no_future_path_admission", "no_held_outcome_selection"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, action="append", default=None)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--bars-root", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
