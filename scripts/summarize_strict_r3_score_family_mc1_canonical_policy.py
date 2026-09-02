#!/usr/bin/env python3
"""Publish the immutable comparison of separately materialised MC1 arms.

The BCF and current-v5 MC1 runs are deliberately executed separately to bound
memory.  This producer verifies that their overlapping candidate identities
carry identical canonical policy outcomes, then aggregates their independently
constrained portfolio replays.  It never refits a score, mapper, or policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_COLUMNS = (
    "policy_path_valid",
    "policy_gross_bps",
    "policy_net_bps",
    "policy_exit_bar_15m",
    "policy_entry_price",
    "policy_exit_price",
    "policy_exit_reason",
    "policy_label_available_ts",
    "policy_outcome_source",
    "policy_cost_bps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_path(run: Path, family: str) -> Path:
    path = run / f"predictions_{family}_mc1_d2.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _policy_identity_audit(bcf: pd.DataFrame, current: pd.DataFrame) -> dict[str, int]:
    cols = ["candidate_id", "__decision_ts__", *POLICY_COLUMNS]
    left = bcf.loc[:, cols].copy()
    right = current.loc[:, cols].copy()
    overlap = left.merge(right, on="candidate_id", suffixes=("_bcf", "_current"), validate="one_to_one")
    if overlap.empty:
        raise AssertionError("BCF/current-v5 candidate routes have no overlap")
    for field in ("__decision_ts__", *POLICY_COLUMNS):
        lhs, rhs = overlap[f"{field}_bcf"], overlap[f"{field}_current"]
        if pd.api.types.is_numeric_dtype(lhs):
            equal = np.isclose(lhs.to_numpy(float), rhs.to_numpy(float), equal_nan=True).all()
        else:
            equal = lhs.fillna("__null__").astype(str).equals(rhs.fillna("__null__").astype(str))
        if not equal:
            raise AssertionError(f"canonical policy mismatch on overlapping candidate IDs: {field}")
    return {
        "bcf_prediction_rows": int(len(bcf)),
        "current_v5_prediction_rows": int(len(current)),
        "overlapping_candidate_ids": int(len(overlap)),
    }


def _month_metrics(decisions: pd.DataFrame, family: str) -> pd.DataFrame:
    decisions = decisions.copy()
    decisions["timestamp"] = pd.to_datetime(decisions["timestamp"], utc=True, errors="raise")
    decisions["month"] = decisions["timestamp"].dt.strftime("%Y-%m")
    rows: list[dict[str, object]] = []
    for month, piece in decisions.groupby("month", sort=True):
        accepted = piece.loc[piece["accepted"].astype(bool)].copy()
        bps = accepted["position_net_return"].to_numpy(float) * 1e4
        rows.append({
            "family": family,
            "month": month,
            "admitted_candidates": int(len(piece)),
            "portfolio_accepted_trades": int(len(accepted)),
            "acceptance_rate": float(len(accepted) / len(piece)) if len(piece) else float("nan"),
            "net_ev_bps_per_trade": float(np.mean(bps)) if len(bps) else float("nan"),
            "net_sum_bps": float(np.sum(bps)) if len(bps) else 0.0,
        })
    return pd.DataFrame(rows)


def _selected_cohorts(bcf_decisions: pd.DataFrame, current_decisions: pd.DataFrame) -> pd.DataFrame:
    left = bcf_decisions.loc[bcf_decisions["accepted"].astype(bool), ["candidate_id", "position_net_return"]].copy()
    right = current_decisions.loc[current_decisions["accepted"].astype(bool), ["candidate_id", "position_net_return"]].copy()
    merged = left.merge(right, on="candidate_id", how="outer", suffixes=("_bcf", "_current"), indicator=True)
    rows: list[dict[str, object]] = []
    cohort_names = {
        "left_only": "bcf_only_selected",
        "right_only": "current_v5_only_selected",
        "both": "both_selected",
    }
    for cohort, piece in merged.groupby("_merge", observed=True):
        values = piece["position_net_return_bcf"].combine_first(piece["position_net_return_current"]).to_numpy(float) * 1e4
        rows.append({
            "portfolio_selected_cohort": cohort_names[str(cohort)],
            "trades": int(len(piece)),
            "net_ev_bps_per_trade": float(np.mean(values)) if len(values) else float("nan"),
            "net_sum_bps": float(np.sum(values)),
        })
    return pd.DataFrame(rows)


def _aggregate_metrics(portfolios: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for family, piece in portfolios.groupby("arm", sort=True):
        trades = int(piece["accepted_rows"].sum())
        total_bps = float(piece["net_sum_bps_realised"].sum())
        rows.append({
            "family": family,
            "period": "2025-02_to_2026-07",
            "portfolio_accepted_trades": trades,
            "net_ev_bps_per_trade": total_bps / trades if trades else float("nan"),
            "net_sum_bps": total_bps,
            "worst_month_bps": float(piece["worst_month_bps"].min()),
            "worst_week_bps": float(piece["worst_week_bps"].min()),
            # Yearly portfolio simulations reset wallet at each calendar year;
            # this is the worst within-year drawdown, not one compounded series.
            "worst_year_reset_max_drawdown": float(piece["max_drawdown"].min()),
        })
    aggregate = pd.DataFrame(rows)
    baseline = aggregate.loc[aggregate["family"].eq("bcf")].iloc[0]
    deltas = aggregate.copy()
    for column in (
        "portfolio_accepted_trades", "net_ev_bps_per_trade", "net_sum_bps",
        "worst_month_bps", "worst_week_bps", "worst_year_reset_max_drawdown",
    ):
        deltas[f"delta_vs_bcf_{column}"] = deltas[column] - baseline[column]
    return aggregate, deltas


def _admission_cohorts(bcf: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    cols = ["candidate_id", "policy_net_bps", "mc1_expected_bps"]
    merged = bcf.loc[:, cols].merge(current.loc[:, cols], on="candidate_id", suffixes=("_bcf", "_current"), validate="one_to_one")
    bcf_admit = merged["mc1_expected_bps_bcf"].ge(50.0)
    current_admit = merged["mc1_expected_bps_current"].ge(50.0)
    cohort = np.select(
        [bcf_admit & current_admit, bcf_admit, current_admit],
        ["both_admit", "bcf_only_admit", "current_v5_only_admit"],
        default="both_reject",
    )
    merged["admission_cohort"] = cohort
    rows: list[dict[str, object]] = []
    for name, piece in merged.groupby("admission_cohort", sort=True):
        # The identity audit above has already established that the canonical
        # policy outcome is identical across the two suffix-qualified fields.
        values = piece["policy_net_bps_bcf"].to_numpy(float)
        rows.append({
            "admission_cohort": name,
            "overlap_candidates": int(len(piece)),
            "realised_net_ev_bps_per_trade": float(np.mean(values)),
            "realised_net_sum_bps": float(np.sum(values)),
            "bcf_mapped_ev_bps": float(piece["mc1_expected_bps_bcf"].mean()),
            "current_v5_mapped_ev_bps": float(piece["mc1_expected_bps_current"].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-run", required=True, type=Path)
    parser.add_argument("--current-run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    bcf_path = _prediction_path(args.bcf_run, "bcf")
    current_path = _prediction_path(args.current_run, "current_v5")
    bcf = pd.read_parquet(bcf_path)
    current = pd.read_parquet(current_path)
    audit = _policy_identity_audit(bcf, current)

    portfolios = pd.concat([
        pd.read_csv(args.bcf_run / "portfolio_metrics.csv"),
        pd.read_csv(args.current_run / "portfolio_metrics.csv"),
    ], ignore_index=True)
    bcf_decisions = pd.concat([pd.read_parquet(path) for path in sorted(args.bcf_run.glob("bcf_*_decisions.parquet"))], ignore_index=True)
    current_decisions = pd.concat([pd.read_parquet(path) for path in sorted(args.current_run.glob("current_v5_*_decisions.parquet"))], ignore_index=True)
    monthly = pd.concat([_month_metrics(bcf_decisions, "bcf"), _month_metrics(current_decisions, "current_v5")], ignore_index=True)
    admission = _admission_cohorts(bcf, current)
    selected = _selected_cohorts(bcf_decisions, current_decisions)
    aggregate, deltas = _aggregate_metrics(portfolios)
    raw_metrics = pd.concat([
        pd.read_parquet(args.bcf_run / "raw_score_metrics.parquet"),
        pd.read_parquet(args.current_run / "raw_score_metrics.parquet"),
    ], ignore_index=True)

    portfolios.to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    portfolios.to_csv(args.out_dir / "portfolio_metrics.csv", index=False)
    monthly.to_parquet(args.out_dir / "monthly_constrained_metrics.parquet", index=False)
    monthly.to_csv(args.out_dir / "monthly_constrained_metrics.csv", index=False)
    admission.to_parquet(args.out_dir / "admission_overlap_metrics.parquet", index=False)
    admission.to_csv(args.out_dir / "admission_overlap_metrics.csv", index=False)
    selected.to_parquet(args.out_dir / "portfolio_selected_overlap_metrics.parquet", index=False)
    selected.to_csv(args.out_dir / "portfolio_selected_overlap_metrics.csv", index=False)
    aggregate.to_parquet(args.out_dir / "aggregate_constrained_metrics.parquet", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_constrained_metrics.csv", index=False)
    deltas.to_parquet(args.out_dir / "delta_vs_bcf_metrics.parquet", index=False)
    deltas.to_csv(args.out_dir / "delta_vs_bcf_metrics.csv", index=False)
    raw_metrics.to_parquet(args.out_dir / "raw_score_metrics.parquet", index=False)
    raw_metrics.to_csv(args.out_dir / "raw_score_metrics.csv", index=False)
    manifest = {
        "schema": "strict_r3_score_family_matched_mc1_canonical_policy_comparison_v1",
        "status": "complete",
        "purpose": "immutable summary of separately materialised BCF/current-v5 MC1 runs on the shared canonical policy substrate",
        "bcf_run": {"path": str(args.bcf_run), "prediction_sha256": _sha256(bcf_path)},
        "current_v5_run": {"path": str(args.current_run), "prediction_sha256": _sha256(current_path)},
        "policy_identity_audit": audit,
        "fixed_contract": {
            "mc1": "frozen depth-2 MC1_d2; no retuning",
            "admission": "mc1_expected_bps >= +50",
            "portfolio": "long-only, 7x, 10% margin slots, two new entries per hour, eight concurrent, 80% wallet cap",
            "invalid_outcomes": "excluded before MC1 fitting and portfolio capacity allocation",
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", **audit}), flush=True)


if __name__ == "__main__":
    main()
