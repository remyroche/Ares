"""Liquidity Regime Run Selector

This script scans liquidity cluster quality reports (liquidity_cluster_quality_*.csv)
produced by ml_liquidity_regime_step and ranks runs according to:

1. Hard constraints:
   - All required regimes (default: 0,1,2,3) exist with minimum support.
2. Soft scoring:
   - Overall quality score
   - Effort/result CoV separation score
   - Returns CoV separation score
   - Class balance score

It does **not** change any labels or models; it only evaluates and ranks
existing runs using the CoV-aware diagnostics.

Usage (from project root):

    python3 scripts/liquidity_regime_run_selector.py \
        --symbol ETHUSDT \
        --outcomes-dir outcomes \
        --min-support-share 0.01 \
        --required-regimes 0,1,2,3 \
        --top-k 10

"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank liquidity regime runs by CoV-aware quality metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument(
        "--outcomes-dir",
        type=str,
        default="outcomes",
        help="Directory containing liquidity_cluster_quality_* CSV reports",
    )
    parser.add_argument(
        "--min-support-share",
        type=float,
        default=0.01,
        help="Minimum share of total samples required per regime",
    )
    parser.add_argument(
        "--required-regimes",
        type=str,
        default="0,1,2,3",
        help="Comma-separated list of required regime IDs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top runs to display",
    )

    return parser.parse_args()


def load_run_from_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    # Expect exactly one row per report
    return df.iloc[0]


def compute_support(row: pd.Series, required_regimes: List[int]) -> Tuple[bool, Dict[int, float]]:
    total = float(row.get("n_samples", 0.0))
    supports: Dict[int, float] = {}

    if total <= 0:
        return False, {k: 0.0 for k in required_regimes}

    for k in required_regimes:
        col = f"regime_{k}_n_samples"
        n_k = float(row.get(col, 0.0))
        supports[k] = n_k / total

    # support_ok will be checked by caller against threshold
    return True, supports


def score_run(
    row: pd.Series,
    supports: Dict[int, float],
    support_threshold: float,
    required_regimes: List[int],
) -> Tuple[float, bool, str]:
    """Compute scalar score and support_ok flag for a single run.

    Returns (score, support_ok, reason_if_rejected).
    """
    # Check support constraints
    support_ok = True
    reason = ""

    for k in required_regimes:
        if supports.get(k, 0.0) < support_threshold:
            support_ok = False
            reason = f"regime {k} support {supports.get(k, 0.0):.4f} < {support_threshold:.4f}"
            break

    overall = float(row.get("overall_quality_score", 0.0))
    cov_effort = float(row.get("effort_result_cov_separation_score", 0.0))
    cov_ret = float(row.get("returns_cov_separation_score", 0.0))
    balance = float(row.get("class_balance_score", 0.0))

    if not support_ok:
        # Still compute a diagnostic score but push it far down
        score = -1e9 + overall
    else:
        # Soft scoring: combine CoV separation and balance with overall quality
        score = (
            0.30 * overall
            + 0.25 * cov_effort
            + 0.25 * cov_ret
            + 0.20 * balance
        )

    return score, support_ok, reason


def main() -> None:
    args = parse_args()

    outcomes_dir = Path(args.outcomes_dir)
    if not outcomes_dir.exists():
        raise SystemExit(f"Outcomes directory does not exist: {outcomes_dir}")

    required_regimes = [int(x) for x in args.required_regimes.split(",") if x.strip() != ""]

    pattern = str(outcomes_dir / f"liquidity_cluster_quality_{args.symbol}_*.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise SystemExit(f"No liquidity_cluster_quality CSVs found for symbol {args.symbol} in {outcomes_dir}")

    runs: List[Dict[str, object]] = []

    for p in paths:
        path = Path(p)
        try:
            row = load_run_from_csv(path)
        except Exception as e:
            print(f"Skipping {path}: failed to load ({e})")
            continue

        ok_total, supports = compute_support(row, required_regimes)
        if not ok_total:
            score = -1e9
            support_ok = False
            reason = "no samples"
        else:
            score, support_ok, reason = score_run(
                row,
                supports,
                support_threshold=args.min_support_share,
                required_regimes=required_regimes,
            )

        # Extract timestamp from filename suffix
        ts = path.stem.split("_")[-1]

        run_info: Dict[str, object] = {
            "path": str(path),
            "timestamp": ts,
            "score": score,
            "support_ok": support_ok,
            "reject_reason": reason,
            "overall_quality_score": float(row.get("overall_quality_score", 0.0)),
            "effort_result_cov_separation_score": float(row.get("effort_result_cov_separation_score", 0.0)),
            "returns_cov_separation_score": float(row.get("returns_cov_separation_score", 0.0)),
            "class_balance_score": float(row.get("class_balance_score", 0.0)),
            "n_samples": float(row.get("n_samples", 0.0)),
        }

        # Add per-regime supports
        for k in required_regimes:
            run_info[f"support_regime_{k}"] = supports.get(k, 0.0)

        runs.append(run_info)

    if not runs:
        raise SystemExit("No valid runs found.")

    runs_df = pd.DataFrame(runs)
    runs_df = runs_df.sort_values("score", ascending=False)

    # Summary
    total_runs = len(runs_df)
    n_support_ok = int(runs_df["support_ok"].sum())

    print("\n=== Liquidity Regime Run Selector ===")
    print(f"Symbol: {args.symbol}")
    print(f"Total runs: {total_runs}")
    print(f"Runs meeting support constraints: {n_support_ok} / {total_runs}")
    print(f"Required regimes: {required_regimes}")
    print(f"Min support share per regime: {args.min_support_share:.4f}\n")

    # Display top-K table
    cols = [
        "timestamp",
        "score",
        "support_ok",
        "overall_quality_score",
        "effort_result_cov_separation_score",
        "returns_cov_separation_score",
        "class_balance_score",
    ]
    for k in required_regimes:
        cols.append(f"support_regime_{k}")

    print("Top runs:")
    print(runs_df[cols].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
