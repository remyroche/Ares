#!/usr/bin/env python3
"""Replay mapped execution-EV arms in the exact inference spread universe.

This is deliberately a *retrospective diagnostic*.  The baseline file used by
the live universe contract was produced in June--July 2026, so applying it to
the May--July OOS candidates is not point-in-time historical evidence.  It is
nevertheless the right way to measure the effect of the currently deployed
static inference universe, provided results are not promoted as causal
training or validation evidence.

For every arm and evaluation window the script holds the mapped score and the
one pooled global top-10% selection contract fixed, and reports three books:

* unrestricted global top 10%;
* the inference-eligible slice of that original book; and
* a separately reranked global top 10% within the eligible universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.universe import (
    DEFAULT_SPREAD_COST_BLACKLIST_THRESHOLD_BPS,
    _normalize_symbol,
    load_spread_cost_excluded_symbols,
)


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SCORE = "canonical_recent_ev_score"
TARGET_COLUMNS = (
    "execution_net_ev_12h",
    "execution_gross_ev_12h",
    "execution_cost_return",
)
DEFAULT_ARMS = (
    "direct_net",
    "hurdle_prob",
    "competing_clean_probability",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pooled_top_fraction(
    frame: pd.DataFrame, *, score: str = SCORE, fraction: float = 0.10
) -> pd.DataFrame:
    """Return one pooled score-ranked book, with no timestamp or side quota."""
    if frame.empty:
        return frame.copy()
    return frame.nlargest(max(1, int(np.ceil(float(fraction) * len(frame)))), score).copy()


def normalized_spread_eligibility(
    symbols: pd.Series, excluded_symbols: set[str]
) -> pd.Series:
    """Match the exact inference contract, including underscore/slash aliases."""
    normalized_excluded = {_normalize_symbol(symbol) for symbol in excluded_symbols}
    return ~symbols.astype(str).map(_normalize_symbol).isin(normalized_excluded)


def summarize(
    selected: pd.DataFrame,
    *,
    candidate_rows: int,
    unrestricted_candidate_rows: int,
    original_top10_rows: int,
    book: str,
) -> dict[str, Any]:
    """Return the economic summary on exact frozen-policy targets."""
    result: dict[str, Any] = {
        "book": book,
        "candidate_rows": int(candidate_rows),
        "unrestricted_candidate_rows": int(unrestricted_candidate_rows),
        "original_global_top10_rows": int(original_top10_rows),
        "selected_rows": int(len(selected)),
        "selected_fraction_of_book_candidates": len(selected) / max(candidate_rows, 1),
        "selected_fraction_of_unrestricted_candidates": len(selected)
        / max(unrestricted_candidate_rows, 1),
        "mean_net_ev_bps": None,
        "mean_gross_ev_bps": None,
        "mean_cost_bps": None,
        "positive_net_rate": None,
        "long_rows": 0,
        "short_rows": 0,
    }
    if selected.empty:
        return result
    result.update(
        {
            "mean_net_ev_bps": float(selected[TARGET_COLUMNS[0]].mean() * 1e4),
            "mean_gross_ev_bps": float(selected[TARGET_COLUMNS[1]].mean() * 1e4),
            "mean_cost_bps": float(selected[TARGET_COLUMNS[2]].mean() * 1e4),
            "positive_net_rate": float(selected[TARGET_COLUMNS[0]].gt(0.0).mean()),
            "long_rows": int(selected["side_name"].eq("long").sum()),
            "short_rows": int(selected["side_name"].eq("short").sum()),
        }
    )
    return result


def _read_predictions(path: Path, arms: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {*IDENTITY, "window", "arm", SCORE}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")
    selected = frame.loc[frame["arm"].isin(arms)].copy()
    absent = sorted(set(arms) - set(selected["arm"].unique()))
    if absent:
        raise ValueError(f"requested arms absent from predictions: {absent}")
    return selected


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    return result


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite {args.output_dir}")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    arms = tuple(dict.fromkeys(args.arms))
    predictions = _prepare(_read_predictions(args.predictions, arms))
    targets = _prepare(
        pd.read_parquet(args.targets, columns=[*IDENTITY, *TARGET_COLUMNS])
    )
    if predictions.duplicated([*IDENTITY, "window", "arm"], keep=False).any():
        raise ValueError("duplicate prediction identity within window/arm")
    if targets.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("duplicate target identity")
    joined = predictions.merge(targets, on=list(IDENTITY), how="left", validate="many_to_one")
    if joined[list(TARGET_COLUMNS)].isna().any().any():
        raise ValueError("prediction-to-exact-policy-target join is incomplete")
    if not np.isfinite(joined[[SCORE, *TARGET_COLUMNS]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite mapped score or exact-policy target")

    # Deliberately call the live implementation rather than reproducing its
    # threshold or symbol parsing here.  The explicit file/threshold freeze
    # makes the diagnostic reproducible regardless of caller environment.
    excluded = load_spread_cost_excluded_symbols(
        path=str(args.spread_baseline), threshold_bps=float(args.threshold_bps)
    )
    if not excluded:
        raise ValueError(
            "inference spread exclusion is empty; check baseline, threshold, and "
            "EPM_DISABLE_SPREAD_BLACKLIST before accepting a no-op diagnostic"
        )
    joined["normalized_inference_symbol"] = joined["__symbol__"].map(_normalize_symbol)
    joined["inference_spread_eligible"] = normalized_spread_eligibility(
        joined["__symbol__"], excluded
    )

    rows: list[dict[str, Any]] = []
    assignments: list[pd.DataFrame] = []
    for (window, arm), group in joined.groupby(["window", "arm"], sort=True):
        unrestricted = pooled_top_fraction(group, fraction=args.fraction)
        eligible = group.loc[group["inference_spread_eligible"]].copy()
        eligible_slice = unrestricted.loc[
            unrestricted["inference_spread_eligible"]
        ].copy()
        eligible_reranked = pooled_top_fraction(eligible, fraction=args.fraction)
        for book, selected, candidates in (
            ("unrestricted_global_top10", unrestricted, group),
            (
                "eligible_slice_of_original_global_top10",
                eligible_slice,
                group,
            ),
            ("eligible_universe_reranked_global_top10", eligible_reranked, eligible),
        ):
            rows.append(
                {
                    "window": str(window),
                    "arm": str(arm),
                    "eligible_candidate_rows": int(len(eligible)),
                    "excluded_candidate_rows": int(len(group) - len(eligible)),
                    **summarize(
                        selected,
                        candidate_rows=len(candidates),
                        unrestricted_candidate_rows=len(group),
                        original_top10_rows=len(unrestricted),
                        book=book,
                    ),
                }
            )
            tagged = selected.loc[
                :,
                [
                    *IDENTITY,
                    "normalized_inference_symbol",
                    "inference_spread_eligible",
                    SCORE,
                    *TARGET_COLUMNS,
                ],
            ].copy()
            tagged["window"] = str(window)
            tagged["arm"] = str(arm)
            tagged["book"] = book
            assignments.append(tagged)

    baseline = pd.read_csv(args.spread_baseline)
    if "symbol" not in baseline or "average_spread_bps" not in baseline:
        raise ValueError("spread baseline must have symbol and average_spread_bps columns")
    baseline["normalized_inference_symbol"] = baseline["symbol"].map(_normalize_symbol)
    universe = baseline.loc[:, ["symbol", "normalized_inference_symbol", "average_spread_bps"]].copy()
    universe["inference_spread_eligible"] = ~universe["normalized_inference_symbol"].isin(
        {_normalize_symbol(symbol) for symbol in excluded}
    )
    universe = universe.sort_values("normalized_inference_symbol").reset_index(drop=True)
    baseline_symbols = set(universe["normalized_inference_symbol"])
    candidate_symbols = set(joined["normalized_inference_symbol"])
    input_symbol_coverage = {
        "unique_candidate_symbols": int(len(candidate_symbols)),
        "candidate_symbols_in_spread_baseline": int(len(candidate_symbols & baseline_symbols)),
        "candidate_symbols_absent_from_spread_baseline": int(
            len(candidate_symbols - baseline_symbols)
        ),
        "candidate_symbols_excluded_by_current_contract": int(
            len(candidate_symbols & {_normalize_symbol(symbol) for symbol in excluded})
        ),
        "interpretation": (
            "If candidate_symbols_excluded_by_current_contract is zero, the "
            "evaluated mapped cohort was already inside the current static "
            "inference universe; this ablation cannot estimate removal effect "
            "without an earlier, pre-universe candidate ledger."
        ),
    }

    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.csv"
    selected_path = args.output_dir / "selected_rows.parquet"
    universe_path = args.output_dir / "inference_spread_universe.csv"
    manifest_path = args.output_dir / "manifest.json"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.concat(assignments, ignore_index=True).to_parquet(selected_path, index=False, compression="zstd")
    universe.to_csv(universe_path, index=False)
    manifest: Mapping[str, Any] = {
        "schema": "execution_ev_inference_spread_universe_ablation_v1",
        "status": "diagnostic_non_PIT_not_promotion_evidence",
        "classification": {
            "historical_causality": "non_PIT_retrospective_only",
            "reason": (
                "The current spread baseline was produced June-July 2026, after "
                "some evaluated May-July decisions. It can measure the current "
                "inference universe but cannot validate historical training or "
                "promotion causally."
            ),
        },
        "contract": {
            "inference_implementation": "extreme_price_movements.universe.load_spread_cost_excluded_symbols",
            "symbol_normalization": "extreme_price_movements.universe._normalize_symbol",
            "baseline_value_column": "average_spread_bps",
            "exclusion_rule": "average_spread_bps > threshold_bps",
            "threshold_bps": float(args.threshold_bps),
            "ranking": f"one pooled global top {args.fraction:.0%} after canonical recent-EV mapping; no timestamp or side quotas",
            "books": [
                "unrestricted_global_top10",
                "eligible_slice_of_original_global_top10",
                "eligible_universe_reranked_global_top10",
            ],
            "exact_target": "frozen deployed-policy gross, cost, and net 12h replay",
        },
        "inputs": {
            "predictions": str(args.predictions),
            "predictions_sha256": sha256(args.predictions),
            "targets": str(args.targets),
            "targets_sha256": sha256(args.targets),
            "spread_baseline": str(args.spread_baseline),
            "spread_baseline_sha256": sha256(args.spread_baseline),
            "spread_baseline_rows": int(len(baseline)),
            "excluded_symbol_count": int(len(excluded)),
            "input_symbol_coverage": input_symbol_coverage,
            "arms": list(arms),
        },
        "outputs": {
            "metrics": str(metrics_path),
            "metrics_sha256": sha256(metrics_path),
            "selected_rows": str(selected_path),
            "selected_rows_sha256": sha256(selected_path),
            "inference_spread_universe": str(universe_path),
            "inference_spread_universe_sha256": sha256(universe_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"metrics": metrics_path, "selected_rows": selected_path, "manifest": manifest_path}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--targets", type=Path, required=True)
    result.add_argument("--spread-baseline", type=Path, required=True)
    result.add_argument(
        "--threshold-bps", type=float, default=DEFAULT_SPREAD_COST_BLACKLIST_THRESHOLD_BPS
    )
    result.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    result.add_argument("--fraction", type=float, default=0.10)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps({key: str(value) for key, value in run(args).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
