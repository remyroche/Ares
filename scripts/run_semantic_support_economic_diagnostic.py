#!/usr/bin/env python3
"""Evaluate semantic supportive controls on one pooled global candidate book.

This is an economic diagnostic, not a policy runner.  It deliberately keeps
timing/wait actions and portfolio constraints out of scope and reports future
oracles separately from learnable OOF controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_top_k(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    valid = frame[score].notna() & np.isfinite(frame[score].to_numpy(dtype=float))
    eligible = frame.loc[valid].sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort")
    count = max(1, int(np.ceil(len(eligible) * fraction)))
    return eligible.head(count)


def _profit_factor(net: pd.Series) -> float:
    positive = float(net[net > 0.0].sum())
    negative = float(-net[net < 0.0].sum())
    return positive / negative if negative > 0.0 else float("inf")


def _book_metrics(selected: pd.DataFrame) -> dict[str, Any]:
    net = pd.to_numeric(selected.execution_net_ev_12h, errors="coerce") * 10_000.0
    gross = pd.to_numeric(selected.execution_gross_ev_12h, errors="coerce") * 10_000.0
    cost = pd.to_numeric(selected.execution_cost_return, errors="coerce") * 10_000.0
    months = pd.to_datetime(selected["__ts__"], utc=True).dt.strftime("%Y-%m")
    return {
        "selected_rows": int(len(selected)),
        "net_bps": float(net.mean()),
        "gross_bps": float(gross.mean()),
        "cost_bps": float(cost.mean()),
        "win_rate": float((net > 0.0).mean()),
        "profit_factor": float(_profit_factor(net)),
        "months_selected": int(months.nunique()),
        "latest_month": str(months.max()) if len(months) else None,
        "latest_month_net_bps": float(net[months.eq(months.max())].mean()) if len(months) else float("nan"),
    }


def _bootstrap(selected: pd.DataFrame, *, reps: int, seed: int) -> dict[str, Any]:
    if selected.empty:
        return {"replicates": 0, "ci05_net_bps": float("nan"), "ci95_net_bps": float("nan"), "probability_positive": float("nan")}
    work = selected.copy()
    # Canonical UTC-date strings avoid mixing tz-aware pandas timestamps with
    # numpy datetime64 keys during resampling.
    work["_day"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m-%d")
    days = np.array(sorted(work["_day"].unique()), dtype=object)
    if len(days) < 2:
        return {"replicates": 0, "ci05_net_bps": float("nan"), "ci95_net_bps": float("nan"), "probability_positive": float("nan")}
    grouped = {day: (work.loc[work["_day"].eq(day), "execution_net_ev_12h"].to_numpy(float) * 10_000.0) for day in days}
    day_sums = np.array([values.sum() for values in grouped.values()], dtype=float)
    day_counts = np.array([len(values) for values in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.integers(0, len(days), size=len(days))
        values[i] = day_sums[sample].sum() / day_counts[sample].sum()
    return {
        "replicates": int(reps),
        "utc_day_blocks": int(len(days)),
        "ci05_net_bps": float(np.quantile(values, 0.05)),
        "ci95_net_bps": float(np.quantile(values, 0.95)),
        "probability_positive": float((values > 0.0).mean()),
    }


def _load_frame(*, ledger: Path, labels: Path, predictions: Path) -> pd.DataFrame:
    pred = pd.read_parquet(predictions)
    led = pd.read_parquet(ledger, columns=[
        "candidate_id", "side_name", "__symbol__", "execution_net_ev_12h",
        "execution_gross_ev_12h", "execution_cost_return",
    ])
    semantic = pd.read_parquet(labels)
    semantic = semantic.drop(columns=["decision_ts", "label_end_ts", "label_available_ts", "symbol", "side"], errors="ignore")
    frame = pred.merge(led, on="candidate_id", how="inner", validate="one_to_one")
    frame = frame.merge(semantic, on="candidate_id", how="inner", validate="one_to_one")
    if len(frame) != len(pred):
        raise ValueError("economic diagnostic lost OOF candidates during joins")
    return frame


def run(*, ledger: Path, labels: Path, predictions: Path, output: Path, bootstrap_reps: int = 200) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    frame = _load_frame(ledger=ledger, labels=labels, predictions=predictions)
    # C0-C4 are predeclared semantic controls.  The heads remain separate in
    # the OOF ledger; compositions are diagnostic and cannot be promoted.
    reach = frame["semantic_oof__meaningful_mfe_reach"].clip(0.0, 1.0)
    opportunity = frame["semantic_oof__opportunity_reach"].clip(0.0, 1.0)
    peak = frame["semantic_oof__conditional_peak_mfe"].clip(lower=0.0)
    mae = frame["semantic_oof__conditional_mae_before_mfe"].clip(lower=0.0)
    persistence = frame["semantic_oof__retention_persistence"].clip(0.0, 1.0)
    adverse = frame["semantic_oof__adverse"].clip(0.0, 1.0)
    frame["C0_opportunity_probability"] = opportunity
    frame["C1_meaningful_reach_probability"] = reach
    frame["C2_reach_x_peak"] = reach * peak
    frame["C3_reach_x_peak_minus_mae"] = reach * (peak - mae)
    frame["C4_reach_x_persistence_x_peak_minus_adverse"] = reach * persistence * peak * (1.0 - adverse)
    frame["O1_future_meaningful_reach"] = frame["target_meaningful_mfe_reached_12h"].astype(float)
    frame["O2_future_unconditional_peak"] = np.where(
        frame["target_meaningful_mfe_reached_12h"].astype(bool),
        pd.to_numeric(frame["target_peak_mfe_atr_given_meaningful_mfe"], errors="coerce"),
        0.0,
    )
    frame["O3_future_exact_net"] = pd.to_numeric(frame["execution_net_ev_12h"], errors="coerce")
    score_columns = [
        "C0_opportunity_probability", "C1_meaningful_reach_probability",
        "C2_reach_x_peak", "C3_reach_x_peak_minus_mae",
        "C4_reach_x_persistence_x_peak_minus_adverse",
        "O1_future_meaningful_reach", "O2_future_unconditional_peak", "O3_future_exact_net",
    ]
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for split, subset in (("development_oof", frame[frame.fold_order.eq(1)]), ("final_oos_diagnostic", frame[frame.fold_order.eq(2)])):
        expected_months = set(pd.to_datetime(subset.__ts__, utc=True).dt.strftime("%Y-%m").unique())
        for score in score_columns:
            for fraction in FRACTIONS:
                eligible = subset.loc[subset[score].notna() & np.isfinite(subset[score].to_numpy(float))]
                selected = _stable_top_k(eligible, score, fraction)
                metrics = _book_metrics(selected)
                selected_months = set(pd.to_datetime(selected.__ts__, utc=True).dt.strftime("%Y-%m").unique())
                summary_rows.append({
                    "split": split, "score": score, "fraction": fraction,
                    "population_rows": int(len(eligible)),
                    "expected_months": int(len(expected_months)),
                    "complete_month_coverage": expected_months.issubset(selected_months),
                    "acceptance_candidate": (
                        split == "development_oof"
                        and score.startswith("C")
                        and metrics["net_bps"] > 0.0
                        and metrics["latest_month_net_bps"] > 0.0
                    ),
                    **metrics,
                })
                bootstrap_rows.append({
                    "split": split, "score": score, "fraction": fraction,
                    **_bootstrap(selected, reps=bootstrap_reps, seed=20260801 + int(round(fraction * 100)) + len(score)),
                })
                if len(selected):
                    selected = selected.assign(_month=pd.to_datetime(selected.__ts__, utc=True).dt.strftime("%Y-%m"))
                    for (month, side), group in selected.groupby(["_month", "side_name"], observed=True):
                        detail_rows.append({
                            "split": split, "score": score, "fraction": fraction,
                            "scope": "month_side", "month": month, "side": side,
                            **_book_metrics(group),
                        })
    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        summary.to_parquet(stage / "global_topk_summary.parquet", index=False, compression="zstd")
        details.to_parquet(stage / "month_side_topk_metrics.parquet", index=False, compression="zstd")
        bootstrap.to_parquet(stage / "global_topk_day_bootstrap.parquet", index=False, compression="zstd")
        manifest = {
            "schema": "semantic_support_economic_diagnostic_v1",
            "status": "RESEARCH_ONLY_NO_PROMOTION",
            "portfolio_constraints_in_scope": False,
            "selection_policy": "one pooled global top-k over each score; no timestamp/side/asset quota",
            "development_selection_only": True,
            "fractions": list(FRACTIONS),
            "controls": {
                "C0": "clean opportunity probability",
                "C1": "meaningful-MFE reach probability",
                "C2": "reach probability times conditional peak MFE",
                "C3": "reach probability times conditional peak MFE minus conditional pre-MFE MAE",
                "C4": "reach probability times persistence times peak, downweighted by adverse probability",
            },
            "oracles": {
                "O1": "future meaningful-MFE reach",
                "O2": "future unconditional peak MFE",
                "O3": "future exact execution net",
            },
            "inputs": {
                "ledger": {"path": str(ledger), "sha256": _sha256(ledger)},
                "labels": {"path": str(labels), "sha256": _sha256(labels)},
                "predictions": {"path": str(predictions), "sha256": _sha256(predictions)},
            },
            "outputs": {
                "global_topk_summary": "global_topk_summary.parquet",
                "month_side_topk_metrics": "month_side_topk_metrics.parquet",
                "global_topk_day_bootstrap": "global_topk_day_bootstrap.parquet",
            },
            "rows": int(len(frame)),
            "bootstrap_replicates": int(bootstrap_reps),
        }
        manifest["outputs_sha256"] = {name: _sha256(stage / name) for name in manifest["outputs"].values()}
        manifest["runner"] = {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__))}
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run(
        ledger=args.ledger,
        labels=args.labels,
        predictions=args.predictions,
        output=args.output,
        bootstrap_reps=args.bootstrap_reps,
    ), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
