#!/usr/bin/env python3
"""Fast HPO over train_meta composite selector scores.

This consumes an existing month-forward prediction parquet from
``run_direct_context_risk_aware_train_meta_smoke.py``.  It does not refit model
heads; it searches leakage-safe meta score combinations using only columns
available at selection time:

* existing OOF/month-forward selector scores;
* predicted EV/full-SL/timeout/clean heads;
* side and source archetype identity.

The objective is top-k precision/EV/path-quality oriented, not AUC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_direct_context_risk_aware_train_meta_smoke import (  # noqa: E402
    TOP_FRACS,
    _json_safe,
    _topk_metrics,
)


DEFAULT_SMOKE_DIR = Path(
    "data_perp/reports/contextual_tp_sl_ablation_workflow_v14_runtime_health_20260701/"
    "direct_cross_asset_meta_context_v1/train_meta_direct_context_feature_set_v1/"
    "train_meta_interaction_context_feature_set_v1/risk_aware_train_meta_smoke_v1"
)
DEFAULT_PREDICTIONS = DEFAULT_SMOKE_DIR / "risk_aware_train_meta_predictions.parquet"
DEFAULT_OUT_DIR = DEFAULT_SMOKE_DIR.parent / "composite_selector_hpo_v1"
REFERENCE_SELECTOR = "s12_ev_clean_strong_risk"
LONG_SELECTOR_CANDIDATES = (
    "s14_cell_prior_fullsl_s12",
    "s15_cell_prior_fullsl_timeout_s12",
    "s16_cell_prior_clean_risk_s12",
    "s18_long_cell_prior_ev_fullsl_s12",
)
STAGE1_ALPHAS = (0.25, 0.50, 0.75, 1.0)
FULL_SL_PENALTIES = (0.0, 0.010, 0.020, 0.035)
TIMEOUT_PENALTIES = (0.0, 0.006, 0.012, 0.020)
LONG_DIST_TIMEOUT_PENALTIES = (0.0, 0.006, 0.012)
TOP_STAGE1_ARMS = 8
PRECISION_SAFE_MAX_PRECISION_LOSS = 0.0005
PRECISION_SAFE_MAX_WEIGHTED_PRECISION_LOSS = 0.0010
PRECISION_SAFE_MAX_FULL_SL_INCREASE = 0.0030
PRECISION_SAFE_MAX_TIMEOUT_INCREASE = 0.0030
PRECISION_SAFE_MAX_WORST_EV_LOSS = 0.0010


def _score_col(selector: str) -> str:
    return f"score_{selector}"


def _required_columns(long_selectors: tuple[str, ...]) -> list[str]:
    cols = [
        "month",
        "side_name",
        "source_archetype",
        "exec_ev_after_1pct_cost",
        "full_sl",
        "timeout",
        "clean_exec_proxy",
        "pred_ev",
        "pred_full_sl",
        "pred_timeout",
        "pred_clean",
        _score_col(REFERENCE_SELECTOR),
    ]
    cols.extend(_score_col(selector) for selector in long_selectors)
    return cols


def _candidate_score(
    frame: pd.DataFrame,
    *,
    long_selector: str,
    alpha: float,
    full_sl_penalty: float,
    timeout_penalty: float,
    long_dist_timeout_penalty: float,
) -> pd.Series:
    fallback = pd.to_numeric(frame[_score_col(REFERENCE_SELECTOR)], errors="coerce")
    focused = pd.to_numeric(frame[_score_col(long_selector)], errors="coerce")
    side = frame["side_name"].astype(str)
    is_long = side.eq("long")
    is_long_dist = is_long & frame["source_archetype"].astype(str).eq("long_dist")
    blended = fallback.mul(1.0 - float(alpha)).add(focused.mul(float(alpha)), fill_value=np.nan)
    score = fallback.where(~is_long | focused.isna(), blended)
    long_penalty = (
        float(full_sl_penalty) * pd.to_numeric(frame["pred_full_sl"], errors="coerce").fillna(0.0)
        + float(timeout_penalty) * pd.to_numeric(frame["pred_timeout"], errors="coerce").fillna(0.0)
    )
    score = score - long_penalty.where(is_long, 0.0)
    score = score - (
        float(long_dist_timeout_penalty)
        * pd.to_numeric(frame["pred_timeout"], errors="coerce").fillna(0.0)
    ).where(is_long_dist, 0.0)
    return score.astype("float32")


def _metrics_for_score(
    frame: pd.DataFrame,
    *,
    score: pd.Series,
    selector_name: str,
    min_group_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    working["_hpo_score"] = score
    aggregate = _topk_metrics(
        working,
        selector_name=selector_name,
        score_col="_hpo_score",
        guards={},
        group_cols=["month"],
        min_group_rows=min_group_rows,
    )
    by_cell = _topk_metrics(
        working,
        selector_name=selector_name,
        score_col="_hpo_score",
        guards={},
        group_cols=["month", "side_name", "source_archetype"],
        min_group_rows=min_group_rows,
    )
    return aggregate, by_cell


def _objective(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty:
        return {
            "objective": float("-inf"),
            "top10_ev": float("nan"),
            "top10_precision": float("nan"),
            "top10_full_sl": float("nan"),
            "top10_timeout": float("nan"),
            "worst_top10_ev": float("nan"),
        }
    score_parts = []
    weights = {0.10: 1.00, 0.20: 0.55, 0.30: 0.30}
    for frac, weight in weights.items():
        cur = metrics[np.isclose(pd.to_numeric(metrics["top_frac"], errors="coerce"), frac)]
        if cur.empty:
            continue
        ev = float(pd.to_numeric(cur["mean_ev_after_1pct"], errors="coerce").mean())
        precision = float(pd.to_numeric(cur["precision_positive_ev"], errors="coerce").mean())
        weighted_precision = float(pd.to_numeric(cur["ev_weighted_precision"], errors="coerce").mean())
        full_sl = float(pd.to_numeric(cur["full_sl_rate"], errors="coerce").mean())
        timeout = float(pd.to_numeric(cur["timeout_rate"], errors="coerce").mean())
        clean = float(pd.to_numeric(cur["clean_exec_proxy_rate"], errors="coerce").mean())
        # Precision is the primary objective because the downstream system
        # trades top-k buckets.  EV remains in the objective, but as a
        # secondary tie-breaker/quality term rather than the dominant force.
        score_parts.append(
            float(weight)
            * (
                0.0060 * weighted_precision
                + 0.0040 * precision
                + 0.1500 * ev
                + 0.0010 * clean
                - 0.0040 * full_sl
                - 0.0030 * timeout
            )
        )
    top10 = metrics[np.isclose(pd.to_numeric(metrics["top_frac"], errors="coerce"), 0.10)].copy()
    worst_top10_ev = float(pd.to_numeric(top10["mean_ev_after_1pct"], errors="coerce").min()) if not top10.empty else float("nan")
    downside_penalty = 0.40 * abs(min(0.0, worst_top10_ev)) if np.isfinite(worst_top10_ev) else 0.0
    objective = float(np.nansum(score_parts) - downside_penalty)
    return {
        "objective": objective,
        "top10_ev": float(pd.to_numeric(top10["mean_ev_after_1pct"], errors="coerce").mean()) if not top10.empty else float("nan"),
        "top10_precision": float(pd.to_numeric(top10["precision_positive_ev"], errors="coerce").mean()) if not top10.empty else float("nan"),
        "top10_weighted_precision": float(pd.to_numeric(top10["ev_weighted_precision"], errors="coerce").mean()) if not top10.empty else float("nan"),
        "top10_full_sl": float(pd.to_numeric(top10["full_sl_rate"], errors="coerce").mean()) if not top10.empty else float("nan"),
        "top10_timeout": float(pd.to_numeric(top10["timeout_rate"], errors="coerce").mean()) if not top10.empty else float("nan"),
        "top10_clean": float(pd.to_numeric(top10["clean_exec_proxy_rate"], errors="coerce").mean()) if not top10.empty else float("nan"),
        "worst_top10_ev": worst_top10_ev,
    }


def _cell_summary(by_cell: pd.DataFrame) -> pd.DataFrame:
    if by_cell.empty:
        return pd.DataFrame()
    top10 = by_cell[np.isclose(pd.to_numeric(by_cell["top_frac"], errors="coerce"), 0.10)].copy()
    if top10.empty:
        return pd.DataFrame()
    return top10.groupby(["selector", "side_name", "source_archetype"], as_index=False).agg(
        months=("month", "nunique"),
        precision_positive_ev=("precision_positive_ev", "mean"),
        ev_weighted_precision=("ev_weighted_precision", "mean"),
        mean_ev_after_1pct=("mean_ev_after_1pct", "mean"),
        full_sl_rate=("full_sl_rate", "mean"),
        timeout_rate=("timeout_rate", "mean"),
        clean_exec_proxy_rate=("clean_exec_proxy_rate", "mean"),
    )


def _run_arm(
    frame: pd.DataFrame,
    *,
    arm_id: str,
    stage: str,
    long_selector: str,
    alpha: float,
    full_sl_penalty: float,
    timeout_penalty: float,
    long_dist_timeout_penalty: float,
    min_group_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    score = _candidate_score(
        frame,
        long_selector=long_selector,
        alpha=alpha,
        full_sl_penalty=full_sl_penalty,
        timeout_penalty=timeout_penalty,
        long_dist_timeout_penalty=long_dist_timeout_penalty,
    )
    aggregate, by_cell = _metrics_for_score(frame, score=score, selector_name=arm_id, min_group_rows=min_group_rows)
    obj = _objective(aggregate)
    record = {
        "arm_id": arm_id,
        "stage": stage,
        "long_selector": long_selector,
        "alpha": float(alpha),
        "full_sl_penalty": float(full_sl_penalty),
        "timeout_penalty": float(timeout_penalty),
        "long_dist_timeout_penalty": float(long_dist_timeout_penalty),
        **obj,
    }
    return record, aggregate, by_cell


def _write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    trials: pd.DataFrame,
    aggregate: pd.DataFrame,
    cell_summary: pd.DataFrame,
) -> None:
    best = trials.sort_values("objective", ascending=False).head(20)
    recommended = pd.DataFrame([manifest["recommended_arm"]]) if manifest.get("recommended_arm") else pd.DataFrame()
    top10 = aggregate[np.isclose(pd.to_numeric(aggregate["top_frac"], errors="coerce"), 0.10)].copy()
    top10_summary = top10.groupby("selector", as_index=False).agg(
        months=("month", "nunique"),
        precision_positive_ev=("precision_positive_ev", "mean"),
        ev_weighted_precision=("ev_weighted_precision", "mean"),
        mean_ev_after_1pct=("mean_ev_after_1pct", "mean"),
        full_sl_rate=("full_sl_rate", "mean"),
        timeout_rate=("timeout_rate", "mean"),
        clean_exec_proxy_rate=("clean_exec_proxy_rate", "mean"),
    )
    top10_summary = top10_summary.merge(
        trials[["arm_id", "objective", "stage", "long_selector", "alpha", "full_sl_penalty", "timeout_penalty", "long_dist_timeout_penalty"]],
        left_on="selector",
        right_on="arm_id",
        how="left",
    ).sort_values("objective", ascending=False)
    long_cells = cell_summary[cell_summary["side_name"].astype(str).eq("long")].copy() if not cell_summary.empty else pd.DataFrame()
    lines = [
        "# Train Meta Composite Selector HPO",
        "",
        "## Scope",
        "",
        f"- Prediction input: `{manifest['prediction_path']}`",
        f"- Trials: `{manifest['trial_count']}`",
        "- Search is hierarchical: long-selector/blend first, then long risk penalties for the best arms.",
        "- Objective uses top10/top20/top30 EV, precision, weighted precision, full-SL, timeout, clean proxy, and worst-month EV.",
        "",
        "## Best Trials",
        "",
        best.to_markdown(index=False) if not best.empty else "No trials.",
        "",
        "## Recommended Arm",
        "",
        recommended.to_markdown(index=False) if not recommended.empty else "No recommended arm.",
        "",
        "## Top10 Metrics By Arm",
        "",
        top10_summary.head(30).to_markdown(index=False) if not top10_summary.empty else "No top10 metrics.",
        "",
        "## Long Cell Top10 Metrics",
        "",
        long_cells.to_markdown(index=False) if not long_cells.empty else "No long cell metrics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommended_arm(trials: pd.DataFrame) -> dict[str, Any]:
    if trials.empty or not trials["arm_id"].eq("ref_s12").any():
        return _json_safe(trials.iloc[0].to_dict()) if not trials.empty else {}
    ref = trials[trials["arm_id"].eq("ref_s12")].iloc[0]
    candidates = trials[~trials["arm_id"].eq("ref_s12")].copy()
    if candidates.empty:
        out = ref.to_dict()
        out["recommendation_reason"] = "reference_only"
        return _json_safe(out)
    mask = (
        candidates["top10_precision"].ge(float(ref["top10_precision"]) - PRECISION_SAFE_MAX_PRECISION_LOSS)
        & candidates["top10_weighted_precision"].ge(
            float(ref["top10_weighted_precision"]) - PRECISION_SAFE_MAX_WEIGHTED_PRECISION_LOSS
        )
        & candidates["top10_ev"].ge(float(ref["top10_ev"]))
        & candidates["top10_full_sl"].le(float(ref["top10_full_sl"]) + PRECISION_SAFE_MAX_FULL_SL_INCREASE)
        & candidates["top10_timeout"].le(float(ref["top10_timeout"]) + PRECISION_SAFE_MAX_TIMEOUT_INCREASE)
        & candidates["worst_top10_ev"].ge(float(ref["worst_top10_ev"]) - PRECISION_SAFE_MAX_WORST_EV_LOSS)
    )
    safe = candidates[mask].copy()
    if safe.empty:
        out = ref.to_dict()
        out["recommendation_reason"] = "no_candidate_met_precision_safe_constraints"
        return _json_safe(out)
    out = safe.sort_values("objective", ascending=False).iloc[0].to_dict()
    out["recommendation_reason"] = "best_precision_safe_candidate"
    return _json_safe(out)


def run(
    *,
    prediction_path: Path,
    output_dir: Path,
    long_selectors: tuple[str, ...],
    min_group_rows: int,
    top_stage1_arms: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(prediction_path)
    missing = sorted(set(_required_columns(long_selectors)).difference(frame.columns))
    if missing:
        raise ValueError(f"prediction parquet missing required columns: {missing}")
    # Keep only columns used for scoring/eval to reduce memory churn.
    keep_cols = list(dict.fromkeys(_required_columns(long_selectors)))
    frame = frame[keep_cols].copy()

    trial_records: list[dict[str, Any]] = []
    aggregate_parts: list[pd.DataFrame] = []
    cell_parts: list[pd.DataFrame] = []

    # Reference arm for direct comparison.
    ref_aggregate, ref_cell = _metrics_for_score(
        frame,
        score=pd.to_numeric(frame[_score_col(REFERENCE_SELECTOR)], errors="coerce").astype("float32"),
        selector_name="ref_s12",
        min_group_rows=min_group_rows,
    )
    ref_record = {"arm_id": "ref_s12", "stage": "reference", "long_selector": REFERENCE_SELECTOR, "alpha": 0.0, "full_sl_penalty": 0.0, "timeout_penalty": 0.0, "long_dist_timeout_penalty": 0.0, **_objective(ref_aggregate)}
    trial_records.append(ref_record)
    aggregate_parts.append(ref_aggregate)
    cell_parts.append(ref_cell)

    stage1_records: list[dict[str, Any]] = []
    for long_selector in long_selectors:
        for alpha in STAGE1_ALPHAS:
            arm_id = f"hpo1_{long_selector}_a{alpha:.2f}".replace(".", "p")
            record, aggregate, by_cell = _run_arm(
                frame,
                arm_id=arm_id,
                stage="stage1_blend",
                long_selector=long_selector,
                alpha=alpha,
                full_sl_penalty=0.0,
                timeout_penalty=0.0,
                long_dist_timeout_penalty=0.0,
                min_group_rows=min_group_rows,
            )
            stage1_records.append(record)
            trial_records.append(record)
            aggregate_parts.append(aggregate)
            cell_parts.append(by_cell)

    stage1_best = sorted(stage1_records, key=lambda row: row["objective"], reverse=True)[: int(top_stage1_arms)]
    for parent_idx, parent in enumerate(stage1_best):
        for full_sl_penalty in FULL_SL_PENALTIES:
            for timeout_penalty in TIMEOUT_PENALTIES:
                for long_dist_timeout_penalty in LONG_DIST_TIMEOUT_PENALTIES:
                    if full_sl_penalty == timeout_penalty == long_dist_timeout_penalty == 0.0:
                        continue
                    arm_id = (
                        f"hpo2_{parent_idx:02d}_{parent['long_selector']}_a{parent['alpha']:.2f}"
                        f"_fs{full_sl_penalty:.3f}_to{timeout_penalty:.3f}_ldto{long_dist_timeout_penalty:.3f}"
                    ).replace(".", "p")
                    record, aggregate, by_cell = _run_arm(
                        frame,
                        arm_id=arm_id,
                        stage="stage2_risk_penalty",
                        long_selector=str(parent["long_selector"]),
                        alpha=float(parent["alpha"]),
                        full_sl_penalty=float(full_sl_penalty),
                        timeout_penalty=float(timeout_penalty),
                        long_dist_timeout_penalty=float(long_dist_timeout_penalty),
                        min_group_rows=min_group_rows,
                    )
                    trial_records.append(record)
                    aggregate_parts.append(aggregate)
                    cell_parts.append(by_cell)

    trials = pd.DataFrame(trial_records).sort_values("objective", ascending=False).reset_index(drop=True)
    aggregate = pd.concat(aggregate_parts, ignore_index=True)
    by_cell = pd.concat(cell_parts, ignore_index=True)
    cell_summary = _cell_summary(by_cell)

    outputs = {
        "trials": output_dir / "composite_selector_hpo_trials.csv",
        "aggregate": output_dir / "composite_selector_hpo_topk_metrics.csv",
        "by_cell": output_dir / "composite_selector_hpo_by_cell.csv",
        "cell_summary": output_dir / "composite_selector_hpo_cell_summary.csv",
        "report": output_dir / "composite_selector_hpo_report.md",
        "manifest": output_dir / "manifest.json",
    }
    trials.to_csv(outputs["trials"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    by_cell.to_csv(outputs["by_cell"], index=False)
    cell_summary.to_csv(outputs["cell_summary"], index=False)
    manifest = {
        "scope": "train_meta_composite_selector_hpo",
        "prediction_path": str(prediction_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "trial_count": int(len(trials)),
        "long_selectors": list(long_selectors),
        "top_stage1_arms": int(top_stage1_arms),
        "objective_contract": "precision-first top-k objective with EV/path-quality diagnostics; no AUC; no refit; uses existing month-forward predictions",
        "best_arm": _json_safe(trials.iloc[0].to_dict()) if not trials.empty else {},
        "recommended_arm": _recommended_arm(trials),
        "recommendation_contract": {
            "reference": "ref_s12",
            "max_top10_precision_loss": PRECISION_SAFE_MAX_PRECISION_LOSS,
            "max_top10_weighted_precision_loss": PRECISION_SAFE_MAX_WEIGHTED_PRECISION_LOSS,
            "min_top10_ev_delta": 0.0,
            "max_top10_full_sl_increase": PRECISION_SAFE_MAX_FULL_SL_INCREASE,
            "max_top10_timeout_increase": PRECISION_SAFE_MAX_TIMEOUT_INCREASE,
            "max_worst_top10_ev_loss": PRECISION_SAFE_MAX_WORST_EV_LOSS,
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(outputs["report"], manifest=manifest, trials=trials, aggregate=aggregate, cell_summary=cell_summary)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-path", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--long-selectors", nargs="+", default=list(LONG_SELECTOR_CANDIDATES))
    parser.add_argument("--min-group-rows", type=int, default=100)
    parser.add_argument("--top-stage1-arms", type=int, default=TOP_STAGE1_ARMS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        prediction_path=args.prediction_path,
        output_dir=args.output_dir,
        long_selectors=tuple(str(selector) for selector in args.long_selectors),
        min_group_rows=int(args.min_group_rows),
        top_stage1_arms=int(args.top_stage1_arms),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
