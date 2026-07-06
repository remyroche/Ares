#!/usr/bin/env python3
"""Month-forward train_meta ablation for direct cross-asset context features.

This consumes ``direct_cross_asset_meta_context_handoff.parquet`` and asks a
simple question: do live-predictable cross-asset / AE-GMM / latent context
blocks improve top-k executable selection beyond baseline score features?

Metrics are top-k precision and EV first.  AUC is intentionally not reported.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONTEXT_DIR = Path(
    "data_perp/reports/contextual_tp_sl_ablation_workflow_v14_runtime_health_20260701/"
    "direct_cross_asset_meta_context_v1"
)
DEFAULT_HANDOFF = DEFAULT_CONTEXT_DIR / "direct_cross_asset_meta_context_handoff.parquet"
DEFAULT_OUT_DIR = DEFAULT_CONTEXT_DIR / "direct_cross_asset_meta_ablation_v1"

TOP_FRACS = (0.30, 0.20, 0.10)
TARGET_COL = "exec_ev_after_1pct_cost"
BASELINE_VARIANT = "m0_score_only"

BASE_SCORE_CANDIDATES = (
    "normalized_rank_score",
    "calibrated_score",
    "reliability_blend_score",
    "source_calibrated_score",
    "source_reliability_blend_score",
    "anchor_score",
    "reliability_anchor_only_score",
    "raw_prediction_score",
    "base_pred",
    "meta_pred",
    "rank_pct",
    "policy_rank_pct",
    "source_policy_rank_pct",
    "auction_rank_pct",
    "base_train_rank_pct",
    "base_batch_rank_pct",
    "meta_train_rank_pct",
    "historical_rank_pct",
    "batch_rank_pct",
    "auction_rank_score",
    "simple_policy_calibrated_good_trade_prob",
    "simple_policy_calibrated_bad_trade_prob",
    "simple_policy_calibrated_expected_net_gain",
    "estimated_hit_rate",
    "estimated_ev_net_return",
    "uncertainty_adjusted_ev_net_return",
    "contextual_tp_sl_score",
    "contextual_tp_mult",
)

VARIANTS = {
    "m0_score_only": ("base",),
    "m1_score_plus_raw_xasset": ("base", "raw_ctx"),
    "m2_score_plus_oof_aegmm": ("base", "oofctx"),
    "m3_score_plus_xctx_latent": ("base", "xctx_latent"),
    "m4_score_plus_xctx_score": ("base", "xctx_score"),
    "m5_score_plus_all_context": ("base", "raw_ctx", "oofctx", "xctx_latent", "xctx_score"),
    "m6_context_only": ("raw_ctx", "oofctx", "xctx_latent", "xctx_score"),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _numeric_existing(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    out: list[str] = []
    for col in cols:
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            s = pd.to_numeric(frame[col], errors="coerce")
            if s.notna().mean() >= 0.05 and s.nunique(dropna=True) > 1:
                out.append(col)
    return out


def _feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    base = _numeric_existing(frame, list(BASE_SCORE_CANDIDATES))
    raw_ctx = _numeric_existing(frame, [c for c in frame.columns if c.startswith("ctx_")])
    oofctx = _numeric_existing(frame, [c for c in frame.columns if c.startswith("oofctx_")])
    xctx_latent = _numeric_existing(
        frame,
        [
            c
            for c in frame.columns
            if c.startswith("xctx_latent_")
            or c in {"xctx_cluster_id", "xctx_cluster_distance", "xctx_cluster_entropy"}
        ],
    )
    xctx_score = _numeric_existing(frame, [c for c in ("xctx_ev_score_oof", "xctx_blend_score")])
    return {
        "base": base,
        "raw_ctx": raw_ctx,
        "oofctx": oofctx,
        "xctx_latent": xctx_latent,
        "xctx_score": xctx_score,
    }


def _variant_features(groups: dict[str, list[str]], variant: str) -> list[str]:
    cols: list[str] = []
    for group in VARIANTS[variant]:
        cols.extend(groups.get(group, []))
    seen: set[str] = set()
    out: list[str] = []
    for col in cols:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _fit_predict_month_forward(
    frame: pd.DataFrame,
    features: list[str],
    *,
    max_fit_rows: int,
    seed: int,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    pred = pd.Series(np.nan, index=frame.index, dtype="float32")
    events: list[dict[str, Any]] = []
    months = sorted(frame["month"].dropna().astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    for month in months[1:]:
        train_idx = frame.index[frame["month"].astype(str) < month]
        val_idx = frame.index[frame["month"].astype(str).eq(month)]
        y = pd.to_numeric(frame.loc[train_idx, TARGET_COL], errors="coerce")
        valid = y.notna()
        train_idx = train_idx[valid.to_numpy()]
        if len(train_idx) < 1000 or len(val_idx) == 0 or not features:
            events.append(
                {
                    "month": month,
                    "status": "skipped",
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(val_idx)),
                    "feature_count": int(len(features)),
                }
            )
            continue
        if len(train_idx) > max_fit_rows:
            train_idx = pd.Index(rng.choice(train_idx.to_numpy(), size=max_fit_rows, replace=False))
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=128,
                learning_rate=0.035,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                min_samples_leaf=120,
                random_state=seed,
            ),
        )
        model.fit(frame.loc[train_idx, features], frame.loc[train_idx, TARGET_COL].astype(float))
        pred.loc[val_idx] = model.predict(frame.loc[val_idx, features]).astype("float32")
        events.append(
            {
                "month": month,
                "status": "fit",
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "feature_count": int(len(features)),
            }
        )
    return pred, events


def _topk_rows(
    frame: pd.DataFrame,
    *,
    score_col: str,
    group_cols: list[str],
    min_group_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = frame[pd.to_numeric(frame[score_col], errors="coerce").notna()].copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    for keys, grp in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(grp) < int(min_group_rows):
            continue
        ordered = grp.sort_values(score_col, ascending=False)
        for frac in TOP_FRACS:
            n = max(1, int(math.ceil(len(ordered) * frac)))
            sel = ordered.head(n)
            ev = pd.to_numeric(sel[TARGET_COL], errors="coerce")
            abs_ev = float(ev.abs().sum())
            rec = {col: key for col, key in zip(group_cols, keys)}
            rec.update(
                {
                    "variant": score_col.removeprefix("score__"),
                    "top_frac": float(frac),
                    "rows": int(len(grp)),
                    "selected_rows": int(len(sel)),
                    "precision_positive_ev": float((ev > 0).mean()),
                    "ev_weighted_precision": float(ev.clip(lower=0).sum() / abs_ev) if abs_ev > 0 else float("nan"),
                    "mean_ev_after_1pct": float(ev.mean()),
                    "sum_ev_after_1pct": float(ev.sum()),
                    "full_sl_rate": float(pd.to_numeric(sel.get("full_sl"), errors="coerce").mean()),
                    "timeout_rate": float(pd.to_numeric(sel.get("timeout"), errors="coerce").mean()),
                    "clean_exec_proxy_rate": float(pd.to_numeric(sel.get("clean_exec_proxy"), errors="coerce").mean()),
                }
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def _delta_vs_baseline(metrics: pd.DataFrame, *, key_cols: list[str]) -> pd.DataFrame:
    base = metrics[metrics["variant"].eq(BASELINE_VARIANT)]
    rows: list[dict[str, Any]] = []
    for _, cur in metrics[~metrics["variant"].eq(BASELINE_VARIANT)].iterrows():
        mask = pd.Series(True, index=base.index)
        for col in key_cols:
            mask &= base[col].eq(cur[col])
        if not mask.any():
            continue
        ref = base[mask].iloc[0]
        rec = {col: cur[col] for col in key_cols}
        rec["variant"] = cur["variant"]
        rec["rows"] = int(cur["rows"])
        rec["selected_rows"] = int(cur["selected_rows"])
        for metric in (
            "precision_positive_ev",
            "ev_weighted_precision",
            "mean_ev_after_1pct",
            "full_sl_rate",
            "timeout_rate",
            "clean_exec_proxy_rate",
        ):
            rec[metric] = float(cur[metric])
            rec[f"baseline_{metric}"] = float(ref[metric])
            rec[f"delta_{metric}"] = float(cur[metric] - ref[metric])
        rows.append(rec)
    return pd.DataFrame(rows)


def _write_report(path: Path, manifest: dict[str, Any], aggregate: pd.DataFrame, deltas: pd.DataFrame) -> None:
    summary = aggregate.groupby(["variant", "top_frac"], as_index=False).agg(
        months=("month", "nunique"),
        precision_positive_ev=("precision_positive_ev", "mean"),
        ev_weighted_precision=("ev_weighted_precision", "mean"),
        mean_ev_after_1pct=("mean_ev_after_1pct", "mean"),
        full_sl_rate=("full_sl_rate", "mean"),
        timeout_rate=("timeout_rate", "mean"),
        clean_exec_proxy_rate=("clean_exec_proxy_rate", "mean"),
    )
    top10 = deltas[deltas["top_frac"].eq(0.10)].copy() if not deltas.empty else pd.DataFrame()
    if not top10.empty:
        delta_summary = top10.groupby("variant", as_index=False).agg(
            cells=("variant", "size"),
            positive_ev_delta_cells=("delta_mean_ev_after_1pct", lambda s: int((s > 0).sum())),
            mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
            mean_delta_precision=("delta_precision_positive_ev", "mean"),
            mean_delta_full_sl=("delta_full_sl_rate", "mean"),
            mean_delta_timeout=("delta_timeout_rate", "mean"),
        )
    else:
        delta_summary = pd.DataFrame()
    lines = [
        "# Direct Cross-Asset Meta Ablation",
        "",
        "## Status",
        "",
        f"- Handoff rows: `{manifest['rows']}`",
        f"- Months: `{', '.join(manifest['months'])}`",
        f"- Variants: `{', '.join(manifest['variants'])}`",
        "- Metrics are top-k precision/EV/path quality; AUC is intentionally not used.",
        "- `full_sl_rate` is the available bad-path proxy in this broad ledger; first-touch bad-MAE is not present.",
        "",
        "## Aggregate Top-k",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No aggregate metrics.",
        "",
        "## Top10 Side x Archetype Deltas",
        "",
        delta_summary.to_markdown(index=False) if not delta_summary.empty else "No delta rows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    handoff_path: Path,
    output_dir: Path,
    max_fit_rows: int,
    min_group_rows: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(handoff_path)
    required = {"month", TARGET_COL, "side_name", "source_archetype"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"handoff missing required columns: {missing}")
    groups = _feature_groups(frame)
    predictions = frame[["__ts__", "__symbol__", "month", "side_name", "source_archetype", TARGET_COL, "full_sl", "timeout", "clean_exec_proxy"]].copy()
    variant_contract: dict[str, Any] = {}
    fit_events: list[dict[str, Any]] = []
    for i, variant in enumerate(VARIANTS):
        features = _variant_features(groups, variant)
        score, events = _fit_predict_month_forward(
            frame,
            features,
            max_fit_rows=max_fit_rows,
            seed=seed + i,
        )
        score_col = f"score__{variant}"
        predictions[score_col] = score
        variant_contract[variant] = {
            "groups": list(VARIANTS[variant]),
            "feature_count": int(len(features)),
            "features": features,
        }
        for event in events:
            event = dict(event)
            event["variant"] = variant
            fit_events.append(event)

    score_cols = [f"score__{variant}" for variant in VARIANTS]
    aggregate = pd.concat(
        [
            _topk_rows(predictions, score_col=score_col, group_cols=["month"], min_group_rows=min_group_rows)
            for score_col in score_cols
        ],
        ignore_index=True,
    )
    cell_metrics = pd.concat(
        [
            _topk_rows(
                predictions,
                score_col=score_col,
                group_cols=["month", "side_name", "source_archetype"],
                min_group_rows=min_group_rows,
            )
            for score_col in score_cols
        ],
        ignore_index=True,
    )
    aggregate_delta = _delta_vs_baseline(aggregate, key_cols=["month", "top_frac"])
    cell_delta = _delta_vs_baseline(cell_metrics, key_cols=["month", "side_name", "source_archetype", "top_frac"])
    useful_cells = (
        cell_delta[
            (cell_delta["top_frac"].eq(0.10))
            & (cell_delta["delta_mean_ev_after_1pct"] > 0)
            & (cell_delta["delta_precision_positive_ev"] >= 0)
        ].copy()
        if not cell_delta.empty
        else pd.DataFrame()
    )

    outputs = {
        "predictions": output_dir / "direct_cross_asset_meta_ablation_predictions.parquet",
        "aggregate": output_dir / "direct_cross_asset_meta_ablation_aggregate.csv",
        "cell_metrics": output_dir / "direct_cross_asset_meta_ablation_by_cell.csv",
        "aggregate_delta": output_dir / "direct_cross_asset_meta_ablation_aggregate_delta.csv",
        "cell_delta": output_dir / "direct_cross_asset_meta_ablation_cell_delta.csv",
        "useful_cells": output_dir / "direct_cross_asset_meta_ablation_useful_cells.csv",
        "fit_events": output_dir / "direct_cross_asset_meta_ablation_fit_events.csv",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "direct_cross_asset_meta_ablation_report.md",
    }
    predictions.to_parquet(outputs["predictions"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    cell_metrics.to_csv(outputs["cell_metrics"], index=False)
    aggregate_delta.to_csv(outputs["aggregate_delta"], index=False)
    cell_delta.to_csv(outputs["cell_delta"], index=False)
    useful_cells.to_csv(outputs["useful_cells"], index=False)
    pd.DataFrame(fit_events).to_csv(outputs["fit_events"], index=False)
    manifest = {
        "scope": "direct_cross_asset_meta_ablation",
        "handoff_path": str(handoff_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "months": sorted(frame["month"].dropna().astype(str).unique().tolist()),
        "variants": list(VARIANTS.keys()),
        "variant_contract": variant_contract,
        "target": TARGET_COL,
        "top_fracs": list(TOP_FRACS),
        "metrics": "top-k precision, EV after 1% cost, full-SL proxy, timeout, clean-exec proxy",
        "leakage_contract": "all models are fit month-forward on strictly earlier months; no stability-prior features are used",
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(outputs["report"], manifest, aggregate, cell_delta)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-path", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-fit-rows", type=int, default=80_000)
    parser.add_argument("--min-group-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        handoff_path=args.handoff_path,
        output_dir=args.output_dir,
        max_fit_rows=int(args.max_fit_rows),
        min_group_rows=int(args.min_group_rows),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
