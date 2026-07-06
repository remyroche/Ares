#!/usr/bin/env python3
"""Feature-family ablation for direct cross-asset train_meta context.

This sits one level below ``run_direct_cross_asset_meta_ablation.py``.  Instead
of only testing large blocks, it asks which live-predictable feature families
are useful in at least one side x archetype cell.

The output is intentionally cell-level.  A family can be accepted as useful
context even if it is not globally positive.
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

from scripts.run_direct_cross_asset_meta_ablation import (
    BASELINE_VARIANT,
    DEFAULT_HANDOFF,
    TARGET_COL,
    _delta_vs_baseline,
    _feature_groups,
    _fit_predict_month_forward,
    _json_safe,
    _topk_rows,
    _variant_features,
)


DEFAULT_OUT_DIR = DEFAULT_HANDOFF.parent / "direct_cross_asset_family_ablation_v1"
MIN_ACCEPTED_DELTA_EV = 0.0
MIN_ACCEPTED_DELTA_PRECISION = 0.0


def _family_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    groups = _feature_groups(frame)
    base = groups["base"]

    def cols(prefixes: tuple[str, ...], contains: tuple[str, ...] = ()) -> list[str]:
        out: list[str] = []
        for col in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[col]):
                continue
            if prefixes and not col.startswith(prefixes):
                continue
            if contains and not any(token in col for token in contains):
                continue
            s = pd.to_numeric(frame[col], errors="coerce")
            if s.notna().mean() >= 0.05 and s.nunique(dropna=True) > 1:
                out.append(col)
        return sorted(set(out))

    families = {
        "f00_score_only": base,
        "f01_raw_breadth": base
        + cols(("ctx_",), ("pct_assets", "market_breadth", "mkt_oi", "tail_fail")),
        "f02_raw_btc_eth": base + cols(("ctx_",), ("btc", "eth")),
        "f03_raw_spectral_dispersion": base
        + cols(("ctx_",), ("spectral", "trend_dispersion", "cs_rank", "asym")),
        "f04_raw_liquidity_funding": base
        + cols(("ctx_",), ("fund", "spread", "liquidity", "basket")),
        "f05_oof_gmm_posterior": base + cols(("oofctx_gmm_prob_",)),
        "f06_oof_gmm_distance": base
        + cols(("oofctx_gmm_dist_center_", "oofctx_gmm_mahal_"))
        + cols(("oofctx_",), ("mahalanobis", "raw_state_min_cluster_distance")),
        "f07_oof_dae_latent": base + cols(("oofctx_dae_b16_",)),
        "f08_oof_dae_error_cluster": base
        + cols(
            ("oofctx_",),
            (
                "dae_reconstruction_error",
                "cluster_entropy",
                "cluster_flip",
                "cluster_t",
                "time_since_cluster_change",
                "rolling_cluster_stability",
            ),
        ),
        "f09_oof_regime_centroid": base + cols(("oofctx_regime_centroid_",)),
        "f10_xctx_latent": base + cols(("xctx_latent_",)) + cols(("xctx_cluster_",)),
        "f11_xctx_scores": base + cols(("xctx_ev_score_oof", "xctx_blend_score")),
    }
    # Drop empty non-baseline variants while preserving baseline.
    cleaned: dict[str, list[str]] = {}
    base_set = set(base)
    for family, family_cols in families.items():
        seen: set[str] = set()
        unique_cols = [c for c in family_cols if not (c in seen or seen.add(c))]
        if family == "f00_score_only" or len(set(unique_cols).difference(base_set)) > 0:
            cleaned[family] = unique_cols
    return cleaned


def _accepted_family_cells(cell_delta: pd.DataFrame) -> pd.DataFrame:
    if cell_delta.empty:
        return pd.DataFrame()
    top10 = cell_delta[cell_delta["top_frac"].eq(0.10)].copy()
    accepted = top10[
        (top10["delta_mean_ev_after_1pct"] > MIN_ACCEPTED_DELTA_EV)
        & (top10["delta_precision_positive_ev"] >= MIN_ACCEPTED_DELTA_PRECISION)
    ].copy()
    if accepted.empty:
        return accepted
    accepted = accepted.sort_values(
        ["month", "side_name", "source_archetype", "delta_mean_ev_after_1pct"],
        ascending=[True, True, True, False],
    )
    accepted["family_rank_in_cell"] = (
        accepted.groupby(["month", "side_name", "source_archetype"]).cumcount() + 1
    )
    return accepted


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    cell_delta: pd.DataFrame,
    accepted: pd.DataFrame,
) -> None:
    agg_summary = aggregate.groupby(["variant", "top_frac"], as_index=False).agg(
        months=("month", "nunique"),
        precision_positive_ev=("precision_positive_ev", "mean"),
        ev_weighted_precision=("ev_weighted_precision", "mean"),
        mean_ev_after_1pct=("mean_ev_after_1pct", "mean"),
        full_sl_rate=("full_sl_rate", "mean"),
        timeout_rate=("timeout_rate", "mean"),
        clean_exec_proxy_rate=("clean_exec_proxy_rate", "mean"),
    )
    top10 = cell_delta[cell_delta["top_frac"].eq(0.10)].copy() if not cell_delta.empty else pd.DataFrame()
    if not top10.empty:
        family_summary = top10.groupby("variant", as_index=False).agg(
            cells=("variant", "size"),
            accepted_cells=(
                "delta_mean_ev_after_1pct",
                lambda s: int((s > MIN_ACCEPTED_DELTA_EV).sum()),
            ),
            mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
            mean_delta_precision=("delta_precision_positive_ev", "mean"),
            mean_delta_full_sl=("delta_full_sl_rate", "mean"),
            mean_delta_timeout=("delta_timeout_rate", "mean"),
        )
    else:
        family_summary = pd.DataFrame()
    if not accepted.empty:
        accepted_summary = accepted.groupby(["variant", "side_name", "source_archetype"], as_index=False).agg(
            months=("month", "nunique"),
            mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
            mean_delta_precision=("delta_precision_positive_ev", "mean"),
            mean_delta_full_sl=("delta_full_sl_rate", "mean"),
            mean_delta_timeout=("delta_timeout_rate", "mean"),
        )
    else:
        accepted_summary = pd.DataFrame()

    lines = [
        "# Direct Cross-Asset Feature-Family Ablation",
        "",
        "## Status",
        "",
        f"- Handoff rows: `{manifest['rows']}`",
        f"- Families: `{', '.join(manifest['families'])}`",
        "- A family is accepted in a cell when top10 EV improves and top10 positive-EV precision does not fall.",
        "- No stability-prior features are used.",
        "",
        "## Aggregate Top-k",
        "",
        agg_summary.to_markdown(index=False) if not agg_summary.empty else "No aggregate metrics.",
        "",
        "## Top10 Family Delta Summary",
        "",
        family_summary.to_markdown(index=False) if not family_summary.empty else "No family delta rows.",
        "",
        "## Accepted Side x Archetype Families",
        "",
        accepted_summary.sort_values("mean_delta_ev", ascending=False)
        .head(40)
        .to_markdown(index=False)
        if not accepted_summary.empty
        else "No accepted family cells.",
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
    if TARGET_COL not in frame.columns:
        raise ValueError(f"handoff missing target column: {TARGET_COL}")
    families = _family_columns(frame)
    if "f00_score_only" not in families:
        # Fallback to the existing baseline feature selection if the synthetic
        # frame has unusual columns.
        families["f00_score_only"] = _variant_features(_feature_groups(frame), BASELINE_VARIANT)

    predictions = frame[
        [
            "__ts__",
            "__symbol__",
            "month",
            "side_name",
            "source_archetype",
            TARGET_COL,
            "full_sl",
            "timeout",
            "clean_exec_proxy",
        ]
    ].copy()
    fit_events: list[dict[str, Any]] = []
    family_contract: dict[str, Any] = {}
    for i, (family, features) in enumerate(families.items()):
        score, events = _fit_predict_month_forward(
            frame,
            features,
            max_fit_rows=max_fit_rows,
            seed=seed + i,
        )
        predictions[f"score__{family}"] = score
        added = sorted(set(features).difference(families["f00_score_only"]))
        family_contract[family] = {
            "feature_count": int(len(features)),
            "added_feature_count": int(len(added)),
            "added_features": added,
            "features": features,
        }
        for event in events:
            event = dict(event)
            event["variant"] = family
            fit_events.append(event)

    score_cols = [f"score__{family}" for family in families]
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
    aggregate = aggregate.replace({"variant": {"f00_score_only": BASELINE_VARIANT}})
    cell_metrics = cell_metrics.replace({"variant": {"f00_score_only": BASELINE_VARIANT}})
    aggregate_delta = _delta_vs_baseline(aggregate, key_cols=["month", "top_frac"])
    cell_delta = _delta_vs_baseline(cell_metrics, key_cols=["month", "side_name", "source_archetype", "top_frac"])
    accepted = _accepted_family_cells(cell_delta)

    outputs = {
        "predictions": output_dir / "direct_cross_asset_family_ablation_predictions.parquet",
        "aggregate": output_dir / "direct_cross_asset_family_ablation_aggregate.csv",
        "cell_metrics": output_dir / "direct_cross_asset_family_ablation_by_cell.csv",
        "aggregate_delta": output_dir / "direct_cross_asset_family_ablation_aggregate_delta.csv",
        "cell_delta": output_dir / "direct_cross_asset_family_ablation_cell_delta.csv",
        "accepted": output_dir / "direct_cross_asset_family_ablation_accepted_cells.csv",
        "fit_events": output_dir / "direct_cross_asset_family_ablation_fit_events.csv",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "direct_cross_asset_family_ablation_report.md",
    }
    predictions.to_parquet(outputs["predictions"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    cell_metrics.to_csv(outputs["cell_metrics"], index=False)
    aggregate_delta.to_csv(outputs["aggregate_delta"], index=False)
    cell_delta.to_csv(outputs["cell_delta"], index=False)
    accepted.to_csv(outputs["accepted"], index=False)
    pd.DataFrame(fit_events).to_csv(outputs["fit_events"], index=False)
    manifest = {
        "scope": "direct_cross_asset_family_ablation",
        "handoff_path": str(handoff_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "months": sorted(frame["month"].dropna().astype(str).unique().tolist()),
        "families": list(families.keys()),
        "baseline_variant": BASELINE_VARIANT,
        "target": TARGET_COL,
        "acceptance_rule": {
            "top_frac": 0.10,
            "delta_mean_ev_after_1pct": f">{MIN_ACCEPTED_DELTA_EV}",
            "delta_precision_positive_ev": f">={MIN_ACCEPTED_DELTA_PRECISION}",
        },
        "family_contract": family_contract,
        "leakage_contract": "all models fit month-forward on strictly earlier months; no stability-prior features are used",
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(outputs["report"], manifest, aggregate, cell_delta, accepted)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-path", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-fit-rows", type=int, default=80_000)
    parser.add_argument("--min-group-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=71)
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
