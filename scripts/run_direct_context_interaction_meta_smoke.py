#!/usr/bin/env python3
"""Learned side/archetype interaction smoke for direct train_meta context.

This is the next step after the hand-tuned cell-prior selector experiment.  It
keeps the same month-forward target heads and S12-style risk score, but changes
the feature set so the heads can learn interactions between live-predictable
side/archetype state and cross-asset / AE-GMM context features.

No future outcomes are used as features.  Side, source archetype, and all
interaction inputs are available at decision time in the handoff.
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

from scripts.run_direct_context_risk_aware_train_meta_smoke import (
    DEFAULT_FEATURE_MANIFEST,
    DEFAULT_FEATURE_SET_DIR,
    DEFAULT_HANDOFF,
    SELECTOR_SPECS,
    TARGETS,
    _add_scores,
    _delta_vs_ev_only,
    _fit_month_forward_heads,
    _json_safe,
    _load_feature_columns,
    _summarize_cell_deltas,
    _topk_metrics,
)


DEFAULT_OUT_DIR = DEFAULT_FEATURE_SET_DIR / "interaction_train_meta_smoke_v1"
VARIANTS = (
    "i0_direct_context",
    "i1_side_archetype_id",
    "i2_xctx_cell_interactions",
    "i3_context_risk_cell_interactions",
)
EVAL_SELECTORS = ("s0_ev_only", "s12_ev_clean_strong_risk", "s13_ev_clean_fullsl_neutral_timeout")
INTERACTION_BASE_FEATURES = (
    "xctx_ev_score_oof",
    "xctx_blend_score",
    "xctx_cluster_entropy",
    "xctx_cluster_distance",
    "oofctx_dae_reconstruction_error",
    "oofctx_dae_reconstruction_error_zscore",
    "oofctx_cluster_entropy",
    "oofctx_min_mahalanobis",
    "oofctx_expected_mahalanobis",
    "oofctx_latent_mahalanobis_drift",
)
RISK_INTERACTION_FEATURES = (
    "oofctx_dae_reconstruction_error",
    "oofctx_dae_reconstruction_error_zscore",
    "oofctx_cluster_entropy",
    "oofctx_min_mahalanobis",
    "oofctx_expected_mahalanobis",
    "xctx_cluster_entropy",
    "xctx_cluster_distance",
)


def _load_feature_families(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    families = payload.get("families", {})
    if not isinstance(families, dict):
        return {}
    return {str(k): [str(c) for c in v] for k, v in families.items() if isinstance(v, list)}


def _safe_feature_name(value: object) -> str:
    out = str(value).strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in out)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "missing"


def _numeric_existing(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    out: list[str] = []
    for col in cols:
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            values = pd.to_numeric(frame[col], errors="coerce")
            if values.notna().mean() >= 0.03 and values.nunique(dropna=True) > 1:
                out.append(col)
    return out


def _add_one_hot_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    out = frame.copy()
    groups: dict[str, list[str]] = {"side": [], "source_archetype": [], "cell": []}
    for col, prefix, group_name in (
        ("side_name", "int_side", "side"),
        ("source_archetype", "int_arch", "source_archetype"),
    ):
        for value in sorted(out[col].dropna().astype(str).unique().tolist()):
            name = f"{prefix}_{_safe_feature_name(value)}"
            out[name] = out[col].astype(str).eq(value).astype("float32")
            groups[group_name].append(name)
    cell_values = (
        out["side_name"].astype(str).fillna("missing") + "__" + out["source_archetype"].astype(str).fillna("missing")
    )
    for value in sorted(cell_values.dropna().unique().tolist()):
        name = f"int_cell_{_safe_feature_name(value)}"
        out[name] = cell_values.eq(value).astype("float32")
        groups["cell"].append(name)
    return out, groups["side"] + groups["source_archetype"] + groups["cell"], groups


def _add_context_interactions(
    frame: pd.DataFrame,
    *,
    one_hot_groups: dict[str, list[str]],
    include_risk_crosses: bool,
) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    new_cols: list[str] = []
    interaction_data: dict[str, pd.Series] = {}
    context_cols = _numeric_existing(out, list(INTERACTION_BASE_FEATURES))
    # Cell interactions are intentionally limited to the most explanatory
    # context features from the weak-cell driver report.
    interaction_one_hots = one_hot_groups.get("side", []) + one_hot_groups.get("source_archetype", []) + one_hot_groups.get("cell", [])
    one_hot_values = {
        one_hot: pd.to_numeric(out[one_hot], errors="coerce").fillna(0.0).astype("float32")
        for one_hot in interaction_one_hots
    }
    for feature in context_cols:
        values = pd.to_numeric(out[feature], errors="coerce").astype("float32")
        for one_hot, encoded in one_hot_values.items():
            name = f"intx_{feature}__{one_hot}"
            interaction_data[name] = (values * encoded).astype("float32")
            new_cols.append(name)
    if include_risk_crosses and "xctx_ev_score_oof" in out.columns:
        ev_score = pd.to_numeric(out["xctx_ev_score_oof"], errors="coerce").astype("float32")
        for risk_col in _numeric_existing(out, list(RISK_INTERACTION_FEATURES)):
            if risk_col == "xctx_ev_score_oof":
                continue
            name = f"intx_xctx_ev_score_oof__{risk_col}"
            interaction_data[name] = (ev_score * pd.to_numeric(out[risk_col], errors="coerce")).astype("float32")
            new_cols.append(name)
    if interaction_data:
        out = pd.concat([out, pd.DataFrame(interaction_data, index=out.index)], axis=1).copy()
    return out, new_cols


def _variant_frame_and_features(
    frame: pd.DataFrame,
    base_features: list[str],
    *,
    variant: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    working = frame.copy()
    features = list(base_features)
    one_hot_cols: list[str] = []
    interaction_cols: list[str] = []
    one_hot_groups: dict[str, list[str]] = {"side": [], "source_archetype": [], "cell": []}
    if variant in {"i1_side_archetype_id", "i2_xctx_cell_interactions", "i3_context_risk_cell_interactions"}:
        working, one_hot_cols, one_hot_groups = _add_one_hot_features(working)
        features.extend(one_hot_cols)
    if variant in {"i2_xctx_cell_interactions", "i3_context_risk_cell_interactions"}:
        working, interaction_cols = _add_context_interactions(
            working,
            one_hot_groups=one_hot_groups,
            include_risk_crosses=variant == "i3_context_risk_cell_interactions",
        )
        features.extend(interaction_cols)
    seen: set[str] = set()
    deduped = []
    for col in features:
        if col in working.columns and col not in seen:
            seen.add(col)
            deduped.append(col)
    metadata = {
        "variant": variant,
        "base_feature_count": int(len(base_features)),
        "one_hot_feature_count": int(len(one_hot_cols)),
        "interaction_feature_count": int(len(interaction_cols)),
        "feature_count": int(len(deduped)),
        "one_hot_features": one_hot_cols,
        "interaction_features": interaction_cols,
    }
    return working, deduped, metadata


def _evaluate_variant(
    *,
    frame: pd.DataFrame,
    features: list[str],
    variant: str,
    output_dir: Path,
    max_fit_rows: int,
    min_group_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    preds, events = _fit_month_forward_heads(frame, features, max_fit_rows=max_fit_rows, seed=seed)
    keep_cols = [
        "__ts__",
        "__symbol__",
        "month",
        "side_name",
        "source_archetype",
        "exec_ev_after_1pct_cost",
        "full_sl",
        "timeout",
        "clean_exec_proxy",
    ]
    predictions = pd.concat([frame[[c for c in keep_cols if c in frame.columns]].copy(), preds], axis=1)
    predictions = _add_scores(predictions)
    metrics = []
    cell_metrics = []
    for selector_name in EVAL_SELECTORS:
        metrics.append(
            _topk_metrics(
                predictions,
                selector_name=selector_name,
                score_col=f"score_{selector_name}",
                guards=dict(SELECTOR_SPECS[selector_name].get("guards", {})),
                group_cols=["month"],
                min_group_rows=min_group_rows,
            )
        )
        cell_metrics.append(
            _topk_metrics(
                predictions,
                selector_name=selector_name,
                score_col=f"score_{selector_name}",
                guards=dict(SELECTOR_SPECS[selector_name].get("guards", {})),
                group_cols=["month", "side_name", "source_archetype"],
                min_group_rows=min_group_rows,
            )
        )
    aggregate = pd.concat(metrics, ignore_index=True)
    aggregate.insert(0, "variant", variant)
    by_cell = pd.concat(cell_metrics, ignore_index=True)
    by_cell.insert(0, "variant", variant)
    aggregate_delta = _delta_vs_ev_only(aggregate, key_cols=["variant", "month", "top_frac"])
    cell_delta = _delta_vs_ev_only(by_cell, key_cols=["variant", "month", "side_name", "source_archetype", "top_frac"])
    cell_delta_summary, worst_cell_tradeoffs = _summarize_cell_deltas(cell_delta)
    cell_delta_summary.insert(0, "variant", variant)
    worst_cell_tradeoffs.insert(0, "variant", variant)
    for event in events:
        event["variant"] = variant
    # Store compact predictions only for the best-followup analysis; this is
    # small enough and avoids rerunning heads for diagnostics.
    pred_cols = [c for c in predictions.columns if c in keep_cols or c.startswith("pred_") or c in {"score_s12_ev_clean_strong_risk", "score_s13_ev_clean_fullsl_neutral_timeout", "score_s0_ev_only"}]
    predictions[pred_cols].to_parquet(output_dir / f"{variant}_predictions.parquet", index=False)
    return aggregate, by_cell, aggregate_delta, cell_delta, cell_delta_summary, worst_cell_tradeoffs, events


def _write_report(path: Path, manifest: dict[str, Any], aggregate: pd.DataFrame, deltas: pd.DataFrame, cell_summary: pd.DataFrame) -> None:
    top10 = aggregate[aggregate["top_frac"].eq(0.10)].copy()
    top10_summary = top10.groupby(["variant", "selector"], as_index=False).agg(
        months=("month", "nunique"),
        precision_positive_ev=("precision_positive_ev", "mean"),
        ev_weighted_precision=("ev_weighted_precision", "mean"),
        mean_ev_after_1pct=("mean_ev_after_1pct", "mean"),
        full_sl_rate=("full_sl_rate", "mean"),
        timeout_rate=("timeout_rate", "mean"),
        clean_exec_proxy_rate=("clean_exec_proxy_rate", "mean"),
    )
    delta10 = deltas[deltas["top_frac"].eq(0.10)].copy() if not deltas.empty else pd.DataFrame()
    if not delta10.empty:
        delta_summary = delta10.groupby(["variant", "selector"], as_index=False).agg(
            mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
            mean_delta_precision=("delta_precision_positive_ev", "mean"),
            mean_delta_full_sl=("delta_full_sl_rate", "mean"),
            mean_delta_timeout=("delta_timeout_rate", "mean"),
            mean_delta_clean=("delta_clean_exec_proxy_rate", "mean"),
        )
    else:
        delta_summary = pd.DataFrame()
    lines = [
        "# Direct Context Interaction Train Meta Smoke",
        "",
        "## Status",
        "",
        f"- Rows: `{manifest['rows']}`",
        f"- Variants: `{', '.join(manifest['variants'])}`",
        "- Heads are fit month-forward on strictly earlier months.",
        "- Interaction features use only live-predictable side/archetype/context fields.",
        "",
        "## Top10 Aggregate",
        "",
        top10_summary.to_markdown(index=False) if not top10_summary.empty else "No top10 metrics.",
        "",
        "## Top10 Delta vs EV-Only Within Variant",
        "",
        delta_summary.to_markdown(index=False) if not delta_summary.empty else "No delta metrics.",
        "",
        "## Cell Delta Coverage",
        "",
        cell_summary.to_markdown(index=False) if not cell_summary.empty else "No cell summary.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    handoff_path: Path,
    feature_manifest_path: Path,
    output_dir: Path,
    variants: tuple[str, ...],
    max_fit_rows: int,
    min_group_rows: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(handoff_path)
    base_features = [c for c in _load_feature_columns(feature_manifest_path) if c in frame.columns]
    missing_targets = sorted(set(TARGETS.values()).difference(frame.columns))
    if missing_targets:
        raise ValueError(f"handoff missing target columns: {missing_targets}")
    aggregate_parts = []
    cell_parts = []
    aggregate_delta_parts = []
    cell_delta_parts = []
    cell_summary_parts = []
    worst_cell_parts = []
    fit_events: list[dict[str, Any]] = []
    variant_metadata: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        variant_frame, features, metadata = _variant_frame_and_features(frame, base_features, variant=variant)
        variant_metadata.append(metadata)
        aggregate, by_cell, aggregate_delta, cell_delta, cell_summary, worst_cells, events = _evaluate_variant(
            frame=variant_frame,
            features=features,
            variant=variant,
            output_dir=output_dir,
            max_fit_rows=max_fit_rows,
            min_group_rows=min_group_rows,
            seed=seed + idx * 17,
        )
        aggregate_parts.append(aggregate)
        cell_parts.append(by_cell)
        aggregate_delta_parts.append(aggregate_delta)
        cell_delta_parts.append(cell_delta)
        cell_summary_parts.append(cell_summary)
        worst_cell_parts.append(worst_cells)
        fit_events.extend(events)
    aggregate_all = pd.concat(aggregate_parts, ignore_index=True)
    cell_all = pd.concat(cell_parts, ignore_index=True)
    aggregate_delta_all = pd.concat(aggregate_delta_parts, ignore_index=True)
    cell_delta_all = pd.concat(cell_delta_parts, ignore_index=True)
    cell_summary_all = pd.concat(cell_summary_parts, ignore_index=True)
    worst_cell_all = pd.concat(worst_cell_parts, ignore_index=True)
    outputs = {
        "aggregate": output_dir / "interaction_train_meta_aggregate.csv",
        "cell_metrics": output_dir / "interaction_train_meta_by_cell.csv",
        "aggregate_delta": output_dir / "interaction_train_meta_aggregate_delta.csv",
        "cell_delta": output_dir / "interaction_train_meta_cell_delta.csv",
        "cell_delta_summary": output_dir / "interaction_train_meta_cell_delta_summary.csv",
        "worst_cell_tradeoffs": output_dir / "interaction_train_meta_worst_cell_tradeoffs.csv",
        "fit_events": output_dir / "interaction_train_meta_fit_events.csv",
        "variant_metadata": output_dir / "interaction_train_meta_variant_metadata.json",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "interaction_train_meta_smoke.md",
    }
    aggregate_all.to_csv(outputs["aggregate"], index=False)
    cell_all.to_csv(outputs["cell_metrics"], index=False)
    aggregate_delta_all.to_csv(outputs["aggregate_delta"], index=False)
    cell_delta_all.to_csv(outputs["cell_delta"], index=False)
    cell_summary_all.to_csv(outputs["cell_delta_summary"], index=False)
    worst_cell_all.to_csv(outputs["worst_cell_tradeoffs"], index=False)
    pd.DataFrame(fit_events).to_csv(outputs["fit_events"], index=False)
    outputs["variant_metadata"].write_text(json.dumps(_json_safe(variant_metadata), indent=2), encoding="utf-8")
    manifest = {
        "scope": "direct_context_interaction_train_meta_smoke",
        "handoff_path": str(handoff_path),
        "feature_manifest_path": str(feature_manifest_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "base_feature_count": int(len(base_features)),
        "variants": list(variants),
        "eval_selectors": list(EVAL_SELECTORS),
        "variant_metadata": variant_metadata,
        "leakage_contract": (
            "month-forward target heads; side/archetype one-hot and interaction features are live-predictable; "
            "no prior/future outcome metadata is used as model input"
        ),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(outputs["report"], manifest, aggregate_all, aggregate_delta_all, cell_summary_all)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-path", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--feature-manifest-path", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--max-fit-rows", type=int, default=80_000)
    parser.add_argument("--min-group-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=191)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        handoff_path=args.handoff_path,
        feature_manifest_path=args.feature_manifest_path,
        output_dir=args.output_dir,
        variants=tuple(str(v) for v in args.variants),
        max_fit_rows=int(args.max_fit_rows),
        min_group_rows=int(args.min_group_rows),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
