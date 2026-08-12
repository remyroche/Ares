#!/usr/bin/env python3
"""Run the frozen one-month GAM disagreement contract on untouched 2026 rows.

This runner is intentionally narrow and prequential:

* the canonical R3 TP6/SL4 panel is the only source of base outputs and labels;
* the residual/path model for each 2026 month is fit before that month;
* each month's archetypes, clusters, CMI context and GAM are fit on the
  immediately preceding month only;
* the residual/meta learner sees one field only, ``gam_delta_bps``;
* transport-invalid target months use the exact matched control score.

The resulting July 2026 metrics are a transport test, not a development
replay.  No 2026 held outcome is used in any fit for its target month.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.cofiring_economic_clusters import materialize_memberships  # noqa: E402
from extreme_price_movements.structural_archetypes import (  # noqa: E402
    match_catalog_to_archetypes,
    materialize_row_archetype_exposures,
)
from scripts.materialize_tp6_sl4_canonical_meta_paths_20260808 import _materialize_month  # noqa: E402
from scripts.run_tp6_sl4_downstream_retrain_2025 import (  # noqa: E402
    _map_base,
    _selected_context,
)
from scripts.run_tp6_sl4_rolling_archetype_gam_oos import (  # noqa: E402
    CLUSTER_ACTIVE_THRESHOLD,
    GAM_CONTEXT_CAP,
    MATCH_TEMPERATURE,
    MATCH_THRESHOLD,
    TOP_N,
    _fit_vc,
    _local_archetypes,
    _select_clusters,
)
from scripts.run_tp6_sl4_rolling_gam_residual_integration import (  # noqa: E402
    _fit_heads,
    _group,
    _pct,
)


INPUT = ROOT / "data_perp/artifacts/r3_tp6_sl4_meta_target_ablation_20260803_v1/r3_meta_target_oof_predictions.parquet"
OLD_BASE = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_base_20260811_v1.parquet"
OLD_RAW = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/strict_base_reasoning"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_oos_2026_20260815_v1"
MONTHS = tuple(f"2026-{m:02d}" for m in range(1, 8))
SIDE = "long"
SEED = 20260815


def _load_panel() -> tuple[pd.DataFrame, list[str], str]:
    context = _selected_context()
    source = pd.read_parquet(INPUT)
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="raise")
    source["label_available_ts"] = pd.to_datetime(source["label_available_ts"], utc=True, errors="raise")
    source = source.loc[source.side_name.astype(str).str.lower().eq(SIDE)].copy()
    source["month"] = source["__ts__"].dt.strftime("%Y-%m")
    source["base_score"] = pd.to_numeric(source["r3_meta_p_clear"], errors="coerce") - 0.5 * pd.to_numeric(source["r3_meta_p_adverse"], errors="coerce")
    source["label_valid"] = source["label_valid"].fillna(False).astype(bool)
    missing = sorted(set(context).difference(source.columns))
    if missing:
        raise ValueError(f"R3 panel is missing frozen context fields: {missing}")
    required = {"candidate_id", "__ts__", "label_available_ts", "exact_net_bps", "exact_gross_bps", "base_score", "label_valid"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"R3 panel is missing required fields: {missing}")
    # Retain a broad pre-2026 history for the residual/meta fit.  Target
    # months are never used in a month-specific fit until their own turn.
    keep_months = sorted(set(source.month.astype(str)) & {f"{y:04d}-{m:02d}" for y in (2024, 2025, 2026) for m in range(1, 13)})
    source = source.loc[source.month.isin(keep_months) & source.label_valid].copy()
    source = source.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if source.candidate_id.duplicated().any():
        raise ValueError("R3 source has duplicate candidate IDs")
    digest = hashlib.sha256("\n".join(context).encode()).hexdigest()
    return source, context, digest


def _materialize_transport_paths(panel: pd.DataFrame, context: list[str], out: Path) -> Path:
    raw_out = out / "raw_paths"
    raw_out.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        audit = raw_out / "fold_audits" / f"month={month}.json"
        eval_path = raw_out / "fold_evaluations" / f"month={month}.parquet"
        catalogue = raw_out / "strict_base_reasoning" / f"month={month}" / "leaf_rule_catalog.parquet"
        contribution = raw_out / "family_contributions" / f"month={month}.parquet"
        if all(p.exists() for p in (audit, eval_path, catalogue, contribution)):
            continue
        start = pd.Timestamp(month, tz="UTC")
        train = panel.loc[
            panel.__ts__.lt(start)
            & panel.label_available_ts.lt(start)
            & panel.label_valid
            & panel.side_name.eq(SIDE)
        ].copy()
        held = panel.loc[panel.month.eq(month) & panel.side_name.eq(SIDE)].copy()
        result = _materialize_month(
            train=train,
            held=held,
            context=context,
            month=month,
            out=raw_out,
            max_trees=64,
            contribution_components=8,
            threshold_bands=4,
        )
        print(json.dumps({"month": month, "status": result.get("status"), "train_rows": result.get("train_rows"), "held_rows": result.get("held_rows")}), flush=True)
    return raw_out


def _base_expected_lookup(panel: pd.DataFrame, raw_out: Path) -> pd.DataFrame:
    old = pd.read_parquet(OLD_BASE, columns=["candidate_id", "__ts__", "base_expected_bps", "base_score"])
    old["__ts__"] = pd.to_datetime(old["__ts__"], utc=True)
    rows = [old]
    for month in MONTHS:
        p = raw_out / "fold_evaluations" / f"month={month}.parquet"
        if p.exists():
            d = pd.read_parquet(p, columns=["candidate_id", "__ts__", "base_expected_bps"])
            d["__ts__"] = pd.to_datetime(d["__ts__"], utc=True)
            rows.append(d)
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["candidate_id", "__ts__"], keep="last")
    return out


def _load_raw(raw_root: Path, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if month in MONTHS:
        folder = raw_root / "strict_base_reasoning" / f"month={month}"
    else:
        folder = OLD_RAW / f"month={month}"
    return pd.read_parquet(folder / "leaf_rule_catalog.parquet"), pd.read_parquet(folder / "leaf_assignments.parquet")


def _rolling_one_month(panel: pd.DataFrame, context: list[str], raw_root: Path, base_lookup: pd.DataFrame, target_month: str) -> tuple[pd.DataFrame, dict[str, object]]:
    target_start = pd.Timestamp(target_month, tz="UTC")
    previous = (target_start - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    train_catalog, train_leaves = _load_raw(raw_root, previous)
    target_catalog, target_leaves = _load_raw(raw_root, target_month)
    train_catalog = train_catalog.copy(); target_catalog = target_catalog.copy()
    train_catalog["fold_id"] = train_catalog.fold_id.astype(str)
    target_catalog["fold_id"] = target_catalog.fold_id.astype(str)
    archetypes, mode, recurrence = _local_archetypes(train_catalog, 1)
    if not archetypes:
        raise RuntimeError(f"no local archetypes for {target_month}")
    local_catalog = pd.concat([train_catalog, target_catalog], ignore_index=True)
    matches, _ = match_catalog_to_archetypes(local_catalog, archetypes, temperature=MATCH_TEMPERATURE, unmatched_threshold=MATCH_THRESHOLD, top_n=TOP_N)
    mass = np.abs(local_catalog.ensemble_tree_contribution.to_numpy(float))
    mapping_quality = float(np.average(matches.best_similarity.to_numpy(float), weights=np.maximum(mass, 1e-12)))
    parts: dict[str, pd.DataFrame] = {}
    for month, cat, leaves in ((previous, train_catalog, train_leaves), (target_month, target_catalog, target_leaves)):
        m = matches.loc[matches.fold_id.astype(str).eq(month)].copy()
        feats, _ = materialize_row_archetype_exposures(leaves, cat, m, archetypes)
        parts[month] = pd.concat([leaves[["candidate_id", "__ts__"]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    lookup_cols = ["candidate_id", "__ts__", "month", "exact_net_bps", "exact_gross_bps", "base_score", *context]
    lookup = panel.loc[panel.month.isin([previous, target_month]), lookup_cols].copy()
    lookup = lookup.merge(base_lookup[["candidate_id", "__ts__", "base_expected_bps"]], on=["candidate_id", "__ts__"], how="left", validate="one_to_one")
    if lookup.base_expected_bps.isna().any():
        # This should only be possible for a missing historical fold artifact;
        # fail rather than silently remapping with held outcomes.
        raise RuntimeError(f"missing train-only base anchors for {target_month}: {int(lookup.base_expected_bps.isna().sum())}")
    rows = {}
    for month in (previous, target_month):
        rows[month] = parts[month].merge(lookup.loc[lookup.month.eq(month)], on=["candidate_id", "__ts__"], how="inner", validate="one_to_one")
    row_train, row_target = rows[previous], rows[target_month]
    archetype_fields = [f"archetype__{a.archetype_id}__abs_contribution" for a in archetypes]
    signed_fields = [f"archetype__{a.archetype_id}__signed_contribution" for a in archetypes]
    abs_train = row_train[archetype_fields].copy(); abs_train.columns = [a.archetype_id for a in archetypes]
    signed_train = row_train[signed_fields].copy(); signed_train.columns = [a.archetype_id for a in archetypes]
    residual = row_train.exact_net_bps.to_numpy(float) - row_train.base_expected_bps.to_numpy(float)
    contracts, cluster_audit, selected = _select_clusters(abs_train, signed_train, residual, row_train.month.astype(str).to_numpy(), mapping_quality)
    if not contracts:
        raise RuntimeError(f"no transportable cluster for {target_month}")
    train_cluster = materialize_memberships(abs_train, contracts)
    abs_target = row_target[archetype_fields].copy(); abs_target.columns = [a.archetype_id for a in archetypes]
    target_cluster = materialize_memberships(abs_target, contracts)
    for col in train_cluster.columns:
        row_train[col] = train_cluster[col].to_numpy()
        row_target[col] = target_cluster[col].to_numpy()
    zero_parts, target_memberships, counts = [], [], []
    for cluster in contracts:
        prefix = f"cluster__{cluster.cluster_id}__"
        mtr = row_train[f"{prefix}membership"].to_numpy(float)
        mte = row_target[f"{prefix}membership"].to_numpy(float)
        etr = row_train[f"{prefix}abs_contribution"].to_numpy(float)
        ete = row_target[f"{prefix}abs_contribution"].to_numpy(float)
        selected_fields, _ = _weighted_cmi_compat(row_train, context, residual, mtr, GAM_CONTEXT_CAP)
        p_zero, _ = _fit_vc(row_train, row_target, selected_fields, etr, ete, mtr, residual, intercept=False)
        zero_parts.append(p_zero); target_memberships.append(mte); counts.append(len(selected_fields))
    mtx = np.column_stack(target_memberships)
    zero = np.divide((mtx * np.column_stack(zero_parts)).sum(axis=1), np.maximum(mtx.sum(axis=1), 1e-8), out=np.zeros(len(row_target)), where=mtx.sum(axis=1) > 1e-8)
    out = row_target[["candidate_id", "__ts__", "month", "exact_net_bps", "exact_gross_bps", "base_expected_bps", "base_score"]].copy()
    out["target_month"] = target_month
    out["train_months"] = json.dumps([previous])
    out["archetype_mode"] = mode
    out["archetype_count"] = len(archetypes)
    out["rolling_cluster_count"] = len(contracts)
    out["rolling_transport_valid"] = bool(selected.get("valid", False))
    out["rolling_gam_zero_residual"] = zero
    out["gam_expected_bps"] = out.base_expected_bps.to_numpy(float) + 0.25 * zero
    out["gam_delta_bps"] = out.gam_expected_bps.to_numpy(float) - out.base_expected_bps.to_numpy(float)
    audit = {
        "target_month": target_month, "train_month": previous,
        "train_rows": len(row_train), "target_rows": len(row_target),
        "archetype_count": len(archetypes), "archetype_mode": mode,
        "mapping_quality": mapping_quality, "cluster_count": len(contracts),
        "transport_valid": bool(selected.get("valid", False)),
        "transport_score": float(selected.get("transport_score", np.nan)),
        "selected_context_mean_count": float(np.mean(counts)) if counts else 0.0,
    }
    return out, audit


def _weighted_cmi_compat(frame: pd.DataFrame, fields: list[str], target: np.ndarray, membership: np.ndarray, cap: int) -> tuple[list[str], pd.DataFrame]:
    # Import lazily to keep this transport utility's import graph small.
    from scripts.run_tp6_sl4_archetype_cluster_vcgam_oof import _weighted_cmi
    return _weighted_cmi(frame, fields, target, membership, cap)


def _score_oos(panel: pd.DataFrame, context: list[str], gam: pd.DataFrame, target_month: str, out: Path) -> pd.DataFrame:
    held = panel.loc[panel.month.eq(target_month)].copy()
    train = panel.loc[(panel.__ts__ < pd.Timestamp(target_month, tz="UTC")) & panel.label_available_ts.lt(pd.Timestamp(target_month, tz="UTC"))].copy()
    base_train, base_held = _map_base(train, held)
    gam_train = pd.DataFrame({"candidate_id": train.candidate_id, "__ts__": train.__ts__, "gam_delta_bps": 0.0})
    # Historical rows retain their prequential GAM output when available; the
    # missing warm-up rows are neutral, never outcome-imputed.
    historical = pd.read_parquet(ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/rolling_oof_predictions.parquet", columns=["candidate_id", "__ts__", "month", "window_months", "rolling_transport_valid", "rolling_gam_zero_gamma025", "base_expected_bps"])
    historical = historical.loc[historical.window_months.eq(1)].copy()
    historical["gam_delta_bps"] = np.where(historical.rolling_transport_valid.astype(bool), historical.rolling_gam_zero_gamma025 - historical.base_expected_bps, 0.0)
    hist = historical[["candidate_id", "__ts__", "gam_delta_bps"]]
    gam_train = gam_train.merge(hist, on=["candidate_id", "__ts__"], how="left", suffixes=("", "_hist"))
    gam_train["gam_delta_bps"] = gam_train["gam_delta_bps_hist"].fillna(gam_train["gam_delta_bps"]).astype(float)
    gam_held = gam.loc[:, ["candidate_id", "__ts__", "gam_delta_bps", "rolling_transport_valid"]].copy()
    held = held.merge(gam_held, on=["candidate_id", "__ts__"], how="inner", validate="one_to_one")
    train = train.merge(gam_train[["candidate_id", "__ts__", "gam_delta_bps"]], on=["candidate_id", "__ts__"], how="left", validate="one_to_one")
    train["gam_delta_bps"] = train.gam_delta_bps.fillna(0.0)
    train.attrs["context_fields"] = context; held.attrs["context_fields"] = context
    # Exact matched residual/meta contract: no BaseEV modulation, one GAM
    # disagreement input, and a hard target-month validity gate.
    control_consensus, control_residual_rank, _, _ = _fit_heads(
        train.copy(), held.copy(), base_train, base_held, use_gam_inputs=False,
        extra_fields=None, feature_fraction=1.0, month=target_month,
        seed_base=SEED,
    )
    consensus, residual_rank, residual_target, _ = _fit_heads(
        train, held, base_train, base_held, use_gam_inputs=False,
        extra_fields=["gam_delta_bps"], feature_fraction=1.0, month=target_month,
        seed_base=SEED,
    )
    base_rank = _pct(held.base_score.to_numpy(float), train.base_score.to_numpy(float))
    enhanced = (0.50 * base_rank + 0.25 * consensus + 0.25 * residual_rank).astype(float)
    control = (0.50 * base_rank + 0.25 * control_consensus + 0.25 * control_residual_rank).astype(float)
    gated = np.where(held.rolling_transport_valid.astype(bool), enhanced, control)
    # The target month is a single globally ranked population.  Keep both raw
    # scores and percentile-normalized scores so the audit can reproduce it.
    result = held[["candidate_id", "__ts__", "month", "side_name", "exact_net_bps", "exact_gross_bps", "rolling_transport_valid", "gam_delta_bps"]].copy()
    result["control_score"] = control
    result["gated_gam_score"] = gated
    result["base_rank"] = base_rank
    result["consensus_rank"] = consensus
    result["residual_rank"] = residual_rank
    result["residual_target_train_mean"] = float(np.mean(residual_target))
    result.to_parquet(out / "untouched_oos_predictions.parquet", index=False, compression="zstd")
    rows = []
    for score in ("control_score", "gated_gam_score"):
        for tail in (0.005, 0.01, 0.02, 0.05, 0.10):
            n = max(1, int(math.ceil(len(result) * tail)))
            top = result.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            rows.append({"target_month": target_month, "arm": score, "tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "transport_valid_fraction": float(result.rolling_transport_valid.mean()), "rank_ic": float(result[score].corr(result.exact_net_bps, method="spearman"))})
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(out / "untouched_oos_metrics.parquet", index=False)
    return metrics


def run(*, output_dir: Path = DEFAULT_OUT, target_month: str = "2026-07") -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    panel, context, context_hash = _load_panel()
    raw_root = _materialize_transport_paths(panel, context, output_dir)
    base_lookup = _base_expected_lookup(panel, raw_root)
    gam_parts, audits = [], []
    for month in MONTHS:
        part, audit = _rolling_one_month(panel, context, raw_root, base_lookup, month)
        gam_parts.append(part); audits.append(audit)
        part.to_parquet(output_dir / f"gam_month_{month}.parquet", index=False, compression="zstd")
    gam = pd.concat(gam_parts, ignore_index=True)
    gam.to_parquet(output_dir / "rolling_gam_2026.parquet", index=False, compression="zstd")
    metrics = _score_oos(panel, context, gam.loc[gam.target_month.eq(target_month)], target_month, output_dir)
    pd.DataFrame(audits).to_parquet(output_dir / "rolling_fit_audit.parquet", index=False)
    target_start = pd.Timestamp(target_month, tz="UTC")
    meta_train_rows = int((panel.__ts__.lt(target_start) & panel.label_available_ts.lt(target_start) & panel.label_valid).sum())
    target_rows = int((panel.month.eq(target_month) & panel.label_valid).sum())
    target_label_min = str(panel.loc[panel.month.eq(target_month), "label_available_ts"].min())
    target_label_max = str(panel.loc[panel.month.eq(target_month), "label_available_ts"].max())
    correctness = {
        "schema": "tp6_sl4_gam_untouched_oos_2026_correctness_v1",
        "target_month": target_month,
        "training_months_for_target_gam": [f"{(pd.Timestamp(target_month)-pd.offsets.MonthBegin(1)).strftime('%Y-%m')}"],
        "target_month_outcomes_used_in_gam_fit": False,
        "target_month_outcomes_used_in_meta_fit": False,
        "meta_train_rows": meta_train_rows,
        "target_rows": target_rows,
        "target_label_available_min": target_label_min,
        "target_label_available_max": target_label_max,
        "train_label_available_max": str(panel.loc[panel.__ts__.lt(target_start) & panel.label_available_ts.lt(target_start) & panel.label_valid, "label_available_ts"].max()),
        "one_canonical_gam_field": True,
        "gam_field": "gam_delta_bps",
        "base_ev_modulation": False,
        "transport_invalid_is_exact_control": True,
        "global_ranking_after_score_generation": True,
    }
    (output_dir / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "tp6_sl4_gam_untouched_oos_2026_v1",
        "status": "COMPLETE",
        "side": SIDE,
        "target_month": target_month,
        "source_panel": str(INPUT),
        "context_sha256": context_hash,
        "context_count": len(context),
        "rolling_window": 1,
        "gamma": 0.25,
        "canonical_field": "gam_delta_bps",
        "residual_meta": "native four-hour x side LambdaRank heads with one GAM disagreement field",
        "base_ev_modulation": False,
        "transport_invalid_rule": "exact control score",
        "target_month_outcomes_used_in_any_fit": False,
        "meta_train_rows": meta_train_rows,
        "target_rows": target_rows,
        "target_label_available_min": target_label_min,
        "target_label_available_max": target_label_max,
        "metrics": str(output_dir / "untouched_oos_metrics.parquet"),
        "correctness": str(output_dir / "correctness_test_report.json"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    report = [
        "# TP6/SL4 untouched 2026 one-month gated GAM OOS",
        "",
        "The target month is scored once. GAM/archetypes use the immediately preceding month; residual/meta uses only rows before the target month.",
        "",
        "## Metrics",
        "",
        metrics.round(3).to_string(index=False),
        "",
        "## Fit audit",
        "",
        pd.DataFrame(audits).round(3).to_string(index=False),
        "",
        "## Correctness",
        "",
        json.dumps(correctness, indent=2),
    ]
    (output_dir / "TP6_SL4_GAM_UNTOUCHED_OOS_2026_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(output_dir), "rows": len(gam), "metrics": len(metrics), "target_month": target_month}, indent=2))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-month", default="2026-07")
    args = parser.parse_args()
    run(output_dir=args.output_dir, target_month=args.target_month)
