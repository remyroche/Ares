#!/usr/bin/env python3
"""Frozen cross-fold cluster residual replay for the canonical TP6/SL4 stack.

This is the stable-contract counterpart to
``run_tp6_sl4_conditional_cluster_residual.py``.  A cluster contract is
discovered once on the designated 2024 development population and then reused
unchanged for every 2025 held month.  The per-cluster target is the requested
soft residual:

    membership(cluster) * (exact TP6/SL4 net - train-only base expected bps)

All configured causal/meta fields (including the strict OOF regime/transition
sidecar fields) are present before train-only CMI selection.  No held-month
outcome is used for cluster discovery, feature selection, or fitting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.conditional_cluster_residual import (
    ClusterContract,
    conditional_mi_scores,
    discover_family_clusters,
    materialize_cluster_features,
    soft_cluster_residual_target,
)
from extreme_price_movements.funnel_selection import global_tail_metrics
from scripts.run_tp6_sl4_conditional_cluster_residual import (
    _is_leak,
    _meta_pool,
)

SIDE = "long"
DEV_END = pd.Timestamp("2025-01-01", tz="UTC")
SEED = 20260812
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
CONTEXT_CAP = 16
MAX_TRAIN_ROWS = 120_000

DEFAULT_BASE = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_base_20260811_v1.parquet"
DEFAULT_FAMILY = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/meta_family_contribution_matrix.parquet"
DEFAULT_META = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_meta_pool_regime_20260811_v1.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_frozen_cluster_residual_20260812_v1"


def _numeric(frame: pd.DataFrame, fields: list[str], med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = frame[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = x.median().fillna(0.0)
    return x.fillna(med).fillna(0.0).astype("float32"), med


def _load(base_path: Path, family_path: Path, meta_path: Path, *, development_end: pd.Timestamp = DEV_END) -> tuple[pd.DataFrame, list[str], list[str]]:
    base = pd.read_parquet(base_path)
    base = base.loc[base.side_name.astype(str).str.lower().eq(SIDE)].copy()
    base["__ts__"] = pd.to_datetime(base["__ts__"], utc=True, errors="raise")
    base["label_available_ts"] = pd.to_datetime(base["label_available_ts"], utc=True, errors="raise")
    base = base.rename(columns={"exact_net_bps": "net_bps", "exact_gross_bps": "gross_bps"})
    required = {"candidate_id", "__ts__", "label_available_ts", "net_bps", "gross_bps", "base_expected_bps"}
    missing = sorted(required.difference(base.columns))
    if missing:
        raise ValueError(f"base panel missing {missing}")

    family = pd.read_parquet(family_path)
    family = family.loc[family.side_name.astype(str).str.lower().eq(SIDE)].copy()
    family["__ts__"] = pd.to_datetime(family["__ts__"], utc=True, errors="raise")
    family_fields = sorted(
        c for c in family.columns
        if c.startswith("meta_structural_family__") and not c.endswith("unassigned_mass")
    )
    abs_fields = [f"family_abs_share__{f}" for f in family_fields]
    conf_fields = [f"family_confidence_share__{f}" for f in family_fields]
    if not family_fields or not all(f in family.columns for f in abs_fields):
        raise ValueError("family matrix has incomplete structural family/absolute-share columns")
    family_keep = ["candidate_id", "__ts__", *family_fields, *abs_fields]
    family_keep += [f for f in conf_fields if f in family.columns]
    for f in ("family_assignment_quality", "family_low_confidence_mass", "family_total_abs_contribution", "family_unassigned_mass"):
        if f in family.columns:
            family_keep.append(f)
    family = family[family_keep].drop_duplicates(["candidate_id", "__ts__"])

    meta = pd.read_parquet(meta_path)
    meta = meta.loc[meta.side_name.astype(str).str.lower().eq(SIDE)].copy()
    meta["__ts__"] = pd.to_datetime(meta["__ts__"], utc=True, errors="raise")
    meta_schema = list(map(str, meta.columns))
    meta_fields, _, _ = _meta_pool(list(family.columns), meta_schema)
    # The pool must be selected from the full configured causal/meta contract;
    # identifiers and outcome-like fields remain audit-only.
    meta_fields = [f for f in meta_fields if f in meta.columns and not _is_leak(f)]
    if not meta_fields:
        raise ValueError("no configured causal/meta fields available")
    meta_keep = ["candidate_id", "__ts__", *meta_fields]
    meta = meta[meta_keep].drop_duplicates(["candidate_id", "__ts__"])

    frame = base.merge(family, on=["candidate_id", "__ts__"], how="inner", validate="one_to_one")
    frame = frame.merge(meta, on=["candidate_id", "__ts__"], how="inner", validate="one_to_one")
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError("empty canonical base/family/meta join")
    return frame, family_fields, meta_fields


def _freeze_contract(dev: pd.DataFrame, family_fields: list[str]) -> tuple[list[ClusterContract], pd.DataFrame]:
    abs_frame = dev[[f"family_abs_share__{f}" for f in family_fields]].copy()
    abs_frame.columns = family_fields
    signed = dev[family_fields].copy()
    discovered, audit = discover_family_clusters(abs_frame, signed, seed=SEED)
    # Stable IDs are independent of the fold-loop position and persist in the
    # output contract.  Family ordering is deterministic from the matrix schema.
    frozen: list[ClusterContract] = []
    for idx, contract in enumerate(discovered):
        frozen.append(ClusterContract(
            cluster_id=f"frozen_cluster_{idx:02d}",
            family_fields=tuple(sorted(contract.family_fields)),
            family_indices=tuple(family_fields.index(f) for f in sorted(contract.family_fields)),
            centroid_distance=contract.centroid_distance,
        ))
    return frozen, audit


def _context_frame(frame: pd.DataFrame, clusters: pd.DataFrame, cluster_ids: list[str], meta_fields: list[str]) -> pd.DataFrame:
    base_map = {
        "p_clear": "r3_meta_p_clear",
        "p_adverse": "r3_meta_p_adverse",
        "p_weak": "r3_meta_p_weak",
        "base_raw": "base_score",
        "base_score": "base_score",
        "base_expected_bps": "base_expected_bps",
    }
    cols: dict[str, pd.Series] = {}
    for dest, source in base_map.items():
        if source in frame.columns:
            cols[dest] = pd.to_numeric(frame[source], errors="coerce")
    out = pd.DataFrame(cols, index=frame.index)
    cluster_part = clusters.apply(pd.to_numeric, errors="coerce").copy()
    context_fields = [field for field in meta_fields if field in frame.columns and field not in out.columns and field not in cluster_part.columns]
    context_part = frame[context_fields].apply(pd.to_numeric, errors="coerce") if context_fields else pd.DataFrame(index=frame.index)
    out = pd.concat([out, cluster_part, context_part], axis=1, copy=False)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    # Explicitly expose cross-cluster competition and trust fields to the CMI
    # screen and downstream learners.
    memberships = [f"cluster__{c}__membership" for c in cluster_ids if f"cluster__{c}__membership" in out]
    if memberships:
        values = out[memberships].to_numpy(float)
        out["cluster_membership_max"] = values.max(axis=1)
        out["cluster_membership_second"] = np.partition(values, -2, axis=1)[:, -2] if values.shape[1] > 1 else 0.0
    return out.reset_index(drop=True)


def _fit_cluster(train_ctx: pd.DataFrame, test_ctx: pd.DataFrame, target: np.ndarray, weight: np.ndarray, selected: list[str], seed: int) -> np.ndarray:
    import lightgbm as lgb

    common = dict(
        objective="huber", alpha=0.85, n_estimators=220, learning_rate=0.03,
        max_depth=4, num_leaves=15, min_child_samples=300, min_sum_hessian_in_leaf=1.0,
        feature_fraction=0.82, bagging_fraction=0.82, bagging_freq=1,
        reg_alpha=0.05, reg_lambda=10.0, max_bin=127,
        random_state=seed, n_jobs=1, verbosity=-1,
    )
    fields = list(dict.fromkeys(selected))
    xtr, med = _numeric(train_ctx, fields)
    xte, _ = _numeric(test_ctx, fields, med)
    model = lgb.LGBMRegressor(**common)
    model.fit(xtr, target.astype(float), sample_weight=np.asarray(weight, dtype=float))
    return np.asarray(model.predict(xte), dtype=float)


def _tails(frame: pd.DataFrame, score: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    blocks = [("all", frame)] + [(str(m), g) for m, g in frame.groupby("month", sort=True)]
    for period, block in blocks:
        ranked = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
        for tail in TAILS:
            n = max(1, int(math.ceil(len(ranked) * tail)))
            selected = ranked.head(n)
            rows.append({
                "arm": score, "period": period, "tail": tail, "trades": int(n),
                "gross_bps_per_trade": float(selected.gross_bps.mean()),
                "net_bps_per_trade": float(selected.net_bps.mean()),
                "rank_ic": float(block[score].corr(block.net_bps, method="spearman")) if block[score].nunique() > 1 else np.nan,
            })
    return rows


def run(*, base_path: Path = DEFAULT_BASE, family_path: Path = DEFAULT_FAMILY, meta_path: Path = DEFAULT_META, out: Path = DEFAULT_OUT, development_end: pd.Timestamp = DEV_END, evaluation_start: pd.Timestamp | None = None, evaluation_end: pd.Timestamp | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    development_end = pd.Timestamp(development_end)
    development_end = development_end.tz_localize("UTC") if development_end.tz is None else development_end.tz_convert("UTC")
    evaluation_start = development_end if evaluation_start is None else pd.Timestamp(evaluation_start)
    evaluation_start = evaluation_start.tz_localize("UTC") if evaluation_start.tz is None else evaluation_start.tz_convert("UTC")
    if evaluation_start < development_end:
        raise ValueError("evaluation_start cannot precede development_end")
    if evaluation_end is not None:
        evaluation_end = pd.Timestamp(evaluation_end)
        evaluation_end = evaluation_end.tz_localize("UTC") if evaluation_end.tz is None else evaluation_end.tz_convert("UTC")
        if evaluation_end <= evaluation_start:
            raise ValueError("evaluation_end must be after evaluation_start")
    frame, family_fields, meta_fields = _load(base_path, family_path, meta_path, development_end=development_end)
    dev = frame.loc[frame.__ts__.lt(development_end) & frame.label_available_ts.lt(development_end)].copy()
    test_mask = frame.__ts__.ge(evaluation_start)
    if evaluation_end is not None:
        test_mask &= frame.__ts__.lt(evaluation_end)
    test = frame.loc[test_mask].copy()
    if len(dev) < 500 or len(test) < 500:
        raise ValueError(f"insufficient frozen-contract support: dev={len(dev)}, test={len(test)}")
    if len(dev) > MAX_TRAIN_ROWS:
        dev = dev.iloc[np.linspace(0, len(dev) - 1, MAX_TRAIN_ROWS, dtype=int)].copy()

    contracts, cluster_audit = _freeze_contract(dev, family_fields)
    cluster_ids = [c.cluster_id for c in contracts]
    contract_payload = {
        "schema": "tp6_sl4_frozen_cross_fold_cluster_contract_v1",
        "side": SIDE, "development_end": str(development_end), "evaluation_start": str(evaluation_start), "evaluation_end": str(evaluation_end) if evaluation_end is not None else None, "family_fields": family_fields,
        "cluster_count": len(contracts), "clusters": [c.to_dict() for c in contracts],
        "discovery": "structure-only coactivation/signed contribution KMeans on 2024 development rows",
        "same_contract_for_all_evaluation_rows": True,
    }
    (out / "frozen_cluster_contract.json").write_text(json.dumps(contract_payload, indent=2) + "\n")
    cluster_audit.to_parquet(out / "frozen_cluster_discovery_audit.parquet", index=False)

    dev_cluster = materialize_cluster_features(dev, contracts, family_fields=family_fields)
    test_cluster = materialize_cluster_features(test, contracts, family_fields=family_fields)
    dev_ctx = _context_frame(dev, dev_cluster, cluster_ids, meta_fields)
    test_ctx = _context_frame(test, test_cluster, cluster_ids, meta_fields)
    train_resid = dev.net_bps.to_numpy(float) - dev.base_expected_bps.to_numpy(float)
    correction = np.zeros(len(test), dtype=float)
    selected_rows: list[dict[str, object]] = []
    for idx, cluster_id in enumerate(cluster_ids):
        membership_field = f"cluster__{cluster_id}__membership"
        membership = dev_cluster[membership_field].to_numpy(float)
        target = soft_cluster_residual_target(train_resid, membership)
        # The selector sees every configured meta/regime field in dev_ctx.  It
        # is fit once on development rows and frozen for all 2025 predictions.
        cmi_fields = [f for f in meta_fields if f in dev_ctx.columns]
        cmi = conditional_mi_scores(dev_ctx, cmi_fields, target, membership)
        selected = cmi.head(CONTEXT_CAP).feature.tolist()
        if not selected:
            selected = []
        common = [
            "base_expected_bps", "p_clear", "p_adverse", "p_weak", "base_raw", "base_score",
            membership_field, f"cluster__{cluster_id}__abs_contribution",
            f"cluster__{cluster_id}__signed_contribution", f"cluster__{cluster_id}__confidence_share",
            f"cluster__{cluster_id}__active", "cluster_path_represented_mass", "cluster_path_unassigned_mass",
            "cluster_path_assignment_quality", "cluster_path_low_confidence_mass", "cluster_path_entropy",
            "cluster_path_top2_margin", "cluster_membership_max", "cluster_membership_second",
        ]
        common = [f for f in common if f in dev_ctx.columns]
        fields = list(dict.fromkeys(common + selected))
        correction += _fit_cluster(dev_ctx, test_ctx, target, 0.25 + 0.75 * membership, fields, SEED + idx)
        selected_rows.append({
            "cluster_id": cluster_id, "target": "membership * (net_bps - base_expected_bps)",
            "active_rows": int((membership > 0.10).sum()), "mean_membership": float(membership.mean()),
            "meta_pool_count": int(len(cmi_fields)), "selected_count": int(len(selected)),
            "selected_fields": json.dumps(selected),
        })
    result = test[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_expected_bps"]].copy()
    result["fold"] = result["month"].astype(str)
    result["base_score"] = result.base_expected_bps.to_numpy(float)
    result["frozen_cluster_correction"] = np.clip(correction, -200.0, 200.0)
    result["frozen_cluster_score"] = result.base_score.to_numpy(float) + result.frozen_cluster_correction.to_numpy(float)
    for field in test_cluster.columns:
        result[field] = test_cluster[field].to_numpy()
    result.to_parquet(out / "frozen_cluster_oos_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(selected_rows).to_parquet(out / "frozen_cluster_feature_selection.parquet", index=False, compression="zstd")
    metrics = pd.DataFrame(_tails(result, "base_score") + _tails(result, "frozen_cluster_score"))
    metrics.to_parquet(out / "frozen_cluster_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "tp6_sl4_frozen_cluster_residual_v1", "status": "complete", "side": SIDE,
        "base": str(base_path), "family": str(family_path), "meta": str(meta_path),
        "development_end": str(development_end), "evaluation_start": str(evaluation_start), "evaluation_end": str(evaluation_end) if evaluation_end is not None else None, "development_rows": int(len(dev)), "oos_rows": int(len(test)),
        "family_count": int(len(family_fields)), "meta_pool_count": int(len(meta_fields)),
        "cluster_count": int(len(contracts)), "context_cap": CONTEXT_CAP,
        "target": "one-cluster soft membership * (exact TP6/SL4 net - train-only base expected bps)",
        "cluster_contract_frozen_before_evaluation": True,
        "global_ranking": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    selected_fields = [field for row in selected_rows for field in json.loads(row["selected_fields"])]
    correctness = {
        "schema": "tp6_sl4_frozen_cluster_residual_correctness_v1",
        "development_end": str(development_end),
        "evaluation_start": str(evaluation_start),
        "development_labels_matured_before_cutoff": bool(dev.label_available_ts.lt(development_end).all()),
        "contract_frozen_before_evaluation": bool(development_end <= evaluation_start),
        "stable_cluster_ids_unique": bool(len(cluster_ids) == len(set(cluster_ids))),
        "stable_family_memberships_unique": bool(len({f for c in contracts for f in c.family_fields}) <= len(family_fields)),
        "meta_pool_before_selection_count": int(len(meta_fields)),
        "target_like_fields_selected": sorted({f for f in selected_fields if _is_leak(f)}),
        "target_like_fields_selected_count": int(sum(_is_leak(f) for f in selected_fields)),
        "held_outcomes_used_for_discovery_or_selection": False,
        "global_ranking_after_score_generation": True,
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    pooled = metrics.query("period == 'all'").sort_values(["tail", "net_bps_per_trade"], ascending=[True, False])
    lines = [
        "# TP6/SL4 frozen cross-fold cluster residual replay", "",
        f"Long-only. Cluster contract discovered once before {development_end.date()} and reused unchanged over [{evaluation_start.date()}, {evaluation_end.date() if evaluation_end is not None else 'end'}].", "",
        f"Family fields: {len(family_fields)}; full configured meta/regime pool before CMI: {len(meta_fields)}; frozen clusters: {len(contracts)}.", "",
        "| arm | tail | trades | gross bps/trade | net bps/trade | rank IC |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.tail:.3g} | {row.trades} | {row.gross_bps_per_trade:.2f} | {row.net_bps_per_trade:.2f} | {row.rank_ic:.4f} |")
    lines += ["", "The cluster contract is a stable feature contract; no held evaluation outcomes enter discovery or CMI selection."]
    (out / "FROZEN_CLUSTER_REPORT.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--development-end", type=str, default=str(DEV_END))
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--evaluation-end", type=str, default=None)
    args = parser.parse_args()
    print(run(
        base_path=args.base, family_path=args.family, meta_path=args.meta, out=args.out,
        development_end=pd.Timestamp(args.development_end),
        evaluation_start=pd.Timestamp(args.evaluation_start) if args.evaluation_start else None,
        evaluation_end=pd.Timestamp(args.evaluation_end) if args.evaluation_end else None,
    ))
