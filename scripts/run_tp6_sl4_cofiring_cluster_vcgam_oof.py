#!/usr/bin/env python3
"""Two-stage co-firing/economic cluster + varying-coefficient GAM replay.

Stage 1 discovers a single frozen structural contract from the pre-2025
development population.  Discovery uses co-firing Jaccard/NPMI, contribution
profile coherence, train-only economic coherence, balance, and a later
development block's held-out active/inactive differentiation.  The selected
K and weights are then refit on all pre-2025 development rows.

Stage 2 fits one causal varying-coefficient GAM per frozen cluster at each
held 2025 month.  The target is the ordinary signed TP6/SL4 residual:

    residual_bps = exact_net_bps - base_expected_bps

Cluster membership is used only as exposure/sample weight.  It is never
multiplied into the target.  The GAM design is deliberately cluster-specific:

    prediction = alpha_k + exposure_k * (beta_k + delta_k(X))

where ``delta_k`` is an additive spline of causal context fields.  This tests
whether conditions modulate the value of this cluster, rather than giving a
free smoother another opportunity to rediscover generic alpha.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.cofiring_economic_clusters import (  # noqa: E402
    CoFiringClusterContract,
    cluster_conditional_differentiation,
    discover_best_contract,
    materialize_memberships,
    refit_contract,
)
from scripts.run_tp6_sl4_frozen_cluster_residual import _load  # noqa: E402
from extreme_price_movements.conditional_cluster_residual import conditional_mi_scores  # noqa: E402


SIDE = "long"
DEV_END = pd.Timestamp("2025-01-01", tz="UTC")
SEED = 20260813
CONTEXT_CAP = 12
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
GAMMAS = (0.25, 0.50, 1.00)
DEFAULT_BASE = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_base_20260811_v1.parquet"
DEFAULT_FAMILY = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/meta_family_contribution_matrix.parquet"
DEFAULT_META = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_meta_pool_regime_20260811_v1.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_cofiring_cluster_vcgam_oof_20260813_v1"


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 32 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def _prep(frame: pd.DataFrame, fields: Sequence[str], med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    z = frame.reindex(columns=list(fields)).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = z.median().fillna(0.0)
        med.attrs["scale"] = ((z - med).abs().median().replace(0.0, 1.0).fillna(1.0)).to_dict()
    scale = pd.Series(med.attrs.get("scale", {}), dtype=float).reindex(fields).fillna(1.0)
    return ((z.fillna(med).fillna(0.0) - med) / scale).clip(-20.0, 20.0).astype("float32"), med


def _fit_vc_gam(
    train: pd.DataFrame,
    held: pd.DataFrame,
    context_fields: Sequence[str],
    exposure_train: np.ndarray,
    exposure_held: np.ndarray,
    membership_train: np.ndarray,
    residual_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit alpha + exposure*(beta + additive spline(context))."""

    fields = list(context_fields)
    if fields:
        xtr, med = _prep(train, fields)
        xte, _ = _prep(held, fields, med)
        spline = SplineTransformer(
            n_knots=3, degree=2, knots="quantile", extrapolation="linear", include_bias=False
        )
        btr = spline.fit_transform(xtr)
        bte = spline.transform(xte)
    else:
        btr = np.zeros((len(train), 0), dtype=np.float32)
        bte = np.zeros((len(held), 0), dtype=np.float32)

    positive = np.asarray(exposure_train, float) > 0
    scale = float(np.nanmedian(np.asarray(exposure_train, float)[positive])) if positive.any() else 1.0
    scale = max(scale, 1e-5)
    etr = np.clip(np.asarray(exposure_train, float) / scale, 0.0, 20.0)
    ete = np.clip(np.asarray(exposure_held, float) / scale, 0.0, 20.0)
    design_tr = np.column_stack([etr, etr[:, None] * btr])
    design_te = np.column_stack([ete, ete[:, None] * bte])
    weights = np.clip(np.asarray(membership_train, float), 0.0, 1.0)
    ok = np.isfinite(residual_train) & np.isfinite(weights) & (weights > 1e-8)
    if int(ok.sum()) < 128 or float(weights[ok].sum()) < 64.0:
        return np.zeros(len(held), dtype=float), np.zeros(len(train), dtype=float)
    model = Ridge(alpha=20.0)
    model.fit(design_tr[ok], np.asarray(residual_train, float)[ok], sample_weight=weights[ok])
    train_pred = np.clip(np.asarray(model.predict(design_tr), float), -1000.0, 1000.0)
    held_pred = np.clip(np.asarray(model.predict(design_te), float), -1000.0, 1000.0)
    return held_pred, train_pred


def _select_context(train_ctx: pd.DataFrame, fields: Sequence[str], residual: np.ndarray, membership: np.ndarray) -> tuple[list[str], pd.DataFrame]:
    available = [f for f in fields if f in train_ctx.columns]
    audit = conditional_mi_scores(train_ctx, available, residual, membership)
    selected = audit.head(CONTEXT_CAP).feature.astype(str).tolist() if not audit.empty else []
    audit = audit.assign(selected=audit.feature.isin(selected)) if not audit.empty else audit
    return selected, audit


def _global_metrics(frame: pd.DataFrame, score_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_rows, month_rows, stability_rows = [], [], []
    for col in score_cols:
        for tail in TAILS:
            n = max(1, int(math.ceil(len(frame) * tail)))
            top = frame.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            global_rows.append({
                "arm": col, "scope": "global_2025", "tail": tail, "trades": n,
                "gross_bps_per_trade": float(top.gross_bps.mean()), "net_bps_per_trade": float(top.net_bps.mean()),
                "rank_ic": _rank_ic(frame[col].to_numpy(float), frame.net_bps.to_numpy(float)),
            })
        values, ics = [], []
        for month, block in frame.groupby("month", sort=True):
            n = max(1, int(math.ceil(len(block) * 0.05)))
            top = block.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            values.append(float(top.net_bps.mean()))
            ics.append(_rank_ic(block[col].to_numpy(float), block.net_bps.to_numpy(float)))
            month_rows.append({
                "arm": col, "month": str(month), "tail": 0.05, "trades": n,
                "gross_bps_per_trade": float(top.gross_bps.mean()), "net_bps_per_trade": values[-1],
                "rank_ic": ics[-1],
            })
        arr = np.asarray(values, float)
        med = float(np.median(arr)); mad = float(np.median(np.abs(arr - med)))
        stability_rows.append({
            "arm": col, "months": len(arr), "mean_top5_net_bps": float(np.mean(arr)),
            "median_top5_net_bps": med, "mad_top5_net_bps": mad,
            "worst_month_top5_net_bps": float(np.min(arr)), "positive_months_top5": int(np.sum(arr > 0)),
            "portability_score_bps": med - 0.75 * mad - max(0.0, -float(np.min(arr))),
            "mean_month_rank_ic": float(np.nanmean(ics)),
        })
    return pd.DataFrame(global_rows), pd.DataFrame(month_rows), pd.DataFrame(stability_rows)


def _cluster_metrics(
    held: pd.DataFrame,
    pred: np.ndarray,
    membership: np.ndarray,
    residual: np.ndarray,
    cluster_id: str,
    month: str,
) -> dict[str, object]:
    active = membership > 0.05
    w = np.clip(membership, 0.0, 1.0)
    return {
        "month": month, "cluster_id": cluster_id, "rows": len(held),
        "active_rows": int(active.sum()), "mean_membership": float(membership.mean()),
        "weighted_residual_mean_bps": float(np.sum(w * residual) / max(np.sum(w), 1e-12)),
        "active_residual_mean_bps": float(np.mean(residual[active])) if active.any() else np.nan,
        "active_rank_ic": _rank_ic(pred[active], residual[active]) if active.sum() >= 32 else np.nan,
        "active_net_mean_bps": float(np.mean(held.loc[active, "net_bps"])) if active.any() else np.nan,
        "active_gross_mean_bps": float(np.mean(held.loc[active, "gross_bps"])) if active.any() else np.nan,
        "active_vs_inactive_residual_bps": float(np.mean(residual[active]) - np.mean(residual[~active])) if active.any() and (~active).any() else np.nan,
    }


def run(
    *,
    base_path: Path = DEFAULT_BASE,
    family_path: Path = DEFAULT_FAMILY,
    meta_path: Path = DEFAULT_META,
    out: Path = DEFAULT_OUT,
    development_end: pd.Timestamp = DEV_END,
) -> Path:
    if out.exists():
        raise FileExistsError(out)
    development_end = pd.Timestamp(development_end)
    development_end = development_end.tz_localize("UTC") if development_end.tz is None else development_end.tz_convert("UTC")
    frame, family_fields, meta_fields = _load(base_path, family_path, meta_path, development_end=development_end)
    all_family_fields = [f for f in family_fields if not f.endswith("unassigned_mass")]
    all_abs_fields = [f"family_abs_share__{f}" for f in all_family_fields]
    if not all(f in frame.columns for f in all_abs_fields):
        raise ValueError("canonical family matrix lacks absolute-share fields")
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    frame["residual_bps"] = frame["net_bps"].to_numpy(float) - frame["base_expected_bps"].to_numpy(float)
    dev = frame.loc[frame.__ts__.lt(development_end) & frame.label_available_ts.lt(development_end)].copy()
    held = frame.loc[frame.__ts__.ge(development_end)].copy()
    months = sorted(dev.month.astype(str).unique())
    if len(months) < 4 or len(held) < 500:
        raise ValueError(f"insufficient development/OOS support: dev={len(dev)}, held={len(held)}, months={months}")
    split = max(2, len(months) - 2)
    discovery_months, validation_months = months[:split], months[split:]
    disc = dev.loc[dev.month.astype(str).isin(discovery_months)].copy()
    validation = dev.loc[dev.month.astype(str).isin(validation_months)].copy()
    family_support_rows = []
    for field in all_family_fields:
        disc_rate = float((disc[f"family_abs_share__{field}"] > 1e-8).mean())
        validation_rate = float((validation[f"family_abs_share__{field}"] > 1e-8).mean())
        family_support_rows.append({"family": field, "discovery_active_rate": disc_rate, "validation_active_rate": validation_rate, "transportable": bool(disc_rate >= 0.05 and validation_rate >= 0.05)})
    family_support_audit = pd.DataFrame(family_support_rows)
    family_fields = family_support_audit.loc[family_support_audit.transportable, "family"].astype(str).tolist()
    if len(family_fields) < 6:
        raise ValueError(f"fewer than six transportable family fields after support audit: {len(family_fields)}")
    abs_fields = [f"family_abs_share__{f}" for f in family_fields]
    abs_disc = disc[abs_fields].copy(); abs_disc.columns = family_fields
    abs_val = validation[abs_fields].copy(); abs_val.columns = family_fields
    contrib_disc = disc[family_fields].copy()
    contracts0, candidate_audit, pair_audit, val_diff = discover_best_contract(
        abs_disc, contrib_disc, disc.residual_bps.to_numpy(float), abs_val, validation.residual_bps.to_numpy(float),
        seed=SEED,
    )
    selected_k = len(contracts0)
    abs_dev = dev[abs_fields].copy(); abs_dev.columns = family_fields
    contrib_dev = dev[family_fields].copy()
    contracts, pair_refit, family_audit = refit_contract(abs_dev, contrib_dev, dev.residual_bps.to_numpy(float), k=selected_k)
    all_abs = frame[abs_fields].copy(); all_abs.columns = family_fields
    cluster_features = materialize_memberships(all_abs, contracts)
    cluster_ids = [c.cluster_id for c in contracts]
    for col in cluster_features.columns:
        frame[col] = cluster_features[col].to_numpy()

    out.mkdir(parents=True)
    contract_payload = {
        "schema": "tp6_sl4_cofiring_economic_cluster_contract_v1",
        "side": SIDE, "development_end": str(development_end),
        "discovery_months": discovery_months, "validation_months": validation_months,
        "family_count": len(family_fields), "meta_pool_count": len(meta_fields),
        "cluster_count": len(contracts), "clusters": [c.to_dict() for c in contracts],
        "similarity": "Jaccard + NPMI + contribution-profile coherence + economic coherence",
        "selection": "compactness + balance + held-out development differentiation + silhouette",
        "membership_is_exposure_not_target": True,
    }
    (out / "cofiring_cluster_contract.json").write_text(json.dumps(contract_payload, indent=2) + "\n")
    candidate_audit.to_parquet(out / "cluster_candidate_audit.parquet", index=False)
    pair_refit.to_parquet(out / "pair_similarity_refit.parquet", index=False)
    pair_audit.to_parquet(out / "pair_similarity_discovery.parquet", index=False)
    family_audit.to_parquet(out / "family_economic_audit.parquet", index=False)
    family_support_audit.to_parquet(out / "family_transport_support_audit.parquet", index=False)
    val_diff.to_parquet(out / "development_validation_differentiation.parquet", index=False)

    all_context_audit: list[pd.DataFrame] = []
    gam_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    held_months = sorted(held.month.astype(str).unique())
    for month in held_months:
        cutoff = pd.Timestamp(month, tz="UTC")
        train = frame.loc[frame.__ts__.lt(cutoff) & frame.label_available_ts.lt(cutoff)].copy()
        test = frame.loc[frame.month.astype(str).eq(month)].copy()
        if len(train) < 500 or test.empty:
            continue
        train_pred_by_cluster: list[np.ndarray] = []
        test_pred_by_cluster: list[np.ndarray] = []
        test_membership: list[np.ndarray] = []
        for cluster in contracts:
            prefix = f"cluster__{cluster.cluster_id}__"
            m_tr = train[f"{prefix}membership"].to_numpy(float)
            m_te = test[f"{prefix}membership"].to_numpy(float)
            e_tr = train[f"{prefix}abs_contribution"].to_numpy(float)
            e_te = test[f"{prefix}abs_contribution"].to_numpy(float)
            ctx = train[meta_fields].copy()
            selected, cmi = _select_context(ctx, meta_fields, train.residual_bps.to_numpy(float), m_tr)
            if not cmi.empty:
                all_context_audit.append(cmi.assign(month=month, cluster_id=cluster.cluster_id, train_rows=len(train)))
            tr_pred, held_pred = _fit_vc_gam(
                train, test, selected, e_tr, e_te, m_tr, train.residual_bps.to_numpy(float)
            )
            # _fit_vc_gam returns held then train; keep the held prediction.
            cluster_pred = tr_pred
            train_pred_by_cluster.append(held_pred)
            test_pred_by_cluster.append(cluster_pred)
            test_membership.append(m_te)
            cluster_rows.append(_cluster_metrics(test, cluster_pred, m_te, test.residual_bps.to_numpy(float), cluster.cluster_id, month))
            gam_rows.append({
                "month": month, "cluster_id": cluster.cluster_id, "train_rows": len(train),
                "held_rows": len(test), "active_train_rows": int((m_tr > 0.05).sum()),
                "active_held_rows": int((m_te > 0.05).sum()), "selected_count": len(selected),
                "selected_fields": json.dumps(selected),
                "target": "exact_net_bps - base_expected_bps; membership is sample weight/exposure",
                "model": "alpha + exposure * (beta + additive spline(context))",
                "held_rank_ic": _rank_ic(cluster_pred, test.residual_bps.to_numpy(float)),
            })
        m = np.column_stack(test_membership)
        p = np.column_stack(test_pred_by_cluster)
        aggregate = np.divide((m * p).sum(axis=1), np.maximum(m.sum(axis=1), 1e-8), out=np.zeros(len(test)), where=m.sum(axis=1) > 1e-8)
        out_frame = test[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_expected_bps", "base_score", "residual_bps"]].copy()
        out_frame["cofire_vcgam_residual"] = aggregate
        out_frame["cofire_vcgam_raw_score"] = out_frame.base_expected_bps.to_numpy(float) + aggregate
        for gamma in GAMMAS:
            out_frame[f"cofire_vcgam_gamma{int(gamma * 100):03d}"] = out_frame.base_expected_bps.to_numpy(float) + gamma * aggregate
        out_frame["cluster_membership_total"] = m.sum(axis=1)
        prediction_parts.append(out_frame)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    score_cols = ["base_expected_bps", "cofire_vcgam_raw_score", *[f"cofire_vcgam_gamma{int(g * 100):03d}" for g in GAMMAS]]
    global_metrics, monthly_metrics, stability_metrics = _global_metrics(predictions, score_cols)
    predictions.to_parquet(out / "cofiring_vcgam_oof_predictions.parquet", index=False, compression="zstd")
    global_metrics.to_parquet(out / "metrics_global.parquet", index=False)
    monthly_metrics.to_parquet(out / "metrics_monthly.parquet", index=False)
    stability_metrics.to_parquet(out / "metrics_stability.parquet", index=False)
    pd.DataFrame(gam_rows).to_parquet(out / "cluster_gam_fit_audit.parquet", index=False)
    pd.DataFrame(cluster_rows).to_parquet(out / "cluster_own_metrics.parquet", index=False)
    if all_context_audit:
        pd.concat(all_context_audit, ignore_index=True).to_parquet(out / "cluster_gam_context_selection.parquet", index=False)

    selected_fields = []
    if all_context_audit:
        ca = pd.concat(all_context_audit, ignore_index=True)
        selected_fields = ca.loc[ca.selected, "feature"].astype(str).tolist()
    correctness = {
        "schema": "tp6_sl4_cofiring_cluster_vcgam_correctness_v1",
        "rows_scored": int(len(predictions)), "candidate_ids_unique": bool(predictions.candidate_id.is_unique),
        "all_scores_finite": bool(np.isfinite(predictions[score_cols].to_numpy(float)).all()),
        "development_labels_matured_before_each_held_month": True,
        "contract_frozen_before_2025": True,
        "membership_multiplied_into_target": False,
        "membership_used_as_sample_weight_or_exposure": True,
        "target": "exact_net_bps - base_expected_bps",
        "target_like_context_fields_selected": sorted(set(x for x in selected_fields if any(t in x.lower() for t in ("net_bps", "gross_bps", "policy_", "label_available", "future", "mfe", "mae")))),
        "held_2025_outcomes_used_for_contract_or_context_selection": False,
        "global_ranking_after_score_generation": True,
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "tp6_sl4_cofiring_cluster_vcgam_oof_v1", "status": "COMPLETE", "side": SIDE,
        "base": str(base_path), "family": str(family_path), "meta": str(meta_path),
        "development_end": str(development_end), "development_rows": len(dev), "held_rows": len(predictions),
        "family_count": len(family_fields), "meta_pool_count": len(meta_fields), "cluster_count": len(contracts),
        "gam_context_cap": CONTEXT_CAP, "gammas": list(GAMMAS),
        "contract": "co-firing Jaccard/NPMI + contribution/economic coherence + balance + held-out differentiation",
        "gam": "cluster-specific alpha + exposure*(beta + additive spline(context)), Ridge(alpha=20), membership-weighted",
        "target": "ordinary signed net residual; membership never multiplied into target",
        "global_ranking": True, "artifacts": sorted(x.name for x in out.iterdir()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# TP6/SL4 co-firing/economic frozen clusters + varying-coefficient GAM OOF replay", "",
        f"Long-only; {len(contracts)} clusters discovered with a frozen contract before 2025. Membership is exposure/weight, not target multiplication.", "",
        "## Global metrics", "", global_metrics.round(3).to_string(index=False), "", "## Stability", "", stability_metrics.round(3).to_string(index=False),
    ]
    (out / "TP6_SL4_COFIRING_CLUSTER_VCGAM_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(out), "rows": len(predictions), "clusters": len(contracts), "global_metric_rows": len(global_metrics)}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--development-end", type=str, default=str(DEV_END))
    args = parser.parse_args()
    run(base_path=args.base, family_path=args.family, meta_path=args.meta, out=args.out, development_end=pd.Timestamp(args.development_end))
