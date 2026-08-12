#!/usr/bin/env python3
"""Causal 2025 replay with downstream layers trained on TP6/SL4 net labels.

The input is the strict-OOS R3 TP6/SL4 panel produced by the Stage-I target
ablation.  The R3 probabilities are treated as a frozen base output.  Before
each 2025 month, a train-only isotonic map converts the base score to bps;
consensus and residual LambdaRank heads are then fit to TP6/SL4 net residual
grades.  Ranking is pooled globally after side/month percentile normalization.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/remyroche/Documents/Ares")
INPUT = ROOT / "data_perp/artifacts/r3_tp6_sl4_meta_target_ablation_20260803_v1/r3_meta_target_oof_predictions.parquet"
SELECTED = ROOT / "data_perp/artifacts/r3_plus_meta_tp6_sl4_ablation_20260803_v1/r3_plus_meta_metrics.parquet"
OUT = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1"
MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 13))
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
SEED = 20260807

IDENTITY = {
    "candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts",
    "label_available_ts", "exact_net_bps", "exact_gross_bps", "label_valid",
    "r3_class", "r3_metric_target", "source_month", "population_segment",
    "selector_month", "selector_economic_bin", "fold",
}
BASE_OUTPUTS = {
    "r3_meta_p_adverse", "r3_meta_p_weak", "r3_meta_p_clear",
    "r3_meta_opportunity_score", "r3_meta2_p_adverse", "r3_meta2_p_weak",
    "r3_meta2_p_clear", "r3_meta2_opportunity_score",
}


def _selected_context() -> list[str]:
    """Freeze the context union from the earlier pre-2025 TP6 selection."""
    if SELECTED.exists():
        m = pd.read_parquet(SELECTED, columns=["selected_context_features"])
        values: list[str] = []
        for value in m["selected_context_features"].dropna():
            if isinstance(value, str):
                value = ast.literal_eval(value)
            values.extend(map(str, value))
        fields = sorted(set(values))
        if len(fields) >= 25:
            return fields
    return []


def _load() -> tuple[pd.DataFrame, list[str], str]:
    context = _selected_context()
    probe = pd.read_parquet(INPUT, columns=None)
    if not context:
        context = sorted(
            c for c in probe.columns
            if c not in IDENTITY and c not in BASE_OUTPUTS
            and not c.startswith("r3_") and pd.api.types.is_numeric_dtype(probe[c])
        )
    missing = sorted(set(context).difference(probe.columns))
    if missing:
        raise ValueError(f"selected context fields absent from panel: {missing}")
    fields = [c for c in context if probe[c].notna().mean() >= 0.90 and probe[c].nunique(dropna=True) > 1]
    if len(fields) < 25:
        raise ValueError(f"too few usable causal context fields: {len(fields)}")
    cols = [
        "candidate_id", "__ts__", "side_name", "label_available_ts",
        "exact_net_bps", "exact_gross_bps", "label_valid",
        "r3_meta_p_adverse", "r3_meta_p_weak", "r3_meta_p_clear", *fields,
    ]
    x = probe.loc[:, list(dict.fromkeys(cols))].copy()
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x["label_available_ts"] = pd.to_datetime(x["label_available_ts"], utc=True)
    x["month"] = x["__ts__"].dt.strftime("%Y-%m")
    x["base_score"] = pd.to_numeric(x["r3_meta_p_clear"], errors="coerce") - 0.5 * pd.to_numeric(x["r3_meta_p_adverse"], errors="coerce")
    valid = x["label_valid"].fillna(False) & np.isfinite(x["exact_net_bps"]) & np.isfinite(x["exact_gross_bps"])
    x = x.loc[valid & x["month"].isin([*MONTHS, *[f"2024-{m:02d}" for m in range(2, 13)]])].copy()
    if x.candidate_id.duplicated().any():
        raise ValueError("input panel contains duplicate candidate IDs")
    if x.empty:
        raise ValueError("empty valid TP6/SL4 panel")
    digest = hashlib.sha256("\n".join(fields).encode()).hexdigest()
    return x.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True), fields, digest


def _prep(frame: pd.DataFrame, fields: list[str], med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    z = frame.reindex(columns=fields).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = z.median().fillna(0.0)
        med.attrs["scale"] = ((z - med).abs().median().replace(0.0, 1.0).fillna(1.0)).to_dict()
    scale = pd.Series(med.attrs.get("scale", {}), dtype=float).reindex(fields).fillna(1.0)
    return ((z.fillna(med).fillna(0.0) - med) / scale).clip(-20.0, 20.0).astype("float32"), med


def _month_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["month"].value_counts()
    w = frame["month"].map((1.0 / counts).to_dict()).to_numpy(float)
    return (w * len(w) / max(w.sum(), 1e-12)).astype("float32")


def _group(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = pd.to_datetime(frame["__ts__"], utc=True).dt.floor("4h").astype("int64").astype(str)
    order = np.argsort(q.to_numpy(), kind="stable")
    qs = q.iloc[order]
    counts = qs.groupby(qs, sort=False).size()
    valid = counts.index[counts.to_numpy() >= 2]
    keep = qs.isin(valid).to_numpy()
    order = order[keep]
    groups = qs.iloc[keep].groupby(qs.iloc[keep], sort=False).size().to_numpy(dtype=np.int32)
    return order, groups


def _rank_fit(
    train: pd.DataFrame,
    held: pd.DataFrame,
    fields: list[str],
    label: np.ndarray,
    *,
    equal_month: bool,
    seed: int,
    return_model: bool = False,
):
    x, med = _prep(train, fields)
    order, groups = _group(train)
    if len(groups) == 0:
        empty = (np.zeros(len(train), dtype=np.float32), np.zeros(len(held), dtype=np.float32))
        return empty
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10,
        n_estimators=140, learning_rate=0.035, max_depth=5, num_leaves=31,
        min_child_samples=180, feature_fraction=0.82, bagging_fraction=0.82,
        bagging_freq=1, lambda_l1=0.02, lambda_l2=2.0, max_bin=127,
        label_gain=[0.0, 0.25, 1.0, 3.0, 7.0], random_state=seed,
        n_jobs=4, verbosity=-1,
    )
    w = _month_weights(train) if equal_month else np.ones(len(train), dtype=np.float32)
    model.fit(x.iloc[order], label[order], group=groups, sample_weight=w[order])
    train_raw = np.asarray(model.predict(x), dtype=np.float32)
    held_x, _ = _prep(held, fields, med)
    held_raw = np.asarray(model.predict(held_x), dtype=np.float32)
    if return_model:
        # The canonical path materializer needs the exact matrices used by
        # LightGBM, not a re-created approximation.  The caller owns the
        # returned model and matrices and releases them after materialisation.
        return train_raw, held_raw, model, x, held_x
    del model, x, held_x
    gc.collect()
    return train_raw, held_raw


def _pct(value: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(ref, dtype=float))
    return (np.searchsorted(ref, value, side="right") / max(len(ref), 1)).astype("float32")


def _map_base(train: pd.DataFrame, held: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(train.base_score) & np.isfinite(train.exact_net_bps)
    if ok.sum() < 100:
        return np.full(len(train), float(train.exact_net_bps.mean()), dtype=np.float32), np.full(len(held), float(train.exact_net_bps.mean()), dtype=np.float32)
    model = IsotonicRegression(out_of_bounds="clip", y_min=-1000.0, y_max=1000.0)
    model.fit(train.loc[ok, "base_score"], train.loc[ok, "exact_net_bps"])
    return model.predict(train.base_score).astype("float32"), model.predict(held.base_score).astype("float32")


def _score_month(
    held: pd.DataFrame,
    base_rank: np.ndarray,
    consensus: np.ndarray,
    residual: np.ndarray,
    base_anchor: np.ndarray,
    context_fields: list[str],
) -> pd.DataFrame:
    keep = [
        "candidate_id", "__ts__", "label_available_ts", "side_name", "month", "exact_net_bps", "exact_gross_bps",
        "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak",
        *context_fields,
    ]
    out = held.loc[:, list(dict.fromkeys(keep))].copy()
    out["base_expected_bps"] = np.asarray(base_anchor, dtype=np.float32)
    out["base_rank"] = base_rank
    out["consensus_rank"] = consensus
    out["residual_rank"] = residual
    out["base_only"] = out.base_rank
    out["base_plus_consensus25"] = 0.75 * out.base_rank + 0.25 * out.consensus_rank
    out["base_plus_residual25"] = 0.75 * out.base_rank + 0.25 * out.residual_rank
    out["full_base_consensus_residual"] = 0.50 * out.base_rank + 0.25 * out.consensus_rank + 0.25 * out.residual_rank
    return out


def _metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arms = ["base_only", "consensus_only", "residual_only", "base_plus_consensus25", "base_plus_residual25", "full_base_consensus_residual"]
    glob, month, stab = [], [], []
    for arm in arms:
        for tail in TAILS:
            n = max(1, int(math.ceil(len(pred) * tail)))
            top = pred.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            glob.append({"arm": arm, "scope": "global_2025", "tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": float(pred[[arm, "exact_net_bps"]].corr(method="spearman").iloc[0, 1])})
        vals = []
        for m, g in pred.groupby("month", sort=True):
            n = max(1, int(math.ceil(len(g) * 0.05)))
            top = g.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            vals.append(float(top.exact_net_bps.mean()))
            month.append({"arm": arm, "month": m, "tail": 0.05, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": vals[-1], "rank_ic": float(g[[arm, "exact_net_bps"]].corr(method="spearman").iloc[0, 1])})
        a = np.asarray(vals, dtype=float)
        med = float(np.nanmedian(a))
        stab.append({"arm": arm, "months": len(a), "mean_top5_net_bps": float(np.nanmean(a)), "median_top5_net_bps": med, "mad_top5_net_bps": float(np.nanmedian(np.abs(a - med))), "worst_month_top5_net_bps": float(np.nanmin(a)), "positive_months_top5": int(np.sum(a > 0.0)), "mean_month_rank_ic": float(np.nanmean([x["rank_ic"] for x in month if x["arm"] == arm]))})
    return pd.DataFrame(glob), pd.DataFrame(month), pd.DataFrame(stab)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, context, digest = _load()
    cap_sizes = (25, 40, 60, min(73, len(context)))
    heads: list[dict] = []
    parts: list[pd.DataFrame] = []
    for month in MONTHS:
        held = x.loc[x.month.eq(month)].copy()
        train = x.loc[(x.__ts__ < pd.Timestamp(month, tz="UTC")) & (x.label_available_ts < pd.Timestamp(month, tz="UTC"))].copy()
        if len(held) == 0 or len(train) < 1000:
            continue
        for side in ("long", "short"):
            tr = train.loc[train.side_name.eq(side)].copy(); te = held.loc[held.side_name.eq(side)].copy()
            if len(tr) < 300 or len(te) == 0:
                continue
            tr_anchor, te_anchor = _map_base(tr, te)
            tr["base_anchor"] = tr_anchor; te["base_anchor"] = te_anchor
            residual = tr.exact_net_bps.to_numpy(float) - tr.base_anchor.to_numpy(float)
            consensus_raws = []
            for cap in cap_sizes:
                fields = ["base_anchor", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context[:cap]]
                for weighting in (False, True):
                    grade = np.digitize(residual, [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
                    raw_tr, raw_te = _rank_fit(tr, te, fields, grade, equal_month=weighting, seed=SEED + int(month[-2:]) * 100 + cap + int(weighting))
                    consensus_raws.append((_pct(raw_te, raw_tr), raw_tr, f"cap{cap}_{'equal_month' if weighting else 'ordinary'}"))
                    heads.append({"month": month, "side": side, "layer": "consensus", "cap": cap, "equal_month": weighting, "rows": len(tr), "query_groups": int(_group(tr)[1].size), "fields": len(fields)})
            te_consensus = np.nanmedian(np.column_stack([r[0] for r in consensus_raws]), axis=1).astype(np.float32)
            rfields = ["base_anchor", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context]
            rgrade = np.digitize(residual, [-100.0, -25.0, 25.0, 100.0]).astype(np.int32)
            rtr, rte = _rank_fit(tr, te, rfields, rgrade, equal_month=True, seed=SEED + int(month[-2:]) * 1000 + 99)
            te_residual = _pct(rte, rtr)
            heads.append({"month": month, "side": side, "layer": "residual", "cap": len(context), "equal_month": True, "rows": len(tr), "query_groups": int(_group(tr)[1].size), "fields": len(rfields)})
            base_rank = _pct(te.base_score.to_numpy(float), tr.base_score.to_numpy(float))
            side_out = _score_month(te, base_rank, te_consensus, te_residual, te_anchor, context)
            side_out["fold_month"] = month
            parts.append(side_out)
    pred = pd.concat(parts, ignore_index=True)
    # Percentile normalization is side-local for portability, then all rows
    # enter one pooled global ranking.
    for col in ("base_only", "consensus_rank", "residual_rank", "base_plus_consensus25", "base_plus_residual25", "full_base_consensus_residual"):
        pred[col] = pred.groupby(["month", "side_name"], sort=False)[col].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    pred["consensus_only"] = pred["consensus_rank"]
    pred["residual_only"] = pred["residual_rank"]
    g, m, s = _metrics(pred)
    pred.to_parquet(OUT / "predictions_2025.parquet", index=False, compression="zstd")
    # This is the hand-off contract for downstream conditional path/cluster
    # layers.  It deliberately persists the mapped bps anchor and causal
    # context, not only percentile ranks; a rank is not an economically
    # meaningful residual baseline.
    pred.to_parquet(OUT / "canonical_cluster_input_2025.parquet", index=False, compression="zstd")
    g.to_parquet(OUT / "metrics_global.parquet", index=False); m.to_parquet(OUT / "metrics_monthly.parquet", index=False); s.to_parquet(OUT / "metrics_stability.parquet", index=False)
    pd.DataFrame(heads).to_parquet(OUT / "head_fit_audit.parquet", index=False)
    coverage = pred.groupby(["month", "side_name"], sort=True).size().rename("rows").reset_index(); coverage.to_parquet(OUT / "coverage.parquet", index=False)
    manifest = {"schema": "tp6_sl4_downstream_retrain_2025_v1", "status": "COMPLETED", "input": str(INPUT), "geometry": "TP6/SL4/H12; 100 bps cost as encoded by exact_net_bps", "panel_scope": "strict-OOS Stage-I panel, 2024-02 through 2025-12 evaluation support", "held_months": list(MONTHS), "rows_scored": len(pred), "context_features": context, "context_sha256": digest, "base_score": "p_clear - 0.5*p_adverse; train-only isotonic map to exact TP6/SL4 net bps", "base_anchor_persisted": "base_expected_bps is the train-only isotonic mapped bps anchor for each held row", "conditional_cluster_input_contract": "canonical_cluster_input_2025.parquet contains base_expected_bps, base score/probabilities, causal context, exact labels for evaluation, and all current rank outputs", "consensus_target": "exact TP6/SL4 net - mapped base, grades thresholds [-150,-50,50,150]", "residual_target": "exact TP6/SL4 net - mapped base, grades thresholds [-100,-25,25,100]", "query": "4-hour UTC x side LambdaRank", "hpo": {"n_estimators": 140, "learning_rate": 0.035, "max_depth": 5, "num_leaves": 31, "min_child_samples": 180, "feature_fraction": 0.82, "bagging_fraction": 0.82, "lambda_l1": 0.02, "lambda_l2": 2.0, "max_bin": 127}, "ranking": "side-local monthly percentile normalization followed by one pooled global top-k", "no_held_outcomes_in_fit": True, "artifacts": ["predictions_2025.parquet", "canonical_cluster_input_2025.parquet", "metrics_global.parquet", "metrics_monthly.parquet", "metrics_stability.parquet", "head_fit_audit.parquet", "coverage.parquet"]}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = ["# TP6/SL4-trained downstream layers — 2025 causal replay", "", "The R3 base output is fixed. Consensus and residual layers are refit before each held 2025 month using only earlier resolved TP6/SL4 labels.", "", "## Global metrics", "", g.round(3).to_string(index=False), "", "## Monthly top-5", "", m.round(3).to_string(index=False), "", "## Stability", "", s.round(3).to_string(index=False), "", "## Coverage", "", coverage.to_string(index=False)]
    (OUT / "TP6_SL4_DOWNSTREAM_RETRAIN_2025_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT), "rows_scored": len(pred), "global_rows": len(g), "months": sorted(pred.month.unique().tolist())}, indent=2))


if __name__ == "__main__":
    run()
