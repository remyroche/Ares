#!/usr/bin/env python3
"""Matched exact TP6/SL4/H12 replay on the approved 170-symbol universe.

The replay keeps the canonical residual objective/query/blend, but isolates
the two requested additions: causal support/OOD fields and base-head
uncertainty.  All five rows in the comparison (base, canonical residual
control, support/OOD, uncertainty, both) are produced on the same exact-label
rows and the same monthly folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import config

SIDES = ("long", "short")
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
BASE_PARAMS = dict(
    objective="multiclass", num_class=3, n_estimators=140, learning_rate=0.05,
    num_leaves=31, min_child_samples=350, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=8.0, n_jobs=2, verbosity=-1,
)
META_PARAMS = dict(
    objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10,
    n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=12,
    min_child_samples=350, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l1=1.0, lambda_l2=10.0, max_bin=63,
    label_gain=[0.0, 0.25, 1.0, 3.0, 7.0], n_jobs=2, verbosity=-1,
)


def _matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)


def _rank_pct(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=np.float32)
    return ((rankdata(values, method="average") - 1.0) / (len(values) - 1.0)).astype(np.float32)


def _query_order(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = pd.to_datetime(frame["__ts__"], utc=True).dt.floor("4h").astype("int64").astype(str)
    q = q + "__" + frame["side_name"].astype(str)
    order = np.argsort(q.to_numpy(), kind="stable")
    qs = q.iloc[order]
    counts = qs.groupby(qs, sort=False).size()
    valid = counts.index[counts.to_numpy() >= 2]
    keep = qs.isin(valid).to_numpy()
    order = order[keep]
    groups = qs.iloc[keep].groupby(qs.iloc[keep], sort=False).size().to_numpy(dtype=np.int32)
    return order, groups


def _fit_ranker(train: pd.DataFrame, held: pd.DataFrame, cols: list[str], target: np.ndarray, seed: int) -> np.ndarray:
    order, groups = _query_order(train)
    if len(groups) == 0 or len(order) < 20:
        return np.zeros(len(held), dtype=np.float32)
    params = dict(META_PARAMS)
    # Match the pre-existing canonical residual helper: the child floor is
    # 3% of the prior-OOF training population, bounded below by 120.  This is
    # the only fold-size-dependent parameter; all other ranker parameters are
    # frozen from the canonical control.
    params["min_child_samples"] = max(120, int(math.ceil(0.03 * len(train))))
    model = lgb.LGBMRanker(random_state=seed, **params)
    model.fit(_matrix(train.iloc[order], cols), target[order], group=groups)
    return np.asarray(model.predict(_matrix(held, cols)), dtype=np.float32)


def _fit_expected_net(history: pd.DataFrame) -> tuple[IsotonicRegression, float]:
    x = pd.to_numeric(history["base_score"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(history["net_bps"], errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 50 or np.unique(x[keep]).size < 3:
        fallback = float(np.nanmean(y[keep])) if keep.any() else 0.0
        return IsotonicRegression(out_of_bounds="clip"), fallback
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x[keep], y[keep])
    return model, float(np.nanmean(y[keep]))


def _feature_columns() -> tuple[list[str], list[str], dict[str, object]]:
    base = list(dict.fromkeys(config.MODEL_DIRECT_BASE_FEATURE_KEYS))
    meta = list(dict.fromkeys(config.MODEL_REGIME_CONTEXT_META_FEATURE_KEYS))
    return base, meta, {"configured_base": len(base), "configured_meta": len(meta)}


def _load_feature_catalog(feature_root: Path) -> list[str]:
    files = sorted(feature_root.glob("symbol=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no hourly feature files under {feature_root}")
    probe = pd.read_parquet(files[0])
    return list(probe.columns)


def _join_features(panel: pd.DataFrame, feature_root: Path, requested: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    available = set(_load_feature_catalog(feature_root))
    cols = [c for c in requested if c in available]
    out = panel.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out["feature_missing_fraction"] = 1.0
    out["feature_row_found"] = 0.0
    symbols = out["__symbol__"].astype(str).unique()
    for symbol in symbols:
        path = feature_root / f"symbol={symbol.replace('/', '_')}.parquet"
        if not path.exists():
            continue
        take = out["__symbol__"].astype(str).eq(symbol)
        times = pd.DatetimeIndex(pd.to_datetime(out.loc[take, "__ts__"], utc=True))
        # A few legacy symbols have a slightly older schema.  Read only the
        # requested fields that exist in that file and leave the rest missing;
        # never substitute another feature or a future-derived value.
        schema_cols = set(pq.ParquetFile(path).schema.names)
        symbol_cols = [c for c in cols if c in schema_cols]
        feat = pd.read_parquet(path, columns=symbol_cols)
        idx = pd.to_datetime(feat.index, utc=True)
        feat.index = idx
        joined = feat.reindex(times)
        joined.index = out.index[take]
        if symbol_cols:
            out.loc[take, symbol_cols] = joined.to_numpy()
        arr = out.loc[take, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        out.loc[take, "feature_missing_fraction"] = np.mean(~np.isfinite(arr), axis=1)
        out.loc[take, "feature_row_found"] = (~np.all(~np.isfinite(arr), axis=1)).astype(float)
    return out, {
        "requested_feature_count": len(requested), "store_feature_count": len(cols),
        "missing_requested": sorted(set(requested) - set(cols)),
        "symbols_with_store_files": int(sum((feature_root / f"symbol={s.replace('/', '_')}.parquet").exists() for s in symbols)),
    }


def _load_month(source: Path, labels_root: Path, month: str, universe: set[str]) -> pd.DataFrame:
    parts = []
    for side in SIDES:
        cand_path = source / f"train_global_{side}_5_{month.replace('-', '_')}.parquet"
        lab_path = labels_root / "parts" / f"month={month}" / f"side={side}.parquet"
        if not cand_path.exists() or not lab_path.exists():
            continue
        identity = pd.read_parquet(cand_path, columns=["candidate_id", "__ts__", "__symbol__", "side_name"])
        identity = identity.loc[identity["__symbol__"].astype(str).isin(universe)].copy()
        identity["__ts__"] = pd.to_datetime(identity["__ts__"], utc=True, errors="raise")
        labels = pd.read_parquet(lab_path)
        labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="raise")
        frame = identity.merge(labels, on=["candidate_id", "__ts__", "__symbol__", "side_name"], how="inner", validate="one_to_one")
        frame["month"] = month
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _add_base_outputs(frame: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    out = frame[["candidate_id", "__ts__", "__symbol__", "side_name", "month", "atr_bps", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "label_valid", "robust_clear_event_b25"]].copy()
    out = out.rename(columns={"t4_tp6_sl4_gross_bps": "gross_bps", "t4_tp6_sl4_net_bps": "net_bps"})
    out["p_adverse"] = proba[:, 0]
    out["p_weak"] = proba[:, 1]
    out["p_clear"] = proba[:, 2]
    pp = np.clip(proba, 1e-8, 1.0)
    out["base_entropy"] = -np.sum(pp * np.log(pp), axis=1)
    so = np.sort(proba, axis=1)
    out["base_top2_margin"] = so[:, -1] - so[:, -2]
    out["base_conviction"] = proba[:, 2] - 0.5 * proba[:, 1]
    out["base_score"] = proba[:, 2] - 0.5 * proba[:, 0]
    return out


def _base_target(frame: pd.DataFrame) -> np.ndarray:
    clear = pd.to_numeric(frame["robust_clear_event_b25"], errors="coerce").fillna(0).to_numpy(float) > 0.5
    net = pd.to_numeric(frame["t4_tp6_sl4_net_bps"], errors="coerce").to_numpy(float)
    adverse = np.isfinite(net) & (net < -50.0)
    return np.select([adverse, clear], [0, 2], default=1).astype(np.int8)


def _metric(block: pd.DataFrame, score: str, tail: float, scope: str, **extra: object) -> dict[str, object]:
    valid = block.loc[block["label_valid"].astype(bool)].copy()
    n = max(1, int(math.ceil(len(valid) * tail)))
    top = valid.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort").head(n)
    return {"scope": scope, "tail": tail, "trades": len(top), "gross_bps_per_trade": float(top.gross_bps.mean()), "net_bps_per_trade": float(top.net_bps.mean()), "rank_ic": float(valid[[score, "net_bps"]].corr(method="spearman").iloc[0, 1]) if len(valid) > 2 else np.nan, **extra}


def _run(*, output: Path, source: Path, labels_root: Path, universe_file: Path, feature_root: Path, start: str, end: str) -> Path:
    if output.exists():
        raise FileExistsError(output)
    universe_frame = pd.read_csv(universe_file)
    universe = set(universe_frame["symbol"].astype(str))
    months = [str(x) for x in pd.period_range(start=start, end=pd.Timestamp(end) - pd.Timedelta(days=1), freq="M")]
    all_panels = []
    for month in months:
        frame = _load_month(source, labels_root, month, universe)
        if not frame.empty:
            all_panels.append(frame)
    panel = pd.concat(all_panels, ignore_index=True) if all_panels else pd.DataFrame()
    if panel.empty:
        raise RuntimeError("no exact labelled candidate rows were found")
    base_cfg, meta_cfg, cfg_audit = _feature_columns()
    panel, feature_audit = _join_features(panel, feature_root, list(dict.fromkeys(base_cfg + meta_cfg)))
    base_cols = [c for c in base_cfg if c in panel.columns and pd.to_numeric(panel[c], errors="coerce").notna().any()]
    meta_cols = [c for c in meta_cfg if c in panel.columns and pd.to_numeric(panel[c], errors="coerce").notna().any()]
    if len(base_cols) < 10:
        raise RuntimeError(f"too few causal base features after join: {len(base_cols)}")
    panel["valid_for_training"] = panel["label_valid"].astype(bool) & panel["feature_row_found"].gt(0)
    panel["candidate_row"] = np.arange(len(panel), dtype=np.int64)
    history = {side: [] for side in SIDES}
    outputs = []
    fold_audit = []
    for month in months:
        start_ts = pd.Timestamp(month + "-01", tz="UTC")
        held_all = panel.loc[panel["month"].eq(month)].copy()
        if held_all.empty:
            continue
        month_out = []
        for side_i, side in enumerate(SIDES):
            held = held_all.loc[held_all["side_name"].eq(side)].copy()
            train = panel.loc[(panel["side_name"].eq(side)) & (pd.to_datetime(panel["__label_available_at__"], utc=True, errors="coerce") <= start_ts) & panel["valid_for_training"]].copy()
            if len(train) < 500 or held.empty:
                fold_audit.append({"month": month, "side": side, "status": "SKIP", "train_rows": len(train), "held_rows": len(held)})
                continue
            y = _base_target(train)
            base = lgb.LGBMClassifier(random_state=20260808 + side_i, **BASE_PARAMS)
            base.fit(_matrix(train, base_cols), y)
            tr = _add_base_outputs(train, base.predict_proba(_matrix(train, base_cols)))
            ho = _add_base_outputs(held, base.predict_proba(_matrix(held, base_cols)))
            # Carry raw causal context into the outputs; it is not an outcome label.
            for c in meta_cols:
                tr[c] = pd.to_numeric(train[c], errors="coerce").to_numpy()
                ho[c] = pd.to_numeric(held[c], errors="coerce").to_numpy()
            tr["feature_missing_fraction"] = train["feature_missing_fraction"].to_numpy()
            ho["feature_missing_fraction"] = held["feature_missing_fraction"].to_numpy()
            tr["context_ood_mean_abs_z"] = 0.0; ho["context_ood_mean_abs_z"] = 0.0
            tr["context_ood_p95_abs_z"] = 0.0; ho["context_ood_p95_abs_z"] = 0.0
            tr["context_ood_outlier_fraction"] = 0.0; ho["context_ood_outlier_fraction"] = 0.0
            tr["context_ood_tail_fraction"] = 0.0; ho["context_ood_tail_fraction"] = 0.0
            # Robust causal support/OOD summary, fit on the prior rows only.
            med = np.nanmedian(train[meta_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float), axis=0) if meta_cols else np.zeros(1)
            mad = np.nanmedian(np.abs(train[meta_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float) - med), axis=0) if meta_cols else np.ones(1)
            mad[~np.isfinite(mad) | (mad < 1e-6)] = 1.0
            for target_frame, source_frame in ((tr, train), (ho, held)):
                z = np.clip((source_frame[meta_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float) - med) / mad, -20.0, 20.0) if meta_cols else np.zeros((len(source_frame), 1))
                az = np.abs(z)
                target_frame["context_ood_mean_abs_z"] = np.nanmean(az, axis=1)
                target_frame["context_ood_p95_abs_z"] = np.nanpercentile(az, 95, axis=1)
                target_frame["context_ood_outlier_fraction"] = np.nanmean(az > 3.0, axis=1)
                target_frame["context_ood_tail_fraction"] = np.nanmean(az > 2.0, axis=1)
                target_frame["support_unseen_bucket_share"] = np.mean(~np.isfinite(z), axis=1)
            prior = pd.concat(history[side], ignore_index=True) if history[side] else pd.DataFrame()
            if prior.empty:
                for arm in ("base", "canonical_residual_control", "support_ood", "uncertainty", "support_ood+uncertainty"):
                    o = ho.copy(); o["arm"] = arm; o["residual_raw"] = 0.0; o["score"] = o["base_score"]
                    month_out.append(o)
                status = "BASE_WARMUP"
            else:
                iso, fallback = _fit_expected_net(prior)
                prior["expected_net_bps"] = np.asarray(iso.predict(prior["base_score"].to_numpy(float)) if hasattr(iso, "X_min_") else np.full(len(prior), fallback), dtype=float)
                ho["expected_net_bps"] = np.asarray(iso.predict(ho["base_score"].to_numpy(float)) if hasattr(iso, "X_min_") else np.full(len(ho), fallback), dtype=float)
                prior["residual_bps"] = prior["net_bps"] - prior["expected_net_bps"]
                grade = np.digitize(prior["residual_bps"].to_numpy(float), [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
                control = ["base_score", "expected_net_bps"]
                support = control + ["context_ood_mean_abs_z", "context_ood_p95_abs_z", "context_ood_outlier_fraction", "context_ood_tail_fraction", "support_unseen_bucket_share", "feature_missing_fraction"]
                uncertainty = control + ["p_clear", "p_adverse", "p_weak", "base_entropy", "base_top2_margin", "base_conviction"]
                arms = {"canonical_residual_control": control, "support_ood": support, "uncertainty": uncertainty, "support_ood+uncertainty": list(dict.fromkeys(support + uncertainty))}
                for arm, cols in arms.items():
                    for c in cols:
                        if c not in prior: prior[c] = 0.0
                        if c not in ho: ho[c] = 0.0
                    raw = _fit_ranker(prior, ho, cols, grade, 20260808 + 1000 * side_i + int(month[-2:]) + len(cols))
                    o = ho.copy(); o["arm"] = arm; o["residual_raw"] = raw; o["score"] = 0.75 * _rank_pct(o["base_score"].to_numpy(float)) + 0.25 * _rank_pct(raw)
                    month_out.append(o)
                o = ho.copy(); o["arm"] = "base"; o["residual_raw"] = 0.0; o["score"] = _rank_pct(o["base_score"].to_numpy(float)); month_out.append(o)
                status = "RESIDUAL_FIT_PRIOR_OOF"
            history[side].append(ho.loc[ho["label_valid"].astype(bool)].copy())
            fold_audit.append({"month": month, "side": side, "status": status, "train_rows": len(train), "held_rows": len(held), "valid_held_rows": int(held["label_valid"].sum()), "base_feature_count": len(base_cols), "meta_feature_count": len(meta_cols), "train_clear_rate": float(np.mean(y == 2))})
        if month_out:
            outputs.extend(month_out)
    pred = pd.concat(outputs, ignore_index=True)
    metrics = []
    monthly = []
    per_side = []
    for arm, block in pred.groupby("arm", sort=True):
        for tail in TAILS:
            metrics.append(_metric(block, "score", tail, "global_exact170", arm=arm))
        for month, b in block.groupby("month", sort=True):
            monthly.append(_metric(b, "score", .05, "monthly_exact170", arm=arm, month=month))
        for side, b in block.groupby("side_name", sort=True):
            for tail in (.01, .05, .10):
                per_side.append(_metric(b, "score", tail, "side_exact170", arm=arm, side=side))
    # Required cost/regime decomposition: valid rows only, coarse but stable bins.
    valid = pred.loc[pred["label_valid"].astype(bool)].copy()
    valid["base_decile"] = valid.groupby(["arm", "month"], observed=False)["base_score"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop"))
    valid["residual_decile"] = valid.groupby(["arm", "month"], observed=False)["residual_raw"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop"))
    valid["cost_to_atr"] = 100.0 / pd.to_numeric(valid["atr_bps"], errors="coerce").replace(0, np.nan)
    valid["cost_atr_bin"] = pd.cut(valid["cost_to_atr"], [-np.inf, .25, .5, 1.0, 2.0, np.inf], labels=["<=.25", ".25-.5", ".5-1", "1-2", ">2"])
    decomposition = valid.groupby(["arm", "side_name", "month", "base_decile", "residual_decile", "cost_atr_bin"], observed=False).agg(rows=("net_bps", "size"), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), atr_bps=("atr_bps", "median")).reset_index()
    output.mkdir(parents=True)
    pred.to_parquet(output / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(output / "metrics.parquet", index=False)
    pd.DataFrame(monthly).to_parquet(output / "monthly_metrics.parquet", index=False)
    pd.DataFrame(per_side).to_parquet(output / "per_side_metrics.parquet", index=False)
    pd.DataFrame(fold_audit).to_parquet(output / "fold_audit.parquet", index=False)
    decomposition.to_parquet(output / "net_ev_decomposition.parquet", index=False)
    manifest = {"schema": "tp6_sl4_exact170_matched_replay_v1", "status": "COMPLETE", "universe_count": len(universe), "months": months, "rows": len(pred), "valid_rows": int(valid.shape[0]), "target": "exact TP6/SL4/H12; R3 robust clear B25/T50; invalid paths excluded from fitting and tails", "base_contract": "configured MODEL_DIRECT_BASE_FEATURE_KEYS intersect raw hourly store", "meta_contract": "configured MODEL_REGIME_CONTEXT_META_FEATURE_KEYS intersect raw hourly store", "residual_control": "same canonical LambdaRank family: prior-OOF net residual around train-only isotonic base-score map, ordinal grades [-150,-50,50,150], 4h UTC x side queries, 75/25 base/residual rank blend", "arms": ["base", "canonical_residual_control", "support_ood", "uncertainty", "support_ood+uncertainty"], "feature_audit": feature_audit, "configured_feature_audit": cfg_audit, "base_features": base_cols, "meta_features": meta_cols, "model_params": {"base": BASE_PARAMS, "residual": META_PARAMS}, "artifacts": ["predictions.parquet", "metrics.parquet", "monthly_metrics.parquet", "per_side_metrics.parquet", "fold_audit.parquet", "net_ev_decomposition.parquet"]}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    report = ["# Exact 170-symbol TP6/SL4 matched replay", "", "## Global metrics", "", pd.DataFrame(metrics).round(3).to_string(index=False), "", "## Monthly top-5", "", pd.DataFrame(monthly).round(3).to_string(index=False), "", "## Per-side", "", pd.DataFrame(per_side).round(3).to_string(index=False), "", "## Fold audit", "", pd.DataFrame(fold_audit).round(3).to_string(index=False), "", "## Contract", "", json.dumps(manifest, indent=2, default=str)]
    (output / "TP6_SL4_EXACT170_MATCHED_REPLAY_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(output), "rows": len(pred), "valid_rows": int(valid.shape[0]), "metrics": metrics}, indent=2, default=str))
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "data_perp/artifacts/tp6_sl4_exact170_matched_replay_20260808_v1")
    p.add_argument("--source", type=Path, default=ROOT / "data_perp/artifacts/20260720_s59_h5_fullthroughjul10_candleclose_trailing_cost100bps_labels/labels")
    p.add_argument("--labels-root", type=Path, default=ROOT / "data_perp/artifacts/tp6_sl4_exact170_labels_20260808_v1")
    p.add_argument("--universe-file", type=Path, default=ROOT / "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_sliding365_meta_hpo150_wf30_20260721_v1/best_full_oos/p90spread_fee15bps_eligible170/eligible_symbols.csv")
    p.add_argument("--feature-root", type=Path, default=ROOT / "data_perp/features/20260711_070000")
    p.add_argument("--start", default="2026-01")
    p.add_argument("--end", default="2026-08-01")
    args = p.parse_args()
    _run(output=args.output, source=args.source, labels_root=args.labels_root, universe_file=args.universe_file, feature_root=args.feature_root, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
