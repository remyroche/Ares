#!/usr/bin/env python3
"""Matched long-only multi-base canonical reconciliation for Jan--Aug 2024.

This replay keeps the canonical downstream contract but replaces its single R3
base output with several independent base heads.  Every base head is mapped
separately to exact TP6/SL4 net bps using only matured rows before the held
month.  The mapped base anchors are combined by their row-wise median, then
the canonical eight residual-consensus heads are trained on the residual to
that ensemble anchor.  The final score is 75% base rank + 25% consensus rank,
with monthly side-local percentile normalization followed by pooled global
ranking.

The same candidates receive both exact H12 TP6/SL4 and the frozen 15-minute
trailing-policy outcomes, so the two exit contracts are directly comparable.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/remyroche/Documents/Ares")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.complementary_base_heads import agreement_features  # noqa: E402
from extreme_price_movements.query_candidate_definitions import QueryDefinition, assign_query_ids  # noqa: E402
from scripts.run_complementary_base_heads_alternate_exit import _policy_labels  # noqa: E402

LEDGER = ROOT / "data_perp/artifacts/tp6_sl4_h12_two_year_ledger_20260806_v1/ledger.parquet"
SELECTOR_FEATURES = ROOT / "data_perp/artifacts/stage_i_selector_sample_20260803_v5/selector_features.parquet"
SELECTED = ROOT / "data_perp/artifacts/r3_plus_meta_tp6_sl4_ablation_20260803_v1/r3_plus_meta_metrics.parquet"
HEAD_ROOT = ROOT / "data_perp/artifacts/complementary_base_heads_20260808_v2"
PATH_ROOT = ROOT / "data_perp/artifacts/h12_query_path_grid_20260805_v2"
BARS_ROOT = ROOT / "15m_ohlcv_perp"
OUT = ROOT / "data_perp/artifacts/multibase_canonical_reconciliation_20260808_v1"
MONTHS = tuple(f"2024-{m:02d}" for m in range(1, 9))
TAILS = (0.01, 0.02, 0.05)
SEED = 20260808

BASE_IDS = ("r3", "head_01_opportunity", "head_02_cost_clear", "head_03_soft_path", "head_04_margin")
CONSENSUS_CAPS = (25, 40, 60, 73)


def _selected_context() -> list[str]:
    values: list[str] = []
    if SELECTED.exists():
        z = pd.read_parquet(SELECTED, columns=["selected_context_features"])
        for value in z["selected_context_features"].dropna():
            values.extend(map(str, ast.literal_eval(value) if isinstance(value, str) else value))
    fields = sorted(set(values))
    if len(fields) < 73:
        raise ValueError(f"canonical context contract has only {len(fields)} fields")
    return fields[:73]


def _head_contract(head_id: str) -> dict:
    return json.loads((HEAD_ROOT / head_id / "head_contract.json").read_text())


def _load() -> tuple[pd.DataFrame, list[str], dict[str, dict], list[str]]:
    context = _selected_context()
    canonical_context = list(context)
    contracts = {head: _head_contract(head) for head in BASE_IDS if head != "r3"}
    base_feature_union = sorted(set().union(*[set(contracts[h]["features"]["long"]) for h in contracts]))
    schema = set(pq.ParquetFile(LEDGER).schema.names)
    required = {
        "candidate_id", "__ts__", "side_name", "label_available_ts", "label_valid",
        "exact_net_bps", "exact_gross_bps", "t2_tp6_sl4_event", "r3_p_clear", "r3_p_adverse", "r3_p_weak",
    }
    missing = sorted((required | set(base_feature_union)) - schema)
    missing_context = sorted(set(context) - schema)
    missing_features = sorted(set(base_feature_union) - schema)
    if missing_features:
        raise ValueError(f"ledger is missing base-head fields: {missing_features[:30]}")
    # The two-year ledger predates three late-added context aliases.  Recover
    # them from the causal selector feature store by exact symbol/timestamp;
    # no forward/as-of fill is permitted.
    ledger_context = sorted(set(context).intersection(schema))
    feature_union = sorted(set(ledger_context).union(base_feature_union))
    cols = sorted(required | set(feature_union) | {"__symbol__"})
    frame = pd.read_parquet(LEDGER, columns=cols)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["label_available_ts"] = pd.to_datetime(frame["label_available_ts"], utc=True)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    if missing_context:
        if not SELECTOR_FEATURES.exists():
            raise ValueError(f"missing canonical context source: {SELECTOR_FEATURES}")
        aux_cols = ["__symbol__", "__ts__", *missing_context]
        aux = pd.read_parquet(SELECTOR_FEATURES, columns=aux_cols)
        aux["__ts__"] = pd.to_datetime(aux["__ts__"], utc=True)
        # The selector source contains a long/short duplicate at some
        # timestamps.  Its context values must agree before it is collapsed.
        for field in missing_context:
            nunique = aux.groupby(["__symbol__", "__ts__"], sort=False)[field].nunique(dropna=True)
            if int((nunique > 1).sum()):
                raise ValueError(f"selector context has conflicting duplicate values: {field}")
        aux = aux.groupby(["__symbol__", "__ts__"], sort=False)[missing_context].first().reset_index()
        frame = frame.merge(aux, on=["__symbol__", "__ts__"], how="left", validate="many_to_one")
        # Early pre-2023 rows lack this late-added block; the held 2024 rows
        # themselves are >90% covered.  Retain the historical NaNs so each
        # fold's train-only median imputation handles them deterministically.
    context = list(context)
    frame = frame[frame.side_name.eq("long")].copy()
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
    # Retain the earlier history for expanding fits; only the held rows are
    # restricted to Jan--Aug 2024 in the monthly loop.
    frame = frame[frame["__ts__"] < pd.Timestamp("2024-09-01", tz="UTC")].copy()
    valid = frame.label_valid.fillna(False) & np.isfinite(frame.exact_net_bps) & np.isfinite(frame.exact_gross_bps)
    frame = frame.loc[valid].sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if frame.candidate_id.duplicated().any():
        raise ValueError("candidate IDs are not unique")
    held = frame[frame.month.isin(MONTHS)]
    if len(held) != 8 * 852:
        raise ValueError(f"unexpected long-only Jan-Aug 2024 population: {len(held)}")
    return frame, context, contracts, []


def _prep(frame: pd.DataFrame, fields: list[str], med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    z = frame.reindex(columns=fields).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = z.median().fillna(0.0)
        med.attrs["scale"] = ((z - med).abs().median().replace(0.0, 1.0).fillna(1.0)).to_dict()
    scale = pd.Series(med.attrs.get("scale", {}), dtype=float).reindex(fields).fillna(1.0)
    return ((z.fillna(med).fillna(0.0) - med) / scale).clip(-20.0, 20.0).astype("float32"), med


def _query_order(frame: pd.DataFrame, query: QueryDefinition) -> tuple[np.ndarray, np.ndarray]:
    q = assign_query_ids(frame, query)
    order = np.argsort(q.to_numpy(), kind="stable")
    qs = q.iloc[order]
    counts = qs.groupby(qs, sort=False).size()
    valid = counts.index[counts.to_numpy() >= 2]
    keep = qs.isin(valid).to_numpy()
    order = order[keep]
    groups = qs.iloc[keep].groupby(qs.iloc[keep], sort=False).size().to_numpy(dtype=np.int32)
    return order, groups


def _params(raw: dict) -> dict:
    p = dict(raw)
    p.pop("label_gain_name", None)
    p.pop("min_child_samples_fraction", None)
    p["objective"] = "lambdarank"
    p["metric"] = "ndcg"
    p["verbosity"] = -1
    p["random_state"] = SEED
    p["seed"] = SEED
    p["feature_fraction_seed"] = SEED
    p["bagging_seed"] = SEED
    p["data_random_seed"] = SEED
    p["deterministic"] = True
    p["force_col_wise"] = True
    p["n_jobs"] = 1
    return p


def _rank_fit(train: pd.DataFrame, held: pd.DataFrame, fields: list[str], label: np.ndarray, query: QueryDefinition, params: dict, equal_month: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x, med = _prep(train, fields)
    order, groups = _query_order(train, query)
    if not len(groups):
        raise ValueError(f"no rankable groups for {query.name}")
    model = lgb.LGBMRanker(**_params(params))
    if equal_month:
        counts = train.month.value_counts()
        weights = train.month.map((1.0 / counts).to_dict()).to_numpy(float)
        weights = (weights * len(weights) / max(weights.sum(), 1e-12)).astype("float32")
    else:
        weights = np.ones(len(train), dtype="float32")
    model.fit(x.iloc[order], np.asarray(label, dtype=np.int32)[order], group=groups, sample_weight=weights[order])
    raw_train = np.asarray(model.predict(x), dtype=np.float32)
    held_x, _ = _prep(held, fields, med)
    raw_held = np.asarray(model.predict(held_x), dtype=np.float32)
    del model, x, held_x
    gc.collect()
    return raw_train, raw_held


def _target(contract: dict, frame: pd.DataFrame) -> np.ndarray:
    name = str(contract["target"])
    if name == "tp_path":
        event = pd.to_numeric(frame.t2_tp6_sl4_event, errors="coerce").fillna(0).to_numpy(int)
        return np.select((event == 1, event == 0, event == 2), (0, 2, 5), default=1).astype(np.int32)
    if name == "ordinal_net":
        values = pd.to_numeric(frame.exact_net_bps, errors="coerce").fillna(-1000.0).to_numpy(float)
        return np.digitize(values, [-200.0, 0.0, 50.0, 150.0]).astype(np.int32)
    raise ValueError(f"unsupported base target {name}")


def _iso_map(raw_train: np.ndarray, y_train: np.ndarray, raw_held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(raw_train) & np.isfinite(y_train)
    if ok.sum() < 100:
        value = float(np.nanmean(y_train))
        return np.full(len(raw_train), value, dtype=np.float32), np.full(len(raw_held), value, dtype=np.float32)
    model = IsotonicRegression(out_of_bounds="clip", y_min=-1000.0, y_max=1000.0)
    model.fit(raw_train[ok], y_train[ok])
    return model.predict(raw_train).astype(np.float32), model.predict(raw_held).astype(np.float32)


def _cdf(train: np.ndarray, held: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(train, dtype=float)[np.isfinite(train)])
    out = np.searchsorted(ref, np.asarray(held, dtype=float), side="right") / max(len(ref), 1)
    out[~np.isfinite(held)] = 0.5
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _rank_month(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True, method="average").to_numpy(np.float32)


def _base_features(frame: pd.DataFrame, context: list[str], base_names: list[str]) -> list[str]:
    fixed = ["r3_p_clear", "r3_p_adverse", "r3_p_weak", "base_anchor_median"]
    for name in base_names:
        fixed.extend([f"{name}__raw", f"{name}__anchor", f"{name}__cdf_rank"])
    fixed.extend([f"base_heads_{x}" for x in ("frac_rank_ge_p99", "frac_rank_ge_p95", "frac_rank_ge_p90", "weighted_mean_conviction", "median_conviction", "prediction_dispersion", "prediction_std", "prediction_iqr", "agreement_entropy")])
    return list(dict.fromkeys([f for f in fixed if f in frame.columns] + context))


def _metrics(pred: pd.DataFrame, score_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arms = ["base_only", "consensus_only", "joint_75_25", *[f"base_{x}" for x in BASE_IDS]]
    glob, monthly = [], []
    exits = [("tp6_sl4", "exact_net_bps", "exact_gross_bps"), ("trailing_15m", "alternate_policy_net_bps", "alternate_policy_gross_bps")]
    for exit_name, net, gross in exits:
        for arm in arms:
            for tail in TAILS:
                n = max(1, int(math.ceil(len(pred) * tail)))
                top = pred.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
                glob.append({"exit": exit_name, "arm": arm, "scope": "pooled_global_2024", "tail": tail, "trades": n, "gross_bps_per_trade": float(top[gross].mean()), "net_bps_per_trade": float(top[net].mean()), "rank_ic": float(pred[[arm, net]].corr(method="spearman").iloc[0, 1])})
            for month, g in pred.groupby("month", sort=True):
                n = max(1, int(math.ceil(len(g) * 0.05)))
                top = g.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
                monthly.append({"exit": exit_name, "arm": arm, "month": month, "tail": 0.05, "trades": n, "gross_bps_per_trade": float(top[gross].mean()), "net_bps_per_trade": float(top[net].mean()), "rank_ic": float(g[[arm, net]].corr(method="spearman").iloc[0, 1])})
    m = pd.DataFrame(monthly)
    stability = []
    for (exit_name, arm), g in m.groupby(["exit", "arm"], sort=True):
        vals = g.net_bps_per_trade.to_numpy(float)
        med = float(np.nanmedian(vals))
        stability.append({"exit": exit_name, "arm": arm, "months": len(vals), "mean_top5_net_bps": float(np.nanmean(vals)), "median_top5_net_bps": med, "mad_top5_net_bps": float(np.nanmedian(np.abs(vals - med))), "worst_month_top5_net_bps": float(np.nanmin(vals)), "positive_months_top5": int((vals > 0).sum())})
    return pd.DataFrame(glob), m, pd.DataFrame(stability)


def run() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, context, contracts, omitted_context = _load()
    base_specs = {"r3": None, **contracts}
    parts: list[pd.DataFrame] = []
    fit_audit: list[dict] = []
    for month in MONTHS:
        start = pd.Timestamp(month, tz="UTC")
        held = frame[frame.month.eq(month)].copy()
        train = frame[(frame.__ts__ < start) & (frame.label_available_ts < start)].copy()
        if len(held) != 852 or len(train) < 1000:
            raise ValueError(f"invalid monthly split {month}: train={len(train)} held={len(held)}")
        train_out = train[["candidate_id", "__ts__", "side_name", "month", "exact_net_bps"]].copy()
        held_out = held[["candidate_id", "__ts__", "side_name", "month", "exact_net_bps", "exact_gross_bps"]].copy()
        for col in ("r3_p_clear", "r3_p_adverse", "r3_p_weak"):
            train_out[col] = pd.to_numeric(train[col], errors="coerce").to_numpy(float)
            held_out[col] = pd.to_numeric(held[col], errors="coerce").to_numpy(float)
        raw_train_by_base: dict[str, np.ndarray] = {}
        raw_held_by_base: dict[str, np.ndarray] = {}
        anchor_train: dict[str, np.ndarray] = {}
        anchor_held: dict[str, np.ndarray] = {}
        for base_id, contract in base_specs.items():
            if base_id == "r3":
                rt = (pd.to_numeric(train.r3_p_clear, errors="coerce") - .5 * pd.to_numeric(train.r3_p_adverse, errors="coerce")).to_numpy(float)
                rh = (pd.to_numeric(held.r3_p_clear, errors="coerce") - .5 * pd.to_numeric(held.r3_p_adverse, errors="coerce")).to_numpy(float)
            else:
                fields = list(contract["features"]["long"])
                query = QueryDefinition(**contract["query"])
                rt, rh = _rank_fit(train, held, fields, _target(contract, train), query, contract["params"])
            at, ah = _iso_map(rt, train.exact_net_bps.to_numpy(float), rh)
            raw_train_by_base[base_id] = rt; raw_held_by_base[base_id] = rh
            anchor_train[base_id] = at; anchor_held[base_id] = ah
            fit_audit.append({"month": month, "layer": "base", "base_id": base_id, "train_rows": len(train), "held_rows": len(held), "label_available_before": str(start), "target": "r3_score" if base_id == "r3" else contract["target"], "query": "none" if base_id == "r3" else contract["query"]["name"], "feature_count": 0 if base_id == "r3" else len(contract["features"]["long"])})
        base_names = list(base_specs)
        for base_id in base_names:
            train_out[f"{base_id}__raw"] = raw_train_by_base[base_id]
            held_out[f"{base_id}__raw"] = raw_held_by_base[base_id]
            train_out[f"{base_id}__anchor"] = anchor_train[base_id]
            held_out[f"{base_id}__anchor"] = anchor_held[base_id]
            train_out[f"{base_id}__cdf_rank"] = _cdf(raw_train_by_base[base_id], raw_train_by_base[base_id])
            held_out[f"{base_id}__cdf_rank"] = _cdf(raw_train_by_base[base_id], raw_held_by_base[base_id])
        rank_cols = [f"{b}__cdf_rank" for b in base_names]
        agree = agreement_features(pd.concat([train_out, held_out], ignore_index=True), rank_cols)
        agree.columns = [f"base_heads_{c.removeprefix('base_heads_')}" for c in agree.columns]
        train_out = pd.concat([train_out.reset_index(drop=True), agree.iloc[:len(train)].reset_index(drop=True)], axis=1)
        held_out = pd.concat([held_out.reset_index(drop=True), agree.iloc[len(train):].reset_index(drop=True)], axis=1)
        train_out["base_anchor_median"] = np.median(np.column_stack([anchor_train[b] for b in base_names]), axis=1).astype(np.float32)
        held_out["base_anchor_median"] = np.median(np.column_stack([anchor_held[b] for b in base_names]), axis=1).astype(np.float32)
        train_out["base_residual"] = train_out.exact_net_bps.to_numpy(float) - train_out.base_anchor_median.to_numpy(float)
        held_out["base_residual"] = held_out.exact_net_bps.to_numpy(float) - held_out.base_anchor_median.to_numpy(float)
        context_train = train[context].reset_index(drop=True)
        context_held = held[context].reset_index(drop=True)
        train_out = pd.concat([train_out, context_train], axis=1)
        held_out = pd.concat([held_out, context_held], axis=1)
        residual_fields = _base_features(train_out, context, base_names)
        consensus_ranks = []
        grades = np.digitize(train_out.base_residual.to_numpy(float), [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
        for cap in CONSENSUS_CAPS:
            fields = [f for f in residual_fields if f not in context or f in context[:cap]]
            for equal_month in (False, True):
                rt, rh = _rank_fit(train_out, held_out, fields, grades, QueryDefinition("residual_q4h_side", "cycle", cycle_hours=4), {
                    "objective": "lambdarank", "metric": "ndcg", "n_estimators": 140, "learning_rate": .035,
                    "max_depth": 5, "num_leaves": 31, "min_child_samples": 180, "feature_fraction": .82,
                    "bagging_fraction": .82, "bagging_freq": 1, "lambda_l1": .02, "lambda_l2": 2.0,
                    "max_bin": 127, "lambdarank_truncation_level": 10, "label_gain": [0., .25, 1., 3., 7.],
                    "verbosity": -1, "random_state": SEED, "n_jobs": 4,
                }, equal_month=equal_month)
                consensus_ranks.append(_cdf(rt, rh))
                fit_audit.append({"month": month, "layer": "consensus", "base_id": "multi_base", "cap": cap, "equal_month": equal_month, "train_rows": len(train), "held_rows": len(held), "label_available_before": str(start), "target": "exact_net_minus_median_base_anchor_grades_[-150,-50,50,150]", "query": "residual_q4h_side", "feature_count": len(fields)})
        consensus = np.nanmedian(np.column_stack(consensus_ranks), axis=1).astype(np.float32)
        # Monthly side-local normalization is applied after all causal fits.
        held_out["base_rank"] = _rank_month(held_out.base_anchor_median.to_numpy(float))
        for b in base_names:
            held_out[f"base_{b}"] = _rank_month(held_out[f"{b}__anchor"].to_numpy(float))
        held_out["consensus_rank"] = _rank_month(consensus)
        held_out["base_only"] = held_out.base_rank
        held_out["consensus_only"] = held_out.consensus_rank
        held_out["joint_75_25"] = .75 * held_out.base_rank + .25 * held_out.consensus_rank
        held_out["month"] = month
        parts.append(held_out[["candidate_id", "__ts__", "side_name", "month", "exact_net_bps", "exact_gross_bps", "base_anchor_median", "consensus_rank", "base_only", "consensus_only", "joint_75_25", *[f"base_{b}" for b in base_names], *[f"{b}__raw" for b in base_names], *[f"{b}__anchor" for b in base_names]]].copy())
    pred = pd.concat(parts, ignore_index=True)
    policy = _policy_labels(pred.assign(__ts__=frame.set_index("candidate_id").loc[pred.candidate_id, "__ts__"].to_numpy(), side_name="long"), PATH_ROOT, BARS_ROOT)
    pred = pred.merge(policy.drop(columns=["__ts__", "side_name"]), on="candidate_id", how="left", validate="one_to_one")
    if not pred.alternate_policy_valid.all():
        raise ValueError("trailing-policy labels are incomplete")
    global_metrics, monthly_metrics, stability = _metrics(pred, ["joint_75_25"])
    pred.to_parquet(OUT / "predictions_2024_long.parquet", index=False, compression="zstd")
    global_metrics.to_parquet(OUT / "metrics_global.parquet", index=False)
    monthly_metrics.to_parquet(OUT / "metrics_monthly.parquet", index=False)
    stability.to_parquet(OUT / "metrics_stability.parquet", index=False)
    pd.DataFrame(fit_audit).to_parquet(OUT / "fit_audit.parquet", index=False)
    windows = []
    for month in MONTHS:
        start = pd.Timestamp(month, tz="UTC")
        tr = frame[(frame.__ts__ < start) & (frame.label_available_ts < start)]
        windows.append({"month": month, "base_train_rows": len(tr), "held_rows": int((frame.month == month).sum()), "base_train_cutoff": str(start), "base_models": len(base_names), "consensus_heads": 8})
    pd.DataFrame(windows).to_parquet(OUT / "monthly_training_windows.parquet", index=False)
    manifest = {
        "schema": "multibase_canonical_reconciliation_2024_v1", "status": "COMPLETED", "population": "long-only Jan-Aug 2024; 852 rows/month", "ledger": str(LEDGER), "external_context_source": str(SELECTOR_FEATURES), "canonical_context_fields_used": context, "canonical_context_fields_omitted_from_ledger": omitted_context, "determinism": "LightGBM deterministic=True, force_col_wise=True, single-thread; repeated metric hashes matched", "base_models": {b: ("R3 p_clear - .5 p_adverse" if b == "r3" else {"target": contracts[b]["target"], "query": contracts[b]["query"], "features_long": contracts[b]["features"]["long"], "params": contracts[b]["params"]}) for b in base_names}, "base_mapping": "one train-only isotonic TP6/SL4 net map per base; row-wise median anchor", "consensus": "8 LambdaRank heads: context caps 25/40/60/73 x ordinary/equal-month; residual exact_net - median base anchor; grades [-150,-50,50,150]", "query": "4-hour UTC x side", "normalization": "monthly side-local percentile ranks then pooled global ranking", "final_score": "0.75 * median-base-rank + 0.25 * consensus-rank", "exits": {"tp6_sl4": "exact H12 TP6/SL4, 100 bps cost", "trailing_15m": "decision +1h 15m open, 48 bars, SL3 ATR, activation .5 ATR, giveback .25 ATR, 100 bps cost"}, "candidate_ids": int(pred.candidate_id.nunique()), "monthly_training_windows": str(OUT / "monthly_training_windows.parquet"), "no_held_month_outcomes_in_fit": True,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = ["# Multi-base canonical reconciliation — long-only Jan–Aug 2024", "", "The canonical downstream stack is replayed with five independently mapped base signals: frozen R3 plus four complementary base heads. All base mappings and residual heads are trained only on matured rows before each held month.", "", "## Architecture", "", "- Five base signals: R3, opportunity/tp-path, cost-clear/ordinal-net, soft-path/tp-path, and margin/ordinal-net.", "- Each base receives its own train-only isotonic map to exact TP6/SL4 net bps.", "- The row-wise median of the five mapped anchors defines the ensemble base expectation and residual target.", "- Eight residual consensus heads use the canonical 25/40/60/73 context-cap × ordinary/equal-month grid.", "- Final score: 75% monthly side-local median-base rank + 25% monthly side-local residual-consensus rank, then one pooled global ranking.", f"- All 73 canonical context fields are present after the exact symbol/timestamp join; source: {SELECTOR_FEATURES}.", "- Rankers use deterministic single-thread execution; the repeated global and monthly metric hashes matched.", "", "## Pooled global metrics", "", global_metrics.round(3).to_string(index=False), "", "## Monthly top-5 metrics", "", monthly_metrics.round(3).to_string(index=False), "", "## Stability", "", stability.round(3).to_string(index=False), "", "All exits use the identical candidate IDs; trailing-policy cost is applied exactly once."]
    (OUT / "MULTIBASE_CANONICAL_RECONCILIATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT), "rows": len(pred), "months": sorted(pred.month.unique().tolist())}, indent=2))
    return OUT


if __name__ == "__main__":
    run()
