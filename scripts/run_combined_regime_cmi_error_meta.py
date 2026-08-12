#!/usr/bin/env python3
"""Combined regime-specialist + CMI + causal 7-day error-history residual arm."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import (
    MARKET_CROSS_SECTIONAL_META_FEATURE_KEYS,
    PERP_META_PRIMARY_FEATURE_KEYS,
    RESIDUAL_META_FEATURE_KEYS,
    T2_FUNNEL_META_CONTEXT_FEATURE_KEYS,
)
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_multiview_specialist_input_ablation import _base, _store_rows, _utc
from scripts.run_frozen_residual_query_hpo import _fit_residual
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS
from scripts.run_meta_incremental_cmi import _cmi
from scripts.run_regime_grouped_larger_specialists import (
    PARAMS as SPECIALIST_PARAMS,
    REGIMES,
    _bins,
    _large_fields,
    _rank_fit,
    _regime_columns,
    _schema,
)
from scripts.run_specialist_error_history_ablation import _rolling_features

OUT = ROOT / "data_perp/artifacts/combined_regime_cmi_error_meta_20260810_v1"
REGIME_CONTRACT = ROOT / "data_perp/artifacts/regime_grouped_larger_specialists_20260810_v1/feature_contract.json"
BASE_FIELDS = ["p_clear", "p_adverse", "p_weak", "base_score", "prequential_base_expected_net_bps"]
RESIDUAL_PARAMS = {
    "n_estimators": 220,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 31,
    "min_child_samples": 893,
    "min_sum_hessian_in_leaf": 1.13,
    "min_gain_to_split": 0.00893,
    "colsample_bytree": 0.79,
    "subsample": 0.87,
    "subsample_freq": 1,
    "reg_alpha": 0.03,
    "reg_lambda": 0.17,
    "max_bin": 63,
    "label_gain": [0.0, 0.25, 1.0, 3.0, 7.0, 12.0],
    "verbosity": -1,
    "random_state": 20260810,
    "n_jobs": 1,
}


def _candidate_fields(store_cols: set[str]) -> list[str]:
    keys = list(dict.fromkeys(
        list(PERP_META_PRIMARY_FEATURE_KEYS)
        + list(RESIDUAL_META_FEATURE_KEYS)
        + list(MARKET_CROSS_SECTIONAL_META_FEATURE_KEYS)
        + list(T2_FUNNEL_META_CONTEXT_FEATURE_KEYS)
    ))
    return [x for x in keys if x in store_cols]


def _query(frame: pd.DataFrame, derived: pd.DataFrame, train_derived: pd.DataFrame, key: str, *, train: bool = False) -> pd.Series:
    symbol = frame.candidate_id.astype(str).str.split("|").str[0]
    source = derived
    if key == "volatility":
        bt = _bins(train_derived.volatility_proxy.to_numpy(float), source.volatility_proxy.to_numpy(float))
        return symbol + "|" + pd.Series(bt, index=frame.index).astype(str)
    if key == "trend":
        bt = _bins(train_derived.trend_proxy.to_numpy(float), source.trend_proxy.to_numpy(float))
        return symbol + "|" + pd.Series(bt, index=frame.index).astype(str)
    if key == "transition":
        bt = _bins(train_derived.transition_intensity.to_numpy(float), source.transition_intensity.to_numpy(float))
        return symbol + "|" + pd.Series(bt, index=frame.index).astype(str)
    if key == "entropy":
        bt = _bins(train_derived.transition_entropy.to_numpy(float), source.transition_entropy.to_numpy(float))
        return symbol + "|" + pd.Series(bt, index=frame.index).astype(str)
    return (
        symbol
        + "|" + source.volatility_proxy.round().astype(str)
        + "|" + source.trend_proxy.round().astype(str)
        + "|" + source.transition_intensity.round().astype(str)
        + "|" + source.transition_entropy.round().astype(str)
    )


def _side_month_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, x in pred.groupby("side_name", sort=False):
        rows.append({"scope": "side", "side": side, **global_tail_metrics(x), **monthly_stability(x)})
    for month, x in pred.assign(_month=pd.to_datetime(pred.__ts__, utc=True).dt.strftime("%Y-%m")).groupby("_month", sort=True):
        rows.append({"scope": "month", "month": month, **global_tail_metrics(x), **monthly_stability(x)})
    return pd.DataFrame(rows)


def _diagnostic(pred: pd.DataFrame) -> pd.DataFrame:
    """Break down monthly failures by side, score decile, and regime proxies."""
    x = pred.copy()
    x["month"] = pd.to_datetime(x["__ts__"], utc=True).dt.strftime("%Y-%m")
    x["score_decile"] = x.groupby(["month", "side_name"], sort=False).score.rank(pct=True, method="first")
    x["score_decile"] = np.ceil(x.score_decile * 10).clip(1, 10).astype(int)
    rows = []
    for (month, side, decile), g in x.groupby(["month", "side_name", "score_decile"], sort=True):
        rows.append({"scope": "score_decile", "month": month, "side": side, "score_decile": decile, "rows": len(g), "mean_net_bps": float(g.net_bps.mean()), "mean_gross_bps": float(g.gross_bps.mean()), "mean_base_expected_net_bps": float(g.prequential_base_expected_net_bps.mean()), "mean_conversion_error_bps": float((g.net_bps-g.prequential_base_expected_net_bps).mean())})
    # Derived regime bins are decision-time fields and are diagnostic only.
    for col in [c for c in x.columns if c.startswith("regime_bin__")]:
        for (month, side, state), g in x.groupby(["month", "side_name", col], sort=True, dropna=False):
            rows.append({"scope": col, "month": month, "side": side, "state": state, "rows": len(g), "mean_net_bps": float(g.net_bps.mean()), "mean_gross_bps": float(g.gross_bps.mean()), "mean_base_expected_net_bps": float(g.prequential_base_expected_net_bps.mean()), "mean_conversion_error_bps": float((g.net_bps-g.prequential_base_expected_net_bps).mean())})
    return pd.DataFrame(rows)


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    base = _base()
    store_cols = _schema()
    contract = json.loads(REGIME_CONTRACT.read_text())
    specialist_fields = list(contract["features"])
    regime_store = [f for f in contract["regime_fields"] if f in store_cols]
    cmi_candidates = _candidate_fields(store_cols)
    # The regime-driver fields are already included under a prefixed name in
    # the meta contract; do not merge them a second time under their raw names.
    cmi_candidates = [f for f in cmi_candidates if f not in regime_store]
    first = LONG_HISTORY_FOLDS[3]
    template = base[base.__ts__.between(_utc(first.train_start), _utc(first.calibration_start), inclusive="left")]
    if len(specialist_fields) != 160:
        specialist_fields = _large_fields(template, store_cols)
    predictions: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    for fold in LONG_HISTORY_FOLDS[3:]:
        a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
        tr = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)]
        ca = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)]
        te = base[base.__ts__.between(c, e, inclusive="left")]
        for side in ("long", "short"):
            train, cal, test = (z[z.side_name.eq(side)].copy() for z in (tr, ca, te))
            train = train.sample(min(150000, len(train)), random_state=20260810)
            train_store = train.merge(_store_rows(train, specialist_fields + regime_store), on="candidate_id", validate="one_to_one")
            cal_store = cal.merge(_store_rows(cal, specialist_fields + regime_store + cmi_candidates), on="candidate_id", validate="one_to_one")
            test_store = test.merge(_store_rows(test, specialist_fields + regime_store + cmi_candidates), on="candidate_id", validate="one_to_one")
            train_reg = _regime_columns(train_store[regime_store])
            cal_reg = _regime_columns(cal_store[regime_store])
            test_reg = _regime_columns(test_store[regime_store])
            train_scores = train[["candidate_id"]].copy()
            cal_scores = cal[["candidate_id"]].copy()
            test_scores = test[["candidate_id"]].copy()
            for key in REGIMES:
                qtr = _query(train_store, train_reg, train_reg, key, train=True)
                qca = _query(cal_store, cal_reg, train_reg, key)
                qte = _query(test_store, test_reg, train_reg, key)
                target = (train_store.net_bps.to_numpy(float) > 50.0).astype(np.int32)
                model, med = _rank_fit(train_store, specialist_fields, target, qtr)
                train_scores["rg__" + key] = model.predict(train_store[specialist_fields].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0))
                cal_scores["rg__" + key] = model.predict(cal_store[specialist_fields].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0))
                test_scores["rg__" + key] = model.predict(test_store[specialist_fields].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0))
                del model
            score_cols = [c for c in cal_scores.columns if c.startswith("rg__")]
            # Causal 7-day error history from the five regime-specialist scores.
            cal_hist = cal[["candidate_id", "__ts__", "side_name"]].merge(cal_scores, on="candidate_id", validate="one_to_one")
            test_hist = test[["candidate_id", "__ts__", "side_name"]].merge(test_scores, on="candidate_id", validate="one_to_one")
            hist_frame = pd.concat([cal_hist, test_hist], ignore_index=True)
            known = pd.concat([(cal.net_bps > 50.0).astype(float).reset_index(drop=True), pd.Series(np.nan, index=np.arange(len(test)))], ignore_index=True)
            hist = _rolling_features(hist_frame, score_cols, known)
            cal_meta = cal[["candidate_id", "__ts__", "net_bps", "gross_bps", "side_name", *BASE_FIELDS]].merge(cal_scores, on="candidate_id", validate="one_to_one")
            test_meta = test[["candidate_id", "__ts__", "net_bps", "gross_bps", "side_name", *BASE_FIELDS]].merge(test_scores, on="candidate_id", validate="one_to_one")
            regime_names = ["regime__" + x for x in regime_store] + ["regime__" + x for x in ["volatility_proxy", "trend_proxy", "transition_intensity", "transition_entropy"]]
            cal_meta = cal_meta.merge(cal_store[["candidate_id", *regime_store]], on="candidate_id", validate="one_to_one")
            test_meta = test_meta.merge(test_store[["candidate_id", *regime_store]], on="candidate_id", validate="one_to_one")
            cal_reg_values = pd.DataFrame({"candidate_id": cal_store.candidate_id.to_numpy()})
            test_reg_values = pd.DataFrame({"candidate_id": test_store.candidate_id.to_numpy()})
            for col in ["volatility_proxy", "trend_proxy", "transition_intensity", "transition_entropy"]:
                cal_reg_values["regime__" + col] = cal_reg[col].to_numpy(float)
                test_reg_values["regime__" + col] = test_reg[col].to_numpy(float)
            cal_meta = cal_meta.merge(cal_reg_values, on="candidate_id", validate="one_to_one")
            test_meta = test_meta.merge(test_reg_values, on="candidate_id", validate="one_to_one")
            cal_meta = cal_meta.merge(_store_rows(cal, cmi_candidates), on="candidate_id", validate="one_to_one")
            test_meta = test_meta.merge(_store_rows(test, cmi_candidates), on="candidate_id", validate="one_to_one")
            cal_meta = cal_meta.merge(hist[hist.candidate_id.isin(cal.candidate_id)], on="candidate_id", validate="one_to_one")
            test_meta = test_meta.merge(hist[hist.candidate_id.isin(test.candidate_id)], on="candidate_id", validate="one_to_one")
            select = cal_meta.iloc[: max(1, len(cal_meta)//2)]
            residual = select.net_bps.to_numpy(float) - select.prequential_base_expected_net_bps.to_numpy(float)
            remaining = []
            for field in cmi_candidates:
                v = pd.to_numeric(select[field], errors="coerce")
                scale = float((v-v.median()).abs().median()) if v.notna().any() else 0.0
                if float(v.notna().mean()) >= .90 and np.isfinite(scale) and scale > 1e-8:
                    remaining.append((field, _cmi(v.to_numpy(float), residual, select.base_score.to_numpy(float))))
            remaining.sort(key=lambda z: (-z[1], z[0]))
            chosen = [x[0] for x in remaining[:5] if np.isfinite(x[1])]
            for i, (field, score) in enumerate(remaining[:5], 1):
                selected_rows.append({"fold": fold.name, "side": side, "step": i, "feature": field, "cmi": score})
            history7 = [c for c in hist.columns if "__7d__" in c]
            meta_fields = list(dict.fromkeys(BASE_FIELDS + score_cols + regime_names + chosen + history7))
            meta_fields = [f for f in meta_fields if f in cal_meta.columns and f in test_meta.columns]
            raw = _fit_residual(cal_meta, test_meta, meta_fields, "q4h_side", RESIDUAL_PARAMS)
            out_frame = test[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "prequential_base_expected_net_bps"]].copy()
            out_frame["score"] = test.prequential_base_expected_net_bps.to_numpy(float) + np.clip(raw, -50.0, 50.0)
            out_frame["fold"] = fold.name
            out_frame["cmi_count"] = len(chosen)
            out_frame["error_history"] = "7d_ic_hr_surprise"
            out_frame["regime_specialist_count"] = len(REGIMES)
            for key, source in (("volatility", test_reg.volatility_proxy), ("trend", test_reg.trend_proxy), ("transition", test_reg.transition_intensity), ("entropy", test_reg.transition_entropy)):
                out_frame["regime_bin__" + key] = pd.qcut(source.rank(method="first"), 4, labels=False, duplicates="drop").to_numpy()
            predictions.append(out_frame)
            del train_store, cal_store, test_store, cal_meta, test_meta, hist, hist_frame, raw
            gc.collect()
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(selected_rows).to_parquet(out / "selected_cmi_features.parquet", index=False)
    pd.DataFrame([{"arm": "combined_regime_cmi_7d_error", **global_tail_metrics(pred), **monthly_stability(pred)}]).to_parquet(out / "metrics.parquet", index=False)
    _side_month_metrics(pred).to_parquet(out / "side_month_metrics.parquet", index=False)
    _diagnostic(pred).to_parquet(out / "bad_month_diagnostic.parquet", index=False)
    (out / "feature_contract.json").write_text(json.dumps({"specialist_feature_count": len(specialist_fields), "specialist_features": specialist_fields, "regime_fields_fed_to_meta": regime_names, "cmi_candidate_families": "config meta keys only", "error_history_features": "five regime heads × 7-day IC/hit-rate/hit-surprise", "meta_max_depth": 4}, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps({"schema": "combined_regime_cmi_error_meta_v1", "regime_specialists": list(REGIMES), "specialist_feature_count": len(specialist_fields), "specialist_target": "binary H12 net > +50 bps", "specialist_queries": "regime-based symbol × causal regime bins", "meta_query": "q4h_side", "meta_target": "ordinal per-row net residual in bps", "cmi": "five greedy binned-CMI additions, selection half only, config meta keys", "error_history": "7-day prior-only IC/hit-rate/hit-surprise", "max_depth": 4, "selection": "global top5 net then month stability then top1"}, indent=2) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    print(run(args.out))
