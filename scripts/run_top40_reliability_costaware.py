#!/usr/bin/env python3
"""Top-40% reliability residual ablation.

The base model remains the opportunity model.  This experiment admits only
the broad base model's top 40% within each 4-hour x side query, retrains the
frozen specialist views with explicit cost-aware labels, and learns a bounded
three-class reliability ranking:

    overconfident base  -> 0
    approximately right -> 1
    underconfident base -> 2

The reliability model is native LambdaRank with 4-hour x side queries.  Its
raw ranking score is converted only to a conservative bounded correction; no
isotonic/calibration mapping is used.  All score selection is side-local and
OOF, and candidates outside the base top-40% receive the unchanged mapped base
score.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_broad_multiview_specialist_lambdarank import (  # noqa: E402
    LONG_HISTORY_FOLDS,
    MAX_TRAIN_ROWS,
    _base,
    _ranker,
    _utc,
)
from scripts.run_frozen_multiview_specialist_input_ablation import (  # noqa: E402
    _schema,
    _store_columns,
    _store_rows,
)
from scripts.run_gated_prior_mapped_residual import (  # noqa: E402
    _base_map,
    _load_frozen_views,
    _select_regime_context_fields,
    _write_json,
)

FROZEN_SPECIALIST_ARTIFACT = ROOT / "data_perp/artifacts/frozen_multiview_specialist_input_ablation_20260805_v1"
OUT = ROOT / "data_perp/artifacts/top40_reliability_costaware_20260805_v1"
TAILS = (.01, .05, .10)
COST_BPS = 100.0
COST_CLEAR_MARGINS = (25.0, 50.0, 75.0)
CORRECTION_CAPS = (25.0, 50.0, 75.0)
LAMBDAS = (0.0, 0.25, 0.5)
CONFIDENCE_THRESHOLDS = (0.0, 0.25, 0.5)
SEED = 20260805


def _top40(frame: pd.DataFrame) -> pd.Series:
    """Causal admission mask using only base score and decision timestamp."""
    q = frame.__ts__.dt.floor("4h")
    work = frame[["side_name", "base_score"]].copy()
    work["q"] = q.to_numpy()
    work["rank"] = work.groupby(["side_name", "q"], sort=False)["base_score"].rank(method="first", ascending=False)
    work["n"] = work.groupby(["side_name", "q"], sort=False)["base_score"].transform("size")
    return (work["rank"] <= np.ceil(work["n"] * 0.40)).astype(bool)


def _reliability_grade(residual: np.ndarray, hurdle: float = 50.0) -> np.ndarray:
    return np.select([residual < -hurdle, residual > hurdle], [0, 2], default=1).astype(np.int32)


def _ensure_store_fields(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Join only absent fields so context/specialist joins never suffix columns."""
    missing = [field for field in fields if field not in frame.columns]
    if not missing:
        return frame.copy()
    return frame.merge(_store_rows(frame, missing), on="candidate_id", validate="one_to_one")


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)) * 1.4826)
    return center, max(mad, 1e-6)


def _signed_reliability(raw: np.ndarray, center: float, scale: float) -> np.ndarray:
    # Positive means underconfident (base value too low); negative means
    # overconfident.  tanh is only a bounded score transform, not calibration.
    return np.tanh((np.asarray(raw, dtype=float) - center) / scale)


def _top_net(values: np.ndarray, net: np.ndarray, fraction: float) -> float:
    good = np.isfinite(values) & np.isfinite(net)
    if int(good.sum()) < 20:
        return -np.inf
    n = max(1, int(np.ceil(good.sum() * fraction)))
    order = np.argsort(-values[good], kind="stable")[:n]
    return float(np.mean(net[good][order]))


def _choose_correction(val: pd.DataFrame, signed: np.ndarray) -> dict[str, float | bool]:
    base = val.causal_base_map_bps.to_numpy(float)
    net = val.net_bps.to_numpy(float)
    admitted = val.admitted_top40.to_numpy(bool)
    noop = _top_net(base, net, .10)
    best = {"lambda": 0.0, "cap_bps": 0.0, "threshold": 1.0, "beats_noop": False, "economic_top10_net_bps": noop, "noop_top10_net_bps": noop}
    for lam in LAMBDAS:
        for cap in CORRECTION_CAPS:
            for threshold in CONFIDENCE_THRESHOLDS:
                active = admitted & (np.abs(signed) >= threshold)
                correction = np.where(active, float(lam) * float(cap) * signed, 0.0)
                score = base + correction
                econ = _top_net(score, net, .10)
                if econ > float(best["economic_top10_net_bps"]) + 1e-9:
                    best = {"lambda": float(lam), "cap_bps": float(cap), "threshold": float(threshold), "beats_noop": bool(econ > noop), "economic_top10_net_bps": econ, "noop_top10_net_bps": noop}
    if not bool(best["beats_noop"]):
        best.update({"lambda": 0.0, "cap_bps": 0.0, "threshold": 1.0})
    return best


def _fit_cost_specialists(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, views: dict[str, list[str]], margin_bps: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fit = train if len(train) <= MAX_TRAIN_ROWS else train.sample(MAX_TRAIN_ROWS, random_state=SEED).sort_values(["__ts__", "candidate_id"], kind="stable")
    cal_out: dict[str, np.ndarray] = {}
    test_out: dict[str, np.ndarray] = {}
    for view, fields in views.items():
        fitx = _ensure_store_fields(fit, fields)
        calx = _ensure_store_fields(cal, fields)
        testx = _ensure_store_fields(test, fields)
        med = fitx[fields].median()
        X = fitx[fields].fillna(med).astype(np.float32)
        C = calx[fields].fillna(med).astype(np.float32)
        T = testx[fields].fillna(med).astype(np.float32)
        # Explicit cost-aware target: gross H12 outcome minus the declared
        # entry/exit cost must exceed the robust-clear margin.
        target = ((fitx.gross_bps.to_numpy(float) - COST_BPS) > float(margin_bps)).astype(np.int8)
        clf = lgb.LGBMClassifier(
            objective="binary", n_estimators=180, learning_rate=.04,
            num_leaves=20, min_child_samples=400, colsample_bytree=.8,
            reg_lambda=20.0, random_state=SEED, n_jobs=1, verbosity=-1,
        ).fit(X, target)
        cal_out[view] = clf.predict_proba(C)[:, 1]
        test_out[view] = clf.predict_proba(T)[:, 1]
        del fitx, calx, testx, X, C, T, clf
        gc.collect()
    return cal_out, test_out


def _metrics(frame: pd.DataFrame, score_col: str, system: str, fold: str, period: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side, sub in [("pooled", frame), *[(s, frame[frame.side_name.eq(s)]) for s in ("long", "short")]]:
        if len(sub) == 0:
            continue
        for tail in TAILS:
            n = max(1, int(np.ceil(len(sub) * tail)))
            top = sub.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            rows.append({
                "system": system, "fold": fold, "period": period, "side": side,
                "tail": tail, "rows": len(sub), "trades": n,
                "admitted_rows": int(sub.admitted_top40.sum()),
                "admission_rate": float(sub.admitted_top40.mean()),
                "gross_bps": float(top.gross_bps.mean()), "net_bps": float(top.net_bps.mean()),
                "rank_ic": float(sub[score_col].rank().corr(sub.net_bps.rank())),
            })
    return rows


def _fit_reliability(fit: pd.DataFrame, val: pd.DataFrame, val_all: pd.DataFrame, test: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool], dict[str, float]]:
    fit_target = _reliability_grade(fit.net_bps.to_numpy(float) - fit.causal_base_map_bps.to_numpy(float))
    fit_frame = pd.concat([fit[["__ts__"]], fit[fields]], axis=1)
    model, usable = _ranker(fit_frame, fit_target, query_id=fit.__ts__.dt.floor("4h"))
    raw_fit = model.predict(fit[usable].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    raw_val = model.predict(val[usable].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    center, scale = _robust_scale(raw_fit)
    signed_val = _signed_reliability(raw_val, center, scale)
    # Select correction parameters on the complete validation population. The
    # residual model is only trained on admitted rows, but global top-k ranking
    # still includes the 60% non-admitted candidates at their unchanged base
    # scores.
    signed_full = np.zeros(len(val_all), dtype=float)
    lookup = pd.Series(signed_val, index=val["candidate_id"].astype(str))
    for i, candidate_id in enumerate(val_all["candidate_id"].astype(str)):
        if candidate_id in lookup.index:
            signed_full[i] = float(lookup.loc[candidate_id])
    val_eval = val_all.copy()
    val_eval["reliability_signed"] = signed_full
    params = _choose_correction(val_eval, signed_full)
    # Final model is fit on all calibration rows, while correction parameters
    # remain selected on the temporally later OOF validation slice.
    all_target = _reliability_grade(np.r_[fit.net_bps.to_numpy(float), val.net_bps.to_numpy(float)] - np.r_[fit.causal_base_map_bps.to_numpy(float), val.causal_base_map_bps.to_numpy(float)])
    all_cal = pd.concat([fit, val], ignore_index=True)
    final, final_fields = _ranker(pd.concat([all_cal[["__ts__"]], all_cal[fields]], axis=1), all_target, query_id=all_cal.__ts__.dt.floor("4h"))
    raw_cal = final.predict(all_cal[final_fields].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    raw_test = final.predict(test[final_fields].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    center_final, scale_final = _robust_scale(raw_cal)
    del model, final
    return raw_test, _signed_reliability(raw_test, center_final, scale_final), params, {"center": center_final, "scale": scale_final, "usable_fields": len(final_fields)}


def run(out: Path = OUT, frozen_artifact: Path = FROZEN_SPECIALIST_ARTIFACT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    folds = LONG_HISTORY_FOLDS[3:]
    base = _base()
    mapped, map_audit, map_manifest = _base_map(base)
    base = base.merge(mapped, on=["candidate_id", "__ts__", "side_name"], validate="one_to_one")
    available = _schema()
    template_probe = base.iloc[: min(30_000, len(base))].merge(_store_rows(base.iloc[: min(30_000, len(base))], available), on="candidate_id", validate="one_to_one")
    regime_fields = _select_regime_context_fields(template_probe, set(available))
    views = {side: _load_frozen_views(frozen_artifact, side) for side in ("long", "short")}
    selected_context = ["ema20_slope_5h", "mkt_volume_z_24h", "funding_abs_z", "mkt_oi_chg_z_24h", "atr_percentile", "distance_to_resistance_daily_vwap_atr"]
    context_fields = list(dict.fromkeys(f for f in selected_context + regime_fields if f in available))
    _write_json(out / "base_map_manifest.json", map_manifest)
    map_audit.to_parquet(out / "base_map_audit.parquet", index=False)
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    target_audit: list[dict[str, object]] = []
    query_audit: list[dict[str, object]] = []
    for fold in folds:
        a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
        tr = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)].copy()
        ca = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)].copy()
        te = base[base.__ts__.between(c, e, inclusive="left")].copy()
        for side in ("long", "short"):
            train = tr[tr.side_name.eq(side)].copy(); cal = ca[ca.side_name.eq(side)].copy(); test = te[te.side_name.eq(side)].copy()
            for frame in (train, cal, test):
                frame["admitted_top40"] = _top40(frame).to_numpy(bool)
            train_admit = train[train.admitted_top40].copy()
            cal_admit = cal[cal.admitted_top40].copy().sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
            test_admit = test[test.admitted_top40].copy()
            if len(train_admit) < 200 or len(cal_admit) < 200 or len(test_admit) < 20:
                continue
            context = _store_rows(pd.concat([cal_admit, test_admit], ignore_index=True), context_fields)
            context_cal = context.iloc[: len(cal_admit)].copy(); context_test = context.iloc[len(cal_admit):].copy()
            cal_admit = cal_admit.merge(context_cal, on="candidate_id", validate="one_to_one")
            test_admit = test_admit.merge(context_test, on="candidate_id", validate="one_to_one")
            split = max(1, int(len(cal_admit) * .60))
            fit = cal_admit.iloc[:split].copy(); val = cal_admit.iloc[split:].copy()
            for margin in COST_CLEAR_MARGINS:
                cal_scores, test_scores = _fit_cost_specialists(train_admit, cal_admit, test_admit, views[side], margin)
                prefix = f"sp_cost_clear_{int(margin)}__"
                specialist_fields = []
                for name in views[side]:
                    field = prefix + name
                    specialist_fields.append(field)
                    cal_admit[field] = cal_scores[name]
                    test_admit[field] = test_scores[name]
                fields = list(dict.fromkeys(["base_score", "p_clear", "p_adverse", "p_weak", "causal_base_map_bps", "map_prior_global_support", "map_prior_bin_support", "map_neutral_fallback", *specialist_fields, *context_fields]))
                fields = [f for f in fields if f in cal_admit.columns and f in test_admit.columns and pd.api.types.is_numeric_dtype(cal_admit[f])]
                fit = cal_admit.iloc[:split].copy(); val = cal_admit.iloc[split:].copy()
                # Query audit proves the requested 4-hour x side construction.
                q = fit.__ts__.dt.floor("4h")
                query_audit.append({"fold": fold.name, "side": side, "cost_clear_margin_bps": margin, "fit_rows": len(fit), "fit_queries": int(q.nunique()), "median_rows_per_query": float(q.value_counts().median()), "min_rows_per_query": int(q.value_counts().min())})
                val_all = cal.iloc[split:].copy()
                raw_test, signed_test, params, scale = _fit_reliability(fit, val, val_all, test_admit, fields)
                # Reconstruct the validation signed score with the same model
                # selection path is deliberately avoided; params were selected
                # OOF and only the untouched test is used below.
                test_correction = np.where(np.abs(signed_test) >= float(params["threshold"]), float(params["lambda"]) * float(params["cap_bps"]) * signed_test, 0.0)
                out_side = test.copy()
                out_side["score"] = out_side.causal_base_map_bps.to_numpy(float)
                out_side["score"] = out_side["score"].to_numpy(float)
                out_side["no_op_score"] = out_side.causal_base_map_bps
                out_side["reliability_score"] = np.nan
                out_side["reliability_correction_bps"] = 0.0
                out_side["reliability_class"] = np.nan
                idx = np.flatnonzero(test.admitted_top40.to_numpy(bool))
                out_side.iloc[idx, out_side.columns.get_loc("score")] = test_admit.causal_base_map_bps.to_numpy(float) + test_correction
                out_side.iloc[idx, out_side.columns.get_loc("reliability_score")] = signed_test
                out_side.iloc[idx, out_side.columns.get_loc("reliability_correction_bps")] = test_correction
                out_side.iloc[idx, out_side.columns.get_loc("reliability_class")] = np.select([signed_test < -.33, signed_test > .33], [0, 2], default=1)
                out_side["fold"] = fold.name; out_side["side_name"] = side; out_side["cost_clear_margin_bps"] = margin
                out_side["lambda"] = float(params["lambda"]); out_side["cap_bps"] = float(params["cap_bps"]); out_side["confidence_threshold"] = float(params["threshold"])
                predictions.append(out_side[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "base_score", "causal_base_map_bps", "admitted_top40", "score", "no_op_score", "reliability_score", "reliability_correction_bps", "reliability_class", "cost_clear_margin_bps", "lambda", "cap_bps", "confidence_threshold", "fold"]])
                combined = pd.concat([out_side], ignore_index=True)
                metrics.extend(_metrics(combined, "score", f"top40_reliability_cost_clear_{int(margin)}", fold.name, "fold"))
                out_side["month"] = pd.to_datetime(out_side.__ts__, utc=True).dt.strftime("%Y-%m")
                for month, month_frame in out_side.groupby("month", sort=True):
                    metrics.extend(_metrics(month_frame, "score", f"top40_reliability_cost_clear_{int(margin)}", fold.name, month))
                target_audit.extend([
                    {"fold": fold.name, "side": side, "cost_clear_margin_bps": margin, "population": name, "rows": len(frame), "admitted_rows": int(frame.admitted_top40.sum()), "target_rate": float(((frame.gross_bps - COST_BPS) > margin).mean())}
                    for name, frame in [("train", train_admit), ("calibration", cal_admit), ("test", test_admit)]
                ])
                del cal_scores, test_scores, raw_test, signed_test, out_side
                gc.collect()
        pd.DataFrame(metrics).to_parquet(out / "metrics.checkpoint.parquet", index=False)
        pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.checkpoint.parquet", index=False)
        _write_json(out / "progress.json", {"status": "running", "completed_fold": fold.name})
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(target_audit).to_parquet(out / "specialist_target_audit.parquet", index=False)
    pd.DataFrame(query_audit).to_parquet(out / "query_audit.parquet", index=False)
    _write_json(out / "residual_feature_contract.json", {"schema": "top40_reliability_costaware_v1", "admission": "raw base_score top 40% within 4h x side query", "reliability_classes": {"0": "base overconfident: residual < -50 bps", "1": "approximately correct: abs residual <= 50 bps", "2": "base underconfident: residual > 50 bps"}, "query_cadence": "4h x side", "cost_bps": COST_BPS, "cost_clear_margins_bps": list(COST_CLEAR_MARGINS), "correction_caps_bps": list(CORRECTION_CAPS), "lambdas": list(LAMBDAS), "confidence_thresholds": list(CONFIDENCE_THRESHOLDS), "context_fields": context_fields, "specialist_fields": {side: {f"data_view_{i:02d}": views[side][f"data_view_{i:02d}"] for i in range(7)} for side in ("long", "short")}})
    _write_json(out / "manifest.json", {"schema": "top40_reliability_costaware_v1", "folds": [f.name for f in folds], "specialist_target": "gross_h12_bps - 100_cost_bps > margin", "reliability_target": "three-class residual reliability", "ranking": "native LambdaRank, 4h x side queries; raw score used for ranking, no calibration", "admission": "base raw score top 40% per 4h x side query", "correction": "OOF-selected bounded lambda x cap x tanh reliability score; outside-admission rows are no-op", "base_map": map_manifest})
    _write_json(out / "progress.json", {"status": "complete", "completed_folds": [f.name for f in folds]})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--frozen-artifact", type=Path, default=FROZEN_SPECIALIST_ARTIFACT)
    args = parser.parse_args()
    print(run(args.out, args.frozen_artifact))
