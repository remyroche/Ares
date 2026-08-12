#!/usr/bin/env python3
"""Side-local conversion model with residual specialists and EV calibration."""
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

from extreme_price_movements.prequential_r3_value_map import (  # noqa: E402
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)
from scripts.run_broad_multiview_specialist_lambdarank import (  # noqa: E402
    LONG_HISTORY_FOLDS,
    MAX_TRAIN_ROWS,
    _base,
    _rank_target,
    _ranker,
    _utc,
)
from scripts.run_frozen_multiview_specialist_input_ablation import (  # noqa: E402
    _schema,
    _store_rows,
)
from scripts.run_gated_prior_mapped_residual import (  # noqa: E402
    _base_map,
    _load_frozen_views,
    _select_regime_context_fields,
    _write_json,
)
from scripts.run_top40_reliability_costaware import _ensure_store_fields, _top40  # noqa: E402

OUT = ROOT / "data_perp/artifacts/side_local_conversion_residual_ev_20260806_v1"
FROZEN = ROOT / "data_perp/artifacts/frozen_multiview_specialist_input_ablation_20260805_v1"
LAMBDA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
TAILS = (.01, .05, .10)


def _residual_grade(residual: np.ndarray) -> np.ndarray:
    """Ordered conversion-error target, explicitly in residual space."""
    return _rank_target(np.asarray(residual, dtype=float)).astype(np.int32)


def _map_score_to_target(frame: pd.DataFrame, raw_score: np.ndarray, target_col: str, side: str) -> pd.DataFrame:
    q = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    score = np.tanh(np.asarray(raw_score, dtype=float)[q.index.to_numpy()])
    values, audit, _ = prequential_same_side_r3_value_map(
        exact_net_bps=q[target_col].to_numpy(float),
        decision_timestamps=q.__ts__,
        label_available_timestamps=q.label_available_ts,
        side=side,
        score=score,
        config=PrequentialR3ValueMapConfig(
            side=side, bins=20, min_global_rows=32, bin_shrink_rows=64,
            mapping_mode="monotone_pava", monotone_min_bin_rows=1,
        ),
    )
    return pd.DataFrame({
        "candidate_id": q.candidate_id.to_numpy(),
        "mapped_target_bps": values.astype(np.float32),
        "map_prior_support": audit.prior_resolved_global_support.to_numpy(np.int32),
    })


def _fit_residual_specialists(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, views: dict[str, list[str]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fit = train if len(train) <= MAX_TRAIN_ROWS else train.sample(MAX_TRAIN_ROWS, random_state=20260806).sort_values(["__ts__", "candidate_id"], kind="stable")
    cal_out: dict[str, np.ndarray] = {}
    test_out: dict[str, np.ndarray] = {}
    for view, fields in views.items():
        fitx = _ensure_store_fields(fit, fields)
        calx = _ensure_store_fields(cal, fields)
        testx = _ensure_store_fields(test, fields)
        med = fitx[fields].median()
        fit_frame = pd.concat([fitx[["__ts__"]], fitx[fields].fillna(med)], axis=1)
        target = _residual_grade(fitx.residual_bps.to_numpy(float))
        model, usable = _ranker(fit_frame, target, query_id=fitx.__ts__.dt.floor("4h"))
        cal_out[view] = model.predict(calx[usable].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        test_out[view] = model.predict(testx[usable].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        del fitx, calx, testx, fit_frame, model
        gc.collect()
    return cal_out, test_out


def _top_net(frame: pd.DataFrame, score_col: str, fraction: float) -> float:
    n = max(1, int(np.ceil(len(frame) * fraction)))
    return float(frame.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable").head(n).net_bps.mean())


def _select_lambda(cal: pd.DataFrame) -> tuple[float, float, float]:
    base = _top_net(cal, "base_ev_bps", .10)
    best = (base, 0.0, base)
    for lam in LAMBDA_GRID:
        col = f"score_lambda_{int(lam*100):03d}"
        value = _top_net(cal, col, .10)
        if value > best[0] + 1e-9:
            best = (value, lam, base)
    return best


def _metrics(frame: pd.DataFrame, score_col: str, system: str, fold: str, period: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side, sub in [("pooled", frame), *[(s, frame[frame.side_name.eq(s)]) for s in ("long", "short")]]:
        if len(sub) == 0:
            continue
        for tail in TAILS:
            n = max(1, int(np.ceil(len(sub) * tail)))
            top = sub.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            rows.append({"system": system, "fold": fold, "period": period, "side": side, "tail": tail, "rows": len(sub), "trades": n, "gross_bps": float(top.gross_bps.mean()), "net_bps": float(top.net_bps.mean()), "rank_ic": float(sub[score_col].rank().corr(sub.net_bps.rank()))})
    return rows


def run(out: Path = OUT, frozen_artifact: Path = FROZEN) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    folds = LONG_HISTORY_FOLDS[3:]
    base = _base()
    mapped, map_audit, map_manifest = _base_map(base)
    base = base.merge(mapped, on=["candidate_id", "__ts__", "side_name"], validate="one_to_one")
    base["residual_bps"] = base.net_bps - base.causal_base_map_bps
    available = _schema()
    probe = base.iloc[: min(30_000, len(base))].merge(_store_rows(base.iloc[: min(30_000, len(base))], available), on="candidate_id", validate="one_to_one")
    regime_fields = _select_regime_context_fields(probe, set(available))
    selected_context = ["ema20_slope_5h", "mkt_volume_z_24h", "funding_abs_z", "mkt_oi_chg_z_24h", "atr_percentile", "distance_to_resistance_daily_vwap_atr"]
    context_fields = list(dict.fromkeys(f for f in selected_context + regime_fields if f in available))
    views = {side: _load_frozen_views(frozen_artifact, side) for side in ("long", "short")}
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    specialists: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    fold_outputs: list[pd.DataFrame] = []
    for fold in folds:
        a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
        tr = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)].copy()
        ca = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)].copy()
        te = base[base.__ts__.between(c, e, inclusive="left")].copy()
        fold_side_outputs: list[pd.DataFrame] = []
        for side in ("long", "short"):
            train = tr[tr.side_name.eq(side)].copy(); cal = ca[ca.side_name.eq(side)].copy(); test = te[te.side_name.eq(side)].copy()
            for frame in (train, cal, test):
                frame["admitted_top40"] = _top40(frame).to_numpy(bool)
            train_admit = train[train.admitted_top40].copy()
            cal_admit = cal[cal.admitted_top40].copy().sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
            test_admit = test[test.admitted_top40].copy()
            if len(train_admit) < 200 or len(cal_admit) < 200 or len(test_admit) < 20:
                continue
            specialist_cal, specialist_test = _fit_residual_specialists(train_admit, cal_admit, test_admit, views[side])
            specialist_fields: list[str] = []
            for name in views[side]:
                field = "sp_residual__" + name
                specialist_fields.append(field)
                cal_admit[field] = specialist_cal[name]
                test_admit[field] = specialist_test[name]
                specialists.append({"fold": fold.name, "side": side, "specialist": name, "target": "ordinalized net_bps - causal_base_map_bps", "train_rows": len(train_admit), "cal_rows": len(cal_admit), "test_rows": len(test_admit)})
            joined = _store_rows(pd.concat([cal_admit, test_admit], ignore_index=True), context_fields)
            n_cal = len(cal_admit)
            cal_admit = cal_admit.merge(joined.iloc[:n_cal], on="candidate_id", validate="one_to_one")
            test_admit = test_admit.merge(joined.iloc[n_cal:], on="candidate_id", validate="one_to_one")
            train_admit = _ensure_store_fields(train_admit, context_fields)
            fields = list(dict.fromkeys(["base_score", "p_clear", "p_adverse", "p_weak", "causal_base_map_bps", "map_prior_global_support", "map_prior_bin_support", "map_neutral_fallback", *specialist_fields, *context_fields]))
            fields = [f for f in fields if f in train_admit.columns and f in cal_admit.columns and f in test_admit.columns and pd.api.types.is_numeric_dtype(train_admit[f])]
            target = _residual_grade(train_admit.residual_bps.to_numpy(float))
            q = train_admit.__ts__.dt.floor("4h")
            queries.append({"fold": fold.name, "side": side, "fit_rows": len(train_admit), "queries": int(q.nunique()), "median_rows_per_query": float(q.value_counts().median()), "min_rows_per_query": int(q.value_counts().min())})
            model, usable = _ranker(pd.concat([train_admit[["__ts__"]], train_admit[fields]], axis=1), target, query_id=q)
            cal_admit["meta_raw_score"] = model.predict(cal_admit[usable].replace([np.inf, -np.inf], np.nan).fillna(0.0))
            test_admit["meta_raw_score"] = model.predict(test_admit[usable].replace([np.inf, -np.inf], np.nan).fillna(0.0))
            eval_admit = pd.concat([cal_admit, test_admit], ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
            mapped_resid = _map_score_to_target(eval_admit, eval_admit.meta_raw_score.to_numpy(float), "residual_bps", side)
            eval_admit = eval_admit.merge(mapped_resid, on="candidate_id", validate="one_to_one")
            cal_ev = eval_admit[eval_admit.__ts__.lt(c)].copy(); test_ev = eval_admit[eval_admit.__ts__.ge(c)].copy()
            cal_eval = cal.copy().merge(cal_ev[["candidate_id", "mapped_target_bps"]], on="candidate_id", how="left", validate="one_to_one")
            test_eval = test.copy().merge(test_ev[["candidate_id", "mapped_target_bps"]], on="candidate_id", how="left", validate="one_to_one")
            cal_eval["base_ev_bps"] = cal_eval.causal_base_map_bps
            test_eval["base_ev_bps"] = test_eval.causal_base_map_bps
            cal_eval["meta_residual_ev_bps"] = cal_eval.mapped_target_bps
            test_eval["meta_residual_ev_bps"] = test_eval.mapped_target_bps
            for lam in LAMBDA_GRID:
                col = f"score_lambda_{int(lam*100):03d}"
                cal_eval[col] = cal_eval.base_ev_bps
                test_eval[col] = test_eval.base_ev_bps
                mask = cal_eval.admitted_top40.to_numpy(bool)
                cal_eval.loc[mask, col] = cal_eval.loc[mask, "base_ev_bps"] + lam * cal_eval.loc[mask, "meta_residual_ev_bps"]
                mask = test_eval.admitted_top40.to_numpy(bool)
                test_eval.loc[mask, col] = test_eval.loc[mask, "base_ev_bps"] + lam * test_eval.loc[mask, "meta_residual_ev_bps"]
            selected_value, selected_lambda, base_value = _select_lambda(cal_eval)
            selections.append({"fold": fold.name, "side": side, "selected_lambda": selected_lambda, "selected_cal_top10_net_bps": selected_value, "base_cal_top10_net_bps": base_value})
            for lam in LAMBDA_GRID:
                col = f"score_lambda_{int(lam*100):03d}"
                system = "base_only" if lam == 0 else f"base_plus_residual_ev_lambda_{lam:.2f}"
                metrics.extend(_metrics(test_eval, col, system, fold.name, "fold"))
                test_eval["month"] = pd.to_datetime(test_eval.__ts__, utc=True).dt.strftime("%Y-%m")
                for month, month_frame in test_eval.groupby("month", sort=True):
                    metrics.extend(_metrics(month_frame, col, system, fold.name, month))
            test_eval["fold"] = fold.name
            keep = ["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "base_score", "admitted_top40", "base_ev_bps", "meta_residual_ev_bps", *[f"score_lambda_{int(lam*100):03d}" for lam in LAMBDA_GRID], "fold"]
            out_side = test_eval[keep].copy(); predictions.append(out_side); fold_side_outputs.append(out_side)
            del model, specialist_cal, specialist_test
            gc.collect()
        if fold_side_outputs:
            fold_outputs.append(pd.concat(fold_side_outputs, ignore_index=True))
        pd.DataFrame(metrics).to_parquet(out / "metrics.checkpoint.parquet", index=False)
        pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.checkpoint.parquet", index=False)
        _write_json(out / "progress.json", {"status": "running", "completed_fold": fold.name})
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(selections).to_parquet(out / "lambda_selection.parquet", index=False)
    pd.DataFrame(specialists).to_parquet(out / "specialist_target_audit.parquet", index=False)
    pd.DataFrame(queries).to_parquet(out / "query_audit.parquet", index=False)
    map_audit.to_parquet(out / "base_map_audit.parquet", index=False)
    _write_json(out / "base_map_manifest.json", map_manifest)
    _write_json(out / "conversion_contract.json", {"schema": "side_local_conversion_residual_ev_v1", "admission": "base raw score top 40% within 4h x side query", "specialist_target": "ordinalized net_bps - causal_base_map_bps", "meta_target": "same residual target; no direct-net or residualized-specialist mismatch", "meta_query": "native LambdaRank, 4h x side", "base_map": "side-local prior-resolved monotone PAVA to expected net bps", "meta_map": "side-local prior-resolved monotone PAVA to expected residual bps", "strict_boundary": "label_available_ts < decision_timestamp", "lambda_grid": list(LAMBDA_GRID), "context_fields": context_fields})
    _write_json(out / "manifest.json", {"schema": "side_local_conversion_residual_ev_v1", "folds": [f.name for f in folds], "specialist_target": "ordinalized H12 net residual bps", "meta_target": "ordinalized H12 net residual bps", "ev_mapping": "side-local prior-resolved base net map plus side-local prior-resolved residual map", "admission": "raw base score top 40% per 4h x side query", "lambda_grid": list(LAMBDA_GRID), "query_objective": "native LambdaRank, 4h x side", "base_map": map_manifest})
    _write_json(out / "progress.json", {"status": "complete", "completed_folds": [f.name for f in folds]})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--frozen-artifact", type=Path, default=FROZEN)
    args = parser.parse_args()
    print(run(args.out, args.frozen_artifact))
