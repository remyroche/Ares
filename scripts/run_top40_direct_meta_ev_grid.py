#!/usr/bin/env python3
"""Direct top-40% meta target with separately EV-mapped score combinations."""
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
    _fixed_residual_contract,
    _load_frozen_views,
    _select_regime_context_fields,
    _write_json,
)
from scripts.run_top40_reliability_costaware import (  # noqa: E402
    _ensure_store_fields,
    _fit_cost_specialists,
    _top40,
)

OUT = ROOT / "data_perp/artifacts/top40_direct_meta_ev_grid_20260805_v1"
FROZEN = ROOT / "data_perp/artifacts/frozen_multiview_specialist_input_ablation_20260805_v1"
COST_CLEAR_MARGIN = 50.0
GRID = ((1.0, 0.0), (0.75, 0.25), (0.50, 0.50), (0.25, 0.75), (0.0, 1.0))
TAILS = (.01, .05, .10)


def _map_eval(frame: pd.DataFrame, raw_score: np.ndarray, side: str) -> pd.DataFrame:
    q = frame.sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    values, audit, _ = prequential_same_side_r3_value_map(
        exact_net_bps=q.net_bps.to_numpy(float),
        decision_timestamps=q.__ts__,
        label_available_timestamps=q.label_available_ts,
        side=side,
        # LambdaRank outputs are unbounded; the existing causal EV mapper
        # accepts an R3-style scalar in [-1, 1].  tanh is a fixed monotone
        # order-preserving transform and does not use labels or future rows.
        score=np.tanh(np.asarray(raw_score, dtype=float)[q.index.to_numpy()]),
        config=PrequentialR3ValueMapConfig(
            side=side, bins=20, min_global_rows=32, bin_shrink_rows=64,
            mapping_mode="monotone_pava", monotone_min_bin_rows=1,
        ),
    )
    return pd.DataFrame({
        "candidate_id": q.candidate_id.to_numpy(),
        "meta_ev_bps": values.astype(np.float32),
        "meta_map_support": audit.prior_resolved_global_support.to_numpy(np.int32),
    })


def _top_net(frame: pd.DataFrame, score_col: str, fraction: float) -> float:
    n = max(1, int(np.ceil(len(frame) * fraction)))
    return float(frame.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable").head(n).net_bps.mean())


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


def _select_grid(cal: pd.DataFrame, grid_cols: list[str]) -> tuple[float, float, float]:
    base = _top_net(cal, "base_ev_bps", .10)
    best = (base, 1.0, 0.0)
    for (wb, wm), col in zip(GRID, grid_cols):
        value = _top_net(cal, col, .10)
        if value > best[0] + 1e-9:
            best = (value, wb, wm)
    return best


def run(out: Path = OUT, frozen_artifact: Path = FROZEN) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    folds = LONG_HISTORY_FOLDS[3:]
    base = _base()
    mapped, map_audit, map_manifest = _base_map(base)
    base = base.merge(mapped, on=["candidate_id", "__ts__", "side_name"], validate="one_to_one")
    available = _schema()
    probe = base.iloc[: min(30_000, len(base))].merge(_store_rows(base.iloc[: min(30_000, len(base))], available), on="candidate_id", validate="one_to_one")
    regime_fields = _select_regime_context_fields(probe, set(available))
    selected_context = ["ema20_slope_5h", "mkt_volume_z_24h", "funding_abs_z", "mkt_oi_chg_z_24h", "atr_percentile", "distance_to_resistance_daily_vwap_atr"]
    context_fields = list(dict.fromkeys(f for f in selected_context + regime_fields if f in available))
    views = {side: _load_frozen_views(frozen_artifact, side) for side in ("long", "short")}
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
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
            specialist_cal, specialist_test = _fit_cost_specialists(train_admit, cal_admit, test_admit, views[side], COST_CLEAR_MARGIN)
            specialist_fields: list[str] = []
            for name in views[side]:
                field = "sp_cost_clear_50__" + name
                specialist_fields.append(field)
                cal_admit[field] = specialist_cal[name]
                test_admit[field] = specialist_test[name]
            # Join the causal context once, after specialist fitting, avoiding
            # duplicate-column suffixes when a specialist uses a context field.
            joined = _store_rows(pd.concat([cal_admit, test_admit], ignore_index=True), context_fields)
            cal_admit = cal_admit.merge(joined.iloc[: len(cal_admit)], on="candidate_id", validate="one_to_one")
            test_admit = test_admit.merge(joined.iloc[len(cal_admit):], on="candidate_id", validate="one_to_one")
            train_admit = _ensure_store_fields(train_admit, context_fields)
            fields = list(dict.fromkeys(["base_score", "p_clear", "p_adverse", "p_weak", "causal_base_map_bps", "map_prior_global_support", "map_prior_bin_support", "map_neutral_fallback", *specialist_fields, *context_fields]))
            fields = [f for f in fields if f in train_admit.columns and f in cal_admit.columns and f in test_admit.columns and pd.api.types.is_numeric_dtype(train_admit[f])]
            split = max(1, int(len(cal_admit) * .60))
            fit = train_admit.copy()
            # Direct economic target: no residual subtraction. The grade is
            # assigned from the realised H12 net outcome itself.
            target = _rank_target(fit.net_bps.to_numpy(float))
            query = fit.__ts__.dt.floor("4h")
            queries.append({"fold": fold.name, "side": side, "fit_rows": len(fit), "queries": int(query.nunique()), "median_rows_per_query": float(query.value_counts().median()), "min_rows_per_query": int(query.value_counts().min())})
            model, usable = _ranker(pd.concat([fit[["__ts__"]], fit[fields]], axis=1), target, query_id=query)
            raw_cal = model.predict(cal_admit[usable].replace([np.inf, -np.inf], np.nan).fillna(0.0))
            raw_test = model.predict(test_admit[usable].replace([np.inf, -np.inf], np.nan).fillna(0.0))
            cal_admit["meta_raw_score"] = raw_cal
            test_admit["meta_raw_score"] = raw_test
            # Each map is formed independently from its own score and only
            # previous resolved rows, then applied to the untouched test rows.
            eval_frame = pd.concat([cal_admit, test_admit], ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
            meta_map = _map_eval(eval_frame, eval_frame.meta_raw_score.to_numpy(float), side)
            eval_frame = eval_frame.merge(meta_map, on="candidate_id", validate="one_to_one")
            cal_ev = eval_frame.iloc[: len(cal_admit)].copy(); test_ev = eval_frame.iloc[len(cal_admit):].copy()
            # The chronological sort above can reorder rows, so select by
            # timestamp boundary rather than positional slicing.
            cal_ev = eval_frame[eval_frame.__ts__.lt(c)].copy(); test_ev = eval_frame[eval_frame.__ts__.ge(c)].copy()
            base_lookup = base[["candidate_id", "causal_base_map_bps"]].drop_duplicates("candidate_id")
            cal_ev = cal_ev.drop(columns=["causal_base_map_bps"], errors="ignore").merge(base_lookup, on="candidate_id", validate="one_to_one")
            test_ev = test_ev.drop(columns=["causal_base_map_bps"], errors="ignore").merge(base_lookup, on="candidate_id", validate="one_to_one")
            cal_eval = cal.copy().merge(cal_ev[["candidate_id", "meta_ev_bps"]], on="candidate_id", how="left", validate="one_to_one")
            test_eval = test.copy().merge(test_ev[["candidate_id", "meta_ev_bps"]], on="candidate_id", how="left", validate="one_to_one")
            cal_eval["base_ev_bps"] = cal_eval.causal_base_map_bps
            test_eval["base_ev_bps"] = test_eval.causal_base_map_bps
            grid_cols: list[str] = []
            for i, (wb, wm) in enumerate(GRID):
                col = f"score_wb{int(wb*100):03d}_wm{int(wm*100):03d}"
                grid_cols.append(col)
                cal_eval[col] = cal_eval.base_ev_bps
                test_eval[col] = test_eval.base_ev_bps
                admitted = cal_eval.admitted_top40.to_numpy(bool)
                cal_eval.loc[admitted, col] = wb * cal_eval.loc[admitted, "base_ev_bps"] + wm * cal_eval.loc[admitted, "meta_ev_bps"]
                admitted = test_eval.admitted_top40.to_numpy(bool)
                test_eval.loc[admitted, col] = wb * test_eval.loc[admitted, "base_ev_bps"] + wm * test_eval.loc[admitted, "meta_ev_bps"]
            selected_net, selected_wb, selected_wm = _select_grid(cal_eval, grid_cols)
            selections.append({"fold": fold.name, "side": side, "selected_wb": selected_wb, "selected_wm": selected_wm, "selected_cal_top10_net_bps": selected_net, "base_cal_top10_net_bps": _top_net(cal_eval, "base_ev_bps", .10)})
            for i, col in enumerate(grid_cols):
                system = "base_only" if i == 0 else ("meta_only" if i == len(grid_cols) - 1 else f"ev_mix_{GRID[i][0]:.2f}_{GRID[i][1]:.2f}")
                metrics.extend(_metrics(test_eval, col, system, fold.name, "fold"))
                test_eval["month"] = pd.to_datetime(test_eval.__ts__, utc=True).dt.strftime("%Y-%m")
                for month, month_frame in test_eval.groupby("month", sort=True):
                    metrics.extend(_metrics(month_frame, col, system, fold.name, month))
            test_eval["fold"] = fold.name
            keep = ["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "base_score", "admitted_top40", "base_ev_bps", "meta_ev_bps", *grid_cols, "fold"]
            predictions.append(test_eval[keep])
            del model, raw_cal, raw_test, specialist_cal, specialist_test
            gc.collect()
        pd.DataFrame(metrics).to_parquet(out / "metrics.checkpoint.parquet", index=False)
        pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.checkpoint.parquet", index=False)
        _write_json(out / "progress.json", {"status": "running", "completed_fold": fold.name})
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(selections).to_parquet(out / "weight_selection.parquet", index=False)
    pd.DataFrame(queries).to_parquet(out / "query_audit.parquet", index=False)
    map_audit.to_parquet(out / "base_map_audit.parquet", index=False)
    _write_json(out / "base_map_manifest.json", map_manifest)
    _write_json(out / "meta_ev_map_contract.json", {"schema": "top40_direct_meta_ev_map_v1", "meta_target": "direct ordinalized H12 net bps; no residual subtraction", "admission": "raw base score top 40% within 4h x side query", "query_cadence": "4h x side", "meta_ev_mapping": "side-local prior-resolved monotone PAVA", "strict_boundary": "label_available_ts < decision_timestamp", "cost_clear_specialist_margin_bps": COST_CLEAR_MARGIN, "grid": [{"base_weight": wb, "meta_weight": wm} for wb, wm in GRID], "context_fields": context_fields, "specialist_fields": specialist_fields})
    _write_json(out / "manifest.json", {"schema": "top40_direct_meta_ev_grid_v1", "folds": [f.name for f in folds], "meta_target": "direct ordinalized H12 net bps; not residual", "admission": "base raw score top 40% per 4h x side query", "query_objective": "native LambdaRank, 4h x side; raw ranking score before EV mapping", "ev_mapping": "independent side-local prior-resolved maps for base and meta", "combination_grid": [{"base_weight": wb, "meta_weight": wm} for wb, wm in GRID], "specialist_target": "gross_h12_bps - 100_cost_bps > 50 bps", "base_map": map_manifest})
    _write_json(out / "progress.json", {"status": "complete", "completed_folds": [f.name for f in folds]})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--frozen-artifact", type=Path, default=FROZEN)
    args = parser.parse_args()
    print(run(args.out, args.frozen_artifact))
