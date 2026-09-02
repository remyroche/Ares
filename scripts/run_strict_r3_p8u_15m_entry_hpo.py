#!/usr/bin/env python3
"""Bounded strict-OOS HPO for the P8U 15m ordinal-entry head (research only).

The target-free feature panel is immutable before policy labels are joined.  A
model is fit on exactly the prior two calendar months whose policy labels were
resolved before the next held month.  The runner compares target scale, loss,
model family and conservative admission interactions; it never promotes a row
below its named dual-MC1 floor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS
from scripts import run_strict_r3_p8u_15m_walkforward as source
from scripts import run_strict_r3_p8u_15m_continuation_walkforward as continuation


PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_ordinal_mc1_threshold_observed25h_20260830_v4_manifested_results/target_free_15m_features.parquet"
STATE_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_predictive_observed25h_20260830_v3/target_free_continuation_state_parts"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_hpo_20260830_v1"
FLOORS = (20.0, 30.0, 40.0, 50.0)
SPECS = ("lgb_l1_bps", "lgb_huber_bps", "lgb_l2_atr", "cat_mae_bps", "cat_rmse_bps", "cat_mae_atr", "lgb_ranker_grade")
INTERACTIONS = ("base", "veto_pred_ge_0", "veto_pred_ge_25", "blend25_veto")
GRADE_BINS = [-np.inf, -100.0, 0.0, 100.0, 250.0, np.inf]


def _fit(spec: str, x: pd.DataFrame, y_bps: pd.Series, atr_bps: pd.Series, timestamps: pd.Series):
    min_child = max(2, math.ceil(len(x) * 0.02))
    if spec == "lgb_l1_bps":
        model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=350, learning_rate=.03, max_depth=4, num_leaves=15, min_child_samples=min_child, subsample=.8, colsample_bytree=.8, reg_lambda=4., random_state=1729, n_jobs=2, verbosity=-1)
        target, kind = y_bps, "bps"
    elif spec == "lgb_huber_bps":
        model = lgb.LGBMRegressor(objective="huber", alpha=.9, n_estimators=350, learning_rate=.03, max_depth=4, num_leaves=15, min_child_samples=min_child, subsample=.8, colsample_bytree=.8, reg_lambda=4., random_state=1729, n_jobs=2, verbosity=-1)
        target, kind = y_bps, "bps"
    elif spec == "lgb_l2_atr":
        model = lgb.LGBMRegressor(objective="regression_l2", n_estimators=350, learning_rate=.03, max_depth=4, num_leaves=15, min_child_samples=min_child, subsample=.8, colsample_bytree=.8, reg_lambda=4., random_state=1729, n_jobs=2, verbosity=-1)
        target, kind = y_bps / atr_bps, "atr"
    elif spec in {"cat_mae_bps", "cat_rmse_bps", "cat_mae_atr"}:
        loss = "RMSE" if spec == "cat_rmse_bps" else "MAE"
        model = CatBoostRegressor(loss_function=loss, iterations=300, depth=4, learning_rate=.04, l2_leaf_reg=5., random_seed=1729, verbose=False, thread_count=2, allow_writing_files=False)
        target, kind = (y_bps / atr_bps, "atr") if spec == "cat_mae_atr" else (y_bps, "bps")
    elif spec == "lgb_ranker_grade":
        model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=350, learning_rate=.03, max_depth=4, num_leaves=15, min_child_samples=min_child, subsample=.8, colsample_bytree=.8, reg_lambda=4., random_state=1729, n_jobs=2, verbosity=-1)
        target = pd.cut(y_bps, bins=GRADE_BINS, labels=False, include_lowest=True).astype(int)
        ordered = pd.DataFrame({"ts": timestamps, "idx": np.arange(len(x))}).sort_values(["ts", "idx"], kind="stable")
        xs = x.iloc[ordered.idx.to_numpy()]
        ys = target.iloc[ordered.idx.to_numpy()]
        groups = ordered.groupby("ts", sort=False).size().to_numpy()
        model.fit(xs, ys, group=groups)
        return model, "ranker"
    else:
        raise ValueError(spec)
    model.fit(x, target)
    return model, kind


def _predict(model, kind: str, x: pd.DataFrame, atr_bps: pd.Series) -> np.ndarray:
    raw = np.asarray(model.predict(x), dtype=float)
    return raw * atr_bps.to_numpy(float) if kind == "atr" else raw


def _binned_calibration(train_score: np.ndarray, train_y: np.ndarray, test_score: np.ndarray) -> np.ndarray:
    """Monotone-free causal decile calibration; only the train fold is used."""
    work = pd.DataFrame({"score": train_score, "y": train_y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 100 or work.score.nunique() < 4:
        return np.full(len(test_score), float(work.y.mean()) if len(work) else np.nan)
    work["bucket"] = pd.qcut(work.score.rank(method="first"), 10, labels=False, duplicates="drop")
    table = work.groupby("bucket", observed=True).agg(score=("score", "median"), y=("y", "mean")).sort_values("score")
    return np.interp(test_score, table.score.to_numpy(float), table.y.to_numpy(float), left=float(table.y.iloc[0]), right=float(table.y.iloc[-1]))


def _read_policy_labels(root: Path) -> pd.DataFrame:
    paths = sorted(Path(root).resolve().glob("policy_parts/symbol=*/policy_labels.parquet"))
    if not paths:
        raise FileNotFoundError(f"No policy label parts under {root}")
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    labels = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    if labels["candidate_id"].duplicated().any():
        raise AssertionError("policy labels contain duplicate candidate identities")
    if not np.isclose(pd.to_numeric(labels.loc[labels.policy_path_valid, "policy_cost_bps"], errors="coerce"), 100.0).all():
        raise AssertionError("policy labels must carry the exact 100-bps cost once")
    return labels


def _load(*, feature_panel: Path, labels_root: Path | None, state_root: Path) -> pd.DataFrame:
    features = pd.read_parquet(feature_panel)
    labels = _read_policy_labels(labels_root) if labels_root is not None else source.load_target_free_scores()[1]
    state = continuation._read_state_panel(state_root)
    state = state.loc[:, ["candidate_id", "entry_price", "signal_atr"]].drop_duplicates("candidate_id")
    panel = features.merge(labels, on="candidate_id", how="left", validate="one_to_one").merge(state, on="candidate_id", how="left", validate="one_to_one")
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True)
    panel["policy_label_available_ts"] = pd.to_datetime(panel["policy_label_available_ts"], utc=True)
    panel["atr_bps"] = pd.to_numeric(panel.signal_atr, errors="coerce") / pd.to_numeric(panel.entry_price, errors="coerce") * 10_000.
    return panel.loc[panel.policy_path_valid.fillna(False) & panel.policy_net_bps.notna() & panel.policy_label_available_ts.notna() & panel.finite_15m_feature_count.ge(50) & panel.atr_bps.gt(0)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-panel", type=Path, default=PANEL)
    parser.add_argument("--labels-root", type=Path, help="Immutable rich-policy label root; defaults to the prior source labels.")
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument("--floor", type=float, action="append", default=[], help="Dual-MC1 floor; repeatable.")
    parser.add_argument("--model-spec", choices=SPECS, action="append", default=[], help="Entry model family; repeatable.")
    parser.add_argument("--interaction", choices=INTERACTIONS, action="append", default=[], help="Frozen entry authority; repeatable.")
    parser.add_argument("--held-start", default="2026-03-01")
    parser.add_argument("--held-end", default="2026-08-01")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    floors = tuple(args.floor) if args.floor else FLOORS
    specs = tuple(args.model_spec) if args.model_spec else SPECS
    interactions = tuple(args.interaction) if args.interaction else INTERACTIONS
    panel = _load(
        feature_panel=args.feature_panel.resolve(),
        labels_root=args.labels_root.resolve() if args.labels_root else None,
        state_root=args.state_root.resolve(),
    )
    results: list[pd.DataFrame] = []; metrics: list[dict[str, object]] = []; processed_held_months: set[str] = set()
    months = pd.date_range(args.held_start, args.held_end, freq="MS", tz="UTC")
    for floor in floors:
        scoped = panel.loc[pd.to_numeric(panel.dual_mc1_min_bps, errors="coerce").ge(floor)].copy()
        for held in months:
            end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=2)
            train = scoped.loc[scoped["__decision_ts__"].ge(start) & scoped["__decision_ts__"].lt(held) & scoped["policy_label_available_ts"].lt(held)].copy()
            test = scoped.loc[scoped["__decision_ts__"].ge(held) & scoped["__decision_ts__"].lt(end)].copy()
            # This runner's OOS contract is two *complete* prior calendar
            # months.  Do not quietly turn the first available fold into a
            # one-month fit when an immutable feature panel begins mid-history.
            required_months = {
                (held - pd.DateOffset(months=1)).strftime("%Y-%m"),
                (held - pd.DateOffset(months=2)).strftime("%Y-%m"),
            }
            observed_months = set(train["__decision_ts__"].dt.strftime("%Y-%m"))
            if not required_months.issubset(observed_months):
                continue
            if len(train) < 200 or test.empty: continue
            processed_held_months.add(held.strftime("%Y-%m"))
            xtr, xte = train.loc[:, FIFTEEN_MINUTE_FEATURE_KEYS], test.loc[:, FIFTEEN_MINUTE_FEATURE_KEYS]
            ytr = pd.to_numeric(train.policy_net_bps, errors="coerce")
            for spec in specs:
                model, kind = _fit(spec, xtr, ytr, train.atr_bps, train["__decision_ts__"])
                train_score = _predict(model, kind, xtr, train.atr_bps)
                test_score = _predict(model, kind, xte, test.atr_bps)
                calibrated = _binned_calibration(train_score, ytr.to_numpy(float), test_score)
                frame = test.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps", "dual_mc1_min_bps", "bcf_final_score"]].copy()
                frame["floor_bps"], frame["held_month"], frame["model_spec"] = floor, held.strftime("%Y-%m"), spec
                frame["head_score"], frame["head_expected_bps"] = test_score, calibrated
                for interaction in interactions:
                    base = pd.to_numeric(frame.dual_mc1_min_bps, errors="coerce").ge(floor)
                    if interaction == "base": selected = base
                    elif interaction == "veto_pred_ge_0": selected = base & frame.head_expected_bps.ge(0.)
                    elif interaction == "veto_pred_ge_25": selected = base & frame.head_expected_bps.ge(25.)
                    else: selected = base & (0.75 * frame.dual_mc1_min_bps + 0.25 * frame.head_expected_bps).ge(floor)
                    chosen = frame.loc[selected]
                    y = pd.to_numeric(chosen.policy_net_bps, errors="coerce")
                    metrics.append({"floor_bps":floor,"held_month":held.strftime("%Y-%m"),"model_spec":spec,"interaction":interaction,"entries":len(chosen),"net_bps_per_trade":y.mean(),"total_net_bps":y.sum(),"positive_rate":(y>0).mean()})
                    frame[f"selected__{interaction}"] = selected.to_numpy(bool)
                results.append(frame)
    predictions = pd.concat(results, ignore_index=True); monthly = pd.DataFrame(metrics)
    aggregate = monthly.groupby(["floor_bps","model_spec","interaction"], as_index=False).agg(held_months=("held_month","nunique"),entries=("entries","sum"),total_net_bps=("total_net_bps","sum"),net_bps_per_trade=("net_bps_per_trade","mean"),worst_month_bps=("net_bps_per_trade","min"),positive_rate=("positive_rate","mean"))
    predictions.to_parquet(out/"walkforward_predictions.parquet",index=False); monthly.to_parquet(out/"monthly_metrics.parquet",index=False); aggregate.to_parquet(out/"aggregate_metrics.parquet",index=False)
    (out/"run_manifest.json").write_text(json.dumps({"schema":"p8u-entry-hpo-v1","scope":"offline strict-OOS only","panel":str(args.feature_panel.resolve()),"labels_root":str(args.labels_root.resolve()) if args.labels_root else "prior_source_labels","state_root":str(args.state_root.resolve()),"fold":"exactly two complete prior calendar months; all labels resolved before held month","floors_bps":floors,"specs":specs,"interactions":interactions,"requested_held_months":[str(month) for month in months],"processed_held_months":sorted(processed_held_months),"models":{"depth":4,"min_leaf_fraction":.02,"lgb_losses":["L1","Huber","L2"],"catboost_losses":["MAE","RMSE"],"ranker":"LambdaRank"},"entry_authority":"veto/blend only; cannot promote below named dual-MC1 floor"},indent=2)+"\n")

if __name__ == "__main__": main()
