#!/usr/bin/env python3
"""Round B0--B3 robustness ablation for one shared TP6 residual expert.

This is deliberately the second round of the sequential funnel, not a
factorial search.  The target remains the current exact-net residual around
the prequential R3 payoff map.  The feature contract is a frozen invariant
core: base simplex, prequential expected bps, side flag, and the fourteen
high-coverage causal market-context fields.  It excludes target columns,
``base_raw`` (whose units differ between source ledgers), soft-regime fields,
and every form of local/per-regime model routing.

B0  fixed pooled Huber model, uniform rows.
B1  same fixed model, square-root training-era weights.
B2  uniform rows, tiny predeclared model-grid selected only on earlier
    chronological held-out eras with a worst-era robust control.
B3  B2 selection plus B1 weights.

For every outer era the candidate grid is selected *without touching that
era*.  The robust control is ``mean + .5 * worst - .25 * std`` of prior
global top-1% net bps.  (The last term is a dispersion penalty; writing it
with a negative sign prevents volatile results from being rewarded.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_shared_residual_b0_b3_20260809_v1"
ERAS = ("2023-07_08", "2023-09_10", "2023-11_12", "2024-01_02", "2024-05_06", "2024-07_08", "2024-09_10", "2024-11")
BASE = ("p_adverse", "p_weak", "p_clear")
CONTEXT = (
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
    "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
    "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
    "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market",
    "deleveraging_without_followthrough", "short_signal_recovery_conflict",
)
FEATURES = ("side_is_long", *BASE, "prequential_base_expected_net_bps", *CONTEXT)
TOPS = (.01, .05, .10)
SEED = 20260809
LABEL_AVAILABILITY_DELAY = pd.Timedelta(hours=13)  # signal close +1h entry + H12

# Predeclared tiny grid.  It is intentionally a capacity/regularisation check,
# not an HPO sweep.  All choices are available in every eligible outer fold.
GRID: dict[str, dict[str, Any]] = {
    "reference": dict(n_estimators=180, learning_rate=.035, num_leaves=24, min_child_samples=400, colsample_bytree=.80, subsample=.80, reg_lambda=12.),
    "conservative": dict(n_estimators=140, learning_rate=.030, num_leaves=12, min_child_samples=800, colsample_bytree=.80, subsample=.80, reg_lambda=20.),
    "moderate": dict(n_estimators=160, learning_rate=.030, num_leaves=16, min_child_samples=600, colsample_bytree=.80, subsample=.80, reg_lambda=16.),
}
ARMS = {
    "B0_ordinary_pooled": {"weight": "uniform", "selection": "fixed"},
    "B1_sqrt_era_balance": {"weight": "sqrt_era", "selection": "fixed"},
    "B2_worst_era_selection": {"weight": "uniform", "selection": "robust"},
    "B3_balance_plus_selection": {"weight": "sqrt_era", "selection": "robust"},
}


def ensure_label_available_ts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "label_available_ts" in out:
        available = pd.to_datetime(out["label_available_ts"], utc=True, errors="raise")
    elif "__label_available_at__" in out:
        available = pd.to_datetime(out["__label_available_at__"], utc=True, errors="raise")
    else:
        available = pd.to_datetime(out["__ts__"], utc=True) + LABEL_AVAILABILITY_DELAY
    if available.isna().any() or (available < pd.to_datetime(out["__ts__"], utc=True) + LABEL_AVAILABILITY_DELAY).any():
        raise ValueError("H12 label availability must be signal-close +13h or later")
    out["label_available_ts"] = available
    return out


def assert_outer_train_resolved(train: pd.DataFrame, test: pd.DataFrame) -> None:
    cutoff = pd.to_datetime(test["__ts__"], utc=True).min()
    latest = pd.to_datetime(train["label_available_ts"], utc=True).max()
    if latest >= cutoff:
        raise ValueError(f"outer training contains unresolved labels: {latest} >= {cutoff}")


def fit_model(params: dict[str, Any], x: pd.DataFrame, y: pd.Series, weight: np.ndarray) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber", alpha=.9, random_state=SEED, n_jobs=1, verbosity=-1,
        **params,
    ).fit(matrix(x), y.to_numpy(float), sample_weight=weight)


def matrix(x: pd.DataFrame) -> np.ndarray:
    value = x.loc[:, FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("non-finite invariant-core value reached model matrix")
    return value


def weight_for(train: pd.DataFrame, profile: str) -> np.ndarray:
    if profile == "uniform":
        return np.ones(len(train), dtype=float)
    if profile != "sqrt_era":
        raise ValueError(profile)
    count = train.groupby("era").size()
    # sqrt(N_total / (K N_e)); then mean one and cap as preregistered.
    value = train.era.map(np.sqrt(len(train) / (len(count) * count))).to_numpy(float)
    return np.clip(value / value.mean(), .25, 4.)


def global_top_net(frame: pd.DataFrame, score: np.ndarray, top: float = .01) -> float:
    order = np.lexsort((frame.candidate_id.to_numpy(str), -np.asarray(score, float)))
    take = frame.iloc[order[:max(1, int(np.ceil(len(frame) * top)))]].net_bps
    return float(take.mean())


def robust_control(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean_top1_net_bps": float(a.mean()),
        "worst_top1_net_bps": float(a.min()),
        "std_top1_net_bps": float(a.std(ddof=0)),
        "robust_control": float(a.mean() + .5 * a.min() - .25 * a.std(ddof=0)),
    }


def metric_rows(frame: pd.DataFrame, score: np.ndarray, common: dict[str, Any], period: str, scope: str) -> list[dict[str, Any]]:
    z = frame.copy()
    z["score_bps"] = score
    rows: list[dict[str, Any]] = []
    for view, q in (("global", z), ("long", z[z.side_name.eq("long")]), ("short", z[z.side_name.eq("short")])):
        ic = spearmanr(q.score_bps, q.net_bps).statistic if len(q) > 1 else np.nan
        for top in TOPS:
            take = q.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort").head(max(1, int(np.ceil(len(q) * top))))
            rows.append({
                **common, "scope": scope, "period": period, "view": view, "top_fraction": top,
                "n": len(take), "net_bps": float(take.net_bps.mean()), "gross_bps": float(take.gross_bps.mean()),
                "all_rows_net_bps": float(q.net_bps.mean()), "score_net_spearman": float(ic),
                "selected_long_fraction": float(take.side_name.eq("long").mean()),
                "positive_net_fraction": float(take.net_bps.gt(0).mean()),
                "clear_50_fraction": float(take.net_bps.gt(50).mean()),
            })
    return rows


def pick_params(data: pd.DataFrame, outer_index: int, profile: str, cache_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    """Use only complete *earlier* rolling held-out eras for grid selection."""
    # No validation exists for the first outer era. One cell is too little to
    # claim worst-era robustness, so retain reference until two inner cells.
    if outer_index < 3:
        return "reference", [{"candidate": "reference", "reason": "insufficient_prior_heldout_eras", "inner_cells": outer_index - 1}]
    records: list[dict[str, Any]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, params in GRID.items():
        values: list[float] = []
        for j in range(1, outer_index):
            cache = cache_dir / f"{profile}_{name}_{ERAS[j]}.json"
            if cache.exists():
                values.append(float(json.loads(cache.read_text())["top1_net_bps"]))
                continue
            train = data[data.era.isin(ERAS[:j])]
            test = data[data.era.eq(ERAS[j])]
            assert_outer_train_resolved(train, test)
            target = train.net_bps - train.prequential_base_expected_net_bps
            model = fit_model(params, train, target, weight_for(train, profile))
            score = test.prequential_base_expected_net_bps.to_numpy(float) + model.predict(matrix(test))
            value = global_top_net(test, score)
            cache.write_text(json.dumps({"profile": profile, "candidate": name, "test_era": ERAS[j], "top1_net_bps": value, "train_eras": list(ERAS[:j])}) + "\n")
            values.append(value)
        record: dict[str, Any] = {"candidate": name, "inner_cells": len(values), **robust_control(values)}
        records.append(record)
    # deterministic tie break favours more conservative capacity then name.
    rank = {"conservative": 0, "moderate": 1, "reference": 2}
    selected = max(records, key=lambda r: (r["robust_control"], r["worst_top1_net_bps"], r["mean_top1_net_bps"], -rank[r["candidate"]]))
    return str(selected["candidate"]), records


def validate(data: pd.DataFrame) -> None:
    missing = set(FEATURES).difference(data.columns)
    if missing:
        raise ValueError(f"missing invariant core: {sorted(missing)}")
    if data.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique")
    if not np.allclose(data.gross_bps.to_numpy(float) - data.net_bps.to_numpy(float), 100., atol=.02):
        raise ValueError("fixed 100-bps cost contract failed")
    if not set(data.era.unique()).issubset(ERAS):
        raise ValueError("unexpected era")
    if data[["p_adverse", "p_weak", "p_clear"]].isna().any().any():
        raise ValueError("base simplex missing on declared contract")
    if not np.allclose(data[["p_adverse", "p_weak", "p_clear"]].sum(axis=1), 1., atol=1e-4):
        raise ValueError("base simplex violates sum-to-one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only-arm", choices=tuple(ARMS))
    parser.add_argument("--only-era", choices=ERAS[1:])
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    stage = args.out.with_name(args.out.name + "_stage")
    if args.finalize:
        checkpoints = sorted((stage / "checkpoints").glob("*.parquet"))
        selections = sorted((stage / "selections").glob("*.parquet"))
        if not checkpoints:
            raise FileNotFoundError("no checkpoint files")
        metrics = pd.concat([pd.read_parquet(p) for p in checkpoints], ignore_index=True).drop_duplicates(
            ["arm", "test_era", "scope", "period", "view", "top_fraction"], keep="last")
        expected = {(a, e) for a in (args.only_arm and (args.only_arm,) or tuple(ARMS)) for e in (args.only_era and (args.only_era,) or ERAS[1:])}
        got = set(metrics.loc[metrics.scope.eq("era"), ["arm", "test_era"]].itertuples(index=False, name=None))
        if expected - got:
            raise ValueError(f"missing outer metrics: {sorted(expected-got)}")
        selected = pd.concat([pd.read_parquet(p) for p in selections], ignore_index=True) if selections else pd.DataFrame()
        outer = metrics[(metrics.scope.eq("era")) & (metrics.view.eq("global")) & (metrics.top_fraction.eq(.01))]
        summary_rows: list[dict[str, Any]] = []
        for arm, q in outer.groupby("arm", sort=True):
            summary_rows.append({"arm": arm, "eras": int(q.test_era.nunique()), "positive_eras": int(q.net_bps.gt(0).sum()), **robust_control(q.net_bps.tolist())})
        summary = pd.DataFrame(summary_rows).sort_values("robust_control", ascending=False, kind="stable")
        # pooled-global, not timestamp-local: join all strict outer predictions
        # through their metrics is impossible, so compute from saved prediction files.
        pooled_rows: list[dict[str, Any]] = []
        for arm in ARMS:
            pp = sorted((stage / "predictions").glob(f"*_{arm}.parquet"))
            if not pp:
                continue
            pred = pd.concat([pd.read_parquet(p) for p in pp], ignore_index=True)
            pooled_rows.extend(metric_rows(pred, pred.score_bps.to_numpy(float), {"arm": arm, "test_era": "ALL_OUTER", "train_through": "rolling", "model_name": "per_outer_selected"}, "ALL_OUTER", "pooled_global"))
        args.out.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(args.out / "metrics.parquet", index=False)
        pd.DataFrame(pooled_rows).to_parquet(args.out / "pooled_global_metrics.parquet", index=False)
        summary.to_parquet(args.out / "summary.parquet", index=False)
        selected.to_parquet(args.out / "model_selection.parquet", index=False)
        report = ["# B0–B3 shared residual robustness ablation", "", "One shared Huber residual expert; no local/per-regime experts or target changes.", "", "## Era-level global top-1% robust control", "", "| arm | eras | positive eras | mean net bps | worst net bps | std | robust control |", "|---|---:|---:|---:|---:|---:|---:|"]
        for r in summary.itertuples(index=False):
            report.append(f"| {r.arm} | {r.eras} | {r.positive_eras} | {r.mean_top1_net_bps:.3f} | {r.worst_top1_net_bps:.3f} | {r.std_top1_net_bps:.3f} | {r.robust_control:.3f} |")
        report += ["", "Robust control = mean top-1% net + 0.5 × worst-era top-1% net − 0.25 × standard deviation. B2/B3 selection uses only earlier rolling held-out eras; first two outer cells retain the predeclared reference model because fewer than two inner cells cannot establish a worst-era criterion.", "", "All detailed per-era, per-month, per-side, top-1/5/10%, score-IC, gross/net, side mix, and pooled-global economics are in Parquet artifacts."]
        (args.out / "REPORT.md").write_text("\n".join(report) + "\n")
        manifest = {"schema": "tp6_shared_residual_b0_b3_v1", "status": "COMPLETED_DIAGNOSTIC_NO_PROMOTION", "input": str(INPUT), "contract": {"geometry": "TP6/SL4/H12", "cost_bps": 100., "target": "net_bps - prequential_base_expected_net_bps", "expert": "one shared Huber LGBM", "ranking": "pooled global per held-out era; pooled global aggregate", "no_local_or_regime_experts": True, "no_target_change": True}, "features": list(FEATURES), "excluded": ["base_raw_unit_incompatible", "target__*", "all soft/hard regime fields", "local/per-regime routing"], "arms": ARMS, "model_grid": GRID, "selection": "mean + 0.5*worst - 0.25*std, prior held-out eras only", "selected_robust_control": summary.iloc[0].to_dict() if len(summary) else None}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(json.dumps(summary.to_dict(orient="records"), indent=2))
        return

    if not args.only_era:
        raise ValueError("pass --only-era for a resumable strict outer checkpoint, then --finalize")
    x = pd.read_parquet(INPUT)
    # Same complete substrate across B0--B3, with no zero imputation.  The
    # shared-regime availability flag is retained as the fixed common cohort
    # because later C/D rounds must not change candidate support.
    x = x[x.shared_regime_contract_complete.astype(bool)].copy()
    x["side_is_long"] = x.side_name.eq("long").astype(float)
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x = ensure_label_available_ts(x)
    # The initial July reference build intentionally has no prior payoff map
    # for its earliest rows.  It cannot define a residual around expected bps,
    # so exclude it rather than inventing a zero baseline.  This finite cohort
    # is fixed once, before comparing any B arm.
    x = x[np.isfinite(x.loc[:, FEATURES].to_numpy(float)).all(axis=1)].copy()
    validate(x)
    x = x.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    outer_index = ERAS.index(args.only_era)
    train = x[x.era.isin(ERAS[:outer_index])].copy()
    test = x[x.era.eq(args.only_era)].copy()
    if train.empty or test.empty:
        raise ValueError("empty rolling outer split")
    assert_outer_train_resolved(train, test)
    arms = (args.only_arm,) if args.only_arm else tuple(ARMS)
    rows: list[dict[str, Any]] = []
    sel_rows: list[dict[str, Any]] = []
    for arm in arms:
        definition = ARMS[arm]
        if definition["selection"] == "fixed":
            selected, detail = "reference", [{"candidate": "reference", "reason": "fixed_control", "inner_cells": 0}]
        else:
            selected, detail = pick_params(x, outer_index, definition["weight"], stage / "selection_cache")
        for entry in detail:
            sel_rows.append({"arm": arm, "test_era": args.only_era, "train_through": ERAS[outer_index - 1], "selected": selected, "weight_profile": definition["weight"], **entry})
        y = train.net_bps - train.prequential_base_expected_net_bps
        model = fit_model(GRID[selected], train, y, weight_for(train, definition["weight"]))
        score = test.prequential_base_expected_net_bps.to_numpy(float) + model.predict(matrix(test))
        common = {"arm": arm, "test_era": args.only_era, "train_through": ERAS[outer_index - 1], "train_rows": len(train), "test_rows": len(test), "model_name": selected, "weight_profile": definition["weight"]}
        rows.extend(metric_rows(test, score, common, args.only_era, "era"))
        for month, q in test.assign(__month__=test.__ts__.dt.strftime("%Y-%m")).groupby("__month__", sort=True):
            rows.extend(metric_rows(q, score[q.index.to_numpy() - test.index.min()], common, month, "month"))
        pred = test[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "era"]].copy()
        pred["score_bps"] = score
        pred["arm"] = arm
        pred["model_name"] = selected
        pred["weight_profile"] = definition["weight"]
        outdir = stage / "predictions"; outdir.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(outdir / f"{args.only_era}_{arm}.parquet", index=False)
    checkpoint = stage / "checkpoints"; checkpoint.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(checkpoint / f"{args.only_era}{'_' + args.only_arm if args.only_arm else ''}.parquet", index=False)
    selections = stage / "selections"; selections.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sel_rows).to_parquet(selections / f"{args.only_era}{'_' + args.only_arm if args.only_arm else ''}.parquet", index=False)
    print(json.dumps({"test_era": args.only_era, "arms": list(arms), "train_rows": len(train), "test_rows": len(test)}, indent=2))


if __name__ == "__main__":
    main()
