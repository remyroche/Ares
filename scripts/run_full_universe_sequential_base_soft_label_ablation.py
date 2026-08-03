#!/usr/bin/env python3
"""Sequential TP6/SL4 H12 base-target ablation, without meta features.

The target/evaluation contract is the exact TP6/SL4 sidecar, not the older
TP3/SL2 panel labels.  Only candidate IDs with a complete exact H12 sidecar
row are admitted to either fitting or scoring.  The incumbent per-side *base*
feature sets remain frozen; path/economic sidecar fields are labels only.

Round 1 is intentionally a sequential funnel, not a factorial search:

* B0 exact first-touch control vs B1 distance-softness tau grid;
* the R1 winner vs B2 raw path-membership tau grid; then
* the R2 winner vs B3 hard-floor/promptness grid around the selected B1 tau.

Each round uses a later, non-overlapping development span and training labels
strictly resolved before that span.  After R3, the frozen winner is refit on
all rows resolved before OOS and replayed once on the untouched OOS period.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extreme_price_movements.tp6_sl4_target_weights import (
    TP6SL4Columns,
    TargetParameters,
    WeightParameters,
    assert_simplex,
    build_target,
    build_weight,
    target_manifest,
)


TOPS = (0.01, 0.05, 0.10)
TAUS = (.10, .25, .50)


@dataclass(frozen=True)
class Stage:
    name: str
    start: str
    end: str


STAGES = (
    Stage("R1_B1_tau", "2024-04-15", "2024-05-15"),
    Stage("R2_B2_membership", "2024-05-15", "2024-06-15"),
    Stage("R3_B3_floor_time", "2024-06-15", "2024-08-01"),
    Stage("R4_magnitude", "2024-08-01", "2024-09-01"),
)


def _read_parts(root: Path, columns: list[str]) -> pd.DataFrame:
    files = sorted((root / "parts").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet parts beneath {root}")
    return pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)


def _feature_contract(base: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for side in ("long", "short"):
        manifest = json.loads((base / side / "target_family_manifest.json").read_text())
        key = f"T2_soft_barrier|tp3_sl2|{side}"
        cols = manifest.get("feature_contract", {}).get(key, [])
        if len(cols) < 30:
            raise ValueError(f"frozen {side} base feature contract missing")
        result[side] = list(cols)
    return result


def _target_columns() -> TP6SL4Columns:
    # The materialised exact sidecar calls the H12 close value
    # `t4_tp6_sl4_terminal_pnl_atr`; the target primitive accepts an override.
    return TP6SL4Columns(terminal_atr="t4_tp6_sl4_terminal_pnl_atr")


def _raw_membership(frame: pd.DataFrame, tau: float) -> np.ndarray:
    """Construct the declared B2 stability/conflict memberships.

    Exact intra-bar path ordering is unavailable in the historical panel, so
    a row whose realised path crosses both barriers is the explicit
    upper/lower ambiguity proxy.  A timeout that approached the upper barrier
    more closely than the lower is the upper/timeout ambiguity proxy.  This
    is a resolved training label only, never a feature.
    """
    mfe = frame.t2_path_mfe_atr.to_numpy(float)
    mae = frame.t2_path_mae_atr.to_numpy(float)
    event = frame.t2_tp6_sl4_event.to_numpy(int)
    out = np.full((len(frame), 3), .025, dtype=float)
    # Stable exact events retain the proposed 95% winning-class mass.
    out[np.arange(len(frame)), event] = .95
    crossed_upper = mfe >= 6.
    crossed_lower = mae >= 4.
    ambiguous_ul = crossed_upper & crossed_lower
    # [0.55, 0.40, 0.05] in side-normalised (upper, lower, timeout) space.
    out[ambiguous_ul] = np.array([.55, .40, .05])
    timeout = event == 2
    upper_closer = (6. - mfe) <= (4. - mae)
    ambiguous_ut = timeout & ~ambiguous_ul & upper_closer
    out[ambiguous_ut] = np.array([.60, .05, .35])
    assert_simplex(out)
    return out


def _prepare(panel: Path, sidecar: Path, features: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_union = list(dict.fromkeys([*features["long"], *features["short"]]))
    panel_cols = ["candidate_id", "__ts__", "side_name", "t2_path_mfe_atr", "t2_path_mae_atr", *feature_union]
    sidecar_cols = ["candidate_id", "__label_available_at__", "t2_tp6_sl4_event", "t2_tp6_sl4_exit_minute", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "t4_tp6_sl4_terminal_pnl_atr"]
    base = _read_parts(panel, panel_cols)
    labels = _read_parts(sidecar, sidecar_cols)
    if base.candidate_id.duplicated().any() or labels.candidate_id.duplicated().any():
        raise ValueError("candidate identity must be unique in panel and exact sidecar")
    merged = base.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("no complete TP6/SL4 labels joined")
    merged["__ts__"] = pd.to_datetime(merged["__ts__"], utc=True, errors="raise")
    merged["__label_available_at__"] = pd.to_datetime(merged["__label_available_at__"], utc=True, errors="raise")
    if not merged.__label_available_at__.gt(merged.__ts__).all():
        raise ValueError("exact TP6/SL4 labels must become available after decision")
    numeric = ["t2_path_mfe_atr", "t2_path_mae_atr", "t2_tp6_sl4_exit_minute", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "t4_tp6_sl4_terminal_pnl_atr"]
    if not np.isfinite(merged.loc[:, numeric].to_numpy(float)).all():
        raise ValueError("complete TP6/SL4 join contains non-finite target fields")
    if not np.isin(merged.t2_tp6_sl4_event.to_numpy(int), (0, 1, 2)).all():
        raise ValueError("TP6/SL4 event convention is invalid")
    if not np.allclose(merged.t4_tp6_sl4_gross_bps.to_numpy(float) - 100., merged.t4_tp6_sl4_net_bps.to_numpy(float), rtol=0., atol=2e-3):
        raise ValueError("TP6/SL4 gross/net cost parity fails")
    coverage = {"panel_rows": int(len(base)), "complete_exact_tp6_sl4_rows": int(len(labels)), "joined_complete_rows": int(len(merged)), "join_fraction_of_panel": float(len(merged) / len(base)), "rows_by_side": merged.side_name.value_counts().to_dict()}
    return merged, coverage


def _arm(name: str, *, tau: float = .25, hard_floor: float = .75, time_decay_hours: float = 4.) -> dict[str, Any]:
    if name not in ("B0", "B1", "B2", "B3"):
        raise ValueError(name)
    parameters = TargetParameters(distance_tau_atr=float(tau), hard_floor=float(hard_floor), time_decay_hours=float(time_decay_hours))
    return {"name": name, "tau": float(tau), "hard_floor": float(hard_floor), "time_decay_hours": float(time_decay_hours), "weight": "BW0", "target_parameters": parameters}


def _label_and_weight(train: pd.DataFrame, arm: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cols = _target_columns()
    # B2's explicit raw-membership fields are constructed from terminal path
    # labels at fit time, then passed to the shared target primitive.
    # B0/B1/B3 only read resolved labels.  Copying the full multi-million-row
    # feature frame for each arm is unnecessary and can exhaust memory.
    work = train
    if arm["name"] == "B2":
        membership = _raw_membership(train, arm["tau"])
        work = pd.DataFrame({
            cols.event: train[cols.event].to_numpy(),
            cols.upper_membership: membership[:, 0],
            cols.lower_membership: membership[:, 1],
            cols.timeout_membership: membership[:, 2],
        })
    target = build_target(work, arm["name"], columns=cols, parameters=arm["target_parameters"])
    weight = build_weight(work, arm["weight"], columns=cols, target=target, target_parameters=arm["target_parameters"], parameters=WeightParameters())
    assert_simplex(target)
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("TP6/SL4 sample weights must be positive")
    lineage = dict(target_manifest(arm["name"], arm["weight"], columns=cols, target_parameters=arm["target_parameters"], weight_parameters=WeightParameters()))
    if arm["name"] == "B2": lineage["B2_membership"] = "stable/conflict memberships: stable event 0.95; crossed-both barrier proxy 0.55/0.40/0.05; upper-close timeout proxy 0.60/0.05/0.35"
    return target, weight, lineage


def _matrix(frame: pd.DataFrame, fields: list[str]) -> np.ndarray:
    return frame.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(np.float32)


def _fit_score(train: pd.DataFrame, evaluation: pd.DataFrame, *, arm: dict[str, Any], features: dict[str, list[str]], seed: int) -> np.ndarray:
    target, weight, _ = _label_and_weight(train, arm)
    score = np.full(len(evaluation), np.nan, dtype=np.float64)
    for side_number, side in enumerate(("long", "short")):
        trpos = np.flatnonzero(train.side_name.eq(side).to_numpy())
        evpos = np.flatnonzero(evaluation.side_name.eq(side).to_numpy())
        if len(trpos) < 10_000 or not len(evpos):
            raise ValueError(f"insufficient {side} complete target support")
        xtr, xev = _matrix(train.iloc[trpos], features[side]), _matrix(evaluation.iloc[evpos], features[side])
        probabilities = []
        for klass in range(3):
            model = lgb.LGBMRegressor(objective="huber", alpha=.90, n_estimators=50, learning_rate=.05, num_leaves=24, min_child_samples=400, colsample_bytree=.80, subsample=.80, reg_lambda=8., random_state=seed + side_number * 10 + klass, n_jobs=1, verbosity=-1).fit(xtr, target[trpos, klass], sample_weight=weight[trpos])
            probabilities.append(np.maximum(model.predict(xev), 0.))
        p = np.column_stack(probabilities); p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
        # Fit-only soft-target conditional payoff conversion uses the exact
        # TP6/SL4 net outcome. No evaluation label enters this score.
        net = train.iloc[trpos].t4_tp6_sl4_net_bps.to_numpy(float)
        means = (target[trpos] * net[:, None] * weight[trpos, None]).sum(axis=0) / np.maximum((target[trpos] * weight[trpos, None]).sum(axis=0), 1.)
        score[evpos] = p @ means
    if not np.isfinite(score).all(): raise ValueError("complete joined evaluation rows require a finite base score")
    return score


def _metrics(frame: pd.DataFrame, score: np.ndarray, scope: str) -> list[dict[str, Any]]:
    ranked = frame.assign(__score__=score).sort_values(["__score__", "candidate_id"], ascending=[False, True], kind="mergesort")
    rows = []
    for top in TOPS:
        x = ranked.iloc[:int(np.ceil(len(ranked) * top))]
        rows.append({"scope": scope, "top_fraction": top, "n": len(x), "gross_bps": float(x.t4_tp6_sl4_gross_bps.mean()), "net_bps": float(x.t4_tp6_sl4_net_bps.mean()), "long_n": int(x.side_name.eq("long").sum()), "short_n": int(x.side_name.eq("short").sum())})
    return rows


def _objective(metrics: list[dict[str, Any]]) -> float:
    lookup = {float(row["top_fraction"]): float(row["net_bps"]) for row in metrics}
    return .70 * lookup[.10] + .30 * lookup[.05]


def _evaluate_arms(train: pd.DataFrame, evaluation: pd.DataFrame, arms: list[dict[str, Any]], *, scope: str, features: dict[str, list[str]], seed: int) -> list[dict[str, Any]]:
    rows = []
    for number, arm in enumerate(arms):
        score = _fit_score(train, evaluation, arm=arm, features=features, seed=seed + 100 * number)
        metrics = _metrics(evaluation, score, scope)
        _, _, lineage = _label_and_weight(train, arm)
        rows.append({"arm": {key: value for key, value in arm.items() if key != "target_parameters"}, "lineage": lineage, "objective": _objective(metrics), "metrics": metrics, "score": score})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    p.add_argument("--tp6-sl4-sidecar", type=Path, default=ROOT / "data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1")
    p.add_argument("--base-contract", type=Path, default=ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1")
    p.add_argument("--oos-start", default="2024-08-01")
    p.add_argument("--oos-end", default="2024-12-01")
    p.add_argument("--out", type=Path, default=ROOT / "data_perp/artifacts/full_universe_sequential_tp6_sl4_base_target_ablation_20260804_v1")
    p.add_argument("--seed", type=int, default=20260804)
    p.add_argument("--run-one", help="Run just one arm as NAME,STAGE[,TAU,FLOOR,LAMBDA]; emits a compact development result. This permits resource-bounded sequential execution.")
    a = p.parse_args()
    features = _feature_contract(a.base_contract)
    frame, coverage = _prepare(a.panel, a.tp6_sl4_sidecar, features)
    if a.run_one:
        values = [item.strip() for item in a.run_one.split(",")]
        if len(values) not in (2, 5):
            raise ValueError("--run-one must be NAME,STAGE or NAME,STAGE,TAU,FLOOR,LAMBDA")
        name, stage_name = values[:2]
        stages = {stage.name: stage for stage in STAGES}
        if name not in ("B0", "B1", "B2", "B3") or stage_name not in stages:
            raise ValueError("invalid --run-one target or stage")
        tau, floor, decay = (.25, .75, 4.) if len(values) == 2 else tuple(map(float, values[2:]))
        arm = _arm(name, tau=tau, hard_floor=floor, time_decay_hours=decay)
        stage = stages[stage_name]; start, end = pd.Timestamp(stage.start, tz="UTC"), pd.Timestamp(stage.end, tz="UTC")
        train = frame.loc[frame.__label_available_at__.lt(start)]
        ev = frame.loc[frame.__ts__.ge(start) & frame.__ts__.lt(end)]
        result = _evaluate_arms(train, ev, [arm], scope=stage.name, features=features, seed=a.seed)[0]
        a.out.mkdir(parents=True, exist_ok=False)
        payload = {"schema": "tp6_sl4_one_arm_development_v1", "coverage": coverage, "stage": stage.__dict__, "train_rows": len(train), "evaluation_rows": len(ev), "candidate": {key: value for key, value in result.items() if key != "score"}}
        (a.out / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))
        print(json.dumps(payload["candidate"], indent=2, default=str))
        return
    records: list[dict[str, Any]] = []
    carried = _arm("B0")
    # R1: B0 remains a control; B1 is selected jointly across a small tau grid.
    s1 = STAGES[0]; start, end = pd.Timestamp(s1.start, tz="UTC"), pd.Timestamp(s1.end, tz="UTC")
    train, ev = frame.loc[frame.__label_available_at__.lt(start)], frame.loc[frame.__ts__.ge(start) & frame.__ts__.lt(end)]
    r1 = _evaluate_arms(train, ev, [carried, *[_arm("B1", tau=tau) for tau in TAUS]], scope=s1.name, features=features, seed=a.seed)
    winner = sorted(r1, key=lambda x: (-x["objective"], x["arm"]["name"], x["arm"]["tau"]))[0]; carried = _arm(**{k: v for k, v in winner["arm"].items() if k != "weight"})
    records.append({"stage": s1.__dict__, "train_rows": len(train), "evaluation_rows": len(ev), "incoming": "B0", "candidates": [{k:v for k,v in row.items() if k != "score"} for row in r1], "winner": winner["arm"]})
    # R2: B2 receives raw distance memberships, intentionally distinct from B1.
    s2 = STAGES[1]; start, end = pd.Timestamp(s2.start, tz="UTC"), pd.Timestamp(s2.end, tz="UTC")
    train, ev = frame.loc[frame.__label_available_at__.lt(start)], frame.loc[frame.__ts__.ge(start) & frame.__ts__.lt(end)]
    r2 = _evaluate_arms(train, ev, [carried, *[_arm("B2", tau=tau) for tau in TAUS]], scope=s2.name, features=features, seed=a.seed + 10_000)
    winner = sorted(r2, key=lambda x: (-x["objective"], x["arm"]["name"], x["arm"]["tau"]))[0]; carried = _arm(**{k: v for k, v in winner["arm"].items() if k != "weight"})
    records.append({"stage": s2.__dict__, "train_rows": len(train), "evaluation_rows": len(ev), "incoming": records[-1]["winner"], "candidates": [{k:v for k,v in row.items() if k != "score"} for row in r2], "winner": winner["arm"]})
    # R3: B3 anchors back to the R1 B1 tau, then searches its declared
    # hard-first-touch and promptness parameters, not side-specific settings.
    s3 = STAGES[2]; start, end = pd.Timestamp(s3.start, tz="UTC"), pd.Timestamp(s3.end, tz="UTC")
    train, ev = frame.loc[frame.__label_available_at__.lt(start)], frame.loc[frame.__ts__.ge(start) & frame.__ts__.lt(end)]
    b1_tau = float(records[0]["winner"]["tau"]) if records[0]["winner"]["name"] == "B1" else .25
    b3 = [_arm("B3", tau=b1_tau, hard_floor=floor, time_decay_hours=decay) for floor in (.60, .75) for decay in (2., 4., 8.)]
    r3 = _evaluate_arms(train, ev, [carried, *b3], scope=s3.name, features=features, seed=a.seed + 20_000)
    winner = sorted(r3, key=lambda x: (-x["objective"], x["arm"]["name"], x["arm"]["hard_floor"], x["arm"]["time_decay_hours"]))[0]; carried = _arm(**{k: v for k, v in winner["arm"].items() if k != "weight"})
    records.append({"stage": s3.__dict__, "train_rows": len(train), "evaluation_rows": len(ev), "incoming": records[-1]["winner"], "candidates": [{k:v for k,v in row.items() if k != "score"} for row in r3], "winner": winner["arm"]})
    oos_start, oos_end = pd.Timestamp(a.oos_start, tz="UTC"), pd.Timestamp(a.oos_end, tz="UTC")
    train, oos = frame.loc[frame.__label_available_at__.lt(oos_start)], frame.loc[frame.__ts__.ge(oos_start) & frame.__ts__.lt(oos_end)]
    final = _evaluate_arms(train, oos, [carried], scope="untouched_oos", features=features, seed=a.seed + 30_000)[0]
    a.out.mkdir(parents=True, exist_ok=True)
    oos.loc[:, ["candidate_id", "__ts__", "side_name", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps"]].assign(score_bps=final["score"], selected_target=carried["name"]).to_parquet(a.out / "oos_predictions.parquet", index=False)
    pd.DataFrame(final["metrics"]).to_parquet(a.out / "oos_global_tp6_sl4_metrics.parquet", index=False)
    manifest = {"schema": "full_universe_sequential_tp6_sl4_base_target_ablation_v1", "status": "COMPLETED_UNTOUCHED_OOS_REPLAY", "coverage": coverage, "contract": {"geometry": "TP6/SL4, H12, exact next-minute entry; adverse same-minute conflict precedence", "population": "only exact complete sidecar candidate IDs, same complete population for each arm", "features": "frozen per-side base-only lists; no meta/outcome/path target fields", "label_availability": "sidecar __label_available_at__ must be strictly before each training boundary", "selection": "sequential R1 B0/B1-tau, R2 B2-tau, R3 B3 hard-floor/promptness; global top5/top10 net objective", "evaluation": "one pooled global complete-candidate book; global top1/5/10 TP6 gross/net", "weights": "every arm calls tp6_sl4_target_weights build_target + build_weight(BW0); weights are training-only"}, "stages": records, "final_winner": {k:v for k,v in carried.items() if k != "target_parameters"}, "final_train_rows": len(train), "oos_rows": len(oos), "oos_metrics": final["metrics"], "final_lineage": final["lineage"]}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"final_winner": manifest["final_winner"], "oos_metrics": final["metrics"], "complete_coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
