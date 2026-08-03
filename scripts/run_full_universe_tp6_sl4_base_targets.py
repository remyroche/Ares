#!/usr/bin/env python3
"""TP6/SL4 H12 base-only B0/B1 target ablation runner.

This runner deliberately has no feature selection, HPO, meta model, payoff
map, portfolio constraint, or side quota.  It uses the frozen 36-feature
long/short base contracts from the existing TP3/SL2 base artifact and trains
only a small side-local three-output event model.

Targets:

* B0: exact hard TP/SL/timeout event simplex;
* B1: literal TP6/SL4 soft barrier simplex.  The distance terms are based on
  full-H12 MFE/MAE from the original panel and *6 ATR / 4 ATR* barriers, never
  the old TP3/SL2 geometry.  A small predeclared tau grid is exposed.

All training rows must have ``__label_available_at__ < train_end``.  Scores
from long and short models are then pooled in one global top-k book.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIDES = ("long", "short")
TP, SL, HORIZON = 6.0, 4.0, 720.0
TOPS = (.01, .05, .10)
PARAMS = dict(
    n_estimators=80, learning_rate=.06, num_leaves=24, min_child_samples=400,
    colsample_bytree=.80, subsample=.80, reg_lambda=8., random_state=20260809,
    n_jobs=1, verbosity=-1,
)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    p.add_argument("--sidecar", type=Path, required=True, help="output directory from materialize_full_universe_tp6_sl4_h12_sidecar.py")
    p.add_argument("--frozen-base", type=Path, default=ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-end", default="2024-04-01")
    p.add_argument("--eval-end", default="2024-08-01")
    p.add_argument("--taus", default="0.15,0.25,0.40", help="comma-separated B1 softness values; B0 is hard and has no tau")
    return p.parse_args()


def _frozen_features(root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for side in SIDES:
        path = root / side / "target_family_manifest.json"
        payload = json.loads(path.read_text())
        key = f"T2_soft_barrier|tp3_sl2|{side}"
        features = payload.get("feature_contract", {}).get(key)
        if not isinstance(features, list) or not 30 <= len(features) <= 40:
            raise ValueError(f"missing frozen 30--40 feature {side} base contract: {path}")
        if len(set(features)) != len(features):
            raise ValueError(f"duplicate frozen {side} features")
        result[side] = list(features)
    return result


def _read_joined(panel: Path, sidecar: Path, features: dict[str, list[str]]) -> pd.DataFrame:
    wanted = sorted(set().union(*features.values()))
    panel_parts = sorted((panel / "parts").glob("*.parquet"))
    if not panel_parts or not (sidecar / "parts").exists():
        raise FileNotFoundError("panel or TP6/SL4 sidecar partitions are missing")
    identity = ["candidate_id", "__ts__", "side_name", "__label_available_at__", "t2_path_mfe_atr", "t2_path_mae_atr"]
    outcome = ["candidate_id", "__ts__", "side_name", "__label_available_at__", "t2_tp6_sl4_event", "t2_tp6_sl4_exit_minute", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "t4_tp6_sl4_terminal_pnl_atr"]
    frames: list[pd.DataFrame] = []
    for part in panel_parts:
        label_part = sidecar / "parts" / part.name
        if not label_part.exists():
            raise FileNotFoundError(f"TP6/SL4 sidecar partition missing: {label_part}")
        base = pd.read_parquet(part, columns=[*identity, *wanted])
        labels = pd.read_parquet(label_part, columns=outcome)
        for frame in (base, labels):
            frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
            frame["__label_available_at__"] = pd.to_datetime(frame["__label_available_at__"], utc=True)
        x = base.merge(labels, on=["candidate_id", "__ts__", "side_name", "__label_available_at__"], validate="one_to_one")
        if len(x) != len(base):
            raise ValueError(f"sidecar did not preserve every panel candidate: {part}")
        frames.append(x)
    data = pd.concat(frames, ignore_index=True)
    if data.candidate_id.duplicated().any():
        raise ValueError("candidate identity is not unique after panel/sidecar join")
    if not data.t2_tp6_sl4_event.isin((0, 1, 2)).all():
        raise ValueError("unexpected TP6/SL4 first-touch event")
    if not np.allclose(data.t4_tp6_sl4_gross_bps.to_numpy(float) - 100., data.t4_tp6_sl4_net_bps.to_numpy(float), atol=1e-4, rtol=0.):
        raise ValueError("TP6/SL4 gross/net cost identity is invalid")
    return data


def _matrix(x: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return x.loc[:, cols].replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(np.float32)


def _b0(frame: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(frame), 3), dtype=np.float64)
    out[np.arange(len(frame)), frame.t2_tp6_sl4_event.to_numpy(int)] = 1.
    return out


def _b1(frame: pd.DataFrame, tau: float) -> np.ndarray:
    """Literal soft TP6/SL4 target using shared H12 MFE/MAE supervision."""
    if tau <= 0.:
        raise ValueError("tau must be positive")
    event = frame.t2_tp6_sl4_event.to_numpy(int)
    exit_minute = frame.t2_tp6_sl4_exit_minute.to_numpy(float)
    mfe = frame.t2_path_mfe_atr.to_numpy(float)
    mae = frame.t2_path_mae_atr.to_numpy(float)
    up = (mfe - TP) / tau
    down = (mae - SL) / tau
    timeout = np.minimum((TP - mfe) / tau, (SL - mae) / tau)
    # Retain exact first touch.  A late win receives less additional evidence,
    # while MFE/MAE closeness still expresses the missed-barrier structure.
    bonus = 2.0 + .75 * (1. - np.minimum(exit_minute, HORIZON) / HORIZON)
    up[event == 0] += bonus[event == 0]
    down[event == 1] += bonus[event == 1]
    timeout[event == 2] += 2.0
    logits = np.column_stack([up, down, timeout])
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(np.clip(logits, -40., 0.))
    return probs / probs.sum(axis=1, keepdims=True)


def _fit_side(train: pd.DataFrame, evaluation: pd.DataFrame, cols: list[str], label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_eval = _matrix(train, cols), _matrix(evaluation, cols)
    raw = np.column_stack([
        np.maximum(lgb.LGBMRegressor(objective="huber", alpha=.90, **PARAMS).fit(x_train, label[:, j]).predict(x_eval), 0.)
        for j in range(3)
    ])
    probability = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-8)
    net = train.t4_tp6_sl4_net_bps.to_numpy(float)
    means = (label * net[:, None]).sum(axis=0) / np.maximum(label.sum(axis=0), 1.)
    return probability @ means, probability, means


def _metrics(frame: pd.DataFrame) -> list[dict]:
    ranked = frame.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort")
    rows = []
    for fraction in TOPS:
        chosen = ranked.head(int(np.ceil(len(ranked) * fraction)))
        rows.append({"top_fraction": fraction, "n": int(len(chosen)), "gross_bps": float(chosen.gross_bps.mean()), "net_bps": float(chosen.net_bps.mean()), "long_n": int(chosen.side_name.eq("long").sum()), "short_n": int(chosen.side_name.eq("short").sum())})
    return rows


def main() -> None:
    a = _args()
    train_end, eval_end = (pd.Timestamp(v, tz="UTC") for v in (a.train_end, a.eval_end))
    if not train_end < eval_end:
        raise ValueError("train-end must precede eval-end")
    taus = tuple(float(v.strip()) for v in a.taus.split(",") if v.strip())
    if not taus or any(v <= 0. for v in taus):
        raise ValueError("taus must contain positive values")
    frozen = _frozen_features(a.frozen_base)
    data = _read_joined(a.panel, a.sidecar, frozen)
    train = data[data.__ts__.lt(train_end) & data.__label_available_at__.lt(train_end)].copy()
    evaluation = data[data.__ts__.ge(train_end) & data.__ts__.lt(eval_end)].copy()
    if train.empty or evaluation.empty:
        raise ValueError("empty strict training or evaluation population")
    if evaluation.__label_available_at__.lt(train_end).any():
        raise AssertionError("evaluation row label was already available to the train cutoff")
    labels: list[tuple[str, float | None, np.ndarray]] = [("B0_hard_event", None, _b0(train))]
    labels += [("B1_tp6_sl4_distance", tau, _b1(train, tau)) for tau in taus]
    all_predictions: list[pd.DataFrame] = []
    metrics: list[dict] = []
    arms: list[dict] = []
    for target_name, tau, full_label in labels:
        rows: list[pd.DataFrame] = []
        contract: dict[str, dict] = {}
        for side in SIDES:
            train_mask = train.side_name.eq(side).to_numpy()
            tr = train.loc[train_mask].copy()
            ev = evaluation[evaluation.side_name.eq(side)].copy()
            score, p, means = _fit_side(tr, ev, frozen[side], full_label[train_mask])
            out = ev[["candidate_id", "__ts__", "side_name", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps"]].copy()
            out.columns = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps"]
            out["target"] = target_name
            out["tau"] = np.nan if tau is None else tau
            out["score_bps"] = score
            out[["p_upper", "p_lower", "p_timeout"]] = p
            rows.append(out)
            contract[side] = {"features": frozen[side], "conditional_net_means_bps": means.tolist(), "train_rows": int(len(tr)), "evaluation_rows": int(len(ev))}
        pred = pd.concat(rows, ignore_index=True)
        for row in _metrics(pred):
            metrics.append({"target": target_name, "tau": tau, **row})
        all_predictions.append(pred)
        arms.append({"target": target_name, "tau": tau, "side_contract": contract})
    a.out.mkdir(parents=True, exist_ok=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(a.out / "predictions.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(a.out / "global_metrics.parquet", index=False)
    manifest = {
        "schema": "full_universe_tp6_sl4_base_targets_v1", "status": "COMPLETED_WHEN_RUN",
        "contract": {"model": "fixed side-local three-output LightGBM regressors", "frozen_features": "TP3/SL2 base artifact feature sets reused unchanged; no feature selection or HPO", "B0": "hard exact TP6/SL4 first-touch event", "B1": "soft literal MFE/MAE distances to TP=6 ATR and SL=4 ATR plus exact first-touch/timing bonus", "entry_exit": "sidecar exact next-minute entry, first TP6/SL4 touch with adverse tie precedence, H12 timeout", "cost": "100 bps", "selection": "global pooled top 1/5/10 percent across long, short, assets and timestamps; no quotas", "no_meta": True},
        "causality": {"train_feature_ts_before": str(train_end), "train_label_available_before": str(train_end), "evaluation": [str(train_end), str(eval_end)], "assertion": "No label whose availability is at/after train_end enters the fit."},
        "rows": {"all": int(len(data)), "train": int(len(train)), "evaluation": int(len(evaluation))},
        "arms": arms, "global_metrics": metrics,
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
