#!/usr/bin/env python3
"""Fit and seal the production Adaptive Exit V1 bundle from prior labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.adaptive_exit_v1 import CONTROLLER, SCHEMA, TARGET

SOURCE = ROOT / "data_perp/artifacts/canonical_a5_source_aligned_hybrid_adaptive_exit_funnel_20260813_v4"
POLICY = ROOT / "data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/winner.json"
OUTPUT = ROOT / "data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1"
SEED = 20260813


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _row_hash(frame: pd.DataFrame) -> str:
    values = frame.loc[:, ["candidate_id", "decision_ts", "path_bar"]].copy()
    values["decision_ts"] = pd.to_datetime(values.decision_ts, utc=True).astype(str)
    return hashlib.sha256(values.sort_values(list(values.columns)).to_csv(index=False).encode()).hexdigest()


def _cap(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.sort_values(["decision_ts", "candidate_id", "path_bar"]).copy()
    work = frame.copy()
    work["__month"] = pd.to_datetime(work.decision_ts, utc=True).dt.to_period("M").astype(str)
    per = max(1, maximum // work.__month.nunique())
    pieces = [g.sample(min(len(g), per), random_state=SEED) for _, g in work.groupby("__month")]
    out = pd.concat(pieces)
    if len(out) < maximum:
        extra = work.drop(out.index).sample(min(maximum - len(out), len(work) - len(out)), random_state=SEED)
        out = pd.concat([out, extra])
    return out.drop(columns="__month").sort_values(["decision_ts", "candidate_id", "path_bar"]).head(maximum)


def _matrix(frame: pd.DataFrame, fields: list[str], medians: pd.Series | None = None):
    x = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median() if medians is None else medians
    return x.fillna(med).fillna(0.0).to_numpy(float), med.astype(float)


def _params(rows: int, trees: int = 700) -> dict:
    return dict(objective="quantile", alpha=.65, n_estimators=int(trees), learning_rate=.03,
                max_depth=4, num_leaves=15, min_child_samples=max(100, int(.01 * rows)),
                subsample=.75, subsample_freq=1, colsample_bytree=.75, reg_lambda=10.0,
                random_state=SEED, n_jobs=-1, verbosity=-1)


def _fit_with_stop(train: pd.DataFrame, fields: list[str]):
    clock = pd.to_datetime(train.decision_ts, utc=True)
    cut = clock.quantile(.8)
    fit = train[clock.lt(cut)]
    stop = train[clock.ge(cut + pd.Timedelta(hours=12))]
    if len(stop) < 100:
        fit, stop = train.iloc[: int(.8 * len(train))], train.iloc[int(.8 * len(train)) :]
    xf, med = _matrix(fit, fields)
    xs, _ = _matrix(stop, fields, med)
    model = LGBMRegressor(**_params(len(fit)))
    model.fit(xf, fit[TARGET].to_numpy(float), eval_set=[(xs, stop[TARGET].to_numpy(float))],
              callbacks=[early_stopping(30, verbose=False)])
    return model, med, int(model.best_iteration_ or 700)


def _refit(train: pd.DataFrame, fields: list[str], trees: int):
    x, med = _matrix(train, fields)
    model = LGBMRegressor(**_params(len(train), trees))
    model.fit(x, train[TARGET].to_numpy(float))
    return model, med


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--policy-json", type=Path, default=POLICY)
    ap.add_argument("--activation-ts", default="2026-08-01T00:00:00Z")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT)
    ap.add_argument("--max-train-states", type=int, default=40000)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    source_manifest = json.loads((args.source / "run_manifest.json").read_text())
    f1 = list(source_manifest["feature_contracts"]["F1"])
    f4 = list(source_manifest["feature_contracts"]["F4"])
    states = pd.read_parquet(args.source / "causal_states.parquet")
    states["decision_ts"] = pd.to_datetime(states.decision_ts, utc=True)
    activation = pd.Timestamp(args.activation_ts)
    activation = activation.tz_localize("UTC") if activation.tzinfo is None else activation.tz_convert("UTC")
    start = max(pd.Timestamp("2025-01-01", tz="UTC"), activation - pd.DateOffset(months=9))
    eligible = states[
        states.decision_ts.ge(start)
        & states.decision_ts.lt(activation - pd.Timedelta(hours=12))
        & states[TARGET].notna()
    ].copy()
    train = _cap(eligible, args.max_train_states).reset_index(drop=True)
    if len(train) < 1000:
        raise RuntimeError("insufficient adaptive-exit training support")
    clock = train.decision_ts
    inner_cut = clock.quantile(.65)
    opportunity = train[clock.lt(inner_cut)]
    gate = train[clock.ge(inner_cut + pd.Timedelta(hours=12))]
    if len(opportunity) < 500 or len(gate) < 300:
        raise RuntimeError("insufficient inner disagreement support")
    inner1, med1, _ = _fit_with_stop(opportunity, f1)
    inner4, med4, _ = _fit_with_stop(opportunity, f4)
    xg1, _ = _matrix(gate, f1, med1)
    xg4, _ = _matrix(gate, f4, med4)
    gate_p1 = inner1.predict(xg1)
    gate_p4 = inner4.predict(xg4)
    disagreement_p80 = float(np.quantile(np.abs(gate_p1 - gate_p4), .8))
    stopped1, _, trees1 = _fit_with_stop(train, f1)
    stopped4, _, trees4 = _fit_with_stop(train, f4)
    model1, final_med1 = _refit(train, f1, trees1)
    model4, final_med4 = _refit(train, f4, trees4)
    policy = json.loads(args.policy_json.read_text())["winner"]
    payload = {
        "f1_model": model1, "f4_model": model4,
        "f1_fields": tuple(f1), "f4_fields": tuple(f4),
        "f1_medians": final_med1.to_numpy(float), "f4_medians": final_med4.to_numpy(float),
        "disagreement_p80": disagreement_p80,
    }
    artifact = args.out_dir / "adaptive_exit_v1.joblib"
    joblib.dump(payload, artifact, compress=3)
    bundle_id = hashlib.sha256((SCHEMA + _sha(artifact) + activation.isoformat()).encode()).hexdigest()[:20]
    manifest = {
        "schema": SCHEMA, "bundle_id": bundle_id, "side": "long", "controller": CONTROLLER,
        "target": TARGET, "objective": "quantile_0.65", "research_canonical": True,
        "live_canonical": False, "activation_ts": activation.isoformat(),
        "training": {"start": start.isoformat(), "end_exclusive": (activation-pd.Timedelta(hours=12)).isoformat(),
                     "purge_hours": 12, "row_cap": args.max_train_states, "eligible_rows": len(eligible),
                     "sampled_rows": len(train), "sampled_row_sha256": _row_hash(train),
                     "equal_month_sampling": True, "inner_fit_fraction": .65,
                     "inner_gate_rows": len(gate), "f1_best_iteration": trees1, "f4_best_iteration": trees4,
                     "seed": SEED},
        "feature_contracts": {"F1": f1, "F4": f4},
        "controller_parameters": {"activation_shrink": .75, "activation_lower_ratio": .5,
                                  "activation_upper_ratio": 1.25, "disagreement_p80": disagreement_p80},
        "policy": {"sl_atr": float(policy["sl_mult"]),
                   "base_activation_atr": float(policy["trailing_activation_mult"]),
                   "fixed_giveback_atr": float(policy["fixed_trailing_gap_mult"]),
                   "timeout_hours": 12, "round_trip_cost_bps": 100.0,
                   "entry": "first_15m_open_at_signal_close_plus_1h",
                   "decision_clock": "completed_hourly_bar", "effective_from": "next_15m_bar",
                   "authority": "trailing_activation_only"},
        "source": {"research_manifest": str((args.source / "run_manifest.json").relative_to(ROOT)),
                   "research_manifest_sha256": _sha(args.source / "run_manifest.json"),
                   "policy_json": str(args.policy_json.relative_to(ROOT)), "policy_json_sha256": _sha(args.policy_json)},
        "sha256": {"model_bundle": _sha(artifact)},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Serialization parity reference: inputs and outputs are immutable and are
    # not performance evidence. They catch preprocessing/model drift at load.
    reference = (
        train.tail(min(2048, len(train)))
        .drop_duplicates("candidate_id", keep="last")
        .tail(512)
        .loc[:, ["candidate_id", "decision_ts", "path_bar", *dict.fromkeys([*f1, *f4])]]
        .copy()
    )
    from extreme_price_movements.adaptive_exit_v1 import AdaptiveExitV1Bundle
    scored = AdaptiveExitV1Bundle.load(args.out_dir).score(reference, decision_ts=activation)
    reference.merge(scored, on="candidate_id", validate="one_to_one").to_parquet(
        args.out_dir / "serialization_parity_reference.parquet", index=False, compression="zstd")
    print(json.dumps({"event": "complete", "bundle_id": bundle_id, "rows": len(train),
                      "disagreement_p80": disagreement_p80, "f1_trees": trees1, "f4_trees": trees4}))


if __name__ == "__main__":
    main()
