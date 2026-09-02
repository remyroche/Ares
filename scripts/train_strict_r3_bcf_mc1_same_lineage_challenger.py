#!/usr/bin/env python3
"""Fit a challenger-only BCF-native MC1 mapper from OOS BCF score rows.

The score family must already have been generated target-free by one frozen
BCF bundle.  Parent-policy labels are attached in a prior step.  This runner
does not score, route, admit, or execute a trade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_bcf_mc1_mapper import CONTRACT, FEATURES, SCHEMA
from extreme_price_movements.strict_r3_mc1_mapper import fit_structural_curve, score_bands


PARAMS = {
    "max_depth": 2,
    "max_iter": 80,
    "learning_rate": 0.04,
    "l2_regularization": 20.0,
    "min_samples_leaf": 100,
    "random_state": 1729,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _equal_day_sample(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    work["__day__"] = work["__decision_ts__"].dt.normalize()
    days = sorted(work["__day__"].unique())
    per_day = max(1, cap // len(days))
    pieces: list[pd.DataFrame] = []
    for day in days:
        group = work.loc[work["__day__"].eq(day)].copy()
        order = pd.util.hash_pandas_object(group["candidate_id"].astype(str), index=False).to_numpy(np.uint64)
        pieces.append(group.iloc[np.argsort(order, kind="stable")[:per_day]])
    sampled = pd.concat(pieces, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    if len(sampled) > cap:
        order = pd.util.hash_pandas_object(sampled["candidate_id"].astype(str), index=False).to_numpy(np.uint64)
        sampled = sampled.iloc[np.argsort(order, kind="stable")[:cap]]
    return sampled.drop(columns="__day__").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--fit-cutoff", required=True)
    parser.add_argument("--train-cap", type=int, default=50_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    cutoff = _utc(args.fit_cutoff)
    columns = [
        "candidate_id", "__decision_ts__", "side_name", *FEATURES,
        "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    ]
    ledger = pd.read_parquet(args.ledger, columns=columns)
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True)
    ledger["policy_label_available_ts"] = pd.to_datetime(ledger["policy_label_available_ts"], utc=True)
    if ledger.empty or ledger["candidate_id"].duplicated().any():
        raise ValueError("MC1 challenger ledger must be nonempty and candidate-unique")
    if not ledger["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("BCF MC1 challenger is long-only")
    x = ledger.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = (
        ledger["__decision_ts__"].lt(cutoff)
        & ledger["policy_label_available_ts"].lt(cutoff)
        & ledger["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(ledger["policy_net_bps"], errors="coerce").notna()
        & np.isfinite(x.to_numpy(float)).all(axis=1)
    )
    history = ledger.loc[valid].copy()
    if len(history) < 1_000:
        raise ValueError("insufficient strictly resolved OOS BCF rows for challenger MC1")
    history["net"] = pd.to_numeric(history["policy_net_bps"], errors="coerce")
    history["score_band"] = score_bands(history)
    curve, global_mean = fit_structural_curve(history)
    training = _equal_day_sample(history, int(args.train_cap))
    target = training["net"].to_numpy(float)
    model = HistGradientBoostingRegressor(**PARAMS).fit(
        training.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce"), target,
    )
    payload = {
        "features_ordered": FEATURES,
        "model": model,
        "structural_curve_bps": np.asarray(curve, dtype=float),
        "structural_global_mean_bps": float(global_mean),
    }
    args.out_dir.mkdir(parents=True)
    model_path = args.out_dir / "bcf_mc1_d2.joblib"
    joblib.dump(payload, model_path)
    manifest = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "bundle_id": hashlib.sha256(
            (str(_sha(args.ledger)) + cutoff.isoformat() + json.dumps(PARAMS, sort_keys=True)).encode()
        ).hexdigest()[:20],
        "status": "challenger_only_not_live_promoted",
        "side": "long",
        "fit_cutoff": cutoff.isoformat(),
        "features_ordered": list(FEATURES),
        "feature_contract": "native_bcf_ten_head_agreement_v1",
        "model": {"type": "HistGradientBoostingRegressor", **PARAMS},
        "dynamic_component": {
            "type": "causal_recent_global_residual_shift",
            "window_days": 21,
            "trim_fraction_each_tail": 0.10,
            "outcome_availability_rule": "policy_label_available_ts < decision_ts",
            "absent_support": "unavailable_fail_closed",
        },
        "admission_threshold_bps": 30.0,
        "training_rows": int(len(training)),
        "history_rows": int(len(history)),
        "history_start_decision_ts": history["__decision_ts__"].min().isoformat(),
        "history_max_label_available_ts": history["policy_label_available_ts"].max().isoformat(),
        "source_ledger": str(args.ledger),
        "source_ledger_sha256": _sha(args.ledger),
        "source_scores_are_strict_oos_from_bcf_bundle": True,
        "sha256": {"model_bundle": _sha(model_path)},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
