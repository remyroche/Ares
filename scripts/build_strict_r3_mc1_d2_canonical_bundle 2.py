#!/usr/bin/env python3
"""Fit and seal the exact frozen MC1_d2 champion at one activation cutoff."""

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

from extreme_price_movements.strict_r3_mc1_mapper import (  # noqa: E402
    CONTRACT, FEATURES, SCHEMA, fit_structural_curve, score_bands,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--champion-config", type=Path, required=True)
    parser.add_argument("--fit-cutoff", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config = json.loads(args.champion_config.read_text())
    if tuple(config["features_ordered"]) != FEATURES:
        raise ValueError("champion feature order changed")
    cutoff = pd.Timestamp(args.fit_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    columns = [
        "candidate_id", "__decision_ts__", "policy_label_available_ts",
        "policy_path_valid", "policy_net_bps", *FEATURES,
    ]
    source = pd.read_parquet(args.ledger, columns=list(dict.fromkeys(columns)))
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True)
    source["policy_label_available_ts"] = pd.to_datetime(source["policy_label_available_ts"], utc=True)
    source = source.loc[
        source["policy_label_available_ts"].le(cutoff)
        & source["policy_path_valid"].fillna(False).astype(bool)
        & source["policy_net_bps"].notna()
    ].copy()
    source["day"] = source["__decision_ts__"].dt.normalize()
    source["score_band"] = score_bands(source)
    # Reproduce the frozen experiment's deterministic day-balanced substrate.
    pieces = []
    for _, group in source.groupby("day", sort=True):
        ordered = group.sort_values(
            ["__decision_ts__", "final_score", "candidate_id"],
            ascending=[True, False, True], kind="stable",
        )
        ordered["rank_n"] = ordered.groupby("__decision_ts__", sort=False).cumcount() + 1
        top = ordered.loc[ordered["rank_n"].le(50)]
        rest = ordered.drop(top.index)
        pieces.append(pd.concat([top, rest.sample(min(250, len(rest)), random_state=1729)]))
    history = pd.concat(pieces, ignore_index=True).sort_values("policy_label_available_ts", kind="stable")
    net = pd.to_numeric(history["policy_net_bps"], errors="coerce")
    lower, upper = net.quantile([0.02, 0.98])
    history["net"] = net.clip(lower, upper)
    curve, global_mean = fit_structural_curve(history)
    train = history.dropna(subset=[*FEATURES, "net"])
    if len(train) > 50_000:
        train = train.sample(50_000, random_state=1729)
    params = config["model"]
    model = HistGradientBoostingRegressor(
        max_depth=int(params["max_depth"]), max_iter=int(params["max_iter"]),
        learning_rate=float(params["learning_rate"]),
        l2_regularization=float(params["l2_regularization"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=int(params["random_state"]),
    ).fit(train.loc[:, FEATURES], train["net"])
    args.out_dir.mkdir(parents=True)
    model_path = args.out_dir / "mc1_d2.joblib"
    joblib.dump({
        "model": model, "features_ordered": FEATURES,
        "structural_curve_bps": curve, "structural_global_bps": global_mean,
    }, model_path, compress=3)
    identity = hashlib.sha256(
        (sha(args.champion_config) + sha(args.ledger) + cutoff.isoformat()).encode()
    ).hexdigest()[:20]
    manifest = {
        "schema": SCHEMA, "contract": CONTRACT, "bundle_id": identity,
        "side": "long", "fit_cutoff": cutoff.isoformat(),
        "champion_config": str(args.champion_config),
        "champion_config_sha256": sha(args.champion_config),
        "features_ordered": list(FEATURES), "model": config["model"],
        "dynamic_component": config["dynamic_component"],
        "admission_threshold_bps": 50.0,
        "authority": "MC1 absolute EV owns admission; Robust-21 is retained control/fallback telemetry; no numerical blend",
        "auction_order": "frozen strict-R3 final_score only",
        "training_rows": int(len(train)), "history_rows": int(len(history)),
        "history_max_label_available_ts": history["policy_label_available_ts"].max().isoformat(),
        "source_ledger": str(args.ledger), "source_ledger_sha256": sha(args.ledger),
        "sha256": {"model_bundle": sha(model_path)},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "bundle_id": identity, "rows": len(train)}))


if __name__ == "__main__":
    main()
