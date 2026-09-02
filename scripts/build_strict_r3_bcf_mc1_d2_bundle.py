#!/usr/bin/env python3
"""Fit and seal the family-specific BCF MC1_d2 absolute-EV mapper."""

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

from extreme_price_movements.strict_r3_bcf_mc1_mapper import CONTRACT, FEATURES, SCHEMA, fit_structural_curve
from extreme_price_movements.strict_r3_mc1_mapper import score_bands


def _sha(path: Path) -> str:
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
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config = json.loads(args.champion_config.read_text())
    config_side = str(config.get("side") or "").strip().lower()
    if config_side != args.side:
        raise ValueError("BCF MC1 champion configuration side must match --side")
    if tuple(config["features_ordered"]) != FEATURES:
        raise ValueError("BCF MC1 feature order changed")
    cutoff = pd.Timestamp(args.fit_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    columns = ["candidate_id", "__decision_ts__", "policy_label_available_ts", "policy_path_valid", "policy_net_bps", "side_name", *FEATURES]
    source = pd.read_parquet(args.ledger, columns=columns)
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True)
    source["policy_label_available_ts"] = pd.to_datetime(source["policy_label_available_ts"], utc=True)
    observed_side = source["side_name"].astype(str).str.strip().str.lower()
    if not observed_side.isin(("long", "short")).all() or not observed_side.eq(args.side).all():
        raise ValueError(
            "BCF MC1 ledger must be side-local and match --side; "
            f"observed={observed_side.value_counts(dropna=False).to_dict()}",
        )
    # Strictly earlier than the live cutoff; no boundary outcome can enter the
    # first live-hour calibration reserve.
    source = source.loc[
        source["policy_label_available_ts"].lt(cutoff)
        & source["policy_path_valid"].fillna(False).astype(bool)
        & source["policy_net_bps"].notna()
        & pd.to_numeric(source["final_score"], errors="coerce").notna()
    ].copy()
    source["day"] = source["__decision_ts__"].dt.normalize()
    source["score_band"] = score_bands(source)
    pieces: list[pd.DataFrame] = []
    for _, group in source.groupby("day", sort=True):
        ordered = group.sort_values(["__decision_ts__", "final_score", "candidate_id"], ascending=[True, False, True], kind="stable")
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
        learning_rate=float(params["learning_rate"]), l2_regularization=float(params["l2_regularization"]),
        min_samples_leaf=int(params["min_samples_leaf"]), random_state=int(params["random_state"]),
    ).fit(train.loc[:, FEATURES], train["net"])
    args.out_dir.mkdir(parents=True)
    model_path = args.out_dir / "bcf_mc1_d2.joblib"
    joblib.dump({"model": model, "features_ordered": FEATURES, "structural_curve_bps": curve, "structural_global_bps": global_mean}, model_path, compress=3)
    identity = hashlib.sha256((_sha(args.champion_config) + _sha(args.ledger) + cutoff.isoformat() + args.side + "bcf-native-v1").encode()).hexdigest()[:20]
    manifest = {
        "schema": SCHEMA, "contract": CONTRACT, "bundle_id": identity, "side": args.side,
        "fit_cutoff": cutoff.isoformat(), "champion_config": str(args.champion_config),
        "champion_config_sha256": _sha(args.champion_config), "features_ordered": list(FEATURES),
        "feature_contract": "native_bcf_ten_head_agreement_v1", "model": config["model"],
        "dynamic_component": {**config["dynamic_component"], "outcome_availability_rule": "policy_label_available_ts < decision_ts"},
        "admission_threshold_bps": 30.0,
        "authority": "BCF MC1 is dual-gate confirmation and BCF-EV auction priority; it never manufactures current-v5 admission",
        "training_rows": int(len(train)), "history_rows": int(len(history)),
        "history_max_label_available_ts": history["policy_label_available_ts"].max().isoformat(),
        "source_ledger": str(args.ledger), "source_ledger_sha256": _sha(args.ledger),
        "sha256": {"model_bundle": _sha(model_path)},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "bundle_id": identity, "rows": len(train)}))


if __name__ == "__main__":
    main()
