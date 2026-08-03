#!/usr/bin/env python3
"""Memory-isolated stages for the strict TP6/SL4 R3→R5 experiment."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_r3_base_meta_r5 import (
    BASE_PARAMS, META_CONTEXT, META_PARAMS, _base_features, _base_probability,
    _classes, _context_for_ids, _map_apply, _map_fit, _matrix, _read_side, _weights,
)

def _load_context(panel: Path, ids: set[str], columns: list[str]) -> pd.DataFrame:
    pieces = []
    for part in sorted((panel / "parts").glob("*.parquet")):
        x = pd.read_parquet(part, columns=["candidate_id", *columns])
        x = x.loc[x.candidate_id.isin(ids)]
        if not x.empty:
            pieces.append(x)
    return pd.concat(pieces, ignore_index=True)


def _fit_meta(panel: Path, oof_paths: list[Path], out: Path, meta_context: list[str], mode: str) -> None:
    if out.exists():
        raise FileExistsError(out)
    oof = pd.concat([pd.read_parquet(path) for path in oof_paths], ignore_index=True)
    oof = oof.sort_values(["fold", "candidate_id"], kind="mergesort").reset_index(drop=True)
    mapped = []
    for fold, chunk in oof.groupby("fold", observed=True):
        history = oof.loc[oof.fold.lt(fold)]
        if history.empty:
            continue
        part = chunk.copy()
        part["base_expected_bps"] = _map_apply(part.base_raw.to_numpy(float), _map_fit(history.base_raw.to_numpy(float), history.net_bps.to_numpy(float)))
        mapped.append(part)
    train = pd.concat(mapped, ignore_index=True)
    context = _load_context(panel, set(train.candidate_id), meta_context)
    train = train.merge(context, on="candidate_id", how="inner", validate="one_to_one")
    train["residual_target"] = train.net_bps - train.base_expected_bps
    features = ["prob_adverse", "prob_weak", "prob_clear", "base_expected_bps", *meta_context]
    if mode == "residual":
        model = lgb.LGBMRegressor(objective="huber", alpha=.9, random_state=20260901, **META_PARAMS)
        model.fit(_matrix(train, features), train.residual_target.to_numpy(float))
    elif mode == "reliability":
        # Correctness-style meta target: conversion to a net-positive realised
        # outcome, conditional on an OOF base opportunity representation.
        model = lgb.LGBMClassifier(objective="binary", random_state=20260903, **META_PARAMS)
        model.fit(_matrix(train, features), train.net_bps.gt(0.).to_numpy(int))
    else:
        raise ValueError(f"unknown meta mode {mode!r}")
    base_map = _map_fit(oof.base_raw.to_numpy(float), oof.net_bps.to_numpy(float))
    out.mkdir(parents=True)
    joblib.dump({"model": model, "features": features, "base_map": base_map, "mode": mode}, out / "meta_artifact.joblib")
    manifest = {"schema": "tp6_sl4_r5_meta_artifact_v1", "status": "COMPLETED",
                "contract": {"source": "strict chronological base OOF only", "target": "exact net minus prior-OOF expected bps" if mode == "residual" else "net-positive correctness conditional on OOF base output"},
                "oof_rows": len(oof), "meta_train_rows": len(train), "feature_count": len(features), "meta_context_features": meta_context}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


def _score(panel: Path, winner: Path, robust: Path, features_root: Path, side: str, artifact: Path, start: pd.Timestamp, end: pd.Timestamp, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    artifact_data = joblib.load(artifact / "meta_artifact.joblib")
    base_cols = _base_features(features_root, side)
    boundary = pd.Timestamp("2024-03-01", tz="UTC")
    train = _read_side(panel, winner, robust, side, base_cols, cutoff=boundary)
    labels = train[["robust_clear_event_b25", "lower_touch_minute"]]
    y = _classes(labels)
    weight = train[["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]]
    base = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=20260900, **BASE_PARAMS)
    base.fit(_matrix(train, base_cols), y, sample_weight=_weights(weight, y))
    del train, labels, weight, y
    gc.collect()
    evaluation = _read_side(panel, winner, robust, side, base_cols, start=start, cutoff=end)
    p = _base_probability(base, evaluation, base_cols)
    result = evaluation[["candidate_id", "__ts__", "side_name", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps"]].copy()
    result.columns = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps"]
    context = _load_context(panel, set(result.candidate_id), artifact_data["features"][4:])
    result = result.merge(context, on="candidate_id", how="inner", validate="one_to_one")
    result["prob_adverse"] = p[:, 0]; result["prob_weak"] = p[:, 1]; result["prob_clear"] = p[:, 2]; result["base_raw"] = p[:, 2] - p[:, 0]
    result["base_expected_bps"] = _map_apply(result.base_raw.to_numpy(float), artifact_data["base_map"])
    if artifact_data.get("mode", "residual") == "reliability":
        result["meta_reliability"] = artifact_data["model"].predict_proba(_matrix(result, artifact_data["features"]))[:, 1]
        result["score_reliability"] = result.meta_reliability
    else:
        result["meta_residual_bps"] = artifact_data["model"].predict(_matrix(result, artifact_data["features"]))
    result["score_base_bps"] = result.base_expected_bps
    if "meta_residual_bps" in result:
        result["score_base_meta_bps"] = result.base_expected_bps + result.meta_residual_bps
    out.mkdir(parents=True)
    result.to_parquet(out / "base_meta_oos_predictions.parquet", index=False)
    manifest = {"schema": "tp6_sl4_r5_score_shard_v1", "status": "COMPLETED", "side": side,
                "evaluation_start": str(start), "evaluation_end": str(end), "rows": len(result),
                "base_train_end": "2024-03-01", "meta_artifact": str(artifact)}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-meta"); fit.add_argument("--oof", type=Path, nargs="+", required=True); fit.add_argument("--out", type=Path, required=True); fit.add_argument("--selection", type=Path); fit.add_argument("--mode", choices=("residual", "reliability"), default="residual")
    score = sub.add_parser("score"); score.add_argument("--side", choices=("long", "short"), required=True); score.add_argument("--artifact", type=Path, required=True); score.add_argument("--start", required=True); score.add_argument("--end", required=True); score.add_argument("--out", type=Path, required=True)
    p.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    p.add_argument("--winner", type=Path, default=ROOT / "data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1")
    p.add_argument("--robust", type=Path, default=ROOT / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1")
    p.add_argument("--features", type=Path, default=ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1")
    a = p.parse_args()
    if a.command == "fit-meta":
        selected = META_CONTEXT if a.selection is None else json.loads(a.selection.read_text())["selection"]["selected_meta_context"]
        _fit_meta(a.panel, a.oof, a.out, selected, a.mode)
    else: _score(a.panel, a.winner, a.robust, a.features, a.side, a.artifact, pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC"), a.out)


if __name__ == "__main__":
    main()
