#!/usr/bin/env python3
"""Separate May--June continuation of the accepted residual layer.

The residual contract, calibration, target and HPO are frozen.  Its only new
inputs are the new base continuation OOF scores and exact 12-hour economics on
the diagnostic common universe.  Historical Jan--Apr artifacts are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_febapr2025_canonical_residual_oof as accepted_residual

SCHEMA = "mayjun2025_canonical_residual_continuation_v1"
TOP = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1"
CONTRACT = ROOT / "data_perp/artifacts/packb_side_local_residual_oof_20260724_v1_31_8"
AE = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
BASE = ROOT / "data_perp/artifacts/mayjun2025_canonical_base_continuation_20260730_v1"
OUT = ROOT / "data_perp/artifacts/mayjun2025_canonical_residual_continuation_20260730_v1"
TARGET = "__first_touch_capture_net__"
WEIGHT = "__w__"


class ResidualContinuationError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")
    os.replace(tmp, path)


def _economics(frame: pd.DataFrame, score: pd.Series) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    ranked = frame.assign(_score=np.asarray(score)).sort_values("_score", ascending=False, kind="stable")
    k = max(1, int(np.ceil(len(ranked) * .10)))
    top = ranked.head(k)
    return {"rows": int(len(ranked)), "top10_global_rows": int(k),
            "top10_global_execution_net_ev": float(top.execution_net_ev_12h.mean()),
            "top10_global_positive_fraction": float((top.execution_net_ev_12h > 0).mean()),
            "score_native_target_spearman": float(ranked[["_score", TARGET]].corr(method="spearman").iloc[0, 1])}


def _feature_matrix(frame: pd.DataFrame, side: str, feature_store: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_contract = json.loads((AE / side / "loader_evidence/frozen_feature_contract.json").read_text())
    requested = json.loads((CONTRACT / side / "feature_contract.json").read_text())["features"]
    point = frame.loc[:, ["candidate_id", "side_name", "__ts__", "__symbol__"]].copy()
    # PIT store keys use the candidate-id symbol spelling, not the display
    # spelling retained by the exact execution label.
    point["__symbol__"] = point.candidate_id.astype(str).str.split("|", n=1).str[0]
    # The accepted helper binds its store through its module global.  Set it
    # explicitly so this continuation's CLI/config path is real rather than a
    # cosmetic argument.  The default remains the accepted frozen store.
    accepted_residual.STORE = Path(feature_store)
    raw, coverage = accepted_residual.point_features(point, raw_contract)
    timestamp = pd.to_datetime(frame["__ts__"], utc=True)
    raw = raw.reset_index(drop=True)
    raw["base_prediction"] = frame.base_oof_score.to_numpy()
    raw["base_rank_pct_timestamp_side"] = frame.base_rank_pct_timestamp_side.to_numpy()
    raw["base_rank_timestamp_side"] = frame.base_rank_timestamp_side.to_numpy()
    raw["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour.to_numpy() / 24)
    raw["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour.to_numpy() / 24)
    matrix = raw.reindex(columns=requested)
    finite = {column: float(matrix[column].notna().mean()) for column in requested}
    bad = {column: value for column, value in finite.items() if value < .95}
    if coverage["exact_key_fraction"] != 1.0 or bad:
        raise ResidualContinuationError(f"{side} PIT residual coverage failed: {bad}")
    return matrix, {"raw": coverage, "residual_feature_finite_fraction": finite, "features": requested}


def _base_rows(base_dir: Path) -> pd.DataFrame:
    contract = json.loads((base_dir / "continuation_contract.json").read_text())
    if contract.get("schema") != "mayjun2025_canonical_base_continuation_v1":
        raise ResidualContinuationError("base continuation schema mismatch")
    frame = pd.read_parquet(base_dir / "oof_predictions.parquet")
    if not frame.selected_top40.all():
        # 30 assets per side means the preserved top-40 handoff accepts every
        # scored row, but never silently admits a rank beyond the contract.
        frame = frame.loc[frame.selected_top40].copy()
    frame["native_label_resolution_utc"] = pd.to_datetime(frame.base_label_resolution_utc, utc=True)
    return frame


def run(*, output_dir: Path = OUT, base_dir: Path = BASE, feature_store: Path = ROOT / "data_perp/features/20260711_070000",
        month: int | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ResidualContinuationError(f"refusing to overwrite existing continuation: {output_dir}")
    months = (month,) if month else (5, 6)
    if any(item not in (5, 6) for item in months):
        raise ResidualContinuationError("May and/or June 2025 are required")
    historical = pd.read_parquet(TOP / "population.parquet")
    historical["__ts__"] = pd.to_datetime(historical["__ts__"], utc=True)
    historical["native_label_resolution_utc"] = pd.to_datetime(historical.native_label_resolution_utc, utc=True)
    continued = _base_rows(Path(base_dir))
    frame = pd.concat([historical, continued], ignore_index=True, sort=False)
    if frame.candidate_id.duplicated().any():
        raise ResidualContinuationError("historical and continuation candidate identities overlap")
    output_dir.mkdir(parents=True)
    all_outputs: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for side in ("long", "short"):
        side_frame = frame.loc[frame.side_name.eq(side)].reset_index(drop=True)
        hp = json.loads((CONTRACT / side / "hpo_contract.json").read_text())
        for value in months:
            start = pd.Timestamp(f"2025-{value:02d}-01", tz="UTC")
            end = start + pd.offsets.MonthBegin(1)
            train = side_frame.loc[side_frame.native_label_resolution_utc.lt(start)].reset_index(drop=True)
            valid = side_frame.loc[side_frame["__ts__"].ge(start) & side_frame["__ts__"].lt(end)].reset_index(drop=True)
            if train.empty or valid.empty:
                raise ResidualContinuationError(f"{side} {value}: train or validation is empty")
            # New scorer rows are top-40 by construction.  Historical rows
            # retain the accepted top-40 handoff population and can never
            # include a comparator score.
            train_x, train_cov = _feature_matrix(train, side, Path(feature_store))
            valid_x, valid_cov = _feature_matrix(valid, side, Path(feature_store))
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
                train.base_oof_score, train[TARGET], sample_weight=train[WEIGHT]
            )
            base_train = iso.predict(train.base_oof_score)
            y = train[TARGET].to_numpy() - base_train
            model = lgb.LGBMRegressor(**hp["params"], n_estimators=int(hp["rounds"]), random_state=9600 + value).fit(
                train_x, y, sample_weight=train[WEIGHT]
            )
            base_expected = iso.predict(valid.base_oof_score)
            delta = model.predict(valid_x)
            out = valid.copy()
            out["base_expected_ev"] = base_expected
            out["residual_delta_ev"] = delta
            out["residual_expected_ev"] = base_expected + float(hp["alpha"]) * delta
            out["residual_fold"] = f"month_2025_{value:02d}"
            out["residual_is_oof"] = True
            fold_dir = output_dir / side / out.residual_fold.iloc[0]
            fold_dir.mkdir(parents=True)
            out.to_parquet(fold_dir / "oof_predictions.parquet", index=False, compression="zstd")
            all_outputs.append(out)
            folds.append({"side": side, "month": f"2025-{value:02d}", "train_rows": int(len(train)), "validation_rows": int(len(valid)),
                          "train_label_resolution_max_utc": train.native_label_resolution_utc.max().isoformat(),
                          "label_cutoff": f"native label resolution < {start.isoformat()}", "feature_selection": "none; frozen residual feature contract",
                          "hpo": "none; frozen residual HPO contract", "hpo_sha256": _sha(CONTRACT / side / "hpo_contract.json"),
                          "train_feature_coverage": train_cov, "validation_feature_coverage": valid_cov})
    output = pd.concat(all_outputs, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable")
    output.to_parquet(output_dir / "oof_predictions.parquet", index=False, compression="zstd")
    per_month: dict[str, Any] = {}
    for key, part in output.groupby(output["__ts__"].dt.strftime("%Y-%m"), sort=True):
        per_month[key] = {"base": _economics(part, part.base_expected_ev), "residual": _economics(part, part.residual_expected_ev)}
    manifest = {"schema": SCHEMA, "status": "CONTINUATION_CONTRACT_UNVALIDATED_AGAINST_ACCEPTED_JAN_APR_SCOPE",
                "scope": "May--June 2025 frozen 30-asset common-universe; exact 1m candidate-local 12h policy economics",
                "historical_inputs": "accepted Feb--Apr top-40 residual population, read-only", "base_input": str(base_dir),
                "residual_training": "only rows with native label resolution strictly before validation-month start",
                "residual_contract": "accepted side-local isotonic base mapping plus frozen residual feature/HPO contract; no new HPO or feature selection",
                "action_layer": "timing, MAE, target price and wait actions remain outside this residual score",
                "folds": folds, "prediction_rows": int(len(output)), "metrics": {"by_month": per_month,
                "aggregate": {"base": _economics(output, output.base_expected_ev), "residual": _economics(output, output.residual_expected_ev)}},
                "inputs_sha256": {"accepted_historical_population": _sha(TOP / "population.parquet"),
                                  "base_continuation_predictions": _sha(Path(base_dir) / "oof_predictions.parquet")},
                "outputs": {"oof_predictions.parquet": _sha(output_dir / "oof_predictions.parquet")}}
    _write_json(output_dir / "continuation_contract.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--feature-store", type=Path, default=ROOT / "data_perp/features/20260711_070000")
    parser.add_argument("--month", type=int, choices=(5, 6))
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir, base_dir=args.base_dir, feature_store=args.feature_store, month=args.month), indent=2, default=str))


if __name__ == "__main__":
    main()
