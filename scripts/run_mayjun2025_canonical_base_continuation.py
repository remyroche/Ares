#!/usr/bin/env python3
"""Versioned May--June 2025 continuation of the frozen canonical base model.

This is deliberately separate from the accepted Jan--Apr runner.  It keeps the
31/8 feature, HPO and AE/GMM contracts fixed, advances only the chronological
label cutoff, and evaluates the frozen 30-asset common-universe on exact 1m
12-hour policy economics.  It is a continuation contract, not a replacement
or a relabelling of the accepted Jan--Apr artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_febapr2025_canonical_base_oof import (
    ECONOMIC, IDENTITY, SIDES, TARGET, WEIGHT, _deterministic_cap,
    _load_contracts, _materialize_features, _sha256,
)
from scripts.run_packb_pre_march_side_fs_hpo import _lgbm_regressor

SCHEMA = "mayjun2025_canonical_base_continuation_v1"
DEFAULT_LABEL_DIR = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
DEFAULT_CANDIDATES = ROOT / "data_perp/artifacts/mayjul2025_execution_ev_common30_policy_inputs_20260727_v2/candidates.parquet"
DEFAULT_EXECUTION = ROOT / "data_perp/artifacts/mayjul2025_execution_ev_common30_labels_20260727_v2/labels.parquet"
DEFAULT_PROMOTION = ROOT / "docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json"
DEFAULT_AE = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/mayjun2025_canonical_base_continuation_20260730_v1"


class ContinuationError(RuntimeError):
    pass


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")
    os.replace(tmp, path)


def _identity_hash(frame: pd.DataFrame) -> str:
    value = frame.loc[:, list(IDENTITY)].copy()
    value["__ts__"] = pd.to_datetime(value["__ts__"], utc=True).astype(str)
    value = value.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(pd.util.hash_pandas_object(value, index=False).values.tobytes()).hexdigest()


def _native(labels_dir: Path) -> pd.DataFrame:
    shards = sorted(labels_dir.glob("train_global_*_5_2025_0[1-6].parquet"))
    if len(shards) != 12:
        raise ContinuationError("exactly twelve Jan--Jun native side label shards are required")
    columns = [*IDENTITY, "__decision_ts__", TARGET, WEIGHT, ECONOMIC]
    out = pd.concat([pd.read_parquet(path, columns=columns) for path in shards], ignore_index=True)
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="raise")
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    if out.candidate_id.duplicated().any():
        raise ContinuationError("native candidate IDs are not unique")
    out["__feature_symbol__"] = out.candidate_id.astype(str).str.split("|", n=1).str[0]
    out["base_label_resolution_utc"] = out["__decision_ts__"] + pd.Timedelta(hours=24)
    if not out["base_label_resolution_utc"].eq(out["__ts__"] + pd.Timedelta(hours=25)).all():
        raise ContinuationError("native label timing is no longer signal+25h")
    return out


def _validation(candidates_path: Path, execution_path: Path, native: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    candidates = pd.read_parquet(candidates_path, columns=list(IDENTITY))
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    candidates = candidates.loc[candidates["__ts__"].ge(start) & candidates["__ts__"].lt(end)].copy()
    exact = pd.read_parquet(execution_path, columns=[*IDENTITY, "execution_net_ev_12h", "execution_label_end_utc", "execution_label_available_at"])
    exact["__ts__"] = pd.to_datetime(exact["__ts__"], utc=True, errors="raise")
    exact["execution_label_end_utc"] = pd.to_datetime(exact["execution_label_end_utc"], utc=True, errors="raise")
    exact["execution_label_available_at"] = pd.to_datetime(exact["execution_label_available_at"], utc=True, errors="raise")
    native_values = native.drop(columns=["side_name", "__symbol__", "__ts__"])
    native_check = native.loc[:, ["candidate_id", "side_name", "__ts__"]].rename(columns={"side_name": "_native_side", "__ts__": "_native_ts"})
    out = candidates.merge(native_values, on="candidate_id", how="left", validate="one_to_one").merge(native_check, on="candidate_id", how="left", validate="one_to_one")
    if not (out.side_name.eq(out.pop("_native_side")) & out["__ts__"].eq(out.pop("_native_ts"))).all():
        raise ContinuationError("common-universe candidate does not exactly match native base supervision")
    exact_values = exact.drop(columns=["side_name", "__symbol__", "__ts__"])
    exact_check = exact.loc[:, ["candidate_id", "side_name", "__ts__"]].rename(columns={"side_name": "_execution_side", "__ts__": "_execution_ts"})
    out = out.merge(exact_values, on="candidate_id", how="left", validate="one_to_one").merge(exact_check, on="candidate_id", how="left", validate="one_to_one")
    if not (out.side_name.eq(out.pop("_execution_side")) & out["__ts__"].eq(out.pop("_execution_ts"))).all():
        raise ContinuationError("common-universe candidate does not exactly match execution economics")
    if out[[TARGET, WEIGHT, ECONOMIC, "execution_net_ev_12h"]].isna().any().any():
        raise ContinuationError("validation has incomplete native or exact-execution labels")
    out["effective_label_resolution_utc"] = out[["base_label_resolution_utc", "execution_label_available_at"]].max(axis=1)
    return out.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _economics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    ranked = frame.sort_values("base_oof_score", ascending=False, kind="stable")
    k = max(1, int(np.ceil(len(ranked) * 0.10)))
    top = ranked.head(k)
    return {"rows": int(len(frame)), "top10_global_rows": int(k),
            "top10_global_execution_net_ev": float(top.execution_net_ev_12h.mean()),
            "top10_global_positive_fraction": float((top.execution_net_ev_12h > 0).mean()),
            "score_target_spearman": float(frame[["base_oof_score", TARGET]].corr(method="spearman").iloc[0, 1])}


def run(*, output_dir: Path = DEFAULT_OUTPUT, labels_dir: Path = DEFAULT_LABEL_DIR,
        candidates_path: Path = DEFAULT_CANDIDATES, execution_path: Path = DEFAULT_EXECUTION,
        promotion_path: Path = DEFAULT_PROMOTION, ae_root: Path = DEFAULT_AE,
        feature_store: Path = DEFAULT_STORE, month: int | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ContinuationError(f"refusing to overwrite existing continuation: {output_dir}")
    months = (month,) if month else (5, 6)
    if any(item not in (5, 6) for item in months):
        raise ContinuationError("continuation can score May and/or June 2025 only")
    native = _native(Path(labels_dir))
    contracts = _load_contracts(Path(promotion_path), Path(ae_root))
    output_dir.mkdir(parents=True)
    records: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for side_index, side in enumerate(SIDES):
        for value in months:
            start = pd.Timestamp(f"2025-{value:02d}-01", tz="UTC")
            end = start + pd.offsets.MonthBegin(1)
            valid = _validation(Path(candidates_path), Path(execution_path), native, start, end)
            valid = valid.loc[valid.side_name.eq(side)].reset_index(drop=True)
            train = native.loc[native.side_name.eq(side) & native.base_label_resolution_utc.lt(start)].copy()
            # This is the accepted deterministic cap, applied only after the
            # new chronological resolution cutoff.
            train = _deterministic_cap(train, 100_000).reset_index(drop=True)
            if train.empty or valid.empty:
                raise ContinuationError(f"{side} {value}: train or validation is empty")
            fold_dir = output_dir / side / f"month_2025_{value:02d}"
            fold_dir.mkdir(parents=True)
            train_x, train_cov = _materialize_features(train, contracts[side], Path(feature_store), fold_dir / "train_features.parquet")
            valid_x, valid_cov = _materialize_features(valid, contracts[side], Path(feature_store), fold_dir / "validation_features.parquet")
            if train_cov["exact_key_fraction"] != 1.0 or valid_cov["exact_key_fraction"] != 1.0:
                raise ContinuationError(f"{side} {value}: point-in-time feature coverage is incomplete")
            train.to_parquet(fold_dir / "train_labels.parquet", index=False, compression="zstd")
            valid.to_parquet(fold_dir / "validation_labels.parquet", index=False, compression="zstd")
            model = _lgbm_regressor(contracts[side]["params"], seed=9500 + side_index * 100 + value)
            model.fit(train_x.loc[:, list(contracts[side]["features"])], train[TARGET], sample_weight=train[WEIGHT])
            out = valid.loc[:, [*IDENTITY, "__decision_ts__", "base_label_resolution_utc", "effective_label_resolution_utc", TARGET, WEIGHT, ECONOMIC, "execution_net_ev_12h", "execution_label_end_utc"]].copy()
            out["fold_id"] = f"month_2025_{value:02d}"
            out["fold_validation_start_utc"] = start
            out["fold_validation_end_utc"] = end
            out["base_oof_score"] = model.predict(valid_x.loc[:, list(contracts[side]["features"])]).astype(np.float64)
            out["base_rank_timestamp_side"] = out.groupby("__ts__")["base_oof_score"].rank(method="first", ascending=False).astype(np.int64)
            out["base_group_rows"] = out.groupby("__ts__")["candidate_id"].transform("size").astype(np.int64)
            out["base_rank_pct_timestamp_side"] = out["base_rank_timestamp_side"] / out["base_group_rows"]
            out["selected_top40"] = out["base_rank_timestamp_side"].le(40)
            out.to_parquet(fold_dir / "oof_predictions.parquet", index=False, compression="zstd")
            records.append(out)
            folds.append({"side": side, "month": f"2025-{value:02d}", "train_rows": int(len(train)), "validation_rows": int(len(valid)),
                          "train_label_resolution_max_utc": train.base_label_resolution_utc.max().isoformat(),
                          "label_cutoff": f"native decision+24h < {start.isoformat()}", "train_feature_coverage": train_cov,
                          "validation_feature_coverage": valid_cov, "hpo_trial_id": contracts[side]["trial_id"],
                          "feature_count": len(contracts[side]["features"])})
    predictions = pd.concat(records, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable")
    predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False, compression="zstd")
    per_month = {key: _economics(value) for key, value in predictions.groupby(predictions["__ts__"].dt.strftime("%Y-%m"), sort=True)}
    contract = {"schema": SCHEMA, "status": "CONTINUATION_CONTRACT_UNVALIDATED_AGAINST_ACCEPTED_JAN_APR_SCOPE",
                "scope": "May--June 2025 frozen 30-asset common-universe; exact 1m candidate-local 12h policy economics",
                "accepted_history": "Jan--Apr artifacts remain unchanged and are not relabelled or overwritten",
                "feature_selection": "none; frozen accepted 31-long/8-short contracts", "hpo": "none; frozen accepted HPO parameters",
                "validation": "chronological month holdout with native base-label resolution strictly before month start",
                "execution_economics": "exact candidate-local 12h replay, diagnostic only for base alpha", "folds": folds,
                "inputs": {"native_label_dir": str(labels_dir), "candidates": str(candidates_path), "exact_execution_labels": str(execution_path),
                           "promotion": str(promotion_path), "ae_root": str(ae_root), "feature_store": str(feature_store)},
                "inputs_sha256": {"candidates": _sha256(Path(candidates_path)), "exact_execution_labels": _sha256(Path(execution_path)), "promotion": _sha256(Path(promotion_path))},
                "prediction_rows": int(len(predictions)), "prediction_identity_sha256": _identity_hash(predictions),
                "economics": {"aggregate": _economics(predictions), "by_month": per_month},
                "outputs": {"oof_predictions.parquet": _sha256(output_dir / "oof_predictions.parquet")}}
    _write_json(output_dir / "continuation_contract.json", contract)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--execution-labels", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--promotion", type=Path, default=DEFAULT_PROMOTION)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--month", type=int, choices=(5, 6))
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir, labels_dir=args.labels_dir, candidates_path=args.candidates,
                         execution_path=args.execution_labels, promotion_path=args.promotion, ae_root=args.ae_root,
                         feature_store=args.feature_store, month=args.month), indent=2, default=str))


if __name__ == "__main__":
    main()
