#!/usr/bin/env python3
"""Materialise a blocked-OOF July 2025 common-30 base/residual bridge.

This deliberately does not extend or relabel the accepted final ledger.  It
reuses its frozen side-local base and residual contracts, fits both layers only
on labels resolved before 2025-07-01, and writes a separately sealed,
common-universe bridge.  All model decisions are hourly; the 1m execution
replay is retained solely as a candidate-local outcome label.
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

from scripts.run_febapr2025_canonical_base_oof import (
    ECONOMIC, IDENTITY, SIDES, TARGET, WEIGHT, _deterministic_cap,
    _load_contracts, _materialize_features, _sha256,
)
from scripts.run_packb_pre_march_side_fs_hpo import _lgbm_regressor
from scripts import run_febapr2025_canonical_residual_oof as accepted_residual
from scripts.run_mayjun2025_canonical_residual_continuation import (
    _feature_matrix, _sha as residual_sha,
)

SCHEMA = "july2025_common30_final_base_residual_oof_bridge_v1"
LABEL_DIR = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
CANDIDATES = ROOT / "data_perp/artifacts/mayjul2025_execution_ev_common30_policy_inputs_20260727_v2/candidates.parquet"
EXECUTION = ROOT / "data_perp/artifacts/mayjul2025_execution_ev_common30_labels_20260727_v2/labels.parquet"
PROMOTION = ROOT / "docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json"
AE = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
STORE = ROOT / "data_perp/features/20260711_070000"
HISTORICAL = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1/population.parquet"
MAYJUN_BASE = ROOT / "data_perp/artifacts/mayjun2025_canonical_base_continuation_20260730_v1"
RESIDUAL_CONTRACT = ROOT / "data_perp/artifacts/packb_side_local_residual_oof_20260724_v1_31_8"
OUT = ROOT / "data_perp/artifacts/july2025_common30_final_base_residual_oof_bridge_20260730_v1"
RESIDUAL_TARGET = "__first_touch_capture_net__"


class BridgeError(RuntimeError):
    pass


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as h:
        tmp = Path(h.name)
        json.dump(value, h, indent=2, sort_keys=True, allow_nan=False, default=str)
        h.write("\n")
    os.replace(tmp, path)


def _identity_hash(frame: pd.DataFrame) -> str:
    value = frame.loc[:, list(IDENTITY)].copy()
    value["__ts__"] = pd.to_datetime(value["__ts__"], utc=True).astype(str)
    value = value.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(pd.util.hash_pandas_object(value, index=False).values.tobytes()).hexdigest()


def _native(labels_dir: Path) -> pd.DataFrame:
    shards = sorted(labels_dir.glob("train_global_*_5_2025_0[1-7].parquet"))
    if len(shards) != 14:
        raise BridgeError("exactly fourteen Jan--Jul native side label shards are required")
    columns = [*IDENTITY, "__decision_ts__", TARGET, WEIGHT, ECONOMIC]
    out = pd.concat([pd.read_parquet(path, columns=columns) for path in shards], ignore_index=True)
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="raise")
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    if out.candidate_id.duplicated().any():
        raise BridgeError("native candidate IDs are not unique")
    out["__feature_symbol__"] = out.candidate_id.astype(str).str.split("|", n=1).str[0]
    out["base_label_resolution_utc"] = out["__decision_ts__"] + pd.Timedelta(hours=24)
    if not out["base_label_resolution_utc"].eq(out["__ts__"] + pd.Timedelta(hours=25)).all():
        raise BridgeError("native label resolution is no longer signal+25h")
    return out


def _july_validation(candidates_path: Path, execution_path: Path, native: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2025-07-01", tz="UTC")
    end = pd.Timestamp("2025-08-01", tz="UTC")
    candidates = pd.read_parquet(candidates_path, columns=list(IDENTITY))
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    candidates = candidates.loc[candidates["__ts__"].ge(start) & candidates["__ts__"].lt(end)].copy()
    exact_columns = [*IDENTITY, "execution_net_ev_12h", "execution_label_end_utc", "execution_label_available_at"]
    exact = pd.read_parquet(execution_path, columns=exact_columns)
    exact["__ts__"] = pd.to_datetime(exact["__ts__"], utc=True, errors="raise")
    exact = exact.loc[exact["__ts__"].ge(start) & exact["__ts__"].lt(end)].copy()
    for column in ("execution_label_end_utc", "execution_label_available_at"):
        exact[column] = pd.to_datetime(exact[column], utc=True, errors="raise")
    native_values = native.drop(columns=["side_name", "__symbol__", "__ts__"])
    native_check = native.loc[:, ["candidate_id", "side_name", "__ts__"]].rename(
        columns={"side_name": "_native_side", "__ts__": "_native_ts"})
    out = candidates.merge(native_values, on="candidate_id", how="left", validate="one_to_one").merge(
        native_check, on="candidate_id", how="left", validate="one_to_one")
    if not (out.side_name.eq(out.pop("_native_side")) & out["__ts__"].eq(out.pop("_native_ts"))).all():
        raise BridgeError("July common-universe candidates do not exactly match native supervision")
    exact_values = exact.drop(columns=["side_name", "__symbol__", "__ts__"])
    exact_check = exact.loc[:, ["candidate_id", "side_name", "__ts__"]].rename(
        columns={"side_name": "_execution_side", "__ts__": "_execution_ts"})
    out = out.merge(exact_values, on="candidate_id", how="left", validate="one_to_one").merge(
        exact_check, on="candidate_id", how="left", validate="one_to_one")
    if not (out.side_name.eq(out.pop("_execution_side")) & out["__ts__"].eq(out.pop("_execution_ts"))).all():
        raise BridgeError("July common-universe candidates do not exactly match exact execution outcomes")
    required = [TARGET, WEIGHT, ECONOMIC, "execution_net_ev_12h", "execution_label_end_utc", "execution_label_available_at"]
    if out[required].isna().any().any() or len(out) != 44_640:
        raise BridgeError("July validation contains incomplete labels or unexpected candidate count")
    out["effective_label_resolution_utc"] = out[["base_label_resolution_utc", "execution_label_available_at"]].max(axis=1)
    return out.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _economics(frame: pd.DataFrame, score: str) -> dict[str, Any]:
    ranked = frame.sort_values(score, ascending=False, kind="stable")
    k = max(1, int(np.ceil(len(ranked) * .10)))
    top = ranked.head(k)
    return {"rows": int(len(frame)), "top10_global_rows": int(k),
            "top10_global_execution_net_ev": float(top.execution_net_ev_12h.mean()),
            "top10_global_positive_fraction": float((top.execution_net_ev_12h > 0).mean()),
            "score_native_target_spearman": float(frame[[score, TARGET]].corr(method="spearman").iloc[0, 1])}


def _base_training_features_with_frozen_cache(
    train: pd.DataFrame, *, side: str, route: Mapping[str, Any], feature_store: Path,
    output: Path, mayjun_base_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse only exact June-continuation PIT rows, then read the new remainder.

    The June cached training matrix was generated from the same immutable base
    feature contract.  Candidate IDs are unique, therefore alignment by ID is
    exact, and the missing July-expanded training sample is still read through
    the normal strict PIT loader.  This avoids re-reading the same 82% of the
    frozen history in a single resource-limited process.
    """
    cache_dir = Path(mayjun_base_dir) / side / "month_2025_06"
    cache_labels = pd.read_parquet(cache_dir / "train_labels.parquet", columns=["candidate_id"])
    cache_features = pd.read_parquet(cache_dir / "train_features.parquet")
    if len(cache_labels) != len(cache_features) or cache_labels.candidate_id.duplicated().any():
        raise BridgeError(f"{side}: invalid June frozen feature cache")
    cached = cache_features.copy()
    cached.index = cache_labels.candidate_id.astype(str)
    if tuple(cached.columns) != tuple(route["features"]):
        raise BridgeError(f"{side}: June cache feature contract differs from frozen July route")
    key = train.candidate_id.astype(str)
    available = key.isin(cached.index)
    missing = train.loc[~available].reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    if not missing.empty:
        fresh, fresh_cov = _materialize_features(missing, route, feature_store, output.with_name("base_train_features_missing.parquet"))
        if fresh_cov["exact_key_fraction"] != 1.0:
            raise BridgeError(f"{side}: new July-expanded base training rows lack exact PIT features")
        fresh.index = missing.candidate_id.astype(str)
        pieces.append(fresh)
    lookup = pd.concat([cached, *pieces], axis=0)
    if lookup.index.duplicated().any() or not key.isin(lookup.index).all():
        raise BridgeError(f"{side}: cache/fresh base feature reconstruction lost identity")
    matrix = lookup.loc[key, list(route["features"])].reset_index(drop=True)
    matrix.to_parquet(output, index=False, compression="zstd")
    return matrix, {"rows": int(len(matrix)), "exact_key_fraction": 1.0,
                    "frozen_june_cache_rows": int(available.sum()), "new_exact_pit_rows": int((~available).sum()),
                    "cache_source": str(cache_dir)}


def _existing_matrix(path: Path, *, rows: int, columns: tuple[str, ...] | None) -> pd.DataFrame | None:
    """Accept only an exact, previously written immutable stage matrix."""
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if len(frame) != rows or (columns is not None and tuple(frame.columns) != columns):
        raise BridgeError(f"partial stage matrix is not the requested frozen matrix: {path}")
    return frame


def run(*, output_dir: Path = OUT, labels_dir: Path = LABEL_DIR, candidates_path: Path = CANDIDATES,
        execution_path: Path = EXECUTION, promotion_path: Path = PROMOTION, ae_root: Path = AE,
        feature_store: Path = STORE, historical_path: Path = HISTORICAL,
        mayjun_base_dir: Path = MAYJUN_BASE, residual_contract: Path = RESIDUAL_CONTRACT,
        stage: str = "full") -> dict[str, Any]:
    output_dir = Path(output_dir)
    if stage not in {"base", "residual", "full", "finalize"}:
        raise BridgeError("stage must be base, residual, full, or finalize")
    if stage == "finalize":
        contract_path = output_dir / "bridge_contract.json"
        predictions_path = output_dir / "oof_predictions.parquet"
        base_path = output_dir / "base_oof_predictions.parquet"
        base_contract_path = output_dir / "base_stage_contract.json"
        required_paths = (contract_path, predictions_path, base_path, base_contract_path)
        if not all(path.is_file() for path in required_paths):
            raise BridgeError("finalize requires complete base and residual stage outputs")
        contract = json.loads(contract_path.read_text())
        if contract.get("schema") != SCHEMA or contract.get("status") != "SEALED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION":
            raise BridgeError("bridge contract is not a completed compatible bridge")
        if contract.get("outputs", {}).get("oof_predictions.parquet") != _sha256(predictions_path):
            raise BridgeError("bridge prediction checksum no longer matches its contract")
        output = pd.read_parquet(
            predictions_path,
            columns=["candidate_id", "__ts__", "side_name", "score_base_alpha", "score_residual_expected_ev"],
        )
        output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="raise")
        if (
            len(output) != 44_640
            or output["candidate_id"].duplicated().any()
            or output[["score_base_alpha", "score_residual_expected_ev"]].isna().any().any()
            or (output["__ts__"].astype("int64") % pd.Timedelta(hours=1).value != 0).any()
            or output["side_name"].value_counts().to_dict() != {"long": 22_320, "short": 22_320}
        ):
            raise BridgeError("completed bridge fails identity, cadence, side, or score validation")
        manifest = {
            **contract,
            "manifest_schema": f"{SCHEMA}_sealed_manifest",
            "manifest_status": "SEALED_CHECKSUM_VERIFIED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION",
            "outputs_sha256": {
                path.name: _sha256(path)
                for path in required_paths
            },
        }
        manifest_path = output_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        (output_dir / "manifest.sha256").write_text(
            f"{_sha256(manifest_path)}  manifest.json\n"
        )
        return manifest
    existing_files = {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()} if output_dir.exists() else set()
    partial_allowed = {"base_oof_predictions.parquet", "base_stage_contract.json"}
    partial_allowed |= {f"{side}/month_2025_07/{name}" for side in SIDES for name in (
        "base_train_features.parquet", "base_train_features_missing.parquet", "base_validation_features.parquet", "base_oof_predictions.parquet")}
    if existing_files - partial_allowed:
        raise BridgeError(f"refusing to overwrite bridge containing non-resumable files: {sorted(existing_files - partial_allowed)}")
    start = pd.Timestamp("2025-07-01", tz="UTC")
    end = pd.Timestamp("2025-08-01", tz="UTC")
    native = _native(Path(labels_dir))
    valid = _july_validation(Path(candidates_path), Path(execution_path), native)
    contracts = _load_contracts(Path(promotion_path), Path(ae_root))
    # A prior interrupted run may have created only empty fold directories.
    # Files above are forbidden; preserving directories is safe and avoids a
    # destructive cleanup step.
    output_dir.mkdir(parents=True, exist_ok=True)
    base_outputs: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for side_index, side in enumerate(SIDES):
        train = _deterministic_cap(native.loc[native.side_name.eq(side) & native.base_label_resolution_utc.lt(start)].copy(), 100_000).reset_index(drop=True)
        side_valid = valid.loc[valid.side_name.eq(side)].reset_index(drop=True)
        if train.empty or side_valid.empty or not train.base_label_resolution_utc.lt(start).all():
            raise BridgeError(f"{side}: invalid blocked-OOF base fold")
        fold_dir = output_dir / side / "month_2025_07"
        fold_dir.mkdir(parents=True, exist_ok=True)
        cached_base = _existing_matrix(fold_dir / "base_oof_predictions.parquet", rows=len(side_valid), columns=None)
        if cached_base is not None:
            base_outputs.append(cached_base)
            continue
        train_x = _existing_matrix(fold_dir / "base_train_features.parquet", rows=len(train), columns=tuple(contracts[side]["features"]))
        if train_x is None:
            train_x, train_cov = _base_training_features_with_frozen_cache(
                train, side=side, route=contracts[side], feature_store=Path(feature_store),
                output=fold_dir / "base_train_features.parquet", mayjun_base_dir=Path(mayjun_base_dir))
        else:
            train_cov = {"rows": int(len(train_x)), "exact_key_fraction": 1.0, "resumed_frozen_matrix": True}
        valid_x = _existing_matrix(fold_dir / "base_validation_features.parquet", rows=len(side_valid), columns=tuple(contracts[side]["features"]))
        if valid_x is None:
            valid_x, valid_cov = _materialize_features(side_valid, contracts[side], Path(feature_store), fold_dir / "base_validation_features.parquet")
        else:
            valid_cov = {"rows": int(len(valid_x)), "exact_key_fraction": 1.0, "resumed_frozen_matrix": True}
        if train_cov["exact_key_fraction"] != 1.0 or valid_cov["exact_key_fraction"] != 1.0:
            raise BridgeError(f"{side}: incomplete PIT base feature coverage")
        params = dict(contracts[side]["params"])
        # The accepted constructor already pins n_jobs=1.  Do not inject a
        # duplicate runtime keyword or change any learned parameter.
        model = _lgbm_regressor(params, seed=9500 + side_index * 100 + 7)
        model.fit(train_x.loc[:, list(contracts[side]["features"])], train[TARGET], sample_weight=train[WEIGHT])
        out = side_valid.copy()
        out["fold_id"] = "month_2025_07"
        out["fold_validation_start_utc"] = start
        out["fold_validation_end_utc"] = end
        out["base_oof_score"] = model.predict(valid_x.loc[:, list(contracts[side]["features"])]).astype(np.float64)
        out["score_base_alpha"] = out["base_oof_score"]
        out["base_rank_timestamp_side"] = out.groupby("__ts__")["base_oof_score"].rank(method="first", ascending=False).astype(np.int64)
        out["base_group_rows"] = out.groupby("__ts__")["candidate_id"].transform("size").astype(np.int64)
        out["base_rank_pct_timestamp_side"] = out["base_rank_timestamp_side"] / out["base_group_rows"]
        out["selected_top40"] = out["base_rank_timestamp_side"].le(40)
        out.to_parquet(fold_dir / "base_oof_predictions.parquet", index=False, compression="zstd")
        base_outputs.append(out)
        folds.append({"layer": "base", "side": side, "train_rows": int(len(train)), "validation_rows": int(len(out)),
                      "train_label_resolution_max_utc": train.base_label_resolution_utc.max().isoformat(),
                      "label_cutoff": "native decision+24h < 2025-07-01T00:00:00+00:00", "feature_count": len(contracts[side]["features"]),
                      "hpo_trial_id": contracts[side]["trial_id"], "train_feature_coverage": train_cov, "validation_feature_coverage": valid_cov})
    base = pd.concat(base_outputs, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if not base.selected_top40.all():
        raise BridgeError("common30 bridge unexpectedly contains rows outside top40")
    base_path = output_dir / "base_oof_predictions.parquet"
    if base_path.exists():
        prior = pd.read_parquet(base_path).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if _identity_hash(prior) != _identity_hash(base) or not np.array_equal(
            prior["score_base_alpha"].to_numpy(), base["score_base_alpha"].to_numpy()
        ):
            raise BridgeError("resumed base-stage score ledger disagrees with immutable per-side checkpoints")
    else:
        base.to_parquet(base_path, index=False, compression="zstd")
    if stage == "base":
        partial = {"schema": SCHEMA, "status": "BASE_STAGE_COMPLETE_RESIDUAL_PENDING", "rows": int(len(base)),
                   "identity_sha256": _identity_hash(base), "decision_timeframe": "1h", "no_2026_outcomes": True,
                   "score_fit_provenance": "base fit rows have native label resolution strictly before 2025-07-01", "folds": folds}
        _write_json(output_dir / "base_stage_contract.json", partial)
        return partial
    historical = pd.read_parquet(historical_path)
    historical["__ts__"] = pd.to_datetime(historical["__ts__"], utc=True)
    historical["native_label_resolution_utc"] = pd.to_datetime(historical.native_label_resolution_utc, utc=True)
    mayjun_contract = json.loads((Path(mayjun_base_dir) / "continuation_contract.json").read_text())
    if mayjun_contract.get("schema") != "mayjun2025_canonical_base_continuation_v1":
        raise BridgeError("May--June base continuation contract mismatch")
    mayjun = pd.read_parquet(Path(mayjun_base_dir) / "oof_predictions.parquet")
    mayjun["native_label_resolution_utc"] = pd.to_datetime(mayjun.base_label_resolution_utc, utc=True)
    base["native_label_resolution_utc"] = pd.to_datetime(base.base_label_resolution_utc, utc=True)
    residual_frame = pd.concat([historical, mayjun, base], ignore_index=True, sort=False)
    if residual_frame.candidate_id.duplicated().any():
        raise BridgeError("residual history and July bridge candidate identities overlap")
    residual_outputs: list[pd.DataFrame] = []
    for side in SIDES:
        side_frame = residual_frame.loc[residual_frame.side_name.eq(side)].reset_index(drop=True)
        train = side_frame.loc[side_frame.native_label_resolution_utc.lt(start)].reset_index(drop=True)
        side_valid = side_frame.loc[side_frame["__ts__"].ge(start) & side_frame["__ts__"].lt(end)].reset_index(drop=True)
        if train.empty or side_valid.empty or not train.native_label_resolution_utc.lt(start).all():
            raise BridgeError(f"{side}: invalid blocked-OOF residual fold")
        hp = json.loads((Path(residual_contract) / side / "hpo_contract.json").read_text())
        train_x, train_cov = _feature_matrix(train, side, Path(feature_store))
        valid_x, valid_cov = _feature_matrix(side_valid, side, Path(feature_store))
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(train.base_oof_score, train[RESIDUAL_TARGET], sample_weight=train[WEIGHT])
        expected_train = iso.predict(train.base_oof_score)
        model = lgb.LGBMRegressor(**hp["params"], n_estimators=int(hp["rounds"]), random_state=9607).fit(
            train_x, train[RESIDUAL_TARGET].to_numpy() - expected_train, sample_weight=train[WEIGHT])
        out = side_valid.copy()
        out["base_expected_ev"] = iso.predict(out.base_oof_score)
        out["residual_delta_ev"] = model.predict(valid_x)
        out["residual_expected_ev"] = out.base_expected_ev + float(hp["alpha"]) * out.residual_delta_ev
        out["score_residual_expected_ev"] = out.residual_expected_ev
        out["residual_fold"] = "month_2025_07"
        out["residual_is_oof"] = True
        residual_outputs.append(out)
        folds.append({"layer": "residual", "side": side, "train_rows": int(len(train)), "validation_rows": int(len(out)),
                      "train_label_resolution_max_utc": train.native_label_resolution_utc.max().isoformat(),
                      "label_cutoff": "native label resolution < 2025-07-01T00:00:00+00:00", "feature_selection": "none; frozen residual feature contract",
                      "hpo": "none; frozen residual HPO contract", "hpo_sha256": residual_sha(Path(residual_contract) / side / "hpo_contract.json"),
                      "train_feature_coverage": train_cov, "validation_feature_coverage": valid_cov})
    output = pd.concat(residual_outputs, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    required = ["score_base_alpha", "score_residual_expected_ev", "execution_label_end_utc", "base_label_resolution_utc"]
    if len(output) != len(valid) or output[required].isna().any().any():
        raise BridgeError("bridge output is incomplete")
    if not output["__ts__"].sort_values().diff().dropna().isin([pd.Timedelta(0), pd.Timedelta(hours=1)]).all():
        raise BridgeError("output does not preserve the hourly candidate clock")
    output.to_parquet(output_dir / "oof_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "SEALED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION",
        "scope": "July 2025, frozen 30-asset common universe; candidate identity exact within that universe, not a replacement for the wider final-ledger population",
        "decision_timeframe": "1h candidate clock and 1h model features/scores", "replay_timeframe": "nested 1m execution labels only; never a model training or assessment clock",
        "no_2026_outcomes": True, "feature_selection": "none; immutable accepted base 31-long/8-short and residual feature contracts",
        "hpo": "none; immutable accepted side-local HPO contracts", "score_fit_provenance": "base and residual fit rows have native label resolution strictly before 2025-07-01",
        "label_contract": "native first-touch target decision+24h (signal+25h); exact execution label end/availability persisted separately", "folds": folds,
        "prediction_rows": int(len(output)), "prediction_identity_sha256": _identity_hash(output),
        "metrics": {"base": _economics(output, "score_base_alpha"), "residual": _economics(output, "score_residual_expected_ev")},
        "inputs": {"native_label_dir": str(labels_dir), "candidates": str(candidates_path), "exact_execution_labels": str(execution_path), "promotion": str(promotion_path), "feature_store": str(feature_store), "historical_residual_population": str(historical_path), "mayjun_base": str(mayjun_base_dir), "residual_contract": str(residual_contract)},
        "inputs_sha256": {"candidates": _sha256(Path(candidates_path)), "exact_execution_labels": _sha256(Path(execution_path)), "promotion": _sha256(Path(promotion_path)), "historical_residual_population": residual_sha(Path(historical_path)), "mayjun_base_predictions": residual_sha(Path(mayjun_base_dir) / "oof_predictions.parquet")},
        "outputs": {"oof_predictions.parquet": _sha256(output_dir / "oof_predictions.parquet")},
    }
    _write_json(output_dir / "bridge_contract.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--labels-dir", type=Path, default=LABEL_DIR)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--execution-labels", type=Path, default=EXECUTION)
    parser.add_argument("--promotion", type=Path, default=PROMOTION)
    parser.add_argument("--ae-root", type=Path, default=AE)
    parser.add_argument("--feature-store", type=Path, default=STORE)
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--mayjun-base", type=Path, default=MAYJUN_BASE)
    parser.add_argument("--residual-contract", type=Path, default=RESIDUAL_CONTRACT)
    parser.add_argument("--stage", choices=("base", "residual", "full", "finalize"), default="full")
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir, labels_dir=args.labels_dir, candidates_path=args.candidates,
                         execution_path=args.execution_labels, promotion_path=args.promotion, ae_root=args.ae_root,
                         feature_store=args.feature_store, historical_path=args.historical, mayjun_base_dir=args.mayjun_base,
                         residual_contract=args.residual_contract, stage=args.stage), indent=2, default=str))


if __name__ == "__main__":
    main()
