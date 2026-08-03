#!/usr/bin/env python3
"""Reconstruct strict Feb--Apr 2025 canonical side-local base OOF scores.

This runner intentionally stops at the base layer.  It scores only the
accepted frozen exact-path identities, reuses the frozen 31/8 feature and HPO
contracts, and never reads an historical model-score archive.  The base target
remains the frozen first-touch soft target used to select the canonical base
models; deployed exact 12-hour execution EV is retained strictly as an
out-of-sample economic gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import transform_ae_gmm_features
from extreme_price_movements.packb_static_point_feature_loader import (
    iter_point_in_time_feature_batches,
)
from scripts.run_packb_pre_march_side_fs_hpo import _lgbm_regressor


SCHEMA = "febapr2025_canonical_base_oof_v1"
SIDES = ("long", "short")
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
TARGET = "__first_touch_target_soft__"
WEIGHT = "__w__"
ECONOMIC = "__first_touch_capture_net__"
DEFAULT_POPULATION = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2"
DEFAULT_LABEL_DIR = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
DEFAULT_EXECUTION_LABELS = ROOT / "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/labels.parquet"
DEFAULT_PROMOTION = ROOT / "docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json"
DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1"


class CanonicalBaseOOFError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalBaseOOFError(f"cannot read {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise CanonicalBaseOOFError(f"JSON object required: {path}")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _identity_hash(frame: pd.DataFrame) -> str:
    payload = frame.loc[:, list(IDENTITY)].copy()
    payload["__ts__"] = pd.to_datetime(payload["__ts__"], utc=True).astype(str)
    payload = payload.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(pd.util.hash_pandas_object(payload, index=False).values.tobytes()).hexdigest()


def _load_contracts(promotion_path: Path, ae_root: Path) -> dict[str, dict[str, Any]]:
    promotion = _json(promotion_path)
    if promotion.get("status") not in {
        "FROZEN_SIDE_ROUTED_FEATURE_SELECTION_AND_HPO",
        "FROZEN_HISTORICAL_FEATURE_EXCEPTION_WITH_STRICT_PRE_MARCH_HPO",
    }:
        raise CanonicalBaseOOFError("frozen promotion status required")
    routes: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        route = promotion.get("sides", {}).get(side)
        if not isinstance(route, Mapping):
            raise CanonicalBaseOOFError(f"promotion route missing for {side}")
        source = ROOT / str(route.get("source_root"))
        feature_path = source / "feature_contract.json"
        hpo_path = source / "hpo_parameters.json"
        if _sha256(feature_path) != route.get("feature_contract_sha256"):
            raise CanonicalBaseOOFError(f"{side} feature contract changed")
        if _sha256(hpo_path) != route.get("hpo_parameters_sha256"):
            raise CanonicalBaseOOFError(f"{side} HPO contract changed")
        feature = _json(feature_path)
        hpo = _json(hpo_path)
        features = tuple(map(str, feature.get("selected_features", ())))
        params = hpo.get("selection", {}).get("selected_params")
        if (feature.get("side") != side or hpo.get("side") != side or not features
                or not isinstance(params, Mapping)):
            raise CanonicalBaseOOFError(f"invalid frozen {side} selection contract")
        state_path = ae_root / side / "ae_gmm/ae_gmm_state.pkl"
        metadata = _json(ae_root / side / "ae_gmm/ae_gmm_state_metadata.json")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, dict) or not state.get("enabled") or state.get("packb_side_scope") != side:
            raise CanonicalBaseOOFError(f"invalid frozen {side} AE/GMM state")
        state_features = tuple(map(str, state.get("feature_columns", ())))
        generated = tuple(name for name in features if name not in state_features)
        supported_generated = {"dae_b16_02", "gmm_ood_score"}
        if set(generated) - supported_generated:
            raise CanonicalBaseOOFError(f"unsupported generated {side} fields: {generated}")
        routes[side] = {
            "features": features,
            "params": dict(params),
            "trial_id": hpo["selection"]["selected_trial_id"],
            "feature_path": feature_path,
            "hpo_path": hpo_path,
            "state_path": state_path,
            "state_sha256": _sha256(state_path),
            "state_metadata_sha256": _sha256(ae_root / side / "ae_gmm/ae_gmm_state_metadata.json"),
            "state": state,
            "state_features": state_features,
            "generated": generated,
            "metadata": metadata,
        }
    if len(routes["long"]["features"]) != 31 or len(routes["short"]["features"]) != 8:
        raise CanonicalBaseOOFError("expected frozen canonical 31-long/8-short contract")
    return routes


def _load_native_base_labels(labels_dir: Path) -> pd.DataFrame:
    """Load native base supervision; it is not an execution-label substitute."""
    shards = sorted(labels_dir.glob("train_global_*_5_2025_0[1-4].parquet"))
    if len(shards) != 8:
        raise CanonicalBaseOOFError("exactly eight Jan-Apr side native base label shards required")
    cols = [*IDENTITY, "__decision_ts__", TARGET, WEIGHT, ECONOMIC]
    native = pd.concat([pd.read_parquet(path, columns=cols) for path in shards], ignore_index=True)
    native["__ts__"] = pd.to_datetime(native["__ts__"], utc=True, errors="raise")
    native["__decision_ts__"] = pd.to_datetime(native["__decision_ts__"], utc=True, errors="raise")
    if native["candidate_id"].duplicated().any():
        raise CanonicalBaseOOFError("native base label source has duplicate candidate IDs")
    # The store files retain the slash spelling embedded in candidate_id even
    # though the historical native label ledger normalises it to underscores.
    native["__feature_symbol__"] = native["candidate_id"].astype(str).str.split("|", n=1).str[0]
    native["base_label_resolution_utc"] = native["__decision_ts__"] + pd.Timedelta(hours=24)
    if not native["base_label_resolution_utc"].eq(native["__ts__"] + pd.Timedelta(hours=25)).all():
        raise CanonicalBaseOOFError("native base label timing changed")
    return native


def _load_labels(population: pd.DataFrame, labels_dir: Path, execution_path: Path) -> pd.DataFrame:
    base = _load_native_base_labels(labels_dir)
    base = base.loc[base["__ts__"].dt.month.isin((2, 3, 4))].copy()
    execution = pd.read_parquet(execution_path, columns=[*IDENTITY, "execution_net_ev_12h", "execution_label_end_utc"])
    for frame in (base, execution):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    population = population.copy()
    population["__ts__"] = pd.to_datetime(population["__ts__"], utc=True, errors="raise")
    # Candidate IDs are the immutable shared identity.  Historical base labels
    # use the feature-store symbol spelling (``BTC_USD:USD``), while accepted
    # execution labels preserve the simulator spelling (``BTC/USD:USD``).
    # Do not equate those presentation fields during the join.
    base_check = base.loc[:, ["candidate_id", "side_name", "__ts__"]].copy()
    base_check = base_check.rename(columns={"side_name": "__base_side__", "__ts__": "__base_ts__"})
    base_values = base.drop(columns=["side_name", "__symbol__", "__ts__"])
    merged = population.loc[:, list(IDENTITY)].merge(base_values, on="candidate_id", how="left", validate="one_to_one")
    merged = merged.merge(base_check, on="candidate_id", how="left", validate="one_to_one")
    if not (merged["side_name"].eq(merged.pop("__base_side__")) & merged["__ts__"].eq(merged.pop("__base_ts__"))).all():
        raise CanonicalBaseOOFError("canonical base label candidate IDs disagree on side or signal time")
    execution_check = execution.loc[:, ["candidate_id", "side_name", "__ts__"]].rename(columns={"side_name": "__execution_side__", "__ts__": "__execution_ts__"})
    execution_values = execution.drop(columns=["side_name", "__symbol__", "__ts__"])
    merged = merged.merge(execution_values, on="candidate_id", how="left", validate="one_to_one")
    merged = merged.merge(execution_check, on="candidate_id", how="left", validate="one_to_one")
    if not (merged["side_name"].eq(merged.pop("__execution_side__")) & merged["__ts__"].eq(merged.pop("__execution_ts__"))).all():
        raise CanonicalBaseOOFError("accepted execution candidate IDs disagree on side or signal time")
    if len(merged) != len(population) or merged[[TARGET, WEIGHT, ECONOMIC, "execution_net_ev_12h"]].isna().any().any():
        raise CanonicalBaseOOFError("frozen population lacks one-to-one canonical labels")
    merged["__ts__"] = pd.to_datetime(merged["__ts__"], utc=True, errors="raise")
    merged["__decision_ts__"] = pd.to_datetime(merged["__decision_ts__"], utc=True, errors="raise")
    merged["execution_label_end_utc"] = pd.to_datetime(merged["execution_label_end_utc"], utc=True, errors="raise")
    # The canonical base label resolves one day after decision.  The accepted
    # execution label is shorter (12h), but both must be known before fitting.
    merged["effective_label_resolution_utc"] = merged[["base_label_resolution_utc", "execution_label_end_utc"]].max(axis=1)
    if not merged["base_label_resolution_utc"].eq(merged["__ts__"] + pd.Timedelta(hours=25)).all():
        raise CanonicalBaseOOFError("canonical base label timing changed")
    return merged.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _materialize_features(
    ledger: pd.DataFrame, route: Mapping[str, Any], feature_store: Path, output: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    requested = tuple(route["features"])
    # This is deliberately an exact point-in-time read.  We do not ask the
    # old contract to validate its original 2026 candidate-universe hash
    # against this frozen historical cohort; hashes are preserved in output.
    contract = _json(ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1" / str(ledger["side_name"].iloc[0]) / "loader_evidence/frozen_feature_contract.json")
    pieces: list[pd.DataFrame] = []
    matched = 0
    point_ledger = ledger.loc[:, ["candidate_id", "side_name", "__feature_symbol__", "__ts__"]].rename(columns={"__feature_symbol__": "__symbol__"})
    for batch in iter_point_in_time_feature_batches(
        point_ledger, feature_store_dir=feature_store,
        # Use the full frozen raw surface.  A generated AE/GMM field must see
        # precisely the state input order, and a reduced contract would have a
        # different content hash.  The selected raw subset is projected only
        # after this exact point-in-time read.
        feature_contract=contract,
        verify_frozen_schema=False,
        max_rows_per_batch=2048,
        max_columns_per_read=64,
    ):
        raw_full = batch.features
        # Full contract load is required for frozen AE/GMM transforms.
        if route["generated"]:
            raw = raw_full.loc[:, list(route["state_features"])]
            finite = np.isfinite(raw.to_numpy(dtype=np.float32, copy=False)).all(axis=1)
            generated = pd.DataFrame(np.nan, index=raw.index, columns=list(route["generated"]), dtype=np.float32)
            if finite.any():
                transformed = transform_ae_gmm_features(raw.loc[finite], route["state"], index=raw.index[finite])
                generated.loc[finite, list(route["generated"])] = transformed.loc[:, list(route["generated"])].to_numpy(dtype=np.float32, copy=False)
            values = pd.concat([raw_full, generated], axis=1).loc[:, list(requested)]
        else:
            values = raw_full.loc[:, list(requested)]
        # iter batches are group-local, but positions make reconstruction exact.
        values["__ledger_row__"] = batch.ledger_row_positions
        pieces.append(values)
        matched += int(batch.matched_exact_keys.sum())
    features = pd.concat(pieces, ignore_index=True).sort_values("__ledger_row__", kind="stable")
    positions = features.pop("__ledger_row__").to_numpy(dtype=np.int64)
    if not np.array_equal(positions, np.arange(len(ledger), dtype=np.int64)):
        raise CanonicalBaseOOFError("point-in-time feature rows lost identity order")
    finite_by_feature = {name: float(np.isfinite(features[name].to_numpy(dtype=np.float32)).mean()) for name in requested}
    features.to_parquet(output, index=False, compression="zstd")
    return features, {"rows": len(features), "exact_key_rows": matched, "exact_key_fraction": matched / max(len(features), 1), "finite_fraction": finite_by_feature}


def _admit_joint_complete(labels: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the frozen canonical joint-complete feature policy."""
    if len(labels) != len(features):
        raise CanonicalBaseOOFError("feature/label rows changed before admission")
    finite = np.isfinite(features.to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    admitted_labels = labels.loc[finite].reset_index(drop=True)
    admitted_features = features.loc[finite].reset_index(drop=True)
    if admitted_labels.empty:
        raise CanonicalBaseOOFError("joint-complete admission removed every row")
    return admitted_labels, admitted_features, {
        "policy": "joint_complete", "input_rows": int(len(labels)), "admitted_rows": int(len(admitted_labels)),
        "rejected_rows": int((~finite).sum()), "admitted_fraction": float(finite.mean()),
    }


def _folds(_labels: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [
        (pd.Timestamp("2025-02-01", tz="UTC"), pd.Timestamp("2025-03-01", tz="UTC")),
        (pd.Timestamp("2025-03-01", tz="UTC"), pd.Timestamp("2025-04-01", tz="UTC")),
        (pd.Timestamp("2025-04-01", tz="UTC"), pd.Timestamp("2025-05-01", tz="UTC")),
    ]


def _deterministic_cap(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    # The locked calendar contract requires calendar-month representation
    # before deterministic hashing; a single global hash would occasionally
    # discard an older month entirely as the expanding history grows.
    work = frame.copy()
    work["__calendar_month__"] = work["__ts__"].dt.strftime("%Y-%m")
    counts = work["__calendar_month__"].value_counts().sort_index()
    exact = counts.astype(float) * (float(maximum) / float(len(work)))
    allocation = np.floor(exact).astype(int)
    remainder = int(maximum - allocation.sum())
    fractional = (exact - allocation).sort_values(ascending=False, kind="stable")
    for month in fractional.index[:remainder]:
        allocation.loc[month] += 1
    selected: list[pd.DataFrame] = []
    for month, wanted in allocation.items():
        ranked = work.loc[work["__calendar_month__"].eq(month), ["candidate_id"]].copy()
        ranked["__digest__"] = ranked["candidate_id"].astype(str).map(lambda value: hashlib.sha256(value.encode()).hexdigest())
        selected.append(ranked.sort_values(["__digest__", "candidate_id"], kind="stable").head(int(wanted)))
    positions = pd.concat(selected).index
    return frame.loc[positions].sort_index(kind="stable")


def _economics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    work = frame.sort_values("base_oof_score", ascending=False, kind="stable")
    k = max(1, int(np.ceil(len(work) * 0.10)))
    top = work.head(k)
    return {
        "rows": int(len(work)),
        "mean_execution_net_ev": float(work["execution_net_ev_12h"].mean()),
        "top10_global_rows": int(k),
        "top10_global_execution_net_ev": float(top["execution_net_ev_12h"].mean()),
        "top10_global_positive_fraction": float((top["execution_net_ev_12h"] > 0).mean()),
        "score_target_spearman": float(work[["base_oof_score", TARGET]].corr(method="spearman").iloc[0, 1]),
    }


def run_phase(*, output_dir: Path, phase: str, side: str, month: str,
              population_dir: Path = DEFAULT_POPULATION, labels_dir: Path = DEFAULT_LABEL_DIR,
              execution_labels: Path = DEFAULT_EXECUTION_LABELS, promotion_path: Path = DEFAULT_PROMOTION,
              ae_root: Path = DEFAULT_AE_ROOT, feature_store: Path = DEFAULT_FEATURE_STORE) -> dict[str, Any]:
    """Execute one bounded materialise/fit phase for a resumable monthly shard."""
    if side not in SIDES or month not in {"2025_02", "2025_03", "2025_04"}:
        raise CanonicalBaseOOFError("explicit canonical side and month are required")
    folds = [item for item in _folds(pd.DataFrame()) if item[0].strftime("%Y_%m") == month]
    if len(folds) != 1:
        raise CanonicalBaseOOFError("canonical month is absent")
    start, end = folds[0]
    fold_id = f"month_{month}"
    fold_dir = Path(output_dir) / side / fold_id
    contracts = _load_contracts(promotion_path, ae_root)
    if phase in {"prepare_train", "prepare_validation"}:
        population = pd.read_parquet(population_dir / "population.parquet")
        labels = _load_labels(population, labels_dir, execution_labels)
        native = _load_native_base_labels(labels_dir)
        source = (native.loc[native["side_name"].eq(side) & native["base_label_resolution_utc"].lt(start)].copy()
                  if phase == "prepare_train" else
                  labels.loc[labels["side_name"].eq(side) & labels["__ts__"].ge(start) & labels["__ts__"].lt(end)].copy())
        if phase == "prepare_train":
            source = _deterministic_cap(source, 100_000)
        if source.empty:
            raise CanonicalBaseOOFError("selected phase has no rows")
        fold_dir.mkdir(parents=True, exist_ok=True)
        label_path = fold_dir / ("train_labels.parquet" if phase == "prepare_train" else "validation_labels.parquet")
        feature_path = fold_dir / ("train_features.parquet" if phase == "prepare_train" else "validation_features.parquet")
        if label_path.exists() or feature_path.exists():
            raise CanonicalBaseOOFError(f"refusing to overwrite completed {phase} shard")
        features, coverage = _materialize_features(source.reset_index(drop=True), contracts[side], feature_store, feature_path)
        source.to_parquet(label_path, index=False, compression="zstd")
        _write_json(fold_dir / f"{phase}_coverage.json", {
            "schema": SCHEMA, "phase": phase, "side": side, "fold_id": fold_id,
            "validation_start_utc": start.isoformat(), "validation_end_utc": end.isoformat(),
            "rows": int(len(source)), "coverage": coverage,
            "sampling": ("per-side calendar-month-stratified SHA256 candidate_id cap after full native-label eligibility" if phase == "prepare_train" else "accepted frozen scored identities only"),
            "feature_rows_sha256": _sha256(feature_path), "label_rows_sha256": _sha256(label_path),
        })
        return {"phase": phase, "side": side, "fold": fold_id, "rows": int(len(source)), "features": int(features.shape[1])}
    if phase != "fit":
        raise CanonicalBaseOOFError("phase must be prepare_train, prepare_validation, or fit")
    train = pd.read_parquet(fold_dir / "train_labels.parquet")
    valid = pd.read_parquet(fold_dir / "validation_labels.parquet")
    train_x = pd.read_parquet(fold_dir / "train_features.parquet")
    valid_x = pd.read_parquet(fold_dir / "validation_features.parquet")
    if len(train) != len(train_x) or len(valid) != len(valid_x):
        raise CanonicalBaseOOFError("persisted fold feature/label alignment changed")
    if not pd.to_datetime(train["base_label_resolution_utc"], utc=True).lt(start).all():
        raise CanonicalBaseOOFError("persisted native training purge failed")
    model = _lgbm_regressor(contracts[side]["params"], seed=9100 + SIDES.index(side) * 100 + int(month[-2:]))
    model.fit(train_x.loc[:, list(contracts[side]["features"])], train[TARGET], sample_weight=train[WEIGHT])
    out = valid.loc[:, [*IDENTITY, "__decision_ts__", "base_label_resolution_utc", "effective_label_resolution_utc", TARGET, WEIGHT, ECONOMIC, "execution_net_ev_12h"]].copy()
    out["fold_id"] = fold_id
    out["fold_validation_start_utc"] = start
    out["fold_validation_end_utc"] = end
    out["base_oof_score"] = model.predict(valid_x.loc[:, list(contracts[side]["features"])]).astype(np.float64)
    out.to_parquet(fold_dir / "oof_predictions.parquet", index=False, compression="zstd")
    econ = _economics(out)
    _write_json(fold_dir / "manifest.json", {
        "schema": SCHEMA, "status": "MATERIALIZED_BASE_ONLY_MONTHLY_OOF_SHARD",
        "side": side, "fold_id": fold_id, "prediction_rows": int(len(out)),
        "target": TARGET, "weight": WEIGHT, "native_economic": ECONOMIC,
        "execution_economic_diagnostic": "execution_net_ev_12h accepted exact-policy label",
        "purge_rule": "native base decision+24h < validation start",
        "hpo_trial_id": contracts[side]["trial_id"], "features": list(contracts[side]["features"]),
        "feature_contract_sha256": _sha256(contracts[side]["feature_path"]),
        "hpo_parameters_sha256": _sha256(contracts[side]["hpo_path"]), "ae_gmm_state_sha256": contracts[side]["state_sha256"],
        "economics": econ, "outputs": {"oof_predictions.parquet": _sha256(fold_dir / "oof_predictions.parquet")},
    })
    return {"phase": phase, "side": side, "fold": fold_id, "rows": int(len(out)), "economics": econ}


def run(*, output_dir: Path = DEFAULT_OUTPUT, population_dir: Path = DEFAULT_POPULATION,
        labels_dir: Path = DEFAULT_LABEL_DIR, execution_labels: Path = DEFAULT_EXECUTION_LABELS,
        promotion_path: Path = DEFAULT_PROMOTION, ae_root: Path = DEFAULT_AE_ROOT,
        feature_store: Path = DEFAULT_FEATURE_STORE, side_filter: str | None = None,
        month_filter: str | None = None) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists():
        raise CanonicalBaseOOFError(f"refusing to overwrite output: {destination}")
    gate_path = population_dir / "population_gate.json"
    gate = _json(gate_path)
    population = pd.read_parquet(population_dir / "population.parquet")
    if not population["eligible_for_fresh_canonical_base_oof"].all() or len(population) != 509868:
        raise CanonicalBaseOOFError("accepted frozen Feb-Apr common identities required")
    contracts = _load_contracts(promotion_path, ae_root)
    labels = _load_labels(population, labels_dir, execution_labels)
    native = _load_native_base_labels(labels_dir)
    if not labels["candidate_id"].isin(native["candidate_id"]).all():
        raise CanonicalBaseOOFError("accepted scoring identities are absent from native base labels")
    destination.mkdir(parents=True)
    all_predictions: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    selected_sides = (side_filter,) if side_filter is not None else SIDES
    if any(side not in SIDES for side in selected_sides):
        raise CanonicalBaseOOFError("side_filter must be long or short")
    selected_folds = [
        item for item in _folds(labels)
        if month_filter is None or item[0].strftime("%Y_%m") == month_filter
    ]
    if not selected_folds:
        raise CanonicalBaseOOFError("month_filter selected no canonical monthly fold")
    for side_number, side in enumerate(selected_sides):
        side_dir = destination / side
        side_dir.mkdir()
        side_labels = labels.loc[labels["side_name"].eq(side)].reset_index(drop=True)
        side_native = native.loc[native["side_name"].eq(side)].reset_index(drop=True)
        coverage[side] = {"folds": []}
        for fold_number, (start, end) in enumerate(selected_folds, start=1):
            valid_mask = side_labels["__ts__"].ge(start) & side_labels["__ts__"].lt(end)
            train = _deterministic_cap(side_native.loc[side_native["base_label_resolution_utc"].lt(start)].copy(), 100_000).reset_index(drop=True)
            valid = side_labels.loc[valid_mask].reset_index(drop=True)
            if train.empty or valid.empty:
                continue
            if not train["base_label_resolution_utc"].lt(start).all():
                raise CanonicalBaseOOFError("label-resolution purge failed")
            # Per-fold materialisation bounds peak memory and proves that both
            # the capped native training sample and frozen scored identities
            # were read directly from the point-in-time store.
            fold_id = f"month_{start.strftime('%Y_%m')}"
            fold_dir = side_dir / fold_id
            fold_dir.mkdir()
            train_x, train_coverage = _materialize_features(
                train, contracts[side], feature_store, fold_dir / "train_features.parquet"
            )
            valid_x, valid_coverage = _materialize_features(
                valid, contracts[side], feature_store, fold_dir / "validation_features.parquet"
            )
            if train_coverage["exact_key_fraction"] != 1.0 or valid_coverage["exact_key_fraction"] != 1.0:
                raise CanonicalBaseOOFError(f"{side} feature exact-key coverage is incomplete")
            coverage[side]["folds"].append({"fold_id": fold_id, "train": train_coverage, "validation": valid_coverage, "admission_policy": "label_complete_rows_lightgbm_native_nan_no_imputation"})
            model = _lgbm_regressor(contracts[side]["params"], seed=9100 + side_number * 100 + fold_number)
            model.fit(train_x, train[TARGET], sample_weight=train[WEIGHT])
            predicted = model.predict(valid_x).astype(np.float64)
            out = valid.loc[:, [*IDENTITY, "__decision_ts__", "base_label_resolution_utc", "effective_label_resolution_utc", TARGET, WEIGHT, ECONOMIC, "execution_net_ev_12h"]].copy()
            out["fold_id"] = fold_id
            out["fold_validation_start_utc"] = start
            out["fold_validation_end_utc"] = end
            out["base_oof_score"] = predicted
            all_predictions.append(out)
            fold_records.append({
                "side": side, "fold_id": fold_id,
                "validation_start_utc": start.isoformat(), "validation_end_utc": end.isoformat(),
                "train_rows_before_feature_admission": int(len(train)), "validation_rows_before_feature_admission": int(len(valid)),
                "train_base_label_resolution_max_utc": train["base_label_resolution_utc"].max().isoformat(),
                "purge_rule": "native base label resolution < validation start; accepted execution labels only evaluate frozen scored rows",
                "train_sampling": "per-side calendar-month-stratified SHA256 candidate_id cap after full native-label eligibility",
                "features": list(contracts[side]["features"]), "hpo_trial_id": contracts[side]["trial_id"],
            })
    predictions = pd.concat(all_predictions, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable")
    if predictions["candidate_id"].duplicated().any():
        raise CanonicalBaseOOFError("OOF predictions overlap")
    predictions.to_parquet(destination / "oof_predictions.parquet", index=False, compression="zstd")
    fold_frame = pd.DataFrame(fold_records)
    fold_frame.to_parquet(destination / "fold_provenance.parquet", index=False, compression="zstd")
    by_side = {side: _economics(predictions.loc[predictions.side_name.eq(side)]) for side in SIDES}
    economic_gate = {"global": _economics(predictions), "by_side": by_side,
                     "latest_month": _economics(predictions.loc[predictions["__ts__"].dt.month.eq(4)])}
    _write_json(destination / "coverage_gate.json", {
        "schema": SCHEMA, "status": "PASS" if all(
            item["exact_key_fraction"] == 1.0
            for side_coverage in coverage.values()
            for fold_coverage in side_coverage["folds"]
            for item in (fold_coverage["train"], fold_coverage["validation"])
        ) else "FAIL",
        "coverage": coverage, "economics": economic_gate,
        "interpretation": "base target is frozen first-touch soft target; execution net EV is a held-out accepted-label economic diagnostic only",
    })
    manifest = {
        "schema": SCHEMA, "status": "MATERIALIZED_BASE_ONLY_STRICT_EXPANDING_OOF",
        "scope": "accepted Feb-Apr 2025 frozen canonical exact-path identities only; no invalidated historical prediction archive",
        "population_gate": str(gate_path), "population_gate_sha256": _sha256(gate_path),
        "population_identity_sha256": _identity_hash(population), "population_rows": int(len(population)),
        "source_label_dir": str(labels_dir), "execution_label_path": str(execution_labels), "execution_label_sha256": _sha256(execution_labels),
        "promotion_path": str(promotion_path), "promotion_sha256": _sha256(promotion_path),
        "feature_store": str(feature_store), "fold_count": int(len(fold_records)), "prediction_rows": int(len(predictions)),
        "side_filter": side_filter, "month_filter": month_filter,
        "contracts": {side: {"features": list(route["features"]), "hpo_trial_id": route["trial_id"], "feature_contract_sha256": _sha256(route["feature_path"]), "hpo_parameters_sha256": _sha256(route["hpo_path"]), "ae_gmm_state_sha256": route["state_sha256"], "ae_gmm_state_metadata_sha256": route["state_metadata_sha256"]} for side, route in contracts.items()},
        "label_purge": "native base label decision+24h < fold validation start; accepted execution labels are only required for frozen scored-row economics",
        "outputs": {name: _sha256(destination / name) for name in ("oof_predictions.parquet", "fold_provenance.parquet", "coverage_gate.json")},
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--population-dir", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--execution-labels", type=Path, default=DEFAULT_EXECUTION_LABELS)
    parser.add_argument("--promotion", type=Path, default=DEFAULT_PROMOTION)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--side", choices=SIDES)
    parser.add_argument("--month", choices=("2025_02", "2025_03", "2025_04"))
    parser.add_argument("--phase", choices=("prepare_train", "prepare_validation", "fit"))
    args = parser.parse_args()
    if args.phase is not None:
        if args.side is None or args.month is None:
            parser.error("--phase requires --side and --month")
        result = run_phase(output_dir=args.output_dir, phase=args.phase, side=args.side, month=args.month,
                           population_dir=args.population_dir, labels_dir=args.labels_dir,
                           execution_labels=args.execution_labels, promotion_path=args.promotion,
                           ae_root=args.ae_root, feature_store=args.feature_store)
    else:
        result = run(output_dir=args.output_dir, population_dir=args.population_dir, labels_dir=args.labels_dir,
                     execution_labels=args.execution_labels, promotion_path=args.promotion,
                     ae_root=args.ae_root, feature_store=args.feature_store,
                     side_filter=args.side, month_filter=args.month)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
