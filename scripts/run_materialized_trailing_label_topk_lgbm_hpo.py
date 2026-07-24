#!/usr/bin/env python3
"""Top-k LGBM HPO for materialized trailing-profit first-touch labels."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from lightgbm import LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - dependency check.
    LGBMRegressor = None
    _LIGHTGBM_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional fast schema reader.
    pa = None
    pq = None

from scripts.run_first_touch_label_training_smoke import (  # noqa: E402
    _first_touch_eval_metrics,
    _target_from_frame,
)
from scripts.run_label_feature_store_model_smoke import (  # noqa: E402
    DEFAULT_AE_GMM_STATE_FEATURE_GMM_MAX_TRAIN_ROWS,
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER,
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS,
    _ae_gmm_smoke_feature_policy_columns,
    _append_fold_ae_gmm_state_features,
    _fit_ae_gmm_state_for_rows,
    _fold_ae_gmm_economic_targets,
    _persist_ae_gmm_state_artifact,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
)
from scripts.run_label_weighted_proxy_ablation import WEIGHT_ARMS, _effective_sample_size, _weight_series  # noqa: E402
from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    AE_GMM_CYCLE_CONTRACT_VERSION,
    AE_GMM_FEATURE_COLUMNS,
    ae_gmm_cycle_reference_indices,
    ae_gmm_cycle_sample_identity_hash,
    ae_gmm_learned_transform_hash,
    fit_ae_gmm_state,
    load_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)
from extreme_price_movements.base_side_target_contract import (  # noqa: E402
    TARGET_MODE as PROMOTED_SIDE_TARGET_MODE,
    build_promoted_side_target,
    promoted_side_target_provenance,
)
from extreme_price_movements.feature_transform_contract import (  # noqa: E402
    build_model_input_numeric_contract,
    model_matrix_hash,
)
from extreme_price_movements.lgbm_pipeline import (  # noqa: E402
    BASE_SINGLE_CYCLE_MDA_SELECTION_CONTRACT,
    LGBM_TWO_PHASE_SELECTION_CONTRACT,
    LGBM_TWO_PHASE_FULL_FIT_ROW_CAP,
    LGBM_HPO_SAMPLE_ROWS,
    LGBM_TWO_PHASE_SELECTION_SAMPLE_ROWS,
    _recent_feature_coverage_survivors,
    canonical_base_feature_selection_recipe,
    cumulative_positive_mda_keep_count,
    materialize_bme_parquet_sample,
    train_lgbm_stability_candidate,
    use_canonical_two_phase_feature_selection,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/materialized_trailing_label_topk_lgbm_hpo_v1")
DEFAULT_FIXED_PARAMS_JSON = Path("docs/promoted_s59_singlecycle_base_params.json")
DEFAULT_FIXED_SELECTED_FEATURES_CSV = None
DEFAULT_FIXED_AE_GMM_STATE_PKL = Path(
    "data_perp/reports/s59_h5_singlecycle_aegmm_bme_fs_fixedparams_wf30_20260716_v1/"
    "ae_gmm_states/cycle__global_state.pkl"
)
TOP_FRACS = (0.10, 0.20, 0.30, 0.40)
BASE_TO_META_HANDOFF_RANK_SCOPE = "timestamp_side"
MODEL_INPUT_PARITY_SCHEMA = "base_oos_model_input_parity_v1"
MODEL_INPUT_PARITY_ANCHOR_ROWS = 12
BASE_OOF_PROVENANCE_SCHEMA = "base_oof_fold_provenance_v1"
BASE_OOF_LABEL_PATH_TIMEFRAME = pd.Timedelta(minutes=15)


def _rank_top_indices_by_side(
    pred: pd.Series | np.ndarray,
    side: pd.Series | np.ndarray,
    top_frac: float,
) -> np.ndarray:
    """Return the top fraction independently within long and short streams."""

    scores = np.asarray(pred, dtype=np.float64)
    sides = np.asarray(pd.to_numeric(pd.Series(side), errors="coerce"), dtype=np.float64)
    if len(scores) != len(sides):
        raise ValueError("Prediction and side arrays must have equal length")
    selected: list[np.ndarray] = []
    normalized_side = np.where(sides < 0.0, -1, 1).astype(np.int8)
    for side_value in (-1, 1):
        positions = np.flatnonzero(normalized_side == side_value)
        if not len(positions):
            continue
        local = _rank_top_indices(scores[positions], float(top_frac))
        if len(local):
            selected.append(positions[local])
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def _timestamp_side_ranks(
    frame: pd.DataFrame,
    pred: pd.Series | np.ndarray,
    side: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Rank scores within decision timestamp and side with stable symbol ties."""

    scores = np.asarray(pred, dtype=np.float64)
    sides = np.asarray(pd.to_numeric(pd.Series(side), errors="coerce"), dtype=np.float64)
    if len(frame) != len(scores) or len(scores) != len(sides):
        raise ValueError("Timestamp-side rank inputs must have equal length")
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    symbols = frame.get("__symbol__", pd.Series("", index=frame.index)).astype(str)
    work = pd.DataFrame(
        {
            "_pos": np.arange(len(frame), dtype=np.int64),
            "_ts": ts.to_numpy(),
            "_side": np.where(sides < 0.0, "short", "long"),
            "_score": np.nan_to_num(scores, nan=-np.inf),
            "_symbol": symbols.to_numpy(copy=False),
        }
    ).sort_values(
        ["_ts", "_side", "_score", "_symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    grouped = work.groupby(["_ts", "_side"], sort=False, dropna=False)
    work["rank"] = grouped.cumcount().add(1).astype(np.int32)
    work["group_rows"] = grouped["_pos"].transform("size").astype(np.int32)
    work["rank_pct"] = (
        work["rank"].to_numpy(dtype=np.float64)
        / work["group_rows"].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    cutoff = grouped["_score"].transform("min").astype(np.float32)
    work["group_min_score"] = cutoff
    return work.sort_values("_pos", kind="mergesort").reset_index(drop=True)
TARGET_MODES = (
    "policy_soft",
    "target_soft",
    "exec_guarded_policy",
    "clean_exec",
    "time_decay_policy",
    "side_continuous_geometry_v1",
    "p90_trailing_blend",
)


def _train_valid_availability_survivors(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    *,
    min_train_non_neutral_frac: float = 0.01,
    neutral_atol: float = 1e-8,
) -> tuple[list[str], dict[str, Any]]:
    """Reject tail-store outages before AE/GMM and feature selection.

    A feature that is materially populated in training but exactly neutral for
    the whole validation window usually indicates an incomplete feature-store
    tail. Keeping it makes OOS replay and live inference use different input
    domains even when their model and policy artifacts match.
    """
    common = [str(c) for c in x_train.columns if str(c) in x_valid.columns]
    survivors: list[str] = []
    collapsed: list[str] = []
    for feature in common:
        train_values = pd.to_numeric(x_train[feature], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        valid_values = pd.to_numeric(x_valid[feature], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        train_finite = train_values[np.isfinite(train_values)]
        valid_finite = valid_values[np.isfinite(valid_values)]
        if train_finite.size == 0 or valid_finite.size == 0:
            collapsed.append(feature)
            continue
        train_non_neutral = float(
            np.mean(np.abs(train_finite) > float(neutral_atol))
        )
        valid_is_neutral = bool(
            np.all(np.abs(valid_finite) <= float(neutral_atol))
        )
        if (
            train_non_neutral >= float(min_train_non_neutral_frac)
            and valid_is_neutral
        ):
            collapsed.append(feature)
            continue
        survivors.append(feature)
    return survivors, {
        "enabled": True,
        "checked_features": int(len(common)),
        "surviving_features": int(len(survivors)),
        "collapsed_tail_features": list(collapsed),
        "collapsed_tail_feature_count": int(len(collapsed)),
        "min_train_non_neutral_frac": float(min_train_non_neutral_frac),
        "neutral_atol": float(neutral_atol),
    }


def _cycle_reference_input_survivors(
    *,
    frame: pd.DataFrame,
    ts_utc: pd.Series,
    reference_window: dict[str, Any],
    candidate_features: Sequence[str],
    payload_max_train_rows: int,
) -> tuple[list[str], dict[str, Any]]:
    """Apply the reference-fold availability contract before fitting AE/GMM.

    The frozen representation cannot be fitted on columns that the reference
    validation tail cannot materialize. This preflight mirrors the downstream
    model basket checks and prevents fitting a cycle state that its own OOS
    transform would immediately reject.
    """
    candidates = [
        str(feature)
        for feature in dict.fromkeys(candidate_features)
        if str(feature) in frame.columns
    ]
    if len(candidates) < 2:
        return [], {
            "candidate_feature_count": int(len(candidates)),
            "surviving_feature_count": 0,
            "reason": "insufficient_available_candidates",
        }

    train_mask = ts_utc.lt(reference_window["valid_start"])
    if reference_window.get("train_start") is not None:
        train_mask = train_mask & ts_utc.ge(reference_window["train_start"])
    valid_mask = ts_utc.ge(reference_window["valid_start"]) & ts_utc.lt(
        reference_window["valid_end"]
    )
    train_positions = np.flatnonzero(train_mask.to_numpy(dtype=bool, copy=False))
    if int(payload_max_train_rows) > 0 and len(train_positions) > int(
        payload_max_train_rows
    ):
        train_positions = train_positions[
            _time_spread_cap_rows(len(train_positions), int(payload_max_train_rows))
        ]
    valid_positions = np.flatnonzero(valid_mask.to_numpy(dtype=bool, copy=False))
    if len(train_positions) < 500 or len(valid_positions) < 100:
        return [], {
            "candidate_feature_count": int(len(candidates)),
            "surviving_feature_count": 0,
            "train_rows": int(len(train_positions)),
            "valid_rows": int(len(valid_positions)),
            "reason": "insufficient_reference_rows",
        }

    x_train = frame.iloc[train_positions].loc[:, candidates].replace(
        [np.inf, -np.inf], np.nan
    )
    x_valid = frame.iloc[valid_positions].loc[:, candidates].replace(
        [np.inf, -np.inf], np.nan
    )
    availability_survivors, availability_diag = (
        _train_valid_availability_survivors(x_train, x_valid)
    )
    if len(availability_survivors) < 2:
        return [], {
            "candidate_feature_count": int(len(candidates)),
            "surviving_feature_count": 0,
            "train_rows": int(len(train_positions)),
            "valid_rows": int(len(valid_positions)),
            "availability": availability_diag,
            "reason": "reference_tail_availability_rejected_all",
        }

    coverage_survivors, coverage_diag = _recent_feature_coverage_survivors(
        x_train.loc[:, availability_survivors],
        frame.iloc[train_positions]["__ts__"].to_numpy(),
        require_joint_complete_case=True,
        min_feature_coverage=0.90,
        coverage_scope="all_post_warmup",
        warmup_days=30,
        warmup_reference_start=frame.iloc[train_positions]["__ts__"].min(),
    )
    survivors = [
        feature for feature in availability_survivors if feature in coverage_survivors
    ]
    return survivors, {
        "enabled": True,
        "candidate_feature_count": int(len(candidates)),
        "surviving_feature_count": int(len(survivors)),
        "train_rows": int(len(train_positions)),
        "valid_rows": int(len(valid_positions)),
        "availability": availability_diag,
        "joint_post_warmup_coverage": coverage_diag,
        "reason": "ok" if len(survivors) >= 2 else "insufficient_joint_coverage",
    }

AE_GMM_INPUT_POLICY = os.environ.get("EPM_LGBM_AE_GMM_INPUT_POLICY", "a0bis").strip().lower()
AE_GMM_A0BIS_MOMENTUM_TOKENS = (
    "lr_",
    "ret",
    "return",
    "trend",
    "mom",
    "adx",
    "impulse",
    "breakout",
    "z_r",
    "zr_",
    "convexity",
    "slope",
    "velocity",
    "speed",
    "thrust",
)
AE_GMM_A0BIS_NORMALIZED_TOKENS = (
    "atr",
    "vol_norm",
    "_z",
    "z_",
    "cp_z",
    "ts_resid",
    "ratio",
    "rank",
    "pct",
    "tanh",
    "bps",
    "rsi",
    "autocorr",
)


def _safe_artifact_stem(value: Any) -> str:
    text = str(value)
    stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    return stem.strip("_") or "artifact"


def _feature_contract_hash(feature_names: list[str]) -> str:
    payload = json.dumps(list(feature_names), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_input_row_keys(valid: pd.DataFrame) -> pd.DataFrame:
    """Return the immutable OOS identity used to bind persisted input hashes."""

    required = ["__ts__", "__symbol__", "side"]
    missing = [name for name in required if name not in valid.columns]
    if missing:
        raise ValueError(f"Model input parity requires row keys: {missing}")
    keys = valid.loc[:, required].copy()
    keys["__ts__"] = pd.to_datetime(keys["__ts__"], utc=True, errors="coerce")
    keys["__symbol__"] = keys["__symbol__"].astype(str)
    keys["side"] = pd.to_numeric(keys["side"], errors="coerce").astype("Int8")
    if keys.isna().any(axis=None) or keys.duplicated(keep=False).any():
        raise ValueError("Model input parity row keys are invalid or non-unique")
    return keys


def _deterministic_anchor_positions(row_count: int, max_rows: int) -> np.ndarray:
    if row_count <= 0 or max_rows <= 0:
        return np.asarray([], dtype=np.int64)
    take = min(int(row_count), int(max_rows))
    return np.unique(
        np.floor(np.linspace(0, row_count - 1, num=take)).astype(np.int64)
    )


def _model_input_row_hashes(matrix: pd.DataFrame) -> list[str]:
    values = matrix.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Cannot persist non-finite model input parity matrix")
    feature_hash = _feature_contract_hash([str(name) for name in matrix.columns])
    return [
        hashlib.sha256(feature_hash.encode("ascii") + row.tobytes()).hexdigest()
        for row in np.ascontiguousarray(values)
    ]


def _persist_oos_model_input_parity(
    *,
    parity_root: Path,
    fold: str,
    valid: pd.DataFrame,
    x_valid: pd.DataFrame,
    valid_sides: np.ndarray,
    feature_contracts: Mapping[str, Sequence[str]],
    model_side_scope: str,
    anchor_rows: int = MODEL_INPUT_PARITY_ANCHOR_ROWS,
) -> dict[str, Any]:
    """Persist exact scored matrices as compact hashes plus B/M/E anchors.

    ``x_valid`` is the already materialized scorer input.  In production it has
    crossed the fold-cache float16 boundary before this function is called; this
    sidecar deliberately observes that matrix rather than rebuilding features.
    """

    scope = str(model_side_scope).strip().lower()
    if scope not in {"shared", "per_side"}:
        raise ValueError(f"Unknown model_side_scope: {model_side_scope!r}")
    keys = _model_input_row_keys(valid)
    sides = np.asarray(valid_sides, dtype=str)
    if len(keys) != len(x_valid) or len(sides) != len(x_valid):
        raise ValueError("Model input parity rows are not aligned with scored inputs")
    if scope == "shared":
        if "shared" not in feature_contracts:
            raise ValueError("Shared model input parity requires the shared contract")
        side_contracts = {"shared": list(feature_contracts["shared"])}
    else:
        observed_sides = set(sides)
        missing_contracts = sorted(observed_sides.difference(feature_contracts))
        if missing_contracts:
            raise ValueError(
                "Per-side model input parity is missing contracts: "
                f"{missing_contracts}"
            )
        side_contracts = {
            side: list(feature_contracts[side])
            for side in ("long", "short")
            if side in observed_sides
        }
    fold_dir = parity_root / _safe_fold_name(str(fold))
    fold_dir.mkdir(parents=True, exist_ok=True)
    hash_frames: list[pd.DataFrame] = []
    anchor_frames: list[pd.DataFrame] = []
    contracts: dict[str, dict[str, Any]] = {}
    for model_side, columns in side_contracts.items():
        positions = (
            np.arange(len(x_valid), dtype=np.int64)
            if model_side == "shared"
            else np.flatnonzero(sides == model_side)
        )
        if not len(positions):
            continue
        missing = [str(name) for name in columns if str(name) not in x_valid.columns]
        if missing:
            raise ValueError(
                f"Model input parity missing {model_side} features: {missing[:20]}"
            )
        matrix = x_valid.iloc[positions].loc[:, columns].reset_index(drop=True)
        contract = build_model_input_numeric_contract(
            matrix.columns,
            reference_matrix_hash=model_matrix_hash(
                matrix, row_ids=keys.iloc[positions].reset_index(drop=True)
            ),
        ).asdict()
        feature_hash = _feature_contract_hash([str(name) for name in matrix.columns])
        keyed_hashes = keys.iloc[positions].reset_index(drop=True)
        keyed_hashes["model_side"] = str(model_side)
        keyed_hashes["feature_contract_hash"] = feature_hash
        keyed_hashes["numeric_contract_hash"] = str(contract["contract_hash"])
        keyed_hashes["model_input_row_hash"] = _model_input_row_hashes(matrix)
        hash_frames.append(keyed_hashes)

        order = np.lexsort(
            (
                keyed_hashes["side"].to_numpy(dtype=np.int8),
                keyed_hashes["__symbol__"].to_numpy(dtype=str),
                keyed_hashes["__ts__"].astype("int64").to_numpy(),
            )
        )
        anchor_positions = order[_deterministic_anchor_positions(len(order), anchor_rows)]
        anchors = keyed_hashes.iloc[anchor_positions].drop(
            columns=["model_input_row_hash"]
        ).reset_index(drop=True)
        for name in matrix.columns:
            anchors[str(name)] = matrix.iloc[anchor_positions][name].to_numpy(
                dtype=np.float32, copy=False
            )
        anchor_frames.append(anchors)
        contracts[str(model_side)] = {
            "feature_names": [str(name) for name in matrix.columns],
            "feature_contract_hash": feature_hash,
            "numeric_contract": contract,
            "rows": int(len(matrix)),
            "matrix_hash": str(contract["reference_matrix_hash"]),
        }
    if not hash_frames:
        raise ValueError("Model input parity received no scored rows")
    hashes = pd.concat(hash_frames, ignore_index=True)
    anchors = pd.concat(anchor_frames, ignore_index=True)
    hashes_path = fold_dir / "row_hashes.parquet"
    anchors_path = fold_dir / "anchors.parquet"
    hashes.to_parquet(hashes_path, index=False, compression="zstd", compression_level=5)
    anchors.to_parquet(anchors_path, index=False, compression="zstd", compression_level=5)
    manifest = {
        "schema": MODEL_INPUT_PARITY_SCHEMA,
        "fold": str(fold),
        "row_key_columns": ["__ts__", "__symbol__", "side"],
        "model_side_scope": scope,
        "anchor_policy": "stable_timestamp_symbol_side_beginning_middle_end_v1",
        "anchor_rows_per_model_side": int(anchor_rows),
        "contracts_by_model_side": contracts,
        "row_hashes_path": str(hashes_path),
        "anchors_path": str(anchors_path),
    }
    manifest_path = fold_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def _finite_median_or_zero(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    finite_values = numeric.dropna()
    if finite_values.empty:
        return 0.0
    median = finite_values.median()
    return float(median) if pd.notna(median) and np.isfinite(median) else 0.0


def _reuse_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable identity for immutable inputs behind a reusable artifact."""

    encoded = json.dumps(
        _json_safe(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_target_weight_provenance(
    *, target_mode: str, weight_arm: str
) -> dict[str, Any]:
    """Serialize the exact target and weighting contract used by base.

    The promoted side-geometry target already owns a richer contract.  The
    incumbent soft-label path still needs explicit provenance so strict meta
    training can reproduce W7 (or another named base weight arm) rather than
    silently interpreting it as target-strength weighting.
    """

    if str(target_mode) == PROMOTED_SIDE_TARGET_MODE:
        return promoted_side_target_provenance()
    target_contract = {
        "schema": "base_soft_label_contract_v1",
        "target_column": "__first_touch_target_soft__",
        "target_mode": str(target_mode),
        "source": "base_scoring_target_from_frame",
    }
    weight_spec = {
        "schema": "base_weight_arm_v1",
        "weight_arm": str(weight_arm),
        "source": "base_weight_series",
    }
    return {
        "base_target_contract": target_contract,
        "base_target_contract_hash": _reuse_fingerprint(target_contract),
        "base_sample_weight_spec": weight_spec,
        "base_sample_weight_spec_hash": _reuse_fingerprint(weight_spec),
    }


def _file_identity(
    path: Path | None, *, include_sha256: bool = True
) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "status": "missing"}
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        identity["sha256"] = _sha256_file(resolved)
    return identity


def _label_source_identity(labels_path: Path) -> dict[str, Any]:
    path = Path(labels_path)
    files = _canonical_label_files(path)
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"No parquet label files found under {path}")
    return {
        "path": str(path.resolve()),
        "file_signature": _source_files_signature(files),
        "file_count": int(len(files)),
        "files": [str(file.name) for file in files],
    }


_MONTHLY_LABEL_PARTITION = re.compile(
    r"^train_global_(?:long|short)_\d+_\d{4}_\d{2}\.parquet$"
)


def _canonical_label_files(labels_path: Path) -> list[Path]:
    """Resolve one unambiguous label population from a materialized store.

    Modern causal label stores are month-partitioned.  Some directories retain
    legacy unsuffixed parquet files for provenance; globbing both populations
    silently adds stale horizons and duplicate keys.  Once monthly partitions
    exist they are the authoritative population.  Legacy-only stores retain
    the historical all-parquet behavior.
    """

    path = Path(labels_path)
    if path.is_file():
        files = [path]
    else:
        all_files = sorted(path.glob("*.parquet"))
        monthly = [
            file for file in all_files if _MONTHLY_LABEL_PARTITION.fullmatch(file.name)
        ]
        files = monthly or all_files
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"No parquet label files found under {path}")
    return files


def _load_canonical_labels(labels_path: Path) -> pd.DataFrame:
    files = _canonical_label_files(labels_path)
    frames = [pd.read_parquet(file) for file in files]
    out = (
        pd.concat(frames, ignore_index=True, copy=False)
        if len(frames) > 1
        else frames[0].copy()
    )
    if "__ts__" not in out.columns or "__symbol__" not in out.columns:
        raise ValueError("Label frame must include __ts__ and __symbol__")
    out["__ts__"] = pd.to_datetime(out["__ts__"], errors="coerce")
    return out.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(
        drop=True
    )


def _fold_reuse_fingerprint(
    *,
    run_fingerprint: str | None,
    window: Mapping[str, Any],
    selected_features: Sequence[str] | None,
    fixed_training_contract: Mapping[str, Any] | None,
    ae_gmm_state_path: Path | None,
    frozen_ae_gmm_output_sidecar_path: Path | None,
) -> str:
    return _reuse_fingerprint(
        {
            "schema": "base_fold_payload_reuse_v1",
            "run_fingerprint": str(run_fingerprint or ""),
            "fold": str(window.get("fold")),
            "month": str(window.get("month")),
            "valid_start": str(window.get("valid_start")),
            "valid_end": str(window.get("valid_end")),
            "train_start": str(window.get("train_start")),
            "selected_features": list(map(str, selected_features or [])),
            "fixed_training_contract": dict(fixed_training_contract or {}),
            "ae_gmm_state": _file_identity(ae_gmm_state_path),
            "frozen_ae_gmm_output_sidecar": _file_identity(
                frozen_ae_gmm_output_sidecar_path, include_sha256=False
            ),
        }
    )


def _label_schema_columns(labels_path: Path) -> list[str]:
    if pq is None:
        return []
    files = _canonical_label_files(Path(labels_path))
    cols: list[str] = []
    for path in files[:8]:
        try:
            cols.extend(str(c) for c in pq.read_schema(path).names)
        except Exception:
            continue
    return list(dict.fromkeys(cols))


def _missing_feature_store_columns(
    frame_columns: Sequence[str],
    requested_features: Sequence[str],
) -> list[str]:
    """Return ordered feature-store columns not already materialized in labels."""

    available = {str(column) for column in frame_columns}
    return list(
        dict.fromkeys(
            str(feature)
            for feature in requested_features
            if str(feature) and str(feature) not in available
        )
    )


def _merge_authoritative_store_features(
    frame: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Replace observable label-embedded columns with static-store values."""

    if feature_matrix.empty:
        return frame
    if len(frame) != len(feature_matrix):
        raise ValueError(
            "Static feature matrix row count does not match the label frame: "
            f"features={len(feature_matrix)} labels={len(frame)}"
        )
    store_values = feature_matrix.reset_index(drop=True).astype(
        np.float32, copy=False
    )
    existing_store_cols = [col for col in store_values.columns if col in frame.columns]
    base = frame.drop(columns=existing_store_cols) if existing_store_cols else frame
    return pd.concat(
        [base.reset_index(drop=True), store_values], axis=1, copy=False
    )


def _resolve_base_model_features(
    frame: pd.DataFrame,
    fixed_selected_features: Sequence[str] | None,
    *,
    authoritative_store_features: Sequence[str] | None = None,
) -> list[str]:
    """Build a model contract from replayable observable features only.

    Label shards retain targets and archetype context for supervised screening,
    but their embedded observable copies are not a valid model source.  When an
    authoritative static-store contract is supplied, only those columns plus
    the decision-time side indicator may enter the base model.
    """

    fixed = list(dict.fromkeys(map(str, fixed_selected_features or [])))
    authoritative = list(
        dict.fromkeys(map(str, authoritative_store_features or []))
    )
    if authoritative:
        allowed = set(authoritative)
        if "side" in frame.columns:
            allowed.add("side")
        if fixed:
            missing = [name for name in fixed if name not in allowed]
            if missing:
                raise RuntimeError(
                    "Fixed base contract contains observable features that are "
                    "not supplied by the authoritative static store: "
                    f"{missing[:20]}"
                )
            candidates = fixed
        else:
            candidates = [*authoritative]
            if "side" in allowed:
                candidates.append("side")
        candidate_frame = frame.loc[
            :, [name for name in dict.fromkeys(candidates) if name in frame.columns]
        ]
        return _feature_columns(candidate_frame)
    if not fixed:
        return _feature_columns(frame)
    missing = [name for name in fixed if name not in frame.columns]
    if missing:
        raise RuntimeError(
            "Projected full-population frame is missing fixed base features: "
            f"{missing[:20]}"
        )
    return fixed


def _load_projected_labels(
    labels_path: Path,
    *,
    selected_features: Sequence[str] | None,
    ae_gmm_input_features: Sequence[str] | None,
    external_feature_sidecar_path: Path | None = None,
    target_sidecar_path: Path | None = None,
    frozen_ae_gmm_output_sidecar_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load wide labels only for selection; project them for full model fits."""
    path = Path(labels_path)
    files = _canonical_label_files(path)
    frozen = list(selected_features or [])
    if not frozen:
        frame = _load_canonical_labels(path).reset_index(drop=True)
        frame, sidecar_contract = _join_external_feature_sidecar(
            frame,
            external_feature_sidecar_path,
            requested_features=None,
        )
        frame, target_sidecar_contract = _join_external_feature_sidecar(
            frame,
            target_sidecar_path,
            requested_features=(
                "__p90_trailing_target_soft__",
                "__p90_trailing_target_hard__",
            ),
        )
        frame, ae_gmm_sidecar_contract = _join_external_feature_sidecar(
            frame,
            frozen_ae_gmm_output_sidecar_path,
            requested_features=None,
        )
        return frame, {
            "mode": "wide_selection_sample",
            "source_column_count": int(len(frame.columns)),
            "loaded_column_count": int(len(frame.columns)),
            "external_feature_sidecar": sidecar_contract,
            "target_sidecar": target_sidecar_contract,
            "frozen_ae_gmm_output_sidecar": ae_gmm_sidecar_contract,
        }

    union_schema: list[str] = []
    schemas_by_file: dict[Path, set[str]] = {}
    for file in files:
        if pq is not None:
            names = set(map(str, pq.read_schema(file).names))
        else:
            import duckdb

            escaped = str(file.resolve()).replace("'", "''")
            names = {
                str(row[0])
                for row in duckdb.connect()
                .execute(f"DESCRIBE SELECT * FROM read_parquet('{escaped}')")
                .fetchall()
            }
        schemas_by_file[file] = names
        union_schema.extend(names)
    available = set(union_schema)
    required = {
        column
        for column in available
        if str(column).startswith("__")
    }
    required.update(
        {
            "side",
            "side_name",
            "timeframe",
            "candidate_id",
            "G_VOL",
        }.intersection(available)
    )
    required.update(set(map(str, frozen)).intersection(available))
    required.update(set(map(str, ae_gmm_input_features or [])).intersection(available))
    ordered_required = [column for column in union_schema if column in required]
    ordered_required = list(dict.fromkeys(ordered_required))
    frames: list[pd.DataFrame] = []
    for file in files:
        local_columns = [
            column for column in ordered_required if column in schemas_by_file[file]
        ]
        part = pd.read_parquet(file, columns=local_columns)
        missing = [column for column in ordered_required if column not in part.columns]
        for column in missing:
            part[column] = np.nan
        frames.append(part.reindex(columns=ordered_required))
    frame = (
        pd.concat(frames, ignore_index=True, copy=False)
        if len(frames) > 1
        else frames[0].reset_index(drop=True)
    )
    if "__ts__" not in frame.columns or "__symbol__" not in frame.columns:
        raise ValueError("Projected label frame must include __ts__ and __symbol__")
    # Monthly label shards may have equivalent UTC timestamps persisted as a
    # mix of timezone-aware and legacy naive values.  Pandas' strict mixed
    # parser can otherwise coerce valid rows to NaT after concatenation.
    frame["__ts__"] = pd.to_datetime(
        frame["__ts__"], format="mixed", utc=True, errors="coerce"
    )
    frame = frame.sort_values(
        ["__ts__", "__symbol__"], kind="mergesort"
    ).reset_index(drop=True)
    frame, sidecar_contract = _join_external_feature_sidecar(
        frame,
        external_feature_sidecar_path,
        requested_features=frozen,
    )
    frame, target_sidecar_contract = _join_external_feature_sidecar(
        frame,
        target_sidecar_path,
        requested_features=(
            "__p90_trailing_target_soft__",
            "__p90_trailing_target_hard__",
        ),
    )
    # The sidecar contains the complete frozen representation contract.  Keep
    # those columns on the candidate ledger for downstream meta selection;
    # ``features`` still limits the base fit matrix to its selected contract.
    frame, ae_gmm_sidecar_contract = _join_external_feature_sidecar(
        frame,
        frozen_ae_gmm_output_sidecar_path,
        requested_features=None,
    )
    return frame, {
        "mode": "narrow_full_population",
        "source_column_count": int(len(available)),
        "loaded_column_count": int(len(ordered_required)),
        "loaded_columns": ordered_required,
        "external_feature_sidecar": sidecar_contract,
        "target_sidecar": target_sidecar_contract,
        "frozen_ae_gmm_output_sidecar": ae_gmm_sidecar_contract,
    }


def _source_files_signature(paths: Sequence[Path]) -> str:
    payload = [
        (str(path.resolve()), int(path.stat().st_size), int(path.stat().st_mtime_ns))
        for path in paths
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _validate_frozen_ae_gmm_output_sidecar(
    *,
    labels_path: Path,
    state_path: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    """Bind a precomputed representation sidecar to its labels and state."""

    manifest_path = sidecar_path.with_suffix(".manifest.json")
    if not sidecar_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Frozen AE/GMM output sidecar requires both parquet and manifest: "
            f"{sidecar_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_files = _canonical_label_files(Path(labels_path))
    expected = {
        "source_signature": _source_files_signature(label_files),
        "state_sha256": _sha256_file(Path(state_path)),
        "source_rows": int(
            sum(pq.ParquetFile(path).metadata.num_rows for path in label_files)
        )
        if pq is not None
        else None,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if value is not None and str(manifest.get(key)) != str(value)
    }
    outputs = [str(value) for value in manifest.get("output_features", []) or []]
    if mismatches or not outputs:
        raise RuntimeError(
            "Precomputed frozen AE/GMM sidecar contract mismatch: "
            f"mismatches={mismatches}, output_count={len(outputs)}"
        )
    return {
        **manifest,
        "status": "validated_precomputed",
        "path": str(sidecar_path),
        "manifest_path": str(manifest_path),
    }


def _materialize_frozen_ae_gmm_output_sidecar(
    *,
    labels_path: Path,
    feature_dir: Path,
    state_path: Path,
    output_path: Path,
    output_features: Sequence[str],
    chunk_rows: int = 250_000,
) -> tuple[Path, dict[str, Any]]:
    """Stream frozen row-independent AE/GMM outputs without a wide full reload."""

    if pq is None or pa is None:
        raise RuntimeError("PyArrow is required for streamed AE/GMM materialization")
    state = load_ae_gmm_state_artifact(state_path)
    if str(state.get("temporal_feature_contract") or "") != "row_independent_v1":
        raise RuntimeError(
            "Chunked frozen AE/GMM materialization requires row_independent_v1; "
            "a temporal transform must provide an explicit cross-chunk state contract"
        )
    inputs = [str(value) for value in state.get("feature_columns", []) or []]
    requested_outputs = list(
        dict.fromkeys(
            str(value)
            for value in output_features
            if str(value) in set(map(str, AE_GMM_FEATURE_COLUMNS))
        )
    )
    if not inputs or not requested_outputs:
        raise ValueError("Frozen AE/GMM sidecar requires inputs and selected outputs")
    label_files = _canonical_label_files(Path(labels_path))
    source_rows = int(
        sum(pq.ParquetFile(path).metadata.num_rows for path in label_files)
    )
    source_signature = _source_files_signature(label_files)
    state_sha256 = _sha256_file(Path(state_path))
    output_hash = _feature_contract_hash(requested_outputs)
    input_source_policy = "shared_static_store_authoritative_v1"
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.is_file() and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            str(cached.get("source_signature")) == source_signature
            and str(cached.get("state_sha256")) == state_sha256
            and str(cached.get("output_feature_hash")) == output_hash
            and str(cached.get("input_source_policy")) == input_source_policy
        ):
            return output_path, {**cached, "status": "reused"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    fill_map = {
        str(key): np.float32(value)
        for key, value in dict(state.get("cycle_input_fill_values", {}) or {}).items()
    }
    try:
        for label_file in label_files:
            schema_names = set(map(str, pq.read_schema(label_file).names))
            read_columns = [
                name
                for name in ["__ts__", "__symbol__", "side"]
                if name in schema_names
            ]
            parquet_file = pq.ParquetFile(label_file)
            for batch in parquet_file.iter_batches(
                batch_size=max(1_000, int(chunk_rows)), columns=read_columns
            ):
                frame = pa.Table.from_batches([batch]).to_pandas()
                fetched, _ = _load_feature_store_columns(
                    frame,
                    feature_dir=feature_dir,
                    selected_features=[name for name in inputs if name != "side"],
                    min_feature_finite_frac=1e-12,
                )
                if "side" in inputs:
                    fetched["side"] = pd.to_numeric(
                        frame["side"], errors="raise"
                    ).to_numpy(dtype=np.float32, copy=False)
                missing_inputs = [name for name in inputs if name not in fetched.columns]
                if missing_inputs:
                    raise RuntimeError(
                        "Frozen AE/GMM sidecar cannot source ordered inputs from "
                        f"the shared static store: {missing_inputs[:20]}"
                    )
                x = fetched.reindex(columns=inputs).apply(pd.to_numeric, errors="coerce")
                x = x.replace([np.inf, -np.inf], np.nan)
                for name in inputs:
                    x[name] = x[name].fillna(fill_map.get(name, np.float32(0.0)))
                x = x.astype(np.float32, copy=False)
                if not bool(np.isfinite(x.to_numpy(dtype=np.float32, copy=False)).all()):
                    raise RuntimeError("Frozen AE/GMM sidecar inputs remain non-finite")
                generated = transform_ae_gmm_features(x, state).loc[:, requested_outputs]
                out = pd.DataFrame(
                    {
                        "__ts__": pd.to_datetime(frame["__ts__"], utc=True),
                        "__symbol__": frame["__symbol__"].astype(str),
                        "side": pd.to_numeric(frame["side"], errors="raise").astype(np.int8),
                    }
                )
                out = pd.concat(
                    [out.reset_index(drop=True), generated.reset_index(drop=True)],
                    axis=1,
                    copy=False,
                )
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path, table.schema, compression="zstd", compression_level=5
                    )
                writer.write_table(table, row_group_size=max(1_000, int(chunk_rows)))
                total_rows += len(out)
                print(
                    "[ae_gmm_sidecar] progress "
                    f"rows={total_rows}/{source_rows} "
                    f"pct={100.0 * total_rows / max(source_rows, 1):.1f} "
                    f"outputs={len(requested_outputs)}",
                    flush=True,
                )
                del frame, x, generated, out, table, batch
                _release_process_memory()
    finally:
        if writer is not None:
            writer.close()
    if total_rows <= 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Frozen AE/GMM sidecar materialized no rows")
    temp_path.replace(output_path)
    contract = {
        "schema": "frozen_ae_gmm_selected_output_sidecar_v1",
        "status": "materialized",
        "path": str(output_path),
        "rows": int(total_rows),
        "source_rows": int(source_rows),
        "chunk_rows": int(chunk_rows),
        "input_feature_count": int(len(inputs)),
        "output_features": requested_outputs,
        "output_feature_count": int(len(requested_outputs)),
        "output_feature_hash": output_hash,
        "state_path": str(state_path),
        "state_sha256": state_sha256,
        "source_signature": source_signature,
        "temporal_feature_contract": "row_independent_v1",
        "input_source_policy": input_source_policy,
        "materialization": "bounded_parquet_batches_then_selected_output_projection",
    }
    manifest_path.write_text(
        json.dumps(_json_safe(contract), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, contract


def _join_external_feature_sidecar(
    frame: pd.DataFrame,
    path: Path | None,
    *,
    requested_features: Sequence[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join immutable representation outputs by UTC timestamp, symbol and side."""

    if path is None:
        return frame, {"enabled": False}
    sidecar_path = Path(path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    schema = list(map(str, pq.read_schema(sidecar_path).names)) if pq is not None else list(
        pd.read_parquet(sidecar_path).columns
    )
    keys = ["__ts__", "__symbol__", "side"]
    missing_keys = [key for key in keys if key not in schema or key not in frame.columns]
    if missing_keys:
        raise ValueError(
            f"External representation sidecar is missing join keys: {missing_keys}"
        )
    available_features = [name for name in schema if name not in keys]
    if requested_features is None:
        selected = available_features
    else:
        requested = set(map(str, requested_features))
        selected = [name for name in available_features if name in requested]
    if not selected:
        return frame, {
            "enabled": True,
            "path": str(sidecar_path),
            "available_feature_count": int(len(available_features)),
            "joined_feature_count": 0,
        }
    left = frame.copy(deep=False)
    left["__ts__"] = pd.to_datetime(left["__ts__"], utc=True, errors="coerce")
    left["__symbol__"] = left["__symbol__"].astype(str)
    left["side"] = pd.to_numeric(left["side"], errors="coerce").astype("Int8")
    if bool(left[keys].isna().any(axis=None)):
        raise ValueError("Model labels contain invalid external-sidecar join keys")
    overlap = sorted(set(selected).intersection(left.columns))
    if overlap:
        raise ValueError(
            f"External representation sidecar would overwrite columns: {overlap[:20]}"
        )

    # The canonical B/M/E feature-selection sample is deliberately small. Avoid
    # materializing a multi-million-row sidecar merely to attach its frozen
    # representation columns to a 40k-row sample. The key-filtered DuckDB path
    # is exact on the requested keys; full population fits retain the pandas
    # path below and its whole-sidecar uniqueness check.
    sidecar_rows = 0
    if pq is not None:
        try:
            sidecar_rows = int(pq.ParquetFile(sidecar_path).metadata.num_rows)
        except Exception:
            sidecar_rows = 0
    use_key_filtered_join = bool(sidecar_rows and len(left) * 20 < sidecar_rows)
    if use_key_filtered_join:
        import duckdb

        connection = duckdb.connect()
        try:
            connection.execute("SET TimeZone='UTC'")
            connection.register("external_sidecar_keys", left[keys])
            quoted_columns = ", ".join(
                f's."{column}" AS "{column}"' for column in [*keys, *selected]
            )
            escaped = str(sidecar_path.resolve()).replace("'", "''")
            sidecar = connection.execute(
                f"""
                SELECT {quoted_columns}
                FROM read_parquet('{escaped}') AS s
                INNER JOIN external_sidecar_keys AS k
                    ON s."__ts__" = k."__ts__"
                    AND CAST(s."__symbol__" AS VARCHAR) = k."__symbol__"
                    AND CAST(s."side" AS SMALLINT) = k."side"
                """
            ).fetchdf()
        finally:
            connection.close()
        join_mode = "key_filtered_duckdb"
    else:
        sidecar = pd.read_parquet(sidecar_path, columns=[*keys, *selected])
        join_mode = "full_sidecar_pandas"
    sidecar["__ts__"] = pd.to_datetime(sidecar["__ts__"], utc=True, errors="coerce")
    sidecar["__symbol__"] = sidecar["__symbol__"].astype(str)
    sidecar["side"] = pd.to_numeric(sidecar["side"], errors="coerce").astype("Int8")
    if bool(sidecar[keys].isna().any(axis=None)):
        raise ValueError("External representation sidecar contains invalid join keys")
    if bool(sidecar.duplicated(keys, keep=False).any()):
        raise ValueError(
            "External representation sidecar must be unique by UTC timestamp, symbol and side"
        )
    merged = left.merge(sidecar, on=keys, how="left", validate="many_to_one", copy=False)
    supported = merged[selected].notna().all(axis=1)
    coverage = float(supported.mean()) if len(merged) else 1.0
    support_policy = str(
        os.environ.get("EPM_EXTERNAL_SIDECAR_SUPPORT_POLICY", "strict")
    ).strip().lower()
    if support_policy not in {"strict", "filter_report"}:
        raise ValueError(
            "EPM_EXTERNAL_SIDECAR_SUPPORT_POLICY must be 'strict' or 'filter_report'"
        )
    input_rows = int(len(merged))
    if coverage < 0.999:
        if support_policy != "filter_report":
            raise ValueError(
                "External representation sidecar does not cover the model population: "
                f"coverage={coverage:.6f}"
            )
        # This is a pre-entry availability restriction, not an outcome-driven
        # selector. It keeps all density arms on the same frozen state support.
        # Downstream feature-store loading uses positional row indices. Reset
        # after this explicit support restriction so labels and fetched feature
        # vectors remain aligned instead of retaining sparse source indices.
        merged = merged.loc[supported].reset_index(drop=True)
    return merged, {
        "enabled": True,
        "path": str(sidecar_path),
        "available_feature_count": int(len(available_features)),
        "joined_features": selected,
        "joined_feature_count": int(len(selected)),
        "join_mode": join_mode,
        "sidecar_rows": int(sidecar_rows),
        "row_coverage": coverage,
        "support_policy": support_policy,
        "input_rows": input_rows,
        "dropped_unsupported_rows": int(input_rows - len(merged)),
        "retained_rows": int(len(merged)),
    }


def _contains_any_token(name: Any, tokens: Sequence[str]) -> bool:
    text = str(name).lower()
    return any(tok in text for tok in tokens)


def _default_ae_gmm_input_features(
    selected_features: Sequence[str] | None,
    available_features: Sequence[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    selected = [str(c) for c in (selected_features or []) if str(c).strip()]
    available = [str(c) for c in (available_features or []) if str(c).strip()]
    generated = {str(c) for c in AE_GMM_FEATURE_COLUMNS}
    selected = [c for c in selected if c not in generated]
    available = [c for c in available if c not in generated]
    policy = str(AE_GMM_INPUT_POLICY or "a0bis").strip().lower()
    if policy in {"a0", "selected", "legacy", "raw"}:
        output = list(dict.fromkeys(selected))
        return output, {
            "policy": policy,
            "selected_input_feature_count_before_policy": int(len(selected)),
            "selected_input_feature_count_after_policy": int(len(output)),
            "removed_raw_momentum_count": 0,
            "added_normalized_momentum_count": 0,
            "removed_raw_momentum_features": [],
            "added_normalized_momentum_features": [],
        }
    raw_momentum = [
        c
        for c in selected
        if _contains_any_token(c, AE_GMM_A0BIS_MOMENTUM_TOKENS)
        and not _contains_any_token(c, AE_GMM_A0BIS_NORMALIZED_TOKENS)
    ]
    raw_set = set(raw_momentum)
    normalized_momentum = [
        c
        for c in available
        if _contains_any_token(c, AE_GMM_A0BIS_MOMENTUM_TOKENS)
        and _contains_any_token(c, AE_GMM_A0BIS_NORMALIZED_TOKENS)
    ]
    output = list(dict.fromkeys([c for c in selected if c not in raw_set] + normalized_momentum))
    return output, {
        "policy": "a0bis",
        "selected_input_feature_count_before_policy": int(len(selected)),
        "selected_input_feature_count_after_policy": int(len(output)),
        "removed_raw_momentum_count": int(len(raw_momentum)),
        "added_normalized_momentum_count": int(len(set(normalized_momentum).difference(selected))),
        "removed_raw_momentum_features": list(raw_momentum),
        "added_normalized_momentum_features": sorted(set(normalized_momentum).difference(selected)),
    }


def _save_base_fold_model(
    *,
    model_dir: Path,
    fold: dict[str, Any],
    model: Any,
    feature_names: list[str] | Mapping[str, Sequence[str]],
    x_train: pd.DataFrame,
    imputation_fill_values: Mapping[str, float] | None = None,
    params: dict[str, Any],
    trial_number: int,
    seed: int,
    train_rows_available: int,
    train_rows_fit: int,
    valid_rows: int,
    reuse_fingerprint: str | None = None,
    base_oof_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fold_dir = model_dir / _safe_artifact_stem(fold.get("fold", "fold"))
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "base_model.joblib"
    joblib.dump(model, model_path, compress=3)
    columns_path = fold_dir / "columns.json"
    if isinstance(feature_names, Mapping):
        names_by_side = {
            str(side): [str(name) for name in names]
            for side, names in feature_names.items()
        }
        flattened = list(
            dict.fromkeys(name for names in names_by_side.values() for name in names)
        )
        columns_payload = {
            "schema": "s59_base_fold_feature_contract_v2_side_local",
            "feature_names": flattened,
            "feature_names_by_side": names_by_side,
            "feature_count": int(len(flattened)),
            "feature_count_by_side": {
                side: int(len(names)) for side, names in names_by_side.items()
            },
            "feature_contract_hash": _reuse_fingerprint(names_by_side),
        }
    else:
        flattened = list(feature_names)
        columns_payload = {
            "schema": "s59_base_fold_feature_contract_v1",
            "feature_names": flattened,
            "feature_count": int(len(flattened)),
            "feature_contract_hash": _feature_contract_hash(flattened),
        }
    columns_path.write_text(json.dumps(_json_safe(columns_payload), indent=2, sort_keys=True), encoding="utf-8")
    missing_train_columns = [name for name in flattened if name not in x_train.columns]
    if missing_train_columns:
        raise ValueError(
            "Cannot persist base imputation contract; fitted matrix is missing "
            f"features: {missing_train_columns[:20]}"
        )
    fill_values: list[float] = []
    for name in flattened:
        configured_fill = (imputation_fill_values or {}).get(name)
        if configured_fill is not None and np.isfinite(configured_fill):
            fill_values.append(float(configured_fill))
            continue
        fill_values.append(_finite_median_or_zero(x_train[name]))
    imputation_contract = {
        "schema": "s60_base_train_median_imputation_v1",
        "strategy": "per_feature_train_median_then_zero_if_all_missing",
        "fit_scope": (
            "fold_train_rows_before_fit_cap"
            if imputation_fill_values is not None
            else "serialized_model_training_rows_only"
        ),
        "feature_names": flattened,
        "feature_order_hash": _feature_contract_hash(flattened),
        "feature_count": int(len(flattened)),
        "fill_values": fill_values,
        "train_rows_fit": int(train_rows_fit),
        "precomputed_fill_values_used": bool(imputation_fill_values is not None),
    }
    imputation_contract["imputation_contract_hash"] = _reuse_fingerprint(
        {
            "schema": imputation_contract["schema"],
            "strategy": imputation_contract["strategy"],
            "feature_names": flattened,
            "fill_values": fill_values,
        }
    )
    imputation_path = fold_dir / "imputation.json"
    imputation_path.write_text(
        json.dumps(_json_safe(imputation_contract), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    imputation_sha256 = _sha256_file(imputation_path)
    manifest = {
        "schema": "s60_base_saved_fold_model_v2",
        "fold": str(fold.get("fold")),
        "calendar_month": str(fold.get("month")),
        "valid_start": fold.get("valid_start"),
        "valid_end": fold.get("valid_end"),
        "max_oos_model_age_days": int(fold.get("max_oos_model_age_days", 0)),
        "trial_number": int(trial_number),
        "seed": int(seed),
        "target_mode": str(params.get("target_mode")),
        "weight_arm": str(params.get("weight_arm")),
        "train_rows_available": int(train_rows_available),
        "train_rows_fit": int(train_rows_fit),
        "valid_rows": int(valid_rows),
        "model_path": str(model_path),
        "columns_path": str(columns_path),
        "imputation_path": str(imputation_path),
        "imputation_sha256": imputation_sha256,
        "imputation_contract_hash": imputation_contract["imputation_contract_hash"],
        "imputation_provenance": {
            "fit_scope": imputation_contract["fit_scope"],
            "strategy": imputation_contract["strategy"],
            "precomputed_fill_values_used": imputation_contract[
                "precomputed_fill_values_used"
            ],
            "feature_order_hash": imputation_contract["feature_order_hash"],
            "train_rows_fit": int(train_rows_fit),
        },
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "feature_count": int(len(flattened)),
        "feature_contract_hash": columns_payload["feature_contract_hash"],
        "reuse_fingerprint": str(reuse_fingerprint or ""),
        "params": _json_safe(params),
        "ae_gmm_generated_features": int(fold.get("ae_gmm_generated_features", 0)),
        "ae_gmm_context_feature_count": int(fold.get("ae_gmm_context_feature_count", 0)),
        "ae_gmm_status": fold.get("ae_gmm_status"),
        "base_oof_provenance": dict(base_oof_provenance or {}),
        "leakage_contract": {
            "fit_scope": "prior_rows_only_for_this_oos_fold",
            "oos_rows": "valid_start <= timestamp < valid_end",
            "feature_contract": "columns.json is the required inference-time feature order",
            "imputation_contract": "imputation.json is fit on serialized model training rows and is required before float16 scoring",
            "target": "materialized trailing-label soft economic target used only on train rows",
        },
    }
    manifest_path = fold_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path), "model_dir": str(fold_dir)}


def _persist_base_oof_provenance(
    manifest_path: Path, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach fresh provenance to a reused model without altering it."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_oof_provenance"] = dict(provenance)
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _append_single_side_ae_gmm_state_features(
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    max_train_rows: int,
    gmm_max_train_rows: int,
    ae_max_iter: int,
    random_state: int,
    state_artifact_dir: Path | None = None,
    state_artifact_name: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    base_features = [
        str(col)
        for col in x_train.columns
        if str(col) not in set(str(v) for v in AE_GMM_FEATURE_COLUMNS)
    ]
    if len(base_features) < 2 or len(x_train) < 500:
        return x_train, x_valid, [], {
            "ae_gmm_state_feature_status": "single_side_insufficient_rows_or_features",
            "ae_gmm_state_feature_count": 0,
            "ae_gmm_state_input_feature_count": int(len(base_features)),
        }
    x_train_base = x_train.reindex(columns=base_features).astype(np.float32, copy=False)
    x_valid_base = x_valid.reindex(columns=base_features).astype(np.float32, copy=False)
    state = fit_ae_gmm_state(
        x_train_base.reset_index(drop=True),
        economic_targets=_fold_ae_gmm_economic_targets(
            train_metrics.reset_index(drop=True),
            train_frame=train_frame.reset_index(drop=True),
        ),
        random_state=int(random_state),
        max_train_rows=int(max_train_rows),
        gmm_max_train_rows=int(gmm_max_train_rows),
        ae_max_iter=int(ae_max_iter),
        require_both_sides=False,
        min_side_cluster_frac=0.02,
        min_side_cluster_rows=10,
    )
    if not bool(state.get("enabled", False)):
        persisted_disabled = _persist_ae_gmm_state_artifact(
            state=state,
            artifact_dir=state_artifact_dir,
            artifact_name=state_artifact_name,
            scope="single_side_disabled",
            train_rows=len(x_train_base),
            valid_rows=len(x_valid_base),
            input_feature_count=len(base_features),
        )
        return x_train, x_valid, [], {
            "ae_gmm_state_feature_status": f"single_side_{state.get('reason', 'state_disabled')}",
            "ae_gmm_state_feature_count": 0,
            "ae_gmm_state_input_feature_count": int(len(base_features)),
            "ae_gmm_state_hpo_report_count": int(state.get("hpo_report_count", 0) or 0),
            **persisted_disabled,
        }
    persisted_artifacts = _persist_ae_gmm_state_artifact(
        state=state,
        artifact_dir=state_artifact_dir,
        artifact_name=state_artifact_name,
        scope="single_side",
        train_rows=len(x_train_base),
        valid_rows=len(x_valid_base),
        input_feature_count=len(base_features),
    )
    valid_generated = transform_ae_gmm_features(x_valid_base, state, index=x_valid.index)
    all_generated = [str(col) for col in valid_generated.columns]
    generated = _ae_gmm_smoke_feature_policy_columns(all_generated)
    generated = list(dict.fromkeys([*generated, "ae_gmm_oof_available"]))
    train_generated = transform_ae_gmm_features(x_train_base, state, index=x_train.index).reindex(
        columns=generated,
        fill_value=0.0,
    )
    valid_generated = valid_generated.reindex(columns=generated, fill_value=0.0)
    train_generated["ae_gmm_oof_available"] = np.float32(1.0)
    valid_generated["ae_gmm_oof_available"] = np.float32(1.0)
    selected_config = dict(state.get("selected_config", {}) or {})
    return (
        pd.concat([x_train, train_generated], axis=1, copy=False).astype(np.float32, copy=False),
        pd.concat([x_valid, valid_generated], axis=1, copy=False).astype(np.float32, copy=False),
        generated,
        {
            "ae_gmm_state_feature_status": "ok_single_side_outer_train",
            "ae_gmm_state_feature_count": int(len(generated)),
            "ae_gmm_state_all_feature_count": int(len(all_generated)),
            "ae_gmm_state_input_feature_count": int(len(base_features)),
            "ae_gmm_state_hpo_report_count": int(state.get("hpo_report_count", 0) or 0),
            "ae_gmm_state_train_rows_available": int(state.get("train_rows_available", len(x_train_base)) or 0),
            "ae_gmm_state_ae_fit_rows": int(state.get("ae_fit_rows", 0) or 0),
            "ae_gmm_state_gmm_fit_rows": int(state.get("gmm_fit_rows", 0) or 0),
            "ae_gmm_state_ae_max_train_rows": int(state.get("ae_max_train_rows", max_train_rows) or 0),
            "ae_gmm_state_gmm_max_train_rows": int(state.get("gmm_max_train_rows", gmm_max_train_rows) or 0),
            "ae_gmm_state_sample_policy": str(state.get("sample_policy", "")),
            "ae_gmm_state_n_components": int(state.get("gmm_n_components", 0) or 0),
            "ae_gmm_state_path_cleanliness_score": float(
                selected_config.get("path_cleanliness_score", float("nan"))
            ),
            "ae_gmm_state_temporal_concentration_score": float(
                selected_config.get("temporal_concentration_score", float("nan"))
            ),
            "ae_gmm_state_train_feature_scope": "outer_train_in_sample",
            "ae_gmm_state_validation_feature_scope": "frozen_outer_train_artifact",
            "ae_gmm_state_artifact_dir": str(state_artifact_dir) if state_artifact_dir is not None else None,
            "ae_gmm_frozen_replay_contract": (
                "single-side AE/GMM state fit on the outer train fold is persisted; "
                "validation/OOS rows are transformed with that frozen train-fitted state"
            ),
            **persisted_artifacts,
        },
    )


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _load_fixed_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params", payload)
    if not isinstance(params, dict):
        raise ValueError(f"Fixed params payload must be a dict or contain a params dict: {path}")
    required = {
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "target_mode",
        "weight_arm",
    }
    missing = sorted(required.difference(params))
    if missing:
        raise ValueError(f"Fixed params missing keys {missing}: {path}")
    optional = {"loss_function", "min_split_gain"}
    # Promoted HPO files also contain historical objectives, ranks and metrics.
    # Those are evidence, not model parameters; carrying them into a new trial
    # can overwrite freshly evaluated values when dictionaries are merged.
    out = {key: params[key] for key in sorted(required.union(optional)) if key in params}
    for key in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        out[key] = int(float(out[key]))
    for key in ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        out[key] = float(out[key])
    out["target_mode"] = str(out["target_mode"])
    out["weight_arm"] = str(out["weight_arm"])
    out["loss_function"] = str(out.get("loss_function", "regression"))
    out["min_split_gain"] = float(out.get("min_split_gain", 0.0))
    if "trial_number" in payload:
        out["_fixed_trial_number"] = int(float(payload["trial_number"]))
    return out


def _load_fixed_selected_features(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("selected_features") if isinstance(payload, dict) else payload
        if isinstance(values, dict):
            values = values.get("features")
        features = [str(v) for v in (values or []) if str(v).strip()]
    else:
        frame = pd.read_csv(path)
        if "feature" not in frame.columns:
            raise ValueError(f"{path} must include a 'feature' column")
        if "selected" in frame.columns:
            selected = frame["selected"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
            frame = frame.loc[selected].copy()
        if "rank" in frame.columns:
            frame = frame.sort_values("rank", kind="mergesort")
        features = [str(v) for v in frame["feature"].dropna().tolist() if str(v).strip()]
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError(f"No fixed selected features found in {path}")
    return features


def _load_fixed_selected_features_by_side(
    path: Path | None,
    selected_union: Sequence[str] | None,
) -> dict[str, list[str]] | None:
    if path is None or path.suffix.lower() != ".json" or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    raw = payload.get("selected_features_by_side")
    if not isinstance(raw, dict):
        return None
    allowed = set(map(str, selected_union or ()))
    result = {
        side: list(
            dict.fromkeys(
                str(feature)
                for feature in raw.get(side, ())
                if str(feature) in allowed
            )
        )
        for side in ("long", "short")
    }
    if not result["long"] or not result["short"]:
        raise ValueError(f"Fixed side feature contract is empty for one side: {path}")
    return result


def _fixed_selected_ae_gmm_features(features: Sequence[str] | None) -> list[str]:
    if not features:
        return []
    generated = set(str(col) for col in AE_GMM_FEATURE_COLUMNS)
    return [str(col) for col in features if str(col) in generated]


def _scored_key_tuples(frame: pd.DataFrame) -> set[tuple[int, str, int]]:
    if frame.empty:
        return set()
    required = {"__ts__", "__symbol__", "side"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot build scored keys; missing columns: {missing}")
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce").astype("int64").to_numpy()
    sym = frame["__symbol__"].astype(str).to_numpy()
    side = pd.to_numeric(frame["side"], errors="coerce").fillna(1.0).astype(np.int8).to_numpy()
    bad_ts = np.iinfo(np.int64).min
    return {
        (int(t), str(s), int(sd))
        for t, s, sd in zip(ts, sym, side, strict=False)
        if int(t) != bad_ts
    }


def _load_existing_scored_keys(path: Path | None) -> set[tuple[int, str, int]]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(path)
    existing = pd.read_parquet(path, columns=["__ts__", "__symbol__", "side"])
    return _scored_key_tuples(existing)


def _missing_against_existing_mask(frame: pd.DataFrame, existing_keys: set[tuple[int, str, int]]) -> np.ndarray:
    if not existing_keys or frame.empty:
        return np.ones(len(frame), dtype=bool)
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce").astype("int64").to_numpy()
    sym = frame["__symbol__"].astype(str).to_numpy()
    side = pd.to_numeric(frame["side"], errors="coerce").fillna(1.0).astype(np.int8).to_numpy()
    bad_ts = np.iinfo(np.int64).min
    missing = np.ones(len(frame), dtype=bool)
    for i, (t, s, sd) in enumerate(zip(ts, sym, side, strict=False)):
        missing[i] = int(t) != bad_ts and (int(t), str(s), int(sd)) not in existing_keys
    return missing


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_fold_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _fold_frame_columns(frame: pd.DataFrame) -> list[str]:
    exact = {
        "__ts__",
        "__symbol__",
        "__side__",
        "side",
        "side_name",
        "candidate_id",
        "month",
        "__w__",
        "__econ_sideaware_execres_clean__",
        "__econ_sideaware_execres_dirty_positive__",
        "__econ_side_resolution_clean__",
        "__econ_side_resolution_dirty_positive__",
    }
    context_tokens = (
        "archetype",
        "source",
        "regime",
        "aegmm",
        "ae_gmm",
        "gmm",
        "cluster",
        "reconstruction",
        "latent",
        "posterior",
        "entropy",
        "mahalanobis",
    )
    keep: list[str] = []
    for col in frame.columns:
        name = str(col)
        lower = name.lower()
        if name in exact or name.startswith("__") or any(token in lower for token in context_tokens):
            keep.append(name)
    return list(dict.fromkeys([col for col in keep if col in frame.columns]))


def _base_oof_provenance_columns(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add stable candidate and resolved-label timestamps for fold auditing.

    The materialized trailing labels resolve on 15-minute replay paths.  Their
    first-touch bar is one-indexed, so the source path's first timestamp plus
    that many path bars is the first instant the target is fully known.
    """

    out = frame.copy(deep=False)
    candidate_id_source = "candidate_id"
    if "candidate_id" not in out.columns:
        ts = pd.to_datetime(out.get("__ts__"), utc=True, errors="coerce")
        symbols = out.get("__symbol__", pd.Series("", index=out.index)).astype(str)
        side = pd.to_numeric(
            out.get("side", out.get("__side__", 1.0)), errors="coerce"
        ).fillna(1.0).astype(np.int8)
        out["candidate_id"] = (
            symbols
            + "|"
            + ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("NaT")
            + "|"
            + side.astype(str)
        )
        candidate_id_source = "derived_symbol_timestamp_side_v1"

    explicit_resolution_columns = (
        "__label_resolution_ts__",
        "label_resolution_ts",
        "__label_available_ts__",
        "label_available_ts",
        "execution_label_available_at",
        "label_end_ts",
        "__label_end_ts__",
    )
    resolution_source_column = next(
        (column for column in explicit_resolution_columns if column in out.columns),
        None,
    )
    resolution_derivation = "source_column"
    if resolution_source_column is not None:
        resolution = pd.to_datetime(
            out[resolution_source_column], utc=True, errors="coerce"
        )
    elif {"__first_path_ts__", "__first_touch_bar__"}.issubset(out.columns):
        first_path = pd.to_datetime(out["__first_path_ts__"], utc=True, errors="coerce")
        bars = pd.to_numeric(out["__first_touch_bar__"], errors="coerce")
        resolution = first_path + pd.to_timedelta(
            bars * BASE_OOF_LABEL_PATH_TIMEFRAME / pd.Timedelta(minutes=1),
            unit="m",
        )
        resolution_derivation = "first_path_plus_first_touch_bars_x_15m"
    else:
        resolution = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        resolution_derivation = "unavailable"

    out["__label_resolution_ts__"] = resolution
    return out, {
        "candidate_id_column": "candidate_id",
        "candidate_id_source": candidate_id_source,
        "label_resolution_column": "__label_resolution_ts__",
        "label_resolution_source_column": str(resolution_source_column or ""),
        "label_resolution_derivation": resolution_derivation,
        "label_path_timeframe": "15m"
        if resolution_derivation == "first_path_plus_first_touch_bars_x_15m"
        else None,
    }


def _base_oof_fit_provenance(
    *,
    fold: Mapping[str, Any],
    train_provenance: pd.DataFrame,
    fit_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return provenance for the exact rows supplied to a fold model."""

    work = train_provenance
    if fit_indices is not None:
        work = work.iloc[np.asarray(fit_indices, dtype=np.int64)]
    decision_source = work.get(
        "__decision_ts__",
        pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]"),
    )
    resolution_source = work.get(
        "__label_resolution_ts__",
        pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]"),
    )
    decision = pd.to_datetime(decision_source, utc=True, errors="coerce")
    resolution = pd.to_datetime(resolution_source, utc=True, errors="coerce")
    return {
        "schema": BASE_OOF_PROVENANCE_SCHEMA,
        "candidate_id_column": "candidate_id",
        "fold_validation_start": fold.get("valid_start"),
        "fold_validation_end": fold.get("valid_end"),
        "train_signal_cutoff_exclusive": fold.get("train_cutoff"),
        "latest_train_decision_cutoff": decision.max() if len(decision) else pd.NaT,
        "latest_train_decision_timestamp": decision.max() if len(decision) else pd.NaT,
        "latest_train_resolved_label_timestamp": resolution.max()
        if len(resolution)
        else pd.NaT,
        "label_resolution_column": str(
            fold.get("label_resolution_column", "__label_resolution_ts__")
        ),
        "label_resolution_derivation": str(
            fold.get("label_resolution_derivation", "unavailable")
        ),
        "label_resolution_source_column": str(
            fold.get("label_resolution_source_column", "")
        ),
        "label_path_timeframe": fold.get("label_path_timeframe"),
        "train_rows_provenance": int(len(work)),
    }


def _ae_gmm_context_columns(columns: Sequence[str]) -> list[str]:
    context_tokens = (
        "ae_gmm",
        "aegmm",
        "gmm_",
        "cluster",
        "posterior",
        "entropy",
        "mahalanobis",
        "dae_",
        "reconstruction",
        "latent",
    )
    out: list[str] = []
    for col in columns:
        name = str(col)
        lower = name.lower()
        if any(token in lower for token in context_tokens):
            out.append(name)
    return list(dict.fromkeys(out))


def _restore_cycle_input_columns(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    *,
    train_source: pd.DataFrame,
    valid_source: pd.DataFrame,
    required_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required = list(dict.fromkeys(str(col) for col in required_columns if str(col)))
    missing_train_source = [col for col in required if col not in train_source.columns]
    missing_valid_source = [col for col in required if col not in valid_source.columns]
    if missing_train_source or missing_valid_source:
        raise RuntimeError(
            "Frozen cycle AE/GMM inputs are absent from the assembled pre-entry frame: "
            f"train_missing={missing_train_source[:20]} ({len(missing_train_source)}), "
            f"valid_missing={missing_valid_source[:20]} ({len(missing_valid_source)})"
        )
    restored = [col for col in required if col not in x_train.columns or col not in x_valid.columns]
    if not restored:
        return x_train, x_valid, []
    train_extra = (
        train_source.loc[:, restored]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32, copy=False)
    )
    valid_extra = (
        valid_source.loc[:, restored]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32, copy=False)
    )
    return (
        pd.concat([x_train, train_extra], axis=1, copy=False),
        pd.concat([x_valid, valid_extra], axis=1, copy=False),
        restored,
    )


def _write_fold_payload(fold: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    fold_dir = cache_dir / _safe_fold_name(str(fold["fold"]))
    fold_dir.mkdir(parents=True, exist_ok=True)
    payload_paths = {
        key: fold_dir / f"{key}.parquet"
        for key in (
            "train",
            "valid",
            "train_metrics",
            "valid_metrics",
            "train_target",
            "train_weight",
            "train_side",
            "train_provenance",
            "x_train",
            "x_valid",
        )
        if key in fold and isinstance(fold.get(key), pd.DataFrame)
    }
    if "ae_gmm_context_valid" in fold and isinstance(fold.get("ae_gmm_context_valid"), pd.DataFrame):
        payload_paths["ae_gmm_context_valid"] = fold_dir / "ae_gmm_context_valid.parquet"
    for key, path in payload_paths.items():
        frame = fold[key]
        if key in {
            "x_train",
            "x_valid",
            "ae_gmm_context_valid",
        }:
            frame = frame.clip(
                lower=float(np.finfo(np.float16).min),
                upper=float(np.finfo(np.float16).max),
                axis=None,
            ).astype(np.float16, copy=False)
        frame.to_parquet(path, index=False, compression="zstd", compression_level=9)
    slim = {key: value for key, value in fold.items() if key not in payload_paths}
    slim["payload_paths"] = {key: str(path) for key, path in payload_paths.items()}
    slim["train_rows"] = int(len(fold["x_train"]))
    slim["valid_rows"] = int(len(fold["x_valid"]))
    manifest_slim = {
        key: value for key, value in slim.items() if key != "feature_selection"
    }
    (fold_dir / "fold_manifest.json").write_text(
        json.dumps(_json_safe(manifest_slim), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return slim


def _reuse_complete_fold_payload(
    *,
    cache_dir: Path,
    window: Mapping[str, Any],
    selected_features: Sequence[str] | None,
    fixed_training_contract: Mapping[str, Any] | None,
    expected_reuse_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    """Reuse a complete cache without loading its wide parquet payloads."""

    if pq is None or not selected_features or not fixed_training_contract:
        return None
    fold_dir = cache_dir / _safe_fold_name(str(window["fold"]))
    required = (
        "valid",
        "valid_metrics",
        "train_target",
            "train_weight",
            "train_provenance",
            "x_train",
        "x_valid",
        "ae_gmm_context_valid",
    )
    payload_paths = {key: fold_dir / f"{key}.parquet" for key in required}
    optional_train_side = fold_dir / "train_side.parquet"
    if optional_train_side.is_file() and optional_train_side.stat().st_size > 0:
        payload_paths["train_side"] = optional_train_side
    fold_manifest_path = fold_dir / "fold_manifest.json"
    if not all(path.is_file() and path.stat().st_size > 0 for path in payload_paths.values()):
        return None
    if not fold_manifest_path.is_file():
        return None
    try:
        cached_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
        if (
            expected_reuse_fingerprint is not None
            and str(cached_manifest.get("reuse_fingerprint") or "")
            != str(expected_reuse_fingerprint)
        ):
            return None
        cached_imputation = dict(
            cached_manifest.get("train_median_imputation_values", {}) or {}
        )
        if set(map(str, selected_features)).difference(cached_imputation):
            return None
        train_pf = pq.ParquetFile(payload_paths["x_train"])
        valid_pf = pq.ParquetFile(payload_paths["x_valid"])
        context_pf = pq.ParquetFile(payload_paths["ae_gmm_context_valid"])
        provenance_pf = pq.ParquetFile(payload_paths["train_provenance"])
        train_columns = [str(col) for col in train_pf.schema.names]
        valid_columns = [str(col) for col in valid_pf.schema.names]
        expected_columns = [str(col) for col in selected_features]
        if train_columns != expected_columns or valid_columns != expected_columns:
            return None
        train_rows = int(train_pf.metadata.num_rows)
        valid_rows = int(valid_pf.metadata.num_rows)
        expected_train_rows_uncapped = int(
            window.get("train_rows_estimate", train_rows)
        )
        cached_train_rows_uncapped = int(
            cached_manifest.get("train_rows_uncapped", train_rows)
        )
        if cached_train_rows_uncapped != expected_train_rows_uncapped:
            return None
        if train_rows <= 0 or train_rows > cached_train_rows_uncapped:
            return None
        provenance_columns = set(map(str, provenance_pf.schema.names))
        if int(provenance_pf.metadata.num_rows) != train_rows or not {
            "candidate_id",
            "__decision_ts__",
            "__label_resolution_ts__",
        }.issubset(provenance_columns):
            return None
        if valid_rows != int(window.get("valid_rows_estimate", valid_rows)):
            return None
        if str(cached_manifest.get("fixed_training_target_mode")) != str(
            fixed_training_contract.get("target_mode")
        ):
            return None
        if str(cached_manifest.get("fixed_training_weight_arm")) != str(
            fixed_training_contract.get("weight_arm")
        ):
            return None
        context_features = [str(col) for col in context_pf.schema.names]
        if int(context_pf.metadata.num_rows) != valid_rows:
            return None
    except Exception:
        return None

    slim = {
        "fold": str(window["fold"]),
        "month": str(window["month"]),
        "valid_start": window["valid_start"],
        "valid_end": window["valid_end"],
        "train_start": window.get("train_start"),
        "train_cutoff": cached_manifest.get("train_cutoff"),
        "ae_gmm_anchor_start": window.get("ae_gmm_anchor_start"),
        "ae_gmm_anchor_end": window.get("ae_gmm_anchor_end"),
        "ae_gmm_anchor_rows": 0,
        "max_oos_model_age_days": int(
            (pd.Timestamp(window["valid_end"]) - pd.Timestamp(window["valid_start"])).days
        ),
        "train_rows_uncapped": cached_train_rows_uncapped,
        "train_rows_payload": train_rows,
        "valid_rows_raw": int(window.get("valid_rows_raw_estimate", valid_rows)),
        "missing_only": False,
        "existing_scored_ledger_path": None,
        "payload_train_sampling": str(
            cached_manifest.get("payload_train_sampling", "full_train_rows")
        ),
        "train_valid_availability_contract": {"status": "reused_complete_cache"},
        "candidate_id_column": str(
            cached_manifest.get("candidate_id_column", "candidate_id")
        ),
        "candidate_id_source": str(
            cached_manifest.get("candidate_id_source", "candidate_id")
        ),
        "label_resolution_column": str(
            cached_manifest.get("label_resolution_column", "__label_resolution_ts__")
        ),
        "label_resolution_source_column": str(
            cached_manifest.get("label_resolution_source_column", "")
        ),
        "label_resolution_derivation": str(
            cached_manifest.get("label_resolution_derivation", "unavailable")
        ),
        "label_path_timeframe": cached_manifest.get("label_path_timeframe"),
        "ae_gmm_generated_features": len(context_features),
        "ae_gmm_context_feature_count": len(context_features),
        "ae_gmm_context_features": context_features,
        "ae_gmm_status": "loaded_fixed_state_artifact",
        "selected_features": list(expected_columns),
        "train_median_imputation_values": {
            str(key): float(value)
            for key, value in dict(
                cached_manifest.get("train_median_imputation_values", {}) or {}
            ).items()
            if np.isfinite(value)
        },
        "feature_selection": pd.DataFrame(),
        "compact_fixed_training_payload": True,
        "fixed_training_target_mode": str(fixed_training_contract.get("target_mode")),
        "fixed_training_weight_arm": str(fixed_training_contract.get("weight_arm")),
        "reuse_fingerprint": str(cached_manifest.get("reuse_fingerprint") or ""),
        "payload_paths": {key: str(path) for key, path in payload_paths.items()},
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "cache_reused": True,
    }
    manifest_slim = {
        key: value for key, value in slim.items() if key != "feature_selection"
    }
    (fold_dir / "fold_manifest.json").write_text(
        json.dumps(_json_safe(manifest_slim), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return slim


def _load_fold_payload(fold: dict[str, Any]) -> dict[str, Any]:
    if "payload_paths" not in fold:
        return fold
    loaded = dict(fold)
    for key, path in dict(fold["payload_paths"]).items():
        frame = pd.read_parquet(path)
        if key in {
            "x_train",
            "x_valid",
            "ae_gmm_context_valid",
        }:
            frame = frame.astype(np.float32, copy=False)
        loaded[key] = frame
    return loaded


def _load_fold_payload_keys(
    fold: dict[str, Any], keys: Sequence[str]
) -> dict[str, Any]:
    """Load only the cached frames needed by one scoring phase."""

    if "payload_paths" not in fold:
        return fold
    loaded = dict(fold)
    wanted = set(str(key) for key in keys)
    for key, path in dict(fold["payload_paths"]).items():
        if key not in wanted:
            continue
        frame = pd.read_parquet(path)
        if key in {"x_train", "x_valid", "ae_gmm_context_valid"}:
            frame = frame.astype(np.float32, copy=False)
        loaded[key] = frame
    return loaded


def _release_process_memory() -> None:
    """Return fold-local Arrow and malloc arenas where the runtime supports it."""

    gc.collect()
    if pa is not None:
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass
    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        pressure_relief = libsystem.malloc_zone_pressure_relief
        pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        pressure_relief.restype = ctypes.c_size_t
        pressure_relief(None, 0)
    except Exception:
        pass


def _cap_rows(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False).astype(np.int64))


def _time_spread_cap_rows(n_rows: int, max_rows: int) -> np.ndarray:
    """Deterministically sample rows from the beginning, middle, and end."""
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    n = int(n_rows)
    k = int(max_rows)
    parts: list[np.ndarray] = []
    base = k // 3
    rem = k - base * 3
    sizes = [base + (1 if i < rem else 0) for i in range(3)]
    windows = [(0, n // 3), (n // 3, (2 * n) // 3), ((2 * n) // 3, n)]
    for size, (start, end) in zip(sizes, windows):
        size = min(int(size), max(int(end - start), 0))
        if size <= 0:
            continue
        if size >= end - start:
            parts.append(np.arange(start, end, dtype=np.int64))
        else:
            parts.append(np.linspace(start, end - 1, size, dtype=np.int64))
    if not parts:
        return np.arange(0, min(n, k), dtype=np.int64)
    return np.unique(np.concatenate(parts).astype(np.int64))


def _fit_cycle_ae_gmm_state(
    *,
    frame: pd.DataFrame,
    ts_utc: pd.Series,
    reference_window: dict[str, Any],
    feature_columns: Sequence[str],
    input_feature_columns: Sequence[str] | None,
    max_train_rows: int,
    gmm_max_train_rows: int,
    ae_max_iter: int,
    artifact_dir: Path,
    seed: int,
    cluster_candidates: Sequence[int] | None = None,
    reg_covar_candidates: Sequence[float] | None = None,
    covariance_type_candidates: Sequence[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Fit and persist the one immutable AE/GMM state for this cycle."""
    excluded = set(map(str, AE_GMM_FEATURE_COLUMNS))
    requested = list(input_feature_columns or feature_columns)
    inputs = [
        str(col)
        for col in dict.fromkeys(requested)
        if str(col) in frame.columns and str(col) not in excluded
    ]
    if len(inputs) < 2:
        raise RuntimeError("Cycle AE/GMM requires at least two available input features")

    mask = ts_utc.lt(reference_window["valid_start"])
    if reference_window.get("train_start") is not None:
        mask = mask & ts_utc.ge(reference_window["train_start"])
    positions = np.flatnonzero(mask.to_numpy(dtype=bool, copy=False))
    if len(positions) < 500:
        raise RuntimeError(f"Cycle AE/GMM reference scope has only {len(positions)} rows")

    order_frame = frame.iloc[positions].loc[:, ["__ts__", "__symbol__", "side"]]
    reference_cap = max(int(max_train_rows), int(gmm_max_train_rows))
    local_sample = ae_gmm_cycle_reference_indices(
        order_frame["__ts__"],
        symbols=order_frame["__symbol__"],
        sides=order_frame["side"],
        max_rows=reference_cap,
    )
    sampled_positions = positions[local_sample]
    x_reference = (
        frame.iloc[sampled_positions]
        .loc[:, inputs]
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32, copy=False)
        .reset_index(drop=True)
    )
    med = x_reference.median(numeric_only=True)
    x_reference = x_reference.fillna(med).fillna(0.0).astype(np.float32, copy=False)
    frame_reference = frame.iloc[sampled_positions].reset_index(drop=True)
    reference_month = pd.to_datetime(
        frame_reference["__ts__"], utc=True, errors="coerce"
    ).dt.to_period("M").astype(str)
    selection_context = {
        "side": pd.to_numeric(
            frame_reference["side"], errors="coerce"
        ).fillna(1.0).to_numpy(dtype=np.float32),
        "time_bucket": pd.factorize(reference_month, sort=True)[0].astype(np.float32),
    }
    state = fit_ae_gmm_state(
        x_reference,
        economic_targets=selection_context,
        random_state=int(seed),
        max_train_rows=int(max_train_rows),
        gmm_max_train_rows=int(gmm_max_train_rows),
        ae_max_iter=int(ae_max_iter),
        cluster_candidates=cluster_candidates,
        reg_covar_candidates=reg_covar_candidates,
        covariance_type_candidates=covariance_type_candidates,
        require_both_sides=True,
        smooth_lambda_candidates=(0.0,),
        path_aware_hpo=False,
        temporal_stability_hpo=False,
        outcome_free=True,
        temporal_feature_contract="row_independent_v1",
    )
    if not bool(state.get("enabled", False)):
        raise RuntimeError("Cycle AE/GMM fit failed: " + str(state.get("reason", "disabled")))

    reference_ts = pd.to_datetime(frame_reference["__ts__"], utc=True, errors="coerce")
    state.update(
        {
            "cycle_contract_version": AE_GMM_CYCLE_CONTRACT_VERSION,
            "temporal_feature_contract": "row_independent_v1",
            "smooth_lambda": 0.0,
            "cycle_reference_fold": str(reference_window["fold"]),
            "cycle_reference_start": str(reference_ts.min()),
            "cycle_reference_end": str(reference_ts.max()),
            "cycle_reference_rows_available": int(len(positions)),
            "cycle_reference_rows_sampled": int(len(sampled_positions)),
            "cycle_reference_sample_policy": "beginning_middle_end_time_spread",
            "cycle_reference_ordering": "timestamp_utc,symbol,side",
            "cycle_reference_sample_identity_hash": ae_gmm_cycle_sample_identity_hash(
                frame_reference["__ts__"],
                symbols=frame_reference["__symbol__"],
                sides=frame_reference["side"],
            ),
            "cycle_reference_symbol_count": int(
                order_frame["__symbol__"].astype(str).nunique(dropna=False)
            ),
            "cycle_reference_sampled_symbol_count": int(
                frame_reference["__symbol__"].astype(str).nunique(dropna=False)
            ),
            "cycle_reference_side_counts": {
                str(key): int(value)
                for key, value in frame_reference["side"]
                .astype(str)
                .value_counts(dropna=False)
                .sort_index()
                .items()
            },
            "cycle_input_fill_values": {
                str(col): float(value) if np.isfinite(value) else 0.0
                for col, value in med.items()
            },
        }
    )
    state["cycle_state_hash"] = ae_gmm_learned_transform_hash(state)
    persisted = _persist_ae_gmm_state_artifact(
        state=state,
        artifact_dir=artifact_dir,
        artifact_name="cycle",
        scope="global",
        train_rows=len(x_reference),
        valid_rows=0,
        input_feature_count=len(inputs),
    )
    state_path = Path(str(persisted.get("ae_gmm_global_state_path", "")))
    if not state_path.is_file():
        raise RuntimeError("Cycle AE/GMM state artifact was not persisted")
    return state_path, {
        "contract_version": str(state["cycle_contract_version"]),
        "state_path": str(state_path),
        "state_hash": str(state["cycle_state_hash"]),
        "reference_fold": str(reference_window["fold"]),
        "reference_rows_available": int(len(positions)),
        "reference_rows_sampled": int(len(sampled_positions)),
        "reference_start": str(state["cycle_reference_start"]),
        "reference_end": str(state["cycle_reference_end"]),
        "input_feature_count": int(len(inputs)),
        "sample_policy": "beginning_middle_end_time_spread",
        "ordering": "timestamp_utc,symbol,side",
        "sample_identity_hash": str(
            state["cycle_reference_sample_identity_hash"]
        ),
        "representation_selection_outcome_free": True,
        "representation_selection_context_keys": ["side", "time_bucket"],
        "representation_selection_outcome_keys": [],
    }


def _auto_mda_keep_count(records: list[dict[str, Any]], requested_top_n: int) -> tuple[int, str, float]:
    """Choose feature count from MDA scores when no explicit top-k is requested."""
    return cumulative_positive_mda_keep_count(
        [float(row.get("score", 0.0) or 0.0) for row in records],
        requested_top_n=int(requested_top_n),
        maximum_feature_count=150,
    )


def _smallest_subset_within_fractional_se(
    rows: Sequence[Mapping[str, Any]], *, se_mult: float
) -> dict[str, Any]:
    valid = [
        dict(row)
        for row in rows
        if int(row.get("feature_count", 0) or 0) > 0
        and math.isfinite(float(row.get("mean_objective", float("nan"))))
    ]
    if not valid:
        return {}
    best = max(valid, key=lambda row: float(row["mean_objective"]))
    floor = float(best["mean_objective"]) - max(
        float(se_mult) * float(best.get("se_objective", 0.0) or 0.0), 0.0
    )
    eligible = [row for row in valid if float(row["mean_objective"]) >= floor]
    chosen = min(
        eligible or [best],
        key=lambda row: (
            int(row["feature_count"]),
            -float(row["mean_objective"]),
        ),
    )
    return {
        **chosen,
        "best_feature_count": int(best["feature_count"]),
        "best_mean_objective": float(best["mean_objective"]),
        "best_se_objective": float(best.get("se_objective", 0.0) or 0.0),
        "selection_floor": float(floor),
        "selection_se_mult": float(se_mult),
    }


def _post_mda_fractional_se_keep_count(
    *,
    ranked: Sequence[Mapping[str, Any]],
    cumulative_keep_n: int,
    x_train: pd.DataFrame,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    target: pd.DataFrame,
    sample_weight: pd.Series,
    params: Mapping[str, Any],
    fold: str,
    seed: int,
    se_mult: float = 0.75,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """Stability-select a nested subset inside the cumulative MDA pool."""

    max_n = max(1, min(int(cumulative_keep_n), len(ranked)))
    fractions = (0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00)
    candidate_sizes = sorted(
        {
            max(1, min(max_n, int(math.ceil(max_n * frac))))
            for frac in fractions
        }
        | {min(max_n, value) for value in (16, 24, 32, 48, 64) if value <= max_n}
    )
    eval_chunks = [
        chunk.astype(np.int64, copy=False)
        for chunk in np.array_split(np.arange(len(eval_idx), dtype=np.int64), 4)
        if len(chunk) >= 100
    ]
    if len(eval_chunks) < 2:
        return max_n, [], {
            "feature_count": max_n,
            "selection_status": "fractional_se_insufficient_eval_blocks",
            "selection_se_mult": float(se_mult),
        }
    fs_target = target.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for candidate_i, feature_count in enumerate(candidate_sizes, start=1):
        features = [str(row["feature"]) for row in ranked[:feature_count]]
        model = _fit_lgbm_model(
            x_train=x_train.iloc[fit_idx].loc[:, features].reset_index(drop=True),
            y_train=fs_target["target_soft"].iloc[fit_idx].reset_index(drop=True),
            w_train=sample_weight.iloc[fit_idx].reset_index(drop=True),
            params=dict(params),
            seed=int(seed) + candidate_i * 101,
        )
        x_eval = (
            x_train.iloc[eval_idx]
            .loc[:, features]
            .reset_index(drop=True)
            .astype(np.float32, copy=False)
        )
        pred = pd.Series(model.predict(x_eval).astype(np.float32))
        block_scores: list[float] = []
        for chunk in eval_chunks:
            block_scores.append(
                _mda_selection_objective(
                    valid=train_frame.iloc[eval_idx[chunk]].reset_index(drop=True),
                    metrics=train_metrics.iloc[eval_idx[chunk]].reset_index(drop=True),
                    target=fs_target.iloc[eval_idx[chunk]].reset_index(drop=True),
                    pred=pred.iloc[chunk].reset_index(drop=True),
                    fold=f"{fold}__se_block",
                )
            )
        score_arr = np.asarray(block_scores, dtype=np.float64)
        finite = score_arr[np.isfinite(score_arr)]
        mean_score = float(np.mean(finite)) if finite.size else float("nan")
        std_score = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        rows.append(
            {
                "feature_count": int(feature_count),
                "mean_objective": mean_score,
                "std_objective": std_score,
                "se_objective": (
                    std_score / math.sqrt(float(finite.size)) if finite.size else 0.0
                ),
                "fold_count": int(finite.size),
            }
        )
    chosen = _smallest_subset_within_fractional_se(rows, se_mult=float(se_mult))
    keep_n = int(chosen.get("feature_count", max_n))
    return max(1, min(keep_n, max_n)), rows, chosen


def _select_features_by_univariate(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    target: pd.Series,
    *,
    top_n: int,
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    if int(top_n) <= 0 or x_train.shape[1] <= int(top_n):
        rows = [
            {
                "fold": str(fold),
                "feature": str(col),
                "score": float("nan"),
                "rank": int(i + 1),
                "selected": True,
                "feature_selection_status": "disabled_or_not_needed",
            }
            for i, col in enumerate(x_train.columns)
        ]
        return x_train, x_valid, list(x_train.columns), pd.DataFrame(rows)
    y = _safe_numeric(target).replace([np.inf, -np.inf], np.nan)
    valid_y = y.notna()
    if int(valid_y.sum()) < 100 or int(y.loc[valid_y].nunique(dropna=True)) < 3:
        keep = list(x_train.columns[: int(top_n)])
        rows = [
            {
                "fold": str(fold),
                "feature": str(col),
                "score": float("nan"),
                "rank": int(i + 1),
                "selected": col in keep,
                "feature_selection_status": "insufficient_target_variation",
            }
            for i, col in enumerate(x_train.columns)
        ]
        return x_train.loc[:, keep], x_valid.loc[:, keep], keep, pd.DataFrame(rows)
    x = x_train.loc[valid_y].astype(np.float32, copy=False)
    yr = y.loc[valid_y].rank(method="average").to_numpy(dtype=np.float32)
    yr -= float(np.nanmean(yr))
    yr_std = float(np.nanstd(yr))
    if yr_std <= 1e-12:
        yr_std = 1.0
    yr /= yr_std
    scores: list[tuple[str, float]] = []
    for col in x.columns:
        ser = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(ser.notna().sum()) < 100 or int(ser.nunique(dropna=True)) < 3:
            scores.append((str(col), 0.0))
            continue
        xr = ser.rank(method="average").fillna(ser.rank(method="average").median()).to_numpy(dtype=np.float32)
        xr -= float(np.nanmean(xr))
        xr_std = float(np.nanstd(xr))
        score = 0.0 if xr_std <= 1e-12 else float(abs(np.nanmean((xr / xr_std) * yr)))
        scores.append((str(col), score if math.isfinite(score) else 0.0))
    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    keep = [name for name, _score in ranked[: int(top_n)]]
    selected = set(keep)
    rows = [
        {
            "fold": str(fold),
            "feature": name,
            "score": float(score),
            "rank": int(rank),
            "selected": name in selected,
            "feature_selection_status": "ok",
        }
        for rank, (name, score) in enumerate(ranked, start=1)
    ]
    return x_train.loc[:, keep], x_valid.loc[:, keep], keep, pd.DataFrame(rows)


def _feature_selection_family(name: str) -> str:
    text = str(name)
    low = text.lower()
    if "gmm" in low or "cluster" in low or "mahal" in low:
        return "ae_gmm_cluster"
    if "ae_" in low or "dae_" in low or "reconstruction" in low or "latent" in low:
        return "ae_gmm_autoencoder"
    if low.startswith("ctx_") or "state_" in low or "regime" in low:
        return "context_regime"
    if "orderbook" in low or "book" in low or "depth" in low:
        return "orderbook"
    if "funding" in low or "open_interest" in low or "_oi" in low or "oi_" in low:
        return "perp_oi_funding"
    if "btc" in low or "eth" in low or "market" in low or "cross" in low:
        return "cross_market"
    if "spread" in low or "liquidity" in low or "volume" in low:
        return "liquidity_volume"
    if "residual" in low or "leaf" in low or "base_score" in low:
        return "model_residual_leaf"
    return "config_feature"


def _mda_selection_objective(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    pred: pd.Series,
    fold: str,
) -> float:
    rows = [
        _selection_metrics(
            valid=valid,
            metrics=metrics,
            target=target,
            pred=pred,
            month=str(fold),
            top_frac=float(frac),
            trial_name="feature_selection_mda",
        )
        for frac in TOP_FRACS
    ]
    return _objective_from_rows(rows)


def _fit_lgbm_model(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    params: dict[str, Any],
    seed: int,
) -> Any:
    if not _LIGHTGBM_AVAILABLE or LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for MDA feature selection")
    model = LGBMRegressor(
        objective=str(params.get("loss_function", "regression")),
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        min_split_gain=float(params.get("min_split_gain", 0.0)),
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(
        x_train.reset_index(drop=True),
        _safe_numeric(y_train).fillna(0.0).to_numpy(dtype=np.float32),
        sample_weight=_safe_numeric(w_train).fillna(1.0).to_numpy(dtype=np.float32),
    )
    return model


def _select_features_by_mda(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    target: pd.DataFrame,
    *,
    top_n: int,
    fold: str,
    seed: int,
    max_train_rows: int = 60_000,
    max_valid_rows: int = 20_000,
    repeats: int = 1,
    post_cumulative_se_mult: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    candidate_features = list(x_train.columns)
    candidate_family_counts = pd.Series([_feature_selection_family(c) for c in candidate_features]).value_counts()
    if int(top_n) > 0 and len(candidate_features) <= int(top_n):
        rows = [
            {
                "fold": str(fold),
                "feature": str(col),
                "feature_family": _feature_selection_family(str(col)),
                "score": float("nan"),
                "rank": int(i + 1),
                "selected": True,
                "feature_selection_method": "mda_topk_permutation",
                "feature_selection_status": "disabled_or_not_needed",
                "candidate_feature_count": int(len(candidate_features)),
                "candidate_family_count": int(candidate_family_counts.get(_feature_selection_family(str(col)), 0)),
            }
            for i, col in enumerate(candidate_features)
        ]
        return x_train, x_valid, candidate_features, pd.DataFrame(rows)
    if not _LIGHTGBM_AVAILABLE or LGBMRegressor is None:
        y = target["target_soft"] if "target_soft" in target else pd.Series(dtype=float)
        xtr, xva, keep, rows = _select_features_by_univariate(
            x_train,
            x_valid,
            y,
            top_n=int(top_n),
            fold=str(fold),
        )
        rows["feature_selection_method"] = "univariate_fallback_lightgbm_unavailable"
        rows["candidate_feature_count"] = int(len(candidate_features))
        return xtr, xva, keep, rows

    y = _safe_numeric(target["target_soft"]).replace([np.inf, -np.inf], np.nan)
    valid_y = y.notna()
    if int(valid_y.sum()) < 1_000 or int(y.loc[valid_y].nunique(dropna=True)) < 3:
        xtr, xva, keep, rows = _select_features_by_univariate(
            x_train,
            x_valid,
            y,
            top_n=int(top_n),
            fold=str(fold),
        )
        rows["feature_selection_method"] = "univariate_fallback_insufficient_target_variation"
        rows["candidate_feature_count"] = int(len(candidate_features))
        return xtr, xva, keep, rows

    valid_positions = np.flatnonzero(valid_y.to_numpy(dtype=bool))
    split_at = int(max(500, round(0.80 * len(valid_positions))))
    split_at = min(split_at, max(len(valid_positions) - 500, 1))
    fit_pos = valid_positions[:split_at]
    eval_pos = valid_positions[split_at:]
    if len(fit_pos) < 500 or len(eval_pos) < 500:
        xtr, xva, keep, rows = _select_features_by_univariate(
            x_train,
            x_valid,
            y,
            top_n=int(top_n),
            fold=str(fold),
        )
        rows["feature_selection_method"] = "univariate_fallback_insufficient_mda_split"
        rows["candidate_feature_count"] = int(len(candidate_features))
        return xtr, xva, keep, rows

    fit_idx = fit_pos[_time_spread_cap_rows(len(fit_pos), int(max_train_rows))]
    eval_idx = eval_pos[_time_spread_cap_rows(len(eval_pos), int(max_valid_rows))]
    print(
        "[feature_selection] mda_start "
        f"fold={fold} candidates={len(candidate_features)} train_rows={len(fit_idx)} eval_rows={len(eval_idx)}",
        flush=True,
    )
    fs_target = target.reset_index(drop=True)
    fs_weights = _weight_series(
        frame=train_frame.reset_index(drop=True),
        metrics=train_metrics.reset_index(drop=True),
        target=fs_target,
        arm="W0_base",
    )
    params = {
        "n_estimators": 180,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 3.0,
    }
    model = _fit_lgbm_model(
        x_train=x_train.iloc[fit_idx].reset_index(drop=True),
        y_train=fs_target["target_soft"].iloc[fit_idx].reset_index(drop=True),
        w_train=fs_weights.iloc[fit_idx].reset_index(drop=True),
        params=params,
        seed=int(seed),
    )
    x_eval = x_train.iloc[eval_idx].reset_index(drop=True).astype(np.float32, copy=False)
    valid_eval = train_frame.iloc[eval_idx].reset_index(drop=True)
    metrics_eval = train_metrics.iloc[eval_idx].reset_index(drop=True)
    target_eval = fs_target.iloc[eval_idx].reset_index(drop=True)
    baseline_pred = pd.Series(model.predict(x_eval).astype(np.float32))
    baseline_objective = _mda_selection_objective(
        valid=valid_eval,
        metrics=metrics_eval,
        target=target_eval,
        pred=baseline_pred,
        fold=str(fold),
    )
    rng = np.random.default_rng(int(seed) + 23)
    records: list[dict[str, Any]] = []
    base_values = x_eval.to_numpy(dtype=np.float32, copy=True)
    for j, feature in enumerate(candidate_features):
        drops: list[float] = []
        for rep in range(max(1, int(repeats))):
            x_perm = base_values.copy()
            order = rng.permutation(x_perm.shape[0])
            x_perm[:, j] = x_perm[order, j]
            pred_perm = pd.Series(model.predict(pd.DataFrame(x_perm, columns=candidate_features)).astype(np.float32))
            perm_objective = _mda_selection_objective(
                valid=valid_eval,
                metrics=metrics_eval,
                target=target_eval,
                pred=pred_perm,
                fold=str(fold),
            )
            drops.append(float(baseline_objective - perm_objective))
        mean_drop = float(np.nanmean(drops)) if drops else 0.0
        records.append(
            {
                "fold": str(fold),
                "feature": str(feature),
                "feature_family": _feature_selection_family(str(feature)),
                "score": mean_drop if math.isfinite(mean_drop) else 0.0,
                "mda_mean": mean_drop if math.isfinite(mean_drop) else 0.0,
                "mda_std": float(np.nanstd(drops)) if len(drops) > 1 else 0.0,
                "mda_repeats": int(max(1, int(repeats))),
                "mda_baseline_objective": float(baseline_objective),
                "mda_eval_rows": int(len(eval_idx)),
                "mda_train_rows": int(len(fit_idx)),
                "feature_selection_method": "mda_topk_permutation",
                "feature_selection_status": "ok",
                "candidate_feature_count": int(len(candidate_features)),
                "candidate_family_count": int(candidate_family_counts.get(_feature_selection_family(str(feature)), 0)),
            }
            )
    ranked = sorted(records, key=lambda row: float(row["score"]), reverse=True)
    keep_n, selection_status, score_floor = _auto_mda_keep_count(ranked, int(top_n))
    subset_rows: list[dict[str, Any]] = []
    subset_choice: dict[str, Any] = {}
    if post_cumulative_se_mult is not None and int(top_n) <= 0:
        keep_n, subset_rows, subset_choice = _post_mda_fractional_se_keep_count(
            ranked=ranked,
            cumulative_keep_n=int(keep_n),
            x_train=x_train,
            fit_idx=np.asarray(fit_idx, dtype=np.int64),
            eval_idx=np.asarray(eval_idx, dtype=np.int64),
            train_frame=train_frame.reset_index(drop=True),
            train_metrics=train_metrics.reset_index(drop=True),
            target=fs_target,
            sample_weight=fs_weights.reset_index(drop=True),
            params=params,
            fold=str(fold),
            seed=int(seed) + 701,
            se_mult=float(post_cumulative_se_mult),
        )
        selection_status = "auto_mda_cumulative_positive_99pct_then_fractional_se"
    keep = [str(row["feature"]) for row in ranked[:keep_n]]
    print(
        "[feature_selection] mda_done "
        f"fold={fold} selected={len(keep)} baseline_objective={float(baseline_objective):.6f}",
        flush=True,
    )
    selected = set(keep)
    rows = []
    for rank, row in enumerate(ranked, start=1):
        out = dict(row)
        out["rank"] = int(rank)
        out["selected"] = str(row["feature"]) in selected
        out["feature_selection_status"] = selection_status
        out["feature_selection_requested_top_n"] = int(top_n)
        out["feature_selection_auto_score_floor"] = float(score_floor)
        out["feature_selection_auto_selected_count"] = int(keep_n)
        out["post_mda_selection_se_mult"] = (
            float(post_cumulative_se_mult)
            if post_cumulative_se_mult is not None
            else np.nan
        )
        out["post_mda_selection_floor"] = float(
            subset_choice.get("selection_floor", np.nan)
        )
        out["post_mda_best_feature_count"] = int(
            subset_choice.get("best_feature_count", keep_n)
        )
        out["post_mda_best_mean_objective"] = float(
            subset_choice.get("best_mean_objective", np.nan)
        )
        out["post_mda_subset_evaluations"] = json.dumps(
            _json_safe(subset_rows), separators=(",", ":")
        )
        rows.append(out)
    return x_train.loc[:, keep], x_valid.loc[:, keep], keep, pd.DataFrame(rows)


def _first_existing_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((str(name) for name in names if str(name) in frame.columns), None)


def _select_features_by_archetype_prescreen_side_mda(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    target: pd.DataFrame,
    *,
    fold: str,
    seed: int,
    report_dir: Path,
    correlation_first: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    """Run archetype-aware prescreens followed by unweighted side MDA.

    The downstream HPO/model fitter remains this runner's implementation. This
    adapter uses lgbm_pipeline only to obtain a feature contract, keeping the
    ablation isolated to feature selection. Archetypes affect only the cheap
    univariate/Relief screens; MDA is fitted per side across all archetypes.
    """
    if len(x_train) != len(train_frame) or len(x_train) != len(train_metrics):
        raise ValueError("staged archetype MDA inputs must be row-aligned")
    archetype_col = _first_existing_column(
        train_frame,
        (
            "__archetype_label_family__",
            "archetype_label_family",
            "policy_archetype",
            "local_side_archetype",
            "source_archetype",
        ),
    )
    if archetype_col is None:
        raise RuntimeError(
            "staged archetype MDA requires an observable archetype label column"
        )
    side_col = _first_existing_column(
        train_frame, ("side_name", "__side__", "side")
    )
    symbol_col = _first_existing_column(
        train_frame, ("__symbol__", "symbol", "asset", "instrument")
    )
    ts_col = _first_existing_column(
        train_frame, ("__ts__", "timestamp", "datetime", "date")
    )
    if side_col is None or ts_col is None:
        raise RuntimeError(
            "staged archetype MDA requires side and timestamp context"
        )

    y_soft = _safe_numeric(target["target_soft"]).fillna(0.0).to_numpy(
        dtype=np.float32
    )
    hard_col = _first_existing_column(
        train_metrics,
        ("first_touch_net_positive", "clean_first_touch_exec", "y_bin"),
    )
    return_col = _first_existing_column(
        train_metrics,
        ("first_touch_net", "ret_net", "u_policy_net", "return"),
    )
    hard = (
        _safe_numeric(train_metrics[hard_col]).fillna(0.0).to_numpy(dtype=np.float32)
        if hard_col is not None
        else np.asarray(y_soft >= 0.5, dtype=np.float32)
    )
    returns = (
        _safe_numeric(train_metrics[return_col]).fillna(0.0).to_numpy(dtype=np.float32)
        if return_col is not None
        else y_soft.copy()
    )
    sample_weight = _weight_series(
        frame=train_frame.reset_index(drop=True),
        metrics=train_metrics.reset_index(drop=True),
        target=target.reset_index(drop=True),
        arm="W0_base",
    ).to_numpy(dtype=np.float32)
    side_raw = train_frame[side_col]
    side_numeric = pd.to_numeric(side_raw, errors="coerce")
    side_text = side_raw.astype(str).str.strip().str.lower()
    side_name = np.where(
        side_text.str.contains("short", regex=False).to_numpy()
        | (side_numeric.fillna(1.0).to_numpy(dtype=np.float32) < 0.0),
        "short",
        "long",
    )
    archetype_family = train_frame[archetype_col].astype(str).to_numpy()
    side_archetype = np.asarray(
        [f"{side}__{archetype}" for side, archetype in zip(side_name, archetype_family)],
        dtype=object,
    )
    label_context: dict[str, Any] = {
        "feature_selection_archetype": side_archetype,
        "side_name": side_name,
        "side": side_name,
        "y_ret": returns,
        "y_bin": hard,
    }
    for source, target_name in (
        ("first_touch_mae_to_sl", "bad_mae_1r"),
        ("first_touch_timeout", "is_timeout"),
    ):
        if source in train_metrics.columns:
            label_context[target_name] = _safe_numeric(
                train_metrics[source]
            ).fillna(0.0).to_numpy(dtype=np.float32)

    report_dir.mkdir(parents=True, exist_ok=True)
    selector_params = {
        "n_estimators": 180,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 3.0,
        "min_split_gain": 0.0,
        "objective": "cross_entropy",
        "verbosity": -1,
        "n_jobs": 4,
    }
    result = train_lgbm_stability_candidate(
        x_train,
        y_soft,
        sample_weight=sample_weight,
        random_state=int(seed),
        mode="classifier",
        timestamps=pd.to_datetime(train_frame[ts_col], utc=True, errors="coerce")
        .astype("int64")
        .to_numpy(),
        assets=(
            train_frame[symbol_col].astype(str).to_numpy()
            if symbol_col is not None
            else np.arange(len(train_frame)).astype(str)
        ),
        returns=returns,
        hard_labels=hard,
        hpo_objective_mode="train_base",
        preset_best_params=selector_params,
        preset_source="archetype_prescreen_side_mda_selection_only",
        reference_artifact_dir=report_dir,
        cfg={
            "mda_config": {
                "archetype_conditioned_enabled": False,
                "side_tail_across_archetypes_unweighted": True,
                "correlation_pruning_before_prescreen": bool(correlation_first),
                "correlation_pruning_floor_ratio": 0.50,
                "correlation_pruning_floor_count": 300,
                "report_dir": str(report_dir / "mda"),
            },
            "lgbm_joint_complete_case_filter_enabled": False,
        },
        label_context=label_context,
    )
    if not result:
        raise RuntimeError("canonical staged archetype selector returned no result")
    selected = [
        str(feature)
        for feature in result.get("selected_feature_names", [])
        if str(feature) in x_train.columns
    ]
    if len(selected) < 2:
        raise RuntimeError(
            f"canonical staged archetype selector retained only {len(selected)} features"
        )
    stats = result.get("feature_stats")
    report = stats.copy() if isinstance(stats, pd.DataFrame) else pd.DataFrame()
    if report.empty or "feature" not in report.columns:
        report = pd.DataFrame({"feature": list(x_train.columns)})
    report["fold"] = str(fold)
    report["selected"] = report["feature"].astype(str).isin(selected)
    report["feature_selection_method"] = (
        "archetype_prescreen_side_mda_corrfirst"
        if correlation_first
        else "archetype_prescreen_side_mda"
    )
    report["feature_selection_status"] = "ok"
    report["candidate_feature_count"] = int(x_train.shape[1])
    report["selected_feature_count"] = int(len(selected))
    metrics = result.get("metrics", {}) or {}
    selected_by_side = dict(
        metrics.get("per_side_feature_selection_selected_features", {}) or {}
    )
    for side_name in ("long", "short"):
        local = {str(value) for value in selected_by_side.get(side_name, [])}
        report[f"selected_{side_name}"] = report["feature"].astype(str).isin(local)
    (report_dir / "staged_archetype_selector_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True)
    )
    pd.DataFrame(result.get("univariate_stats", pd.DataFrame())).to_csv(
        report_dir / "univariate_archetype_stats.csv", index=False
    )
    pd.DataFrame(result.get("relief_stats", pd.DataFrame())).to_csv(
        report_dir / "relief_archetype_stats.csv", index=False
    )
    return (
        x_train.loc[:, selected],
        x_valid.reindex(columns=selected),
        selected,
        report,
    )


def _fit_predict_lgbm(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
) -> pd.Series:
    if not _LIGHTGBM_AVAILABLE or LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for this HPO")
    model = LGBMRegressor(
        objective=str(params.get("loss_function", "regression")),
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        min_split_gain=float(params.get("min_split_gain", 0.0)),
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(
        x_train.reset_index(drop=True),
        _safe_numeric(y_train).fillna(0.0).to_numpy(dtype=np.float32),
        sample_weight=_safe_numeric(w_train).fillna(1.0).to_numpy(dtype=np.float32),
    )
    return pd.Series(model.predict(x_valid.reset_index(drop=True)).astype(np.float32))


def _side_name_array(frame: pd.DataFrame) -> np.ndarray:
    if "side_name" in frame.columns:
        values = frame["side_name"].astype(str).str.lower().to_numpy()
        return np.where(values == "short", "short", "long")
    raw = pd.to_numeric(
        frame.get("__side__", frame.get("side", pd.Series(1.0, index=frame.index))),
        errors="coerce",
    ).to_numpy(dtype=np.float64, copy=False)
    return np.where(raw < 0.0, "short", "long")


def _fit_lgbm_models(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    train_sides: np.ndarray,
    params: dict[str, Any],
    seed: int,
    model_side_scope: str,
    features_by_side: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Any, dict[str, list[str]]]:
    scope = str(model_side_scope).strip().lower()
    if scope == "shared":
        return (
            _fit_lgbm_model(
                x_train=x_train,
                y_train=y_train,
                w_train=w_train,
                params=params,
                seed=seed,
            ),
            {"shared": list(x_train.columns)},
        )
    if scope != "per_side":
        raise ValueError(f"Unknown model_side_scope: {model_side_scope!r}")
    models: dict[str, Any] = {}
    contracts: dict[str, list[str]] = {}
    side_values = np.asarray(train_sides, dtype=str)
    for offset, side_name in enumerate(("long", "short"), start=1):
        mask = side_values == side_name
        if int(mask.sum()) < 500:
            raise RuntimeError(
                f"Per-side base fit has insufficient {side_name} rows: {int(mask.sum())}"
            )
        requested = list((features_by_side or {}).get(side_name, ()))
        columns = [str(col) for col in requested if str(col) in x_train.columns]
        if not columns:
            columns = list(x_train.columns)
        contracts[side_name] = columns
        models[side_name] = _fit_lgbm_model(
            x_train=x_train.loc[mask, columns].reset_index(drop=True),
            y_train=y_train.loc[mask].reset_index(drop=True),
            w_train=w_train.loc[mask].reset_index(drop=True),
            params=params,
            seed=int(seed) + offset * 10_003,
        )
    return models, contracts


def _predict_lgbm_models(
    *,
    models: Any,
    x_valid: pd.DataFrame,
    valid_sides: np.ndarray,
    model_side_scope: str,
    feature_contracts: Mapping[str, Sequence[str]],
) -> pd.Series:
    scope = str(model_side_scope).strip().lower()
    if scope == "shared":
        return pd.Series(
            models.predict(x_valid.loc[:, list(feature_contracts["shared"])]).astype(
                np.float32
            )
        )
    output = np.full(len(x_valid), np.nan, dtype=np.float32)
    sides = np.asarray(valid_sides, dtype=str)
    for side_name in ("long", "short"):
        mask = sides == side_name
        if not mask.any():
            continue
        columns = list(feature_contracts[side_name])
        output[mask] = models[side_name].predict(
            x_valid.loc[mask, columns].reset_index(drop=True)
        ).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("Per-side base scoring emitted non-finite predictions")
    return pd.Series(output)


def _selection_metrics(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    pred: pd.Series,
    month: str,
    top_frac: float,
    trial_name: str,
) -> dict[str, Any]:
    side_values = valid.get(
        "__side__", valid.get("side", pd.Series(1.0, index=valid.index))
    )
    ranks = _timestamp_side_ranks(valid, pred, side_values)
    selected_mask = ranks["rank"].to_numpy(dtype=np.int64) <= np.ceil(
        ranks["group_rows"].to_numpy(dtype=np.float64) * float(top_frac)
    ).astype(np.int64)
    idx = np.flatnonzero(selected_mask)
    selected = valid.iloc[idx].reset_index(drop=True) if len(idx) else valid.iloc[:0].copy()
    sm = metrics.iloc[idx].reset_index(drop=True) if len(idx) else metrics.iloc[:0].copy()
    st = target.iloc[idx].reset_index(drop=True) if len(idx) else target.iloc[:0].copy()
    clean = _safe_numeric(st.get("target_hard", pd.Series(dtype=float))).fillna(0.0).clip(0.0, 1.0)
    net = _safe_numeric(sm.get("first_touch_net", sm.get("u_policy_net", pd.Series(dtype=float)))).fillna(0.0)
    cost = _safe_numeric(sm.get("round_trip_cost", pd.Series(0.0, index=sm.index))).fillna(0.0)
    gross = (net + cost).clip(lower=0.0)
    gross_denom = float(gross.sum())
    side = _safe_numeric(sm.get("side", pd.Series(1.0, index=sm.index))).fillna(1.0)
    return {
        "trial_name": str(trial_name),
        "month": str(month),
        "top_frac": float(top_frac),
        "rows": int(len(valid)),
        "selected_rows": int(len(selected)),
        "selected_symbols": int(selected["__symbol__"].nunique(dropna=True)) if "__symbol__" in selected.columns else 0,
        "clean_precision": _safe_mean(clean),
        "gross_ev_weighted_clean_precision": float((clean * gross).sum() / gross_denom) if gross_denom > 0.0 else float("nan"),
        "mean_first_touch_net": _safe_mean(net),
        "mean_first_touch_gross": _safe_mean(net + cost),
        "q10_first_touch_net": _safe_quantile(net, 0.10),
        "hit_first_touch_net": _safe_mean(net > 0.0),
        "first_touch_stop_rate": _safe_mean(sm.get("first_touch_stop", pd.Series(dtype=float))),
        "first_touch_timeout_rate": _safe_mean(sm.get("first_touch_timeout", pd.Series(dtype=float))),
        "first_touch_bad_mae_to_sl_rate": _safe_mean(
            _safe_numeric(sm.get("first_touch_mae_to_sl", pd.Series(dtype=float))).ge(1.0)
        ),
        "p90_first_touch_mae_to_sl": _safe_quantile(sm.get("first_touch_mae_to_sl", pd.Series(dtype=float)), 0.90),
        "p90_first_touch_bar": _safe_quantile(sm.get("first_touch_bar", pd.Series(dtype=float)), 0.90),
        "bad_mae_1r_rate": _safe_mean(_safe_numeric(sm.get("mae_norm", pd.Series(dtype=float))).ge(1.0)),
        "long_share": _safe_mean(side.ge(0.0)) if len(side) else float("nan"),
        "short_share": _safe_mean(side.lt(0.0)) if len(side) else float("nan"),
        "top_symbol_share": float(selected["__symbol__"].astype(str).value_counts(normalize=True).iloc[0])
        if len(selected) and "__symbol__" in selected.columns
        else float("nan"),
    }


def _validation_windows(
    months: Sequence[str],
    *,
    max_oos_model_age_days: int,
    single_fit_oos_window: bool,
) -> list[dict[str, Any]]:
    periods = sorted(pd.Period(month) for month in months)
    if not periods:
        return []
    if bool(single_fit_oos_window):
        expected = [periods[0] + i for i in range(len(periods))]
        if periods != expected:
            raise ValueError(
                "single_fit_oos_window requires contiguous evaluation months; "
                f"got={[str(period) for period in periods]}"
            )
        start = pd.Timestamp(periods[0].start_time, tz="UTC")
        end = pd.Timestamp((periods[-1] + 1).start_time, tz="UTC")
        return [
            {
                "fold": f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                "month": f"{periods[0]}_to_{periods[-1]}",
                "valid_start": start,
                "valid_end": end,
            }
        ]

    windows: list[dict[str, Any]] = []
    contiguous = periods == [periods[0] + i for i in range(len(periods))]
    if int(max_oos_model_age_days) > 0 and contiguous:
        start = pd.Timestamp(periods[0].start_time, tz="UTC")
        scope_end = pd.Timestamp((periods[-1] + 1).start_time, tz="UTC")
        step = pd.Timedelta(days=int(max_oos_model_age_days))
        while start < scope_end:
            end = min(start + step, scope_end)
            windows.append(
                {
                    "fold": f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                    "month": f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                    "valid_start": start,
                    "valid_end": end,
                }
            )
            start = end
        return windows

    for period in periods:
        month_start = pd.Timestamp(period.start_time, tz="UTC")
        month_end = pd.Timestamp((period + 1).start_time, tz="UTC")
        if int(max_oos_model_age_days) > 0:
            start = month_start
            step = pd.Timedelta(days=int(max_oos_model_age_days))
            while start < month_end:
                end = min(start + step, month_end)
                windows.append(
                    {
                        "fold": f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                        "month": str(period),
                        "valid_start": start,
                        "valid_end": end,
                    }
                )
                start = end
        else:
            windows.append(
                {
                    "fold": str(period),
                    "month": str(period),
                    "valid_start": month_start,
                    "valid_end": month_end,
                }
            )
    return windows


def _prepare_folds(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_gmm_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    feature_selection_top_n: int,
    feature_selection_target_mode: str,
    feature_selection_method: str,
    max_oos_model_age_days: int,
    single_fit_oos_window: bool,
    train_window_days: int,
    label_path_purge_hours: float,
    ae_gmm_anchor_days: int,
    payload_max_train_rows: int,
    fold_cache_dir: Path | None,
    fixed_selected_features: list[str] | None,
    fixed_selected_features_by_side: Mapping[str, Sequence[str]] | None,
    fixed_selected_features_path: Path | None,
    fixed_ae_gmm_state_pkl: Path | None,
    ae_gmm_input_features: list[str] | None,
    freeze_ae_gmm_state_after_reference: bool,
    existing_scored_ledger_path: Path | None,
    missing_only: bool,
    seed: int,
    selection_only: bool = False,
    fixed_training_contract: Mapping[str, Any] | None = None,
    external_feature_sidecar_path: Path | None = None,
    target_sidecar_path: Path | None = None,
    frozen_ae_gmm_output_sidecar_path: Path | None = None,
    run_reuse_fingerprint: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame, label_projection = _load_projected_labels(
        labels_path,
        selected_features=fixed_selected_features,
        ae_gmm_input_features=(
            []
            if frozen_ae_gmm_output_sidecar_path is not None
            else ae_gmm_input_features
        ),
        external_feature_sidecar_path=external_feature_sidecar_path,
        target_sidecar_path=target_sidecar_path,
        frozen_ae_gmm_output_sidecar_path=frozen_ae_gmm_output_sidecar_path,
    )
    frame, label_resolution_contract = _base_oof_provenance_columns(frame)
    all_months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    if not months:
        months = all_months[1:]
    folds: list[dict[str, Any]] = []
    ts_utc = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    existing_scored_keys = _load_existing_scored_keys(existing_scored_ledger_path) if bool(missing_only) else set()
    periods = sorted(pd.Period(month) for month in months)
    contiguous_months = bool(periods) and periods == [periods[0] + i for i in range(len(periods))]
    validation_windows = _validation_windows(
        months,
        max_oos_model_age_days=int(max_oos_model_age_days),
        single_fit_oos_window=bool(single_fit_oos_window),
    )
    global_selected_features: list[str] | None = list(fixed_selected_features or []) or None
    global_selected_features_by_side: dict[str, list[str]] | None = None
    if fixed_selected_features_by_side:
        global_selected_features_by_side = {
            side: [str(col) for col in fixed_selected_features_by_side.get(side, ())]
            for side in ("long", "short")
        }
    elif global_selected_features is not None:
        global_selected_features_by_side = {
            "long": list(global_selected_features),
            "short": list(global_selected_features),
        }
    global_feature_selection_df: pd.DataFrame | None = None
    if global_selected_features is not None:
        global_feature_selection_df = pd.DataFrame(
            {
                "fold": ["fixed_selected_features"] * len(global_selected_features),
                "feature": list(global_selected_features),
                "score": np.nan,
                "rank": np.arange(1, len(global_selected_features) + 1, dtype=np.int32),
                "selected": True,
                "feature_selection_method": "fixed_selected_features",
                "feature_selection_status": "fixed_replay",
            }
        )
    eligible_windows: list[dict[str, Any]] = []
    for window in validation_windows:
        train_start = (
            window["valid_start"] - pd.Timedelta(days=int(train_window_days))
            if int(train_window_days) > 0
            else None
        )
        train_cutoff = window["valid_start"] - pd.Timedelta(
            hours=float(label_path_purge_hours)
        )
        train_mask_window = ts_utc.lt(train_cutoff)
        if train_start is not None:
            train_mask_window = train_mask_window & ts_utc.ge(train_start)
        train_rows = int(train_mask_window.sum())
        valid_mask_window = ts_utc.ge(window["valid_start"]) & ts_utc.lt(window["valid_end"])
        valid_rows_raw = int(valid_mask_window.sum())
        if bool(missing_only) and existing_scored_keys:
            valid_frame_window = frame.loc[valid_mask_window, ["__ts__", "__symbol__", "side"]]
            valid_rows = int(np.sum(_missing_against_existing_mask(valid_frame_window, existing_scored_keys)))
        else:
            valid_rows = int(valid_rows_raw)
        if train_rows < 500 or valid_rows < 100:
            continue
        enriched = dict(window)
        enriched["train_start"] = train_start
        enriched["train_cutoff"] = train_cutoff
        enriched["label_path_purge_hours"] = float(label_path_purge_hours)
        enriched["ae_gmm_anchor_start"] = (
            train_start - pd.Timedelta(days=int(ae_gmm_anchor_days))
            if train_start is not None and int(ae_gmm_anchor_days) > 0
            else None
        )
        enriched["ae_gmm_anchor_end"] = train_start if train_start is not None and int(ae_gmm_anchor_days) > 0 else None
        enriched["train_rows_estimate"] = int(train_rows)
        enriched["valid_rows_estimate"] = int(valid_rows)
        enriched["valid_rows_raw_estimate"] = int(valid_rows_raw)
        eligible_windows.append(enriched)
    fs_window_fold = None
    if eligible_windows:
        fs_window = max(eligible_windows, key=lambda w: (int(w["train_rows_estimate"]), int(w["valid_rows_estimate"])))
        fs_window_fold = str(fs_window["fold"])
        ordered_windows = [fs_window] + [w for w in eligible_windows if str(w["fold"]) != fs_window_fold]
    else:
        ordered_windows = []
    early_manifest: dict[str, Any] = {
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)) if "__symbol__" in frame.columns else 0,
        "timestamp_min": frame["__ts__"].min() if "__ts__" in frame.columns else None,
        "timestamp_max": frame["__ts__"].max() if "__ts__" in frame.columns else None,
        "fold_months_requested": list(months),
        "missing_only": bool(missing_only),
        "existing_scored_ledger_path": str(existing_scored_ledger_path) if existing_scored_ledger_path is not None else None,
        "existing_scored_key_count": int(len(existing_scored_keys)),
        "eligible_window_count": int(len(ordered_windows)),
        "eligible_windows": [
            {
                "fold": str(window["fold"]),
                "month": str(window["month"]),
                "valid_start": window["valid_start"],
                "valid_end": window["valid_end"],
                "train_start": window.get("train_start"),
                "ae_gmm_anchor_start": window.get("ae_gmm_anchor_start"),
                "ae_gmm_anchor_end": window.get("ae_gmm_anchor_end"),
                "train_rows_estimate": int(window.get("train_rows_estimate", 0)),
                "valid_rows_estimate": int(window.get("valid_rows_estimate", 0)),
                "valid_rows_raw_estimate": int(window.get("valid_rows_raw_estimate", 0)),
            }
            for window in ordered_windows
        ],
        "label_projection": label_projection,
    }
    if not ordered_windows:
        return [], early_manifest
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    if fixed_selected_features:
        required_source_features = set(map(str, fixed_selected_features))
        if frozen_ae_gmm_output_sidecar_path is None:
            required_source_features.update(map(str, ae_gmm_input_features or []))
        selected_features = [
            feature for feature in selected_features if feature in required_source_features
        ]
    # The shared static store is the authoritative source for observable model
    # features. Label shards may carry an older materialized copy of some
    # columns; reusing those values makes training depend on label-generation
    # time while replay/inference read the current static endpoint. Request the
    # full candidate contract and overwrite every column the store can supply.
    # Outcome/archetype-only columns are retained from labels when the static
    # endpoint does not expose them.
    embedded_candidate_features = {
        str(feature) for feature in selected_features if str(feature) in frame.columns
    }
    store_features = list(dict.fromkeys(map(str, selected_features)))
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=store_features,
    )
    authoritative_store_features = list(map(str, feature_matrix.columns))
    excluded_embedded_candidate_features = sorted(
        embedded_candidate_features.difference(authoritative_store_features)
    )
    feature_report.update(
        {
            "candidate_features_before_existing_filter": int(len(selected_features)),
            "embedded_candidate_feature_count": int(len(embedded_candidate_features)),
            "authoritative_store_feature_count": int(len(authoritative_store_features)),
            "excluded_embedded_candidate_feature_count": int(
                len(excluded_embedded_candidate_features)
            ),
            "excluded_embedded_candidate_features": excluded_embedded_candidate_features,
            "requested_store_features": int(len(store_features)),
            # Backward-compatible report keys. Embedded observable copies are
            # retained in the label frame for diagnostics only and are never
            # reused by the model contract.
            "existing_frame_features_reused": 0,
            "requested_missing_features": int(len(store_features)),
            "existing_columns_reused": False,
            "observable_feature_source_policy": "shared_static_store_authoritative_v1",
        }
    )
    if not feature_matrix.empty:
        frame = _merge_authoritative_store_features(frame, feature_matrix)
    metrics = _first_touch_eval_metrics(frame, _path_metrics(frame)).reset_index(drop=True)
    # A frozen model contract may deliberately select AE/GMM outputs.  Those
    # are generated after the raw feature-store matrix is assembled, so only
    # validate/load the raw subset at this point.  The complete contract is
    # validated immediately after the frozen state transform below.
    fixed_generated_features = set(
        _fixed_selected_ae_gmm_features(fixed_selected_features)
    )
    raw_fixed_features = (
        [
            str(feature)
            for feature in fixed_selected_features
            if str(feature) not in fixed_generated_features
        ]
        if fixed_selected_features
        else None
    )
    features = _resolve_base_model_features(
        frame,
        raw_fixed_features,
        authoritative_store_features=authoritative_store_features,
    )
    fold_frame_columns = _fold_frame_columns(frame)
    if fold_cache_dir is not None:
        fold_cache_dir.mkdir(parents=True, exist_ok=True)
    active_fixed_ae_gmm_state_pkl = fixed_ae_gmm_state_pkl
    frozen_ae_gmm_reference_fold: str | None = (
        "external_fixed_state" if fixed_ae_gmm_state_pkl is not None else None
    )
    frozen_ae_gmm_reference_state_path: str | None = (
        str(fixed_ae_gmm_state_pkl) if fixed_ae_gmm_state_pkl is not None else None
    )
    cycle_ae_gmm_contract: dict[str, Any] = {
        "contract_version": "external_fixed_state"
        if fixed_ae_gmm_state_pkl is not None
        else "disabled",
        "state_path": str(fixed_ae_gmm_state_pkl)
        if fixed_ae_gmm_state_pkl is not None
        else None,
    }
    cycle_input_preflight: dict[str, Any] = {
        "enabled": False,
        "reason": "external_fixed_state"
        if fixed_ae_gmm_state_pkl is not None
        else "ae_gmm_disabled",
    }
    if bool(include_ae_gmm_state_features) and active_fixed_ae_gmm_state_pkl is None:
        if fold_cache_dir is None:
            raise RuntimeError("Cycle AE/GMM fitting requires a fold cache artifact directory")
        # Feature selection/HPO use the largest available training fold. Fit
        # the cycle representation on that same reference scope, with rows
        # sampled across its beginning, middle and end, then replay the frozen
        # transform backward/forward for every growing OOS window.
        reference_window = max(
            ordered_windows,
            key=lambda window: (
                int(window.get("train_rows_estimate", 0) or 0),
                pd.Timestamp(window["valid_start"]).value,
            ),
        )
        cycle_input_candidates = list(ae_gmm_input_features or features)
        cycle_input_features, cycle_input_preflight = (
            _cycle_reference_input_survivors(
                frame=frame,
                ts_utc=ts_utc,
                reference_window=reference_window,
                candidate_features=cycle_input_candidates,
                payload_max_train_rows=int(payload_max_train_rows),
            )
        )
        if len(cycle_input_features) < 2:
            raise RuntimeError(
                "Cycle AE/GMM has fewer than two inputs after the reference "
                "train/OOS availability and joint-coverage contract: "
                f"{cycle_input_preflight}"
            )
        print(
            "[prepare_folds] cycle_input_preflight "
            f"candidates={len(cycle_input_candidates)} "
            f"survivors={len(cycle_input_features)} "
            f"status={cycle_input_preflight.get('reason')}",
            flush=True,
        )
        active_fixed_ae_gmm_state_pkl, cycle_ae_gmm_contract = _fit_cycle_ae_gmm_state(
            frame=frame,
            ts_utc=ts_utc,
            reference_window=reference_window,
            feature_columns=features,
            input_feature_columns=cycle_input_features,
            max_train_rows=int(ae_gmm_state_feature_max_train_rows),
            gmm_max_train_rows=int(ae_gmm_state_feature_gmm_max_train_rows),
            ae_max_iter=int(ae_gmm_state_feature_max_iter),
            artifact_dir=fold_cache_dir.parent / "ae_gmm_states",
            seed=int(seed),
        )
        cycle_ae_gmm_contract["input_preflight"] = cycle_input_preflight
        frozen_ae_gmm_reference_fold = str(reference_window["fold"])
        frozen_ae_gmm_reference_state_path = str(active_fixed_ae_gmm_state_pkl)
        print(
            "[prepare_folds] cycle_ae_gmm_ready "
            f"fold={frozen_ae_gmm_reference_fold} "
            f"rows={cycle_ae_gmm_contract.get('reference_rows_sampled')} "
            f"hash={cycle_ae_gmm_contract.get('state_hash')}",
            flush=True,
        )
    cycle_input_fill_values: dict[str, float] = {}
    cycle_input_features_effective: list[str] = []
    if bool(include_ae_gmm_state_features) and active_fixed_ae_gmm_state_pkl is not None:
        cycle_state_for_preprocessing = load_ae_gmm_state_artifact(
            active_fixed_ae_gmm_state_pkl
        )
        cycle_input_fill_values = {
            str(key): float(value)
            for key, value in dict(
                cycle_state_for_preprocessing.get("cycle_input_fill_values", {}) or {}
            ).items()
            if np.isfinite(value)
        }
        cycle_input_features_effective = [
            str(value)
            for value in cycle_state_for_preprocessing.get("feature_columns", []) or []
        ]
    for fold_id, window in enumerate(ordered_windows):
        expected_fold_reuse_fingerprint = (
            _fold_reuse_fingerprint(
                run_fingerprint=run_reuse_fingerprint,
                window=window,
                selected_features=global_selected_features,
                fixed_training_contract=fixed_training_contract,
                ae_gmm_state_path=active_fixed_ae_gmm_state_pkl,
                frozen_ae_gmm_output_sidecar_path=frozen_ae_gmm_output_sidecar_path,
            )
            if global_selected_features is not None
            else None
        )
        if fold_cache_dir is not None:
            cached_fold = _reuse_complete_fold_payload(
                cache_dir=fold_cache_dir,
                window=window,
                selected_features=global_selected_features,
                fixed_training_contract=fixed_training_contract,
                expected_reuse_fingerprint=expected_fold_reuse_fingerprint,
            )
            if cached_fold is not None:
                folds.append(cached_fold)
                print(
                    "[prepare_folds] reused_cache "
                    f"{window['fold']} train={cached_fold['train_rows']} "
                    f"valid={cached_fold['valid_rows']} "
                    f"features={len(cached_fold['selected_features'])}",
                    flush=True,
                )
                continue
        print(
            "[prepare_folds] start "
            f"{window['fold']} train_est={int(window.get('train_rows_estimate', 0))} "
            f"valid_est={int(window.get('valid_rows_estimate', 0))}",
            flush=True,
        )
        valid_mask = ts_utc.ge(window["valid_start"]) & ts_utc.lt(window["valid_end"])
        valid_rows_raw = int(valid_mask.sum())
        if bool(missing_only) and existing_scored_keys:
            valid_frame_window = frame.loc[valid_mask, ["__ts__", "__symbol__", "side"]]
            missing_mask_window = _missing_against_existing_mask(valid_frame_window, existing_scored_keys)
            valid_mask_values = valid_mask.to_numpy(dtype=bool, copy=True)
            valid_positions = np.flatnonzero(valid_mask_values)
            valid_mask_values[valid_positions] = missing_mask_window
            valid_mask = pd.Series(valid_mask_values, index=frame.index)
        train_mask = ts_utc.lt(
            window.get("train_cutoff", window["valid_start"])
        )
        if window.get("train_start") is not None:
            train_mask = train_mask & ts_utc.ge(window["train_start"])
        train_full_uncapped = frame.loc[train_mask]
        valid_full = frame.loc[valid_mask]
        train_metrics_uncapped = metrics.loc[train_mask].reset_index(drop=True)
        if int(payload_max_train_rows) > 0 and len(train_full_uncapped) > int(payload_max_train_rows):
            payload_idx = _time_spread_cap_rows(len(train_full_uncapped), int(payload_max_train_rows))
            train_full = train_full_uncapped.iloc[payload_idx]
            train_metrics = train_metrics_uncapped.iloc[payload_idx].reset_index(drop=True)
        else:
            payload_idx = np.arange(len(train_full_uncapped), dtype=np.int64)
            train_full = train_full_uncapped
            train_metrics = train_metrics_uncapped
        train = train_full.loc[:, fold_frame_columns].reset_index(drop=True)
        valid = valid_full.loc[:, fold_frame_columns].reset_index(drop=True)
        train_provenance = train.loc[
            :, ["candidate_id", "__decision_ts__", "__label_resolution_ts__"]
        ].copy()
        valid_metrics = metrics.loc[valid_mask].reset_index(drop=True)
        print(
            "[prepare_folds] build_matrix "
            f"{window['fold']} feature_count={len(features)}",
            flush=True,
        )
        x_train = train_full.loc[:, features].replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        x_valid = valid_full.loc[:, features].replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        if frozen_ae_gmm_output_sidecar_path is None:
            x_train, x_valid, restored_cycle_inputs = _restore_cycle_input_columns(
                x_train,
                x_valid,
                train_source=train_full,
                valid_source=valid_full,
                required_columns=cycle_input_features_effective,
            )
        else:
            restored_cycle_inputs = []
        if restored_cycle_inputs:
            print(
                "[prepare_folds] restored_cycle_inputs "
                f"{window['fold']} count={len(restored_cycle_inputs)}",
                flush=True,
            )
        availability_diag: dict[str, Any] = {
            "checked_features": 0,
            "surviving_features": int(x_train.shape[1]),
            "collapsed_tail_features": [],
            "collapsed_tail_feature_count": 0,
        }
        if global_selected_features is None:
            availability_survivors, availability_diag = (
                _train_valid_availability_survivors(x_train, x_valid)
            )
            if len(availability_survivors) < 2:
                raise RuntimeError(
                    "Base feature-selection reference fold has fewer than two "
                    "features after the train/validation availability contract."
                )
            if len(availability_survivors) < x_train.shape[1]:
                print(
                    "[prepare_folds] availability_guard "
                    f"{window['fold']} {x_train.shape[1]}->{len(availability_survivors)} "
                    "features; collapsed_tail="
                    f"{availability_diag['collapsed_tail_feature_count']}",
                    flush=True,
                )
                x_train = x_train.loc[:, availability_survivors]
                x_valid = x_valid.loc[:, availability_survivors]
        med_idx = _time_spread_cap_rows(len(x_train), 300_000)
        med = x_train.iloc[med_idx].median(numeric_only=True)
        if cycle_input_fill_values:
            for col, value in cycle_input_fill_values.items():
                if col in med.index:
                    med.loc[col] = np.float32(value)
        train_median_imputation_values = {
            str(column): (
                float(value) if pd.notna(value) and np.isfinite(value) else 0.0
            )
            for column, value in med.items()
        }
        x_train = x_train.fillna(med).fillna(0.0).astype(np.float32, copy=False)
        x_valid = x_valid.fillna(med).fillna(0.0).astype(np.float32, copy=False)
        ae_gmm_anchor_rows = int(cycle_ae_gmm_contract.get("reference_rows_sampled", 0) or 0)
        print(
            "[prepare_folds] ae_gmm_start "
            f"{window['fold']} train_rows={len(x_train)} valid_rows={len(x_valid)}",
            flush=True,
        )
        full_valid_context: dict[str, Any] = {}
        if frozen_ae_gmm_output_sidecar_path is not None:
            generated_features = _fixed_selected_ae_gmm_features(
                global_selected_features or list(x_train.columns)
            )
            if generated_features:
                missing_from_full = [
                    name
                    for name in generated_features
                    if name not in train_full.columns or name not in valid_full.columns
                ]
                if missing_from_full:
                    raise RuntimeError(
                        "Frozen AE/GMM output sidecar is missing selected outputs: "
                        f"{missing_from_full[:20]}"
                    )
                generated_train = (
                    train_full.loc[:, generated_features]
                    .replace([np.inf, -np.inf], np.nan)
                    .astype(np.float32, copy=False)
                )
                generated_valid = (
                    valid_full.loc[:, generated_features]
                    .replace([np.inf, -np.inf], np.nan)
                    .astype(np.float32, copy=False)
                )
                x_train = pd.concat(
                    [
                        x_train.drop(
                            columns=[
                                name for name in generated_features if name in x_train.columns
                            ],
                            errors="ignore",
                        ),
                        generated_train,
                    ],
                    axis=1,
                    copy=False,
                )
                x_valid = pd.concat(
                    [
                        x_valid.drop(
                            columns=[
                                name for name in generated_features if name in x_valid.columns
                            ],
                            errors="ignore",
                        ),
                        generated_valid,
                    ],
                    axis=1,
                    copy=False,
                )
            missing_generated = [
                name
                for name in generated_features
                if name not in x_train.columns or name not in x_valid.columns
            ]
            if missing_generated:
                raise RuntimeError(
                    "Frozen AE/GMM output sidecar is missing selected outputs: "
                    f"{missing_generated[:20]}"
                )
            context_features = [
                name
                for name in AE_GMM_FEATURE_COLUMNS
                if name in valid_full.columns
            ]
            full_valid_generated = valid_full.loc[:, context_features].copy(deep=False)
            ae_diag = {
                "ae_gmm_state_feature_status": "precomputed_frozen_selected_output_sidecar",
                "ae_gmm_state_feature_count": int(len(generated_features)),
                "ae_gmm_context_feature_count": int(len(context_features)),
                "ae_gmm_output_sidecar_path": str(frozen_ae_gmm_output_sidecar_path),
            }
        else:
            x_train, x_valid, generated_features, ae_diag = _append_fold_ae_gmm_state_features(
                x_train=x_train,
                x_valid=x_valid,
                train_frame=train,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                enabled=bool(include_ae_gmm_state_features),
                max_train_rows=int(ae_gmm_state_feature_max_train_rows),
                gmm_max_train_rows=int(ae_gmm_state_feature_gmm_max_train_rows),
                ae_max_iter=int(ae_gmm_state_feature_max_iter),
                random_state=int(seed) + fold_id,
                state_artifact_dir=(fold_cache_dir.parent / "ae_gmm_states") if fold_cache_dir is not None else None,
                state_artifact_name=str(window["fold"]),
                fixed_state_path=active_fixed_ae_gmm_state_pkl,
                output_feature_subset=(
                    list(global_selected_features)
                    if global_selected_features is not None
                    else None
                ),
                valid_context_output=full_valid_context,
                input_feature_cols=(cycle_input_features_effective or ae_gmm_input_features),
            )
            full_valid_generated = full_valid_context.get("frame")
        full_context_features = _ae_gmm_context_columns(
            full_valid_generated.columns
            if isinstance(full_valid_generated, pd.DataFrame)
            else generated_features
        )
        if bool(include_ae_gmm_state_features) and not full_context_features:
            raise RuntimeError(
                "Frozen cycle AE/GMM state emitted no context features for fold "
                f"{window['fold']}: {ae_diag.get('ae_gmm_state_feature_status')}"
            )
        print(
            "[prepare_folds] ae_gmm_done "
            f"{window['fold']} generated={len(generated_features)} status={ae_diag.get('ae_gmm_state_feature_status')}",
            flush=True,
        )
        ae_gmm_context_features = list(full_context_features)
        ae_gmm_context_valid = (
            (
                full_valid_generated
                if isinstance(full_valid_generated, pd.DataFrame)
                else x_valid
            ).reindex(columns=ae_gmm_context_features, fill_value=0.0)
            .astype(np.float32, copy=False)
            .reset_index(drop=True)
            if ae_gmm_context_features
            else pd.DataFrame(index=np.arange(len(x_valid)))
        )
        if global_selected_features is None:
            raw_presence_columns = [
                feature for feature in x_train.columns if feature in train_full.columns
            ]
            presence_frame = train_full.loc[:, raw_presence_columns].replace(
                [np.inf, -np.inf], np.nan
            )
            generated_presence_columns = [
                feature
                for feature in x_train.columns
                if feature not in presence_frame.columns
            ]
            if generated_presence_columns:
                presence_frame = pd.concat(
                    [
                        presence_frame.reset_index(drop=True),
                        x_train.loc[:, generated_presence_columns].reset_index(drop=True),
                    ],
                    axis=1,
                    copy=False,
                )
            coverage_survivors, coverage_diag = _recent_feature_coverage_survivors(
                presence_frame,
                train["__ts__"].to_numpy(),
                require_joint_complete_case=True,
                min_feature_coverage=0.90,
                coverage_scope="all_post_warmup",
                warmup_days=30,
                warmup_reference_start=train_full_uncapped["__ts__"].min(),
            )
            del presence_frame
            if len(coverage_survivors) < 2:
                raise RuntimeError(
                    "Base feature-selection basket has fewer than two features "
                    "after the 90% joint post-warm-up coverage contract."
                )
            x_train = x_train.loc[:, coverage_survivors]
            x_valid = x_valid.reindex(columns=coverage_survivors)
            fs_target_frame = _target_from_frame(train, train_metrics, target_mode=str(feature_selection_target_mode))
            fs_fold_name = f"largest_train_before_{window['valid_start']:%Y-%m-%d}"
            selection_method = str(feature_selection_method).strip().lower()
            if selection_method in {"mda", "mda_cum99_se075"}:
                x_train, x_valid, selected_features_fold, feature_selection_df = _select_features_by_mda(
                    x_train,
                    x_valid,
                    train,
                    train_metrics,
                    fs_target_frame,
                    top_n=int(feature_selection_top_n),
                    fold=fs_fold_name,
                    seed=int(seed) + fold_id,
                    post_cumulative_se_mult=(
                        0.75 if selection_method == "mda_cum99_se075" else None
                    ),
                )
            elif selection_method in {
                "archetype_prescreen_side_mda",
                "archetype_prescreen_side_mda_corrfirst",
            }:
                if fold_cache_dir is not None:
                    staged_report_dir = fold_cache_dir.parent / "staged_archetype_feature_selection"
                else:
                    staged_report_dir = Path(output_dir) / "staged_archetype_feature_selection"
                x_train, x_valid, selected_features_fold, feature_selection_df = (
                    _select_features_by_archetype_prescreen_side_mda(
                        x_train,
                        x_valid,
                        train,
                        train_metrics,
                        fs_target_frame,
                        fold=fs_fold_name,
                        seed=int(seed) + fold_id,
                        report_dir=staged_report_dir,
                        correlation_first=selection_method.endswith("_corrfirst"),
                    )
                )
            else:
                x_train, x_valid, selected_features_fold, feature_selection_df = _select_features_by_univariate(
                    x_train,
                    x_valid,
                    fs_target_frame["target_soft"],
                    top_n=int(feature_selection_top_n),
                    fold=fs_fold_name,
                )
                feature_selection_df["feature_selection_method"] = "univariate_rank_corr"
            for key, value in coverage_diag.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    feature_selection_df[f"coverage_{key}"] = value
            global_selected_features = list(selected_features_fold)
            if {"selected_long", "selected_short"}.issubset(
                feature_selection_df.columns
            ):
                global_selected_features_by_side = {
                    side_name: feature_selection_df.loc[
                        feature_selection_df[f"selected_{side_name}"].fillna(False),
                        "feature",
                    ]
                    .astype(str)
                    .tolist()
                    for side_name in ("long", "short")
                }
            else:
                global_selected_features_by_side = {
                    "long": list(global_selected_features),
                    "short": list(global_selected_features),
                }
            global_feature_selection_df = feature_selection_df
        else:
            missing_fixed_outputs = [
                feature
                for feature in global_selected_features
                if feature not in x_train.columns or feature not in x_valid.columns
            ]
            if missing_fixed_outputs:
                raise RuntimeError(
                    "Frozen model contract is missing generated/raw features after "
                    "the AE/GMM transform: "
                    f"{missing_fixed_outputs[:20]}"
                )
            x_train = x_train.reindex(columns=global_selected_features, fill_value=0.0).astype(np.float32, copy=False)
            x_valid = x_valid.reindex(columns=global_selected_features, fill_value=0.0).astype(np.float32, copy=False)
            selected_features_fold = list(global_selected_features)
            feature_selection_df = pd.DataFrame(
                columns=["fold", "feature", "score", "rank", "selected", "feature_selection_status"]
            )
        compact_fixed_training = bool(
            fixed_training_contract
            and global_selected_features is not None
            and not bool(selection_only)
        )
        train_target_payload: pd.DataFrame | None = None
        train_weight_payload: pd.DataFrame | None = None
        train_side_payload = pd.DataFrame(
            {"side_name": _side_name_array(train).astype(str)}
        )
        if compact_fixed_training:
            target_mode = str(fixed_training_contract.get("target_mode", ""))
            weight_arm = str(fixed_training_contract.get("weight_arm", ""))
            if target_mode not in TARGET_MODES or weight_arm not in WEIGHT_ARMS:
                raise ValueError(
                    "Compact fixed-training payload requires valid target_mode and "
                    f"weight_arm, got target={target_mode!r} weight={weight_arm!r}"
                )
            prepared_target = _target_from_frame(
                train,
                train_metrics,
                target_mode=target_mode,
            )
            prepared_weight = _weight_series(
                frame=train,
                metrics=train_metrics,
                target=prepared_target,
                arm=weight_arm,
            )
            train_target_payload = prepared_target.loc[
                :, ["target_soft", "target_hard"]
            ].reset_index(drop=True)
            train_weight_payload = pd.DataFrame(
                {"sample_weight": prepared_weight.to_numpy(dtype=np.float32)}
            )
        fold_payload = {
                "fold": str(window["fold"]),
                "month": str(window["month"]),
                "valid_start": window["valid_start"],
                "valid_end": window["valid_end"],
                "train_start": window.get("train_start"),
                "train_cutoff": window.get("train_cutoff"),
                "ae_gmm_anchor_start": window.get("ae_gmm_anchor_start"),
                "ae_gmm_anchor_end": window.get("ae_gmm_anchor_end"),
                "ae_gmm_anchor_rows": int(ae_gmm_anchor_rows),
                "max_oos_model_age_days": int(max_oos_model_age_days),
                "train_rows_uncapped": int(len(train_full_uncapped)),
                "train_rows_payload": int(len(train_full)),
                "valid_rows_raw": int(valid_rows_raw),
                "missing_only": bool(missing_only),
                "existing_scored_ledger_path": str(existing_scored_ledger_path) if existing_scored_ledger_path is not None else None,
                "payload_train_sampling": (
                    "beginning_middle_end_time_spread"
                    if int(payload_max_train_rows) > 0 and len(train_full_uncapped) > int(payload_max_train_rows)
                    else "full_train_rows"
                ),
                "reuse_fingerprint": _fold_reuse_fingerprint(
                    run_fingerprint=run_reuse_fingerprint,
                    window=window,
                    selected_features=selected_features_fold,
                    fixed_training_contract=fixed_training_contract,
                    ae_gmm_state_path=active_fixed_ae_gmm_state_pkl,
                    frozen_ae_gmm_output_sidecar_path=frozen_ae_gmm_output_sidecar_path,
                ),
                "train_valid_availability_contract": availability_diag,
                "candidate_id_column": label_resolution_contract["candidate_id_column"],
                "candidate_id_source": label_resolution_contract["candidate_id_source"],
                "label_resolution_column": label_resolution_contract[
                    "label_resolution_column"
                ],
                "label_resolution_derivation": label_resolution_contract[
                    "label_resolution_derivation"
                ],
                "label_resolution_source_column": label_resolution_contract[
                    "label_resolution_source_column"
                ],
                "label_path_timeframe": label_resolution_contract[
                    "label_path_timeframe"
                ],
                "valid": valid,
                "valid_metrics": valid_metrics,
                "train_provenance": train_provenance,
                "train_side": train_side_payload,
                "x_train": x_train,
                "x_valid": x_valid,
                "train_median_imputation_values": {
                    str(feature): float(
                        train_median_imputation_values.get(
                            feature, _finite_median_or_zero(x_train[feature])
                        )
                    )
                    for feature in selected_features_fold
                },
                "ae_gmm_generated_features": int(len(generated_features)),
                "ae_gmm_context_feature_count": int(len(ae_gmm_context_features)),
                "ae_gmm_context_features": list(ae_gmm_context_features),
                "ae_gmm_context_valid": ae_gmm_context_valid,
                "ae_gmm_status": ae_diag.get("ae_gmm_state_feature_status"),
                "selected_features": selected_features_fold,
                "selected_features_by_side": dict(
                    global_selected_features_by_side or {}
                ),
                "feature_selection": feature_selection_df,
                "compact_fixed_training_payload": compact_fixed_training,
                "fixed_training_target_mode": (
                    str(fixed_training_contract.get("target_mode"))
                    if compact_fixed_training
                    else None
                ),
                "fixed_training_weight_arm": (
                    str(fixed_training_contract.get("weight_arm"))
                    if compact_fixed_training
                    else None
                ),
            }
        if compact_fixed_training:
            fold_payload["train_target"] = train_target_payload
            fold_payload["train_weight"] = train_weight_payload
        else:
            fold_payload["train"] = train
            fold_payload["train_metrics"] = train_metrics
        folds.append(_write_fold_payload(fold_payload, fold_cache_dir) if fold_cache_dir is not None else fold_payload)
        print(
            "[prepare_folds] cached "
            f"{window['fold']} train={int(len(train_full))}/{int(len(train_full_uncapped))} valid={int(len(valid_full))} "
            f"features={int(x_train.shape[1])} ae_gmm={int(len(generated_features))}",
            flush=True,
        )
        del train_full_uncapped, train_full, valid_full, train, valid, train_provenance, train_metrics, train_metrics_uncapped, valid_metrics, x_train, x_valid
        del ae_gmm_context_valid, full_valid_context, full_valid_generated, fold_payload
        gc.collect()
        if bool(selection_only):
            break
    selected_by_fold = {str(fold["fold"]): list(fold["selected_features"]) for fold in folds}
    selected_sets = [set(features) for features in selected_by_fold.values()]
    selected_union = sorted(set().union(*selected_sets)) if selected_sets else []
    selected_intersection = sorted(set.intersection(*selected_sets)) if selected_sets else []
    manifest = {
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "feature_count": int(len(features)),
        "feature_store": feature_report,
        "fold_payload_storage": "parquet_cache" if fold_cache_dir is not None else "memory",
        "fold_cache_dir": str(fold_cache_dir) if fold_cache_dir is not None else None,
        "fold_cache_feature_dtype_on_disk": "float16_clipped_to_finite_range",
        "fold_frame_column_count": int(len(fold_frame_columns)),
        "fold_months": sorted({str(fold["month"]) for fold in folds}),
        "fold_windows": [
            {
                "fold": str(fold["fold"]),
                "month": str(fold["month"]),
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "train_start": fold.get("train_start"),
                "train_signal_cutoff_exclusive": fold.get("train_cutoff"),
                "candidate_id_column": fold.get("candidate_id_column"),
                "label_resolution_column": fold.get("label_resolution_column"),
                "label_resolution_derivation": fold.get("label_resolution_derivation"),
                "label_resolution_source_column": fold.get(
                    "label_resolution_source_column"
                ),
                "ae_gmm_anchor_start": fold.get("ae_gmm_anchor_start"),
                "ae_gmm_anchor_end": fold.get("ae_gmm_anchor_end"),
                "ae_gmm_anchor_rows": int(fold.get("ae_gmm_anchor_rows", 0)),
                "max_oos_model_age_days": int(fold["max_oos_model_age_days"]),
                "train_rows_uncapped": int(fold.get("train_rows_uncapped", fold.get("train_rows", 0))),
                "train_rows_payload": int(fold.get("train_rows_payload", fold.get("train_rows", 0))),
                "payload_train_sampling": str(fold.get("payload_train_sampling", "full_train_rows")),
            }
            for fold in folds
        ],
        "fold_count": int(len(folds)),
        "max_oos_model_age_days": int(max_oos_model_age_days),
        "train_window_days": int(train_window_days),
        "ae_gmm_anchor_days": int(ae_gmm_anchor_days),
        "ae_gmm_input_features_path": None,
        "ae_gmm_input_feature_count": int(len(cycle_input_features_effective)),
        "ae_gmm_input_features": list(cycle_input_features_effective),
        "ae_gmm_cycle_input_preflight": cycle_input_preflight,
        "payload_max_train_rows": int(payload_max_train_rows),
        "compact_fixed_training_payload": bool(fixed_training_contract),
        "validation_windowing": (
            "continuous_rolling_max_age_windows"
            if int(max_oos_model_age_days) > 0 and contiguous_months
            else "calendar_month_windows"
        ),
        "oos_model_age_contract": (
            "validation windows are capped by --max-oos-model-age-days"
            if int(max_oos_model_age_days) > 0
            else "month_forward_legacy"
        ),
        "ae_gmm_generated_features_by_fold": [fold["ae_gmm_generated_features"] for fold in folds],
        "feature_selection_scope": "single_global_largest_train_window",
        "feature_selection_calibration_fold": fs_window_fold,
        "feature_selection_global_calibration_note": (
            "features are selected once on the largest train fold and reused for all OOS scoring folds"
        ),
        "feature_selection_method": str(feature_selection_method),
        "fixed_selected_features_path": str(fixed_selected_features_path) if fixed_selected_features_path is not None else None,
        "fixed_selected_features_count": int(len(fixed_selected_features or [])),
        "fixed_ae_gmm_state_pkl": str(fixed_ae_gmm_state_pkl) if fixed_ae_gmm_state_pkl is not None else None,
        "ae_gmm_state_freeze_after_reference": bool(freeze_ae_gmm_state_after_reference),
        "ae_gmm_state_reference_fold": frozen_ae_gmm_reference_fold,
        "ae_gmm_state_reference_state_path": frozen_ae_gmm_reference_state_path,
        "ae_gmm_cycle_contract": cycle_ae_gmm_contract,
        "frozen_ae_gmm_output_sidecar_path": (
            str(frozen_ae_gmm_output_sidecar_path)
            if frozen_ae_gmm_output_sidecar_path is not None
            else None
        ),
        "ae_gmm_state_ae_max_train_rows": int(ae_gmm_state_feature_max_train_rows),
        "ae_gmm_state_gmm_max_train_rows": int(ae_gmm_state_feature_gmm_max_train_rows),
        "ae_gmm_state_sample_policy": "cycle_reference_beginning_middle_end_time_spread",
        "missing_only": bool(missing_only),
        "existing_scored_ledger_path": str(existing_scored_ledger_path) if existing_scored_ledger_path is not None else None,
        "existing_scored_key_count": int(len(existing_scored_keys)),
        "feature_selection_top_n": int(feature_selection_top_n),
        "feature_selection_target_mode": str(feature_selection_target_mode),
        "global_feature_selection_fold": (
            str(global_feature_selection_df["fold"].iloc[0])
            if global_feature_selection_df is not None and not global_feature_selection_df.empty
            else None
        ),
        "selected_features_by_fold": selected_by_fold,
        "selected_feature_union": selected_union,
        "selected_feature_intersection": selected_intersection,
        "selected_feature_union_count": int(len(selected_union)),
        "selected_feature_intersection_count": int(len(selected_intersection)),
        "selected_features_by_side": dict(global_selected_features_by_side or {}),
    }
    return folds, manifest


def _suggest_params(
    trial: Any,
    rng: np.random.Generator,
    *,
    target_modes: Sequence[str] = TARGET_MODES,
) -> dict[str, Any]:
    available_target_modes = tuple(dict.fromkeys(map(str, target_modes)))
    if not available_target_modes:
        raise ValueError("HPO requires at least one available target mode")
    if trial is None:
        return {
            "n_estimators": int(rng.integers(120, 321)),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.015), np.log(0.08)))),
            "num_leaves": int(rng.choice([15, 23, 31, 47, 63])),
            "max_depth": int(rng.choice([-1, 4, 5, 6, 8])),
            "min_child_samples": int(rng.integers(25, 101)),
            "subsample": float(rng.uniform(0.65, 0.95)),
            "colsample_bytree": float(rng.uniform(0.55, 0.95)),
            "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(3.0)))),
            "reg_lambda": float(np.exp(rng.uniform(np.log(0.3), np.log(12.0)))),
            # The atom-heavy soft target requires a conditional-mean estimator.
            # L1 can collapse to the target median and destroy tail ranking.
            "loss_function": "regression",
            "min_split_gain": float(
                rng.choice([0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
            ),
            "target_mode": str(rng.choice(available_target_modes)),
            "weight_arm": str(rng.choice(["W0_base", "W7_timestamp_balanced", "W8_combined_conservative", "W12_tail_timestamp_balanced"])),
        }
    return {
        "n_estimators": int(trial.suggest_int("n_estimators", 120, 360)),
        "learning_rate": float(trial.suggest_float("learning_rate", 0.015, 0.08, log=True)),
        "num_leaves": int(trial.suggest_categorical("num_leaves", [15, 23, 31, 47, 63])),
        "max_depth": int(trial.suggest_categorical("max_depth", [-1, 4, 5, 6, 8])),
        "min_child_samples": int(trial.suggest_int("min_child_samples", 25, 110)),
        "subsample": float(trial.suggest_float("subsample", 0.65, 0.95)),
        "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.55, 0.95)),
        "reg_alpha": float(trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True)),
        "reg_lambda": float(trial.suggest_float("reg_lambda", 0.3, 12.0, log=True)),
        "loss_function": "regression",
        "min_split_gain": float(
            trial.suggest_categorical(
                "min_split_gain",
                [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            )
        ),
        "target_mode": str(
            trial.suggest_categorical("target_mode", list(available_target_modes))
        ),
        "weight_arm": str(
            trial.suggest_categorical(
                "weight_arm",
                ["W0_base", "W7_timestamp_balanced", "W8_combined_conservative", "W12_tail_timestamp_balanced"],
            )
        ),
    }


def _objective_from_rows(rows: list[dict[str, Any]]) -> float:
    df = pd.DataFrame(rows)
    if df.empty:
        return float("-inf")
    def m(frac: float, col: str) -> float:
        return _safe_mean(df.loc[df["top_frac"].eq(frac), col])

    top10 = m(0.10, "gross_ev_weighted_clean_precision")
    top20 = m(0.20, "gross_ev_weighted_clean_precision")
    top30 = m(0.30, "gross_ev_weighted_clean_precision")
    clean30 = m(0.30, "clean_precision")
    net30 = m(0.30, "mean_first_touch_net")
    q10_net30 = m(0.30, "q10_first_touch_net")
    timeout30 = m(0.30, "first_touch_timeout_rate")
    bad30 = m(0.30, "first_touch_bad_mae_to_sl_rate")
    objective = (
        1.00 * top30
        + 0.55 * top20
        + 0.30 * top10
        + 0.25 * clean30
        + 10.00 * net30
        + 3.00 * min(q10_net30, 0.0)
        - 0.20 * timeout30
        - 0.12 * bad30
    )
    return float(objective) if math.isfinite(float(objective)) else float("-inf")


def _run_trial(
    *,
    folds: list[dict[str, Any]],
    params: dict[str, Any],
    trial_number: int,
    max_train_rows: int,
    seed: int,
    model_side_scope: str = "shared",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for fold_id, fold in enumerate(folds):
        payload = _load_fold_payload(fold)
        if bool(payload.get("compact_fixed_training_payload")):
            if str(params["target_mode"]) != str(
                payload.get("fixed_training_target_mode")
            ) or str(params["weight_arm"]) != str(
                payload.get("fixed_training_weight_arm")
            ):
                raise ValueError("Trial parameters do not match compact fold payload")
            train_target = payload["train_target"]
            weights = payload["train_weight"]["sample_weight"]
        else:
            train_target = _target_from_frame(
                payload["train"],
                payload["train_metrics"],
                target_mode=str(params["target_mode"]),
            )
            if str(params["weight_arm"]) not in WEIGHT_ARMS:
                raise ValueError(f"Unknown weight arm: {params['weight_arm']}")
            weights = _weight_series(
                frame=payload["train"],
                metrics=payload["train_metrics"],
                target=train_target,
                arm=str(params["weight_arm"]),
            )
        valid_target = _target_from_frame(
            payload["valid"],
            payload["valid_metrics"],
            target_mode=str(params["target_mode"]),
        )
        uncapped_fit = int(max_train_rows) <= 0 or len(payload["x_train"]) <= int(
            max_train_rows
        )
        idx = (
            None
            if uncapped_fit
            else _time_spread_cap_rows(len(payload["x_train"]), int(max_train_rows))
        )
        x_train_fit = (
            payload["x_train"]
            if idx is None
            else payload["x_train"].iloc[idx].reset_index(drop=True)
        )
        y_train_fit = (
            train_target["target_soft"]
            if idx is None
            else train_target["target_soft"].iloc[idx].reset_index(drop=True)
        )
        w_train_fit = (
            weights if idx is None else weights.iloc[idx].reset_index(drop=True)
        )
        train_side = payload.get("train_side")
        if not isinstance(train_side, pd.DataFrame):
            train_side = pd.DataFrame(
                {"side_name": _side_name_array(payload["train"])}
            )
        train_sides_fit = train_side["side_name"].to_numpy()[
            slice(None) if idx is None else idx
        ]
        models, feature_contracts = _fit_lgbm_models(
            x_train=x_train_fit,
            y_train=y_train_fit,
            w_train=w_train_fit,
            train_sides=train_sides_fit,
            params=params,
            seed=int(seed) + 1000 * int(trial_number) + fold_id,
            model_side_scope=model_side_scope,
            features_by_side=payload.get("selected_features_by_side"),
        )
        pred = _predict_lgbm_models(
            models=models,
            x_valid=payload["x_valid"],
            valid_sides=_side_name_array(payload["valid"]),
            model_side_scope=model_side_scope,
            feature_contracts=feature_contracts,
        )
        trial_name = f"trial_{int(trial_number):03d}"
        for frac in TOP_FRACS:
            metric = _selection_metrics(
                valid=payload["valid"],
                metrics=payload["valid_metrics"],
                target=valid_target,
                pred=pred,
                month=str(fold["fold"]),
                top_frac=float(frac),
                trial_name=trial_name,
            )
            metric.update(
                {
                    "trial_number": int(trial_number),
                    "calendar_month": str(fold["month"]),
                    "valid_start": fold["valid_start"],
                    "valid_end": fold["valid_end"],
                    "max_oos_model_age_days": int(fold["max_oos_model_age_days"]),
                    **params,
                }
            )
            rows.append(metric)
        diagnostics.append(
            {
                "trial_number": int(trial_number),
                "month": str(fold["fold"]),
                "calendar_month": str(fold["month"]),
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "max_oos_model_age_days": int(fold["max_oos_model_age_days"]),
                "train_rows": int(
                    len(payload["x_train"]) if idx is None else len(idx)
                ),
                "train_rows_uncapped": int(len(payload["x_train"])),
                "valid_rows": int(len(payload["x_valid"])),
                "target_train_mean": _safe_mean(train_target["target_soft"]),
                "target_valid_mean": _safe_mean(valid_target["target_soft"]),
                "weight_mean": _safe_mean(weights),
                "weight_effective_frac": _effective_sample_size(weights) / max(float(len(weights)), 1.0),
                "ae_gmm_generated_features": int(fold["ae_gmm_generated_features"]),
                "ae_gmm_status": fold.get("ae_gmm_status"),
                "model_side_scope": str(model_side_scope),
                **params,
            }
        )
        del payload, train_target, valid_target, weights, pred
        _release_process_memory()
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "trial_number": int(trial_number),
        "trial_name": f"trial_{int(trial_number):03d}",
        **params,
        "model_side_scope": str(model_side_scope),
        "objective": _objective_from_rows(rows),
        "folds": int(len(folds)),
    }
    for frac in TOP_FRACS:
        tag = f"top{int(round(frac * 100))}"
        subset = df[df["top_frac"].eq(frac)]
        for col in (
            "gross_ev_weighted_clean_precision",
            "clean_precision",
            "mean_first_touch_net",
            "mean_first_touch_gross",
            "q10_first_touch_net",
            "hit_first_touch_net",
            "first_touch_stop_rate",
            "first_touch_timeout_rate",
            "first_touch_bad_mae_to_sl_rate",
            "bad_mae_1r_rate",
            "selected_rows",
            "selected_symbols",
        ):
            summary[f"mean_{tag}_{col}"] = _safe_mean(subset[col]) if col in subset else float("nan")
    return summary, rows, diagnostics


def _best_params_from_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "loss_function",
        "min_split_gain",
        "target_mode",
        "weight_arm",
    )
    out = {key: row[key] for key in keys if key in row}
    for key in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        if key in out:
            out[key] = int(float(out[key]))
    for key in (
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
    ):
        if key in out:
            out[key] = float(out[key])
    if "target_mode" in out:
        out["target_mode"] = str(out["target_mode"])
    if "loss_function" in out:
        out["loss_function"] = str(out["loss_function"])
    if "weight_arm" in out:
        out["weight_arm"] = str(out["weight_arm"])
    return out


def _score_best_oos_ledger(
    *,
    folds: list[dict[str, Any]],
    params: dict[str, Any],
    trial_number: int,
    max_train_rows: int,
    seed: int,
    save_fold_models_dir: Path | None = None,
    output_path: Path | None = None,
    model_side_scope: str = "shared",
) -> pd.DataFrame:
    scored_paths: list[tuple[pd.Timestamp, Path]] = []
    saved_models: list[dict[str, Any]] = []
    model_input_parity_sidecars: list[dict[str, Any]] = []
    model_input_parity_root = (
        output_path.parent / "model_input_parity" if output_path is not None else None
    )
    scored_cache_dir = (
        save_fold_models_dir.parent / "_scored_fold_cache"
        if save_fold_models_dir is not None
        else (
            output_path.parent / "_scored_fold_cache"
            if output_path is not None
            else None
        )
    )
    if scored_cache_dir is not None:
        scored_cache_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, fold in enumerate(folds):
        fold_name = _safe_fold_name(str(fold["fold"]))
        scored_path = (
            scored_cache_dir / f"{fold_name}.parquet"
            if scored_cache_dir is not None
            else None
        )
        model_path = (
            save_fold_models_dir / fold_name / "base_model.joblib"
            if save_fold_models_dir is not None
            else None
        )
        model_manifest_path = (
            save_fold_models_dir / fold_name / "manifest.json"
            if save_fold_models_dir is not None
            else None
        )
        fold_reuse_fingerprint = str(fold.get("reuse_fingerprint") or "")
        model_reuse_fingerprint = _reuse_fingerprint(
            {
                "schema": "base_fold_model_reuse_v1",
                "fold_payload_fingerprint": fold_reuse_fingerprint,
                "params": dict(params),
                "trial_number": int(trial_number),
                "seed": int(seed) + 1000 * int(trial_number) + fold_id,
                "max_train_rows": int(max_train_rows),
                "model_side_scope": str(model_side_scope),
            }
        )
        scored_reuse_fingerprint = _reuse_fingerprint(
            {
                "schema": "base_scored_fold_reuse_v2",
                "model_fingerprint": model_reuse_fingerprint,
                "fold_payload_fingerprint": fold_reuse_fingerprint,
                "top_fracs": list(TOP_FRACS),
                "base_oof_provenance_schema": BASE_OOF_PROVENANCE_SCHEMA,
            }
        )
        scored_manifest_path = (
            scored_path.with_suffix(".manifest.json") if scored_path is not None else None
        )
        model_input_parity_manifest_path = (
            model_input_parity_root / fold_name / "manifest.json"
            if model_input_parity_root is not None
            else None
        )
        scored_reusable = False
        if (
            scored_path is not None
            and scored_manifest_path is not None
            and scored_path.is_file()
            and scored_path.stat().st_size > 0
            and scored_manifest_path.is_file()
        ):
            try:
                scored_manifest = json.loads(scored_manifest_path.read_text(encoding="utf-8"))
                scored_reusable = str(scored_manifest.get("reuse_fingerprint") or "") == scored_reuse_fingerprint
            except Exception:
                scored_reusable = False
        if scored_reusable and model_input_parity_manifest_path is not None:
            try:
                persisted_parity = json.loads(
                    model_input_parity_manifest_path.read_text(encoding="utf-8")
                )
                row_hashes_path = Path(str(persisted_parity.get("row_hashes_path") or ""))
                anchors_path = Path(str(persisted_parity.get("anchors_path") or ""))
                scored_reusable = (
                    str(persisted_parity.get("schema") or "")
                    == MODEL_INPUT_PARITY_SCHEMA
                    and row_hashes_path.is_file()
                    and anchors_path.is_file()
                )
            except Exception:
                scored_reusable = False
        if scored_reusable and model_manifest_path is not None:
            try:
                saved_manifest = json.loads(
                    model_manifest_path.read_text(encoding="utf-8")
                )
                scored_reusable = _has_persisted_imputation_artifact(saved_manifest)
            except Exception:
                scored_reusable = False
        if scored_reusable:
            scored_paths.append((pd.Timestamp(fold["valid_start"]), scored_path))
            model_input_parity_sidecars.append(
                {**persisted_parity, "manifest_path": str(model_input_parity_manifest_path)}
            )
            if model_manifest_path is not None and model_manifest_path.is_file():
                saved_models.append(json.loads(model_manifest_path.read_text()))
            print(f"[score_oos] reused_scored {fold['fold']}", flush=True)
            continue

        reuse_model = False
        if (
            model_path is not None
            and model_manifest_path is not None
            and model_path.is_file()
            and model_manifest_path.is_file()
        ):
            try:
                model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
                reuse_model = (
                    str(model_manifest.get("reuse_fingerprint") or "")
                    == model_reuse_fingerprint
                    and _has_persisted_imputation_artifact(model_manifest)
                )
            except Exception:
                reuse_model = False
        payload_keys = [
            "valid",
            "valid_metrics",
            "x_valid",
            "ae_gmm_context_valid",
            "train_provenance",
        ]
        if not reuse_model:
            payload_keys.extend(["x_train", "train_target", "train_weight", "train_side", "train", "train_metrics"])
        payload = _load_fold_payload_keys(fold, payload_keys)
        if reuse_model:
            model = joblib.load(model_path)
            saved_columns = json.loads(
                Path(model_manifest["columns_path"]).read_text(encoding="utf-8")
            )
            feature_contracts = dict(
                saved_columns.get("feature_names_by_side")
                or {"shared": saved_columns.get("feature_names", [])}
            )
            train_rows_available = int(fold.get("train_rows", 0))
            train_rows_fit = train_rows_available
            print(f"[score_oos] reused_model {fold['fold']}", flush=True)
        else:
            if bool(payload.get("compact_fixed_training_payload")):
                if str(params["target_mode"]) != str(
                    payload.get("fixed_training_target_mode")
                ) or str(params["weight_arm"]) != str(
                    payload.get("fixed_training_weight_arm")
                ):
                    raise ValueError(
                        "Scoring parameters do not match compact fold payload"
                    )
                train_target = payload["train_target"]
                weights = payload["train_weight"]["sample_weight"]
            else:
                train_target = _target_from_frame(
                    payload["train"],
                    payload["train_metrics"],
                    target_mode=str(params["target_mode"]),
                )
                weights = _weight_series(
                    frame=payload["train"],
                    metrics=payload["train_metrics"],
                    target=train_target,
                    arm=str(params["weight_arm"]),
                )
            uncapped_fit = int(max_train_rows) <= 0 or len(payload["x_train"]) <= int(
                max_train_rows
            )
            if uncapped_fit:
                x_train_fit = payload["x_train"]
                y_train_fit = train_target["target_soft"]
                w_train_fit = weights
            else:
                idx = _time_spread_cap_rows(
                    len(payload["x_train"]), int(max_train_rows)
                )
                x_train_fit = payload["x_train"].iloc[idx].reset_index(drop=True)
                y_train_fit = train_target["target_soft"].iloc[idx].reset_index(
                    drop=True
                )
                w_train_fit = weights.iloc[idx].reset_index(drop=True)
            train_side = payload.get("train_side")
            if not isinstance(train_side, pd.DataFrame):
                train_side = pd.DataFrame(
                    {"side_name": _side_name_array(payload["train"])}
                )
            train_sides_fit = train_side["side_name"].to_numpy()[
                slice(None) if uncapped_fit else idx
            ]
            model, feature_contracts = _fit_lgbm_models(
                x_train=x_train_fit,
                y_train=y_train_fit,
                w_train=w_train_fit,
                train_sides=train_sides_fit,
                params=params,
                seed=int(seed) + 1000 * int(trial_number) + fold_id,
                model_side_scope=model_side_scope,
                features_by_side=payload.get("selected_features_by_side"),
            )
            train_rows_available = int(len(payload["x_train"]))
            train_rows_fit = int(len(x_train_fit))
        train_provenance = payload.get("train_provenance")
        if not isinstance(train_provenance, pd.DataFrame):
            train_provenance = pd.DataFrame(
                index=np.arange(train_rows_available),
                columns=["candidate_id", "__decision_ts__", "__label_resolution_ts__"],
            )
        provenance_indices = None
        if int(max_train_rows) > 0 and len(train_provenance) > int(max_train_rows):
            provenance_indices = _time_spread_cap_rows(
                len(train_provenance), int(max_train_rows)
            )
        base_oof_provenance = _base_oof_fit_provenance(
            fold=fold,
            train_provenance=train_provenance,
            fit_indices=provenance_indices,
        )
        pred = _predict_lgbm_models(
            models=model,
            x_valid=payload["x_valid"],
            valid_sides=_side_name_array(payload["valid"]),
            model_side_scope=model_side_scope,
            feature_contracts=feature_contracts,
        )
        if model_input_parity_root is None:
            raise RuntimeError("OOS model input parity requires an output path")
        model_input_parity_sidecars.append(
            _persist_oos_model_input_parity(
                parity_root=model_input_parity_root,
                fold=str(fold["fold"]),
                valid=payload["valid"],
                x_valid=payload["x_valid"],
                valid_sides=_side_name_array(payload["valid"]),
                feature_contracts=feature_contracts,
                model_side_scope=model_side_scope,
            )
        )
        if save_fold_models_dir is not None and not reuse_model:
            saved_models.append(
                _save_base_fold_model(
                    model_dir=save_fold_models_dir,
                    fold=fold,
                    model=model,
                    feature_names=feature_contracts,
                    x_train=x_train_fit,
                    imputation_fill_values=fold.get(
                        "train_median_imputation_values"
                    ),
                    params=params,
                    trial_number=int(trial_number),
                    seed=int(seed) + 1000 * int(trial_number) + fold_id,
                    train_rows_available=train_rows_available,
                    train_rows_fit=train_rows_fit,
                    valid_rows=int(len(payload["x_valid"])),
                    reuse_fingerprint=model_reuse_fingerprint,
                    base_oof_provenance=base_oof_provenance,
                )
            )
        elif model_manifest_path is not None and model_manifest_path.is_file():
            saved_models.append(
                _persist_base_oof_provenance(
                    model_manifest_path, base_oof_provenance
                )
            )
        scored = payload["valid"].copy()
        scored, _ = _base_oof_provenance_columns(scored)
        scored["score"] = pred.to_numpy(dtype=np.float32, copy=False)
        scored["oos_fold"] = str(fold["fold"])
        ts_scored = pd.to_datetime(scored["__ts__"], errors="coerce", utc=True)
        scored["fold_window"] = str(fold["month"])
        scored["calendar_month"] = ts_scored.dt.strftime("%Y-%m")
        scored["month"] = scored["calendar_month"]
        scored["valid_start"] = fold["valid_start"]
        scored["valid_end"] = fold["valid_end"]
        for key in (
            "fold_validation_start",
            "fold_validation_end",
            "latest_train_decision_cutoff",
            "latest_train_decision_timestamp",
            "latest_train_resolved_label_timestamp",
            "label_resolution_column",
            "label_resolution_source_column",
        ):
            scored[key] = base_oof_provenance[key]
        scored["max_oos_model_age_days"] = int(fold["max_oos_model_age_days"])
        scored["base_model_trial_number"] = int(trial_number)
        scored["base_model_target_mode"] = str(params["target_mode"])
        scored["base_model_weight_arm"] = str(params["weight_arm"])
        scored["base_model_side_scope"] = str(model_side_scope)
        valid_target = _target_from_frame(
            payload["valid"],
            payload["valid_metrics"],
            target_mode=str(params["target_mode"]),
        )
        if str(params["target_mode"]) == PROMOTED_SIDE_TARGET_MODE:
            promoted = build_promoted_side_target(scored)
            scored["__first_touch_target_soft__"] = promoted["target_soft"].to_numpy(
                dtype=np.float32, copy=False
            )
        else:
            scored["__first_touch_target_soft__"] = valid_target[
                "target_soft"
            ].to_numpy(dtype=np.float32, copy=False)
        provenance = _base_target_weight_provenance(
            target_mode=str(params["target_mode"]),
            weight_arm=str(params["weight_arm"]),
        )
        scored["base_target_contract_json"] = json.dumps(
            provenance["base_target_contract"], sort_keys=True, separators=(",", ":")
        )
        scored["base_sample_weight_spec_json"] = json.dumps(
            provenance["base_sample_weight_spec"], sort_keys=True, separators=(",", ":")
        )
        scored["base_target_contract_hash"] = provenance[
            "base_target_contract_hash"
        ]
        scored["base_sample_weight_spec_hash"] = provenance[
            "base_sample_weight_spec_hash"
        ]
        ae_gmm_context = payload.get("ae_gmm_context_valid")
        if isinstance(ae_gmm_context, pd.DataFrame) and len(ae_gmm_context) == len(scored):
            for col in _ae_gmm_context_columns(ae_gmm_context.columns):
                if col not in scored.columns:
                    scored[col] = ae_gmm_context[col].to_numpy(copy=False)
        side = pd.to_numeric(scored.get("__side__", scored.get("side", np.nan)), errors="coerce")
        if "side_name" not in scored.columns:
            scored["side_name"] = np.where(side.to_numpy(dtype=np.float64, copy=False) < 0.0, "short", "long")
        scored["candidate_handoff_rank_scope"] = BASE_TO_META_HANDOFF_RANK_SCOPE
        ranks = _timestamp_side_ranks(scored, pred, side)
        scored["base_rank_within_timestamp_side"] = ranks["rank"].to_numpy(
            dtype=np.int32, copy=False
        )
        scored["base_rank_pct_timestamp_side"] = ranks["rank_pct"].to_numpy(
            dtype=np.float32, copy=False
        )
        for frac in TOP_FRACS:
            col = f"selected_top{int(round(frac * 100))}"
            mask = ranks["rank"].to_numpy(dtype=np.int64) <= np.ceil(
                ranks["group_rows"].to_numpy(dtype=np.float64) * float(frac)
            ).astype(np.int64)
            scored[col] = mask
            if np.isclose(float(frac), 0.30):
                cutoff_frame = pd.DataFrame(
                    {
                        "__ts__": pd.to_datetime(scored["__ts__"], utc=True),
                        "side_name": scored["side_name"].astype(str),
                        "score": scored["score"].where(mask),
                    }
                )
                scored["base_cutoff_score_timestamp_side"] = cutoff_frame.groupby(
                    ["__ts__", "side_name"], sort=False, dropna=False
                )["score"].transform("min").to_numpy(dtype=np.float32)
        scored = scored.sort_values(
            ["__ts__", "__symbol__", "side_name"], kind="mergesort"
        ).reset_index(drop=True)
        scored.attrs.clear()
        if scored_path is not None:
            scored.to_parquet(
                scored_path,
                index=False,
                compression="zstd",
                compression_level=5,
            )
            scored_manifest_path = scored_path.with_suffix(".manifest.json")
            scored_manifest_path.write_text(
                json.dumps(
                    _json_safe({
                        "schema": "base_scored_fold_reuse_v2",
                        "reuse_fingerprint": scored_reuse_fingerprint,
                        "model_reuse_fingerprint": model_reuse_fingerprint,
                        "fold_payload_fingerprint": fold_reuse_fingerprint,
                        "fold": str(fold["fold"]),
                        "base_oof_provenance": base_oof_provenance,
                        "model_input_parity_manifest": str(
                            model_input_parity_sidecars[-1]["manifest_path"]
                        ),
                    }),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            scored_paths.append((pd.Timestamp(fold["valid_start"]), scored_path))
        else:
            raise RuntimeError("Streaming OOS scoring requires a fold cache path")
        print(
            f"[score_oos] cached {fold['fold']} rows={len(scored)} "
            f"model_reused={reuse_model}",
            flush=True,
        )
        del payload, pred, scored, model
        if not reuse_model:
            del train_target, weights, x_train_fit, y_train_fit, w_train_fit
        _release_process_memory()
    if not scored_paths:
        return pd.DataFrame()
    if output_path is None or pq is None:
        raise RuntimeError("Streaming OOS scoring requires pyarrow and output_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for _start, part_path in sorted(scored_paths, key=lambda item: item[0]):
            parquet_file = pq.ParquetFile(part_path)
            for batch in parquet_file.iter_batches(batch_size=50_000):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression="zstd",
                        compression_level=5,
                    )
                writer.write_table(table, row_group_size=50_000)
                del table, batch
                _release_process_memory()
            del parquet_file
            _release_process_memory()
    finally:
        if writer is not None:
            writer.close()
    out = pd.DataFrame()
    if saved_models:
        out.attrs["saved_fold_models"] = saved_models
    out.attrs["model_input_parity_sidecars"] = model_input_parity_sidecars
    out.attrs["streamed_to_output_path"] = str(output_path)
    return out



def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_persisted_imputation_artifact(manifest: Mapping[str, Any]) -> bool:
    path = Path(str(manifest.get("imputation_path") or ""))
    expected_sha256 = str(manifest.get("imputation_sha256") or "")
    if not path.is_file() or not expected_sha256:
        return False
    try:
        return _sha256_file(path) == expected_sha256
    except OSError:
        return False


def _package_final_ae_gmm_contract(
    *,
    final_dir: Path,
    state_path: Path | None,
    input_features: Sequence[str] | None,
) -> dict[str, Any]:
    """Copy the exact cycle state and transform inputs beside the final model."""

    if state_path is None or not Path(state_path).is_file():
        return {"status": "missing", "source_state": str(state_path or "")}
    source = Path(state_path)
    target_dir = final_dir / "ae_gmm_state"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_state = target_dir / "ae_gmm_state.pkl"
    shutil.copy2(source, target_state)
    ordered_inputs = list(dict.fromkeys(map(str, input_features or [])))
    input_contract = {
        "schema": "single_cycle_frozen_ae_gmm_input_contract_v1",
        "ordered_input_features": ordered_inputs,
        "input_feature_count": len(ordered_inputs),
        "input_feature_order_hash": _feature_contract_hash(ordered_inputs),
    }
    input_path = target_dir / "input_features.json"
    input_path.write_text(
        json.dumps(input_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_manifest_candidates = (
        source.with_name(source.stem.replace("_state", "_manifest") + ".json"),
        source.with_suffix(".json"),
    )
    copied_source_manifest = None
    for candidate in source_manifest_candidates:
        if candidate.is_file():
            copied_source_manifest = target_dir / "source_state_manifest.json"
            shutil.copy2(candidate, copied_source_manifest)
            break
    contract = {
        "schema": "single_cycle_frozen_ae_gmm_bundle_v3",
        "status": "packaged",
        "source_state": str(source),
        "state_path": str(target_state),
        "state_sha256": _sha256_file(target_state),
        "input_contract_path": str(input_path),
        "input_feature_count": len(ordered_inputs),
        "input_feature_order_hash": input_contract["input_feature_order_hash"],
        "source_manifest_path": (
            str(copied_source_manifest) if copied_source_manifest is not None else None
        ),
        "reuse_contract": (
            "exact pickle and ordered transforms must be reused for OOS, final "
            "refit, replay, and inference"
        ),
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(_json_safe(contract), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def _fit_final_all_rows_base_model(
    *,
    folds: list[dict[str, Any]],
    params: dict[str, Any],
    trial_number: int,
    max_train_rows: int,
    train_window_days: int = 0,
    model_side_scope: str = "shared",
    seed: int,
    model_dir: Path,
    fixed_ae_gmm_state_pkl: Path | None = None,
    ae_gmm_input_features: Sequence[str] | None = None,
    reuse_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Fit the deployable base model after OOS scoring, without contaminating it."""

    if not folds:
        return {"status": "skipped_no_folds"}
    final_dir = model_dir / "final_all_rows"
    model_path = final_dir / "base_model.joblib"
    manifest_path = final_dir / "manifest.json"
    if model_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            str(manifest.get("reuse_fingerprint") or "")
            == str(reuse_fingerprint or "")
            and _has_persisted_imputation_artifact(manifest)
        ):
            manifest["status"] = "reused"
            return manifest

    fold = max(folds, key=lambda item: pd.Timestamp(item["valid_end"]))
    payload_paths = dict(fold.get("payload_paths", {}) or {})
    full_refit_payloads = {
        "x_train",
        "train_target",
        "train_weight",
        "train_side",
        "train",
        "train_metrics",
        "x_valid",
        "valid",
        "valid_metrics",
    }
    if bool(fold.get("compact_fixed_training_payload")) and payload_paths and not full_refit_payloads.issubset(
        payload_paths
    ):
        # Compact fold caches intentionally omit the multi-gigabyte wide row
        # frames.  The dedicated packager reconstructs exact row identities,
        # restores the purge gap, and enforces the trailing-window boundary.
        # Do not crash after valid OOS scoring or silently fit an approximate
        # 365-day model from x_train+x_valid.
        return {
            "status": "deferred_to_compact_side_specific_packager",
            "packager": "scripts/refit_package_side_specific_base_champion.py",
            "latest_fold": str(fold.get("fold")),
            "missing_payloads": sorted(full_refit_payloads.difference(payload_paths)),
            "excluded_from_oos_metrics": True,
        }
    payload = _load_fold_payload_keys(
        fold,
        [
            "x_train",
            "train_target",
            "train_weight",
            "train_side",
            "train",
            "train_metrics",
            "x_valid",
            "valid",
            "valid_metrics",
        ],
    )
    if bool(payload.get("compact_fixed_training_payload")):
        if str(params["target_mode"]) != str(payload.get("fixed_training_target_mode")):
            raise ValueError("Final-refit target mode does not match compact payload")
        if str(params["weight_arm"]) != str(payload.get("fixed_training_weight_arm")):
            raise ValueError("Final-refit weight arm does not match compact payload")
        train_target = payload["train_target"]
        train_weight = payload["train_weight"]["sample_weight"]
    else:
        train_target = _target_from_frame(
            payload["train"],
            payload["train_metrics"],
            target_mode=str(params["target_mode"]),
        )
        train_weight = _weight_series(
            frame=payload["train"],
            metrics=payload["train_metrics"],
            target=train_target,
            arm=str(params["weight_arm"]),
        )
    valid_target = _target_from_frame(
        payload["valid"],
        payload["valid_metrics"],
        target_mode=str(params["target_mode"]),
    )
    valid_weight = _weight_series(
        frame=payload["valid"],
        metrics=payload["valid_metrics"],
        target=valid_target,
        arm=str(params["weight_arm"]),
    )
    x_full = pd.concat(
        [payload["x_train"], payload["x_valid"]],
        ignore_index=True,
        copy=False,
    ).astype(np.float32, copy=False)
    y_full = pd.concat(
        [train_target["target_soft"], valid_target["target_soft"]],
        ignore_index=True,
    )
    w_full = pd.concat([train_weight, valid_weight], ignore_index=True)
    train_side = payload.get("train_side")
    if isinstance(train_side, pd.DataFrame) and "side_name" in train_side:
        train_sides = train_side["side_name"].astype(str).str.lower().to_numpy()
    else:
        train_sides = _side_name_array(payload["train"])
    full_sides = np.concatenate(
        [train_sides, _side_name_array(payload["valid"])]
    )
    train_ts = pd.to_datetime(
        payload["train"]["__ts__"], utc=True, errors="coerce"
    )
    valid_ts = pd.to_datetime(
        payload["valid"]["__ts__"], utc=True, errors="coerce"
    )
    timestamps = pd.concat([train_ts, valid_ts], ignore_index=True)
    if len(timestamps) != len(x_full):
        raise ValueError("Final-refit timestamp and feature rows are misaligned")
    final_end = timestamps.max()
    final_start = timestamps.min()
    excluded_outside_train_window = 0
    if int(train_window_days) > 0:
        if pd.isna(final_end):
            raise ValueError("Final refit has no valid timestamp for sliding-window trim")
        final_start = final_end - pd.Timedelta(days=int(train_window_days))
        keep = timestamps.ge(final_start) & timestamps.le(final_end)
        excluded_outside_train_window = int((~keep).sum())
        x_full = x_full.loc[keep].reset_index(drop=True)
        y_full = y_full.loc[keep].reset_index(drop=True)
        w_full = w_full.loc[keep].reset_index(drop=True)
        full_sides = full_sides[keep.to_numpy()]
    train_rows_available = int(len(x_full))
    if int(max_train_rows) > 0 and len(x_full) > int(max_train_rows):
        idx = _time_spread_cap_rows(len(x_full), int(max_train_rows))
        x_fit = x_full.iloc[idx].reset_index(drop=True)
        y_fit = y_full.iloc[idx].reset_index(drop=True)
        w_fit = w_full.iloc[idx].reset_index(drop=True)
        sides_fit = full_sides[idx]
    else:
        x_fit, y_fit, w_fit = x_full, y_full, w_full
        sides_fit = full_sides
    final_seed = int(seed) + 9_000_000 + int(trial_number)
    model, feature_contracts = _fit_lgbm_models(
        x_train=x_fit,
        y_train=y_fit,
        w_train=w_fit,
        train_sides=sides_fit,
        params=params,
        seed=final_seed,
        model_side_scope=model_side_scope,
        features_by_side=fold.get("selected_features_by_side"),
    )
    pseudo_fold = dict(fold)
    pseudo_fold.update(
        {
            "fold": "final_all_rows",
            "month": "deployment",
            "valid_start": None,
            "valid_end": None,
            "max_oos_model_age_days": 0,
        }
    )
    manifest = _save_base_fold_model(
        model_dir=model_dir,
        fold=pseudo_fold,
        model=model,
        feature_names=feature_contracts,
        x_train=x_fit,
        imputation_fill_values=fold.get("train_median_imputation_values"),
        params=params,
        trial_number=int(trial_number),
        seed=final_seed,
        train_rows_available=train_rows_available,
        train_rows_fit=int(len(x_fit)),
        valid_rows=0,
        reuse_fingerprint=reuse_fingerprint,
    )
    manifest.update(
        {
            "status": "fitted",
            "schema": "s59_base_final_all_rows_model_v1",
            "train_start": final_start,
            "train_end": final_end,
            "train_window_days": int(train_window_days),
            "model_side_scope": str(model_side_scope),
            "excluded_outside_train_window_rows": int(
                excluded_outside_train_window
            ),
            "excluded_from_oos_metrics": True,
            "source_oos_fold": str(fold.get("fold")),
            "leakage_contract": {
                "fit_scope": (
                    f"trailing {int(train_window_days)} days through the latest "
                    "resolved labelled row"
                    if int(train_window_days) > 0
                    else "all labelled rows available after OOS scoring"
                ),
                "oos_metrics": "excluded; never used to generate reported OOS predictions",
                "feature_contract": "identical selected columns and frozen cycle AE/GMM state used by OOS folds",
                "target": "materialized trailing-label soft economic target",
            },
        }
    )
    manifest["ae_gmm_bundle"] = _package_final_ae_gmm_contract(
        final_dir=final_dir,
        state_path=fixed_ae_gmm_state_pkl,
        input_features=ae_gmm_input_features,
    )
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest

def _write_report(output_dir: Path, summary: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "materialized_trailing_label_topk_lgbm_hpo.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "rank",
        "trial_name",
        "objective",
        "target_mode",
        "weight_arm",
        "mean_top10_gross_ev_weighted_clean_precision",
        "mean_top20_gross_ev_weighted_clean_precision",
        "mean_top30_gross_ev_weighted_clean_precision",
        "mean_top10_clean_precision",
        "mean_top10_mean_first_touch_net",
        "mean_top10_q10_first_touch_net",
        "mean_top10_first_touch_timeout_rate",
        "mean_top10_first_touch_bad_mae_to_sl_rate",
        "mean_top10_selected_rows",
        "mean_top10_selected_symbols",
        "num_leaves",
        "min_child_samples",
        "learning_rate",
        "reg_lambda",
    ]
    best = str(summary.iloc[0]["trial_name"]) if not summary.empty else ""
    fold_cols = [
        "trial_name",
        "month",
        "top_frac",
        "gross_ev_weighted_clean_precision",
        "clean_precision",
        "mean_first_touch_net",
        "q10_first_touch_net",
        "hit_first_touch_net",
        "first_touch_timeout_rate",
        "first_touch_bad_mae_to_sl_rate",
        "selected_rows",
        "selected_symbols",
    ]
    lines = [
        "# Materialized Trailing Label Top-k LGBM HPO",
        "",
        "Scope: month-forward base-model HPO against already materialized trailing-profit labels. Primary metrics are top10/top20/top30 clean precision and gross-EV-weighted clean precision; net EV and path-risk rates are diagnostics.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Months: `{', '.join(manifest['fold_months'])}`",
        f"Features: `{manifest['feature_count']}` plus AE/GMM generated features `{manifest['ae_gmm_generated_features_by_fold']}` by fold.",
        "",
        "## Winner",
        "",
        table(summary.head(1), cols),
        "",
        "## Trial Ranking",
        "",
        table(summary, cols, limit=40),
        "",
        "## Winner Fold Detail",
        "",
        table(folds[folds["trial_name"].eq(best)], fold_cols),
        "",
        "## Outputs",
        "",
        f"- Trial summary: `{manifest['outputs']['trial_summary']}`",
        f"- Fold metrics: `{manifest['outputs']['fold_metrics']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Best params: `{manifest['outputs']['best_params']}`",
        f"- Best OOS scored ledger: `{manifest['outputs']['best_oos_scored_ledger']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_hpo(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    months: list[str],
    max_feature_store_features: int | None,
    max_train_rows: int,
    feature_selection_sample_rows: int,
    hpo_max_train_rows: int,
    n_trials: int,
    seed: int,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_gmm_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    feature_selection_top_n: int,
    feature_selection_target_mode: str,
    feature_selection_method: str,
    model_side_scope: str,
    max_oos_model_age_days: int,
    single_fit_oos_window: bool = False,
    train_window_days: int = 0,
    label_path_purge_hours: float = 25.0,
    ae_gmm_anchor_days: int = 0,
    ae_gmm_input_features_csv: Path | None = None,
    fixed_params_json: Path | None = None,
    fixed_selected_features_csv: Path | None = None,
    fixed_ae_gmm_state_pkl: Path | None = None,
    allow_refit_ae_gmm_with_fixed_features: bool = False,
    refit_ae_gmm_per_window: bool = False,
    existing_scored_ledger_path: Path | None = None,
    missing_only: bool = False,
    rerun_hpo: bool = False,
    rerun_ae_gmm_hpo: bool = False,
    fresh_feature_selection_requested: bool = False,
    refit_cycle_ae_gmm_requested: bool = False,
    save_fold_models: bool = False,
    save_final_model: bool = True,
    two_phase_wide_feature_selection: bool = True,
    hpo_only: bool = False,
    external_feature_sidecar_path: Path | None = None,
    target_sidecar_path: Path | None = None,
    frozen_ae_gmm_output_sidecar_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_params_path = (
        Path(fixed_params_json) if fixed_params_json is not None else None
    )
    fixed_training_contract = (
        _load_fixed_params(fixed_params_path)
        if fixed_params_path is not None and not bool(rerun_hpo)
        else None
    )
    if bool(refit_ae_gmm_per_window):
        raise ValueError(
            "--refit-ae-gmm-per-window is incompatible with the production cycle "
            "contract. AE/GMM is fitted once on the feature-selection/HPO reference "
            "sample and reused for every growing window and final refit."
        )
    fold_cache_dir = output_dir / "_fold_cache"
    fixed_selected_features = _load_fixed_selected_features(fixed_selected_features_csv)
    fixed_selected_features_by_side = _load_fixed_selected_features_by_side(
        fixed_selected_features_csv,
        fixed_selected_features,
    )
    had_frozen_feature_contract = fixed_selected_features is not None
    diagnostic_single_phase_opt_out = bool(
        not had_frozen_feature_contract and not bool(two_phase_wide_feature_selection)
    )
    ae_gmm_input_features = _load_fixed_selected_features(ae_gmm_input_features_csv)
    ae_gmm_input_policy_diag: dict[str, Any] = {
        "policy": "explicit_csv" if ae_gmm_input_features_csv is not None else str(AE_GMM_INPUT_POLICY or "a0bis"),
        "selected_input_feature_count_before_policy": int(len(fixed_selected_features or [])),
        "selected_input_feature_count_after_policy": int(len(ae_gmm_input_features or [])),
        "removed_raw_momentum_count": 0,
        "added_normalized_momentum_count": 0,
        "removed_raw_momentum_features": [],
        "added_normalized_momentum_features": [],
    }
    if bool(include_ae_gmm_state_features) and fixed_ae_gmm_state_pkl is not None:
        frozen_state = load_ae_gmm_state_artifact(Path(fixed_ae_gmm_state_pkl))
        frozen_state_inputs = [
            str(value)
            for value in (
                frozen_state.get("feature_columns")
                or frozen_state.get("input_feature_columns")
                or []
            )
            if str(value)
        ]
        if not frozen_state_inputs:
            raise ValueError(
                "Frozen AE/GMM state does not persist its ordered input-feature "
                f"contract: {fixed_ae_gmm_state_pkl}"
            )
        if (
            ae_gmm_input_features_csv is not None
            and list(ae_gmm_input_features or []) != frozen_state_inputs
        ):
            raise ValueError(
                "Explicit AE/GMM inputs do not exactly match the frozen state's "
                "ordered input contract. Remove --ae-gmm-input-features-csv or "
                "supply the exact state inputs."
            )
        ae_gmm_input_features = frozen_state_inputs
        ae_gmm_input_policy_diag = {
            **ae_gmm_input_policy_diag,
            "policy": "frozen_state_ordered_input_contract",
            "selected_input_feature_count_after_policy": int(
                len(ae_gmm_input_features)
            ),
            "frozen_state_path": str(fixed_ae_gmm_state_pkl),
        }
    if (
        bool(include_ae_gmm_state_features)
        and ae_gmm_input_features_csv is None
        and fixed_ae_gmm_state_pkl is None
        and fixed_selected_features
    ):
        ae_gmm_input_features, ae_gmm_input_policy_diag = _default_ae_gmm_input_features(
            fixed_selected_features,
            list(dict.fromkeys([*_label_schema_columns(labels_path), *_read_feature_list(feature_list_csv)])),
        )
    fixed_selected_ae_gmm = _fixed_selected_ae_gmm_features(fixed_selected_features)
    if (
        bool(include_ae_gmm_state_features)
        and fixed_selected_features_csv is not None
        and fixed_selected_ae_gmm
        and fixed_ae_gmm_state_pkl is None
        and not bool(allow_refit_ae_gmm_with_fixed_features)
    ):
        preview = ", ".join(fixed_selected_ae_gmm[:12])
        raise ValueError(
            "Refusing to refit AE/GMM while reusing a fixed selected-feature list "
            "that contains AE/GMM-generated columns. This can change feature "
            "semantics versus the feature-selection/HPO artifact and confuse "
            "downstream frozen models. Pass --fixed-ae-gmm-state-pkl with the "
            "train-fitted state from the source artifact, rerun feature selection/HPO, "
            "or explicitly pass --allow-refit-ae-gmm-with-fixed-features for a "
            f"diagnostic-only run. AE/GMM fixed features include: {preview}"
        )
    two_phase_manifest: dict[str, Any] | None = None
    selection_feature_table: pd.DataFrame | None = None
    frozen_ae_gmm_output_sidecar_contract: dict[str, Any] | None = None
    if frozen_ae_gmm_output_sidecar_path is not None:
        if fixed_ae_gmm_state_pkl is None:
            raise ValueError(
                "--frozen-ae-gmm-output-sidecar requires --fixed-ae-gmm-state-pkl"
            )
        frozen_ae_gmm_output_sidecar_path = Path(
            frozen_ae_gmm_output_sidecar_path
        )
        frozen_ae_gmm_output_sidecar_contract = (
            _validate_frozen_ae_gmm_output_sidecar(
                labels_path=labels_path,
                state_path=Path(fixed_ae_gmm_state_pkl),
                sidecar_path=frozen_ae_gmm_output_sidecar_path,
            )
        )
    use_two_phase_selection = use_canonical_two_phase_feature_selection(
        has_frozen_feature_contract=had_frozen_feature_contract,
        diagnostic_single_phase=not bool(two_phase_wide_feature_selection),
    )
    if use_two_phase_selection:
        two_phase_union_cap = (
            None
            if str(feature_selection_method).startswith("archetype_prescreen_side_mda")
            else 150
        )
        selection_sample_dir = output_dir / "_feature_selection_bme_sample"
        selection_sample_path = selection_sample_dir / "labels.parquet"
        selection_sample_dir.mkdir(parents=True, exist_ok=True)
        sample_rows = max(int(feature_selection_sample_rows), 300)
        sample_manifest_path = selection_sample_dir / "manifest.json"
        selection_result_path = selection_sample_dir / "selection_result.json"
        selected_contract_path = selection_sample_dir / "selected_features.json"
        if sample_manifest_path.is_file() and selection_sample_path.is_file():
            sample_contract = json.loads(sample_manifest_path.read_text())
        else:
            selection_end = min(
                pd.Timestamp(pd.Period(month).start_time, tz="UTC")
                for month in months
            )
            selection_start = (
                selection_end - pd.Timedelta(days=int(train_window_days))
                if int(train_window_days) > 0
                else None
            )
            sample_contract = materialize_bme_parquet_sample(
                labels_path,
                selection_sample_path,
                max_rows=sample_rows,
                seed=int(seed),
                timestamp_column="__ts__",
                identity_columns=("__symbol__", "side"),
                min_timestamp=selection_start,
                max_timestamp_exclusive=selection_end,
            )
            sample_manifest_path.write_text(
                json.dumps(_json_safe(sample_contract), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        reusable_selection: dict[str, Any] | None = None
        if selection_result_path.is_file() and selected_contract_path.is_file():
            candidate = json.loads(selection_result_path.read_text(encoding="utf-8"))
            candidate_state = Path(str(candidate.get("frozen_ae_gmm_state") or ""))
            candidate_features = _load_fixed_selected_features(selected_contract_path)
            if (
                str(candidate.get("training_contract"))
                == str(LGBM_TWO_PHASE_SELECTION_CONTRACT)
                and candidate_state.is_file()
                and candidate_features
            ):
                reusable_selection = candidate
        selection_cache_dir = output_dir / "_feature_selection_phase/_fold_cache"
        if reusable_selection is not None:
            selection_manifest = dict(
                reusable_selection.get("selection_manifest", {}) or {}
            )
            fixed_selected_features = list(
                _load_fixed_selected_features(selected_contract_path) or []
            )
            if two_phase_union_cap is not None:
                fixed_selected_features = fixed_selected_features[:two_phase_union_cap]
            fixed_selected_features_by_side = {
                side: [
                    str(feature)
                    for feature in dict(
                        reusable_selection.get("selected_features_by_side") or {}
                    ).get(side, [])
                    if str(feature) in fixed_selected_features
                ]
                for side in ("long", "short")
            }
            selected_contract_path.write_text(
                json.dumps(
                    {
                        "selected_features": fixed_selected_features,
                        "selected_features_by_side": fixed_selected_features_by_side,
                        "selected_feature_count": len(fixed_selected_features),
                        "source": "reused_two_phase_selection_capped_150",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            fixed_selected_features_csv = selected_contract_path
            fixed_ae_gmm_state_pkl = Path(
                str(reusable_selection["frozen_ae_gmm_state"])
            )
            ae_gmm_input_features = list(
                selection_manifest.get("ae_gmm_input_features", []) or []
            )
            selection_feature_table = pd.DataFrame(
                {
                    "fold": ["reused_two_phase_selection"]
                    * len(fixed_selected_features),
                    "feature": fixed_selected_features,
                    "score": np.nan,
                    "rank": np.arange(
                        1, len(fixed_selected_features) + 1, dtype=np.int32
                    ),
                    "selected": True,
                    "feature_selection_method": "reused_exact_contract",
                    "feature_selection_status": "reused_after_interruption",
                }
            )
            two_phase_manifest = dict(reusable_selection)
            two_phase_manifest["resume_status"] = "reused_exact_selection_and_state"
            print(
                "[two_phase] reused_selection "
                f"features={len(fixed_selected_features)} "
                f"state={fixed_ae_gmm_state_pkl}",
                flush=True,
            )
        else:
            # The B/M/E selection artifact is intentionally bounded strictly
            # before the requested OOS scope.  Its selector therefore needs an
            # internal chronological validation month from the end of the
            # training period; asking it for the external OOS months produces
            # no eligible fold and, more importantly, would mix model selection
            # with the controlled evaluation period.
            selection_end = min(
                pd.Timestamp(pd.Period(month).start_time, tz="UTC")
                for month in months
            )
            selection_validation_month = str(
                (selection_end - pd.Timedelta(nanoseconds=1)).to_period("M")
            )
            selection_folds, selection_manifest = _prepare_folds(
                labels_path=selection_sample_path,
                feature_dir=feature_dir,
                feature_list_csv=feature_list_csv,
                max_feature_store_features=max_feature_store_features,
                months=[selection_validation_month],
                include_ae_gmm_state_features=include_ae_gmm_state_features,
                ae_gmm_state_feature_max_train_rows=ae_gmm_state_feature_max_train_rows,
                ae_gmm_state_feature_gmm_max_train_rows=ae_gmm_state_feature_gmm_max_train_rows,
                ae_gmm_state_feature_max_iter=ae_gmm_state_feature_max_iter,
                feature_selection_top_n=feature_selection_top_n,
                feature_selection_target_mode=feature_selection_target_mode,
                feature_selection_method=feature_selection_method,
                max_oos_model_age_days=int(max_oos_model_age_days),
                single_fit_oos_window=bool(single_fit_oos_window),
                train_window_days=int(train_window_days),
                label_path_purge_hours=float(label_path_purge_hours),
                ae_gmm_anchor_days=int(ae_gmm_anchor_days),
                payload_max_train_rows=sample_rows,
                fold_cache_dir=selection_cache_dir,
                fixed_selected_features=None,
                fixed_selected_features_by_side=None,
                fixed_selected_features_path=None,
                fixed_ae_gmm_state_pkl=fixed_ae_gmm_state_pkl,
                ae_gmm_input_features=ae_gmm_input_features,
                freeze_ae_gmm_state_after_reference=True,
                existing_scored_ledger_path=None,
                missing_only=False,
                seed=seed,
                selection_only=True,
                external_feature_sidecar_path=external_feature_sidecar_path,
                target_sidecar_path=target_sidecar_path,
            )
            fixed_selected_features = list(
                selection_manifest.get("selected_feature_union", []) or []
            )
            if two_phase_union_cap is not None:
                fixed_selected_features = fixed_selected_features[:two_phase_union_cap]
            fixed_selected_features_by_side = {
                side: [
                    str(feature)
                    for feature in dict(
                        selection_manifest.get("selected_features_by_side") or {}
                    ).get(side, [])
                    if str(feature) in fixed_selected_features
                ]
                for side in ("long", "short")
            }
            if not fixed_selected_features:
                raise RuntimeError("Two-phase base selection produced no feature contract")
            fixed_selected_features_csv = selected_contract_path
            fixed_selected_features_csv.write_text(
                json.dumps(
                    {
                        "selected_features": fixed_selected_features,
                        "selected_features_by_side": fixed_selected_features_by_side,
                        "selected_feature_count": len(fixed_selected_features),
                        "source": "two_phase_bme_feature_selection",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            fixed_ae_gmm_state_pkl = Path(
                str(selection_manifest.get("ae_gmm_state_reference_state_path") or "")
            )
            if include_ae_gmm_state_features and not fixed_ae_gmm_state_pkl.is_file():
                raise RuntimeError(
                    "Two-phase base selection did not persist its frozen AE/GMM state"
                )
            ae_gmm_input_features = list(
                selection_manifest.get("ae_gmm_input_features", []) or []
            )
            if selection_folds:
                selection_feature_table = selection_folds[0].get("feature_selection")
            if selection_feature_table is not None and not selection_feature_table.empty:
                selection_feature_table.to_csv(
                    selection_sample_dir / "feature_selection_importance.csv",
                    index=False,
                )
            two_phase_manifest = {
                "schema": "base_two_phase_wide_selection_v1",
                "training_contract": LGBM_TWO_PHASE_SELECTION_CONTRACT,
                "sample": sample_contract,
                "selection_manifest": selection_manifest,
                "selected_feature_count": len(fixed_selected_features),
                "selected_features_by_side": fixed_selected_features_by_side,
                "frozen_ae_gmm_state": str(fixed_ae_gmm_state_pkl),
                "full_population_reload": "selected_raw_columns_plus_frozen_ae_gmm_inputs",
                "full_population_train_row_cap": int(max_train_rows),
            }
            selection_result_path.write_text(
                json.dumps(_json_safe(two_phase_manifest), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        if include_ae_gmm_state_features:
            selected_ae_gmm_outputs = _fixed_selected_ae_gmm_features(
                fixed_selected_features
            )
            if selected_ae_gmm_outputs and frozen_ae_gmm_output_sidecar_path is None:
                # Persist the complete frozen representation for the meta
                # handoff.  The base matrix remains restricted to the selected
                # subset, so this does not widen the full base fit payload.
                sidecar_output_features = list(AE_GMM_FEATURE_COLUMNS)
                frozen_ae_gmm_output_sidecar_path, frozen_ae_gmm_output_sidecar_contract = (
                    _materialize_frozen_ae_gmm_output_sidecar(
                        labels_path=labels_path,
                        feature_dir=feature_dir,
                        state_path=Path(fixed_ae_gmm_state_pkl),
                        output_path=(
                            output_dir
                            / "_frozen_ae_gmm_outputs"
                            / "selected_outputs.parquet"
                        ),
                        output_features=sidecar_output_features,
                    )
                )
                if two_phase_manifest is not None:
                    two_phase_manifest["frozen_ae_gmm_output_sidecar"] = (
                        frozen_ae_gmm_output_sidecar_contract
                    )
            elif selected_ae_gmm_outputs and two_phase_manifest is not None:
                two_phase_manifest["frozen_ae_gmm_output_sidecar"] = (
                    frozen_ae_gmm_output_sidecar_contract
                )
    run_reuse_fingerprint = _reuse_fingerprint(
        {
            "schema": "base_run_reuse_v1",
            "labels": _label_source_identity(labels_path),
            "feature_dir": str(Path(feature_dir).resolve()),
            "feature_list": _file_identity(Path(feature_list_csv)),
            "external_feature_sidecar": _file_identity(
                external_feature_sidecar_path, include_sha256=False
            ),
            "target_sidecar": _file_identity(target_sidecar_path, include_sha256=False),
            "selected_features": list(map(str, fixed_selected_features or [])),
            "fixed_params": _file_identity(fixed_params_path),
            "ae_gmm_state": _file_identity(fixed_ae_gmm_state_pkl),
            "frozen_ae_gmm_output_sidecar": _file_identity(
                frozen_ae_gmm_output_sidecar_path, include_sha256=False
            ),
            "months": list(months),
            "max_train_rows": int(max_train_rows),
            "train_window_days": int(train_window_days),
            "max_oos_model_age_days": int(max_oos_model_age_days),
            "single_fit_oos_window": bool(single_fit_oos_window),
            "missing_only": bool(missing_only),
            "seed": int(seed),
            "fixed_training_contract": dict(fixed_training_contract or {}),
        }
    )
    folds, manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        months=months,
        include_ae_gmm_state_features=include_ae_gmm_state_features,
        ae_gmm_state_feature_max_train_rows=ae_gmm_state_feature_max_train_rows,
        ae_gmm_state_feature_gmm_max_train_rows=ae_gmm_state_feature_gmm_max_train_rows,
        ae_gmm_state_feature_max_iter=ae_gmm_state_feature_max_iter,
        feature_selection_top_n=feature_selection_top_n,
        feature_selection_target_mode=feature_selection_target_mode,
        feature_selection_method=feature_selection_method,
        max_oos_model_age_days=int(max_oos_model_age_days),
        single_fit_oos_window=bool(single_fit_oos_window),
        train_window_days=int(train_window_days),
        label_path_purge_hours=float(label_path_purge_hours),
        ae_gmm_anchor_days=int(ae_gmm_anchor_days),
        payload_max_train_rows=int(max_train_rows),
        fold_cache_dir=fold_cache_dir,
        fixed_selected_features=fixed_selected_features,
        fixed_selected_features_by_side=fixed_selected_features_by_side,
        fixed_selected_features_path=fixed_selected_features_csv,
        fixed_ae_gmm_state_pkl=fixed_ae_gmm_state_pkl,
        ae_gmm_input_features=ae_gmm_input_features,
        freeze_ae_gmm_state_after_reference=not bool(refit_ae_gmm_per_window),
        existing_scored_ledger_path=existing_scored_ledger_path,
        missing_only=bool(missing_only),
        seed=seed,
        fixed_training_contract=fixed_training_contract,
        external_feature_sidecar_path=external_feature_sidecar_path,
        target_sidecar_path=target_sidecar_path,
        frozen_ae_gmm_output_sidecar_path=frozen_ae_gmm_output_sidecar_path,
        run_reuse_fingerprint=run_reuse_fingerprint,
    )
    if selection_feature_table is not None and folds:
        folds[0]["feature_selection"] = selection_feature_table
    if two_phase_manifest is not None:
        manifest["two_phase_feature_selection"] = two_phase_manifest
    manifest["feature_materialization_contract"] = {
        "schema": LGBM_TWO_PHASE_SELECTION_CONTRACT,
        "canonical_default": True,
        "fresh_selection_used_two_phase": bool(use_two_phase_selection),
        "diagnostic_single_phase_opt_out": diagnostic_single_phase_opt_out,
        "selection_sample_rows": int(feature_selection_sample_rows),
        "full_fit_projection": (
            "selected_raw_columns_plus_precomputed_selected_ae_gmm_outputs"
            if frozen_ae_gmm_output_sidecar_path is not None
            else "selected_raw_columns_plus_frozen_ae_gmm_inputs"
        ),
        "full_fit_train_row_cap": int(max_train_rows),
        "full_fit_population": (
            "all_prior_rows" if int(max_train_rows) <= 0 else "explicitly_capped"
        ),
    }
    if not folds:
        if bool(missing_only):
            paths = {
                "trial_summary": output_dir / "topk_lgbm_hpo_trials.csv",
                "fold_metrics": output_dir / "topk_lgbm_hpo_folds.csv",
                "diagnostics": output_dir / "topk_lgbm_hpo_diagnostics.csv",
                "feature_selection": output_dir / "topk_lgbm_feature_selection_by_fold.csv",
                "best_oos_scored_ledger": output_dir / "best_oos_scored_ledger.parquet",
                "best_params": output_dir / "topk_lgbm_hpo_best.json",
                "manifest": output_dir / "manifest.json",
            }
            pd.DataFrame().to_csv(paths["trial_summary"], index=False)
            pd.DataFrame().to_csv(paths["fold_metrics"], index=False)
            pd.DataFrame().to_csv(paths["diagnostics"], index=False)
            pd.DataFrame().to_csv(paths["feature_selection"], index=False)
            pd.DataFrame().to_parquet(paths["best_oos_scored_ledger"], index=False)
            best_payload = {"status": "no_missing_rows", "params": {}}
            paths["best_params"].write_text(json.dumps(_json_safe(best_payload), indent=2), encoding="utf-8")
            manifest.update(
                {
                    "scope": "materialized_trailing_label_topk_lgbm_hpo",
                    "status": "no_missing_rows",
                    "labels_path": str(labels_path),
                    "feature_dir": str(feature_dir),
                    "feature_list_csv": str(feature_list_csv),
                    "output_dir": str(output_dir),
                    "months": list(months),
                    "train_window_days": int(train_window_days),
                    "ae_gmm_anchor_days": int(ae_gmm_anchor_days),
                    "ae_gmm_input_features_csv": str(ae_gmm_input_features_csv) if ae_gmm_input_features_csv is not None else None,
                    "ae_gmm_input_feature_count": int(len(ae_gmm_input_features or [])),
                    "fixed_params_json": str(fixed_params_json) if fixed_params_json is not None else None,
                    "fixed_selected_features_csv": str(fixed_selected_features_csv) if fixed_selected_features_csv is not None else None,
                    "fixed_ae_gmm_state_pkl": str(fixed_ae_gmm_state_pkl) if fixed_ae_gmm_state_pkl is not None else None,
                    "existing_scored_ledger_path": str(existing_scored_ledger_path) if existing_scored_ledger_path is not None else None,
                    "missing_only": True,
                    "outputs": {key: str(value) for key, value in paths.items()},
                }
            )
            paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
            return manifest
        raise RuntimeError("No valid OOS folds prepared")
    hpo_folds = [max(folds, key=lambda fold: int(fold.get("train_rows", 0)))]
    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    available_target_modes = [
        mode
        for mode in TARGET_MODES
        if mode != "p90_trailing_blend" or target_sidecar_path is not None
    ]
    try:
        import optuna
    except Exception:
        optuna = None

    def evaluate(params: dict[str, Any], trial_number: int) -> float:
        print(
            "[hpo] evaluating "
            f"trial={int(trial_number)} target={params.get('target_mode')} weight={params.get('weight_arm')}",
            flush=True,
        )
        summary, rows, diag = _run_trial(
            folds=hpo_folds,
            params=params,
            trial_number=trial_number,
            max_train_rows=int(hpo_max_train_rows),
            seed=seed,
            model_side_scope=model_side_scope,
        )
        summaries.append(summary)
        fold_rows.extend(rows)
        diagnostics.extend(diag)
        print(
            "[hpo] completed "
            f"trial={int(trial_number)} objective={float(summary['objective']):.6f}",
            flush=True,
        )
        return float(summary["objective"])

    trial_counter = 0
    if fixed_params_path is not None and not bool(rerun_hpo):
        fixed_params = dict(fixed_training_contract or _load_fixed_params(fixed_params_path))
        fixed_trial_number = int(fixed_params.pop("_fixed_trial_number", trial_counter))
        evaluate(fixed_params, fixed_trial_number)
        trial_counter += 1
    else:
        baselines: list[dict[str, Any]] = []
        if fixed_params_path is not None and fixed_params_path.is_file():
            incumbent = dict(_load_fixed_params(fixed_params_path))
            incumbent.pop("_fixed_trial_number", None)
            # A fresh HPO run always evaluates its incumbent with the canonical
            # L2 loss, regardless of the historical artifact's fitted loss.
            incumbent["loss_function"] = "regression"
            incumbent.setdefault("min_split_gain", 0.0)
            baselines.append(incumbent)
        baselines.extend([
            {
                "n_estimators": 180,
                "learning_rate": 0.035,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 45,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.05,
                "reg_lambda": 2.0,
                "loss_function": "regression",
                "min_split_gain": 0.0,
                "target_mode": "policy_soft",
                "weight_arm": "W0_base",
            },
            {
                "n_estimators": 220,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 55,
                "subsample": 0.82,
                "colsample_bytree": 0.78,
                "reg_alpha": 0.10,
                "reg_lambda": 4.0,
                "loss_function": "regression",
                "min_split_gain": 1e-3,
                "target_mode": "exec_guarded_policy",
                "weight_arm": "W8_combined_conservative",
            },
        ])
        for params in baselines:
            evaluate(dict(params), trial_counter)
            trial_counter += 1
    if (fixed_params_path is None or bool(rerun_hpo)) and optuna is not None and int(n_trials) > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            nonlocal trial_counter
            params = _suggest_params(
                trial,
                rng,
                target_modes=available_target_modes,
            )
            value = evaluate(params, trial_counter)
            if summaries:
                for key, val in summaries[-1].items():
                    if isinstance(val, (int, float)) and math.isfinite(float(val)):
                        trial.set_user_attr(key, float(val))
            trial_counter += 1
            return value

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    elif fixed_params_path is None or bool(rerun_hpo):
        for _ in range(int(n_trials)):
            evaluate(
                _suggest_params(
                    None,
                    rng,
                    target_modes=available_target_modes,
                ),
                trial_counter,
            )
            trial_counter += 1

    summary_df = pd.DataFrame(summaries).sort_values("objective", ascending=False).reset_index(drop=True)
    if "rank" in summary_df.columns:
        summary_df = summary_df.drop(columns=["rank"])
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1, dtype=np.int32))
    folds_df = pd.DataFrame(fold_rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    feature_selection_df = (
        pd.concat([fold["feature_selection"] for fold in folds], ignore_index=True)
        if folds
        else pd.DataFrame(columns=["fold", "feature", "score", "rank", "selected", "feature_selection_status"])
    )
    paths = {
        "trial_summary": output_dir / "topk_lgbm_hpo_trials.csv",
        "fold_metrics": output_dir / "topk_lgbm_hpo_folds.csv",
        "diagnostics": output_dir / "topk_lgbm_hpo_diagnostics.csv",
        "feature_selection": output_dir / "topk_lgbm_feature_selection_by_fold.csv",
        "best_oos_scored_ledger": output_dir / "best_oos_scored_ledger.parquet",
        "best_params": output_dir / "topk_lgbm_hpo_best.json",
        "manifest": output_dir / "manifest.json",
    }
    summary_df.to_csv(paths["trial_summary"], index=False)
    folds_df.to_csv(paths["fold_metrics"], index=False)
    diagnostics_df.to_csv(paths["diagnostics"], index=False)
    feature_selection_df.to_csv(paths["feature_selection"], index=False)
    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    final_model_manifest: dict[str, Any] = {
        "status": "disabled" if not bool(save_final_model) else "skipped_no_best"
    }
    if best and not bool(hpo_only):
        best_trial_number = int(float(best.get("trial_number", 0)))
        best_params_for_scoring = _best_params_from_summary_row(best)
        best_ledger = _score_best_oos_ledger(
            folds=folds,
            params=best_params_for_scoring,
            trial_number=best_trial_number,
            max_train_rows=int(max_train_rows),
            seed=int(seed),
            save_fold_models_dir=(output_dir / "models") if bool(save_fold_models) else None,
            output_path=paths["best_oos_scored_ledger"],
            model_side_scope=model_side_scope,
        )
        saved_fold_models = list(best_ledger.attrs.get("saved_fold_models", []))
        model_input_parity_sidecars = list(
            best_ledger.attrs.get("model_input_parity_sidecars", [])
        )
        # Parquet stores DataFrame.attrs as JSON metadata. Fold manifests contain
        # pandas Timestamps, so keep them in the run manifest and strip attrs from
        # the tabular ledger before serialization.
        streamed_output = best_ledger.attrs.get("streamed_to_output_path")
        best_ledger.attrs.clear()
        if not streamed_output:
            best_ledger.to_parquet(paths["best_oos_scored_ledger"], index=False)
        if bool(save_final_model):
            final_model_manifest = _fit_final_all_rows_base_model(
                folds=folds,
                params=best_params_for_scoring,
                trial_number=best_trial_number,
                max_train_rows=int(max_train_rows),
                train_window_days=int(train_window_days),
                model_side_scope=str(model_side_scope),
                seed=int(seed),
                model_dir=output_dir / "models",
                fixed_ae_gmm_state_pkl=fixed_ae_gmm_state_pkl,
                ae_gmm_input_features=ae_gmm_input_features,
                reuse_fingerprint=_reuse_fingerprint(
                    {
                        "schema": "base_final_model_reuse_v1",
                        "run_fingerprint": run_reuse_fingerprint,
                        "params": best_params_for_scoring,
                        "trial_number": best_trial_number,
                        "max_train_rows": int(max_train_rows),
                        "seed": int(seed),
                        "source_fold_fingerprint": str(
                            max(
                                folds,
                                key=lambda item: pd.Timestamp(item["valid_end"]),
                            ).get("reuse_fingerprint")
                            or ""
                        ),
                    }
                ),
            )
    else:
        saved_fold_models = []
        model_input_parity_sidecars = []
        pd.DataFrame().to_parquet(paths["best_oos_scored_ledger"], index=False)
    paths["best_params"].write_text(json.dumps(_json_safe(best), indent=2), encoding="utf-8")
    manifest.update(
        {
            "scope": "materialized_trailing_label_topk_lgbm_hpo",
            "base_feature_selection_contract": BASE_SINGLE_CYCLE_MDA_SELECTION_CONTRACT,
            "base_feature_selection_recipe": canonical_base_feature_selection_recipe(),
            "labels_path": str(labels_path),
            "run_reuse_fingerprint": run_reuse_fingerprint,
            "feature_dir": str(feature_dir),
            "feature_list_csv": str(feature_list_csv),
            "external_feature_sidecar_path": (
                str(external_feature_sidecar_path)
                if external_feature_sidecar_path is not None
                else None
            ),
            "external_feature_sidecar_sha256": (
                _sha256_file(Path(external_feature_sidecar_path))
                if external_feature_sidecar_path is not None
                else None
            ),
            "output_dir": str(output_dir),
            "max_feature_store_features": max_feature_store_features,
            "max_train_rows": int(max_train_rows),
            "hpo_scope": "single_largest_train_fold",
            "hpo_sampling": "beginning_middle_end_time_spread",
            "hpo_global_calibration_note": (
                "parameters are selected once on the largest train fold and reused for all OOS scoring folds"
            ),
            "hpo_calibration_fold": str(hpo_folds[0].get("fold")) if hpo_folds else None,
            "hpo_calibration_train_rows": int(hpo_folds[0].get("train_rows", 0)) if hpo_folds else 0,
            "hpo_max_train_rows": int(hpo_max_train_rows),
            "feature_selection_sample_rows": int(feature_selection_sample_rows),
            "n_trials_requested": int(n_trials),
            "fixed_params_json": str(fixed_params_path) if fixed_params_path is not None else None,
            "fixed_selected_features_csv": str(fixed_selected_features_csv) if fixed_selected_features_csv is not None else None,
            "fixed_selected_features_count": int(len(fixed_selected_features or [])),
            "fixed_ae_gmm_state_pkl": str(fixed_ae_gmm_state_pkl) if fixed_ae_gmm_state_pkl is not None else None,
            "train_window_days": int(train_window_days),
            "single_fit_oos_window": bool(single_fit_oos_window),
            "ae_gmm_anchor_days": int(ae_gmm_anchor_days),
            "ae_gmm_input_features_csv": str(ae_gmm_input_features_csv) if ae_gmm_input_features_csv is not None else None,
            "ae_gmm_input_feature_count": int(len(ae_gmm_input_features or [])),
            "ae_gmm_input_policy": str(ae_gmm_input_policy_diag.get("policy", "")),
            "ae_gmm_input_feature_count_before_policy": int(
                ae_gmm_input_policy_diag.get("selected_input_feature_count_before_policy", 0) or 0
            ),
            "ae_gmm_input_removed_raw_momentum_count": int(
                ae_gmm_input_policy_diag.get("removed_raw_momentum_count", 0) or 0
            ),
            "ae_gmm_input_added_normalized_momentum_count": int(
                ae_gmm_input_policy_diag.get("added_normalized_momentum_count", 0) or 0
            ),
            "ae_gmm_input_removed_raw_momentum_features": list(
                ae_gmm_input_policy_diag.get("removed_raw_momentum_features", []) or []
            ),
            "ae_gmm_input_added_normalized_momentum_features": list(
                ae_gmm_input_policy_diag.get("added_normalized_momentum_features", []) or []
            ),
            "ae_gmm_state_ae_max_train_rows": int(ae_gmm_state_feature_max_train_rows),
            "ae_gmm_state_gmm_max_train_rows": int(ae_gmm_state_feature_gmm_max_train_rows),
            "ae_gmm_refit_per_window": False,
            "ae_gmm_state_reuse_policy": "single_cycle_state_for_all_windows_and_final_refit",
            "existing_scored_ledger_path": str(existing_scored_ledger_path) if existing_scored_ledger_path is not None else None,
            "missing_only": bool(missing_only),
            "rerun_hpo": bool(rerun_hpo),
            "rerun_ae_gmm_hpo": bool(rerun_ae_gmm_hpo),
            "fresh_feature_selection_requested": bool(
                fresh_feature_selection_requested
            ),
            "refit_cycle_ae_gmm_requested": bool(
                refit_cycle_ae_gmm_requested
            ),
            "save_fold_models": bool(save_fold_models),
            "saved_fold_models": _json_safe(saved_fold_models),
            "model_input_parity": {
                "schema": MODEL_INPUT_PARITY_SCHEMA,
                "root": str(output_dir / "model_input_parity"),
                "fold_sidecars": _json_safe(model_input_parity_sidecars),
            },
            "save_final_model": bool(save_final_model),
            "hpo_only": bool(hpo_only),
            "final_model": _json_safe(final_model_manifest),
            "search_mode": "fixed_params_eval" if fixed_params_path is not None and not bool(rerun_hpo) else "hpo",
            "seed": int(seed),
            "top_fracs": list(TOP_FRACS),
            "target_modes": list(TARGET_MODES),
            "primary_objective": "top30_first_gross_ev_weighted_clean_precision_plus_net_ev_penalties",
            "model_side_scope": str(model_side_scope),
            "outputs": {key: str(value) for key, value in paths.items()},
        }
    )
    report = _write_report(output_dir, summary_df, folds_df, manifest)
    manifest["outputs"]["report"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=LGBM_TWO_PHASE_FULL_FIT_ROW_CAP,
        help=(
            "Rows used by each growing-window model fit. Default 0 uses every "
            "available prior row; positive values are diagnostic compute caps."
        ),
    )
    parser.add_argument(
        "--target-sidecar",
        type=Path,
        default=None,
        help="Keyed target-only sidecar. It is excluded from model feature contracts.",
    )
    parser.add_argument(
        "--feature-selection-sample-rows",
        type=int,
        default=LGBM_TWO_PHASE_SELECTION_SAMPLE_ROWS,
        help=(
            "Rows materialized for the one-shot beginning/middle/end feature-"
            "selection phase. The default 300k restores the evidence population "
            "used by the strongest base feature contract; MDA internally caps "
            "its model/evaluation samples at 60k/20k."
        ),
    )
    parser.add_argument(
        "--hpo-max-train-rows",
        type=int,
        default=LGBM_HPO_SAMPLE_ROWS,
        help=(
            "Rows used by each one-shot HPO trial, sampled from the beginning/"
            "middle/end of the largest calibration scope. This is independent "
            "from --feature-selection-sample-rows."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-params-json",
        type=Path,
        default=DEFAULT_FIXED_PARAMS_JSON,
        help="Evaluate this fixed parameter recipe instead of launching HPO. Pass an empty path with --rerun-hpo to search.",
    )
    parser.add_argument(
        "--fixed-selected-features-csv",
        type=Path,
        default=DEFAULT_FIXED_SELECTED_FEATURES_CSV,
        help="CSV/JSON of previously selected features. When set, skips feature selection and reuses these columns.",
    )
    parser.add_argument(
        "--single-phase-wide-feature-selection",
        action="store_true",
        help=(
            "Diagnostic opt-out. The default two-phase path performs MDA/AE-GMM "
            "on a wide 45k B/M/E sample, then reloads all model-fit rows with "
            "only selected raw columns and frozen AE/GMM inputs."
        ),
    )
    parser.add_argument(
        "--fixed-ae-gmm-state-pkl",
        type=Path,
        default=DEFAULT_FIXED_AE_GMM_STATE_PKL,
        help=(
            "Persisted AE/GMM state artifact to reuse for generated AE/GMM features. "
            "When set, the runner does not refit the global AE/GMM state on the scoring fold."
        ),
    )
    parser.add_argument(
        "--frozen-ae-gmm-output-sidecar",
        type=Path,
        default=None,
        help=(
            "Precomputed parquet of frozen AE/GMM outputs. Its adjacent manifest "
            "must match the exact label-row signature and frozen-state SHA-256."
        ),
    )
    parser.add_argument(
        "--allow-refit-ae-gmm-with-fixed-features",
        action="store_true",
        help=(
            "Diagnostic-only escape hatch. Allows refitting AE/GMM while reusing a "
            "fixed selected-feature list that contains AE/GMM-generated columns. "
            "For frozen replay/inference parity, pass --fixed-ae-gmm-state-pkl instead."
        ),
    )
    parser.add_argument(
        "--existing-scored-ledger",
        type=Path,
        default=None,
        help="Existing scored ledger used to identify already-scored __ts__/__symbol__/side rows.",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Score only OOS rows absent from --existing-scored-ledger while keeping the full train window.",
    )
    parser.add_argument(
        "--rerun-hpo",
        action="store_true",
        help="Ignore --fixed-params-json for search control and run the baseline/Optuna HPO arms.",
    )
    parser.add_argument(
        "--rerun-ae-gmm-hpo",
        action="store_true",
        help=(
            "Refit and retune the cycle AE/GMM representation instead of reusing "
            "the promoted frozen state. This also clears the promoted selected-"
            "feature contract because generated AE/GMM columns are state-specific."
        ),
    )
    parser.add_argument(
        "--fresh-feature-selection",
        action="store_true",
        help=(
            "Ignore the promoted selected-feature CSV and rerun the configured "
            "feature-selection process for this model cycle."
        ),
    )
    parser.add_argument(
        "--refit-cycle-ae-gmm",
        action="store_true",
        help=(
            "Ignore the promoted AE/GMM state and fit one fresh cycle-reference "
            "state without labelling the run as an AE/GMM-HPO experiment."
        ),
    )
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument(
        "--ae-gmm-state-feature-max-train-rows",
        type=int,
        default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS,
        help="Rows used to fit the denoising AE state on the train-only reference sample.",
    )
    parser.add_argument(
        "--ae-gmm-state-feature-gmm-max-train-rows",
        type=int,
        default=DEFAULT_AE_GMM_STATE_FEATURE_GMM_MAX_TRAIN_ROWS,
        help="Latent train-only rows used to fit/HPO the GMM after the AE is frozen.",
    )
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER)
    parser.add_argument(
        "--refit-ae-gmm-per-window",
        action="store_true",
        help=(
            "Removed legacy mode. Passing this flag now fails because production uses "
            "one AE/GMM state per feature-selection/HPO cycle."
        ),
    )
    parser.add_argument(
        "--feature-selection-top-n",
        type=int,
        default=0,
        help=(
            "Legacy explicit selected-feature cap. Default 0 keeps the native MDA "
            "auto-count path; positive values are ignored unless "
            "--force-feature-selection-top-n is also set."
        ),
    )
    parser.add_argument(
        "--force-feature-selection-top-n",
        action="store_true",
        help="Honor --feature-selection-top-n as an explicit cap instead of MDA auto-count.",
    )
    parser.add_argument("--feature-selection-target-mode", choices=TARGET_MODES, default="time_decay_policy")
    parser.add_argument(
        "--feature-selection-method",
        choices=(
            "univariate",
            "mda",
            "mda_cum99_se075",
            "archetype_prescreen_side_mda",
            "archetype_prescreen_side_mda_corrfirst",
        ),
        default="mda",
        help=(
            "First-window feature selector. 'mda' restores the incumbent global "
            "top-k permutation pathway; 'archetype_prescreen_side_mda' runs "
            "archetype-aware univariate and Relief screens followed by unweighted "
            "per-side MDA across all archetypes; 'mda_cum99_se075' adds a "
            "chronological nested-subset 0.75-SE stability rule after the 99%% "
            "cumulative positive-MDA pool."
        ),
    )
    parser.add_argument(
        "--model-side-scope",
        choices=("shared", "per_side"),
        default="shared",
        help=(
            "Fit one shared base estimator or independent long/short estimators. "
            "Per-side mode consumes the independently selected side contracts."
        ),
    )
    parser.add_argument(
        "--max-oos-model-age-days",
        type=int,
        default=0,
        help="When positive, split each requested OOS month into windows no longer than this many days.",
    )
    parser.add_argument(
        "--single-fit-oos-window",
        action="store_true",
        help=(
            "Fit one base model before the first requested month and score the "
            "entire contiguous requested month range without growing-window refits. "
            "This is intended for controlled representation ablations."
        ),
    )
    parser.add_argument(
        "--train-window-days",
        type=int,
        default=0,
        help="When positive, train each OOS fold only on rows in [valid_start-N days, valid_start).",
    )
    parser.add_argument(
        "--label-path-purge-hours",
        type=float,
        default=25.0,
        help=(
            "Exclude train labels this many hours before each OOS boundary. "
            "The corrected S59 path uses a one-hour decision offset plus up to "
            "96 fifteen-minute path bars, so the canonical default is 25h."
        ),
    )
    parser.add_argument(
        "--ae-gmm-anchor-days",
        type=int,
        default=0,
        help=(
            "When positive with --train-window-days, fit the AE/GMM state on the N days "
            "immediately before the train window, then transform train/OOS rows with it."
        ),
    )
    parser.add_argument(
        "--ae-gmm-input-features-csv",
        type=Path,
        default=None,
        help="CSV/JSON feature list used only as AE/GMM state inputs; model columns remain controlled separately.",
    )
    parser.add_argument(
        "--external-feature-sidecar",
        type=Path,
        default=None,
        help=(
            "Optional immutable parquet of additional model features, unique by "
            "__ts__/__symbol__/side. Used by representation ablations without "
            "copying the label store."
        ),
    )
    parser.add_argument(
        "--save-fold-models",
        action="store_true",
        help="Persist each final OOS scoring fold's fitted base model plus columns.json and leakage manifest.",
    )
    parser.add_argument(
        "--save-final-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After OOS scoring, fit and persist a deployment model on all labelled "
            "rows using the frozen selected features, parameters, and AE/GMM state. "
            "This model is explicitly excluded from OOS metrics."
        ),
    )
    parser.add_argument(
        "--hpo-only",
        action="store_true",
        help=(
            "Persist the completed search and winning contract without fitting "
            "walk-forward/final models. Use the saved winner in a separate full run."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixed_selected_features_csv = args.fixed_selected_features_csv
    fixed_ae_gmm_state_pkl = args.fixed_ae_gmm_state_pkl
    if bool(args.fresh_feature_selection):
        fixed_selected_features_csv = None
    if bool(args.refit_cycle_ae_gmm):
        fixed_ae_gmm_state_pkl = None
    if bool(args.rerun_ae_gmm_hpo):
        fixed_selected_features_csv = None
        fixed_ae_gmm_state_pkl = None
        print(
            "[ae_gmm] explicit HPO requested; ignoring promoted frozen state and "
            "selected-feature contract",
            flush=True,
        )
    feature_selection_top_n = int(args.feature_selection_top_n)
    if feature_selection_top_n > 0 and not bool(args.force_feature_selection_top_n):
        print(
            "[feature_selection] ignoring explicit --feature-selection-top-n="
            f"{feature_selection_top_n}; using MDA auto-count. Pass "
            "--force-feature-selection-top-n to cap intentionally.",
            flush=True,
        )
        feature_selection_top_n = 0
    manifest = run_hpo(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, ()),
        max_feature_store_features=args.max_feature_store_features,
        max_train_rows=int(args.max_train_rows),
        feature_selection_sample_rows=int(args.feature_selection_sample_rows),
        hpo_max_train_rows=int(args.hpo_max_train_rows),
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_gmm_max_train_rows=int(args.ae_gmm_state_feature_gmm_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
        feature_selection_top_n=int(feature_selection_top_n),
        feature_selection_target_mode=str(args.feature_selection_target_mode),
        feature_selection_method=str(args.feature_selection_method),
        model_side_scope=str(args.model_side_scope),
        max_oos_model_age_days=int(args.max_oos_model_age_days),
        single_fit_oos_window=bool(args.single_fit_oos_window),
        train_window_days=int(args.train_window_days),
        label_path_purge_hours=float(args.label_path_purge_hours),
        ae_gmm_anchor_days=int(args.ae_gmm_anchor_days),
        ae_gmm_input_features_csv=args.ae_gmm_input_features_csv,
        fixed_params_json=args.fixed_params_json if str(args.fixed_params_json).strip() else None,
        fixed_selected_features_csv=fixed_selected_features_csv,
        fixed_ae_gmm_state_pkl=fixed_ae_gmm_state_pkl,
        allow_refit_ae_gmm_with_fixed_features=bool(args.allow_refit_ae_gmm_with_fixed_features),
        refit_ae_gmm_per_window=bool(args.refit_ae_gmm_per_window),
        existing_scored_ledger_path=args.existing_scored_ledger,
        missing_only=bool(args.missing_only),
        rerun_hpo=bool(args.rerun_hpo),
        rerun_ae_gmm_hpo=bool(args.rerun_ae_gmm_hpo),
        fresh_feature_selection_requested=bool(args.fresh_feature_selection),
        refit_cycle_ae_gmm_requested=bool(args.refit_cycle_ae_gmm),
        save_fold_models=bool(args.save_fold_models),
        save_final_model=bool(args.save_final_model),
        two_phase_wide_feature_selection=not bool(
            args.single_phase_wide_feature_selection
        ),
        hpo_only=bool(args.hpo_only),
        external_feature_sidecar_path=args.external_feature_sidecar,
        target_sidecar_path=args.target_sidecar,
        frozen_ae_gmm_output_sidecar_path=args.frozen_ae_gmm_output_sidecar,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
