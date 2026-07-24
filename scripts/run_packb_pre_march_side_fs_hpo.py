#!/usr/bin/env python3
"""Run strict long/short Pack-B feature selection and 150-trial HPO.

This production adapter binds the fixed pre-March population, the side-local
raw feature contracts frozen by ``run_packb_pre_march_side_ae.py``, and the
canonical label inventory to ``packb_side_local_fs_hpo_stage``.  It deliberately
fits long then short so only one side's matrices and models are resident.

Feature selection is supervised but fixed-calendar.  It reuses the recent
winning Pack-B selector family independently for each side:

* pre-November fit rows and the fixed November selector rows are the only
  supervised selector population;
* correlation-first pruning retains the recent-winner 300-feature floor;
* archetype-aware univariate and Relief screens precede iterative MDA; and
* MDA and automatic stopping are fitted independently for long and short.

HPO evaluates 150 explicit, deterministic LightGBM arms on December, January,
and February.  The objective combines rank IC, top-decile executable net-return
lift, and weighted soft-target error.  No post-cutoff row, shared-side selector,
shared-side parameter study, or default fallback is available.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import packb_side_stage_manifest as stage_manifest
from extreme_price_movements.features_gmm_ae import (
    AE_GMM_LATENT_DIM,
    ae_gmm_input_feature_order_hash,
    ae_gmm_learned_transform_hash,
    transform_ae_gmm_features,
)
from extreme_price_movements.packb_side_local_fs_hpo_stage import (
    FeatureSelectionInput,
    HPOFoldInput,
    HPOFoldLedger,
    HPOTrial,
    HPOTrialEvaluation,
    fit_side_local_fs_hpo_stages,
)
from extreme_price_movements.packb_static_point_feature_loader import (
    LoaderEvidenceBundle,
    _feature_contract_digest,
    load_point_in_time_features,
    make_packb_static_feature_loader,
)
from extreme_price_movements.training_resource_guard import (
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts.audit_full_pipeline_migration import hash_path
from scripts.prepare_packb_pre_march_side_contracts import parse_locked_dec09
from scripts.run_packb_pre_march_side_ae import (
    DEFAULT_DECISIONS,
    DEFAULT_FEATURE_INVENTORY,
    DEFAULT_FEATURE_STORE,
    DEFAULT_POPULATION_ROOT,
    _feature_inventory_binding,
    _source_contracts,
)

DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_LABELS = (
    ROOT / "data_perp/artifacts/"
    "20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/packb_side_local_fs_hpo_20260724_v2"
DEFAULT_TRIALS = 150
POINT_LOADER_MAX_ROWS_PER_BATCH = 2_048
POINT_LOADER_MAX_COLUMNS_PER_READ = 64
POINT_LOADER_MAX_OUTPUT_BYTES = 512 * 1024**2
TARGET_COLUMN = "__first_touch_target_soft__"
WEIGHT_COLUMN = "__w__"
ECONOMIC_COLUMN = "__first_touch_capture_net__"
ARCHETYPE_COLUMN = "__archetype_label_family__"
NET_POSITIVE_COLUMN = "__first_touch_net_positive__"
MAE_TO_SL_COLUMN = "__first_touch_mae_to_sl__"
TIMEOUT_COLUMN = "__first_touch_timeout__"
RECENT_WINNER_SELECTOR_CONTRACT = (
    "packb_pre_march_recent_winner_archetype_prescreen_side_mda_corrfirst_v1"
)
# The reference selector used a one-year base burn-in because its fit history
# exceeded one year.  This production stage is intentionally locked to the
# legal Jan-Nov pre-March population, so six months is the longest simple
# half-year burn-in that leaves a substantial, strictly later validation tail.
# Short-history fallback remains disabled.
RECENT_WINNER_SELECTOR_FORWARD_BURN_IN_DAYS = 180.0
RECENT_WINNER_PROCESS_MANIFEST = (
    ROOT / "data_perp/reports/"
    "s59_h5_signalclose_causal_stagec_packb_sliding365_wf30_20260721_v1/"
    "manifest.json"
)
RECENT_WINNER_FEATURE_CONTRACT = (
    ROOT / "data_perp/reports/"
    "weighted_packb_july_frozen_oos_scoring_validation_20260721_v1/"
    "base_fold_models/columns.json"
)
LABEL_IDENTITY_COLUMNS = (
    "candidate_id",
    "side_name",
    "__symbol__",
    "__ts__",
)


class PackBSideFSHPORunnerError(RuntimeError):
    """Raised when production FS/HPO evidence cannot be proven."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackBSideFSHPORunnerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackBSideFSHPORunnerError(f"JSON object required: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_revision() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PackBSideFSHPORunnerError("cannot resolve source revision") from exc
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise PackBSideFSHPORunnerError("source revision is not a full Git SHA")
    if dirty:
        raise PackBSideFSHPORunnerError(
            "production FS/HPO requires a clean tracked source revision"
        )
    return revision


def _release_memory() -> None:
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def _canonical_label_files(
    labels_dir: Path, population_manifest: Mapping[str, Any]
) -> tuple[Path, ...]:
    names = population_manifest.get("input", {}).get("canonical_shards")
    if not isinstance(names, list) or not names:
        raise PackBSideFSHPORunnerError(
            "population manifest has no canonical label inventory"
        )
    paths = tuple(Path(labels_dir) / str(name) for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PackBSideFSHPORunnerError(
            "canonical label inventory is incomplete: " + ", ".join(missing[:4])
        )
    actual_names = {path.name for path in Path(labels_dir).glob("*.parquet")}
    explicitly_excluded = set(
        population_manifest.get("population_preflight", {})
        .get("label_inventory", {})
        .get("explicitly_excluded_shards", [])
    )
    extras = sorted(actual_names - set(map(str, names)) - explicitly_excluded)
    if extras:
        raise PackBSideFSHPORunnerError(
            "unlisted label shards are forbidden: " + ", ".join(extras[:8])
        )
    audit_path = Path(
        str(population_manifest.get("input", {}).get("causal_audit_path") or "")
    )
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path
    expected_audit_hash = str(
        population_manifest.get("input", {}).get("causal_audit_sha256") or ""
    )
    if (
        not audit_path.is_file()
        or stage_manifest.sha256_file(audit_path) != expected_audit_hash
    ):
        raise PackBSideFSHPORunnerError(
            "bound causal label audit is missing or has changed"
        )
    audit = _json(audit_path)
    if audit.get("status") != "PASS" or audit.get("failures"):
        raise PackBSideFSHPORunnerError("bound causal label audit did not pass")
    audited_rows = {
        str(item["file"]): int(item["rows"])
        for item in audit.get("per_file", [])
        if isinstance(item, Mapping) and item.get("file") is not None
    }
    if set(audited_rows) != set(map(str, names)):
        raise PackBSideFSHPORunnerError(
            "causal audit and canonical label inventory disagree"
        )
    try:
        import pyarrow.parquet as pq

        for path in paths:
            parquet = pq.ParquetFile(path)
            if int(parquet.metadata.num_rows) != audited_rows[path.name]:
                raise PackBSideFSHPORunnerError(
                    f"label shard row count changed since audit: {path.name}"
                )
            missing_columns = sorted(
                {
                    *LABEL_IDENTITY_COLUMNS,
                    TARGET_COLUMN,
                    WEIGHT_COLUMN,
                    ECONOMIC_COLUMN,
                    ARCHETYPE_COLUMN,
                    NET_POSITIVE_COLUMN,
                    MAE_TO_SL_COLUMN,
                    TIMEOUT_COLUMN,
                }
                - set(parquet.schema.names)
            )
            if missing_columns:
                raise PackBSideFSHPORunnerError(
                    f"label shard {path.name} lacks columns: {missing_columns}"
                )
    except PackBSideFSHPORunnerError:
        raise
    except Exception as exc:
        raise PackBSideFSHPORunnerError(
            f"cannot validate canonical label metadata: {exc}"
        ) from exc
    return paths


class ExactLabelLoader:
    """Bounded exact-key label reader retaining only its most recent slice."""

    def __init__(
        self,
        files: Sequence[Path],
        *,
        resource_guard: TrainingResourceGuard | Any | None = None,
    ) -> None:
        self.files = tuple(str(Path(path)) for path in files)
        self.resource_guard = resource_guard
        self._key: tuple[str, ...] | None = None
        self._frame: pd.DataFrame | None = None

    def load(self, ledger: pd.DataFrame) -> pd.DataFrame:
        candidate_ids = tuple(ledger["candidate_id"].astype(str))
        if self._key == candidate_ids and self._frame is not None:
            return self._frame.copy()
        if self.resource_guard is not None:
            self.resource_guard.checkpoint("packb_side_fs_hpo:before_label_join")
        requested = ledger.loc[
            :, ["candidate_id", "side_name", "__symbol__", "__ts__"]
        ].copy()
        requested["__order__"] = np.arange(len(requested), dtype=np.int64)
        requested["__ts__"] = pd.to_datetime(
            requested["__ts__"], utc=True, errors="raise"
        )
        try:
            import duckdb

            connection = duckdb.connect(database=":memory:")
            try:
                connection.register("requested", requested)
                frame = connection.execute(
                    """
                    SELECT
                        r.__order__,
                        l.candidate_id,
                        l.side_name,
                        l.__symbol__,
                        l.__decision_ts__,
                        l.__first_touch_target_soft__,
                        l.__w__,
                        l.__first_touch_capture_net__,
                        l.__archetype_label_family__,
                        l.__first_touch_net_positive__,
                        l.__first_touch_mae_to_sl__,
                        l.__first_touch_timeout__
                    FROM requested AS r
                    INNER JOIN read_parquet(?, union_by_name=true) AS l
                    USING (candidate_id)
                    ORDER BY r.__order__
                    """,
                    [list(self.files)],
                ).fetchdf()
            finally:
                connection.close()
        except Exception as exc:
            raise PackBSideFSHPORunnerError(
                f"cannot exact-join labels for {len(requested):,} rows: {exc}"
            ) from exc
        if len(frame) != len(requested):
            raise PackBSideFSHPORunnerError(
                "label join is not one-to-one and complete "
                f"(requested={len(requested):,}, joined={len(frame):,})"
            )
        observed_decision = pd.to_datetime(
            frame["__decision_ts__"], utc=True, errors="coerce"
        )
        expected_decision = requested["__ts__"].reset_index(drop=True) + pd.Timedelta(
            hours=1
        )
        exact = (
            frame["candidate_id"]
            .astype(str)
            .reset_index(drop=True)
            .eq(requested["candidate_id"].astype(str).reset_index(drop=True))
            & frame["side_name"]
            .astype(str)
            .str.lower()
            .reset_index(drop=True)
            .eq(requested["side_name"].astype(str).str.lower().reset_index(drop=True))
            & frame["__symbol__"]
            .astype(str)
            .reset_index(drop=True)
            .eq(requested["__symbol__"].astype(str).reset_index(drop=True))
            & observed_decision.reset_index(drop=True).eq(expected_decision)
        )
        if not exact.all():
            raise PackBSideFSHPORunnerError(
                "candidate_id label join disagrees on side, symbol, or signal timestamp"
            )
        for column in (
            TARGET_COLUMN,
            WEIGHT_COLUMN,
            ECONOMIC_COLUMN,
            NET_POSITIVE_COLUMN,
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
                raise PackBSideFSHPORunnerError(
                    f"label column {column!r} contains non-finite values"
                )
        # These are optional selector context diagnostics, not training
        # targets.  The recent winning selector's adapter used
        # ``_safe_numeric(...).fillna(0.0)`` for both fields.
        for column in (MAE_TO_SL_COLUMN, TIMEOUT_COLUMN):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
                raise PackBSideFSHPORunnerError(
                    f"selector context column {column!r} contains infinite values"
                )
            frame[column] = numeric.fillna(0.0)
        if (frame[WEIGHT_COLUMN] < 0.0).any() or frame[WEIGHT_COLUMN].sum() <= 0.0:
            raise PackBSideFSHPORunnerError("label weights must be non-negative")
        archetype = frame[ARCHETYPE_COLUMN].astype("string").str.strip()
        if archetype.isna().any() or archetype.eq("").any():
            raise PackBSideFSHPORunnerError(
                "feature-selection archetype labels must be complete"
            )
        frame[ARCHETYPE_COLUMN] = archetype.astype(str)
        keep = frame.loc[
            :,
            [
                TARGET_COLUMN,
                WEIGHT_COLUMN,
                ECONOMIC_COLUMN,
                ARCHETYPE_COLUMN,
                NET_POSITIVE_COLUMN,
                MAE_TO_SL_COLUMN,
                TIMEOUT_COLUMN,
            ],
        ].reset_index(drop=True)
        self._key = candidate_ids
        self._frame = keep
        return keep.copy()

    def target(self, ledger: pd.DataFrame) -> pd.Series:
        return self.load(ledger)[TARGET_COLUMN].copy()

    def weights(self, ledger: pd.DataFrame, _target: pd.Series) -> pd.Series:
        return self.load(ledger)[WEIGHT_COLUMN].copy()

    def economic(self, ledger: pd.DataFrame) -> np.ndarray:
        return self.load(ledger)[ECONOMIC_COLUMN].to_numpy(dtype=np.float64, copy=True)

    def selection_context(self, ledger: pd.DataFrame) -> pd.DataFrame:
        return (
            self.load(ledger)
            .loc[
                :,
                [
                    ARCHETYPE_COLUMN,
                    NET_POSITIVE_COLUMN,
                    MAE_TO_SL_COLUMN,
                    TIMEOUT_COLUMN,
                    ECONOMIC_COLUMN,
                ],
            ]
            .copy()
        )


def _active_ae_gmm_columns(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the non-temporal generated representation contract."""

    components = int(state.get("gmm_n_components", 0) or 0)
    if components < 1:
        raise PackBSideFSHPORunnerError("AE/GMM state has no fitted components")
    return tuple(
        [
            *(f"dae_b16_{index:02d}" for index in range(AE_GMM_LATENT_DIM)),
            *(f"gmm_cluster_posterior_{index}" for index in range(components)),
            *(f"gmm_dist_center_{index}" for index in range(components)),
            *(f"gmm_mahal_{index}" for index in range(components)),
            "gmm_cluster_id",
            "gmm_posterior_max",
            "gmm_posterior_margin",
            "gmm_unknown_probability",
            "gmm_ood_score",
            "gmm_entropy",
            "cluster_entropy_norm",
            "mahalanobis_distance",
            "expected_mahalanobis",
            "dae_reconstruction_error",
            "dae_reconstruction_error_zscore",
        ]
    )


def _load_side_ae_state(
    state_path: Path,
    *,
    expected_side: str,
    expected_sha256: str,
    raw_features: Sequence[str],
) -> dict[str, Any]:
    if (
        not state_path.is_file()
        or stage_manifest.sha256_file(state_path) != expected_sha256
    ):
        raise PackBSideFSHPORunnerError(
            f"{expected_side} AE/GMM state is missing or changed"
        )
    try:
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
    except Exception as exc:
        raise PackBSideFSHPORunnerError(
            f"cannot load {expected_side} AE/GMM state: {exc}"
        ) from exc
    if (
        not isinstance(state, dict)
        or not bool(state.get("enabled", False))
        or state.get("packb_side_scope") != expected_side
        or state.get("representation_selection_outcome_free") is not True
        or state.get("temporal_feature_contract") != "row_independent_v1"
    ):
        raise PackBSideFSHPORunnerError(
            f"{expected_side} AE/GMM state violates its outcome-free side contract"
        )
    feature_columns = tuple(map(str, state.get("feature_columns", [])))
    if feature_columns != tuple(map(str, raw_features)):
        raise PackBSideFSHPORunnerError(
            f"{expected_side} AE/GMM input features differ from the raw contract"
        )
    expected_input_hash = ae_gmm_input_feature_order_hash(feature_columns)
    if state.get("input_feature_order_hash") != expected_input_hash:
        raise PackBSideFSHPORunnerError(
            f"{expected_side} AE/GMM input-order hash is invalid"
        )
    if state.get("cycle_state_hash") != ae_gmm_learned_transform_hash(state):
        raise PackBSideFSHPORunnerError(
            f"{expected_side} AE/GMM learned-transform hash is invalid"
        )
    _active_ae_gmm_columns(state)
    return state


class SideRepresentationFeatureLoader:
    """Append frozen side-local AE/GMM outputs to exact raw point features."""

    def __init__(
        self,
        *,
        raw_loader: Any,
        raw_features: Sequence[str],
        state: Mapping[str, Any],
        generated_features: Sequence[str],
    ) -> None:
        self.raw_loader = raw_loader
        self.raw_features = tuple(map(str, raw_features))
        self.raw_set = set(self.raw_features)
        self.state = dict(state)
        self.generated_features = tuple(map(str, generated_features))
        self.generated_set = set(self.generated_features)
        if self.raw_set.intersection(self.generated_set):
            raise PackBSideFSHPORunnerError(
                "raw and generated feature contracts overlap"
            )

    def __call__(
        self, ledger: pd.DataFrame, requested_features: Sequence[str]
    ) -> pd.DataFrame:
        requested = tuple(map(str, requested_features))
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(
                feature not in self.raw_set and feature not in self.generated_set
                for feature in requested
            )
        ):
            raise PackBSideFSHPORunnerError(
                "representation loader received an invalid feature subset"
            )
        requested_generated = [
            feature for feature in requested if feature in self.generated_set
        ]
        raw_request = (
            self.raw_features
            if requested_generated
            else tuple(feature for feature in requested if feature in self.raw_set)
        )
        raw = self.raw_loader(ledger, list(raw_request))
        if not requested_generated:
            return raw.loc[:, list(requested)].reset_index(drop=True)
        raw_contract = raw.loc[:, list(self.raw_features)]
        complete = np.isfinite(raw_contract.to_numpy(dtype=np.float32, copy=False)).all(
            axis=1
        )
        generated = pd.DataFrame(
            np.nan,
            index=raw.index,
            columns=requested_generated,
            dtype=np.float32,
        )
        if complete.any():
            transformed = transform_ae_gmm_features(
                raw_contract.loc[complete],
                self.state,
                index=raw.index[complete],
            ).loc[:, requested_generated]
            generated.loc[complete, requested_generated] = transformed.to_numpy(
                dtype=np.float32,
                copy=False,
            )
        joined = pd.concat([raw, generated], axis=1, copy=False)
        return joined.loc[:, list(requested)].reset_index(drop=True)


def _derived_raw_subset_contract(
    parent_contract: Mapping[str, Any],
    requested_features: Sequence[str],
) -> dict[str, Any]:
    """Derive a content-validated raw subset without changing feature semantics."""

    parent_features = tuple(map(str, parent_contract["feature_columns"]))
    parent_set = frozenset(parent_features)
    requested = tuple(map(str, requested_features))
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(feature not in parent_set for feature in requested)
    ):
        raise PackBSideFSHPORunnerError(
            "raw subset loader received an invalid feature subset"
        )
    subset = sorted(requested)
    result = dict(parent_contract)
    result["feature_columns"] = subset
    result["feature_contract_sha256"] = _feature_contract_digest(
        feature_columns=subset,
        candidate_universe_sha256=str(result["candidate_universe_sha256"]),
        source_schema_sha256=str(result["source_schema_sha256"]),
        raw_allowlist_sha256=str(result["raw_allowlist_sha256"]),
        generator_registry_sha256=str(result["generator_registry_sha256"]),
        store_scan_manifest_sha256=str(result["store_scan_manifest_sha256"]),
        coverage_profile_sha256=(
            str(result["coverage_profile_sha256"])
            if result.get("coverage_profile_sha256") is not None
            else None
        ),
        min_exact_key_coverage=float(result["min_exact_key_coverage"]),
        min_non_null_feature_coverage=float(result["min_non_null_feature_coverage"]),
        max_feature_columns=(
            int(result["max_feature_columns"])
            if result.get("max_feature_columns") is not None
            else None
        ),
        coverage_admission_rejections=[
            (str(item[0]), str(item[1]))
            for item in result.get("coverage_admission_rejections", [])
        ],
    )
    return result


def make_fs_hpo_raw_feature_loader(
    *,
    feature_store_dir: Path,
    feature_contract: Mapping[str, Any],
    evidence_bundle: LoaderEvidenceBundle,
    resource_guard: TrainingResourceGuard,
):
    """Load raw-only HPO subsets narrowly while preserving the frozen parent.

    The canonical full-contract loader remains the sole path whenever the
    requested columns equal the AE input surface.  This is required for every
    generated AE/GMM feature.  Once feature selection freezes a raw-only
    subset, a derived contract reads only those columns and then restores the
    caller's order.  Exact keys, causal registry hashes, coverage policy, dtype,
    missingness policy, and the final dataset hash are unchanged.
    """

    parent = dict(feature_contract)
    parent_features = tuple(map(str, parent["feature_columns"]))
    parent_set = frozenset(parent_features)
    canonical_loader = make_packb_static_feature_loader(
        feature_store_dir=feature_store_dir,
        feature_contract=parent,
        max_rows_per_batch=POINT_LOADER_MAX_ROWS_PER_BATCH,
        max_columns_per_read=POINT_LOADER_MAX_COLUMNS_PER_READ,
        max_output_bytes=POINT_LOADER_MAX_OUTPUT_BYTES,
        evidence_bundle=evidence_bundle,
        resource_guard=resource_guard,
    )
    subset_contracts: dict[tuple[str, ...], dict[str, Any]] = {}

    def _loader(ledger: pd.DataFrame, input_features: Sequence[str]) -> pd.DataFrame:
        requested = tuple(map(str, input_features))
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(feature not in parent_set for feature in requested)
        ):
            raise PackBSideFSHPORunnerError(
                "raw subset loader received an invalid feature subset"
            )
        if requested == parent_features:
            return canonical_loader(ledger, requested)
        contract_key = tuple(sorted(requested))
        subset_contract = subset_contracts.get(contract_key)
        if subset_contract is None:
            subset_contract = _derived_raw_subset_contract(parent, contract_key)
            subset_contracts[contract_key] = subset_contract
        matrix = load_point_in_time_features(
            ledger,
            feature_store_dir=feature_store_dir,
            feature_contract=subset_contract,
            max_rows_per_batch=POINT_LOADER_MAX_ROWS_PER_BATCH,
            max_columns_per_read=POINT_LOADER_MAX_COLUMNS_PER_READ,
            max_output_bytes=POINT_LOADER_MAX_OUTPUT_BYTES,
            resource_guard=resource_guard,
        )
        return matrix.loc[:, list(requested)]

    _loader.fs_hpo_subset_loading_contract_sha256 = _canonical_sha256(
        {
            "schema": "packb_fs_hpo_raw_subset_loading_v1",
            "parent_feature_contract_sha256": parent["feature_contract_sha256"],
            "policy": ("full_parent_for_ae_gmm_inputs_else_sorted_derived_raw_subset"),
            "exact_join": "__symbol__+__ts__",
            "max_rows_per_batch": POINT_LOADER_MAX_ROWS_PER_BATCH,
            "max_columns_per_read": POINT_LOADER_MAX_COLUMNS_PER_READ,
            "max_output_bytes": POINT_LOADER_MAX_OUTPUT_BYTES,
            "imputation": "forbidden_joint_complete_rows_only",
        }
    )
    return _loader


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total > 0.0 else float("nan")


def _weighted_correlation(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    left_mean = _weighted_mean(left, weights)
    right_mean = _weighted_mean(right, weights)
    left_centered = left - left_mean
    right_centered = right - right_mean
    covariance = float(np.dot(weights, left_centered * right_centered) / total)
    left_var = float(np.dot(weights, left_centered**2) / total)
    right_var = float(np.dot(weights, right_centered**2) / total)
    denominator = math.sqrt(max(left_var * right_var, 0.0))
    return covariance / denominator if denominator > 1e-15 else 0.0


def _weighted_rank_ic(
    predictions: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> float:
    from scipy.stats import rankdata

    return _weighted_correlation(
        rankdata(predictions, method="average"),
        rankdata(target, method="average"),
        weights,
    )


def _top_fraction_indices(
    predictions: np.ndarray,
    *,
    timestamps: Sequence[Any] | None,
    symbols: Sequence[Any] | None,
    fraction: float = 0.10,
) -> np.ndarray:
    """Select the exact per-timestamp side-local fraction with lexical ties."""

    predictions = np.asarray(predictions, dtype=np.float64)
    if timestamps is None:
        timestamp_values = pd.Series(
            pd.Timestamp("1970-01-01T00:00:00Z"), index=range(len(predictions))
        )
    else:
        timestamp_values = pd.to_datetime(
            pd.Series(timestamps).reset_index(drop=True),
            utc=True,
            errors="coerce",
        )
    if timestamp_values.isna().any():
        raise PackBSideFSHPORunnerError(
            "economic objective received invalid ranking timestamps"
        )
    symbol_values = (
        np.arange(len(predictions)).astype(str)
        if symbols is None
        else np.asarray(symbols, dtype=str)
    )
    if len(timestamp_values) != len(predictions) or len(symbol_values) != len(
        predictions
    ):
        raise PackBSideFSHPORunnerError(
            "economic ranking timestamps/symbols are not aligned"
        )
    selected: list[np.ndarray] = []
    encoded = timestamp_values.astype("int64").to_numpy()
    for timestamp in np.unique(encoded):
        rows = np.flatnonzero(encoded == timestamp)
        count = max(1, int(math.ceil(float(fraction) * len(rows))))
        order = np.lexsort((symbol_values[rows], -predictions[rows]))
        selected.append(rows[order[:count]])
    return (
        np.concatenate(selected).astype(np.int64, copy=False)
        if selected
        else np.asarray([], dtype=np.int64)
    )


def _economic_objective(
    predictions: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    net_return: np.ndarray,
    *,
    timestamps: Sequence[Any] | None = None,
    symbols: Sequence[Any] | None = None,
) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    net_return = np.asarray(net_return, dtype=np.float64)
    rank_ic = _weighted_rank_ic(predictions, target, weights)
    residual = predictions - target
    rmse = math.sqrt(_weighted_mean(residual**2, weights))
    target_mean = _weighted_mean(target, weights)
    baseline_rmse = math.sqrt(_weighted_mean((target - target_mean) ** 2, weights))
    top = _top_fraction_indices(
        predictions,
        timestamps=timestamps,
        symbols=symbols,
        fraction=0.10,
    )
    top10_net = float(np.mean(net_return[top]))
    overall_net = float(np.mean(net_return))
    net_lift = top10_net - overall_net
    error_gain = (baseline_rmse - rmse) / max(baseline_rmse, 1e-9)
    objective = (
        0.45 * rank_ic
        + 0.35 * math.tanh(net_lift / 0.01)
        + 0.20 * math.tanh(error_gain / 0.10)
    )
    return {
        "objective": float(objective),
        "weighted_rank_ic": float(rank_ic),
        "weighted_rmse": float(rmse),
        "weighted_baseline_rmse": float(baseline_rmse),
        "relative_rmse_gain": float(error_gain),
        "top10_mean_net_return": float(top10_net),
        "overall_mean_net_return": float(overall_net),
        "top10_net_return_lift": float(net_lift),
        "top10_rows": int(len(top)),
        "ranking_scope": "within_utc_timestamp_and_side",
        "ranking_tie_break": "score_desc_symbol_asc",
    }


def _lgbm_regressor(params: Mapping[str, Any], *, seed: int):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise PackBSideFSHPORunnerError("LightGBM is required for Pack-B") from exc
    return LGBMRegressor(
        objective="regression",
        verbosity=-1,
        n_jobs=1,
        random_state=int(seed),
        deterministic=True,
        force_col_wise=True,
        **dict(params),
    )


def _fit_predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_weight: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    valid_weight: pd.Series,
    params: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Any, np.ndarray, int]:
    import lightgbm as lgb

    model = _lgbm_regressor(params, seed=seed)
    model.fit(
        train_x,
        train_y,
        sample_weight=train_weight,
        eval_set=[(valid_x, valid_y)],
        eval_sample_weight=[valid_weight],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(model.best_iteration_ or params["n_estimators"])
    prediction = model.predict(valid_x, num_iteration=best_iteration)
    return model, np.asarray(prediction, dtype=np.float64), best_iteration


def _normalise_scores(values: Mapping[str, float]) -> dict[str, float]:
    series = (
        pd.Series(values, dtype=np.float64)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if series.empty:
        return {}
    low = float(series.min())
    high = float(series.max())
    if high <= low:
        return {str(key): 0.0 for key in series.index}
    return {
        str(key): float((value - low) / (high - low)) for key, value in series.items()
    }


def _regression_relief_scores(
    features: pd.DataFrame, target: pd.Series, *, max_rows: int = 2_500
) -> dict[str, float]:
    from sklearn.neighbors import NearestNeighbors

    rows = min(len(features), int(max_rows))
    positions = np.linspace(0, len(features) - 1, num=rows, dtype=np.int64)
    values = features.iloc[positions].to_numpy(dtype=np.float32, copy=True)
    y = target.iloc[positions].to_numpy(dtype=np.float32, copy=True)
    median = np.median(values, axis=0)
    scale = np.subtract(*np.percentile(values, [75.0, 25.0], axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    values = np.clip((values - median) / scale, -8.0, 8.0)
    neighbors = min(11, len(values))
    indices = (
        NearestNeighbors(n_neighbors=neighbors, n_jobs=1)
        .fit(values)
        .kneighbors(values, return_distance=False)[:, 1:]
    )
    if indices.shape[1] == 0:
        return {column: 0.0 for column in features.columns}
    row = np.repeat(np.arange(len(values)), indices.shape[1])
    near = indices.reshape(-1)
    target_distance = np.abs(y[row] - y[near]).astype(np.float64)
    target_distance -= target_distance.mean()
    target_scale = float(np.sqrt(np.dot(target_distance, target_distance)))
    scores: dict[str, float] = {}
    for index, column in enumerate(features.columns):
        distance = np.abs(values[row, index] - values[near, index]).astype(np.float64)
        distance -= distance.mean()
        denominator = target_scale * float(np.sqrt(np.dot(distance, distance)))
        scores[str(column)] = (
            float(np.dot(distance, target_distance) / denominator)
            if denominator > 1e-12
            else 0.0
        )
    return scores


def _screen_features(
    train_x: pd.DataFrame, train_y: pd.Series, *, seed: int
) -> tuple[list[str], dict[str, Any]]:
    from sklearn.feature_selection import mutual_info_regression

    correlations: dict[str, float] = {}
    for column in train_x.columns:
        raw_correlation = pd.Series(train_x[column], copy=False).corr(
            train_y, method="spearman"
        )
        correlations[str(column)] = (
            abs(float(raw_correlation))
            if pd.notna(raw_correlation) and np.isfinite(raw_correlation)
            else 0.0
        )
    mi_rows = min(len(train_x), 20_000)
    positions = np.linspace(0, len(train_x) - 1, num=mi_rows, dtype=np.int64)
    mi_values = mutual_info_regression(
        train_x.iloc[positions],
        train_y.iloc[positions],
        random_state=int(seed),
        n_neighbors=5,
    )
    mutual_information = {
        str(column): float(mi_values[index])
        for index, column in enumerate(train_x.columns)
    }
    relief = _regression_relief_scores(train_x, train_y)
    top_univariate = sorted(correlations, key=lambda x: (-correlations[x], x))[:64]
    top_mi = sorted(mutual_information, key=lambda x: (-mutual_information[x], x))[:48]
    top_relief = sorted(relief, key=lambda x: (-relief[x], x))[:48]
    union = list(dict.fromkeys([*top_univariate, *top_mi, *top_relief]))
    norm_corr = _normalise_scores(correlations)
    norm_mi = _normalise_scores(mutual_information)
    norm_relief = _normalise_scores(relief)
    combined = {
        column: norm_corr.get(column, 0.0)
        + norm_mi.get(column, 0.0)
        + norm_relief.get(column, 0.0)
        for column in union
    }
    ordered = sorted(union, key=lambda x: (-combined[x], x))
    correlation_matrix = train_x.loc[:, ordered].corr(method="spearman").abs()
    retained: list[str] = []
    rejected: list[dict[str, Any]] = []
    for column in ordered:
        conflict = next(
            (
                prior
                for prior in retained
                if float(correlation_matrix.loc[column, prior]) >= 0.95
            ),
            None,
        )
        if conflict is None:
            retained.append(column)
        else:
            rejected.append(
                {
                    "feature": column,
                    "correlated_with": conflict,
                    "absolute_spearman": float(
                        correlation_matrix.loc[column, conflict]
                    ),
                }
            )
    retained = retained[:96]
    return retained, {
        "univariate_abs_spearman": correlations,
        "mutual_information": mutual_information,
        "regression_relief": relief,
        "prescreen_union": union,
        "combined_order": ordered,
        "redundancy_threshold": 0.95,
        "redundancy_rejections": rejected,
        "mda_candidates": retained,
    }


class SideFeatureSelector:
    def __init__(
        self,
        *,
        side: str,
        labels: ExactLabelLoader,
        seed: int,
        resource_guard: TrainingResourceGuard | Any | None = None,
    ) -> None:
        self.side = side
        self.labels = labels
        self.seed = int(seed)
        self.resource_guard = resource_guard

    def __call__(self, value: FeatureSelectionInput) -> dict[str, Any]:
        if value.side != self.side:
            raise PackBSideFSHPORunnerError("feature selector received wrong side")
        if self.resource_guard is not None:
            self.resource_guard.checkpoint(
                f"packb_side_fs_hpo:{self.side}:feature_prescreen"
            )
        candidates, diagnostics = _screen_features(
            value.train.features, value.train.target, seed=self.seed
        )
        if len(candidates) < 8:
            raise PackBSideFSHPORunnerError(
                f"{self.side} feature prescreen retained fewer than eight features"
            )
        params = {
            "n_estimators": 500,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 80,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.10,
            "reg_lambda": 3.0,
            "min_split_gain": 0.0,
        }
        train_x = value.train.features.loc[:, candidates]
        valid_x = value.validation.features.loc[:, candidates]
        model, prediction, best_iteration = _fit_predict(
            train_x,
            value.train.target,
            value.train.weights,
            valid_x,
            value.validation.target,
            value.validation.weights,
            params,
            seed=self.seed,
        )
        net_return = self.labels.economic(value.validation.ledger)
        baseline = _economic_objective(
            prediction,
            value.validation.target.to_numpy(dtype=np.float64),
            value.validation.weights.to_numpy(dtype=np.float64),
            net_return,
            timestamps=value.validation.ledger["__ts__"],
            symbols=value.validation.ledger["__symbol__"],
        )
        rng = np.random.default_rng(self.seed)
        mda_rows: list[dict[str, Any]] = []
        repeats = 3
        for feature in candidates:
            drops: list[float] = []
            position = candidates.index(feature)
            for _repeat in range(repeats):
                permuted = valid_x.to_numpy(dtype=np.float32, copy=True)
                permuted[:, position] = permuted[
                    rng.permutation(len(permuted)), position
                ]
                permuted_prediction = model.predict(
                    permuted, num_iteration=best_iteration
                )
                score = _economic_objective(
                    permuted_prediction,
                    value.validation.target.to_numpy(dtype=np.float64),
                    value.validation.weights.to_numpy(dtype=np.float64),
                    net_return,
                    timestamps=value.validation.ledger["__ts__"],
                    symbols=value.validation.ledger["__symbol__"],
                )
                drops.append(float(baseline["objective"] - score["objective"]))
            mda_rows.append(
                {
                    "feature": feature,
                    "importance_mean": float(np.mean(drops)),
                    "importance_std": float(np.std(drops, ddof=1)),
                    "repeat_drops": drops,
                }
            )
            if self.resource_guard is not None:
                self.resource_guard.checkpoint(
                    f"packb_side_fs_hpo:{self.side}:mda:{feature}"
                )
        ranked = sorted(
            mda_rows,
            key=lambda row: (-row["importance_mean"], row["feature"]),
        )
        positive = [row for row in ranked if row["importance_mean"] > 0.0]
        if len(positive) < 8:
            raise PackBSideFSHPORunnerError(
                f"{self.side} has fewer than eight positive-MDA features"
            )
        positive_mass = sum(row["importance_mean"] for row in positive)
        cumulative = 0.0
        mass_count = len(positive)
        for index, row in enumerate(positive, start=1):
            cumulative += row["importance_mean"]
            if cumulative >= 0.99 * positive_mass:
                mass_count = index
                break
        max_prefix = min(64, len(positive))
        prefix_counts = sorted(
            {
                max(8, min(max_prefix, value))
                for value in (8, 12, 16, 24, 32, 48, 64, mass_count)
            }
        )
        prefix_rows: list[dict[str, Any]] = []
        for count in prefix_counts:
            columns = [row["feature"] for row in positive[:count]]
            _prefix_model, prefix_prediction, prefix_iteration = _fit_predict(
                value.train.features.loc[:, columns],
                value.train.target,
                value.train.weights,
                value.validation.features.loc[:, columns],
                value.validation.target,
                value.validation.weights,
                params,
                seed=self.seed + count,
            )
            metrics = _economic_objective(
                prefix_prediction,
                value.validation.target.to_numpy(dtype=np.float64),
                value.validation.weights.to_numpy(dtype=np.float64),
                net_return,
                timestamps=value.validation.ledger["__ts__"],
                symbols=value.validation.ledger["__symbol__"],
            )
            prefix_rows.append(
                {
                    "feature_count": int(count),
                    "features": columns,
                    "best_iteration": int(prefix_iteration),
                    **metrics,
                }
            )
        best_objective = max(row["objective"] for row in prefix_rows)
        tolerance = 0.005
        selected_row = min(
            (
                row
                for row in prefix_rows
                if row["objective"] >= best_objective - tolerance
            ),
            key=lambda row: row["feature_count"],
        )
        return {
            "side": self.side,
            "selected_features": list(selected_row["features"]),
            "selection_scope": "side_local",
            "fallback_used": False,
            "selection_methods": [
                "univariate",
                "mutual_information",
                "regression_relief",
                "correlation_redundancy_pruning",
                "mda",
                "mda_prefix_confirmation",
            ],
            "search_breadth": int(
                len(value.candidate_features)
                + len(mda_rows) * repeats
                + len(prefix_rows)
            ),
            "target_contract": {
                "column": TARGET_COLUMN,
                "economic_validation_column": ECONOMIC_COLUMN,
                "weight_column": WEIGHT_COLUMN,
            },
            "prescreen": diagnostics,
            "mda": {
                "baseline": baseline,
                "repeats": repeats,
                "rows": ranked,
                "positive_importance_mass": float(positive_mass),
                "positive_mass_99pct_count": int(mass_count),
            },
            "prefix_confirmation": {
                "tolerance_from_best_objective": tolerance,
                "evaluations": prefix_rows,
                "selected_feature_count": int(selected_row["feature_count"]),
                "best_objective": float(best_objective),
            },
        }


class RecentWinnerSideFeatureSelector:
    """Re-run the selector family that produced the recent 55/37 Pack-B pair.

    The historical fitted feature lists are not reused: they were selected
    after the DEC-09 cutoff.  Only the process is reused, on the fixed
    pre-March population and the matching side-local AE/GMM representation.
    Each callback contains one side only, so every prescreen, redundancy
    decision, MDA fit, and automatic stop is side-local by construction.
    """

    _MODEL_PARAMS = {
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

    def __init__(
        self,
        *,
        side: str,
        labels: ExactLabelLoader,
        seed: int,
        resource_guard: TrainingResourceGuard | Any | None = None,
    ) -> None:
        self.side = str(side)
        self.labels = labels
        self.seed = int(seed)
        self.resource_guard = resource_guard

    @staticmethod
    def _finite_metric_snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
        keep = (
            "J_final",
            "J_base",
            "J_meta",
            "feature_selection_candidate_count",
            "feature_selection_selected_count",
            "feature_selection_cluster_count",
            "archetype_univariate_prescreen_enabled",
            "archetype_relief_prescreen_enabled",
            "correlation_pruning_before_prescreen",
            "per_side_feature_selection_reason",
        )
        result: dict[str, Any] = {}
        for key in keep:
            value = metrics.get(key)
            if isinstance(value, (bool, str, int)):
                result[key] = value
            elif isinstance(value, (float, np.floating)) and np.isfinite(value):
                result[key] = float(value)
        return result

    @staticmethod
    def _feature_stats_snapshot(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, pd.DataFrame) or value.empty:
            return []
        columns = [
            column
            for column in (
                "feature",
                "feature_score",
                "mda_mean",
                "mda_std",
                "hard_drop",
                "selected_by_sides",
                "selection_side",
            )
            if column in value.columns
        ]
        rows: list[dict[str, Any]] = []
        for record in value.loc[:, columns].to_dict(orient="records"):
            clean: dict[str, Any] = {}
            for key, raw in record.items():
                if pd.isna(raw):
                    continue
                if isinstance(raw, (bool, np.bool_)):
                    clean[str(key)] = bool(raw)
                elif isinstance(raw, (int, np.integer)):
                    clean[str(key)] = int(raw)
                elif isinstance(raw, (float, np.floating)):
                    if np.isfinite(raw):
                        clean[str(key)] = float(raw)
                else:
                    clean[str(key)] = str(raw)
            rows.append(clean)
        return rows

    def __call__(self, value: FeatureSelectionInput) -> dict[str, Any]:
        if value.side != self.side:
            raise PackBSideFSHPORunnerError(
                "recent-winner selector received the wrong side"
            )
        if self.resource_guard is not None:
            self.resource_guard.checkpoint(
                f"packb_side_fs_hpo:{self.side}:recent_winner_selector_start"
            )

        # The fixed November cohort is the declared selector-validation
        # population.  Joining it to the legal pre-November fit rows gives the
        # old selector its internal selection/evaluation split without making
        # any December-or-later label available.
        features = pd.concat(
            [value.train.features, value.validation.features],
            ignore_index=True,
            copy=False,
        ).loc[:, list(value.candidate_features)]
        target = pd.concat(
            [value.train.target, value.validation.target],
            ignore_index=True,
        ).to_numpy(dtype=np.float32, copy=False)
        ledger = pd.concat(
            [value.train.ledger, value.validation.ledger],
            ignore_index=True,
            copy=False,
        )
        context = pd.concat(
            [
                self.labels.selection_context(value.train.ledger),
                self.labels.selection_context(value.validation.ledger),
            ],
            ignore_index=True,
            copy=False,
        )
        side_names = np.full(len(ledger), self.side, dtype=object)
        returns = context[ECONOMIC_COLUMN].to_numpy(dtype=np.float32, copy=False)
        hard = context[NET_POSITIVE_COLUMN].to_numpy(dtype=np.float32, copy=False)
        label_context = {
            "feature_selection_archetype": context[ARCHETYPE_COLUMN]
            .astype(str)
            .to_numpy(),
            "side_name": side_names,
            "side": side_names,
            "y_ret": returns,
            "y_bin": hard,
            "bad_mae_1r": context[MAE_TO_SL_COLUMN].to_numpy(
                dtype=np.float32, copy=False
            ),
            "is_timeout": context[TIMEOUT_COLUMN].to_numpy(
                dtype=np.float32, copy=False
            ),
        }

        from extreme_price_movements import lgbm_pipeline

        original_burn_in_days = lgbm_pipeline.LGBM_BASE_FORWARD_BURN_IN_DAYS
        original_short_history_fallback = (
            lgbm_pipeline.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK
        )
        try:
            # The selector implementation reads these process settings at
            # runtime.  Scope the locked-calendar adaptation to this call and
            # restore the shared module even when fitting fails.
            lgbm_pipeline.LGBM_BASE_FORWARD_BURN_IN_DAYS = (
                RECENT_WINNER_SELECTOR_FORWARD_BURN_IN_DAYS
            )
            lgbm_pipeline.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK = False
            result = lgbm_pipeline.train_lgbm_stability_candidate(
                features,
                target,
                # The recent winning per-side tail deliberately removed
                # cross-archetype weighting before MDA.
                sample_weight=np.ones(len(features), dtype=np.float32),
                random_state=self.seed,
                mode="classifier",
                timestamps=pd.to_datetime(
                    ledger["__ts__"], utc=True, errors="raise"
                ).astype("int64"),
                assets=ledger["__symbol__"].astype(str).to_numpy(),
                returns=returns,
                hard_labels=hard,
                hpo_objective_mode="train_base",
                preset_best_params=dict(self._MODEL_PARAMS),
                preset_source=RECENT_WINNER_SELECTOR_CONTRACT,
                cfg={
                    "mda_config": {
                        "archetype_conditioned_enabled": False,
                        "side_tail_across_archetypes_unweighted": True,
                        "correlation_pruning_before_prescreen": True,
                        "correlation_pruning_floor_ratio": 0.50,
                        "correlation_pruning_floor_count": 300,
                    },
                    "lgbm_joint_complete_case_filter_enabled": False,
                },
                label_context=label_context,
            )
        finally:
            lgbm_pipeline.LGBM_BASE_FORWARD_BURN_IN_DAYS = original_burn_in_days
            lgbm_pipeline.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK = (
                original_short_history_fallback
            )
        if not result:
            raise PackBSideFSHPORunnerError(
                f"{self.side} recent-winner selector returned no fitted result"
            )
        selected = [
            str(feature)
            for feature in result.get("selected_feature_names", ())
            if str(feature) in value.candidate_features
        ]
        if len(selected) < 8:
            raise PackBSideFSHPORunnerError(
                f"{self.side} recent-winner selector retained fewer than eight features"
            )
        if len(selected) != len(set(selected)):
            raise PackBSideFSHPORunnerError(
                f"{self.side} recent-winner selector returned duplicate features"
            )
        stats = self._feature_stats_snapshot(result.get("feature_stats"))
        history = result.get(
            "pruning_history",
            result.get("selection_history", result.get("history", ())),
        )
        history_count = len(history) if isinstance(history, Sequence) else 0
        metrics = (
            dict(result.get("metrics", {}))
            if isinstance(result.get("metrics"), Mapping)
            else {}
        )
        if self.resource_guard is not None:
            self.resource_guard.checkpoint(
                f"packb_side_fs_hpo:{self.side}:recent_winner_selector_complete"
            )
        return {
            "side": self.side,
            "selected_features": selected,
            "selection_scope": "side_local",
            "fallback_used": False,
            "selection_methods": [
                "correlation_pruning",
                "archetype_univariate",
                "archetype_relief",
                "mda",
                "automatic_iterative_stopping",
            ],
            "search_breadth": int(
                len(value.candidate_features) + len(stats) + history_count
            ),
            "target_contract": {
                "column": TARGET_COLUMN,
                "economic_validation_column": ECONOMIC_COLUMN,
                "hard_label_column": NET_POSITIVE_COLUMN,
                "archetype_column": ARCHETYPE_COLUMN,
                "weighting_for_side_mda": "uniform_across_archetypes",
            },
            "recent_winner_alignment": {
                "contract": RECENT_WINNER_SELECTOR_CONTRACT,
                "historical_feature_lists_reused": False,
                "historical_fitted_state_reused": False,
                "correlation_first": True,
                "correlation_pruning_floor_ratio": 0.50,
                "correlation_pruning_floor_count": 300,
                "archetype_aware_prescreens": ["univariate", "relief"],
                "mda_scope": self.side,
                "mda_weighting": "uniform_across_archetypes",
                "automatic_stopping": "iterative_mda",
                "forward_validation": {
                    "mode": "forward_burnin",
                    "burn_in_days": (RECENT_WINNER_SELECTOR_FORWARD_BURN_IN_DAYS),
                    "short_history_fallback": False,
                    "adaptation_reason": (
                        "locked selector history is shorter than the "
                        "reference process one-year burn-in"
                    ),
                },
                "selector_model_params": dict(self._MODEL_PARAMS),
                "reference_process_manifest": (
                    str(RECENT_WINNER_PROCESS_MANIFEST.relative_to(ROOT))
                ),
                "reference_feature_contract": (
                    str(RECENT_WINNER_FEATURE_CONTRACT.relative_to(ROOT))
                ),
                "reference_selected_counts": {"long": 55, "short": 37},
                "selector_population": (
                    "legal_pre_november_fit_plus_fixed_november_validation"
                ),
            },
            "selector_metrics": self._finite_metric_snapshot(metrics),
            "feature_stats": stats,
        }


def make_hpo_trials(*, side: str, count: int = DEFAULT_TRIALS) -> tuple[HPOTrial, ...]:
    """Return a predeclared deterministic random design with no default arm."""

    if count < 2:
        raise PackBSideFSHPORunnerError("HPO requires at least two trials")
    seed = 20260724 + (0 if side == "long" else 10_000)
    rng = np.random.default_rng(seed)
    trials: list[HPOTrial] = []
    seen: set[str] = set()
    while len(trials) < int(count):
        max_depth = int(rng.integers(3, 10))
        max_leaves = min(127, 2**max_depth - 1)
        params = {
            "n_estimators": int(rng.integers(240, 901)),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.008), np.log(0.08)))),
            "num_leaves": int(rng.integers(4, max_leaves + 1)),
            "max_depth": max_depth,
            "min_child_samples": int(rng.integers(30, 181)),
            "subsample": float(rng.uniform(0.65, 1.0)),
            "subsample_freq": 1,
            "colsample_bytree": float(rng.uniform(0.60, 1.0)),
            "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(3.0)))),
            "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(12.0)))),
            "min_split_gain": float(rng.uniform(0.0, 0.08)),
        }
        fingerprint = _canonical_sha256(params)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        trials.append(HPOTrial(f"trial_{len(trials):03d}", params))
    return tuple(trials)


class SideHPOEvaluator:
    def __init__(
        self,
        *,
        side: str,
        labels: ExactLabelLoader,
        seed: int,
    ) -> None:
        self.side = side
        self.labels = labels
        self.seed = int(seed)

    def __call__(self, value: HPOFoldInput) -> dict[str, Any]:
        if value.side != self.side:
            raise PackBSideFSHPORunnerError("HPO evaluator received wrong side")
        _model, prediction, best_iteration = _fit_predict(
            value.train.features,
            value.train.target,
            value.train.weights,
            value.validation.features,
            value.validation.target,
            value.validation.weights,
            value.trial.params,
            seed=self.seed
            + int(value.trial.trial_id.rsplit("_", 1)[-1])
            + 1_000 * int(value.fold_name.rsplit("_", 1)[-1]),
        )
        metrics = _economic_objective(
            prediction,
            value.validation.target.to_numpy(dtype=np.float64),
            value.validation.weights.to_numpy(dtype=np.float64),
            self.labels.economic(value.validation.ledger),
            timestamps=value.validation.ledger["__ts__"],
            symbols=value.validation.ledger["__symbol__"],
        )
        return {
            **metrics,
            "best_iteration": int(best_iteration),
            "train_rows": int(len(value.train.ledger)),
            "validation_rows": int(len(value.validation.ledger)),
            "selected_feature_count": int(len(value.selected_features)),
        }


class SideHPOSelector:
    def __init__(self, *, side: str, trials: Sequence[HPOTrial]) -> None:
        self.side = side
        self.trials = {trial.trial_id: trial for trial in trials}

    def __call__(self, evaluations: Sequence[HPOTrialEvaluation]) -> dict[str, Any]:
        by_trial: dict[str, list[float]] = defaultdict(list)
        fold_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for evaluation in evaluations:
            by_trial[evaluation.trial_id].append(float(evaluation.result["objective"]))
            fold_metrics[evaluation.trial_id].append(dict(evaluation.result))
        if set(by_trial) != set(self.trials):
            raise PackBSideFSHPORunnerError("not every HPO trial was evaluated")
        if any(len(values) != 3 for values in by_trial.values()):
            raise PackBSideFSHPORunnerError(
                "every HPO arm must have exactly three chronological folds"
            )
        summaries = []
        for trial_id, values in by_trial.items():
            summaries.append(
                {
                    "trial_id": trial_id,
                    "mean_objective": float(np.mean(values)),
                    "worst_fold_objective": float(np.min(values)),
                    "objective_std": float(np.std(values, ddof=1)),
                    "fold_results": fold_metrics[trial_id],
                }
            )
        ranked = sorted(
            summaries,
            key=lambda row: (
                -row["mean_objective"],
                -row["worst_fold_objective"],
                row["objective_std"],
                row["trial_id"],
            ),
        )
        winner = ranked[0]
        trial = self.trials[winner["trial_id"]]
        return {
            "side": self.side,
            "selected_trial_id": trial.trial_id,
            "selected_params": dict(trial.params),
            "selection_scope": "side_local",
            "fallback_used": False,
            "selection_metric": (
                "mean_three_fold_cost_aware_economic_objective_then_"
                "worst_fold_then_stability"
            ),
            "evaluated_trial_count": int(len(ranked)),
            "evaluated_fold_count_per_trial": 3,
            "ranking": ranked,
        }


def _load_loader_contract(
    loader_root: Path, *, source_revision: str
) -> tuple[dict[str, Any], LoaderEvidenceBundle, dict[str, str]]:
    contract_path = loader_root / "frozen_feature_contract.json"
    evidence_path = loader_root / "loader_evidence.json"
    universe_path = loader_root / "raw_feature_universe.json"
    coverage_path = loader_root / "coverage_profile.json"
    contract = _json(contract_path)
    evidence = _json(evidence_path)
    if evidence.get("source_revision") != source_revision:
        raise PackBSideFSHPORunnerError(
            "AE loader evidence revision does not match this source revision"
        )
    bundle = LoaderEvidenceBundle(
        raw_universe_sha256=str(evidence["raw_universe_sha256"]),
        coverage_profile_sha256=(
            str(evidence["coverage_profile_sha256"])
            if evidence.get("coverage_profile_sha256")
            else None
        ),
        feature_contract_sha256=str(evidence["feature_contract_sha256"]),
        loader_contract_sha256=str(evidence["loader_contract_sha256"]),
        loader_module_sha256=str(evidence["loader_module_sha256"]),
        source_schema_sha256=str(evidence["source_schema_sha256"]),
        source_revision=str(evidence["source_revision"]),
        path=str(evidence_path),
    )
    hashes = {
        "raw_universe_sha256": stage_manifest.sha256_file(universe_path),
        "coverage_profile_sha256": stage_manifest.sha256_file(coverage_path),
        "feature_loader_contract_sha256": stage_manifest.sha256_file(evidence_path),
        "frozen_feature_contract_sha256": stage_manifest.sha256_file(contract_path),
    }
    return contract, bundle, hashes


def _feature_provenance(
    contract: Mapping[str, Any],
    bundle: LoaderEvidenceBundle,
    *,
    state: Mapping[str, Any] | None = None,
    generated_features: Sequence[str] = (),
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for feature in contract["feature_columns"]:
        result[str(feature)] = {
            "causal_definition_sha256": _canonical_sha256(
                {
                    "feature": feature,
                    "generator_registry_sha256": contract["generator_registry_sha256"],
                    "raw_allowlist_sha256": contract["raw_allowlist_sha256"],
                    "selection_provenance": contract["selection_provenance"],
                }
            ),
            "inference_availability_sha256": _canonical_sha256(
                {
                    "feature": feature,
                    "source_schema_sha256": contract["source_schema_sha256"],
                    "store_scan_manifest_sha256": contract[
                        "store_scan_manifest_sha256"
                    ],
                    "loader_contract_sha256": bundle.loader_contract_sha256,
                    "exact_join": "__symbol__+__ts__",
                }
            ),
            "units_contract_sha256": _canonical_sha256(
                {
                    "feature": feature,
                    "storage_units": "canonical_feature_store_native_units",
                    "training_dtype": "float32",
                    "imputation": "forbidden_joint_complete_rows_only",
                    "feature_contract_sha256": contract["feature_contract_sha256"],
                }
            ),
        }
    if generated_features:
        if state is None:
            raise PackBSideFSHPORunnerError(
                "generated features require an AE/GMM state"
            )
        state_hash = str(state["cycle_state_hash"])
        transform_module_sha256 = stage_manifest.sha256_file(
            ROOT / "extreme_price_movements/features_gmm_ae.py"
        )
        for feature in generated_features:
            result[str(feature)] = {
                "causal_definition_sha256": _canonical_sha256(
                    {
                        "feature": feature,
                        "transform": (
                            "frozen_side_local_outcome_free_ae_gmm_row_independent_v1"
                        ),
                        "cycle_state_hash": state_hash,
                        "transform_module_sha256": transform_module_sha256,
                    }
                ),
                "inference_availability_sha256": _canonical_sha256(
                    {
                        "feature": feature,
                        "raw_input_feature_order_hash": state[
                            "input_feature_order_hash"
                        ],
                        "cycle_state_hash": state_hash,
                        "inference_transform": "transform_ae_gmm_features",
                    }
                ),
                "units_contract_sha256": _canonical_sha256(
                    {
                        "feature": feature,
                        "training_dtype": "float32",
                        "units": "frozen_ae_gmm_representation_native_units",
                        "temporal_feature_contract": "row_independent_v1",
                    }
                ),
            }
    return result


def _cohort(population_root: Path, side: str, name: str) -> tuple[pd.DataFrame, Path]:
    path = Path(population_root) / f"cohorts/{side}/{name}.parquet"
    frame = pd.read_parquet(path)
    return frame, path


def _folds(population_root: Path, side: str) -> tuple[HPOFoldLedger, ...]:
    result = []
    for index in range(1, 4):
        train, train_path = _cohort(population_root, side, f"hpo_{index}_train")
        valid, valid_path = _cohort(population_root, side, f"hpo_{index}_valid")
        result.append(
            HPOFoldLedger(
                name=f"hpo_{index}",
                train_ledger=train,
                train_ledger_path=train_path,
                valid_ledger=valid,
                valid_ledger_path=valid_path,
            )
        )
    return tuple(result)


def run(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    population_root: Path = DEFAULT_POPULATION_ROOT,
    ae_root: Path = DEFAULT_AE_ROOT,
    labels_dir: Path = DEFAULT_LABELS,
    feature_store: Path = DEFAULT_FEATURE_STORE,
    feature_inventory_path: Path = DEFAULT_FEATURE_INVENTORY,
    decisions_path: Path = DEFAULT_DECISIONS,
    hpo_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists():
        raise PackBSideFSHPORunnerError(
            f"refusing to overwrite production FS/HPO output: {destination}"
        )
    for reference in (
        RECENT_WINNER_PROCESS_MANIFEST,
        RECENT_WINNER_FEATURE_CONTRACT,
    ):
        if not reference.is_file():
            raise PackBSideFSHPORunnerError(
                f"recent-winner feature-selection reference is missing: {reference}"
            )
    revision = _git_revision()
    ae_summary = _json(Path(ae_root) / "summary.json")
    ae_revision = str(ae_summary.get("source_revision") or "")
    if ae_summary.get("status") != "FROZEN_LONG_AND_SHORT_AE_GMM":
        raise PackBSideFSHPORunnerError("side-local AE summary is absent or incomplete")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ae_revision, revision],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PackBSideFSHPORunnerError(
            "side-local AE source revision is not an ancestor of this runner"
        ) from exc
    population_manifest, source_hashes, calendar_sha256, feature_binding = (
        _source_contracts(
            population_root=Path(population_root),
            feature_inventory_path=Path(feature_inventory_path),
            decisions_path=Path(decisions_path),
        )
    )
    expected_tree = _feature_inventory_binding(Path(feature_inventory_path))
    current_tree = hash_path(Path(feature_store))
    if (
        current_tree.get("sha256") != expected_tree["tree_sha256"]
        or current_tree.get("bytes") != expected_tree["bytes"]
        or current_tree.get("files") != expected_tree["files"]
    ):
        raise PackBSideFSHPORunnerError(
            "canonical feature store changed since the immutable inventory"
        )
    dec09 = parse_locked_dec09(Path(decisions_path))
    if stage_manifest.canonical_json_sha256(dec09["calendar"]) != calendar_sha256:
        raise PackBSideFSHPORunnerError("fixed calendar binding changed")
    label_files = _canonical_label_files(Path(labels_dir), population_manifest)
    label_inventory_hash = _canonical_sha256(
        [
            {
                "name": path.name,
                "sha256": stage_manifest.sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in label_files
        ]
    )
    stage_root = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    stage_root.mkdir(parents=True, exist_ok=False)
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=destination.parent,
        telemetry_path=stage_root / "training_resource_telemetry.jsonl",
    )
    guard.preflight("packb_side_fs_hpo:preflight")
    reports: dict[str, Any] = {}
    try:
        for side_index, side in enumerate(("long", "short")):
            guard.checkpoint(f"packb_side_fs_hpo:{side}:before_inputs")
            loader_root = Path(ae_root) / side / "loader_evidence"
            contract, bundle, extra_hashes = _load_loader_contract(
                loader_root, source_revision=ae_revision
            )
            raw_feature_loader = make_fs_hpo_raw_feature_loader(
                feature_store_dir=feature_store,
                feature_contract=contract,
                evidence_bundle=bundle,
                resource_guard=guard,
            )
            ae_manifest_path = (
                Path(ae_root) / side / "ae_gmm" / "side_stage_manifest.json"
            )
            ae_manifest = stage_manifest.validate_side_stage_manifest(
                ae_manifest_path,
                expected_side=side,
                expected_stage="ae_gmm",
                expected_source_hashes=source_hashes,
                expected_fixed_calendar_sha256=calendar_sha256,
            )
            state_path = (
                Path(ae_root) / side / "ae_gmm" / str(ae_manifest["artifact"]["path"])
            )
            state = _load_side_ae_state(
                state_path,
                expected_side=side,
                expected_sha256=str(ae_manifest["artifact"]["sha256"]),
                raw_features=contract["feature_columns"],
            )
            generated_features = _active_ae_gmm_columns(state)
            candidate_features = tuple(
                [*contract["feature_columns"], *generated_features]
            )
            feature_loader = SideRepresentationFeatureLoader(
                raw_loader=raw_feature_loader,
                raw_features=contract["feature_columns"],
                state=state,
                generated_features=generated_features,
            )
            label_loader = ExactLabelLoader(label_files, resource_guard=guard)
            fs_train, fs_train_path = _cohort(
                population_root, side, "feature_selection_train"
            )
            fs_valid, fs_valid_path = _cohort(
                population_root, side, "feature_selection_valid"
            )
            folds = _folds(population_root, side)
            trials = make_hpo_trials(side=side, count=int(hpo_trials))
            seed = 20260724 + 1_000 * side_index
            report = fit_side_local_fs_hpo_stages(
                side=side,
                fs_train_ledger=fs_train,
                fs_train_ledger_path=fs_train_path,
                fs_valid_ledger=fs_valid,
                fs_valid_ledger_path=fs_valid_path,
                hpo_folds=folds,
                authorized_population_ledger_path=(
                    Path(population_root)
                    / population_manifest["ledgers"]["authorized_population"]["path"]
                ),
                feature_loader=feature_loader,
                target_loader=label_loader.target,
                weight_loader=label_loader.weights,
                candidate_features=list(candidate_features),
                feature_provenance=_feature_provenance(
                    contract,
                    bundle,
                    state=state,
                    generated_features=generated_features,
                ),
                feature_selection_callback=RecentWinnerSideFeatureSelector(
                    side=side,
                    labels=label_loader,
                    seed=seed,
                    resource_guard=guard,
                ),
                hpo_trials=trials,
                hpo_trial_evaluator=SideHPOEvaluator(
                    side=side, labels=label_loader, seed=seed
                ),
                hpo_selection_callback=SideHPOSelector(side=side, trials=trials),
                output_dir=stage_root / side,
                published_output_dir=destination / side,
                source_hashes=source_hashes,
                source_revision=revision,
                fixed_calendar_sha256=calendar_sha256,
                extra_provenance_hashes={
                    **extra_hashes,
                    "canonical_label_content_inventory_sha256": label_inventory_hash,
                    "ae_summary_sha256": stage_manifest.sha256_file(
                        Path(ae_root) / "summary.json"
                    ),
                    "side_ae_manifest_sha256": stage_manifest.sha256_file(
                        ae_manifest_path
                    ),
                    "side_ae_state_sha256": stage_manifest.sha256_file(state_path),
                    "side_ae_transform_contract_sha256": _canonical_sha256(
                        {
                            "cycle_state_hash": state["cycle_state_hash"],
                            "generated_features": list(generated_features),
                            "temporal_feature_contract": (
                                state["temporal_feature_contract"]
                            ),
                        }
                    ),
                    "fs_hpo_raw_subset_loading_contract_sha256": str(
                        raw_feature_loader.fs_hpo_subset_loading_contract_sha256
                    ),
                    "recent_winner_process_manifest_sha256": (
                        stage_manifest.sha256_file(RECENT_WINNER_PROCESS_MANIFEST)
                    ),
                    "recent_winner_feature_contract_sha256": (
                        stage_manifest.sha256_file(RECENT_WINNER_FEATURE_CONTRACT)
                    ),
                },
                fs_train_max_rows=60_000,
                fs_valid_max_rows=20_000,
                hpo_train_max_rows=10_000,
                hpo_valid_max_rows=10_000,
                resource_guard=guard,
            )
            reports[side] = report
            del report, trials, folds, fs_valid, fs_train
            del label_loader, feature_loader, raw_feature_loader
            del state, bundle, contract
            _release_memory()
            guard.checkpoint(f"packb_side_fs_hpo:{side}:released")
        summary = {
            "schema": "packb_pre_march_side_fs_hpo_runner_v1",
            "status": "FROZEN_LONG_AND_SHORT_FEATURE_SELECTION_AND_HPO",
            "source_revision": revision,
            "upstream_ae_source_revision": ae_revision,
            "source_hashes": source_hashes,
            "fixed_calendar_sha256": calendar_sha256,
            "feature_store_revalidation": current_tree,
            "feature_store_inventory": feature_binding,
            "label_contract": {
                "canonical_file_count": len(label_files),
                "canonical_content_inventory_sha256": label_inventory_hash,
                "target_column": TARGET_COLUMN,
                "weight_column": WEIGHT_COLUMN,
                "economic_validation_column": ECONOMIC_COLUMN,
                "cost_accounting": (
                    "economic validation column is canonical first-touch net "
                    "return with stored round-trip cost applied once"
                ),
            },
            "search_contract": {
                "side_local": True,
                "shared_selector_or_study": False,
                "representation": (
                    "raw causal point features plus matching frozen "
                    "side-local outcome-free AE/GMM outputs"
                ),
                "ae_gmm_temporal_contract": "row_independent_v1",
                "raw_subset_load_optimization": (
                    "full frozen parent for AE/GMM outputs; content-validated "
                    "derived subset for raw-only selected features"
                ),
                "feature_selection_validation": "2025-11",
                "feature_selection_contract": RECENT_WINNER_SELECTOR_CONTRACT,
                "feature_selection_process_reference": (
                    "recent 55-long/37-short Pack-B winner; process only, "
                    "historical fitted features and state are not reused"
                ),
                "hpo_validation_months": ["2025-12", "2026-01", "2026-02"],
                "explicit_trials_per_side": int(hpo_trials),
                "fallback": "FORBIDDEN",
            },
            "sides": reports,
        }
        summary_path = stage_root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        guard.checkpoint("packb_side_fs_hpo:complete")
        os.replace(stage_root, destination)
        return {
            **summary,
            "summary_path": str(destination / "summary.json"),
            "summary_sha256": stage_manifest.sha256_file(destination / "summary.json"),
        }
    except Exception:
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hpo-trials", type=int, default=DEFAULT_TRIALS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run(output_dir=args.output_dir, hpo_trials=args.hpo_trials)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
