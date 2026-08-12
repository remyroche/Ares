"""Hash-bound immutable base-offset cache for Stage-I meta MDA.

Meta permutations assess only the incremental correction.  The same-side base
OOF simplex and its causal expected-EV mapping are frozen inputs; refitting or
re-mapping them inside a feature permutation changes the question and can leak
the permuted feature into the base comparator.  This module persists those
arrays once and reconstructs every score as ``fixed_offset + meta_prediction``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_target_neutral_cache import (
    atomic_cache_staging,
    array_sha256,
    canonical_sha256,
    frame_identity_sha256,
)


SCHEMA = "stage_i_meta_fixed_base_offset_cache_v1"


@dataclass(frozen=True)
class MetaOffsetCache:
    root: Path
    manifest: Mapping[str, Any]
    probabilities: np.ndarray
    raw_score: np.ndarray
    expected_net_bps: np.ndarray
    fold_id: np.ndarray
    mapping_support: pd.DataFrame

    def reconstructed_score(self, meta_prediction_bps: Any) -> np.ndarray:
        correction = np.asarray(meta_prediction_bps, dtype=np.float32).reshape(-1)
        if len(correction) != len(self.expected_net_bps) or not np.isfinite(correction).all():
            raise ValueError("meta correction must be finite and cache-row aligned")
        # Copy prevents a caller from obtaining a writable view of the frozen
        # mmap and mutating later permutation baselines in-place.
        return self.expected_net_bps.astype(np.float32, copy=True) + correction


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validated_fold_ids(fold_ids: Any, rows: int) -> np.ndarray:
    """Accept only resolved OOF folds; burn-in/unresolved rows fail closed."""
    raw = np.asarray(fold_ids).reshape(-1)
    if len(raw) != int(rows):
        raise ValueError("meta offset/fold arrays are not finite and row aligned")
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce")
    if numeric.notna().all():
        values = numeric.to_numpy(np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or (values < 0).any():
            raise ValueError("meta offset cache rejects burn-in/unresolved fold ids")
        return values.astype(np.int32)
    values = pd.Series(raw, dtype=object).astype(str).str.strip()
    forbidden = values.str.lower().str.contains(
        r"(^$|burn[ _-]?in|unresolved|unknown|invalid|none|nan|^-1$)", regex=True
    )
    if forbidden.any():
        raise ValueError("meta offset cache rejects burn-in/unresolved fold ids")
    return values.to_numpy(dtype=str)


def _validate_same_side_identity(identity: pd.DataFrame, side: str) -> None:
    if "side_name" not in identity:
        raise ValueError("meta offset cache identity must carry side_name")
    observed = identity["side_name"].astype(str).str.lower()
    if observed.isna().any() or not observed.eq(side).all():
        raise ValueError("meta offset cache base handoff is not same-side")


def _validate_causal_support(support: pd.DataFrame, identity: pd.DataFrame, side: str) -> None:
    required = {
        "candidate_id", "side", "prior_resolved_global_support",
        "prior_resolved_bin_support", "value_map_fallback",
        "value_map_max_label_available_ts",
    }
    missing = sorted(required.difference(support.columns))
    if missing:
        raise ValueError(f"causal mapping support lacks required fields: {missing}")
    if not support["side"].astype(str).str.lower().eq(side).all():
        raise ValueError("causal mapping support is not same-side")
    decision = pd.to_datetime(identity["decision_ts"], utc=True, errors="raise")
    resolved = pd.to_datetime(
        support["value_map_max_label_available_ts"], utc=True, errors="coerce"
    )
    global_support = pd.to_numeric(
        support["prior_resolved_global_support"], errors="coerce"
    ).to_numpy(np.float64)
    bin_support = pd.to_numeric(
        support["prior_resolved_bin_support"], errors="coerce"
    ).to_numpy(np.float64)
    if (
        not np.isfinite(global_support).all() or not np.isfinite(bin_support).all()
        or (global_support < 0).any() or (bin_support < 0).any()
    ):
        raise ValueError("causal mapping support counts are invalid")
    # A neutral cold-start row may report the number of labels visible to the
    # mapper while deliberately consuming none of them because minimum global
    # support has not been reached.  Only non-neutral mapped values must carry
    # an exact last-consumed-label cutoff.
    fallback = support["value_map_fallback"].astype(str)
    consumes_prior_labels = ~fallback.eq("neutral_no_prior_resolved_support")
    if resolved[consumes_prior_labels].isna().any():
        raise ValueError("non-neutral causal map rows lack a resolution cutoff")
    finite = resolved.notna().to_numpy()
    if np.any(resolved[finite].to_numpy() >= decision[finite].to_numpy()):
        raise ValueError("causal mapping support uses future/equal label resolution")


def materialize_meta_offset_cache(
    root: str | Path,
    *,
    identity: pd.DataFrame,
    base_oof_probabilities: Any,
    base_expected_net_bps: Any,
    fold_ids: Any,
    mapping_support: pd.DataFrame,
    target_contract_sha256: str,
    economics_sha256: str,
    base_oof_sha256: str,
    folds_sha256: str,
    feature_contract: Sequence[str],
    side: str,
) -> MetaOffsetCache:
    """Materialise the immutable same-side base handoff for meta MDA."""

    root = Path(root)
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"long", "short"}:
        raise ValueError("meta offset cache side must be long or short")
    _validate_same_side_identity(identity, normalized_side)
    n = len(identity)
    probability = np.asarray(base_oof_probabilities, dtype=np.float32)
    if probability.shape != (n, 3) or not np.isfinite(probability).all():
        raise ValueError("meta offset cache requires an aligned finite 3-state base simplex")
    if (probability < 0.0).any() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("meta offset cache base probabilities are not a simplex")
    offset = np.asarray(base_expected_net_bps, dtype=np.float32).reshape(-1)
    folds = _validated_fold_ids(fold_ids, n)
    if len(offset) != n or not np.isfinite(offset).all():
        raise ValueError("meta offset/fold arrays are not finite and row aligned")
    if len(mapping_support) != n:
        raise ValueError("meta mapping support is not row aligned")
    support = mapping_support.reset_index(drop=True).copy()
    # Support is causal-map context, not a second offset source.  Refuse an
    # ambiguous duplicate expected-EV column.
    forbidden = {"expected_net_bps", "base_expected_net_bps", "prediction_offset"}
    if forbidden.intersection(map(str, support.columns)):
        raise ValueError("mapping support may not contain a competing base offset")
    if "candidate_id" not in support:
        raise ValueError("mapping support must carry candidate_id identity")
    if not np.array_equal(
        support["candidate_id"].astype(str).to_numpy(),
        identity["candidate_id"].astype(str).to_numpy(),
    ):
        raise ValueError("mapping support candidate identity/order differs")
    _validate_causal_support(support, identity, normalized_side)
    features = list(map(str, feature_contract))
    request = {
        "schema": SCHEMA,
        "side": normalized_side,
        "rows": n,
        "identity_sha256": frame_identity_sha256(identity),
        "target_contract_sha256": str(target_contract_sha256),
        "economics_sha256": str(economics_sha256),
        "base_oof_sha256": str(base_oof_sha256),
        "folds_sha256": str(folds_sha256),
        "feature_contract": features,
        "feature_contract_sha256": canonical_sha256(features),
        "probability_sha256": array_sha256(probability),
        "raw_score_sha256": array_sha256(probability[:, 2] - probability[:, 0]),
        "base_expected_net_bps_sha256": array_sha256(offset),
        "fold_id_sha256": array_sha256(folds),
        "mapping_support_sha256": sha256(
            pd.util.hash_pandas_object(support, index=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        ).hexdigest(),
        "offset_semantics": "same_side_strict_oof_causal_expected_net_bps",
        "permutation_semantics": "fixed_base_offset_plus_recomputed_meta_correction",
    }
    request_sha = canonical_sha256(request)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        return load_meta_offset_cache(root, expected_request_sha256=request_sha)
    with atomic_cache_staging(root) as staging:
        if staging is None:
            return load_meta_offset_cache(root, expected_request_sha256=request_sha)
        probability_path = staging / "base_oof_probabilities.npy"
        raw_score_path = staging / "base_raw_score.npy"
        offset_path = staging / "base_expected_net_bps.npy"
        fold_path = staging / "fold_id.npy"
        support_path = staging / "causal_mapping_support.parquet"
        np.save(probability_path, probability, allow_pickle=False)
        np.save(raw_score_path, probability[:, 2] - probability[:, 0], allow_pickle=False)
        np.save(offset_path, offset, allow_pickle=False)
        np.save(fold_path, folds, allow_pickle=False)
        support.to_parquet(support_path, index=False, compression="zstd")
        paths = [probability_path, raw_score_path, offset_path, fold_path, support_path]
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "request": request,
            "request_sha256": request_sha,
            "artifact_sha256": {path.name: _file_sha(path) for path in paths},
            "immutability": {
                "base_offset_permuted": False,
                "base_model_refit_per_permutation": False,
                "causal_map_refit_per_permutation": False,
                "only_meta_prediction_recomputed": True,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return load_meta_offset_cache(root, expected_request_sha256=request_sha)


def load_meta_offset_cache(
    root: str | Path, *, expected_request_sha256: str | None = None
) -> MetaOffsetCache:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("meta base-offset cache manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("meta base-offset cache schema/status drift")
    if expected_request_sha256 is not None and manifest.get("request_sha256") != expected_request_sha256:
        raise ValueError("meta base-offset cache request lineage drift")
    paths = {
        "base_oof_probabilities.npy": root / "base_oof_probabilities.npy",
        "base_raw_score.npy": root / "base_raw_score.npy",
        "base_expected_net_bps.npy": root / "base_expected_net_bps.npy",
        "fold_id.npy": root / "fold_id.npy",
        "causal_mapping_support.parquet": root / "causal_mapping_support.parquet",
    }
    artifacts = manifest.get("artifact_sha256") or {}
    for name, path in paths.items():
        if not path.is_file() or _file_sha(path) != artifacts.get(name):
            raise ValueError(f"meta base-offset cache artifact drift: {name}")
    probability = np.load(paths["base_oof_probabilities.npy"], mmap_mode="r", allow_pickle=False)
    raw = np.load(paths["base_raw_score.npy"], mmap_mode="r", allow_pickle=False)
    offset = np.load(paths["base_expected_net_bps.npy"], mmap_mode="r", allow_pickle=False)
    folds = np.load(paths["fold_id.npy"], mmap_mode="r", allow_pickle=False)
    support = pd.read_parquet(paths["causal_mapping_support.parquet"])
    request = manifest.get("request") or {}
    checks = {
        "probability_sha256": array_sha256(probability),
        "raw_score_sha256": array_sha256(raw),
        "base_expected_net_bps_sha256": array_sha256(offset),
        "fold_id_sha256": array_sha256(folds),
        "mapping_support_sha256": sha256(
            pd.util.hash_pandas_object(support, index=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        ).hexdigest(),
    }
    for key, actual in checks.items():
        if request.get(key) != actual:
            raise ValueError(f"meta base-offset cache semantic drift: {key}")
    if not np.array_equal(raw, probability[:, 2] - probability[:, 0]):
        raise ValueError("meta base-offset cache raw score is inconsistent with simplex")
    return MetaOffsetCache(root, manifest, probability, raw, offset, folds, support)


def assert_fixed_offset_parity(
    cache: MetaOffsetCache,
    baseline_meta_prediction: Any,
    permuted_meta_prediction: Any,
) -> dict[str, float]:
    """Audit that a permutation changed only the meta correction."""
    baseline = cache.reconstructed_score(baseline_meta_prediction)
    permuted = cache.reconstructed_score(permuted_meta_prediction)
    base = cache.expected_net_bps.astype(np.float32, copy=False)
    baseline_correction = np.asarray(baseline_meta_prediction, dtype=np.float32)
    permuted_correction = np.asarray(permuted_meta_prediction, dtype=np.float32)
    if not np.allclose(
        baseline - baseline_correction, base, rtol=0.0, atol=1e-5
    ):
        raise AssertionError("baseline meta reconstruction changed the frozen base offset")
    if not np.allclose(
        permuted - permuted_correction, base, rtol=0.0, atol=1e-5
    ):
        raise AssertionError("permuted meta reconstruction changed the frozen base offset")
    delta_error = float(np.max(np.abs((permuted - baseline) - (permuted_correction - baseline_correction))))
    if delta_error > 1e-5:
        raise AssertionError("meta score delta is not exactly the correction delta")
    return {"rows": float(len(base)), "max_delta_parity_error_bps": delta_error}


def meta_mda_fixed_offset_kwargs(
    cache: MetaOffsetCache,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Narrow adapter for ``run_stage_i_head_selection`` candidate kwargs.

    The returned array is copied and read-only.  The lower MDA scorer already
    implements ``prediction_offset + predicted correction`` and never
    permutes ``prediction_offset``; this hook prevents a runner from passing a
    separately recomputed offset with weaker lineage.
    """
    offset = cache.expected_net_bps.astype(np.float32, copy=True)
    offset.setflags(write=False)
    request = cache.manifest["request"]
    candidate_kwargs = {
        "frozen_base_expected_net_bps": offset,
        "frozen_base_expected_net_units": "bps",
    }
    provenance = {
        "schema": SCHEMA,
        "request_sha256": cache.manifest["request_sha256"],
        "target_contract_sha256": request["target_contract_sha256"],
        "economics_sha256": request["economics_sha256"],
        "base_oof_sha256": request["base_oof_sha256"],
        "folds_sha256": request["folds_sha256"],
        "feature_contract_sha256": request["feature_contract_sha256"],
        "permutation_semantics": request["permutation_semantics"],
    }
    return candidate_kwargs, provenance


def causal_mapping_support_frame(
    map_audit: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    """Project a value-map audit to immutable causal support, excluding EV.

    The mapped expected-net value is cached only in the dedicated offset
    array.  This projection prevents a second writable copy from entering the
    support payload while preserving the prior-row/bin/fallback lineage needed
    to audit each offset.
    """
    if len(map_audit) != len(identity) or "candidate_id" not in map_audit:
        raise ValueError("causal map audit lacks aligned candidate identity")
    if not np.array_equal(
        map_audit["candidate_id"].astype(str).to_numpy(),
        identity["candidate_id"].astype(str).to_numpy(),
    ):
        raise ValueError("causal map audit identity/order differs")
    required = {
        "candidate_id", "side", "prior_resolved_global_support",
        "prior_resolved_bin_support", "value_map_fallback",
        "value_map_max_label_available_ts",
    }
    missing = sorted(required.difference(map_audit.columns))
    if missing:
        raise ValueError(f"causal map audit lacks required support fields: {missing}")
    side_values = map_audit["side"].astype(str).str.lower()
    if "side_name" not in identity or not np.array_equal(
        side_values.to_numpy(), identity["side_name"].astype(str).str.lower().to_numpy()
    ):
        raise ValueError("causal map audit side differs from same-side identity")
    excluded = {
        "prequential_base_expected_net_bps", "expected_net_bps",
        "base_expected_net_bps", "prediction_offset",
    }
    columns = [str(column) for column in map_audit.columns if str(column) not in excluded]
    support = map_audit.loc[:, columns].reset_index(drop=True).copy()
    _validate_causal_support(support, identity, str(side_values.iloc[0]))
    return support


__all__ = [
    "SCHEMA", "MetaOffsetCache", "assert_fixed_offset_parity",
    "load_meta_offset_cache", "materialize_meta_offset_cache",
    "causal_mapping_support_frame", "meta_mda_fixed_offset_kwargs",
]
