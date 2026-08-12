#!/usr/bin/env python3
"""Run the link-defined feature/model-reasoning portability ablation.

This is deliberately a sequential funnel, not a full factorial.  It freezes
the existing side-local TP6/SL4/R3 base control, then executes:

1. feature lineage/portability diagnostics and F0--F4 base contracts;
2. strict-OOF G1/G2/G3 base-reasoning materialisation;
3. leaf health, portability, clusters and causal covariance diagnostics;
4. staged L/H/C/S meta arms, with grouped chronological MDA only at H6; and
5. one final untouched November 2024 replay after all development choices.

The program is cache-oriented and only writes its declared destination.  It
does not overwrite an existing run, use per-timestamp ranking, or fit a
post-test map.  Labels are H12 from the next-hour entry and therefore resolve
thirteen hours after the candidate decision bar.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable, Mapping, Sequence
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Scripts are launched directly from the repository in production.  Make the
# package import explicit rather than depending on an ambient PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.feature_portability import (
    causal_rolling_portability_transform_batches,
    classify_feature_roles,
    estimate_causal_rolling_transform_memory,
)
from extreme_price_movements.feature_portability_audit import (
    ChronologicalAuditPolicy,
    run_chronological_feature_portability_audit,
    write_feature_portability_artifacts,
)
from extreme_price_movements.strict_oof_base_reasoning import (
    STRICT_OOF_BASE_REASONING_SCHEMA,
    StrictOOFBaseReasoningConfig,
    StrictOOFContributionCacheCapacityError,
    build_strict_oof_multiclass_contribution_cache,
    materialize_strict_oof_base_reasoning,
)
from extreme_price_movements.tp6_portability_data import (
    FROZEN_META_CONTEXT,
    SIDES,
    TP6PortabilityContract,
    TP6_SL4_COST_BPS,
    all_frozen_base_features,
    frozen_input_columns,
    load_tp6_population,
)
from extreme_price_movements.tp6_transport_validation import (
    FINAL_OOS,
    TRANSPORT_A,
    TRANSPORT_B,
    TimeWindow,
    TransportSpec,
    evaluate_transport,
)


SCHEMA = "feature_leaf_reasoning_portability_v1"
TOP_FRACTIONS = (0.01, 0.05, 0.10)
BASE_PARAMS = dict(
    n_estimators=140, learning_rate=0.05, num_leaves=31, min_child_samples=350,
    subsample=0.80, colsample_bytree=0.80, reg_lambda=8.0, n_jobs=1, verbosity=-1,
)
META_PARAMS = dict(
    n_estimators=180, learning_rate=0.035, num_leaves=15, min_child_samples=500,
    subsample=0.80, colsample_bytree=0.80, reg_lambda=15.0, n_jobs=1, verbosity=-1,
)
# F3 has 32 source fields per side and eight derived columns per source.  At
# one million rows that is ~0.95 GiB of unavoidable float32 output.  The
# transform itself is therefore one-source-batch at a time, and the run fails
# before allocation if a wider input would cross this explicit boundary.
F3_TRANSFORM_FEATURE_BATCH_SIZE = 1
F3_MAX_GENERATED_BYTES = 1024 * 1024 * 1024
F3_MAX_BATCH_WORKING_BYTES = 384 * 1024 * 1024
# One R3 fold can otherwise retain three dense contribution matrices for both
# train and evaluation.  Prefer a 384MiB RAM cache.  Later expanding folds can
# use one explicitly bounded temporary memmap instead of repeating all-class
# LightGBM contribution passes once per semantic head.  The scratch map is
# removed immediately after its adverse/weak/clear artifacts are committed.
STRICT_REASONING_MULTICLASS_CACHE_MAX_BYTES = 384 * 1024 * 1024
STRICT_REASONING_MULTICLASS_SPILL_MAX_BYTES = 4 * 1024 * 1024 * 1024
STRICT_REASONING_MULTICLASS_SPILL_MAX_WORKING_BYTES = 128 * 1024 * 1024
STRICT_REASONING_REQUIRED_OUTPUTS = (
    "base_reasoning_features.parquet",
    "base_reasoning_predictions.parquet",
    "base_reasoning_labels.parquet",
    "leaf_assignments.parquet",
    "leaf_rule_catalog.parquet",
    "contribution_bundle.parquet",
)


class PortabilityRunError(RuntimeError):
    """Raised for a failure of the experiment's frozen causal contract."""


@dataclass(frozen=True)
class TransportRun:
    """One continuous development or terminal chronological transport."""

    name: str
    train_start: str
    train_end: str
    eval_start: str
    eval_end: str
    terminal: bool = False

    def spec(self) -> TransportSpec:
        return TransportSpec(
            name=self.name,
            train_windows=(TimeWindow(self.train_start, self.train_end),),
            test_windows=(TimeWindow(self.eval_start, self.eval_end),),
        )


DEVELOPMENT_TRANSPORTS: tuple[TransportRun, ...] = (
    TransportRun("transport_a_2023q4_to_2024h1", "2023-04-01", "2024-01-01", "2024-01-01", "2024-07-01"),
    # December 2024 is absent from the retained source.  This is explicitly
    # H2-to-date development rather than an invented full-H2 claim.
    TransportRun("transport_b_2024h1_to_2024h2_to_date", "2023-04-01", "2024-07-01", "2024-07-01", "2024-11-01"),
)
UNTOUCHED_OOS = TransportRun("final_untouched_oos_2024_11", "2023-04-01", "2024-11-01", "2024-11-01", "2024-12-01", terminal=True)


def _utc(value: object) -> pd.Timestamp:
    out = pd.Timestamp(value)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_contract_payload(contract: TP6PortabilityContract) -> dict[str, object]:
    """Use the exact JSON-safe source-contract representation stored in runs."""
    return json.loads(json.dumps(asdict(contract), default=str, sort_keys=True))


def _fitted_model_sha256(model: lgb.LGBMClassifier) -> str:
    """Fingerprint a fitted base model without depending on private helpers.

    A strict artifact may only be reused when the model fitted on the resumed
    invocation is byte-for-byte the same LightGBM model recorded by the
    committed artifact.  This catches changed source values, labels, model
    parameters, seeds, or library serialisation before stale reasoning output
    can be mixed into a resumed meta-training population.
    """
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise PortabilityRunError("cannot resume strict reasoning from an unfitted base model")
    try:
        encoded = booster.model_to_string(num_iteration=-1).encode("utf-8")
    except Exception as exc:  # pragma: no cover - LightGBM implementation failure
        raise PortabilityRunError("cannot fingerprint resumed fitted base model") from exc
    return hashlib.sha256(encoded).hexdigest()


def _resume_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise persisted/current candidate keys before exact comparison."""
    required = ["candidate_id", "__ts__", "side_name"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise PortabilityRunError(f"strict resume identity lacks required keys: {missing}")
    result = frame.loc[:, required].copy().reset_index(drop=True)
    result["candidate_id"] = result["candidate_id"].astype("string")
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="coerce")
    result["side_name"] = result["side_name"].astype("string").str.lower()
    if result.isna().any().any() or result.duplicated(required).any():
        raise PortabilityRunError("strict resume identity is null or non-unique")
    return result


def _equal_resume_values(stored: pd.Series, expected: pd.Series) -> bool:
    """Exact equality allowing matched nulls, including UTC label timestamps."""
    if len(stored) != len(expected):
        return False
    if pd.api.types.is_datetime64_any_dtype(stored) or pd.api.types.is_datetime64_any_dtype(expected):
        left = pd.to_datetime(stored, utc=True, errors="coerce")
        right = pd.to_datetime(expected, utc=True, errors="coerce")
        return bool(left.equals(right))
    left = pd.to_numeric(stored, errors="coerce").to_numpy(np.float64)
    right = pd.to_numeric(expected, errors="coerce").to_numpy(np.float64)
    return bool(np.array_equal(left, right, equal_nan=True))


def _validate_reusable_strict_reasoning_head(
    artifact_dir: Path,
    *,
    head_name: str,
    class_index: int,
    side: str,
    fold_id: str,
    model: lgb.LGBMClassifier,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    identity: pd.DataFrame,
    probabilities: np.ndarray,
    labels: pd.DataFrame,
    feature_columns: Sequence[str],
    config: StrictOOFBaseReasoningConfig,
) -> bool:
    """Return whether one existing head is safe to reuse; otherwise fail closed.

    The strict materialiser commits a complete head atomically.  A resume must
    still establish that this exact invocation would produce the same fold:
    schema, semantic head, fitted-model fingerprint, training/evaluation time
    bounds, feature contract, candidate identity, prediction slice, labels,
    and every persisted output hash are checked.  A missing head is the only
    state that asks the caller to materialise it; a present but incompatible or
    damaged head can never be overwritten in place.
    """
    if not artifact_dir.exists():
        return False
    if not artifact_dir.is_dir():
        raise PortabilityRunError(f"strict resume path is not a directory: {artifact_dir}")
    manifest_path = artifact_dir / "base_reasoning_manifest.json"
    if not manifest_path.exists():
        raise PortabilityRunError(
            f"strict resume found an incomplete head without a manifest: {artifact_dir}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortabilityRunError(f"strict resume cannot read head manifest: {artifact_dir}") from exc

    expected_feature_hash = _sha256_json(list(feature_columns))
    expected_model_hash = _fitted_model_sha256(model)
    expected_config = asdict(config)
    provenance = manifest.get("provenance", {})
    rows = manifest.get("rows", {})
    invariants = {
        "schema": manifest.get("schema") == STRICT_OOF_BASE_REASONING_SCHEMA,
        "status": manifest.get("status") == "MATERIALIZED_STRICT_OOF",
        "head_name": manifest.get("head_name") == str(head_name),
        "side_name": manifest.get("side_name") == str(side).lower(),
        "fold_id": manifest.get("fold_id") == str(fold_id),
        "class_index": provenance.get("class_index") == int(class_index),
        "feature_contract_sha256": provenance.get("feature_contract_sha256") == expected_feature_hash,
        "feature_contract": provenance.get("feature_contract") == list(feature_columns),
        "model_hashes": provenance.get("model_hashes") == [expected_model_hash],
        "config": manifest.get("config") == expected_config,
        "train_rows": rows.get("train") == int(len(train)),
        "eval_rows": rows.get("eval") == int(len(evaluate)),
        "train_start": provenance.get("train_start_utc") == _utc(train["decision_ts"].min()).isoformat(),
        "train_end": provenance.get("train_end_utc") == _utc(train["decision_ts"].max()).isoformat(),
        "eval_start": provenance.get("eval_start_utc") == _utc(evaluate["decision_ts"].min()).isoformat(),
        "eval_end": provenance.get("eval_end_utc") == _utc(evaluate["decision_ts"].max()).isoformat(),
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise PortabilityRunError(
            f"strict resume artifact is incompatible and cannot be overwritten ({artifact_dir}): {failed}"
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PortabilityRunError(f"strict resume artifact lacks output hashes: {artifact_dir}")
    for name in STRICT_REASONING_REQUIRED_OUTPUTS:
        path = artifact_dir / name
        recorded = outputs.get(name)
        if not isinstance(recorded, str) or not path.is_file() or _sha256_file(path) != recorded:
            raise PortabilityRunError(
                f"strict resume artifact output is missing or hash-mismatched: {path}"
            )

    expected_identity = _resume_identity(identity)
    prediction_path = artifact_dir / "base_reasoning_predictions.parquet"
    stored_prediction = pd.read_parquet(
        prediction_path,
        columns=["candidate_id", "__ts__", "side_name", "base_prediction", "head_name", "class_index", "fold_id"],
    )
    if not _resume_identity(stored_prediction).equals(expected_identity):
        raise PortabilityRunError(f"strict resume candidate identity mismatch: {prediction_path}")
    if (
        not stored_prediction["head_name"].astype("string").eq(str(head_name)).all()
        or not pd.to_numeric(stored_prediction["class_index"], errors="coerce").eq(int(class_index)).all()
        or not stored_prediction["fold_id"].astype("string").eq(str(fold_id)).all()
        or not np.array_equal(
            pd.to_numeric(stored_prediction["base_prediction"], errors="coerce").to_numpy(np.float32),
            np.asarray(probabilities[:, int(class_index)], dtype=np.float32),
            equal_nan=False,
        )
    ):
        raise PortabilityRunError(f"strict resume prediction provenance mismatch: {prediction_path}")

    label_path = artifact_dir / "base_reasoning_labels.parquet"
    stored_labels = pd.read_parquet(label_path)
    if not _resume_identity(stored_labels).equals(expected_identity):
        raise PortabilityRunError(f"strict resume label identity mismatch: {label_path}")
    for column in labels:
        stored_name = f"label__{column}"
        if stored_name not in stored_labels or not _equal_resume_values(stored_labels[stored_name], labels[column].reset_index(drop=True)):
            raise PortabilityRunError(f"strict resume label provenance mismatch: {label_path}/{stored_name}")
    if (
        not stored_labels["head_name"].astype("string").eq(str(head_name)).all()
        or not stored_labels["fold_id"].astype("string").eq(str(fold_id)).all()
    ):
        raise PortabilityRunError(f"strict resume label head provenance mismatch: {label_path}")
    return True


def _as_float_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Preserve the declared ordered contract and make LGBM input reproducible."""
    if not columns:
        raise PortabilityRunError("an ablation cannot fit an empty feature contract")
    missing = [name for name in columns if name not in frame]
    if missing:
        raise PortabilityRunError(f"feature contract misses columns: {missing[:10]}")
    matrix = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return matrix.astype(np.float32, copy=False)


def _r3_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["r3_class"].to_numpy(np.int8)
    agreement = frame[["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]].nunique(axis=1).eq(1).to_numpy(float)
    certainty = 0.5 + 0.5 * agreement
    counts = np.bincount(labels, minlength=3).astype(float)
    class_weight = np.sqrt(len(frame) / np.maximum(counts, 1.0))[labels]
    class_weight /= max(class_weight.mean(), 1e-12)
    weight = np.clip(certainty * class_weight, 0.25, 4.0)
    return weight / max(weight.mean(), 1e-12)


def _fit_base(frame: pd.DataFrame, columns: Sequence[str], *, seed: int) -> lgb.LGBMClassifier:
    if len(frame) < 10_000:
        raise PortabilityRunError(f"base train support too small: {len(frame)}")
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, random_state=int(seed), **BASE_PARAMS
    )
    model.fit(
        _as_float_matrix(frame, columns), frame["r3_class"].to_numpy(np.int8),
        sample_weight=_r3_weights(frame),
    )
    if not np.array_equal(np.asarray(model.classes_, dtype=np.int8), np.array([0, 1, 2], dtype=np.int8)):
        raise PortabilityRunError("frozen R3 base class order must be adverse=0, weak=1, clear=2")
    return model


def _base_prediction(model: lgb.LGBMClassifier, frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    value = np.asarray(model.predict_proba(_as_float_matrix(frame, columns)), dtype=np.float32)
    if value.shape != (len(frame), 3) or not np.isfinite(value).all():
        raise PortabilityRunError("base must emit finite adverse/weak/clear probabilities")
    return value


def _class_payoff_map(train: pd.DataFrame) -> np.ndarray:
    """Prior-resolved causal bps map: a model probability simplex -> common bps."""
    global_mean = float(train["net_bps"].mean())
    means = np.array(
        [
            float(train.loc[train["r3_class"].eq(k), "net_bps"].mean())
            if train["r3_class"].eq(k).any() else global_mean
            for k in range(3)
        ],
        dtype=np.float32,
    )
    if not np.isfinite(means).all():
        raise PortabilityRunError("prior-resolved class payoff map is non-finite")
    return means


def _inner_windows(run: TransportRun) -> tuple[TimeWindow, ...]:
    """Two-month evaluation blocks with a three-month minimum warm-up.

    The only source of any fold statistic is rows whose labels resolve before
    that block's start.  No fold owns a history value derived from itself.
    """
    start, end = _utc(run.train_start), _utc(run.train_end)
    cursor = start + pd.DateOffset(months=3)
    values: list[TimeWindow] = []
    while cursor < end:
        next_cursor = min(cursor + pd.DateOffset(months=2), end)
        if next_cursor > cursor:
            values.append(TimeWindow(cursor, next_cursor))
        cursor = next_cursor
    if len(values) < 2:
        raise PortabilityRunError(f"{run.name} has insufficient inner chronological folds")
    return tuple(values)


def _require_prior(train: pd.DataFrame, eval_frame: pd.DataFrame) -> None:
    if train.empty or eval_frame.empty:
        raise PortabilityRunError("chronological fold has empty train/evaluation support")
    if not train["label_available_ts"].lt(eval_frame["decision_ts"].min()).all():
        raise PortabilityRunError("a base fold includes a label unresolved at its evaluation start")


def _reasoning_for_fold(
    *,
    model: lgb.LGBMClassifier,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    columns: Sequence[str],
    side: str,
    fold_id: str,
    probabilities: np.ndarray,
    destination: Path,
    return_payload: bool = True,
    resume_existing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run G1/G2/G3 separately for clear/adverse/weak; never mix heads."""
    feature_tables: list[pd.DataFrame] = []
    assignments: list[pd.DataFrame] = []
    catalogs: list[pd.DataFrame] = []
    train_matrix = _as_float_matrix(train, columns)
    eval_matrix = _as_float_matrix(evaluate, columns)
    identity = evaluate.loc[:, ["candidate_id", "decision_ts", "side_name"]].rename(columns={"decision_ts": "__ts__"})
    labels = evaluate.loc[:, ["gross_bps", "net_bps", "r3_class", "label_available_ts"]]
    config = StrictOOFBaseReasoningConfig(max_trees_per_model=64, contribution_components=12)
    requested_heads = (("p_adverse", 0), ("p_weak", 1), ("p_clear", 2))
    reusable_heads: set[str] = set()
    if resume_existing:
        for head, index in requested_heads:
            head_dir = destination / head
            if _validate_reusable_strict_reasoning_head(
                head_dir,
                head_name=head,
                class_index=index,
                side=side,
                fold_id=fold_id,
                model=model,
                train=train,
                evaluate=evaluate,
                identity=identity,
                probabilities=probabilities,
                labels=labels,
                feature_columns=columns,
                config=config,
            ):
                reusable_heads.add(head)
    # LightGBM multiclass ``pred_contrib`` returns all three classes in one
    # pass.  Reusing a bounded per-fold bundle avoids repeating that expensive
    # work for adverse/weak/clear, while materialisation still receives an
    # explicit class index for every semantic head.  The bounded memmap fallback
    # is temporary implementation state, not a persisted artifact.  Capacity is
    # deliberately a normal optimisation miss, never a reason to alter the
    # strict contract.
    contribution_cache = None
    if len(reusable_heads) != len(requested_heads):
        try:
            contribution_cache = build_strict_oof_multiclass_contribution_cache(
                [model], train_matrix, eval_matrix,
                batch_rows=config.contribution_batch_rows,
                max_cache_bytes=STRICT_REASONING_MULTICLASS_CACHE_MAX_BYTES,
                spill_directory=destination.parent,
                max_spill_bytes=STRICT_REASONING_MULTICLASS_SPILL_MAX_BYTES,
                spill_max_working_bytes=STRICT_REASONING_MULTICLASS_SPILL_MAX_WORKING_BYTES,
            )
        except StrictOOFContributionCacheCapacityError:
            contribution_cache = None
    try:
        for head, index in requested_heads:
            if head in reusable_heads:
                if return_payload:
                    head_dir = destination / head
                    result_features = pd.read_parquet(head_dir / "base_reasoning_features.parquet")
                    feature_tables.append(result_features.drop(columns=["head_name", "fold_id"]).rename(columns={
                        name: f"{head}__{name}"
                        for name in result_features
                        if name.startswith("base_reasoning__")
                    }))
                    assignments.append(pd.read_parquet(head_dir / "leaf_assignments.parquet").assign(head_name=head))
                    catalogs.append(pd.read_parquet(head_dir / "leaf_rule_catalog.parquet").assign(head_name=head))
                continue
            # One model contains three tree groups.  The materialiser's explicit
            # head index prevents the former class-1-only contribution shortcut.
            result = materialize_strict_oof_base_reasoning(
                [model], train_matrix, eval_matrix,
                head_name=head, class_index=index, side_name=side, fold_id=fold_id,
                train_timestamps=train["decision_ts"], eval_timestamps=evaluate["decision_ts"],
                eval_identity=identity, eval_predictions=probabilities[:, index],
                eval_labels=labels, train_targets=train["r3_class"].to_numpy(np.float32),
                artifact_dir=destination / head,
                config=config,
                contribution_cache=contribution_cache,
            )
            # The materialiser has atomically written complete per-head/fold G1,
            # G2 and G3 artifacts.  Retaining all three wide assignment tables
            # for every chronological fold would turn a valid full-data run into
            # a multi-GB in-memory concatenation.  The disk index is therefore
            # canonical; callers request payloads only for small diagnostics.
            if not return_payload:
                del result
                continue
            feats = result.features.drop(columns=["head_name", "fold_id"]).copy()
            renames = {name: f"{head}__{name}" for name in feats if name.startswith("base_reasoning__")}
            feature_tables.append(feats.rename(columns=renames))
            assignments.append(result.leaf_assignments.assign(head_name=head))
            catalogs.append(result.leaf_rule_catalog.assign(head_name=head))
    finally:
        if contribution_cache is not None:
            contribution_cache.release()
    if not return_payload:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Candidate identity is identical across heads by construction.  Merge
    # compact features only; full per-head labels remain their own artifacts.
    merged = feature_tables[0]
    keys = ["candidate_id", "__ts__", "side_name"]
    for table in feature_tables[1:]:
        merged = merged.merge(table, on=keys, how="inner", validate="one_to_one")
    return merged, pd.concat(assignments, ignore_index=True), pd.concat(catalogs, ignore_index=True)


def _frozen_feature_roles(columns: Sequence[str]) -> pd.DataFrame:
    """Static role inventory.  Outcome fields are never sent here."""
    return classify_feature_roles(list(dict.fromkeys(columns)))


def _add_causal_relative_transforms(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add the coverage-safe causal rank, robust-z, and delta F3 bundle.

    This happens before a fold split but every generated row only uses its
    own/past asset observations.  It is valid for the next-open convention:
    inputs are already complete at ``decision_ts``.  Crucially, this mutates
    the already-owned side frame in one source-feature batch at a time; it
    does not make a second full raw panel or sort every source column.
    """
    estimate = estimate_causal_rolling_transform_memory(
        rows=len(frame), source_features=len(columns), rank_windows=(90, 180),
        robust_z_windows=(90, 180), change_periods=(4, 24), include_relative_change=False,
        feature_batch_size=F3_TRANSFORM_FEATURE_BATCH_SIZE,
    )
    if estimate.materialized_output_bytes > F3_MAX_GENERATED_BYTES:
        raise PortabilityRunError(
            "F3 generated matrix exceeds its explicit memory contract "
            f"({estimate.materialized_output_bytes:,} > {F3_MAX_GENERATED_BYTES:,} bytes); "
            "reduce the predeclared F3 source contract before running"
        )
    generated_names: list[str] = []
    # Incremental insertion avoids a second full, multi-gigabyte DataFrame
    # copy.  Pandas warns once the intentionally wide F3 contract becomes
    # fragmented; that layout is expected here and the warning is not a data
    # quality issue.  LGBM receives a dense float32 matrix only per fold.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        for generated in causal_rolling_portability_transform_batches(
            frame,
            feature_names=list(columns), timestamp_column="decision_ts", group_columns=["asset"],
            rank_windows=(90, 180), robust_z_windows=(90, 180), change_periods=(4, 24),
            include_relative_change=False,
            minimum_periods=30,
            feature_batch_size=F3_TRANSFORM_FEATURE_BATCH_SIZE,
            max_batch_working_bytes=F3_MAX_BATCH_WORKING_BYTES,
        ):
            for name in generated:
                frame[name] = generated[name].to_numpy(copy=False)
                generated_names.append(name)
            del generated
            gc.collect()
    return frame, generated_names


def _feature_contracts(
    frame: pd.DataFrame,
    base: Sequence[str],
    portability_manifest: pd.DataFrame,
) -> dict[str, tuple[list[str], str]]:
    """The F0--F4 sequential feature funnel (F4 is filled after grouped MDA)."""
    approved = set(
        portability_manifest.loc[
            portability_manifest["disposition"].isin(("INVARIANT_RAW", "INVARIANT_NORMALIZED", "INVARIANT_RELATIVE", "KEEP_PORTABLE")),
            "feature",
        ].astype(str)
    )
    f0 = list(base)
    f1 = [name for name in base if name in approved]
    # Do not silently replace a small portable subset with F0.  A weak F1 is
    # still a meaningful outcome of the Stage-A gate; its result tells us
    # whether the frozen model depended on non-portable/unstable raw inputs.
    f1_note = "portable_raw_subset"
    # Existing selected fields already express distances in ATR, returns,
    # ratios, residuals or ranks.  Add a raw/ATR ratio only when a truly raw
    # level exists; never divide non-price fractions or rates by ATR.
    raw_levels = [
        name for name in f1
        if name in frame and not any(token in name.lower() for token in ("atr", "ratio", "pct", "fraction", "rank", "resid", "norm", "z_", "per_hour"))
        and any(token in name.lower() for token in ("price", "distance", "level", "vwap", "donchian"))
    ]
    f2 = f1.copy()
    for name in raw_levels:
        generated = f"{name}__atr_normalized"
        atr = pd.to_numeric(frame["atr_1h"], errors="coerce")
        frame[generated] = pd.to_numeric(frame[name], errors="coerce").div(atr.where(atr.abs() > 1e-12)).astype(np.float32)
        f2.append(generated)
    return {
        "F0_current_frozen": (f0, "frozen_side_local_control"),
        "F1_portable_raw": (f1, f1_note),
        "F2_portable_plus_atr": (f2, "raw_atr_normalization" if raw_levels else "no_selected_raw_level_required_atr_normalization"),
    }


def _prequential_feature_residual(frame: pd.DataFrame) -> pd.Series:
    """A label-safe provisional residual for Stage-A diagnostics.

    The per-side expanding estimate is shifted by one decision timestamp and
    only includes rows whose H12 labels are available.  The definitive Stage-A
    residual replaces it after strict OOF base scores are materialised.
    """
    ordered = frame.sort_values(["side_name", "label_available_ts", "candidate_id"], kind="stable").copy()
    prior = ordered.groupby("side_name", observed=True)["net_bps"].transform(lambda value: value.expanding().mean().shift(1))
    ordered["__residual__"] = ordered["net_bps"] - prior
    return ordered.set_index("candidate_id")["__residual__"].reindex(frame["candidate_id"]).to_numpy()


def run_feature_audit(
    *,
    destination: Path,
    contract: TP6PortabilityContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage A: one causal source/coverage/drift audit over both sides."""
    input_columns = frozen_input_columns(contract)
    frame = load_tp6_population(contract=contract, columns=input_columns, start="2023-04-01", end="2024-11-01")
    frame["era"] = frame["decision_ts"].dt.to_period("Q").astype(str)
    frame["economic_residual_provisional"] = _prequential_feature_residual(frame)
    features = list(dict.fromkeys([*all_frozen_base_features(contract)["long"], *all_frozen_base_features(contract)["short"], *FROZEN_META_CONTEXT]))
    roles = _frozen_feature_roles(features)
    result = run_chronological_feature_portability_audit(
        frame, feature_names=features, timestamp_column="decision_ts", era_column="era",
        strata_columns=("side_name",), target_column="net_bps",
        economic_residual_column="economic_residual_provisional",
        policy=ChronologicalAuditPolicy(min_reference_rows=1000),
    )
    # Preserve the exact link-named canonical artifact in addition to the
    # module's more explicit era/disposition companions.
    write_feature_portability_artifacts(result, destination)
    audit = result.era_audit
    manifest = result.dispositions
    audit.to_parquet(destination / "feature_portability_audit.parquet", index=False, compression="zstd")
    roles.to_csv(destination / "feature_role_manifest.csv", index=False)
    (destination / "feature_role_manifest.yaml").write_text(
        "schema: feature_leaf_reasoning_portability_v1\nroles:\n" + "\n".join(
            f"  {row.feature}: {row.role}" for row in roles.itertuples(index=False)
        ) + "\n", encoding="utf-8"
    )
    (destination / "portable_feature_manifest.json").write_text(
        json.dumps({"schema": SCHEMA, "coverage_gate": 0.99, "rows": manifest.to_dict("records")}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return frame, manifest


def _fold_base_rows(
    *,
    frame: pd.DataFrame,
    run: TransportRun,
    feature_columns: Sequence[str],
    side: str,
    seed: int,
    artifact_root: Path | None,
    retain_oof: bool = True,
    retain_reasoning_payload: bool = True,
    resume_existing_reasoning: bool = False,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Materialise inner OOF base scores and one outer-test base fit.

    Returned OOF rows are the *only* allowed meta-training population.  Each
    row's base model and class-to-bps map use labels resolved before that
    row's decision interval.  The outer test model is fit before the outer
    evaluation begins and is never refit during the evaluation period.
    """
    start, train_end, eval_end = _utc(run.train_start), _utc(run.train_end), _utc(run.eval_end)
    # ``_arm_frame`` has already loaded exactly this interval.  Keep a view
    # where possible: a full copy here plus one train/eval copy per fold was
    # enough to make the all-arm run susceptible to memory termination.
    scoped = frame.loc[
        frame["decision_ts"].ge(start) & frame["decision_ts"].lt(eval_end)
    ]
    if scoped.empty:
        raise PortabilityRunError(f"{run.name}/{side} has no frozen candidates")
    oof_rows: list[pd.DataFrame] = []
    reasoning_rows: list[pd.DataFrame] = []
    assignment_rows: list[pd.DataFrame] = []
    catalog_rows: list[pd.DataFrame] = []
    for number, window in enumerate(_inner_windows(run)):
        evaluate = scoped.loc[
            scoped["decision_ts"].ge(window.start) & scoped["decision_ts"].lt(window.end)
        ]
        train = scoped.loc[
            scoped["decision_ts"].lt(window.start) & scoped["label_available_ts"].lt(window.start)
        ]
        _require_prior(train, evaluate)
        model = _fit_base(train, feature_columns, seed=seed + number)
        probabilities = _base_prediction(model, evaluate, feature_columns)
        class_map = _class_payoff_map(train)
        if retain_oof:
            output_columns = list(dict.fromkeys([
                "candidate_id", "decision_ts", "label_available_ts", "side_name", "asset",
                "gross_bps", "net_bps", "r3_class", *FROZEN_META_CONTEXT,
            ]))
            output = evaluate.loc[:, output_columns].copy()
            output["fold_id"] = f"{run.name}_inner_{number:02d}"
            output["p_adverse"] = probabilities[:, 0]
            output["p_weak"] = probabilities[:, 1]
            output["p_clear"] = probabilities[:, 2]
            output["base_raw"] = probabilities[:, 2] - probabilities[:, 0]
            output["base_expected_bps"] = probabilities @ class_map
            output["base_fit_cutoff_ts"] = train["label_available_ts"].max()
            output["base_map_cutoff_ts"] = train["label_available_ts"].max()
            output["feature_generation_ts"] = output["decision_ts"]
            oof_rows.append(output)
        if artifact_root is not None:
            if not retain_oof:
                raise PortabilityRunError("reasoning materialisation requires strict OOF output retention")
            features, assignments, catalog = _reasoning_for_fold(
                model=model, train=train, evaluate=evaluate, columns=feature_columns,
                side=side, fold_id=f"{run.name}_inner_{number:02d}", probabilities=probabilities,
                destination=artifact_root / "folds" / side / f"inner_{number:02d}",
                return_payload=retain_reasoning_payload,
                resume_existing=resume_existing_reasoning,
            )
            if retain_reasoning_payload:
                reasoning_rows.append(features)
                assignment_rows.append(assignments)
                catalog_rows.append(catalog)

    outer_train = scoped.loc[
        scoped["decision_ts"].lt(train_end) & scoped["label_available_ts"].lt(train_end)
    ]
    outer_eval = scoped.loc[
        scoped["decision_ts"].ge(train_end) & scoped["decision_ts"].lt(eval_end)
    ]
    _require_prior(outer_train, outer_eval)
    final_model = _fit_base(outer_train, feature_columns, seed=seed + 10_000)
    final_probabilities = _base_prediction(final_model, outer_eval, feature_columns)
    final_map = _class_payoff_map(outer_train)
    final_columns = list(dict.fromkeys([
        "candidate_id", "decision_ts", "label_available_ts", "side_name", "asset",
        "gross_bps", "net_bps", "r3_class", *FROZEN_META_CONTEXT,
    ]))
    final = outer_eval.loc[:, final_columns].copy()
    final["fold_id"] = f"{run.name}_outer"
    final["p_adverse"] = final_probabilities[:, 0]
    final["p_weak"] = final_probabilities[:, 1]
    final["p_clear"] = final_probabilities[:, 2]
    final["base_raw"] = final_probabilities[:, 2] - final_probabilities[:, 0]
    final["base_expected_bps"] = final_probabilities @ final_map
    final["base_fit_cutoff_ts"] = outer_train["label_available_ts"].max()
    final["base_map_cutoff_ts"] = outer_train["label_available_ts"].max()
    final["feature_generation_ts"] = final["decision_ts"]
    if artifact_root is not None:
        features, assignments, catalog = _reasoning_for_fold(
            model=final_model, train=outer_train, evaluate=outer_eval, columns=feature_columns,
            side=side, fold_id=str(final["fold_id"].iat[0]), probabilities=final_probabilities,
            destination=artifact_root / "folds" / side / "outer",
            return_payload=retain_reasoning_payload,
            resume_existing=resume_existing_reasoning,
        )
        if retain_reasoning_payload:
            reasoning_rows.append(features)
            assignment_rows.append(assignments)
            catalog_rows.append(catalog)
    return (
        pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame(), final,
        pd.concat(reasoning_rows, ignore_index=True) if reasoning_rows else pd.DataFrame(),
        pd.concat(assignment_rows, ignore_index=True) if assignment_rows else pd.DataFrame(),
    ), (pd.concat(catalog_rows, ignore_index=True) if catalog_rows else pd.DataFrame())


def _fit_control_meta(
    oof: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    seed: int,
    extra_features: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str], pd.Timestamp]:
    """Current residual meta control, fit strictly on per-row base OOF output."""
    required = ["p_adverse", "p_weak", "p_clear", "base_expected_bps", *FROZEN_META_CONTEXT, *extra_features]
    matrix_train = _as_float_matrix(oof, required)
    matrix_eval = _as_float_matrix(evaluate, required)
    target = (oof["net_bps"] - oof["base_expected_bps"]).to_numpy(np.float32)
    model = lgb.LGBMRegressor(objective="huber", alpha=0.9, random_state=seed, **META_PARAMS)
    model.fit(matrix_train, target)
    output = evaluate.copy()
    output["meta_residual_bps"] = model.predict(matrix_eval).astype(np.float32)
    output["score_base_bps"] = output["base_expected_bps"]
    output["score_base_meta_bps"] = output["base_expected_bps"] + output["meta_residual_bps"]
    cutoff = oof["label_available_ts"].max()
    if not cutoff < output["decision_ts"].min():
        raise PortabilityRunError("meta OOF labels are not all resolved before the outer evaluation")
    output["meta_fit_cutoff_ts"] = cutoff
    return output, required, cutoff


def _base_metric_summary(
    rows: pd.DataFrame, *, run: TransportRun, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the shared evaluator so score ranks once globally before splits."""
    result = evaluate_transport(
        # Side-local base contracts differ by design.  Their coverage is
        # audited separately before pooled evaluation; the only common global
        # ranking input here is the already causal-mapped bps score.
        rows, transport=run.spec(), score_column=score_column, feature_columns=[score_column],
        prior_resolved_columns=("base_fit_cutoff_ts", "base_map_cutoff_ts"),
        expected_cost_bps=TP6_SL4_COST_BPS, min_feature_coverage=0.99,
        top_fractions=TOP_FRACTIONS, enforce_h12=True,
    )
    return result.metrics, result.transport_gates


def _arm_frame(
    *,
    side: str,
    run: TransportRun,
    contract: TP6PortabilityContract,
    base_columns: Sequence[str],
    feature_manifest: pd.DataFrame,
    arm: str,
    f4_feature_contract: Mapping[str, Sequence[str]] | None = None,
    source_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], str]:
    """Load one side and generate only the feature representation requested."""
    columns = list(dict.fromkeys([*frozen_input_columns(contract)]))
    frame = (
        source_frame.copy()
        if source_frame is not None
        else load_tp6_population(
            contract=contract, columns=columns, start=run.train_start, end=run.eval_end,
            sides=(side,), valid_labels_only=True,
        )
    )
    contracts = _feature_contracts(frame, base_columns, feature_manifest)
    if arm not in {"F0_current_frozen", "F1_portable_raw", "F2_portable_plus_atr", "F3_plus_relative", "F4_transport_selected_compact"}:
        raise PortabilityRunError(f"unsupported sequential base arm: {arm}")
    if arm in {"F3_plus_relative", "F4_transport_selected_compact"}:
        # F3 is the explicit rehabilitation test: it may transform a raw
        # field that was not stable in its native units, but never an era
        # shortcut or rejected-lineage field.  The transform itself is causal
        # per asset and receives a fresh Stage-A/MDA gate later.
        disposition = feature_manifest.set_index("feature")["disposition"]
        base = [
            name for name in base_columns
            if str(disposition.get(name, "REJECTED_LINEAGE")) not in {"ERA_SHORTCUT", "REJECTED_LINEAGE"}
        ]
        if not base:
            raise PortabilityRunError(f"{side} has no lineage-safe source fields for F3")
        frame, generated = _add_causal_relative_transforms(frame, base)
        if arm == "F4_transport_selected_compact":
            if not isinstance(f4_feature_contract, Mapping) or set(f4_feature_contract) != set(SIDES):
                raise PortabilityRunError("selected F4 reasoning requires exact long/short compact feature lists")
            requested = [str(value) for value in f4_feature_contract[side]]
            if not requested or len(requested) != len(set(requested)):
                raise PortabilityRunError(f"selected F4 {side} feature list must be non-empty and unique")
            available = set(base).union(generated)
            unexpected = sorted(set(requested).difference(available))
            if unexpected:
                raise PortabilityRunError(
                    f"selected F4 {side} manifest is not a subset of the regenerated F3 contract: {unexpected[:8]}"
                )
            return frame, requested, "transport_selected_compact_f4_manifest_after_f0_f3_controls"
        return frame, [*base, *generated], "coverage_safe_causal_rank90_180_robust_z90_180_delta4_24_from_nonshortcut_sources"
    features, note = contracts[arm]
    return frame, features, note


def run_base_feature_funnel(
    *,
    destination: Path,
    contract: TP6PortabilityContract,
    feature_manifest: pd.DataFrame,
    seed: int,
    runs: Sequence[TransportRun] = DEVELOPMENT_TRANSPORTS,
    arms: Sequence[str] = ("F0_current_frozen", "F1_portable_raw", "F2_portable_plus_atr", "F3_plus_relative"),
) -> pd.DataFrame:
    """Stage A/F: F0--F3 use only frozen side-specific base controls.

    F2 can legitimately equal F1 if no selected raw level requires an ATR
    conversion; it is recorded as a tied representational control rather than
    refit needlessly.  F4 is deferred until its required grouped MDA is run
    after the feature-portability gate has been observed.
    """
    result_rows: list[pd.DataFrame] = []
    gate_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    supported_arms = {"F0_current_frozen", "F1_portable_raw", "F2_portable_plus_atr", "F3_plus_relative"}
    unknown_arms = sorted(set(arms).difference(supported_arms))
    if unknown_arms:
        raise PortabilityRunError(f"unsupported base arms: {unknown_arms}")
    if not runs or not arms:
        raise PortabilityRunError("base funnel requires at least one transport and arm")
    for run_index, run in enumerate(runs):
        for arm_index, arm in enumerate(arms):
            # F1/F2 are a global-book comparison.  If either side has no
            # Stage-A-approved source feature, fitting the other side would
            # create an incomparable one-sided book and waste four/five long
            # chronological fits.  Record the rejection before loading a
            # single panel instead of silently falling back to F0.
            if arm in {"F1_portable_raw", "F2_portable_plus_atr"}:
                approved = set(
                    feature_manifest.loc[
                        feature_manifest["disposition"].isin(
                            ("INVARIANT_RAW", "INVARIANT_NORMALIZED", "INVARIANT_RELATIVE", "KEEP_PORTABLE")
                        ), "feature",
                    ].astype(str)
                )
                preflight = {
                    side: [name for name in all_frozen_base_features(contract)[side] if name in approved]
                    for side in SIDES
                }
                absent = [side for side, fields in preflight.items() if not fields]
                if absent:
                    rejected_rows.append({
                        "arm": arm, "transport": run.name, "side_model": "side_local_models_global_pool",
                        "status": "NOT_RUN_NO_PORTABLE_FEATURES",
                        "reason": (
                            "Stage-A has no invariant raw features for side(s) "
                            f"{','.join(absent)}; an asymmetric global book was not invented"
                        ),
                    })
                    for side in SIDES:
                        fields = preflight[side]
                        lineage.append({
                            "arm": arm, "run": run.name, "side": side, "feature_count": len(fields),
                            "features": fields, "feature_signature": _sha256_json(fields),
                            "representation_note": "portable_raw_subset_preflight",
                            "reused_identical_contract": False, "oof_rows": 0,
                            "oof_materialised": False, "outer_eval_rows": 0,
                            "target": contract.target, "entry": contract.entry,
                        })
                    _write_base_funnel_checkpoint(
                        destination=destination, results=result_rows, gates=gate_rows,
                        coverage_rows=coverage_rows, rejected_rows=rejected_rows, lineage=lineage,
                        requested_runs=runs, requested_arms=arms,
                    )
                    continue
            # These are compact strict-OOF score rows (identity, economics,
            # score and frozen context), never the original wide feature
            # matrix.  They are necessary evidence for the shared transport
            # validator: it must see real prior-resolved training rows rather
            # than accept a test-only score surface.
            arm_oof: list[pd.DataFrame] = []
            arm_final: list[pd.DataFrame] = []
            arm_lineage: list[dict[str, object]] = []
            missing_side_contracts: list[str] = []
            for side_index, side in enumerate(SIDES):
                base = all_frozen_base_features(contract)[side]
                frame, features, note = _arm_frame(
                    side=side, run=run, contract=contract, base_columns=base,
                    feature_manifest=feature_manifest, arm=arm,
                )
                signature = _sha256_json(features)
                if not features:
                    missing_side_contracts.append(side)
                    rejected_rows.append({
                        "arm": arm, "transport": run.name, "side_model": side,
                        "status": "NOT_RUN_NO_PORTABLE_FEATURES",
                        "reason": "Stage-A has no invariant raw features for this side; no substitute was invented",
                    })
                    arm_lineage.append({
                        "arm": arm, "run": run.name, "side": side, "feature_count": 0,
                        "features": [], "feature_signature": signature, "representation_note": note,
                        "reused_identical_contract": False, "oof_rows": 0, "outer_eval_rows": 0,
                        "target": contract.target, "entry": contract.entry,
                    })
                    continue
                packed, catalog = _fold_base_rows(
                    frame=frame, run=run, feature_columns=features, side=side,
                    seed=seed + 1000 * run_index + 100 * side_index + arm_index,
                    artifact_root=None, retain_oof=True,
                )
                oof, final, _, _ = packed
                reused = False
                # ``final`` deliberately carries only score/context columns
                # to bound memory.  Coverage, however, is a property of the
                # original inference contract, so measure it on the raw
                # outer rows before releasing the side panel.
                test = frame.loc[
                    frame["decision_ts"].ge(_utc(run.eval_start)) & frame["decision_ts"].lt(_utc(run.eval_end))
                ]
                finite = np.isfinite(_as_float_matrix(test, features).to_numpy(float))
                for offset, feature in enumerate(features):
                    coverage_rows.append({
                        "arm": arm, "transport": run.name, "side_model": side, "feature": feature,
                        "test_rows": int(len(test)), "finite_coverage": float(finite[:, offset].mean()),
                        "passes_99pct_coverage": bool(finite[:, offset].mean() >= 0.99),
                    })
                arm_oof.append(oof); arm_final.append(final)
                arm_lineage.append({
                    "arm": arm, "run": run.name, "side": side, "feature_count": len(features),
                    "features": features, "feature_signature": signature, "representation_note": note,
                    "reused_identical_contract": reused,
                    "oof_rows": len(oof), "oof_materialised": True,
                    "outer_eval_rows": len(final),
                    "target": contract.target, "entry": contract.entry,
                })
                # ``oof``/``final`` are compact score/context frames.  Release
                # the full raw side panel and the wide F3 transform matrix
                # before loading/fitting the other side.
                del frame, packed, catalog
                gc.collect()
            if missing_side_contracts:
                lineage.extend(arm_lineage)
                _write_base_funnel_checkpoint(
                    destination=destination,
                    results=result_rows,
                    gates=gate_rows,
                    coverage_rows=coverage_rows,
                    rejected_rows=rejected_rows,
                    lineage=lineage,
                    requested_runs=runs,
                    requested_arms=arms,
                )
                continue
            # Rank once after both sides have been mapped to common bps.  The
            # helper's side/month/quarter rows describe this identical global
            # selection; they cannot create a side-local top-k book.
            # Inner OOF candidates are causal for their own decision, but an
            # observation made on the final pre-outer-test day resolves 13h
            # later.  It must not masquerade as fully resolved *outer train*
            # provenance in the transport validator.  This filter affects
            # only the audit input, not the strict OOF rows retained for the
            # subsequent selected-winner meta stage.
            outer_start = _utc(run.eval_start)
            resolved_oof = [
                table.loc[table["label_available_ts"].lt(outer_start)]
                for table in arm_oof
            ]
            scores = pd.concat([*resolved_oof, *arm_final], ignore_index=True)
            metrics, gates = _base_metric_summary(scores, run=run, score_column="base_expected_bps")
            coverage = pd.DataFrame(coverage_rows)
            arm_coverage = coverage.loc[
                coverage["arm"].eq(arm) & coverage["transport"].eq(run.name)
            ]
            coverage_pass = bool(arm_coverage["passes_99pct_coverage"].all())
            metrics["arm"] = arm; metrics["layer"] = "base"
            metrics["side_model"] = "side_local_models_global_pool"
            metrics["representation_note"] = ";".join(sorted({str(x["representation_note"]) for x in arm_lineage}))
            metrics["feature_contract_99pct_pass"] = coverage_pass
            metrics["arm_status"] = "ELIGIBLE" if coverage_pass else "REJECTED_FEATURE_COVERAGE"
            gates["arm"] = arm; gates["layer"] = "base"
            gates["feature_contract_99pct_pass"] = coverage_pass
            result_rows.append(metrics); gate_rows.append(gates)
            lineage.extend(arm_lineage)
            # A resource termination must not erase completed arms.  These
            # files are intentionally overwritten *within the newly created
            # destination only*, and always represent the completed prefix.
            _write_base_funnel_checkpoint(
                destination=destination,
                results=result_rows,
                gates=gate_rows,
                coverage_rows=coverage_rows,
                rejected_rows=rejected_rows,
                lineage=lineage,
                requested_runs=runs,
                requested_arms=arms,
            )
            del scores, metrics, gates, resolved_oof, arm_oof, arm_final
            gc.collect()
    results = _write_base_funnel_checkpoint(
        destination=destination,
        results=result_rows,
        gates=gate_rows,
        coverage_rows=coverage_rows,
        rejected_rows=rejected_rows,
        lineage=lineage,
        requested_runs=runs,
        requested_arms=arms,
    )
    return results


def _write_base_funnel_checkpoint(
    *,
    destination: Path,
    results: Sequence[pd.DataFrame],
    gates: Sequence[pd.DataFrame],
    coverage_rows: Sequence[dict[str, object]],
    rejected_rows: Sequence[dict[str, object]],
    lineage: Sequence[dict[str, object]],
    requested_runs: Sequence[TransportRun],
    requested_arms: Sequence[str],
) -> pd.DataFrame:
    """Persist the exact completed prefix of a sharded base funnel."""
    # Empty F1/F2 shards are valid evidence: a global pooled arm cannot run
    # when one side has no Stage-A-approved feature.  Give those empty metric
    # tables a stable schema so Parquet and the immutable merger can represent
    # the rejection rather than treating it as a runner failure.
    result_table = pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["transport", "arm", "scope", "period", "side_name", "top_fraction"]
    )
    gate_table = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame(
        columns=["transport", "arm", "scope", "period", "side_name", "top_fraction", "gate"]
    )
    result_table.to_parquet(destination / "base_feature_ablation_results.parquet", index=False, compression="zstd")
    gate_table.to_parquet(destination / "base_feature_transport_gates.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage_rows).to_parquet(
        destination / "base_feature_contract_coverage.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(rejected_rows).to_parquet(
        destination / "base_feature_rejected_arms.parquet", index=False, compression="zstd"
    )
    _write_json(destination / "base_feature_arm_lineage.json", list(lineage))
    _write_json(destination / "base_feature_funnel_progress.json", {
        "requested_transports": [item.name for item in requested_runs],
        "requested_arms": list(requested_arms),
        "completed_arm_transport_pairs": sorted({
            f"{str(row['transport'])}:{str(row['arm'])}"
            for table in results
            for _, row in table.loc[:, ["transport", "arm"]].drop_duplicates().iterrows()
        }),
        "result_rows": int(len(result_table)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return result_table


def materialize_selected_base_reasoning(
    *,
    destination: Path,
    contract: TP6PortabilityContract,
    feature_manifest: pd.DataFrame,
    winner: str,
    seed: int,
    runs: Sequence[TransportRun] = DEVELOPMENT_TRANSPORTS,
    f4_feature_contract: Mapping[str, Sequence[str]] | None = None,
    resume_existing: bool = False,
) -> pd.DataFrame:
    """Write strict-OOF G1/G2/G3 artifacts for the selected base contract.

    Every head/fold artifact is written atomically by the strict materialiser.
    This coordinator deliberately keeps only one side panel and one fold's
    temporary reasoning state in memory.  Prediction shards remain compact
    and are the only inputs a later per-row meta fit may read; leaf artifacts
    are indexed by path rather than concatenated into an unsafe mega-frame.
    """
    supported = {"F0_current_frozen", "F1_portable_raw", "F2_portable_plus_atr", "F3_plus_relative", "F4_transport_selected_compact"}
    if winner not in supported:
        raise PortabilityRunError(
            f"strict reasoning requires a selected frozen F0--F4 contract, got {winner!r}"
        )
    if winner == "F4_transport_selected_compact" and f4_feature_contract is None:
        raise PortabilityRunError("strict reasoning refuses F4 without its selected compact manifest field lists")
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    heads = (("p_adverse", 0), ("p_weak", 1), ("p_clear", 2))
    if not runs:
        raise PortabilityRunError("strict reasoning materialisation requires at least one transport")
    for run_index, run in enumerate(runs):
        run_root = destination / "strict_oof_base_reasoning" / run.name
        for side_index, side in enumerate(SIDES):
            base = all_frozen_base_features(contract)[side]
            frame, features, note = _arm_frame(
                side=side, run=run, contract=contract, base_columns=base,
                feature_manifest=feature_manifest, arm=winner, f4_feature_contract=f4_feature_contract,
            )
            side_seed = int(seed + 10_000 * run_index + 1_000 * side_index)
            packed, _ = _fold_base_rows(
                frame=frame, run=run, feature_columns=features, side=side,
                seed=side_seed, artifact_root=run_root, retain_oof=True,
                retain_reasoning_payload=False,
                resume_existing_reasoning=resume_existing,
            )
            oof, final, _, _ = packed
            # The explicit layer/seed provenance travels with the compact
            # score rows, while per-head rule manifests record model hashes
            # and exact class-strided selected-tree indices.
            oof = oof.copy()
            final = final.copy()
            for table in (oof, final):
                table["layer"] = "base"
                table["base_representation"] = winner
                table["feature_contract_sha256"] = _sha256_json(features)
                table["base_model_seed"] = np.where(
                    table["fold_id"].astype(str).str.endswith("_outer"),
                    side_seed + 10_000,
                    side_seed + table["fold_id"].astype(str).str.extract(r"(\d+)$")[0].fillna(0).astype(int),
                ).astype(np.int32)
            output_root = destination / "base_prediction_shards" / run.name / side
            output_root.mkdir(parents=True, exist_ok=True)
            oof.to_parquet(output_root / "strict_oof_predictions.parquet", index=False, compression="zstd")
            final.to_parquet(output_root / "outer_predictions.parquet", index=False, compression="zstd")

            outer = frame.loc[
                frame["decision_ts"].ge(_utc(run.eval_start)) & frame["decision_ts"].lt(_utc(run.eval_end))
            ]
            finite = np.isfinite(_as_float_matrix(outer, features).to_numpy(float))
            coverage_rows.extend({
                "transport": run.name, "side_model": side, "feature": feature,
                "outer_rows": int(len(outer)), "finite_coverage": float(finite[:, offset].mean()),
                "passes_99pct_coverage": bool(finite[:, offset].mean() >= 0.99),
            } for offset, feature in enumerate(features))

            fold_names = [f"inner_{index:02d}" for index, _ in enumerate(_inner_windows(run))] + ["outer"]
            for fold_slot, fold_name in enumerate(fold_names):
                fold_dir = run_root / "folds" / side / fold_name
                for head, class_index in heads:
                    head_dir = fold_dir / head
                    manifest_path = head_dir / "base_reasoning_manifest.json"
                    if not manifest_path.exists():
                        raise PortabilityRunError(f"strict reasoning artifact is missing: {manifest_path}")
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    rows.append({
                        "transport": run.name, "side_model": side, "layer": "base_reasoning",
                        "head_name": head, "class_index": class_index,
                        "fold_name": fold_name, "fold_id": f"{run.name}_{'outer' if fold_name == 'outer' else fold_name}",
                        "artifact_dir": str(head_dir.relative_to(destination)),
                        "base_representation": winner, "representation_note": note,
                        "feature_count": int(len(features)),
                        "feature_contract_sha256": _sha256_json(features),
                        "runner_side_seed": int(side_seed + (10_000 if fold_name == "outer" else fold_slot)),
                        "strict_status": str(payload.get("status", "")),
                        "train_start_utc": payload.get("provenance", {}).get("train_start_utc"),
                        "train_end_utc": payload.get("provenance", {}).get("train_end_utc"),
                        "eval_start_utc": payload.get("provenance", {}).get("eval_start_utc"),
                        "eval_end_utc": payload.get("provenance", {}).get("eval_end_utc"),
                    })
            del outer, finite, oof, final, frame, packed
            gc.collect()
    index = pd.DataFrame(rows)
    index.to_parquet(destination / "strict_oof_reasoning_artifact_index.parquet", index=False, compression="zstd")
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_parquet(destination / "strict_oof_reasoning_feature_coverage.parquet", index=False, compression="zstd")
    if coverage.empty or not coverage["passes_99pct_coverage"].all():
        raise PortabilityRunError("selected reasoning representation violates the 99% outer coverage gate")
    _write_json(destination / "strict_oof_reasoning_manifest.json", {
        "status": "STRICT_OOF_BASE_REASONING_MATERIALIZED",
        "winner": winner,
        "transports": [item.name for item in runs],
        "heads": [{"head_name": name, "class_index": index} for name, index in heads],
        "artifact_index": "strict_oof_reasoning_artifact_index.parquet",
        "prediction_shards": "base_prediction_shards/<transport>/<side>/",
        "leaf_policy": "opaque leaf tokens remain scoped to side/head/fold/model/tree; only G2 rule signatures may recur across folds",
        "coverage_gate": "all frozen winner features >=99% finite on each outer side/transport",
    })
    return index


def select_base_feature_winner(results: pd.DataFrame, *, destination: Path) -> dict[str, object]:
    """Choose a development representation without looking at final OOS.

    Selection is lexicographic: full 99%-coverage eligibility in both
    transports, no catastrophic top-10 net transport, then median top-10 net
    and top-5 net.  The table is stored even when no arm advances.
    """
    eligible = results.loc[
        results["scope"].eq("global")
        & results["top_fraction"].isin((0.05, 0.10))
    ].copy()
    rows: list[dict[str, object]] = []
    for arm, local in eligible.groupby("arm", sort=True, observed=True):
        top10 = local.loc[local["top_fraction"].eq(0.10)]
        top5 = local.loc[local["top_fraction"].eq(0.05)]
        expected = {item.name for item in DEVELOPMENT_TRANSPORTS}
        coverage = bool(top10["feature_contract_99pct_pass"].all()) if not top10.empty else False
        has_both = set(top10["transport"]) == expected
        rows.append({
            "arm": str(arm), "has_both_transports": has_both, "coverage_pass": coverage,
            "top10_median_net_bps": float(top10["net_bps_per_trade"].median()) if not top10.empty else float("nan"),
            "top10_worst_net_bps": float(top10["net_bps_per_trade"].min()) if not top10.empty else float("nan"),
            "top5_median_net_bps": float(top5["net_bps_per_trade"].median()) if not top5.empty else float("nan"),
            "top5_worst_net_bps": float(top5["net_bps_per_trade"].min()) if not top5.empty else float("nan"),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        decision = {"status": "NO_BASE_FEATURE_ARM_COMPLETED", "winner": None}
    else:
        candidates = table.loc[table["has_both_transports"] & table["coverage_pass"]].copy()
        if candidates.empty:
            decision = {
                "status": "NO_BASE_FEATURE_ARM_ADVANCES_COVERAGE_GATE",
                "winner": None,
                "fallback_for_diagnostic_reasoning_only": "F0_current_frozen",
            }
        else:
            ranked = candidates.sort_values(
                ["top10_worst_net_bps", "top10_median_net_bps", "top5_worst_net_bps"],
                ascending=False, kind="stable",
            )
            winner = str(ranked.iloc[0]["arm"])
            decision = {
                "status": "BASE_FEATURE_WINNER_SELECTED_ON_DEVELOPMENT_ONLY",
                "winner": winner,
                "selection_order": ["top10_worst_net_bps", "top10_median_net_bps", "top5_worst_net_bps"],
            }
    table.to_parquet(destination / "base_feature_selection_summary.parquet", index=False, compression="zstd")
    _write_json(destination / "base_feature_selection_decision.json", decision)
    return decision


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _prepare_destination(path: Path, *, resume: bool = False) -> bool:
    """Create a new run directory or explicitly reopen one interrupted strict run.

    Returning ``True`` means the caller must retain the original manifest and
    Stage-A copy.  Resume is deliberately opt-in and limited by ``main`` to
    the strict reasoning stage; base/audit arms retain the normal immutable
    new-destination rule.
    """
    if path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite an experiment artifact: {path}")
        if not path.is_dir():
            raise PortabilityRunError(f"resume destination is not a directory: {path}")
        manifest_path = path / "run_manifest.json"
        if not manifest_path.exists():
            raise PortabilityRunError(f"resume destination lacks run_manifest.json: {path}")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PortabilityRunError(f"resume destination has an unreadable manifest: {path}") from exc
        if payload.get("schema") != SCHEMA or payload.get("status") != "RUNNING":
            raise PortabilityRunError(
                "resume accepts only an interrupted RUNNING strict-OOF destination; "
                "completed or incompatible artifacts are immutable"
            )
        return True
    if resume:
        raise PortabilityRunError(f"--resume requires an existing interrupted destination: {path}")
    path.mkdir(parents=True)
    return False


def _load_certified_f4_feature_contract(path: Path) -> dict[str, list[str]]:
    """Read only the immutable, F0/F3-controlled F4 reasoning contract.

    This deliberately validates the small selector artifact chain rather than
    trusting a field-list JSON supplied by a caller.  It does *not* make F4 a
    meta default: the manifest's separate meta-control gate remains pending
    until the L0/F4 compact ablation is complete.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "F4_TRANSPORT_SELECTED_COMPACT_FEATURE_MANIFEST":
        raise PortabilityRunError("strict reasoning accepts only a successfully selected F4 compact feature manifest")
    if not bool(payload.get("base_control_verified")) or not bool(payload.get("full_f3_control_verified")):
        raise PortabilityRunError("F4 manifest has not verified its frozen F0 and full-F3 base controls")
    f3_eligible = payload.get("full_f3_control_eligible")
    f3_status = payload.get("full_f3_control_status")
    if not isinstance(f3_eligible, bool):
        raise PortabilityRunError("F4 manifest must state whether full F3 was coverage-eligible")
    if f3_eligible and f3_status != "ELIGIBLE_NONINFERIORITY_PASSED":
        raise PortabilityRunError("coverage-eligible full F3 lacks a verified non-inferiority status")
    if not f3_eligible and f3_status != "FULL_F3_DIAGNOSTIC_INELIGIBLE_COVERAGE_NOT_A_PROMOTION_GATE":
        raise PortabilityRunError("coverage-ineligible full F3 lacks the required diagnostic-only status")
    if payload.get("meta_control_gate") is None:
        raise PortabilityRunError("F4 manifest lacks the required later meta-control gate")
    selection_artifact = payload.get("selection_artifact")
    expected_selection = Path(path).parent / "f4_selected_feature_contract.json"
    if (
        not isinstance(selection_artifact, Mapping)
        or selection_artifact.get("path") != expected_selection.name
        or not expected_selection.exists()
        or selection_artifact.get("sha256") != _sha256_file(expected_selection)
    ):
        raise PortabilityRunError("F4 compact manifest is not cryptographically linked to its immutable development selection")
    selected_payload = json.loads(expected_selection.read_text(encoding="utf-8"))
    if selected_payload.get("representation") != payload.get("selected_representation"):
        raise PortabilityRunError("F4 compact manifest selection does not match its selection artifact")
    expected_run_manifest = Path(path).parent / "f4_run_manifest.json"
    if not expected_run_manifest.exists():
        raise PortabilityRunError("F4 compact manifest lacks its immutable F4 run manifest")
    f4_run = json.loads(expected_run_manifest.read_text(encoding="utf-8"))
    if (
        f4_run.get("status") != "F4_FEATURE_CONTRACT_SELECTED"
        or f4_run.get("compact_manifest_status") != "F4_TRANSPORT_SELECTED_COMPACT_FEATURE_MANIFEST"
    ):
        raise PortabilityRunError("F4 run manifest does not certify a stable selected compact contract")
    candidate = payload.get("feature_contract")
    if not isinstance(candidate, Mapping) or set(candidate) != set(SIDES):
        raise PortabilityRunError("F4 manifest lacks exact long/short compact feature lists")
    result = {side: [str(field) for field in candidate[side]] for side in SIDES}
    if any(not values or len(values) != len(set(values)) for values in result.values()):
        raise PortabilityRunError("F4 manifest has an empty or duplicate compact side feature list")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new destination under data_perp/artifacts")
    parser.add_argument("--stage", choices=("audit", "base", "reasoning", "all"), default="all")
    parser.add_argument("--audit-dir", type=Path, help="completed Stage-A directory to reuse for --stage base")
    parser.add_argument(
        "--base-funnel-dir", type=Path,
        help="merged development-only F0--F3 selection required for --stage reasoning",
    )
    parser.add_argument(
        "--f4-feature-manifest", type=Path,
        help="selected portable_feature_manifest.json; accepted only for --stage reasoning after F0/F3-controlled F4 selection",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--transport", action="append", choices=[item.name for item in DEVELOPMENT_TRANSPORTS],
        help="run one or more exact development transports in this fresh shard; default is both",
    )
    parser.add_argument(
        "--arm", action="append",
        choices=("F0_current_frozen", "F1_portable_raw", "F2_portable_plus_atr", "F3_plus_relative"),
        help="run one or more sequential base arms in this fresh shard; default is all",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume only an interrupted --stage reasoning destination after exact per-head integrity checks",
    )
    args = parser.parse_args()
    if args.f4_feature_manifest is not None and args.stage != "reasoning":
        raise PortabilityRunError("--f4-feature-manifest is valid only for explicit --stage reasoning consumption")
    if args.resume and args.stage != "reasoning":
        raise PortabilityRunError("--resume is intentionally supported only for --stage reasoning")
    resuming = _prepare_destination(args.out, resume=bool(args.resume))
    contract = TP6PortabilityContract()
    if not resuming:
        _write_json(args.out / "run_manifest.json", {
            "schema": SCHEMA,
            "status": "RUNNING",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_contract": _source_contract_payload(contract),
            "entry": "bar close decision -> next hourly open entry",
            "target_horizon_hours_from_entry": 12,
            "label_resolution_hours_from_decision": 13,
            "cost_bps_exactly_once": TP6_SL4_COST_BPS,
            "global_ranking": "pooled across side and timestamp after common-bps mapping",
            "transports": [asdict(item) for item in DEVELOPMENT_TRANSPORTS],
            "final_untouched_oos": asdict(UNTOUCHED_OOS),
            "stages": {"A": "pending", "B_to_E": "pending", "final_oos": "pending"},
        })
    if resuming:
        audit_path = args.out / "feature_portability_dispositions.parquet"
        if not audit_path.exists():
            raise PortabilityRunError("resume destination lacks its frozen Stage-A dispositions")
        status = json.loads((args.out / "run_manifest.json").read_text(encoding="utf-8"))
        if args.audit_dir is not None and status.get("stage_a_reused_from") != str(args.audit_dir):
            raise PortabilityRunError(
                "resume --audit-dir does not match the immutable Stage-A source recorded by the interrupted run"
            )
        if status.get("source_contract") != _source_contract_payload(contract):
            raise PortabilityRunError("resume source contract differs from the interrupted strict-OOF run")
        manifest = pd.read_parquet(audit_path)
        resume_events = list(status.get("resume_events", []))
        resume_events.append({
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "reasoning",
            "mode": "per_head_hash_model_identity_prediction_label_verified",
        })
        status["resume_events"] = resume_events
        _write_json(args.out / "run_manifest.json", status)
    elif args.stage in {"base", "reasoning"} and args.audit_dir is not None:
        source = args.audit_dir
        audit_path = source / "feature_portability_dispositions.parquet"
        if not audit_path.exists():
            raise PortabilityRunError(f"--audit-dir lacks completed dispositions: {audit_path}")
        manifest = pd.read_parquet(audit_path)
        # Keep a self-contained experiment directory while recording the
        # immutable source instead of recomputing a broad Stage-A scan.
        for name in (
            "feature_portability_audit.parquet", "feature_portability_era_audit.parquet",
            "feature_portability_dispositions.parquet", "feature_portability_role_disposition_manifest.json",
            "feature_role_manifest.csv", "feature_role_manifest.yaml", "portable_feature_manifest.json",
        ):
            candidate = source / name
            if candidate.exists():
                shutil.copy2(candidate, args.out / name)
        status = json.loads((args.out / "run_manifest.json").read_text())
        status["stage_a_reused_from"] = str(source)
        _write_json(args.out / "run_manifest.json", status)
    else:
        _, manifest = run_feature_audit(destination=args.out, contract=contract)
    if args.stage == "audit":
        status = json.loads((args.out / "run_manifest.json").read_text())
        status["status"] = "STAGE_A_COMPLETED"
        status["stages"]["A"] = "completed"
        _write_json(args.out / "run_manifest.json", status)
        return
    if args.stage == "reasoning":
        f4_feature_contract: Mapping[str, Sequence[str]] | None = None
        if args.f4_feature_manifest is not None:
            f4_feature_contract = _load_certified_f4_feature_contract(args.f4_feature_manifest)
            winner = "F4_transport_selected_compact"
        else:
            if args.base_funnel_dir is None:
                raise PortabilityRunError("--stage reasoning requires --base-funnel-dir or --f4-feature-manifest")
            decision_path = args.base_funnel_dir / "base_feature_selection_decision.json"
            if not decision_path.exists():
                raise PortabilityRunError(f"--base-funnel-dir lacks development selection: {decision_path}")
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            winner = decision.get("winner")
            if decision.get("status") != "BASE_FEATURE_WINNER_SELECTED_ON_DEVELOPMENT_ONLY" or not winner:
                raise PortabilityRunError("reasoning requires a completed development-only base winner")
        run_lookup = {item.name: item for item in DEVELOPMENT_TRANSPORTS}
        selected_runs = tuple(run_lookup[name] for name in (args.transport or list(run_lookup)))
        index = materialize_selected_base_reasoning(
            destination=args.out, contract=contract, feature_manifest=manifest,
            winner=str(winner), seed=args.seed, runs=selected_runs, f4_feature_contract=f4_feature_contract,
            resume_existing=resuming,
        )
        status = json.loads((args.out / "run_manifest.json").read_text())
        full_reasoning = {item.name for item in selected_runs} == {item.name for item in DEVELOPMENT_TRANSPORTS}
        status["status"] = "STRICT_OOF_BASE_REASONING_COMPLETED" if full_reasoning else "STRICT_OOF_BASE_REASONING_SHARD_COMPLETED"
        status["stages"]["A"] = "completed"
        status["stages"]["B_to_E"] = (
            "strict_oof_g1_g2_g3_materialized; health_cluster_covariance_pending"
            if full_reasoning else "strict_oof_g1_g2_g3_transport_shard_materialized; transport-union analyses pending"
        )
        status["base_feature_selection_source"] = (
            str(args.f4_feature_manifest) if args.f4_feature_manifest is not None else str(args.base_funnel_dir)
        )
        status["base_feature_winner"] = winner
        status["strict_oof_reasoning_artifact_rows"] = int(len(index))
        status["strict_oof_reasoning_transports"] = [item.name for item in selected_runs]
        _write_json(args.out / "run_manifest.json", status)
        return
    run_lookup = {item.name: item for item in DEVELOPMENT_TRANSPORTS}
    selected_runs = tuple(run_lookup[name] for name in (args.transport or list(run_lookup)))
    selected_arms = tuple(args.arm or ("F0_current_frozen", "F1_portable_raw", "F2_portable_plus_atr", "F3_plus_relative"))
    base_results = run_base_feature_funnel(
        destination=args.out, contract=contract, feature_manifest=manifest, seed=args.seed,
        runs=selected_runs, arms=selected_arms,
    )
    full_funnel = (
        {item.name for item in selected_runs} == {item.name for item in DEVELOPMENT_TRANSPORTS}
        and set(selected_arms) == {"F0_current_frozen", "F1_portable_raw", "F2_portable_plus_atr", "F3_plus_relative"}
    )
    base_decision = (
        select_base_feature_winner(base_results, destination=args.out)
        if full_funnel
        else {
            "status": "BASE_FEATURE_SHARD_COMPLETED_NO_SELECTION",
            "winner": None,
            "reason": "Selection waits for the immutable union of all requested development transport shards",
        }
    )
    status = json.loads((args.out / "run_manifest.json").read_text())
    status["status"] = (
        "STAGE_A_AND_BASE_FEATURE_FUNNEL_COMPLETED"
        if full_funnel else "BASE_FEATURE_FUNNEL_SHARD_COMPLETED"
    )
    status["stages"]["A"] = "completed"
    status["stages"]["B_to_E"] = (
        "base_feature_f0_to_f3_completed; reasoning_meta_pending"
        if full_funnel else "base_feature_shard_completed; immutable transport-union selection pending"
    )
    status["base_feature_result_rows"] = int(len(base_results))
    status["base_feature_decision"] = base_decision
    status["base_feature_shard"] = {
        "transports": [item.name for item in selected_runs], "arms": list(selected_arms),
    }
    _write_json(args.out / "run_manifest.json", status)


if __name__ == "__main__":
    main()
