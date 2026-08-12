"""Immutable grouped chronological MDA for leaf-reasoning funnel outputs.

The meta funnel deliberately does not perform feature selection.  Its output
is development evidence only, and this sidecar supplies the missing evidence
required to advance a non-control arm: a transport-local, chronological,
joint-feature permutation test on the same global common-bps top-k decision
surface.  It never opens final OOS and it never modifies the sealed funnel
directory it audits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .leaf_reasoning_meta_funnel import (
    ArmSpec,
    FrozenMetaModelSpec,
    MetaFunnelColumns,
    MetaFunnelConfig,
    MetaFunnelError,
    MetaModelFactory,
    MetaTransportGateConfig,
    _attach_transport_ids,
    _fixed_lgbm_factory,
    _matrix,
    _ranking_tie_columns,
    _train_median_impute,
    build_meta_ablation_gates,
    validate_base_oof_rows,
)


SCHEMA = "leaf_reasoning_grouped_transport_mda_v1"
_IDENTITY = (
    "candidate_id", "decision_ts", "side_name",
    "__strict_fold_id__", "__strict_transport__", "__strict_meta_partition__",
)


class LeafReasoningGroupedMDAError(ValueError):
    """Raised when post-funnel MDA evidence cannot be proved strict."""


@dataclass(frozen=True)
class GroupedMDAConfig:
    """Pre-declared cost-aware MDA contract; no tuning surface is exposed."""

    repeats: int = 3
    phantom_draws: int = 8
    top_fraction: float = 0.10
    random_seed: int = 20260805

    def __post_init__(self) -> None:
        if self.repeats < 2:
            raise LeafReasoningGroupedMDAError("grouped MDA requires at least two real permutations")
        if self.phantom_draws < 8:
            raise LeafReasoningGroupedMDAError("grouped MDA requires at least eight phantom draws for a q95 gate")
        if not 0.0 < self.top_fraction <= 1.0:
            raise LeafReasoningGroupedMDAError("top_fraction must be in (0, 1]")


@dataclass(frozen=True)
class GroupedMDAResult:
    summary: pd.DataFrame
    real_repeats: pd.DataFrame
    phantom_draws: pd.DataFrame
    advancement: pd.DataFrame
    gates: pd.DataFrame
    source_manifest: Mapping[str, Any]
    source_root: Path
    config: GroupedMDAConfig
    cluster_source: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _FittedMeta:
    model: Any
    features: tuple[str, ...]
    medians: np.ndarray | None
    lightgbm_native_nan: bool


def _ledger_aliases(columns: MetaFunnelColumns) -> Mapping[str, tuple[str, ...]]:
    """Keep the projected reader aligned with ``validate_base_oof_rows``.

    The normalizer is deliberately still the authority for aliases and all
    provenance checks.  This map merely prevents a parquet projection from
    dropping an accepted spelling before that normalizer can see it.
    """

    return {
        columns.decision: ("__decision_ts__", "__ts__"),
        columns.label_available: ("__label_available_at__", "label_available_at"),
        columns.realized_gross_bps: ("gross_bps",),
        columns.realized_cost_bps: ("cost_bps",),
        columns.realized_net_bps: ("net_bps", "exact_net_bps"),
        columns.base_fit_end: ("base_fit_cutoff_ts", "prediction_fit_end_ts"),
        columns.base_generated: ("feature_generation_ts", "base_map_cutoff_ts", "prediction_generated_ts"),
        columns.base_strict_oof: ("strict_oof", "is_strict_oof", "strict_prequential_oof"),
    }


def _projected_ledger_columns(
    available: set[str], *, features: Sequence[str], columns: MetaFunnelColumns,
    config: MetaFunnelConfig,
) -> list[str]:
    """Return the smallest source projection that can still be fully audited."""

    requested = {
        columns.candidate_id, columns.side, columns.decision, columns.label_available,
        columns.base_expected_bps, columns.realized_gross_bps, columns.realized_cost_bps,
        columns.realized_net_bps, columns.base_fit_end, columns.base_generated,
        columns.base_strict_oof, columns.fold_id, config.transport_column,
        config.meta_partition_column, *map(str, features),
    }
    result: set[str] = set()
    aliases = _ledger_aliases(columns)
    for name in requested:
        if name in available:
            result.add(name)
            continue
        # Only the validated canonical fields have aliases.  Missing feature
        # columns remain absent so the normal strict error is preserved.
        for alternative in aliases.get(name, ()):
            if alternative in available:
                result.add(alternative)
                break
    return sorted(result)


@dataclass(frozen=True)
class _LedgerReader:
    """Narrow, transport-local ledger reader.

    The old implementation validated a 7.3m-row, very-wide parquet ledger in
    one dataframe before discovering which selected feature contracts were
    actually needed.  This reader retains the same validation function but
    only materializes the current transport and current arm's declared fields.
    """

    source: pd.DataFrame | Path
    config: MetaFunnelConfig
    columns: MetaFunnelColumns
    available: frozenset[str]
    cluster_features: "_ClusterFeatureReader | None" = None

    @classmethod
    def create(cls, source: pd.DataFrame | str | Path, *, config: MetaFunnelConfig,
               columns: MetaFunnelColumns, cluster_features: "_ClusterFeatureReader | None" = None) -> "_LedgerReader":
        if isinstance(source, pd.DataFrame):
            return cls(source, config, columns, frozenset(map(str, source.columns)), cluster_features)
        path = Path(source)
        if path.suffix.lower() != ".parquet":
            raise LeafReasoningGroupedMDAError(
                "memory-safe grouped MDA requires a parquet ledger path; pass a dataframe only for in-memory diagnostics"
            )
        if not path.is_file():
            raise LeafReasoningGroupedMDAError(f"MDA ledger does not exist: {path}")
        return cls(path, config, columns, frozenset(map(str, pq.ParquetFile(path).schema_arrow.names)), cluster_features)

    def transports(self) -> tuple[str, ...]:
        name = self.config.transport_column
        if name not in self.available:
            # Preserve the legacy normalization behavior for in-memory unit
            # diagnostics; a production parquet without transport cannot
            # prove the required transport-local contract.
            if isinstance(self.source, pd.DataFrame):
                return ("UNSPECIFIED",)
            raise LeafReasoningGroupedMDAError("MDA parquet ledger lacks required transport_id column")
        if isinstance(self.source, pd.DataFrame):
            values = self.source[name]
        else:
            # A full string Series for a multi-million-row ledger is an
            # avoidable allocation before the actual arm-local scan starts.
            seen: set[str] = set()
            for batch in pq.ParquetFile(self.source).iter_batches(columns=[name], batch_size=131_072):
                values = batch.column(0).to_pylist()
                if any(value is None or not str(value).strip() for value in values):
                    raise LeafReasoningGroupedMDAError("transport_id must be non-null and non-empty when supplied")
                seen.update(map(str, values))
            return tuple(sorted(seen))
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise LeafReasoningGroupedMDAError("transport_id must be non-null and non-empty when supplied")
        return tuple(sorted(values.astype(str).unique().tolist()))

    def read_transport(self, transport: str, *, features: Sequence[str], side: str | None = None) -> pd.DataFrame:
        local_features = tuple(name for name in features if name in self.available)
        external_features = tuple(name for name in features if name not in self.available)
        projection = _projected_ledger_columns(
            set(self.available), features=local_features, columns=self.columns, config=self.config,
        )
        if isinstance(self.source, pd.DataFrame):
            raw = self.source.loc[:, projection]
            if self.config.transport_column in raw:
                raw = raw.loc[raw[self.config.transport_column].astype(str).eq(str(transport))]
            if side is not None:
                raw = raw.loc[raw[self.columns.side].astype(str).eq(str(side))]
            normalized = self._validate(raw)
            return self._attach_external(normalized, transport, side, external_features)
        if self.config.transport_column not in projection:
            raise LeafReasoningGroupedMDAError("MDA parquet ledger lacks required transport_id column")
        filters: list[tuple[str, str, str]] = [(self.config.transport_column, "==", str(transport))]
        if side is not None:
            filters.append((self.columns.side, "==", str(side)))
        raw = pd.read_parquet(self.source, columns=projection, filters=filters)
        if raw.empty:
            raise LeafReasoningGroupedMDAError(f"MDA ledger has no rows for transport {transport}")
        normalized = self._validate(raw)
        return self._attach_external(normalized, transport, side, external_features)

    def _attach_external(
        self, normalized: pd.DataFrame, transport: str, side: str | None,
        external_features: Sequence[str],
    ) -> pd.DataFrame:
        if not external_features:
            return normalized
        if self.cluster_features is None:
            return normalized
        return self.cluster_features.join(
            normalized, transport=str(transport), side=side, features=external_features,
        )

    def _validate(self, raw: pd.DataFrame) -> pd.DataFrame:
        try:
            normalized = validate_base_oof_rows(raw, columns=self.columns, config=self.config)
            return _attach_transport_ids(normalized, config=self.config)
        except MetaFunnelError as exc:
            raise LeafReasoningGroupedMDAError(str(exc)) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_CLUSTER_NATIVE_IDENTITY = (
    "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
)


@dataclass(frozen=True)
class _ClusterFeatureReader:
    """Hash-bound, projected hand-off from an immutable C candidate root.

    Cluster columns intentionally do not live in the wide base-to-meta ledger.
    Loading the whole candidate table would defeat the bounded MDA reader, so
    this object proves the immutable source once and reads only the current
    transport/side/arm columns before an exact strict-identity join.
    """

    root: Path
    table: Path
    available: frozenset[str]
    source_binding: Mapping[str, Any]

    @classmethod
    def create(
        cls, root: str | Path, *, funnel_manifest: Mapping[str, Any], arms: Sequence[ArmSpec],
    ) -> "_ClusterFeatureReader":
        source = Path(root).resolve()
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise LeafReasoningGroupedMDAError(f"cluster root lacks manifest.json: {source}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LeafReasoningGroupedMDAError("cluster manifest is not valid JSON") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema") != "leaf_reasoning_candidate_cluster_materializer_v1" or manifest.get("status") != "STRICT_OOF_CANDIDATE_CLUSTER_FEATURES_MATERIALIZED":
            raise LeafReasoningGroupedMDAError("cluster root is not a completed immutable candidate-cluster artifact")
        outputs = manifest.get("outputs")
        required = {
            "candidate_cluster_features.parquet", "cluster_groups.json",
            "cluster_taxonomy_contract.json", "cluster_feature_manifest.json",
        }
        if not isinstance(outputs, Mapping) or not required.issubset(outputs):
            raise LeafReasoningGroupedMDAError("cluster manifest lacks required immutable feature-contract hashes")
        # Hash every declared output rather than trusting a mutable sidecar
        # merely because the selected table itself happens to hash correctly.
        for name, expected in outputs.items():
            path = source / str(name)
            if not isinstance(expected, str) or not path.is_file() or _sha256_file(path) != expected:
                raise LeafReasoningGroupedMDAError(f"cluster artifact hash is missing or invalid for {name}")
        try:
            groups = json.loads((source / "cluster_groups.json").read_text(encoding="utf-8"))
            taxonomy = json.loads((source / "cluster_taxonomy_contract.json").read_text(encoding="utf-8"))
            feature_manifest = json.loads((source / "cluster_feature_manifest.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LeafReasoningGroupedMDAError("cluster feature-contract JSON is invalid") from exc
        if not isinstance(groups, Mapping) or not isinstance(taxonomy, Mapping) or not isinstance(feature_manifest, Mapping):
            raise LeafReasoningGroupedMDAError("cluster feature-contract JSON has an invalid shape")
        mapping = feature_manifest.get("cluster_id_to_feature")
        if feature_manifest.get("schema") != manifest.get("schema") or not isinstance(mapping, Mapping):
            raise LeafReasoningGroupedMDAError("cluster feature manifest is incompatible with its candidate root")
        mapping = {str(key): str(value) for key, value in mapping.items()}
        if not mapping or any(not key or not value for key, value in mapping.items()) or len(set(mapping.values())) != len(mapping):
            raise LeafReasoningGroupedMDAError("cluster feature manifest has an invalid ID-to-feature mapping")
        funnel_taxonomy = funnel_manifest.get("cluster_taxonomy")
        if not isinstance(funnel_taxonomy, Mapping):
            raise LeafReasoningGroupedMDAError("C funnel lacks a frozen cluster taxonomy contract")
        for key in ("linkage", "cluster_ids_by_arm"):
            if funnel_taxonomy.get(key) != taxonomy.get(key):
                raise LeafReasoningGroupedMDAError("cluster root taxonomy does not match the sealed C funnel")
        for key in ("threshold_by_arm", "selection_phase"):
            if key in funnel_taxonomy and funnel_taxonomy.get(key) != taxonomy.get(key):
                raise LeafReasoningGroupedMDAError("cluster root threshold contract does not match the sealed C funnel")
        for arm in arms:
            if arm.stage != "C" or arm.arm == "C0":
                continue
            declared = tuple(map(str, groups.get(arm.arm, ())))
            ids = tuple(map(str, taxonomy.get("cluster_ids_by_arm", {}).get(arm.arm, ())))
            if not declared or tuple(mapping.get(cluster_id, "") for cluster_id in ids) != declared:
                raise LeafReasoningGroupedMDAError(f"cluster root {arm.arm} group does not match its immutable taxonomy mapping")
            present = tuple(field for field in arm.features if field in set(mapping.values()))
            if present != declared:
                raise LeafReasoningGroupedMDAError(f"sealed C funnel {arm.arm} fields do not match the supplied cluster root")
        table = source / "candidate_cluster_features.parquet"
        available = frozenset(map(str, pq.ParquetFile(table).schema_arrow.names))
        expected_columns = {*_CLUSTER_NATIVE_IDENTITY, *mapping.values()}
        if set(available) != expected_columns:
            raise LeafReasoningGroupedMDAError("candidate cluster feature table does not exactly match its immutable feature manifest")
        return cls(
            source, table, available,
            {
                "root": str(source),
                "manifest_sha256": _sha256_file(manifest_path),
                "candidate_cluster_features_sha256": str(outputs["candidate_cluster_features.parquet"]),
                "cluster_feature_manifest_sha256": str(outputs["cluster_feature_manifest.json"]),
                "cluster_taxonomy_contract_sha256": str(outputs["cluster_taxonomy_contract.json"]),
                "cluster_groups_sha256": str(outputs["cluster_groups.json"]),
            },
        )

    def join(
        self, ledger: pd.DataFrame, *, transport: str, side: str | None, features: Sequence[str],
    ) -> pd.DataFrame:
        requested = tuple(map(str, features))
        missing = sorted(set(requested).difference(self.available))
        if missing:
            raise LeafReasoningGroupedMDAError(f"cluster feature source lacks sealed fields: {missing[:16]}")
        filters: list[tuple[str, str, str]] = [("transport", "==", str(transport))]
        if side is not None:
            filters.append(("side_name", "==", str(side)))
        cluster = pd.read_parquet(self.table, columns=[*_CLUSTER_NATIVE_IDENTITY, *requested], filters=filters)
        if cluster.empty:
            raise LeafReasoningGroupedMDAError(f"cluster feature source has no rows for transport {transport}")
        cluster["decision_ts"] = pd.to_datetime(cluster["decision_ts"], utc=True, errors="coerce")
        if cluster["decision_ts"].isna().any():
            raise LeafReasoningGroupedMDAError("cluster feature source has invalid decision_ts")
        cluster["side_name"] = cluster["side_name"].astype(str).str.lower()
        rename = {
            "decision_ts": "decision_ts", "side_name": "side_name", "fold_id": "__strict_fold_id__",
            "transport": "__strict_transport__", "meta_partition": "__strict_meta_partition__",
        }
        cluster = cluster.rename(columns=rename)
        if cluster.duplicated(list(_IDENTITY)).any():
            raise LeafReasoningGroupedMDAError("cluster feature source duplicates full strict identity")
        values = cluster.loc[:, list(requested)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise LeafReasoningGroupedMDAError("cluster feature source has non-finite selected values")
        overlap = sorted(set(requested).intersection(ledger.columns))
        if overlap:
            raise LeafReasoningGroupedMDAError(f"cluster feature source would overwrite ledger fields: {overlap}")
        merged = ledger.merge(cluster, on=list(_IDENTITY), how="outer", validate="one_to_one", indicator=True)
        if not merged["_merge"].eq("both").all():
            raise LeafReasoningGroupedMDAError(
                "cluster feature and ledger identities are not exact "
                f"(missing_cluster={int(merged['_merge'].eq('left_only').sum())}, "
                f"extra_cluster={int(merged['_merge'].eq('right_only').sum())})"
            )
        return merged.drop(columns="_merge")


def _safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _load_sealed_funnel(root: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(root).resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise LeafReasoningGroupedMDAError(f"funnel root lacks manifest.json: {source}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LeafReasoningGroupedMDAError("funnel manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("immutable_output") is not True or manifest.get("artifact_state") != "COMPLETE":
        raise LeafReasoningGroupedMDAError("grouped MDA requires a sealed immutable funnel output")
    if str(manifest.get("selection_status", "")).startswith("DEVELOPMENT") is False:
        raise LeafReasoningGroupedMDAError("grouped MDA only accepts an explicitly development-only funnel output")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, Mapping):
        raise LeafReasoningGroupedMDAError("funnel manifest has no table hashes")
    for name in ("predictions.parquet", "metrics.parquet", "month_metrics.parquet", "complexity.parquet"):
        expected = hashes.get(name)
        path = source / name
        if not isinstance(expected, str) or not path.is_file() or _sha256_file(path) != expected:
            raise LeafReasoningGroupedMDAError(f"funnel artifact hash is missing or invalid for {name}")
    if not isinstance(manifest.get("arms"), list) or not isinstance(manifest.get("frozen_meta_model"), Mapping):
        raise LeafReasoningGroupedMDAError("funnel manifest lacks arm/model contracts")
    return source, manifest


def _arms(manifest: Mapping[str, Any]) -> tuple[ArmSpec, ...]:
    result: list[ArmSpec] = []
    for raw in manifest["arms"]:
        if not isinstance(raw, Mapping):
            raise LeafReasoningGroupedMDAError("funnel arm contract is invalid")
        try:
            result.append(ArmSpec(
                arm=str(raw["arm"]),
                feature_groups=tuple(map(str, raw["feature_groups"])),
                features=tuple(map(str, raw["features"])),
                stage=str(raw["stage"]),
                control_arm=str(raw["control_arm"]),
                h6_train_selected=bool(raw.get("h6_train_selected", False)),
                h6_fixed_features=tuple(map(str, raw.get("h6_fixed_features", ()))),
                h6_candidate_features=tuple(map(str, raw.get("h6_candidate_features", ()))),
                cluster_similarity_threshold=(
                    float(raw["cluster_similarity_threshold"])
                    if raw.get("cluster_similarity_threshold") is not None else None
                ),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise LeafReasoningGroupedMDAError("funnel arm contract cannot be reconstructed") from exc
    names = [spec.arm for spec in result]
    if len(names) != len(set(names)):
        raise LeafReasoningGroupedMDAError("funnel arm contract contains duplicate arm names")
    return tuple(result)


def _frozen_spec(manifest: Mapping[str, Any]) -> FrozenMetaModelSpec:
    raw = manifest["frozen_meta_model"]
    try:
        return FrozenMetaModelSpec(str(raw["family"]), dict(raw["params"]), str(raw["contract_id"]))
    except (KeyError, TypeError, MetaFunnelError) as exc:
        raise LeafReasoningGroupedMDAError("funnel frozen meta model contract is invalid") from exc


def _funnel_config(manifest: Mapping[str, Any]) -> MetaFunnelConfig:
    raw = manifest.get("config")
    if not isinstance(raw, Mapping):
        raise LeafReasoningGroupedMDAError("funnel manifest lacks the fit config")
    try:
        return MetaFunnelConfig(**dict(raw))
    except (TypeError, MetaFunnelError) as exc:
        raise LeafReasoningGroupedMDAError("funnel fit config is invalid") from exc


def _funnel_gate_config(manifest: Mapping[str, Any]) -> MetaTransportGateConfig:
    raw = manifest.get("transport_gate_config")
    if not isinstance(raw, Mapping):
        raise LeafReasoningGroupedMDAError("funnel manifest lacks the transport gate config")
    try:
        return MetaTransportGateConfig(**dict(raw))
    except (TypeError, MetaFunnelError) as exc:
        raise LeafReasoningGroupedMDAError("funnel transport gate config is invalid") from exc


def _strict_key(frame: pd.DataFrame) -> list[str]:
    missing = sorted(set(_IDENTITY).difference(frame.columns))
    if missing:
        raise LeafReasoningGroupedMDAError(f"strict identity is absent from prediction output: {missing}")
    return list(_IDENTITY)


def _expected_prediction_surface(
    predictions: pd.DataFrame,
    *,
    arm: str,
    transport: str,
    side: str,
) -> pd.DataFrame:
    work = predictions.loc[
        predictions["arm"].eq(arm)
        & predictions["__transport__"].astype(str).eq(transport)
        & predictions["side_name"].astype(str).eq(side)
    ].copy()
    if work.empty:
        raise LeafReasoningGroupedMDAError(f"funnel prediction panel lacks {arm}/{transport}/{side}")
    _strict_key(work)
    if work.duplicated(list(_IDENTITY)).any():
        raise LeafReasoningGroupedMDAError("funnel prediction panel has duplicate strict identities")
    selected = work["selected_features_json"].dropna().astype(str).unique().tolist()
    if len(selected) != 1:
        raise LeafReasoningGroupedMDAError(f"{arm}/{transport}/{side} has inconsistent selected feature contracts")
    try:
        selected_features = tuple(map(str, json.loads(selected[0])))
    except (json.JSONDecodeError, TypeError) as exc:
        raise LeafReasoningGroupedMDAError("selected_features_json is not an ordered JSON feature list") from exc
    if not selected_features or len(selected_features) != len(set(selected_features)):
        raise LeafReasoningGroupedMDAError("selected feature contract is blank or duplicated")
    work.attrs["selected_features"] = selected_features
    return work


def _read_expected_prediction_surface(
    source: Path,
    *, arm: str, transport: str, side: str, columns: MetaFunnelColumns,
) -> pd.DataFrame:
    """Read one sealed arm/transport/side panel instead of the whole panel."""

    projection = list(dict.fromkeys([
        *_IDENTITY, "arm", "__transport__", "selected_features_json", "common_bps_score",
        columns.realized_gross_bps, columns.realized_cost_bps, columns.realized_net_bps,
    ]))
    work = pd.read_parquet(
        source, columns=projection,
        filters=[("arm", "==", arm), ("__transport__", "==", transport), ("side_name", "==", side)],
    )
    # Reuse the sole authority for strict identity and selected-contract
    # validation.  It is intentionally called on this narrow cell only.
    return _expected_prediction_surface(work, arm=arm, transport=transport, side=side)


def _fit_meta(
    train: pd.DataFrame,
    *,
    features: Sequence[str],
    columns: MetaFunnelColumns,
    model_spec: FrozenMetaModelSpec,
    model_factory: MetaModelFactory,
) -> _FittedMeta:
    features = tuple(map(str, features))
    target = train[columns.realized_net_bps].to_numpy(float) - train[columns.base_expected_bps].to_numpy(float)
    train_x = _matrix(train, features, allow_nan=True)
    if model_factory is _fixed_lgbm_factory:
        return _FittedMeta(model_factory(model_spec).fit(train_x, target), features, None, True)
    # This is exactly the injected-test-factory imputation policy used by the
    # funnel itself.  It is never used by the production CLI.
    train_x, _unused, audit = _train_median_impute(train_x, train_x)
    medians = np.zeros(train_x.shape[1], dtype=float)
    # `_train_median_impute` has already applied the train-only medians.  The
    # values can be recovered from original finite columns without depending on
    # the test surface.
    raw = _matrix(train, features, allow_nan=True)
    for index in range(raw.shape[1]):
        finite = raw[:, index][np.isfinite(raw[:, index])]
        if len(finite):
            medians[index] = float(np.median(finite))
    _ = audit
    return _FittedMeta(model_factory(model_spec).fit(train_x, target), features, medians, False)


def _predict_meta(fitted: _FittedMeta, test: pd.DataFrame) -> np.ndarray:
    matrix = _matrix(test, fitted.features, allow_nan=True)
    if not fitted.lightgbm_native_nan:
        assert fitted.medians is not None
        matrix = np.where(np.isnan(matrix), fitted.medians, matrix)
    values = np.asarray(fitted.model.predict(matrix), dtype=float)
    if len(values) != len(test) or not np.isfinite(values).all():
        raise LeafReasoningGroupedMDAError("grouped MDA model emitted non-finite or misaligned residuals")
    return values


def _predict_meta_joint_permuted(
    fitted: _FittedMeta,
    test: pd.DataFrame,
    features: Sequence[str],
    *, rng: np.random.Generator,
) -> np.ndarray:
    """Predict a joint MDA permutation without copying the whole dataframe."""

    if len(test) < 2:
        raise LeafReasoningGroupedMDAError("joint grouped MDA requires at least two rows per side")
    matrix = _matrix(test, fitted.features, allow_nan=True)
    positions = [fitted.features.index(str(feature)) for feature in features]
    permutation = rng.permutation(len(test))
    # Advanced indexing makes one compact ``n_rows x group_width`` temporary,
    # not a second copy of all source and non-feature ledger columns.
    matrix[:, positions] = matrix[permutation][:, positions]
    if not fitted.lightgbm_native_nan:
        assert fitted.medians is not None
        matrix = np.where(np.isnan(matrix), fitted.medians, matrix)
    values = np.asarray(fitted.model.predict(matrix), dtype=float)
    if len(values) != len(test) or not np.isfinite(values).all():
        raise LeafReasoningGroupedMDAError("grouped MDA model emitted non-finite or misaligned residuals")
    return values


def _score_pooled(
    scored: Sequence[pd.DataFrame],
    *,
    columns: MetaFunnelColumns,
    top_fraction: float,
) -> tuple[float, int]:
    work = pd.concat(scored, ignore_index=True)
    if work.empty or set(work[columns.side].astype(str)) != {"long", "short"}:
        raise LeafReasoningGroupedMDAError("grouped MDA needs both sides in every pooled transport surface")
    tie = _ranking_tie_columns(columns)
    ordered = work.sort_values(["common_bps_score", *tie], ascending=[False, *([True] * len(tie))], kind="mergesort")
    selected = ordered.head(max(1, int(np.ceil(len(ordered) * top_fraction))))
    return float(selected[columns.realized_net_bps].mean()), int(len(selected))


def _same_surface(actual: pd.DataFrame, expected: pd.DataFrame, *, columns: MetaFunnelColumns) -> None:
    key = _strict_key(actual)
    if set(key) != set(_strict_key(expected)):
        raise AssertionError("strict identity helper drifted")
    merged = actual.loc[:, [*key, "common_bps_score", columns.realized_gross_bps, columns.realized_cost_bps, columns.realized_net_bps]].merge(
        expected.loc[:, [*key, "common_bps_score", columns.realized_gross_bps, columns.realized_cost_bps, columns.realized_net_bps]],
        on=key,
        how="outer",
        validate="one_to_one",
        suffixes=("_mda", "_sealed"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise LeafReasoningGroupedMDAError("MDA and sealed funnel do not score identical strict identities")
    for name in ("common_bps_score", columns.realized_gross_bps, columns.realized_cost_bps, columns.realized_net_bps):
        if not np.allclose(merged[f"{name}_mda"], merged[f"{name}_sealed"], rtol=0.0, atol=1e-10):
            raise LeafReasoningGroupedMDAError(
                f"MDA baseline does not reproduce sealed funnel {name}; refusing non-joinable evidence"
            )


def _scoring_frame(frame: pd.DataFrame, *, columns: MetaFunnelColumns) -> pd.DataFrame:
    """Project one unique copy of every field required for global economics."""

    fields = list(dict.fromkeys([
        *_IDENTITY, columns.base_expected_bps,
        columns.realized_gross_bps, columns.realized_cost_bps, columns.realized_net_bps,
    ]))
    return frame.loc[:, fields].copy()


def _phantom_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    control_features: Sequence[str],
    group_features: Sequence[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Create train-only shuffled shadow fields matching group dimension.

    A phantom is fitted in place of the candidate incremental group, never
    appended to it.  Its fit/evaluation values are independently shuffled
    within the same side, so no outcome or cross-side information enters the
    null distribution while model capacity remains comparable.
    """

    fit = train.loc[:, list(control_features)].copy()
    evaluate = test.loc[:, list(control_features)].copy()
    rng = np.random.default_rng(seed)
    names: list[str] = []
    for index, source in enumerate(group_features):
        name = f"__mda_phantom_{index:03d}"
        names.append(name)
        fit[name] = train[source].to_numpy()[rng.permutation(len(train))]
        evaluate[name] = test[source].to_numpy()[rng.permutation(len(test))]
    return fit, evaluate, tuple(names)


def _positive_environment_rate(
    month_metrics: pd.DataFrame,
    *,
    arm: str,
    control_arm: str,
    transport: str,
    top_fraction: float,
) -> float:
    local = month_metrics.loc[
        month_metrics["side_name"].eq("ALL")
        & month_metrics["transport_id"].astype(str).eq(transport)
        & month_metrics["top_fraction"].eq(float(top_fraction))
        & month_metrics["arm"].isin((arm, control_arm)),
        ["arm", "month", "net_bps"],
    ].copy()
    candidate = local.loc[local["arm"].eq(arm), ["month", "net_bps"]].rename(columns={"net_bps": "candidate"})
    control = local.loc[local["arm"].eq(control_arm), ["month", "net_bps"]].rename(columns={"net_bps": "control"})
    paired = candidate.merge(control, on="month", how="inner", validate="one_to_one")
    return float((paired["candidate"] > paired["control"]).mean()) if len(paired) else float("nan")


def materialize_leaf_reasoning_grouped_mda(
    ledger: pd.DataFrame | str | Path,
    *,
    funnel_root: str | Path,
    cluster_root: str | Path | None = None,
    config: GroupedMDAConfig = GroupedMDAConfig(),
    columns: MetaFunnelColumns = MetaFunnelColumns(),
    model_factory: MetaModelFactory | None = None,
) -> GroupedMDAResult:
    """Re-fit strict prior-resolved arms and materialise joinable MDA evidence."""

    source_root, manifest = _load_sealed_funnel(funnel_root)
    model_spec = _frozen_spec(manifest)
    funnel_config = _funnel_config(manifest)
    gate_config = _funnel_gate_config(manifest)
    if funnel_config.fit_protocol != "transport_outer_frozen":
        raise LeafReasoningGroupedMDAError("grouped MDA currently requires transport_outer_frozen funnel outputs")
    if model_factory is None:
        model_factory = _fixed_lgbm_factory
    arms = _arms(manifest)
    c_stage_present = any(spec.stage == "C" for spec in arms)
    cluster_reader = None
    if c_stage_present:
        if cluster_root is None:
            raise LeafReasoningGroupedMDAError(
                "sealed C funnels require --cluster-root: C candidate features live in a separate immutable artifact"
            )
        cluster_reader = _ClusterFeatureReader.create(cluster_root, funnel_manifest=manifest, arms=arms)
    elif cluster_root is not None:
        raise LeafReasoningGroupedMDAError("--cluster-root is only valid for a sealed C funnel")
    by_arm = {spec.arm: spec for spec in arms}
    metrics = pd.read_parquet(source_root / "metrics.parquet")
    month_metrics = pd.read_parquet(source_root / "month_metrics.parquet")
    complexity = pd.read_parquet(source_root / "complexity.parquet")
    ledger_reader = _LedgerReader.create(
        ledger, config=funnel_config, columns=columns, cluster_features=cluster_reader,
    )
    real_rows: list[dict[str, Any]] = []
    phantom_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for transport in ledger_reader.transports():
        for spec in arms:
            control = by_arm.get(spec.control_arm)
            if control is None:
                raise LeafReasoningGroupedMDAError(f"{spec.arm} names an absent control arm")
            all_group_features = tuple(field for field in spec.features if field not in set(control.features))
            if not all_group_features:
                continue  # Stage controls have no incremental group to test.
            # Read the selected feature contract from the sealed output first;
            # it determines the *only* fields needed from the very-wide
            # ledger.  This is deliberately per arm and per transport.
            expected_by_side = {
                side: _read_expected_prediction_surface(
                    source_root / "predictions.parquet", arm=spec.arm,
                    transport=str(transport), side=side, columns=columns,
                )
                for side in ("long", "short")
            }
            actual_mdas: list[float] = []
            phantom_mdas: list[float] = []
            baseline_rows: list[pd.DataFrame] = []
            train_by_side: dict[str, pd.DataFrame] = {}
            test_by_side: dict[str, pd.DataFrame] = {}
            features_by_side: dict[str, tuple[str, ...]] = {}
            group_by_side: dict[str, tuple[str, ...]] = {}
            fitted_by_side: dict[str, _FittedMeta] = {}
            # Establish the global chronological boundary on a core-only
            # projection.  We then release it before loading side-local model
            # matrices, avoiding an expensive union of every selected field.
            actual_partitions: set[str] = set()
            outer_starts: list[pd.Timestamp] = []
            for side in ("long", "short"):
                core = ledger_reader.read_transport(str(transport), features=(), side=side)
                partition = core[funnel_config.meta_partition_column].astype(str)
                actual_partitions.update(partition.unique().tolist())
                outer_core = core.loc[partition.eq(funnel_config.outer_partition_value)]
                inner_core = core.loc[partition.eq(funnel_config.inner_partition_value)]
                if outer_core.empty or inner_core.empty:
                    raise LeafReasoningGroupedMDAError(f"{transport} has no complete strict MDA train/evaluation partition")
                outer_starts.append(outer_core[columns.decision].min())
                del core, outer_core, inner_core
            if actual_partitions != {funnel_config.inner_partition_value, funnel_config.outer_partition_value}:
                raise LeafReasoningGroupedMDAError("MDA ledger does not have the sealed inner/outer partition contract")
            outer_start = min(outer_starts)
            for side_index, side in enumerate(("long", "short")):
                # Side-local contracts can differ.  Read only this side's
                # declared model fields, never a cross-side feature union.
                features = tuple(expected_by_side[side].attrs["selected_features"])
                cell = ledger_reader.read_transport(str(transport), features=features, side=side)
                partition = cell[funnel_config.meta_partition_column].astype(str)
                inner = cell.loc[partition.eq(funnel_config.inner_partition_value)]
                outer = cell.loc[partition.eq(funnel_config.outer_partition_value)]
                train = inner.loc[
                    inner[columns.label_available].lt(outer_start)
                ].copy()
                test = outer.copy()
                if train.empty or test.empty or not train[columns.label_available].lt(outer_start).all():
                    raise LeafReasoningGroupedMDAError(f"{spec.arm}/{transport}/{side} violates strict prior-resolved MDA support")
                expected = expected_by_side[side]
                features = tuple(expected.attrs["selected_features"])
                missing = sorted(set(features).difference(train.columns))
                if missing:
                    raise LeafReasoningGroupedMDAError(f"MDA ledger lacks sealed {spec.arm} features: {missing[:16]}")
                group = tuple(field for field in features if field not in set(control.features))
                if not group:
                    raise LeafReasoningGroupedMDAError(f"{spec.arm}/{transport}/{side} lost its declared incremental group")
                fitted = _fit_meta(train, features=features, columns=columns, model_spec=model_spec, model_factory=model_factory)
                base = _scoring_frame(test, columns=columns)
                base["common_bps_score"] = base[columns.base_expected_bps].to_numpy(float) + _predict_meta(fitted, test)
                _same_surface(base, expected, columns=columns)
                baseline_rows.append(base)
                train_by_side[side] = train
                test_by_side[side] = test
                features_by_side[side] = features
                group_by_side[side] = group
                fitted_by_side[side] = fitted
                # ``train``/``test`` are the only retained views for this
                # side.  Drop the validated source cell and partition masks
                # before loading the other side's selected contract.
                del cell, partition, inner, outer
            baseline_bps, selected_rows = _score_pooled(baseline_rows, columns=columns, top_fraction=config.top_fraction)
            for repeat in range(config.repeats):
                perturbed_rows: list[pd.DataFrame] = []
                for side_index, side in enumerate(("long", "short")):
                    values = _predict_meta_joint_permuted(
                        fitted_by_side[side], test_by_side[side], group_by_side[side],
                        rng=np.random.default_rng(config.random_seed + 1_000_003 * repeat + 10_007 * side_index + 101 * len(spec.arm)),
                    )
                    scored = _scoring_frame(test_by_side[side], columns=columns)
                    scored["common_bps_score"] = scored[columns.base_expected_bps].to_numpy(float) + values
                    perturbed_rows.append(scored)
                permuted_bps, _ = _score_pooled(perturbed_rows, columns=columns, top_fraction=config.top_fraction)
                actual_mda = baseline_bps - permuted_bps
                actual_mdas.append(actual_mda)
                real_rows.append({
                    "arm": spec.arm, "control_arm": spec.control_arm, "transport_id": transport,
                    "repeat": repeat, "baseline_top10_net_bps": baseline_bps,
                    "permuted_top10_net_bps": permuted_bps, "group_mda_bps": actual_mda,
                    "top10_selected_rows": selected_rows, "group_features_json": json.dumps({side: list(group_by_side[side]) for side in ("long", "short")}, sort_keys=True),
                    "ranking_scope": "one_pooled_global_post_common_bps_top_k_per_transport",
                    "permutation_style": "joint_row_shuffle_within_side_on_outer_test_rows",
                    "strict_prior_resolved": True,
                })
            for draw in range(config.phantom_draws):
                phantom_scored: list[pd.DataFrame] = []
                phantom_permuted: list[pd.DataFrame] = []
                for side_index, side in enumerate(("long", "short")):
                    phantom_train, phantom_test, phantom_features = _phantom_frames(
                        train_by_side[side], test_by_side[side], control_features=control.features,
                        group_features=group_by_side[side],
                        seed=config.random_seed + 10_000_019 * draw + 101 * side_index + len(spec.arm),
                    )
                    # Keep target/base columns out of the model matrix while
                    # supplying the exact side-local labels to the refit.
                    phantom_train[columns.realized_net_bps] = train_by_side[side][columns.realized_net_bps].to_numpy()
                    phantom_train[columns.base_expected_bps] = train_by_side[side][columns.base_expected_bps].to_numpy()
                    phantom_test[columns.base_expected_bps] = test_by_side[side][columns.base_expected_bps].to_numpy()
                    fitted_phantom = _fit_meta(
                        phantom_train,
                        features=[*control.features, *phantom_features],
                        columns=columns,
                        model_spec=model_spec,
                        model_factory=model_factory,
                    )
                    baseline = _scoring_frame(test_by_side[side], columns=columns)
                    baseline["common_bps_score"] = baseline[columns.base_expected_bps].to_numpy(float) + _predict_meta(fitted_phantom, phantom_test)
                    values = _predict_meta_joint_permuted(
                        fitted_phantom, phantom_test, phantom_features,
                        rng=np.random.default_rng(config.random_seed + 20_000_033 * draw + 103 * side_index + len(spec.arm)),
                    )
                    permuted = _scoring_frame(test_by_side[side], columns=columns)
                    permuted["common_bps_score"] = permuted[columns.base_expected_bps].to_numpy(float) + values
                    phantom_scored.append(baseline)
                    phantom_permuted.append(permuted)
                phantom_control, _ = _score_pooled(phantom_scored, columns=columns, top_fraction=config.top_fraction)
                phantom_permuted_bps, _ = _score_pooled(phantom_permuted, columns=columns, top_fraction=config.top_fraction)
                phantom_mda = phantom_control - phantom_permuted_bps
                phantom_mdas.append(phantom_mda)
                phantom_rows.append({
                    "arm": spec.arm, "control_arm": spec.control_arm, "transport_id": transport,
                    "phantom_draw": draw, "phantom_control_top10_net_bps": phantom_control,
                    "phantom_permuted_top10_net_bps": phantom_permuted_bps,
                    "phantom_mda_bps": phantom_mda, "phantom_group_size": int(max(map(len, group_by_side.values()))),
                    "phantom_definition": "train_and_test_side_local_shuffled_shadow_group_replacing_incremental_group",
                    "strict_prior_resolved": True,
                })
            actual = np.asarray(actual_mdas, dtype=float)
            phantom = np.asarray(phantom_mdas, dtype=float)
            median = float(np.median(actual))
            mad = float(np.median(np.abs(actual - median)))
            stable = median - .5 * mad
            taxonomy = manifest.get("cluster_taxonomy")
            ids = []
            if isinstance(taxonomy, Mapping) and isinstance(taxonomy.get("cluster_ids_by_arm"), Mapping):
                ids = list(map(str, taxonomy["cluster_ids_by_arm"].get(spec.arm, ())))
            summary_rows.append({
                "arm": spec.arm, "control_arm": spec.control_arm, "stage": spec.stage,
                "transport_id": transport, "feature_count": int(max(map(len, features_by_side.values()))),
                "incremental_feature_count": int(max(map(len, group_by_side.values()))),
                "group_features_json": json.dumps({side: list(group_by_side[side]) for side in ("long", "short")}, sort_keys=True),
                "cluster_ids_json": json.dumps(ids),
                "baseline_top10_net_bps": baseline_bps,
                "transport_mda_bps": median,
                "transport_mda_mean_bps": float(actual.mean()),
                "transport_mda_mad_bps": mad,
                "stable_transport_mda_bps": stable,
                "phantom_q95_bps": float(np.quantile(phantom, .95)),
                "phantom_median_bps": float(np.median(phantom)),
                "real_repeat_count": int(len(actual)), "phantom_draw_count": int(len(phantom)),
                "positive_environment_rate": _positive_environment_rate(
                    month_metrics, arm=spec.arm, control_arm=spec.control_arm,
                    transport=transport, top_fraction=config.top_fraction,
                ),
                "mda_above_phantom": bool(stable > float(np.quantile(phantom, .95))),
                "stable_mda_positive": bool(stable > 0.0),
                "strict_prior_resolved": True,
                "ranking_scope": "one_pooled_global_post_common_bps_top_k_per_transport",
            })

    if not summary_rows:
        raise LeafReasoningGroupedMDAError("funnel declares no incremental arm feature group eligible for grouped MDA")
    summary = pd.DataFrame(summary_rows).sort_values(["arm", "transport_id"], kind="mergesort").reset_index(drop=True)
    real = pd.DataFrame(real_rows).sort_values(["arm", "transport_id", "repeat"], kind="mergesort").reset_index(drop=True)
    phantom = pd.DataFrame(phantom_rows).sort_values(["arm", "transport_id", "phantom_draw"], kind="mergesort").reset_index(drop=True)
    ablation, gates = build_meta_ablation_gates(
        metrics, month_metrics, complexity, arms=arms, config=gate_config, grouped_mda=summary,
    )
    advancement = ablation.merge(
        gates.loc[:, ["arm", "grouped_transport_mda_evidence_present", "grouped_transport_mda_pass", "passes_all_advancement_gates", "promotion_status"]],
        on="arm", how="left", validate="one_to_one",
    )
    return GroupedMDAResult(
        summary, real, phantom, advancement, gates, manifest, source_root, config,
        cluster_source=cluster_reader.source_binding if cluster_reader is not None else None,
    )


def write_immutable_leaf_reasoning_grouped_mda(
    result: GroupedMDAResult,
    output_dir: str | Path,
) -> Path:
    """Publish hash-bound post-funnel grouped-MDA evidence atomically."""

    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite grouped MDA artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        tables = {
            "grouped_mda_summary.parquet": result.summary,
            "grouped_mda_real_repeats.parquet": result.real_repeats,
            "grouped_mda_phantom_draws.parquet": result.phantom_draws,
            "advancement_evidence.parquet": result.advancement,
            "transport_gates_with_grouped_mda.parquet": result.gates,
        }
        hashes: dict[str, str] = {}
        for name, table in tables.items():
            path = temporary / name
            table.to_parquet(path, index=False, compression="zstd")
            hashes[name] = _sha256_file(path)
        source_hashes = result.source_manifest["sha256"]
        manifest = {
            "schema": SCHEMA,
            "status": "STRICT_CHRONOLOGICAL_TRANSPORT_GROUPED_MDA_COMPLETE",
            "immutable_output": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_funnel_root": str(result.source_root),
            "source_funnel_manifest_sha256": _sha256_file(result.source_root / "manifest.json"),
            "source_funnel_table_sha256": {
                name: source_hashes[name]
                for name in ("predictions.parquet", "metrics.parquet", "month_metrics.parquet", "complexity.parquet")
            },
            "source_cluster_feature_artifact": result.cluster_source,
            "config": asdict(result.config),
            "contract": {
                "fit": "same frozen side-local meta model, fit only on inner base-OOF labels resolved before the outer decision boundary",
                "real_mda": "joint row shuffle of each arm's declared incremental feature group within side on outer evaluation rows",
                "phantom": "same-dimensional shuffled shadow group replaces the incremental group during strict train/evaluation refits",
                "ranking": "one pooled cross-side common-bps top-k ranking per chronological transport",
                "advancement": "stable MDA = median(real repeats) - 0.5*MAD; every transport must be positive and above its phantom q95",
                "final_oos": "not read or used",
            },
            "sha256": hashes,
        }
        (temporary / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "GroupedMDAConfig", "GroupedMDAResult", "LeafReasoningGroupedMDAError",
    "materialize_leaf_reasoning_grouped_mda", "write_immutable_leaf_reasoning_grouped_mda",
]
