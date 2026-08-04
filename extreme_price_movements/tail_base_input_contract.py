"""Bounded production input materialisation for the broad-to-tail ranker.

The broad R3 model, tail base model and residual ranker must see the same
decision-time raw and frozen AE/GMM representation.  This module makes that
join explicit without loading the full candidate population into memory.

The static spread input is deliberately narrow: a caller supplies a *pooled
p90* map.  Average-spread columns are rejected rather than treated as an
approximation.  This is a research-universe screen, not a claim that p90 was
known point-in-time historically; callers must disclose that scope in their
experiment manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .features_gmm_ae import transform_ae_gmm_features
from .packb_static_point_feature_loader import (
    FrozenFeatureContract,
    _feature_contract_digest,
    discover_causal_feature_universe,
    freeze_feature_contract,
)
from .stage_i_production_data_adapter import (
    MonthlyReferencePartition,
    _canonical_symbol,
    _read_partition,
    make_static_pit_feature_loader,
)
from .tail_base_targets import grade_atr_normalized_net, grade_exact_net_bps


SCHEMA = "tail_base_input_contract_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
REQUIRED_SPREAD_COLUMNS = ("__symbol__", "p90_spread_bps")
TBM_PATH_COLUMNS = (
    "first_tp4_minute", "first_tp6_minute", "first_sl4_minute", "first_sl6_minute",
)


class TailBaseInputContractError(ValueError):
    """Raised when the materialised broad-to-tail substrate is not provable."""


PointFeatureLoader = Callable[[pd.DataFrame, Sequence[str]], pd.DataFrame]
AEGMMTransformer = Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]


@dataclass(frozen=True)
class PooledP90SpreadMap:
    """Explicit static research-universe screen; no average-spread fallback."""

    values: pd.DataFrame
    threshold_bps: float = 90.0

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, *, threshold_bps: float = 90.0
    ) -> "PooledP90SpreadMap":
        if not np.isfinite(float(threshold_bps)) or float(threshold_bps) <= 0.0:
            raise TailBaseInputContractError("p90 spread threshold must be positive and finite")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("p90 spread map must be a DataFrame")
        missing = sorted(set(REQUIRED_SPREAD_COLUMNS).difference(frame.columns))
        if missing:
            # Explicitly guard the most tempting but invalid substitute.
            average_columns = [name for name in frame.columns if "average" in str(name).lower()]
            hint = "; average spread is not an eligible substitute" if average_columns else ""
            raise TailBaseInputContractError(f"p90 spread map missing {missing}{hint}")
        out = frame.loc[:, list(REQUIRED_SPREAD_COLUMNS)].copy()
        out["__symbol__"] = out["__symbol__"].map(_canonical_symbol)
        out["p90_spread_bps"] = pd.to_numeric(out["p90_spread_bps"], errors="coerce")
        if out["__symbol__"].isna().any() or out["p90_spread_bps"].isna().any():
            raise TailBaseInputContractError("p90 spread map has null/non-numeric values")
        if (~np.isfinite(out["p90_spread_bps"].to_numpy(dtype=float))).any() or (out["p90_spread_bps"] < 0).any():
            raise TailBaseInputContractError("p90 spread map has invalid p90_spread_bps")
        if out["__symbol__"].duplicated().any():
            raise TailBaseInputContractError("p90 spread map has duplicate canonical symbols")
        return cls(out.sort_values("__symbol__", kind="stable").reset_index(drop=True), float(threshold_bps))

    @classmethod
    def from_path(cls, path: str | Path, *, threshold_bps: float = 90.0) -> "PooledP90SpreadMap":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(source)
        elif source.suffix.lower() == ".csv":
            frame = pd.read_csv(source)
        else:
            raise TailBaseInputContractError("p90 spread map must be parquet or csv")
        return cls.from_frame(frame, threshold_bps=threshold_bps)

    def eligible(self, ledger: pd.DataFrame) -> pd.DataFrame:
        result = ledger.merge(self.values, on="__symbol__", how="left", validate="many_to_one", sort=False)
        result["p90_spread_map_matched"] = result["p90_spread_bps"].notna()
        result["p90_spread_eligible"] = result["p90_spread_map_matched"] & (
            result["p90_spread_bps"] < float(self.threshold_bps)
        )
        return result


def _json_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_feature_contract(path: str | Path) -> FrozenFeatureContract:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return FrozenFeatureContract.from_mapping(json.loads(source.read_text(encoding="utf-8")))


def build_aegmm_projection_source_contract(
    *,
    partitions: Sequence[MonthlyReferencePartition],
    p90_spread_map: PooledP90SpreadMap,
    feature_store_dir: str | Path,
    side_model_features: Mapping[str, Sequence[str]],
    aegmm_state: Mapping[str, Any],
) -> tuple[FrozenFeatureContract, dict[str, list[str]], dict[str, Any]]:
    """Freeze an outcome-free source contract for selected fields + AE inputs.

    This is deliberately distinct from side MDA selection: it merely proves
    which registered causal store columns can supply an already-frozen latent
    transform.  A bounded first-valid-row-per-symbol identity sample makes the
    schema check inexpensive and avoids reading the wide store values here.
    """

    if set(map(str, side_model_features)) != {"long", "short"}:
        raise TailBaseInputContractError("side_model_features must include long and short")
    identities: list[pd.DataFrame] = []
    for partition in partitions:
        ledger, _audit = _read_materialisable_partition(partition)
        eligible = p90_spread_map.eligible(ledger)
        eligible = eligible.loc[eligible["p90_spread_eligible"], ["candidate_id", "__ts__", "__symbol__"]]
        if not eligible.empty:
            identities.append(eligible.drop_duplicates("__symbol__", keep="first"))
    if not identities:
        raise TailBaseInputContractError("no p90-eligible label-valid identities for AE/GMM source discovery")
    identity = pd.concat(identities, ignore_index=True).drop_duplicates("__symbol__", keep="first")
    universe = discover_causal_feature_universe(
        identity, feature_store_dir=feature_store_dir, coverage_discovery=False,
    )
    available = set(universe.feature_columns)
    state_inputs = set(map(str, aegmm_state.get("feature_columns", ()))) - {"side"}
    state_available = state_inputs.intersection(available)
    normalized = {side: sorted(set(map(str, fields))) for side, fields in side_model_features.items()}
    requested = set().union(*map(set, normalized.values()), state_available)
    missing_model = sorted(set().union(*map(set, normalized.values())).difference(available))
    if missing_model:
        raise TailBaseInputContractError(
            "selected R3 model fields are absent from fresh causal source universe: " + ", ".join(missing_model[:12])
        )
    base = freeze_feature_contract(
        universe, min_exact_key_coverage=0.0, min_non_null_feature_coverage=0.0, max_feature_columns=None,
    )
    ordered = tuple(sorted(requested))
    projection_contract = replace(
        base, feature_columns=ordered,
        feature_contract_sha256=_feature_contract_digest(
            feature_columns=ordered, candidate_universe_sha256=base.candidate_universe_sha256,
            source_schema_sha256=base.source_schema_sha256, raw_allowlist_sha256=base.raw_allowlist_sha256,
            generator_registry_sha256=base.generator_registry_sha256,
            store_scan_manifest_sha256=base.store_scan_manifest_sha256,
            coverage_profile_sha256=base.coverage_profile_sha256,
            min_exact_key_coverage=base.min_exact_key_coverage,
            min_non_null_feature_coverage=base.min_non_null_feature_coverage,
            max_feature_columns=base.max_feature_columns,
            coverage_admission_rejections=base.coverage_admission_rejections,
        ),
    )
    per_side = {side: sorted(set(fields).union(state_available)) for side, fields in normalized.items()}
    audit = {
        "schema_discovery_rows": int(len(identity)), "schema_discovery_symbols": int(identity["__symbol__"].nunique()),
        "frozen_aegmm_state_inputs": int(len(state_inputs) + ("side" in aegmm_state.get("feature_columns", ()))),
        "aegmm_source_inputs_available": int(len(state_available) + ("side" in aegmm_state.get("feature_columns", ()))),
        "aegmm_source_overlap": float(
            (len(state_available) + ("side" in aegmm_state.get("feature_columns", ())))
            / max(1, len(state_inputs) + ("side" in aegmm_state.get("feature_columns", ())))
        ),
        "projection_source_fields": int(len(ordered)),
        "projection_source_contract_sha256": projection_contract.feature_contract_sha256,
    }
    return projection_contract, per_side, audit


def _partition_files(partition: MonthlyReferencePartition) -> list[Path]:
    path = Path(partition.path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.parquet")) if path.is_dir() else []


def _read_partition_auxiliary(
    partition: MonthlyReferencePartition, *, fields: Sequence[str]
) -> tuple[pd.DataFrame, set[str], int, int]:
    """Read only auxiliary label fields alongside the existing R3 ledger reader."""

    files = _partition_files(partition)
    if not files:
        raise FileNotFoundError(partition.path)
    available: set[str] | None = None
    rows = 0
    valid_rows = 0
    pieces: list[pd.DataFrame] = []
    for path in files:
        import pyarrow.parquet as pq

        schema = set(pq.ParquetFile(path).schema_arrow.names)
        available = schema if available is None else available.intersection(schema)
        cols = ["candidate_id", "label_valid", *[name for name in fields if name in schema]]
        part = pd.read_parquet(path, columns=cols)
        rows += len(part)
        valid_rows += int(part["label_valid"].astype(bool).sum())
        pieces.append(part)
    out = pd.concat(pieces, ignore_index=True)
    if out["candidate_id"].duplicated().any():
        raise TailBaseInputContractError(f"duplicate candidate_id in label partition: {partition.path}")
    return out, set(available or ()), rows, valid_rows


def _read_materialisable_partition(
    partition: MonthlyReferencePartition,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read valid R3 rows plus ATR; retain invalid counts only in audits."""

    aux, common_columns, source_rows, source_valid_rows = _read_partition_auxiliary(
        partition, fields=("atr_bps", *TBM_PATH_COLUMNS)
    )
    if "atr_bps" not in common_columns:
        raise TailBaseInputContractError(f"{partition.path} lacks decision-time atr_bps required by target 2")
    ledger = _read_partition(partition)
    aux = aux.loc[:, ["candidate_id", "atr_bps"]].copy()
    ledger = ledger.merge(aux, on="candidate_id", how="left", validate="one_to_one", sort=False)
    ledger["atr_bps"] = pd.to_numeric(ledger["atr_bps"], errors="coerce")
    if ledger["atr_bps"].isna().any() or (ledger["atr_bps"] <= 0.0).any():
        raise TailBaseInputContractError(f"{partition.path} has invalid decision-time atr_bps on valid rows")
    audit = {
        "source_month": str(partition.source_month),
        "population_segment": str(partition.population),
        "source_path": str(Path(partition.path)),
        "source_rows": int(source_rows),
        "source_label_valid_rows": int(source_valid_rows),
        "source_label_invalid_rows": int(source_rows - source_valid_rows),
        "materialisable_r3_rows": int(len(ledger)),
        "tbm_input_columns_available": bool(set(TBM_PATH_COLUMNS).issubset(common_columns)),
        "tbm_missing_columns": json.dumps(sorted(set(TBM_PATH_COLUMNS).difference(common_columns))),
    }
    return ledger, audit


def _validate_raw_identity(ledger: pd.DataFrame, raw: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    needed = {"candidate_id", "__ts__", "__symbol__", *map(str, fields)}
    missing = sorted(needed.difference(raw.columns))
    if missing:
        raise TailBaseInputContractError(f"PIT loader omitted declared raw fields: {missing}")
    result = raw.loc[:, ["candidate_id", "__ts__", "__symbol__", *map(str, fields)]].copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="coerce")
    if result["__ts__"].isna().any() or result["candidate_id"].duplicated().any():
        raise TailBaseInputContractError("PIT loader returned invalid or duplicate candidate identity")
    left = pd.MultiIndex.from_frame(ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]])
    right = pd.MultiIndex.from_frame(result.loc[:, ["candidate_id", "__ts__", "__symbol__"]])
    if not left.equals(right):
        raise TailBaseInputContractError("PIT loader did not preserve exact candidate identity order")
    return result


def _row_independent_aegmm(
    raw: pd.DataFrame,
    ledger: pd.DataFrame,
    state: Mapping[str, Any],
    *,
    prefix: str,
    min_source_overlap: float,
    transformer: AEGMMTransformer,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(state, Mapping) or not bool(state.get("enabled", False)):
        raise TailBaseInputContractError("frozen AE/GMM state is disabled or invalid")
    projection = dict(state)
    source_temporal_contract = str(projection.get("temporal_feature_contract") or "unspecified")
    projection["temporal_feature_contract"] = "row_independent_v1"
    inputs = tuple(map(str, projection.get("feature_columns", ())))
    if not inputs:
        raise TailBaseInputContractError("frozen AE/GMM state has no ordered input fields")
    source = raw.reindex(columns=list(inputs)).copy()
    if "side" in inputs:
        side = ledger["side_name"].astype(str).str.lower().map({"long": 1.0, "short": -1.0})
        if side.isna().any():
            raise TailBaseInputContractError("cannot reconstruct AE/GMM side input")
        source["side"] = side.to_numpy(dtype=np.float32)
    present = [name for name in inputs if name == "side" or name in raw.columns]
    overlap = float(len(present) / len(inputs))
    if overlap < float(min_source_overlap):
        raise TailBaseInputContractError(
            f"frozen AE/GMM input overlap {overlap:.4f} is below {float(min_source_overlap):.4f}"
        )
    generated = transformer(source, projection).replace([np.inf, -np.inf], np.nan)
    if len(generated) != len(raw):
        raise TailBaseInputContractError("AE/GMM projection did not preserve row count")
    generated = generated.copy()
    generated.columns = [f"{prefix}{name}" if not str(name).startswith(prefix) else str(name) for name in generated.columns]
    return generated.reset_index(drop=True), {
        "state_input_count": int(len(inputs)),
        "state_input_present_count": int(len(present)),
        "state_input_overlap": overlap,
        "state_input_missing": [name for name in inputs if name not in present],
        "source_temporal_contract": source_temporal_contract,
        "projection_temporal_contract": "row_independent_v1",
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _update_coverage(
    counts: dict[str, int], frame: pd.DataFrame, fields: Sequence[str]
) -> None:
    for field in fields:
        counts[str(field)] = counts.get(str(field), 0) + int(frame[str(field)].notna().sum())


def materialize_tail_base_input_contract(
    *,
    partitions: Sequence[MonthlyReferencePartition],
    raw_feature_contract: FrozenFeatureContract,
    p90_spread_map: PooledP90SpreadMap,
    aegmm_state: Mapping[str, Any],
    output_dir: str | Path,
    feature_store_dir: str | Path | None = None,
    batch_rows: int = 8_000,
    raw_features: Sequence[str] | None = None,
    side_raw_features: Mapping[str, Sequence[str]] | None = None,
    source_raw_features: Sequence[str] | None = None,
    aegmm_prefix: str = "aegmm_",
    min_aegmm_source_overlap: float = 0.50,
    pit_feature_loader: PointFeatureLoader | None = None,
    aegmm_transformer: AEGMMTransformer = transform_ae_gmm_features,
) -> dict[str, Any]:
    """Materialise valid T1/T2 rows in bounded chunks.

    ``T3`` is deliberately not approximated.  Its four exact first-touch
    minute inputs are checked in the source schema and reported as an explicit
    materialisation requirement in the manifest.
    """

    if not partitions:
        raise TailBaseInputContractError("at least one R3 production label partition is required")
    if int(batch_rows) < 1:
        raise TailBaseInputContractError("batch_rows must be positive")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable input contract: {destination}")
    if raw_features is not None and side_raw_features is not None:
        raise TailBaseInputContractError("supply raw_features or side_raw_features, not both")
    side_feature_audit: dict[str, list[str]] | None = None
    if side_raw_features is not None:
        invalid_sides = sorted(set(map(str, side_raw_features)).difference({"long", "short"}))
        if invalid_sides:
            raise TailBaseInputContractError(f"unknown side-specific raw feature lists: {invalid_sides}")
        if set(side_raw_features) != {"long", "short"}:
            raise TailBaseInputContractError("side_raw_features must declare both long and short")
        normalized = {
            side: tuple(sorted(set(map(str, values))))
            for side, values in side_raw_features.items()
        }
        if not normalized["long"] or not normalized["short"]:
            raise TailBaseInputContractError("each side-specific raw feature list must be non-empty")
        side_feature_audit = {side: list(values) for side, values in normalized.items()}
        model_declared = tuple(sorted(set(normalized["long"]).union(normalized["short"])))
    else:
        model_declared = tuple(map(str, raw_features or raw_feature_contract.feature_columns))
    declared = tuple(sorted(set(map(str, source_raw_features or model_declared))))
    if not model_declared or len(set(model_declared)) != len(model_declared):
        raise TailBaseInputContractError("model raw feature fields must be non-empty and unique")
    if not declared or len(set(declared)) != len(declared):
        raise TailBaseInputContractError("raw feature fields must be non-empty and unique")
    if not set(model_declared).issubset(declared):
        raise TailBaseInputContractError("source raw feature fields must include all model raw fields")
    unavailable = sorted(set(declared).difference(raw_feature_contract.feature_columns))
    if unavailable:
        raise TailBaseInputContractError(f"raw feature fields are outside frozen feature contract: {unavailable}")
    if pit_feature_loader is None:
        if feature_store_dir is None:
            raise TailBaseInputContractError("feature_store_dir is required when no PIT loader is supplied")
        # The canonical loader intentionally requires an exact frozen field list.
        from dataclasses import replace
        from .packb_static_point_feature_loader import _feature_contract_digest

        ordered = tuple(sorted(declared))
        narrowed = replace(
            raw_feature_contract,
            feature_columns=ordered,
            feature_contract_sha256=_feature_contract_digest(
                feature_columns=ordered,
                candidate_universe_sha256=raw_feature_contract.candidate_universe_sha256,
                source_schema_sha256=raw_feature_contract.source_schema_sha256,
                raw_allowlist_sha256=raw_feature_contract.raw_allowlist_sha256,
                generator_registry_sha256=raw_feature_contract.generator_registry_sha256,
                store_scan_manifest_sha256=raw_feature_contract.store_scan_manifest_sha256,
                coverage_profile_sha256=raw_feature_contract.coverage_profile_sha256,
                min_exact_key_coverage=raw_feature_contract.min_exact_key_coverage,
                min_non_null_feature_coverage=raw_feature_contract.min_non_null_feature_coverage,
                max_feature_columns=raw_feature_contract.max_feature_columns,
                coverage_admission_rejections=raw_feature_contract.coverage_admission_rejections,
            ),
        )
        pit_feature_loader = make_static_pit_feature_loader(
            feature_store_dir=feature_store_dir, feature_contract=narrowed,
            max_rows_per_batch=min(int(batch_rows), 8_000), max_columns_per_read=min(256, len(ordered)),
            verify_frozen_schema=True,
        )
    destination.mkdir(parents=True, exist_ok=False)
    coverage: dict[str, int] = {}
    source_coverage: dict[str, int] = {}
    written_rows = 0
    output_parts: list[str] = []
    audits: list[dict[str, Any]] = []
    ae_columns: tuple[str, ...] | None = None
    ae_summary: dict[str, Any] | None = None
    try:
        for partition_index, partition in enumerate(partitions):
            ledger, audit = _read_materialisable_partition(partition)
            screened = p90_spread_map.eligible(ledger)
            audit["p90_map_matched_rows"] = int(screened["p90_spread_map_matched"].sum())
            audit["p90_eligible_rows"] = int(screened["p90_spread_eligible"].sum())
            audit["p90_unmapped_rows"] = int((~screened["p90_spread_map_matched"]).sum())
            audit["p90_ineligible_rows"] = int((~screened["p90_spread_eligible"]).sum())
            eligible = screened.loc[screened["p90_spread_eligible"]].copy()
            audits.append(audit)
            for batch_index, start in enumerate(range(0, len(eligible), int(batch_rows))):
                batch = eligible.iloc[start:start + int(batch_rows)].reset_index(drop=True)
                raw = _validate_raw_identity(batch, pit_feature_loader(batch, declared), declared)
                ae, current_ae_summary = _row_independent_aegmm(
                    raw, batch, aegmm_state, prefix=aegmm_prefix,
                    min_source_overlap=float(min_aegmm_source_overlap), transformer=aegmm_transformer,
                )
                if ae_columns is None:
                    ae_columns = tuple(map(str, ae.columns))
                    ae_summary = current_ae_summary
                elif tuple(map(str, ae.columns)) != ae_columns:
                    raise TailBaseInputContractError("frozen AE/GMM output schema changed between chunks")
                net_grade = grade_exact_net_bps(batch["exact_net_bps"].to_numpy(), batch["label_valid"].to_numpy())
                atr_grade, atr_z = grade_atr_normalized_net(
                    batch["exact_net_bps"].to_numpy(), batch["atr_bps"].to_numpy(), batch["label_valid"].to_numpy()
                )
                label_columns = [
                    *IDENTITY, "label_available_ts", "label_valid", "exact_gross_bps", "exact_net_bps",
                    "atr_bps", "t2_tp6_sl4_event", "robust_clear_event_b25", "robust_clear_soft_b25_t50",
                    "p90_spread_bps", "p90_spread_map_matched", "p90_spread_eligible",
                ]
                result = batch.loc[:, label_columns].copy()
                result["tail_target_t1_valid"] = batch["label_valid"].astype(bool).to_numpy()
                result["tail_target_net_grade_0_5"] = net_grade
                result["tail_target_t2_valid"] = batch["label_valid"].astype(bool).to_numpy()
                result["tail_target_atr_grade_0_5"] = atr_grade
                result["tail_target_atr_z"] = atr_z
                result = pd.concat([result.reset_index(drop=True), raw.loc[:, list(model_declared)].reset_index(drop=True), ae], axis=1)
                if result.columns.duplicated().any():
                    duplicate = result.columns[result.columns.duplicated()].tolist()
                    raise TailBaseInputContractError(f"materialised input has duplicate columns: {duplicate}")
                _update_coverage(source_coverage, raw, declared)
                _update_coverage(coverage, result, [*model_declared, *ae.columns])
                relative = Path("parts") / f"month={partition.source_month}" / f"part-{partition_index:04d}-{batch_index:05d}.parquet"
                _atomic_parquet(result, destination / relative)
                output_parts.append(str(relative))
                written_rows += len(result)
        if written_rows < 1 or ae_columns is None or ae_summary is None:
            raise TailBaseInputContractError("no p90-eligible valid rows were materialised")
        feature_columns = [*declared, *ae_columns]
        coverage_frame = pd.DataFrame({
            "feature": feature_columns,
            "non_null_rows": [coverage.get(name, 0) for name in feature_columns],
            "rows": int(written_rows),
        })
        coverage_frame["coverage"] = coverage_frame["non_null_rows"] / float(written_rows)
        coverage_frame["coverage_ge_90pct"] = coverage_frame["coverage"] >= 0.90
        coverage_frame.to_parquet(destination / "feature_coverage_audit.parquet", index=False)
        source_coverage_frame = pd.DataFrame({
            "source_feature": list(declared),
            "non_null_rows": [source_coverage.get(name, 0) for name in declared],
            "rows": int(written_rows),
        })
        source_coverage_frame["coverage"] = source_coverage_frame["non_null_rows"] / float(written_rows)
        source_coverage_frame["used_as_model_input"] = source_coverage_frame["source_feature"].isin(model_declared)
        source_coverage_frame.to_parquet(destination / "aegmm_source_coverage_audit.parquet", index=False)
        pd.DataFrame(audits).to_parquet(destination / "label_spread_audit.parquet", index=False)
        t3_ready = bool(all(bool(row["tbm_input_columns_available"]) for row in audits))
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "complete",
            "rows": int(written_rows),
            "parts": output_parts,
            "identity_contract": list(IDENTITY),
            "raw_feature_contract_sha256": raw_feature_contract.feature_contract_sha256,
            "raw_feature_columns": list(model_declared),
            "raw_feature_count": len(model_declared),
            "aegmm_source_raw_feature_columns": list(declared),
            "aegmm_source_raw_feature_count": len(declared),
            "side_raw_feature_contracts": side_feature_audit,
            "frozen_aegmm_output_columns": list(ae_columns),
            "frozen_aegmm_output_count": len(ae_columns),
            "frozen_aegmm": ae_summary,
            "p90_spread_screen": {
                "semantics": "static_pooled_research_universe_screen_not_point_in_time",
                "column": "p90_spread_bps",
                "threshold_bps": float(p90_spread_map.threshold_bps),
                "eligible_rule": "p90_spread_bps < threshold_bps",
                "average_spread_fallback": "forbidden",
                "map_symbols": int(len(p90_spread_map.values)),
            },
            "tail_targets": {
                "t1_exact_net_grades": "ready; <=-50,(-50,50],(50,150],(150,250],(250,350],>350 bps",
                "t2_atr_normalized_net_grades": "ready; exact_net_bps / decision_time_atr_bps",
                "t3_first_touch_tbm": (
                    "ready" if t3_ready else "not_materialised; requires exact decision-time path fields " + ",".join(TBM_PATH_COLUMNS)
                ),
                "t3_path_fields": list(TBM_PATH_COLUMNS),
                "t3_never_approximated_from_tp6_sl4_labels": True,
            },
            "coverage_audit": "feature_coverage_audit.parquet",
            "aegmm_source_coverage_audit": "aegmm_source_coverage_audit.parquet",
            "label_spread_audit": "label_spread_audit.parquet",
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        # Do not write a false complete manifest. Retain checkpoint parts for
        # diagnosis rather than silently deleting possibly valuable evidence.
        raise


__all__ = [
    "IDENTITY", "PooledP90SpreadMap", "SCHEMA", "TailBaseInputContractError",
    "build_aegmm_projection_source_contract", "load_frozen_feature_contract", "materialize_tail_base_input_contract",
]
