"""Deterministic disk-to-production execution boundary for Stage I.

The selector and winner freezer intentionally stop before loading the complete
2022--2026 population.  This module closes that gap without broadening the
frozen model contract:

* materialise only the selected raw base/meta union in atomic monthly parts;
* preserve and report source-calendar gaps (notably 2024-12);
* construct the exact two :class:`StageISideProductionInput` objects; and
* cache each expensive strict-OOF side result behind content hashes.

No feature discovery, HPO, target changes, as-of joins, or missing-value fills
are permitted here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .packb_static_point_feature_loader import (
    FrozenFeatureContract,
    _feature_contract_digest,
)
from .stage_i_feature_selection import STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
from .stage_i_production_data_adapter import (
    MonthlyReferencePartition,
    PointFeatureLoader,
    _read_partition,
    make_static_pit_feature_loader,
)
from .stage_i_production_oos import (
    StageIProductionOOSError,
    StageIProductionWinnerBundle,
    StageISideProductionInput,
    run_stage_i_production_oos,
)
from .stage_i_strict_oof import (
    StageIStrictOOFPlan,
    StageIStrictOOFResult,
    generate_stage_i_strict_oof,
)
from .stage_i_winner_bundle import load_stage_i_production_source_binding


PANEL_SCHEMA = "stage_i_selected_raw_production_panel_v1"
CACHE_SCHEMA = "stage_i_side_strict_oof_cache_v1"
SIDES = ("long", "short")
IDENTITY_COLUMNS = ("candidate_id", "__ts__", "__symbol__")
LEDGER_COLUMNS = (
    *IDENTITY_COLUMNS,
    "side_name",
    "label_available_ts",
    "exact_gross_bps",
    "exact_net_bps",
    "t2_tp6_sl4_event",
    "robust_clear_event_b25",
    "robust_clear_soft_b25_t50",
    "r3_target",
)


class StageIProductionExecutionError(StageIProductionOOSError):
    """Raised when disk materialisation/execution differs from the winner."""


def _canonical_bytes(value: Any) -> bytes:
    def default(item: Any) -> Any:
        if isinstance(item, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(item).isoformat()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=default
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_binding(bundle: StageIProductionWinnerBundle) -> dict[str, Any]:
    values = [dict(cell.source_manifest) for cell in bundle.cells]
    if not values or any(_canonical_bytes(value) != _canonical_bytes(values[0]) for value in values[1:]):
        raise StageIProductionExecutionError(
            "all four winner cells must bind the same immutable production source"
        )
    expected = _digest(values[0])
    if any(cell.source_manifest_sha256 != expected for cell in bundle.cells):
        raise StageIProductionExecutionError("winner source hashes disagree")
    return values[0]


def _selected_raw_by_side(
    bundle: StageIProductionWinnerBundle,
) -> dict[str, tuple[str, ...]]:
    generated = set(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES)
    output: dict[str, tuple[str, ...]] = {}
    for side in SIDES:
        ordered: list[str] = []
        for layer in ("base", "meta"):
            for feature in bundle.cell(layer=layer, side=side).selected_feature_names:
                if feature not in generated and feature not in ordered:
                    ordered.append(feature)
        if not ordered:
            raise StageIProductionExecutionError(f"{side} has no selected raw production fields")
        output[side] = tuple(ordered)
    return output


def _selected_readiness_by_side(
    bundle: StageIProductionWinnerBundle,
    selected: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, str]]:
    """Resolve one exact readiness boundary for every selected raw field."""
    source_start = pd.to_datetime(
        _source_binding(bundle)["production_input_manifest"]["min_signal_ts"],
        utc=True, errors="raise",
    )
    evaluation_start = pd.to_datetime(
        bundle.calendar.evaluation_start_utc, utc=True, errors="raise"
    )
    output: dict[str, dict[str, str]] = {}
    for side in SIDES:
        declared: dict[str, pd.Timestamp] = {}
        for layer in ("base", "meta"):
            cell = bundle.cell(layer=layer, side=side)
            raw = cell.selector_manifest.get("selected_feature_readiness", {})
            if not isinstance(raw, Mapping):
                raise StageIProductionExecutionError(
                    f"{side}/{layer} selected_feature_readiness must be a mapping"
                )
            for feature, record in raw.items():
                if not isinstance(record, Mapping) or "first_ready_timestamp_utc" not in record:
                    raise StageIProductionExecutionError(
                        f"{side}/{layer}/{feature} readiness lacks first_ready_timestamp_utc"
                    )
                boundary = pd.to_datetime(
                    record["first_ready_timestamp_utc"], utc=True, errors="raise"
                )
                if feature in declared and declared[feature] != boundary:
                    raise StageIProductionExecutionError(
                        f"{side}/{feature} readiness differs across base/meta selectors"
                    )
                declared[str(feature)] = boundary
        side_map: dict[str, str] = {}
        for feature in selected[side]:
            boundary = declared.get(str(feature), source_start)
            if boundary < source_start or boundary > evaluation_start:
                raise StageIProductionExecutionError(
                    f"{side}/{feature} readiness is outside source start/required evaluation boundary"
                )
            side_map[str(feature)] = boundary.isoformat()
        unknown = sorted(set(declared) - set(selected[side]))
        if unknown:
            raise StageIProductionExecutionError(
                f"{side} readiness declares unselected raw fields: {unknown[:12]}"
            )
        output[side] = side_map
    return output


def _subset_contract(
    contract: FrozenFeatureContract, fields: Sequence[str]
) -> FrozenFeatureContract:
    selected = tuple(sorted(set(map(str, fields))))
    if not selected or not set(selected).issubset(contract.feature_columns):
        missing = sorted(set(selected) - set(contract.feature_columns))
        raise StageIProductionExecutionError(
            f"selected production fields are absent from frozen store contract: {missing[:12]}"
        )
    digest = _feature_contract_digest(
        feature_columns=selected,
        candidate_universe_sha256=contract.candidate_universe_sha256,
        source_schema_sha256=contract.source_schema_sha256,
        raw_allowlist_sha256=contract.raw_allowlist_sha256,
        generator_registry_sha256=contract.generator_registry_sha256,
        store_scan_manifest_sha256=contract.store_scan_manifest_sha256,
        coverage_profile_sha256=contract.coverage_profile_sha256,
        min_exact_key_coverage=contract.min_exact_key_coverage,
        min_non_null_feature_coverage=contract.min_non_null_feature_coverage,
        max_feature_columns=contract.max_feature_columns,
        coverage_admission_rejections=contract.coverage_admission_rejections,
    )
    return replace(contract, feature_columns=selected, feature_contract_sha256=digest)


def _source_partitions(input_contract_dir: Path) -> list[MonthlyReferencePartition]:
    path = input_contract_dir / "reference_partitions.parquet"
    if not path.is_file():
        raise StageIProductionExecutionError(f"reference partition ledger is missing: {path}")
    frame = pd.read_parquet(path)
    required = {"path", "source_month", "population"}
    if required - set(frame.columns) or frame.empty:
        raise StageIProductionExecutionError("reference partition ledger has an invalid schema")
    output = [
        MonthlyReferencePartition(str(row.path), str(row.source_month), str(row.population))
        for row in frame.itertuples(index=False)
    ]
    if len({(str(item.path), item.source_month, item.population) for item in output}) != len(output):
        raise StageIProductionExecutionError("reference partition ledger contains duplicates")
    return output


def _calendar_gaps(partitions: Sequence[MonthlyReferencePartition]) -> list[str]:
    observed = pd.PeriodIndex(sorted({item.source_month for item in partitions}), freq="M")
    if observed.empty:
        return []
    expected = pd.period_range(observed.min(), observed.max(), freq="M")
    return [str(value) for value in expected.difference(observed)]


def _source_files(partitions: Sequence[MonthlyReferencePartition]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in partitions:
        root = Path(item.path)
        files = [root] if root.is_file() else sorted(root.glob("*.parquet"))
        if not files:
            raise StageIProductionExecutionError(f"source partition is missing: {root}")
        records.append({
            "path": str(root.resolve()),
            "source_month": item.source_month,
            "population": item.population,
            "files": [
                {"path": str(path.resolve()), "sha256": _file_sha256(path)} for path in files
            ],
        })
    return records


def _r3_target(frame: pd.DataFrame) -> np.ndarray:
    adverse = pd.to_numeric(frame["t2_tp6_sl4_event"], errors="coerce").eq(1.0)
    clear = pd.to_numeric(frame["robust_clear_event_b25"], errors="coerce").eq(1.0)
    if adverse.isna().any() or clear.isna().any() or (adverse & clear).any():
        raise StageIProductionExecutionError("R3 primitives are missing or contradictory")
    return np.select([adverse, clear], [0, 2], default=1).astype(np.int8)


def _validate_loaded_features(
    ledger: pd.DataFrame, loaded: pd.DataFrame, fields: Sequence[str]
) -> pd.DataFrame:
    expected = [*IDENTITY_COLUMNS, *map(str, fields)]
    if list(loaded.columns) != expected:
        raise StageIProductionExecutionError("PIT loader returned a widened/reordered feature matrix")
    left = ledger.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
    right = loaded.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
    if not left.equals(right):
        raise StageIProductionExecutionError("PIT loader changed exact candidate/symbol/signal identity")
    result = loaded.loc[:, list(map(str, fields))].copy()
    for feature in fields:
        result[feature] = pd.to_numeric(result[feature], errors="coerce").astype(np.float32)
    return result


@dataclass
class _Coverage:
    whole_rows: int = 0
    whole_finite: int = 0
    post_rows: int = 0
    post_finite: int = 0
    evaluation_rows: int = 0
    evaluation_finite: int = 0
    pre_readiness_finite: int = 0
    minimum: float = np.inf
    maximum: float = -np.inf

    def update(
        self, values: pd.Series, signal: pd.Series, *,
        readiness: pd.Timestamp, evaluation_start: pd.Timestamp,
        evaluation_end: pd.Timestamp,
    ) -> None:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(numeric)
        before = signal.lt(readiness).to_numpy()
        post = signal.ge(readiness).to_numpy()
        evaluation = signal.between(
            evaluation_start, evaluation_end, inclusive="both"
        ).to_numpy()
        self.whole_rows += len(numeric)
        self.whole_finite += int(finite.sum())
        self.post_rows += int(post.sum())
        self.post_finite += int((finite & post).sum())
        self.evaluation_rows += int(evaluation.sum())
        self.evaluation_finite += int((finite & evaluation).sum())
        self.pre_readiness_finite += int((finite & before).sum())
        valid = numeric[finite & post]
        if len(valid):
            self.minimum = min(self.minimum, float(valid.min()))
            self.maximum = max(self.maximum, float(valid.max()))


def _coverage_records(
    parts: Mapping[str, Sequence[Path]], selected: Mapping[str, Sequence[str]],
    readiness: Mapping[str, Mapping[str, str]],
    bundle: StageIProductionWinnerBundle,
) -> pd.DataFrame:
    start = pd.Timestamp(bundle.calendar.evaluation_start_utc)
    end = pd.Timestamp(bundle.calendar.evaluation_end_utc)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    states: dict[tuple[str, str], _Coverage] = {}
    for side in SIDES:
        for path in parts[side]:
            columns = ["__ts__", *selected[side]]
            frame = pd.read_parquet(path, columns=columns)
            signal = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
            if signal.isna().any():
                raise StageIProductionExecutionError(f"invalid signal timestamps in {path}")
            for feature in selected[side]:
                states.setdefault((side, feature), _Coverage()).update(
                    frame[feature], signal,
                    readiness=pd.Timestamp(readiness[side][feature]),
                    evaluation_start=start, evaluation_end=end,
                )
    rows: list[dict[str, Any]] = []
    for (side, feature), state in sorted(states.items()):
        post_coverage = state.post_finite / state.post_rows if state.post_rows else 0.0
        eval_coverage = (
            state.evaluation_finite / state.evaluation_rows
            if state.evaluation_rows else 0.0
        )
        nonconstant = bool(state.post_finite and state.minimum < state.maximum)
        boundary = pd.Timestamp(readiness[side][feature])
        passed = bool(
            boundary <= start
            and state.pre_readiness_finite == 0
            and post_coverage >= 0.90
            and eval_coverage >= 0.90
            and nonconstant
        )
        common = {
            "side_name": side, "feature_name": feature,
            "first_ready_timestamp_utc": boundary,
            "pre_readiness_finite_rows": state.pre_readiness_finite,
            "post_readiness_rows": state.post_rows,
            "post_readiness_finite_rows": state.post_finite,
            "post_readiness_finite_coverage": post_coverage,
            "required_evaluation_rows": state.evaluation_rows,
            "required_evaluation_finite_rows": state.evaluation_finite,
            "required_evaluation_finite_coverage": eval_coverage,
            "minimum": state.minimum if state.post_finite else np.nan,
            "maximum": state.maximum if state.post_finite else np.nan,
            "nonconstant": nonconstant,
            "gate": "post_readiness_and_required_evaluation_aggregate_ge_0.90",
            "status": "pass" if passed else "fail",
        }
        rows.extend([
            {
                **common, "scope": "whole_side",
                "rows": state.whole_rows, "finite_rows": state.whole_finite,
                "finite_coverage": (
                    state.whole_finite / state.whole_rows if state.whole_rows else 0.0
                ),
            },
            {
                **common, "scope": "evaluation_window",
                "rows": state.evaluation_rows,
                "finite_rows": state.evaluation_finite,
                "finite_coverage": eval_coverage,
            },
        ])
    audit = pd.DataFrame(rows)
    if audit.empty or audit.status.ne("pass").any():
        failures = audit.loc[audit.status.ne("pass")].head(12).to_dict(orient="records")
        raise StageIProductionExecutionError(f"selected panel coverage audit failed: {failures}")
    return audit


LoaderFactory = Callable[[FrozenFeatureContract, bool], PointFeatureLoader]


def materialize_stage_i_selected_panels(
    bundle: StageIProductionWinnerBundle,
    *,
    input_contract_dir: str | Path,
    output_dir: str | Path,
    resume: bool = False,
    max_rows_per_batch: int = 4_000,
    max_columns_per_read: int = 64,
    loader_factory: LoaderFactory | None = None,
) -> Mapping[str, Any]:
    """Create/reuse atomic monthly selected-only PIT panels."""
    source = _source_binding(bundle)
    current_source = load_stage_i_production_source_binding(input_contract_dir)
    if _canonical_bytes(source) != _canonical_bytes(current_source):
        raise StageIProductionExecutionError("winner source differs from production input contract")
    root = Path(output_dir)
    final_manifest = root / "manifest.json"
    if final_manifest.exists():
        if not resume:
            raise FileExistsError(f"selected panel already exists without --resume: {root}")
        manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
        if manifest.get("winner_bundle_sha256") != bundle.sha256:
            raise FileExistsError("selected panel belongs to another winner bundle")
        for record in manifest.get("source_partitions", []):
            for source_file in record.get("files", []):
                source_path = Path(source_file["path"])
                if not source_path.is_file() or _file_sha256(source_path) != source_file["sha256"]:
                    raise StageIProductionExecutionError(
                        f"selected panel source partition drift: {source_path}"
                    )
        for record in manifest.get("parts", []):
            path = root / str(record["relative_path"])
            if not path.is_file() or _file_sha256(path) != record["sha256"]:
                raise StageIProductionExecutionError(f"selected panel checkpoint drift: {path}")
            checkpoint_path = path.parent / "manifest.json"
            if (
                not checkpoint_path.is_file()
                or _file_sha256(checkpoint_path) != record["checkpoint_manifest_sha256"]
            ):
                raise StageIProductionExecutionError(
                    f"selected panel checkpoint manifest drift: {checkpoint_path}"
                )
        coverage = manifest.get("coverage_audit", {})
        coverage_path = root / str(coverage.get("relative_path", ""))
        if not coverage_path.is_file() or _file_sha256(coverage_path) != coverage.get("sha256"):
            raise StageIProductionExecutionError("selected panel coverage audit checksum drift")
        return {**manifest, "restart_status": "reused_verified_selected_panel"}
    if root.exists() and not resume:
        raise FileExistsError(f"selected panel path exists without --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)

    input_root = Path(input_contract_dir)
    contract = FrozenFeatureContract.from_mapping(json.loads(
        (input_root / "frozen_feature_contract.json").read_text(encoding="utf-8")
    ))
    selected = _selected_raw_by_side(bundle)
    readiness = _selected_readiness_by_side(bundle, selected)
    union = tuple(sorted(set(selected["long"]) | set(selected["short"])))
    subset = _subset_contract(contract, union)
    input_manifest = source["production_input_manifest"]

    if loader_factory is None:
        def loader_factory(local_contract: FrozenFeatureContract, verify: bool) -> PointFeatureLoader:
            return make_static_pit_feature_loader(
                feature_store_dir=input_manifest["feature_store"],
                feature_contract=local_contract,
                max_rows_per_batch=max_rows_per_batch,
                max_columns_per_read=max_columns_per_read,
                verify_frozen_schema=verify,
            )

    partitions = _source_partitions(input_root)
    source_records = _source_files(partitions)
    source_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in source_records:
        source_by_group.setdefault(
            (record["source_month"], record["population"]), []
        ).append(record)
    for records in source_by_group.values():
        records.sort(key=lambda record: record["path"])
    by_month: dict[tuple[str, str], list[MonthlyReferencePartition]] = {}
    for partition in partitions:
        by_month.setdefault((partition.source_month, partition.population), []).append(partition)
    part_records: list[dict[str, Any]] = []
    parts: dict[str, list[Path]] = {side: [] for side in SIDES}
    schema_verified = False
    for (month, population), group in sorted(by_month.items()):
        source_record = source_by_group[(month, population)]
        checkpoint_contract = {
            "schema": "stage_i_selected_raw_month_checkpoint_v1",
            "winner_bundle_sha256": bundle.sha256,
            "source_binding_sha256": _digest(source),
            "month": month, "population": population,
            "source_record": source_record,
            "subset_feature_contract_sha256": subset.feature_contract_sha256,
        }
        reusable: dict[str, tuple[Path, dict[str, Any]]] = {}
        for side in SIDES:
            destination = root / "parts" / f"side={side}" / f"month={month}" / "panel.parquet"
            checkpoint_path = destination.parent / "manifest.json"
            if not destination.exists() or not checkpoint_path.exists():
                continue
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            expected = {
                **checkpoint_contract,
                "side": side, "selected_raw_features": list(selected[side]),
            }
            for key, value in expected.items():
                if checkpoint.get(key) != value:
                    raise StageIProductionExecutionError(
                        f"monthly selected-panel checkpoint contract drift: {checkpoint_path}"
                    )
            if checkpoint.get("panel_sha256") != _file_sha256(destination):
                raise StageIProductionExecutionError(
                    f"monthly selected-panel checkpoint checksum drift: {destination}"
                )
            reusable[side] = (destination, checkpoint)
        if len(reusable) == len(SIDES):
            if not resume:
                raise FileExistsError(
                    f"monthly selected panels exist without --resume: {month}/{population}"
                )
            for side in SIDES:
                destination, checkpoint = reusable[side]
                part_records.append({
                    "side": side, "month": month, "population": population,
                    "relative_path": destination.relative_to(root).as_posix(),
                    "rows": int(checkpoint["rows"]),
                    "sha256": str(checkpoint["panel_sha256"]),
                    "checkpoint_manifest_sha256": _file_sha256(destination.parent / "manifest.json"),
                })
                parts[side].append(destination)
            continue
        month_ledger = pd.concat([_read_partition(item) for item in group], ignore_index=True)
        month_ledger = month_ledger.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if month_ledger.candidate_id.duplicated().any():
            raise StageIProductionExecutionError(f"duplicate candidate ids in {month}/{population}")
        month_ledger["r3_target"] = _r3_target(month_ledger)
        loader = loader_factory(subset, not schema_verified)
        loaded = _validate_loaded_features(month_ledger, loader(month_ledger, union), union)
        schema_verified = True
        for side in SIDES:
            mask = month_ledger.side_name.eq(side).to_numpy()
            if not mask.any():
                continue
            destination = root / "parts" / f"side={side}" / f"month={month}" / "panel.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame = month_ledger.loc[mask, list(LEDGER_COLUMNS)].reset_index(drop=True)
            selected_values = loaded.loc[mask, list(selected[side])].reset_index(drop=True)
            signal = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
            for feature in selected[side]:
                boundary = pd.Timestamp(readiness[side][feature])
                before = signal.lt(boundary).to_numpy()
                numeric = pd.to_numeric(selected_values[feature], errors="coerce").to_numpy(float)
                if np.isfinite(numeric[before]).any():
                    raise StageIProductionExecutionError(
                        f"{side}/{feature} has a finite value before its frozen readiness boundary"
                    )
                selected_values.loc[before, feature] = np.nan
            frame.loc[:, list(selected[side])] = selected_values
            expected_columns = [*LEDGER_COLUMNS, *selected[side]]
            frame = frame.loc[:, expected_columns]
            if destination.exists() and side in reusable:
                # The checkpoint contract and full file checksum were already
                # verified above.  Recomputing the peer side must not rewrite
                # this immutable monthly result.
                pass
            else:
                if destination.exists() and not resume:
                    raise FileExistsError(f"monthly selected panel exists without --resume: {destination}")
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".panel.", suffix=".parquet.tmp", dir=destination.parent
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    frame.to_parquet(temporary, index=False, compression="zstd")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            record = {
                "side": side, "month": month, "population": population,
                "relative_path": destination.relative_to(root).as_posix(),
                "rows": int(len(frame)), "sha256": _file_sha256(destination),
            }
            checkpoint = {
                **checkpoint_contract, "status": "complete", "side": side,
                "selected_raw_features": list(selected[side]),
                "rows": int(len(frame)), "panel_sha256": record["sha256"],
            }
            _write_json_atomic(destination.parent / "manifest.json", checkpoint)
            record["checkpoint_manifest_sha256"] = _file_sha256(
                destination.parent / "manifest.json"
            )
            part_records.append(record)
            parts[side].append(destination)

    coverage = _coverage_records(parts, selected, readiness, bundle)
    coverage_path = root / "selected_raw_feature_coverage.parquet"
    coverage.to_parquet(coverage_path, index=False, compression="zstd")
    manifest: dict[str, Any] = {
        "schema": PANEL_SCHEMA, "status": "complete",
        "winner_bundle_sha256": bundle.sha256,
        "source_binding": source, "source_binding_sha256": _digest(source),
        "frozen_subset_feature_contract": subset.to_dict(),
        "selected_raw_features_by_side": {side: list(selected[side]) for side in SIDES},
        "selected_feature_readiness_by_side": readiness,
        "controls": {
            "exact_pit_join": "feature ts == candidate signal-close __ts__",
            "no_asof": True, "no_fill": True, "n_validation_folds": 4,
            "min_train_rows": 500,
        },
        "source_partitions": source_records,
        "parts": sorted(part_records, key=lambda row: (row["side"], row["month"])),
        "calendar_gaps": _calendar_gaps(partitions),
        "calendar_gap_disposition": "preserved_and_disclosed_no_fabrication_or_backfill",
        "coverage_audit": {
            "relative_path": coverage_path.relative_to(root).as_posix(),
            "sha256": _file_sha256(coverage_path), "minimum": 0.90,
            "scopes": ["whole_side", "evaluation_window"],
        },
    }
    _write_json_atomic(final_manifest, manifest)
    return manifest


def load_stage_i_side_production_inputs(
    bundle: StageIProductionWinnerBundle,
    *, selected_panel_dir: str | Path,
) -> list[StageISideProductionInput]:
    """Load selected-only monthly parts into the existing production API."""
    root = Path(selected_panel_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != PANEL_SCHEMA or manifest.get("status") != "complete":
        raise StageIProductionExecutionError("selected production panel is incomplete")
    if manifest.get("winner_bundle_sha256") != bundle.sha256:
        raise StageIProductionExecutionError("selected panel/winner bundle mismatch")
    source = _source_binding(bundle)
    if manifest.get("source_binding_sha256") != _digest(source):
        raise StageIProductionExecutionError("selected panel source binding mismatch")
    coverage = manifest.get("coverage_audit", {})
    coverage_path = root / str(coverage.get("relative_path", ""))
    if not coverage_path.is_file() or _file_sha256(coverage_path) != coverage.get("sha256"):
        raise StageIProductionExecutionError("selected panel coverage audit checksum drift")
    controls = manifest.get("controls", {})
    selected = _selected_raw_by_side(bundle)
    readiness = manifest.get("selected_feature_readiness_by_side", {})
    if not isinstance(readiness, Mapping):
        raise StageIProductionExecutionError("selected panel readiness contract is missing")
    materialized_manifest_sha256 = _digest(manifest)
    outputs: list[StageISideProductionInput] = []
    for side in SIDES:
        records = [record for record in manifest["parts"] if record["side"] == side]
        records.sort(key=lambda row: (row["month"], row["relative_path"]))
        pieces: list[pd.DataFrame] = []
        for record in records:
            path = root / record["relative_path"]
            if _file_sha256(path) != record["sha256"]:
                raise StageIProductionExecutionError(f"selected panel part checksum drift: {path}")
            checkpoint_path = path.parent / "manifest.json"
            if _file_sha256(checkpoint_path) != record["checkpoint_manifest_sha256"]:
                raise StageIProductionExecutionError(
                    f"selected panel checkpoint manifest drift: {checkpoint_path}"
                )
            pieces.append(pd.read_parquet(path, columns=[*LEDGER_COLUMNS, *selected[side]]))
        if not pieces:
            raise StageIProductionExecutionError(f"selected panel has no {side} parts")
        ledger = pd.concat(pieces, ignore_index=True).sort_values(
            ["__ts__", "candidate_id"], kind="stable"
        ).reset_index(drop=True)
        if ledger.candidate_id.duplicated().any() or not ledger.side_name.eq(side).all():
            raise StageIProductionExecutionError(f"selected panel {side} identity/side mismatch")
        signal = pd.to_datetime(ledger["__ts__"], utc=True, errors="raise")
        outputs.append(StageISideProductionInput(
            side=side,
            candidate_ids=ledger.candidate_id.to_numpy(dtype=object),
            symbols=ledger["__symbol__"].astype(str).to_numpy(dtype=object),
            signal_close_timestamps=signal,
            decision_timestamps=signal + pd.Timedelta(hours=1),
            label_available_timestamps=pd.to_datetime(ledger.label_available_ts, utc=True, errors="raise"),
            frame=ledger.loc[:, list(selected[side])].astype(np.float32, copy=False),
            r3_target=ledger.r3_target.to_numpy(dtype=np.int8),
            exact_net_bps=ledger.exact_net_bps.to_numpy(dtype=np.float32),
            exact_gross_bps=ledger.exact_gross_bps.to_numpy(dtype=np.float32),
            panel_manifest=source,
            panel_manifest_sha256=_digest(source),
            sample_weight=None,
            n_validation_folds=int(controls.get("n_validation_folds", 4)),
            min_train_rows=int(controls.get("min_train_rows", 500)),
            materialized_panel_manifest_sha256=materialized_manifest_sha256,
            materialized_panel_content_sha256=_digest({
                "side": side,
                "parts": [
                    {
                        "relative_path": record["relative_path"],
                        "rows": int(record["rows"]),
                        "sha256": record["sha256"],
                        "checkpoint_manifest_sha256": record["checkpoint_manifest_sha256"],
                    }
                    for record in records
                ],
                "coverage_audit_sha256": coverage["sha256"],
                "selected_feature_readiness": dict(readiness[side]),
            }),
            selected_feature_readiness=dict(readiness[side]),
        ))
    return outputs


def _cache_key(
    plan: StageIStrictOOFPlan,
    *, bundle: StageIProductionWinnerBundle,
    selected_panel_manifest: Mapping[str, Any],
) -> str:
    side_parts = [
        {key: row[key] for key in ("relative_path", "rows", "sha256")}
        for row in selected_panel_manifest["parts"] if row["side"] == plan.side
    ]
    return _digest({
        "schema": CACHE_SCHEMA, "winner_bundle_sha256": bundle.sha256,
        "side": plan.side, "parts": side_parts,
        "base_features": list(plan.base_feature_names),
        "meta_features": list(plan.meta_feature_names),
        "base_params": dict(plan.base_params), "residual_params": dict(plan.residual_params),
        "n_validation_folds": int(plan.n_validation_folds),
        "min_train_rows": int(plan.min_train_rows),
        "value_map": asdict(plan.value_map) if plan.value_map is not None else None,
        "sample_weight_contract": "uniform_none_from_selected_production_panel",
    })


def make_cached_stage_i_strict_generator(
    *,
    bundle: StageIProductionWinnerBundle,
    selected_panel_dir: str | Path,
    cache_dir: str | Path,
    generate: Callable[[StageIStrictOOFPlan], StageIStrictOOFResult] = generate_stage_i_strict_oof,
) -> Callable[[StageIStrictOOFPlan], StageIStrictOOFResult]:
    """Return a side-result cache compatible with ``run_stage_i_production_oos``."""
    panel_root = Path(selected_panel_dir)
    panel_manifest = json.loads((panel_root / "manifest.json").read_text(encoding="utf-8"))
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    def cached(plan: StageIStrictOOFPlan) -> StageIStrictOOFResult:
        key = _cache_key(plan, bundle=bundle, selected_panel_manifest=panel_manifest)
        destination = cache_root / f"side={plan.side}" / key
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != CACHE_SCHEMA or manifest.get("cache_key") != key:
                raise StageIProductionExecutionError(f"strict OOF cache manifest mismatch: {destination}")
            for name, checksum in manifest["checksums"].items():
                if _file_sha256(destination / name) != checksum:
                    raise StageIProductionExecutionError(f"strict OOF cache checksum drift: {destination / name}")
            metadata = {
                "value_map_provenance": manifest["value_map_provenance"],
                "plan_summary": manifest["plan_summary"],
            }
            if manifest.get("metadata_sha256") != _digest(metadata):
                raise StageIProductionExecutionError(
                    f"strict OOF cache metadata drift: {manifest_path}"
                )
            result = StageIStrictOOFResult(
                side=plan.side,
                predictions=pd.read_parquet(destination / "predictions.parquet"),
                fold_provenance=pd.read_parquet(destination / "fold_provenance.parquet"),
                value_map_provenance=manifest["value_map_provenance"],
                plan_summary=manifest["plan_summary"],
            )
            if (
                len(result.predictions) != len(plan.frame)
                or result.side != plan.side
                or set(result.predictions.get("candidate_id", ()))
                != set(np.asarray(plan.candidate_ids, dtype=object))
            ):
                raise StageIProductionExecutionError(
                    f"strict OOF cache population mismatch: {destination}"
                )
            return result
        if destination.exists():
            raise FileExistsError(f"incomplete strict OOF cache exists: {destination}")
        result = generate(plan)
        if result.side != plan.side or len(result.predictions) != len(plan.frame):
            raise StageIProductionExecutionError(
                f"strict OOF generator changed the frozen {plan.side} population"
            )
        result = StageIStrictOOFResult(
            side=result.side,
            predictions=result.predictions,
            fold_provenance=result.fold_provenance,
            value_map_provenance=result.value_map_provenance,
            plan_summary={**dict(result.plan_summary), "strict_oof_cache_key": key},
        )
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary_parent = Path(tempfile.mkdtemp(prefix=f".{key}.tmp-", dir=parent))
        temporary = temporary_parent / key
        try:
            temporary.mkdir()
            result.predictions.to_parquet(temporary / "predictions.parquet", index=False, compression="zstd")
            result.fold_provenance.to_parquet(temporary / "fold_provenance.parquet", index=False, compression="zstd")
            checksums = {
                name: _file_sha256(temporary / name)
                for name in ("predictions.parquet", "fold_provenance.parquet")
            }
            metadata = {
                "value_map_provenance": dict(result.value_map_provenance),
                "plan_summary": dict(result.plan_summary),
            }
            _write_json_atomic(temporary / "manifest.json", {
                "schema": CACHE_SCHEMA, "status": "complete", "side": plan.side,
                "cache_key": key, "checksums": checksums,
                **metadata, "metadata_sha256": _digest(metadata),
            })
            os.replace(temporary, destination)
            return result
        except Exception:
            shutil.rmtree(temporary_parent, ignore_errors=True)
            raise
        finally:
            if temporary_parent.exists():
                temporary_parent.rmdir()

    return cached


def execute_stage_i_production_oos(
    bundle: StageIProductionWinnerBundle,
    *,
    input_contract_dir: str | Path,
    selected_panel_dir: str | Path,
    strict_oof_cache_dir: str | Path,
    output_dir: str | Path,
    resume: bool = False,
    max_rows_per_batch: int = 4_000,
    max_columns_per_read: int = 64,
) -> Mapping[str, Any]:
    """Materialise selected inputs, resume side OOF, and publish final Stage I."""
    materialize_stage_i_selected_panels(
        bundle, input_contract_dir=input_contract_dir, output_dir=selected_panel_dir,
        resume=resume, max_rows_per_batch=max_rows_per_batch,
        max_columns_per_read=max_columns_per_read,
    )
    inputs = load_stage_i_side_production_inputs(bundle, selected_panel_dir=selected_panel_dir)
    generate = make_cached_stage_i_strict_generator(
        bundle=bundle, selected_panel_dir=selected_panel_dir,
        cache_dir=strict_oof_cache_dir,
    )
    return run_stage_i_production_oos(bundle, inputs, output_dir, generate=generate)


__all__ = [
    "CACHE_SCHEMA", "PANEL_SCHEMA", "StageIProductionExecutionError",
    "execute_stage_i_production_oos", "load_stage_i_side_production_inputs",
    "make_cached_stage_i_strict_generator", "materialize_stage_i_selected_panels",
]
