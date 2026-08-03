#!/usr/bin/env python3
"""Extend the exact historical transition context through December 2023.

This is a deliberately narrow research-only continuation of the frozen-v3
transition contract.  It rebuilds the *entire* August-2022--December-2023
hourly market spine from the immutable compact source using the already-fitted
v3 geometry; rebuilding from January 2023 alone is invalid because
``state_context__state_age_hours`` depends on preceding state history.

Before publishing, the rebuilt August--December 2022 prefix must reproduce
every frozen decision-time feature bit-for-bit (NaNs equal).  Candidate rows
are then joined only when ``__ts__ + 1h == execution_decision_utc``.  Targets,
event identities and all outcome fields from the intermediate transition panel
never appear in the published candidate sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import candidate_identity_sha256  # noqa: E402
from scripts.materialize_frozen_regime_transition_extension import (  # noqa: E402
    DEFAULT_FROZEN_SOURCE,
    DEFAULT_MARKET_SOURCE,
    materialize_extension,
)
from scripts.materialize_historical_exact_transition_context_sidecar import (  # noqa: E402
    EXPECTED_RETAINED_MARKET_FEATURES,
    EXPECTED_TRANSITION_FEATURES,
    EXPECTED_ROWS,
    IDENTITY,
    STATE_CONTEXT_FIELDS,
    _candidate_binding,
    _candidate_identity,
    _sha256,
    build_sidecar,
    coverage_by_year_month_side,
    decision_feature_catalog,
)


SCHEMA = "historical_exact_transition_context_continuation_v1"
START = "2022-08-30T00:00:00Z"
END = "2024-01-01T00:00:00Z"
FROZEN_PREFIX_END = "2023-01-01T00:00:00Z"
EXPECTED_COVERED_ROWS = EXPECTED_ROWS
SOURCE_FAMILY = "frozen_v3_geometry_reconstruction_2022aug_2023dec"
DEFAULT_CANDIDATES = ROOT / (
    "data_perp/artifacts/failure_2022_2023_pf_exact1m_label_inputs_20260730_v2/candidates.parquet"
)
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_CANDIDATES.with_name("manifest.json")
DEFAULT_FROZEN_PREFIX = ROOT / (
    "data_perp/artifacts/regime_transition_research_2022augdec_frozen_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_continuation_20260730_v1"
)


class HistoricalTransitionContinuationError(RuntimeError):
    """Raised when the continuous frozen-geometry reconstruction is unproven."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _manifest(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HistoricalTransitionContinuationError(f"{name} is not an object")
    return value


def _bound_frozen_prefix(directory: Path) -> tuple[Path, dict[str, Any]]:
    """Verify the immutable 2022 prefix that a reconstruction must reproduce."""

    panel = directory / "hourly_transition_dataset.parquet"
    manifest_path = directory / "manifest.json"
    for path in (panel, manifest_path, directory / "manifest.sha256"):
        if not path.is_file():
            raise FileNotFoundError(path)
    signed = (directory / "manifest.sha256").read_text(encoding="utf-8").split()
    if not signed or signed[0] != _sha256(manifest_path):
        raise HistoricalTransitionContinuationError("frozen-prefix manifest detached checksum fails")
    manifest = _manifest(manifest_path, name="frozen-prefix manifest")
    if manifest.get("schema") != "frozen_regime_transition_market_extension_v1":
        raise HistoricalTransitionContinuationError("unexpected frozen-prefix schema")
    if manifest.get("research_only") is not True or manifest.get("promotion_evidence") is not False:
        raise HistoricalTransitionContinuationError("frozen prefix is not marked research-only")
    if manifest.get("full_schema_matches_frozen_v3") is not True:
        raise HistoricalTransitionContinuationError("frozen prefix lacks v3-schema proof")
    if manifest.get("outputs_sha256", {}).get(panel.name) != _sha256(panel):
        raise HistoricalTransitionContinuationError("frozen-prefix panel hash is not manifest-bound")
    interval = manifest.get("source_interval", {})
    if interval.get("start_utc") != "2022-08-30 00:00:00+00:00" or interval.get("end_utc_exclusive") != "2023-01-01 00:00:00+00:00":
        raise HistoricalTransitionContinuationError("frozen-prefix interval is not the required 2022 contract")
    geometry = manifest.get("source_hashes", {}).get("frozen_geometry", {})
    geometry_path = Path(str(geometry.get("path", "")))
    if not geometry_path.is_file() or geometry.get("sha256") != _sha256(geometry_path):
        raise HistoricalTransitionContinuationError(
            "frozen-prefix geometry provenance is absent or no longer hash-bound"
        )
    return panel, {
        "path": str(panel),
        "sha256": _sha256(panel),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "geometry_path": str(geometry_path),
        "geometry_sha256": _sha256(geometry_path),
    }


def assert_frozen_prefix_feature_parity(
    frozen: pd.DataFrame,
    rebuilt: pd.DataFrame,
    *,
    fields: Sequence[str],
) -> dict[str, Any]:
    """Prove feature parity on the frozen 2022 prefix, treating matching NaNs equal."""

    required = {"source_utc", "execution_decision_utc", *fields}
    for name, frame in (("frozen", frozen), ("rebuilt", rebuilt)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise HistoricalTransitionContinuationError(f"{name} panel lacks required fields: {missing}")
    old = frozen.loc[:, ["source_utc", "execution_decision_utc", *fields]].copy()
    new = rebuilt.loc[:, ["source_utc", "execution_decision_utc", *fields]].copy()
    for frame in (old, new):
        frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
        frame["execution_decision_utc"] = pd.to_datetime(frame["execution_decision_utc"], utc=True, errors="raise")
    if old["source_utc"].duplicated().any() or new["source_utc"].duplicated().any():
        raise HistoricalTransitionContinuationError("frozen-prefix source times must be unique")
    new = new.set_index("source_utc").reindex(old["source_utc"])
    if new.index.isna().any() or len(new) != len(old):
        raise HistoricalTransitionContinuationError("rebuilt panel does not cover every frozen-prefix source hour")
    expected_decision = old["source_utc"] + pd.Timedelta(hours=1)
    if not old["execution_decision_utc"].eq(expected_decision).all() or not new["execution_decision_utc"].eq(expected_decision.to_numpy()).all():
        raise HistoricalTransitionContinuationError("frozen-prefix decision-time contract is invalid")
    differences: list[str] = []
    for field in fields:
        left = pd.to_numeric(old[field], errors="coerce").to_numpy(dtype=np.float64)
        right = pd.to_numeric(new[field], errors="coerce").to_numpy(dtype=np.float64)
        same = (left == right) | (np.isnan(left) & np.isnan(right))
        if not bool(np.all(same)):
            differences.append(field)
    if differences:
        raise HistoricalTransitionContinuationError(
            "reconstructed frozen-prefix features differ: " + ", ".join(differences)
        )
    return {
        "frozen_rows": int(len(old)),
        "feature_count": int(len(fields)),
        "compared_source_start_utc": old["source_utc"].min(),
        "compared_source_end_utc": old["source_utc"].max(),
        "status": "EXACT_FEATURE_PARITY",
    }


def build_continuation_sidecar(
    candidates: pd.DataFrame,
    rebuilt: pd.DataFrame,
    *,
    fields: Sequence[str],
    expected_rows: int | None = EXPECTED_ROWS,
    expected_covered_rows: int | None = EXPECTED_COVERED_ROWS,
) -> pd.DataFrame:
    """Build the final candidate sidecar, then require full historical coverage."""

    result = build_sidecar(
        candidates,
        rebuilt.loc[:, ["source_utc", "execution_decision_utc", *fields]],
        feature_columns=fields,
        expected_rows=expected_rows,
    )
    covered = int(result["transition_context_available"].sum())
    if expected_covered_rows is not None and covered != int(expected_covered_rows):
        raise HistoricalTransitionContinuationError(
            f"covered rows must equal {expected_covered_rows}; got {covered}"
        )
    result["source_family"] = np.where(
        result["transition_context_available"], SOURCE_FAMILY, "unavailable_exact_transition_hour"
    )
    if not result["transition_context_available"].all():
        raise HistoricalTransitionContinuationError("continuation left an historical candidate without exact context")
    if result.loc[:, list(fields)].isna().all(axis=1).any():
        raise HistoricalTransitionContinuationError("available context row has no usable frozen-contract features")
    return result


def run(
    *,
    candidates_path: Path = DEFAULT_CANDIDATES,
    candidate_manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    frozen_prefix_dir: Path = DEFAULT_FROZEN_PREFIX,
    frozen_source_dir: Path = DEFAULT_FROZEN_SOURCE,
    market_source: Path = DEFAULT_MARKET_SOURCE,
    destination: Path = DEFAULT_OUTPUT,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_covered_rows: int | None = EXPECTED_COVERED_ROWS,
) -> dict[str, Any]:
    """Atomically rebuild the continuous spine and publish the exact sidecar."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite transition continuation: {destination}")
    for path in (candidates_path, candidate_manifest_path, market_source):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    frozen_prefix_path, frozen_binding = _bound_frozen_prefix(Path(frozen_prefix_dir))
    frozen_template = Path(frozen_source_dir) / "hourly_transition_dataset.parquet"
    for path in (frozen_template, Path(frozen_source_dir) / "pooled_state_geometry.joblib", Path(frozen_source_dir) / "manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    fields = decision_feature_catalog(
        pq.ParquetFile(frozen_template).schema_arrow.names, strict_frozen_contract=True
    )
    sources = {
        "candidates": _candidate_binding(Path(candidates_path), Path(candidate_manifest_path)),
        "frozen_prefix": frozen_binding,
        "frozen_v3_template": {
            "path": str(frozen_template),
            "sha256": _sha256(frozen_template),
            "geometry_path": str(Path(frozen_source_dir) / "pooled_state_geometry.joblib"),
            "geometry_sha256": _sha256(Path(frozen_source_dir) / "pooled_state_geometry.joblib"),
            "manifest_path": str(Path(frozen_source_dir) / "manifest.json"),
            "manifest_sha256": _sha256(Path(frozen_source_dir) / "manifest.json"),
        },
        "compact_market_source": {"path": str(market_source), "sha256": _sha256(Path(market_source))},
    }
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True)
        spine_dir = stage / "frozen_v3_market_spine"
        extension_report = materialize_extension(
            frozen_source_dir=Path(frozen_source_dir),
            market_source=Path(market_source),
            output_dir=spine_dir,
            start=START,
            end=END,
        )
        rebuilt_path = spine_dir / "hourly_transition_dataset.parquet"
        rebuilt = pd.read_parquet(rebuilt_path)
        frozen = pd.read_parquet(frozen_prefix_path)
        parity = assert_frozen_prefix_feature_parity(frozen, rebuilt, fields=fields)
        candidates = pd.read_parquet(candidates_path)
        sidecar = build_continuation_sidecar(
            candidates, rebuilt, fields=fields,
            expected_rows=expected_rows, expected_covered_rows=expected_covered_rows,
        )
        output_path = stage / "context.parquet"
        sidecar.to_parquet(output_path, index=False, compression="zstd", compression_level=5)
        output = {
            "path": str(destination / output_path.name),
            "sha256": _sha256(output_path),
            "rows": int(len(sidecar)),
            "columns": int(len(sidecar.columns)),
            "candidate_identity_sha256": candidate_identity_sha256(sidecar, columns=IDENTITY),
        }
        report = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_EXACT_FROZEN_GEOMETRY_2022_2023_CONTEXT",
            "research_only": True,
            "promotion_eligible": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": _git_revision(),
            "sources": sources,
            "reconstruction": {
                "source_interval": {"start_utc": START, "end_utc_exclusive": END},
                "frozen_geometry_reused": bool(extension_report["frozen_geometry_reused"]),
                "full_schema_matches_frozen_v3": bool(extension_report["full_schema_matches_frozen_v3"]),
                "causal_input_buffer_hours": int(extension_report["causal_input_buffer_hours"]),
                "forward_target_buffer_hours": int(extension_report["forward_target_buffer_hours"]),
                "spine_panel": {
                    "path": str(destination / spine_dir.name / rebuilt_path.name),
                    "sha256": _sha256(rebuilt_path),
                    "rows": int(len(rebuilt)),
                },
                "frozen_prefix_feature_parity": parity,
                "extension_manifest_sha256": _sha256(spine_dir / "manifest.json"),
            },
            "identity_contract": "ordered exact four-key candidate identity is unchanged",
            "time_contract": "candidate __decision_ts__ (or __ts__ + 1h, asserted equal when both exist) exactly joins execution_decision_utc; execution_decision_utc == source_utc + 1h; reconstruction reads a 24h causal source buffer; no asof/fill/interpolation",
            "feature_catalog": {
                "authoritative_contract": "frozen v3 schema, feature order and fitted geometry reused unchanged",
                "retained_market_feature_count": EXPECTED_RETAINED_MARKET_FEATURES,
                "transition_feature_count": EXPECTED_TRANSITION_FEATURES,
                "state_context_fields": list(STATE_CONTEXT_FIELDS),
                "output_feature_count": len(fields),
                "fields": list(fields),
            },
            "coverage": {
                "candidate_rows": int(len(sidecar)),
                "covered_rows": int(sidecar["transition_context_available"].sum()),
                "unavailable_rows": int((~sidecar["transition_context_available"]).sum()),
                "by_year_month_side": coverage_by_year_month_side(sidecar),
            },
            "target_exclusion": "target__*, event IDs, target availability, before/after labels, economics and every outcome column are excluded from context.parquet; intermediate labels exist only to maintain the frozen transition computation contract",
            "caveat": "This restores a frozen-geometry market-state feature continuation, not comparable historical economics, OOF evidence, a policy input, or promotion evidence.",
            "output": output,
        }
        _write_json(stage / "report.json", report)
        manifest = {
            "schema": SCHEMA,
            "status": report["status"],
            "report": {"path": str(destination / "report.json"), "sha256": _sha256(stage / "report.json")},
            "sources": sources,
            "reconstruction": report["reconstruction"],
            "output": output,
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            f"{_sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8"
        )
        os.replace(stage, destination)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    value.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    value.add_argument("--frozen-prefix-dir", type=Path, default=DEFAULT_FROZEN_PREFIX)
    value.add_argument("--frozen-source-dir", type=Path, default=DEFAULT_FROZEN_SOURCE)
    value.add_argument("--market-source", type=Path, default=DEFAULT_MARKET_SOURCE)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    value.add_argument("--expected-covered-rows", type=int, default=EXPECTED_COVERED_ROWS)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(_jsonable(run(
        candidates_path=arguments.candidates,
        candidate_manifest_path=arguments.candidate_manifest,
        frozen_prefix_dir=arguments.frozen_prefix_dir,
        frozen_source_dir=arguments.frozen_source_dir,
        market_source=arguments.market_source,
        destination=arguments.output_dir,
        expected_rows=arguments.expected_rows,
        expected_covered_rows=arguments.expected_covered_rows,
    )), indent=2, sort_keys=True))
