#!/usr/bin/env python3
"""Materialize a strict current-2026 transition-feature sidecar for Primary100.

The cross-era transition panel contains targets and future-only research
columns, so this materializer deliberately reads only the manifest-declared
``feature_columns``.  Current 12-hour transition rows are broadcast to the
frozen 134,889-row Primary100 candidate universe on an *exact* decision-time
timestamp match.  It never performs an as-of join or fills an unavailable
anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "primary100_current_transition_feature_sidecar_v1"
EXPECTED_ROWS = 134_889
EXPECTED_UNMATCHED_ROWS = 23_177
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DECISION = "execution_decision_utc"
ANCHOR = "cohort_anchor_utc"
AVAILABILITY = "transition_context_available"
CURRENT_SOURCE = "current_exact_spread_mayjul2026"
HORIZON_HOURS = 12
PANEL_SCHEMA = "cross_era_global_book_transition_research_panel_v4"
PANEL_STATUS = "SOURCE_SEPARATED_TRANSITION_RESEARCH_PANEL_COMPLETE"
UNIVERSE_SCHEMA = "exact_policy_capture_causal_feature_universe_v1"
UNIVERSE_STATUS = "completed_outcome_free_feature_universe"

DEFAULT_UNIVERSE = (
    ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2"
    / "capture_feature_universe.parquet"
)
DEFAULT_UNIVERSE_MANIFEST = DEFAULT_UNIVERSE.with_name("manifest.json")
DEFAULT_PANEL_DIR = ROOT / "data_perp/artifacts/cross_era_global_book_transition_research_panel_20260730_v4"
DEFAULT_PANEL = DEFAULT_PANEL_DIR / "transition_research_panel.parquet"
DEFAULT_PANEL_MANIFEST = DEFAULT_PANEL_DIR / "manifest.json"
DEFAULT_PANEL_MANIFEST_SIDECAR = DEFAULT_PANEL_DIR / "manifest.sha256"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/primary100_current_transition_feature_sidecar_20260730_v1"

# These semantic namespaces are a second fail-closed line of defence.  The
# manifest is authoritative, but a corrupted / incorrectly authored manifest
# must not turn a target or future research field into a decision-time input.
# Do not use broad word substrings here: raw causal state features legitimately
# include names such as ``dir_path_edge`` and ``spread_proxy_abs_return``.
PROHIBITED_FEATURE_PREFIXES = (
    "target__",
    "future__",
    "before_",
    "after_",
    "delta_",
    "outcome_",
    "label_",
    "execution_",
    "realized_",
    "realised_",
)
PROHIBITED_FEATURE_NAMESPACES = (
    "__target__",
    "__future__",
    "__before_",
    "__after_",
    "__outcome_",
    "__label_",
    "__execution_",
    "__realized_",
    "__realised_",
)


class TransitionSidecarError(RuntimeError):
    """Raised when point-in-time transition context cannot be proven."""


def _is_prohibited_feature(field: str) -> bool:
    """Reject outcome namespaces, without rejecting causal raw-name substrings."""

    name = field.lower()
    return name.startswith(PROHIBITED_FEATURE_PREFIXES) or any(
        token in name for token in PROHIBITED_FEATURE_NAMESPACES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
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


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is absent: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TransitionSidecarError(f"{name} must contain an object")
    return loaded


def _verify_panel_contract(
    panel_path: Path,
    manifest_path: Path,
    manifest_sidecar_path: Path,
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """Verify the immutable source panel and extract its sole feature whitelist."""

    if not panel_path.is_file() or not manifest_sidecar_path.is_file():
        raise FileNotFoundError("transition panel, manifest, or manifest sidecar is absent")
    manifest = _load_json(manifest_path, name="transition panel manifest")
    signed = manifest_sidecar_path.read_text(encoding="utf-8").split()
    if not signed or signed[0] != _sha256(manifest_path):
        raise TransitionSidecarError("transition panel manifest detached checksum fails")
    if manifest.get("schema") != PANEL_SCHEMA:
        raise TransitionSidecarError("unexpected transition panel schema")
    if manifest.get("status") != PANEL_STATUS:
        raise TransitionSidecarError("transition panel is not in complete status")
    declared = manifest.get("outputs", {}).get("panel", {})
    if declared.get("sha256") != _sha256(panel_path):
        raise TransitionSidecarError("transition panel parquet hash is not manifest-bound")
    features = manifest.get("feature_columns")
    if not isinstance(features, list) or not features or any(not isinstance(x, str) for x in features):
        raise TransitionSidecarError("transition panel manifest has no valid feature_columns")
    if len(features) != len(set(features)) or manifest.get("feature_count") != len(features):
        raise TransitionSidecarError("transition panel feature whitelist is inconsistent")
    targets = manifest.get("target_columns")
    if not isinstance(targets, list) or set(features).intersection(targets):
        raise TransitionSidecarError("transition panel feature whitelist overlaps target_columns")
    prohibited = [field for field in features if _is_prohibited_feature(field)]
    if prohibited:
        raise TransitionSidecarError(
            "transition panel manifest attempts to whitelist prohibited fields: "
            + ", ".join(sorted(prohibited))
        )
    return manifest, list(features), {
        "path": str(panel_path),
        "sha256": _sha256(panel_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_sidecar_path": str(manifest_sidecar_path),
        "schema": str(manifest["schema"]),
        "status": str(manifest["status"]),
    }


def _verify_universe_contract(
    universe_path: Path, manifest_path: Path
) -> dict[str, str]:
    if not universe_path.is_file():
        raise FileNotFoundError(f"Primary100 universe is absent: {universe_path}")
    manifest = _load_json(manifest_path, name="Primary100 universe manifest")
    if manifest.get("schema") != UNIVERSE_SCHEMA:
        raise TransitionSidecarError("unexpected Primary100 universe schema")
    if manifest.get("status") != UNIVERSE_STATUS:
        raise TransitionSidecarError("Primary100 universe is not outcome-free complete")
    declared = manifest.get("outputs", {}).get("universe", {})
    if declared.get("sha256") != _sha256(universe_path):
        raise TransitionSidecarError("Primary100 universe parquet hash is not manifest-bound")
    return {
        "path": str(universe_path),
        "sha256": _sha256(universe_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "schema": str(manifest["schema"]),
        "status": str(manifest["status"]),
    }


def _canonical_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = [*IDENTITY, DECISION]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise TransitionSidecarError(f"Primary100 universe lacks required columns: {missing}")
    result = frame.loc[:, required].copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result[DECISION] = pd.to_datetime(result[DECISION], utc=True, errors="raise")
    for field in ("__symbol__", "candidate_id"):
        result[field] = result[field].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.strip().str.lower()
    if not result["side_name"].isin(("long", "short")).all():
        raise TransitionSidecarError("Primary100 universe has noncanonical side values")
    if result[list(IDENTITY)].isna().any().any() or result.duplicated(list(IDENTITY)).any():
        raise TransitionSidecarError("Primary100 universe candidate identity is null or duplicated")
    if result[DECISION].isna().any():
        raise TransitionSidecarError("Primary100 universe has null execution decisions")
    return result


def _current_h12_context(panel: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    required = {"source_family", "horizon_hours", ANCHOR, "context_available", *features}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise TransitionSidecarError(f"transition panel lacks required columns: {missing}")
    selected = panel.loc[
        panel["source_family"].eq(CURRENT_SOURCE)
        & pd.to_numeric(panel["horizon_hours"], errors="raise").eq(HORIZON_HOURS),
        [ANCHOR, "context_available", *features],
    ].copy()
    if selected.empty:
        raise TransitionSidecarError("transition panel has no current-source H12 rows")
    selected[ANCHOR] = pd.to_datetime(selected[ANCHOR], utc=True, errors="raise")
    if selected[ANCHOR].isna().any() or selected.duplicated(ANCHOR).any():
        raise TransitionSidecarError("current H12 panel must have exactly one row per cohort anchor")
    if selected["context_available"].isna().any():
        raise TransitionSidecarError("current H12 panel has null context availability")
    selected["context_available"] = selected["context_available"].astype(bool)
    values = selected.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise TransitionSidecarError("current H12 panel feature columns contain infinity")
    # A whitelist field which is entirely absent has no usable decision-time
    # semantics, even if a broad context flag happens to be true.
    empty_features = [field for field in features if values[field].notna().sum() == 0]
    if empty_features:
        raise TransitionSidecarError(
            "current H12 panel has entirely unavailable whitelisted features: "
            + ", ".join(empty_features)
        )
    return selected


def _identity_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame.loc[:, list(IDENTITY)], index=False)
    return hashlib.sha256(values.to_numpy(dtype="uint64").tobytes()).hexdigest()


def build_sidecar(
    universe: pd.DataFrame,
    transition_panel: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_unmatched_rows: int | None = EXPECTED_UNMATCHED_ROWS,
) -> pd.DataFrame:
    """Join current H12 context with exact-anchor semantics and no fill."""

    candidates = _canonical_candidates(universe)
    if expected_rows is not None and len(candidates) != int(expected_rows):
        raise TransitionSidecarError(
            f"Primary100 universe must contain exactly {expected_rows} rows; got {len(candidates)}"
        )
    features = list(feature_columns)
    if not features or len(features) != len(set(features)):
        raise TransitionSidecarError("feature_columns must be a nonempty unique whitelist")
    selected = _current_h12_context(transition_panel, features)
    payload = selected.rename(columns={"context_available": AVAILABILITY})
    left = candidates.copy()
    left["__input_order__"] = np.arange(len(left), dtype=np.int64)
    joined = left.merge(
        payload,
        how="left",
        left_on=DECISION,
        right_on=ANCHOR,
        sort=False,
        validate="many_to_one",
    ).sort_values("__input_order__", kind="stable")
    if len(joined) != len(candidates):
        raise TransitionSidecarError("exact transition join changed Primary100 row count")
    if not joined.loc[:, list(IDENTITY)].reset_index(drop=True).equals(candidates.loc[:, list(IDENTITY)].reset_index(drop=True)):
        raise TransitionSidecarError("exact transition join changed Primary100 identity order")
    if joined[ANCHOR].notna().any() and not joined.loc[joined[ANCHOR].notna(), DECISION].eq(joined.loc[joined[ANCHOR].notna(), ANCHOR]).all():
        raise TransitionSidecarError("transition context was not joined at the exact decision anchor")
    joined[AVAILABILITY] = joined[AVAILABILITY].eq(True)
    output = joined.loc[:, [*IDENTITY, DECISION, AVAILABILITY, *features]].copy()
    matched = joined[ANCHOR].notna()
    if output.loc[~matched, list(features)].notna().any().any():
        raise TransitionSidecarError("unmatched Primary100 candidates received transition context")
    if output.loc[matched, AVAILABILITY].ne(payload.set_index(ANCHOR).reindex(joined.loc[matched, DECISION])[AVAILABILITY].to_numpy()).any():
        raise TransitionSidecarError("transition availability is not the source anchor availability")
    if output.duplicated(list(IDENTITY)).any() or len(output) != len(candidates):
        raise TransitionSidecarError("sidecar no longer has one row per Primary100 candidate")
    if expected_unmatched_rows is not None and int((~matched).sum()) != int(expected_unmatched_rows):
        raise TransitionSidecarError(
            f"expected {expected_unmatched_rows} unmatched Primary100 candidates; got {int((~matched).sum())}"
        )
    return output.reset_index(drop=True)


def _coverage(frame: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    decisions = pd.to_datetime(work[DECISION], utc=True, errors="raise")
    work["decision_hour_utc"] = decisions.dt.floor("h")
    work["decision_month_utc"] = decisions.dt.strftime("%Y-%m")
    work["feature_non_null_count"] = work.loc[:, list(features)].notna().sum(axis=1)
    aggregations = {"rows": (AVAILABILITY, "size"), "available_rows": (AVAILABILITY, "sum"), "feature_non_null_mean": ("feature_non_null_count", "mean")}
    by_hour = work.groupby("decision_hour_utc", dropna=False).agg(**aggregations).reset_index()
    by_month = work.groupby("decision_month_utc", dropna=False).agg(**aggregations).reset_index()
    by_side = work.groupby("side_name", dropna=False).agg(**aggregations).reset_index()
    for coverage in (by_hour, by_month, by_side):
        coverage["unavailable_rows"] = coverage["rows"] - coverage["available_rows"]
        coverage["available_rate"] = coverage["available_rows"] / coverage["rows"]
    feature = pd.DataFrame({
        "feature": list(features),
        "non_null_rows": [int(frame[field].notna().sum()) for field in features],
        "available_rows": [int(frame.loc[frame[AVAILABILITY], field].notna().sum()) for field in features],
    })
    feature["non_null_rate"] = feature["non_null_rows"] / len(frame)
    return by_hour, by_month, by_side, feature


def run(
    *,
    universe_path: Path = DEFAULT_UNIVERSE,
    universe_manifest_path: Path = DEFAULT_UNIVERSE_MANIFEST,
    panel_path: Path = DEFAULT_PANEL,
    panel_manifest_path: Path = DEFAULT_PANEL_MANIFEST,
    panel_manifest_sidecar_path: Path = DEFAULT_PANEL_MANIFEST_SIDECAR,
    destination: Path = DEFAULT_OUTPUT,
    expected_rows: int = EXPECTED_ROWS,
    expected_unmatched_rows: int = EXPECTED_UNMATCHED_ROWS,
) -> dict[str, Any]:
    """Atomically publish a hash-bound immutable Primary100 transition sidecar."""

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite transition sidecar: {destination}")
    _, features, panel_binding = _verify_panel_contract(
        panel_path, panel_manifest_path, panel_manifest_sidecar_path
    )
    universe_binding = _verify_universe_contract(universe_path, universe_manifest_path)
    universe = pd.read_parquet(universe_path, columns=[*IDENTITY, DECISION])
    panel = pd.read_parquet(
        panel_path,
        columns=["source_family", "horizon_hours", ANCHOR, "context_available", *features],
    )
    sidecar = build_sidecar(
        universe,
        panel,
        features,
        expected_rows=expected_rows,
        expected_unmatched_rows=expected_unmatched_rows,
    )
    by_hour, by_month, by_side, feature_coverage = _coverage(sidecar, features)
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True)
        output_path = stage / "transition_context.parquet"
        sidecar.to_parquet(output_path, index=False, compression="zstd", compression_level=5)
        coverage_paths = {
            "coverage_by_hour": stage / "coverage_by_hour.csv",
            "coverage_by_month": stage / "coverage_by_month.csv",
            "coverage_by_side": stage / "coverage_by_side.csv",
            "feature_coverage": stage / "feature_coverage.csv",
        }
        for frame, path in zip((by_hour, by_month, by_side, feature_coverage), coverage_paths.values(), strict=True):
            frame.to_csv(path, index=False)
        output = {
            "path": str(destination / output_path.name),
            "sha256": _sha256(output_path),
            "rows": int(len(sidecar)),
            "columns": int(len(sidecar.columns)),
            "candidate_identity_sha256": _identity_hash(sidecar),
            "available_rows": int(sidecar[AVAILABILITY].sum()),
            "unavailable_rows": int((~sidecar[AVAILABILITY]).sum()),
        }
        report = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_EXACT_CURRENT_H12_TRANSITION_CONTEXT",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity_contract": "exact ordered four-key Primary100 identity; transition context broadcast only where execution_decision_utc == cohort_anchor_utc",
            "sources": {"primary100_universe": universe_binding, "transition_research_panel": panel_binding},
            "transition_selection": {"source_family": CURRENT_SOURCE, "horizon_hours": HORIZON_HOURS, "one_row_per_anchor": True},
            "feature_columns": features,
            "feature_contract": "manifest feature_columns is the sole decision-time whitelist; target_columns and all other panel fields are excluded",
            "missingness": "unmatched anchors are false with raw NaNs; no as-of join or fill is permitted",
            "output": output,
        }
        _write_json(stage / "report.json", report)
        manifest = {
            "schema": SCHEMA,
            "status": report["status"],
            "sources": report["sources"],
            "feature_columns": features,
            "output": output,
            "outputs": {
                "report": {"path": "report.json", "sha256": _sha256(stage / "report.json")},
                **{key: {"path": path.name, "sha256": _sha256(path)} for key, path in coverage_paths.items()},
            },
            "atomic_publication": "all files are staged, hash-bound, then directory-renamed once",
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
    value.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    value.add_argument("--universe-manifest", type=Path, default=DEFAULT_UNIVERSE_MANIFEST)
    value.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    value.add_argument("--panel-manifest", type=Path, default=DEFAULT_PANEL_MANIFEST)
    value.add_argument("--panel-manifest-sidecar", type=Path, default=DEFAULT_PANEL_MANIFEST_SIDECAR)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    value.add_argument("--expected-unmatched-rows", type=int, default=EXPECTED_UNMATCHED_ROWS)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(_jsonable(run(
        universe_path=arguments.universe,
        universe_manifest_path=arguments.universe_manifest,
        panel_path=arguments.panel,
        panel_manifest_path=arguments.panel_manifest,
        panel_manifest_sidecar_path=arguments.panel_manifest_sidecar,
        destination=arguments.output_dir,
        expected_rows=arguments.expected_rows,
        expected_unmatched_rows=arguments.expected_unmatched_rows,
    )), indent=2, sort_keys=True))
