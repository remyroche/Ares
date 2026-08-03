#!/usr/bin/env python3
"""Materialize an exact, diagnostic-only older transition-context sidecar.

The 2022--2023 exact candidate population is retained in its input order and
left-joined *only* at an exact hourly decision key.  The frozen transition
source ends at 2023-01-01; rows outside that source deliberately retain null
feature values.  This is a research packet, not a model, policy, or promotion
input, and must not be presented as a 2023 continuation.
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


SCHEMA = "historical_exact_transition_context_sidecar_v1"
EXPECTED_ROWS = 118_734
EXPECTED_COVERED_ROWS = 5_932
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SOURCE_FAMILY_AVAILABLE = "regime_transition_research_2022augdec_frozen_v1"
SOURCE_FAMILY_UNAVAILABLE = "unavailable_no_2023_transition_continuation"
DEFAULT_CANDIDATES = ROOT / (
    "data_perp/artifacts/failure_2022_2023_pf_exact1m_label_inputs_20260730_v2/candidates.parquet"
)
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_CANDIDATES.with_name("manifest.json")
DEFAULT_TRANSITION = ROOT / (
    "data_perp/artifacts/regime_transition_research_2022augdec_frozen_v1/hourly_transition_dataset.parquet"
)
DEFAULT_TRANSITION_MANIFEST = DEFAULT_TRANSITION.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_sidecar_20260730_v1"
)

# The frozen-v3 manifest is the authoritative contract: 58 retained market
# fields, 148 existing/new transition fields, and the six descriptive packet
# fields named in configs/frozen_transition_research_stack_20260729_v1.json.
EXPECTED_RETAINED_MARKET_FEATURES = 58
EXPECTED_TRANSITION_FEATURES = 148
STATE_CONTEXT_FIELDS = (
    "state_context__current_state",
    "state_context__nearest_distance",
    "state_context__top2_margin",
    "state_context__switch_1h",
    "state_context__switch_count_6h",
    "state_context__state_age_hours",
)
TRANSITION_PREFIXES = ("mkt_regime_change__", "transition_new__")
TRANSITION_IDENTITY_FIELDS = {"source_utc", "execution_decision_utc", "segment_id"}
FORBIDDEN_FEATURE_TOKENS = (
    "target", "availability", "available", "future", "label", "outcome",
    "event", "phase", "destination", "time_to", "economic_",
    "execution_", "return", "gross", "net_ev", "mfe", "mae", "path_",
)


class HistoricalTransitionContextError(RuntimeError):
    """Raised if the exact, outcome-free sidecar cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


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


def _candidate_identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise HistoricalTransitionContextError(f"{source} lacks identity fields: {missing}")
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.strip().str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    if not result["side_name"].isin(("long", "short")).all():
        raise HistoricalTransitionContextError(f"{source} has a noncanonical side")
    if result[list(IDENTITY)].isna().any().any() or result.duplicated(list(IDENTITY)).any():
        raise HistoricalTransitionContextError(f"{source} identity is null or duplicated")
    return result


def candidate_decision_time(frame: pd.DataFrame) -> pd.Series:
    """Return an explicit UTC decision key and prove any duplicate encoding agrees."""

    has_signal = "__ts__" in frame
    has_explicit = "__decision_ts__" in frame
    if not has_signal and not has_explicit:
        raise HistoricalTransitionContextError("candidate input needs __decision_ts__ or __ts__")
    derived = (
        pd.to_datetime(frame["__ts__"], utc=True, errors="raise") + pd.Timedelta(hours=1)
        if has_signal else None
    )
    explicit = (
        pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if has_explicit else None
    )
    if derived is not None and explicit is not None and not explicit.equals(derived):
        raise HistoricalTransitionContextError(
            "__decision_ts__ must equal __ts__ + 1h exactly"
        )
    return (explicit if explicit is not None else derived).rename("__decision_ts__")


def decision_feature_catalog(columns: Sequence[str], *, strict_frozen_contract: bool) -> tuple[str, ...]:
    """Whitelist only known decision-time fields from the frozen schema contract."""

    names = tuple(str(name) for name in columns)
    ordinary = [
        name for name in names
        if name not in TRANSITION_IDENTITY_FIELDS
        and not name.startswith(TRANSITION_PREFIXES)
        and name not in STATE_CONTEXT_FIELDS
        and not name.startswith("target__")
    ]
    transition = [name for name in names if name.startswith(TRANSITION_PREFIXES)]
    state_context = [name for name in names if name in STATE_CONTEXT_FIELDS]
    selected = (*ordinary, *transition, *state_context)
    illegal = [
        name for name in selected
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if illegal:
        raise HistoricalTransitionContextError(
            "transition feature catalog includes prohibited target/outcome fields: "
            + ", ".join(sorted(illegal))
        )
    if len(set(selected)) != len(selected):
        raise HistoricalTransitionContextError("transition feature catalog is duplicated")
    if strict_frozen_contract:
        if len(ordinary) != EXPECTED_RETAINED_MARKET_FEATURES:
            raise HistoricalTransitionContextError("frozen manifest market-feature count disagrees with schema")
        if len(transition) != EXPECTED_TRANSITION_FEATURES:
            raise HistoricalTransitionContextError("frozen manifest transition-feature count disagrees with schema")
        if tuple(state_context) != STATE_CONTEXT_FIELDS:
            raise HistoricalTransitionContextError("frozen descriptive state-context catalog disagrees with schema")
        unexpected = set(names).difference(
            set(TRANSITION_IDENTITY_FIELDS) | set(selected) |
            {name for name in names if name.startswith("target__")}
        )
        if unexpected:
            raise HistoricalTransitionContextError(f"unclassified frozen transition fields: {sorted(unexpected)}")
    return tuple(selected)


def _validate_transition(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"source_utc", "execution_decision_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise HistoricalTransitionContextError(f"transition source lacks required times: {missing}")
    result = frame.copy()
    result["source_utc"] = pd.to_datetime(result["source_utc"], utc=True, errors="raise")
    result["execution_decision_utc"] = pd.to_datetime(
        result["execution_decision_utc"], utc=True, errors="raise"
    )
    if result["source_utc"].isna().any() or result["execution_decision_utc"].isna().any():
        raise HistoricalTransitionContextError("transition timestamps must be non-null")
    if result["source_utc"].duplicated().any() or result["execution_decision_utc"].duplicated().any():
        raise HistoricalTransitionContextError("transition source-time or execution-time key is duplicated")
    if not result["execution_decision_utc"].eq(result["source_utc"] + pd.Timedelta(hours=1)).all():
        raise HistoricalTransitionContextError(
            "transition execution_decision_utc must equal source_utc + 1h exactly"
        )
    return result


def build_sidecar(
    candidates: pd.DataFrame,
    transition: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    expected_rows: int | None = EXPECTED_ROWS,
) -> pd.DataFrame:
    """Create an ordered left sidecar without filling or inferring missing context."""

    candidate = _candidate_identity(candidates, source="exact older candidates")
    if expected_rows is not None and len(candidate) != int(expected_rows):
        raise HistoricalTransitionContextError(
            f"candidate rows must equal {expected_rows}; got {len(candidate)}"
        )
    candidate["__decision_ts__"] = candidate_decision_time(candidates).to_numpy()
    source = _validate_transition(transition)
    fields = tuple(feature_columns) if feature_columns is not None else decision_feature_catalog(
        source.columns, strict_frozen_contract=False
    )
    missing = sorted(set(fields).difference(source.columns))
    if missing:
        raise HistoricalTransitionContextError(f"transition source lacks whitelisted features: {missing}")
    prohibited = [
        name for name in fields
        if any(token in str(name).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if prohibited:
        raise HistoricalTransitionContextError("output feature whitelist is prohibited: " + ", ".join(prohibited))

    # This is intentionally an exact equality merge, not merge_asof, resample,
    # ffill, bfill, interpolation, or a same-day approximation.
    lookup = source.loc[:, ["source_utc", "execution_decision_utc", *fields]].copy()
    lookup["__transition_exact_match__"] = True
    output = candidate.loc[:, [*IDENTITY, "__decision_ts__"]].merge(
        lookup,
        left_on="__decision_ts__",
        right_on="execution_decision_utc",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(output) != len(candidate) or not output[list(IDENTITY)].equals(candidate[list(IDENTITY)].reset_index(drop=True)):
        raise HistoricalTransitionContextError("exact transition join changed candidate identity order")
    output["transition_context_available"] = output["__transition_exact_match__"].eq(True).astype(bool)
    output["source_family"] = np.where(
        output["transition_context_available"], SOURCE_FAMILY_AVAILABLE, SOURCE_FAMILY_UNAVAILABLE
    )
    unavailable = ~output["transition_context_available"]
    if output.loc[unavailable, list(fields)].notna().any().any():
        raise HistoricalTransitionContextError("unavailable context must retain feature NaNs")
    values = output.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    if np.isinf(values.to_numpy(float)).any():
        raise HistoricalTransitionContextError("transition context contains infinite feature values")
    # source_utc/execution_decision_utc prove the time contract but are internal
    # audit columns, not model/context features in the published sidecar.
    output = output.loc[:, [*IDENTITY, "__decision_ts__", "source_family", "transition_context_available", *fields]]
    return output.reset_index(drop=True)


def coverage_by_year_month_side(sidecar: pd.DataFrame) -> list[dict[str, Any]]:
    indexed = sidecar.copy()
    decision = pd.to_datetime(indexed["__decision_ts__"], utc=True, errors="raise")
    indexed["year"] = decision.dt.year.astype(int)
    indexed["month"] = decision.dt.month.astype(int)
    grouped = (
        indexed.groupby(["year", "month", "side_name"], sort=True, dropna=False)["transition_context_available"]
        .agg(candidate_rows="size", covered_rows="sum")
        .reset_index()
    )
    grouped["unavailable_rows"] = grouped["candidate_rows"] - grouped["covered_rows"]
    grouped["coverage"] = grouped["covered_rows"] / grouped["candidate_rows"]
    return [_jsonable(row) for row in grouped.to_dict(orient="records")]


def _candidate_binding(path: Path, manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("outputs", {}).get("candidates", {}).get("sha256")
    actual = _sha256(path)
    if expected != actual:
        raise HistoricalTransitionContextError("candidate manifest does not bind candidates.parquet hash")
    return {"path": str(path), "sha256": actual, "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path)}


def _transition_binding(path: Path, manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("outputs_sha256", {}).get("hourly_transition_dataset.parquet")
    actual = _sha256(path)
    if expected != actual:
        raise HistoricalTransitionContextError("transition manifest does not bind hourly_transition_dataset.parquet hash")
    if manifest.get("research_only") is not True or manifest.get("promotion_evidence") is not False:
        raise HistoricalTransitionContextError("transition source must be frozen diagnostic-only research")
    if manifest.get("full_schema_matches_frozen_v3") is not True:
        raise HistoricalTransitionContextError("transition source lacks frozen-v3 schema proof")
    return {"path": str(path), "sha256": actual, "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path)}


def run(
    *,
    candidates_path: Path = DEFAULT_CANDIDATES,
    candidate_manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    transition_path: Path = DEFAULT_TRANSITION,
    transition_manifest_path: Path = DEFAULT_TRANSITION_MANIFEST,
    destination: Path = DEFAULT_OUTPUT,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_covered_rows: int | None = EXPECTED_COVERED_ROWS,
) -> dict[str, Any]:
    """Write a hash-bound sidecar atomically, refusing any overwrite."""

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite transition context sidecar: {destination}")
    for path in (candidates_path, candidate_manifest_path, transition_path, transition_manifest_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    sources = {
        "candidates": _candidate_binding(candidates_path, candidate_manifest_path),
        "frozen_transition": _transition_binding(transition_path, transition_manifest_path),
    }
    transition_columns = pq.ParquetFile(transition_path).schema_arrow.names
    features = decision_feature_catalog(transition_columns, strict_frozen_contract=True)
    candidates = pd.read_parquet(candidates_path)
    transition = pd.read_parquet(transition_path, columns=["source_utc", "execution_decision_utc", *features])
    sidecar = build_sidecar(candidates, transition, feature_columns=features, expected_rows=expected_rows)
    covered = int(sidecar["transition_context_available"].sum())
    if expected_covered_rows is not None and covered != int(expected_covered_rows):
        raise HistoricalTransitionContextError(
            f"covered rows must equal {expected_covered_rows}; got {covered}"
        )
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True)
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
            "status": "MATERIALIZED_DIAGNOSTIC_ONLY_NO_2023_TRANSITION_CONTINUATION",
            "research_only": True,
            "promotion_eligible": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": _git_revision(),
            "sources": sources,
            "output": output,
            "identity_contract": "ordered exact four-key candidate identity is unchanged",
            "time_contract": "candidate __decision_ts__ (or __ts__ + 1h, asserted equal when both exist) exactly joins execution_decision_utc; execution_decision_utc == source_utc + 1h is asserted; no asof/fill/interpolation",
            "feature_catalog": {
                "authoritative_contract": "frozen v3 manifest counts plus frozen-stack descriptive packet catalog",
                "retained_market_feature_count": EXPECTED_RETAINED_MARKET_FEATURES,
                "transition_feature_count": EXPECTED_TRANSITION_FEATURES,
                "state_context_fields": list(STATE_CONTEXT_FIELDS),
                "output_feature_count": len(features),
                "fields": list(features),
            },
            "coverage": {
                "candidate_rows": int(len(sidecar)),
                "covered_rows": covered,
                "unavailable_rows": int(len(sidecar) - covered),
                "by_year_month_side": coverage_by_year_month_side(sidecar),
            },
            "missingness": "unavailable exact decision hours retain raw NaNs for every transition feature; no asof or other fill is performed",
            "prohibited": "all target__*, availability/future labels/outcomes, paths, returns, and execution outcomes are excluded",
            "caveat": "The frozen transition source ends at 2023-01-01; this sidecar does not create or imply a 2023 continuation.",
        }
        _write_json(stage / "report.json", report)
        manifest = {
            "schema": SCHEMA,
            "status": report["status"],
            "report": {"path": str(destination / "report.json"), "sha256": _sha256(stage / "report.json")},
            "sources": sources,
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
    value.add_argument("--transition", type=Path, default=DEFAULT_TRANSITION)
    value.add_argument("--transition-manifest", type=Path, default=DEFAULT_TRANSITION_MANIFEST)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    value.add_argument("--expected-covered-rows", type=int, default=EXPECTED_COVERED_ROWS)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(_jsonable(run(
        candidates_path=arguments.candidates,
        candidate_manifest_path=arguments.candidate_manifest,
        transition_path=arguments.transition,
        transition_manifest_path=arguments.transition_manifest,
        destination=arguments.output_dir,
        expected_rows=arguments.expected_rows,
        expected_covered_rows=arguments.expected_covered_rows,
    )), indent=2, sort_keys=True))
