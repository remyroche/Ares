#!/usr/bin/env python3
"""Materialise an immutable non-promotable July 20--23 H12 transition extension.

The source is a hash-bound retrospective scorer, exact 1m 12h policy labels,
and frozen decision-time candidate surface.  The scorer's own 21-day
side-local isotonic map is used only as supplied.  Global mapped-EV
coordinates have a separate causal, day-snapshot reference and deliberately
report their warm-up rather than borrowing prior scores from another model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.materialize_canonical_global_book_conversion_transition_labels import (
        add_causal_global_mapped_ev_coordinates,
        materialize_global_book_labels,
    )
    from scripts.materialize_historical_current_common_transition_geometry import (
        CANONICAL_FEATURES,
        build_historical_hourly_state,
    )
    from scripts.materialize_pooled_historical_current_transition_panel import (
        _base_targets,
        _filter_complete_h12,
    )
except ModuleNotFoundError:
    from materialize_canonical_global_book_conversion_transition_labels import (
        add_causal_global_mapped_ev_coordinates,
        materialize_global_book_labels,
    )
    from materialize_historical_current_common_transition_geometry import (
        CANONICAL_FEATURES,
        build_historical_hourly_state,
    )
    from materialize_pooled_historical_current_transition_panel import _base_targets, _filter_complete_h12


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/july20_23_exact_h12_transition_inputs_20260730_v1"
SCHEMA = "july20_23_exact_h12_transition_extension_v1"
HORIZON_HOURS = 12
BOOK_FRACTION = 0.10
SOURCE_FAMILY = "current_july20_23_retrospective_causal_mapping"
PROVENANCE = "retrospective_causal_21d_non_oof"


class MaterializationError(RuntimeError):
    """Frozen source does not meet the extension contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_manifest(path: Path, *, label: str) -> tuple[dict[str, Any], Path]:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"{label} manifest absent: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MaterializationError(f"{label} manifest is not an object")
    return payload, manifest


def _bound_path(root: Path, relative: str, expected: str | None, *, label: str) -> Path:
    path = root / relative
    if not path.is_file() or not expected or sha256(path) != expected:
        raise MaterializationError(f"{label} is not hash-bound: {path}")
    return path


def load_frozen_source(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the score, labels and raw surface after schema/timing checks."""

    scored_manifest, scored_manifest_path = _read_manifest(root / "scored", label="retrospective scorer")
    labels_manifest, labels_manifest_path = _read_manifest(root / "labels_12h", label="exact H12 labels")
    candidates_manifest, candidates_manifest_path = _read_manifest(root / "candidates_with_frozen_representation", label="frozen candidate surface")
    if scored_manifest.get("schema") != "execution_ev_retrospective_scored_population_v1" or not scored_manifest.get("retrospective"):
        raise MaterializationError("a retrospective frozen execution-EV scorer is required")
    contract = scored_manifest.get("contract", {})
    required_contract = {
        "fixed_lookback_days": 21,
        "mapping": "causal_recent_side_isotonic_ev_21d",
        "ranking": "one pooled global top10 across timestamps and sides after causal mapping",
    }
    if any(contract.get(key) != value for key, value in required_contract.items()):
        raise MaterializationError("frozen scorer causal-map/global-ranking contract changed")
    scored_path = _bound_path(root, "scored/scored_population.parquet", scored_manifest.get("outputs", {}).get("scored_population", {}).get("sha256"), label="scored population")
    support_path = _bound_path(root, "scored/calibration_support.parquet", scored_manifest.get("outputs", {}).get("calibration_support", {}).get("sha256"), label="calibration support")
    labels_path = _bound_path(root, "labels_12h/execution_ev_policy_labels.parquet", labels_manifest.get("output", {}).get("sha256"), label="exact H12 labels")
    candidate_path = _bound_path(root, "candidates_with_frozen_representation/candidate_features_with_representation.parquet", candidates_manifest.get("output", {}).get("sha256"), label="candidate surface")
    scored, labels, candidates, support = (pd.read_parquet(path) for path in (scored_path, labels_path, candidate_path, support_path))
    needed_score = {"candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc", "feature_available_at", "mapping_available_at", "mapped_execution_ev"}
    needed_label = {"candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc", "execution_label_end_utc", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "execution_exit_reason"}
    missing = sorted((needed_score - set(scored)) | (needed_label - set(labels)))
    if missing:
        raise MaterializationError(f"frozen scorer/labels lack fields: {missing}")
    identities = ["candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc"]
    for frame, label in ((scored, "scored"), (labels, "labels")):
        if frame.duplicated("candidate_id").any():
            raise MaterializationError(f"{label} candidate identity is duplicate")
        for timestamp in ("__ts__", "execution_decision_utc"):
            frame[timestamp] = pd.to_datetime(frame[timestamp], utc=True, errors="raise")
    for timestamp in ("feature_available_at", "mapping_available_at"):
        scored[timestamp] = pd.to_datetime(scored[timestamp], utc=True, errors="raise")
        if not scored[timestamp].le(scored["execution_decision_utc"]).all():
            raise MaterializationError(f"{timestamp} is unavailable at decision")
    for timestamp in ("execution_label_end_utc",):
        labels[timestamp] = pd.to_datetime(labels[timestamp], utc=True, errors="raise")
    support["execution_decision_utc"] = pd.to_datetime(support["execution_decision_utc"], utc=True, errors="raise")
    for column in ("history_resolution_min_utc", "history_resolution_max_utc"):
        support[column] = pd.to_datetime(support[column], utc=True, errors="raise")
    if not support["history_resolved_strictly_before_decision"].astype(bool).all() or not support["history_resolution_max_utc"].lt(support["execution_decision_utc"]).all():
        raise MaterializationError("the frozen 21d map support contains a noncausal outcome")
    merged = scored.merge(labels, on=identities, how="inner", validate="one_to_one", suffixes=("", "_label"))
    if len(merged) != len(scored) or len(merged) != len(labels):
        raise MaterializationError("score and exact H12 label identities do not match exactly")
    candidate_required = {"candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc", "feature_available_at"}
    candidate_required.update({feature.split("__")[-1] for feature in []})
    if not candidate_required.issubset(candidates.columns):
        raise MaterializationError("frozen candidate surface lacks decision-time identity")
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    candidates["execution_decision_utc"] = pd.to_datetime(candidates["execution_decision_utc"], utc=True, errors="raise")
    candidates["feature_available_at"] = pd.to_datetime(candidates["feature_available_at"], utc=True, errors="raise")
    if not candidates["feature_available_at"].le(candidates["execution_decision_utc"]).all():
        raise MaterializationError("candidate surface has post-decision fields")
    return merged, candidates, support, {
        "scored_manifest": str(scored_manifest_path), "scored_manifest_sha256": sha256(scored_manifest_path),
        "labels_manifest": str(labels_manifest_path), "labels_manifest_sha256": sha256(labels_manifest_path),
        "candidates_manifest": str(candidates_manifest_path), "candidates_manifest_sha256": sha256(candidates_manifest_path),
        "scored_population": {"path": str(scored_path), "sha256": sha256(scored_path)},
        "calibration_support": {"path": str(support_path), "sha256": sha256(support_path)},
        "labels_12h": {"path": str(labels_path), "sha256": sha256(labels_path)},
        "candidate_surface": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
    }


def build_causal_mapping(scored_labels: pd.DataFrame, support: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce the canonical label input plus outcome-free causal coordinates."""

    work = scored_labels.copy()
    work["candidate_month"] = work["execution_decision_utc"].dt.strftime("%Y-%m")
    work["mapped_direct_net"] = pd.to_numeric(work["mapped_execution_ev"], errors="raise")
    work["mapped_eligible"] = True
    work["execution_exit_class"] = work["execution_exit_reason"].astype(str).replace({"full_sl": "full_stop"})
    allowed = {"trailing", "timeout", "full_stop", "adverse_exit"}
    if not work["execution_exit_class"].isin(allowed).all():
        raise MaterializationError("unknown exact H12 exit class")
    work["opportunity_gross_above_cost_0bps"] = work["execution_gross_ev_12h"].gt(work["execution_cost_return"]).astype(float)
    work["opportunity_gross_above_cost_25bps"] = work["execution_gross_ev_12h"].gt(work["execution_cost_return"] + 0.0025).astype(float)
    support_fields = support.loc[:, ["execution_decision_utc", "side_name", "history_rows"]].rename(columns={"history_rows": "map_side_reference_rows"})
    work = work.merge(support_fields, on=["execution_decision_utc", "side_name"], how="left", validate="many_to_one")
    if work["map_side_reference_rows"].isna().any():
        raise MaterializationError("scored candidates lack mapping support")
    work["map_side_reference_rows"] = work["map_side_reference_rows"].astype(int)
    work["map_reference_rows"] = work.groupby("execution_decision_utc", sort=False)["map_side_reference_rows"].transform("sum").astype(int)
    work["map_cell_reference_rows"] = work["map_side_reference_rows"]
    prepared, audit = add_causal_global_mapped_ev_coordinates(work, minimum_reference_rows=1_000)
    identity = ["candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc"]
    coordinate_columns = [*identity, "feature_available_at", "mapping_available_at", "mapped_execution_ev", "mapped_direct_net", "mapped_eligible", "map_reference_rows", "map_side_reference_rows", "map_cell_reference_rows", "causal_global_mapped_ev_percentile", "causal_global_mapped_ev_band", "causal_global_mapped_ev_reference_rows", "causal_global_mapped_ev_cutoff_p90", "causal_global_mapped_ev_margin_to_p90"]
    return prepared, prepared.loc[:, coordinate_columns].copy(), audit


def build_geometry(candidates: pd.DataFrame) -> pd.DataFrame:
    geometry = build_historical_hourly_state(candidates)
    if list(geometry.columns[1:]) != list(CANONICAL_FEATURES):
        raise MaterializationError("strict common geometry feature order changed")
    geometry["common_transition_context_available"] = geometry.loc[:, list(CANONICAL_FEATURES)].notna().all(axis=1)
    geometry["feature_available_at"] = geometry["signal_context_utc"]
    return geometry


def build_extension_panel(labels: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    selected = _filter_complete_h12(labels)
    selected = selected.merge(geometry, on="signal_context_utc", how="left", validate="one_to_one")
    selected["source_family"] = SOURCE_FAMILY
    selected["economics_tier"] = "exact_1m_spread_aware_current_retrospective"
    selected["policy_cost_contract"] = "deployed_1m_policy_replay_12h"
    selected["path_frequency"] = "1m"
    selected["promotion_use"] = "diagnostic_only_non_oof"
    selected["mapping_provenance_role"] = PROVENANCE
    selected["provenance_oof_share"] = 0.0
    selected["provenance_forward_oos_share"] = 0.0
    selected["source_domain"] = "current_2026_retrospective"
    selected["context_available"] = selected["common_transition_context_available"].fillna(False).astype(bool)
    selected.loc[~selected["context_available"], list(CANONICAL_FEATURES)] = np.nan
    selected = _base_targets(selected)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    block = ((selected["cohort_anchor_utc"] - epoch) // pd.Timedelta(days=7)).astype(int)
    selected["cv_block_start_utc"] = epoch + pd.to_timedelta(block * 7, unit="D")
    selected["cv_group_id"] = "utc7d_" + block.astype(str)
    selected["nonoverlap_anchor_flag"] = (selected["cohort_anchor_utc"].astype("int64") // 3_600_000_000_000 % (2 * HORIZON_HOURS)).eq(0)
    selected["target_available_utc"] = pd.to_datetime(selected["target__adverse_onset_within_3h_available_utc"], utc=True)
    return selected.sort_values("cohort_anchor_utc", kind="stable").reset_index(drop=True)


def run(*, source: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    scored_labels, candidates, support, source_hashes = load_frozen_source(source)
    mapping, coordinates, coordinate_audit = build_causal_mapping(scored_labels, support)
    global_labels = materialize_global_book_labels(mapping, prepared=True)
    global_labels = global_labels.loc[(global_labels["horizon_hours"].eq(HORIZON_HOURS)) & np.isclose(global_labels["book_fraction"], BOOK_FRACTION)].copy()
    if global_labels.empty:
        raise MaterializationError("no H12 top10 global-book labels materialized")
    geometry = build_geometry(candidates)
    panel = build_extension_panel(global_labels, geometry)
    if not panel["context_available"].any():
        raise MaterializationError("no strict 90-field geometry survives in the extension")
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        outputs: dict[str, dict[str, Any]] = {}
        for name, table in (("causal_mapped_candidates", mapping), ("candidate_global_mapped_ev_coordinates", coordinates), ("causal_global_mapped_ev_coordinate_audit", coordinate_audit), ("global_book_transition_labels", global_labels), ("strict_common_geometry", geometry), ("transition_panel", panel)):
            path = stage / f"{name}.parquet"
            table.to_parquet(path, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / path.name), "rows": int(len(table)), "sha256": sha256(path)}
        feature_nulls = {column: int(panel[column].isna().sum()) for column in CANONICAL_FEATURES}
        manifest = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_RETROSPECTIVE_CAUSAL_EXTENSION_DIAGNOSTIC_ONLY_NO_PROMOTION",
            "promotion_eligible": False,
            "source_family": SOURCE_FAMILY,
            "mapping_provenance_role": PROVENANCE,
            "feature_columns": list(CANONICAL_FEATURES),
            "feature_count": len(CANONICAL_FEATURES),
            "horizon_hours": HORIZON_HOURS,
            "book_fraction": BOOK_FRACTION,
            "sources": source_hashes,
            "contracts": {
                "mapping": "frozen retrospective causal 21d side isotonic map; support proves every source outcome resolves strictly before each decision",
                "coordinates": "separate causal 21d global mapped-EV coordinates from this scorer population only, minimum 1000 prior-day reference rows; warm-up retained as unavailable",
                "selection": "one pooled global mapped-EV top10 before/after labels, candidate-id deterministic tie break; no timestamp, side, asset, or regime quotas",
                "labels": "exact deployed 1m spread-aware H12 policy economics, available only after full replay horizon",
                "geometry": "strict 90-field common decision-time state geometry at signal+1h; hourly lags are exact reindex joins with no as-of fill",
                "provenance": "retrospective/non-OOF diagnostic extension only; it must not be treated as OOF/OOS or promoted",
            },
            "coverage": {
                "mapping_rows": int(len(mapping)), "coordinate_available_rows": int(coordinates["causal_global_mapped_ev_percentile"].notna().sum()),
                "global_h12_top10_anchors": int(len(global_labels)), "strict_geometry_rows": int(geometry["common_transition_context_available"].sum()),
                "panel_rows": int(len(panel)), "panel_context_available_rows": int(panel["context_available"].sum()),
                "panel_first_anchor_utc": panel["cohort_anchor_utc"].min(), "panel_last_anchor_utc": panel["cohort_anchor_utc"].max(), "feature_null_counts": feature_nulls,
            },
            "outputs": outputs,
            "outputs_sha256": {f"{name}.parquet": value["sha256"] for name, value in outputs.items()},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(_safe(run(source=args.source, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
