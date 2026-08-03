#!/usr/bin/env python3
"""Build a causal global recent-EV mapping source for the 2022--23 backcast.

The frozen historical base score is retained as-is.  At each UTC-day snapshot,
only exact policy outcomes resolved before that snapshot during the preceding
21 days fit the score-to-net-EV isotonic map.  The resulting score is global:
there is no timestamp, side, asset, or regime quota.  This creates a
diagnostic-only mapping source for the shared global-book before/after label
materializer; it is neither OOF nor promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_execution_ev_recent_mapping_ablation import causal_mappings  # noqa: E402


SCHEMA = "causal_score_economics_conversion_mapping_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
RAW_SCORE = "frozen_backcast_base_score"
MAPPED_SCORE = "causal_global_recent_isotonic_ev"
WINDOW_DAYS = 21
MINIMUM_REFERENCE_ROWS = 1_000
SIDE_SUPPORT_TARGET = 0.0
EXPECTED_ROWS = 118_734
EXIT_CLASS_MAP = {"full_sl": "full_stop"}
EXIT_CLASSES = {"trailing", "timeout", "full_stop", "adverse_exit"}

DEFAULT_SCORE_ROOT = ROOT / "data_perp/reports/failure_2022_2023_pf_baseonly_backcast_20260730_v1"
DEFAULT_CANDIDATES = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_label_inputs_20260730_v2/candidates.parquet"
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_CANDIDATES.with_name("manifest.json")
DEFAULT_POLICY_LABELS = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_policy_labels_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_causal_global_book_mapping_20260730_v1"


class HistoricalGlobalMappingError(RuntimeError):
    """Raised when causal score/economics mapping cannot be proven."""


def _sha256(path: Path) -> str:
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
        temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise_identity(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise HistoricalGlobalMappingError(f"{name} lacks identity fields: {missing}")
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    if not result["side_name"].isin(("long", "short")).all():
        raise HistoricalGlobalMappingError(f"{name} has noncanonical side values")
    if result[list(IDENTITY)].isna().any().any() or result.duplicated(list(IDENTITY)).any():
        raise HistoricalGlobalMappingError(f"{name} identity is null or duplicated")
    return result


def _candidate_binding(path: Path, manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("outputs", {}).get("candidates", {}).get("sha256") != _sha256(path):
        raise HistoricalGlobalMappingError("candidate manifest does not bind candidates.parquet")
    return {"path": str(path), "sha256": _sha256(path), "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path)}


def _load_scores(score_root: Path, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = score_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "three_year_frozen_failure_backcast_v1":
        raise HistoricalGlobalMappingError("unexpected frozen base-backcast schema")
    if manifest.get("evidence_scope") != "frozen_backcast_diagnostic_not_oos":
        raise HistoricalGlobalMappingError("base backcast provenance is not explicitly diagnostic")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise HistoricalGlobalMappingError("base backcast has no chunks")
    paths = [score_root / "candidate_shards" / f"candidates_{str(row.get('key'))}.parquet" for row in chunks]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("one or more declared frozen score shards are absent")
    score_rows: list[pd.DataFrame] = []
    for path in paths:
        source = pd.read_parquet(path, columns=["__ts__", "__symbol__", "side_name", "base_score", "selected_for_monitor"])
        source = source.loc[source["selected_for_monitor"].fillna(False).astype(bool), ["__ts__", "__symbol__", "side_name", "base_score"]].copy()
        score_rows.append(source)
    raw = pd.concat(score_rows, ignore_index=True)
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True, errors="raise")
    raw["__symbol__"] = raw["__symbol__"].astype(str)
    raw["side_name"] = raw["side_name"].astype(str).str.lower()
    if raw.duplicated(["__ts__", "__symbol__", "side_name"]).any():
        raise HistoricalGlobalMappingError("selected frozen score rows duplicate signal identity")
    joined = candidates.loc[:, list(IDENTITY)].merge(
        raw, on=["__ts__", "__symbol__", "side_name"], how="left", validate="one_to_one"
    )
    if joined["base_score"].isna().any():
        raise HistoricalGlobalMappingError("frozen base score shards do not cover every exact candidate")
    if len(raw) != len(candidates) or len(joined) != len(candidates):
        raise HistoricalGlobalMappingError("frozen score/candidate row count changed")
    joined[RAW_SCORE] = pd.to_numeric(joined.pop("base_score"), errors="coerce")
    if not np.isfinite(joined[RAW_SCORE].to_numpy(float)).all():
        raise HistoricalGlobalMappingError("frozen base score contains non-finite values")
    return joined, {
        "path": str(score_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "shards": [{"path": str(path), "sha256": _sha256(path)} for path in paths],
        "evidence_scope": manifest["evidence_scope"],
        "selected_for_monitor_rows": int(manifest.get("selected_for_monitor_rows", -1)),
        "score_column": "base_score",
    }


def _load_policy_labels(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    manifest_path = root / "manifest.json"
    path = root / "execution_policy_labels.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "execution_ev_deployed_policy_1m_labels_v1":
        raise HistoricalGlobalMappingError("unexpected exact policy-label schema")
    if manifest.get("historical_lineage", {}).get("evidence_scope") != "frozen_backcast_diagnostic_not_oof":
        raise HistoricalGlobalMappingError("policy labels are not explicitly frozen-backcast diagnostic")
    if manifest.get("output", {}).get("sha256") != _sha256(path):
        raise HistoricalGlobalMappingError("policy-label manifest does not bind output")
    columns = [*IDENTITY, "execution_decision_utc", "execution_label_end_utc", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "execution_exit_reason"]
    labels = _normalise_identity(pd.read_parquet(path, columns=columns), name="exact policy labels")
    for field in ("execution_decision_utc", "execution_label_end_utc"):
        labels[field] = pd.to_datetime(labels[field], utc=True, errors="raise")
    for field in ("execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"):
        labels[field] = pd.to_numeric(labels[field], errors="coerce")
    if not np.isfinite(labels[["execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"]].to_numpy(float)).all():
        raise HistoricalGlobalMappingError("exact policy labels have non-finite economics")
    if not np.allclose(labels["execution_gross_ev_12h"] - labels["execution_cost_return"], labels["execution_net_ev_12h"], rtol=0.0, atol=1e-7):
        raise HistoricalGlobalMappingError("historical exact policy gross-cost-net accounting fails")
    labels["execution_exit_class"] = labels["execution_exit_reason"].astype(str).str.lower().replace(EXIT_CLASS_MAP)
    if not labels["execution_exit_class"].isin(EXIT_CLASSES).all():
        raise HistoricalGlobalMappingError("exact policy exit classes are incompatible with global-book labels")
    return labels.drop(columns="execution_exit_reason"), {"path": str(path), "sha256": _sha256(path), "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path)}


def build_mapping_source(scores: pd.DataFrame, labels: pd.DataFrame, *, minimum_reference_rows: int = MINIMUM_REFERENCE_ROWS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join frozen scores to exact labels and fit only prior-resolved global maps."""

    scores = _normalise_identity(scores, name="frozen base scores")
    labels = _normalise_identity(labels, name="exact policy labels")
    merged = scores.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    if merged["execution_net_ev_12h"].isna().any():
        raise HistoricalGlobalMappingError("exact labels do not cover all frozen score rows")
    if not merged["execution_decision_utc"].eq(merged["__ts__"] + pd.Timedelta(hours=1)).all():
        raise HistoricalGlobalMappingError("signal-to-decision contract is not exactly one hour")
    if not merged["execution_label_end_utc"].eq(merged["execution_decision_utc"] + pd.Timedelta(hours=12)).all():
        raise HistoricalGlobalMappingError("exact labels are not 12-hour decision horizons")
    merged = merged.sort_values(["execution_decision_utc", "candidate_id"], kind="stable").reset_index(drop=True)
    mapped, audit_rows = causal_mappings(
        merged, score_col=RAW_SCORE, window_days=WINDOW_DAYS,
        min_reference_rows=int(minimum_reference_rows), side_support_target=SIDE_SUPPORT_TARGET,
    )
    mapped["mapped_eligible"] = np.isfinite(pd.to_numeric(mapped["causal_recent_isotonic_ev"], errors="coerce"))
    mapped["mapped_direct_net"] = mapped["causal_recent_isotonic_ev"]
    mapped["map_reference_rows"] = 0
    mapped["map_side_reference_rows"] = 0
    mapped["map_cell_reference_rows"] = 0
    audit = pd.DataFrame.from_records(audit_rows)
    if audit.empty:
        raise HistoricalGlobalMappingError("causal mapping had no eligible snapshots")
    audit = audit.rename(columns={"snapshot": "snapshot_utc"})
    audit["snapshot_utc"] = pd.to_datetime(audit["snapshot_utc"], utc=True, errors="raise")
    for field in ("reference_rows", "long_reference_rows", "short_reference_rows"):
        audit[field] = pd.to_numeric(audit[field], errors="raise").astype(np.int64)
    resolved = pd.to_datetime(mapped["execution_label_end_utc"], utc=True, errors="raise")
    for row in audit.itertuples(index=False):
        mask = mapped["execution_decision_utc"].dt.floor("D").eq(row.snapshot_utc)
        mapped.loc[mask, "map_reference_rows"] = int(row.reference_rows)
        mapped.loc[mask & mapped["side_name"].eq("long"), "map_side_reference_rows"] = int(row.long_reference_rows)
        mapped.loc[mask & mapped["side_name"].eq("short"), "map_side_reference_rows"] = int(row.short_reference_rows)
        mapped.loc[mask, "map_cell_reference_rows"] = int(row.reference_rows)
        ref = resolved.lt(row.snapshot_utc) & resolved.ge(row.snapshot_utc - pd.Timedelta(days=WINDOW_DAYS))
        audit.loc[audit["snapshot_utc"].eq(row.snapshot_utc), "reference_label_end_max_utc"] = resolved.loc[ref].max() if ref.any() else pd.NaT
    available = mapped["mapped_eligible"]
    if not mapped.loc[available, "map_reference_rows"].ge(int(minimum_reference_rows)).all():
        raise HistoricalGlobalMappingError("mapped candidates lack required prior-resolved support")
    mapped["candidate_month"] = mapped["execution_decision_utc"].dt.strftime("%Y-%m")
    mapped["opportunity_gross_above_cost_0bps"] = (mapped["execution_gross_ev_12h"] > mapped["execution_cost_return"]).astype(float)
    mapped["opportunity_gross_above_cost_25bps"] = (mapped["execution_gross_ev_12h"] > mapped["execution_cost_return"] + 0.0025).astype(float)
    output = mapped.loc[:, [
        "candidate_id", "__symbol__", "side_name", "__ts__", "execution_decision_utc", "execution_label_end_utc", "candidate_month", "mapped_eligible", "mapped_direct_net", "map_reference_rows", "map_side_reference_rows", "map_cell_reference_rows", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "execution_exit_class", "opportunity_gross_above_cost_0bps", "opportunity_gross_above_cost_25bps", RAW_SCORE,
    ]].sort_values(["execution_decision_utc", "candidate_id"], kind="stable").reset_index(drop=True)
    audit = audit.loc[:, ["snapshot_utc", "reference_rows", "long_reference_rows", "short_reference_rows", "current_rows", "reference_label_end_max_utc"]].sort_values("snapshot_utc", kind="stable").reset_index(drop=True)
    return output, audit


def run(*, score_root: Path = DEFAULT_SCORE_ROOT, candidates_path: Path = DEFAULT_CANDIDATES, candidate_manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST, policy_labels_root: Path = DEFAULT_POLICY_LABELS, destination: Path = DEFAULT_OUTPUT, expected_rows: int = EXPECTED_ROWS, minimum_reference_rows: int = MINIMUM_REFERENCE_ROWS) -> dict[str, Any]:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable mapping source: {destination}")
    candidates = _normalise_identity(pd.read_parquet(candidates_path), name="exact historical candidates")
    if len(candidates) != int(expected_rows):
        raise HistoricalGlobalMappingError(f"expected {expected_rows} candidate rows; got {len(candidates)}")
    scores, score_binding = _load_scores(Path(score_root), candidates)
    labels, label_binding = _load_policy_labels(Path(policy_labels_root))
    mapped, audit = build_mapping_source(scores, labels, minimum_reference_rows=minimum_reference_rows)
    if len(mapped) != len(candidates):
        raise HistoricalGlobalMappingError("mapping output row count changed")
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True)
        mapped_path, audit_path = stage / "causal_mapped_candidates.parquet", stage / "causal_snapshot_audit.parquet"
        mapped.to_parquet(mapped_path, index=False, compression="zstd", compression_level=5)
        audit.to_parquet(audit_path, index=False, compression="zstd", compression_level=5)
        sources = {"candidates": _candidate_binding(Path(candidates_path), Path(candidate_manifest_path)), "frozen_base_scores": score_binding, "exact_policy_labels": label_binding}
        manifest = {
            "schema": SCHEMA,
            "status": "HISTORICAL_FROZEN_BACKCAST_CAUSAL_GLOBAL_MAPPING_DIAGNOSTIC_ONLY",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "lineage": "historical_frozen_backcast_exact1m_research_only",
            "evidence_scope": "frozen_backcast_diagnostic_not_oof",
            "promotion_eligible": False,
            "causal_contract": {"window_days": WINDOW_DAYS, "minimum_reference_rows": int(minimum_reference_rows), "reference_rule": "execution_label_end_utc < snapshot", "snapshot_frequency": "UTC day", "raw_score": RAW_SCORE, "mapping": {"global": MAPPED_SCORE, "mapped_direct_net_alias": MAPPED_SCORE}, "no_same_day_or_future_outcomes": True},
            "selection_contract": {"primary": "one pooled global top-k", "not_per_timestamp": True, "not_per_side": True, "not_per_asset": True, "tie_break": "candidate_id ascending", "ranking_stage": "after causal global recent-EV mapping"},
            "economics_contract": {"target": "exact 1m frozen-policy spread-counterfactual execution_net_ev_12h", "gross_minus_cost_equals_net": True, "exit_class_map": EXIT_CLASS_MAP, "cost_counted_once": True},
            "source_contract": {"frozen_base_score": "base_score from monitored frozen-backcast candidate shards", "score_is_oof": False, "score_usage": "frozen raw score only; no score refit", "diagnostic_only": True},
            "sources": sources,
            "population": {"rows": int(len(mapped)), "mapped_eligible_rows": int(mapped["mapped_eligible"].sum()), "warmup_unmapped_rows": int((~mapped["mapped_eligible"]).sum()), "mapped_start_utc": mapped.loc[mapped["mapped_eligible"], "execution_decision_utc"].min(), "mapped_end_utc": mapped.loc[mapped["mapped_eligible"], "execution_decision_utc"].max()},
            "outputs": {"mapped": {"path": "causal_mapped_candidates.parquet", "sha256": _sha256(mapped_path)}, "audit": {"path": "causal_snapshot_audit.parquet", "sha256": _sha256(audit_path)}},
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, destination)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--score-root", type=Path, default=DEFAULT_SCORE_ROOT)
    value.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    value.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    value.add_argument("--policy-labels-root", type=Path, default=DEFAULT_POLICY_LABELS)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    value.add_argument("--minimum-reference-rows", type=int, default=MINIMUM_REFERENCE_ROWS)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(_safe(run(score_root=args.score_root, candidates_path=args.candidates, candidate_manifest_path=args.candidate_manifest, policy_labels_root=args.policy_labels_root, destination=args.output_dir, expected_rows=args.expected_rows, minimum_reference_rows=args.minimum_reference_rows)), indent=2, sort_keys=True))
