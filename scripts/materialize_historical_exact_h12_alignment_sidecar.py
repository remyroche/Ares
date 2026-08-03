#!/usr/bin/env python3
"""Bind the minimal exact-H12 research alignment contract for 2023--2024.

This is intentionally small.  It does not claim factual historical execution,
full-universe coverage, or legacy-score OOF provenance.  It recovers the
policy/cost/timing fields discarded by the raw panel materializer so a bounded
candidate-level target experiment can fail closed on incompatible labels,
features, or replay policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = ROOT / "data_perp/artifacts/long_exact_h12_raw_base_panel_20260730_v2"
PANEL = PANEL_DIR / "raw_base_panel.parquet"
FEATURE_CONTRACT = PANEL_DIR / "raw_feature_contract.json"
STAGES = (
    ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_request_stage_20260730_v1/staged_candidates.parquet",
    ROOT / "data_perp/artifacts/failure_2024_transition_exact1m_request_stage_20260730_v2/staged_candidates.parquet",
)
LABELS = (
    ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet",
    ROOT / "data_perp/artifacts/failure_2024_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet",
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/historical_exact_h12_alignment_sidecar_research_only_20260731_v1"

TARGET_ID = "exact_h12_net_current_frozen_spread_counterfactual_v1"
EXECUTION_POLICY_ID = "historical_current_frozen_spread_counterfactual_h12_v1"
COST_MODEL_ID = "current_frozen_spread_counterfactual_row_cost_v1"
FORBIDDEN_FEATURE_TOKENS = (
    "future_", "label", "target", "outcome", "actual_",
    "execution_net", "execution_gross", "execution_cost", "exit_reason",
    "entry_price", "first_event", "recommended_action", "action_value",
)

STAGE_COLUMNS = (
    "candidate_id", "symbol", "side_name", "__barrier_pct__",
    "archetype_policy_key", "policy_archetype_assignment_source",
    "source_row_number", "signal_timestamp", "source_shard_sha256",
    "source_shard_path", "decision_timestamp", "path_end_exclusive",
)
LABEL_COLUMNS = (
    "candidate_id", "__ts__", "__symbol__", "side_name", "__barrier_pct__",
    "__decision_ts__", "__label_end_ts__", "__label_available_at__",
    "execution_decision_utc", "execution_label_end_utc",
    "execution_label_available_at", "policy_archetype", "execution_geometry_key",
    "execution_geometry_source", "execution_gross_ev_12h", "execution_cost_return",
    "execution_net_ev_12h", "execution_entry_price",
    "execution_expected_spread_bps", "execution_entry_half_spread_bps",
    "execution_exit_half_spread_bps", "execution_exit_reason", "execution_exit_hour",
    "__soft_tb_first_event__",
)
PANEL_COLUMNS = (
    "candidate_id", "__ts__", "__symbol__", "side_name", "frozen_base_score",
    "__decision_ts__", "__label_end_ts__", "__label_available_at__",
    "execution_label_end_utc", "execution_label_available_at",
    "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_many(paths: tuple[Path, ...], columns: tuple[str, ...]) -> pd.DataFrame:
    parts = []
    for path in paths:
        part = pd.read_parquet(path, columns=list(columns))
        if part.candidate_id.duplicated().any():
            raise ValueError(f"duplicate candidate identity in {path}")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    if frame.candidate_id.duplicated().any():
        raise ValueError("overlapping candidate identities across source eras")
    return frame


def _as_utc(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for name in columns:
        frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")


def _prefix(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(columns={name: f"{prefix}_{name}" for name in frame.columns if name != "candidate_id"})


def _geometry_id(frame: pd.DataFrame) -> pd.Series:
    values = frame.loc[:, ["side", "policy_archetype", "execution_geometry_key", "execution_geometry_source", "barrier_pct"]].astype(str)
    return values.agg("|".join, axis=1).map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def _feature_set_id(path: Path) -> tuple[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = list(payload["raw_feature_columns"])
    forbidden = [name for name in features if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    if forbidden:
        raise ValueError(f"feature contract contains forbidden target/outcome fields: {forbidden}")
    if len(features) != len(set(features)) or not features:
        raise ValueError("feature contract is empty or non-unique")
    return f"raw_380_{_sha256(path)[:16]}", features


def validate_alignment(frame: pd.DataFrame, *, feature_set_id: str) -> None:
    required = {
        "candidate_id", "side", "decision_ts", "feature_cutoff_ts", "entry_ts",
        "label_end_ts", "label_available_ts", "target_id", "execution_policy_id",
        "replay_execution_policy_id", "cost_model_id", "feature_set_id",
        "exact_h12_gross_bps", "row_cost_bps", "exact_h12_net_bps",
        "execution_geometry_id", "source_row_number", "source_shard_sha256",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"alignment sidecar misses {missing}")
    if frame.candidate_id.isna().any() or frame.candidate_id.duplicated().any():
        raise ValueError("candidate ID must be unique and present")
    if not frame.side.isin(("long", "short")).all():
        raise ValueError("invalid side")
    if not frame.feature_cutoff_ts.le(frame.decision_ts).all():
        raise ValueError("feature cutoff follows decision time")
    if not frame.entry_ts.eq(frame.decision_ts).all():
        raise ValueError("historical exact path entry must equal execution decision")
    if not frame.label_end_ts.eq(frame.decision_ts + pd.Timedelta(hours=12)).all():
        raise ValueError("label end is not exact H12")
    if not frame.label_available_ts.eq(frame.label_end_ts).all():
        raise ValueError("label availability must equal resolved H12 end")
    gross = frame.exact_h12_gross_bps.to_numpy(float)
    cost = frame.row_cost_bps.to_numpy(float)
    net = frame.exact_h12_net_bps.to_numpy(float)
    if not np.isfinite(np.c_[gross, cost, net]).all() or (cost < 0.0).any():
        raise ValueError("non-finite or negative economics")
    if not np.allclose(gross - cost, net, rtol=0.0, atol=1e-6):
        raise ValueError("exact H12 net does not equal gross minus row cost exactly once")
    if not frame.target_id.eq(TARGET_ID).all() or not frame.cost_model_id.eq(COST_MODEL_ID).all():
        raise ValueError("target/cost contract mismatch")
    if not frame.execution_policy_id.eq(frame.replay_execution_policy_id).all():
        raise ValueError("target execution policy does not match replay policy")
    if not frame.feature_set_id.eq(feature_set_id).all():
        raise ValueError("feature set contract mismatch")
    if frame.execution_geometry_id.isna().any() or frame.source_shard_sha256.isna().any():
        raise ValueError("policy/source lineage is incomplete")


def run(*, panel: Path, feature_contract: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    feature_set_id, features = _feature_set_id(feature_contract)
    panel_frame = pd.read_parquet(panel, columns=list(PANEL_COLUMNS))
    stage = _read_many(STAGES, STAGE_COLUMNS)
    labels = _read_many(LABELS, LABEL_COLUMNS)
    _as_utc(stage, ("signal_timestamp", "decision_timestamp", "path_end_exclusive"))
    _as_utc(labels, ("__ts__", "__decision_ts__", "__label_end_ts__", "__label_available_at__", "execution_decision_utc", "execution_label_end_utc", "execution_label_available_at"))
    _as_utc(panel_frame, ("__ts__", "__decision_ts__", "__label_end_ts__", "__label_available_at__", "execution_label_end_utc", "execution_label_available_at"))
    stage = _prefix(stage, "stage")
    labels = _prefix(labels, "label")
    panel_frame = _prefix(panel_frame, "panel")
    joined = stage.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    if len(joined) != len(stage) or len(joined) != len(labels):
        raise ValueError("stage/label exact identity join lost candidates")
    stage_label_identity = (
        ("stage_signal_timestamp", "label___ts__"), ("stage_symbol", "label___symbol__"),
        ("stage_side_name", "label_side_name"),
        ("stage___barrier_pct__", "label___barrier_pct__"),
    )
    for stage_name, label_name in stage_label_identity:
        if not joined[stage_name].astype(str).eq(joined[label_name].astype(str)).all():
            raise ValueError(f"stage/label identity disagreement in {stage_name}/{label_name}")
    result = panel_frame.merge(joined, on="candidate_id", how="inner", validate="one_to_one")
    if len(result) != len(panel_frame):
        raise ValueError("alignment join lost raw-panel candidates")
    panel_label_identity = (
        ("panel___ts__", "label___ts__"), ("panel___symbol__", "label___symbol__"),
        ("panel_side_name", "label_side_name"),
    )
    for panel_name, label_name in panel_label_identity:
        if not result[panel_name].astype(str).eq(result[label_name].astype(str)).all():
            raise ValueError(f"panel/source identity disagreement in {panel_name}/{label_name}")
    # The raw-panel outcome copies and the signed policy replay must be exactly
    # the same target; do not silently choose one source.
    for name in ("execution_label_end_utc", "execution_label_available_at"):
        if not result[f"panel_{name}"].eq(result[f"label_{name}"]).all():
            raise ValueError(f"panel/source timing disagreement in {name}")
    for name in ("execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"):
        if not np.allclose(result[f"panel_{name}"].to_numpy(float), result[f"label_{name}"].to_numpy(float), rtol=0.0, atol=1e-10):
            raise ValueError(f"panel/source economics disagreement in {name}")
    if not result.stage_signal_timestamp.le(result.stage_decision_timestamp).all():
        raise ValueError("stage signal timestamp follows decision timestamp")
    if not result.label_execution_decision_utc.eq(result.stage_decision_timestamp).all():
        raise ValueError("signed policy decision differs from stage decision")
    if not result.stage_path_end_exclusive.eq(result.label_execution_label_end_utc).all():
        raise ValueError("stage exact path end differs from policy label end")
    if not result["panel___decision_ts__"].eq(result.label_execution_decision_utc).all():
        raise ValueError("panel decision timestamp differs from replay decision")

    output_frame = pd.DataFrame({
        "candidate_id": result.candidate_id,
        "symbol": result.label___symbol__.astype(str),
        "side": result.label_side_name.astype(str).str.lower(),
        "decision_ts": result.label_execution_decision_utc,
        "feature_cutoff_ts": result.stage_signal_timestamp,
        "entry_ts": result.label_execution_decision_utc,
        "label_end_ts": result.label_execution_label_end_utc,
        "label_available_ts": result.label_execution_label_available_at,
        "target_id": TARGET_ID,
        "execution_policy_id": EXECUTION_POLICY_ID,
        "replay_execution_policy_id": EXECUTION_POLICY_ID,
        "cost_model_id": COST_MODEL_ID,
        "feature_set_id": feature_set_id,
        "policy_archetype": result.label_policy_archetype.astype(str),
        "execution_geometry_key": result.label_execution_geometry_key.astype(str),
        "execution_geometry_source": result.label_execution_geometry_source.astype(str),
        "barrier_pct": pd.to_numeric(result.label___barrier_pct__, errors="raise"),
        "exact_h12_gross_bps": result.label_execution_gross_ev_12h.to_numpy(float) * 10_000.0,
        "row_cost_bps": result.label_execution_cost_return.to_numpy(float) * 10_000.0,
        "exact_h12_net_bps": result.label_execution_net_ev_12h.to_numpy(float) * 10_000.0,
        "execution_entry_price": result.label_execution_entry_price.to_numpy(float),
        "estimated_spread_bps": result.label_execution_expected_spread_bps.to_numpy(float),
        "entry_half_spread_bps": result.label_execution_entry_half_spread_bps.to_numpy(float),
        "exit_half_spread_bps": result.label_execution_exit_half_spread_bps.to_numpy(float),
        "exit_reason": result.label_execution_exit_reason.astype(str),
        "exit_hour": pd.to_numeric(result.label_execution_exit_hour, errors="coerce"),
        "event_first": result.label___soft_tb_first_event__.astype(str),
        "source_row_number": pd.to_numeric(result.stage_source_row_number, errors="raise"),
        "source_shard_sha256": result.stage_source_shard_sha256.astype(str),
        "source_shard_path": result.stage_source_shard_path.astype(str),
        # The archived score has no row-level OOF lineage in this historical
        # panel; preserve the fact rather than turning it into an OOF claim.
        "historical_archived_score": result.panel_frozen_base_score.to_numpy(float),
        "historical_archived_score_lineage": "unproven_not_eligible_as_oof_input",
    }).sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    output_frame["execution_geometry_id"] = _geometry_id(output_frame)
    validate_alignment(output_frame, feature_set_id=feature_set_id)
    if not np.isfinite(output_frame.execution_entry_price.to_numpy(float)).all() or (output_frame.execution_entry_price <= 0.0).any():
        raise ValueError("execution entry price is invalid")
    for name in ("estimated_spread_bps", "entry_half_spread_bps", "exit_half_spread_bps"):
        values = output_frame[name].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"invalid spread input: {name}")

    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        sidecar = temporary / "alignment_sidecar.parquet"
        report = temporary / "alignment_report.md"
        output_frame.to_parquet(sidecar, index=False, compression="zstd")
        report.write_text(
            "# Historical exact-H12 alignment report\n\n"
            f"- Rows: {len(output_frame):,}; IDs are unique and exact joins passed.\n"
            f"- Target/policy/cost: `{TARGET_ID}` / `{EXECUTION_POLICY_ID}` / `{COST_MODEL_ID}`.\n"
            f"- Feature set: `{feature_set_id}` ({len(features)} raw decision-time columns).\n"
            "- Economics assertion: exact H12 gross minus row cost equals net exactly once.\n"
            "- This is candidate-conditioned current-spread-counterfactual research only; historical L2, full-universe coverage, and archived-score OOF proof are unavailable.\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": "historical_exact_h12_alignment_sidecar_v1",
            "status": "RESEARCH_ONLY_COUNTERFACTUAL_CANDIDATE_CONDITIONED_NO_PROMOTION",
            "contract": {"target_id": TARGET_ID, "execution_policy_id": EXECUTION_POLICY_ID, "cost_model_id": COST_MODEL_ID, "feature_set_id": feature_set_id},
            "assertions": ["feature_cutoff_ts <= decision_ts", "entry_ts == decision_ts", "label_end_ts == decision_ts + 12h", "label_available_ts == label_end_ts", "gross - row_cost == net exactly once", "target policy id == replay policy id", "raw feature contract excludes outcome/path/score/action fields"],
            "limitations": ["candidate-conditioned old selected/monitor population", "current frozen-spread counterfactual", "historical L2 unavailable", "pre-2025 deployed geometry not bit exact", "archived frozen score is unproven and forbidden as OOF execution input"],
            "sources": {str(path): _sha256(path) for path in (panel, feature_contract, *STAGES, *LABELS)},
            "rows": int(len(output_frame)),
            "outputs": {"alignment_sidecar.parquet": _sha256(sidecar), "alignment_report.md": _sha256(report)},
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--feature-contract", type=Path, default=FEATURE_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(panel=args.panel, feature_contract=args.feature_contract, output=args.output), indent=2))


if __name__ == "__main__":
    main()
