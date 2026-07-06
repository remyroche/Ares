#!/usr/bin/env python3
"""Preflight side-aware label/archetype diagnostic artifacts.

This is a report-only guard for the Stage 0/1 label/archetype roadmap. It checks
whether completed diagnostic artifacts preserve the row-level side contract
(`side`, `side_name`, `timeframe`, `candidate_id`) and whether aggregate
scorecards expose side concentration metrics. It does not train, select,
promote, or regenerate diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import _json_safe  # noqa: E402


DEFAULT_INPUT_DIR = Path("data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "side_contract_preflight"
SIDE_RAW_COLUMNS = ("side", "__side__", "side_name")
CONTRACT_COLUMNS = ("side", "side_name", "timeframe", "candidate_id")
SIDE_METRIC_COLUMNS = ("top_side_share", "max_side_top_share", "max_week_side_top_share", "overall_side_top_share")
SIDE_AWARE_JOIN_MODES = {
    "candidate_id",
    "timestamp_symbol_timeframe_side",
    "timestamp_symbol_side",
    "timestamp_symbol_broadcast_label_side",
    "timestamp_symbol_broadcast_quality_side",
}
SELECTED_ROW_SCOPE_COLUMNS = (
    "candidate",
    "label",
    "risk_target",
    "risk_heads",
    "feature_set",
    "source_bucket",
    "causal_gate",
    "risk_gate",
    "selection",
    "selection_mode",
    "selector",
    "fraction",
    "top_frac",
    "period",
    "seed",
)


@dataclass(frozen=True)
class ArtifactSpec:
    role: str
    relpath: str
    kind: str
    required: bool = True


ARTIFACT_SPECS = (
    ArtifactSpec(
        "weekly_candidate_selected_rows",
        "utility_risk_gate_candidate_weekly/candidate_selected_rows.csv",
        "row_contract",
    ),
    ArtifactSpec(
        "path_risk_selected_rows",
        "utility_path_risk_dual_head/source_utility_path_risk_dual_head_selected_rows.csv",
        "row_contract",
    ),
    ArtifactSpec(
        "joint_path_timeout_selected_rows",
        "utility_path_timeout_joint_risk/source_utility_path_timeout_risk_selected_rows.csv",
        "row_contract",
    ),
    ArtifactSpec(
        "archetype_materialized_rows",
        "source_archetypes_v2/candidate_source_archetypes_v2.parquet",
        "row_contract",
    ),
    ArtifactSpec(
        "archetype_scorecard",
        "source_archetypes_v2/source_archetypes_v2_scorecard.csv",
        "side_metrics",
    ),
    ArtifactSpec(
        "timeout_stage1_weekaware_selected_rows",
        "timeout_holding_risk_stage1_weekaware_v1/timeout_holding_risk_selected_rows.csv",
        "row_contract",
        required=False,
    ),
    ArtifactSpec(
        "timeout_stage1_weekaware_aggregate",
        "timeout_holding_risk_stage1_weekaware_v1/timeout_holding_risk_label_aggregate.csv",
        "side_metrics",
        required=False,
    ),
    ArtifactSpec(
        "timeout_stage1_selected_rows",
        "timeout_holding_risk_stage1_metrics_v1/timeout_holding_risk_selected_rows.csv",
        "row_contract",
        required=False,
    ),
    ArtifactSpec(
        "timeout_stage1_aggregate",
        "timeout_holding_risk_stage1_metrics_v1/timeout_holding_risk_label_aggregate.csv",
        "side_metrics",
        required=False,
    ),
    ArtifactSpec(
        "archetype_manifest",
        "source_archetypes_v2/manifest.json",
        "manifest",
        required=False,
    ),
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table file: {path}")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _side_names(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        text = frame["side_name"].fillna("").astype(str).str.strip().str.lower()
        text = text.where(text.isin({"long", "short"}), "")
        if text.ne("").any():
            return text[text.ne("")]
    if "side" in frame.columns:
        numeric = _safe_numeric(frame["side"])
    elif "__side__" in frame.columns:
        numeric = _safe_numeric(frame["__side__"])
    else:
        return pd.Series(dtype=object)
    numeric = numeric.dropna()
    if numeric.empty:
        return pd.Series(dtype=object)
    return pd.Series(np.where(numeric < 0.0, "short", "long"), index=numeric.index, dtype=object)


def _side_counts(frame: pd.DataFrame) -> dict[str, int]:
    names = _side_names(frame)
    counts = names.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _top_side_share(frame: pd.DataFrame) -> float:
    names = _side_names(frame)
    if names.empty:
        return float("nan")
    shares = names.value_counts(normalize=True)
    return float(shares.iloc[0]) if len(shares) else float("nan")


def _row_identity_scope(spec: ArtifactSpec, frame: pd.DataFrame) -> list[str]:
    if spec.role == "archetype_materialized_rows":
        return []
    return [col for col in SELECTED_ROW_SCOPE_COLUMNS if col in frame.columns]


def _duplicate_count(frame: pd.DataFrame, columns: list[str]) -> int | None:
    if not columns or not set(columns) <= set(frame.columns):
        return None
    return int(frame.duplicated(columns).sum())


def _row_contract_report(spec: ArtifactSpec, path: Path) -> dict[str, Any]:
    frame = _read_table(path)
    cols = set(frame.columns)
    side_cols = [col for col in SIDE_RAW_COLUMNS if col in cols]
    missing_contract = [col for col in CONTRACT_COLUMNS if col not in cols]
    side_counts = _side_counts(frame)
    identity_scope = _row_identity_scope(spec, frame)
    candidate_key = [*identity_scope, "candidate_id"] if "candidate_id" in frame.columns else []
    ts_side_key = [*identity_scope, "__ts__", "__symbol__", "side"] if {"__ts__", "__symbol__", "side"} <= cols else []
    if "timeframe" in frame.columns and ts_side_key:
        ts_side_key = [*identity_scope, "__ts__", "__symbol__", "timeframe", "side"]
    duplicate_candidate_ids = _duplicate_count(frame, candidate_key)
    duplicate_ts_symbol_side = _duplicate_count(frame, ts_side_key)
    duplicate_ts_symbol = _duplicate_count(frame, [*identity_scope, "__ts__", "__symbol__"])
    has_side = bool(side_cols) and bool(side_counts)
    has_candidate_id = "candidate_id" in cols
    has_timeframe = "timeframe" in cols
    candidate_unique_ok = duplicate_candidate_ids in (None, 0)
    ts_symbol_side_unique_ok = duplicate_ts_symbol_side in (None, 0)
    passes = bool(has_side and has_candidate_id and has_timeframe and candidate_unique_ok and ts_symbol_side_unique_ok)
    failures: list[str] = []
    if not has_side:
        failures.append("missing_side")
    if not has_candidate_id:
        failures.append("missing_candidate_id")
    if not has_timeframe:
        failures.append("missing_timeframe")
    if duplicate_candidate_ids not in (None, 0):
        failures.append("duplicate_candidate_id")
    if duplicate_ts_symbol_side not in (None, 0):
        failures.append("duplicate_timestamp_symbol_side")
    return {
        "role": spec.role,
        "path": str(path),
        "kind": spec.kind,
        "required": bool(spec.required),
        "exists": True,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "side_columns": side_cols,
        "missing_contract_columns": missing_contract,
        "side_counts": side_counts,
        "top_side_share": _top_side_share(frame),
        "identity_scope_columns": identity_scope,
        "candidate_id_present": has_candidate_id,
        "timeframe_present": has_timeframe,
        "duplicate_candidate_ids": duplicate_candidate_ids,
        "duplicate_timestamp_symbol": duplicate_ts_symbol,
        "duplicate_timestamp_symbol_side": duplicate_ts_symbol_side,
        "passes": passes,
        "status": "ok" if passes else "side_contract_refresh_required",
        "failures": failures,
    }


def _side_metric_report(spec: ArtifactSpec, path: Path) -> dict[str, Any]:
    frame = _read_table(path)
    cols = set(frame.columns)
    present = [col for col in SIDE_METRIC_COLUMNS if col in cols]
    finite_metric_rows = 0
    if present and len(frame):
        finite_metric_rows = int(pd.concat([_safe_numeric(frame[col]) for col in present], axis=1).notna().any(axis=1).sum())
    passes = bool(present and (len(frame) == 0 or finite_metric_rows > 0))
    return {
        "role": spec.role,
        "path": str(path),
        "kind": spec.kind,
        "required": bool(spec.required),
        "exists": True,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "side_metric_columns": present,
        "finite_side_metric_rows": finite_metric_rows,
        "passes": passes,
        "status": "ok" if passes else "side_metric_refresh_required",
        "failures": [] if passes else ["missing_side_metric"],
    }


def _manifest_report(spec: ArtifactSpec, path: Path) -> dict[str, Any]:
    payload = _load_manifest(path)
    join_report = payload.get("join_report", {}) if isinstance(payload.get("join_report"), dict) else {}
    join_mode = str(join_report.get("join_mode", ""))
    side_counts_full = payload.get("side_counts_full", {})
    side_counts_joined = payload.get("side_counts_joined", payload.get("side_counts", {}))
    side_counts_ok = bool(side_counts_full or side_counts_joined)
    join_side_ok = join_mode in SIDE_AWARE_JOIN_MODES
    passes = bool(side_counts_ok and join_side_ok)
    failures: list[str] = []
    if not side_counts_ok:
        failures.append("missing_side_counts")
    if not join_side_ok:
        failures.append("join_mode_not_side_aware")
    return {
        "role": spec.role,
        "path": str(path),
        "kind": spec.kind,
        "required": bool(spec.required),
        "exists": True,
        "rows": None,
        "columns": None,
        "join_mode": join_mode,
        "side_counts_full": side_counts_full,
        "side_counts_joined": side_counts_joined,
        "passes": passes,
        "status": "ok" if passes else "manifest_refresh_required",
        "failures": failures,
    }


def _missing_report(spec: ArtifactSpec, path: Path) -> dict[str, Any]:
    return {
        "role": spec.role,
        "path": str(path),
        "kind": spec.kind,
        "required": bool(spec.required),
        "exists": False,
        "rows": None,
        "columns": None,
        "passes": not bool(spec.required),
        "status": "missing_required_artifact" if spec.required else "missing_optional_artifact",
        "failures": ["missing"] if spec.required else [],
    }


def _artifact_report(spec: ArtifactSpec, input_dir: Path) -> dict[str, Any]:
    path = input_dir / spec.relpath
    if not path.exists():
        return _missing_report(spec, path)
    try:
        if spec.kind == "row_contract":
            return _row_contract_report(spec, path)
        if spec.kind == "side_metrics":
            return _side_metric_report(spec, path)
        if spec.kind == "manifest":
            return _manifest_report(spec, path)
    except Exception as exc:
        return {
            "role": spec.role,
            "path": str(path),
            "kind": spec.kind,
            "required": bool(spec.required),
            "exists": True,
            "rows": None,
            "columns": None,
            "passes": False,
            "status": "read_error",
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    raise ValueError(f"Unsupported artifact kind: {spec.kind}")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if isinstance(value, str):
        return value
    number = _finite(value)
    if math.isfinite(number):
        return f"{number:.4f}".rstrip("0").rstrip(".")
    return ""


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(label for _, label in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    return "\n".join(lines)


def _write_report(output_dir: Path, result: dict[str, Any]) -> Path:
    path = output_dir / "label_archetype_side_contract_preflight.md"
    rows = list(result["artifacts"])
    lines = [
        "# Label / Archetype Side Contract Preflight",
        "",
        "Report-only guard for Stage 0/1 diagnostics. It checks whether refreshed artifacts preserve side-aware candidate identity.",
        "",
        f"Input root: `{result['input_dir']}`",
        f"Decision: `{result['decision']}`",
        f"Ready for side-aware Stage 0 evidence: `{result['ready_for_side_aware_stage0']}`",
        "",
        "## Blocking Artifacts",
        "",
        _markdown_table(
            list(result["blocking_artifacts"]),
            [
                ("role", "role"),
                ("kind", "kind"),
                ("status", "status"),
                ("failures", "failures"),
                ("path", "path"),
            ],
        ),
        "",
        "## Artifact Checks",
        "",
        _markdown_table(
            rows,
            [
                ("role", "role"),
                ("kind", "kind"),
                ("required", "required"),
                ("exists", "exists"),
                ("passes", "passes"),
                ("rows", "rows"),
                ("side_counts", "side_counts"),
                ("top_side_share", "top_side"),
                ("side_metric_columns", "side_metric_cols"),
                ("join_mode", "join_mode"),
                ("status", "status"),
            ],
        ),
        "",
        "## Outputs",
        "",
        f"- JSON: `{result['outputs']['json']}`",
        f"- CSV: `{result['outputs']['csv']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_preflight(*, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [_artifact_report(spec, input_dir) for spec in ARTIFACT_SPECS]
    blocking = [
        row
        for row in reports
        if bool(row.get("required", False)) and not bool(row.get("passes", False))
    ]
    ready = not blocking
    paths = {
        "json": output_dir / "label_archetype_side_contract_preflight.json",
        "csv": output_dir / "label_archetype_side_contract_preflight.csv",
    }
    result = {
        "scope": "label_archetype_side_contract_preflight",
        "input_dir": str(input_dir),
        "ready_for_side_aware_stage0": ready,
        "decision": "ready" if ready else "refresh_required",
        "artifacts": reports,
        "blocking_artifacts": blocking,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["json"].write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    pd.DataFrame(reports).to_csv(paths["csv"], index=False)
    markdown = _write_report(output_dir, result)
    result["outputs"]["markdown"] = str(markdown)
    paths["json"].write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_preflight(input_dir=args.input_dir, output_dir=args.output_dir)
    print(json.dumps(_json_safe({"decision": result["decision"], "outputs": result["outputs"]}), indent=2))
    return 0 if bool(result["ready_for_side_aware_stage0"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
