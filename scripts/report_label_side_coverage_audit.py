#!/usr/bin/env python3
"""Audit side coverage for diagnostic label/archetype evidence.

This report is intentionally separate from the side-contract preflight. The
contract preflight checks whether artifacts preserve side identity. This audit
checks whether both long and short sides are actually represented in the
current evidence bundle. It is report-only and does not generate synthetic
opposite-side rows.
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

from scripts.report_label_archetype_side_contract_preflight import DEFAULT_INPUT_DIR  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import DEFAULT_LABELS_DIR, _json_safe  # noqa: E402


DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "side_coverage_audit"
SIDE_COLUMNS = ("side_name", "side", "__side__", "trade_side")


@dataclass(frozen=True)
class TableSpec:
    role: str
    path: Path
    required: bool = True


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table path: {path}")


def _load_label_ledger(path: Path) -> pd.DataFrame:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet label files found under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()


def _side_names(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    for col in ("side_name", "trade_side"):
        if col in frame.columns:
            text = frame[col].fillna("").astype(str).str.strip().str.lower()
            text = text.where(text.isin({"long", "short"}), "")
            if text.ne("").any():
                return text[text.ne("")]
    raw_col = "side" if "side" in frame.columns else "__side__" if "__side__" in frame.columns else None
    if raw_col is None:
        return pd.Series(dtype=object)
    numeric = _safe_numeric(frame[raw_col]).dropna()
    if numeric.empty:
        return pd.Series(dtype=object)
    return pd.Series(np.where(numeric < 0.0, "short", "long"), index=numeric.index, dtype=object)


def _counts(names: pd.Series) -> dict[str, int]:
    counts = names.value_counts(dropna=False).to_dict()
    out = {"long": 0, "short": 0}
    for key, value in counts.items():
        if str(key) in out:
            out[str(key)] = int(value)
    return out


def _top_share(counts: dict[str, int]) -> float:
    total = int(sum(counts.values()))
    if total <= 0:
        return float("nan")
    return float(max(counts.values()) / total)


def _table_report(spec: TableSpec) -> dict[str, Any]:
    if not spec.path.exists():
        return {
            "role": spec.role,
            "path": str(spec.path),
            "required": bool(spec.required),
            "exists": False,
            "rows": None,
            "side_columns": [],
            "side_counts": {"long": 0, "short": 0},
            "top_side_share": float("nan"),
            "bidirectional": False,
            "status": "missing_required" if spec.required else "missing_optional",
            "failures": ["missing"] if spec.required else [],
        }
    try:
        frame = _read_table(spec.path)
        side_cols = [col for col in SIDE_COLUMNS if col in frame.columns]
        side_counts = _counts(_side_names(frame))
        bidirectional = side_counts["long"] > 0 and side_counts["short"] > 0
        failures: list[str] = []
        if not side_cols:
            failures.append("missing_side_column")
        if side_counts["long"] <= 0:
            failures.append("missing_long_rows")
        if side_counts["short"] <= 0:
            failures.append("missing_short_rows")
        return {
            "role": spec.role,
            "path": str(spec.path),
            "required": bool(spec.required),
            "exists": True,
            "rows": int(len(frame)),
            "side_columns": side_cols,
            "side_counts": side_counts,
            "top_side_share": _top_share(side_counts),
            "bidirectional": bidirectional,
            "status": "ok" if bidirectional else "long_only_or_single_side",
            "failures": [] if bidirectional else failures,
        }
    except Exception as exc:
        return {
            "role": spec.role,
            "path": str(spec.path),
            "required": bool(spec.required),
            "exists": True,
            "rows": None,
            "side_columns": [],
            "side_counts": {"long": 0, "short": 0},
            "top_side_share": float("nan"),
            "bidirectional": False,
            "status": "read_error",
            "failures": [f"{type(exc).__name__}: {exc}"],
        }


def _label_report(path: Path) -> dict[str, Any]:
    spec = TableSpec("label_ledger", path, True)
    if not path.exists():
        return _table_report(spec)
    try:
        frame = _load_label_ledger(path)
    except Exception as exc:
        return {
            "role": spec.role,
            "path": str(path),
            "required": True,
            "exists": True,
            "rows": None,
            "side_columns": [],
            "side_counts": {"long": 0, "short": 0},
            "top_side_share": float("nan"),
            "bidirectional": False,
            "status": "read_error",
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    side_cols = [col for col in SIDE_COLUMNS if col in frame.columns]
    side_counts = _counts(_side_names(frame))
    bidirectional = side_counts["long"] > 0 and side_counts["short"] > 0
    failures = []
    if not side_cols:
        failures.append("missing_side_column")
    if side_counts["long"] <= 0:
        failures.append("missing_long_rows")
    if side_counts["short"] <= 0:
        failures.append("missing_short_rows")
    return {
        "role": spec.role,
        "path": str(path),
        "required": True,
        "exists": True,
        "rows": int(len(frame)),
        "side_columns": side_cols,
        "side_counts": side_counts,
        "top_side_share": _top_share(side_counts),
        "bidirectional": bidirectional,
        "status": "ok" if bidirectional else "long_only_or_single_side",
        "failures": [] if bidirectional else failures,
    }


def _registry_reports(artifact_root: Path, *, max_reports: int = 500) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    paths = sorted(
        artifact_root.glob("*/strategy_registry/selected_single_head_strategy_registry.csv"),
        key=lambda path: (path.stat().st_mtime if path.exists() else 0.0, str(path)),
        reverse=True,
    )
    for path in paths:
        reports.append(_table_report(TableSpec(f"strategy_registry:{path.parent.parent.name}", path, False)))
        if len(reports) >= int(max_reports):
            break
    return reports


def _artifact_specs(input_dir: Path) -> list[TableSpec]:
    return [
        TableSpec("weekly_candidate_selected_rows", input_dir / "utility_risk_gate_candidate_weekly/candidate_selected_rows.csv"),
        TableSpec(
            "path_risk_selected_rows",
            input_dir / "utility_path_risk_dual_head/source_utility_path_risk_dual_head_selected_rows.csv",
        ),
        TableSpec(
            "joint_path_timeout_selected_rows",
            input_dir / "utility_path_timeout_joint_risk/source_utility_path_timeout_risk_selected_rows.csv",
        ),
        TableSpec("archetype_materialized_rows", input_dir / "source_archetypes_v2/candidate_source_archetypes_v2.parquet"),
        TableSpec(
            "timeout_stage1_weekaware_selected_rows",
            input_dir / "timeout_holding_risk_stage1_weekaware_v1/timeout_holding_risk_selected_rows.csv",
            required=False,
        ),
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isfinite(number):
        return f"{number:.4f}".rstrip("0").rstrip(".")
    return ""


def _markdown_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], *, limit: int | None = None) -> str:
    if not rows:
        return "_No rows._"
    view = rows[:limit] if limit is not None else rows
    lines = ["| " + " | ".join(label for _, label in cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in view:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in cols) + " |")
    return "\n".join(lines)


def _write_markdown(output_dir: Path, result: dict[str, Any]) -> Path:
    path = output_dir / "label_side_coverage_audit.md"
    required = [row for row in result["artifacts"] if bool(row.get("required"))]
    registries = result["strategy_registries"]
    blocking = result["blocking_artifacts"]
    registry_summary = result["registry_summary"]
    lines = [
        "# Label Side Coverage Audit",
        "",
        "Report-only audit for actual long/short coverage in the diagnostic label/archetype evidence bundle.",
        "",
        f"Input root: `{result['input_dir']}`",
        f"Labels path: `{result['labels_path']}`",
        f"Decision: `{result['decision']}`",
        f"Bidirectional evidence ready: `{result['bidirectional_evidence_ready']}`",
        "",
        "## Blocking Coverage Gaps",
        "",
        _markdown_table(
            blocking,
            [
                ("role", "role"),
                ("status", "status"),
                ("failures", "failures"),
                ("side_counts", "side_counts"),
                ("path", "path"),
            ],
        ),
        "",
        "## Required Evidence",
        "",
        _markdown_table(
            required,
            [
                ("role", "role"),
                ("rows", "rows"),
                ("side_counts", "side_counts"),
                ("top_side_share", "top_side"),
                ("bidirectional", "bidirectional"),
                ("status", "status"),
            ],
        ),
        "",
        "## Strategy Registry Summary",
        "",
        _markdown_table(
            [registry_summary],
            [
                ("registries", "registries"),
                ("registries_with_long", "with_long"),
                ("registries_with_short", "with_short"),
                ("total_long_rows", "long_rows"),
                ("total_short_rows", "short_rows"),
                ("bidirectional_registries", "bidirectional"),
            ],
        ),
        "",
        "## Recent Registry Samples",
        "",
        _markdown_table(
            registries[:25],
            [
                ("role", "role"),
                ("rows", "rows"),
                ("side_counts", "side_counts"),
                ("bidirectional", "bidirectional"),
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


def build_audit(
    *,
    input_dir: Path,
    labels_path: Path,
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [_label_report(labels_path)]
    artifacts.extend(_table_report(spec) for spec in _artifact_specs(input_dir))
    registries = _registry_reports(artifact_root)
    required_blocking = [
        row for row in artifacts if bool(row.get("required")) and not bool(row.get("bidirectional"))
    ]
    registry_summary = {
        "registries": int(len(registries)),
        "registries_with_long": int(sum(1 for row in registries if row["side_counts"]["long"] > 0)),
        "registries_with_short": int(sum(1 for row in registries if row["side_counts"]["short"] > 0)),
        "total_long_rows": int(sum(row["side_counts"]["long"] for row in registries)),
        "total_short_rows": int(sum(row["side_counts"]["short"] for row in registries)),
        "bidirectional_registries": int(sum(1 for row in registries if bool(row.get("bidirectional")))),
    }
    registry_ready = registry_summary["registries_with_short"] > 0 and registry_summary["registries_with_long"] > 0
    ready = not required_blocking and registry_ready
    decision = "ready_bidirectional" if ready else "long_only_or_missing_short_evidence"
    paths = {
        "json": output_dir / "label_side_coverage_audit.json",
        "csv": output_dir / "label_side_coverage_audit.csv",
    }
    result = {
        "scope": "label_side_coverage_audit",
        "input_dir": str(input_dir),
        "labels_path": str(labels_path),
        "artifact_root": str(artifact_root),
        "bidirectional_evidence_ready": bool(ready),
        "decision": decision,
        "artifacts": artifacts,
        "strategy_registries": registries,
        "registry_summary": registry_summary,
        "blocking_artifacts": required_blocking,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    pd.DataFrame(artifacts + registries).to_csv(paths["csv"], index=False)
    markdown = _write_markdown(output_dir, result)
    result["outputs"]["markdown"] = str(markdown)
    paths["json"].write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--artifact-root", type=Path, default=Path("data_perp/artifacts"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_audit(
        input_dir=args.input_dir,
        labels_path=args.labels_path,
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe({"decision": result["decision"], "outputs": result["outputs"]}), indent=2))
    return 0 if bool(result["bidirectional_evidence_ready"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
