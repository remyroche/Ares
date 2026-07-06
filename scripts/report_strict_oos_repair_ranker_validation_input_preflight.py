#!/usr/bin/env python3
"""Preflight strict-OOS repair-ranker frozen validation inputs.

This is a diagnostic-only readiness report. It checks whether the frozen
profile validation periods are present in the strict-OOS inputs needed by
``run_strict_oos_repair_ranker_frozen_validation.py``. It does not train,
select, validate, or promote profiles.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import _json_safe  # noqa: E402
from scripts.run_strict_oos_repair_ranker_ablation import (  # noqa: E402
    DEFAULT_EVENT_ROWS,
    DEFAULT_PREDICTIONS,
    DEFAULT_QUALITY_LABELS,
)
from scripts.run_strict_oos_repair_ranker_frozen_validation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
)


DEFAULT_RUNNER_MANIFEST = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_frozen_profile_run/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_validation_input_preflight"
)
TABLE_SUFFIXES = {".parquet", ".csv"}
TIME_COL_CANDIDATES = ("__ts__", "timestamp", "period", "ts", "datetime", "date")
PATH_DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _parse_csv(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _as_periods(values: Any, col: str) -> pd.Series:
    series = pd.Series(values)
    if col == "period":
        text = series.dropna().astype(str).str.strip()
        month_like = text.str.extract(r"(\d{4}-\d{2})", expand=False)
        if month_like.notna().any():
            return month_like.dropna()
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dropna().dt.strftime("%Y-%m")


def _table_files(
    path: Path,
    *,
    max_files: int,
    exclude_path_parts: set[str] | None = None,
) -> tuple[list[Path], bool]:
    if path.is_file():
        return ([path] if path.suffix.lower() in TABLE_SUFFIXES else []), False
    if not path.exists() or not path.is_dir():
        return [], False
    excluded = set(exclude_path_parts or set())
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in TABLE_SUFFIXES
        and not (excluded & set(file.relative_to(path).parts[:-1]))
    )
    truncated = len(files) > int(max_files)
    return files[: int(max_files)], truncated


def _read_table_columns(path: Path) -> tuple[list[str], pd.DataFrame]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            schema_cols = list(parquet_file.schema.names)
            pandas_meta = (parquet_file.metadata.metadata or {}).get(b"pandas") if parquet_file.metadata else None
            index_cols: list[str] = []
            if pandas_meta:
                try:
                    pandas_info = json.loads(pandas_meta.decode("utf-8"))
                    index_cols = [
                        str(col)
                        for col in pandas_info.get("index_columns", [])
                        if isinstance(col, str) and col in TIME_COL_CANDIDATES
                    ]
                except (TypeError, ValueError, UnicodeDecodeError):
                    index_cols = []
            read_cols = list(dict.fromkeys(col for col in [*TIME_COL_CANDIDATES, *index_cols] if col in schema_cols))
            frame = pd.read_parquet(path, columns=read_cols if read_cols else [])
            if not read_cols and len(frame) == 0 and parquet_file.metadata is not None:
                frame = pd.DataFrame(index=range(int(parquet_file.metadata.num_rows)))
        except Exception:
            frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        return [], pd.DataFrame()
    return list(frame.columns), frame


def _period_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = PATH_DATE_RE.search(part)
        if match:
            year, month, _day = match.groups()
            return f"{year}-{month}"
    return None


def _one_dimensional(values: Any) -> Any:
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0] if values.shape[1] else pd.Series(dtype="object")
    return values


def summarize_periods(
    path: Path,
    *,
    expected_periods: list[str],
    max_files: int = 500,
    exclude_path_parts: set[str] | None = None,
) -> dict[str, Any]:
    files, truncated = _table_files(
        path,
        max_files=max_files,
        exclude_path_parts=exclude_path_parts,
    )
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "file_count": 0,
            "rows": 0,
            "timestamp_col": None,
            "periods": [],
            "period_counts": {},
            "expected_periods": expected_periods,
            "missing_expected_periods": expected_periods,
            "status": "missing_path",
            "passes": False,
            "truncated_file_scan": False,
        }
    if not files:
        return {
            "path": str(path),
            "exists": True,
            "file_count": 0,
            "rows": 0,
            "timestamp_col": None,
            "periods": [],
            "period_counts": {},
            "expected_periods": expected_periods,
            "missing_expected_periods": expected_periods,
            "status": "no_table_files",
            "passes": False,
            "truncated_file_scan": truncated,
        }

    period_counts: dict[str, int] = {}
    rows = 0
    timestamp_cols: list[str] = []
    read_errors: list[str] = []
    for file in files:
        try:
            cols, frame = _read_table_columns(file)
        except Exception as exc:  # pragma: no cover - defensive report path
            read_errors.append(f"{file}: {type(exc).__name__}: {exc}")
            continue
        rows += int(len(frame))
        time_col = next((col for col in TIME_COL_CANDIDATES if col in cols), None)
        if time_col:
            timestamp_cols.append(time_col)
            periods = _as_periods(_one_dimensional(frame[time_col]), time_col)
        elif isinstance(frame.index, pd.DatetimeIndex):
            timestamp_cols.append("__index__")
            periods = _as_periods(frame.index.to_series(index=frame.index), "__index__")
        else:
            path_period = _period_from_path(file)
            if not path_period:
                continue
            timestamp_cols.append("__path_date__")
            periods = pd.Series([path_period] * len(frame))
        counts = periods.value_counts(dropna=True).to_dict()
        for period, count in counts.items():
            period_counts[str(period)] = int(period_counts.get(str(period), 0) + int(count))

    periods_sorted = sorted(period_counts)
    missing = [period for period in expected_periods if period not in period_counts]
    status = "ok" if not missing and not read_errors else "missing_expected_periods"
    if read_errors:
        status = "read_errors"
    return {
        "path": str(path),
        "exists": True,
        "file_count": int(len(files)),
        "rows": int(rows),
        "timestamp_col": sorted(set(timestamp_cols))[0] if timestamp_cols else None,
        "periods": periods_sorted,
        "period_counts": {key: int(period_counts[key]) for key in periods_sorted},
        "expected_periods": expected_periods,
        "missing_expected_periods": missing,
        "status": status,
        "passes": not missing and not read_errors,
        "truncated_file_scan": truncated,
        "read_errors": read_errors,
    }


def _existence_summary(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "file_count": 1 if path.exists() else 0,
        "rows": None,
        "timestamp_col": None,
        "periods": [],
        "expected_periods": [],
        "missing_expected_periods": [],
        "status": "ok" if path.exists() else "missing_path",
        "passes": bool(path.exists()),
        "truncated_file_scan": False,
    }


def _period_summary(
    role: str,
    path: Path,
    expected_periods: list[str],
    *,
    max_files: int,
    exclude_path_parts: set[str] | None = None,
) -> dict[str, Any]:
    summary = summarize_periods(
        path,
        expected_periods=expected_periods,
        max_files=max_files,
        exclude_path_parts=exclude_path_parts,
    )
    return {"role": role, **summary}


def build_preflight(
    *,
    runner_manifest_path: Path,
    output_dir: Path,
    validation_periods: list[str] | None = None,
    history_periods: list[str] | None = None,
    quality_labels_path: Path | None = None,
    labels_path: Path | None = None,
    predictions_path: Path | None = None,
    event_rows_path: Path | None = None,
    feature_dir: Path | None = None,
    feature_list_csv: Path | None = None,
    max_files: int = 500,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_manifest = _load_json(runner_manifest_path)
    validation_manifest = runner_manifest.get("validation_manifest", {})
    validation_default = (
        validation_manifest.get("validation_periods")
        if isinstance(validation_manifest, dict) and validation_manifest.get("validation_periods")
        else runner_manifest.get("validation_periods", [])
    )
    history_default = runner_manifest.get("history_periods", [])
    validation = validation_periods if validation_periods is not None else [str(v) for v in validation_default]
    history = history_periods if history_periods is not None else [str(v) for v in history_default]

    quality_labels = Path(quality_labels_path or runner_manifest.get("quality_labels_path") or DEFAULT_QUALITY_LABELS)
    labels = Path(labels_path or runner_manifest.get("labels_path") or "")
    predictions = Path(predictions_path or runner_manifest.get("predictions_path") or DEFAULT_PREDICTIONS)
    events = Path(event_rows_path or runner_manifest.get("event_rows_path") or DEFAULT_EVENT_ROWS)
    features = Path(feature_dir or runner_manifest.get("feature_dir") or DEFAULT_FEATURE_DIR)
    feature_list = Path(feature_list_csv or runner_manifest.get("feature_list_csv") or DEFAULT_FEATURE_LIST_CSV)

    role_rows = [
        _period_summary("quality_labels", quality_labels, validation, max_files=max_files),
        _period_summary("labels", labels, validation, max_files=max_files),
        _period_summary("predictions", predictions, validation, max_files=max_files),
        _period_summary(
            "feature_store",
            features,
            validation,
            max_files=max_files,
            exclude_path_parts={"_live_latest_matrix"},
        ),
        _period_summary("event_rows_history", events, history, max_files=max_files),
        _existence_summary(feature_list, "feature_list_csv"),
    ]
    required_roles = {"quality_labels", "labels", "predictions", "feature_store", "event_rows_history", "feature_list_csv"}
    blocking = [
        {
            "role": row["role"],
            "status": row["status"],
            "missing_expected_periods": row.get("missing_expected_periods", []),
            "path": row["path"],
        }
        for row in role_rows
        if row["role"] in required_roles and not row["passes"]
    ]
    ready = not blocking and bool(validation)
    if not validation:
        blocking.append(
            {
                "role": "validation_periods",
                "status": "missing_validation_periods_config",
                "missing_expected_periods": [],
                "path": str(runner_manifest_path),
            }
        )
        ready = False

    outputs = {
        "json": output_dir / "strict_oos_repair_ranker_validation_input_preflight.json",
        "markdown": output_dir / "strict_oos_repair_ranker_validation_input_preflight.md",
        "inputs": output_dir / "strict_oos_repair_ranker_validation_input_preflight_inputs.csv",
    }
    inputs_frame = pd.DataFrame(role_rows)
    csv_frame = inputs_frame.copy()
    for col in ["periods", "expected_periods", "missing_expected_periods"]:
        if col in csv_frame.columns:
            csv_frame[col] = csv_frame[col].map(lambda value: ",".join(value) if isinstance(value, list) else value)
    if "period_counts" in csv_frame.columns:
        csv_frame["period_counts"] = csv_frame["period_counts"].map(json.dumps)
    csv_frame.to_csv(outputs["inputs"], index=False)

    result = {
        "scope": "strict_oos_repair_ranker_validation_input_preflight",
        "runner_manifest_path": str(runner_manifest_path),
        "validation_periods": validation,
        "history_periods": history,
        "ready_to_run_frozen_validation": ready,
        "decision": "ready" if ready else "blocked",
        "blocking_inputs": blocking,
        "inputs": role_rows,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["json"].write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    _write_markdown(outputs["markdown"], result)
    return result


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    rows = []
    for item in result["inputs"]:
        rows.append(
            {
                "role": item["role"],
                "status": item["status"],
                "passes": item["passes"],
                "rows": item.get("rows"),
                "periods": ",".join(item.get("periods", [])),
                "missing": ",".join(item.get("missing_expected_periods", [])),
                "path": item["path"],
            }
        )
    table = pd.DataFrame(rows).to_markdown(index=False) if rows else "No inputs."
    lines = [
        "# Strict OOS Repair Ranker Validation Input Preflight",
        "",
        f"- Runner manifest: `{result['runner_manifest_path']}`",
        f"- Decision: `{result['decision']}`",
        f"- Ready to run frozen validation: `{result['ready_to_run_frozen_validation']}`",
        f"- Validation periods: `{', '.join(result['validation_periods'])}`",
        f"- History periods: `{', '.join(result['history_periods'])}`",
        "",
        "## Inputs",
        "",
        table,
        "",
        "## Blocking Inputs",
        "",
        json.dumps(_json_safe(result["blocking_inputs"]), indent=2),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-manifest", type=Path, default=DEFAULT_RUNNER_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-periods", type=str, default=None)
    parser.add_argument("--history-periods", type=str, default=None)
    parser.add_argument("--quality-labels-path", type=Path, default=None)
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--event-rows-path", type=Path, default=None)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--feature-list-csv", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_preflight(
        runner_manifest_path=args.runner_manifest,
        output_dir=args.output_dir,
        validation_periods=_parse_csv(args.validation_periods) or None,
        history_periods=_parse_csv(args.history_periods) or None,
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        predictions_path=args.predictions_path,
        event_rows_path=args.event_rows_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_files=args.max_files,
    )
    print(json.dumps(_json_safe(result), indent=2))
    if args.fail_on_blocked and not result["ready_to_run_frozen_validation"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
