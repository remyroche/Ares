#!/usr/bin/env python3
"""Describe future path geometry by *separate* regime and transition labels.

This is a research diagnostic.  A persistent regime state is an observable
decision-time context; an adaptive transition phase is an ex-post catalogue
label with its own availability time.  The runner intentionally never joins
them into a single purported state, and does not create a model or policy
input.  It only asks whether the physical/economic path labels differ across
each taxonomy separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LEDGER = ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1"
DEFAULT_PHASES = ROOT / "data_perp/artifacts/transition_pattern_catalogue_20260730_v5/adaptive_phase_labels.parquet"
DEFAULT_LABELS = (
    ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet",
    ROOT / "data_perp/artifacts/failure_2024_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet",
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/regime_transition_path_geometry_diagnostic_20260730_v1"
SCHEMA = "regime_transition_path_geometry_diagnostic_v1"


# Every metric is explicitly unconditional or opportunity-conditional.  This
# prevents a lower time-to-MFE mean, for example, from being confused with a
# higher probability of ever reaching meaningful MFE.
METRICS: tuple[dict[str, str], ...] = (
    {"name": "peak_mfe_atr", "column": "__peak_mfe_atr_12h__", "condition": "all_valid"},
    {"name": "mae_before_meaningful_mfe_atr", "column": "__mae_before_meaningful_mfe_atr_12h__", "condition": "opportunity_only"},
    {"name": "opportunity_probability", "column": "__opportunity_occurred_12h__", "condition": "all_valid"},
    {"name": "time_to_meaningful_mfe_hours", "column": "__time_to_first_meaningful_mfe_hours_12h__", "condition": "opportunity_only"},
    {"name": "future_slope_atr_per_hour", "column": "__future_slope_atr_per_hour_12h__", "condition": "all_valid"},
    {"name": "timeout_probability", "column": "__timeout_outcome_12h__", "condition": "all_valid"},
    {"name": "exit_conversion_failure_probability", "column": "__exit_conversion_failure_proxy_12h__", "condition": "all_valid"},
    {"name": "exit_conversion_loss_return", "column": "__exit_conversion_loss_return_12h__", "condition": "all_valid"},
    {"name": "net_ev_12h", "column": "execution_net_ev_12h", "condition": "all_valid"},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: Path) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks required compatible label columns: {missing}")


def _load_context(ledger_dir: Path, phase_path: Path) -> pd.DataFrame:
    hourly_path = ledger_dir / "hourly_state_calendar.parquet"
    if not hourly_path.exists() or not phase_path.exists():
        raise FileNotFoundError(hourly_path if not hourly_path.exists() else phase_path)
    hourly = pd.read_parquet(hourly_path, columns=["source_utc", "target__pooled_state"])
    phases = pd.read_parquet(phase_path, columns=["source_utc", "target__pattern_phase", "target__pattern_phase_available_utc"])
    hourly["source_utc"] = pd.to_datetime(hourly["source_utc"], utc=True)
    phases["source_utc"] = pd.to_datetime(phases["source_utc"], utc=True)
    if hourly["source_utc"].duplicated().any() or phases["source_utc"].duplicated().any():
        raise ValueError("context source timestamps must be unique before label joins")
    context = hourly.merge(phases, on="source_utc", how="inner", validate="one_to_one")
    if len(context) != len(hourly):
        raise ValueError("phase output does not exactly cover the hourly state calendar")
    return context.rename(columns={"target__pooled_state": "regime_state", "target__pattern_phase": "transition_phase"})


def _load_compatible_labels(label_paths: Iterable[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required = {"__ts__", "__decision_ts__", "side_name", "candidate_id", "__opportunity_occurred_12h__"}
    records: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for path in label_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        available = set(pd.read_parquet(path).columns)
        _require_columns(pd.DataFrame(columns=list(available)), required, source=path)
        included = [item for item in METRICS if item["column"] in available]
        columns = sorted(required.union({item["column"] for item in included}))
        frame = pd.read_parquet(path, columns=columns).copy()
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="coerce")
        if frame["__ts__"].isna().any() or frame["__decision_ts__"].isna().any():
            raise ValueError(f"{path} has invalid label timestamps")
        # Source time is the decision context: exact labels are one hour later
        # at execution, so joining it to `execution_decision_utc` would look
        # aligned while actually shifting the observed context by an hour.
        frame["label_source"] = str(path)
        records.append(frame)
        sources.append({
            "path": str(path),
            "rows_input": int(len(frame)),
            "available_metrics": [item["name"] for item in included],
            "sha256": _sha256(path),
            "time_start": frame["__ts__"].min(),
            "time_end": frame["__ts__"].max(),
        })
    if not records:
        raise ValueError("at least one label source is required")
    return pd.concat(records, ignore_index=True, sort=False), sources


def join_path_geometry_context(
    *, ledger_dir: Path,
    phase_path: Path,
    label_paths: Iterable[Path],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Perform the exact source-time join and return only covered candidates."""

    context = _load_context(ledger_dir, phase_path)
    labels, sources = _load_compatible_labels(label_paths)
    joined = labels.merge(context, left_on="__ts__", right_on="source_utc", how="inner", validate="many_to_one")
    if joined.empty:
        raise ValueError("none of the compatible labels overlap the hourly state coverage")
    joined["side_name"] = joined["side_name"].astype(str).str.lower()
    valid_sides = {"long", "short"}
    if not set(joined["side_name"]).issubset(valid_sides):
        raise ValueError("only long/short labels are supported")
    return joined, sources


def _cluster_mean_ci(values: pd.Series, clusters: pd.Series) -> tuple[int, int, float, float, float]:
    clean = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "cluster": clusters}).dropna()
    if clean.empty:
        return 0, 0, np.nan, np.nan, np.nan
    hourly = clean.groupby("cluster", sort=False)["value"].mean()
    mean = float(clean["value"].mean())
    n_hours = int(len(hourly))
    if n_hours < 2:
        return int(len(clean)), n_hours, mean, np.nan, np.nan
    standard_error = float(hourly.std(ddof=1) / np.sqrt(n_hours))
    half_width = 1.96 * standard_error
    return int(len(clean)), n_hours, mean, mean - half_width, mean + half_width


def summarize_path_geometry(joined: pd.DataFrame) -> pd.DataFrame:
    """Summarize each metric by side and by each taxonomy independently."""

    rows: list[dict[str, Any]] = []
    for taxonomy, column, availability in (
        ("regime_state_at_decision", "regime_state", "decision_time_observable"),
        ("transition_phase_ex_post", "transition_phase", "target_available_after_phase_window"),
    ):
        for (side, bucket), group in joined.groupby(["side_name", column], dropna=False, sort=True):
            for metric in METRICS:
                name, source_column, condition = metric["name"], metric["column"], metric["condition"]
                if source_column not in group:
                    rows.append({
                        "taxonomy": taxonomy, "context_value": str(bucket), "side_name": side,
                        "metric": name, "condition": condition, "metric_available": False,
                        "n_candidates": 0, "n_decision_hours": 0, "mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                    })
                    continue
                selected = group
                if condition == "opportunity_only":
                    selected = selected.loc[pd.to_numeric(selected["__opportunity_occurred_12h__"], errors="coerce").eq(1)]
                count, hours, mean, low, high = _cluster_mean_ci(selected[source_column], selected["__ts__"])
                rows.append({
                    "taxonomy": taxonomy,
                    "taxonomy_availability": availability,
                    "context_value": str(bucket),
                    "side_name": side,
                    "metric": name,
                    "condition": condition,
                    "metric_available": True,
                    "n_candidates": count,
                    "n_decision_hours": hours,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                })
    return pd.DataFrame.from_records(rows).sort_values(["taxonomy", "side_name", "metric", "context_value"], ignore_index=True)


def materialize_regime_transition_path_geometry_diagnostic(
    *,
    ledger_dir: Path = DEFAULT_LEDGER,
    phase_path: Path = DEFAULT_PHASES,
    label_paths: Iterable[Path] = DEFAULT_LABELS,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Write a provenance-bound, descriptive path-geometry diagnostic."""

    ledger_dir, phase_path, output_dir = Path(ledger_dir), Path(phase_path), Path(output_dir)
    label_paths = tuple(Path(path) for path in label_paths)
    joined, sources = join_path_geometry_context(ledger_dir=ledger_dir, phase_path=phase_path, label_paths=label_paths)
    summary = summarize_path_geometry(joined)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "path_geometry_by_context.csv"
    support_path = output_dir / "context_support.csv"
    joined_path = output_dir / "joined_path_geometry_context.parquet"
    summary.to_csv(summary_path, index=False)
    support = (
        joined.groupby(["side_name", "regime_state", "transition_phase"], dropna=False)
        .agg(candidates=("candidate_id", "size"), decision_hours=("__ts__", "nunique"))
        .reset_index()
    )
    # The joint table is support-only.  It deliberately has no outcome metric:
    # no reporting group may imply that a phase is equivalent to a regime.
    support.to_csv(support_path, index=False)
    keep = [
        "__ts__", "__decision_ts__", "side_name", "candidate_id", "label_source", "regime_state", "transition_phase",
        "target__pattern_phase_available_utc",
    ] + [item["column"] for item in METRICS if item["column"] in joined]
    joined.loc[:, keep].to_parquet(joined_path, index=False)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "purpose": "descriptive side-specific future-path geometry by regime state and transition phase; no model or policy routing",
        "research_only": True,
        "promotion_eligible": False,
        "source_time_join": "labels.__ts__ == hourly_state.source_utc; execution is one hour later",
        "taxonomies": {
            "regime_state_at_decision": "observable state context at source time",
            "transition_phase_ex_post": "adaptive event phase, target available after its labelled window; not a decision-time feature",
            "joint_table": "support only; outcomes are not grouped jointly to avoid equating regime with transition",
        },
        "metric_contract": {
            "conditional_metrics": [item["name"] for item in METRICS if item["condition"] == "opportunity_only"],
            "uncertainty": "candidate means; 95% normal approximation from decision-hour cluster means",
            "net_ev": "current-frozen-spread counterfactual label where present; descriptive only, not execution-parity or OOF evidence",
        },
        "sources": {
            "ledger_dir": str(ledger_dir),
            "ledger_hourly_sha256": _sha256(ledger_dir / "hourly_state_calendar.parquet"),
            "phase_path": str(phase_path),
            "phase_sha256": _sha256(phase_path),
            "labels": sources,
        },
        "counts": {
            "joined_candidates": int(len(joined)),
            "joined_decision_hours": int(joined["__ts__"].nunique()),
            "summary_rows": int(len(summary)),
            "support_rows": int(len(support)),
        },
        "outputs_sha256": {
            summary_path.name: _sha256(summary_path),
            support_path.name: _sha256(support_path),
            joined_path.name: _sha256(joined_path),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--phase-path", type=Path, default=DEFAULT_PHASES)
    parser.add_argument("--labels", type=Path, nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    print(json.dumps(_safe(materialize_regime_transition_path_geometry_diagnostic(
        ledger_dir=arguments.ledger_dir,
        phase_path=arguments.phase_path,
        label_paths=arguments.labels,
        output_dir=arguments.output_dir,
    )), indent=2, sort_keys=True))
