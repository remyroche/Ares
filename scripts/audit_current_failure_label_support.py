#!/usr/bin/env python3
"""Audit exact failure-label support without changing the canonical policy.

The canonical cohort is one pooled global top-k over the combined strict
model-OOS history.  A forward-role-local top-k is computed only as a
diagnostic upper bound on available July labels; it is never promotion
eligible and never replaces the canonical selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from materialize_current_lineage_exact_failure_labels import (
    FORWARD_FLAG,
    MAPPED_SCORE,
    build_current_exact_failure_labels,
)
from materialize_historical_exact_model_health import stable_global_top_k


DEFAULT_OVERLAY = (
    "data_perp/artifacts/failure_first_current_strict_model_oos_history_20260726_v1/"
    "strict_model_oos_history.parquet"
)
DEFAULT_HEALTH = (
    "data_perp/artifacts/current_lineage_extended_model_health_20260729_v1/"
    "hourly_model_health.parquet"
)
DEFAULT_LABELS = (
    "data_perp/artifacts/"
    "current_lineage_exact_failure_labels_resolved_july19_20260729_v1/"
    "hourly_current_health_and_failure_labels.parquet"
)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def selection_cutoff_diagnostics(
    frame: pd.DataFrame,
    *,
    fraction: float,
    score_column: str = MAPPED_SCORE,
) -> dict[str, Any]:
    work = frame.loc[pd.to_numeric(frame[score_column], errors="coerce").notna()].copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="raise")
    selected = stable_global_top_k(work, score_column=score_column, fraction=fraction)
    cutoff = float(selected[score_column].min())
    return {
        "eligible_rows": int(len(work)),
        "selected_rows": int(len(selected)),
        "cutoff": cutoff,
        "rows_strictly_above_cutoff": int(work[score_column].gt(cutoff).sum()),
        "rows_equal_cutoff": int(work[score_column].eq(cutoff).sum()),
        "selected_rows_equal_cutoff": int(selected[score_column].eq(cutoff).sum()),
        "unique_score_values": int(work[score_column].nunique()),
    }


def role_score_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, local in frame.groupby("failure_first_history_role", sort=True):
        score = pd.to_numeric(local[MAPPED_SCORE], errors="coerce").dropna()
        rows.append(
            {
                "role": role,
                "rows": int(len(local)),
                "mapped_rows": int(len(score)),
                "start_utc": pd.to_datetime(local["__ts__"], utc=True).min(),
                "end_utc": pd.to_datetime(local["__ts__"], utc=True).max(),
                "mapped_mean": float(score.mean()),
                "mapped_std": float(score.std()),
                "mapped_p50": float(score.quantile(0.50)),
                "mapped_p90": float(score.quantile(0.90)),
                "mapped_p99": float(score.quantile(0.99)),
                "mapped_max": float(score.max()),
                "realized_net_mean": float(
                    pd.to_numeric(
                        local["execution_net_ev_12h"], errors="coerce"
                    ).mean()
                ),
                "positive_net_rate": float(
                    pd.to_numeric(
                        local["execution_net_ev_12h"], errors="coerce"
                    ).gt(0.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def daily_selection(frame: pd.DataFrame, selected: pd.DataFrame, cohort: str) -> pd.DataFrame:
    eligible = frame.copy()
    eligible["date"] = pd.to_datetime(eligible["__ts__"], utc=True).dt.strftime("%Y-%m-%d")
    chosen = selected.copy()
    chosen["date"] = pd.to_datetime(chosen["__ts__"], utc=True).dt.strftime("%Y-%m-%d")
    all_counts = eligible.groupby(
        ["failure_first_history_role", "date"], observed=True
    ).size().rename("eligible_rows")
    selected_counts = chosen.groupby(
        ["failure_first_history_role", "date"], observed=True
    ).size().rename("selected_rows")
    output = pd.concat([all_counts, selected_counts], axis=1).fillna(0).reset_index()
    output["selected_rows"] = output["selected_rows"].astype(int)
    output["selection_rate"] = output["selected_rows"] / output["eligible_rows"]
    output.insert(0, "cohort", cohort)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    overlay_path = Path(args.overlay)
    health_path = Path(args.health)
    labels_path = Path(args.labels)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    overlay = pd.read_parquet(overlay_path)
    health = pd.read_parquet(health_path)
    labels = pd.read_parquet(labels_path)
    overlay["__ts__"] = pd.to_datetime(overlay["__ts__"], utc=True, errors="raise")
    labels["source_utc"] = pd.to_datetime(
        labels["source_utc"], utc=True, errors="raise"
    )
    if overlay["candidate_id"].duplicated().any():
        raise ValueError("overlay candidate IDs must be unique")

    mapped = overlay.loc[pd.to_numeric(overlay[MAPPED_SCORE], errors="coerce").notna()].copy()
    canonical_selected = stable_global_top_k(
        mapped, score_column=MAPPED_SCORE, fraction=float(args.top_k_fraction)
    )
    forward = mapped.loc[
        mapped[FORWARD_FLAG].fillna(False).astype(bool)
    ].copy()
    if forward.empty:
        raise ValueError("no resolved forward-OOS rows exist")
    forward_selected = stable_global_top_k(
        forward, score_column=MAPPED_SCORE, fraction=float(args.top_k_fraction)
    )
    forward_labels, forward_events, _ = build_current_exact_failure_labels(
        forward,
        health,
        top_k_fraction=float(args.top_k_fraction),
        allow_resolved_forward=True,
    )

    role_summary = role_score_summary(mapped)
    daily = pd.concat(
        [
            daily_selection(mapped, canonical_selected, "canonical_combined_global"),
            daily_selection(forward, forward_selected, "diagnostic_forward_role_local"),
        ],
        ignore_index=True,
    )
    label_month = (
        labels.assign(month=labels["source_utc"].dt.strftime("%Y-%m"))
        .groupby("month", sort=True)
        .agg(
            health_rows=("source_utc", "size"),
            complete_label_rows=("label_window_complete", "sum"),
            broad_active_hours=("target__economic_failure_broad_active", "sum"),
            strict_active_hours=("target__economic_failure_strict_active", "sum"),
        )
        .reset_index()
    )
    forward_complete = forward_labels["label_window_complete"].astype(bool)
    forward_event_counts = {
        label: int(
            forward_events.loc[
                forward_events["failure_label"].eq(label), "economic_event_id"
            ].nunique()
        )
        for label in ("broad", "strict")
    }

    output.mkdir(parents=True, exist_ok=False)
    role_summary.to_csv(output / "role_score_summary.csv", index=False)
    daily.to_csv(output / "daily_selection_support.csv", index=False)
    label_month.to_csv(output / "canonical_label_month_coverage.csv", index=False)
    forward_labels.loc[
        :,
        [
            "source_utc",
            "label_window_complete",
            "pre_12h_selected_rows",
            "post_12h_selected_rows",
            "target__economic_failure_broad_active",
            "target__economic_failure_strict_active",
        ],
    ].to_parquet(
        output / "diagnostic_forward_role_local_hourly.parquet",
        index=False,
        compression="zstd",
    )

    combined_cutoff = selection_cutoff_diagnostics(
        mapped, fraction=float(args.top_k_fraction)
    )
    forward_cutoff = selection_cutoff_diagnostics(
        forward, fraction=float(args.top_k_fraction)
    )
    selected_role_counts = {
        str(role): int(count)
        for role, count in canonical_selected[
            "failure_first_history_role"
        ].value_counts().items()
    }
    result = {
        "schema": "current_failure_label_support_audit_v1",
        "status": "SUPPORT_LIMIT_DIAGNOSED",
        "canonical_policy_changed": False,
        "canonical_contract": (
            "one pooled global top 10% after causal recent side-EV mapping "
            "over the combined strict model-OOS evaluation population"
        ),
        "canonical_selection": {
            **combined_cutoff,
            "selected_role_counts": selected_role_counts,
            "resolved_forward_selected_rows": int(
                canonical_selected[FORWARD_FLAG].fillna(False).astype(bool).sum()
            ),
        },
        "resolved_forward_population": {
            **forward_cutoff,
            "rows": int(len(forward)),
            "start_utc": forward["__ts__"].min(),
            "end_utc": forward["__ts__"].max(),
            "realized_net_mean": float(
                pd.to_numeric(
                    forward["execution_net_ev_12h"], errors="coerce"
                ).mean()
            ),
        },
        "diagnostic_forward_role_local_top10": {
            "promotion_eligible": False,
            "reason": (
                "role-local selection changes the canonical combined-period "
                "global book and is used only to bound available July support"
            ),
            "selected_rows": int(len(forward_selected)),
            "complete_label_rows": int(forward_complete.sum()),
            "complete_start_utc": forward_labels.loc[
                forward_complete, "source_utc"
            ].min(),
            "complete_end_utc": forward_labels.loc[
                forward_complete, "source_utc"
            ].max(),
            "failure_events": forward_event_counts,
        },
        "finding": (
            "July support is limited primarily by economically lower mapped "
            "scores and sparse global selection, not missing resolved outcomes; "
            "a role-local diagnostic still yields too little independent event "
            "support and must not replace the canonical policy"
        ),
        "required_next_data": (
            "more strict model-OOS resolved history under the unchanged global "
            "selection contract; do not create episodes with timestamp, side, "
            "calendar, or regime quotas"
        ),
        "sources": {
            "overlay": {"path": str(overlay_path), "sha256": _sha256(overlay_path)},
            "health": {"path": str(health_path), "sha256": _sha256(health_path)},
            "labels": {"path": str(labels_path), "sha256": _sha256(labels_path)},
        },
        "outputs": {},
    }
    for name in (
        "role_score_summary.csv",
        "daily_selection_support.csv",
        "canonical_label_month_coverage.csv",
        "diagnostic_forward_role_local_hourly.parquet",
    ):
        path = output / name
        result["outputs"][name] = {"path": str(path), "sha256": _sha256(path)}
    _write_json(output / "manifest.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--overlay", default=DEFAULT_OVERLAY)
    value.add_argument("--health", default=DEFAULT_HEALTH)
    value.add_argument("--labels", default=DEFAULT_LABELS)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--top-k-fraction", type=float, default=0.10)
    return value


if __name__ == "__main__":
    print(json.dumps(_safe(run(parser().parse_args())), indent=2, sort_keys=True))
