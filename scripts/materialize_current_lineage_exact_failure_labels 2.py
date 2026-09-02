#!/usr/bin/env python3
"""Materialize exact before/after failure labels on current execution-EV OOF.

Admission is one pooled global top-k after the causal recent side-EV mapping;
it is never per timestamp or side.  The existing 29-field current-lineage
health panel remains the inference feature source.  This materializer adds
only exact, candidate-weighted economic labels and their availability times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materialize_historical_exact_model_health import (  # noqa: E402
    _safe,
    _sha256,
    _write_json,
    add_failure_labels,
    stable_global_top_k,
)


MAPPED_SCORE = "causal_recent_side_isotonic_ev"
OOF_FLAG = f"{MAPPED_SCORE}__is_oof"
FORWARD_FLAG = f"{MAPPED_SCORE}__is_forward_oos"
DEFAULT_OVERLAY = ROOT / (
    "data_perp/artifacts/failure_first_detector_current_transfer_20260726_v5/"
    "candidate_overlay.parquet"
)
DEFAULT_HEALTH = ROOT / (
    "data_perp/artifacts/regime_transition_current_model_health_20260727_v1/"
    "hourly_model_health.parquet"
)


def build_current_exact_failure_labels(
    overlay: pd.DataFrame,
    health: pd.DataFrame,
    *,
    top_k_fraction: float,
    allow_resolved_forward: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_overlay = {
        "candidate_id",
        "__ts__",
        "__symbol__",
        "side_name",
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
        MAPPED_SCORE,
        OOF_FLAG,
        FORWARD_FLAG,
    }
    missing = sorted(required_overlay.difference(overlay.columns))
    if missing:
        raise ValueError(f"current overlay lacks {missing}")
    required_health = {"source_utc", "execution_decision_utc"}
    missing = sorted(required_health.difference(health.columns))
    if missing:
        raise ValueError(f"current health lacks {missing}")
    work = overlay.copy()
    for column in ("__ts__", "execution_decision_utc", "execution_label_end_utc"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    if work["candidate_id"].duplicated().any():
        raise ValueError("current overlay candidate IDs must be unique")
    work[MAPPED_SCORE] = pd.to_numeric(work[MAPPED_SCORE], errors="coerce")
    work = work.loc[work[MAPPED_SCORE].notna()].copy()
    if work.empty:
        raise ValueError("no mapped execution-EV rows remain for failure labels")
    oof = work[OOF_FLAG].fillna(False).astype(bool)
    forward = work[FORWARD_FLAG].fillna(False).astype(bool)
    if (oof & forward).any():
        raise ValueError("current mapped OOF and forward flags must be exclusive")
    if allow_resolved_forward:
        combined_flag = "failure_first_score_is_strict_model_oos"
        if combined_flag not in work:
            raise ValueError("resolved forward extension lacks strict model-OOS flag")
        combined = work[combined_flag].fillna(False).astype(bool)
        if not combined.all() or not (oof ^ forward).all():
            raise ValueError(
                "resolved extension requires exclusive OOF/forward strict model-OOS"
            )
    elif not oof.all() or forward.any():
        raise ValueError("this exact current-label artifact requires strict OOF only")
    if not work["execution_decision_utc"].eq(
        work["__ts__"] + pd.Timedelta(hours=1)
    ).all():
        raise ValueError("current overlay violates source-to-decision timing")
    if not work["execution_label_end_utc"].gt(
        work["execution_decision_utc"]
    ).all():
        raise ValueError("current labels must resolve after the decision")
    for column in (
        MAPPED_SCORE,
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if not np.isfinite(
        work[
            [
                MAPPED_SCORE,
                "execution_gross_ev_12h",
                "execution_net_ev_12h",
            ]
        ].to_numpy(float)
    ).all():
        raise ValueError("current overlay score/economics must be finite")

    selected = stable_global_top_k(
        work, score_column=MAPPED_SCORE, fraction=float(top_k_fraction)
    )
    selected["execution_cost_return"] = (
        selected["execution_gross_ev_12h"]
        - selected["execution_net_ev_12h"]
    )
    grouped = selected.groupby("__ts__", observed=True, sort=True)
    economics = grouped.agg(
        health__selected_rows=("candidate_id", "size"),
        realized_net_mean=("execution_net_ev_12h", "mean"),
        realized_net_sum=("execution_net_ev_12h", "sum"),
        expected_mapped_net_mean=(MAPPED_SCORE, "mean"),
        expected_mapped_net_sum=(MAPPED_SCORE, "sum"),
        outcome_available_utc=("execution_label_end_utc", "max"),
    )
    economics["mapping_residual_mean"] = (
        economics["realized_net_mean"]
        - economics["expected_mapped_net_mean"]
    )
    economics["mapping_residual_sum"] = (
        economics["realized_net_sum"]
        - economics["expected_mapped_net_sum"]
    )

    current_health = health.copy()
    for column in ("source_utc", "execution_decision_utc"):
        current_health[column] = pd.to_datetime(
            current_health[column], utc=True, errors="raise"
        )
    if current_health["source_utc"].duplicated().any():
        raise ValueError("current health must have one row per source hour")
    hourly = current_health.merge(
        economics.reset_index().rename(columns={"__ts__": "source_utc"}),
        on="source_utc",
        how="left",
        validate="one_to_one",
    )
    hourly["health__selected_rows"] = pd.to_numeric(
        hourly["health__selected_rows"], errors="coerce"
    ).fillna(0.0)
    for column in (
        "realized_net_sum",
        "expected_mapped_net_sum",
        "mapping_residual_sum",
    ):
        hourly[column] = pd.to_numeric(
            hourly[column], errors="coerce"
        ).fillna(0.0)
    labelled, events = add_failure_labels(
        hourly,
        thresholds={"broad": -0.5, "strict": -1.0},
        selection_contract=(
            "one pooled global top10 after causal recent side-EV mapping"
        ),
    )
    return labelled, events, selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    overlay_path = Path(args.overlay)
    health_path = Path(args.health)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    labelled, events, selected = build_current_exact_failure_labels(
        pd.read_parquet(overlay_path),
        pd.read_parquet(health_path),
        top_k_fraction=float(args.top_k_fraction),
        allow_resolved_forward=bool(args.allow_resolved_forward),
    )
    output.mkdir(parents=True, exist_ok=False)
    hourly_path = output / "hourly_current_health_and_failure_labels.parquet"
    event_path = output / "economic_failure_events.parquet"
    selected_path = output / "frozen_global_top10_mapped_candidates.parquet"
    labelled.to_parquet(hourly_path, index=False, compression="zstd")
    events.to_parquet(event_path, index=False, compression="zstd")
    selected.to_parquet(selected_path, index=False, compression="zstd")
    complete = labelled["label_window_complete"].fillna(False).astype(bool)
    exact_current_lineage = args.lineage_kind == "current_exact"
    manifest = {
        "schema": "current_lineage_exact_failure_labels_v1",
        "status": (
            "HISTORICAL_RECONSTRUCTED_EXECUTION_EXACT_LABELS_COMPLETE"
            if not exact_current_lineage
            else (
                "CURRENT_LINEAGE_RESOLVED_STRICT_MODEL_OOS_EXACT_LABELS_COMPLETE"
                if args.allow_resolved_forward
                else "CURRENT_LINEAGE_STRICT_OOF_EXACT_LABELS_COMPLETE"
            )
        ),
        "lineage_kind": args.lineage_kind,
        "current_lineage": exact_current_lineage,
        "lineage_disclosure": (
            "exact frozen/current execution lineage"
            if exact_current_lineage
            else (
                "historical reconstruction of current execution architecture; "
                "not a frozen current-model backcast"
            )
        ),
        "strict_oof_only": not bool(args.allow_resolved_forward),
        "resolved_forward_included": bool(args.allow_resolved_forward),
        "selection_contract": (
            "one pooled global top 10% after causal recent side-EV mapping; "
            "candidate-ID tie break; never per timestamp or side"
        ),
        "label_contract": (
            "candidate-row-weighted exact hourly pre[-12h,0) versus "
            "post[0,+12h); missing source-hour windows ineligible; exact "
            "policy net and mapped residual; negative post net; 2-of-next-3 "
            "persistence; targets available after every post-window outcome"
        ),
        "rows": int(len(labelled)),
        "complete_label_window_rows": int(complete.sum()),
        "complete_label_window_fraction": float(complete.mean()),
        "selected_candidates": int(len(selected)),
        "start_utc": labelled["source_utc"].min(),
        "end_utc": labelled["source_utc"].max(),
        "failure_events": {
            label: int(
                events.loc[
                    events["failure_label"].eq(label), "economic_event_id"
                ].nunique()
            )
            for label in ("broad", "strict")
        },
        "sources": {
            "overlay": {
                "path": str(overlay_path),
                "sha256": _sha256(overlay_path),
            },
            "health": {
                "path": str(health_path),
                "sha256": _sha256(health_path),
            },
        },
        "outputs": {
            "hourly": {
                "path": str(hourly_path),
                "sha256": _sha256(hourly_path),
            },
            "events": {
                "path": str(event_path),
                "sha256": _sha256(event_path),
            },
            "selected_candidates": {
                "path": str(selected_path),
                "sha256": _sha256(selected_path),
            },
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--health", type=Path, default=DEFAULT_HEALTH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--allow-resolved-forward", action="store_true")
    parser.add_argument(
        "--lineage-kind",
        choices=("current_exact", "historical_reconstructed"),
        default="current_exact",
    )
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
