#!/usr/bin/env python3
"""Build a separate five-month common30 opportunity-support lineage.

The source is the strict two-layer March--July 2025 execution-EV OOF stream on
the frozen 30-asset exact-path universe.  It is never pooled with the
full-universe historical raw-alpha lineage or the current 2026 execution-EV
lineage.  The runner adds causal recent-EV mapping, exact hourly health/failure
labels, and resolved descriptive opportunity packets.  This is a retrospective
12-hour/100-bps counterfactual lineage: it is not an incumbent-policy replay and
cannot contribute to a prospective/current-policy promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from materialize_economic_opportunity_state_packets import (  # noqa: E402
    PRIMARY_STATE_NAMES,
    attach_context,
    build_event_packets,
    build_hourly_components,
)
from materialize_historical_exact_model_health import (  # noqa: E402
    HEALTH_COLUMNS,
    build_health_and_labels,
)
from run_execution_ev_recent_mapping_ablation import causal_mappings  # noqa: E402


DEFAULT_SOURCE = ROOT / (
    "data_perp/artifacts/febjul2025_execution_ev_common30_two_layer_oof_"
    "20260727_v3"
)
DEFAULT_ACTIVE = ROOT / (
    "data_perp/artifacts/regime_transition_active_head_chronological_oos_"
    "20260729_v2/chronological_oos.parquet"
)
DEFAULT_DESTINATION = ROOT / (
    "data_perp/artifacts/regime_transition_destination_chronological_oos_"
    "20260729_v1/destination_chronological_oos.parquet"
)
DEFAULT_BOCPD = ROOT / (
    "data_perp/artifacts/regime_transition_changepoint_ablation_20260727_v2/"
    "grouped_oof_predictions_and_changepoint_context.parquet"
)
DEFAULT_ORIGIN = ROOT / (
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "hourly_transition_dataset.parquet"
)
LINEAGE = "historical_2025_common30_12h_cost100bps_direct_ev_oof"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def prepare_exact_candidates(
    oof: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    window_days: int = 21,
    minimum_reference_rows: int = 500,
    side_support_target: float = 500.0,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Join strict OOF scores to exact labels and apply causal recent mapping."""

    required_oof = {
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "candidate_month",
        "historical_base_soft_oof",
        "historical_direct_ev_oof",
        "direct_oof_fold_start_utc",
        "direct_oof_train_cutoff_utc",
    }
    required_labels = {
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_exit_reason",
        "execution_exit_minute",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
    }
    missing_oof = sorted(required_oof.difference(oof.columns))
    missing_labels = sorted(required_labels.difference(labels.columns))
    if missing_oof or missing_labels:
        raise ValueError(
            f"source contract missing oof={missing_oof}, labels={missing_labels}"
        )
    if oof.candidate_id.astype(str).duplicated().any():
        raise ValueError("strict common30 OOF candidate IDs must be unique")
    if labels.candidate_id.astype(str).duplicated().any():
        raise ValueError("common30 exact label candidate IDs must be unique")
    oof["__ts__"] = pd.to_datetime(oof["__ts__"], utc=True, errors="raise")
    for column in ("direct_oof_fold_start_utc", "direct_oof_train_cutoff_utc"):
        oof[column] = pd.to_datetime(oof[column], utc=True, errors="raise")
    if not oof["direct_oof_train_cutoff_utc"].le(oof["__ts__"]).all():
        raise ValueError("common30 direct-model cutoff must not follow signal time")
    label_columns = [
        "candidate_id",
        *[
            name
            for name in required_labels
            if name not in {*IDENTITY, "candidate_id", "execution_decision_utc", "execution_label_end_utc"}
        ],
    ]
    # The strict-OOF table may carry convenience copies of resolved outcomes.
    # Always replace those copies from the frozen exact-label ledger so the
    # materialized lineage has one authoritative outcome source and no merge
    # suffix ambiguity.
    outcome_columns = [
        column
        for column in label_columns
        if column != "candidate_id" and column in oof.columns
    ]
    joined = oof.drop(columns=outcome_columns).merge(
        labels.loc[:, label_columns],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if len(joined) != len(oof) or joined.execution_net_ev_12h.isna().any():
        raise ValueError("exact labels do not cover every strict common30 OOF row")
    label_identity = labels.set_index("candidate_id")
    for column in ("__ts__", "__symbol__", "side_name"):
        expected = joined.candidate_id.map(label_identity[column])
        if column == "__ts__":
            observed = pd.to_datetime(joined[column], utc=True)
            expected = pd.to_datetime(expected, utc=True)
        else:
            observed = joined[column].astype(str)
            expected = expected.astype(str)
        if not observed.eq(expected).all():
            raise ValueError(f"common30 OOF/label identity mismatch: {column}")
    for column in ("execution_decision_utc", "execution_label_end_utc"):
        joined[column] = pd.to_datetime(joined[column], utc=True, errors="raise")
        expected = pd.to_datetime(
            joined.candidate_id.map(label_identity[column]), utc=True, errors="raise"
        )
        if not joined[column].eq(expected).all():
            raise ValueError(f"common30 OOF/label timing mismatch: {column}")
    if not joined.execution_decision_utc.eq(
        pd.to_datetime(joined["__ts__"], utc=True) + pd.Timedelta(hours=1)
    ).all():
        raise ValueError("common30 decision must equal signal+1h")
    if not joined["direct_oof_train_cutoff_utc"].lt(
        joined["execution_decision_utc"]
    ).all():
        raise ValueError("common30 direct-model cutoff must precede decision time")
    if not joined.execution_label_end_utc.eq(
        joined.execution_decision_utc + pd.Timedelta(hours=12)
    ).all():
        raise ValueError("common30 exact labels must resolve at decision+12h")
    accounting_error = (
        joined.execution_gross_ev_12h
        - joined.execution_cost_return
        - joined.execution_net_ev_12h
    ).abs()
    if float(accounting_error.max()) > 1e-12:
        raise ValueError("common30 gross-cost-net accounting identity failed")
    joined["score_raw"] = pd.to_numeric(
        joined["historical_direct_ev_oof"], errors="raise"
    )
    mapped, audit = causal_mappings(
        joined,
        score_col="score_raw",
        window_days=int(window_days),
        min_reference_rows=int(minimum_reference_rows),
        side_support_target=float(side_support_target),
    )
    reference = pd.DataFrame(audit)
    if len(reference):
        reference["snapshot"] = pd.to_datetime(reference["snapshot"], utc=True)
        mapped["snapshot"] = mapped.execution_decision_utc.dt.floor("D")
        mapped = mapped.merge(
            reference.loc[
                :,
                [
                    "snapshot",
                    "reference_rows",
                    "long_reference_rows",
                    "short_reference_rows",
                ],
            ],
            on="snapshot",
            how="left",
            validate="many_to_one",
        ).drop(columns="snapshot")
    else:
        mapped["reference_rows"] = np.nan
        mapped["long_reference_rows"] = np.nan
        mapped["short_reference_rows"] = np.nan
    mapped["mapped_eligible"] = mapped[
        "causal_recent_side_isotonic_ev"
    ].notna()
    mapped["mapped_direct_net"] = mapped[
        "causal_recent_side_isotonic_ev"
    ]
    mapped["causal_score_percentile"] = mapped["causal_recent_percentile"]
    mapped["map_reference_rows"] = mapped["reference_rows"]
    mapped["effective_label_resolution_utc"] = mapped[
        "execution_label_end_utc"
    ]
    mapped["execution_exit_class"] = (
        mapped["execution_exit_reason"]
        .astype(str)
        .replace({"full_sl": "full_stop"})
    )
    mapped["opportunity_gross_above_cost_0bps"] = mapped[
        "execution_gross_ev_12h"
    ].gt(mapped["execution_cost_return"])
    mapped["exit_is_timeout"] = mapped.execution_exit_class.eq("timeout")
    mapped["score_base"] = mapped["historical_base_soft_oof"]
    mapped["score_alpha_residual"] = np.nan
    mapped["score_direct_execution_ev"] = mapped["score_raw"]
    mapped["score_mapped_execution_ev"] = mapped["mapped_direct_net"]
    return mapped, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    oof_path = source_root / "two_layer_direct_ev_strict_oof.parquet"
    labels_path = source_root / "exact_1m_execution_ev_12h_labels.parquet"
    candidates, mapping_audit = prepare_exact_candidates(
        pd.read_parquet(oof_path),
        pd.read_parquet(labels_path),
        window_days=int(args.window_days),
        minimum_reference_rows=int(args.minimum_reference_rows),
        side_support_target=float(args.side_support_target),
    )
    hourly, events, selected = build_health_and_labels(
        candidates,
        top_k_fraction=float(args.top_k_fraction),
        low_support_rows=int(args.minimum_reference_rows),
        selection_score_column="mapped_direct_net",
        selection_contract=(
            "one pooled global top10 by causal 21-day side-shrunk isotonic "
            "mapped EV; candidate-ID tie break"
        ),
    )
    components = build_hourly_components(selected, LINEAGE)
    components = attach_context(
        components,
        hourly,
        pd.read_parquet(args.active),
        pd.read_parquet(args.destination),
        pd.read_parquet(args.bocpd),
        pd.read_parquet(args.origin_state),
    )
    packets, trajectories = build_event_packets(
        components,
        events,
        LINEAGE,
        reference_days=int(args.packet_reference_days),
        minimum_reference_hours=int(args.packet_minimum_reference_hours),
    )
    state_columns = [
        *[f"state__{name}" for name in PRIMARY_STATE_NAMES],
        "state__normal_opportunity",
        "state__mixed",
        "state__unclassified",
    ]
    taxonomy = (
        packets.groupby("lineage", observed=True)[state_columns]
        .agg(["sum", "mean"])
        .reset_index()
    )
    taxonomy.columns = [
        (
            left
            if not right
            else f"{left}__{right}"
        )
        for left, right in taxonomy.columns
    ]
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "candidates": output / "causal_mapped_strict_oof_candidates.parquet",
        "hourly": output / "hourly_exact_model_health_and_failure_labels.parquet",
        "events": output / "economic_failure_events.parquet",
        "selected": output / "frozen_global_top10_candidates.parquet",
        "components": output / "hourly_opportunity_state_components.parquet",
        "packets": output / "strict_event_packets.parquet",
        "trajectories": output / "strict_event_trajectories.parquet",
        "taxonomy": output / "taxonomy_summary.parquet",
        "health_catalog": output / "health_feature_catalog.csv",
    }
    frames = {
        "candidates": candidates,
        "hourly": hourly,
        "events": events,
        "selected": selected,
        "components": components,
        "packets": packets,
        "trajectories": trajectories,
        "taxonomy": taxonomy,
    }
    for key, frame in frames.items():
        frame.to_parquet(paths[key], index=False, compression="zstd")
    pd.DataFrame({"feature": HEALTH_COLUMNS}).to_csv(
        paths["health_catalog"], index=False
    )
    event_counts = {
        label: int(
            events.loc[events.failure_label.eq(label), "economic_event_id"].nunique()
        )
        for label in ("broad", "strict")
    }
    report = {
        "schema": "common30_opportunity_support_extension_v1",
        "status": "SEPARATE_OOF_COUNTERFACTUAL_LINEAGE_PACKETS_COMPLETE",
        "lineage": LINEAGE,
        "lineage_boundary": (
            "frozen 30-asset March-July 2025 exact-path 12-hour/100-bps "
            "counterfactual with two-layer direct-EV OOF; never pool with "
            "full-universe historical raw-alpha or current 2026 execution-EV "
            "lineages"
        ),
        "policy_parity": {
            "incumbent_parity_claimed": False,
            "admission_calibrator_replayed": False,
            "portfolio_constraints_replayed": False,
            "prospective_support_gate_eligible": False,
            "permitted_use": (
                "within-lineage retrospective descriptive recurrence and "
                "failure-packet research only"
            ),
        },
        "calendar": {
            "start": str(candidates.__ts__.min()),
            "end": str(candidates.__ts__.max()),
            "months": {
                str(month): int(len(local))
                for month, local in candidates.groupby("candidate_month", sort=True)
            },
        },
        "rows": {
            "strict_oof_candidates": int(len(candidates)),
            "causal_mapped_eligible": int(candidates.mapped_eligible.sum()),
            "selected_causal_mapped_global_top10": int(len(selected)),
            "hourly": int(len(hourly)),
            "strict_containing_incidents": int(len(packets)),
        },
        "failure_events": event_counts,
        "selection_contract": (
            "one pooled global top10 after causal 21-day side-shrunk isotonic "
            "mapping; candidate-ID tie break; never per timestamp/month/side; "
            "no admission or portfolio replay"
        ),
        "mapping_contract": {
            "window_days": int(args.window_days),
            "minimum_reference_rows": int(args.minimum_reference_rows),
            "side_support_target": float(args.side_support_target),
            "reference": "labels resolved before each UTC-day snapshot",
            "audit_days": int(len(mapping_audit)),
        },
        "packet_contract": {
            "descriptive_multilabel_only": True,
            "recovery_horizon_hours": 72,
            "supervised_router_authorized": False,
        },
        "sources": {
            "oof": {"path": str(oof_path.resolve()), "sha256": sha256(oof_path)},
            "labels": {
                "path": str(labels_path.resolve()),
                "sha256": sha256(labels_path),
            },
            **{
                f"source_contract__{path.name}": {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
                for path in (
                    source_root / "summary.json",
                    source_root / "coverage_preflight.json",
                    source_root / "base_fold_audit.json",
                    source_root / "direct_ev_fold_audit.json",
                )
            },
            **{
                key: {
                    "path": str(Path(getattr(args, key)).resolve()),
                    "sha256": sha256(Path(getattr(args, key))),
                }
                for key in ("active", "destination", "bocpd", "origin_state")
            },
        },
        "outputs": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "promotion_eligible": False,
    }
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest.sha256").write_text(sha256(manifest) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    result.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    result.add_argument("--bocpd", type=Path, default=DEFAULT_BOCPD)
    result.add_argument("--origin-state", type=Path, default=DEFAULT_ORIGIN)
    result.add_argument("--window-days", type=int, default=21)
    result.add_argument("--minimum-reference-rows", type=int, default=500)
    result.add_argument("--side-support-target", type=float, default=500.0)
    result.add_argument("--top-k-fraction", type=float, default=0.10)
    result.add_argument("--packet-reference-days", type=int, default=30)
    result.add_argument("--packet-minimum-reference-hours", type=int, default=168)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    print(json.dumps(safe(run(parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
