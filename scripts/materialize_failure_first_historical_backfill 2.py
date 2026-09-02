#!/usr/bin/env python3
"""Materialize a continuous 2025 failure-first comparator ledger.

This source is deliberately a separate model generation.  It combines the
common-30 strict two-layer execution-EV OOF score with exact one-minute
current-policy counterfactual outcomes, applies the same causal 21-day
side-local EV correction, and attaches point-in-time raw candidate state.
It must never be described as current-model OOF or factual historical spread.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_market_state import (  # noqa: E402
    MARKET_STATE_COLUMNS,
    attach_decision_time_market_state,
)
from scripts.run_exact_policy_capture_support_ablation import (  # noqa: E402
    apply_recent_mapping_frame,
)


DEFAULT_SOURCE = Path(
    "data_perp/artifacts/"
    "feb2025_jul2026_execution_ev_common30_transfer_oof_20260727_v4/"
    "two_layer_direct_ev_strict_oof.parquet"
)
DEFAULT_LABELS = Path(
    "data_perp/artifacts/"
    "feb2025_jul2026_execution_ev_common30_transfer_oof_20260727_v4/"
    "exact_1m_execution_ev_12h_labels.parquet"
)
DEFAULT_EARLY_SOURCE = Path(
    "data_perp/artifacts/"
    "janfeb2025_execution_ev_exact1m_two_layer_oof_20260727_v2/"
    "two_layer_direct_ev_strict_oof.parquet"
)
DEFAULT_EARLY_LABELS = Path(
    "data_perp/artifacts/"
    "janfeb2025_execution_ev_exact1m_two_layer_oof_20260727_v2/"
    "exact_1m_execution_ev_12h_labels.parquet"
)
DEFAULT_FEATURE_STORE = Path("data_perp/features/20260711_070000")
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/failure_first_historical_backfill_20260726_v3"
)
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--exact-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--early-source-oof", type=Path, default=DEFAULT_EARLY_SOURCE
    )
    parser.add_argument(
        "--early-exact-labels", type=Path, default=DEFAULT_EARLY_LABELS
    )
    parser.add_argument(
        "--feature-store-root", type=Path, default=DEFAULT_FEATURE_STORE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2025-01-15T00:00:00Z")
    parser.add_argument("--end", default="2025-12-01T00:00:00Z")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    primary = pd.read_parquet(args.source_oof)
    allowed_symbols = set(primary["__symbol__"].astype(str))
    early = pd.read_parquet(args.early_source_oof)
    early = early.loc[early["__symbol__"].astype(str).isin(allowed_symbols)].copy()
    primary["__source_generation__"] = "common30_transfer_oof_2025"
    early["__source_generation__"] = "janfeb_two_layer_oof_common30_slice"
    source = pd.concat([early, primary], ignore_index=True)
    label_columns = [
        *IDENTITY,
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
    ]
    early_labels = pd.read_parquet(
        args.early_exact_labels, columns=label_columns
    )
    primary_labels = pd.read_parquet(args.exact_labels, columns=label_columns)
    early_labels["__source_generation__"] = (
        "janfeb_two_layer_oof_common30_slice"
    )
    primary_labels["__source_generation__"] = "common30_transfer_oof_2025"
    labels = pd.concat([early_labels, primary_labels], ignore_index=True)
    for frame, name, key in (
        (source, "source OOF", ["candidate_id"]),
        (
            labels,
            "exact labels",
            ["candidate_id", "__source_generation__"],
        ),
    ):
        if frame["candidate_id"].isna().any() or frame.duplicated(key).any():
            raise ValueError(f"{name} candidate_id must be unique")
    source["execution_decision_utc"] = pd.to_datetime(
        source["execution_decision_utc"], utc=True, errors="raise"
    )
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("start/end must be ordered timezone-aware timestamps")
    source = source.loc[
        source["execution_decision_utc"].ge(start)
        & source["execution_decision_utc"].lt(end)
    ].copy()
    if source.empty:
        raise ValueError("historical failure-first period is empty")
    label_fields = labels.loc[
        :,
        [
            "candidate_id",
            "__source_generation__",
            "execution_gross_ev_12h",
            "execution_cost_return",
            "execution_net_ev_12h",
        ],
    ].rename(columns={"execution_net_ev_12h": "__label_net__"})
    source = source.merge(
        label_fields,
        on=["candidate_id", "__source_generation__"],
        how="inner",
        validate="one_to_one",
    )
    if len(source) == 0:
        raise ValueError("exact label join is empty")
    if not np.allclose(
        source["execution_net_ev_12h"],
        source["__label_net__"],
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("strict OOF and exact-label net outcomes disagree")
    if not np.allclose(
        source["execution_gross_ev_12h"]
        - source["execution_cost_return"],
        source["execution_net_ev_12h"],
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("historical gross-cost-net accounting does not reconcile")

    source["__raw_score__"] = pd.to_numeric(
        source["historical_direct_ev_oof"], errors="raise"
    )
    source["causal_recent_side_isotonic_ev"] = np.nan
    mapping_report: dict[str, Any] = {}
    for generation, positions in source.groupby(
        "__source_generation__", sort=True
    ).groups.items():
        local = source.loc[list(positions)].copy()
        mapped, report = apply_recent_mapping_frame(
            local,
            local["__raw_score__"].to_numpy(np.float64),
            scope="side",
        )
        source.loc[list(positions), "causal_recent_side_isotonic_ev"] = mapped
        mapping_report[str(generation)] = report
    source["causal_recent_side_isotonic_ev__is_oof"] = True
    source["causal_recent_side_isotonic_ev__is_forward_oos"] = False
    source["catboost__residual__without_hpo__all_features"] = source[
        "__raw_score__"
    ]
    source["existing_alpha_ev"] = source["historical_base_soft_oof"]
    source["base_oof_score"] = source["historical_base_soft_oof"]
    source["evaluation_origin"] = source["__source_generation__"]
    source["__ts__"] = pd.to_datetime(
        source["__ts__"], utc=True, errors="raise"
    )

    state_candidates = source.loc[
        :,
        [
            "candidate_id",
            "__ts__",
            "__symbol__",
            "side_name",
            "execution_decision_utc",
        ],
    ].copy()
    joined_state = attach_decision_time_market_state(
        state_candidates,
        feature_store_root=args.feature_store_root,
    )
    coverage = joined_state.coverage.copy()
    retained_source_columns = coverage.loc[
        coverage["finite_fraction"].ge(0.90), "source_column"
    ].astype(str).tolist()
    if len(retained_source_columns) < 20:
        raise ValueError(
            "fewer than 20 canonical raw-H0 market-state fields clear 90% coverage"
        )
    retained_columns = [
        f"mkt_state__{name}" for name in retained_source_columns
    ]
    state = joined_state.frame.loc[
        :,
        [
            "candidate_id",
            "__ts__",
            "__symbol__",
            "side_name",
            "execution_decision_utc",
            "mkt_state_source_utc",
            *retained_columns,
        ],
    ].copy()
    raw_coverage = float(
        state[retained_columns].notna().any(axis=1).mean()
    )
    if raw_coverage < 0.99:
        raise ValueError(
            f"canonical raw-H0 row coverage {raw_coverage:.4f} is below 99%"
        )
    state = state.rename(
        columns={"mkt_state_source_utc": "raw_state_source_utc_h0"}
    )
    state = state.rename(
        columns={
            name: f"{name}__h0" for name in retained_columns
        }
    )
    coverage.to_csv(output / "canonical_h0_coverage.csv", index=False)
    joined_state.source_audit.to_parquet(
        output / "canonical_h0_source_audit.parquet", index=False
    )

    ledger_columns = [
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
        "execution_cost_return",
        "causal_recent_side_isotonic_ev",
        "causal_recent_side_isotonic_ev__is_oof",
        "causal_recent_side_isotonic_ev__is_forward_oos",
        "catboost__residual__without_hpo__all_features",
        "existing_alpha_ev",
        "base_oof_score",
        "base_margin_to_cutoff",
        "base_margin_to_cutoff_z",
        "evaluation_origin",
    ]
    ledger = source.loc[:, ledger_columns].sort_values(
        ["execution_decision_utc", "candidate_id"], kind="stable"
    )
    ledger_path = output / "mapped_strict_oof_ledger.parquet"
    state_path = output / "observable_h0_state.parquet"
    ledger.to_parquet(ledger_path, index=False)
    state.to_parquet(state_path, index=False)
    _write_json(output / "causal_mapping_report.json", mapping_report)
    manifest = {
        "schema": "failure_first_historical_backfill_v2",
        "status": "historical_comparator_not_current_model_oof",
        "period": {"start": start, "end_exclusive": end},
        "rows": int(len(ledger)),
        "hours": int(ledger["execution_decision_utc"].nunique()),
        "days": int(ledger["execution_decision_utc"].dt.floor("D").nunique()),
        "sides": sorted(ledger["side_name"].astype(str).unique()),
        "assets": int(ledger["__symbol__"].nunique()),
        "raw_state_features": [
            f"{name}__h0" for name in retained_columns
        ],
        "raw_state_join_coverage": raw_coverage,
        "mapping_scope": "causal_side_21d",
        "evaluation_origins": sorted(
            ledger["evaluation_origin"].astype(str).unique()
        ),
        "economic_interpretation": (
            "current frozen spread counterfactual on historical exact-one-minute "
            "paths; not factual historical execution costs"
        ),
        "source": {
            "oof": str(Path(args.source_oof).resolve()),
            "oof_sha256": _sha256(Path(args.source_oof)),
            "labels": str(Path(args.exact_labels).resolve()),
            "labels_sha256": _sha256(Path(args.exact_labels)),
            "early_oof": str(Path(args.early_source_oof).resolve()),
            "early_oof_sha256": _sha256(Path(args.early_source_oof)),
            "early_labels": str(Path(args.early_exact_labels).resolve()),
            "early_labels_sha256": _sha256(Path(args.early_exact_labels)),
            "feature_store": str(Path(args.feature_store_root).resolve()),
        },
        "outputs": {
            ledger_path.name: {
                "rows": int(len(ledger)),
                "sha256": _sha256(ledger_path),
            },
            state_path.name: {
                "rows": int(len(state)),
                "sha256": _sha256(state_path),
            },
        },
    }
    _write_json(output / "manifest.json", manifest)
    return {
        "output_dir": output,
        "rows": int(len(ledger)),
        "days": manifest["days"],
        "raw_state_features": len(retained_columns),
    }


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
