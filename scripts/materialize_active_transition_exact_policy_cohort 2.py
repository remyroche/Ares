#!/usr/bin/env python3
"""Materialize an exact-policy cohort for active-transition policy research.

This is deliberately a *cohort builder*, not an economic policy result.  It
joins three independent, stable contracts without manufacturing any execution
paths:

* causal global 21-day recent-EV scores on candidate IDs;
* the canonical exact 1m policy labels, including actual entry/exit prices,
  cost, exit cause and realized exit time; and
* the grouped-OOF active-transition probability on the hourly source time.

The active head's grouped OOF is useful research evidence but is not
chronological.  ``evidence_gate.json`` therefore explicitly prevents a
promotion claim even when all execution fields join exactly.  It also records
whether the resulting window has enough active-transition support for a
meaningful overlay comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (
    load_portfolio_policy_params,
    normalise_candidate_table,
    replay_candidates,
)


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SCORE_COLUMN = "causal_recent_isotonic_ev"
SCORE_OOF_FLAG = "causal_recent_isotonic_ev__is_oof"
SCORE_FORWARD_FLAG = "causal_recent_isotonic_ev__is_forward_oos"
DEFAULT_SCORES = Path(
    "data_perp/artifacts/"
    "execution_ev_context_clean_recent_mapping_forward_july19_20260726_v1/"
    "mapped_oof.parquet"
)
DEFAULT_EXACT_LABELS = Path(
    "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/"
    "execution_ev_policy_labels.parquet"
)
DEFAULT_ACTIVE_OOF = Path(
    "data_perp/artifacts/regime_transition_active_head_20260726_v1/"
    "grouped_oof.parquet"
)
DEFAULT_PORTFOLIO_CONFIG = Path(
    "data_perp/artifacts/"
    "s59_s52_finalfit_meta_repairedcoverage_v9tail95_mlp_hierev_20260715_v3/"
    "policy_params/optimized_portfolio_policy_config.json"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/active_transition_exact_policy_cohort_20260727_v2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _require_unique(frame: pd.DataFrame, *, name: str) -> None:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing identity columns: {missing}")
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{name} has duplicate candidate identities")


def _rank_pct(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="raise").rank(method="max", pct=True)


def materialize_cohort(
    scores: pd.DataFrame,
    exact_labels: pd.DataFrame,
    active_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return replay-ready candidate rows, active-only rows and an evidence gate.

    Only candidate-score rows explicitly marked OOF are retained.  The active
    probability remains labelled research-grouped-OOF: it is not upgraded to a
    chronological OOS claim by this join.
    """

    _require_unique(scores, name="mapped score ledger")
    _require_unique(exact_labels, name="exact policy labels")
    required_scores = {SCORE_COLUMN, SCORE_OOF_FLAG, SCORE_FORWARD_FLAG}
    missing_scores = sorted(required_scores.difference(scores.columns))
    if missing_scores:
        raise ValueError(f"mapped score ledger missing columns: {missing_scores}")
    required_labels = {
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_exit_hour",
        "execution_exit_reason",
        "execution_entry_price",
        "execution_exit_price",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
    }
    missing_labels = sorted(required_labels.difference(exact_labels.columns))
    if missing_labels:
        raise ValueError(f"exact policy labels missing columns: {missing_labels}")
    required_active = {
        "source_utc",
        "target__event_id",
        "target__transition_active",
        "prediction",
    }
    missing_active = sorted(required_active.difference(active_oof.columns))
    if missing_active:
        raise ValueError(f"active-transition OOF missing columns: {missing_active}")

    score_work = scores.loc[
        scores[SCORE_OOF_FLAG].fillna(False).astype(bool),
        [*IDENTITY, SCORE_COLUMN, SCORE_OOF_FLAG, SCORE_FORWARD_FLAG],
    ].copy()
    score_work["__ts__"] = pd.to_datetime(score_work["__ts__"], utc=True, errors="raise")
    if score_work[SCORE_COLUMN].isna().any():
        raise ValueError("explicit OOF candidate scores contain null global EV values")
    if score_work[SCORE_FORWARD_FLAG].fillna(False).astype(bool).any():
        raise ValueError("candidate score rows cannot be both OOF and forward OOS")

    labels = exact_labels.loc[:, [*IDENTITY, *sorted(required_labels)]].copy()
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="raise")
    exact_join = score_work.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    missing_exact = exact_join["execution_entry_price"].isna()
    if missing_exact.any():
        raise ValueError(f"{int(missing_exact.sum())} OOF candidate rows lack exact-policy labels")

    active = active_oof.loc[:, sorted(required_active)].copy()
    active["source_utc"] = pd.to_datetime(active["source_utc"], utc=True, errors="raise")
    if active["source_utc"].duplicated().any():
        raise ValueError("active-transition OOF must have exactly one score per source hour")
    joined = exact_join.merge(
        active.rename(
            columns={
                "prediction": "active_transition_probability_grouped_oof",
                "target__transition_active": "expost__transition_active",
            }
        ),
        left_on="__ts__",
        right_on="source_utc",
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        raise ValueError("there is no overlap between exact OOF candidate scores and active OOF")
    probability = pd.to_numeric(
        joined["active_transition_probability_grouped_oof"], errors="raise"
    )
    if probability.lt(0.0).any() or probability.gt(1.0).any():
        raise ValueError("active-transition probabilities must be within [0, 1]")
    accounting_error = (
        pd.to_numeric(joined["execution_gross_ev_12h"], errors="raise")
        - pd.to_numeric(joined["execution_cost_return"], errors="raise")
        - pd.to_numeric(joined["execution_net_ev_12h"], errors="raise")
    ).abs()
    if float(accounting_error.max()) > 1e-7:
        raise ValueError("exact policy labels violate gross - cost = net")
    decision = pd.to_datetime(joined["execution_decision_utc"], utc=True, errors="raise")
    exit_hour = pd.to_numeric(joined["execution_exit_hour"], errors="raise")
    exit_timestamp = decision + pd.to_timedelta(exit_hour, unit="h")
    label_end = pd.to_datetime(joined["execution_label_end_utc"], utc=True, errors="raise")
    if (exit_timestamp > label_end).any():
        raise ValueError("exact exit timestamp exceeds the label horizon")
    side = joined["side_name"].astype(str).str.lower()
    score = pd.to_numeric(joined[SCORE_COLUMN], errors="raise")
    replay = pd.DataFrame(
        {
            "timestamp": decision,
            "symbol": joined["__symbol__"].astype(str),
            "side": side,
            "strategy_id": np.where(
                side.eq("short"),
                "short_execution_ev_residual",
                "long_execution_ev_residual",
            ),
            "base_strategy_threshold": 0.90,
            "calibrated_score": score,
            "normalized_rank_score": _rank_pct(score),
            "entry_price": pd.to_numeric(joined["execution_entry_price"], errors="raise"),
            "exit_timestamp": exit_timestamp,
            "exit_price": pd.to_numeric(joined["execution_exit_price"], errors="raise"),
            "net_return": pd.to_numeric(joined["execution_net_ev_12h"], errors="raise"),
            "gross_return": pd.to_numeric(joined["execution_gross_ev_12h"], errors="raise"),
            "holding_bars": exit_hour * 4.0,
            "simple_policy_exit_reason": joined["execution_exit_reason"].astype(str),
            # Exact cost is already included in net_return.  Keep it as audit
            # metadata, while zeroing the *expected* friction path so replay
            # does not apply it to priority a second time.
            "fees_bps": pd.to_numeric(joined["execution_cost_return"], errors="raise") * 10_000.0,
            "price_gap_bps": 0.0,
            "expected_friction_bps": 0.0,
            "candidate_id": joined["candidate_id"].astype(str),
            "score_source_utc": joined["__ts__"],
            "candidate_score_is_oof": True,
            "candidate_score_is_forward_oos": False,
            "active_transition_probability_grouped_oof": probability,
            "active_head_validation": "grouped_oof_non_chronological_research_only",
            "expost__transition_active": joined["expost__transition_active"].astype(np.int8),
            "target__event_id": joined["target__event_id"],
            "exact_policy_label_end_utc": label_end,
            "exact_policy_cost_return": pd.to_numeric(joined["execution_cost_return"], errors="raise"),
            "exact_policy_exit_hour": exit_hour,
        }
    )
    replay = normalise_candidate_table(replay)
    active_rows = replay.loc[replay["expost__transition_active"].astype(bool)].copy()
    top_k_count = int(np.ceil(0.10 * len(replay)))
    baseline_ids = set(
        replay.sort_values(
            ["calibrated_score", "timestamp", "symbol", "side", "candidate_id"],
            ascending=[False, True, True, True, True],
            kind="stable",
        )
        .head(top_k_count)["candidate_id"]
        .astype(str)
    )
    active_events = active_rows["target__event_id"].dropna().astype(str).nunique()
    gate = {
        "schema": "active_transition_exact_policy_evidence_gate_v1",
        "cohort_contract": {
            "identity": list(IDENTITY),
            "candidate_score": "causal_global_21d_recent_ev_isotonic_mapping; explicit candidate OOF only",
            "execution": "exact 1m policy labels; actual entry/exit price, cost, exit reason and exit timestamp",
            "active_score": "grouped OOF hourly active-transition probability; non-chronological research validation",
            "selection": "one pooled global top 10% score rank across sides and timestamps; no timestamp quota",
            "portfolio_replay": "normalized shared replay schema with true entry/exit path fields",
        },
        "coverage": {
            "candidate_rows": int(len(replay)),
            "candidate_hours": int(replay["timestamp"].nunique()),
            "min_score_source_utc": replay["score_source_utc"].min(),
            "max_score_source_utc": replay["score_source_utc"].max(),
            "exact_entry_price_coverage": float(replay["entry_price"].notna().mean()),
            "exact_exit_price_coverage": float(replay["exit_price"].notna().mean()),
            "exact_exit_timestamp_coverage": float(replay["exit_timestamp"].notna().mean()),
            "exact_exit_reason_coverage": float(replay["simple_policy_exit_reason"].notna().mean()),
            "active_candidate_rows": int(len(active_rows)),
            "active_hours": int(active_rows["timestamp"].nunique()),
            "active_transition_events": int(active_events),
            "active_rows_in_frozen_global_top10": int(
                active_rows["candidate_id"].astype(str).isin(baseline_ids).sum()
            ),
            "active_events_in_frozen_global_top10": int(
                active_rows.loc[
                    active_rows["candidate_id"].astype(str).isin(baseline_ids),
                    "target__event_id",
                ]
                .dropna()
                .astype(str)
                .nunique()
            ),
        },
        "validity": {
            "exact_execution_lineage": True,
            "causal_global_recent_ev_score": True,
            "candidate_score_strict_oof": True,
            "active_probability_research_oof": True,
            "active_probability_chronological_oos": False,
            "portfolio_replay_schema_valid": True,
            "policy_sweep_mechanically_valid": True,
            "policy_sweep_economic_effectiveness_informative": bool(
                active_events >= 2
                and int(active_rows["candidate_id"].astype(str).isin(baseline_ids).sum()) > 0
            ),
            "promotion_valid": False,
        },
        "blocking_reasons": [
            "active probability is grouped OOF rather than chronological/causal OOS",
            *(
                [
                    "only one active-transition event overlaps the exact OOF candidate-score window; comparative economic protection is not replicated",
                ]
                if active_events < 2
                else []
            ),
            *(
                [
                    "the frozen global top-10% book contains no active-transition candidate rows, so a baseline protection effect cannot be estimated",
                ]
                if int(active_rows["candidate_id"].astype(str).isin(baseline_ids).sum()) == 0
                else []
            ),
        ],
    }
    return replay, active_rows, gate


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    candidates, active_rows, gate = materialize_cohort(
        pd.read_parquet(args.scores),
        pd.read_parquet(args.exact_labels),
        pd.read_parquet(args.active_oof),
    )
    args.output_dir.mkdir(parents=True)
    candidates.to_parquet(args.output_dir / "cohort_candidates.parquet", index=False)
    active_rows.to_parquet(args.output_dir / "active_transition_candidate_rows.parquet", index=False)
    if not 0.0 < float(args.top_k_fraction) <= 1.0:
        raise ValueError("top-k-fraction must lie in (0, 1]")
    frozen_count = int(np.ceil(float(args.top_k_fraction) * len(candidates)))
    frozen = (
        candidates.sort_values(
            ["calibrated_score", "timestamp", "symbol", "side", "candidate_id"],
            ascending=[False, True, True, True, True],
            kind="stable",
        )
        .head(frozen_count)
        .reset_index(drop=True)
    )
    # This is a compatibility/reproducibility replay only.  It preserves the
    # frozen pooled top-k selection; it does not optimize a policy parameter.
    params = replace(
        load_portfolio_policy_params(args.portfolio_config),
        enforce_position_count_cap=True,
    )
    identity_curve = {
        "schema": "monotone_ev_curve_v1",
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "ev_span": 1.0,
        "n_rows": 0,
    }
    decisions, equity, replay_metrics = replay_candidates(
        frozen,
        params,
        mode="global_auction",
        ev_curve=identity_curve,
        initial_wallet=float(args.initial_wallet),
        market_mode="perps",
    )
    accepted = decisions.loc[decisions["accepted"].astype(bool)]
    accepted_index = pd.to_numeric(accepted["candidate_index"], errors="raise").astype(int)
    accepted_active_rows = int(
        frozen.iloc[accepted_index]["expost__transition_active"].astype(bool).sum()
    )
    frozen.to_parquet(args.output_dir / "frozen_global_top10_candidates.parquet", index=False)
    decisions.to_parquet(args.output_dir / "frozen_global_top10_decisions.parquet", index=False)
    equity.to_parquet(args.output_dir / "frozen_global_top10_equity.parquet", index=False)
    gate["portfolio_replay_validation"] = {
        "executed": True,
        "selection": "frozen one pooled global top-k; no timestamp quota",
        "top_k_fraction": float(args.top_k_fraction),
        "selected_rows": int(len(frozen)),
        "accepted_rows": int(len(accepted)),
        "active_selected_rows": int(frozen["expost__transition_active"].astype(bool).sum()),
        "active_accepted_rows": accepted_active_rows,
        "constraints": {
            "count_cap_explicitly_enabled": True,
            "max_concurrent_positions": int(params.max_concurrent_positions),
            "max_concurrent_per_symbol": int(params.max_concurrent_per_symbol),
            "max_new_entries_per_bar": int(params.max_new_entries_per_bar),
            "max_total_wallet_allocation_pct": float(params.max_total_wallet_allocation_pct),
        },
        "metrics": {
            key: value
            for key, value in replay_metrics.items()
            if isinstance(value, (str, int, float, bool, np.generic))
        },
    }
    gate["validity"]["constrained_portfolio_replay_executed"] = True
    _write_json(args.output_dir / "evidence_gate.json", gate)
    manifest = {
        "schema": "active_transition_exact_policy_cohort_v1",
        "research_only": True,
        "sources": {
            "scores": {"path": str(args.scores), "sha256": _sha256(args.scores)},
            "exact_labels": {
                "path": str(args.exact_labels),
                "sha256": _sha256(args.exact_labels),
            },
            "active_oof": {"path": str(args.active_oof), "sha256": _sha256(args.active_oof)},
            "portfolio_config": {
                "path": str(args.portfolio_config),
                "sha256": _sha256(args.portfolio_config),
            },
        },
        "outputs": {
            "cohort_candidates": {
                "path": str(args.output_dir / "cohort_candidates.parquet"),
                "sha256": _sha256(args.output_dir / "cohort_candidates.parquet"),
            },
            "active_transition_candidate_rows": {
                "path": str(args.output_dir / "active_transition_candidate_rows.parquet"),
                "sha256": _sha256(args.output_dir / "active_transition_candidate_rows.parquet"),
            },
            "frozen_global_top10_candidates": {
                "path": str(args.output_dir / "frozen_global_top10_candidates.parquet"),
                "sha256": _sha256(args.output_dir / "frozen_global_top10_candidates.parquet"),
            },
        },
        "evidence_gate": gate,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--exact-labels", type=Path, default=DEFAULT_EXACT_LABELS)
    parser.add_argument("--active-oof", type=Path, default=DEFAULT_ACTIVE_OOF)
    parser.add_argument("--portfolio-config", type=Path, default=DEFAULT_PORTFOLIO_CONFIG)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    manifest = run(_parser().parse_args())
    print(json.dumps(_safe(manifest["evidence_gate"]["coverage"]), indent=2))


if __name__ == "__main__":
    main()
