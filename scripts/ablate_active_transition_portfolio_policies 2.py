#!/usr/bin/env python3
"""Research policy sweep for grouped-OOF active-transition probabilities.

The supplied candidate scores and transition probabilities must carry their
own explicit evidence contracts.  Historical and current evidence tiers are
never pooled economically.  Each tier starts from its frozen pooled-global
top-k book and compares:

* trust discount: ``adjusted_ev = mapped_ev * (1 - lambda * p_active)`` and
  reselect the same number of global rows;
* threshold increase: retain frozen-book rows satisfying
  ``mapped_ev >= baseline_cutoff + lambda * p_active``;
* exposure reduction: keep the frozen book and apply
  ``size_multiplier = 1 - lambda * p_active``.

All selected arms are subsequently replayed through the same count, wallet,
new-entry and per-symbol constraints.  A source directory containing an
``ECONOMIC_INVALIDATION.json`` marker is rejected before results are written.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    load_portfolio_policy_params,
    normalise_candidate_table,
    replay_candidates,
)


DEFAULT_ASSIGNMENTS = Path(
    "data_perp/artifacts/exact_history_state_recurrence_20260727_v1/"
    "state_candidate_assignments.parquet"
)
DEFAULT_ACTIVE_OOF = Path(
    "data_perp/artifacts/regime_transition_active_head_20260726_v1/"
    "grouped_oof.parquet"
)
DEFAULT_PORTFOLIO = Path(
    "data_perp/artifacts/"
    "s59_s52_finalfit_meta_repairedcoverage_v9tail95_mlp_hierev_20260715_v3/"
    "policy_params/optimized_portfolio_policy_config.json"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/active_transition_portfolio_policy_ablation_20260727_v1"
)
IDENTITY_EV_CURVE = {
    "schema": "monotone_ev_curve_v1",
    "x": [0.0, 1.0],
    "y": [0.0, 1.0],
    "ev_span": 1.0,
    "n_rows": 0,
}


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
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(text).split(",") if item.strip())


def _arm_id(policy: str, value: float) -> str:
    return f"{policy}_{value:.4f}".replace(".", "p")


def assert_economic_source_is_valid(path: Path) -> None:
    """Refuse inputs whose owning artifact explicitly invalidates economics."""

    marker = path.parent / "ECONOMIC_INVALIDATION.json"
    if not marker.exists():
        return
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot verify economic validity because {marker} is unreadable"
        ) from exc
    status = str(payload.get("economic_status", "invalidated")).strip()
    reason = payload.get("reason", "unspecified")
    raise ValueError(
        f"refusing economically invalidated source {path}: "
        f"status={status}; reason={reason}"
    )


def _rank_pct(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    return numeric.rank(method="max", pct=True).astype(np.float32)


def attach_active_oof(
    assignments: pd.DataFrame,
    active_oof: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one causal hourly OOF transition score to every candidate."""

    work = assignments.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    transition = active_oof.copy()
    transition["source_utc"] = pd.to_datetime(
        transition["source_utc"], utc=True, errors="raise"
    )
    if transition["source_utc"].duplicated().any():
        raise ValueError("active-transition OOF requires one row per source hour")
    transition = transition.rename(
        columns={
            "prediction": "active_transition_probability_oof",
            "target__transition_active": "expost__transition_active",
        }
    )
    columns = [
        "source_utc",
        "active_transition_probability_oof",
        "expost__transition_active",
        "target__event_id",
    ]
    work = work.merge(
        transition[columns],
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    )
    if work["active_transition_probability_oof"].isna().any():
        missing = int(work["active_transition_probability_oof"].isna().sum())
        raise ValueError(f"{missing} candidate rows lack active-transition OOF coverage")
    probability = pd.to_numeric(
        work["active_transition_probability_oof"], errors="raise"
    )
    if probability.lt(0).any() or probability.gt(1).any():
        raise ValueError("active-transition probabilities must lie in [0,1]")
    return work


def select_policy_arm(
    tier: pd.DataFrame,
    *,
    policy: str,
    value: float,
) -> pd.DataFrame:
    """Select one arm without timestamp-local quotas."""

    work = tier.copy()
    score = pd.to_numeric(work["mapped_score"], errors="raise")
    risk = pd.to_numeric(
        work["active_transition_probability_oof"], errors="raise"
    ).clip(0.0, 1.0)
    baseline = work["selected_global_top10"].fillna(False).astype(bool)
    baseline_count = int(baseline.sum())
    if baseline_count < 1:
        raise ValueError("evidence tier has no frozen global-top-k rows")
    work["policy_score"] = score
    work["portfolio_size_multiplier"] = 1.0
    if policy == "baseline":
        selected = baseline
    elif policy == "trust_discount":
        work["policy_score"] = score * (1.0 - float(value) * risk)
        order = work.sort_values(
            ["policy_score", "__ts__", "__symbol__", "side_name", "candidate_id"],
            ascending=[False, True, True, True, True],
            kind="stable",
        ).index[:baseline_count]
        selected = work.index.isin(order)
    elif policy == "threshold_increase":
        baseline_cutoff = float(score.loc[baseline].min())
        selected = baseline & score.ge(baseline_cutoff + float(value) * risk)
    elif policy == "exposure_reduction":
        selected = baseline
        work["portfolio_size_multiplier"] = np.clip(
            1.0 - float(value) * risk, 0.0, 1.0
        )
    else:
        raise ValueError(f"unknown policy {policy!r}")
    selected_frame = work.loc[selected].copy()
    selected_frame["original_global_rank"] = _rank_pct(score).loc[selected_frame.index]
    selected_frame["policy_global_rank"] = _rank_pct(
        work["policy_score"]
    ).loc[selected_frame.index]
    selected_frame["policy"] = policy
    selected_frame["policy_value"] = float(value)
    return selected_frame.reset_index(drop=True)


def to_replay_candidates(selected: pd.DataFrame) -> pd.DataFrame:
    """Translate exact-return transition rows into the shared replay schema."""

    side = selected["side_name"].astype(str).str.lower()
    net = pd.to_numeric(selected["execution_net_ev_12h"], errors="raise")
    gross = net + 0.01
    decision = pd.to_datetime(
        selected["execution_decision_utc"], utc=True, errors="raise"
    )
    exit_timestamp = pd.to_datetime(
        selected["execution_label_end_utc"], utc=True, errors="raise"
    )
    holding_hours = (exit_timestamp - decision) / pd.Timedelta(hours=1)
    frame = pd.DataFrame(
        {
            "timestamp": decision,
            "symbol": selected["__symbol__"].astype(str),
            "side": side,
            "strategy_id": np.where(
                side.eq("short"),
                "short_s52_meta_threshold_handoff",
                "long_s52_meta_threshold_handoff",
            ),
            "base_strategy_threshold": 0.90,
            "calibrated_score": pd.to_numeric(
                selected["policy_score"], errors="raise"
            ),
            "normalized_rank_score": pd.to_numeric(
                selected["policy_global_rank"], errors="raise"
            ),
            "entry_price": 1.0,
            "exit_timestamp": exit_timestamp,
            "exit_price": np.maximum(
                np.where(side.eq("short"), 1.0 - gross, 1.0 + gross),
                1e-9,
            ),
            "net_return": net,
            "gross_return": gross,
            "holding_bars": np.maximum(holding_hours * 4.0, 1.0),
            "simple_policy_exit_reason": "exact_12h_policy_label",
            "fees_bps": 100.0,
            "price_gap_bps": 0.0,
            "expected_friction_bps": 0.0,
            "candidate_id": selected["candidate_id"].astype(str),
            "portfolio_size_multiplier": pd.to_numeric(
                selected["portfolio_size_multiplier"], errors="raise"
            ),
            "active_transition_probability_oof": pd.to_numeric(
                selected["active_transition_probability_oof"], errors="raise"
            ),
            "expost__transition_active": selected[
                "expost__transition_active"
            ].fillna(0).astype(np.int8),
            "target__event_id": selected["target__event_id"],
        }
    )
    return normalise_candidate_table(frame)


def _accepted_metadata(
    decisions: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return accepted
    index = pd.to_numeric(
        accepted["candidate_index"], errors="raise"
    ).astype(int)
    metadata = candidates.iloc[index][
        [
            "candidate_id",
            "active_transition_probability_oof",
            "expost__transition_active",
            "target__event_id",
        ]
    ].reset_index(drop=True)
    accepted = accepted.reset_index(drop=True)
    for name in metadata:
        accepted[name] = metadata[name]
    accepted["month"] = pd.to_datetime(
        accepted["timestamp"], utc=True
    ).dt.strftime("%Y-%m")
    accepted["week"] = pd.to_datetime(
        accepted["timestamp"], utc=True
    ).dt.to_period("W").astype(str)
    return accepted


def _conditional_metrics(accepted: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if accepted.empty:
        return rows
    conditions = {
        "all": np.ones(len(accepted), dtype=bool),
        "true_active_transition": accepted[
            "expost__transition_active"
        ].astype(bool).to_numpy(),
        "outside_true_transition": ~accepted[
            "expost__transition_active"
        ].astype(bool).to_numpy(),
        "predicted_active_ge_0p5": pd.to_numeric(
            accepted["active_transition_probability_oof"], errors="coerce"
        )
        .ge(0.5)
        .to_numpy(),
    }
    for condition, mask in conditions.items():
        local = accepted.loc[mask]
        net = pd.to_numeric(local["position_net_return"], errors="coerce")
        size = pd.to_numeric(local["position_size"], errors="coerce")
        rows.append(
            {
                "condition": condition,
                "trades": int(len(local)),
                "mean_net_return": float(net.mean()) if len(local) else np.nan,
                "positive_rate": float(net.gt(0).mean()) if len(local) else np.nan,
                "net_pnl": float((net * size).sum()) if len(local) else 0.0,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    assert_economic_source_is_valid(Path(args.assignments))
    output.mkdir(parents=True)
    assignments = pd.read_parquet(args.assignments)
    active_oof = pd.read_parquet(args.active_oof)
    joined = attach_active_oof(assignments, active_oof)
    params = load_portfolio_policy_params(args.portfolio_config)
    # The roadmap explicitly requires concurrency, exposure and asset limits.
    params = replace(params, enforce_position_count_cap=True)
    specifications: list[tuple[str, float]] = [("baseline", 0.0)]
    specifications += [
        ("trust_discount", value) for value in _floats(args.trust_lambdas)
    ]
    specifications += [
        ("threshold_increase", value)
        for value in _floats(args.threshold_lambdas)
    ]
    specifications += [
        ("exposure_reduction", value)
        for value in _floats(args.exposure_lambdas)
    ]
    summary_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    accepted_by_tier_arm: dict[tuple[str, str], pd.DataFrame] = {}
    selected_by_tier_arm: dict[tuple[str, str], pd.DataFrame] = {}
    for tier_name, tier in joined.groupby("evidence_tier", sort=True):
        tier_dir = output / str(tier_name)
        tier_dir.mkdir()
        for policy, value in specifications:
            arm = _arm_id(policy, value)
            selected = select_policy_arm(tier, policy=policy, value=value)
            candidates = to_replay_candidates(selected)
            decisions, equity, metrics = replay_candidates(
                candidates,
                params,
                mode="global_auction",
                ev_curve=IDENTITY_EV_CURVE,
                initial_wallet=float(args.initial_wallet),
                market_mode="perps",
            )
            accepted = _accepted_metadata(decisions, candidates)
            selected_by_tier_arm[(str(tier_name), arm)] = selected
            accepted_by_tier_arm[(str(tier_name), arm)] = accepted
            accepted.to_parquet(tier_dir / f"{arm}_accepted.parquet", index=False)
            equity.to_parquet(tier_dir / f"{arm}_equity.parquet", index=False)
            summary_rows.append(
                {
                    "evidence_tier": str(tier_name),
                    "arm": arm,
                    "policy": policy,
                    "value": float(value),
                    "source_rows": int(len(tier)),
                    "selected_rows": int(len(selected)),
                    "accepted_trades": int(len(accepted)),
                    **{
                        key: item
                        for key, item in metrics.items()
                        if isinstance(item, (str, int, float, bool, np.generic))
                    },
                }
            )
            for row in _conditional_metrics(accepted):
                conditional_rows.append(
                    {
                        "evidence_tier": str(tier_name),
                        "arm": arm,
                        "policy": policy,
                        "value": float(value),
                        **row,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    for tier_name in summary["evidence_tier"].unique():
        baseline_arm = _arm_id("baseline", 0.0)
        baseline_row = summary.loc[
            summary["evidence_tier"].eq(tier_name)
            & summary["arm"].eq(baseline_arm)
        ].iloc[0]
        baseline_accepted = accepted_by_tier_arm[(tier_name, baseline_arm)]
        baseline_ids = set(baseline_accepted["candidate_id"].astype(str))
        baseline_returns = baseline_accepted.set_index("candidate_id")[
            "position_net_return"
        ]
        mask = summary["evidence_tier"].eq(tier_name)
        for index in summary.index[mask]:
            arm = str(summary.loc[index, "arm"])
            accepted = accepted_by_tier_arm[(tier_name, arm)]
            accepted_ids = set(accepted["candidate_id"].astype(str))
            missed_ids = baseline_ids.difference(accepted_ids)
            missed_returns = pd.to_numeric(
                baseline_returns.reindex(list(missed_ids)), errors="coerce"
            )
            summary.loc[index, "missed_baseline_trades"] = len(missed_ids)
            summary.loc[index, "missed_profitable_trades"] = int(
                missed_returns.gt(0).sum()
            )
            summary.loc[index, "missed_profitable_return_sum"] = float(
                missed_returns.loc[missed_returns.gt(0)].sum()
            )
            for metric in (
                "net_pnl",
                "compounded_return",
                "sortino",
                "max_drawdown",
                "worst_week",
                "notional_turnover",
                "trade_count",
            ):
                if metric in summary:
                    summary.loc[index, f"delta_{metric}"] = (
                        float(summary.loc[index, metric])
                        - float(baseline_row[metric])
                    )
    summary.to_csv(output / "policy_summary.csv", index=False)
    pd.DataFrame(conditional_rows).to_csv(
        output / "conditional_economics.csv", index=False
    )
    manifest = {
        "schema": "active_transition_portfolio_policy_ablation_v1",
        "research_only": True,
        "ranking_contract": (
            "one pooled global book per non-comparable evidence tier; "
            "never per timestamp"
        ),
        "transition_score_contract": (
            "grouped OOF hourly active-transition probability, joined on "
            "source __ts__; no target used by policy"
        ),
        "portfolio_contract": {
            "configuration": str(args.portfolio_config),
            "count_cap_explicitly_enabled": True,
            "max_concurrent_positions": params.max_concurrent_positions,
            "max_concurrent_per_symbol": params.max_concurrent_per_symbol,
            "max_new_entries_per_bar": params.max_new_entries_per_bar,
            "max_total_wallet_allocation_pct": (
                params.max_total_wallet_allocation_pct
            ),
        },
        "sources": {
            "assignments": {
                "path": str(args.assignments),
                "sha256": _sha256(args.assignments),
            },
            "active_oof": {
                "path": str(args.active_oof),
                "sha256": _sha256(args.active_oof),
            },
            "portfolio_config": {
                "path": str(args.portfolio_config),
                "sha256": _sha256(args.portfolio_config),
            },
        },
        "arms": [
            {"policy": policy, "value": value, "arm": _arm_id(policy, value)}
            for policy, value in specifications
        ],
        "evidence_tiers": sorted(summary["evidence_tier"].unique().tolist()),
    }
    (output / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--active-oof", type=Path, default=DEFAULT_ACTIVE_OOF)
    parser.add_argument("--portfolio-config", type=Path, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trust-lambdas", default="0.25,0.50,0.75,1.00")
    parser.add_argument(
        "--threshold-lambdas", default="0.0005,0.0010,0.0025,0.0050,0.0100"
    )
    parser.add_argument("--exposure-lambdas", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    return parser


def main() -> None:
    manifest = run(_parser().parse_args())
    print(json.dumps(_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
