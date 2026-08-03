#!/usr/bin/env python3
"""Research-only active-transition policy sweep on canonical exact economics.

The candidate score is frozen/OOF and the active-transition probability is
grouped OOF.  The latter is not chronological policy-OOS, so this runner may
support mechanism research but never promotion.

For each declared score stream, the runner freezes one pooled global top-k book
across both sides and all timestamps, then compares:

* baseline;
* sign-safe trust discount with same-count global reselection;
* additive risk premium with same-count global reselection;
* risk-dependent threshold increase within the frozen baseline book; and
* exposure reduction on the frozen baseline book.

Every selected book is passed through the same concurrency, wallet-allocation,
new-entry and per-symbol portfolio constraints.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ARES_ROOT = Path(
    os.environ.get("ARES_ROOT", "/Users/remyroche/Documents/Ares")
).resolve()
if str(ARES_ROOT) not in sys.path:
    sys.path.insert(0, str(ARES_ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    load_portfolio_policy_params,
    normalise_candidate_table,
    replay_candidates,
)

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _active_validation_metadata(
    contract: str, policy_selection_contract: str = "same_cohort_grid"
) -> dict[str, Any]:
    if policy_selection_contract == "same_cohort_grid":
        policy_blocker = "lambda grid is evaluated on the same cohort"
    elif policy_selection_contract == "prior_frozen":
        policy_blocker = (
            "policy and lambda are declared frozen from a prior cohort"
        )
    else:
        raise ValueError(
            f"unsupported policy selection contract {policy_selection_contract!r}"
        )
    if contract == "grouped_oof":
        return {
            "status": "RESEARCH_ONLY_GROUPED_OOF_POLICY_SWEEP_COMPLETE",
            "blocker": (
                "active-transition probability is grouped OOF, not "
                f"chronological policy-OOS; {policy_blocker}"
            ),
            "model_score_contract": "grouped OOF; non-chronological",
        }
    if contract == "chronological_label_oos_pooled_geometry":
        return {
            "status": (
                "RESEARCH_ONLY_CHRONOLOGICAL_LABEL_OOS_POLICY_SWEEP_COMPLETE"
            ),
            "blocker": (
                "active-head model folds are chronological and label-purged, "
                f"but upstream state geometry is pooled; {policy_blocker}"
            ),
            "model_score_contract": (
                "expanding-month chronological label OOS; pooled upstream "
                "state geometry"
            ),
        }
    raise ValueError(f"unsupported active validation contract {contract!r}")


def _floats(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in str(text).split(",") if value.strip())
    if not values or any(value < 0.0 for value in values):
        raise ValueError("lambda grid must contain nonnegative values")
    return values


def _stable_top_k(
    frame: pd.DataFrame,
    *,
    score_column: str,
    count: int | None = None,
    fraction: float | None = None,
) -> pd.DataFrame:
    if (count is None) == (fraction is None):
        raise ValueError("provide exactly one of count or fraction")
    score = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(score).all():
        raise ValueError(f"{score_column} contains non-finite values")
    if frame["candidate_id"].astype(str).duplicated().any():
        raise ValueError("candidate_id must be unique for global tie-breaking")
    if count is None:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("fraction must be in (0,1]")
        count = max(1, int(math.ceil(float(fraction) * len(frame))))
    count = min(int(count), len(frame))
    order = np.lexsort((frame["candidate_id"].astype(str).to_numpy(), -score))
    return frame.iloc[order[:count]].copy()


def _rank_pct(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="raise").rank(method="max", pct=True)


def attach_active_scores(
    candidates: pd.DataFrame,
    active_oof: pd.DataFrame,
) -> pd.DataFrame:
    required_candidates = {
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_exit_minute",
        "execution_exit_class",
        "mapped_eligible",
    }
    missing = sorted(required_candidates.difference(candidates.columns))
    if missing:
        raise ValueError(f"canonical candidates miss fields: {missing}")
    required_active = {
        "source_utc",
        "prediction",
        "target__transition_active",
        "target__event_id",
    }
    missing_active = sorted(required_active.difference(active_oof.columns))
    if missing_active:
        raise ValueError(f"active OOF misses fields: {missing_active}")
    work = candidates.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    active = active_oof.copy()
    active["source_utc"] = pd.to_datetime(active["source_utc"], utc=True, errors="raise")
    if active["source_utc"].duplicated().any():
        raise ValueError("active OOF must have exactly one row per source hour")
    active = active.rename(
        columns={
            "prediction": "active_transition_probability_oof",
            "target__transition_active": "expost_transition_active",
            "target__event_id": "transition_event_id",
        }
    )
    work = work.merge(
        active[
            [
                "source_utc",
                "active_transition_probability_oof",
                "expost_transition_active",
                "transition_event_id",
            ]
        ],
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    )
    if work["active_transition_probability_oof"].isna().any():
        raise ValueError("canonical cohort lacks complete active-score coverage")
    probability = pd.to_numeric(
        work["active_transition_probability_oof"], errors="raise"
    )
    if probability.lt(0.0).any() or probability.gt(1.0).any():
        raise ValueError("active-transition probability is outside [0,1]")
    gross = pd.to_numeric(work["execution_gross_ev_12h"], errors="raise")
    cost = pd.to_numeric(work["execution_cost_return"], errors="raise")
    net = pd.to_numeric(work["execution_net_ev_12h"], errors="raise")
    if not np.allclose(gross - cost, net, rtol=0.0, atol=1e-7):
        raise ValueError("canonical gross-cost-net reconciliation failed")
    exits = work["execution_exit_class"].astype(str)
    if not exits.isin(("trailing", "timeout", "full_stop", "adverse_exit")).all():
        raise ValueError("canonical exit class is invalid")
    return work


def _robust_score_scale(frame: pd.DataFrame, score_column: str) -> float:
    values = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    return max(scale, 1e-9)


def select_arm(
    eligible: pd.DataFrame,
    *,
    score_column: str,
    baseline_ids: set[str],
    baseline_count: int,
    score_scale: float,
    policy: str,
    value: float,
) -> pd.DataFrame:
    work = eligible.copy()
    score = pd.to_numeric(work[score_column], errors="raise")
    risk = pd.to_numeric(
        work["active_transition_probability_oof"], errors="raise"
    ).clip(0.0, 1.0)
    baseline = work["candidate_id"].astype(str).isin(baseline_ids)
    work["policy_score"] = score
    work["portfolio_size_multiplier"] = 1.0
    if policy == "baseline":
        selected = work.loc[baseline].copy()
    elif policy == "trust_discount":
        # A plain score*(1-lambda*p) promotes negative EV scores toward zero.
        # Subtracting lambda*p*max(|score|, scale) is sign-safe and is identical
        # to the requested multiplicative discount for positive scores when
        # |score| dominates the robust scale.
        work["policy_score"] = score - (
            float(value)
            * risk
            * np.maximum(np.abs(score.to_numpy(float)), float(score_scale))
        )
        selected = _stable_top_k(
            work, score_column="policy_score", count=baseline_count
        )
    elif policy == "risk_premium":
        work["policy_score"] = score - float(value) * float(score_scale) * risk
        selected = _stable_top_k(
            work, score_column="policy_score", count=baseline_count
        )
    elif policy == "threshold_increase":
        cutoff = float(score.loc[baseline].min())
        required = cutoff + float(value) * float(score_scale) * risk
        selected = work.loc[baseline & score.ge(required)].copy()
    elif policy == "exposure_reduction":
        selected = work.loc[baseline].copy()
        selected["portfolio_size_multiplier"] = np.clip(
            1.0
            - float(value)
            * selected["active_transition_probability_oof"].to_numpy(float),
            0.0,
            1.0,
        )
    else:
        raise ValueError(f"unknown policy {policy!r}")
    selected["policy"] = policy
    selected["policy_value"] = float(value)
    selected["score_stream"] = score_column
    selected["policy_global_rank_pct"] = _rank_pct(selected["policy_score"])
    return selected.sort_values(
        ["execution_decision_utc", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def to_replay_candidates(selected: pd.DataFrame) -> pd.DataFrame:
    side = selected["side_name"].astype(str).str.lower()
    decision = pd.to_datetime(
        selected["execution_decision_utc"], utc=True, errors="raise"
    )
    exit_minutes = pd.to_numeric(
        selected["execution_exit_minute"], errors="raise"
    ).clip(lower=1.0)
    exit_timestamp = decision + pd.to_timedelta(exit_minutes, unit="m")
    gross = pd.to_numeric(selected["execution_gross_ev_12h"], errors="raise")
    net = pd.to_numeric(selected["execution_net_ev_12h"], errors="raise")
    cost = pd.to_numeric(selected["execution_cost_return"], errors="raise")
    frame = pd.DataFrame(
        {
            "timestamp": decision,
            "symbol": selected["__symbol__"].astype(str),
            "side": side,
            "strategy_id": np.where(
                side.eq("short"),
                "short_active_transition_research",
                "long_active_transition_research",
            ),
            "base_strategy_threshold": 0.90,
            "calibrated_score": pd.to_numeric(
                selected["policy_score"], errors="raise"
            ),
            "normalized_rank_score": pd.to_numeric(
                selected["policy_global_rank_pct"], errors="raise"
            ),
            "entry_price": 1.0,
            "exit_timestamp": exit_timestamp,
            "exit_price": np.maximum(
                np.where(side.eq("short"), 1.0 - gross, 1.0 + gross),
                1e-9,
            ),
            "net_return": net,
            "gross_return": gross,
            "holding_bars": np.maximum(exit_minutes / 15.0, 1.0),
            "simple_policy_exit_reason": selected["execution_exit_class"].astype(str),
            "fees_bps": cost * 10_000.0,
            "price_gap_bps": 0.0,
            # Canonical net already includes the exact realized policy cost.
            # The shared exact-policy replay contract uses zero expected
            # friction so no tie-break or priority path can consume it again.
            "expected_friction_bps": 0.0,
            "candidate_id": selected["candidate_id"].astype(str),
            "portfolio_size_multiplier": pd.to_numeric(
                selected["portfolio_size_multiplier"], errors="raise"
            ),
            "active_transition_probability_oof": pd.to_numeric(
                selected["active_transition_probability_oof"], errors="raise"
            ),
            "expost_transition_active": selected[
                "expost_transition_active"
            ].fillna(0).astype(np.int8),
            "transition_event_id": selected["transition_event_id"],
        }
    )
    return normalise_candidate_table(frame)


def _accepted_with_metadata(
    decisions: pd.DataFrame,
    replay_candidates_frame: pd.DataFrame,
) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return accepted
    candidate_index = pd.to_numeric(
        accepted["candidate_index"], errors="raise"
    ).astype(int)
    metadata = replay_candidates_frame.iloc[candidate_index][
        [
            "candidate_id",
            "active_transition_probability_oof",
            "expost_transition_active",
            "transition_event_id",
        ]
    ].reset_index(drop=True)
    accepted = accepted.reset_index(drop=True)
    for column in metadata:
        accepted[column] = metadata[column]
    accepted["month"] = pd.to_datetime(
        accepted["timestamp"], utc=True
    ).dt.strftime("%Y-%m")
    accepted["week"] = pd.to_datetime(
        accepted["timestamp"], utc=True
    ).dt.to_period("W").astype(str)
    return accepted


def _book_metrics(frame: pd.DataFrame, *, prefix: str) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_mean_net_bps": np.nan,
            f"{prefix}_positive_rate": np.nan,
            f"{prefix}_active_rows": 0,
            f"{prefix}_active_events": 0,
        }
    net = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise")
    active = frame["expost_transition_active"].fillna(0).astype(bool)
    return {
        f"{prefix}_rows": int(len(frame)),
        f"{prefix}_mean_gross_bps": float(
            10_000.0 * frame["execution_gross_ev_12h"].mean()
        ),
        f"{prefix}_mean_cost_bps": float(
            10_000.0 * frame["execution_cost_return"].mean()
        ),
        f"{prefix}_mean_net_bps": float(10_000.0 * net.mean()),
        f"{prefix}_positive_rate": float(net.gt(0.0).mean()),
        f"{prefix}_active_rows": int(active.sum()),
        f"{prefix}_active_hours": int(
            frame.loc[active, "__ts__"].nunique()
        ),
        f"{prefix}_active_events": int(
            frame.loc[active, "transition_event_id"].dropna().nunique()
        ),
        f"{prefix}_active_mean_net_bps": float(
            10_000.0 * net.loc[active].mean()
        ),
        f"{prefix}_outside_mean_net_bps": float(
            10_000.0 * net.loc[~active].mean()
        ),
        f"{prefix}_long_rows": int(frame["side_name"].eq("long").sum()),
        f"{prefix}_short_rows": int(frame["side_name"].eq("short").sum()),
    }


def replacement_attribution(
    cohort: pd.DataFrame,
    *,
    baseline_ids: set[str],
    selected_ids: set[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    relation = pd.DataFrame(
        {
            "candidate_id": sorted(baseline_ids.union(selected_ids)),
        }
    )
    relation["in_baseline"] = relation["candidate_id"].isin(baseline_ids)
    relation["in_challenger"] = relation["candidate_id"].isin(selected_ids)
    relation["selection_relation"] = np.select(
        [
            relation["in_baseline"] & relation["in_challenger"],
            relation["in_baseline"] & ~relation["in_challenger"],
            ~relation["in_baseline"] & relation["in_challenger"],
        ],
        ["kept", "removed", "newly_added"],
        default="invalid",
    )
    economics = cohort[
        [
            "candidate_id",
            "__ts__",
            "__symbol__",
            "side_name",
            "execution_gross_ev_12h",
            "execution_cost_return",
            "execution_net_ev_12h",
            "execution_exit_class",
            "active_transition_probability_oof",
            "expost_transition_active",
            "transition_event_id",
        ]
    ]
    relation = relation.merge(
        economics, on="candidate_id", how="left", validate="one_to_one"
    )
    if relation["execution_net_ev_12h"].isna().any():
        raise ValueError("replacement attribution misses canonical economics")
    metrics: dict[str, Any] = {}
    for name in ("kept", "removed", "newly_added"):
        local = relation.loc[relation["selection_relation"].eq(name)]
        net = pd.to_numeric(local["execution_net_ev_12h"], errors="raise")
        active = local["expost_transition_active"].astype(bool)
        metrics[f"{name}_rows"] = int(len(local))
        metrics[f"{name}_mean_net_bps"] = (
            float(10_000.0 * net.mean()) if len(local) else np.nan
        )
        metrics[f"{name}_sum_net_return"] = float(net.sum()) if len(local) else 0.0
        metrics[f"{name}_positive_rate"] = (
            float(net.gt(0.0).mean()) if len(local) else np.nan
        )
        metrics[f"{name}_active_rows"] = int(active.sum())
        metrics[f"{name}_active_events"] = int(
            local.loc[active, "transition_event_id"].dropna().nunique()
        )
    union = len(baseline_ids.union(selected_ids))
    metrics["selection_jaccard"] = (
        len(baseline_ids.intersection(selected_ids)) / union if union else np.nan
    )
    metrics["replacement_sum_net_delta"] = (
        metrics["newly_added_sum_net_return"] - metrics["removed_sum_net_return"]
    )
    return metrics, relation


def _conditional_accepted_metrics(accepted: pd.DataFrame) -> list[dict[str, Any]]:
    if accepted.empty:
        return []
    active = accepted["expost_transition_active"].astype(bool)
    probability = pd.to_numeric(
        accepted["active_transition_probability_oof"], errors="raise"
    )
    conditions = {
        "all": np.ones(len(accepted), dtype=bool),
        "true_active_transition": active.to_numpy(),
        "outside_true_transition": (~active).to_numpy(),
        "predicted_active_ge_0p5": probability.ge(0.5).to_numpy(),
    }
    rows: list[dict[str, Any]] = []
    for condition, mask in conditions.items():
        local = accepted.loc[mask]
        net = pd.to_numeric(local["position_net_return"], errors="coerce")
        size = pd.to_numeric(local["position_size"], errors="coerce")
        rows.append(
            {
                "condition": condition,
                "trades": int(len(local)),
                "mean_net_return_bps": float(10_000.0 * net.mean())
                if len(local)
                else np.nan,
                "positive_rate": float(net.gt(0.0).mean())
                if len(local)
                else np.nan,
                "net_pnl": float((net * size).sum()) if len(local) else 0.0,
                "unique_transition_events": int(
                    local["transition_event_id"].dropna().nunique()
                ),
            }
        )
    return rows


def _monthly_accepted_metrics(accepted: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month, local in accepted.groupby("month", sort=True):
        net = pd.to_numeric(local["position_net_return"], errors="coerce")
        size = pd.to_numeric(local["position_size"], errors="coerce")
        active = local["expost_transition_active"].astype(bool)
        rows.append(
            {
                "month": str(month),
                "trades": int(len(local)),
                "mean_net_return_bps": float(10_000.0 * net.mean()),
                "positive_rate": float(net.gt(0.0).mean()),
                "net_pnl": float((net * size).sum()),
                "active_trades": int(active.sum()),
                "active_transition_events": int(
                    local.loc[active, "transition_event_id"].dropna().nunique()
                ),
                "active_net_pnl": float((net.loc[active] * size.loc[active]).sum()),
                "outside_net_pnl": float(
                    (net.loc[~active] * size.loc[~active]).sum()
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    candidates_path = Path(args.mapped_candidates)
    active_path = Path(args.active_oof)
    portfolio_path = Path(args.portfolio_config)
    candidates = pd.read_parquet(candidates_path)
    active_oof = pd.read_parquet(active_path)
    cohort = attach_active_scores(candidates, active_oof)
    cohort = cohort.loc[cohort["mapped_eligible"].astype(bool)].copy()
    if args.evaluation_start is not None:
        cohort = cohort.loc[
            cohort["__ts__"].ge(pd.Timestamp(args.evaluation_start, tz="UTC"))
        ].copy()
    if args.evaluation_end is not None:
        cohort = cohort.loc[
            cohort["__ts__"].lt(pd.Timestamp(args.evaluation_end, tz="UTC"))
        ].copy()
    if cohort.empty:
        raise ValueError("evaluation date filters leave no canonical candidates")
    overlap_events = int(cohort["transition_event_id"].dropna().nunique())
    active_events = int(
        cohort.loc[
            cohort["expost_transition_active"].eq(1),
            "transition_event_id",
        ].dropna().nunique()
    )
    if overlap_events < int(args.minimum_overlap_events):
        raise ValueError(
            f"only {overlap_events} transition events overlap the canonical cohort"
        )
    if active_events < int(args.minimum_active_events):
        raise ValueError(
            f"only {active_events} active events overlap the canonical cohort"
        )
    params = replace(
        load_portfolio_policy_params(portfolio_path),
        enforce_position_count_cap=True,
    )
    specifications = [("baseline", 0.0)]
    if args.policy_selection_contract == "prior_frozen":
        if args.frozen_policy is None or args.frozen_value is None:
            raise ValueError("prior_frozen requires --frozen-policy and --frozen-value")
        specifications.append((args.frozen_policy, float(args.frozen_value)))
    else:
        lambdas = _floats(args.lambdas)
        for policy in (
            "trust_discount",
            "risk_premium",
            "threshold_increase",
            "exposure_reduction",
        ):
            specifications.extend((policy, value) for value in lambdas)
    output.mkdir(parents=True, exist_ok=False)
    summary_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    baseline_accepted_by_score: dict[str, pd.DataFrame] = {}
    pending_missed: list[tuple[int, str, pd.DataFrame]] = []
    for score_column in args.score_columns:
        if score_column not in cohort:
            raise ValueError(f"canonical cohort lacks score {score_column!r}")
        score_dir = output / score_column
        score_dir.mkdir()
        baseline = _stable_top_k(
            cohort,
            score_column=score_column,
            fraction=float(args.top_k_fraction),
        )
        baseline_ids = set(baseline["candidate_id"].astype(str))
        baseline_count = int(len(baseline))
        baseline_active_events = int(
            baseline.loc[
                baseline["expost_transition_active"].eq(1),
                "transition_event_id",
            ].dropna().nunique()
        )
        if baseline_active_events < int(args.minimum_selected_active_events):
            raise ValueError(
                f"{score_column} baseline has only {baseline_active_events} "
                "selected active events"
            )
        scale = _robust_score_scale(cohort, score_column)
        for policy, value in specifications:
            arm = f"{policy}_{value:.4f}".replace(".", "p")
            selected = select_arm(
                cohort,
                score_column=score_column,
                baseline_ids=baseline_ids,
                baseline_count=baseline_count,
                score_scale=scale,
                policy=policy,
                value=value,
            )
            selected_ids = set(selected["candidate_id"].astype(str))
            replacement_metrics, replacement = replacement_attribution(
                cohort,
                baseline_ids=baseline_ids,
                selected_ids=selected_ids,
            )
            relation_by_id = replacement.set_index("candidate_id")[
                "selection_relation"
            ]
            selected["selection_relation"] = selected["candidate_id"].map(
                relation_by_id
            )
            replay_frame = to_replay_candidates(selected)
            decisions, equity, metrics = replay_candidates(
                replay_frame,
                params,
                mode="global_auction",
                ev_curve=IDENTITY_EV_CURVE,
                initial_wallet=float(args.initial_wallet),
                market_mode="perps",
            )
            accepted = _accepted_with_metadata(decisions, replay_frame)
            selected.loc[
                :,
                [
                    "candidate_id",
                    "__ts__",
                    "__symbol__",
                    "side_name",
                    "policy_score",
                    "portfolio_size_multiplier",
                    "active_transition_probability_oof",
                    "expost_transition_active",
                    "transition_event_id",
                    "selection_relation",
                    "execution_gross_ev_12h",
                    "execution_cost_return",
                    "execution_net_ev_12h",
                    "execution_exit_class",
                ],
            ].to_parquet(score_dir / f"{arm}_selected.parquet", index=False)
            replacement.to_parquet(
                score_dir / f"{arm}_replacement_attribution.parquet",
                index=False,
            )
            accepted.to_parquet(score_dir / f"{arm}_accepted.parquet", index=False)
            equity.to_parquet(score_dir / f"{arm}_equity.parquet", index=False)
            row = {
                "score_stream": score_column,
                "score_scale": scale,
                "arm": arm,
                "policy": policy,
                "value": float(value),
                **_book_metrics(selected, prefix="selected"),
                **replacement_metrics,
                **{
                    key: metric
                    for key, metric in metrics.items()
                    if isinstance(metric, (str, int, float, bool, np.generic))
                },
            }
            summary_rows.append(row)
            summary_index = len(summary_rows) - 1
            if policy == "baseline":
                baseline_accepted_by_score[score_column] = accepted
            else:
                pending_missed.append((summary_index, score_column, accepted))
            for conditional in _conditional_accepted_metrics(accepted):
                conditional_rows.append(
                    {
                        "score_stream": score_column,
                        "arm": arm,
                        "policy": policy,
                        "value": float(value),
                        **conditional,
                    }
                )
            for monthly in _monthly_accepted_metrics(accepted):
                monthly_rows.append(
                    {
                        "score_stream": score_column,
                        "arm": arm,
                        "policy": policy,
                        "value": float(value),
                        **monthly,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    for summary_index, score_column, accepted in pending_missed:
        baseline_accepted = baseline_accepted_by_score[score_column]
        baseline_by_id = baseline_accepted.set_index("candidate_id")
        missed_ids = set(baseline_by_id.index.astype(str)).difference(
            accepted["candidate_id"].astype(str)
        )
        missed_returns = pd.to_numeric(
            baseline_by_id.reindex(list(missed_ids))["position_net_return"],
            errors="coerce",
        )
        summary.loc[summary_index, "missed_baseline_accepted_trades"] = len(
            missed_ids
        )
        summary.loc[summary_index, "missed_profitable_trades"] = int(
            missed_returns.gt(0.0).sum()
        )
        summary.loc[summary_index, "missed_profitable_return_sum"] = float(
            missed_returns.loc[missed_returns.gt(0.0)].sum()
        )
    for score_column in args.score_columns:
        baseline_row = summary.loc[
            summary["score_stream"].eq(score_column)
            & summary["policy"].eq("baseline")
        ].iloc[0]
        mask = summary["score_stream"].eq(score_column)
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
                summary.loc[mask, f"delta_{metric}"] = (
                    pd.to_numeric(summary.loc[mask, metric], errors="coerce")
                    - float(baseline_row[metric])
                )
    summary_path = output / "policy_summary.csv"
    conditional_path = output / "conditional_economics.csv"
    monthly_path = output / "monthly_economics.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(conditional_rows).to_csv(conditional_path, index=False)
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
    catalog_rows: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            catalog_rows.append(
                {
                    "relative_path": str(path.relative_to(output)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    catalog_path = output / "output_catalog.csv"
    pd.DataFrame(catalog_rows).to_csv(catalog_path, index=False)
    active_validation = _active_validation_metadata(
        args.active_validation_contract, args.policy_selection_contract
    )
    manifest = {
        "schema": "active_transition_canonical_exact_policy_sweep_v2",
        "status": active_validation["status"],
        "promotion_eligible": False,
        "promotion_blocker": active_validation["blocker"],
        "active_validation_contract": {
            "name": args.active_validation_contract,
            "model_score_contract": active_validation["model_score_contract"],
        },
        "policy_selection_contract": {
            "name": args.policy_selection_contract,
            "frozen_policy": args.frozen_policy,
            "frozen_value": args.frozen_value,
        },
        "evaluation_window": {
            "start_inclusive": args.evaluation_start,
            "end_exclusive": args.evaluation_end,
        },
        "cohort": {
            "rows": int(len(cohort)),
            "hours": int(cohort["__ts__"].nunique()),
            "transition_events": overlap_events,
            "active_transition_events": active_events,
            "active_hours": int(
                cohort.loc[cohort["expost_transition_active"].eq(1), "__ts__"].nunique()
            ),
            "active_rows": int(cohort["expost_transition_active"].sum()),
        },
        "score_streams": list(args.score_columns),
        "policies": [
            {"policy": policy, "value": value}
            for policy, value in specifications
        ],
        "policy_contract": {
            "selection": "one pooled global top-k; never per timestamp or side",
            "top_k_fraction": float(args.top_k_fraction),
            "trust_discount": (
                "score - lambda*p_active*max(abs(score), robust_score_IQR); "
                "same-count global reselection"
            ),
            "risk_premium": (
                "score - lambda*robust_score_IQR*p_active; same-count global reselection"
            ),
            "threshold_increase": (
                "frozen baseline rows only; score >= baseline_cutoff + "
                "lambda*robust_score_IQR*p_active"
            ),
            "exposure_reduction": (
                "frozen baseline; size_multiplier=1-lambda*p_active"
            ),
        },
        "execution_contract": {
            "entry": "normalized price 1 at exact execution_decision_utc",
            "exit": "decision + canonical execution_exit_minute",
            "gross": "canonical exact-policy gross return",
            "cost": "canonical execution_cost_return exactly once",
            "net": "canonical exact-policy net return",
            "exit_class": "canonical trailing/timeout/full_stop/adverse_exit",
            "expected_friction_bps": 0.0,
            "mtm_limitation": (
                "no exact intratrade path in this ledger; MTM equity and "
                "drawdown use shared replay interpolation"
            ),
        },
        "portfolio_contract": {
            "configuration": str(portfolio_path),
            "count_cap_explicitly_enabled": True,
            "max_concurrent_positions": params.max_concurrent_positions,
            "max_concurrent_per_symbol": params.max_concurrent_per_symbol,
            "max_new_entries_per_bar": params.max_new_entries_per_bar,
            "max_total_wallet_allocation_pct": params.max_total_wallet_allocation_pct,
        },
        "sources": {
            "mapped_candidates": {
                "path": str(candidates_path),
                "sha256": _sha256(candidates_path),
            },
            "active_oof": {
                "path": str(active_path),
                "sha256": _sha256(active_path),
            },
            "portfolio_config": {
                "path": str(portfolio_path),
                "sha256": _sha256(portfolio_path),
            },
        },
        "outputs": {
            "policy_summary": {
                "path": str(summary_path),
                "sha256": _sha256(summary_path),
            },
            "conditional_economics": {
                "path": str(conditional_path),
                "sha256": _sha256(conditional_path),
            },
            "monthly_economics": {
                "path": str(monthly_path),
                "sha256": _sha256(monthly_path),
            },
            "output_catalog": {
                "path": str(catalog_path),
                "sha256": _sha256(catalog_path),
            },
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapped-candidates",
        type=Path,
        default=ARES_ROOT
        / (
            "data_perp/artifacts/historical_causal_score_economics_mapping_20260729_v1/"
            "canonical_base__score_base_alpha/causal_mapped_candidates.parquet"
        ),
    )
    parser.add_argument(
        "--active-oof",
        type=Path,
        default=ARES_ROOT
        / "data_perp/artifacts/regime_transition_active_head_20260726_v1/grouped_oof.parquet",
    )
    parser.add_argument(
        "--active-validation-contract",
        choices=("grouped_oof", "chronological_label_oos_pooled_geometry"),
        default="grouped_oof",
    )
    parser.add_argument(
        "--portfolio-config",
        type=Path,
        default=ARES_ROOT
        / (
            "data_perp/artifacts/"
            "s59_s52_finalfit_meta_repairedcoverage_v9tail95_mlp_hierev_20260715_v3/"
            "policy_params/optimized_portfolio_policy_config.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-columns",
        nargs="+",
        default=("score_raw", "mapped_direct_net"),
    )
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--lambdas", default="0.25,0.50,1.00")
    parser.add_argument(
        "--policy-selection-contract",
        choices=("same_cohort_grid", "prior_frozen"),
        default="same_cohort_grid",
    )
    parser.add_argument(
        "--frozen-policy",
        choices=(
            "trust_discount",
            "risk_premium",
            "threshold_increase",
            "exposure_reduction",
        ),
    )
    parser.add_argument("--frozen-value", type=float)
    parser.add_argument("--evaluation-start")
    parser.add_argument("--evaluation-end")
    parser.add_argument("--minimum-overlap-events", type=int, default=5)
    parser.add_argument("--minimum-active-events", type=int, default=5)
    parser.add_argument("--minimum-selected-active-events", type=int, default=3)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    return parser


def main() -> None:
    manifest = run(_parser().parse_args())
    print(json.dumps(_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
