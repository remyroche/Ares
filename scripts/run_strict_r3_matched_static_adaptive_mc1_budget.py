#!/usr/bin/env python3
"""Matched static-versus-adaptive dual-MC1 evaluation at fixed admission budget.

This is deliberately a *selection* experiment rather than a generic mapper
regression benchmark.  It preserves the same target-free BCF and Current
scores, common policy labels, and global portfolio constraints.  For each
timestamp, every arm receives exactly the same number of admissions; only the
MC1-derived gate/priority ordering can change which candidates occupy them.

The capacity schedules are intentionally anchored before looking at outcomes:

* ``static_ge50``: number of candidates passing both static maps at +50 bps;
* ``adaptive_ge50``: number passing both unshrunk 21-day maps at +50 bps.

Each arm is evaluated under both schedules.  This avoids treating a larger
admitted population as mapper uplift.  Outcomes are attached only after the
target-free ranking/selection has been persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _params,
)


SCHEMA = "strict_r3_matched_static_adaptive_mc1_budget_v1"
DEFAULT_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_"
    "sixmonth_aug25_aug26_20260828_v4"
)
DEFAULT_HISTORY_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_"
    "nov25_jul26_fullprehistory_20260828_v1"
)
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_matched_static_adaptive_mc1_"
    "budget_20260828_v2"
)
START = pd.Timestamp("2026-02-01", tz="UTC")
END = pd.Timestamp("2026-08-01", tz="UTC")
THRESHOLD_BPS = 50.0
WINDOW = pd.Timedelta(days=21)
TRIM = 0.10
EPS = 1e-12


def _once_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _robust_mean(values: Iterable[float]) -> float:
    data = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    if not len(data):
        return float("nan")
    count = int(math.floor(TRIM * len(data)))
    if count and len(data) > 2 * count:
        data = data[count:-count]
    return float(data.mean())


def _load_family(path: Path, family: str) -> pd.DataFrame:
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "enhanced_base_routed", "final_score", "static_expected_bps",
        "score_band_curve_bps", "recent_shift_bps", "mc1_expected_bps",
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    }
    frame = pd.read_parquet(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{family} prediction ledger lacks: {missing}")
    frame = frame.loc[:, sorted(required)].copy()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{family} has duplicate candidate IDs")
    return frame.rename(columns={
        "final_score": f"{family}_final_score",
        "static_expected_bps": f"{family}_static_bps",
        "score_band_curve_bps": f"{family}_band_curve_bps",
        "recent_shift_bps": f"{family}_shift_bps",
        "mc1_expected_bps": f"{family}_adaptive_bps",
    })


def _policy_equal(left: pd.Series, right: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
        a = pd.to_numeric(left, errors="coerce").to_numpy(float)
        b = pd.to_numeric(right, errors="coerce").to_numpy(float)
        return bool(np.allclose(a, b, equal_nan=True, atol=1e-10, rtol=0.0))
    return bool(left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)))


def _load_matched(root: Path) -> pd.DataFrame:
    current = _load_family(root / "enhanced_current_mc1_predictions.parquet", "current")
    bcf = _load_family(root / "enhanced_bcf_mc1_predictions.parquet", "bcf")
    shared = [
        "__symbol__", "side_name", "enhanced_base_routed", "policy_path_valid",
        "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason",
        "policy_label_available_ts", "policy_cost_bps",
    ]
    left = current.rename(columns={field: f"{field}__current" for field in shared})
    right = bcf.rename(columns={field: f"{field}__bcf" for field in shared})
    frame = left.merge(right, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    if len(frame) != len(current) or len(frame) != len(bcf):
        raise AssertionError("BCF and Current ledgers do not have identical candidate identities")
    for field in shared:
        if not _policy_equal(frame[f"{field}__current"], frame[f"{field}__bcf"]):
            raise AssertionError(f"BCF/Current mismatch in shared field {field}")
        frame[field] = frame.pop(f"{field}__current")
        frame.pop(f"{field}__bcf")
    if not frame["side_name"].eq("long").all():
        raise AssertionError("matched panel is expected to be long-only")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _score_bands(frame: pd.DataFrame, score_field: str) -> np.ndarray:
    """Timestamp-local deciles under the frozen MC1 shift contract."""
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score_field]].copy()
    work["__position__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", score_field, "candidate_id"],
        ascending=[True, False, True], kind="stable", na_position="last",
    )
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["score_band"] = np.minimum(9, ((rank - .5) / count * 10.0).astype(np.int8))
    return work.sort_values("__position__", kind="stable")["score_band"].to_numpy(np.int8)


def _load_history(root: Path) -> pd.DataFrame:
    """Load only the pre-score inputs required to reproduce shift uncertainty.

    This is deliberately separate from the evaluation ledger: the historical
    panel supplies prior resolved residuals, while the selected evaluation
    ledger remains the source of the fixed static/adaptive MC1 values.
    """
    shared = [
        "__symbol__", "side_name", "enhanced_base_routed", "policy_path_valid",
        "policy_net_bps", "policy_label_available_ts",
    ]
    required = ["candidate_id", "__decision_ts__", "final_score", *shared]
    pieces: dict[str, pd.DataFrame] = {}
    for family in ("bcf", "current"):
        path = root / f"enhanced_{family}_mc1_predictions.parquet"
        raw = pd.read_parquet(path, columns=required)
        if raw["candidate_id"].duplicated().any():
            raise AssertionError(f"{family} history has duplicate candidate IDs")
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
        raw["policy_label_available_ts"] = pd.to_datetime(
            raw["policy_label_available_ts"], utc=True, errors="coerce",
        )
        pieces[family] = raw.rename(columns={
            "final_score": f"{family}_final_score",
            **{field: f"{field}__{family}" for field in shared},
        })
    frame = pieces["current"].merge(
        pieces["bcf"], on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one",
    )
    if len(frame) != len(pieces["current"]) or len(frame) != len(pieces["bcf"]):
        raise AssertionError("history BCF/Current candidate identities differ")
    for field in shared:
        if not _policy_equal(frame[f"{field}__current"], frame[f"{field}__bcf"]):
            raise AssertionError(f"history BCF/Current mismatch in shared field {field}")
        frame[field] = frame.pop(f"{field}__current")
        frame.pop(f"{field}__bcf")
    if not frame["side_name"].eq("long").all():
        raise AssertionError("history panel is expected to be long-only")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _load_band_curves(root: Path, family: str) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for path in sorted((root / "mc1_packages" / f"family={family}").glob("month=*/band_curve.json")):
        values = np.asarray(json.loads(path.read_text(encoding="utf-8"))["expected_bps"], dtype=float)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise AssertionError(f"invalid {family} band curve: {path}")
        curves[path.parent.name.removeprefix("month=")] = values
    expected = {f"{day:%Y-%m}" for day in pd.date_range(START, END - pd.Timedelta(days=1), freq="D", tz="UTC")}
    missing = sorted(expected.difference(curves))
    if missing:
        raise AssertionError(f"missing {family} band curves for {missing}")
    return curves


def _causal_uncertainty(
    history: pd.DataFrame, family: str, curves: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return prior-21d residual sampling uncertainty at the daily cadence.

    The production shift uses the robust all-row residual location relative to
    the package's frozen score-band curve.  To avoid understating uncertainty
    in a dense cross-section, this estimate uses the standard error of prior
    *daily* residual means, not raw-row standard error.  Every row admitted to
    a day has a label availability timestamp strictly before that day.
    """
    work = history.loc[:, [
        "candidate_id", "__decision_ts__", "policy_label_available_ts", "policy_path_valid",
        "policy_net_bps", f"{family}_final_score",
    ]].copy()
    work["day"] = work["__decision_ts__"].dt.normalize()
    work["score_band"] = _score_bands(work, f"{family}_final_score")
    valid = (
        work["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(work["policy_net_bps"], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
        & work["policy_label_available_ts"].notna()
    )
    work = work.loc[valid].sort_values("policy_label_available_ts", kind="stable")
    days = pd.date_range(START.normalize(), (END - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    for day in days:
        curve = curves[f"{day:%Y-%m}"]
        eligible = work.loc[
            work["__decision_ts__"].ge(day - WINDOW)
            & work["__decision_ts__"].lt(day)
            & work["policy_label_available_ts"].lt(day)
        ].copy()
        eligible["residual"] = (
            pd.to_numeric(eligible["policy_net_bps"], errors="coerce").to_numpy(float)
            - curve[eligible["score_band"].to_numpy(int)]
        )
        daily = eligible.groupby("day", sort=True)["residual"].mean().to_numpy(float)
        if len(daily) >= 2:
            standard_error = float(np.std(daily, ddof=1) / math.sqrt(len(daily)))
        else:
            standard_error = float("inf")
        rows.append({
            "decision_day": day,
            f"{family}_uncertainty_se_bps": standard_error,
            f"{family}_uncertainty_days": int(len(daily)),
            f"{family}_uncertainty_rows": int(len(eligible)),
            f"{family}_reconstructed_shift_bps": _robust_mean(eligible["residual"].to_numpy(float)),
            f"{family}_uncertainty_max_available_ts": eligible["policy_label_available_ts"].max() if len(eligible) else pd.NaT,
        })
    return pd.DataFrame(rows)


def _arm_predictions(
    frame: pd.DataFrame, history: pd.DataFrame, curves: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_state = _causal_uncertainty(history, "current", curves["current"])
    bcf_state = _causal_uncertainty(history, "bcf", curves["bcf"])
    state = current_state.merge(bcf_state, on="decision_day", how="inner", validate="one_to_one")
    work = frame.copy()
    work["decision_day"] = work["__decision_ts__"].dt.normalize()
    work = work.merge(state, on="decision_day", how="left", validate="many_to_one")
    if work[["current_uncertainty_se_bps", "bcf_uncertainty_se_bps"]].isna().any().any():
        raise AssertionError("missing daily causal uncertainty state")
    for family in ("bcf", "current"):
        static = pd.to_numeric(work[f"{family}_static_bps"], errors="coerce").to_numpy(float)
        shift = pd.to_numeric(work[f"{family}_shift_bps"], errors="coerce").fillna(0.0).to_numpy(float)
        se = pd.to_numeric(work[f"{family}_uncertainty_se_bps"], errors="coerce").to_numpy(float)
        work[f"{family}_ev_static"] = static
        for weight in (0.25, 0.50, 0.75, 1.00):
            token = str(weight).replace(".", "")
            work[f"{family}_ev_lambda{token}"] = static + weight * shift
        work[f"{family}_shift_active_1se"] = np.abs(shift) > se
        work[f"{family}_ev_uncertainty_1se"] = static + np.where(np.abs(shift) > se, shift, 0.0)
    return work, state


ARMS = (
    ("static", "ev_static"),
    ("shrink_025", "ev_lambda025"),
    ("shrink_050", "ev_lambda05"),
    ("shrink_075", "ev_lambda075"),
    ("adaptive_21d", "ev_lambda10"),
    ("adaptive_uncertainty_1se", "ev_uncertainty_1se"),
)


def _capacity(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    dual = np.minimum(
        pd.to_numeric(frame[f"bcf_{suffix}"], errors="coerce").to_numpy(float),
        pd.to_numeric(frame[f"current_{suffix}"], errors="coerce").to_numpy(float),
    )
    eligible = (
        frame["enhanced_base_routed"].fillna(False).astype(bool).to_numpy()
        & np.isfinite(dual)
    )
    base = frame.loc[:, ["__decision_ts__"]].copy()
    base["passes"] = eligible & (dual >= THRESHOLD_BPS)
    return base.groupby("__decision_ts__", as_index=False, sort=True).agg(
        budget_k=("passes", "sum"),
    )


def _select_fixed_budget(frame: pd.DataFrame, arm: str, suffix: str, schedule: str, capacity: pd.DataFrame) -> pd.DataFrame:
    work = frame.merge(capacity, on="__decision_ts__", how="left", validate="many_to_one").copy()
    work["budget_k"] = pd.to_numeric(work["budget_k"], errors="coerce").fillna(0).astype(int)
    bcf = pd.to_numeric(work[f"bcf_{suffix}"], errors="coerce")
    current = pd.to_numeric(work[f"current_{suffix}"], errors="coerce")
    work["bcf_expected_bps"] = bcf
    work["current_expected_bps"] = current
    work["dual_gate_priority_bps"] = np.minimum(bcf, current)
    eligible = (
        work["enhanced_base_routed"].fillna(False).astype(bool)
        & np.isfinite(work["dual_gate_priority_bps"])
        & np.isfinite(work["bcf_expected_bps"])
    )
    work = work.loc[eligible].copy()
    work = work.sort_values(
        ["__decision_ts__", "dual_gate_priority_bps", "bcf_expected_bps", "candidate_id"],
        ascending=[True, False, False, True], kind="stable",
    )
    work["selection_rank"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    selected = work.loc[work["selection_rank"].le(work["budget_k"])].copy()
    selected["arm"] = arm
    selected["capacity_schedule"] = schedule
    selected["budget_exact"] = True
    return selected


def _valid_outcome(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_exit_bar_15m"], errors="coerce"))
    )


def _top_metrics(selected: pd.DataFrame) -> dict[str, float]:
    valid = selected.loc[_valid_outcome(selected)].copy()
    values: dict[str, float] = {}
    for count in (1, 2):
        top = valid.loc[valid["selection_rank"].le(count)]
        values[f"top_{count}_realized_net_bps"] = float(pd.to_numeric(top["policy_net_bps"], errors="coerce").mean()) if len(top) else float("nan")
        values[f"top_{count}_outcome_rows"] = int(len(top))
    return values


def _portfolio_candidates(selected: pd.DataFrame) -> pd.DataFrame:
    valid = selected.loc[_valid_outcome(selected)].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["auction_rank"] = valid.groupby("__decision_ts__", sort=False)["bcf_expected_bps"].rank(pct=True, method="average")
    decision = pd.to_datetime(valid["__decision_ts__"], utc=True)
    exit_bar = pd.to_numeric(valid["policy_exit_bar_15m"], errors="coerce").astype(int)
    table = pd.DataFrame({
        "timestamp": decision,
        "symbol": valid["__symbol__"].astype(str),
        "side": "long",
        "strategy_id": "strict_r3_matched_static_adaptive_mc1_budget",
        "policy_archetype": "strict_r3_matched_static_adaptive_mc1_budget",
        "normalized_rank_score": valid["auction_rank"].to_numpy(float),
        "strategy_rank_pct": valid["auction_rank"].to_numpy(float),
        "base_strategy_threshold": 0.0,
        "calibrated_score": valid["bcf_expected_bps"].to_numpy(float),
        "entry_price": pd.to_numeric(valid["policy_entry_price"], errors="coerce"),
        "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "exit_price": pd.to_numeric(valid["policy_exit_price"], errors="coerce"),
        "net_return": pd.to_numeric(valid["policy_net_bps"], errors="coerce") / 10_000.0,
        "gross_return": pd.to_numeric(valid["policy_gross_bps"], errors="coerce") / 10_000.0,
        "holding_bars": exit_bar + 1,
        "simple_policy_exit_reason": valid["policy_exit_reason"].astype(str),
        "fees_bps": 100.0,
        "slippage_bps": 0.0,
        "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0,
        "source_month": decision.dt.strftime("%Y-%m"),
        "candidate_id": valid["candidate_id"].astype(str),
        "mapped_expected_net_bps": valid["bcf_expected_bps"].to_numpy(float),
        "policy_outcome_available": np.ones(len(valid), dtype=bool),
    })
    return normalise_candidate_table(table)


def _weekly_metrics(selected: pd.DataFrame, arm: str, schedule: str) -> pd.DataFrame:
    valid = selected.loc[_valid_outcome(selected)].copy()
    if valid.empty:
        return pd.DataFrame(columns=["arm", "capacity_schedule", "week", "trades", "net_ev_bps_per_trade", "total_net_bps"])
    valid["week"] = valid["__decision_ts__"].dt.strftime("%G-W%V")
    result = valid.groupby("week", as_index=False, sort=True).agg(
        trades=("candidate_id", "size"),
        net_ev_bps_per_trade=("policy_net_bps", "mean"),
        total_net_bps=("policy_net_bps", "sum"),
    )
    result.insert(0, "capacity_schedule", schedule)
    result.insert(0, "arm", arm)
    return result


def _monthly_metrics(selected: pd.DataFrame, arm: str, schedule: str) -> pd.DataFrame:
    valid = selected.loc[_valid_outcome(selected)].copy()
    if valid.empty:
        return pd.DataFrame(columns=["arm", "capacity_schedule", "month", "trades", "net_ev_bps_per_trade", "total_net_bps"])
    valid["month"] = valid["__decision_ts__"].dt.strftime("%Y-%m")
    result = valid.groupby("month", as_index=False, sort=True).agg(
        trades=("candidate_id", "size"),
        net_ev_bps_per_trade=("policy_net_bps", "mean"),
        total_net_bps=("policy_net_bps", "sum"),
    )
    result.insert(0, "capacity_schedule", schedule)
    result.insert(0, "arm", arm)
    return result


def _portfolio_summary(decisions: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float | int]:
    """Summarise a fully outcome-valid candidate replay without assumptions.

    This ablation intentionally excludes unresolved/incomplete policy rows
    before its constrained replay.  ``replay_candidates`` does not propagate
    arbitrary source provenance fields into decisions, so use that declared
    all-valid contract directly rather than labelling every decision unknown.
    """
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    net = pd.to_numeric(accepted.get("position_net_return"), errors="coerce") * 10_000.0
    wallet = pd.to_numeric(equity.get("wallet"), errors="coerce").dropna()
    drawdown = float((wallet / wallet.cummax() - 1.0).min()) if len(wallet) else float("nan")
    return {
        "accepted_trades": int(len(accepted)),
        "net_bps_per_trade": float(net.mean()) if len(net) else float("nan"),
        "net_sum_bps": float(net.sum()) if len(net) else float("nan"),
        "max_drawdown": drawdown,
    }


def _summarize(selected: pd.DataFrame, arm: str, schedule: str, out: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = selected.loc[_valid_outcome(selected)].copy()
    weekly = _weekly_metrics(selected, arm, schedule)
    monthly = _monthly_metrics(selected, arm, schedule)
    candidates = _portfolio_candidates(selected)
    if candidates.empty:
        decisions = pd.DataFrame()
        equity = pd.DataFrame()
        portfolio = {"accepted_trades": 0, "net_bps_per_trade": float("nan"), "net_sum_bps": float("nan"), "max_drawdown": float("nan")}
    else:
        decisions, equity, _ = replay_candidates(
            candidates,
            _params(),
            mode="global_auction",
            ev_curve=CAUSAL_AUCTION_CURVE,
            market_mode="perps",
            initial_wallet=1_000.0,
        )
        portfolio = _portfolio_summary(decisions, equity)
    top = _top_metrics(selected)
    summary: dict[str, object] = {
        "arm": arm,
        "capacity_schedule": schedule,
        "selected_admissions": int(len(selected)),
        "outcome_valid_admissions": int(len(valid)),
        "outcome_coverage": float(len(valid) / max(len(selected), 1)),
        "admitted_net_ev_bps_per_trade": float(pd.to_numeric(valid["policy_net_bps"], errors="coerce").mean()) if len(valid) else float("nan"),
        "total_admitted_utility_bps": float(pd.to_numeric(valid["policy_net_bps"], errors="coerce").sum()) if len(valid) else float("nan"),
        "weekly_mean_net_bps": float(weekly["net_ev_bps_per_trade"].mean()) if len(weekly) else float("nan"),
        "weekly_worst_net_bps": float(weekly["net_ev_bps_per_trade"].min()) if len(weekly) else float("nan"),
        "weekly_positive_fraction": float(weekly["net_ev_bps_per_trade"].gt(0).mean()) if len(weekly) else float("nan"),
        "monthly_worst_net_bps": float(monthly["net_ev_bps_per_trade"].min()) if len(monthly) else float("nan"),
        **top,
        **{f"portfolio_{key}": value for key, value in portfolio.items()},
    }
    if not decisions.empty:
        decisions = decisions.copy()
        decisions["arm"] = arm
        decisions["capacity_schedule"] = schedule
    return summary, weekly, monthly, decisions


def _render_report(metrics: pd.DataFrame, out: Path) -> None:
    ordered = metrics.sort_values(
        ["capacity_schedule", "top_1_realized_net_bps", "top_2_realized_net_bps", "total_admitted_utility_bps"],
        ascending=[True, False, False, False], kind="stable",
    )
    columns = [
        "capacity_schedule", "arm", "selected_admissions", "admitted_net_ev_bps_per_trade",
        "top_1_realized_net_bps", "top_2_realized_net_bps", "total_admitted_utility_bps",
        "weekly_worst_net_bps", "weekly_positive_fraction", "portfolio_accepted_trades",
        "portfolio_net_bps_per_trade", "portfolio_net_sum_bps", "portfolio_max_drawdown",
    ]
    table = ordered.loc[:, columns].copy()
    for field in table.columns:
        if pd.api.types.is_numeric_dtype(table[field]) and field not in {"selected_admissions", "portfolio_accepted_trades"}:
            table[field] = table[field].map(lambda value: "—" if not np.isfinite(value) else f"{value:.2f}")
    headers = [str(field) for field in table.columns]
    markdown_table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        markdown_table.append("| " + " | ".join(str(value) for value in row) + " |")
    lines = [
        "# Matched Static vs Adaptive MC1: Fixed-Budget Ablation",
        "",
        "Research only. The target-free BCF/Current score panels, rich-policy labels, and global portfolio constraints are fixed. The maps are compared at the exact same per-timestamp admission count, rather than at a common +50-bps threshold.",
        "",
        "## Capacity schedules",
        "",
        "- `static_ge50`: each timestamp’s capacity equals the number passing both static BCF and Current maps at +50 bps.",
        "- `adaptive_ge50`: each timestamp’s capacity equals the number passing both unshrunk 21-day-shift maps at +50 bps.",
        "- Within every schedule, every arm receives that exact timestamp-local capacity. Ranking uses the conservative dual-map value `min(BCF EV, Current EV)`; the portfolio then preserves BCF mapped EV as auction priority.",
        "",
        "## Results",
        "",
        *markdown_table,
        "",
        "## Interpretation rule",
        "",
        "Select using realized Top-1/Top-2 EV, total admitted utility, and weekly stability together. Do not promote an arm on mapper MAE alone or on a capacity change.",
        "",
        "## Causality",
        "",
        "The static score is a persisted strict-prequential MC1 output. The 21-day shift is persisted from resolved labels only. The uncertainty gate uses only prior-resolved daily residual means; a shift activates only when its absolute value exceeds one daily-clustered standard error. Policy outcomes are joined only after target-free score construction and are not an input to any static or shift prediction.",
        "",
    ]
    (out / "MATCHED_STATIC_ADAPTIVE_MC1_BUDGET_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, history_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    root = root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # The evaluation ledger preserves the static/current 21d MC1 values that
    # are being compared.  Its companion full-prehistory score ledger supplies
    # only prior resolved residuals for the uncertainty gate.
    full_frame = _load_matched(root)
    frame = full_frame.loc[
        full_frame["__decision_ts__"].ge(START) & full_frame["__decision_ts__"].lt(END)
    ].copy()
    history = _load_history(history_root.resolve())
    curves = {
        "bcf": _load_band_curves(root, "bcf"),
        "current": _load_band_curves(root, "current"),
    }
    frame, uncertainty = _arm_predictions(frame, history, curves)
    if frame.empty:
        raise ValueError("evaluation window has no matched rows")
    # The historical residual substrate must represent the same BCF/Current
    # score coordinates as the evaluation panel, not a nearby score family.
    probe = frame.loc[:, ["candidate_id", "__decision_ts__", "bcf_final_score", "current_final_score"]].merge(
        history.loc[:, ["candidate_id", "__decision_ts__", "bcf_final_score", "current_final_score"]],
        on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one", suffixes=("_eval", "_history"),
    )
    if len(probe) != len(frame) or probe[["bcf_final_score_history", "current_final_score_history"]].isna().any().any():
        raise AssertionError("full-prehistory ledger does not cover every evaluation score identity")
    for family in ("bcf", "current"):
        if not np.allclose(
            probe[f"{family}_final_score_eval"].to_numpy(float),
            probe[f"{family}_final_score_history"].to_numpy(float),
            rtol=0.0, atol=1e-12, equal_nan=True,
        ):
            raise AssertionError(f"full-prehistory {family} final_score differs from the evaluation ledger")
    persisted_shift = frame.loc[:, ["decision_day", "bcf_shift_bps", "current_shift_bps"]].drop_duplicates("decision_day")
    uncertainty = uncertainty.merge(persisted_shift, on="decision_day", how="left", validate="one_to_one")
    for family in ("bcf", "current"):
        delta = np.abs(
            pd.to_numeric(uncertainty[f"{family}_reconstructed_shift_bps"], errors="coerce")
            - pd.to_numeric(uncertainty[f"{family}_shift_bps"], errors="coerce")
        )
        if not np.isfinite(delta).all() or float(delta.max()) > 1e-8:
            raise AssertionError(
                f"{family} reconstructed 21d residual shift does not match persisted MC1 shift; "
                f"max delta={float(delta.max()):.6g} bps"
            )
    schedules = {
        "static_ge50": _capacity(frame, "ev_static"),
        "adaptive_ge50": _capacity(frame, "ev_lambda10"),
    }
    metrics: list[dict[str, object]] = []
    selected_parts: list[pd.DataFrame] = []
    weekly_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    decisions_parts: list[pd.DataFrame] = []
    budget_audits: list[pd.DataFrame] = []
    for schedule, capacity in schedules.items():
        audit = capacity.copy()
        audit["capacity_schedule"] = schedule
        budget_audits.append(audit)
        for arm, suffix in ARMS:
            selected = _select_fixed_budget(frame, arm, suffix, schedule, capacity)
            observed = selected.groupby("__decision_ts__", as_index=False).size().rename(columns={"size": "selected_k"})
            check = capacity.merge(observed, on="__decision_ts__", how="left", validate="one_to_one")
            check["selected_k"] = check["selected_k"].fillna(0).astype(int)
            if not check["selected_k"].eq(check["budget_k"]).all():
                failing = check.loc[~check["selected_k"].eq(check["budget_k"])].head(5).to_dict("records")
                raise AssertionError(f"{arm}/{schedule}: fixed budget mismatch {failing}")
            summary, weekly, monthly, decisions = _summarize(selected, arm, schedule, out)
            metrics.append(summary)
            selected_parts.append(selected)
            weekly_parts.append(weekly)
            monthly_parts.append(monthly)
            if not decisions.empty:
                decisions_parts.append(decisions)
    selected_all = pd.concat(selected_parts, ignore_index=True)
    selected_all.to_parquet(out / "selected_candidates.parquet", index=False, compression="zstd")
    pd.concat(budget_audits, ignore_index=True).to_parquet(out / "budget_audit.parquet", index=False, compression="zstd")
    uncertainty.to_parquet(out / "causal_uncertainty_state.parquet", index=False, compression="zstd")
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_parquet(out / "metrics.parquet", index=False, compression="zstd")
    pd.concat(weekly_parts, ignore_index=True).to_parquet(out / "weekly_metrics.parquet", index=False, compression="zstd")
    pd.concat(monthly_parts, ignore_index=True).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    (pd.concat(decisions_parts, ignore_index=True) if decisions_parts else pd.DataFrame()).to_parquet(
        out / "portfolio_decisions.parquet", index=False, compression="zstd",
    )
    active = uncertainty.loc[
        uncertainty["decision_day"].ge(START) & uncertainty["decision_day"].lt(END)
    ].copy()
    _render_report(metrics_frame, out)
    _once_json(out / "correctness_report.json", {
        "schema": SCHEMA,
        "identical_bcf_current_candidate_ids": True,
        "static_and_shift_inputs_persisted_before_this_ablation": True,
        "policy_fields_checked_identical_between_families": True,
        "static_and_adaptive_capacity_schedules_are_timestamp_local": True,
        "all_arms_exactly_match_each_schedule_capacity": True,
        "uncertainty_uses_prior_resolved_daily_residuals_only": bool(
            (
                active["current_uncertainty_se_bps"].notna()
                & active["bcf_uncertainty_se_bps"].notna()
            ).all()
        ),
        "uncertainty_label_availability_strictly_precedes_decision_day": bool(
            (
                (
                    active["current_uncertainty_max_available_ts"].isna()
                    | active["current_uncertainty_max_available_ts"].lt(active["decision_day"])
                )
                & (
                    active["bcf_uncertainty_max_available_ts"].isna()
                    | active["bcf_uncertainty_max_available_ts"].lt(active["decision_day"])
                )
            ).all()
        ),
        "no_live_or_exchange_mutation": True,
    })
    _once_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline matched static-versus-adaptive dual-MC1 fixed-budget research; no live or exchange mutation",
        "prediction_root": str(root),
        "uncertainty_history_root": str(history_root.resolve()),
        "prediction_hashes": {
            "bcf": _sha256(root / "enhanced_bcf_mc1_predictions.parquet"),
            "current": _sha256(root / "enhanced_current_mc1_predictions.parquet"),
        },
        "uncertainty_history_hashes": {
            "bcf": _sha256(history_root / "enhanced_bcf_mc1_predictions.parquet"),
            "current": _sha256(history_root / "enhanced_current_mc1_predictions.parquet"),
        },
        "evaluation": {"start": START.isoformat(), "end_exclusive": END.isoformat()},
        "threshold_anchor_bps": THRESHOLD_BPS,
        "capacity_schedules": {
            "static_ge50": "per-timestamp count where min(static BCF, static Current) >= 50",
            "adaptive_ge50": "per-timestamp count where min(unshrunk BCF, unshrunk Current) >= 50",
        },
        "arms": [
            "static", "shrink_025", "shrink_050", "shrink_075", "adaptive_21d", "adaptive_uncertainty_1se",
        ],
        "priority": "selection=min(BCF expected EV, Current expected EV); portfolio auction=BCF expected EV",
        "uncertainty": "prior-21d daily-clustered residual standard error; activate only abs(shift) > 1 SE",
        "portfolio": "same global chronological constraints as the P8U dual-MC1 package",
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--uncertainty-history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.prediction_root, args.uncertainty_history_root, args.out))


if __name__ == "__main__":
    main()
