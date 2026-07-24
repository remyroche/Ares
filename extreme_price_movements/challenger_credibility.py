"""Leakage-safe statistical checks for frozen policy challengers.

The functions in this module work on already materialized, true-OOS decision
rows.  They never refit a trading model or alter ranks.  Daily observations are
the unit of evidence so cross-asset rows from a common market day cannot
artificially inflate confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import t as student_t


@dataclass(frozen=True)
class PosteriorConfig:
    """Small, deliberately conservative Student-t hierarchical sampler."""

    draws: int = 8_000
    burn_in: int = 2_000
    thin: int = 4
    seed: int = 7
    degrees_of_freedom: float = 4.0


def consecutive_event_blocks(day: pd.Series, adverse: pd.Series) -> pd.Series:
    """Return causal contiguous event-block ids; normal days receive ``normal``."""

    days = pd.to_datetime(day, utc=True).dt.floor("D")
    flagged = adverse.fillna(0).astype(bool).to_numpy()
    values = np.full(len(days), "normal", dtype=object)
    block = 0
    previous: pd.Timestamp | None = None
    active = False
    for idx, current in enumerate(days):
        if not flagged[idx]:
            active = False
            previous = current
            continue
        contiguous = previous is not None and (current - previous) == pd.Timedelta(days=1)
        if not active or not contiguous:
            block += 1
        values[idx] = f"event_{block:03d}"
        active = True
        previous = current
    return pd.Series(values, index=day.index, dtype="string")


def daily_decision_deltas(
    rows: pd.DataFrame,
    *,
    parent_rank: str,
    challenger_rank: str,
    top_threshold: float = 0.90,
    ev_column: str = "ev_after_1pct",
    clean_column: str = "clean_exec",
    adverse_column: str = "adverse_calendar_cell",
) -> pd.DataFrame:
    """Aggregate parent/challenger decisions into independent daily evidence."""

    required = {"__ts__", parent_rank, challenger_rank, ev_column, clean_column}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise KeyError(f"Missing decision-delta columns: {missing}")
    frame = rows.loc[:, [*required, *([adverse_column] if adverse_column in rows else [])]].copy()
    frame["day"] = pd.to_datetime(frame["__ts__"], utc=True).dt.floor("D")
    ev = pd.to_numeric(frame[ev_column], errors="coerce").to_numpy(np.float64)
    clean = pd.to_numeric(frame[clean_column], errors="coerce").to_numpy(np.float64)
    parent = pd.to_numeric(frame[parent_rank], errors="coerce").to_numpy(np.float64) >= top_threshold
    challenger = pd.to_numeric(frame[challenger_rank], errors="coerce").to_numpy(np.float64) >= top_threshold
    frame["parent_selected"] = parent
    frame["challenger_selected"] = challenger
    frame["parent_ev"] = np.where(parent, ev, 0.0)
    frame["challenger_ev"] = np.where(challenger, ev, 0.0)
    frame["parent_clean"] = np.where(parent, clean, np.nan)
    frame["challenger_clean"] = np.where(challenger, clean, np.nan)
    grouped = frame.groupby("day", observed=True, sort=True)
    result = grouped.agg(
        parent_total_ev=("parent_ev", "sum"),
        challenger_total_ev=("challenger_ev", "sum"),
        parent_selected=("parent_selected", "sum"),
        challenger_selected=("challenger_selected", "sum"),
        parent_clean_sum=("parent_clean", "sum"),
        challenger_clean_sum=("challenger_clean", "sum"),
        parent_clean=("parent_clean", "mean"),
        challenger_clean=("challenger_clean", "mean"),
    ).reset_index()
    result["parent_ev_per_trade"] = result["parent_total_ev"] / result["parent_selected"].clip(lower=1)
    result["challenger_ev_per_trade"] = result["challenger_total_ev"] / result["challenger_selected"].clip(lower=1)
    result["delta_total_ev"] = result["challenger_total_ev"] - result["parent_total_ev"]
    result["delta_ev_per_trade"] = result["challenger_ev_per_trade"] - result["parent_ev_per_trade"]
    result["delta_clean_precision"] = result["challenger_clean"] - result["parent_clean"]
    result["activity_ratio"] = result["challenger_selected"] / result["parent_selected"].clip(lower=1)
    if adverse_column in frame:
        event = grouped[adverse_column].max().reset_index(drop=True)
    else:
        event = pd.Series(0, index=result.index)
    result["is_adverse_day"] = pd.to_numeric(event, errors="coerce").fillna(0).gt(0).to_numpy(np.int8)
    result["event_block"] = consecutive_event_blocks(result["day"], result["is_adverse_day"])
    result["month"] = result["day"].dt.strftime("%Y-%m")
    result["event_family"] = np.where(result["is_adverse_day"].astype(bool), "adverse", "normal")
    return result


def _log_student_t(residual: np.ndarray, sigma: float, nu: float) -> float:
    scaled = residual / max(sigma, 1e-8)
    return float(
        np.sum(
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
            - np.log(max(sigma, 1e-8))
            - ((nu + 1.0) / 2.0) * np.log1p((scaled * scaled) / nu)
        )
    )


def hierarchical_student_t_posterior(
    daily: pd.DataFrame,
    *,
    value_column: str = "delta_ev_per_trade",
    config: PosteriorConfig = PosteriorConfig(),
) -> pd.DataFrame:
    """Sample a partial-pooled Student-t model with month and event-family effects.

    The model is deliberately small: ``d ~ t_nu(mu + month + event_family,
    sigma)``. Effects have zero-centred Gaussian priors with learned scales.
    This avoids treating correlated rows within a market day as independent.
    """

    local = daily.dropna(subset=[value_column, "month", "event_family"]).copy()
    if len(local) < 4:
        raise ValueError("At least four independent observations are required")
    value = local[value_column].to_numpy(np.float64)
    scale = float(np.median(np.abs(value - np.median(value))) * 1.4826)
    scale = max(scale, 1e-5)
    y = value / scale
    month_codes, months = pd.factorize(local["month"], sort=True)
    family_codes, families = pd.factorize(local["event_family"], sort=True)
    n_months, n_families = len(months), len(families)
    # mu, log_sigma, log_tau_month, log_tau_family, month effects, family effects.
    size = 4 + n_months + n_families

    def log_posterior(theta: np.ndarray) -> float:
        mu, log_sigma, log_tm, log_tf = theta[:4]
        sigma, tau_month, tau_family = np.exp(log_sigma), np.exp(log_tm), np.exp(log_tf)
        month_effect = theta[4 : 4 + n_months]
        family_effect = theta[4 + n_months :]
        mean = mu + month_effect[month_codes] + family_effect[family_codes]
        prior = -0.5 * (mu / 2.0) ** 2
        prior += -0.5 * np.sum((month_effect / tau_month) ** 2) - n_months * log_tm
        prior += -0.5 * np.sum((family_effect / tau_family) ** 2) - n_families * log_tf
        # Weak half-normal-like regularisation on scale parameters in log space.
        prior += -0.5 * (sigma / 3.0) ** 2 + log_sigma
        prior += -0.5 * (tau_month / 2.0) ** 2 + log_tm
        prior += -0.5 * (tau_family / 2.0) ** 2 + log_tf
        return _log_student_t(y - mean, sigma, config.degrees_of_freedom) + prior

    rng = np.random.default_rng(config.seed)
    theta = np.zeros(size, dtype=np.float64)
    theta[1:4] = np.log([1.0, 0.5, 0.5])
    current = log_posterior(theta)
    proposal = np.full(size, 0.08, dtype=np.float64)
    proposal[0] = 0.04
    samples: list[np.ndarray] = []
    accepted = 0
    total = config.draws + config.burn_in
    for step in range(total):
        candidate = theta + rng.normal(0.0, proposal)
        score = log_posterior(candidate)
        if np.log(rng.uniform()) < score - current:
            theta, current = candidate, score
            accepted += 1
        if step == max(250, config.burn_in // 2):
            rate = accepted / max(step + 1, 1)
            proposal *= np.clip(rate / 0.28, 0.55, 1.7)
        if step >= config.burn_in and (step - config.burn_in) % config.thin == 0:
            samples.append(theta.copy())
    posterior = pd.DataFrame(np.asarray(samples), columns=[
        "mu_standardized", "log_sigma", "log_tau_month", "log_tau_event_family",
        *[f"month_effect__{month}" for month in months],
        *[f"event_family_effect__{family}" for family in families],
    ])
    posterior["mu"] = posterior["mu_standardized"] * scale
    posterior.attrs.update({
        "scale": scale,
        "acceptance_rate": accepted / max(total, 1),
        "months": list(months),
        "event_families": list(families),
        "degrees_of_freedom": config.degrees_of_freedom,
    })
    return posterior


def bayesian_bootstrap_contract_probability(
    daily: pd.DataFrame,
    *,
    draws: int = 20_000,
    seed: int = 17,
    minimum_ev_per_trade: float = 0.0,
) -> dict[str, float]:
    """Estimate the joint contract probability using common day-level weights."""

    local = daily.dropna(subset=["delta_ev_per_trade", "delta_total_ev", "activity_ratio"]).copy()
    if len(local) < 8:
        raise ValueError("At least eight complete days are required")
    rng = np.random.default_rng(seed)
    columns = [
        "parent_total_ev", "challenger_total_ev", "parent_selected", "challenger_selected",
        "parent_clean_sum", "challenger_clean_sum",
    ]
    values = local[columns].to_numpy(np.float64)
    passes = np.zeros(draws, dtype=bool)
    for start in range(0, draws, 512):
        size = min(512, draws - start)
        weight = rng.dirichlet(np.ones(len(values)), size=size)
        parent_total = weight @ values[:, 0]
        challenger_total = weight @ values[:, 1]
        parent_count = weight @ values[:, 2]
        challenger_count = weight @ values[:, 3]
        parent_clean = weight @ values[:, 4]
        challenger_clean = weight @ values[:, 5]
        mean_ev = challenger_total / np.maximum(challenger_count, 1e-12) - parent_total / np.maximum(parent_count, 1e-12)
        total_ev = challenger_total - parent_total
        clean = challenger_clean / np.maximum(challenger_count, 1e-12) - parent_clean / np.maximum(parent_count, 1e-12)
        activity = challenger_count / np.maximum(parent_count, 1e-12)
        passes[start : start + size] = (
            (mean_ev > minimum_ev_per_trade)
            & (total_ev > 0.0)
            & (clean >= 0.0)
            & (activity >= 0.90)
        )
    return {
        "joint_pass_probability": float(passes.mean()),
        "draws": int(draws),
        "minimum_ev_per_trade": float(minimum_ev_per_trade),
    }


def leave_group_out(daily: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Recompute aggregate deltas after removing each day, event block, or month."""

    required = {group_column, "delta_total_ev", "delta_ev_per_trade", "delta_clean_precision", "activity_ratio"}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise KeyError(f"Missing leave-out columns: {missing}")
    rows: list[dict[str, object]] = []
    def aggregate(frame: pd.DataFrame) -> dict[str, float]:
        parent_count = float(frame["parent_selected"].sum())
        challenger_count = float(frame["challenger_selected"].sum())
        parent_total = float(frame["parent_total_ev"].sum())
        challenger_total = float(frame["challenger_total_ev"].sum())
        return {
            "delta_total_ev": challenger_total - parent_total,
            "delta_ev_per_trade": challenger_total / max(challenger_count, 1) - parent_total / max(parent_count, 1),
            "delta_clean_precision": float(frame["challenger_clean_sum"].sum()) / max(challenger_count, 1)
            - float(frame["parent_clean_sum"].sum()) / max(parent_count, 1),
            "activity_ratio": challenger_count / max(parent_count, 1),
        }
    overall = aggregate(daily)
    for group, held in daily.groupby(group_column, observed=True, sort=True):
        kept = daily.loc[daily[group_column].ne(group)]
        if kept.empty:
            continue
        metrics = aggregate(kept)
        total = metrics["delta_total_ev"]
        mean = metrics["delta_ev_per_trade"]
        rows.append({
            "group_column": group_column,
            "held_out": str(group),
            "held_out_days": int(len(held)),
            "delta_total_ev_without": total,
            "delta_ev_per_trade_without": mean,
            "delta_clean_precision_without": metrics["delta_clean_precision"],
            "activity_ratio_without": metrics["activity_ratio"],
            "total_ev_sign_reversal": bool((overall["delta_total_ev"] > 0.0) != (total > 0.0)),
            "ev_per_trade_sign_reversal": bool((overall["delta_ev_per_trade"] > 0.0) != (mean > 0.0)),
            "influence_share": float(
                (overall["delta_total_ev"] - total) / max(abs(overall["delta_total_ev"]), 1e-12)
            ),
        })
    return pd.DataFrame(rows)


__all__ = [
    "PosteriorConfig",
    "bayesian_bootstrap_contract_probability",
    "consecutive_event_blocks",
    "daily_decision_deltas",
    "hierarchical_student_t_posterior",
    "leave_group_out",
]
