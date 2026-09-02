#!/usr/bin/env python3
"""Strict-OOS supportive-label consensus comparison, research only.

This is the bounded downstream test for the repaired long supportive-label
workstream.  Its two independently varied axes are:

* input architecture: B0 alone, one of the two development-selected direct
  scores alone, or their compact three-score stack; and
* supervisory anchor: ordinary policy residual, direct, frozen-path,
  causal-regime, or causal×path residual.

All upstream scores and archetype probabilities are already strict-OOS.  For
every evaluation month, the LambdaRank consensus fit uses the preceding three
calendar months only, with H12 labels resolved before the month starts.  Held
predictions are written target-free before policy outcomes are joined for the
offline dual-MC1/portfolio evaluation.  The 2025 July--December result selects
at most two challengers; only those are then evaluated on 2026 April--July.

This runner is deliberately research-only.  It never changes canonical or
live artifacts, MC1, admission, sizing, or execution code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "strict_r3_long_supportive_consensus_walkforward_v2"
SEED = 1729
# Evaluation is deliberately robust to the admission decision.  A challenger
# must work at both the ordinary +50-bps gate and the less conservative
# +30-bps gate.  Development selection uses their unweighted mean rather than
# optimising against one threshold after the fact.
MC1_THRESHOLDS_BPS = (30.0, 50.0)
BASE_ROUTE_PERCENTILE = 0.70
RESIDUAL_BANDS = (-100.0, -30.0, 30.0, 90.0)
LABEL_GAIN = (0.0, 1.0, 2.0, 4.0, 7.0)
QUERY_HOURS = 4

# These two were selected strictly on the April--September 2025 development
# OOS score panels versus B0, before the consensus experiment is run.
SELECTED_DIRECT_ARMS = {
    "direct_et": "S3_direct_efficiency_time_equal",
    "direct_etb": "S3_direct_efficiency_time_base_equal",
}

ARCHITECTURES = {
    "I0_b0_single": ("b0_bps",),
    "I1_direct_et_single": ("direct_et_bps",),
    "I2_direct_etb_single": ("direct_etb_bps",),
    "I3_b0_direct_stack": ("b0_bps", "direct_et_bps", "direct_etb_bps"),
}

TARGETS = {
    "L0_policy_residual": "input_anchor_bps",
    "L1_direct_residual": "direct_etb_bps",
    "L2_path_residual": "path_anchor_bps",
    "L3_causal_regime_residual": "causal_state_anchor_bps",
    "L4_causal_path_residual": "causal_path_anchor_bps",
}

EVAL_MONTHS_2025 = tuple(pd.date_range("2025-07-01", "2025-12-01", freq="MS", tz="UTC"))
EVAL_MONTHS_2026 = tuple(pd.date_range("2026-04-01", "2026-07-01", freq="MS", tz="UTC"))


@dataclass(frozen=True)
class ResearchPortfolioContract:
    """Narrow offline mirror of the frozen MC1 global-auction constraints.

    It intentionally contains only the evaluation constraints requested for
    this ablation and does not import the live model/parity stack.  A separate
    matched replay through the full production adapter remains required before
    any promotion decision.
    """

    max_concurrent_positions: int = 8
    max_new_entries_per_timestamp: int = 2
    max_per_symbol: int = 1
    max_wallet_margin_fraction: float = 0.80
    margin_slot_fraction: float = 0.10
    leverage: float = 7.0
    initial_wallet: float = 1000.0


PORTFOLIO = ResearchPortfolioContract()


@dataclass(frozen=True)
class Sources:
    stage1: Path
    direct: Path
    stage2: Path
    causal_joint: Path
    current_mc1: Path
    bcf_mc1: Path
    policy: Path


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    expanded: list[Path] = []
    for path in paths:
        expanded.extend(sorted(path.rglob("*.parquet")) if path.is_dir() else [path])
    for path in sorted(expanded):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _read_matching_part(folder: Path, arm: str, columns: Sequence[str]) -> pd.DataFrame:
    for path in sorted(folder.glob("*.parquet")):
        probe = pd.read_parquet(path, columns=["arm"])
        if len(probe) and str(probe["arm"].iloc[0]) == arm:
            return pd.read_parquet(path, columns=list(columns))
    raise FileNotFoundError(f"{folder}: arm {arm!r} not found")


def _read_stage1_b0(root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    cols = ["candidate_id", "__decision_ts__", "fold", "cohort", "predicted_policy_net_bps"]
    for folder in sorted((root / "stage1_oof_prediction_parts").glob("fold=*")):
        frame = _read_matching_part(folder, "B0_prequential_upstream", cols)
        pieces.append(frame.rename(columns={"predicted_policy_net_bps": "b0_bps"}))
    return pd.concat(pieces, ignore_index=True)


def _read_direct(root: Path, arm: str, name: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    cols = ["candidate_id", "__decision_ts__", "fold", "cohort", "predicted_policy_net_bps"]
    for folder in sorted((root / "oof_prediction_parts").glob("fold=*")):
        frame = _read_matching_part(folder, arm, cols)
        pieces.append(frame.rename(columns={"predicted_policy_net_bps": name}))
    return pd.concat(pieces, ignore_index=True)


def _read_path(root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for folder in sorted((root / "oof_prediction_parts").glob("fold=*")):
        cols = [
            "candidate_id", "__decision_ts__", "predicted_policy_net_bps", "path_entropy", "path_max_probability",
            *[f"frozen_path_p_{index:02d}" for index in range(8)],
        ]
        frame = _read_matching_part(folder, "S2_frozen_gmm_k8", cols)
        pieces.append(frame.rename(columns={"predicted_policy_net_bps": "path_anchor_bps"}))
    return pd.concat(pieces, ignore_index=True)


def _read_causal_joint(root: Path) -> pd.DataFrame:
    cols = [
        "candidate_id", "__decision_ts__", "__symbol__",
        "C1_ward_k4_state_expected_ev",
        "C1_ward_k4_J2_soft_base_equal_causal120_plus_oof_stack",
        *[f"C1_ward_k4__q_{index:02d}" for index in range(4)],
        *[f"P3_path_gmm_k8_causal120_plus_oof_stack__p_{index:02d}" for index in range(8)],
        "P3_path_gmm_k8_causal120_plus_oof_stack__entropy",
    ]
    pieces = [pd.read_parquet(path, columns=cols) for path in sorted((root / "causal_joint_oof_predictions").glob("fold=*.parquet"))]
    result = pd.concat(pieces, ignore_index=True)
    return result.rename(columns={
        "C1_ward_k4_state_expected_ev": "causal_state_anchor_bps",
        "C1_ward_k4_J2_soft_base_equal_causal120_plus_oof_stack": "causal_path_anchor_bps",
        "P3_path_gmm_k8_causal120_plus_oof_stack__entropy": "causal_path_entropy",
        **{f"C1_ward_k4__q_{index:02d}": f"causal_state_q_{index:02d}" for index in range(4)},
        **{f"P3_path_gmm_k8_causal120_plus_oof_stack__p_{index:02d}": f"causal_path_p_{index:02d}" for index in range(8)},
    })


def _merge_exact(left: pd.DataFrame, right: pd.DataFrame, *, name: str) -> pd.DataFrame:
    keys = ["candidate_id", "__decision_ts__"]
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise AssertionError(f"identity mismatch while joining {name}: {len(left)} / {len(right)} / {len(merged)}")
    return merged


def _load_population(source: Sources) -> pd.DataFrame:
    base = _read_stage1_b0(source.stage1)
    direct_et = _read_direct(source.direct, SELECTED_DIRECT_ARMS["direct_et"], "direct_et_bps")
    direct_etb = _read_direct(source.direct, SELECTED_DIRECT_ARMS["direct_etb"], "direct_etb_bps")
    path = _read_path(source.stage2)
    causal = _read_causal_joint(source.causal_joint)
    result = _merge_exact(base, direct_et, name="direct_et")
    result = _merge_exact(result, direct_etb, name="direct_etb")
    result = _merge_exact(result, path, name="frozen_path")
    result = _merge_exact(result, causal, name="causal_joint")
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result["candidate_id"].duplicated().any():
        raise AssertionError("duplicate base candidate identity")
    return result


def _load_policy_and_mc1(source: Sources) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_cols = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    policy = pd.read_parquet(source.policy, columns=policy_cols)
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy has duplicate IDs")
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    mc1_cols = ["candidate_id", "mc1_expected_bps"]
    current = pd.read_parquet(source.current_mc1, columns=mc1_cols).rename(columns={"mc1_expected_bps": "current_mc1_expected_bps"})
    bcf = pd.read_parquet(source.bcf_mc1, columns=mc1_cols).rename(columns={"mc1_expected_bps": "bcf_mc1_expected_bps"})
    if current["candidate_id"].duplicated().any() or bcf["candidate_id"].duplicated().any():
        raise AssertionError("MC1 source identities must be unique")
    return policy, current.merge(bcf, on="candidate_id", how="inner", validate="one_to_one")


def _rank_pct(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("__decision_ts__", sort=False)[column].rank(pct=True, method="average")


def _query_id(timestamp: pd.Series) -> pd.Series:
    return pd.to_datetime(timestamp, utc=True).dt.floor(f"{QUERY_HOURS}h").astype(str) + "|long"


def _grade(value: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(value, errors="coerce").to_numpy(float)
    return np.digitize(raw, RESIDUAL_BANDS, right=True).astype(np.int32)


def _model() -> lgb.LGBMRanker:
    # Frozen compact residual control: no HPO in this target comparison.
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 2, 5],
        n_estimators=220, learning_rate=0.03, max_depth=4, num_leaves=31,
        min_child_samples=893, subsample=0.867, colsample_bytree=0.788,
        reg_alpha=0.031, reg_lambda=0.170, max_bin=63,
        lambdarank_norm=True, lambdarank_truncation_level=8,
        label_gain=list(LABEL_GAIN), bagging_freq=1, bagging_by_query=True,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def _feature_columns(architecture: str) -> list[str]:
    scores = list(ARCHITECTURES[architecture])
    fields: list[str] = []
    for score in scores:
        fields.extend((score, f"{score}__rank"))
    fields.extend((
        "path_entropy", "path_max_probability",
        *[f"frozen_path_p_{index:02d}" for index in range(8)],
        "causal_path_entropy",
        *[f"causal_state_q_{index:02d}" for index in range(4)],
        *[f"causal_path_p_{index:02d}" for index in range(8)],
    ))
    return fields


def _prepare(frame: pd.DataFrame, architecture: str) -> pd.DataFrame:
    result = frame.copy()
    score_fields = ARCHITECTURES[architecture]
    for field in score_fields:
        result[f"{field}__rank"] = _rank_pct(result, field)
    rank_fields = [f"{field}__rank" for field in score_fields]
    result["input_anchor_bps"] = result.loc[:, list(score_fields)].mean(axis=1)
    result["input_anchor_rank"] = result.loc[:, rank_fields].mean(axis=1)
    # A multi-input route is the union of its timestamp-local top-30% base
    # candidates.  A single-base route is exactly that base's top-30%.
    result["base_routed"] = result.loc[:, rank_fields].max(axis=1).ge(BASE_ROUTE_PERCENTILE)
    result["query_id"] = _query_id(result["__decision_ts__"])
    return result


def _month_sample(frame: pd.DataFrame, max_rows: int = 180_000) -> pd.DataFrame:
    """Deterministically sample whole queries, balanced by calendar month."""
    if len(frame) <= max_rows:
        return frame
    month = pd.to_datetime(frame["__decision_ts__"], utc=True).dt.to_period("M").astype(str)
    result: list[pd.DataFrame] = []
    per_month = max_rows // max(1, month.nunique())
    for label, part in frame.groupby(month, sort=True):
        sizes = part.groupby("query_id", sort=False).size().rename("size").reset_index()
        # Stable hash avoids looking at target values while selecting queries.
        sizes["h"] = sizes["query_id"].map(lambda value: int(hashlib.sha256(f"{SEED}|{label}|{value}".encode()).hexdigest()[:16], 16))
        sizes = sizes.sort_values("h", kind="stable")
        running = sizes["size"].cumsum()
        keep = sizes.loc[running.le(per_month), "query_id"]
        if keep.empty:
            keep = sizes.iloc[:1]["query_id"]
        result.append(part.loc[part["query_id"].isin(set(keep))])
    sampled = pd.concat(result, ignore_index=True)
    return sampled if len(sampled) <= max_rows + 2_000 else sampled.iloc[:max_rows].copy()


def _fit_predict(train: pd.DataFrame, held: pd.DataFrame, *, architecture: str, target: str) -> tuple[np.ndarray, dict[str, object]]:
    fields = _feature_columns(architecture)
    anchor = TARGETS[target]
    fit = train.copy()
    outcome = pd.to_numeric(fit["policy_net_bps"], errors="coerce")
    value = outcome - pd.to_numeric(fit[anchor], errors="coerce")
    complete = (
        np.isfinite(value.to_numpy(float))
        & fit["policy_path_valid"].fillna(False).astype(bool).to_numpy(bool)
        & fit.loc[:, fields].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy(bool)
    )
    fit = fit.loc[complete].copy()
    fit["grade"] = _grade(value.loc[fit.index])
    # LambdaRank cannot derive a gradient from singleton queries.  This is a
    # training-only eligibility rule, not a held-candidate filter.
    counts = fit.groupby("query_id", sort=False)["candidate_id"].transform("size")
    fit = fit.loc[counts.ge(2)].copy()
    fit = _month_sample(fit)
    fit = fit.sort_values("query_id", kind="stable")
    group = fit.groupby("query_id", sort=False).size().to_numpy(np.int32)
    if len(fit) < 1_000 or len(group) < 8:
        raise ValueError(f"insufficient strict prequential LambdaRank support for {architecture}/{target}")
    model = _model()
    xfit = fit.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    model.fit(xfit, fit["grade"].to_numpy(np.int32), group=group)
    score = np.full(len(held), np.nan, dtype=np.float32)
    finite = held.loc[:, fields].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy(bool)
    if finite.any():
        xheld = held.loc[finite, fields].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        score[finite] = model.predict(xheld).astype(np.float32)
    return score, {
        "train_rows": int(len(fit)), "train_queries": int(len(group)), "target_anchor": anchor,
        "feature_count": int(len(fields)), "held_feature_complete_rows": int(finite.sum()),
    }


def _tail_rows(frame: pd.DataFrame, arm: str, period: str) -> list[dict[str, object]]:
    actual = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    score = pd.to_numeric(frame["consensus_final_score"], errors="coerce").to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(score) & frame["policy_path_valid"].fillna(False).to_numpy(bool)
    rows: list[dict[str, object]] = []
    for tail in (0.01, 0.02, 0.03, 0.05):
        n = max(1, int(math.ceil(tail * int(valid.sum())))) if valid.any() else 0
        selected = actual[valid][np.argsort(score[valid], kind="stable")[-n:]] if n else np.array([], dtype=float)
        rows.append({"arm": arm, "period": period, "metric": f"top_{tail:.0%}_net_ev_bps", "selected_rows": int(len(selected)), "value": float(selected.mean()) if len(selected) else float("nan"), "net_sum_bps": float(selected.sum()) if len(selected) else 0.0})
    return rows


def _candidate_table(frame: pd.DataFrame, *, threshold_bps: float) -> pd.DataFrame:
    valid = (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_gross_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_exit_bar_15m"], errors="coerce"))
    )
    admitted = frame.loc[
        frame["base_routed"].fillna(False).astype(bool)
        & pd.to_numeric(frame["current_mc1_expected_bps"], errors="coerce").ge(float(threshold_bps))
        & pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="coerce").ge(float(threshold_bps))
        & valid
        & np.isfinite(pd.to_numeric(frame["consensus_final_score"], errors="coerce"))
    ].copy()
    if admitted.empty:
        return pd.DataFrame(columns=[
            "timestamp", "symbol", "candidate_id", "consensus_final_score", "policy_net_bps", "policy_gross_bps",
            "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
            "current_mc1_expected_bps", "bcf_mc1_expected_bps", "admission_threshold_bps",
        ])
    # The new consensus score controls auction priority.  MC1 retains solely
    # its frozen dual +50-bps admission authority.
    admitted["auction_rank"] = admitted.groupby("__decision_ts__", sort=False)["consensus_final_score"].rank(pct=True, method="first")
    decision = pd.to_datetime(admitted["__decision_ts__"], utc=True)
    exit_bar = pd.to_numeric(admitted["policy_exit_bar_15m"], errors="raise").astype(int)
    return pd.DataFrame({
        "timestamp": decision, "symbol": admitted["__symbol__"].astype(str),
        "candidate_id": admitted["candidate_id"].astype(str),
        "consensus_final_score": pd.to_numeric(admitted["consensus_final_score"], errors="raise"),
        "policy_net_bps": pd.to_numeric(admitted["policy_net_bps"], errors="raise"),
        "policy_gross_bps": pd.to_numeric(admitted["policy_gross_bps"], errors="raise"),
        "policy_exit_bar_15m": exit_bar,
        "policy_entry_price": pd.to_numeric(admitted["policy_entry_price"], errors="raise"),
        "policy_exit_price": pd.to_numeric(admitted["policy_exit_price"], errors="raise"),
        "policy_exit_reason": admitted["policy_exit_reason"].astype(str),
        "policy_cost_bps": pd.to_numeric(admitted.get("policy_cost_bps", 100.0), errors="coerce").fillna(100.0),
        "current_mc1_expected_bps": pd.to_numeric(admitted["current_mc1_expected_bps"], errors="raise"),
        "bcf_mc1_expected_bps": pd.to_numeric(admitted["bcf_mc1_expected_bps"], errors="raise"),
        "admission_threshold_bps": float(threshold_bps),
    })


def _replay_research_contract(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the declared global-auction contract without live dependencies.

    Candidate outcomes are already complete and valid here.  At each decision
    timestamp the routine closes prior positions first, ranks fresh candidates
    by the consensus score, then applies the frozen 2-entry / 8-position /
    one-symbol / 80%-wallet / 10%-margin-slot rules.
    """
    columns = [
        "candidate_id", "timestamp", "symbol", "accepted", "reject_reason", "exit_timestamp",
        "position_net_return", "position_gross_return", "margin_allocated", "entry_wallet", "exit_wallet",
        "policy_exit_reason", "consensus_final_score",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=["timestamp", "wallet"])
    data = candidates.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data["exit_timestamp"] = data["timestamp"] + pd.to_timedelta((pd.to_numeric(data["policy_exit_bar_15m"], errors="raise").astype(int) + 1) * 15, unit="min")
    data = data.sort_values(["timestamp", "consensus_final_score", "candidate_id"], ascending=[True, False, True], kind="stable")
    wallet = float(PORTFOLIO.initial_wallet)
    active: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    equity: list[dict[str, object]] = []

    def settle(until: pd.Timestamp) -> None:
        nonlocal wallet, active
        due = sorted((item for item in active if pd.Timestamp(item["exit_timestamp"]) <= until), key=lambda item: pd.Timestamp(item["exit_timestamp"]))
        for item in due:
            pnl = float(item["margin_allocated"]) * PORTFOLIO.leverage * float(item["position_net_return"])
            wallet = max(0.0, wallet + pnl)
            item["exit_wallet"] = wallet
            equity.append({"timestamp": item["exit_timestamp"], "wallet": wallet})
        active = [item for item in active if pd.Timestamp(item["exit_timestamp"]) > until]

    for timestamp, group in data.groupby("timestamp", sort=True):
        settle(pd.Timestamp(timestamp))
        accepted_this_timestamp = 0
        for _, row in group.iterrows():
            same_symbol = any(str(item["symbol"]) == str(row["symbol"]) for item in active)
            used_margin = sum(float(item["margin_allocated"]) for item in active)
            slot = PORTFOLIO.margin_slot_fraction * wallet
            remaining = PORTFOLIO.max_wallet_margin_fraction * wallet - used_margin
            reason: str | None = None
            if accepted_this_timestamp >= PORTFOLIO.max_new_entries_per_timestamp:
                reason = "timestamp_entry_cap"
            elif len(active) >= PORTFOLIO.max_concurrent_positions:
                reason = "concurrent_position_cap"
            elif same_symbol:
                reason = "symbol_concentration_cap"
            elif remaining + 1e-12 < slot:
                reason = "wallet_margin_cap"
            record = {
                "candidate_id": str(row["candidate_id"]), "timestamp": pd.Timestamp(timestamp), "symbol": str(row["symbol"]),
                "accepted": reason is None, "reject_reason": reason,
                "exit_timestamp": pd.Timestamp(row["exit_timestamp"]),
                "position_net_return": float(row["policy_net_bps"]) / 10_000.0,
                "position_gross_return": float(row["policy_gross_bps"]) / 10_000.0,
                "margin_allocated": slot if reason is None else 0.0, "entry_wallet": wallet,
                "exit_wallet": np.nan, "policy_exit_reason": str(row["policy_exit_reason"]),
                "consensus_final_score": float(row["consensus_final_score"]),
            }
            records.append(record)
            if reason is None:
                active.append(record)
                accepted_this_timestamp += 1
        equity.append({"timestamp": pd.Timestamp(timestamp), "wallet": wallet})
    settle(pd.Timestamp.max.tz_localize("UTC"))
    return pd.DataFrame(records, columns=columns), pd.DataFrame(equity, columns=["timestamp", "wallet"])


def _research_metrics(
    decisions: pd.DataFrame,
    equity: pd.DataFrame,
    arm: str,
    period: str,
    *,
    threshold_bps: float,
) -> dict[str, object]:
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy() if not decisions.empty else decisions
    net = pd.to_numeric(accepted.get("position_net_return", pd.Series(dtype=float)), errors="coerce") * 10_000.0
    timestamp = pd.to_datetime(accepted.get("timestamp", pd.Series(dtype="datetime64[ns, UTC]")), utc=True)
    monthly = pd.DataFrame({"month": timestamp.dt.strftime("%Y-%m"), "net": net}).groupby("month", sort=True)["net"].mean() if len(accepted) else pd.Series(dtype=float)
    weekly = pd.DataFrame({"week": timestamp.dt.strftime("%G-W%V"), "net": net}).groupby("week", sort=True)["net"].mean() if len(accepted) else pd.Series(dtype=float)
    wallet = pd.to_numeric(equity.get("wallet", pd.Series(dtype=float)), errors="coerce").dropna()
    drawdown = float((wallet / wallet.cummax() - 1.0).min()) if len(wallet) else 0.0
    return {
        "arm": arm, "period": period, "admission_threshold_bps": float(threshold_bps),
        "accepted_rows": int(len(accepted)), "realised_rows": int(len(accepted)), "outcome_coverage": 1.0 if len(accepted) else float("nan"),
        "net_ev_bps_per_realised_trade": float(net.mean()) if len(net) else float("nan"), "net_sum_bps_realised": float(net.sum()) if len(net) else 0.0,
        "net_ev_bps_per_selected_trade": float(net.mean()) if len(net) else float("nan"), "net_sum_bps_selected": float(net.sum()) if len(net) else 0.0,
        "worst_month_bps": float(monthly.min()) if len(monthly) else float("nan"), "worst_week_bps": float(weekly.min()) if len(weekly) else float("nan"),
        "positive_month_fraction": float(monthly.gt(0).mean()) if len(monthly) else float("nan"), "max_drawdown": drawdown,
        "final_wallet": float(wallet.iloc[-1]) if len(wallet) else PORTFOLIO.initial_wallet,
    }


def _portfolio_rows(
    frame: pd.DataFrame,
    arm: str,
    period: str,
    out: Path,
    *,
    threshold_bps: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    candidate = _candidate_table(frame, threshold_bps=threshold_bps)
    decisions, equity = _replay_research_contract(candidate)
    token = f"threshold_{float(threshold_bps):.0f}"
    decisions.to_parquet(out / f"{arm}__{period}__{token}__decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{arm}__{period}__{token}__equity.parquet", index=False, compression="zstd")
    metrics = _research_metrics(decisions, equity, arm, period, threshold_bps=threshold_bps)
    rows: list[dict[str, object]] = [metrics]
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy() if not decisions.empty else decisions
    if not accepted.empty:
        timestamp = pd.to_datetime(accepted["timestamp"], utc=True)
        net = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
        monthly = pd.DataFrame({"month": timestamp.dt.strftime("%Y-%m"), "net": net}).groupby("month", sort=True)["net"].agg(["count", "mean", "sum"])
        for month, item in monthly.iterrows():
            rows.append({"arm": arm, "period": period, "admission_threshold_bps": float(threshold_bps), "metric_type": "monthly", "month": month, "accepted_rows": int(item["count"]), "net_ev_bps": float(item["mean"]), "net_sum_bps": float(item["sum"])})
        symbols = accepted.get("symbol", pd.Series(dtype=str)).astype(str)
        shares = symbols.value_counts(normalize=True)
        rows.append({"arm": arm, "period": period, "admission_threshold_bps": float(threshold_bps), "metric_type": "concentration", "symbol_hhi": float((shares ** 2).sum()), "top_symbol_share": float(shares.iloc[0]) if len(shares) else float("nan")})
    return rows, decisions


def _evaluate(frame: pd.DataFrame, arm: str, period: str, out: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = _tail_rows(frame, arm, period)
    counts: list[dict[str, object]] = []
    for threshold_bps in MC1_THRESHOLDS_BPS:
        dual = (
            pd.to_numeric(frame["current_mc1_expected_bps"], errors="coerce").ge(float(threshold_bps))
            & pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="coerce").ge(float(threshold_bps))
        )
        counts.append({
            "arm": arm, "period": period, "admission_threshold_bps": float(threshold_bps),
            "scored_rows": int(len(frame)), "base_routed_rows": int(frame["base_routed"].sum()),
            "dual_mc1_admitted_rows": int((dual & frame["base_routed"]).sum()),
        })
        portfolio, _ = _portfolio_rows(frame, arm, period, out, threshold_bps=float(threshold_bps))
        rows.extend(portfolio)
    return rows, counts


def _score_month(prepared: pd.DataFrame, month: pd.Timestamp, *, architecture: str, target: str) -> tuple[pd.DataFrame, dict[str, object]]:
    end = month + pd.offsets.MonthBegin(1)
    train_start = month - pd.DateOffset(months=3)
    train = prepared.loc[
        prepared["__decision_ts__"].ge(train_start) & prepared["__decision_ts__"].lt(month)
        & prepared["policy_label_available_ts"].lt(month)
    ].copy()
    held = prepared.loc[prepared["__decision_ts__"].ge(month) & prepared["__decision_ts__"].lt(end)].copy()
    if held.empty:
        raise ValueError(f"no held rows for {month}")
    score, audit = _fit_predict(train, held, architecture=architecture, target=target)
    held["consensus_raw_score"] = score
    held["consensus_rank"] = _rank_pct(held, "consensus_raw_score")
    held["consensus_final_score"] = 0.75 * held["input_anchor_rank"] + 0.25 * held["consensus_rank"]
    held["architecture"] = architecture
    held["target"] = target
    held["score_month"] = month
    audit.update({"architecture": architecture, "target": target, "train_start": train_start, "held_start": month, "held_end_exclusive": end, "held_rows": int(len(held))})
    return held, audit


def _target_free_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    prohibited = {
        "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m", "policy_entry_price",
        "policy_exit_price", "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
        "current_mc1_expected_bps", "bcf_mc1_expected_bps",
    }
    return frame.loc[:, [column for column in frame.columns if column not in prohibited]].copy()


def _selection_table(metrics: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    # Tail diagnostics intentionally have no threshold or accepted-row field;
    # they are descriptive only and must never enter admission-robust model
    # selection.
    overall = metrics.loc[
        (metrics.get("period") == "2025_juldec")
        & metrics.get("metric_type", pd.Series(index=metrics.index, dtype=str)).isna()
        & metrics.get("accepted_rows", pd.Series(index=metrics.index, dtype=float)).notna()
    ].copy()
    if overall.empty:
        overall = metrics.loc[metrics.get("period").eq("2025_juldec")].copy()
    # `_metrics` has one global record per arm / threshold.  Select against
    # the unweighted 30/50-threshold average, which prevents a challenger
    # from advancing solely because it happens to fit one admission cut.
    metric_columns = [
        "accepted_rows", "net_sum_bps_realised", "net_ev_bps_per_realised_trade",
        "worst_month_bps", "worst_week_bps", "max_drawdown", "final_wallet",
    ]
    by_threshold = overall.loc[:, ["arm", "admission_threshold_bps", *metric_columns]].copy()
    by_threshold = by_threshold.sort_values(["arm", "admission_threshold_bps"], kind="stable")
    expected = set(float(value) for value in MC1_THRESHOLDS_BPS)
    support = by_threshold.groupby("arm", sort=False)["admission_threshold_bps"].agg(lambda values: set(map(float, values)))
    if not support.map(lambda values: values == expected).all():
        missing = support.loc[~support.map(lambda values: values == expected)].to_dict()
        raise AssertionError(f"selection requires both MC1 thresholds for every arm: {missing}")
    selected = by_threshold.groupby("arm", sort=False)[metric_columns].mean().reset_index()
    selected = selected.rename(columns={column: f"robust_avg_{column}" for column in metric_columns})
    wide = by_threshold.pivot(index="arm", columns="admission_threshold_bps", values=metric_columns)
    wide.columns = [f"{metric}_threshold_{float(threshold):.0f}" for metric, threshold in wide.columns]
    selected = selected.merge(wide.reset_index(), on="arm", how="left", validate="one_to_one")
    count_columns = ["base_routed_rows", "dual_mc1_admitted_rows"]
    counts25 = counts.loc[counts["period"].eq("2025_juldec"), ["arm", "admission_threshold_bps", *count_columns]].copy()
    count_mean = counts25.groupby("arm", sort=False)[count_columns].mean().reset_index().rename(columns={column: f"robust_avg_{column}" for column in count_columns})
    selected = selected.merge(count_mean, on="arm", how="left", validate="one_to_one")
    selected = selected.sort_values(
        ["robust_avg_net_sum_bps_realised", "robust_avg_net_ev_bps_per_realised_trade", "robust_avg_worst_week_bps"],
        ascending=False, kind="stable",
    ).reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["selected_for_2026"] = selected["selection_rank"].le(2)
    return selected


def _run_months(
    population: pd.DataFrame,
    months: Sequence[pd.Timestamp],
    arms: Sequence[tuple[str, str]],
    *,
    out: Path,
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_root = out / f"{stage}_target_free_predictions"
    prediction_root.mkdir(parents=True, exist_ok=False)
    audit_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    for architecture, target in arms:
        prepared = _prepare(population, architecture)
        arm = f"{architecture}__{target}"
        pieces: list[pd.DataFrame] = []
        for month in months:
            held, audit = _score_month(prepared, month, architecture=architecture, target=target)
            audit_rows.append(audit)
            target_free = _target_free_prediction(held)
            path = prediction_root / f"architecture={architecture}" / f"target={target}" / f"month={month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            target_free.to_parquet(path, index=False, compression="zstd")
            pieces.append(held)
            print(json.dumps({"event": "scored", "stage": stage, "arm": arm, "month": str(month), **audit}, default=str), flush=True)
        frame = pd.concat(pieces, ignore_index=True)
        metrics, counts = _evaluate(frame, arm, "2025_juldec" if stage == "selection_2025" else "2026_aprjul", out)
        metrics_rows.extend(metrics); count_rows.extend(counts)
    return pd.DataFrame(audit_rows), pd.DataFrame(metrics_rows), pd.DataFrame(count_rows)


def run(sources: Sources, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    population = _load_population(sources)
    policy, mc1 = _load_policy_and_mc1(sources)
    population = population.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    population = population.merge(mc1, on="candidate_id", how="left", validate="one_to_one")
    if population["policy_path_valid"].isna().mean() > 0.80:
        raise AssertionError("canonical policy coverage is unexpectedly absent")
    # Policy is a target/evaluation sidecar.  Its fields are prohibited from
    # target-free held prediction artifacts by `_target_free_prediction`.
    all_arms = tuple((architecture, target) for architecture in ARCHITECTURES for target in TARGETS)
    audit25, metrics25, counts25 = _run_months(population, EVAL_MONTHS_2025, all_arms, out=out, stage="selection_2025")
    selection = _selection_table(metrics25, counts25)
    selection.to_parquet(out / "selection_2025.parquet", index=False, compression="zstd")
    selected_arms = tuple(tuple(str(value).split("__", 1)) for value in selection.loc[selection["selected_for_2026"], "arm"])
    if not selected_arms:
        raise AssertionError("no 2025 consensus arm selected for 2026")
    audit26, metrics26, counts26 = _run_months(population, EVAL_MONTHS_2026, selected_arms, out=out, stage="portability_2026")
    pd.concat([audit25.assign(stage="selection_2025"), audit26.assign(stage="portability_2026")], ignore_index=True).to_parquet(out / "walkforward_fit_audit.parquet", index=False, compression="zstd")
    pd.concat([metrics25.assign(stage="selection_2025"), metrics26.assign(stage="portability_2026")], ignore_index=True).to_parquet(out / "portfolio_and_tail_metrics.parquet", index=False, compression="zstd")
    pd.concat([counts25.assign(stage="selection_2025"), counts26.assign(stage="portability_2026")], ignore_index=True).to_parquet(out / "admission_counts.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "scope": "offline research only; no live/canonical mutation",
        "upstream_selection": "development-only strict-OOS top-2 direct configurations versus B0",
        "selected_direct_arms": SELECTED_DIRECT_ARMS,
        "architectures": ARCHITECTURES, "targets": TARGETS,
        "training": "monthly walk-forward: preceding 3 calendar months, labels fully available before held-month start; source base/archetype predictions are strict-OOS",
        "evaluation": {"selection": "2025-07 through 2025-12", "portability": "2026-04 through 2026-07; only 2025-selected top two arms", "selection_rule": "unweighted mean across dual-MC1 30 and 50 bps thresholds"},
        "dual_admission": {"thresholds_bps": list(MC1_THRESHOLDS_BPS), "rule": "current MC1 >= threshold AND BCF MC1 >= threshold"},
        "portfolio": "narrow offline global chronological constraint mirror: 8 concurrent, 2 entries/timestamp, 1 asset, 80% margin, 10% slots, 7x; full production-adapter match still required before promotion",
        "sources": {key: str(value.resolve()) for key, value in vars(sources).items()},
        "source_sha256": {key: _sha256([value]) for key, value in vars(sources).items()},
        "held_prediction_contract": "target-free; policy/MC1 joins occur only after OOS scores are persisted",
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--causal-joint", type=Path, required=True)
    parser.add_argument("--current-mc1", type=Path, required=True)
    parser.add_argument("--bcf-mc1", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    sources = Sources(
        stage1=args.stage1.resolve(), direct=args.direct.resolve(), stage2=args.stage2.resolve(), causal_joint=args.causal_joint.resolve(),
        current_mc1=args.current_mc1.resolve(), bcf_mc1=args.bcf_mc1.resolve(), policy=args.policy.resolve(),
    )
    print(json.dumps({"status": "ok", "out": str(run(sources, args.out.resolve()))}))


if __name__ == "__main__":
    main()
