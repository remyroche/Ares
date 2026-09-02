#!/usr/bin/env python3
"""Strict-OOS long-only base-to-meta conversion ablation, research only.

This experiment tests whether a consensus layer adds value *conditional on the
canonical B0 base route*, rather than by introducing a broad union of direct
head routes.  It deliberately keeps the following contracts fixed:

* B0 timestamp-local top-30% route;
* strict-OOS upstream B0 / efficiency / timing predictions;
* three preceding calendar months of fully resolved policy labels for each
  LambdaRank fit;
* frozen dual current/BCF MC1 admission at both +30 and +50 bps; and
* one chronological constrained research portfolio.

Stage A isolates the target and information question:

* ``rich_policy_net_bps``: direct rich-policy net utility; versus
* ``policy_residual_bps``: the simpler incumbent consensus target,
  ``rich_policy_net_bps - B0 predicted policy value``.

Both are tested with B0-only inputs and with B0 plus direct-head disagreement
geometry.  Stage B takes the best Stage-A parent selected on 2025 Jul--Dec and
tests a demotion-only severe-residual overlay and a bounded recovery overlay.
Neither overlay changes the B0 route or creates an MC1 admission; they can only
re-rank already-routed candidates.  April--July 2026 is confirmation only.

The runner is offline research only.  It does not mutate canonical, live,
admission, sizing, portfolio, policy, or exchange artifacts.
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

from scripts.run_strict_r3_long_supportive_consensus_walkforward import (  # noqa: E402
    BASE_ROUTE_PERCENTILE,
    MC1_THRESHOLDS_BPS,
    PORTFOLIO,
    QUERY_HOURS,
    RESIDUAL_BANDS,
    SEED,
    Sources,
    _candidate_table,
    _load_policy_and_mc1,
    _load_population,
    _merge_exact,
    _month_sample,
    _query_id,
    _rank_pct,
    _replay_research_contract,
)


SCHEMA = "strict_r3_long_base_meta_conversion_ablation_v1"
EVAL_MONTHS_2025 = tuple(pd.date_range("2025-07-01", "2025-12-01", freq="MS", tz="UTC"))
EVAL_MONTHS_2026 = tuple(pd.date_range("2026-04-01", "2026-07-01", freq="MS", tz="UTC"))
LABEL_GAIN = (0.0, 1.0, 2.0, 4.0, 7.0)
DEMODE_RESIDUAL_BPS = -100.0
RECOVERY_RESIDUAL_BPS = 90.0
DEMODE_AUTHORITY = 0.15
RECOVERY_AUTHORITY = 0.05


@dataclass(frozen=True)
class Arm:
    name: str
    target: str
    features: str
    overlay: str


CONTROL = Arm("M0_b0_control", "none", "b0", "none")
STAGE_A = (
    Arm("M1_simple_residual_b0", "simple_residual", "b0", "none"),
    Arm("M1_rich_policy_net_b0", "rich_policy_net", "b0", "none"),
    Arm("M2_simple_residual_disagreement", "simple_residual", "disagreement", "none"),
    Arm("M2_rich_policy_net_disagreement", "rich_policy_net", "disagreement", "none"),
)


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


def _read_direct_components(root: Path) -> pd.DataFrame:
    """Read the strict-OOS base/efficiency/timing components once.

    The selected direct blend persists all three component predictions.  Those
    are target-free base outputs, unlike the realised column in the same source
    files, which is intentionally never requested.
    """
    pieces: list[pd.DataFrame] = []
    columns = [
        "candidate_id", "__decision_ts__", "base_bps", "efficiency_bps", "timing_bps",
    ]
    for folder in sorted((root / "oof_prediction_parts").glob("fold=*")):
        pieces.append(_read_matching_part(folder, "S3_direct_efficiency_time_base_equal", columns))
    frame = pd.concat(pieces, ignore_index=True)
    return frame.rename(columns={"base_bps": "direct_base_bps"})


def _load_frame(source: Sources) -> pd.DataFrame:
    frame = _load_population(source)
    direct = _read_direct_components(source.direct)
    frame = _merge_exact(frame, direct, name="direct_components")
    policy, mc1 = _load_policy_and_mc1(source)
    frame = frame.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    frame = frame.merge(mc1, on="candidate_id", how="left", validate="one_to_one")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("duplicate candidate identity after source joins")
    return frame


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    score_fields = ("b0_bps", "efficiency_bps", "timing_bps")
    for column in score_fields:
        result[f"{column}__rank"] = _rank_pct(result, column)
    result["direct_et_bps"] = 0.5 * (result["efficiency_bps"] + result["timing_bps"])
    result["direct_etb_bps"] = (result["b0_bps"] + result["efficiency_bps"] + result["timing_bps"]) / 3.0
    result["b0_routed"] = result["b0_bps__rank"].ge(BASE_ROUTE_PERCENTILE)
    result["query_id"] = _query_id(result["__decision_ts__"])
    result["delta_e_minus_t"] = result["efficiency_bps"] - result["timing_bps"]
    result["delta_e_minus_b0"] = result["efficiency_bps"] - result["b0_bps"]
    result["delta_t_minus_b0"] = result["timing_bps"] - result["b0_bps"]
    matrix = result.loc[:, list(score_fields)].to_numpy(np.float64)
    result["score_std"] = np.nanstd(matrix, axis=1)
    result["score_min"] = np.nanmin(matrix, axis=1)
    result["score_max"] = np.nanmax(matrix, axis=1)
    result["score_range"] = result["score_max"] - result["score_min"]
    result["rich_policy_net_bps"] = pd.to_numeric(result["policy_net_bps"], errors="coerce")
    result["simple_residual_bps"] = result["rich_policy_net_bps"] - pd.to_numeric(result["b0_bps"], errors="coerce")
    return result


def _feature_columns(mode: str) -> list[str]:
    b0 = ["b0_bps", "b0_bps__rank"]
    if mode == "b0":
        return b0
    if mode != "disagreement":
        raise ValueError(f"unknown feature mode {mode!r}")
    return b0 + [
        "efficiency_bps", "efficiency_bps__rank", "timing_bps", "timing_bps__rank",
        "direct_et_bps", "direct_etb_bps",
        "delta_e_minus_t", "delta_e_minus_b0", "delta_t_minus_b0",
        "score_std", "score_min", "score_max", "score_range",
    ]


def _target_value(frame: pd.DataFrame, target: str) -> pd.Series:
    if target == "rich_policy_net":
        return pd.to_numeric(frame["rich_policy_net_bps"], errors="coerce")
    if target == "simple_residual":
        return pd.to_numeric(frame["simple_residual_bps"], errors="coerce")
    raise ValueError(f"unknown target {target!r}")


def _grade(value: pd.Series) -> np.ndarray:
    return np.digitize(pd.to_numeric(value, errors="coerce").to_numpy(float), RESIDUAL_BANDS, right=True).astype(np.int32)


def _ranker() -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 2, 5],
        n_estimators=220, learning_rate=0.03, max_depth=4, num_leaves=31,
        min_child_samples=893, subsample=0.867, colsample_bytree=0.788,
        reg_alpha=0.031, reg_lambda=0.170, max_bin=63,
        lambdarank_norm=True, lambdarank_truncation_level=8,
        label_gain=list(LABEL_GAIN), bagging_freq=1, bagging_by_query=True,
        random_state=SEED, n_jobs=-1, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )


def _classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=160, learning_rate=0.03,
        max_depth=3, num_leaves=15, min_child_samples=893,
        subsample=0.867, colsample_bytree=0.788,
        reg_alpha=0.031, reg_lambda=0.170, max_bin=63,
        bagging_freq=1, random_state=SEED, n_jobs=-1, verbosity=-1,
        deterministic=True, force_col_wise=True,
        class_weight="balanced",
    )


def _eligible_train(train: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    complete = (
        train["b0_routed"].fillna(False).astype(bool)
        & train["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(train["rich_policy_net_bps"], errors="coerce"))
        & train.loc[:, fields].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    )
    fit = train.loc[complete].copy()
    counts = fit.groupby("query_id", sort=False)["candidate_id"].transform("size")
    return fit.loc[counts.ge(2)].copy()


def _fit_ranker(train: pd.DataFrame, held: pd.DataFrame, arm: Arm) -> tuple[np.ndarray, dict[str, object]]:
    fields = _feature_columns(arm.features)
    fit = _eligible_train(train, fields)
    target = _target_value(fit, arm.target)
    fit = fit.loc[np.isfinite(target.to_numpy(float))].copy()
    fit["grade"] = _grade(_target_value(fit, arm.target))
    fit = _month_sample(fit)
    fit = fit.sort_values("query_id", kind="stable")
    group = fit.groupby("query_id", sort=False).size().to_numpy(np.int32)
    if len(fit) < 1_000 or len(group) < 8:
        raise ValueError(f"insufficient training support for {arm.name}")
    model = _ranker()
    model.fit(
        fit.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32),
        fit["grade"].to_numpy(np.int32), group=group,
    )
    score = np.full(len(held), np.nan, dtype=np.float32)
    usable = held["b0_routed"].fillna(False).astype(bool).to_numpy(bool) & held.loc[:, fields].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy(bool)
    if usable.any():
        score[usable] = model.predict(held.loc[usable, fields].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)).astype(np.float32)
    return score, {
        "train_rows": int(len(fit)), "train_queries": int(len(group)),
        "feature_count": int(len(fields)), "held_usable_rows": int(usable.sum()),
    }


def _fit_binary_overlay(train: pd.DataFrame, held: pd.DataFrame, features: str, *, kind: str) -> tuple[np.ndarray, dict[str, object]]:
    fields = _feature_columns(features)
    fit = _eligible_train(train, fields)
    residual = pd.to_numeric(fit["simple_residual_bps"], errors="coerce")
    if kind == "demotion":
        label = residual.le(DEMODE_RESIDUAL_BPS)
    elif kind == "recovery":
        label = residual.ge(RECOVERY_RESIDUAL_BPS)
    else:
        raise ValueError(kind)
    finite = np.isfinite(residual.to_numpy(float))
    fit = fit.loc[finite].copy()
    label = label.loc[fit.index].astype(np.int8)
    # A degenerate fold has no authority: zero penalty/bonus, never a fitted
    # held-period fallback.
    score = np.zeros(len(held), dtype=np.float32)
    if label.nunique() < 2 or len(fit) < 1_000:
        return score, {"overlay_train_rows": int(len(fit)), "overlay_positive_rows": int(label.sum()), "overlay_fitted": False}
    fit = _month_sample(fit)
    label = label.loc[fit.index].astype(np.int8)
    model = _classifier()
    model.fit(fit.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32), label.to_numpy(np.int8))
    usable = held["b0_routed"].fillna(False).astype(bool).to_numpy(bool) & held.loc[:, fields].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy(bool)
    if usable.any():
        score[usable] = model.predict_proba(held.loc[usable, fields].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32))[:, 1].astype(np.float32)
    return score, {"overlay_train_rows": int(len(fit)), "overlay_positive_rows": int(label.sum()), "overlay_fitted": True}


def _score_month(prepared: pd.DataFrame, month: pd.Timestamp, arm: Arm) -> tuple[pd.DataFrame, dict[str, object]]:
    end = month + pd.offsets.MonthBegin(1)
    train_start = month - pd.DateOffset(months=3)
    train = prepared.loc[
        prepared["__decision_ts__"].ge(train_start) & prepared["__decision_ts__"].lt(month)
        & prepared["policy_label_available_ts"].lt(month)
    ].copy()
    held = prepared.loc[prepared["__decision_ts__"].ge(month) & prepared["__decision_ts__"].lt(end)].copy()
    if held.empty:
        raise ValueError(f"no held population for {month}")
    held["base_routed"] = held["b0_routed"]
    if arm == CONTROL:
        held["meta_raw_score"] = held["b0_bps"]
        held["meta_rank"] = held["b0_bps__rank"]
        held["consensus_final_score"] = held["b0_bps__rank"]
        audit: dict[str, object] = {"train_rows": 0, "train_queries": 0, "feature_count": 0, "held_usable_rows": int(held["b0_routed"].sum())}
    else:
        raw, audit = _fit_ranker(train, held, arm)
        held["meta_raw_score"] = raw
        held["meta_rank"] = _rank_pct(held, "meta_raw_score")
        held["consensus_final_score"] = 0.75 * held["b0_bps__rank"] + 0.25 * held["meta_rank"]
        if arm.overlay in {"demotion", "demotion_recovery"}:
            demotion, overlay_audit = _fit_binary_overlay(train, held, arm.features, kind="demotion")
            held["demotion_probability"] = demotion
            held["demotion_rank"] = _rank_pct(held, "demotion_probability")
            held["consensus_final_score"] = held["consensus_final_score"] - DEMODE_AUTHORITY * held["demotion_rank"].fillna(0.0)
            audit.update({f"demotion_{key}": value for key, value in overlay_audit.items()})
        else:
            held["demotion_probability"] = 0.0
            held["demotion_rank"] = 0.0
        if arm.overlay == "demotion_recovery":
            recovery, overlay_audit = _fit_binary_overlay(train, held, arm.features, kind="recovery")
            held["recovery_probability"] = recovery
            held["recovery_rank"] = _rank_pct(held, "recovery_probability")
            held["consensus_final_score"] = held["consensus_final_score"] + RECOVERY_AUTHORITY * held["recovery_rank"].fillna(0.0)
            audit.update({f"recovery_{key}": value for key, value in overlay_audit.items()})
        else:
            held["recovery_probability"] = 0.0
            held["recovery_rank"] = 0.0
    held["arm"] = arm.name
    held["score_month"] = month
    audit.update({"arm": arm.name, "target": arm.target, "features": arm.features, "overlay": arm.overlay, "train_start": train_start, "held_start": month, "held_end_exclusive": end, "held_rows": int(len(held))})
    return held, audit


def _target_free(frame: pd.DataFrame) -> pd.DataFrame:
    prohibited = {
        "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m", "policy_entry_price",
        "policy_exit_price", "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
        "rich_policy_net_bps", "simple_residual_bps", "current_mc1_expected_bps", "bcf_mc1_expected_bps",
    }
    return frame.loc[:, [column for column in frame.columns if column not in prohibited]].copy()


def _tail_metrics(frame: pd.DataFrame, arm: str, period: str) -> list[dict[str, object]]:
    actual = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    score = pd.to_numeric(frame["consensus_final_score"], errors="coerce").to_numpy(float)
    valid = (
        frame["b0_routed"].fillna(False).to_numpy(bool)
        & frame["policy_path_valid"].fillna(False).to_numpy(bool)
        & np.isfinite(actual) & np.isfinite(score)
    )
    rows: list[dict[str, object]] = []
    for tail in (0.01, 0.02, 0.03, 0.05):
        n = max(1, int(math.ceil(tail * int(valid.sum())))) if valid.any() else 0
        selected = actual[valid][np.argsort(score[valid], kind="stable")[-n:]] if n else np.array([], dtype=float)
        rows.append({
            "arm": arm, "period": period, "metric_type": "routed_global_tail", "metric": f"top_{tail:.0%}_net_ev_bps",
            "selected_rows": int(len(selected)), "value": float(selected.mean()) if len(selected) else float("nan"),
            "net_sum_bps": float(selected.sum()) if len(selected) else 0.0,
        })
    return rows


def _portfolio_metrics(decisions: pd.DataFrame, equity: pd.DataFrame, arm: str, period: str, threshold: float) -> list[dict[str, object]]:
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy() if not decisions.empty else decisions
    net = pd.to_numeric(accepted.get("position_net_return", pd.Series(dtype=float)), errors="coerce") * 10_000.0
    timestamp = pd.to_datetime(accepted.get("timestamp", pd.Series(dtype="datetime64[ns, UTC]")), utc=True)
    monthly = pd.DataFrame({"month": timestamp.dt.strftime("%Y-%m"), "net": net}).groupby("month", sort=True)["net"].agg(["count", "mean", "sum"]) if len(accepted) else pd.DataFrame()
    weekly = pd.DataFrame({"week": timestamp.dt.strftime("%G-W%V"), "net": net}).groupby("week", sort=True)["net"].mean() if len(accepted) else pd.Series(dtype=float)
    wallet = pd.to_numeric(equity.get("wallet", pd.Series(dtype=float)), errors="coerce").dropna()
    max_dd = float((wallet / wallet.cummax() - 1.0).min()) if len(wallet) else 0.0
    rows: list[dict[str, object]] = [{
        "arm": arm, "period": period, "metric_type": "portfolio", "admission_threshold_bps": float(threshold),
        "accepted_rows": int(len(accepted)), "net_ev_bps_per_trade": float(net.mean()) if len(net) else float("nan"),
        "net_sum_bps": float(net.sum()) if len(net) else 0.0,
        "worst_month_bps": float(monthly["mean"].min()) if len(monthly) else float("nan"),
        "worst_week_bps": float(weekly.min()) if len(weekly) else float("nan"),
        "positive_month_fraction": float(monthly["mean"].gt(0).mean()) if len(monthly) else float("nan"),
        "max_drawdown": max_dd, "final_wallet": float(wallet.iloc[-1]) if len(wallet) else float(PORTFOLIO.initial_wallet),
    }]
    for month, item in monthly.iterrows():
        rows.append({
            "arm": arm, "period": period, "metric_type": "monthly", "admission_threshold_bps": float(threshold),
            "month": month, "accepted_rows": int(item["count"]), "net_ev_bps": float(item["mean"]), "net_sum_bps": float(item["sum"]),
        })
    if not accepted.empty:
        shares = accepted["symbol"].astype(str).value_counts(normalize=True)
        rows.append({"arm": arm, "period": period, "metric_type": "concentration", "admission_threshold_bps": float(threshold), "symbol_hhi": float((shares ** 2).sum()), "top_symbol_share": float(shares.iloc[0])})
    return rows


def _evaluate(frame: pd.DataFrame, arm: Arm, period: str, out: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = _tail_metrics(frame, arm.name, period)
    counts: list[dict[str, object]] = []
    # `_candidate_table` treats `base_routed` as route eligibility and leaves
    # MC1 as a frozen admission authority.  The consensus score only controls
    # the auction ordering of already admitted B0 candidates.
    for threshold in MC1_THRESHOLDS_BPS:
        dual = (
            pd.to_numeric(frame["current_mc1_expected_bps"], errors="coerce").ge(float(threshold))
            & pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="coerce").ge(float(threshold))
        )
        counts.append({
            "arm": arm.name, "period": period, "admission_threshold_bps": float(threshold),
            "scored_rows": int(len(frame)), "b0_routed_rows": int(frame["b0_routed"].sum()),
            "dual_mc1_admitted_rows": int((dual & frame["b0_routed"]).sum()),
        })
        candidate = _candidate_table(frame, threshold_bps=float(threshold))
        decisions, equity = _replay_research_contract(candidate)
        token = f"threshold_{int(threshold)}"
        decisions.to_parquet(out / "decisions" / f"{arm.name}__{period}__{token}.parquet", index=False, compression="zstd")
        equity.to_parquet(out / "equity" / f"{arm.name}__{period}__{token}.parquet", index=False, compression="zstd")
        rows.extend(_portfolio_metrics(decisions, equity, arm.name, period, float(threshold)))
    return rows, counts


def _run_arms(
    prepared: pd.DataFrame,
    arms: Sequence[Arm],
    months: Sequence[pd.Timestamp],
    *,
    stage: str,
    period: str,
    out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_free_root = out / f"{stage}_target_free_predictions"
    target_free_root.mkdir(parents=True, exist_ok=False)
    audit_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    for arm in arms:
        pieces: list[pd.DataFrame] = []
        for month in months:
            held, audit = _score_month(prepared, month, arm)
            audit_rows.append(audit)
            path = target_free_root / f"arm={arm.name}" / f"month={month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            _target_free(held).to_parquet(path, index=False, compression="zstd")
            pieces.append(held)
            print(json.dumps({"event": "scored", "stage": stage, "arm": arm.name, "month": str(month), **audit}, default=str), flush=True)
        frame = pd.concat(pieces, ignore_index=True)
        metrics, counts = _evaluate(frame, arm, period, out)
        metrics_rows.extend(metrics); count_rows.extend(counts)
    return pd.DataFrame(audit_rows), pd.DataFrame(metrics_rows), pd.DataFrame(count_rows)


def _portfolio_summary(metrics: pd.DataFrame, period: str) -> pd.DataFrame:
    rows = metrics.loc[(metrics["period"].eq(period)) & (metrics["metric_type"].eq("portfolio"))].copy()
    columns = ["accepted_rows", "net_ev_bps_per_trade", "net_sum_bps", "worst_month_bps", "worst_week_bps", "max_drawdown", "final_wallet"]
    expected = set(map(float, MC1_THRESHOLDS_BPS))
    support = rows.groupby("arm", sort=False)["admission_threshold_bps"].agg(lambda values: set(map(float, values)))
    if not support.map(lambda values: values == expected).all():
        raise AssertionError(f"each arm requires both frozen MC1 thresholds: {support.to_dict()}")
    result = rows.groupby("arm", sort=False)[columns].mean().reset_index().rename(columns={column: f"robust_avg_{column}" for column in columns})
    result = result.sort_values(["robust_avg_net_sum_bps", "robust_avg_net_ev_bps_per_trade", "robust_avg_worst_week_bps"], ascending=False, kind="stable").reset_index(drop=True)
    result["selection_rank"] = np.arange(1, len(result) + 1)
    return result


def _stage_b_arms(parent: Arm) -> tuple[Arm, Arm]:
    return (
        Arm("M3_" + parent.name.removeprefix("M2_") + "_demotion", parent.target, parent.features, "demotion"),
        Arm("M4_" + parent.name.removeprefix("M2_") + "_demotion_recovery", parent.target, parent.features, "demotion_recovery"),
    )


def _causality_audit(prepared: pd.DataFrame, audits: pd.DataFrame) -> dict[str, object]:
    if (pd.to_datetime(prepared["policy_label_available_ts"], utc=True, errors="coerce") < pd.to_datetime(prepared["__decision_ts__"], utc=True)).any():
        raise AssertionError("policy label availability precedes decision timestamp")
    if audits.empty or audits.loc[audits["arm"].ne(CONTROL.name), "held_usable_rows"].le(0).any():
        raise AssertionError("a meta arm has no held feature-complete B0-routed rows")
    return {
        "base_route": "B0 timestamp-local rank >= 70th percentile; direct-only union disabled",
        "meta_training": "preceding 3 calendar months; policy labels resolved before held-month start",
        "held_predictions": "persisted target-free before MC1/policy portfolio evaluation",
        "mc1": "frozen current and BCF maps, dual thresholds +30/+50 bps; no refit",
        "overlays": "strict-prior training only; bounded reranking inside B0 route; never create admission",
        "portfolio": "one global chronological constrained research mirror",
    }


def run(sources: Sources, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "decisions").mkdir(); (out / "equity").mkdir()
    prepared = _prepare(_load_frame(sources))
    audit_a, metrics_a, counts_a = _run_arms(
        prepared, (CONTROL, *STAGE_A), EVAL_MONTHS_2025,
        stage="selection_2025", period="2025_juldec", out=out,
    )
    summary_a = _portfolio_summary(metrics_a, "2025_juldec")
    stage_a_noncontrol = summary_a.loc[summary_a["arm"].ne(CONTROL.name)].copy()
    parent_name = str(stage_a_noncontrol.iloc[0]["arm"])
    parents = {arm.name: arm for arm in STAGE_A}
    parent = parents[parent_name]
    stage_b = _stage_b_arms(parent)
    audit_b, metrics_b, counts_b = _run_arms(
        prepared, stage_b, EVAL_MONTHS_2025,
        stage="selection_2025_overlays", period="2025_juldec", out=out,
    )
    summary_full = _portfolio_summary(pd.concat([metrics_a, metrics_b], ignore_index=True), "2025_juldec")
    # The frozen B0 control always travels to confirmation, independently of
    # its development selection rank.  A challenger cannot claim portability
    # without a contemporaneous matched control on identical held rows.
    challenger_names = summary_full.loc[summary_full["arm"].ne(CONTROL.name) & summary_full["selection_rank"].le(3), "arm"].tolist()[:2]
    confirmation_names = [CONTROL.name, *challenger_names]
    all_arms = {CONTROL.name: CONTROL, **{arm.name: arm for arm in STAGE_A}, **{arm.name: arm for arm in stage_b}}
    confirmation = tuple(all_arms[name] for name in confirmation_names)
    audit_c, metrics_c, counts_c = _run_arms(
        prepared, confirmation, EVAL_MONTHS_2026,
        stage="portability_2026", period="2026_aprjul", out=out,
    )
    audits = pd.concat([audit_a.assign(stage="stage_a_2025"), audit_b.assign(stage="stage_b_2025"), audit_c.assign(stage="confirmation_2026")], ignore_index=True)
    metrics = pd.concat([metrics_a.assign(stage="stage_a_2025"), metrics_b.assign(stage="stage_b_2025"), metrics_c.assign(stage="confirmation_2026")], ignore_index=True)
    counts = pd.concat([counts_a.assign(stage="stage_a_2025"), counts_b.assign(stage="stage_b_2025"), counts_c.assign(stage="confirmation_2026")], ignore_index=True)
    selection = summary_full.copy()
    selection["selected_for_2026"] = selection["arm"].isin(confirmation_names)
    selection.to_parquet(out / "selection_2025.parquet", index=False, compression="zstd")
    audits.to_parquet(out / "walkforward_fit_audit.parquet", index=False, compression="zstd")
    metrics.to_parquet(out / "portfolio_tail_monthly_metrics.parquet", index=False, compression="zstd")
    counts.to_parquet(out / "admission_counts.parquet", index=False, compression="zstd")
    causality = _causality_audit(prepared, audits)
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": SCHEMA,
        "scope": "offline research only; no live/canonical mutation",
        "question": "does the consensus layer improve policy conversion conditional on the B0 base route?",
        "stage_a": {
            "control": CONTROL.__dict__, "arms": [arm.__dict__ for arm in STAGE_A],
            "comparison": "direct rich_policy_net_bps versus incumbent simple policy residual target",
        },
        "stage_b": {"parent_selected_on_2025_only": parent.__dict__, "arms": [arm.__dict__ for arm in stage_b]},
        "confirmation": {"months": [str(value) for value in EVAL_MONTHS_2026], "arms": confirmation_names, "control_required": CONTROL.name},
        "training": "three preceding calendar months; each label must be resolved before the held month; B0-routed training rows only; train-only deterministic query sampling",
        "query": f"{QUERY_HOURS}-hour UTC x long", "base_route_percentile": BASE_ROUTE_PERCENTILE,
        "ranker": "frozen LambdaRank compact residual parameters; no HPO", "residual_bands_bps": list(RESIDUAL_BANDS),
        "overlay": {"demotion_residual_bps": DEMODE_RESIDUAL_BPS, "recovery_residual_bps": RECOVERY_RESIDUAL_BPS, "demotion_authority": DEMODE_AUTHORITY, "recovery_authority": RECOVERY_AUTHORITY},
        "admission": {"rule": "frozen current MC1 >= threshold AND frozen BCF MC1 >= threshold", "thresholds_bps": list(MC1_THRESHOLDS_BPS)},
        "portfolio": {"contract": PORTFOLIO.__dict__, "purpose": "narrow offline research mirror; not production promotion evidence"},
        "causality": causality,
        "sources": {key: str(value.resolve()) for key, value in vars(sources).items()},
        "source_sha256": {key: _sha256([value]) for key, value in vars(sources).items()},
    }, indent=2, default=str) + "\n")
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
