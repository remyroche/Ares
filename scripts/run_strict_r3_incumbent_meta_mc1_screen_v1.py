#!/usr/bin/env python3
"""Strict-prequential MC1 added-value screen for incumbent meta heads.

The retained upstream stays fixed throughout:

``incumbent_upstream_bps = .50 * efficiency_bps + .50 * timing_bps``.

This runner consumes target-free parent Current/BCF score receipts and one
target-free OOF meta-head receipt at a time.  It first persists their joined
target-free score panels, then attaches the canonical rich-policy ledger only
for strictly earlier supervised MC1 fits and final economics.  The meta output
is an *additional mapper coordinate*: it has no direct ranking or admission
authority.  Current and BCF MC1 maps remain separate and both must clear the
retained +50-bps gate before one chronological constrained portfolio auction.

It is deliberately a small target/query-selection tool.  Full meta feature
selection and HPO happen only after this screen identifies credible family
candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_strict_r3_o3v2_target_funnel as target_contract  # noqa: E402


SCHEMA = "strict_r3_incumbent_meta_mc1_screen_v1"
SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
PARENT_FEATURES = (
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)
MC1_MONTHS = 6
SHIFT_DAYS = 21
ADMISSION_BPS = 50.0
EVALUATION_START = pd.Timestamp("2026-04-01", tz="UTC")
EVALUATION_END = pd.Timestamp("2026-08-01", tz="UTC")
DEFAULT_PARENT = ROOT / "data_perp/artifacts/strict_r3_enhanced_base_live_stack_challenger_20260823_v10/target_free_scores"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/canonical_reconciled_policy_labels.parquet"

POLICY_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
)
PROHIBITED = set(target_contract.PROHIBITED_SCORE_COLUMNS)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    for target in (sorted(path.rglob("*.parquet")) if path.is_dir() else [path]):
        digest.update(str(target).encode())
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _parse_months(raw: str) -> tuple[pd.Timestamp, ...]:
    output = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in raw.split(",") if item.strip())
    if not output or tuple(sorted(set(output))) != output:
        raise ValueError("--months must be unique, chronological calendar months")
    return output


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _target_free(path: Path) -> None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    leaked = sorted(PROHIBITED.intersection(names))
    if leaked:
        raise AssertionError(f"{path}: target-free score receipt leaks {leaked}")


def _read_parent(parent_root: Path, family: str, month: pd.Timestamp) -> pd.DataFrame:
    path = parent_root / family / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    _target_free(path)
    columns = [*IDENTITY, "enhanced_base_routed", *PARENT_FEATURES]
    result = pd.read_parquet(path, columns=columns)
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result.duplicated(IDENTITY).any() or not result.side_name.eq("long").all():
        raise AssertionError(f"{path}: invalid parent score identities")
    if result.enhanced_base_routed.isna().any():
        raise AssertionError(f"{path}: missing canonical stored route state")
    result["enhanced_base_routed"] = result.enhanced_base_routed.astype(bool)
    return result


def _read_meta(root: Path, arm: str, month: pd.Timestamp) -> pd.DataFrame:
    path = root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    _target_free(path)
    result = pd.read_parquet(path, columns=[*IDENTITY, "meta_raw_score", "meta_rank_ts", "arm", "target_family", "query_contract"])
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result.duplicated(IDENTITY).any() or not result.side_name.eq("long").all():
        raise AssertionError(f"{path}: invalid meta score identities")
    if not result.arm.eq(arm).all():
        raise AssertionError(f"{path}: arm identity changed")
    return result


def _target_free_panels(
    *, parent_root: Path, meta_root: Path, arm: str, months: Sequence[pd.Timestamp], out: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    panels: dict[str, list[pd.DataFrame]] = {"current": [], "bcf": []}
    audit: list[dict[str, Any]] = []
    for month in months:
        meta = _read_meta(meta_root, arm, month)
        for family in panels:
            parent = _read_parent(parent_root, family, month)
            merged = parent.merge(meta, on=list(IDENTITY), how="inner", validate="one_to_one")
            if merged.empty:
                raise AssertionError(f"{arm} {family} {month:%Y-%m}: no common target-free rows")
            if not merged.enhanced_base_routed.fillna(False).astype(bool).all():
                raise AssertionError(f"{arm} {family} {month:%Y-%m}: meta scores outside the persisted canonical base route")
            if len(merged) != len(meta):
                raise AssertionError(f"{arm} {family} {month:%Y-%m}: parent source misses target-free meta identities")
            panel_path = out / "target_free_panels" / arm / family / f"month={month:%Y-%m}.parquet"
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(panel_path, index=False, compression="zstd")
            panels[family].append(merged)
            audit.append({
                "arm": arm, "family": family, "month": f"{month:%Y-%m}",
                "parent_rows": int(len(parent)), "meta_rows": int(len(meta)), "common_rows": int(len(merged)),
                "canonical_stored_route_rows": int(parent.enhanced_base_routed.sum()),
                "target_free_before_policy_join": True, "path": str(panel_path),
            })
    return {key: pd.concat(value, ignore_index=True) for key, value in panels.items()}, pd.DataFrame(audit)


def _read_policy(path: Path) -> pd.DataFrame:
    result = pd.read_parquet(path, columns=list(POLICY_COLUMNS))
    if result.candidate_id.duplicated().any():
        raise AssertionError("canonical policy ledger has duplicate candidate IDs")
    result["policy_path_valid"] = result.policy_path_valid.fillna(False).astype(bool)
    result["policy_net_bps"] = pd.to_numeric(result.policy_net_bps, errors="coerce")
    result["policy_label_available_ts"] = pd.to_datetime(result.policy_label_available_ts, utc=True, errors="coerce")
    return result


def _rank_bands(frame: pd.DataFrame) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "final_score"]].copy()
    work["row"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "final_score", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    work["score_band"] = np.minimum(9, (10.0 * (ordinal + .5) / count).astype(np.int8))
    return work.sort_values("row", kind="stable")["score_band"].to_numpy(np.int8)


def _robust_mean(values: Iterable[float], trim: float = .10) -> float:
    x = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    if not len(x):
        return float("nan")
    k = int(math.floor(len(x) * trim))
    if k and len(x) > 2 * k:
        x = x[k:len(x) - k]
    return float(x.mean())


def _fit_mc1(train: pd.DataFrame, features: Sequence[str]) -> tuple[HistGradientBoostingRegressor, np.ndarray, np.ndarray, tuple[float, float]]:
    fit = train.copy()
    fit["score_band"] = _rank_bands(fit)
    fit["day"] = fit.__decision_ts__.dt.normalize()
    samples: list[pd.DataFrame] = []
    for _day, group in fit.groupby("day", sort=True):
        ordered = group.sort_values(["__decision_ts__", "final_score", "candidate_id"], ascending=[True, False, True], kind="stable")
        tail = ordered.iloc[50:]
        samples.append(pd.concat([ordered.head(50), tail.sample(min(250, len(tail)), random_state=SEED)]))
    work = pd.concat(samples, ignore_index=True)
    y = pd.to_numeric(work.policy_net_bps, errors="coerce")
    low, high = y.quantile([.02, .98])
    work["target"] = y.clip(low, high)
    if len(work) > 50_000:
        work = work.sample(50_000, random_state=SEED)
    medians = work.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").median().to_numpy(float)
    x = work.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").fillna(pd.Series(medians, index=features))
    model = HistGradientBoostingRegressor(
        max_depth=2, max_iter=80, learning_rate=.04, l2_regularization=20.0,
        min_samples_leaf=100, random_state=SEED,
    ).fit(x, work.target)
    global_mean = _robust_mean(work.target)
    curve = np.full(10, global_mean, dtype=float)
    for band, group in work.groupby("score_band", sort=True):
        mean, sd, n = float(group.target.mean()), max(float(group.target.std(ddof=0)), 1.0), len(group)
        precision = n / (sd * sd + 1.0)
        prior = 80.0 / (250.0 ** 2)
        curve[int(band)] = (precision * mean + prior * global_mean) / (precision + prior)
    curve = -IsotonicRegression(increasing=True).fit_transform(np.arange(10), -curve)
    return model, medians, np.asarray(curve, dtype=float), (float(low), float(high))


def _mc1_predict(frame: pd.DataFrame, *, family: str, features: Sequence[str], months: Sequence[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["score_band"] = _rank_bands(work)
    outputs: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for month in months:
        if month < EVALUATION_START:
            continue
        end = _month_end(month)
        fit_start = month - pd.DateOffset(months=MC1_MONTHS)
        fit = work.loc[
            work.__decision_ts__.ge(fit_start) & work.__decision_ts__.lt(month)
            & work.policy_path_valid & work.policy_label_available_ts.lt(month)
            & np.isfinite(pd.to_numeric(work.policy_net_bps, errors="coerce"))
        ].copy()
        held = work.loc[work.__decision_ts__.ge(month) & work.__decision_ts__.lt(end)].copy()
        if len(fit) < 5_000 or held.empty:
            audit.append({"family": family, "month": f"{month:%Y-%m}", "status": "insufficient", "train_rows": int(len(fit)), "held_rows": int(len(held))})
            continue
        model, medians, curve, clip = _fit_mc1(fit, features)
        x = held.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").fillna(pd.Series(medians, index=features))
        held["static_expected_bps"] = model.predict(x)
        shifts: dict[pd.Timestamp, float] = {}
        for day in pd.date_range(month.normalize(), (end - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC"):
            history = work.loc[
                work.__decision_ts__.ge(day - pd.Timedelta(days=SHIFT_DAYS)) & work.__decision_ts__.lt(day)
                & work.policy_path_valid & work.policy_label_available_ts.lt(day)
                & np.isfinite(pd.to_numeric(work.policy_net_bps, errors="coerce"))
            ]
            residual = pd.to_numeric(history.policy_net_bps, errors="coerce").to_numpy(float) - curve[history.score_band.to_numpy(int)]
            shifts[day] = _robust_mean(residual, trim=.10) if len(residual) else 0.0
        held["recent_shift_bps"] = held.__decision_ts__.dt.normalize().map(shifts).fillna(0.0)
        held["mc1_expected_bps"] = held.static_expected_bps + held.recent_shift_bps
        held["mc1_family"] = family
        outputs.append(held)
        audit.append({"family": family, "month": f"{month:%Y-%m}", "status": "scored", "train_rows": int(len(fit)), "held_rows": int(len(held)), "clip_low": clip[0], "clip_high": clip[1]})
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audit)


def _portfolio(frame: pd.DataFrame, *, arm: str, out: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    from extreme_price_movements.portfolio_policy_replay import normalise_candidate_table, replay_candidates
    from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _metrics, _params

    gate = (
        frame.policy_path_valid & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
        & np.isfinite(pd.to_numeric(frame.policy_exit_bar_15m, errors="coerce"))
        & pd.to_numeric(frame.current_mc1_expected_bps, errors="coerce").ge(ADMISSION_BPS)
        & pd.to_numeric(frame.bcf_mc1_expected_bps, errors="coerce").ge(ADMISSION_BPS)
    )
    admitted = frame.loc[gate].copy().reset_index(drop=True)
    admitted["auction_rank"] = admitted.groupby("__decision_ts__", sort=False).bcf_mc1_expected_bps.rank(pct=True, method="average")
    decision = pd.to_datetime(admitted.__decision_ts__, utc=True)
    candidates = pd.DataFrame({
        "timestamp": decision,
        "symbol": admitted.candidate_id.astype(str).str.split("|", n=1, expand=True)[0],
        "side": "long", "strategy_id": "strict_r3_incumbent_meta_mc1", "policy_archetype": "strict_r3_incumbent_meta_mc1",
        "normalized_rank_score": admitted.auction_rank.to_numpy(float), "strategy_rank_pct": admitted.auction_rank.to_numpy(float),
        "base_strategy_threshold": 0.0, "calibrated_score": admitted.bcf_mc1_expected_bps.to_numpy(float),
        "entry_price": pd.to_numeric(admitted.policy_entry_price, errors="coerce"),
        "exit_timestamp": decision + pd.to_timedelta((pd.to_numeric(admitted.policy_exit_bar_15m, errors="coerce").astype(int) + 1) * 15, unit="min"),
        "exit_price": pd.to_numeric(admitted.policy_exit_price, errors="coerce"),
        "net_return": pd.to_numeric(admitted.policy_net_bps, errors="coerce") / 10_000.0,
        "gross_return": pd.to_numeric(admitted.policy_gross_bps, errors="coerce") / 10_000.0,
        "holding_bars": pd.to_numeric(admitted.policy_exit_bar_15m, errors="coerce").astype(int) + 1,
        "simple_policy_exit_reason": admitted.policy_exit_reason.astype(str),
        "fees_bps": 100.0, "slippage_bps": 0.0, "expected_friction_bps": 100.0, "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0, "source_month": decision.dt.strftime("%Y-%m"), "candidate_id": admitted.candidate_id.astype(str),
        "mapped_expected_net_bps": admitted.bcf_mc1_expected_bps.to_numpy(float), "policy_outcome_available": True,
    })
    normalized = normalise_candidate_table(candidates)
    decisions, equity, _ = replay_candidates(normalized, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perps", initial_wallet=1000.0)
    # The canonical replay normalizer does not retain this research coverage
    # marker.  Every candidate reaching this adapter was explicitly joined to
    # a valid canonical policy outcome, so restore the marker solely for the
    # shared reporting utility; it is not an auction input.
    if "policy_outcome_available" not in decisions.columns:
        decisions["policy_outcome_available"] = True
    decisions.to_parquet(out / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
    metric = _metrics(decisions, equity, arm, "2026_aprjul")
    metric.update({"arm": arm, "candidate_admitted_rows": int(len(admitted)), "admission_threshold_bps": ADMISSION_BPS})
    return metric, admitted.loc[:, ["candidate_id", "__decision_ts__", "current_mc1_expected_bps", "bcf_mc1_expected_bps"]]


def _evaluate_arm(*, arm: str, parent_root: Path, meta_root: Path, policy: pd.DataFrame, months: Sequence[pd.Timestamp], out: Path) -> tuple[list[dict[str, Any]], list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    target_free, panel_audit = _target_free_panels(parent_root=parent_root, meta_root=meta_root, arm=arm, months=months, out=out)
    results: list[dict[str, Any]] = []
    admissions: list[pd.DataFrame] = []
    mc1_audits: list[pd.DataFrame] = []
    for has_meta in (False, True):
        maps: dict[str, pd.DataFrame] = {}
        audits: list[pd.DataFrame] = []
        for family, panel in target_free.items():
            labelled = panel.merge(policy, on="candidate_id", how="left", validate="one_to_one")
            if len(labelled) != len(panel):
                raise AssertionError("policy join altered target-free score identities")
            feature_list = [*PARENT_FEATURES, "meta_rank_ts"] if has_meta else list(PARENT_FEATURES)
            pred, audit = _mc1_predict(labelled, family=family, features=feature_list, months=months)
            maps[family] = pred.rename(columns={"mc1_expected_bps": f"{family}_mc1_expected_bps"})
            audit["arm"] = arm; audit["has_meta"] = has_meta; audits.append(audit)
        current = maps["current"]
        bcf = maps["bcf"]
        left = current.loc[:, ["candidate_id", "__decision_ts__", "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason", "current_mc1_expected_bps"]]
        right = bcf.loc[:, ["candidate_id", "__decision_ts__", "bcf_mc1_expected_bps"]]
        combined = left.merge(right, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
        label = f"{arm}__{'with_meta' if has_meta else 'control_no_meta'}"
        metric, admitted = _portfolio(combined, arm=label, out=out)
        metric["meta_enabled"] = has_meta
        metric["meta_arm"] = arm
        metric["screen_label"] = label
        results.append(metric)
        admitted["arm"] = arm; admitted["meta_enabled"] = has_meta; admissions.append(admitted)
        mc1_audits.append(pd.concat(audits, ignore_index=True))
    return results, admissions, panel_audit, pd.concat(mc1_audits, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--arms", required=True, help="comma-separated selected arm names")
    parser.add_argument("--months", default="2025-09,2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out}: immutable output root already exists")
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("--arms must contain unique non-empty values")
    months = _parse_months(args.months)
    if months[0] > EVALUATION_START - pd.DateOffset(months=MC1_MONTHS):
        raise ValueError("months omit the six-month prequential MC1 history")
    args.out.mkdir(parents=True)
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline research-only strict-prequential meta-to-MC1 selection; no live, inference, admission, or exchange mutation",
        "incumbent_upstream": "0.50 * efficiency_bps + 0.50 * timing_bps",
        "meta_authority": "additional MC1 feature only; no direct score, route, or admission authority",
        "parent_score_source": str(args.parent_root), "meta_score_source": str(args.meta_root), "policy_source": str(args.policy),
        "arms": list(arms), "months": [f"{month:%Y-%m}" for month in months],
        "evaluation": [EVALUATION_START.isoformat(), EVALUATION_END.isoformat()],
        "mc1": {"fit_months": MC1_MONTHS, "prior_resolved_shift_days": SHIFT_DAYS, "features": list(PARENT_FEATURES) + ["meta_rank_ts"]},
        "admission": {"current_and_bcf_expected_net_bps_gte": ADMISSION_BPS, "priority": "bcf_mc1_expected_bps"},
        "portfolio": "one chronological constrained canonical rich-policy replay",
        "causality": "parent/meta score panels are persisted target-free before policy join; MC1 fits only earlier resolved labels; no held-month labels in model or shift",
        "source_hashes": {"parent": _sha(args.parent_root), "meta": _sha(args.meta_root), "policy": _sha(args.policy)},
    })
    policy = _read_policy(args.policy)
    results: list[dict[str, Any]] = []
    admissions: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    mc1_audits: list[pd.DataFrame] = []
    for arm in arms:
        arm_results, arm_admissions, panel_audit, mc1_audit = _evaluate_arm(arm=arm, parent_root=args.parent_root, meta_root=args.meta_root, policy=policy, months=months, out=args.out)
        results.extend(arm_results); admissions.extend(arm_admissions); audits.append(panel_audit); mc1_audits.append(mc1_audit)
    metrics = pd.DataFrame(results)
    # Each arm has its own matched no-meta control.  This intentionally avoids
    # comparing a meta arm with a different routed universe.
    controls = metrics.loc[~metrics.meta_enabled].set_index("meta_arm")
    deltas: list[dict[str, Any]] = []
    for _, row in metrics.loc[metrics.meta_enabled].iterrows():
        control = controls.loc[row.meta_arm]
        payload: dict[str, Any] = {"arm": row.meta_arm}
        for field in ("accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown", "candidate_admitted_rows"):
            if field in row and field in control:
                payload[f"delta_{field}"] = float(row[field]) - float(control[field])
        deltas.append(payload)
    metrics.to_parquet(args.out / "mc1_portfolio_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(deltas).to_parquet(args.out / "mc1_added_value_deltas.parquet", index=False, compression="zstd")
    pd.concat(admissions, ignore_index=True).to_parquet(args.out / "mc1_admission_provenance.parquet", index=False, compression="zstd")
    pd.concat(audits, ignore_index=True).to_parquet(args.out / "target_free_panel_audit.parquet", index=False, compression="zstd")
    pd.concat(mc1_audits, ignore_index=True).to_parquet(args.out / "mc1_fit_audit.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    main()
