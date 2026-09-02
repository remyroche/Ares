#!/usr/bin/env python3
"""Audit frozen Router50 single-Base finalists without changing any score.

This terminal research reporter consumes only immutable target-free Base/R/U
receipts and the already-produced dual-MC1 outcome panels.  It joins canonical
policy outcomes *after* scores have been read and reports the metrics required
to select a Router50 Base architecture: MC1 capture, R/U component overlap,
residual-band attribution, rescue economics, and constrained portfolio risk.

It is a diagnostic.  It never fits a model, alters admission, changes a
portfolio decision, writes an inference bundle, or performs exchange I/O.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
THRESHOLD_BPS = 50.0
MONTHS = ("2026-06", "2026-07")
COMPONENT_ARMS = ("base_only", "r_only", "u_only", "base_ru")


def _write_once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _parse_arm(value: str) -> tuple[str, Path, Path]:
    try:
        name, mc1, ru = value.split("::", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--arm must be NAME::MC1_ROOT::RU_ROOT") from error
    if not name or not mc1 or not ru:
        raise argparse.ArgumentTypeError("--arm values must be non-empty")
    return name, Path(mc1).resolve(), Path(ru).resolve()


def _admission(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & pd.to_numeric(frame["current_mc1_expected_bps"], errors="coerce").ge(THRESHOLD_BPS)
        & pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="coerce").ge(THRESHOLD_BPS)
    )


def _policy_valid(frame: pd.DataFrame) -> pd.Series:
    return frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(
        pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    )


def _top_timestamp_net(frame: pd.DataFrame, score: str, top_n: int) -> float:
    work = frame.loc[_policy_valid(frame), ["__decision_ts__", "candidate_id", score, "policy_net_bps"]].copy()
    work = work.sort_values(["__decision_ts__", score, "candidate_id"], ascending=[True, False, True], kind="stable")
    selected = work.groupby("__decision_ts__", sort=False).head(top_n)
    return float(pd.to_numeric(selected["policy_net_bps"], errors="coerce").mean()) if len(selected) else float("nan")


def _capture(frame: pd.DataFrame, admitted: pd.Series) -> dict[str, float | int]:
    valid = _policy_valid(frame)
    y = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    result: dict[str, float | int] = {
        "eligible_rows": int(valid.sum()),
        "dual_admitted_rows": int(admitted.sum()),
        "dual_admitted_ev_bps": float(y.loc[admitted].mean()) if admitted.any() else float("nan"),
        "dual_admitted_total_bps": float(y.loc[admitted].sum()),
    }
    for threshold in (50.0, 100.0):
        winners = valid & y.gt(threshold)
        result[f"winner{int(threshold)}_rows"] = int(winners.sum())
        result[f"winner{int(threshold)}_recall"] = float((admitted & winners).sum() / winners.sum()) if winners.any() else float("nan")
    excess_all = np.maximum(y.loc[valid].to_numpy(float) - THRESHOLD_BPS, 0.0).sum()
    excess_admitted = np.maximum(y.loc[admitted].to_numpy(float) - THRESHOLD_BPS, 0.0).sum()
    result["er50_captured"] = float(excess_admitted / excess_all) if excess_all > 0.0 else float("nan")
    result["timestamp_top1_net_bps"] = _top_timestamp_net(frame, "bcf_mc1_expected_bps", 1)
    result["timestamp_top2_net_bps"] = _top_timestamp_net(frame, "bcf_mc1_expected_bps", 2)
    return result


def _bands(frame: pd.DataFrame, admitted: pd.Series) -> pd.DataFrame:
    rank = pd.to_numeric(frame["base_rank_ts"], errors="coerce")
    labels = np.select(
        [rank.ge(.95), rank.ge(.90), rank.ge(.85), rank.ge(.80)],
        ["0_5", "5_10", "10_15", "15_20"], default="20_50",
    )
    work = frame.loc[:, ["policy_net_bps"]].copy()
    work["base_band"] = labels
    work["admitted"] = admitted.to_numpy(bool)
    work["winner50"] = pd.to_numeric(work["policy_net_bps"], errors="coerce").gt(50.0)
    rows: list[dict[str, float | int | str]] = []
    for label in ("0_5", "5_10", "10_15", "15_20", "20_50"):
        part = work.loc[(work.base_band == label) & work.admitted].copy()
        y = pd.to_numeric(part.policy_net_bps, errors="coerce")
        rows.append({
            "base_band": label,
            "mc1_admitted": int(len(part)),
            "final_ev_bps": float(y.mean()) if len(part) else float("nan"),
            "winner50_captured": int(part.winner50.sum()),
            "total_bps": float(y.sum()),
        })
    return pd.DataFrame(rows)


def _mi(x: np.ndarray, y: np.ndarray) -> float:
    """Discrete mutual information in nats without an outcome-aware model."""
    if not len(x):
        return float("nan")
    table = pd.crosstab(pd.Series(x), pd.Series(y), normalize=True).to_numpy(float)
    px = table.sum(axis=1, keepdims=True)
    py = table.sum(axis=0, keepdims=True)
    positive = table > 0.0
    return float((table[positive] * np.log(table[positive] / (px @ py)[positive])).sum())


def _conditional_information(frame: pd.DataFrame) -> dict[str, float]:
    valid = _policy_valid(frame)
    work = frame.loc[valid, ["base_rank_ts", "r_residual_sqrt_atr_quintile_rank", "u_unexpected_trailing_atr1_rank", "policy_net_bps"]].copy()
    base = np.minimum((pd.to_numeric(work.base_rank_ts, errors="coerce").fillna(0.0).to_numpy(float) * 10).astype(int), 9)
    r = np.minimum((pd.to_numeric(work.r_residual_sqrt_atr_quintile_rank, errors="coerce").fillna(0.0).to_numpy(float) * 10).astype(int), 9)
    u = np.minimum((pd.to_numeric(work.u_unexpected_trailing_atr1_rank, errors="coerce").fillna(0.0).to_numpy(float) * 10).astype(int), 9)
    y_raw = pd.to_numeric(work.policy_net_bps, errors="coerce").to_numpy(float)
    y = np.select([y_raw <= 0.0, y_raw <= 50.0, y_raw <= 100.0, y_raw <= 200.0], [0, 1, 2, 3], default=4)
    result = {"cmi_r_policy_given_base": 0.0, "cmi_u_policy_given_base": 0.0, "cmi_ru_policy_given_base": 0.0}
    for band in range(10):
        mask = base == band
        if int(mask.sum()) < 100:
            continue
        weight = float(mask.mean())
        result["cmi_r_policy_given_base"] += weight * _mi(r[mask], y[mask])
        result["cmi_u_policy_given_base"] += weight * _mi(u[mask], y[mask])
        result["cmi_ru_policy_given_base"] += weight * _mi(r[mask] * 10 + u[mask], y[mask])
    return result


def _rescue(frame: pd.DataFrame) -> pd.DataFrame:
    valid = _policy_valid(frame)
    work = frame.loc[valid, ["__decision_ts__", "candidate_id", "base_rank_ts", "r_residual_sqrt_atr_quintile_rank", "u_unexpected_trailing_atr1_rank", "policy_net_bps"]].copy()
    work = work.loc[work.base_rank_ts.ge(.80) & work.base_rank_ts.lt(.95)].copy()
    work["ru_combo"] = .5 * work.r_residual_sqrt_atr_quintile_rank + .5 * work.u_unexpected_trailing_atr1_rank
    work["base_select"] = work.groupby("__decision_ts__", sort=False).base_rank_ts.rank(pct=True, method="average").ge(.75)
    work["ru_select"] = work.groupby("__decision_ts__", sort=False).ru_combo.rank(pct=True, method="average").ge(.75)
    rows: list[dict[str, float | int | str]] = []
    for name, mask in (("all_5_20", pd.Series(True, index=work.index)), ("base_equivalent_top25", work.base_select), ("ru_combo_top25", work.ru_select)):
        part = work.loc[mask]
        y = pd.to_numeric(part.policy_net_bps, errors="coerce")
        monthly = part.assign(month=part.__decision_ts__.dt.strftime("%Y-%m")).groupby("month", sort=True).policy_net_bps.mean()
        all50 = work.policy_net_bps.gt(50.0).sum()
        all100 = work.policy_net_bps.gt(100.0).sum()
        rows.append({
            "selection": name,
            "rows": int(len(part)),
            "ev_bps": float(y.mean()) if len(part) else float("nan"),
            "total_bps": float(y.sum()),
            "winner50_recall": float(part.policy_net_bps.gt(50.0).sum() / all50) if all50 else float("nan"),
            "winner100_recall": float(part.policy_net_bps.gt(100.0).sum() / all100) if all100 else float("nan"),
            "excess_over50_bps": float(np.maximum(y.to_numpy(float) - 50.0, 0.0).sum()),
            "q25_month_ev_bps": float(monthly.quantile(.25)) if len(monthly) else float("nan"),
        })
    return pd.DataFrame(rows)


def _component_cohorts(root: Path) -> pd.DataFrame:
    table: dict[str, pd.DataFrame] = {}
    for arm in ("r_only", "u_only"):
        frame = pd.read_parquet(root / f"{arm}_dual_predictions.parquet")
        table[arm] = frame.loc[:, ["candidate_id", "policy_net_bps"]].assign(**{arm: _admission(frame).to_numpy(bool)})
    work = table["r_only"].merge(table["u_only"].drop(columns="policy_net_bps"), on="candidate_id", validate="one_to_one")
    rows: list[dict[str, float | int | str]] = []
    for name, mask in (
        ("R_only_admitted", work.r_only & ~work.u_only),
        ("U_only_admitted", ~work.r_only & work.u_only),
        ("R_U_shared_admitted", work.r_only & work.u_only),
    ):
        y = pd.to_numeric(work.loc[mask, "policy_net_bps"], errors="coerce")
        rows.append({"cohort": name, "rows": int(mask.sum()), "ev_bps": float(y.mean()) if len(y) else float("nan"), "total_bps": float(y.sum())})
    return pd.DataFrame(rows)


def _portfolio_risk(root: Path) -> dict[str, float | int]:
    decisions = pd.read_parquet(root / "base_ru_202606_202607_decisions.parquet")
    equity = pd.read_parquet(root / "base_ru_202606_202607_equity.parquet")
    accepted = decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
    timestamps = pd.to_datetime(accepted.timestamp, utc=True)
    period_days = max(1, int((pd.to_datetime(equity.timestamp, utc=True).max().normalize() - pd.to_datetime(equity.timestamp, utc=True).min().normalize()).days))
    returns = pd.to_numeric(equity.mtm_equity, errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    downside = np.minimum(returns, 0.0)
    sortino = float(returns.mean() / np.sqrt(np.mean(downside ** 2))) if len(returns) and np.sqrt(np.mean(downside ** 2)) > 1e-12 else float("nan")
    net_bps = pd.to_numeric(accepted.position_net_return, errors="coerce") * 10_000.0
    losses = accepted.loc[net_bps < 0.0, ["symbol"]].copy()
    losses["loss_bps"] = -net_bps.loc[net_bps < 0.0].to_numpy(float)
    sym = losses.groupby("symbol", sort=False).loss_bps.sum()
    daily = pd.DataFrame({"day": timestamps.dt.normalize(), "loss_bps": np.maximum(-net_bps.to_numpy(float), 0.0)}).groupby("day", sort=False).loss_bps.sum()
    return {
        "portfolio_entries": int(len(accepted)),
        "trades_per_calendar_day": float(len(accepted) / period_days),
        "portfolio_sortino_hourly_mtm": sortino,
        "portfolio_final_wallet": float(pd.to_numeric(equity.wallet, errors="coerce").iloc[-1]),
        "symbol_hhi": float((accepted.symbol.value_counts(normalize=True) ** 2).sum()) if len(accepted) else float("nan"),
        "largest_symbol_loss_share": float(sym.max() / sym.sum()) if len(sym) and sym.sum() > 0.0 else 0.0,
        "largest_day_loss_share": float(daily.max() / daily.sum()) if len(daily) and daily.sum() > 0.0 else 0.0,
    }


def _read_ru(root: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(root / "target_free_combined" / f"month={month}.parquet") for month in MONTHS]
    frame = pd.concat(parts, ignore_index=True)
    forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts"}
    if forbidden.intersection(frame.columns):
        raise AssertionError("R/U target-free receipt contains policy outcome fields")
    if frame.duplicated("candidate_id").any():
        raise AssertionError("R/U target-free receipt duplicates identity")
    return frame


def _audit_one(name: str, mc1_root: Path, ru_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ru = _read_ru(ru_root)
    result_rows: list[dict[str, object]] = []
    band_rows: list[pd.DataFrame] = []
    base_ru: pd.DataFrame | None = None
    for arm in COMPONENT_ARMS:
        dual = pd.read_parquet(mc1_root / f"{arm}_dual_predictions.parquet")
        if dual.duplicated("candidate_id").any():
            raise AssertionError(f"{name}/{arm}: dual MC1 duplicates identity")
        frame = dual.merge(ru.loc[:, [*IDENTITY, "base_rank_ts", "r_residual_sqrt_atr_quintile_rank", "u_unexpected_trailing_atr1_rank"]], on=list(IDENTITY), how="inner", validate="one_to_one")
        if len(frame) != len(dual) or len(frame) != len(ru):
            raise AssertionError(f"{name}/{arm}: target-free R/U to MC1 identity mismatch")
        admitted = _admission(frame)
        result_rows.append({"stack": name, "component_arm": arm, **_capture(frame, admitted)})
        if arm == "base_ru":
            base_ru = frame.assign(__admitted__=admitted.to_numpy(bool))
            bands = _bands(base_ru, base_ru.__admitted__)
            bands.insert(0, "stack", name)
            band_rows.append(bands)
    assert base_ru is not None
    rescue = _rescue(base_ru); rescue.insert(0, "stack", name)
    cohorts = _component_cohorts(mc1_root); cohorts.insert(0, "stack", name)
    risk = _portfolio_risk(mc1_root)
    risk["stack"] = name
    information = _conditional_information(base_ru)
    information["stack"] = name
    return pd.DataFrame(result_rows), pd.concat(band_rows, ignore_index=True), rescue, cohorts, {**risk, **information}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=_parse_arm, required=True, help="NAME::MC1_ROOT::RU_ROOT; repeat for every matched Base stack")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    names = [name for name, _, _ in args.arm]
    if len(set(names)) != len(names):
        raise ValueError("duplicate final-audit stack name")
    args.out.mkdir(parents=True)
    mc1_rows: list[pd.DataFrame] = []
    bands: list[pd.DataFrame] = []
    rescue: list[pd.DataFrame] = []
    cohorts: list[pd.DataFrame] = []
    risk: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    for name, mc1_root, ru_root in args.arm:
        rows, one_bands, one_rescue, one_cohorts, info = _audit_one(name, mc1_root, ru_root)
        mc1_rows.append(rows); bands.append(one_bands); rescue.append(one_rescue); cohorts.append(one_cohorts); risk.append(info)
        sources[name] = {"mc1_root": str(mc1_root), "ru_root": str(ru_root)}
    pd.concat(mc1_rows, ignore_index=True).to_parquet(args.out / "mc1_component_metrics.parquet", index=False, compression="zstd")
    pd.concat(bands, ignore_index=True).to_parquet(args.out / "residual_band_attribution.parquet", index=False, compression="zstd")
    pd.concat(rescue, ignore_index=True).to_parquet(args.out / "residual_rescue_metrics.parquet", index=False, compression="zstd")
    pd.concat(cohorts, ignore_index=True).to_parquet(args.out / "component_admission_cohorts.parquet", index=False, compression="zstd")
    pd.DataFrame(risk).to_parquet(args.out / "portfolio_risk_metrics.parquet", index=False, compression="zstd")
    _write_once(args.out / "correctness_report.json", {
        "target_free_ru_checked_before_outcome_join": True,
        "ru_to_dual_identity_exact": True,
        "dual_mc1_threshold_bps": THRESHOLD_BPS,
        "all_rows_are_existing_frozen_scores": True,
        "scope": "offline diagnostic only; no fit, score, admission, portfolio, live, or exchange mutation",
    })
    _write_once(args.out / "run_manifest.json", {
        "schema": "strict_r3_router_single_base_finalists_audit_v1",
        "scope": "offline final-pipeline attribution only; no score/model/exchange mutation",
        "evaluation_months": list(MONTHS),
        "threshold_bps": THRESHOLD_BPS,
        "base_bands": ["0_5", "5_10", "10_15", "15_20", "20_50"],
        "rescue_pool": "Base ranks 5-20%; compare fixed R/U average top quartile with equal-count Base-only top quartile",
        "conditional_information": "outcome-joined diagnostic: fixed ten-bin Base/R/U ranks and fixed five-bin policy-net grade",
        "sources": sources,
    })


if __name__ == "__main__":
    main()
