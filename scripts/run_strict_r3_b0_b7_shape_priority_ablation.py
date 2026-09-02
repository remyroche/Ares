#!/usr/bin/env python3
"""Focused B0--B7 MC1 level-versus-shape ablation.

This is deliberately a narrow successor to the sealed A0--A10 study.  It
holds the target-free BCF/Current candidate panels, frozen package-static MC1
maps, rich-policy labels, and portfolio replay fixed.  It asks only whether
the *causal score-band residual state* should be used as:

* an absolute admission-level adjustment; or
* a relative, uncertainty-shrunk priority correction after the static gate.

The static dual +50-bps gate is the primary population.  Unlike the older
``priority_only`` attribution, this runner ranks that fixed population by the
declared BCF priority directly; it never keeps the static dual gate score as
an implicit first sort key.  Thus it is a real test of the two-entry auction
use case.

Research only.  It never reads exchange state or mutates an inference bundle.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_mc1_inference_package import score_bands  # noqa: E402
from extreme_price_movements.portfolio_policy_replay import normalise_candidate_table, replay_candidates  # noqa: E402
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params  # noqa: E402


INPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_a0_a10_hierarchical_mc1_20260828_v2"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_p8u_b0_b7_shape_priority_20260829_v1"
THRESHOLD_BPS = 50.0
EPS = 1e-12
SCHEMA = "strict_r3_p8u_b0_b7_shape_priority_v1"


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


def _eb(location: float, se: float, tau2: float, prior: float = 0.0) -> tuple[float, float, float]:
    if not np.isfinite(location) or not np.isfinite(se) or se <= 0.0 or tau2 <= 0.0:
        return float(prior), 0.0, float(max(tau2, 0.0))
    se2 = se * se
    weight = float(tau2 / (tau2 + se2))
    return float(prior + weight * (location - prior)), weight, float(tau2 * se2 / (tau2 + se2))


def _isotonic_by_timestamp(frame: pd.DataFrame, *, static: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Non-decreasing projection used only for the anchored gate curve.

    This preserves the static-score ordering and permits a precise +50 anchor
    afterwards.  The raw B1 priority intentionally remains unprojected.
    """
    result = np.asarray(raw, dtype=float).copy()
    for _, group in frame.groupby("__decision_ts__", sort=False):
        idx = group.index.to_numpy()
        x = static[idx]
        y = raw[idx]
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() >= 2 and np.ptp(x[good]) > 1e-10:
            result[idx[good]] = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(x[good], y[good])
    return result


def _read(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The source panel is deliberately wide.  This ablation needs only the
    # target-free score coordinates, static/A6 values, and the policy fields
    # needed *after scoring* for replay; limiting the read keeps the whole
    # study comfortably below the research host's memory budget.
    fields = [
        "candidate_id", "__decision_ts__", "__symbol__", "enhanced_base_routed",
        "policy_path_valid", "policy_net_bps", "policy_exit_bar_15m", "policy_entry_price",
        "policy_exit_price", "policy_exit_reason", "policy_gross_bps", "policy_label_available_ts",
        "bcf_static_bps", "current_static_bps", "bcf_final_score", "current_final_score",
        "bcf__A6_band_eb", "current__A6_band_eb",
    ]
    panel = pd.read_parquet(input_root / "a0_a10_predictions.parquet", columns=fields)
    state = pd.read_parquet(input_root / "causal_residual_state.parquet")
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
    panel["policy_label_available_ts"] = pd.to_datetime(panel["policy_label_available_ts"], utc=True, errors="coerce")
    panel["decision_day"] = panel["__decision_ts__"].dt.normalize()
    state["decision_day"] = pd.to_datetime(state["decision_day"], utc=True, errors="coerce")
    required = {
        "candidate_id", "__decision_ts__", "decision_day", "enhanced_base_routed",
        "policy_path_valid", "policy_net_bps", "policy_exit_bar_15m", "policy_entry_price",
        "policy_exit_price", "policy_exit_reason", "policy_gross_bps", "__symbol__",
        "bcf_static_bps", "current_static_bps", "bcf_final_score", "current_final_score",
        "bcf__A6_band_eb", "current__A6_band_eb",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"A0/A6 panel lacks: {sorted(missing)}")
    if panel[["candidate_id", "__decision_ts__"]].duplicated().any():
        raise AssertionError("duplicate candidate identities")
    return panel.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), state


def _state_row(state: pd.DataFrame, family: str, day: pd.Timestamp, horizon: int, band: int) -> tuple[pd.Series, pd.Series]:
    now = state.loc[
        state["family"].eq(family)
        & state["decision_day"].eq(day)
        & state["horizon_days"].eq(float(horizon))
    ]
    glob = now.loc[now["scope"].eq("global")]
    local = now.loc[now["scope"].eq("score_band") & now["score_band"].eq(float(band))]
    if len(glob) != 1 or len(local) != 1:
        raise AssertionError(f"state missing or duplicated: {family}/{day}/{horizon}/band{band}")
    return glob.iloc[0], local.iloc[0]


def _family_state(panel: pd.DataFrame, state: pd.DataFrame, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct causal A6 raw corrections plus shape and confidence state."""
    static_field = f"{family}_static_bps"
    final_field = f"{family}_final_score"
    view = panel.loc[:, ["candidate_id", "__decision_ts__", "decision_day", static_field, final_field]].copy()
    view["final_score"] = view.pop(final_field)
    view["score_band"] = score_bands(view).astype(np.int8)
    rows: list[dict[str, Any]] = []
    for day in sorted(view["decision_day"].unique()):
        for band in range(10):
            global21, local21 = _state_row(state, family, day, 21, band)
            g21, _gweight, gvar = _eb(float(global21["location"]), float(global21["se"]), float(global21["tau2"]))
            raw, local_weight, local_var = _eb(
                float(local21["location"]), float(local21["se"]), float(local21["tau2"]), prior=g21,
            )
            shape = raw - g21
            hshape: list[float] = []
            for horizon in (7, 21, 42):
                glob, local = _state_row(state, family, day, horizon, band)
                g, _gw, _gv = _eb(float(glob["location"]), float(glob["se"]), float(glob["tau2"]))
                l, _lw, _lv = _eb(float(local["location"]), float(local["se"]), float(local["tau2"]), prior=g)
                hshape.append(float(l - g))
            harray = np.asarray(hshape, dtype=float)
            agreement = float(1.0 / (1.0 + np.std(harray) / 50.0)) if np.isfinite(harray).all() else 0.0
            shape_se = math.sqrt(max(local_var, 0.0) + max(gvar, 0.0))
            snr = abs(shape) / max(shape_se, EPS)
            snr_weight = float(snr * snr / (1.0 + snr * snr)) if np.isfinite(snr) else 0.0
            predictive_sd = math.sqrt(max(float(local21["mad_sd"]), 0.0) ** 2 + max(local_var, 0.0) + max(gvar, 0.0))
            rows.append({
                "family": family, "decision_day": day, "score_band": band, f"{family}_global21": g21,
                f"{family}_raw_a6": raw, f"{family}_shape": shape,
                f"{family}_shape_se": shape_se, f"{family}_shape_snr": snr,
                f"{family}_shape_snr_weight": snr_weight,
                f"{family}_shape_agreement": agreement,
                f"{family}_shape_confidence": snr_weight * agreement,
                f"{family}_predictive_sd": predictive_sd,
                f"{family}_local_eb_weight": local_weight,
            })
    merged = view.merge(pd.DataFrame(rows), on=["decision_day", "score_band"], how="left", validate="many_to_one")
    if merged.filter(regex=f"^{family}_").isna().any().any():
        raise AssertionError(f"missing reconstructed {family} state")
    return merged.rename(columns={"score_band": f"{family}_score_band"}).drop(columns=["final_score"]), pd.DataFrame(rows)


def _anchor_curve(frame: pd.DataFrame, *, family: str, static: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Project the shape curve and make its value exactly 50 at static EV=50."""
    projected = _isotonic_by_timestamp(frame, static=static, raw=raw)
    anchor = np.zeros(len(frame), dtype=float)
    audit: list[dict[str, Any]] = []
    for ts, group in frame.groupby("__decision_ts__", sort=False):
        idx = group.index.to_numpy()
        x = static[idx]
        y = projected[idx]
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 2:
            value = 50.0
        else:
            order = np.argsort(x[good], kind="stable")
            value = float(np.interp(50.0, x[good][order], y[good][order]))
        anchor[idx] = value
        audit.append({"family": family, "__decision_ts__": ts, "anchor_raw_value_bps": value, "anchor_error_before_bps": value - 50.0})
    return projected + (50.0 - anchor), pd.DataFrame(audit)


def _construct(panel: pd.DataFrame, state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = panel.copy()
    state_audits: list[pd.DataFrame] = []
    anchors: list[pd.DataFrame] = []
    for family in ("bcf", "current"):
        f, state_summary = _family_state(work, state, family)
        fields = [column for column in f.columns if column.startswith(f"{family}_") and column not in work.columns]
        work = work.merge(f.loc[:, ["candidate_id", "__decision_ts__", *fields]], on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
        static = pd.to_numeric(work[f"{family}_static_bps"], errors="coerce").to_numpy(float)
        raw = static + pd.to_numeric(work[f"{family}_raw_a6"], errors="coerce").to_numpy(float)
        # B1 is deliberately raw A6 (global level + local correction).
        work[f"{family}__B0_static"] = static
        work[f"{family}__B1_raw_a6"] = raw
        shape = pd.to_numeric(work[f"{family}_shape"], errors="coerce").to_numpy(float)
        for lam in (.25, .50, .75, 1.0):
            token = f"{int(lam * 100):03d}"
            work[f"{family}__B2_shape_l{token}"] = static + lam * shape
        # Anchor an isotonic shape curve at the static +50 decision boundary.
        anchored, anchor_audit = _anchor_curve(work, family=family, static=static, raw=static + shape)
        anchors.append(anchor_audit)
        work[f"{family}__B3_anchored_shape"] = anchored
        # B4: predictable downside-risk penalty has priority authority only.
        sigma = pd.to_numeric(work[f"{family}_predictive_sd"], errors="coerce").to_numpy(float)
        for kappa in (.10, .25, .50):
            token = f"{int(kappa * 100):02d}"
            work[f"{family}__B4_shape_risk_k{token}"] = anchored - kappa * sigma
        confidence = pd.to_numeric(work[f"{family}_shape_confidence"], errors="coerce").to_numpy(float)
        work[f"{family}__B7_multi_conf_shape"] = static + confidence * shape
        # Validate reconstruction against the sealed A6 projected score.  This
        # is the actual raw A6 curve with its global level retained.
        rec_a6 = _isotonic_by_timestamp(work, static=static, raw=raw)
        stored = pd.to_numeric(work[f"{family}__A6_band_eb"], errors="coerce").to_numpy(float)
        max_delta = float(np.nanmax(np.abs(rec_a6 - stored)))
        if max_delta > 1e-8:
            raise AssertionError(f"{family} A6 reconstruction parity failed: {max_delta} bps")
        state_summary["a6_reconstruction_max_abs_delta_bps"] = max_delta
        state_audits.append(state_summary)
    return work, pd.concat(state_audits, ignore_index=True), pd.concat(anchors, ignore_index=True)


ARM_LABELS = {
    "B0_static": "B0 static gate + static priority",
    "B1_raw_a6": "B1 static gate + raw A6 priority",
    "B2_shape_l025": "B2 static gate + mean-zero shape λ=.25",
    "B2_shape_l050": "B2 static gate + mean-zero shape λ=.50",
    "B2_shape_l075": "B2 static gate + mean-zero shape λ=.75",
    "B2_shape_l100": "B2 static gate + mean-zero shape λ=1.00",
    "B3_anchored_shape": "B3 static gate + +50-anchored shape priority",
    "B4_shape_risk_k10": "B4 static gate + shape − .10σ priority",
    "B4_shape_risk_k25": "B4 static gate + shape − .25σ priority",
    "B4_shape_risk_k50": "B4 static gate + shape − .50σ priority",
    "B5_anchored_gate_static_priority": "B5 anchored shape gate + static priority",
    "B6_anchored_gate_priority": "B6 anchored shape gate + anchored priority",
    "B7_multi_conf_shape": "B7 static gate + multi-horizon-confidence shape",
}


def _dual(frame: pd.DataFrame, name: str) -> np.ndarray:
    return np.minimum(
        pd.to_numeric(frame[f"bcf__{name}"], errors="coerce").to_numpy(float),
        pd.to_numeric(frame[f"current__{name}"], errors="coerce").to_numpy(float),
    )


def _static_gate(frame: pd.DataFrame) -> np.ndarray:
    return frame["enhanced_base_routed"].fillna(False).astype(bool).to_numpy() & np.isfinite(_dual(frame, "B0_static")) & (_dual(frame, "B0_static") >= THRESHOLD_BPS)


def _select(frame: pd.DataFrame, arm: str, mode: str) -> pd.DataFrame:
    """Select with the declared gate and a single explicit BCF priority."""
    if mode == "static_gate_priority":
        gate = _static_gate(frame)
        priority_name = arm
        gate_name = "B0_static"
    elif mode == "anchored_gate_static_priority":
        gate = frame["enhanced_base_routed"].fillna(False).astype(bool).to_numpy() & (_dual(frame, "B3_anchored_shape") >= THRESHOLD_BPS)
        priority_name = "B0_static"
        gate_name = "B3_anchored_shape"
    elif mode == "anchored_gate_priority":
        gate = frame["enhanced_base_routed"].fillna(False).astype(bool).to_numpy() & (_dual(frame, "B3_anchored_shape") >= THRESHOLD_BPS)
        priority_name = "B3_anchored_shape"
        gate_name = "B3_anchored_shape"
    else:
        raise ValueError(mode)
    out = frame.loc[gate].copy()
    out["gate_score_bps"] = _dual(out, gate_name)
    out["bcf_priority_bps"] = pd.to_numeric(out[f"bcf__{priority_name}"], errors="coerce")
    out = out.loc[np.isfinite(out["bcf_priority_bps"])].copy()
    out = out.sort_values(["__decision_ts__", "bcf_priority_bps", "candidate_id"], ascending=[True, False, True], kind="stable")
    out["selection_rank"] = out.groupby("__decision_ts__", sort=False).cumcount() + 1
    out["arm"] = arm
    out["mode"] = mode
    return out


def _outcome_valid(frame: pd.DataFrame) -> pd.Series:
    return frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce")) & np.isfinite(pd.to_numeric(frame["policy_exit_bar_15m"], errors="coerce"))


def _portfolio_input(selected: pd.DataFrame) -> pd.DataFrame:
    valid = selected.loc[_outcome_valid(selected)].copy()
    if valid.empty:
        return pd.DataFrame()
    timestamp = pd.to_datetime(valid["__decision_ts__"], utc=True)
    exit_bar = pd.to_numeric(valid["policy_exit_bar_15m"], errors="coerce").astype(int)
    rank = valid.groupby("__decision_ts__", sort=False)["bcf_priority_bps"].rank(pct=True, method="average")
    return normalise_candidate_table(pd.DataFrame({
        "timestamp": timestamp, "symbol": valid["__symbol__"].astype(str), "side": "long",
        "strategy_id": "strict_r3_b0_b7", "policy_archetype": "strict_r3_b0_b7",
        "normalized_rank_score": rank.to_numpy(float), "strategy_rank_pct": rank.to_numpy(float),
        "base_strategy_threshold": 0.0, "calibrated_score": valid["bcf_priority_bps"].to_numpy(float),
        "entry_price": pd.to_numeric(valid["policy_entry_price"], errors="coerce"),
        "exit_timestamp": timestamp + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "exit_price": pd.to_numeric(valid["policy_exit_price"], errors="coerce"),
        "net_return": pd.to_numeric(valid["policy_net_bps"], errors="coerce") / 10_000.0,
        "gross_return": pd.to_numeric(valid["policy_gross_bps"], errors="coerce") / 10_000.0,
        "holding_bars": exit_bar + 1, "simple_policy_exit_reason": valid["policy_exit_reason"].astype(str),
        "fees_bps": 100.0, "slippage_bps": 0.0, "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0, "liquidity_capacity_weight": 1.0,
        "source_month": timestamp.dt.strftime("%Y-%m"), "candidate_id": valid["candidate_id"].astype(str),
        "mapped_expected_net_bps": valid["bcf_priority_bps"].to_numpy(float),
    }))


def _portfolio_metrics(selected: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    candidates = _portfolio_input(selected)
    if candidates.empty:
        return {"portfolio_trades": 0, "portfolio_ev_bps": float("nan"), "portfolio_total_bps": float("nan"), "portfolio_max_dd": float("nan")}, pd.DataFrame()
    decisions, equity, _ = replay_candidates(candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perps", initial_wallet=1_000.0)
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    net = pd.to_numeric(accepted.get("position_net_return"), errors="coerce") * 10_000.0
    wallet = pd.to_numeric(equity.get("wallet"), errors="coerce").dropna()
    dd = float((wallet / wallet.cummax() - 1.0).min()) if len(wallet) else float("nan")
    return {
        "portfolio_trades": int(len(accepted)), "portfolio_ev_bps": float(net.mean()) if len(net) else float("nan"),
        "portfolio_total_bps": float(net.sum()) if len(net) else float("nan"), "portfolio_max_dd": dd,
    }, decisions


def _summary(selected: pd.DataFrame, arm: str, mode: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = selected.loc[_outcome_valid(selected)].copy()
    value = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
    top1 = pd.to_numeric(valid.loc[valid["selection_rank"].le(1), "policy_net_bps"], errors="coerce")
    top2 = pd.to_numeric(valid.loc[valid["selection_rank"].le(2), "policy_net_bps"], errors="coerce")
    valid["week"] = valid["__decision_ts__"].dt.strftime("%G-W%V")
    valid["month"] = valid["__decision_ts__"].dt.strftime("%Y-%m")
    weekly = valid.groupby("week", as_index=False, sort=True).agg(trades=("candidate_id", "size"), net_ev_bps=("policy_net_bps", "mean"), total_bps=("policy_net_bps", "sum"))
    monthly = valid.groupby("month", as_index=False, sort=True).agg(trades=("candidate_id", "size"), net_ev_bps=("policy_net_bps", "mean"), total_bps=("policy_net_bps", "sum"))
    portfolio, decisions = _portfolio_metrics(selected)
    result = {
        "arm": arm, "arm_label": ARM_LABELS[arm], "mode": mode,
        "selected": int(len(selected)), "resolved": int(len(valid)), "coverage": float(len(valid) / max(len(selected), 1)),
        "admitted_ev_bps": float(value.mean()), "total_utility_bps": float(value.sum()),
        "hit_gt50": float(value.gt(50).mean()), "hit_gt100": float(value.gt(100).mean()),
        "top1_ev_bps": float(top1.mean()), "top2_ev_bps": float(top2.mean()),
        "weekly_mean_ev_bps": float(weekly["net_ev_bps"].mean()), "weekly_q10_ev_bps": float(weekly["net_ev_bps"].quantile(.10)),
        "weekly_worst_ev_bps": float(weekly["net_ev_bps"].min()), "weekly_positive_fraction": float(weekly["net_ev_bps"].gt(0).mean()),
        "monthly_worst_ev_bps": float(monthly["net_ev_bps"].min()), **portfolio,
    }
    for part in (weekly, monthly, decisions):
        if len(part):
            part.insert(0, "mode", mode)
            part.insert(0, "arm", arm)
    return result, weekly, monthly, decisions


def _markdown(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    view = frame.loc[:, columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(view[column]) and column not in {"selected", "resolved", "portfolio_trades"}:
            view[column] = view[column].map(lambda x: "—" if not np.isfinite(x) else f"{x:.2f}")
    return [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *["| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False, name=None)],
    ]


def run(input_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    panel, state = _read(input_root)
    work, state_audit, anchors = _construct(panel, state)
    # Static B0 reproduction must select exactly the old static gate IDs.
    old_static = _static_gate(work)
    legacy_dual = np.minimum(
        pd.to_numeric(panel["bcf_static_bps"], errors="coerce").to_numpy(float),
        pd.to_numeric(panel["current_static_bps"], errors="coerce").to_numpy(float),
    )
    legacy_static = panel["enhanced_base_routed"].fillna(False).astype(bool).to_numpy() & np.isfinite(legacy_dual) & (legacy_dual >= THRESHOLD_BPS)
    if int(old_static.sum()) != int(legacy_static.sum()):
        raise AssertionError("static gate count changed during B construction")
    plans: list[tuple[str, str]] = [
        ("B0_static", "static_gate_priority"), ("B1_raw_a6", "static_gate_priority"),
        *[(f"B2_shape_l{int(l * 100):03d}", "static_gate_priority") for l in (.25, .50, .75, 1.0)],
        ("B3_anchored_shape", "static_gate_priority"),
        *[(f"B4_shape_risk_k{int(k * 100):02d}", "static_gate_priority") for k in (.10, .25, .50)],
        ("B5_anchored_gate_static_priority", "anchored_gate_static_priority"),
        ("B6_anchored_gate_priority", "anchored_gate_priority"),
        ("B7_multi_conf_shape", "static_gate_priority"),
    ]
    summaries: list[dict[str, Any]] = []
    selected_parts: list[pd.DataFrame] = []
    weekly_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    static_ids: set[str] | None = None
    for position, (arm, mode) in enumerate(plans, start=1):
        print(f"[B0-B7] {position}/{len(plans)} {arm} {mode}", flush=True)
        chosen = _select(work, arm, mode)
        if mode == "static_gate_priority":
            ids = set(chosen["candidate_id"].astype(str))
            if static_ids is None:
                static_ids = ids
            elif ids != static_ids:
                raise AssertionError(f"{arm} changed static-gate candidate identities")
        summary, weekly, monthly, decisions = _summary(chosen, arm, mode)
        summaries.append(summary)
        selected_parts.append(chosen.loc[:, [
            "candidate_id", "__decision_ts__", "__symbol__", "arm", "mode", "selection_rank",
            "gate_score_bps", "bcf_priority_bps", "policy_path_valid", "policy_net_bps",
            "policy_label_available_ts",
        ]].copy())
        weekly_parts.append(weekly)
        monthly_parts.append(monthly)
        if len(decisions):
            keep = [field for field in (
                "arm", "mode", "candidate_id", "timestamp", "symbol", "accepted", "reason",
                "position_net_return", "mapped_expected_net_bps",
            ) if field in decisions.columns]
            decision_parts.append(decisions.loc[:, keep].copy())
        del chosen, summary, weekly, monthly, decisions
        gc.collect()
    metrics = pd.DataFrame(summaries)
    metrics.to_parquet(out / "metrics.parquet", index=False, compression="zstd")
    work.to_parquet(out / "b0_b7_predictions.parquet", index=False, compression="zstd")
    state_audit.to_parquet(out / "shape_state_audit.parquet", index=False, compression="zstd")
    anchors.to_parquet(out / "anchor_audit.parquet", index=False, compression="zstd")
    pd.concat(selected_parts, ignore_index=True).to_parquet(out / "selected_candidates.parquet", index=False, compression="zstd")
    pd.concat(weekly_parts, ignore_index=True).to_parquet(out / "weekly_metrics.parquet", index=False, compression="zstd")
    pd.concat(monthly_parts, ignore_index=True).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    (pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()).to_parquet(out / "portfolio_decisions.parquet", index=False, compression="zstd")
    static = metrics.loc[metrics["mode"].eq("static_gate_priority")].sort_values(["top1_ev_bps", "top2_ev_bps", "portfolio_ev_bps"], ascending=False)
    gates = metrics.loc[metrics["mode"].ne("static_gate_priority")].sort_values(["portfolio_ev_bps", "portfolio_total_bps"], ascending=False)
    report = [
        "# B0--B7 MC1 Level-versus-Shape Ablation",
        "",
        "Strict-prequential Feb--Jul 2026 research. Static package maps, target-free BCF/Current inputs, rich-policy labels, and the global portfolio auction are fixed. No live or exchange component is touched.",
        "",
        "## Static dual +50 gate; only BCF priority varies",
        "",
        "Every row in this section has exactly the same static admitted candidate IDs per timestamp. `selection_rank` and the two-entry portfolio auction use the declared BCF priority directly; no hidden static-dual sort key remains.",
        "",
        *_markdown(static, ["arm", "arm_label", "selected", "admitted_ev_bps", "top1_ev_bps", "top2_ev_bps", "total_utility_bps", "weekly_q10_ev_bps", "weekly_worst_ev_bps", "portfolio_trades", "portfolio_ev_bps", "portfolio_total_bps", "portfolio_max_dd"]),
        "",
        "## Anchored adaptive-gate diagnostics",
        "",
        *_markdown(gates, ["arm", "arm_label", "selected", "admitted_ev_bps", "top1_ev_bps", "top2_ev_bps", "total_utility_bps", "weekly_q10_ev_bps", "weekly_worst_ev_bps", "portfolio_trades", "portfolio_ev_bps", "portfolio_total_bps", "portfolio_max_dd"]),
        "",
        "## Interpretation contract",
        "",
        "- B1 has full raw A6 (level + band) priority authority after the static gate.",
        "- B2 removes the causal global EB level and varies only band shape authority.",
        "- B3 is a monotone score-band curve anchored at static EV=+50; it has priority authority only here.",
        "- B4 subtracts a posterior-predictive residual-risk penalty only from priority.",
        "- B5/B6 are explicitly diagnostic adaptive-gate arms; they do not alter the static-gate result.",
        "- B7 uses 7/21/42-day agreement and score-band SNR only to shrink 21-day shape authority.",
    ]
    (out / "B0_B7_SHAPE_PRIORITY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    correctness = {
        "schema": SCHEMA,
        "research_only_no_live_or_exchange_mutation": True,
        "input_a0_a10_correctness_receipt": str((input_root / "correctness_report.json").resolve()),
        "same_static_gate_candidate_ids_for_all_priority_arms": True,
        "static_gate_is_dual_bcf_current_ge_50_and_base_routed": True,
        "priority_is_direct_bcf_coordinate_without_hidden_static_gate_sort": True,
        "shape_state_uses_only_prior_resolved_policy_labels": True,
        "a6_raw_reconstruction_matches_sealed_projected_a6": bool(state_audit["a6_reconstruction_max_abs_delta_bps"].le(1e-8).all()),
        "anchored_curve_is_timestamp_local_and_uses_no_outcome_inputs": True,
        "policy_outcomes_join_only_for_metrics_and_portfolio_replay": True,
        "portfolio_policy_and_constraints_unchanged": True,
    }
    _once_json(out / "correctness_report.json", correctness)
    _once_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline B0--B7 MC1 shape-priority research; no live mutation",
        "input_root": str(input_root.resolve()), "input_hashes": {
            "predictions": _sha256(input_root / "a0_a10_predictions.parquet"),
            "causal_state": _sha256(input_root / "causal_residual_state.parquet"),
            "correctness": _sha256(input_root / "correctness_report.json"),
        },
        "static_gate": "enhanced_base_routed and min(BCF_static, Current_static) >= +50 bps",
        "shape": "A6 raw score-band EB correction minus posterior global-21d EB correction",
        "B3_anchor": "monotone timestamp-local shape curve anchored at static EV +50 bps",
        "B4_uncertainty": "posterior predictive residual standard deviation, priority-only penalty",
        "B7_confidence": "21d score-band shape authority shrunk by score-band SNR and 7/21/42d agreement",
        "policy_target": "canonical rich policy net bps; embedded cost exactly once",
        "threshold_bps": THRESHOLD_BPS,
        "arms": ARM_LABELS,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.input, args.out))


if __name__ == "__main__":
    main()
