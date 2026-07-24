#!/usr/bin/env python3
"""Run a fixed-row factorial diagnosis of label versus optimized execution.

No parameters are fitted or tuned.  The script holds signals, scores, delayed
entries, one-minute paths, horizon, and cost assumptions fixed, then replaces
one execution component at a time before testing the largest adverse pairs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.diagnostics.execution_factorial import (  # noqa: E402
    EXIT_TYPES,
    assert_fixed_stage_keys,
    exit_transition_matrix,
    interaction_delta,
    paired_variant_summary,
)
from scripts.replay_current_policy_july_1m import _side_params  # noqa: E402
from scripts.run_execution_transfer_waterfall import (  # noqa: E402
    DEFAULT_LABELS,
    DEFAULT_PARENT_SUMMARY,
    DEFAULT_POLICY,
    DEFAULT_SELECTED,
    DEFAULT_STORE,
    _build_multistage_paths,
    _causal_entry_atr,
    _load_deployed_side_params,
    _load_population,
    _simulate_canonical_label_geometry,
)


DEFAULT_OUT = Path(
    "data_perp/reports/may_july_failure_diagnosis_20260722_v1/"
    "execution_policy_factorial_v1"
)

R_TIMEOUT = 0
R_FULL_STOP = 1
R_HARD_TP = 2
R_TRAILING = 3
R_CAPITAL = 4
R_ADVERSE = 5

F_TIMEOUT = 1 << 0
F_TP_LEVEL = 1 << 1
F_TP_ACTIVATION = 1 << 2
F_INITIAL_STOP = 1 << 3
F_TRAILING_ACTIVATION = 1 << 4
F_TRAILING_GAP = 1 << 5
F_CAPITAL = 1 << 6
F_ADVERSE = 1 << 7
F_SPREAD_ANCHOR = 1 << 8
F_EXIT_PRICE = 1 << 9

COMPONENT_FLAGS = {
    "timeout_handling": F_TIMEOUT,
    "tp_level": F_TP_LEVEL,
    "tp_activation": F_TP_ACTIVATION,
    "initial_stop_distance": F_INITIAL_STOP,
    "trailing_activation": F_TRAILING_ACTIVATION,
    "trailing_gap": F_TRAILING_GAP,
    "capital_protection": F_CAPITAL,
    "adverse_exit": F_ADVERSE,
    "spread_adjusted_barrier_anchoring": F_SPREAD_ANCHOR,
    "exit_price_treatment": F_EXIT_PRICE,
}

RUNTIME_ORDER = [
    "spread_adjusted_barrier_anchoring",
    "initial_stop_distance",
    "tp_activation",
    "trailing_activation",
    "trailing_gap",
    "capital_protection",
    "adverse_exit",
    "timeout_handling",
    "exit_price_treatment",
]


@njit(cache=True, inline="always")
def _stop_fill(
    side: float,
    stop: float,
    high: float,
    low: float,
    half_spread_bps: float,
    pessimistic_fill: bool,
) -> float:
    quote = stop
    if pessimistic_fill:
        quote *= 1.0 - side * max(half_spread_bps, 0.0) / 10_000.0
        through = max(stop - low, 0.0) if side > 0.0 else max(high - stop, 0.0)
        gap = min(stop * 15.0 / 10_000.0 + 0.05 * through, stop * 75.0 / 10_000.0)
        quote -= side * gap
    return quote


@njit(cache=True, parallel=True)
def _simulate_factorial(
    open0: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    side: np.ndarray,
    barrier_frac: np.ndarray,
    ref_tp_r: np.ndarray,
    ref_sl_r: np.ndarray,
    ref_trail_r: np.ndarray,
    ref_activation_minutes: np.ndarray,
    atr_frac: np.ndarray,
    opt_sl_mult: np.ndarray,
    opt_activation_mult: np.ndarray,
    opt_activation_cap_frac: np.ndarray,
    opt_trail_power: np.ndarray,
    opt_trail_divisor: np.ndarray,
    opt_giveback_beta: np.ndarray,
    opt_adverse_enabled: np.ndarray,
    opt_adverse_min_mae: np.ndarray,
    opt_adverse_min_speed: np.ndarray,
    opt_adverse_theta: np.ndarray,
    opt_adverse_fast_minutes: np.ndarray,
    opt_adverse_max_mfe: np.ndarray,
    half_spread_bps: np.ndarray,
    flags: int,
    horizon: int,
    deduplicate_stop_spread: bool,
) -> tuple[np.ndarray, ...]:
    n = len(open0)
    gross = np.full(n, np.nan, dtype=np.float64)
    exit_price = np.full(n, np.nan, dtype=np.float64)
    exit_bar = np.full(n, -1, dtype=np.int32)
    reason = np.full(n, R_TIMEOUT, dtype=np.int8)
    mfe = np.full(n, np.nan, dtype=np.float64)
    mae = np.full(n, np.nan, dtype=np.float64)
    time_to_mfe = np.full(n, -1, dtype=np.int32)
    activation_bar = np.full(n, -1, dtype=np.int32)
    valid = np.zeros(n, dtype=np.bool_)
    use_opt_timeout = (flags & F_TIMEOUT) != 0
    use_opt_tp_level = (flags & F_TP_LEVEL) != 0
    use_opt_tp_activation = (flags & F_TP_ACTIVATION) != 0
    use_opt_stop = (flags & F_INITIAL_STOP) != 0
    use_opt_trail_activation = (flags & F_TRAILING_ACTIVATION) != 0
    use_opt_trail_gap = (flags & F_TRAILING_GAP) != 0
    use_opt_capital = (flags & F_CAPITAL) != 0
    use_opt_adverse = (flags & F_ADVERSE) != 0
    use_spread_anchor = (flags & F_SPREAD_ANCHOR) != 0
    use_opt_exit_fill = (flags & F_EXIT_PRICE) != 0

    for i in prange(n):
        raw_entry = float(open0[i])
        if not np.isfinite(raw_entry) or raw_entry <= 0.0:
            continue
        sign = 1.0 if side[i] >= 0.0 else -1.0
        spread = max(float(half_spread_bps[i]), 0.0)
        entry = raw_entry * (1.0 + sign * spread / 10_000.0) if use_spread_anchor else raw_entry
        barrier = entry * max(float(barrier_frac[i]), 1e-8)
        atr = entry * max(float(atr_frac[i]), 1e-8)

        stop_dist = max(float(ref_sl_r[i]), 0.0) * barrier
        if use_opt_stop:
            stop_dist = max(float(opt_sl_mult[i]), 0.1) * atr
        full_stop = entry - sign * stop_dist

        # The deployed winner has hard_tp_abs_pct=0.  Keeping this arm explicit
        # proves that replacing the absent hard TP is an identity operation.
        hard_tp_dist = 0.0 if use_opt_tp_level else 0.0

        activation_dist = max(float(ref_tp_r[i]), 0.0) * barrier
        if use_opt_trail_activation:
            activation_dist = max(float(opt_activation_mult[i]), 0.05) * atr
            cap = max(float(opt_activation_cap_frac[i]), 0.0)
            if cap > 0.0:
                activation_dist = min(activation_dist, entry * cap)
        activation_deadline = max(int(ref_activation_minutes[i]), 1)
        if use_opt_tp_activation:
            activation_deadline = horizon

        max_fav = 0.0
        max_adv = 0.0
        best_fav_px = entry
        activated = False
        completed = False
        capital_active = False
        capital_stop = full_stop

        for j in range(horizon):
            hi = float(high[i, j])
            lo = float(low[i, j])
            cl = float(close[i, j])
            op = entry if j == 0 else float(close[i, j - 1])
            if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(cl) and np.isfinite(op)):
                break

            cur_fav = max(hi - entry, 0.0) if sign > 0.0 else max(entry - lo, 0.0)
            cur_adv = max(entry - lo, 0.0) if sign > 0.0 else max(hi - entry, 0.0)

            # Existing state acts on this bar; state crossed during this bar is
            # only available on the next bar.
            active_stop = capital_stop if capital_active else full_stop
            trigger = active_stop
            if use_spread_anchor:
                trigger = (
                    active_stop / max(1.0 - spread / 10_000.0, 1e-12)
                    if sign > 0.0
                    else active_stop / (1.0 + spread / 10_000.0)
                )
            stop_hit = lo <= trigger if sign > 0.0 else hi >= trigger

            trail_stop = full_stop
            trail_hit = False
            if activated:
                if use_opt_trail_gap:
                    u = max_fav / max(atr, 1e-12)
                    dynamic = min(max((max(u, 0.0) / max(float(opt_trail_divisor[i]), 0.05)) ** max(float(opt_trail_power[i]), 0.05), 0.0), 1.0)
                    gap = max(max_fav * max(float(opt_giveback_beta[i]), 0.0) * (1.0 - dynamic), entry * 0.003)
                else:
                    gap = max(float(ref_trail_r[i]), 1e-8) * barrier
                trail_stop = best_fav_px - sign * gap
                trail_trigger = trail_stop
                if use_spread_anchor:
                    trail_trigger = (
                        trail_stop / max(1.0 - spread / 10_000.0, 1e-12)
                        if sign > 0.0
                        else trail_stop / (1.0 + spread / 10_000.0)
                    )
                trail_hit = lo <= trail_trigger if sign > 0.0 else hi >= trail_trigger

            if stop_hit or trail_hit:
                chosen = trail_stop
                chosen_reason = R_TRAILING
                if stop_hit and (not trail_hit or abs(active_stop - op) <= abs(trail_stop - op)):
                    chosen = active_stop
                    chosen_reason = R_CAPITAL if capital_active else R_FULL_STOP
                fill_spread = (
                    0.0
                    if deduplicate_stop_spread or not use_spread_anchor
                    else spread
                )
                exit_price[i] = _stop_fill(sign, chosen, hi, lo, fill_spread, use_opt_exit_fill)
                exit_bar[i] = j
                reason[i] = chosen_reason
                completed = True
                break

            if hard_tp_dist > 0.0:
                target = entry + sign * hard_tp_dist
                target_hit = hi >= target if sign > 0.0 else lo <= target
                if target_hit:
                    exit_price[i] = target * (1.0 - sign * spread / 10_000.0) if use_opt_exit_fill and use_spread_anchor else target
                    exit_bar[i] = j
                    reason[i] = R_HARD_TP
                    completed = True
                    break

            if use_opt_adverse and bool(opt_adverse_enabled[i]) and j <= int(opt_adverse_fast_minutes[i]):
                adv_mfe_atr = max(max_fav, cur_fav) / max(atr, 1e-12)
                adv_mae_atr = max(max_adv, cur_adv) / max(atr, 1e-12)
                speed = adv_mae_atr / max((j + 1) / 15.0, 1.0 / 15.0)
                score = np.log1p(0.75) + np.log1p(max(adv_mae_atr, 0.0)) + np.log1p(max(speed, 0.0))
                eligible = (
                    adv_mae_atr >= float(opt_adverse_min_mae[i])
                    and speed >= float(opt_adverse_min_speed[i])
                    and adv_mfe_atr <= float(opt_adverse_max_mfe[i])
                )
                if eligible and score > float(opt_adverse_theta[i]):
                    exit_price[i] = cl * (1.0 - sign * spread / 10_000.0) if use_opt_exit_fill and use_spread_anchor else cl
                    exit_bar[i] = j
                    reason[i] = R_ADVERSE
                    completed = True
                    break

            if cur_fav > max_fav:
                max_fav = cur_fav
                time_to_mfe[i] = j
            max_adv = max(max_adv, cur_adv)

            if not activated and max_fav >= activation_dist and j + 1 <= activation_deadline:
                activated = True
                activation_bar[i] = j
            if activated:
                best_fav_px = max(best_fav_px, hi) if sign > 0.0 else min(best_fav_px, lo)

            # The promoted policy has capital protection disabled.  This branch
            # remains explicit so the identity substitution is measured.
            if use_opt_capital:
                capital_active = False

        if not completed:
            last = -1
            for j in range(horizon - 1, -1, -1):
                if np.isfinite(close[i, j]):
                    last = j
                    break
            if last < 0:
                continue
            px = float(close[i, last])
            if use_opt_timeout and use_opt_exit_fill and use_spread_anchor:
                px *= 1.0 - sign * spread / 10_000.0
            exit_price[i] = px
            exit_bar[i] = last
            reason[i] = R_TIMEOUT
        gross[i] = sign * (exit_price[i] / entry - 1.0)
        mfe[i] = max_fav / entry
        mae[i] = max_adv / entry
        valid[i] = True
    return gross, exit_price, exit_bar, reason, mfe, mae, time_to_mfe, activation_bar, valid


def _side_arrays(rows: pd.DataFrame, payload: dict[str, Any]) -> dict[str, np.ndarray]:
    params = _side_params(payload)
    names = rows["side_name"].astype(str).to_numpy()

    def values(key: str, default: float) -> np.ndarray:
        return np.asarray([float(params[name].get(key, default)) for name in names], dtype=np.float64)

    # Preserve the exact prior bridge behavior.  It passed raw `*_bars` keys to
    # a kernel expecting `*_minutes`, so these defaults were used.
    return {
        "sl_mult": values("sl_mult", 2.5),
        "activation_mult": values("trailing_activation_mult", 1.5),
        "activation_cap": values("trailing_activation_cap_pct", 0.0),
        "trail_power": values("trailing_power", 1.5),
        "trail_divisor": values("trailing_squash_divisor", 2.0),
        "giveback_beta": values("giveback_beta", 0.5),
        "adverse_enabled": np.asarray([bool(params[name].get("adverse_exit_enabled", False)) for name in names]),
        "adverse_min_mae": values("adverse_exit_min_mae_atr", 1.0),
        "adverse_min_speed": values("adverse_exit_min_speed_per_15m", 0.3),
        "adverse_theta": values("adverse_exit_theta", 1e9),
        "adverse_fast_minutes": values("adverse_exit_fast_minutes", 0.0),
        "adverse_max_mfe": values("adverse_exit_max_mfe_atr", 0.25),
    }


def _make_frame(
    rows: pd.DataFrame,
    *,
    variant: str,
    flags: int,
    result: tuple[np.ndarray, ...],
    fee_round_trip: float,
) -> pd.DataFrame:
    gross, exit_price, exit_bar, reason, mfe, mae, time_to_mfe, activation_bar, valid = result
    out = rows[["timestamp", "symbol", "side_name", "archetype", "rank_score"]].copy()
    out["row_id"] = np.arange(len(rows), dtype=np.int32)
    out["variant"] = variant
    out["flags"] = int(flags)
    out["gross_return"] = gross
    fee_side = float(fee_round_trip) / 2.0
    out["net_return"] = gross - fee_side - fee_side * (1.0 + gross)
    out["exit_price"] = exit_price
    out["exit_bar"] = exit_bar
    out["holding_minutes"] = exit_bar + 1
    out["exit_reason_code"] = reason
    mapping = np.asarray(EXIT_TYPES, dtype=object)
    out["exit_type"] = mapping[np.clip(reason, 0, len(mapping) - 1)]
    out["mfe"] = mfe
    out["mae"] = mae
    out["time_to_mfe_minutes"] = time_to_mfe + 1
    out["activation_bar"] = activation_bar
    out["valid"] = valid & np.isfinite(gross)
    return out


def _slice_specs(frame: pd.DataFrame) -> list[tuple[str, list[str], pd.DataFrame]]:
    frame = frame.copy()
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    ranked = frame.loc[frame["variant"].eq("reference")].sort_values(
        ["rank_score", "timestamp", "symbol", "side_name"],
        ascending=[False, True, True, True], kind="stable",
    )
    key_cols = ["timestamp", "symbol", "side_name"]
    tails = []
    for name, frac in (("top01", .01), ("top02", .02), ("top05", .05), ("top10", .10), ("top20", .20)):
        part = ranked.iloc[:max(1, int(np.ceil(len(ranked) * frac)))][key_cols].copy()
        part["score_tail"] = name
        tails.append(part)
    with_tails = frame.merge(pd.concat(tails, ignore_index=True), on=key_cols, how="inner")
    return [
        ("overall", [], frame),
        ("side", ["side_name"], frame),
        ("month", ["month"], frame),
        ("side_month", ["side_name", "month"], frame),
        ("archetype", ["side_name", "archetype"], frame),
        ("score_tail", ["score_tail"], with_tails),
        ("side_score_tail", ["side_name", "score_tail"], with_tails),
        ("month_score_tail", ["month", "score_tail"], with_tails),
        ("archetype_score_tail", ["side_name", "archetype", "score_tail"], with_tails),
    ]


def _summaries(ledger: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for scope, columns, source in _slice_specs(ledger):
        iterator = [((), source)] if not columns else source.groupby(columns, observed=True, dropna=False)
        for key, group in iterator:
            key = key if isinstance(key, tuple) else (key,)
            identity = dict(zip(columns, key))
            reference = group.loc[group["variant"].eq("reference")].sort_values(
                ["timestamp", "symbol", "side_name"], kind="stable"
            )
            for variant, part in group.groupby("variant", observed=True, sort=False):
                part = part.sort_values(["timestamp", "symbol", "side_name"], kind="stable")
                records.append({"scope": scope, **identity, "variant": variant, **paired_variant_summary(part, reference)})
    return pd.DataFrame(records)


def _timeout_diagnostics(
    ledger: pd.DataFrame,
    eight_hour_results: dict[int, tuple[np.ndarray, ...]],
    full_results: dict[int, tuple[np.ndarray, ...]],
    barrier_frac: np.ndarray,
    activation_distance_frac: dict[int, np.ndarray],
    frozen_target_reached: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, part in ledger.groupby("variant", observed=True, sort=False):
        if variant == "reference_24h_original_geometry" or variant.startswith("audit__"):
            continue
        part = part.reset_index(drop=True)
        timeout = part["exit_type"].eq("timeout").to_numpy()
        if not timeout.any():
            continue
        full = full_results[int(part["flags"].iloc[0])]
        gross24, _, exit24, reason24, mfe24, _, ttm24, activation24, valid24 = full
        for i in np.flatnonzero(timeout):
            row_id = int(part.at[i, "row_id"])
            flags = int(part.at[i, "flags"])
            mfe8 = float(part.at[i, "mfe"])
            barrier = max(float(barrier_frac[row_id]), 1e-12)
            activation = max(float(activation_distance_frac[flags][row_id]), 1e-12)
            no_spread_flags = flags & ~F_SPREAD_ANCHOR
            no_spread_reason = eight_hour_results[no_spread_flags][3][row_id]
            rows.append({
                "row_id": row_id,
                "variant": variant,
                "timestamp": part.at[i, "timestamp"],
                "symbol": part.at[i, "symbol"],
                "side_name": part.at[i, "side_name"],
                "archetype": part.at[i, "archetype"],
                "final_return_8h": float(part.at[i, "gross_return"]),
                "mfe_8h": mfe8,
                "mfe_r": mfe8 / barrier,
                "reached_0_25r": int(mfe8 >= 0.25 * barrier),
                "reached_0_50r": int(mfe8 >= 0.50 * barrier),
                "reached_1_00r": int(mfe8 >= barrier),
                "reached_target_equivalent_mfe": int(mfe8 >= activation),
                "time_to_mfe_8h": int(part.at[i, "time_to_mfe_minutes"]),
                "never_reached_activation": int(part.at[i, "activation_bar"] < 0),
                "reached_activation_no_trailing_exit": int(part.at[i, "activation_bar"] >= 0),
                "profitable_at_timeout": int(part.at[i, "gross_return"] > 0.0),
                "losing_at_timeout": int(part.at[i, "gross_return"] <= 0.0),
                "favorable_excursion_insufficient": int(mfe8 < activation),
                "barrier_unreachable_after_delayed_entry": int(
                    bool(frozen_target_reached[row_id]) and part.at[i, "activation_bar"] < 0
                ),
                "spread_adjustment_moved_effective_target": int(
                    bool(flags & F_SPREAD_ANCHOR) and no_spread_reason != R_TIMEOUT
                ),
                "favorable_excursion_after_8h": int(
                    valid24[row_id]
                    and ttm24[row_id] >= 480
                    and mfe24[row_id] > mfe8 + 1e-12
                ),
                "late_success_after_8h": int(valid24[row_id] and exit24[row_id] >= 480 and reason24[row_id] != R_TIMEOUT and gross24[row_id] > 0.0),
                "full_24h_return": float(gross24[row_id]),
                "full_24h_mfe": float(mfe24[row_id]),
                "full_24h_time_to_mfe": int(ttm24[row_id] + 1),
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--parent-summary", type=Path, default=DEFAULT_PARENT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--horizon-minutes", type=int, default=480)
    parser.add_argument("--counterfactual-horizon-minutes", type=int, default=1440)
    parser.add_argument("--fee-round-trip", type=float, default=0.003)
    parser.add_argument("--negative-components", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_population(args.selected, args.labels_dir)
    paths, path_manifest = _build_multistage_paths(
        rows,
        store_root=args.store,
        offsets_minutes={"signal_reference": 0, "signal_close": 60, "delay_5m": 65},
        horizon_minutes=int(args.counterfactual_horizon_minutes),
    )
    path = paths["delay_5m"]
    signal_reference_path = paths["signal_reference"]
    signal_close_path = paths["signal_close"]
    policy_rows = rows.copy()
    policy_rows["timestamp"] = rows["timestamp"] + pd.Timedelta(hours=1)
    policy_rows["side"] = rows["side_sign"]
    policy_rows["rank_pct"] = rows["rank_score"]
    deployed, _ = _load_deployed_side_params(args.parent_summary)
    atr, atr_audit, atr_manifest = _causal_entry_atr(
        policy_rows, store_root=args.store, deployed_by_side=deployed,
        parent_summary=args.parent_summary, warmup_hours=48,
    )
    payload = json.loads(args.policy.read_text(encoding="utf-8"))
    opt = _side_arrays(rows, payload)
    inputs = (
        path[0][:, 0].astype(np.float64), path[1], path[2], path[3],
        rows["side_sign"].to_numpy(float),
        pd.to_numeric(rows["__barrier_pct__"], errors="coerce").to_numpy(float),
        pd.to_numeric(rows["__archetype_policy_tp_r__"], errors="coerce").to_numpy(float),
        pd.to_numeric(rows["__archetype_policy_sl_r__"], errors="coerce").to_numpy(float),
        pd.to_numeric(rows["__archetype_policy_trail_r__"], errors="coerce").to_numpy(float),
        (15.0 * pd.to_numeric(rows["__archetype_policy_max_bars_to_mfe__"], errors="coerce")).to_numpy(float),
        atr, opt["sl_mult"], opt["activation_mult"], opt["activation_cap"],
        opt["trail_power"], opt["trail_divisor"], opt["giveback_beta"],
        opt["adverse_enabled"], opt["adverse_min_mae"], opt["adverse_min_speed"],
        opt["adverse_theta"], opt["adverse_fast_minutes"], opt["adverse_max_mfe"],
        pd.to_numeric(rows["p90_spread_bps"], errors="coerce").fillna(0.0).to_numpy(float) / 2.0,
    )

    complete = np.isfinite(path[0]).all(axis=1) & np.isfinite(path[1]).all(axis=1) & np.isfinite(path[2]).all(axis=1) & np.isfinite(path[3]).all(axis=1) & np.isfinite(atr)
    # Match the already established all-stage common population exactly.
    common_path = args.output_dir.parent / "execution_transfer_waterfall_v1/identical_row_stage_ledger.parquet"
    common = pd.read_parquet(common_path, columns=["timestamp", "symbol", "side_name", "stage"])
    common = common.loc[common["stage"].eq("label_1m_delay_5m_reanchored_8h"), ["timestamp", "symbol", "side_name"]]
    common_idx = pd.MultiIndex.from_frame(common)
    row_idx = pd.MultiIndex.from_frame(rows[["timestamp", "symbol", "side_name"]])
    keep = complete & row_idx.isin(common_idx)
    rows = rows.loc[keep].reset_index(drop=True)
    inputs = tuple(value[keep] for value in inputs)
    signal_inputs = (
        signal_close_path[0][keep, 0].astype(np.float64),
        signal_close_path[1][keep], signal_close_path[2][keep],
        signal_close_path[3][keep], *inputs[4:],
    )
    if len(rows) != 3560:
        raise RuntimeError(f"Expected fixed 3,560-row population, found {len(rows):,}")

    def simulate(flags: int, horizon: int) -> tuple[np.ndarray, ...]:
        return _simulate_factorial(*inputs, int(flags), int(horizon), False)

    def simulate_signal_close(flags: int, horizon: int) -> tuple[np.ndarray, ...]:
        return _simulate_factorial(*signal_inputs, int(flags), int(horizon), False)

    variants: list[tuple[str, int, str]] = [("reference", 0, "reference")]
    variants.extend((f"single__{name}", flag, "single") for name, flag in COMPONENT_FLAGS.items())
    frames: list[pd.DataFrame] = []
    result8: dict[int, tuple[np.ndarray, ...]] = {}
    result24: dict[int, tuple[np.ndarray, ...]] = {}
    for name, flags, _ in variants:
        result8[flags] = simulate(flags, int(args.horizon_minutes))
        frames.append(_make_frame(rows, variant=name, flags=flags, result=result8[flags], fee_round_trip=float(args.fee_round_trip)))
    initial = pd.concat(frames, ignore_index=True)
    initial_summary = _summaries(initial)
    singles = initial_summary.loc[
        initial_summary["scope"].eq("overall") & initial_summary["variant"].str.startswith("single__")
    ].nsmallest(int(args.negative_components), "ev_delta")
    negative = [str(value).removeprefix("single__") for value in singles["variant"]]

    pair_variants = []
    for a, b in itertools.combinations(negative, 2):
        pair_variants.append((f"pair__{a}__{b}", COMPONENT_FLAGS[a] | COMPONENT_FLAGS[b], "pair"))
    required_pair = (
        "spread_adjusted_barrier_anchoring", "capital_protection"
    )
    required_pair_name = f"pair__{required_pair[0]}__{required_pair[1]}"
    if all(name != required_pair_name for name, _, _ in pair_variants):
        pair_variants.append((
            required_pair_name,
            COMPONENT_FLAGS[required_pair[0]] | COMPONENT_FLAGS[required_pair[1]],
            "pair_priority",
        ))
    cumulative = 0
    reconstruction = []
    for number, name in enumerate(RUNTIME_ORDER, start=1):
        cumulative |= COMPONENT_FLAGS[name]
        reconstruction.append((f"runtime_{number:02d}__{name}", cumulative, "runtime_reconstruction"))
    all_components = 0
    for flag in COMPONENT_FLAGS.values():
        all_components |= flag
    reconstruction.append(("optimized_all_components", all_components, "full"))

    ideal_fill_geometry_flags = all_components & ~F_SPREAD_ANCHOR & ~F_EXIT_PRICE
    raw_geometry_flags = all_components & ~F_SPREAD_ANCHOR

    for name, flags, _ in [*pair_variants, *reconstruction]:
        if flags not in result8:
            result8[flags] = simulate(flags, int(args.horizon_minutes))
        frames.append(_make_frame(rows, variant=name, flags=flags, result=result8[flags], fee_round_trip=float(args.fee_round_trip)))
    if ideal_fill_geometry_flags not in result8:
        result8[ideal_fill_geometry_flags] = simulate(
            ideal_fill_geometry_flags, int(args.horizon_minutes)
        )
    frames.append(_make_frame(
        rows, variant="optimized_geometry_ideal_fill", flags=ideal_fill_geometry_flags,
        result=result8[ideal_fill_geometry_flags], fee_round_trip=float(args.fee_round_trip),
    ))
    if raw_geometry_flags not in result8:
        result8[raw_geometry_flags] = simulate(
            raw_geometry_flags, int(args.horizon_minutes)
        )
    frames.append(_make_frame(
        rows, variant="optimized_raw_geometry", flags=raw_geometry_flags,
        result=result8[raw_geometry_flags], fee_round_trip=float(args.fee_round_trip),
    ))
    deduplicated_stop_spread = _simulate_factorial(
        *inputs, int(all_components), int(args.horizon_minutes), True
    )
    frames.append(_make_frame(
        rows, variant="audit__full_without_duplicate_stop_spread",
        flags=all_components, result=deduplicated_stop_spread,
        fee_round_trip=float(args.fee_round_trip),
    ))
    reference_24h_result = simulate(0, int(args.counterfactual_horizon_minutes))
    frames.append(_make_frame(
        rows, variant="reference_24h_original_geometry", flags=0,
        result=reference_24h_result, fee_round_trip=float(args.fee_round_trip),
    ))
    ledger = pd.concat(frames, ignore_index=True)
    assert_fixed_stage_keys(ledger, key_columns=["timestamp", "symbol", "side_name"])
    summary = _summaries(ledger)

    overall = summary.loc[summary["scope"].eq("overall")].set_index("variant")
    interaction_rows = []
    for name, _, _ in pair_variants:
        _, a, b = name.split("__", 2)
        interaction_rows.append({
            "pair": name, "component_a": a, "component_b": b,
            "gross_interaction_delta": interaction_delta(
                overall.at[name, "gross_ev"], overall.at[f"single__{a}", "gross_ev"],
                overall.at[f"single__{b}", "gross_ev"], overall.at["reference", "gross_ev"],
            ),
            "net_interaction_delta": interaction_delta(
                overall.at[name, "net_ev"], overall.at[f"single__{a}", "net_ev"],
                overall.at[f"single__{b}", "net_ev"], overall.at["reference", "net_ev"],
            ),
        })
    interactions = pd.DataFrame(interaction_rows)

    # Required runtime interactions that involve the already-fixed horizon or
    # delayed-entry contract are assessed as auxiliary two-factor contrasts.
    ref8 = float(overall.at["reference", "gross_ev"])
    ref24_result = simulate(0, int(args.counterfactual_horizon_minutes))
    ref24_ev = float(np.nanmean(ref24_result[0]))
    no_delay_ref = simulate_signal_close(0, int(args.horizon_minutes))
    no_delay_ref_ev = float(np.nanmean(no_delay_ref[0]))
    priority_rows = []
    for component in ("tp_activation", "trailing_activation"):
        flag = COMPONENT_FLAGS[component]
        pair_ev = float(np.nanmean(simulate(flag, int(args.counterfactual_horizon_minutes))[0]))
        priority_rows.append({
            "pair": f"priority__horizon24h__{component}",
            "component_a": "horizon24h", "component_b": component,
            "gross_interaction_delta": interaction_delta(
                pair_ev, ref24_ev, float(overall.at[f"single__{component}", "gross_ev"]), ref8
            ),
            "net_interaction_delta": interaction_delta(
                pair_ev - args.fee_round_trip, ref24_ev - args.fee_round_trip,
                float(overall.at[f"single__{component}", "net_ev"]),
                float(overall.at["reference", "net_ev"]),
            ),
        })
    for component in ("initial_stop_distance", "tp_level"):
        flag = COMPONENT_FLAGS[component]
        pair_ev = float(np.nanmean(simulate_signal_close(flag, int(args.horizon_minutes))[0]))
        priority_rows.append({
            "pair": f"priority__remove_delay5m__{component}",
            "component_a": "remove_delay5m", "component_b": component,
            "gross_interaction_delta": interaction_delta(
                pair_ev, no_delay_ref_ev, float(overall.at[f"single__{component}", "gross_ev"]), ref8
            ),
            "net_interaction_delta": interaction_delta(
                pair_ev - args.fee_round_trip, no_delay_ref_ev - args.fee_round_trip,
                float(overall.at[f"single__{component}", "net_ev"]),
                float(overall.at["reference", "net_ev"]),
            ),
        })
    interactions = pd.concat([interactions, pd.DataFrame(priority_rows)], ignore_index=True)

    transitions = []
    reference = ledger.loc[ledger["variant"].eq("reference")].sort_values(["timestamp", "symbol", "side_name"])
    for variant, part in ledger.groupby("variant", observed=True, sort=False):
        if variant == "reference":
            continue
        part = part.sort_values(["timestamp", "symbol", "side_name"])
        transitions.append(exit_transition_matrix(part, reference, variant=variant))
    transition_frame = pd.concat(transitions, ignore_index=True)

    # Full 24-hour paths for timeout counterfactuals.  Also save exact returns
    # at 12h and 16h for rows that timed out in the corresponding 8h variant.
    timeout_frames = []
    for flags in sorted(set(int(value) for value in ledger["flags"].unique())):
        result24[flags] = simulate(flags, int(args.counterfactual_horizon_minutes))
        no_spread_flags = flags & ~F_SPREAD_ANCHOR
        if no_spread_flags not in result8:
            result8[no_spread_flags] = simulate(
                no_spread_flags, int(args.horizon_minutes)
            )
    barrier_frac = inputs[5]
    activation_distance_frac: dict[int, np.ndarray] = {}
    for flags in sorted(set(int(value) for value in ledger["flags"].unique())):
        distance = inputs[6] * barrier_frac
        if flags & F_TRAILING_ACTIVATION:
            distance = inputs[12] * inputs[10]
            distance = np.where(
                inputs[13] > 0.0,
                np.minimum(distance, inputs[13]),
                distance,
            )
        activation_distance_frac[flags] = distance
    signal_close_entry = signal_close_path[0][keep, 0].astype(np.float64)
    delayed_high = inputs[1][:, : int(args.horizon_minutes)]
    delayed_low = inputs[2][:, : int(args.horizon_minutes)]
    side_sign = inputs[4]
    frozen_target = signal_close_entry * (
        1.0 + side_sign * inputs[6] * barrier_frac
    )
    frozen_target_reached = np.where(
        side_sign > 0.0,
        np.nanmax(delayed_high, axis=1) >= frozen_target,
        np.nanmin(delayed_low, axis=1) <= frozen_target,
    )
    timeout_rows = _timeout_diagnostics(
        ledger, result8, result24, barrier_frac, activation_distance_frac,
        frozen_target_reached,
    )
    for horizon in (720, 960):
        cache: dict[int, tuple[np.ndarray, ...]] = {}
        for flags in sorted(set(int(value) for value in timeout_rows.merge(
            ledger[["variant", "flags"]].drop_duplicates(), on="variant", how="left"
        )["flags"].dropna().astype(int))):
            cache[flags] = simulate(flags, horizon)
        mapped = timeout_rows.merge(ledger[["variant", "flags"]].drop_duplicates(), on="variant", how="left")
        mapped[f"return_{horizon // 60}h"] = [
            cache[int(flag)][0][int(row_id)]
            for row_id, flag in zip(mapped["row_id"], mapped["flags"])
        ]
        timeout_rows = mapped.drop(columns="flags")

    timeout_rows["month"] = pd.to_datetime(timeout_rows["timestamp"], utc=True).dt.strftime("%Y-%m")
    timeout_summaries = []
    for scope, group_columns in (
        ("overall", ["variant"]),
        ("side", ["variant", "side_name"]),
        ("month", ["variant", "month"]),
        ("archetype", ["variant", "side_name", "archetype"]),
    ):
        part = timeout_rows.groupby(group_columns, observed=True, dropna=False).agg(
            timeout_trades=("symbol", "size"),
            mean_final_return_8h=("final_return_8h", "mean"),
            mean_mfe_8h=("mfe_8h", "mean"),
            mean_time_to_mfe_8h=("time_to_mfe_8h", "mean"),
            never_reached_activation_rate=("never_reached_activation", "mean"),
            activated_without_trailing_exit_rate=("reached_activation_no_trailing_exit", "mean"),
            insufficient_favorable_excursion_rate=("favorable_excursion_insufficient", "mean"),
            barrier_unreachable_after_delay_rate=("barrier_unreachable_after_delayed_entry", "mean"),
            spread_moved_target_rate=("spread_adjustment_moved_effective_target", "mean"),
            favorable_excursion_after_8h_rate=("favorable_excursion_after_8h", "mean"),
            profitable_timeout_rate=("profitable_at_timeout", "mean"),
            losing_timeout_rate=("losing_at_timeout", "mean"),
            late_success_after_8h_rate=("late_success_after_8h", "mean"),
            reached_0_25r_rate=("reached_0_25r", "mean"),
            reached_0_50r_rate=("reached_0_50r", "mean"),
            reached_1_00r_rate=("reached_1_00r", "mean"),
            reached_target_equivalent_rate=("reached_target_equivalent_mfe", "mean"),
            mean_return_12h=("return_12h", "mean"),
            mean_return_16h=("return_16h", "mean"),
            mean_return_24h=("full_24h_return", "mean"),
        ).reset_index()
        part.insert(0, "scope", scope)
        timeout_summaries.append(part)
    timeout_summary = pd.concat(timeout_summaries, ignore_index=True, sort=False)

    capture = ledger.copy()
    capture["mfe_capture_ratio"] = np.divide(
        capture["gross_return"], capture["mfe"],
        out=np.full(len(capture), np.nan), where=capture["mfe"].to_numpy() > 1e-12,
    )
    capture_summary = capture.groupby(
        ["variant", "side_name", "exit_type"], observed=True, dropna=False
    ).agg(
        trades=("symbol", "size"), gross_ev=("gross_return", "mean"),
        mean_mfe=("mfe", "mean"), mean_mae=("mae", "mean"),
        mean_capture_ratio=("mfe_capture_ratio", "mean"),
        mean_holding_minutes=("holding_minutes", "mean"),
    ).reset_index()

    # Barrier/spread accounting audit.
    audit_rows = []
    raw_entry = inputs[0]
    spread = inputs[-1]
    signal_reference_entry = signal_reference_path[0][keep, 0].astype(np.float64)
    anchors = (
        ("signal_reference", signal_reference_entry),
        ("causal_signal_close", signal_close_entry),
        ("delayed_executable_mid", raw_entry),
        ("delayed_executable_quote", raw_entry * (1.0 + inputs[4] * spread / 10_000.0)),
    )
    for label, anchor_price in anchors:
        target_level = anchor_price * (1.0 + inputs[4] * inputs[6] * barrier_frac)
        stop_level = anchor_price * (1.0 - inputs[4] * inputs[7] * barrier_frac)
        effective_target_bps = inputs[4] * (target_level / raw_entry - 1.0) * 10_000.0
        effective_stop_bps = -inputs[4] * (stop_level / raw_entry - 1.0) * 10_000.0
        anchor = barrier_frac
        audit_rows.append(pd.DataFrame({
            "timestamp": rows["timestamp"], "symbol": rows["symbol"], "side_name": rows["side_name"],
            "anchor": label, "anchor_price": anchor_price,
            "barrier_distance_bps": 10_000.0 * anchor,
            "barrier_distance_atr": anchor / np.maximum(inputs[10], 1e-12),
            "effective_target_distance_bps_from_delayed_mid": effective_target_bps,
            "effective_stop_distance_bps_from_delayed_mid": effective_stop_bps,
        }))
    barrier_audit = pd.concat(audit_rows, ignore_index=True)
    spread_audit = pd.DataFrame([
        {"channel": "executable_entry_price", "reference": False, "optimized": True, "economic_role": "entry half-spread"},
        {"channel": "barrier_placement", "reference": False, "optimized": True, "economic_role": "geometry reanchored to executable entry"},
        {"channel": "trigger_comparison", "reference": False, "optimized": True, "economic_role": "exit quote must cross stop"},
        {"channel": "trailing_reference_high_low", "reference": False, "optimized": False, "economic_role": "raw OHLC state; no spread"},
        {"channel": "exit_fill", "reference": False, "optimized": True, "economic_role": "exit half-spread plus stop-gap proxy"},
        {"channel": "posthoc_pnl_deduction", "reference": False, "optimized": False, "economic_role": "forbidden; spread already in prices"},
        {"channel": "round_trip_fee", "reference": True, "optimized": True, "economic_role": f"charged once at {args.fee_round_trip:.6f}"},
    ])

    ledger.to_parquet(args.output_dir / "factorial_row_ledger.parquet", index=False)
    summary.to_csv(args.output_dir / "factorial_metrics.csv", index=False)
    interactions.to_csv(args.output_dir / "pairwise_interactions.csv", index=False)
    transition_frame.to_csv(args.output_dir / "exit_transition_matrices.csv", index=False)
    timeout_rows.to_parquet(args.output_dir / "timeout_counterfactuals.parquet", index=False)
    timeout_summary.to_csv(args.output_dir / "timeout_decomposition.csv", index=False)
    capture_summary.to_csv(args.output_dir / "mfe_capture_by_exit.csv", index=False)
    barrier_audit.to_parquet(args.output_dir / "barrier_anchor_audit.parquet", index=False)
    spread_audit.to_csv(args.output_dir / "spread_accounting_audit.csv", index=False)
    atr_audit.to_parquet(args.output_dir / "causal_atr_audit.parquet", index=False)

    manifest = {
        "reference": "label_1m_delay_5m_reanchored_8h",
        "rows": len(rows), "entry_delay_minutes": 5,
        "horizon_minutes": int(args.horizon_minutes),
        "counterfactual_horizons_minutes": [720, 960, int(args.counterfactual_horizon_minutes)],
        "fee_round_trip": float(args.fee_round_trip),
        "fixed_scores": True, "fitted_or_tuned": False,
        "largest_negative_components": negative,
        "path_manifest": path_manifest, "atr_manifest": atr_manifest,
        "component_semantics": {
            "tp_level": "identity: hard_tp_abs_pct is zero in both reference and deployed winner",
            "tp_activation": "replace finite label arming deadline with deployed unlimited arming window; threshold held fixed",
            "trailing_activation": "replace side/archetype barrier-relative activation distance with side-parent ATR-relative distance",
            "capital_protection": "identity: disabled in deployed winner",
            "timeout_handling": "identity unless combined with optimized exit-price/spread treatment",
        },
        "bridge_compatibility": {
            "preserved": "raw artifact keys are passed exactly as in the prior waterfall",
            "known_key_mismatch": "*_bars fields do not populate constrained-kernel *_minutes fields; effective decay and adverse window use kernel defaults",
        },
        "spread_double_count_audit": {
            "variant": "audit__full_without_duplicate_stop_spread",
            "definition": "keep entry spread, quote-aware trigger, timeout/adverse exit spread and stop-gap proxy; remove the second half-spread subtraction from stop/trailing fill",
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    report = [
        "# Execution Policy Factorial Diagnostic", "", "## Contract", "",
        f"- Fixed identical population: **{len(rows):,} rows**.",
        "- Reference: `label_1m_delay_5m_reanchored_8h`.",
        "- Five-minute delayed entry, one-minute paths, eight-hour decision horizon.",
        f"- Net metrics charge the fixed {100 * args.fee_round_trip:.2f}% round-trip fee once.",
        "- No model, threshold, or geometry parameter was fitted or tuned.", "",
        "## Single Components", "",
        "| Component | Gross EV | Net EV | Gross delta | Win delta | Exit changes | >25bp | >50bp | >100bp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    single_table = overall.loc[["reference", *[f"single__{name}" for name in COMPONENT_FLAGS]]]
    for name, row in single_table.iterrows():
        report.append(
            f"| `{name}` | {100 * row['gross_ev']:+.4f}% | {100 * row['net_ev']:+.4f}% | "
            f"{100 * row['ev_delta']:+.4f} pp | {100 * row['win_rate_delta']:+.2f} pp | "
            f"{int(row['exit_type_changes']):,} | {int(row['return_change_gt_25bps']):,} | "
            f"{int(row['return_change_gt_50bps']):,} | {int(row['return_change_gt_100bps']):,} |"
        )
    report += ["", "## Largest Negative Components", "", ", ".join(f"`{x}`" for x in negative), "", "## Pairwise Interactions", ""]
    if len(interactions):
        report += ["| Pair | Gross interaction | Net interaction |", "|---|---:|---:|"]
        for _, row in interactions.sort_values("gross_interaction_delta").iterrows():
            report.append(f"| `{row['pair']}` | {100 * row['gross_interaction_delta']:+.4f} pp | {100 * row['net_interaction_delta']:+.4f} pp |")
    report += [
        "", "## Interpretation Boundary", "",
        "The July short slice is descriptive because it contains only 39 rows. This report attributes the fixed execution bridge; it does not propose or tune replacement geometry.",
        "Detailed overall/side/month/tail/archetype metrics are in `factorial_metrics.csv`; row transitions, timeout counterfactuals, and spread/barrier audits are separate artifacts.",
    ]
    (args.output_dir / "FACTORIAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    print(single_table[["rows", "gross_ev", "net_ev", "ev_delta", "win_rate", "timeout_rate", "exit_type_changes"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
