#!/usr/bin/env python3
"""Strict causal gradual continuation-grid research on exact one-minute paths.

The base continuation score is calibrated first, using a prior resolved
calendar reserve.  A grid then maps calibrated probability to a continuous
next-interval exit multiplier.  Each control is evaluated independently:
trailing activation, giveback, and stop distance.  No exchange, live-bundle,
or order-submission code is imported.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_h4_overlay_research import replay_exact_1m_gradual_h4_overlay
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as h4_features
from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as h4_hpo
from scripts.run_strict_r3_p8u_v58_e2_50_exact_matched import (
    DEFAULT_PATH_ROOT,
    DEFAULT_POLICY,
    _candidate_table,
    _load_policy,
    _replay,
)


STATE_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_exact_h4_states_20260830_v1"
FEATURE_CONTRACT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2/stable_selected_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_gradual_exit_grid_20260830_v1"
SEED = 1729
MONTHS = (
    pd.Timestamp("2026-05-01", tz="UTC"),
    pd.Timestamp("2026-06-01", tz="UTC"),
    pd.Timestamp("2026-07-01", tz="UTC"),
)
CONTROL_UNIT = {"activation": 0.25, "giveback": 0.10, "stop": 0.10}

# Forked workers inherit these read-only arrays and score schedules.  This
# avoids serialising the complete one-minute path panel once per trial.
_WORKER_CONTEXT: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(contract: Path, held: pd.Timestamp) -> tuple[str, ...]:
    selected = pd.read_parquet(contract)
    rows = selected.loc[
        selected["arm"].eq("C4_normalized_vwap_fs")
        & selected["held_month"].eq(held.strftime("%Y-%m"))
    ].sort_values("position", kind="stable")
    fields = tuple(rows["feature"].astype(str))
    if len(fields) != 45 or len(set(fields)) != len(fields):
        raise AssertionError(f"{held:%Y-%m}: invalid frozen C4 contract")
    return fields


def _fit(train: pd.DataFrame, fields: tuple[str, ...]) -> lgb.LGBMRegressor:
    spec = h4_hpo.SPECS["H4_l1_d4_l15_leaf5_reg20"]
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=int(spec["n_estimators"]),
        learning_rate=float(spec["learning_rate"]),
        max_depth=int(spec["max_depth"]),
        num_leaves=int(spec["num_leaves"]),
        min_child_samples=max(8, int(np.ceil(len(train) * float(spec["min_child_fraction"])))),
        subsample=.80,
        colsample_bytree=.80,
        reg_lambda=float(spec["reg_lambda"]),
        random_state=SEED,
        n_jobs=2,
        verbosity=-1,
    )
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    model.fit(train.loc[:, fields], train["activation50_advantage_bps"].to_numpy(float), sample_weight=weights)
    return model


def _prepare_calibrated_schedule(
    *,
    state_root: Path,
    feature_contract: Path,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, dict[pd.Timestamp, float]], pd.DataFrame]:
    exact = pd.read_parquet(state_root / "exact_parent_h4_state_features_target_free.parquet")
    exact["candidate_id"] = exact["candidate_id"].astype(str)
    exact["entry_decision_ts"] = pd.to_datetime(exact["entry_decision_ts"], utc=True, errors="raise")
    exact["state_decision_ts"] = pd.to_datetime(exact["state_decision_ts"], utc=True, errors="raise")
    labels = h4_features._load_panel(h4_features.TARGET_PANEL, h4_features.VWAP_PANEL)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    labels["entry_decision_ts"] = pd.to_datetime(labels["entry_decision_ts"], utc=True, errors="raise")
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True, errors="raise")

    score_rows: list[pd.DataFrame] = []
    receipts: list[dict[str, object]] = []
    for held in MONTHS:
        end = held + pd.offsets.MonthBegin(1)
        start = held - pd.DateOffset(months=4)
        reserve_start = max(start, held - pd.Timedelta(days=28))
        fields = _fields(feature_contract, held)
        fit = labels.loc[
            labels["entry_decision_ts"].ge(start)
            & labels["entry_decision_ts"].lt(reserve_start)
            & labels["policy_label_available_ts"].lt(reserve_start)
            & pd.to_numeric(labels["MC1_expected_bps"], errors="coerce").ge(30.0)
        ].copy()
        reserve = labels.loc[
            labels["entry_decision_ts"].ge(reserve_start)
            & labels["entry_decision_ts"].lt(held)
            & labels["policy_label_available_ts"].lt(held)
            & pd.to_numeric(labels["MC1_expected_bps"], errors="coerce").ge(30.0)
        ].copy()
        test = exact.loc[
            exact["entry_decision_ts"].ge(held)
            & exact["entry_decision_ts"].lt(end)
        ].copy()
        missing = set(fields).difference(fit.columns) | set(fields).difference(reserve.columns) | set(fields).difference(test.columns)
        if missing or fit.candidate_id.nunique() < 100 or reserve.candidate_id.nunique() < 50 or test.empty:
            raise RuntimeError(f"{held:%Y-%m}: incomplete strict fit/reserve/test fold: {sorted(missing)}")
        model = _fit(fit, fields)
        reserve_raw = model.predict(reserve.loc[:, fields])
        reserve_y = pd.to_numeric(reserve["activation50_advantage_bps"], errors="raise").to_numpy(float)
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(reserve_raw, (reserve_y > 0.0).astype(float))
        raw = model.predict(test.loc[:, fields])
        probability = np.asarray(calibrator.predict(raw), dtype=float)
        score = test.loc[:, ["candidate_id", "entry_decision_ts", "state_decision_ts", "state_minute"]].copy()
        score["raw_h4_advantage_prediction"] = raw
        score["calibrated_h4_probability_positive"] = probability
        score["held_month"] = held.strftime("%Y-%m")
        score_rows.append(score)
        joblib.dump(
            {
                "model": model,
                "calibrator": calibrator,
                "fields": fields,
                "fit_start": start,
                "calibration_reserve_start": reserve_start,
                "held_month": held,
                "target": "activation50_advantage_bps > 0",
            },
            output / f"calibrated_h4_bundle_{held:%Y%m}.joblib",
        )
        reserve_prob = np.asarray(calibrator.predict(reserve_raw), dtype=float)
        receipts.append(
            {
                "held_month": held.strftime("%Y-%m"),
                "fit_states": len(fit),
                "fit_candidates": int(fit.candidate_id.nunique()),
                "reserve_states": len(reserve),
                "reserve_candidates": int(reserve.candidate_id.nunique()),
                "test_states": len(test),
                "test_candidates": int(test.candidate_id.nunique()),
                "reserve_positive_rate": float((reserve_y > 0.0).mean()),
                "reserve_raw_brier": float(np.mean((np.clip(reserve_raw, 0.0, 1.0) - (reserve_y > 0.0)) ** 2)),
                "reserve_isotonic_brier": float(np.mean((reserve_prob - (reserve_y > 0.0)) ** 2)),
            }
        )
    scores = pd.concat(score_rows, ignore_index=True)
    if scores.duplicated(["candidate_id", "state_decision_ts"]).any():
        raise AssertionError("calibrated H4 schedule has duplicate state identities")
    schedule = {
        candidate: {
            pd.Timestamp(ts): float(prob)
            for ts, prob in zip(group["state_decision_ts"], group["calibrated_h4_probability_positive"], strict=True)
        }
        for candidate, group in scores.groupby("candidate_id", sort=False)
    }
    return scores, schedule, pd.DataFrame(receipts)


def _grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for control in ("activation", "giveback", "stop"):
        for threshold in (.20, .40, .60, .80):
            for strength in (1.0, 2.0, 3.0):
                for power in (.50, 1.0, 2.0):
                    rows.append({"control": control, "mode": "shrink", "extension_ratio": 0.0, "threshold": threshold, "strength": strength, "power": power})
                    rows.append({"control": control, "mode": "extend", "extension_ratio": 1.0, "threshold": threshold, "strength": strength, "power": power})
                    for ratio in (.50, 1.00):
                        rows.append({"control": control, "mode": "both", "extension_ratio": ratio, "threshold": threshold, "strength": strength, "power": power})
    return rows


def _multipliers(probability: float, config: dict[str, object]) -> dict[str, float]:
    p = float(np.clip(probability, 0.0, 1.0))
    threshold = float(config["threshold"])
    power = float(config["power"])
    strength = float(config["strength"])
    high = ((p - threshold) / max(1.0 - threshold, 1.0e-12)) ** power if p >= threshold else 0.0
    low = ((threshold - p) / max(threshold, 1.0e-12)) ** power if p < threshold else 0.0
    unit = CONTROL_UNIT[str(config["control"])]
    shrink = min(0.80, unit * strength * high)
    extend = min(0.80, unit * strength * float(config["extension_ratio"]) * low)
    mode = str(config["mode"])
    if mode == "shrink":
        signed = -shrink
    elif mode == "extend":
        signed = extend
    elif mode == "both":
        signed = extend - shrink
    else:
        raise ValueError(f"unknown modulation mode {mode}")
    result = {
        "activation_multiplier": 1.0,
        "giveback_multiplier": 1.0,
        "sl_distance_multiplier": 1.0,
    }
    key = {
        "activation": "activation_multiplier",
        "giveback": "giveback_multiplier",
        "stop": "sl_distance_multiplier",
    }[str(config["control"])]
    result[key] = float(np.clip(1.0 + signed, 0.20, 1.80))
    return result


def _run_candidates(config: dict[str, object], candidate_ids: list[str]) -> pd.DataFrame:
    context = _WORKER_CONTEXT
    route = context["route_by_id"]
    schedule = context["schedule"]
    entry = context["entry"]
    atr = context["atr"]
    high = context["high"]
    low = context["low"]
    close = context["close"]
    params = context["params"]
    median = context["median"]
    records: list[dict[str, object]] = []
    allow_extension = str(config["mode"]) in {"extend", "both"} and str(config["control"]) == "stop"
    for candidate in candidate_ids:
        row = route[candidate]
        path_i = int(row["__path_index__"])
        candidate_schedule = schedule.get(candidate, {})

        def modulator(state: dict[str, float], candidate_schedule=candidate_schedule, config=config) -> dict[str, float] | None:
            probability = candidate_schedule.get(pd.Timestamp(state["state_decision_ts"]))
            return None if probability is None else _multipliers(float(probability), config)

        trace = replay_exact_1m_gradual_h4_overlay(
            entry_price=float(entry[path_i]),
            signal_atr=float(atr[path_i]),
            entry_ts=row["entry_ts"],
            highs=high[path_i],
            lows=low[path_i],
            closes=close[path_i],
            params=params,
            median_atr_fraction=float(median),
            mc1_expected_bps=float(row["bcf_mc1_expected_bps"]),
            state_modulator=modulator,
            allow_stop_extension=allow_extension,
            max_stop_loss_fraction=.05,
            emit_states=False,
        )
        records.append(
            {
                "candidate_id": candidate,
                "timestamp": row["timestamp"],
                "exact_net_bps": float(trace["net_bps"]),
                "exact_gross_bps": float(trace["gross_bps"]),
                "exact_exit_ts": pd.Timestamp(trace["exit_timestamp"]),
                "exact_exit_price": float(trace["exit_price"]),
                "exact_exit_minute": int(trace["exit_minute"]),
                "exact_exit_reason": str(trace["exit_reason"]),
                # Match the shared portfolio replacement contract exactly.
                "exact_entry_price": float(entry[path_i]),
            }
        )
    return pd.DataFrame(records)


def _screen_one(config: dict[str, object], candidate_ids: list[str], parent_bps: dict[str, float]) -> dict[str, object]:
    outcome = _run_candidates(config, candidate_ids)
    outcome["parent_net_bps"] = outcome["candidate_id"].map(parent_bps)
    outcome["delta_bps"] = outcome["exact_net_bps"] - outcome["parent_net_bps"]
    outcome["month"] = pd.to_datetime(outcome["timestamp"], utc=True).dt.strftime("%Y-%m")
    monthly = outcome.groupby("month", sort=True)["delta_bps"].mean()
    return {
        **config,
        "screen_candidates": len(outcome),
        "screen_net_bps_per_trade": float(outcome["exact_net_bps"].mean()),
        "screen_delta_bps_per_trade": float(outcome["delta_bps"].mean()),
        "screen_total_delta_bps": float(outcome["delta_bps"].sum()),
        "screen_worst_month_delta_bps": float(monthly.min()),
        "screen_positive_month_fraction": float((monthly > 0.0).mean()),
        "screen_score": float(outcome["delta_bps"].mean() - .50 * max(0.0, -monthly.min())),
    }


def _parent_frame(route: pd.DataFrame, parent: pd.DataFrame, entry: np.ndarray) -> pd.DataFrame:
    enriched = parent.merge(route.loc[:, ["candidate_id", "__path_index__"]], on="candidate_id", how="inner", validate="one_to_one")
    enriched["entry_price"] = np.asarray([entry[int(index)] for index in enriched["__path_index__"]], dtype=float)
    frame = route.loc[:, ["candidate_id", "timestamp", "entry_ts", "symbol", "bcf_mc1_expected_bps"]].merge(enriched, on="candidate_id", how="inner", validate="one_to_one")
    frame["exact_entry_price"] = frame["entry_price"].to_numpy(float)
    frame["exact_net_bps"] = frame["parent_exact_net_bps"].to_numpy(float)
    frame["exact_gross_bps"] = frame["parent_exact_gross_bps"].to_numpy(float)
    frame["exact_exit_ts"] = pd.to_datetime(frame["parent_exit_ts"], utc=True)
    frame["exact_exit_price"] = frame["exact_entry_price"] * (1.0 + frame["exact_gross_bps"] / 10_000.0)
    frame["exact_exit_minute"] = frame["parent_exit_minute"].to_numpy(int)
    frame["exact_exit_reason"] = frame["parent_exit_reason"].astype(str)
    return frame.loc[:, ["candidate_id", "timestamp", "entry_ts", "symbol", "bcf_mc1_expected_bps", "exact_entry_price", "exact_net_bps", "exact_gross_bps", "exact_exit_ts", "exact_exit_price", "exact_exit_minute", "exact_exit_reason"]]


def _portfolio(frame: pd.DataFrame, *, output: Path, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    table = _candidate_table(frame, frame["bcf_mc1_expected_bps"])
    decisions, accepted, equity, metrics = _replay(table)
    table.to_parquet(output / f"{arm}_portfolio_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(output / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(output / f"{arm}_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(output / f"{arm}_equity.parquet", index=False, compression="zstd")
    return table, decisions, accepted, metrics


def _accepted_metrics(table: pd.DataFrame, accepted: pd.DataFrame, *, start: pd.Timestamp | None = None) -> dict[str, float]:
    lookup = table.reset_index(names="candidate_index")
    # The shared auction materialiser expresses realised PnL as a decimal
    # ``net_return``.  Keep this research helper in bps, regardless of
    # whether it receives a pre-materialised exact column or the auction
    # table's canonical return column.
    if "exact_net_bps" not in lookup.columns:
        if "net_return" not in lookup.columns:
            raise KeyError("portfolio table has neither exact_net_bps nor net_return")
        lookup["exact_net_bps"] = pd.to_numeric(lookup["net_return"], errors="raise") * 10_000.0
    # ``accepted`` already owns the execution timestamp.  Do not join the
    # candidate-table timestamp a second time (which would create suffixes
    # and silently break the held-period filter).
    use = accepted.merge(lookup[["candidate_index", "exact_net_bps"]], on="candidate_index", how="left", validate="one_to_one")
    if start is not None:
        use = use.loc[pd.to_datetime(use["timestamp"], utc=True).ge(start)].copy()
    values = pd.to_numeric(use["exact_net_bps"], errors="raise")
    return {
        "accepted": float(len(use)),
        "net_bps_per_trade": float(values.mean()) if len(values) else float("nan"),
        "total_net_bps": float(values.sum()),
    }


def _replace_month(parent: pd.DataFrame, outcome: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    result = parent.copy()
    mask = pd.to_datetime(result["timestamp"], utc=True).ge(month)
    replacement = outcome.set_index("candidate_id")
    columns = ["exact_entry_price", "exact_net_bps", "exact_gross_bps", "exact_exit_ts", "exact_exit_price", "exact_exit_minute", "exact_exit_reason"]
    mapped = result.loc[mask, "candidate_id"].map(replacement[columns].to_dict("index"))
    if mapped.isna().any():
        raise AssertionError("variant outcome misses a routed candidate")
    for column in columns:
        result.loc[mask, column] = mapped.map(lambda value, column=column: value[column]).to_numpy()
    return result


def _full_outcome(config: dict[str, object]) -> pd.DataFrame:
    context = _WORKER_CONTEXT
    return _run_candidates(config, list(context["route_by_id"]))


def _sample_ids(route: pd.DataFrame, *, months: tuple[pd.Timestamp, ...], per_month: int) -> list[str]:
    rng = np.random.default_rng(SEED)
    ids: list[str] = []
    stamp = pd.to_datetime(route["timestamp"], utc=True)
    for month in months:
        end = month + pd.offsets.MonthBegin(1)
        values = route.loc[stamp.ge(month) & stamp.lt(end), "candidate_id"].astype(str).drop_duplicates().to_numpy()
        if len(values) <= per_month:
            ids.extend(values.tolist())
        else:
            ids.extend(rng.choice(values, size=per_month, replace=False).tolist())
    return sorted(set(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument("--feature-contract", type=Path, default=FEATURE_CONTRACT)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--screen-sample-per-month", type=int, default=100)
    parser.add_argument("--top-per-control", type=int, default=6)
    parser.add_argument("--workers", type=int, default=min(4, max(1, (mp.cpu_count() or 2) - 1)))
    parser.add_argument("--grid-start", type=int, default=0, help="inclusive coarse-grid offset for resumable screens")
    parser.add_argument("--grid-stop", type=int, default=None, help="exclusive coarse-grid offset for resumable screens")
    parser.add_argument("--screen-only", action="store_true", help="write an immutable coarse screen block then stop")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    if args.screen_sample_per_month < 25 or args.top_per_control < 1:
        raise ValueError("screen sample must be >=25 and at least one finalist is required")
    output.mkdir(parents=True, exist_ok=False)

    route = pd.read_parquet(args.state_root.resolve() / "target_free_route.parquet")
    route["candidate_id"] = route["candidate_id"].astype(str)
    route["timestamp"] = pd.to_datetime(route["timestamp"], utc=True, errors="raise")
    route["entry_ts"] = pd.to_datetime(route["entry_ts"], utc=True, errors="raise")
    parent = pd.read_parquet(args.state_root.resolve() / "exact_parent_outcomes.parquet")
    if parent.candidate_id.nunique() != len(route):
        raise AssertionError("parent outcomes do not cover the exact route")

    rows = pd.read_parquet(args.path_root.resolve() / "valid_exact_paths_rows.parquet")
    path_index = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    route["__path_index__"] = route["candidate_id"].map(path_index)
    if route["__path_index__"].isna().any():
        raise AssertionError("route/path identity mismatch")
    archive = np.load(args.path_root.resolve() / "exact_paths.npz", allow_pickle=False)
    entry = np.asarray(archive["entry"], dtype=float)
    atr = np.asarray(archive["atr"], dtype=float)
    high = np.asarray(archive["high"], dtype=np.float32)
    low = np.asarray(archive["low"], dtype=np.float32)
    close = np.asarray(archive["close"], dtype=np.float32)
    params, median, _ = _load_policy(args.policy.resolve())
    parent_frame = _parent_frame(route, parent, entry)

    score_panel, schedule, calibration_receipts = _prepare_calibrated_schedule(
        state_root=args.state_root.resolve(),
        feature_contract=args.feature_contract.resolve(),
        output=output,
    )
    score_panel.to_parquet(output / "calibrated_h4_scores_target_free.parquet", index=False, compression="zstd")
    calibration_receipts.to_parquet(output / "calibration_receipts.parquet", index=False)

    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "route_by_id": {str(row.candidate_id): row.to_dict() for _, row in route.iterrows()},
        "schedule": schedule,
        "entry": entry,
        "atr": atr,
        "high": high,
        "low": low,
        "close": close,
        "params": params,
        "median": median,
    }
    sample_ids = _sample_ids(route, months=MONTHS[:2], per_month=int(args.screen_sample_per_month))
    parent_bps = parent_frame.set_index("candidate_id")["exact_net_bps"].to_dict()
    full_grid = _grid()
    grid_start = int(args.grid_start)
    grid_stop = len(full_grid) if args.grid_stop is None else int(args.grid_stop)
    if not 0 <= grid_start < grid_stop <= len(full_grid):
        raise ValueError(f"invalid grid slice [{grid_start}, {grid_stop}) for {len(full_grid)} configurations")
    grid = full_grid[grid_start:grid_stop]
    # Serial mode is deliberately first-class: it is the deterministic
    # smoke-test path and avoids a forked child hiding a Python exception.
    # The regular research run keeps the read-only forked-array fast path.
    if int(args.workers) == 1:
        screen = [_screen_one(config, sample_ids, parent_bps) for config in grid]
    else:
        screen = []
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=context) as pool:
            futures = [pool.submit(_screen_one, config, sample_ids, parent_bps) for config in grid]
            for future in as_completed(futures):
                screen.append(future.result())
    screen_frame = pd.DataFrame(screen).sort_values(["control", "screen_score", "screen_delta_bps_per_trade"], ascending=[True, False, False], kind="stable")
    screen_frame.to_parquet(output / "coarse_grid_screen.parquet", index=False)
    if args.screen_only:
        (output / "run_manifest.json").write_text(json.dumps({
            "schema": "strict-r3-p8u-v58-gradual-exit-grid-screen-v1",
            "scope": "offline exact one-minute research only; no live mutation, exchange IO, or order submission",
            "grid_slice": [grid_start, grid_stop],
            "grid_total": len(full_grid),
            "screen_sample_per_month": int(args.screen_sample_per_month),
            "workers": int(args.workers),
            "calibration": "prior-resolved final-28-calendar-day isotonic probability calibration",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        return

    finalists: list[dict[str, object]] = []
    for control, group in screen_frame.groupby("control", sort=True):
        finalists.extend(group.head(int(args.top_per_control)).drop(columns=[column for column in group.columns if column.startswith("screen_")]).to_dict("records"))
    finalist_frame = pd.DataFrame(finalists)
    finalist_frame.to_parquet(output / "selection_finalists.parquet", index=False)

    july = pd.Timestamp("2026-07-01", tz="UTC")
    parent_table, _, parent_accepted, _ = _portfolio(parent_frame, output=output, arm="C0_parent")
    parent_july = _accepted_metrics(parent_table, parent_accepted, start=july)
    holdout_rows: list[dict[str, object]] = []
    full_outcomes: dict[str, pd.DataFrame] = {}
    for number, config in enumerate(finalists):
        outcome = _full_outcome(config)
        frame = _replace_month(parent_frame, outcome, july)
        arm = f"finalist_{number:02d}_{config['control']}_{config['mode']}"
        table, _, accepted, _ = _portfolio(frame, output=output, arm=arm)
        observed = _accepted_metrics(table, accepted, start=july)
        holdout_rows.append({**config, "arm": arm, **{f"july_{key}": value for key, value in observed.items()}, "july_delta_net_bps_per_trade": observed["net_bps_per_trade"] - parent_july["net_bps_per_trade"], "july_delta_total_net_bps": observed["total_net_bps"] - parent_july["total_net_bps"]})
        full_outcomes[arm] = outcome
    holdout = pd.DataFrame(holdout_rows)
    holdout["holdout_score"] = holdout["july_delta_total_net_bps"] + 50.0 * holdout["july_delta_net_bps_per_trade"]
    holdout.to_parquet(output / "july_holdout_finalists.parquet", index=False)

    winners: list[dict[str, object]] = []
    for control, group in holdout.groupby("control", sort=True):
        winner = group.sort_values(["holdout_score", "july_delta_net_bps_per_trade"], ascending=False, kind="stable").iloc[0].to_dict()
        winners.append(winner)
    winner_frame = pd.DataFrame(winners)
    winner_frame.to_parquet(output / "per_control_winners.parquet", index=False)

    summaries: list[dict[str, object]] = []
    _, _, c0_accepted, c0_metrics = _portfolio(parent_frame, output=output, arm="C0_parent_full")
    summaries.append({"arm": "C0_parent_full", "control": "parent", **c0_metrics})
    for winner in winners:
        config = {key: winner[key] for key in ("control", "mode", "extension_ratio", "threshold", "strength", "power")}
        outcome = full_outcomes[str(winner["arm"])]
        frame = _replace_month(parent_frame, outcome, pd.Timestamp("2026-05-01", tz="UTC"))
        arm = f"winner_{config['control']}"
        _, _, accepted, metrics = _portfolio(frame, output=output, arm=arm)
        summaries.append({"arm": arm, **config, **metrics})
        accepted.to_parquet(output / f"{arm}_accepted_full.parquet", index=False, compression="zstd")
    summary = pd.DataFrame(summaries)
    c0 = summary.loc[summary.arm.eq("C0_parent_full")].iloc[0]
    for metric in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "worst_week", "sortino", "compounded_return"):
        summary[f"delta_vs_parent_{metric}"] = summary[metric] - c0[metric]
    summary.to_parquet(output / "full_exact_portfolio_summary.parquet", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "schema": "strict-r3-p8u-v58-gradual-exit-grid-v1",
        "scope": "offline exact one-minute research only; no live mutation, exchange IO, or order submission",
        "route": "Router50, dual BCF/current MC1 >=50, BCF priority, +5 minute entry, shared full chronological portfolio constraints",
        "parent_policy": str(args.policy.resolve()),
        "parent_policy_sha256": _sha256(args.policy.resolve()),
        "state_source": str(args.state_root.resolve()),
        "calibration": "prior-resolved final-28-calendar-day isotonic probability calibration before any control modulation",
        "model_target": "activation50_advantage_bps > 0",
        "grid": {
            "thresholds": [.20, .40, .60, .80],
            "strength": [1.0, 2.0, 3.0],
            "power": [.50, 1.0, 2.0],
            "modes": ["shrink", "extend", "both: extension ratios .5 and 1.0"],
            "control_unit_authority": CONTROL_UNIT,
            "stop_max_loss_fraction": .05,
        },
        "selection": "coarse May-June candidate-balanced screen; top finalists per control; July exact constrained portfolio holdout; no cross-control composition",
        "h4_label_caveat": "the H4 target source is historical prior-resolved activation50 advantage; exact test paths are full +5-minute one-minute replays",
        "workers": int(args.workers),
        "seed": SEED,
    }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
