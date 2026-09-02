#!/usr/bin/env python3
"""State-specialist direct activation-50 continuation ablation.

The action under test is intentionally narrow: bring the rich parent policy's
trailing activation forward by 50%, beginning on the *next* 15-minute
interval.  That action has material authority only before either the trailing
or smooth-protection state is already armed.  This research-only challenger
therefore trains a specialist on the pre-activation state and deterministically
keeps the parent policy in already-protected states.

Feature subsets are the prior-only stable selections produced by the companion
v2 direct-advantage study.  Each held-month specialist trains only on the two
complete preceding months and only where the corresponding action outcome was
resolved before the held boundary.  It never modifies a live or canonical
artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_continuation_state import replay_open_long_policy_with_continuation_modulator
from extreme_price_movements.portfolio_policy_replay import compute_replay_metrics, replay_candidates
from scripts import replay_strict_r3_p8u_15m_continuation_portfolio as port
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1
from scripts import run_strict_r3_p8u_15m_continuation_v2_advantage_ablation as full
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params as portfolio_params


DIRECT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_v2_advantage_ablation_20260830_v1"
TARGET_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_activation50_advantage_20260830_v1/activation50_advantage_states.parquet"
ENTRY_PREDICTIONS = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_policycap_retrain_20260830_v3_control/walkforward_predictions.parquet"
STATE_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_predictive_observed25h_20260830_v3/target_free_continuation_state_parts"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_v2_preactivation_ablation_20260830_v1"

ARM_SPECS = (
    ("A2_global_mean0_reference", "mean", 0.0, False),
    ("A5_preactivation_mean0", "mean", 0.0, True),
    ("A6_preactivation_q20_0", "q20", 0.0, True),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pre_activation(frame: pd.DataFrame) -> pd.Series:
    """Action is relevant only before trailing and smooth protection arm."""
    return (
        pd.to_numeric(frame["is_trailing_active"], errors="coerce").fillna(0.0).eq(0.0)
        & pd.to_numeric(frame["current_protection_state"], errors="coerce").fillna(0.0).eq(0.0)
    )


def _features_by_held(path: Path) -> dict[str, tuple[str, ...]]:
    selected = pd.read_parquet(path)
    required = {"held_month", "feature", "position"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"stable feature selection lacks {sorted(missing)}")
    return {
        str(month): tuple(group.sort_values("position", kind="stable").feature.astype(str))
        for month, group in selected.groupby("held_month", sort=True)
    }


def _simulate(
    group: pd.DataFrame,
    bars: c1.CompactBars,
    model,
    features: tuple[str, ...],
    threshold: float,
    params,
    median: float,
    specialist: bool,
) -> dict[str, object] | None:
    first = group.iloc[0]
    path = c1._bar_path(bars, pd.Timestamp(first["entry_decision_ts"]))
    if path is None:
        return None
    high, low, close = path
    static = {
        int(row.state_bar_15m): (
            row.loc[list(features)].to_numpy(float),
            bool(float(row.is_trailing_active) == 0.0 and float(row.current_protection_state) == 0.0),
        )
        for _, row in group.iterrows()
    }
    buffer = np.empty(len(features), dtype=float)
    calls = actions = ineligible_state_calls = 0

    def callback(dynamic: dict[str, float]) -> float | None:
        nonlocal calls, actions, ineligible_state_calls
        bar = int(dynamic.pop("state_bar_15m"))
        item = static.get(bar)
        if item is None:
            return None
        values, can_act = item
        if specialist and not can_act:
            ineligible_state_calls += 1
            return 2.0
        calls += 1
        buffer[:] = values
        prediction = float(model.booster_.predict(buffer.reshape(1, -1))[0])
        action = prediction >= threshold
        actions += int(action)
        return 0.0 if action else 2.0

    trace = replay_open_long_policy_with_continuation_modulator(
        entry=float(first["entry_price"]), signal_atr=float(first["signal_atr"]), highs=high, lows=low, closes=close,
        params=params, median_atr_fraction=median, prediction_for_completed_bar=callback,
        sl_tighten=0.0, giveback_tighten=0.0, activation_earlier=0.50,
    )
    return {
        "candidate_id": str(first.candidate_id), "__symbol__": str(first.__symbol__),
        "entry_decision_ts": pd.Timestamp(first.entry_decision_ts),
        "baseline_net_bps": float(first.policy_net_bps), "baseline_gross_bps": float(first.policy_gross_bps),
        "baseline_exit_bar": int(first.policy_exit_bar_15m), "baseline_exit_reason": str(first.policy_exit_reason),
        "c1_gross_bps": float(trace.terminal_gross_bps), "c1_net_bps": float(trace.terminal_gross_bps - 100.0),
        "c1_exit_bar": int(trace.terminal_exit_bar), "c1_exit_reason": str(trace.terminal_reason),
        "model_calls": calls, "action_calls": actions, "ineligible_state_calls": ineligible_state_calls,
    }


def _replay_portfolio(rows: pd.DataFrame, arm: str, output: Path) -> dict[str, object]:
    prices = port._entry_prices(STATE_ROOT)
    priorities = port._bcf_priority()
    tagged = rows.copy()
    tagged["arm"] = arm
    tagged["mc1_threshold_bps"] = 30.0
    candidates = port._candidate_table(tagged, prices, priorities)
    params = portfolio_params()
    decisions, equity, _ = replay_candidates(candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp")
    decisions = port._attach_ids(decisions, candidates)
    accepted = decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
    for name, frame in (("candidates", candidates), ("decisions", decisions), ("accepted", accepted), ("equity", equity)):
        frame.to_parquet(output / f"{arm}_{name}.parquet", index=False, compression="zstd")
    port._period_metrics(accepted, "month").assign(arm=arm).to_parquet(output / f"{arm}_monthly.parquet", index=False)
    returns = pd.to_numeric(candidates.iloc[pd.to_numeric(accepted.candidate_index, errors="raise").astype(int).to_numpy()].net_return, errors="raise") * 10_000.0
    return {
        "arm": arm, "routed_candidates": len(candidates), "portfolio_accepted": len(accepted),
        "policy_net_bps_per_trade": float(returns.mean()) if len(returns) else np.nan,
        "total_policy_net_bps": float(returns.sum()),
        **compute_replay_metrics(candidates, decisions, equity, params=params),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-output", type=Path, default=DIRECT_OUTPUT)
    parser.add_argument("--target-panel", type=Path, default=TARGET_PANEL)
    parser.add_argument("--entry-predictions", type=Path, default=ENTRY_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Apr--Aug 2026")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    direct = args.direct_output.resolve()
    feature_sets = _features_by_held(direct / "stable_selected_features.parquet")
    panel_path = args.target_panel.resolve()
    panel = pd.read_parquet(panel_path)
    panel["candidate_id"] = panel.candidate_id.astype(str)
    for name in ("entry_decision_ts", "state_decision_ts", "policy_label_available_ts"):
        panel[name] = pd.to_datetime(panel[name], utc=True, errors="raise")
    selected_ids = full._selection_ids(args.entry_predictions.resolve())
    held_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-04-01", "2026-08-01", freq="MS", tz="UTC"))
    params, median, policy = c1.base._load_policy()
    outcomes: dict[str, list[pd.DataFrame]] = defaultdict(list)
    audit_rows: list[dict[str, object]] = []
    bars_cache: dict[str, c1.CompactBars | None] = {}
    for held in held_months:
        key, held_end, train_start = held.strftime("%Y-%m"), held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=2)
        features = feature_sets.get(key)
        if not features:
            continue
        train_all = panel.loc[
            pd.to_numeric(panel.MC1_expected_bps, errors="coerce").ge(30.0)
            & panel.entry_decision_ts.ge(train_start) & panel.entry_decision_ts.lt(held)
            & panel.policy_label_available_ts.lt(held)
        ].copy()
        train_preactivation = train_all.loc[_pre_activation(train_all)].copy()
        test = panel.loc[
            panel.entry_decision_ts.ge(held) & panel.entry_decision_ts.lt(held_end)
            & panel.candidate_id.isin(selected_ids)
        ].copy()
        required = {(held - pd.DateOffset(months=1)).strftime("%Y-%m"), (held - pd.DateOffset(months=2)).strftime("%Y-%m")}
        observed = set(train_preactivation.entry_decision_ts.dt.strftime("%Y-%m"))
        if not required.issubset(observed) or train_preactivation.candidate_id.nunique() < 100 or test.empty:
            continue
        if set(features).difference(train_all.columns):
            raise AssertionError(f"{key}: selected feature absent from pre-activation train frame")
        models = {
            # Preserve a true global training control.  The specialists use
            # only states where bringing activation forward retains authority.
            "global_mean": full._fit(train_all, features, "mean"),
            "preactivation_mean": full._fit(train_preactivation, features, "mean"),
            "preactivation_q20": full._fit(train_preactivation, features, "q20"),
        }
        outcomes["C0_parent"].append(full._parent_rows(test).assign(held_month=key))
        ordered = test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable")
        for arm, kind, threshold, specialist in ARM_SPECS:
            model_key = "global_mean" if not specialist else f"preactivation_{kind}"
            rows: list[dict[str, object]] = []
            for symbol, symbol_rows in ordered.groupby("__symbol__", sort=True):
                if str(symbol) not in bars_cache:
                    bars_cache[str(symbol)] = c1._load_symbol_bars(str(symbol))
                bars = bars_cache[str(symbol)]
                if bars is None:
                    continue
                for _, group in symbol_rows.groupby("candidate_id", sort=True):
                    result = _simulate(group, bars, models[model_key], features, threshold, params, median, specialist)
                    if result is not None:
                        rows.append(result)
            frame = pd.DataFrame(rows)
            if frame.empty:
                raise RuntimeError(f"{key}/{arm}: no policy outcomes")
            frame["held_month"] = key
            outcomes[arm].append(frame)
            audit_rows.append({
                "held_month": key, "arm": arm, "train_rows_all_states": len(train_all),
                "train_candidates_all_states": train_all.candidate_id.nunique(),
                "train_rows_pre_activation": len(train_preactivation),
                "train_candidates_pre_activation": train_preactivation.candidate_id.nunique(),
                "test_rows": len(frame), "test_candidates": frame.candidate_id.nunique(),
                "selected_feature_count": len(features), "model_calls": int(frame.model_calls.sum()),
                "action_calls": int(frame.action_calls.sum()), "ineligible_state_calls": int(frame.ineligible_state_calls.sum()),
            })
    expected = {"C0_parent", *(arm for arm, *_ in ARM_SPECS)}
    if set(outcomes) != expected:
        raise RuntimeError(f"missing strict-OOS outcome arms: {sorted(expected.difference(outcomes))}")
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for arm, frames in outcomes.items():
        frame = pd.concat(frames, ignore_index=True)
        if frame.candidate_id.duplicated().any():
            raise AssertionError(f"{arm}: duplicate candidate OOS outcome")
        frame.to_parquet(output / f"{arm}_entry_outcomes.parquet", index=False, compression="zstd")
        summaries.append(_replay_portfolio(frame, arm, output))
    summary = pd.DataFrame(summaries)
    control = summary.loc[summary.arm.eq("C0_parent")].iloc[0]
    for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "compounded_return", "sortino", "max_drawdown", "worst_week"):
        summary[f"delta_vs_C0_{metric}"] = summary[metric] - control[metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    pd.DataFrame(audit_rows).to_parquet(output / "state_specialist_audit.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-continuation-v2-preactivation-specialist-v1",
        "scope": "offline strict-OOS research only; no live/canonical mutation",
        "target": "direct activation-50 net advantage over parent rich policy; action begins only on next 15m interval",
        "fold": "two full preceding calendar months; all parent action labels resolved before held boundary",
        "training_populations": {
            "A2_global_mean0_reference": "all valid resolved prior states",
            "A5_preactivation_mean0": "only prior pre-activation states",
            "A6_preactivation_q20_0": "only prior pre-activation states",
        },
        "state_routing": {
            "pre_activation": "is_trailing_active == 0 and current_protection_state == 0; specialist may act",
            "trailing_or_smooth_armed": "deterministic parent/no-action because activation-50 is already inert or has minimal remaining authority",
        },
        "feature_selection": "hash-bound prior-only stable selections from direct v2 study", "direct_output": str(direct),
        "stable_selected_features_sha256": _sha256(direct / "stable_selected_features.parquet"),
        "target_panel": str(panel_path), "target_panel_sha256": _sha256(panel_path),
        "entry_predictions": str(args.entry_predictions.resolve()), "entry_predictions_sha256": _sha256(args.entry_predictions.resolve()),
        "cost": "100 bps embedded once in baseline and each action-policy outcome",
        "policy": policy["params"], "portfolio": asdict(portfolio_params()),
        "arms": [arm for arm, *_ in ARM_SPECS],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
