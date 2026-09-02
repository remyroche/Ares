#!/usr/bin/env python3
"""Strict-OOS feature-contract comparison for the stable-mean continuation arm.

This runner deliberately holds the direct action fixed: the target is the
activation-50 advantage over the rich parent and the action threshold is zero.
It compares feature contracts only.  April--July 2026 is the model-selection
period; August is written separately as a never-used-for-selection holdout.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS, VWAP_15M_FEATURE_KEYS
from extreme_price_movements.p8u_continuation_state import CONTINUATION_STATE_FEATURE_KEYS
from extreme_price_movements.p8u_continuation_v2_features import EXTENDED_STATE_FEATURE_KEYS, MANDATORY_STATE_FEATURE_KEYS
from scripts import run_strict_r3_p8u_15m_continuation_v2_advantage_ablation as stable
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1


TARGET_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_activation50_advantage_20260830_v1/activation50_advantage_states.parquet"
VWAP_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_v2_vwap_features_20260830_v1/continuation_v2_state_features.parquet"
ENTRY_PREDICTIONS = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_policycap_retrain_20260830_v3_control/walkforward_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_panel(target_path: Path, vwap_path: Path) -> pd.DataFrame:
    target = pd.read_parquet(target_path)
    vwap = pd.read_parquet(vwap_path)
    keys = ["candidate_id", "state_decision_ts", "state_bar_15m"]
    for frame in (target, vwap):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["state_decision_ts"] = pd.to_datetime(frame["state_decision_ts"], utc=True, errors="raise")
    if target.duplicated(keys).any() or vwap.duplicated(keys).any():
        raise AssertionError("continuation state identity is not unique")
    overlay = vwap.loc[:, [*keys, *VWAP_15M_FEATURE_KEYS]].copy()
    joined = target.merge(overlay, on=keys, how="inner", validate="one_to_one")
    if len(joined) != len(target):
        raise AssertionError("VWAP panel does not cover the exact activation-target state identity")
    for name in ("entry_decision_ts", "policy_label_available_ts"):
        joined[name] = pd.to_datetime(joined[name], utc=True, errors="raise")
    return joined


def _available(panel: pd.DataFrame, features: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in features if name in panel and panel[name].notna().mean() > 0.50)


def _candidate_feature_sets(panel: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    old = tuple(name for name in (*FIFTEEN_MINUTE_FEATURE_KEYS, *CONTINUATION_STATE_FEATURE_KEYS) if name in panel)
    mandatory = tuple(name for name in (*MANDATORY_STATE_FEATURE_KEYS, "MC1_expected_bps") if name in panel)
    if len(mandatory) != len((*MANDATORY_STATE_FEATURE_KEYS, "MC1_expected_bps")):
        raise AssertionError("normalised mandatory continuation state is incomplete")
    normal_optional = tuple(
        name for name in _available(panel, EXTENDED_STATE_FEATURE_KEYS)
        if name not in mandatory and name not in VWAP_15M_FEATURE_KEYS
    )
    vwap_optional = tuple((*normal_optional, *[name for name in VWAP_15M_FEATURE_KEYS if name in panel and panel[name].notna().mean() > 0.50]))
    normal_all = tuple(dict.fromkeys((*mandatory, *normal_optional)))
    vwap_all = tuple(dict.fromkeys((*mandatory, *vwap_optional)))
    if len(old) < 80 or len(normal_all) < 60 or len(vwap_all) != len(normal_all) + len(VWAP_15M_FEATURE_KEYS):
        raise AssertionError("unexpected feature-contract availability")
    return {"C0_old": old, "C1_normalized": normal_all, "C3_normalized_vwap": vwap_all}


def _simulate(group: pd.DataFrame, bars: c1.CompactBars, model, features: tuple[str, ...], params, median: float) -> dict[str, object] | None:
    first = group.iloc[0]
    path = c1._bar_path(bars, pd.Timestamp(first["entry_decision_ts"]))
    if path is None:
        return None
    high, low, close = path
    static = {int(row.state_bar_15m): row.loc[list(features)].to_numpy(float) for _, row in group.iterrows()}
    buffer = np.empty(len(features), dtype=float)
    calls = 0
    actions = 0

    def callback(dynamic: dict[str, float]) -> float | None:
        nonlocal calls, actions
        row = static.get(int(dynamic.pop("state_bar_15m")))
        if row is None:
            return None
        calls += 1
        buffer[:] = row
        prediction = float(model.booster_.predict(buffer.reshape(1, -1))[0])
        actions += int(prediction >= 0.0)
        return 0.0 if prediction >= 0.0 else 2.0

    trace = stable.replay_open_long_policy_with_continuation_modulator(
        entry=float(first.entry_price), signal_atr=float(first.signal_atr), highs=high, lows=low, closes=close,
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
        "model_calls": calls, "action_calls": actions,
    }


def _run_arm(panel: pd.DataFrame, arm: str, *, features: tuple[str, ...], select: bool, output: Path, subspace_runs: int, held_months: tuple[pd.Timestamp, ...], train_months: int) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    selected_ids = stable._selection_ids(ENTRY_PREDICTIONS)
    params, median, _ = c1.base._load_policy()
    bars_cache: dict[str, c1.CompactBars | None] = {}
    outcomes: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    trace_rows: list[pd.DataFrame] = []
    ranking_rows: list[pd.DataFrame] = []
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=train_months)
        train = panel.loc[
            pd.to_numeric(panel.MC1_expected_bps, errors="coerce").ge(30.0)
            & panel.entry_decision_ts.ge(start) & panel.entry_decision_ts.lt(held)
            & panel.policy_label_available_ts.lt(held)
        ].copy()
        test = panel.loc[
            panel.entry_decision_ts.ge(held) & panel.entry_decision_ts.lt(end)
            & panel.candidate_id.isin(selected_ids)
        ].copy()
        if train.candidate_id.nunique() < 100 or test.empty:
            raise RuntimeError(f"{arm} has incomplete strict-OOS fold {held:%Y-%m}")
        if select:
            mandatory = tuple(name for name in (*MANDATORY_STATE_FEATURE_KEYS, "MC1_expected_bps") if name in train)
            optional = tuple(name for name in features if name not in mandatory)
            chosen, trace, ranking = stable._select_features(train, optional, subspace_runs)
            # The selector adds its own fixed mandatory core.  It must equal
            # this arm's core rather than accidentally retain an old schema.
            if tuple(chosen[:len(mandatory)]) != mandatory:
                raise AssertionError("feature-selection mandatory core drifted")
            features_fold = chosen
            selected_rows.append(pd.DataFrame({"arm": arm, "held_month": held.strftime("%Y-%m"), "feature": chosen, "position": np.arange(len(chosen))}))
            trace["arm"] = arm; trace["held_month"] = held.strftime("%Y-%m"); trace_rows.append(trace)
            ranking["arm"] = arm; ranking["held_month"] = held.strftime("%Y-%m"); ranking_rows.append(ranking)
        else:
            features_fold = features
        model = stable._fit(train, features_fold, "mean")
        rows: list[dict[str, object]] = []
        for symbol, group in test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable").groupby("__symbol__", sort=True):
            token = str(symbol)
            if token not in bars_cache:
                bars_cache[token] = c1._load_symbol_bars(token)
            bars = bars_cache[token]
            if bars is None:
                continue
            for _, candidate in group.groupby("candidate_id", sort=True):
                replay = _simulate(candidate, bars, model, features_fold, params, median)
                if replay is not None:
                    rows.append(replay)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError(f"{arm} has no executable held outcomes for {held:%Y-%m}")
        frame["held_month"] = held.strftime("%Y-%m")
        outcomes.append(frame)
    detail = pd.concat(outcomes, ignore_index=True)
    if detail.candidate_id.duplicated().any():
        raise AssertionError(f"{arm} duplicated strict-OOS candidate outcomes")
    detail.to_parquet(output / f"{arm}_entry_outcomes.parquet", index=False, compression="zstd")
    return detail, selected_rows, trace_rows, ranking_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-panel", type=Path, default=TARGET_PANEL)
    parser.add_argument("--vwap-panel", type=Path, default=VWAP_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subspace-runs", type=int, default=200)
    parser.add_argument("--train-months", type=int, default=4, help="maximum trailing resolved calendar months for each strict-OOS fit")
    parser.add_argument("--held-month", action="append", help="optional repeatable YYYY-MM; default Apr--Aug 2026")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    panel = _load_panel(args.target_panel.resolve(), args.vwap_panel.resolve())
    contracts = _candidate_feature_sets(panel)
    held_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-04-01", "2026-08-01", freq="MS", tz="UTC"))
    output.mkdir(parents=True, exist_ok=False)
    arms = {
        "C0_old_all": (contracts["C0_old"], False),
        "C1_normalized_all": (contracts["C1_normalized"], False),
        "C2_normalized_fs": (contracts["C1_normalized"], True),
        "C3_normalized_vwap_all": (contracts["C3_normalized_vwap"], False),
        "C4_normalized_vwap_fs": (contracts["C3_normalized_vwap"], True),
    }
    summaries: list[dict[str, object]] = []
    all_selection: list[pd.DataFrame] = []
    all_trace: list[pd.DataFrame] = []
    all_ranking: list[pd.DataFrame] = []
    for arm, (features, select) in arms.items():
        detail, selected, traces, rankings = _run_arm(panel, arm, features=features, select=select, output=output, subspace_runs=args.subspace_runs, held_months=held_months, train_months=args.train_months)
        for scope, scoped in (
            ("selection_apr_jul", detail.loc[pd.to_datetime(detail["entry_decision_ts"], utc=True).lt(SELECTION_END)].copy()),
            ("august_holdout", detail.loc[pd.to_datetime(detail["entry_decision_ts"], utc=True).ge(SELECTION_END)].copy()),
            ("all_oos", detail),
        ):
            if scoped.empty:
                continue
            replay = stable._replay_portfolio(scoped, f"{arm}__{scope}", output)
            replay["model_arm"] = arm
            replay["evaluation_scope"] = scope
            summaries.append(replay)
        all_selection.extend(selected); all_trace.extend(traces); all_ranking.extend(rankings)
    summary = pd.DataFrame(summaries)
    summary["total_ev_per_abs_drawdown"] = summary["total_policy_net_bps"] / summary["max_drawdown"].abs().replace(0.0, np.nan)
    selection = summary.loc[summary["evaluation_scope"].eq("selection_apr_jul")].copy()
    # August contributes no model choice.  Summaries are nevertheless written
    # both for Apr--Jul (selection) and Aug (untouched final holdout).
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    selection.sort_values(["total_ev_per_abs_drawdown", "policy_net_bps_per_trade", "worst_week"], ascending=[False, False, False], kind="stable").to_parquet(output / "selection_ranking_apr_jul.parquet", index=False)
    monthly = pd.concat([pd.read_parquet(output / f"{arm}__all_oos_monthly.parquet") for arm in arms], ignore_index=True)
    monthly["month"] = pd.to_datetime(monthly["period"], utc=True, errors="coerce").dt.strftime("%Y-%m") if "period" in monthly else pd.NaT
    monthly.to_parquet(output / "monthly_metrics.parquet", index=False)
    if all_selection:
        pd.concat(all_selection, ignore_index=True).to_parquet(output / "stable_selected_features.parquet", index=False)
        pd.concat(all_trace, ignore_index=True).to_parquet(output / "selection_trace.parquet", index=False)
        pd.concat(all_ranking, ignore_index=True).to_parquet(output / "subspace_stability.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-continuation-feature-contract-ablation-v1",
        "scope": "offline strict-OOS research only; no live/canonical mutation",
        "direct_action": "stable LightGBM L1 mean target of activation-50 rich-policy advantage; action applies next 15m bar iff prediction >= 0",
        "fold": f"up to {args.train_months} trailing complete preceding calendar months, with parent action labels resolved before held boundary",
        "selection_period": "2026-04 through 2026-07 only; August is an untouched holdout",
        "feature_contracts": {name: list(value[0]) for name, value in arms.items()},
        "feature_selection": "200 random subspaces, 30--60% optional features, stability score then CMI/OOF greedy selection; 30--45 final features",
        "target_panel": str(args.target_panel.resolve()), "target_panel_sha256": _sha256(args.target_panel.resolve()),
        "vwap_panel": str(args.vwap_panel.resolve()), "vwap_panel_sha256": _sha256(args.vwap_panel.resolve()),
        "vwap": "completed-bar, 24-hour volume-weighted typical-price overlay; zero-volume rows never receive fabricated values",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
