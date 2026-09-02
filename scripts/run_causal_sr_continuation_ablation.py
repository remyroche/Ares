#!/usr/bin/env python3
"""Strict-OOS test of causal S/R features in the frozen H4 continuation arm.

The entry population, parent policy, 50%-earlier trailing authority and 20%
giveback tightening are fixed.  This script tests only whether independently
OOF S/R snapshot features improve the continuation decision.  It is offline
research and has no execution or live-model side effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as h4
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as study
from scripts import run_strict_r3_p8u_15m_continuation_v2_advantage_ablation as stable
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1


SR_ROOT = ROOT / "data_perp/artifacts/causal_sr_heads_oof_20260830_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_continuation_h4_giveback20_20260830_v1"
FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2"
ENTRY_SELECTION = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_agreement_ablation_20260830_v1/E2_q50_agreement_selection_target_free.parquet"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")

SR_FEATURES = (
    "sr_long_support_hold_strength", "sr_long_resistance_break_probability",
    "sr_long_downside_break_probability", "sr_long_resistance_rejection_strength",
    "sr_long_structure_balance", "sr_long_support_distance_atr",
    "sr_long_resistance_distance_atr", "sr_support_prior_strength",
    "sr_resistance_prior_strength", "sr_support_reaction_magnitude_q50",
    "sr_resistance_reaction_magnitude_q50",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge_sr(panel: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sr = pd.read_parquet(path)
    sr["snapshot_ts"] = pd.to_datetime(sr.snapshot_ts, utc=True, errors="raise")
    sr["state_bar_15m"] = pd.to_numeric(sr.state_bar_15m, errors="raise").astype(int)
    keys = ["candidate_id", "state_bar_15m", "snapshot_ts"]
    if sr.duplicated(keys).any():
        raise AssertionError("S/R continuation snapshots have duplicate state identities")
    work = panel.copy()
    work["snapshot_ts"] = pd.to_datetime(work.state_decision_ts, utc=True, errors="raise")
    work["state_bar_15m"] = pd.to_numeric(work.state_bar_15m, errors="raise").astype(int)
    merged = work.merge(sr.loc[:, [*keys, *SR_FEATURES]], on=keys, how="left", validate="one_to_one")
    coverage = merged.assign(sr_available=merged.loc[:, list(SR_FEATURES)].notna().any(axis=1)).groupby(
        pd.to_datetime(merged.entry_decision_ts, utc=True).dt.to_period("M"), observed=True
    ).agg(rows=("candidate_id", "size"), sr_available=("sr_available", "sum")).reset_index(names="entry_month")
    return merged, coverage


def _simulate(group: pd.DataFrame, bars: c1.CompactBars, model, features: tuple[str, ...], params, median: float) -> dict[str, object] | None:
    first = group.iloc[0]
    path = c1._bar_path(bars, pd.Timestamp(first.entry_decision_ts))
    if path is None:
        return None
    high, low, close = path
    static = {int(row.state_bar_15m): row.loc[list(features)].to_numpy(float) for _, row in group.iterrows()}
    buffer = np.empty(len(features), dtype=float)
    calls = 0
    actions = 0

    def callback(dynamic: dict[str, float]) -> float | None:
        nonlocal calls, actions
        values = static.get(int(dynamic.pop("state_bar_15m")))
        if values is None:
            return None
        calls += 1
        buffer[:] = values
        prediction = float(model.booster_.predict(buffer.reshape(1, -1))[0])
        actions += int(prediction >= 0.0)
        return 0.0 if prediction >= 0.0 else 2.0

    trace = stable.replay_open_long_policy_with_continuation_modulator(
        entry=float(first.entry_price), signal_atr=float(first.signal_atr), highs=high, lows=low, closes=close,
        params=params, median_atr_fraction=median, prediction_for_completed_bar=callback,
        sl_tighten=0.0, giveback_tighten=0.20, activation_earlier=0.50,
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


def _scope(detail: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope, frame in (
        ("selection_jun_jul", detail.loc[pd.to_datetime(detail.entry_decision_ts, utc=True).lt(SELECTION_END)]),
        ("august_holdout", detail.loc[pd.to_datetime(detail.entry_decision_ts, utc=True).ge(SELECTION_END)]),
        ("all_oos", detail),
    ):
        if frame.empty:
            continue
        metric = stable._replay_portfolio(frame.copy(), f"{arm}__{scope}", output)
        metric["model_arm"], metric["evaluation_scope"] = arm, scope
        rows.append(metric)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-root", type=Path, default=SR_ROOT)
    parser.add_argument("--feature-study", type=Path, default=FEATURE_STUDY)
    parser.add_argument("--entry-selection", type=Path, default=ENTRY_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    args = parser.parse_args()
    if args.train_months < 2:
        raise ValueError("requires at least two strict prior months")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    selected_ids = h4._selected_ids(args.entry_selection.resolve())
    panel = study._load_panel(study.TARGET_PANEL, study.VWAP_PANEL)
    panel, coverage = _merge_sr(panel, args.sr_root.resolve() / "continuation_sr_oof_features.parquet")
    # The frozen E2 entry selection begins in June; April--May have no
    # selected-entry continuation states and must not be represented as a
    # zero-result fold.  June--July remain the predeclared selection window;
    # August is untouched.
    held_months = tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    feature_map = h4._features_by_month(args.feature_study.resolve() / "stable_selected_features.parquet", "C4_normalized_vwap_fs", held_months)
    params, median, _ = c1.base._load_policy()
    bars_cache: dict[str, c1.CompactBars | None] = {}
    bar_coverage: dict[str, dict[str, object]] = {}
    output.mkdir(parents=True, exist_ok=False)
    all_rows: dict[str, list[pd.DataFrame]] = {"H4_giveback20_control": [], "H4_plus_sr_giveback20": []}
    fold_trace: list[dict[str, object]] = []
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=args.train_months)
        train = panel.loc[
            pd.to_numeric(panel.MC1_expected_bps, errors="coerce").ge(30.0)
            & panel.entry_decision_ts.ge(start) & panel.entry_decision_ts.lt(held)
            & panel.policy_label_available_ts.lt(held)
        ].copy()
        test = panel.loc[panel.entry_decision_ts.ge(held) & panel.entry_decision_ts.lt(end) & panel.candidate_id.isin(selected_ids)].copy()
        base_features = feature_map[held]
        if train.empty or test.empty:
            raise RuntimeError(f"incomplete strict OOS fold {held:%Y-%m}")
        specs = {
            "H4_giveback20_control": tuple(base_features),
            "H4_plus_sr_giveback20": tuple((*base_features, *SR_FEATURES)),
        }
        for arm, features in specs.items():
            missing = set(features).difference(train.columns) | set(features).difference(test.columns)
            if missing:
                raise AssertionError(f"{arm} features absent: {sorted(missing)}")
            model = h4._fit(train, features, h4.SPECS["H4_l1_d4_l15_leaf5_reg20"])
            rows: list[dict[str, object]] = []
            for symbol, group in test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable").groupby("__symbol__", sort=True):
                token = str(symbol)
                if token not in bars_cache:
                    try:
                        bars_cache[token] = c1._load_symbol_bars(token)
                        bar_coverage[token] = {"__symbol__": token, "source_available": bars_cache[token] is not None}
                    except Exception as exc:
                        # Historical replay must fail closed per symbol.  Do
                        # not substitute another market feed or impute a path.
                        bars_cache[token] = None
                        bar_coverage[token] = {
                            "__symbol__": token, "source_available": False,
                            "exception_type": type(exc).__name__, "exception": str(exc),
                        }
                if bars_cache[token] is None:
                    continue
                for _, candidate in group.groupby("candidate_id", sort=True):
                    replay = _simulate(candidate, bars_cache[token], model, features, params, median)
                    if replay is not None:
                        rows.append(replay)
            frame = pd.DataFrame(rows)
            if frame.empty:
                raise RuntimeError(f"{arm} yielded no executable outcomes in {held:%Y-%m}")
            frame["held_month"], frame["model_arm"] = held.strftime("%Y-%m"), arm
            all_rows[arm].append(frame)
        fold_trace.append({"held_month": held.strftime("%Y-%m"), "train_rows": len(train), "test_rows": len(test), "sr_available_train": int(train.loc[:, list(SR_FEATURES)].notna().any(axis=1).sum()), "sr_available_test": int(test.loc[:, list(SR_FEATURES)].notna().any(axis=1).sum())})
    summary_rows: list[dict[str, object]] = []
    for arm, frames in all_rows.items():
        detail = pd.concat(frames, ignore_index=True)
        if detail.candidate_id.duplicated().any():
            raise AssertionError(f"{arm} duplicated candidate identities")
        detail.to_parquet(output / f"{arm}_outcomes.parquet", index=False, compression="zstd")
        summary_rows.extend(_scope(detail, arm, output))
    summary = pd.DataFrame(summary_rows)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        base = group.loc[group.model_arm.eq("H4_giveback20_control")].iloc[0]
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_control_{metric}"] = group[metric] - base[metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    coverage.to_parquet(output / "sr_merge_coverage.parquet", index=False)
    pd.DataFrame(bar_coverage.values()).sort_values("__symbol__", kind="stable").to_parquet(output / "outcome_source_coverage.parquet", index=False)
    pd.DataFrame(fold_trace).to_parquet(output / "fold_trace.parquet", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "schema": "causal-sr-h4-continuation-ablation-v1", "scope": "offline strict-OOS research only; does not modify live execution",
        "sr_root": str(args.sr_root.resolve()), "sr_manifest_sha256": _sha256(args.sr_root.resolve() / "run_manifest.json"),
        "entry_selection": str(args.entry_selection.resolve()), "entry_selection_sha256": _sha256(args.entry_selection.resolve()),
        "base_head": "H4_l1_d4_l15_leaf5_reg20", "authority": "unchanged: activation50 earlier trailing only; parent rich policy and 20% giveback tightening fixed",
        "held_months": [f"{x:%Y-%m}" for x in held_months], "fold_trace": fold_trace, "sr_features": list(SR_FEATURES),
        "outcome_source_contract": "candidate-local unavailable on corrupt/missing historical 15m path; no substitute or imputation",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
