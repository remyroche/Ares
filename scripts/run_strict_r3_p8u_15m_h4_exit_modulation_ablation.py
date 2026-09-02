#!/usr/bin/env python3
"""Strict-OOS H4 continuation giveback/stop modulation ablation.

H4 remains the selected stable L1 mean continuation model with its C4
normalised/VWAP feature selection and its causal next-bar action ordering.
This offline study changes only the *tightening authority* after a nonnegative
H4 advantage prediction: earlier trailing activation, tighter trailing
giveback, tighter hard stop, or bounded combinations.  No arm can loosen an
exit, apply a same-bar action, change sizing, or modify the parent policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as study
from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as h4
from scripts import run_strict_r3_p8u_15m_continuation_v2_advantage_ablation as stable


DEFAULT_ENTRY_SELECTION = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_postfs_hpo_20260830_v1/H0_q50_d3_l7_baseline_selection_target_free.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_h4_exit_modulation_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
H4_SPEC = h4.SPECS["H4_l1_d4_l15_leaf5_reg20"]

# All variants retain the selected 50% earlier-activation action.  The source
# model returns a risk action only for predicted activation50 advantage >= 0.
CONFIGS: dict[str, dict[str, float]] = {
    # Exact rich-parent control.  The H4 model is intentionally not given any
    # effective authority: all modifier magnitudes are zero.  Keeping it in
    # the same replay makes the total H4 uplift distinguishable from the
    # incremental giveback-only uplift below.
    "C0_parent_rich_policy": {"sl_tighten": 0.0, "giveback_tighten": 0.0, "activation_earlier": 0.0},
    "H4_control_activation50": {"sl_tighten": 0.0, "giveback_tighten": 0.0, "activation_earlier": 0.50},
    "H4_giveback10": {"sl_tighten": 0.0, "giveback_tighten": 0.10, "activation_earlier": 0.50},
    "H4_giveback20": {"sl_tighten": 0.0, "giveback_tighten": 0.20, "activation_earlier": 0.50},
    "H4_sl10": {"sl_tighten": 0.10, "giveback_tighten": 0.0, "activation_earlier": 0.50},
    "H4_sl20": {"sl_tighten": 0.20, "giveback_tighten": 0.0, "activation_earlier": 0.50},
    "H4_joint10": {"sl_tighten": 0.10, "giveback_tighten": 0.10, "activation_earlier": 0.50},
    "H4_joint20": {"sl_tighten": 0.20, "giveback_tighten": 0.20, "activation_earlier": 0.50},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _selected_ids(path: Path) -> set[str]:
    frame = pd.read_parquet(path)
    if "candidate_id" not in frame:
        raise ValueError("target-free entry receipt lacks candidate_id")
    ids = frame.candidate_id.astype(str)
    if ids.duplicated().any():
        raise AssertionError("target-free entry receipt duplicates identity")
    return set(ids)


def _safe_load_symbol_bars(
    symbol: str,
    cache: dict[str, c1.CompactBars | None],
    failures: list[dict[str, object]],
) -> c1.CompactBars | None:
    """Load a historical path source without allowing one archive failure to abort every arm.

    A missing, unhydrated, or corrupt symbol archive is candidate-local: it is
    unavailable to every compared arm for the affected held month.  This is a
    strict fail-closed treatment--the runner neither reconstitutes bars nor
    substitutes a later source--and retains a receipt for review.
    """
    if symbol in cache:
        return cache[symbol]
    try:
        value = c1._load_symbol_bars(symbol)
    except (ArrowInvalid, OSError, ValueError) as exc:
        failures.append(
            {
                "__symbol__": symbol,
                "source_path": str(c1.base.BARS_ROOT / c1.base._symbol_filename(symbol)),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
        )
        value = None
    cache[symbol] = value
    return value


def _simulate(group: pd.DataFrame, bars: c1.CompactBars, model, features: tuple[str, ...], params, median: float, config: dict[str, float]) -> dict[str, object] | None:
    first = group.iloc[0]
    path = c1._bar_path(bars, pd.Timestamp(first.entry_decision_ts))
    if path is None:
        return None
    high, low, close = path
    static = {int(row.state_bar_15m): row.loc[list(features)].to_numpy(float) for _, row in group.iterrows()}
    buffer = np.empty(len(features), dtype=float)
    calls = actions = 0

    def callback(dynamic: dict[str, float]) -> float | None:
        nonlocal calls, actions
        values = static.get(int(dynamic.pop("state_bar_15m")))
        if values is None:
            return None
        calls += 1
        buffer[:] = values
        prediction = float(model.booster_.predict(buffer.reshape(1, -1))[0])
        active = prediction >= 0.0
        actions += int(active)
        # The modulator maps 0 to full declared tightening and 2 to neutral.
        # Therefore this is exactly the frozen H4 binary authority, not a
        # newly tuned continuous policy score.
        return 0.0 if active else 2.0

    trace = stable.replay_open_long_policy_with_continuation_modulator(
        entry=float(first.entry_price), signal_atr=float(first.signal_atr), highs=high, lows=low, closes=close,
        params=params, median_atr_fraction=median, prediction_for_completed_bar=callback, **config,
    )
    return {
        "candidate_id": str(first.candidate_id), "__symbol__": str(first.__symbol__), "entry_decision_ts": pd.Timestamp(first.entry_decision_ts),
        "baseline_net_bps": float(first.policy_net_bps), "baseline_gross_bps": float(first.policy_gross_bps),
        "baseline_exit_bar": int(first.policy_exit_bar_15m), "baseline_exit_reason": str(first.policy_exit_reason),
        "c1_gross_bps": float(trace.terminal_gross_bps), "c1_net_bps": float(trace.terminal_gross_bps - 100.0),
        "c1_exit_bar": int(trace.terminal_exit_bar), "c1_exit_reason": str(trace.terminal_reason),
        "model_calls": calls, "action_calls": actions,
    }


def _scope_replay(detail: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    ts = pd.to_datetime(detail.entry_decision_ts, utc=True, errors="raise")
    for scope, subset in (
        ("selection_jun_jul", detail.loc[ts.lt(SELECTION_END)].copy()),
        ("august_holdout", detail.loc[ts.ge(SELECTION_END)].copy()),
        ("all_oos", detail),
    ):
        if subset.empty:
            continue
        metrics = stable._replay_portfolio(subset, f"{arm}__{scope}", output)
        metrics["model_arm"], metrics["evaluation_scope"] = arm, scope
        summaries.append(metrics)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-study", type=Path, default=h4.FEATURE_STUDY)
    parser.add_argument("--entry-selection", type=Path, default=DEFAULT_ENTRY_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Jun--Aug 2026")
    parser.add_argument("--arm", choices=tuple(CONFIGS), action="append", help="repeatable; defaults to every bounded arm")
    args = parser.parse_args()
    if args.train_months < 2:
        raise ValueError("strict H4 fold needs at least two complete prior months")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    held_months = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in args.held_month) if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    features_path = args.feature_study.resolve() / "stable_selected_features.parquet"
    feature_map = h4._features_by_month(features_path, "C4_normalized_vwap_fs", held_months)
    selected_ids = _selected_ids(args.entry_selection.resolve())
    panel = study._load_panel(study.TARGET_PANEL, study.VWAP_PANEL)
    params, median, _ = c1.base._load_policy()
    configs = {name: CONFIGS[name] for name in (args.arm or list(CONFIGS))}
    out.mkdir(parents=True, exist_ok=False)
    bars_cache: dict[str, c1.CompactBars | None] = {}
    source_failures: list[dict[str, object]] = []
    source_coverage: list[dict[str, object]] = []
    outcomes: dict[str, list[pd.DataFrame]] = {name: [] for name in configs}
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=args.train_months)
        train = panel.loc[
            pd.to_numeric(panel.MC1_expected_bps, errors="coerce").ge(30.0)
            & panel.entry_decision_ts.ge(start) & panel.entry_decision_ts.lt(held)
            & panel.policy_label_available_ts.lt(held)
        ].copy()
        test = panel.loc[panel.entry_decision_ts.ge(held) & panel.entry_decision_ts.lt(end) & panel.candidate_id.astype(str).isin(selected_ids)].copy()
        features = feature_map[held]
        observed = set(pd.to_datetime(train.entry_decision_ts, utc=True).dt.strftime("%Y-%m"))
        required = {(held - pd.DateOffset(months=n)).strftime("%Y-%m") for n in range(1, args.train_months + 1)}
        missing = set(features).difference(train.columns) | set(features).difference(test.columns)
        if missing or train.candidate_id.nunique() < 100 or test.empty or not required.issubset(observed):
            raise RuntimeError(f"{held:%Y-%m}: strict H4 modulation fold incomplete; fields={sorted(missing)}")
        model = h4._fit(train, features, H4_SPEC)
        rows: dict[str, list[dict[str, object]]] = {name: [] for name in configs}
        for symbol, symbol_rows in test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable").groupby("__symbol__", sort=True):
            token = str(symbol)
            failure_start = len(source_failures)
            bars = _safe_load_symbol_bars(token, bars_cache, source_failures)
            source_coverage.append(
                {
                    "held_month": held.strftime("%Y-%m"),
                    "__symbol__": token,
                    "candidate_count": int(symbol_rows.candidate_id.nunique()),
                    "path_source_available": bars is not None,
                }
            )
            if bars is None:
                for failure in source_failures[failure_start:]:
                    failure["held_month"] = held.strftime("%Y-%m")
                    failure["candidate_count"] = int(symbol_rows.candidate_id.nunique())
                continue
            for _, candidate in symbol_rows.groupby("candidate_id", sort=True):
                for name, config in configs.items():
                    result = _simulate(candidate, bars, model, features, params, median, config)
                    if result is not None:
                        rows[name].append(result)
        for name, values in rows.items():
            frame = pd.DataFrame(values)
            if frame.empty:
                raise RuntimeError(f"{held:%Y-%m}/{name}: no executable H4 outcomes")
            frame["held_month"], frame["arm"] = held.strftime("%Y-%m"), name
            frame["net_delta_bps"] = frame.c1_net_bps - frame.baseline_net_bps
            outcomes[name].append(frame)
    summaries: list[dict[str, object]] = []
    for name, frames in outcomes.items():
        detail = pd.concat(frames, ignore_index=True)
        if detail.candidate_id.duplicated().any():
            raise AssertionError(f"{name}: H4 outcome identity duplicated")
        detail.to_parquet(out / f"{name}_entry_outcomes.parquet", index=False, compression="zstd")
        summaries.extend(_scope_replay(detail, name, out))
    summary = pd.DataFrame(summaries)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        control = group.loc[group.model_arm.eq("H4_control_activation50")]
        if len(control) != 1:
            raise AssertionError(f"{scope}: H4 control is missing")
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "sortino", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_control_{metric}"] = group[metric] - control.iloc[0][metric]
        parent = group.loc[group.model_arm.eq("C0_parent_rich_policy")]
        if len(parent) != 1:
            raise AssertionError(f"{scope}: rich-parent control is missing")
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "sortino", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_parent_{metric}"] = group[metric] - parent.iloc[0][metric]
    summary.to_parquet(out / "portfolio_summary.parquet", index=False)
    pd.DataFrame(source_failures).to_parquet(out / "source_unavailable_symbols.parquet", index=False)
    pd.DataFrame(source_coverage).to_parquet(out / "source_coverage.parquet", index=False)
    ranking = summary.loc[summary.evaluation_scope.eq("selection_jun_jul") & ~summary.model_arm.isin(["C0_parent_rich_policy", "H4_control_activation50"])].sort_values(
        ["total_ev_per_abs_drawdown", "total_policy_net_bps", "policy_net_bps_per_trade", "worst_week"], ascending=[False, False, False, False], kind="stable"
    )
    ranking.to_parquet(out / "selection_ranking_jun_jul.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-h4-exit-modulation-v1",
        "scope": "offline strict-OOS research only; no live/canonical mutation, exchange IO, or order submission",
        "selection_period": "2026-06 through 2026-07 only; August untouched until selection is frozen",
        "held_months": [f"{value:%Y-%m}" for value in held_months],
        "entry_selection": str(args.entry_selection.resolve()), "entry_selection_sha256": _sha256(args.entry_selection.resolve()),
        "h4_model": {"spec": H4_SPEC, "feature_arm": "C4_normalized_vwap_fs", "features_sha256": _sha256(features_path)},
        "training": f"up to {args.train_months} preceding complete months; activation labels resolve before held boundary",
        "action": "H4 prediction >=0 only; state from completed 15-minute bar applies to the next bar; tightening only",
        "configs": configs,
        "parent_policy": "unchanged rich parent policy with one embedded 100-bps cost",
        "portfolio": "unchanged normal global constrained auction",
        "source_availability": "candidate-local fail-closed; every arm excludes the same symbol-month path rows when an archived source cannot be read",
        "source_failure_count": len(source_failures),
        "source_coverage_rows": len(source_coverage),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
