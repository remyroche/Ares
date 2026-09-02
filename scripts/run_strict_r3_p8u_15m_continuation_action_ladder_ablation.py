#!/usr/bin/env python3
"""Strict-OOS test of a conservative continuation action ladder.

The incumbent remains the stable H4 direct 50%-earlier-trailing action.  This
challenger may choose 25%, 50%, or 75% earlier activation only when its
separate causal advantage estimate for that action is positive; otherwise it
leaves the rich parent policy unchanged.  It cannot tighten stops, change
sizing, manufacture entries, or change a same-bar outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as features
from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as hpo
from scripts import run_strict_r3_p8u_15m_continuation_v2_advantage_ablation as stable
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1


ACTION_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_action_ladder_20260830_v1/continuation_action_ladder_states.parquet"
FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2"
ENTRY_SELECTION = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_postfs_hpo_20260830_v1/H0_q50_d3_l7_baseline_selection_target_free.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_action_ladder_ablation_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
SEED = 1729
ACTIONS = (0.25, 0.50, 0.75)
H4 = hpo.SPECS["H4_l1_d4_l15_leaf5_reg20"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_column(action: float) -> str:
    return f"activation{int(round(action * 100)):02d}_advantage_bps"


def _fit(train: pd.DataFrame, fields: tuple[str, ...], target: str) -> lgb.LGBMRegressor:
    min_child = max(8, int(np.ceil(len(train) * float(H4["min_child_fraction"]))))
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=int(H4["n_estimators"]), learning_rate=float(H4["learning_rate"]),
        max_depth=int(H4["max_depth"]), num_leaves=int(H4["num_leaves"]), min_child_samples=min_child,
        subsample=.80, colsample_bytree=.80, reg_lambda=float(H4["reg_lambda"]), random_state=SEED, n_jobs=2, verbosity=-1,
    )
    model.fit(train.loc[:, fields], pd.to_numeric(train[target], errors="raise"), sample_weight=stable._weights(train))
    return model


def _simulate(group: pd.DataFrame, bars: c1.CompactBars, models: dict[float, lgb.LGBMRegressor], fields: tuple[str, ...], params, median: float, *, ladder: bool) -> dict[str, object] | None:
    first = group.iloc[0]
    path = c1._bar_path(bars, pd.Timestamp(first.entry_decision_ts))
    if path is None:
        return None
    high, low, close = path
    static = {int(row.state_bar_15m): row.loc[list(fields)].to_numpy(float) for _, row in group.iterrows()}
    buffer = np.empty(len(fields), dtype=float)
    calls = 0
    action_counts = {action: 0 for action in ACTIONS}

    def callback(dynamic: dict[str, float]) -> float | None:
        nonlocal calls
        values = static.get(int(dynamic.pop("state_bar_15m")))
        if values is None:
            return None
        calls += 1
        buffer[:] = values
        if not ladder:
            action = 0.50 if float(models[0.50].booster_.predict(buffer.reshape(1, -1))[0]) >= 0.0 else None
        else:
            predictions = {action: float(model.booster_.predict(buffer.reshape(1, -1))[0]) for action, model in models.items()}
            action = max(predictions, key=predictions.get)
            if predictions[action] < 0.0:
                action = None
        if action is None:
            return 2.0
        action_counts[action] += 1
        # The replay's score convention maps 0 -> full authority and 2 ->
        # neutral.  The ladder uses a 75% maximum, where this represents
        # 25/50/75 exactly.  The direct control has a 50% maximum, so its
        # sole active action must return zero rather than the ladder scale.
        return 2.0 * (1.0 - action / 0.75) if ladder else 0.0

    trace = stable.replay_open_long_policy_with_continuation_modulator(
        entry=float(first.entry_price), signal_atr=float(first.signal_atr), highs=high, lows=low, closes=close,
        params=params, median_atr_fraction=median, prediction_for_completed_bar=callback,
        sl_tighten=0.0, giveback_tighten=0.0, activation_earlier=0.75 if ladder else 0.50,
    )
    return {
        "candidate_id": str(first.candidate_id), "__symbol__": str(first.__symbol__), "entry_decision_ts": pd.Timestamp(first.entry_decision_ts),
        "baseline_net_bps": float(first.policy_net_bps), "baseline_gross_bps": float(first.policy_gross_bps),
        "baseline_exit_bar": int(first.policy_exit_bar_15m), "baseline_exit_reason": str(first.policy_exit_reason),
        "c1_gross_bps": float(trace.terminal_gross_bps), "c1_net_bps": float(trace.terminal_gross_bps - 100.0),
        "c1_exit_bar": int(trace.terminal_exit_bar), "c1_exit_reason": str(trace.terminal_reason),
        "model_calls": calls, "action_calls": int(sum(action_counts.values())),
        "action25_calls": action_counts[0.25], "action50_calls": action_counts[0.50], "action75_calls": action_counts[0.75],
    }


def _scope_replays(detail: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for scope, frame in (
        ("selection_jun_jul", detail.loc[pd.to_datetime(detail.entry_decision_ts, utc=True).lt(SELECTION_END)].copy()),
        ("august_holdout", detail.loc[pd.to_datetime(detail.entry_decision_ts, utc=True).ge(SELECTION_END)].copy()),
        ("all_oos", detail),
    ):
        if frame.empty:
            continue
        metrics = stable._replay_portfolio(frame, f"{arm}__{scope}", output)
        metrics["model_arm"], metrics["evaluation_scope"] = arm, scope
        records.append(metrics)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-panel", type=Path, default=ACTION_PANEL)
    parser.add_argument("--feature-study", type=Path, default=FEATURE_STUDY)
    parser.add_argument("--entry-selection", type=Path, default=ENTRY_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Jun--Aug 2026")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    if args.train_months < 2:
        raise ValueError("strict action-ladder fits require at least two completed prior months")
    held = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))

    action_path = args.action_panel.resolve()
    # Reuse the same causal VWAP overlay and H4 field selection as the
    # retained continuation experiment; only action outcomes differ.
    panel = features._load_panel(action_path, features.VWAP_PANEL)
    for name in ("entry_decision_ts", "policy_label_available_ts"):
        panel[name] = pd.to_datetime(panel[name], utc=True, errors="raise")
    required_targets = {_target_column(action) for action in ACTIONS}
    if required_targets.difference(panel.columns):
        raise AssertionError("action-ladder panel lacks a required causal target")
    field_map = hpo._features_by_month(args.feature_study.resolve() / "stable_selected_features.parquet", "C4_normalized_vwap_fs", held)
    selection = pd.read_parquet(args.entry_selection.resolve())
    if "candidate_id" not in selection or selection.candidate_id.duplicated().any():
        raise AssertionError("entry selection must provide unique target-free candidate identities")
    selected_ids = set(selection.candidate_id.astype(str))
    params, median, _ = c1.base._load_policy()
    output.mkdir(parents=True, exist_ok=False)
    bars_cache: dict[str, c1.CompactBars | None] = {}
    outcomes: dict[str, list[pd.DataFrame]] = {"P0_parent": [], "C50_direct": [], "L75_action_ladder": []}

    for month in held:
        end, start = month + pd.offsets.MonthBegin(1), month - pd.DateOffset(months=args.train_months)
        train = panel.loc[
            pd.to_numeric(panel.MC1_expected_bps, errors="coerce").ge(30.0)
            & panel.entry_decision_ts.ge(start) & panel.entry_decision_ts.lt(month)
            & panel.policy_label_available_ts.lt(month)
        ].copy()
        test = panel.loc[panel.entry_decision_ts.ge(month) & panel.entry_decision_ts.lt(end) & panel.candidate_id.isin(selected_ids)].copy()
        fields = field_map[month]
        if train.candidate_id.nunique() < 100 or test.empty:
            raise RuntimeError(f"incomplete strict-OOS action-ladder fold {month:%Y-%m}")
        missing = set(fields).difference(train.columns) | set(fields).difference(test.columns)
        if missing:
            raise AssertionError(f"H4 feature contract absent from action panel: {sorted(missing)}")
        models = {action: _fit(train, fields, _target_column(action)) for action in ACTIONS}
        parent = stable._parent_rows(test)
        parent["held_month"] = month.strftime("%Y-%m")
        outcomes["P0_parent"].append(parent)
        direct_rows: list[dict[str, object]] = []
        ladder_rows: list[dict[str, object]] = []
        for symbol, group in test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable").groupby("__symbol__", sort=True):
            token = str(symbol)
            if token not in bars_cache:
                bars_cache[token] = c1._load_symbol_bars(token)
            bars = bars_cache[token]
            if bars is None:
                continue
            for _, candidate in group.groupby("candidate_id", sort=True):
                direct = _simulate(candidate, bars, {0.50: models[0.50]}, fields, params, median, ladder=False)
                ladder = _simulate(candidate, bars, models, fields, params, median, ladder=True)
                if direct is not None:
                    direct_rows.append(direct)
                if ladder is not None:
                    ladder_rows.append(ladder)
        for arm, rows in (("C50_direct", direct_rows), ("L75_action_ladder", ladder_rows)):
            frame = pd.DataFrame(rows)
            if frame.empty or frame.candidate_id.duplicated().any():
                raise RuntimeError(f"{arm} has incomplete candidate replay for {month:%Y-%m}")
            frame["held_month"] = month.strftime("%Y-%m")
            outcomes[arm].append(frame)

    summaries: list[dict[str, object]] = []
    action_usage: list[pd.DataFrame] = []
    for arm, frames in outcomes.items():
        detail = pd.concat(frames, ignore_index=True)
        if detail.candidate_id.duplicated().any():
            raise AssertionError(f"{arm} duplicated a strict-OOS candidate")
        detail.to_parquet(output / f"{arm}_entry_outcomes.parquet", index=False, compression="zstd")
        summaries.extend(_scope_replays(detail, arm, output))
        if arm == "L75_action_ladder":
            action_usage.append(detail.groupby("held_month", as_index=False).agg(
                candidates=("candidate_id", "size"), model_calls=("model_calls", "sum"),
                action25_calls=("action25_calls", "sum"), action50_calls=("action50_calls", "sum"), action75_calls=("action75_calls", "sum"),
            ))
    summary = pd.DataFrame(summaries)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    parent = summary.loc[summary.model_arm.eq("P0_parent")].set_index("evaluation_scope")
    for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "total_ev_per_abs_drawdown"):
        summary[f"delta_vs_parent_{metric}"] = summary.apply(lambda row: row[metric] - parent.loc[row.evaluation_scope, metric], axis=1)
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    summary.loc[summary.evaluation_scope.eq("selection_jun_jul")].sort_values(
        ["total_ev_per_abs_drawdown", "policy_net_bps_per_trade", "worst_week"], ascending=[False, False, False], kind="stable",
    ).to_parquet(output / "selection_ranking_jun_jul.parquet", index=False)
    pd.concat(action_usage, ignore_index=True).to_parquet(output / "ladder_action_usage.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-continuation-action-ladder-ablation-v1",
        "scope": "offline strict-OOS research only; no live/canonical mutation",
        "action_panel": str(action_path), "action_panel_sha256": _sha256(action_path),
        "entry_selection": str(args.entry_selection.resolve()), "entry_selection_sha256": _sha256(args.entry_selection.resolve()),
        "feature_study": str(args.feature_study.resolve()), "feature_selection_arm": "C4_normalized_vwap_fs",
        "fold": f"up to {args.train_months} trailing completed months; all policy/action labels resolve before held boundary",
        "selection_period": "June--July 2026; August is an untouched holdout", "held_months": [f"{item:%Y-%m}" for item in held],
        "arms": {
            "P0_parent": "unchanged rich policy",
            "C50_direct": "retained H4 direct activation-50 action, predicted advantage >= 0",
            "L75_action_ladder": "per-state argmax among causal 25/50/75 activation advantages if positive; otherwise neutral",
        },
        "authority": "trailing activation only; no stop, giveback, sizing, entry, or same-bar modification",
        "cost": "100 bps embedded exactly once in every target and replay outcome", "seed": SEED,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
