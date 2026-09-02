#!/usr/bin/env python3
"""Bounded strict-OOS HPO for the selected stable-mean continuation head.

The continuation target and authority are intentionally frozen: predict the
activation-50 advantage, then only enable the existing earlier-trailing action
on the following 15-minute interval when the L1 mean prediction is nonnegative.
This runner changes no stop, trailing, sizing, or policy parameter.  It tests
only well-supported LightGBM geometry after per-fold causal feature selection.
April--July are the model-selection period; August is an untouched holdout.
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

from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as study
from scripts import run_strict_r3_p8u_15m_continuation_v2_advantage_ablation as stable
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1


FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_postfs_hpo_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
SEED = 1729

# L1 mean is the frozen stable direct-action family.  The remaining grid is
# deliberately small and support-first; H0 exactly recreates its prior fit.
SPECS: dict[str, dict[str, float | int]] = {
    "H0_l1_d3_l7_baseline": {"max_depth": 3, "num_leaves": 7, "min_child_fraction": .025, "reg_lambda": 8., "learning_rate": .03, "n_estimators": 350},
    "H1_l1_d2_l3_strict": {"max_depth": 2, "num_leaves": 3, "min_child_fraction": .04, "reg_lambda": 12., "learning_rate": .03, "n_estimators": 350},
    "H2_l1_d3_l7_leaf4_reg16": {"max_depth": 3, "num_leaves": 7, "min_child_fraction": .04, "reg_lambda": 16., "learning_rate": .03, "n_estimators": 350},
    "H3_l1_d3_l7_leaf5_reg20": {"max_depth": 3, "num_leaves": 7, "min_child_fraction": .05, "reg_lambda": 20., "learning_rate": .025, "n_estimators": 420},
    "H4_l1_d4_l15_leaf5_reg20": {"max_depth": 4, "num_leaves": 15, "min_child_fraction": .05, "reg_lambda": 20., "learning_rate": .025, "n_estimators": 420},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _features_by_month(path: Path, arm: str, held_months: tuple[pd.Timestamp, ...]) -> dict[pd.Timestamp, tuple[str, ...]]:
    selected = pd.read_parquet(path)
    scoped = selected.loc[selected["arm"].eq(arm)].copy()
    result: dict[pd.Timestamp, tuple[str, ...]] = {}
    for held in held_months:
        rows = scoped.loc[scoped["held_month"].eq(held.strftime("%Y-%m"))].sort_values("position", kind="stable")
        fields = tuple(rows["feature"].astype(str))
        if not 30 <= len(fields) <= 45 or len(set(fields)) != len(fields):
            raise AssertionError(f"{arm} feature contract for {held:%Y-%m} is not a 30--45 unique-field selection")
        result[held] = fields
    return result


def _selected_ids(path: Path) -> set[str]:
    """Read either a frozen predictive selector or a target-free selection receipt.

    Both formats contain only decision-time candidate identities.  The latter
    lets a separately selected entry arm feed continuation without joining any
    outcomes back into its eligibility set.
    """
    frame = pd.read_parquet(path)
    if "candidate_id" not in frame:
        raise ValueError("entry-selection source lacks candidate_id")
    if {"floor_bps", "model_spec", "selected__veto_pred_ge_0"}.issubset(frame.columns):
        return stable._selection_ids(path)
    ids = frame["candidate_id"].astype(str)
    if ids.duplicated().any():
        raise AssertionError("target-free entry selection contains duplicate candidate identities")
    return set(ids)


def _fit(train: pd.DataFrame, features: tuple[str, ...], spec: dict[str, float | int]) -> lgb.LGBMRegressor:
    min_child = max(8, int(np.ceil(len(train) * float(spec["min_child_fraction"]))))
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=int(spec["n_estimators"]), learning_rate=float(spec["learning_rate"]),
        max_depth=int(spec["max_depth"]), num_leaves=int(spec["num_leaves"]), min_child_samples=min_child,
        subsample=.80, colsample_bytree=.80, reg_lambda=float(spec["reg_lambda"]),
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    weights = stable._weights(train)
    if "__recency_weight__" in train:
        weights = weights * pd.to_numeric(train["__recency_weight__"], errors="raise").to_numpy(float)
    model.fit(train.loc[:, features], train["activation50_advantage_bps"], sample_weight=weights)
    return model


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
        values = static.get(int(dynamic.pop("state_bar_15m")))
        if values is None:
            return None
        calls += 1
        buffer[:] = values
        prediction = float(model.booster_.predict(buffer.reshape(1, -1))[0])
        active = prediction >= 0.0
        actions += int(active)
        return 0.0 if active else 2.0

    trace = stable.replay_open_long_policy_with_continuation_modulator(
        entry=float(first.entry_price), signal_atr=float(first.signal_atr), highs=high, lows=low, closes=close,
        params=params, median_atr_fraction=median, prediction_for_completed_bar=callback,
        sl_tighten=0.0, giveback_tighten=0.0, activation_earlier=0.50,
    )
    return {
        "candidate_id": str(first.candidate_id), "__symbol__": str(first.__symbol__), "entry_decision_ts": pd.Timestamp(first.entry_decision_ts),
        "baseline_net_bps": float(first.policy_net_bps), "baseline_gross_bps": float(first.policy_gross_bps),
        "baseline_exit_bar": int(first.policy_exit_bar_15m), "baseline_exit_reason": str(first.policy_exit_reason),
        "c1_gross_bps": float(trace.terminal_gross_bps), "c1_net_bps": float(trace.terminal_gross_bps - 100.0),
        "c1_exit_bar": int(trace.terminal_exit_bar), "c1_exit_reason": str(trace.terminal_reason),
        "model_calls": calls, "action_calls": actions,
    }


def _scope_replays(detail: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for scope, frame in (
        ("selection_apr_jul", detail.loc[pd.to_datetime(detail.entry_decision_ts, utc=True).lt(SELECTION_END)].copy()),
        ("august_holdout", detail.loc[pd.to_datetime(detail.entry_decision_ts, utc=True).ge(SELECTION_END)].copy()),
        ("all_oos", detail),
    ):
        if frame.empty:
            continue
        metrics = stable._replay_portfolio(frame, f"{arm}__{scope}", output)
        metrics["model_arm"], metrics["evaluation_scope"] = arm, scope
        summaries.append(metrics)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-study", type=Path, default=FEATURE_STUDY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Apr--Aug 2026")
    parser.add_argument("--spec", choices=tuple(SPECS), action="append", default=[], help="repeatable bounded model arm")
    parser.add_argument("--entry-selection", type=Path, help="optional immutable target-free entry-selection receipt")
    parser.add_argument("--recency-half-life-days", type=float, help="optional causal exponential sample-weight half-life; omitted is exact prior behavior")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    if args.train_months < 2:
        raise ValueError("strict training requires at least two completed prior months")
    if args.recency_half_life_days is not None and not np.isfinite(args.recency_half_life_days) or args.recency_half_life_days is not None and args.recency_half_life_days <= 0.0:
        raise ValueError("recency half-life must be finite and positive")
    held_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-04-01", "2026-08-01", freq="MS", tz="UTC"))
    study_root = args.feature_study.resolve()
    feature_path = study_root / "stable_selected_features.parquet"
    panel = study._load_panel(study.TARGET_PANEL, study.VWAP_PANEL)
    feature_map = _features_by_month(feature_path, "C4_normalized_vwap_fs", held_months)
    entry_selection = args.entry_selection.resolve() if args.entry_selection else study.ENTRY_PREDICTIONS.resolve()
    selected_ids = _selected_ids(entry_selection)
    params, median, _ = c1.base._load_policy()
    specs = {name: SPECS[name] for name in (args.spec or list(SPECS))}
    output.mkdir(parents=True, exist_ok=False)
    bars_cache: dict[str, c1.CompactBars | None] = {}
    all_rows: dict[str, list[pd.DataFrame]] = {name: [] for name in specs}
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=args.train_months)
        train = panel.loc[
            pd.to_numeric(panel.MC1_expected_bps, errors="coerce").ge(30.0)
            & panel.entry_decision_ts.ge(start) & panel.entry_decision_ts.lt(held)
            & panel.policy_label_available_ts.lt(held)
        ].copy()
        if args.recency_half_life_days is not None:
            age_days = (held - pd.to_datetime(train.entry_decision_ts, utc=True, errors="raise")).dt.total_seconds() / 86_400.0
            if (age_days < 0.0).any():
                raise AssertionError("causal recency age became negative")
            train["__recency_weight__"] = np.exp(-np.log(2.0) * age_days.to_numpy(float) / args.recency_half_life_days)
        test = panel.loc[
            panel.entry_decision_ts.ge(held) & panel.entry_decision_ts.lt(end) & panel.candidate_id.isin(selected_ids)
        ].copy()
        if train.candidate_id.nunique() < 100 or test.empty:
            raise RuntimeError(f"incomplete strict-OOS continuation fold {held:%Y-%m}")
        features = feature_map[held]
        missing = set(features).difference(train.columns) | set(features).difference(test.columns)
        if missing:
            raise AssertionError(f"selected features are absent: {sorted(missing)}")
        for name, spec in specs.items():
            model = _fit(train, features, spec)
            rows: list[dict[str, object]] = []
            for symbol, group in test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable").groupby("__symbol__", sort=True):
                token = str(symbol)
                if token not in bars_cache:
                    bars_cache[token] = c1._load_symbol_bars(token)
                bars = bars_cache[token]
                if bars is None:
                    continue
                for _, candidate in group.groupby("candidate_id", sort=True):
                    replay = _simulate(candidate, bars, model, features, params, median)
                    if replay is not None:
                        rows.append(replay)
            frame = pd.DataFrame(rows)
            if frame.empty:
                raise RuntimeError(f"{name} produced no executable continuation outcomes for {held:%Y-%m}")
            frame["held_month"], frame["hpo_arm"] = held.strftime("%Y-%m"), name
            all_rows[name].append(frame)
    summaries: list[dict[str, object]] = []
    for name, frames in all_rows.items():
        detail = pd.concat(frames, ignore_index=True)
        if detail.candidate_id.duplicated().any():
            raise AssertionError(f"{name} duplicated a strict-OOS candidate")
        detail.to_parquet(output / f"{name}_entry_outcomes.parquet", index=False, compression="zstd")
        summaries.extend(_scope_replays(detail, name, output))
    summary = pd.DataFrame(summaries)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    selection = summary.loc[summary.evaluation_scope.eq("selection_apr_jul")].sort_values(["total_ev_per_abs_drawdown", "policy_net_bps_per_trade", "worst_week"], ascending=[False, False, False], kind="stable")
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    selection.to_parquet(output / "selection_ranking_apr_jul.parquet", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "schema": "strict-r3-p8u-continuation-postfs-hpo-v1", "scope": "offline strict-OOS research only; no live/canonical mutation",
        "feature_study": str(study_root), "feature_selection_arm": "C4_normalized_vwap_fs", "feature_selection_sha256": _sha256(feature_path),
        "fold": f"up to {args.train_months} trailing complete prior months; activation labels resolved before each held boundary",
        "selection_period": "2026-04 through 2026-07 only; August untouched", "held_months": [f"{x:%Y-%m}" for x in held_months],
        "entry_selection_source": str(entry_selection), "entry_selection_sha256": _sha256(entry_selection),
        "direct_action": "frozen stable L1 mean activation50 advantage; prediction >= 0 activates only existing 50% earlier trailing overlay on next completed 15m interval",
        "recency_half_life_days": args.recency_half_life_days,
        "specs": specs, "seed": SEED,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
