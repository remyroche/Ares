#!/usr/bin/env python3
"""Strict-OOS capacity-aware pairwise entry replacement challenger.

For each timestamp, the ordinary BCF-priority top two MC1>=30 candidates are
the incumbent capacity.  A 20--30 bps reserve candidate may replace only the
marginal incumbent, never expand the candidate count.  Its authority is a
lower quantile of realised policy-net advantage over that marginal candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS
from extreme_price_movements.portfolio_policy_replay import (
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params as portfolio_params
from scripts.replay_strict_r3_p8u_15m_continuation_portfolio import _attach_ids, _period_metrics
from scripts.run_strict_r3_p8u_15m_entry_ordinal_replacement_ablation import _labels


FEATURE_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_ordinal_mc1_threshold_observed25h_20260830_v4_manifested_results/target_free_15m_features.parquet"
LABEL_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_rich_policy_labels_20260830_v1_control"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_pairwise_replacement_20260830_v1"
CORE_FLOOR = 30.0
RESERVE_FLOOR = 20.0
MAX_NEW_ENTRIES = 2
MARGIN_FEATURES = (
    "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_mc1_min_bps",
    "path_efficiency_15m_1h", "return_acceleration_15m_1h", "rv_15m_1h",
    "latest_impulse_size_atr_15m", "pullback_depth_atr_15m", "relative_volume_15m_1h",
    "volume_acceleration_15m_1h", "adverse_efficiency_15m", "micro_regime_flip_score_15m",
)
PAIR_FEATURES = (*FIFTEEN_MINUTE_FEATURE_KEYS, *(f"margin__{name}" for name in MARGIN_FEATURES), "incumbent_bcf_mc1_expected_bps")


def _candidate_frame(features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "dual_mc1_min_bps",
        "bcf_mc1_expected_bps", "bcf_final_score", "finite_15m_feature_count",
        *FIFTEEN_MINUTE_FEATURE_KEYS,
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"target-free feature panel lacks {sorted(missing)}")
    output = features.copy()
    output["candidate_id"] = output.candidate_id.astype(str)
    output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True, errors="raise")
    output = output.loc[
        pd.to_numeric(output["dual_mc1_min_bps"], errors="coerce").ge(RESERVE_FLOOR)
        & pd.to_numeric(output["finite_15m_feature_count"], errors="coerce").ge(50)
    ].copy()
    if output.candidate_id.duplicated().any():
        raise AssertionError("target-free input has duplicate candidate identities")
    return output


def _marginal_incumbent(group: pd.DataFrame) -> pd.Series | None:
    core = group.loc[pd.to_numeric(group["dual_mc1_min_bps"], errors="coerce").ge(CORE_FLOOR)].copy()
    if core.empty:
        return None
    core = core.sort_values(["bcf_mc1_expected_bps", "bcf_final_score", "candidate_id"], ascending=[False, False, True], kind="stable")
    return core.iloc[min(len(core), MAX_NEW_ENTRIES) - 1]


def _pairs(frame: pd.DataFrame, *, require_labels: bool) -> pd.DataFrame:
    """One reserve-vs-marginal-incumbent row per candidate and timestamp."""
    records: list[dict[str, object]] = []
    for timestamp, group in frame.groupby("__decision_ts__", sort=True):
        incumbent = _marginal_incumbent(group)
        if incumbent is None:
            continue
        reserves = group.loc[
            pd.to_numeric(group["dual_mc1_min_bps"], errors="coerce").ge(RESERVE_FLOOR)
            & pd.to_numeric(group["dual_mc1_min_bps"], errors="coerce").lt(CORE_FLOOR)
        ]
        for _, reserve in reserves.iterrows():
            row: dict[str, object] = {
                "reserve_candidate_id": str(reserve["candidate_id"]),
                "incumbent_candidate_id": str(incumbent["candidate_id"]),
                "__decision_ts__": timestamp,
                "__symbol__": str(reserve["__symbol__"]),
                "incumbent_bcf_mc1_expected_bps": float(incumbent["bcf_mc1_expected_bps"]),
                "reserve_bcf_mc1_expected_bps": float(reserve["bcf_mc1_expected_bps"]),
                "reserve_dual_mc1_min_bps": float(reserve["dual_mc1_min_bps"]),
            }
            for feature in FIFTEEN_MINUTE_FEATURE_KEYS:
                row[feature] = reserve[feature]
            for feature in MARGIN_FEATURES:
                row[f"margin__{feature}"] = float(reserve[feature]) - float(incumbent[feature])
            if require_labels:
                row["pair_advantage_bps"] = float(reserve["policy_net_bps"]) - float(incumbent["policy_net_bps"])
                row["pair_label_available_ts"] = max(
                    pd.Timestamp(reserve["policy_label_available_ts"]),
                    pd.Timestamp(incumbent["policy_label_available_ts"]),
                )
            records.append(row)
    return pd.DataFrame(records)


def _fit(train: pd.DataFrame, reserve_weight: float, quantile: float) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=quantile, n_estimators=350, learning_rate=0.03,
        max_depth=3, num_leaves=7, min_child_samples=max(8, int(np.ceil(len(train) * 0.03))),
        subsample=0.8, colsample_bytree=0.8, reg_lambda=8.0, random_state=1729, n_jobs=2, verbosity=-1,
    )
    # A constant sample weight would be inert because this population contains
    # reserve pairs only.  Weight *within* the declared 20--30 reserve band:
    # 20 bps retains unit weight, while the upper edge receives the requested
    # 1.5x/2x emphasis.  This is known at decision time and changes the fit.
    reserve_position = np.clip((pd.to_numeric(train["reserve_dual_mc1_min_bps"], errors="raise").to_numpy(float) - RESERVE_FLOOR) / (CORE_FLOOR - RESERVE_FLOOR), 0.0, 1.0)
    weights = 1.0 + (float(reserve_weight) - 1.0) * reserve_position
    model.fit(train.loc[:, PAIR_FEATURES], pd.to_numeric(train["pair_advantage_bps"], errors="raise"), sample_weight=weights)
    return model


def _incumbent_top2(frame: pd.DataFrame) -> pd.DataFrame:
    core = frame.loc[pd.to_numeric(frame["dual_mc1_min_bps"], errors="coerce").ge(CORE_FLOOR)].copy()
    core = core.sort_values(["__decision_ts__", "bcf_mc1_expected_bps", "bcf_final_score", "candidate_id"], ascending=[True, False, False, True], kind="stable")
    return core.loc[core.groupby("__decision_ts__", sort=False).cumcount().lt(MAX_NEW_ENTRIES)].copy()


def _apply_replacement(frame: pd.DataFrame, predictions: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = _incumbent_top2(frame)
    candidates = predictions.loc[pd.to_numeric(predictions["pair_lcb_advantage_bps"], errors="coerce").ge(threshold)].copy()
    if candidates.empty:
        return chosen, candidates.assign(promoted=False, reason="lcb_below_threshold")
    candidates = candidates.sort_values(["__decision_ts__", "pair_lcb_advantage_bps", "reserve_bcf_mc1_expected_bps", "reserve_candidate_id"], ascending=[True, False, False, True], kind="stable")
    proposals: list[dict[str, object]] = []
    output = chosen.set_index("candidate_id", drop=False).copy()
    for timestamp, group in candidates.groupby("__decision_ts__", sort=True):
        reserve = group.iloc[0]
        incumbents = output.loc[output["__decision_ts__"].eq(timestamp)].copy().reset_index(drop=True)
        if incumbents.empty:
            proposals.append({**reserve.to_dict(), "promoted": False, "reason": "no_incumbent"})
            continue
        marginal = incumbents.sort_values(["bcf_mc1_expected_bps", "bcf_final_score", "candidate_id"], ascending=[True, True, True], kind="stable").iloc[0]
        if str(reserve["incumbent_candidate_id"]) != str(marginal["candidate_id"]):
            proposals.append({**reserve.to_dict(), "promoted": False, "reason": "marginal_changed"})
            continue
        replacement = frame.loc[frame["candidate_id"].astype(str).eq(str(reserve["reserve_candidate_id"]))]
        if len(replacement) != 1:
            raise AssertionError("reserve proposal does not resolve to one target-free candidate")
        output = output.drop(index=str(marginal["candidate_id"]))
        output.loc[str(reserve["reserve_candidate_id"])] = replacement.iloc[0]
        proposals.append({**reserve.to_dict(), "promoted": True, "reason": "replaced_marginal_incumbent"})
    chosen_out = output.reset_index(drop=True)
    if chosen_out.groupby("__decision_ts__").size().gt(MAX_NEW_ENTRIES).any():
        raise AssertionError("replacement expanded timestamp capacity")
    return chosen_out, pd.DataFrame(proposals)


def _candidate_table(selection: pd.DataFrame, labels: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, int]:
    before = len(selection)
    frame = selection.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    valid = frame["policy_path_valid"].fillna(False).astype(bool)
    frame = frame.loc[valid].copy()
    exit_bar = pd.to_numeric(frame["policy_exit_bar_15m"], errors="raise").astype(int)
    gross_bps = pd.to_numeric(frame["policy_gross_bps"], errors="raise")
    candidates = pd.DataFrame({
        "timestamp": frame["__decision_ts__"], "candidate_id": frame["candidate_id"], "symbol": frame["__symbol__"], "side": "long",
        "strategy_id": "strict_r3_p8u_15m_pairwise_replacement_long", "policy_archetype": arm,
        "normalized_rank_score": 1.0, "strategy_rank_pct": 1.0, "base_strategy_threshold": 0.0, "calibrated_score": 1.0,
        "portfolio_priority_adjustment": pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="raise"),
        "entry_price": pd.to_numeric(frame["policy_entry_price"], errors="raise"),
        "exit_timestamp": frame["__decision_ts__"] + pd.to_timedelta((exit_bar + 1) * 15, unit="m"),
        "exit_price": pd.to_numeric(frame["policy_exit_price"], errors="raise"),
        "net_return": pd.to_numeric(frame["policy_net_bps"], errors="raise") / 10_000.0,
        "gross_return": gross_bps / 10_000.0, "holding_bars": exit_bar + 1,
        "simple_policy_exit_reason": frame["policy_exit_reason"].astype(str), "fees_bps": 100.0,
        "expected_friction_bps": 0.0, "price_gap_bps": 0.0, "liquidity_capacity_weight": 1.0,
    })
    return normalise_candidate_table(candidates), before - len(frame)


def _replay(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> dict[str, object]:
    candidates, unlabelled = _candidate_table(selection, labels, arm)
    params = portfolio_params()
    decisions, equity, _ = replay_candidates(candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp")
    decisions = _attach_ids(decisions, candidates)
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    candidates.to_parquet(output / f"{arm}_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(output / f"{arm}_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(output / f"{arm}_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(output / f"{arm}_equity.parquet", index=False, compression="zstd")
    _period_metrics(accepted, "month").assign(arm=arm).to_parquet(output / f"{arm}_monthly.parquet", index=False)
    metrics = compute_replay_metrics(candidates, decisions, equity, params=params)
    returns = pd.to_numeric(candidates.iloc[pd.to_numeric(accepted["candidate_index"], errors="raise").astype(int).to_numpy()]["net_return"], errors="raise") * 10_000.0
    return {
        "arm": arm, "selected_before_outcomes": len(selection), "outcome_unavailable": unlabelled,
        "portfolio_routed": len(candidates), "portfolio_accepted": len(accepted),
        "policy_net_bps_per_trade": float(returns.mean()) if len(returns) else np.nan,
        "total_policy_net_bps": float(returns.sum()),
        **metrics,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-panel", type=Path, default=FEATURE_PANEL)
    parser.add_argument("--labels-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Apr--Aug 2026")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    target_free = _candidate_frame(pd.read_parquet(args.feature_panel.resolve()))
    labels = _labels(args.labels_root.resolve())
    labelled = target_free.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled["policy_path_valid"].fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(labelled["policy_label_available_ts"], utc=True, errors="raise")
    held_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-04-01", "2026-08-01", freq="MS", tz="UTC"))
    variant_specs = ((1.0, 25.0), (1.5, 25.0), (2.0, 25.0), (2.0, 50.0))
    all_predictions: list[pd.DataFrame] = []
    controls: list[pd.DataFrame] = []
    promotion_logs: list[pd.DataFrame] = []
    selections: dict[str, list[pd.DataFrame]] = {f"W{weight:g}_LCB{int(threshold)}": [] for weight, threshold in variant_specs}
    for held in held_months:
        held_end = held + pd.offsets.MonthBegin(1)
        train_start = held - pd.DateOffset(months=2)
        train_raw = labelled.loc[
            labelled["__decision_ts__"].ge(train_start) & labelled["__decision_ts__"].lt(held)
        ].copy()
        train_pairs = _pairs(train_raw, require_labels=True)
        train_pairs = train_pairs.loc[train_pairs["pair_label_available_ts"].lt(held)].copy()
        required = {(held - pd.DateOffset(months=1)).strftime("%Y-%m"), (held - pd.DateOffset(months=2)).strftime("%Y-%m")}
        observed = set(pd.to_datetime(train_pairs["__decision_ts__"], utc=True).dt.strftime("%Y-%m")) if not train_pairs.empty else set()
        test = target_free.loc[target_free["__decision_ts__"].ge(held) & target_free["__decision_ts__"].lt(held_end)].copy()
        test_pairs = _pairs(test, require_labels=False)
        if not required.issubset(observed) or len(train_pairs) < 100 or test.empty:
            continue
        base_selection = _incumbent_top2(test)
        base_selection["held_month"] = held.strftime("%Y-%m")
        controls.append(base_selection)
        for weight, threshold in variant_specs:
            model = _fit(train_pairs, reserve_weight=weight, quantile=0.20)
            predicted = test_pairs.loc[:, ["reserve_candidate_id", "incumbent_candidate_id", "__decision_ts__", "__symbol__", "reserve_bcf_mc1_expected_bps", "incumbent_bcf_mc1_expected_bps"]].copy()
            predicted["pair_lcb_advantage_bps"] = model.predict(test_pairs.loc[:, PAIR_FEATURES])
            predicted["reserve_weight"] = weight
            predicted["promotion_threshold_bps"] = threshold
            predicted["held_month"] = held.strftime("%Y-%m")
            all_predictions.append(predicted)
            selection, log = _apply_replacement(test, predicted, threshold)
            selection["held_month"] = held.strftime("%Y-%m")
            tag = f"W{weight:g}_LCB{int(threshold)}"
            selection["arm"] = tag
            log["arm"] = tag
            promotion_logs.append(log)
            selections[tag].append(selection)
    if not controls or not all_predictions:
        raise RuntimeError("no complete strict-OOS pairwise replacement folds")
    output.mkdir(parents=True, exist_ok=False)
    predictions = pd.concat(all_predictions, ignore_index=True)
    promotion = pd.concat(promotion_logs, ignore_index=True) if promotion_logs else pd.DataFrame()
    control = pd.concat(controls, ignore_index=True)
    if control.duplicated("candidate_id").any():
        raise AssertionError("BCF top-two control duplicated a candidate")
    results: list[dict[str, object]] = []
    results.append(_replay(control, labels, "B0_bcf_top2", output))
    control.to_parquet(output / "B0_bcf_top2_selection_target_free.parquet", index=False)
    for arm, frames in selections.items():
        selection = pd.concat(frames, ignore_index=True)
        if selection.duplicated("candidate_id").any():
            raise AssertionError(f"{arm}: replacement selection duplicated a candidate")
        if selection.groupby("__decision_ts__").size().gt(MAX_NEW_ENTRIES).any():
            raise AssertionError(f"{arm}: replacement expanded timestamp capacity")
        selection.to_parquet(output / f"{arm}_selection_target_free.parquet", index=False)
        results.append(_replay(selection, labels, arm, output))
    summary = pd.DataFrame(results)
    control_metrics = summary.loc[summary["arm"].eq("B0_bcf_top2")]
    if len(control_metrics) != 1:
        raise AssertionError("one BCF top-two replay control is required")
    for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "compounded_return", "sortino", "max_drawdown", "worst_week"):
        summary[f"delta_vs_B0_{metric}"] = summary[metric] - control_metrics.iloc[0][metric]
    predictions.to_parquet(output / "pairwise_predictions.parquet", index=False)
    promotion.to_parquet(output / "promotion_log.parquet", index=False)
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    monthly = []
    for arm in summary["arm"]:
        monthly.append(pd.read_parquet(output / f"{arm}_monthly.parquet"))
    pd.concat(monthly, ignore_index=True).to_parquet(output / "monthly_metrics.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-pairwise-capacity-replacement-v1",
        "scope": "offline strict-OOS challenger only; no live/canonical mutation",
        "target_free_population": "complete 15-minute source rows with dual MC1 >=20; labels are joined only after selection for outcome evaluation",
        "incumbent": "BCF MC1 priority top two dual-MC1>=30 candidates per timestamp",
        "pair_target": "reserve rich-policy net bps minus timestamp marginal incumbent rich-policy net bps",
        "training": "two complete prior calendar months; both policy labels resolved before held boundary",
        "reserve_weight_ablations": {
            "multipliers": [weight for weight, _ in variant_specs],
            "formula": "1 + (multiplier - 1) * clip((reserve_dual_mc1_min_bps - 20) / 10, 0, 1)",
        },
        "authority": "one reserve can replace the marginal BCF incumbent only; it cannot increase timestamp candidate count",
        "lower_confidence_bound": "LightGBM quantile alpha=0.20; thresholds 25/50 bps",
        "priority_after_replacement": "unchanged BCF MC1 expected bps in the normal global portfolio auction",
        "portfolio": asdict(portfolio_params()),
        "cost": "100 bps embedded once in source-aligned rich-policy outcomes",
        "feature_panel": str(args.feature_panel.resolve()), "feature_panel_sha256": _sha256(args.feature_panel.resolve()),
        "labels_root": str(args.labels_root.resolve()),
        "held_months": [value.strftime("%Y-%m") for value in held_months],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
