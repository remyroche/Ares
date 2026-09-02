#!/usr/bin/env python3
"""Non-causal S/R ceiling test on the frozen E2 -> H4 continuation contract.

``H4_plus_oracle_sr`` is intentionally invalid for deployment: it receives the
future outcome of the first policy-relevant S/R interaction.  The arm exists
only to answer whether materially better S/R heads could have policy value.
All arms share candidate IDs, E2 entry selection, H4 folds, rich-policy
ordering, 20% giveback tightening, costs, and portfolio auction.
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

from scripts import run_causal_sr_continuation_ablation as causal
from scripts import run_causal_sr_oracle_audit as audit
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as study
from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as h4
from scripts import run_strict_r3_p8u_15m_continuation_v2_advantage_ablation as stable
from scripts import replay_strict_r3_p8u_15m_continuation_portfolio as port
from extreme_price_movements.portfolio_policy_replay import compute_replay_metrics, replay_candidates
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params as portfolio_params


ORACLE_ROOT = ROOT / "data_perp/artifacts/causal_sr_oracle_audit_20260830_v1"
HEADS_ROOT = ROOT / "data_perp/artifacts/causal_sr_heads_oof_20260830_v3_entrypivotfix"
FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2"
ENTRY_SELECTION = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_agreement_ablation_20260830_v1/E2_q50_agreement_selection_target_free.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_oracle_continuation_ceiling_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
H4_SPEC = h4.SPECS["H4_l1_d4_l15_leaf5_reg20"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _merge(panel: pd.DataFrame, heads_root: Path, oracle_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    value = panel.copy()
    value["snapshot_ts"] = pd.to_datetime(value.state_decision_ts, utc=True, errors="raise")
    value["__state_key"] = pd.to_numeric(value.state_bar_15m, errors="raise").astype("int16")
    keys = ["candidate_id", "snapshot_ts", "__state_key"]
    predicted = pd.read_parquet(heads_root / "continuation_sr_oof_features.parquet")
    predicted["snapshot_ts"] = pd.to_datetime(predicted.snapshot_ts, utc=True, errors="raise")
    predicted["__state_key"] = pd.to_numeric(predicted.state_bar_15m, errors="raise").astype("int16")
    predicted = predicted.rename(columns={"state_bar_15m": "__head_state_bar"})
    if predicted.duplicated(keys).any():
        raise AssertionError("predicted S/R continuation identity is duplicated")
    oracle = pd.read_parquet(oracle_root / "continuation_oracle_labels_NONCAUSAL_DIAGNOSTIC_ONLY.parquet")
    oracle["snapshot_ts"] = pd.to_datetime(oracle.snapshot_ts, utc=True, errors="raise")
    oracle["__state_key"] = pd.to_numeric(oracle.__state_key, errors="raise").astype("int16")
    if oracle.duplicated(keys).any():
        raise AssertionError("oracle S/R continuation identity is duplicated")
    prediction_cols = [item for item in causal.SR_FEATURES if item in predicted]
    oracle_cols = [item for item in audit.ORACLE_FEATURES if item in oracle]
    result = value.merge(predicted.loc[:, [*keys, *prediction_cols]], on=keys, how="left", validate="one_to_one")
    result = result.merge(oracle.loc[:, [*keys, *oracle_cols]], on=keys, how="left", validate="one_to_one")
    coverage = result.assign(
        predicted_sr_available=result.loc[:, prediction_cols].notna().any(axis=1),
        oracle_sr_available=result.loc[:, oracle_cols].notna().any(axis=1),
    ).groupby(pd.to_datetime(result.entry_decision_ts, utc=True).dt.to_period("M"), observed=True).agg(
        rows=("candidate_id", "size"), predicted_sr_available=("predicted_sr_available", "sum"),
        oracle_sr_available=("oracle_sr_available", "sum"),
    ).reset_index(names="entry_month")
    return result, coverage


def _simulate(group: pd.DataFrame, bars: c1.CompactBars, model, features: tuple[str, ...], params, median: float) -> dict[str, object] | None:
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
        return 0.0 if active else 2.0

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


def _replay_portfolio(rows: pd.DataFrame, prices: pd.DataFrame, arm: str, output: Path) -> dict[str, object]:
    """Use the frozen portfolio contract without reopening a mutable part tree.

    The historical state-part directory is not a source of truth for this
    challenger: every exact entry price already exists in the materialised H4
    panel that generated the outcomes.  Reading that candidate-local source
    avoids letting an unrelated corrupt state part break a matched replay.
    """
    priorities = port._bcf_priority()
    tagged = rows.copy()
    tagged["arm"], tagged["mc1_threshold_bps"] = arm, 30.0
    candidates = port._candidate_table(tagged, prices, priorities)
    params = portfolio_params()
    decisions, equity, _ = replay_candidates(candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp")
    decisions = port._attach_ids(decisions, candidates)
    accepted = decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
    candidates.to_parquet(output / f"{arm}_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(output / f"{arm}_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(output / f"{arm}_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(output / f"{arm}_equity.parquet", index=False, compression="zstd")
    port._period_metrics(accepted, "month").assign(arm=arm).to_parquet(output / f"{arm}_monthly.parquet", index=False)
    returns = pd.to_numeric(candidates.iloc[pd.to_numeric(accepted.candidate_index, errors="raise").astype(int).to_numpy()].net_return, errors="raise") * 10_000.0
    return {
        "arm": arm, "routed_candidates": len(candidates), "portfolio_accepted": len(accepted),
        "policy_net_bps_per_trade": float(returns.mean()) if len(returns) else np.nan,
        "total_policy_net_bps": float(returns.sum()),
        **compute_replay_metrics(candidates, decisions, equity, params=params),
    }


def _scope(detail: pd.DataFrame, prices: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ts = pd.to_datetime(detail.entry_decision_ts, utc=True, errors="raise")
    for scope, frame in (
        ("selection_jun_jul", detail.loc[ts.lt(SELECTION_END)]),
        ("august_holdout", detail.loc[ts.ge(SELECTION_END)]),
        ("all_oos", detail),
    ):
        if frame.empty:
            continue
        metric = _replay_portfolio(frame.copy(), prices, f"{arm}__{scope}", output)
        metric["model_arm"], metric["evaluation_scope"] = arm, scope
        rows.append(metric)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    parser.add_argument("--feature-study", type=Path, default=FEATURE_STUDY)
    parser.add_argument("--entry-selection", type=Path, default=ENTRY_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; defaults June--August")
    args = parser.parse_args()
    if args.train_months < 2:
        raise ValueError("strict continuation training requires at least two prior months")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    held = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    features_by_month = h4._features_by_month(args.feature_study.resolve() / "stable_selected_features.parquet", "C4_normalized_vwap_fs", held)
    selected_ids = set(pd.read_parquet(args.entry_selection.resolve(), columns=["candidate_id"]).candidate_id.astype(str))
    panel, coverage = _merge(study._load_panel(study.TARGET_PANEL, study.VWAP_PANEL), args.heads_root.resolve(), args.oracle_root.resolve())
    panel["entry_decision_ts"] = pd.to_datetime(panel.entry_decision_ts, utc=True, errors="raise")
    panel["policy_label_available_ts"] = pd.to_datetime(panel.policy_label_available_ts, utc=True, errors="raise")
    params, median, _ = c1.base._load_policy()
    prices = panel.loc[:, ["candidate_id", "entry_price"]].copy()
    prices["candidate_id"] = prices.candidate_id.astype(str)
    prices["entry_price"] = pd.to_numeric(prices.entry_price, errors="coerce")
    if prices.entry_price.isna().any() or prices.groupby("candidate_id").entry_price.nunique(dropna=True).gt(1).any():
        raise AssertionError("H4 panel does not contain one finite immutable entry price per candidate")
    prices = prices.drop_duplicates("candidate_id", keep="first")
    out.mkdir(parents=True, exist_ok=False)
    arms = {
        "H4_giveback20_control": None,
        "H4_plus_predicted_sr_giveback20": tuple(causal.SR_FEATURES),
        "H4_plus_ORACLE_sr_NONCAUSAL_giveback20": tuple(audit.ORACLE_FEATURES),
    }
    rows_by_arm: dict[str, list[pd.DataFrame]] = {name: [] for name in arms}
    source_coverage: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    bars_cache: dict[str, c1.CompactBars | None] = {}
    fold_trace: list[dict[str, object]] = []
    for month in held:
        end, start = month + pd.offsets.MonthBegin(1), month - pd.DateOffset(months=args.train_months)
        train = panel.loc[
            pd.to_numeric(panel.MC1_expected_bps, errors="coerce").ge(30.0)
            & panel.entry_decision_ts.ge(start) & panel.entry_decision_ts.lt(month)
            & panel.policy_label_available_ts.lt(month)
        ].copy()
        test = panel.loc[
            panel.entry_decision_ts.ge(month) & panel.entry_decision_ts.lt(end)
            & panel.candidate_id.astype(str).isin(selected_ids)
        ].copy()
        base_features = tuple(features_by_month[month])
        if train.empty or test.empty:
            raise RuntimeError(f"{month:%Y-%m}: incomplete E2/H4 oracle fold")
        for arm, added in arms.items():
            features = base_features if added is None else tuple((*base_features, *added))
            missing = set(features).difference(train.columns) | set(features).difference(test.columns)
            if missing:
                raise AssertionError(f"{arm}: feature contract missing {sorted(missing)}")
            model = h4._fit(train, features, H4_SPEC)
            records: list[dict[str, object]] = []
            for symbol, symbol_rows in test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable").groupby("__symbol__", sort=True):
                token = str(symbol)
                if token not in bars_cache:
                    try:
                        bars_cache[token] = c1._load_symbol_bars(token)
                    except (ArrowInvalid, OSError, ValueError) as exc:
                        bars_cache[token] = None
                        failures.append({"__symbol__": token, "exception_type": type(exc).__name__, "exception": str(exc)})
                bars = bars_cache[token]
                source_coverage.append({"held_month": month.strftime("%Y-%m"), "__symbol__": token, "source_available": bars is not None, "candidates": int(symbol_rows.candidate_id.nunique())})
                if bars is None:
                    continue
                for _, group in symbol_rows.groupby("candidate_id", sort=True):
                    result = _simulate(group, bars, model, features, params, median)
                    if result is not None:
                        records.append(result)
            frame = pd.DataFrame(records)
            if frame.empty:
                raise RuntimeError(f"{month:%Y-%m}: {arm} had no executable outcomes")
            frame["held_month"], frame["model_arm"] = month.strftime("%Y-%m"), arm
            rows_by_arm[arm].append(frame)
        fold_trace.append({
            "held_month": month.strftime("%Y-%m"), "train_rows": len(train), "test_rows": len(test),
            "oracle_interaction_test": int(pd.to_numeric(test.sr_oracle_any_interaction, errors="coerce").fillna(0.0).sum()),
        })
    metrics: list[dict[str, object]] = []
    for arm, pieces in rows_by_arm.items():
        detail = pd.concat(pieces, ignore_index=True)
        if detail.candidate_id.duplicated().any():
            raise AssertionError(f"{arm} duplicated candidate identity")
        detail.to_parquet(out / f"{arm}_outcomes.parquet", index=False, compression="zstd")
        metrics.extend(_scope(detail, prices, arm, out))
    summary = pd.DataFrame(metrics)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        reference = group.loc[group.model_arm.eq("H4_giveback20_control")].iloc[0]
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_control_{metric}"] = group[metric] - reference[metric]
    summary.to_parquet(out / "portfolio_summary.parquet", index=False)
    coverage.to_parquet(out / "sr_merge_coverage.parquet", index=False)
    pd.DataFrame(fold_trace).to_parquet(out / "fold_trace.parquet", index=False)
    pd.DataFrame(source_coverage).to_parquet(out / "outcome_source_coverage.parquet", index=False)
    pd.DataFrame(failures).to_parquet(out / "outcome_source_failures.parquet", index=False)
    manifest = {
        "schema": "causal-sr-oracle-continuation-ceiling-v1",
        "scope": "offline non-causal ceiling diagnostic only; no live/canonical/execution mutation",
        "oracle_root": str(args.oracle_root.resolve()), "oracle_manifest_sha256": _sha256(args.oracle_root.resolve() / "run_manifest.json"),
        "heads_root": str(args.heads_root.resolve()), "heads_manifest_sha256": _sha256(args.heads_root.resolve() / "run_manifest.json"),
        "entry_selection": str(args.entry_selection.resolve()), "entry_selection_sha256": _sha256(args.entry_selection.resolve()),
        "policy": "exact H4 50%-earlier activation plus 20% tighter giveback; parent rich-policy ordering, costs, and global portfolio auction unchanged",
        "oracle_arm_warning": "H4_plus_ORACLE_sr_NONCAUSAL_giveback20 receives future S/R interaction outcomes and is forbidden from inference, live policy, or causal performance claims",
        "held_months": [f"{item:%Y-%m}" for item in held], "fold_trace": fold_trace,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
