#!/usr/bin/env python3
"""August-only gradual H4 holdout on the canonical E2-selected contract.

This is an offline research evaluation.  It uses the exact candidate IDs,
15-minute bars, rich parent policy, and normal portfolio auction of the
canonical H4 handover.  The gradual parameters were frozen from the separate
v58 screen; the August result below does not retune them.
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
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_continuation_state import replay_open_long_policy_with_gradual_continuation_modulator
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as study
from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as h4
from scripts import run_strict_r3_p8u_15m_continuation_v2_advantage_ablation as stable

ENTRY_SELECTION = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_e2_demotion_residual_20260830_v1/E2_q50_agreement_selection_target_free.parquet"
FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2"
PARENT_RECEIPT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_h4_parent_vs_giveback20_e2_20260830_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_e2_h4_gradual_august_holdout_20260830_v1"
HELD = pd.Timestamp("2026-08-01", tz="UTC")
SPEC = h4.SPECS["H4_l1_d4_l15_leaf5_reg20"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_ids(path: Path) -> set[str]:
    ids = pd.read_parquet(path)["candidate_id"].astype(str)
    if ids.duplicated().any():
        raise AssertionError("E2 selection contains duplicate candidate IDs")
    return set(ids)


def _load_bars(symbol: str, cache: dict[str, c1.CompactBars | None], failures: list[dict[str, str]]) -> c1.CompactBars | None:
    if symbol not in cache:
        try:
            cache[symbol] = c1._load_symbol_bars(symbol)
        except (ArrowInvalid, OSError, ValueError) as exc:
            cache[symbol] = None
            failures.append({"symbol": symbol, "exception": f"{type(exc).__name__}: {exc}"})
    return cache[symbol]


def _probability(raw: float, *, threshold: float, power: float) -> tuple[float, float]:
    p = float(np.clip(raw, 0.0, 1.0))
    high = ((p - threshold) / (1.0 - threshold)) ** power if p >= threshold else 0.0
    low = ((threshold - p) / threshold) ** power if p < threshold else 0.0
    return float(high), float(low)


def _multipliers(p: float) -> dict[str, float]:
    # Frozen from the preceding v58 gradual screen.  There is no August
    # parameter tuning here.  High probability moves exits earlier/tighter;
    # low probability postpones/widens only before a floor has ratcheted.
    high_a, low_a = _probability(p, threshold=.20, power=2.0)
    high_g, low_g = _probability(p, threshold=.20, power=.50)
    return {
        "activation_multiplier": float(np.clip(1.0 - .75 * high_a + .75 * low_a, .20, 1.80)),
        "giveback_multiplier": float(np.clip(1.0 - .30 * high_g + .30 * low_g, .20, 1.80)),
        "sl_distance_multiplier": 1.0,
    }


def _replay_candidate(
    group: pd.DataFrame,
    *,
    bars: c1.CompactBars,
    model,
    calibrator: IsotonicRegression,
    features: tuple[str, ...],
    params,
    median: float,
    gradual: bool,
) -> dict[str, object] | None:
    first = group.iloc[0]
    path = c1._bar_path(bars, pd.Timestamp(first.entry_decision_ts))
    if path is None:
        return None
    high, low, close = path
    static = {int(row.state_bar_15m): row.loc[list(features)].to_numpy(float) for _, row in group.iterrows()}
    buffer = np.empty(len(features), dtype=float)
    calls = 0

    def callback(state: dict[str, float]) -> dict[str, float] | None:
        nonlocal calls
        values = static.get(int(state["state_bar_15m"]))
        if values is None or not gradual:
            return None
        calls += 1
        buffer[:] = values
        raw = float(model.booster_.predict(buffer.reshape(1, -1))[0])
        probability = float(calibrator.predict(np.asarray([raw]))[0])
        return _multipliers(probability)

    trace = replay_open_long_policy_with_gradual_continuation_modulator(
        entry=float(first.entry_price), signal_atr=float(first.signal_atr), highs=high, lows=low, closes=close,
        params=params, median_atr_fraction=median, modulation_for_completed_bar=callback,
        allow_stop_extension=False, max_stop_loss_fraction=.05,
    )
    return {
        "candidate_id": str(first.candidate_id), "__symbol__": str(first.__symbol__),
        "entry_decision_ts": pd.Timestamp(first.entry_decision_ts),
        "baseline_net_bps": float(first.policy_net_bps), "baseline_gross_bps": float(first.policy_gross_bps),
        "baseline_exit_bar": int(first.policy_exit_bar_15m), "baseline_exit_reason": str(first.policy_exit_reason),
        "c1_gross_bps": float(trace.terminal_gross_bps), "c1_net_bps": float(trace.terminal_gross_bps - 100.0),
        "c1_exit_bar": int(trace.terminal_exit_bar), "c1_exit_reason": str(trace.terminal_reason),
        "model_calls": calls,
    }


def _metrics(detail: pd.DataFrame, arm: str, output: Path) -> dict[str, object]:
    """Replay against the parent receipt's immutable August entry prices.

    The general continuation-state root was subsequently hydrated with a few
    invalid placeholder parquet partitions.  It is not an admissible source
    for this matched August comparison.  The canonical parent candidate
    receipt already contains the exact source-aligned entry price for every
    selected August identity, so it is both narrower and the authoritative
    source for this four-arm policy comparison.
    """
    price_source = pd.read_parquet(
        PARENT_RECEIPT / "C0_parent_rich_policy__august_holdout_candidates.parquet",
        columns=["candidate_id", "entry_price"],
    ).copy()
    price_source["candidate_id"] = price_source["candidate_id"].astype(str)
    if price_source.candidate_id.duplicated().any() or price_source.entry_price.isna().any():
        raise AssertionError("canonical parent receipt lacks one immutable entry price per candidate")
    if set(detail.candidate_id.astype(str)).difference(price_source.candidate_id):
        raise AssertionError("gradual outcome has an identity absent from the canonical parent receipt")
    priorities = stable.port._bcf_priority()
    tagged = detail.copy()
    tagged["arm"] = arm
    tagged["mc1_threshold_bps"] = 30.0
    candidates = stable.port._candidate_table(tagged, price_source, priorities)
    params = stable.portfolio_params()
    decisions, equity, _ = stable.replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=stable.CAUSAL_AUCTION_CURVE,
        market_mode="perp",
    )
    decisions = stable.port._attach_ids(decisions, candidates)
    accepted = decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
    candidates.to_parquet(output / f"{arm}_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(output / f"{arm}_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(output / f"{arm}_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(output / f"{arm}_equity.parquet", index=False, compression="zstd")
    returns = pd.to_numeric(
        candidates.iloc[pd.to_numeric(accepted.candidate_index, errors="raise").astype(int).to_numpy()].net_return,
        errors="raise",
    ) * 10_000.0
    result = {
        "arm": arm,
        "routed_candidates": len(candidates),
        "portfolio_accepted": len(accepted),
        "policy_net_bps_per_trade": float(returns.mean()) if len(returns) else np.nan,
        "total_policy_net_bps": float(returns.sum()),
        **stable.compute_replay_metrics(candidates, decisions, equity, params=params),
        "entry_price_source": "canonical_parent_august_candidate_receipt",
    }
    result["model_arm"] = arm
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    selected = _selected_ids(ENTRY_SELECTION)
    panel = study._load_panel(study.TARGET_PANEL, study.VWAP_PANEL)
    features = h4._features_by_month(FEATURE_STUDY / "stable_selected_features.parquet", "C4_normalized_vwap_fs", (HELD,))[HELD]
    reserve_start = HELD - pd.Timedelta(days=28)
    fit_start = HELD - pd.DateOffset(months=4)
    eligible = pd.to_numeric(panel["MC1_expected_bps"], errors="coerce").ge(30.0)
    fit = panel.loc[eligible & panel.entry_decision_ts.ge(fit_start) & panel.entry_decision_ts.lt(reserve_start) & panel.policy_label_available_ts.lt(reserve_start)].copy()
    reserve = panel.loc[eligible & panel.entry_decision_ts.ge(reserve_start) & panel.entry_decision_ts.lt(HELD) & panel.policy_label_available_ts.lt(HELD)].copy()
    end = HELD + pd.offsets.MonthBegin(1)
    test = panel.loc[panel.entry_decision_ts.ge(HELD) & panel.entry_decision_ts.lt(end) & panel.candidate_id.astype(str).isin(selected)].copy()
    missing = set(features).difference(fit.columns) | set(features).difference(reserve.columns) | set(features).difference(test.columns)
    if missing or fit.candidate_id.nunique() < 100 or reserve.candidate_id.nunique() < 50 or test.empty:
        raise RuntimeError(f"incomplete August gradual fold: fields={sorted(missing)}")
    model = h4._fit(fit, features, SPEC)
    reserve_raw = model.booster_.predict(reserve.loc[:, list(features)].to_numpy(float))
    reserve_y = pd.to_numeric(reserve["activation50_advantage_bps"], errors="raise").to_numpy(float)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(reserve_raw, (reserve_y > 0.0).astype(float))
    pd.DataFrame({"raw": reserve_raw, "target_positive": reserve_y > 0.0, "probability": calibrator.predict(reserve_raw)}).to_parquet(output / "calibration_reserve.parquet", index=False)

    params, median, _ = c1.base._load_policy()
    bars_cache: dict[str, c1.CompactBars | None] = {}
    failures: list[dict[str, str]] = []
    neutral_rows: list[dict[str, object]] = []
    gradual_rows: list[dict[str, object]] = []
    for symbol, rows in test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable").groupby("__symbol__", sort=True):
        bars = _load_bars(str(symbol), bars_cache, failures)
        if bars is None:
            continue
        for _, candidate in rows.groupby("candidate_id", sort=True):
            neutral = _replay_candidate(candidate, bars=bars, model=model, calibrator=calibrator, features=features, params=params, median=median, gradual=False)
            gradual = _replay_candidate(candidate, bars=bars, model=model, calibrator=calibrator, features=features, params=params, median=median, gradual=True)
            if neutral is not None and gradual is not None:
                neutral_rows.append(neutral)
                gradual_rows.append(gradual)
    neutral_frame, gradual_frame = pd.DataFrame(neutral_rows), pd.DataFrame(gradual_rows)
    if neutral_frame.empty or neutral_frame.candidate_id.duplicated().any() or not neutral_frame.candidate_id.equals(gradual_frame.candidate_id):
        raise AssertionError("neutral/gradual August candidate identity failure")
    neutral_frame.to_parquet(output / "C0_gradual_adapter_neutral_entry_outcomes.parquet", index=False, compression="zstd")
    gradual_frame.to_parquet(output / "gradual_activation_giveback_entry_outcomes.parquet", index=False, compression="zstd")
    # Every policy arm is then replayed on the *same* source-valid population
    # as the gradual arm.  This avoids presenting an incidental source-archive
    # difference (seven CRCLX paths) as an exit-policy difference.
    matched_ids = set(neutral_frame.candidate_id.astype(str))
    canonical_details: dict[str, pd.DataFrame] = {}
    for canonical_arm in ("C0_parent_rich_policy", "H4_control_activation50", "H4_giveback20"):
        detail = pd.read_parquet(PARENT_RECEIPT / f"{canonical_arm}_entry_outcomes.parquet")
        detail["candidate_id"] = detail.candidate_id.astype(str)
        detail["entry_decision_ts"] = pd.to_datetime(detail.entry_decision_ts, utc=True, errors="raise")
        detail = detail.loc[detail.entry_decision_ts.ge(HELD) & detail.candidate_id.isin(matched_ids)].copy()
        if set(detail.candidate_id) != matched_ids:
            raise AssertionError(f"{canonical_arm}: canonical outcome population does not match gradual source-valid IDs")
        canonical_details[canonical_arm] = detail

    matched_metrics = []
    for canonical_arm, detail in canonical_details.items():
        metrics = _metrics(detail, f"{canonical_arm}__matched_source_valid", output)
        metrics["model_arm"] = canonical_arm
        matched_metrics.append(metrics)
    neutral_metrics = _metrics(neutral_frame, "C0_gradual_adapter_neutral__matched_source_valid", output)
    neutral_metrics["model_arm"] = "C0_gradual_adapter_neutral"
    gradual_metrics = _metrics(gradual_frame, "gradual_activation_giveback__matched_source_valid", output)
    gradual_metrics["model_arm"] = "gradual_activation_giveback"
    summary = pd.DataFrame([*matched_metrics, neutral_metrics, gradual_metrics])
    reference = summary.loc[summary.model_arm.eq("C0_parent_rich_policy")].iloc[0]
    for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "sortino"):
        summary[f"delta_vs_parent_{metric}"] = summary[metric] - reference[metric]
    summary.to_parquet(output / "august_matched_comparison_summary.parquet", index=False)
    # Exact parent parity against canonical C0 means the new adapter can be
    # treated as a fourth arm, not an independent policy implementation.
    canonical_parent = canonical_details["C0_parent_rich_policy"]
    joined = neutral_frame.merge(canonical_parent[["candidate_id", "c1_net_bps", "c1_exit_bar", "c1_exit_reason"]], on="candidate_id", suffixes=("_new", "_canonical"), validate="one_to_one")
    receipt = {
        "candidate_count": int(len(joined)),
        "max_abs_net_bps_delta": float(np.abs(joined.c1_net_bps_new - joined.c1_net_bps_canonical).max()),
        "exit_bar_differences": int((joined.c1_exit_bar_new != joined.c1_exit_bar_canonical).sum()),
        "exit_reason_differences": int((joined.c1_exit_reason_new != joined.c1_exit_reason_canonical).sum()),
        "result": "pass" if np.abs(joined.c1_net_bps_new - joined.c1_net_bps_canonical).max() == 0.0 and (joined.c1_exit_bar_new == joined.c1_exit_bar_canonical).all() and (joined.c1_exit_reason_new == joined.c1_exit_reason_canonical).all() else "fail",
    }
    (output / "neutral_parent_parity.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if receipt["result"] != "pass":
        raise RuntimeError("gradual adapter neutral arm does not reproduce canonical parent")
    (output / "run_manifest.json").write_text(json.dumps({
        "scope": "offline August holdout only; no exchange or live mutation",
        "entry_selection": str(ENTRY_SELECTION), "entry_selection_sha256": _sha256(ENTRY_SELECTION),
        "parent_receipt": str(PARENT_RECEIPT),
        "held_month": "2026-08", "fit_window": [str(fit_start), str(reserve_start)], "reserve_window": [str(reserve_start), str(HELD)],
        "calibrator": "isotonic P(activation50_advantage_bps > 0) fitted only on preceding resolved 28-day reserve",
        "gradual_params": {"activation": {"threshold": .20, "strength": 3.0, "power": 2.0, "mode": "both", "unit": .25}, "giveback": {"threshold": .20, "strength": 3.0, "power": .50, "mode": "both", "unit": .10}, "stop": "unchanged; 5% max loss cap enforced"},
        "h4_spec": SPEC, "features_sha256": _sha256(FEATURE_STUDY / "stable_selected_features.parquet"),
        "source_failures": failures,
    }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        argv = sys.argv[1:]
        if "--output" in argv:
            output = Path(argv[argv.index("--output") + 1]).resolve()
            if output.exists():
                (output / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
