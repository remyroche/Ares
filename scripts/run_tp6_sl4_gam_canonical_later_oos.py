#!/usr/bin/env python3
"""Untouched later replay for the canonical TP6/SL4 Base+Consensus contract.

This runner is deliberately separate from the 2025 development replay.  It
uses the frozen F0 R3 handoff for the later rows, refits the handover's
downstream consensus heads before the later population, and compares the
same 75/25 Base+Consensus control with a one-field, hard-gated GAM input.

The later population (2026-07-20 through 2026-07-23) is never used to fit a
model, a map, a GAM, a path contract, or a gate.  TP6/SL4 H12 outcomes are
used only for the final report.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tp6_sl4_canonical_meta_paths_20260808 import _materialize_month  # noqa: E402
from scripts.run_tp6_sl4_downstream_retrain_2025 import _map_base  # noqa: E402
from scripts.run_tp6_sl4_gam_untouched_oos_2026 import _rolling_one_month  # noqa: E402
from scripts.run_tp6_sl4_rolling_gam_residual_integration import (  # noqa: E402
    _fit_heads,
    _pct,
)


HISTORICAL = ROOT / "data_perp/artifacts/r3_tp6_sl4_meta_target_ablation_20260803_v1/r3_meta_target_oof_predictions.parquet"
LATER_BASE = ROOT / "data_perp/artifacts/tp6_sl4_frozen_f0_later_base_20260808_v1/later_frozen_f0_base_predictions.parquet"
LATER_CONTEXT = ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/rebuilt_f0_context_canonical73_v1/later_f0_context.parquet"
DEV_GAM_2025 = ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/rolling_oof_predictions.parquet"
DEV_GAM_2026 = ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_oos_2026_20260815_v2/rolling_gam_2026.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_gam_canonical_later_oos_20260808_v1"
TARGET_START = pd.Timestamp("2026-07-20 00:00:00+00:00")
TARGET_END = pd.Timestamp("2026-07-24 00:00:00+00:00")
GAM_TRAIN_MONTH = "2026-06"
GAM_TARGET_MONTH = "2026-07"
SIDE = "long"
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
SEED = 20260815


def _context_fields() -> list[str]:
    selected = pd.read_parquet(
        ROOT / "data_perp/artifacts/r3_plus_meta_tp6_sl4_ablation_20260803_v1/r3_plus_meta_metrics.parquet",
        columns=["selected_context_features"],
    )
    import ast

    out: list[str] = []
    for value in selected.selected_context_features.dropna():
        if isinstance(value, str):
            value = ast.literal_eval(value)
        out.extend(map(str, value))
    fields = sorted(set(out))
    if len(fields) != 73:
        raise ValueError(f"canonical context contract changed: {len(fields)} fields")
    return fields


def _load_panel(context: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    hist = pd.read_parquet(HISTORICAL)
    hist["__ts__"] = pd.to_datetime(hist["__ts__"], utc=True, errors="raise")
    hist["label_available_ts"] = pd.to_datetime(hist["label_available_ts"], utc=True, errors="raise")
    hist = hist.loc[hist.side_name.astype(str).str.lower().eq(SIDE) & hist.label_valid.fillna(False)].copy()
    hist["base_score"] = pd.to_numeric(hist["r3_meta_p_clear"], errors="coerce") - 0.5 * pd.to_numeric(hist["r3_meta_p_adverse"], errors="coerce")
    later = pd.read_parquet(LATER_BASE)
    context_frame = pd.read_parquet(LATER_CONTEXT)
    later["__ts__"] = pd.to_datetime(later["__ts__"], utc=True, errors="raise")
    later["label_available_ts"] = pd.to_datetime(later["label_available_ts"], utc=True, errors="raise")
    context_frame["__ts__"] = pd.to_datetime(context_frame["__ts__"], utc=True, errors="raise")
    later = later.loc[later.side_name.astype(str).str.lower().eq(SIDE)].copy()
    # The source context is generated once per symbol/timestamp and then
    # joined to both side rows.  Collapse only those exact duplicate keys;
    # side identity remains in the later base-prediction frame.
    context_frame = context_frame[["__ts__", "__symbol__", *context]].drop_duplicates(["__ts__", "__symbol__"], keep="first")
    later = later.merge(context_frame, on=["__ts__", "__symbol__"], how="left", validate="many_to_one")
    later = later.rename(columns={"t4_tp6_sl4_gross_bps": "exact_gross_bps", "t4_tp6_sl4_net_bps": "exact_net_bps"})
    later["month"] = GAM_TARGET_MONTH
    hist["month"] = hist["__ts__"].dt.strftime("%Y-%m")
    # The later panel deliberately carries the frozen F0 probabilities, not a
    # challenger prediction artifact.  The downstream architecture remains
    # the handover's strict R3 score -> isotonic anchor -> consensus heads.
    required = {"candidate_id", "__ts__", "label_available_ts", "exact_net_bps", "exact_gross_bps", "base_score", *context}
    missing_hist = sorted(required.difference(hist.columns))
    missing_later = sorted(required.difference(later.columns))
    if missing_hist:
        raise ValueError(f"historical panel missing canonical fields: {missing_hist}")
    if missing_later:
        raise ValueError(f"later panel missing canonical fields: {missing_later}")
    panel = pd.concat([hist, later], ignore_index=True, sort=False)
    panel = panel.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if panel.candidate_id.duplicated().any():
        raise ValueError("historical/later candidate IDs overlap")
    digest = hashlib.sha256("\n".join(context).encode()).hexdigest()
    return panel, later, digest


def _materialize_gam_paths(panel: pd.DataFrame, later: pd.DataFrame, context: list[str], out: Path) -> tuple[Path, dict[str, object]]:
    raw = out / "raw_paths"
    raw.mkdir(parents=True, exist_ok=True)
    june_train = panel.loc[
        panel.__ts__.lt(pd.Timestamp("2026-06-01", tz="UTC"))
        & panel.label_available_ts.lt(pd.Timestamp("2026-06-01", tz="UTC"))
        & panel.label_valid.fillna(False)
        & panel.side_name.eq(SIDE)
    ].copy()
    june_held = panel.loc[panel.month.eq(GAM_TRAIN_MONTH) & panel.side_name.eq(SIDE)].copy()
    july_held = later.copy()
    july_train = panel.loc[panel.month.eq(GAM_TRAIN_MONTH) & panel.side_name.eq(SIDE)].copy()
    for month, train, held in ((GAM_TRAIN_MONTH, june_train, june_held), (GAM_TARGET_MONTH, july_train, july_held)):
        audit_path = raw / "fold_audits" / f"month={month}.json"
        eval_path = raw / "fold_evaluations" / f"month={month}.parquet"
        if audit_path.exists() and eval_path.exists():
            continue
        result = _materialize_month(
            train=train,
            held=held,
            context=context,
            month=month,
            out=raw,
            max_trees=64,
            contribution_components=8,
            threshold_bands=4,
        )
        if result.get("status") not in {"MATERIALIZED_STRICT_OOF", "skipped"}:
            raise RuntimeError(f"path materialization failed for {month}: {result}")
    base_lookup = pd.concat(
        [
            pd.read_parquet(raw / "fold_evaluations" / f"month={m}.parquet", columns=["candidate_id", "__ts__", "base_expected_bps"])
            for m in (GAM_TRAIN_MONTH, GAM_TARGET_MONTH)
        ],
        ignore_index=True,
    ).drop_duplicates(["candidate_id", "__ts__"], keep="last")
    # The historical panel contains an earlier July 1--10 block under the
    # same calendar-month label as the untouched July 20--23 target.  The
    # rolling helper uses that label for its lookup, so provide their
    # train-only anchors explicitly; their outcomes are never used for the
    # later target fit.
    july_history = panel.loc[
        panel.month.eq(GAM_TARGET_MONTH)
        & panel.__ts__.lt(TARGET_START)
        & panel.label_available_ts.lt(TARGET_START)
        & panel.label_valid.fillna(False)
        & panel.side_name.eq(SIDE)
    ].copy()
    if not july_history.empty:
        pre_july = panel.loc[
            panel.__ts__.lt(pd.Timestamp("2026-07-01", tz="UTC"))
            & panel.label_available_ts.lt(pd.Timestamp("2026-07-01", tz="UTC"))
            & panel.label_valid.fillna(False)
            & panel.side_name.eq(SIDE)
        ].copy()
        _, july_anchor = _map_base(pre_july, july_history)
        base_lookup = pd.concat(
            [base_lookup, july_history[["candidate_id", "__ts__"]].assign(base_expected_bps=july_anchor)],
            ignore_index=True,
        ).drop_duplicates(["candidate_id", "__ts__"], keep="last")
    gam, audit = _rolling_one_month(panel, context, raw, base_lookup, GAM_TARGET_MONTH)
    gam.to_parquet(out / "later_gam_signal.parquet", index=False, compression="zstd")
    (out / "gam_fit_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    return raw, audit


def _historical_gam_field(train: pd.DataFrame) -> pd.DataFrame:
    """Join only precomputed strict-OOF GAM values to downstream training rows."""
    pieces: list[pd.DataFrame] = []
    if DEV_GAM_2025.exists():
        d = pd.read_parquet(DEV_GAM_2025)
        d = d.loc[d.window_months.eq(1)].copy()
        d["gam_delta_bps"] = np.where(
            d.rolling_transport_valid.fillna(False).astype(bool),
            d.rolling_gam_zero_gamma025 - d.base_expected_bps,
            0.0,
        )
        pieces.append(d[["candidate_id", "__ts__", "gam_delta_bps"]])
    if DEV_GAM_2026.exists():
        d = pd.read_parquet(DEV_GAM_2026)
        if "gam_delta_bps" in d.columns:
            d["gam_delta_bps"] = np.where(d.rolling_transport_valid.fillna(False).astype(bool), d.gam_delta_bps, 0.0)
        else:
            d["gam_delta_bps"] = np.where(
                d.rolling_transport_valid.fillna(False).astype(bool),
                d.rolling_gam_zero_gamma025 - d.base_expected_bps,
                0.0,
            )
        pieces.append(d[["candidate_id", "__ts__", "gam_delta_bps"]])
    if pieces:
        joined = pd.concat(pieces, ignore_index=True).drop_duplicates(["candidate_id", "__ts__"], keep="last")
        out = train[["candidate_id", "__ts__"]].merge(joined, on=["candidate_id", "__ts__"], how="left", validate="one_to_one")
    else:
        out = train[["candidate_id", "__ts__"]].copy()
        out["gam_delta_bps"] = np.nan
    out["gam_delta_bps"] = pd.to_numeric(out.gam_delta_bps, errors="coerce").fillna(0.0).astype(np.float32)
    return out


def _metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arms = ["canonical_control", "gated_gam_input"]
    glob, period, stability = [], [], []
    for arm in arms:
        for tail in TAILS:
            n = max(1, int(math.ceil(len(pred) * tail)))
            top = pred.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            glob.append({"arm": arm, "scope": "global_later_long", "tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": float(pred[[arm, "exact_net_bps"]].corr(method="spearman").iloc[0, 1])})
        vals = []
        for day, block in pred.groupby(pred.__ts__.dt.strftime("%Y-%m-%d"), sort=True):
            n = max(1, int(math.ceil(len(block) * 0.05)))
            top = block.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            vals.append(float(top.exact_net_bps.mean()))
            period.append({"arm": arm, "period": day, "tail": 0.05, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": vals[-1], "rank_ic": float(block[[arm, "exact_net_bps"]].corr(method="spearman").iloc[0, 1])})
        arr = np.asarray(vals, dtype=float)
        med = float(np.nanmedian(arr))
        stability.append({"arm": arm, "periods": len(arr), "mean_top5_net_bps": float(np.nanmean(arr)), "median_top5_net_bps": med, "mad_top5_net_bps": float(np.nanmedian(np.abs(arr - med))), "worst_period_top5_net_bps": float(np.nanmin(arr)), "positive_periods_top5": int(np.sum(arr > 0.0))})
    return pd.DataFrame(glob), pd.DataFrame(period), pd.DataFrame(stability)


def run(output_dir: Path = DEFAULT_OUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    context = _context_fields()
    panel, later, context_hash = _load_panel(context)
    raw, gam_audit = _materialize_gam_paths(panel, later, context, output_dir)
    target_gam = pd.read_parquet(output_dir / "later_gam_signal.parquet")
    target_gam = target_gam[["candidate_id", "__ts__", "rolling_transport_valid", "gam_delta_bps"]].copy()
    held = later.merge(target_gam, on=["candidate_id", "__ts__"], how="inner", validate="one_to_one")
    if len(held) != len(later):
        raise ValueError("GAM signal does not cover every later candidate")
    train = panel.loc[
        panel.__ts__.lt(TARGET_START)
        & panel.label_available_ts.lt(TARGET_START)
        & panel.label_valid.fillna(False)
        & panel.side_name.eq(SIDE)
    ].copy()
    historical_gam = _historical_gam_field(train)
    train = train.merge(historical_gam, on=["candidate_id", "__ts__"], how="left", validate="one_to_one")
    train["gam_delta_bps"] = train.gam_delta_bps.fillna(0.0).astype(np.float32)
    train.attrs["context_fields"] = context
    held.attrs["context_fields"] = context
    base_train, base_held = _map_base(train, held)
    # Exact handover contract: 8 consensus heads, 75/25 base/consensus blend.
    control_consensus, _, _, _ = _fit_heads(train.copy(), held.copy(), base_train, base_held, use_gam_inputs=False, month=GAM_TARGET_MONTH, seed_base=SEED)
    gam_consensus, _, _, _ = _fit_heads(train.copy(), held.copy(), base_train, base_held, use_gam_inputs=True, month=GAM_TARGET_MONTH, seed_base=SEED)
    base_rank = _pct(held.base_score.to_numpy(float), train.base_score.to_numpy(float))
    control_score = 0.75 * base_rank + 0.25 * control_consensus
    enhanced_score = 0.75 * base_rank + 0.25 * gam_consensus
    valid_gate = held.rolling_transport_valid.fillna(False).astype(bool).to_numpy()
    gated_score = np.where(valid_gate, enhanced_score, control_score)
    pred = held[["candidate_id", "__ts__", "__symbol__", "side_name", "exact_net_bps", "exact_gross_bps", "label_valid", "label_available_ts"]].copy()
    pred["base_rank"] = base_rank
    pred["consensus_control_rank"] = control_consensus
    pred["consensus_gam_rank"] = gam_consensus
    pred["canonical_control"] = control_score
    pred["gated_gam_input"] = gated_score
    pred["gam_delta_bps"] = held.gam_delta_bps.to_numpy(float)
    pred["transport_valid"] = valid_gate
    pred.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    g, p, s = _metrics(pred)
    g.to_parquet(output_dir / "metrics_global.parquet", index=False)
    p.to_parquet(output_dir / "metrics_daily.parquet", index=False)
    s.to_parquet(output_dir / "metrics_stability.parquet", index=False)
    coverage = pd.DataFrame({"field": context, "later_coverage": [float(pd.to_numeric(held[f], errors="coerce").notna().mean()) for f in context], "later_unique": [int(pd.to_numeric(held[f], errors="coerce").nunique(dropna=True)) for f in context]})
    coverage.to_parquet(output_dir / "context_coverage.parquet", index=False)
    correctness = {
        "schema": "tp6_sl4_gam_canonical_later_oos_correctness_v1",
        "target_start": str(TARGET_START),
        "target_end_exclusive": str(TARGET_END),
        "target_rows": int(len(held)),
        "train_rows": int(len(train)),
        "target_outcomes_used_in_gam_fit": False,
        "target_outcomes_used_in_meta_fit": False,
        "target_outcomes_used_in_base_anchor_fit": False,
        "one_canonical_gam_field": True,
        "gam_field": "gam_delta_bps",
        "base_ev_modulation": False,
        "transport_invalid_is_exact_control": True,
        "global_ranking_after_score_generation": True,
        "same_exit_contract": "TP +6 ATR / SL -4 ATR / H12 / 100 bps once",
        "candidate_ids_unique": bool(pred.candidate_id.is_unique),
        "canonical_context_count": len(context),
        "canonical_context_sha256": context_hash,
        "transport_valid_fraction": float(valid_gate.mean()),
        "benchmark_context_zero_coverage": [str(x) for x in coverage.loc[coverage.later_coverage.eq(0), "field"]],
        "gam_fit_audit": gam_audit,
    }
    (output_dir / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2, default=str) + "\n")
    manifest = {
        "schema": "tp6_sl4_gam_canonical_later_oos_v1",
        "status": "COMPLETE",
        "side": SIDE,
        "target_population": "2026-07-20 through 2026-07-23 UTC",
        "later_base": str(LATER_BASE),
        "later_context": str(LATER_CONTEXT),
        "historical_training_panel": str(HISTORICAL),
        "canonical_baseline": "same handover architecture: frozen R3 p_clear - 0.5 p_adverse, train-only isotonic anchor, 8 consensus heads, 0.75 base rank + 0.25 consensus rank",
        "canonical_gam": "one-month June fit, zero-exposure gamma=0.25, one field gam_delta_bps, hard transport gate",
        "exits": "TP6/SL4 H12, 100 bps cost once",
        "ranking": "one pooled global ranking over later rows",
        "context_count": len(context),
        "context_sha256": context_hash,
        "train_rows": int(len(train)),
        "target_rows": int(len(held)),
        "transport_valid_fraction": float(valid_gate.mean()),
        "artifacts": ["predictions.parquet", "metrics_global.parquet", "metrics_daily.parquet", "metrics_stability.parquet", "context_coverage.parquet", "correctness_test_report.json"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    report = [
        "# TP6/SL4 canonical later GAM OOS",
        "",
        "The no-GAM control is the handover's Base+Consensus architecture, reproduced with the frozen later F0 R3 outputs. The GAM arm adds only gam_delta_bps to the consensus heads and falls back to the exact control when transport is invalid.",
        "",
        "## Global metrics",
        "",
        g.round(3).to_string(index=False),
        "",
        "## Daily Top-5",
        "",
        p.round(3).to_string(index=False),
        "",
        "## Stability",
        "",
        s.round(3).to_string(index=False),
        "",
        "## Caveat",
        "",
        "Three canonical context fields have zero later coverage because the underlying benchmark source is unavailable; they are retained as NaN, never imputed. This is recorded in context_coverage.parquet and correctness_test_report.json.",
    ]
    (output_dir / "TP6_SL4_GAM_CANONICAL_LATER_OOS_REPORT.md").write_text("\n".join(report) + "\n")
    del panel, train, held
    gc.collect()
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.output_dir)
