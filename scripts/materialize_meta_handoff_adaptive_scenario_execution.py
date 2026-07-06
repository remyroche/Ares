#!/usr/bin/env python3
"""Materialize an adaptive-scenario guarded handoff for frozen replay.

Input is a passing `validate_meta_handoff_adaptive_scenario_guard.py` report.
The output mirrors the fixed-scenario guarded handoff:

- clean decision-time handoff without realized outcome columns;
- offline replay candidate parquet with outcomes, for verification only;
- frozen prior-fold EV-curve training candidates, so replay does not reselect
  scenarios or refit the scenario guard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_optimiser import _json_safe  # noqa: E402
from scripts.materialize_meta_handoff_guarded_execution import (  # noqa: E402
    FUTURE_OR_LABEL_TOKENS,
    _decision_time_columns,
    _feature_audit_rows,
    _hash_payload,
    _sha256_path,
    _thresholds_from_args,
)


DEFAULT_DIAGNOSTIC_DIR = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_v3"
)
DEFAULT_OUT_DIR = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_materialized_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _require_path(value: str | Path | None, *, fallback: Path, label: str) -> Path:
    path = Path(value) if value else fallback
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _forbidden_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if any(token in col for token in FUTURE_OR_LABEL_TOKENS)]


def _with_decision_contract(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["signal_timestamp"] = pd.to_datetime(out.get("timestamp"), utc=True, errors="coerce")
    if "delayed_entry_effective_ts" in out.columns:
        decision_ts = pd.to_datetime(out["delayed_entry_effective_ts"], utc=True, errors="coerce")
    else:
        decision_ts = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    has_delayed_observation = decision_ts.notna()
    out["requires_delayed_entry_observation"] = has_delayed_observation
    out["guard_decision_timestamp"] = decision_ts.fillna(out["signal_timestamp"])
    out["guard_decision_contract"] = np.where(
        has_delayed_observation.to_numpy(dtype=bool),
        "post_delayed_entry_window",
        "signal_time_no_delay_window",
    )
    return out


def _add_handoff_metadata(
    frame: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    feature_set_hash: str,
    source_hash: str,
) -> pd.DataFrame:
    out = _with_decision_contract(frame)
    out["base_selector"] = out.get("strategy_id", "")
    out["base_score_oof"] = pd.to_numeric(out.get("rank_pct", np.nan), errors="coerce")
    out["meta_score_oof"] = pd.to_numeric(out.get("calibrated_score", np.nan), errors="coerce")
    out["scenario_id"] = out.get("scenario", "adaptive_scenario")
    out["scenario_family"] = "adaptive_scenario"
    out["horizon_bars"] = pd.to_numeric(out.get("path_len", np.nan), errors="coerce")
    out["barrier_mult"] = pd.to_numeric(out.get("barrier_multiplier", np.nan), errors="coerce")
    out["stop_mult"] = pd.to_numeric(out.get("policy_sl_mult", np.nan), errors="coerce")
    out["accepted"] = True
    out["guard_accepted"] = True
    out["exec_guard_score_oof"] = pd.to_numeric(
        out.get("adaptive_guard_score_oof", np.nan),
        errors="coerce",
    )
    out["exec_guard_threshold"] = pd.to_numeric(
        out.get("adaptive_guard_threshold", np.nan),
        errors="coerce",
    )
    out["exec_guard_keep_frac"] = pd.to_numeric(
        out.get("adaptive_guard_keep_frac", np.nan),
        errors="coerce",
    )
    out["exec_guard_method"] = str(manifest.get("method") or "")
    out["exec_guard_model_name"] = out.get("adaptive_guard_model_name", "")
    out["exec_guard_feature_set_hash"] = feature_set_hash
    out["exec_guard_model_hash"] = [
        _hash_payload(
            {
                "adaptive_manifest": manifest.get("outputs", {}).get("manifest"),
                "source_hash": source_hash,
                "validation_week": week,
                "base_opportunity_key": key,
                "scenario": scenario,
                "score": score,
                "threshold": threshold,
            }
        )
        for week, key, scenario, score, threshold in zip(
            out.get("validation_week", pd.Series("", index=out.index)).astype(str),
            out.get("base_opportunity_key", pd.Series("", index=out.index)).astype(str),
            out.get("scenario", pd.Series("", index=out.index)).astype(str),
            out["exec_guard_score_oof"],
            out["exec_guard_threshold"],
        )
    ]
    out["threshold_source"] = "prior_train_keep_fraction_selection"
    out["decision_fold"] = pd.to_numeric(out.get("decision_fold", np.nan), errors="coerce")
    if out["decision_fold"].isna().all() and "validation_week" in out.columns:
        week_order = {week: idx for idx, week in enumerate(sorted(out["validation_week"].astype(str).unique()))}
        out["decision_fold"] = out["validation_week"].astype(str).map(week_order).astype("int16")
    out["train_window_end"] = pd.to_datetime(
        out.get("validation_week", pd.Series("", index=out.index)).astype(str),
        utc=True,
        errors="coerce",
    ).map(lambda ts: ts.isoformat() if pd.notna(ts) else "")
    return out


def _handoff_columns(frame: pd.DataFrame) -> list[str]:
    adaptive_cols = [
        "scenario_family",
        "base_opportunity_key",
        "adaptive_guard_score_oof",
        "adaptive_guard_threshold",
        "adaptive_guard_margin",
        "adaptive_guard_keep_frac",
        "adaptive_guard_keep",
        "adaptive_guard_model_name",
        "adaptive_guard_method",
        "base_selector",
        "base_score_oof",
        "meta_score_oof",
        "scenario_id",
        "horizon_bars",
        "barrier_mult",
        "stop_mult",
        "accepted",
        "guard_accepted",
        "exec_guard_score_oof",
        "exec_guard_threshold",
        "exec_guard_keep_frac",
        "exec_guard_method",
        "exec_guard_model_name",
        "exec_guard_feature_set_hash",
        "exec_guard_model_hash",
        "threshold_source",
        "decision_fold",
        "validation_week",
        "train_window_end",
    ]
    return [
        col
        for col in dict.fromkeys(_decision_time_columns(frame) + adaptive_cols)
        if col in frame.columns and not any(token in col for token in FUTURE_OR_LABEL_TOKENS)
    ]


def _timing_audit(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "missing_guard_decision_timestamp_rows": 0,
            "missing_required_delayed_entry_timestamp_rows": 0,
            "guard_decision_before_signal_rows": 0,
            "guard_decision_equals_signal_rows": 0,
            "delayed_observation_decision_rows": 0,
            "signal_time_no_delay_rows": 0,
        }
    signal_ts = pd.to_datetime(frame["signal_timestamp"], utc=True, errors="coerce")
    decision_ts = pd.to_datetime(frame["guard_decision_timestamp"], utc=True, errors="coerce")
    requires_delay = frame["requires_delayed_entry_observation"].fillna(False).astype(bool)
    return {
        "missing_guard_decision_timestamp_rows": int(decision_ts.isna().sum()),
        "missing_required_delayed_entry_timestamp_rows": int((requires_delay & decision_ts.isna()).sum()),
        "guard_decision_before_signal_rows": int((decision_ts < signal_ts).fillna(False).sum()),
        "guard_decision_equals_signal_rows": int((decision_ts == signal_ts).fillna(False).sum()),
        "delayed_observation_decision_rows": int(requires_delay.sum()),
        "signal_time_no_delay_rows": int((~requires_delay).sum()),
    }


def _duplicate_count(frame: pd.DataFrame) -> int:
    keys = [col for col in ["timestamp", "symbol", "side", "strategy_id", "scenario_id"] if col in frame.columns]
    return int(frame.duplicated(keys).sum()) if keys else 0


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.6f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-net-pnl", type=float, default=0.0)
    parser.add_argument("--min-mean-objective", type=float, default=0.0)
    parser.add_argument("--min-positive-fold-share", type=float, default=0.67)
    parser.add_argument("--max-full-sl-rate", type=float, default=0.22)
    parser.add_argument("--max-timeout-rate", type=float, default=0.55)
    parser.add_argument("--min-worst-fold-net-pnl", type=float, default=-250.0)
    parser.add_argument("--max-no-trade-folds", type=int, default=0)
    args = parser.parse_args()

    diagnostic_dir = Path(args.diagnostic_dir)
    diagnostic_manifest_path = diagnostic_dir / "manifest.json"
    diagnostic_manifest = _read_json(diagnostic_manifest_path)
    outputs = diagnostic_manifest.get("outputs") if isinstance(diagnostic_manifest.get("outputs"), dict) else {}
    selected_path = _require_path(
        outputs.get("selected_candidates"),
        fallback=diagnostic_dir / "adaptive_scenario_guard_selected_candidates.parquet",
        label="adaptive selected candidates",
    )
    train_ev_path = _require_path(
        outputs.get("train_ev_candidates"),
        fallback=diagnostic_dir / "adaptive_scenario_guard_train_ev_candidates.parquet",
        label="adaptive train EV candidates",
    )
    decisions_path = _require_path(
        outputs.get("decisions"),
        fallback=diagnostic_dir / "adaptive_scenario_guard_decisions.parquet",
        label="adaptive decisions",
    )
    summary_path = _require_path(
        outputs.get("summary"),
        fallback=diagnostic_dir / "adaptive_scenario_guard_summary.csv",
        label="adaptive summary",
    )
    folds_path = _require_path(
        outputs.get("folds"),
        fallback=diagnostic_dir / "adaptive_scenario_guard_folds.csv",
        label="adaptive folds",
    )
    source_path = _require_path(
        diagnostic_manifest.get("candidates"),
        fallback=Path(""),
        label="source candidates",
    )

    selected = pd.read_parquet(selected_path)
    train_ev = pd.read_parquet(train_ev_path)
    decisions = pd.read_parquet(decisions_path)
    summary = pd.read_csv(summary_path)
    folds = pd.read_csv(folds_path)
    if not folds.empty and {"validation_week", "keep_frac"}.issubset(folds.columns):
        fold_keep = folds[["validation_week", "keep_frac", "score_threshold"]].copy()
        fold_keep["validation_week"] = fold_keep["validation_week"].astype(str)
        fold_keep = fold_keep.rename(
            columns={
                "keep_frac": "frozen_fold_keep_frac",
                "score_threshold": "frozen_fold_score_threshold",
            }
        )
        selected["validation_week"] = selected["validation_week"].astype(str)
        selected = selected.merge(fold_keep, on="validation_week", how="left")
    source_hash = _sha256_path(source_path)
    feature_columns = list(diagnostic_manifest.get("feature_columns") or [])
    feature_set_hash = _hash_payload(
        {
            "feature_mode": diagnostic_manifest.get("feature_mode"),
            "features": feature_columns,
            "selection_mode": diagnostic_manifest.get("selection_mode"),
            "anchor_scenario": diagnostic_manifest.get("anchor_scenario"),
            "require_anchor_admission": diagnostic_manifest.get("require_anchor_admission"),
            "switch_margin_buffer": diagnostic_manifest.get("switch_margin_buffer"),
        }
    )

    materialized = _add_handoff_metadata(
        selected,
        manifest=diagnostic_manifest,
        feature_set_hash=feature_set_hash,
        source_hash=source_hash,
    )
    clean = materialized[_handoff_columns(materialized)].copy()
    feature_audit_df, feature_audit = _feature_audit_rows(feature_columns, frame=materialized)
    timing_audit = _timing_audit(clean)
    forbidden = _forbidden_columns(clean)
    duplicate_count = _duplicate_count(clean)
    train_ev_duplicate_count = int(
        train_ev.duplicated(
            [col for col in ["validation_week", "timestamp", "symbol", "side", "strategy_id"] if col in train_ev.columns]
        ).sum()
    )
    feature_audit.update(
        {
            **timing_audit,
            "guarded_candidate_rows": int(len(clean)),
            "portfolio_accepted_rows": int(decisions.get("accepted", pd.Series(False, index=decisions.index)).astype(bool).sum())
            if not decisions.empty and "accepted" in decisions.columns
            else 0,
            "duplicate_timestamp_symbol_side_strategy_scenario_rows": duplicate_count,
            "train_ev_candidate_rows": int(len(train_ev)),
            "train_ev_duplicate_decision_rows": train_ev_duplicate_count,
            "clean_handoff_forbidden_columns": forbidden,
            "scenario_selection": "frozen_from_adaptive_diagnostic",
            "scenario": "adaptive_scenario",
            "source_scenarios": list(diagnostic_manifest.get("scenarios") or []),
            "method": str(diagnostic_manifest.get("method") or ""),
            "feature_mode": str(diagnostic_manifest.get("feature_mode") or ""),
            "leakage_safe_for_frozen_replay": bool(
                not forbidden
                and duplicate_count == 0
                and train_ev_duplicate_count == 0
                and timing_audit["missing_guard_decision_timestamp_rows"] == 0
                and timing_audit["missing_required_delayed_entry_timestamp_rows"] == 0
                and timing_audit["guard_decision_before_signal_rows"] == 0
            ),
            "leakage_safe_caveat": (
                "Guard is decision-time safe. Rows with delayed-entry observations must be evaluated "
                "after that observation window; signal-time fallback rows have no delayed-entry observation."
            ),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "guarded_candidates": args.out_dir / "meta_handoff_adaptive_guarded_candidates.parquet",
        "offline_replay_candidates": args.out_dir / "meta_handoff_adaptive_guarded_offline_replay_candidates.parquet",
        "ev_curve_train_candidates": args.out_dir / "meta_handoff_adaptive_guarded_train_ev_candidates.parquet",
        "execution_plan": args.out_dir / "meta_handoff_adaptive_guarded_execution_plan.csv",
        "summary": args.out_dir / "meta_handoff_adaptive_guarded_summary.csv",
        "feature_audit": args.out_dir / "meta_handoff_adaptive_guarded_feature_audit.csv",
        "leakage_audit": args.out_dir / "meta_handoff_adaptive_guarded_leakage_audit.json",
        "manifest": args.out_dir / "meta_handoff_adaptive_guarded_manifest.json",
        "report": args.out_dir / "meta_handoff_adaptive_guarded_report.md",
    }
    clean.to_parquet(paths["guarded_candidates"], index=False)
    materialized.to_parquet(paths["offline_replay_candidates"], index=False)
    train_ev.to_parquet(paths["ev_curve_train_candidates"], index=False)
    decisions.to_csv(paths["execution_plan"], index=False)
    summary.to_csv(paths["summary"], index=False)
    feature_audit_df.to_csv(paths["feature_audit"], index=False)
    paths["leakage_audit"].write_text(
        json.dumps(_json_safe(feature_audit), indent=2),
        encoding="utf-8",
    )

    manifest = {
        "generated_by": "materialize_meta_handoff_adaptive_scenario_execution",
        "diagnostic_manifest": str(diagnostic_manifest_path),
        "diagnostic_dir": str(diagnostic_dir),
        "source_candidates": str(source_path),
        "source_candidates_sha256": source_hash,
        "out_dir": str(args.out_dir),
        "scenario": "adaptive_scenario",
        "source_scenarios": list(diagnostic_manifest.get("scenarios") or []),
        "method": str(diagnostic_manifest.get("method") or ""),
        "selection_mode": str(diagnostic_manifest.get("selection_mode") or ""),
        "anchor_scenario": str(diagnostic_manifest.get("anchor_scenario") or ""),
        "switch_margin_buffer": float(diagnostic_manifest.get("switch_margin_buffer", 0.0) or 0.0),
        "require_anchor_admission": bool(diagnostic_manifest.get("require_anchor_admission")),
        "feature_mode": str(diagnostic_manifest.get("feature_mode") or ""),
        "feature_columns": feature_columns,
        "feature_set_hash": feature_set_hash,
        "thresholds": _thresholds_from_args(args),
        "market_mode": "perps",
        "global_threshold_floor": 0.0,
        "gate_summary": summary.to_dict(orient="records"),
        "leakage_audit": feature_audit,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Adaptive Scenario Guarded Handoff Materialization",
        "",
        "This materializes a passing anchored adaptive-scenario diagnostic into a clean handoff and offline replay package. Scenario choices and EV-training rows are frozen from the diagnostic output.",
        "",
        "## Summary",
        "",
        _fmt_table(
            summary,
            [
                "scenario",
                "variant",
                "pass_simple_policy_gate",
                "sum_net_pnl",
                "mean_objective",
                "worst_fold_net_pnl",
                "positive_fold_share",
                "accepted_trades",
                "weighted_full_sl_rate",
                "weighted_timeout_rate",
            ],
        ),
        "",
        "## Leakage Audit",
        "",
        _fmt_table(
            pd.DataFrame([feature_audit]),
            [
                "leakage_safe_for_frozen_replay",
                "all_guard_inputs_signal_time_safe",
                "all_guard_inputs_decision_time_safe",
                "guarded_candidate_rows",
                "portfolio_accepted_rows",
                "train_ev_candidate_rows",
                "duplicate_timestamp_symbol_side_strategy_scenario_rows",
                "train_ev_duplicate_decision_rows",
                "delayed_observation_decision_rows",
                "signal_time_no_delay_rows",
            ],
        ),
    ]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            _json_safe(
                {
                    "summary": summary.to_dict(orient="records"),
                    "leakage_audit": feature_audit,
                    "outputs": {key: str(value) for key, value in paths.items()},
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
