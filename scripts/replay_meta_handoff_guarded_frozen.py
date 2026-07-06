#!/usr/bin/env python3
"""Frozen replay for a materialized guarded meta handoff.

This validator consumes the output of
`materialize_meta_handoff_guarded_execution.py`. It does not refit the guard,
change keep fractions, select thresholds, or choose scenarios. It replays the
already-guarded candidate rows fold by fold, using only pre-validation source
rows to fit EV curves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
)
from extreme_price_movements.simple_policy_optimiser import _json_safe  # noqa: E402
from scripts.materialize_meta_handoff_guarded_execution import (  # noqa: E402
    FUTURE_OR_LABEL_TOKENS,
)
from scripts.validate_meta_handoff_execution_guard_walkforward import (  # noqa: E402
    _fold_result,
    _prepare_frame,
    _replay_with_train_curve,
    _summarise,
)


DEFAULT_HANDOFF_DIR = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "meta_handoff_guarded_execution_h9x4_execnet_v1"
)
DEFAULT_OUT_DIR = DEFAULT_HANDOFF_DIR / "frozen_replay_v1"
SUMMARY_COMPARE_COLUMNS = (
    "sum_net_pnl",
    "mean_objective",
    "worst_fold_net_pnl",
    "positive_folds",
    "positive_fold_share",
    "no_trade_folds",
    "accepted_trades",
    "mean_keep_frac",
    "weighted_full_sl_rate",
    "weighted_timeout_rate",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _gate_thresholds(manifest: Mapping[str, Any]) -> dict[str, float | int]:
    raw = manifest.get("thresholds") if isinstance(manifest.get("thresholds"), dict) else {}
    return {
        "min_net_pnl": float(raw.get("min_net_pnl", 0.0)),
        "min_mean_objective": float(raw.get("min_mean_objective", 0.0)),
        "min_positive_fold_share": float(raw.get("min_positive_fold_share", 0.67)),
        "max_full_sl_rate": float(raw.get("max_full_sl_rate", 0.22)),
        "max_timeout_rate": float(raw.get("max_timeout_rate", 0.55)),
        "min_worst_fold_net_pnl": float(raw.get("min_worst_fold_net_pnl", -250.0)),
        "max_no_trade_folds": int(raw.get("max_no_trade_folds", 0)),
    }


def _attach_candidate_payload(decisions: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "candidate_index" not in decisions.columns:
        return decisions.copy()
    out = decisions.copy()
    out["candidate_index"] = pd.to_numeric(out["candidate_index"], errors="coerce").astype("Int64")
    payload_cols = [
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "scenario",
        "archetype_handoff_row_id",
        "validation_week",
        "exec_guard_score_oof",
        "exec_guard_threshold",
        "exec_guard_keep_frac",
        "adaptive_guard_score_oof",
        "adaptive_guard_threshold",
        "adaptive_guard_margin",
        "adaptive_guard_keep_frac",
        "rank_pct",
        "strategy_rank_pct",
        "normalized_rank_score",
        "calibrated_score",
        "net_return",
        "gross_return",
        "simple_policy_exit_reason",
        "holding_bars",
    ]
    payload_cols = [col for col in payload_cols if col in validation.columns]
    payload = validation[payload_cols].reset_index(names="candidate_index")
    return out.merge(payload, on="candidate_index", how="left", suffixes=("", "_candidate"))


def _forbidden_handoff_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if any(token in col for token in FUTURE_OR_LABEL_TOKENS)]


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "archetype_handoff_row_id",
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "scenario_id",
    ]
    cols = [col for col in preferred if col in frame.columns]
    if "scenario_id" not in cols and "scenario" in frame.columns:
        cols.append("scenario")
    return frame[cols].copy() if cols else pd.DataFrame(index=frame.index)


def _handoff_key_mismatch(clean: pd.DataFrame, offline: pd.DataFrame) -> dict[str, Any]:
    if clean.empty or offline.empty:
        return {
            "clean_handoff_rows": int(len(clean)),
            "offline_replay_rows": int(len(offline)),
            "row_count_match": bool(len(clean) == len(offline)),
            "key_set_match": False,
        }
    left = _key_frame(clean).astype(str)
    right = _key_frame(offline).rename(columns={"scenario": "scenario_id"}).astype(str)
    common = [col for col in left.columns if col in right.columns]
    if not common:
        return {
            "clean_handoff_rows": int(len(clean)),
            "offline_replay_rows": int(len(offline)),
            "row_count_match": bool(len(clean) == len(offline)),
            "key_set_match": False,
            "common_key_columns": [],
        }
    left_keys = set(map(tuple, left[common].to_numpy(dtype=object)))
    right_keys = set(map(tuple, right[common].to_numpy(dtype=object)))
    return {
        "clean_handoff_rows": int(len(clean)),
        "offline_replay_rows": int(len(offline)),
        "row_count_match": bool(len(clean) == len(offline)),
        "common_key_columns": common,
        "key_set_match": bool(left_keys == right_keys),
        "keys_only_in_clean": int(len(left_keys - right_keys)),
        "keys_only_in_offline": int(len(right_keys - left_keys)),
    }


def _summary_parity(
    frozen_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    *,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frozen_summary.empty or reference_summary.empty:
        return (
            pd.DataFrame(),
            {
                "reference_available": bool(not reference_summary.empty),
                "max_abs_diff": float("nan"),
                "passes": False,
            },
        )
    rows: list[dict[str, Any]] = []
    frozen = frozen_summary.iloc[0]
    ref = reference_summary.iloc[0]
    for col in SUMMARY_COMPARE_COLUMNS:
        if col not in frozen_summary.columns or col not in reference_summary.columns:
            continue
        fval = pd.to_numeric(pd.Series([frozen[col]]), errors="coerce").iloc[0]
        rval = pd.to_numeric(pd.Series([ref[col]]), errors="coerce").iloc[0]
        diff = float(fval - rval) if pd.notna(fval) and pd.notna(rval) else float("nan")
        rows.append(
            {
                "metric": col,
                "frozen": float(fval) if pd.notna(fval) else np.nan,
                "reference": float(rval) if pd.notna(rval) else np.nan,
                "abs_diff": abs(diff) if np.isfinite(diff) else np.nan,
                "within_tolerance": bool(np.isfinite(diff) and abs(diff) <= tolerance),
            }
        )
    parity = pd.DataFrame(rows)
    finite_diffs = pd.to_numeric(parity.get("abs_diff", pd.Series(dtype=float)), errors="coerce")
    max_abs_diff = float(finite_diffs.max()) if finite_diffs.notna().any() else float("nan")
    return (
        parity,
        {
            "reference_available": True,
            "tolerance": float(tolerance),
            "max_abs_diff": max_abs_diff,
            "passes": bool(len(parity) > 0 and parity["within_tolerance"].all()),
        },
    )


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.6f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--parity-tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    handoff_dir = Path(args.handoff_dir)
    manifest_path = _require_path(
        args.manifest,
        fallback=handoff_dir / "meta_handoff_guarded_manifest.json",
        label="guarded manifest",
    )
    manifest = _read_json(manifest_path)
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    clean_handoff_path = _require_path(
        outputs.get("guarded_candidates"),
        fallback=handoff_dir / "meta_handoff_guarded_candidates.parquet",
        label="clean guarded handoff",
    )
    offline_path = _require_path(
        outputs.get("offline_replay_candidates"),
        fallback=handoff_dir / "meta_handoff_guarded_offline_replay_candidates.parquet",
        label="offline replay candidates",
    )
    source_path = _require_path(
        manifest.get("source_candidates"),
        fallback=Path(""),
        label="source candidates",
    )
    leakage_audit_path = _require_path(
        outputs.get("leakage_audit"),
        fallback=handoff_dir / "meta_handoff_guarded_leakage_audit.json",
        label="leakage audit",
    )
    materialized_summary_path = _require_path(
        outputs.get("summary"),
        fallback=handoff_dir / "meta_handoff_guarded_summary.csv",
        label="materialized summary",
    )
    ev_train_path = None
    if outputs.get("ev_curve_train_candidates"):
        ev_train_path = _require_path(
            outputs.get("ev_curve_train_candidates"),
            fallback=Path(""),
            label="frozen EV-curve training candidates",
        )

    clean_handoff = pd.read_parquet(clean_handoff_path)
    offline = pd.read_parquet(offline_path)
    offline["timestamp"] = pd.to_datetime(offline["timestamp"], utc=True, errors="coerce")
    if "validation_week" not in offline.columns:
        raise ValueError(f"{offline_path} missing validation_week")
    offline["validation_week"] = offline["validation_week"].astype(str)

    source_hash_ok = True
    expected_hash = str(manifest.get("source_candidates_sha256") or "")
    actual_hash = _sha256_path(source_path)
    if expected_hash:
        source_hash_ok = actual_hash == expected_hash

    scenario = str(manifest.get("scenario") or offline.get("scenario", pd.Series([""])).iloc[0])
    feature_mode = str(manifest.get("feature_mode") or "execution_known")
    ev_train = pd.DataFrame()
    if ev_train_path is not None:
        ev_train = pd.read_parquet(ev_train_path)
        ev_train["timestamp"] = pd.to_datetime(ev_train["timestamp"], utc=True, errors="coerce")
        if "validation_week" not in ev_train.columns:
            raise ValueError(f"{ev_train_path} missing validation_week")
        ev_train["validation_week"] = ev_train["validation_week"].astype(str)
        source_frame = pd.DataFrame()
    else:
        source_frame, _features = _prepare_frame(source_path, feature_mode=feature_mode)
        source_scenarios = [str(v) for v in manifest.get("source_scenarios", []) or []]
        if source_scenarios:
            source_frame = source_frame.loc[
                source_frame["scenario"].astype(str).isin(source_scenarios)
            ].copy().reset_index(drop=True)
        else:
            source_frame = source_frame.loc[source_frame["scenario"].eq(scenario)].copy().reset_index(drop=True)
        if source_frame.empty:
            raise ValueError(f"No source candidates found for scenario={scenario!r}")

    fixed_params = PortfolioPolicyParams(
        global_threshold_floor=float(manifest.get("global_threshold_floor", 0.0) or 0.0)
    )
    fold_rows = []
    decision_frames: list[pd.DataFrame] = []
    ev_train_rows = 0
    ev_train_boundary_violations = 0
    for fold_id, validation_week_str in enumerate(sorted(offline["validation_week"].dropna().unique())):
        validation_week = pd.Timestamp(validation_week_str, tz="UTC")
        if ev_train_path is not None:
            train = ev_train.loc[ev_train["validation_week"].eq(validation_week_str)].copy().reset_index(drop=True)
        else:
            train = source_frame.loc[source_frame["timestamp"].lt(validation_week)].copy().reset_index(drop=True)
        ev_train_rows += int(len(train))
        if not train.empty and "timestamp" in train.columns:
            ev_train_boundary_violations += int(
                pd.to_datetime(train["timestamp"], utc=True, errors="coerce")
                .ge(validation_week)
                .fillna(False)
                .sum()
            )
        validation = offline.loc[offline["validation_week"].eq(validation_week_str)].copy().reset_index(drop=True)
        if train.empty or validation.empty:
            continue
        decisions, _equity, metrics = _replay_with_train_curve(
            train_candidates=train,
            eval_candidates=validation,
            market_mode=str(manifest.get("market_mode") or "perps"),
            global_threshold_floor=float(fixed_params.global_threshold_floor),
        )
        attached = _attach_candidate_payload(decisions, validation)
        if not attached.empty:
            attached["fold_id"] = int(fold_id)
            attached["validation_week"] = validation_week_str
            decision_frames.append(attached)
        fold_rows.append(
            _fold_result(
                scenario=scenario,
                fold_id=int(fold_id),
                validation_week=validation_week.date().isoformat(),
                variant="frozen_guarded_replay",
                train_rows=len(train),
                validation_rows=len(validation),
                keep_frac=float(validation["frozen_fold_keep_frac"].iloc[0])
                if "frozen_fold_keep_frac" in validation.columns and len(validation)
                else float(validation["exec_guard_keep_frac"].iloc[0])
                if "exec_guard_keep_frac" in validation.columns and len(validation)
                else float(validation["adaptive_guard_keep_frac"].iloc[0])
                if "adaptive_guard_keep_frac" in validation.columns and len(validation)
                else float("nan"),
                score_threshold=float(validation["frozen_fold_score_threshold"].iloc[0])
                if "frozen_fold_score_threshold" in validation.columns and len(validation)
                else float(validation["exec_guard_threshold"].iloc[0])
                if "exec_guard_threshold" in validation.columns and len(validation)
                else float(validation["adaptive_guard_threshold"].iloc[0])
                if "adaptive_guard_threshold" in validation.columns and len(validation)
                else float("nan"),
                filtered_validation=validation,
                decisions=decisions,
                metrics=metrics,
                train_selector_score=float("nan"),
            )
        )

    fold_df = pd.DataFrame([row.__dict__ for row in fold_rows])
    summary_df = _summarise(fold_df, thresholds=_gate_thresholds(manifest))
    decisions_df = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    reference_summary = pd.read_csv(materialized_summary_path)
    parity_df, parity_summary = _summary_parity(
        summary_df,
        reference_summary,
        tolerance=float(args.parity_tolerance),
    )
    leakage_audit = _read_json(leakage_audit_path)
    forbidden_cols = _forbidden_handoff_columns(clean_handoff)
    handoff_match = _handoff_key_mismatch(clean_handoff, offline)
    audit = {
        "generated_by": "replay_meta_handoff_guarded_frozen",
        "manifest_path": str(manifest_path),
        "clean_handoff_path": str(clean_handoff_path),
        "offline_replay_candidates_path": str(offline_path),
        "source_candidates_path": str(source_path),
        "source_candidates_sha256": actual_hash,
        "source_candidates_hash_matches_manifest": bool(source_hash_ok),
        "scenario": scenario,
        "feature_mode": feature_mode,
        "guard_method": str(manifest.get("method") or ""),
        "fixed_global_threshold_floor": float(fixed_params.global_threshold_floor),
        "replayed_folds": int(len(fold_df)),
        "ev_curve_train_candidates_path": str(ev_train_path) if ev_train_path is not None else None,
        "ev_curve_train_source": "frozen_manifest_rows" if ev_train_path is not None else "source_candidates_prior_window",
        "ev_curve_train_rows_replayed": int(ev_train_rows),
        "ev_curve_train_boundary_violations": int(ev_train_boundary_violations),
        "clean_handoff_forbidden_columns": forbidden_cols,
        "clean_handoff_has_no_realized_outcomes": not forbidden_cols,
        "handoff_key_match": handoff_match,
        "prior_window_ev_curve_by_fold": True,
        "guard_refit": False,
        "threshold_reselection": False,
        "scenario_reselection": False,
        "leakage_audit_input": leakage_audit,
        "summary_parity": parity_summary,
    }
    pass_gate = bool(
        not summary_df.empty
        and bool(summary_df["pass_simple_policy_gate"].iloc[0])
        and bool(parity_summary.get("passes"))
        and bool(source_hash_ok)
        and bool(audit["clean_handoff_has_no_realized_outcomes"])
        and bool(handoff_match.get("row_count_match"))
        and bool(handoff_match.get("key_set_match"))
        and ev_train_boundary_violations == 0
        and bool(leakage_audit.get("leakage_safe_for_frozen_replay"))
    )
    audit["pass_frozen_guarded_replay_gate"] = pass_gate
    audit["status"] = "pass" if pass_gate else "fail"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "folds": args.out_dir / "frozen_replay_fold_summary.csv",
        "summary": args.out_dir / "frozen_replay_summary.csv",
        "decisions": args.out_dir / "frozen_replay_decisions.parquet",
        "parity": args.out_dir / "frozen_replay_parity.csv",
        "audit": args.out_dir / "frozen_replay_audit.json",
        "fixed_policy_config": args.out_dir / "fixed_portfolio_policy_config.json",
        "manifest": args.out_dir / "frozen_replay_manifest.json",
        "report": args.out_dir / "frozen_replay_report.md",
    }
    fold_df.to_csv(paths["folds"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    decisions_df.to_parquet(paths["decisions"], index=False)
    parity_df.to_csv(paths["parity"], index=False)
    paths["audit"].write_text(json.dumps(_json_safe(audit), indent=2), encoding="utf-8")
    paths["fixed_policy_config"].write_text(
        json.dumps(_json_safe(fixed_params.to_live_config()), indent=2),
        encoding="utf-8",
    )
    replay_manifest = {
        **audit,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(_json_safe(replay_manifest), indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Frozen Guarded Handoff Replay",
        "",
        "This replay uses stored guarded candidates only. The guard is not refit, thresholds are not reselected, and the scenario is fixed from the materialized manifest.",
        "",
        "## Gate",
        "",
        pd.DataFrame(
            [
                {
                    "status": audit["status"],
                    "pass_frozen_guarded_replay_gate": pass_gate,
                    "source_hash_ok": source_hash_ok,
                    "clean_handoff_no_outcomes": audit["clean_handoff_has_no_realized_outcomes"],
                    "handoff_key_set_match": handoff_match.get("key_set_match"),
                    "parity_pass": parity_summary.get("passes"),
                }
            ]
        ).to_markdown(index=False),
        "",
        "## Summary",
        "",
        _fmt_table(
            summary_df,
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
        "## Parity",
        "",
        _fmt_table(parity_df, ["metric", "frozen", "reference", "abs_diff", "within_tolerance"]),
        "",
        "## Fold Detail",
        "",
        _fmt_table(
            fold_df,
            [
                "validation_week",
                "net_pnl",
                "objective",
                "accepted_trades",
                "full_sl_rate",
                "timeout_rate",
                "hit_rate",
                "keep_frac",
            ],
            max_rows=80,
        ),
    ]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            _json_safe(
                {
                    "status": audit["status"],
                    "summary": summary_df.to_dict(orient="records"),
                    "parity": parity_summary,
                    "outputs": {key: str(value) for key, value in paths.items()},
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
