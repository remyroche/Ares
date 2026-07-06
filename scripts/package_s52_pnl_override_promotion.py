#!/usr/bin/env python3
"""Package S52 PnL-override candidates into a meta/execution handoff decision.

The package consumes:

* Gate 3 readiness rows with explicit PnL override status;
* frozen clean-action replay metrics;
* source/regime replay breakdowns;
* the existing regime policy recommendation table, when available.

It does not promote hard gates.  It converts the replay-passing S52 candidates
into a conservative production-integration plan: default candidate, benchmark
candidate, source/regime action hints, and explicit caveats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_HANDOFF_ROOT = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_v1"
)
DEFAULT_REPLAY_DIR = DEFAULT_HANDOFF_ROOT / "s52_frozen_action_replay_current_v1"
DEFAULT_READINESS_DIR = DEFAULT_HANDOFF_ROOT / "s52_meta_handoff_gate3_readiness_current_v3_pnl_override"
DEFAULT_OUT_DIR = DEFAULT_HANDOFF_ROOT / "s52_pnl_override_promotion_package_v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _num(values: Any, *, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        if index is None:
            return pd.Series(dtype=np.float32)
        return pd.Series(default, index=index, dtype=np.float32)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val * 100:.2f}%"


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _candidate_role(summary: pd.DataFrame) -> pd.Series:
    rows = summary.copy()
    rows["capacity_score"] = (
        _num(rows.get("sum_net_pnl"), index=rows.index).fillna(-np.inf)
        + 10.0 * _num(rows.get("worst_month_ret_net"), index=rows.index).fillna(-np.inf)
        + 2.0 * _num(rows.get("hit_rate_ret_net"), index=rows.index).fillna(0.0)
    )
    rows["precision_score"] = (
        _num(rows.get("mean_ret_net"), index=rows.index).fillna(-np.inf)
        + _num(rows.get("worst_month_ret_net"), index=rows.index).fillna(-np.inf)
        - 0.25 * _num(rows.get("full_path_bad_mae"), index=rows.index).fillna(1.0)
        - 0.10 * _num(rows.get("dominant_side_share"), index=rows.index).fillna(1.0)
    )
    default_idx = rows["capacity_score"].idxmax() if len(rows) else None
    precision_idx = rows["precision_score"].idxmax() if len(rows) else None
    roles = pd.Series("diagnostic", index=summary.index, dtype=object)
    if default_idx is not None:
        roles.loc[default_idx] = "default_capacity_candidate"
    if precision_idx is not None and precision_idx != default_idx:
        roles.loc[precision_idx] = "conservative_precision_benchmark"
    elif precision_idx is not None:
        roles.loc[precision_idx] = str(roles.loc[precision_idx]) + "+precision_benchmark"
    return roles


def _candidate_table(readiness: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    rows = replay.copy()
    if not readiness.empty and "variant" in readiness.columns:
        keep = [
            "variant",
            "gate3_status",
            "path_risk_status",
            "bad_mae_accepted_by_pnl_override",
            "failed_checks",
        ]
        rows = rows.merge(readiness[[col for col in keep if col in readiness.columns]], on="variant", how="left")
    rows["promotion_role"] = _candidate_role(rows)
    rows["promotion_status"] = np.where(
        rows.get("gate3_status", "").astype(str).eq("pnl_override_candidate")
        & rows.get("clean_handoff_has_no_realized_outcomes", False).astype(bool)
        & rows.get("offline_parity_key_set_match", False).astype(bool)
        & _num(rows.get("unmatched_rows"), index=rows.index).fillna(1).eq(0)
        & _num(rows.get("mean_ret_net"), index=rows.index).ge(0.010)
        & _num(rows.get("worst_month_ret_net"), index=rows.index).ge(0.005),
        "promote_to_meta_execution_integration_candidate",
        "blocked_or_diagnostic",
    )
    rows["recommended_use"] = np.where(
        rows["promotion_status"].astype(str).str.startswith("promote"),
        "feed_clean_action_policy_to_train_meta_execution_shadow; keep_bad_mae_as_sizing_risk",
        "diagnostic_only",
    )
    return rows.sort_values(["promotion_status", "promotion_role", "sum_net_pnl"], ascending=[True, True, False])


def _risk_tier(frame: pd.DataFrame) -> pd.Series:
    ret = _num(frame.get("mean_ret_net"), index=frame.index)
    bad = _num(frame.get("full_path_bad_mae"), index=frame.index)
    timeout = _num(frame.get("timeout"), index=frame.index)
    support = _num(frame.get("rows"), index=frame.index)
    rows = pd.Series("diagnostic_only", index=frame.index, dtype=object)
    rows.loc[ret.lt(0.0)] = "avoid_or_strong_downweight"
    rows.loc[ret.ge(0.0) & bad.ge(0.60)] = "positive_but_high_path_risk_size_down"
    rows.loc[ret.ge(0.0) & bad.ge(0.50) & bad.lt(0.60)] = "positive_path_risk_monitor"
    rows.loc[ret.ge(0.0) & bad.lt(0.50)] = "positive_context_candidate"
    rows.loc[timeout.gt(0.12)] = "timeout_risk_downweight"
    rows.loc[support.lt(20)] = "low_support_feature_only"
    return rows


def _action_for_tier(tier: str) -> tuple[str, str, float, float]:
    if tier == "low_support_feature_only":
        return ("feature_only_low_support", "diagnostic_only_wait_for_more_support", 1.00, 1.00)
    if tier == "positive_context_candidate":
        return ("feature_plus_normal_size", "normal_meta_threshold", 1.05, 1.00)
    if tier == "positive_path_risk_monitor":
        return ("feature_plus_size_down", "require_meta_confirmation", 0.90, 0.75)
    if tier == "positive_but_high_path_risk_size_down":
        return ("feature_plus_strong_size_down", "require_high_meta_confidence", 0.75, 0.50)
    if tier == "timeout_risk_downweight":
        return ("feature_plus_timeout_downweight", "require_timeout_model_agreement", 0.70, 0.50)
    if tier == "avoid_or_strong_downweight":
        return ("feature_plus_strong_downweight", "do_not_hard_gate_without_fresh_oos", 0.50, 0.25)
    return ("feature_only", "diagnostic_only", 1.00, 1.00)


def _source_regime_action_table(replay_breakdown: pd.DataFrame) -> pd.DataFrame:
    if replay_breakdown.empty:
        return pd.DataFrame()
    groupings = {
        "side_source",
        "side_aegmm",
        "side_side_aegmm",
        "side_reconstruction",
        "side_leaf_exec_margin",
    }
    rows = replay_breakdown[replay_breakdown["grouping"].astype(str).isin(groupings)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["risk_tier"] = _risk_tier(rows)
    actions = rows["risk_tier"].map(_action_for_tier)
    rows["recommended_action"] = [item[0] for item in actions]
    rows["meta_threshold_action"] = [item[1] for item in actions]
    rows["sample_weight_multiplier_hint"] = [item[2] for item in actions]
    rows["size_multiplier_hint"] = [item[3] for item in actions]
    support = _num(rows.get("rows"), index=rows.index)
    rows["support_status"] = np.select(
        [support.ge(100), support.ge(50), support.ge(20)],
        ["high_support", "medium_support", "minimum_support"],
        default="low_support_diagnostic",
    )
    rows["action_confidence"] = np.select(
        [
            rows["support_status"].eq("high_support") & rows["risk_tier"].eq("positive_context_candidate"),
            rows["support_status"].isin(["high_support", "medium_support"]),
            rows["support_status"].eq("minimum_support"),
        ],
        ["medium_high", "medium", "low_medium"],
        default="low",
    )
    rows["execution_policy"] = "P_current_trailing_tp075_sl050_tr035"
    rows["execution_policy_status"] = "fixed_current_policy_replay_passed; menu_policy_search_pending"
    rows["hard_gate_allowed"] = False
    rows["promotion_note"] = np.where(
        rows["risk_tier"].astype(str).str.contains("positive"),
        "use_as_meta_context_and_sizing_hint",
        "diagnostic_or_downweight_only_until_fresh_oos_confirms",
    )
    cols = [
        "variant",
        "grouping",
        "month",
        "side_name",
        "source_semantic_family",
        "aegmm_cluster",
        "side_aegmm_cluster",
        "reconstruction_bin",
        "regime_lgbm_leaf_exec_margin_k4",
        "rows",
        "symbols",
        "mean_ret_net",
        "worst_month_ret_net",
        "mean_exec_margin",
        "hit_rate_ret_net",
        "positive_exec_margin_rate",
        "full_path_bad_mae",
        "timeout",
        "inferred_decision_stop_touch",
        "inferred_full_path_stop_touch",
        "support_status",
        "action_confidence",
        "risk_tier",
        "recommended_action",
        "meta_threshold_action",
        "sample_weight_multiplier_hint",
        "size_multiplier_hint",
        "execution_policy",
        "execution_policy_status",
        "hard_gate_allowed",
        "promotion_note",
    ]
    return rows[[col for col in cols if col in rows.columns]].sort_values(
        ["variant", "grouping", "mean_ret_net", "rows"],
        ascending=[True, True, False, False],
    )


def _regime_recommendation_summary(policy_recs: pd.DataFrame) -> pd.DataFrame:
    if policy_recs.empty:
        return pd.DataFrame()
    rows = policy_recs.copy()
    if "promotion_status" not in rows.columns:
        return pd.DataFrame()
    group_cols = ["regime_model", "recommended_action", "promotion_status", "validation_status"]
    present = [col for col in group_cols if col in rows.columns]
    agg = rows.groupby(present, dropna=False).agg(
        cells=("source_tag", "count"),
        fit_rows=("fit_rows", "sum"),
        holdout_rows=("holdout_rows", "sum"),
        mean_expected_delta_exec_margin=("expected_delta_exec_margin", "mean"),
        mean_expected_delta_full_path_bad_mae=("expected_delta_full_path_bad_mae", "mean"),
    )
    return agg.reset_index().sort_values(["cells", "fit_rows"], ascending=[False, False])


def _write_report(
    path: Path,
    candidates: pd.DataFrame,
    actions: pd.DataFrame,
    regime_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# S52 PnL-Override Promotion Package",
        "",
        "## Scope",
        "",
        "Packages the replay-passing S52 meta-threshold candidates for train_meta / execution integration.",
        "This does not create hard regime gates. It converts source/regime evidence into meta features, sample-weight hints, threshold hints, and sizing/risk hints.",
        "",
        "## Inputs",
        "",
        f"- readiness summary: `{manifest['inputs']['readiness_summary']}`",
        f"- frozen replay summary: `{manifest['inputs']['replay_summary']}`",
        f"- frozen replay breakdown: `{manifest['inputs']['replay_breakdown']}`",
        f"- policy recommendations: `{manifest['inputs'].get('policy_recommendation_table')}`",
        "",
        "## Candidate Decision",
        "",
        candidates[
            [
                "variant",
                "promotion_role",
                "promotion_status",
                "rows",
                "symbols",
                "mean_ret_net",
                "worst_month_ret_net",
                "sum_net_pnl",
                "full_path_bad_mae",
                "max_month_full_path_bad_mae",
                "timeout",
                "dominant_side_share",
                "recommended_use",
            ]
        ].to_markdown(index=False),
        "",
        "## Source/Regime Actions",
        "",
    ]
    if actions.empty:
        lines.append("_No source/regime actions._")
    else:
        cols = [
            "variant",
            "grouping",
            "side_name",
            "source_semantic_family",
            "aegmm_cluster",
            "side_aegmm_cluster",
            "reconstruction_bin",
            "regime_lgbm_leaf_exec_margin_k4",
            "rows",
            "mean_ret_net",
            "full_path_bad_mae",
            "timeout",
            "risk_tier",
            "recommended_action",
            "size_multiplier_hint",
            "hard_gate_allowed",
        ]
        lines.append(actions[[col for col in cols if col in actions.columns]].head(80).to_markdown(index=False))
    lines += [
        "",
        "## Regime Recommendation Rollup",
        "",
        regime_summary.head(40).to_markdown(index=False) if not regime_summary.empty else "_No regime recommendation rollup._",
        "",
        "## Decision",
        "",
        "- promote `top10 sidecap80` as the default capacity candidate for train_meta/execution shadow integration",
        "- keep `top5 sidecap80` as the conservative precision benchmark",
        "- use bad-MAE/stop-touch as sizing and risk diagnostics, not as an automatic veto under the current PnL-first rule",
        "- do not introduce hard source/regime gates from this package alone",
        "- next required validation remains fresh production-parity shadow or raw-OHLC replay before deployment promotion",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_package(
    *,
    handoff_root: Path,
    readiness_dir: Path,
    replay_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    readiness_path = readiness_dir / "s52_meta_handoff_gate3_readiness_summary.csv"
    replay_summary_path = replay_dir / "s52_frozen_action_replay_summary.csv"
    replay_breakdown_path = replay_dir / "s52_frozen_action_replay_breakdown.csv"
    policy_recs_path = handoff_root / "policy_recommendation_table.csv"
    readiness = _read_csv(readiness_path)
    replay_summary = _read_csv(replay_summary_path)
    replay_breakdown = _read_csv(replay_breakdown_path)
    policy_recs = _read_csv(policy_recs_path, required=False)

    candidates = _candidate_table(readiness, replay_summary)
    actions = _source_regime_action_table(replay_breakdown)
    regime_summary = _regime_recommendation_summary(policy_recs)
    paths = {
        "candidate_decision": out_dir / "s52_pnl_override_candidate_decision.csv",
        "source_regime_actions": out_dir / "s52_pnl_override_source_regime_action_table.csv",
        "regime_recommendation_rollup": out_dir / "s52_pnl_override_regime_recommendation_rollup.csv",
        "report": out_dir / "s52_pnl_override_promotion_package.md",
        "manifest": out_dir / "manifest.json",
    }
    candidates.to_csv(paths["candidate_decision"], index=False)
    actions.to_csv(paths["source_regime_actions"], index=False)
    regime_summary.to_csv(paths["regime_recommendation_rollup"], index=False)
    manifest = {
        "generated_by": "package_s52_pnl_override_promotion",
        "status": "promotion_package_ready_for_train_meta_execution_shadow",
        "decision": {
            "default_candidate": candidates.loc[
                candidates["promotion_role"].astype(str).str.contains("default_capacity_candidate"),
                "variant",
            ].astype(str).head(1).tolist(),
            "benchmark_candidate": candidates.loc[
                candidates["promotion_role"].astype(str).str.contains("precision_benchmark"),
                "variant",
            ].astype(str).head(1).tolist(),
            "hard_gates_allowed": False,
            "bad_mae_policy": "risk_diagnostic_and_sizing_hint_not_absolute_veto_when_pnl_override_passes",
        },
        "inputs": {
            "handoff_root": str(handoff_root),
            "readiness_summary": str(readiness_path),
            "replay_summary": str(replay_summary_path),
            "replay_breakdown": str(replay_breakdown_path),
            "policy_recommendation_table": str(policy_recs_path) if policy_recs_path.exists() else None,
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "candidate_rows": int(len(candidates)),
        "source_regime_action_rows": int(len(actions)),
        "regime_recommendation_rollup_rows": int(len(regime_summary)),
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    _write_report(paths["report"], candidates, actions, regime_summary, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_package(
        handoff_root=args.handoff_root,
        readiness_dir=args.readiness_dir,
        replay_dir=args.replay_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
