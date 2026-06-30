#!/usr/bin/env python3
"""Compare exact-state size-action learning runs.

The size-action experiments intentionally produce many sparse arms. This
utility summarizes action quality and selected precondition/gate thresholds
across one or more run directories so a larger panel can be compared against
the current reference runs without hand-written notebook snippets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MATERIAL_DELTA_EPS = 1e-6


DEFAULT_ARMS = [
    "C0_baseline",
    "C1_exact_state_oracle_full",
    "C3e_intervention_classifier_action_selector",
    "C3f_calibrated_positive_value_gate",
    "C3g_immediate_capacity_decomposition",
    "C3h_bagged_consensus_gate",
    "C3bn_bagged_safety_stage1_oracle_zero_or_half_diagnostic",
    "C3bq_bagged_safety_stage1_calibrated_zero_or_half_gate",
    "C3bt_bagged_safety_stage1_hard_portfolio_veto_zero_or_half_gate",
    "C3bu_bagged_safety_stage1_learned_no_cut_risk_veto_zero_or_half_gate",
    "C3bv_bagged_safety_stage1_hard_portfolio_precondition_zero_or_half_gate",
    "C3bw_bagged_safety_stage1_fold_fitted_precondition_zero_or_half_gate",
    "C3br_bagged_safety_stage1_oracle_group_zero_or_half_diagnostic",
    "C3bx_bagged_safety_stage1_direct_fixed_zero_or_half_top_gate",
    "C3by_bagged_safety_stage1_oracle_group_predicted_action_gate",
    "C3bz_bagged_safety_stage1_oracle_group_conservative_predicted_action_gate",
    "C3ca_bagged_safety_stage1_oracle_group_ranker_action_gate",
    "C3cb_bagged_safety_stage1_ranker_action_gate",
    "C3cc_bagged_safety_stage1_ranker_action_acceptance_gate",
    "C3cd_bagged_safety_stage1_oracle_group_recall_ranker_action_gate",
    "C3ce_bagged_safety_stage1_oracle_group_recall_ranker_acceptance_gate",
    "C3cf_bagged_safety_stage1_oracle_group_recall_economic_action_gate",
    "C3cg_bagged_safety_stage1_oracle_group_recall_group_action_gate",
    "C3ch_bagged_safety_stage1_oracle_group_recall_group_action_acceptance_gate",
    "C3ci_bagged_safety_stage1_decomposed_value_acceptance_gate",
    "C3cj_bagged_safety_stage1_decomposed_calibrated_gate",
    "C3ck_bagged_safety_stage1_decomposed_calibrated_acceptance_gate",
    "C3cl_stage1_calibrated_fixed_half_gate",
    "C3cm_stage1_calibrated_fixed_zero_gate",
    "C3dn_strategy_stage1_calibrated_fixed_zero_gate",
    "C3cn_bagged_safety_stage1_oracle_group_recall_group_action_family_acceptance_gate",
    "C3co_bagged_safety_stage1_oracle_group_recall_high_p_group_action_family_gate",
    "C3cp_bagged_safety_stage1_oracle_group_recall_high_p_group_action_ranker_gate",
    "C3cq_bagged_safety_stage1_direct_group_action_family_gate",
    "C3cr_bagged_safety_stage1_direct_group_action_family_acceptance_gate",
    "C3cs_bagged_safety_stage1_direct_group_action_family_zero_acceptance_gate",
    "C3ct_bagged_safety_stage1_direct_group_action_family_zero_strict_acceptance_gate",
    "C3cu_bagged_safety_stage1_direct_group_action_family_zero_consensus_acceptance_gate",
    "C3cv_bagged_safety_stage1_direct_group_action_family_zero_value_consensus_gate",
    "C3cw_bagged_safety_stage1_direct_group_action_family_value_consensus_gate",
    "C3cx_bagged_safety_stage1_high_conf_multiplier_value_gate",
    "C3cy_bagged_safety_stage1_family_supported_multiplier_value_gate",
    "C3cz_bagged_safety_stage1_family_supported_multiplier_value_v2_gate",
    "C3da_bagged_safety_stage1_c3cw_or_c3cz_union_gate",
    "C3db_bagged_safety_stage1_c3cw_or_c3cz_nonoverlap_union_gate",
    "C3do_bagged_safety_c3db_or_strategy_calibrated_zero_union_gate",
    "C3dp_bagged_safety_c3db_or_filtered_strategy_calibrated_zero_union_gate",
    "C3dq_bagged_safety_c3db_or_exposure_guarded_strategy_calibrated_zero_union_gate",
    "C3dr_bagged_safety_c3db_or_calibrated_mean_recall_zero_union_gate",
    "C3ds_bagged_safety_c3db_or_unfiltered_calibrated_mean_recall_zero_union_gate",
    "C3dt_bagged_safety_c3db_or_calibrated_lcb_recall_zero_union_gate",
    "C3du_bagged_safety_c3db_or_strategy_filtered_mean_recall_zero_union_gate",
    "C3dv_bagged_safety_c3db_or_zero_harm_mean_recall_zero_union_gate",
    "C3ea_bagged_safety_c3db_or_rank_tail_zero_harm_mean_recall_zero_union_gate",
    "C3eb_bagged_safety_c3ea_or_pressure_opportunity_zero_harm_union_gate",
    "C3ec_bagged_safety_c3ea_or_pressure_opportunity_acceptance_zero_harm_union_gate",
    "C3ed_bagged_safety_c3ea_or_pressure_opportunity_action_tail_veto_union_gate",
    "C3er_bagged_safety_c3ea_or_recall_opportunity_group_action_family_tail_union_gate",
    "C3es_bagged_safety_c3ea_or_recall_opportunity_group_action_ranker_tail_union_gate",
    "C3et_bagged_safety_c3ea_or_recall_opportunity_zero_no_tail_union_gate",
    "C3eu_bagged_safety_c3ed_or_high_rank_group_classifier_tail_union_gate",
    "C3ev_bagged_safety_c3ed_or_high_rank_group_classifier_no_tail_union_gate",
    "C3ew_bagged_safety_c3ed_or_high_rank_group_classifier_moderate_no_tail_union_gate",
    "C3fc_bagged_safety_c3ed_or_capacity_release_opportunity_action_tail_union_gate",
    "C3fd_bagged_safety_c3ed_or_low_stage1_missed_oracle_classifier_tail_union_gate",
    "C3fe_bagged_safety_c3ed_or_relaxed_low_stage1_missed_oracle_tail_union_gate",
    "C3ff_bagged_safety_c3ed_or_relaxed_low_stage1_missed_oracle_precondition_only_union_gate",
    "C3fg_bagged_safety_c3ed_or_action_tail_ranked_low_stage1_missed_oracle_union_gate",
    "C3fh_bagged_safety_c3ed_or_action_tail_ranked_low_stage1_missed_oracle_prefer_secondary_union_gate",
    "C3fi_bagged_safety_c3ed_or_action_tail_ranked_low_stage1_missed_oracle_lcb_guard_union_gate",
    "C3fj_bagged_safety_c3ed_stage1_lcb_guard_gate",
    "C3fk_bagged_safety_c3ed_or_calibrated_group_hurdle_opportunity_union_gate",
    "C3fl_bagged_safety_c3ed_or_high_rank_calibrated_group_hurdle_consensus_union_gate",
    "C3fm_bagged_safety_c3ed_or_calibrated_group_hurdle_no_action_tail_union_gate",
    "C3fn_bagged_safety_c3ed_or_calibrated_group_hurdle_foldfit_action_acceptance_union_gate",
    "C3fo_bagged_safety_c3ed_or_calibrated_group_hurdle_positive_value_acceptance_union_gate",
    "C3fp_bagged_safety_c3ed_or_rank_calibrated_value_secondary_union_gate",
    "C3fq_calibrated_stage1_group_action_selector",
    "C3fa_calibrated_group_hurdle_decomposed_gate",
    "C3fb_high_rank_group_classifier_moderate_secondary_only_diagnostic",
    "C3ef_bagged_safety_c3ed_or_direct_action_recall_zero_harm_union_gate",
    "C3eg_bagged_safety_c3ed_global_action_tail_veto_gate",
    "C3eh_bagged_safety_c3ed_or_direct_zero_half_recall_zero_harm_union_gate",
    "C3ei_bagged_safety_c3ed_or_direct_zero_pnl_recall_union_gate",
    "C3ej_bagged_safety_c3ed_or_high_value_zero_classifier_union_gate",
    "C3ek_bagged_safety_c3ed_or_high_value_zero_classifier_relaxed_union_gate",
    "C3el_bagged_safety_c3ed_or_high_value_zero_classifier_broad_union_gate",
    "C3em_bagged_safety_c3ed_or_high_value_zero_classifier_expanded_union_gate",
    "C3dw_bagged_safety_c3db_or_zero_harm_fixed_zero_union_gate",
    "C3dx_bagged_safety_c3db_or_strategy_zero_harm_mean_recall_zero_union_gate",
    "C3dy_bagged_safety_c3db_or_strategy_opportunity_zero_harm_union_gate",
    "C3dz_bagged_safety_c3db_or_action_economic_zero_harm_union_gate",
]


def _resolve_arms(value: str) -> set[str] | None:
    text = str(value).strip()
    if not text or text.lower() == "all":
        return None
    aliases = {arm: arm for arm in DEFAULT_ARMS}
    aliases.update({arm.lower(): arm for arm in DEFAULT_ARMS})
    aliases.update({arm.split("_", 1)[0]: arm for arm in DEFAULT_ARMS})
    aliases.update({arm.split("_", 1)[0].lower(): arm for arm in DEFAULT_ARMS})
    resolved: set[str] = set()
    unknown: list[str] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        arm = aliases.get(token) or aliases.get(token.lower())
        if arm is None:
            unknown.append(token)
        else:
            resolved.add(arm)
    if unknown:
        valid = ", ".join(arm.split("_", 1)[0] for arm in DEFAULT_ARMS)
        raise SystemExit(f"Unknown arm(s): {', '.join(unknown)}. Valid short codes: {valid}")
    return resolved


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_transfer_diagnostics(run_dir: Path) -> pd.DataFrame:
    enriched = run_dir / "size_action_selector_transfer_diagnostics_enriched.csv"
    if enriched.exists():
        return _read_csv(enriched)
    return _read_csv(run_dir / "size_action_selector_transfer_diagnostics.csv")


def _run_label(path: Path) -> str:
    return path.name.replace("exact_state_size_action_learning_", "")


def _summarize_quality(run_dir: Path, arms: set[str] | None = None) -> pd.DataFrame:
    quality = _read_csv(run_dir / "size_action_action_quality.csv")
    if quality.empty:
        return pd.DataFrame()
    if arms:
        quality = quality[quality["arm"].isin(arms)].copy()
    if quality.empty:
        return pd.DataFrame()
    out = quality.groupby("arm", dropna=False).agg(
        folds=("fold_id", "nunique"),
        scheduled_groups=("scheduled_groups", "sum"),
        interventions=("intervention_count", "sum"),
        positive_actions=("positive_action_count", "sum"),
        realized_delta_full_J_sum=("realized_delta_full_J_sum", "sum"),
        realized_delta_immediate_J_sum=("realized_delta_immediate_J_sum", "sum"),
        mean_positive_action_rate=("positive_action_rate", "mean"),
        mean_intervention_rate=("intervention_rate", "mean"),
        positive_folds=("realized_delta_full_J_sum", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > MATERIAL_DELTA_EPS).sum())),
        mean_oracle_capture=("oracle_positive_group_capture_rate", "mean"),
        oracle_gain_capture_ratio_sum=("oracle_gain_capture_ratio", "sum"),
    ).reset_index()
    out.insert(0, "run", _run_label(run_dir))
    out["precision_total"] = out["positive_actions"] / out["interventions"].replace(0, np.nan)
    out["precision_total"] = out["precision_total"].fillna(0.0)
    return out


def _summarize_binding_opportunities(run_dir: Path) -> pd.DataFrame:
    panel = _read_csv(run_dir / "size_action_exact_panel.csv")
    if panel.empty:
        return pd.DataFrame()
    required = {"fold_id", "timestamp", "strategy_id", "group_can_bind"}
    if not required.issubset(panel.columns):
        return pd.DataFrame()
    work = panel.copy()
    if "split" in work.columns:
        work = work.loc[work["split"].astype(str).eq("eval")].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["strategy_id"] = work["strategy_id"].astype(str)
    work["group_can_bind"] = pd.to_numeric(work["group_can_bind"], errors="coerce").fillna(0.0)
    groups = work.drop_duplicates(["fold_id", "timestamp", "strategy_id"]).copy()
    if groups.empty:
        return pd.DataFrame()
    out = groups.groupby("fold_id", dropna=False).agg(
        scheduled_groups_from_panel=("timestamp", "size"),
        binding_opportunity_groups=("group_can_bind", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum())),
    ).reset_index()
    out.insert(0, "run", _run_label(run_dir))
    return out


def _summarize_replay(run_dir: Path, arms: set[str] | None = None) -> pd.DataFrame:
    summary = _read_csv(run_dir / "size_action_fold_summary.csv")
    if summary.empty:
        return pd.DataFrame()
    if arms:
        summary = summary[summary["arm"].isin(arms)].copy()
    if summary.empty:
        return pd.DataFrame()

    numeric_cols = [
        "net_pnl",
        "gross_pnl",
        "cost_pnl",
        "trade_count",
        "full_sl_rate",
        "timeout_rate",
        "avg_open_positions",
        "mean_multiplier",
        "min_multiplier",
    ]
    for col in numeric_cols:
        if col not in summary.columns:
            default = 1.0 if col in {"mean_multiplier", "min_multiplier"} else 0.0
            summary[col] = default
    for col in numeric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    if "fold_id" not in summary.columns:
        summary["fold_id"] = np.arange(len(summary), dtype=int)

    baseline = summary.loc[summary["arm"].eq("C0_baseline"), ["fold_id", "net_pnl", "trade_count", "full_sl_rate", "timeout_rate"]].copy()
    baseline = baseline.rename(
        columns={
            "net_pnl": "baseline_net_pnl",
            "trade_count": "baseline_trade_count",
            "full_sl_rate": "baseline_full_sl_rate",
            "timeout_rate": "baseline_timeout_rate",
        }
    )
    work = summary.merge(baseline, on="fold_id", how="left")
    work["delta_net_pnl"] = pd.to_numeric(work["net_pnl"], errors="coerce").fillna(0.0) - pd.to_numeric(
        work.get("baseline_net_pnl"), errors="coerce"
    ).fillna(0.0)
    work["delta_trade_count"] = pd.to_numeric(work.get("trade_count"), errors="coerce").fillna(0.0) - pd.to_numeric(
        work.get("baseline_trade_count"), errors="coerce"
    ).fillna(0.0)
    work["delta_full_sl_rate"] = pd.to_numeric(work.get("full_sl_rate"), errors="coerce").fillna(0.0) - pd.to_numeric(
        work.get("baseline_full_sl_rate"), errors="coerce"
    ).fillna(0.0)
    work["delta_timeout_rate"] = pd.to_numeric(work.get("timeout_rate"), errors="coerce").fillna(0.0) - pd.to_numeric(
        work.get("baseline_timeout_rate"), errors="coerce"
    ).fillna(0.0)

    out = work.groupby("arm", dropna=False).agg(
        folds=("fold_id", "nunique"),
        net_pnl_sum=("net_pnl", "sum"),
        baseline_net_pnl_sum=("baseline_net_pnl", "sum"),
        delta_net_pnl_sum=("delta_net_pnl", "sum"),
        median_delta_net_pnl=("delta_net_pnl", "median"),
        q25_delta_net_pnl=("delta_net_pnl", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).quantile(0.25))),
        positive_delta_folds=("delta_net_pnl", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum())),
        trade_count_sum=("trade_count", "sum"),
        delta_trade_count_sum=("delta_trade_count", "sum"),
        mean_full_sl_rate=("full_sl_rate", "mean"),
        mean_delta_full_sl_rate=("delta_full_sl_rate", "mean"),
        mean_timeout_rate=("timeout_rate", "mean"),
        mean_delta_timeout_rate=("delta_timeout_rate", "mean"),
        mean_multiplier=("mean_multiplier", "mean"),
        min_multiplier=("min_multiplier", "min"),
    ).reset_index()
    out.insert(0, "run", _run_label(run_dir))
    return out


def _threshold_summary(run_dir: Path, arms: set[str] | None = None) -> pd.DataFrame:
    thresholds = _read_csv(run_dir / "size_action_gate_thresholds.csv")
    if thresholds.empty:
        return pd.DataFrame()
    if arms:
        thresholds = thresholds[thresholds["arm"].isin(arms)].copy()
    if thresholds.empty:
        return pd.DataFrame()
    keep = [
        "fold_id",
        "arm",
        "threshold_source",
        "train_interventions",
        "train_precision",
        "train_delta_full_J_sum",
        "zero_half_p_min",
        "zero_half_top_fraction",
        "zero_half_vote_min",
        "diagnostic_veto_thresholds",
        "diagnostic_precondition_thresholds",
        "diagnostic_precondition_enabled",
        "no_cut_risk_veto_enabled",
        "no_cut_risk_max",
        "precondition_feature_count",
        "oracle_group_p_min",
        "oracle_group_top_fraction",
        "predicted_action_action_score_min",
        "predicted_action_action_margin_min",
        "predicted_action_action_positive_min",
        "predicted_action_pred_delta_min",
        "predicted_action_top_fraction",
        "direct_fixed_multiplier",
        "direct_fixed_top_fraction",
        "direct_fixed_p_min",
        "zero_consensus_multiplier_confidence_min",
        "zero_consensus_pred_delta_max",
        "multiplier_consensus_allowed_multipliers",
        "multiplier_consensus_confidence_min",
        "multiplier_consensus_pred_delta_max",
        "high_conf_multiplier_allowed_multipliers",
        "high_conf_multiplier_p_intervene_min",
        "high_conf_multiplier_confidence_min",
        "high_conf_multiplier_pred_delta_max",
        "high_conf_multiplier_require_family_support",
        "high_conf_multiplier_family_noop_confidence_min",
        "high_conf_multiplier_family_agreement_confidence_min",
        "high_conf_multiplier_low_conf_family_agreement_pred_delta_min",
        "union_primary_arm",
        "union_secondary_arm",
    ]
    keep = [c for c in keep if c in thresholds.columns]
    out = thresholds[keep].copy()
    out.insert(0, "run", _run_label(run_dir))
    return out


def _active_selected_rows(transfer: pd.DataFrame, arm: str) -> pd.DataFrame:
    if transfer.empty or "arm" not in transfer.columns:
        return pd.DataFrame()
    rows = transfer.loc[transfer["arm"].eq(arm)].copy()
    if rows.empty:
        return rows
    selected = rows.get("selected", False)
    if isinstance(selected, pd.Series):
        selected = selected.fillna(False).astype(bool)
    else:
        selected = pd.Series(bool(selected), index=rows.index)
    return rows.loc[selected].copy()


def _schedule_rows(run_dir: Path, arm: str) -> pd.DataFrame:
    schedules = _read_csv(run_dir / "size_action_schedules.csv")
    if schedules.empty or "arm" not in schedules.columns:
        return pd.DataFrame()
    rows = schedules.loc[schedules["arm"].eq(arm)].copy()
    if rows.empty:
        return rows
    if "timestamp" in rows.columns:
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    if "strategy_id" in rows.columns:
        rows["strategy_id"] = rows["strategy_id"].astype(str)
    return rows


def _summarize_bottlenecks(run_dir: Path, arms: set[str] | None = None) -> pd.DataFrame:
    transfer = _read_transfer_diagnostics(run_dir)
    if transfer.empty or "arm" not in transfer.columns:
        return pd.DataFrame()
    if "timestamp" in transfer.columns:
        transfer["timestamp"] = pd.to_datetime(transfer["timestamp"], utc=True, errors="coerce")
    if "strategy_id" in transfer.columns:
        transfer["strategy_id"] = transfer["strategy_id"].astype(str)

    key_cols = ["fold_id", "timestamp", "strategy_id"]
    oracle_arms = [a for a in transfer["arm"].dropna().unique() if str(a).startswith("C1_exact_state_oracle_full")]
    if not oracle_arms:
        return pd.DataFrame()
    oracle = _active_selected_rows(transfer, oracle_arms[0])
    if oracle.empty:
        return pd.DataFrame()
    oracle["oracle_positive"] = pd.to_numeric(oracle.get("selected_delta_full_J"), errors="coerce").fillna(0.0) > 0.0
    oracle = oracle.loc[oracle["oracle_positive"], key_cols + ["selected_delta_full_J"]].rename(
        columns={"selected_delta_full_J": "oracle_delta_full_J"}
    )
    oracle_index = pd.MultiIndex.from_frame(oracle[key_cols])

    candidate_arms = [a for a in transfer["arm"].dropna().unique() if not str(a).startswith(("C0_", "C1_"))]
    if arms:
        candidate_arms = [a for a in candidate_arms if a in arms]

    rows: list[dict[str, Any]] = []
    for arm in candidate_arms:
        selected = _active_selected_rows(transfer, arm)
        selected_index = pd.MultiIndex.from_frame(selected[key_cols]) if not selected.empty else pd.MultiIndex.from_arrays([[], [], []])
        selected_delta = pd.to_numeric(selected.get("selected_delta_full_J"), errors="coerce").fillna(0.0)
        selected_positive = selected.loc[selected_delta > 0.0].copy()
        selected_false = selected.loc[selected_delta <= 0.0].copy()

        missed = oracle.loc[~oracle_index.isin(selected_index)].copy()
        missed_gain = pd.to_numeric(missed.get("oracle_delta_full_J"), errors="coerce").fillna(0.0)
        sched = _schedule_rows(run_dir, arm)
        if not sched.empty and not missed.empty:
            sched_cols = key_cols + [
                c
                for c in [
                    "p_intervene",
                    "pred_delta_J",
                    "pred_multiplier",
                    "pred_multiplier_confidence",
                    "eligible_action",
                    "multiplier",
                    "nonoverlap_suppressed",
                ]
                if c in sched.columns
            ]
            missed = missed.merge(sched[sched_cols], on=key_cols, how="left")
        for col in ["p_intervene", "pred_delta_J", "pred_multiplier", "pred_multiplier_confidence"]:
            if col in missed.columns:
                missed[col] = pd.to_numeric(missed[col], errors="coerce")

        miss_n = max(int(len(missed)), 1)
        p_intervene = missed.get("p_intervene", pd.Series(np.nan, index=missed.index))
        pred_multiplier = missed.get("pred_multiplier", pd.Series(np.nan, index=missed.index))
        pred_conf = missed.get("pred_multiplier_confidence", pd.Series(np.nan, index=missed.index))
        pred_delta = missed.get("pred_delta_J", pd.Series(np.nan, index=missed.index))
        nonoverlap = missed.get("nonoverlap_suppressed", pd.Series(False, index=missed.index))
        if isinstance(nonoverlap, pd.Series):
            nonoverlap = nonoverlap.map(lambda value: bool(value) if pd.notna(value) else False)

        rows.append(
            {
                "run": _run_label(run_dir),
                "arm": arm,
                "oracle_positive_groups": int(len(oracle)),
                "selected_groups": int(len(selected)),
                "selected_positive_groups": int(len(selected_positive)),
                "selected_false_groups": int(len(selected_false)),
                "selected_precision": float(len(selected_positive) / len(selected)) if len(selected) else 0.0,
                "selected_positive_delta_sum": float(
                    pd.to_numeric(selected_positive.get("selected_delta_full_J"), errors="coerce").fillna(0.0).sum()
                ),
                "selected_false_delta_sum": float(
                    pd.to_numeric(selected_false.get("selected_delta_full_J"), errors="coerce").fillna(0.0).sum()
                ),
                "missed_oracle_groups": int(len(missed)),
                "missed_oracle_gain_sum": float(missed_gain.sum()),
                "missed_oracle_gain_median": float(missed_gain.median()) if len(missed_gain) else 0.0,
                "missed_stage1_p_median": float(p_intervene.median(skipna=True)) if len(missed) else np.nan,
                "missed_stage1_low_share_p_lt_0_40": float((p_intervene.fillna(0.0) < 0.40).sum() / miss_n),
                "missed_stage1_very_low_share_p_lt_0_10": float((p_intervene.fillna(0.0) < 0.10).sum() / miss_n),
                "missed_pred_nonbaseline_share": float((pred_multiplier.fillna(1.0) < 1.0).sum() / miss_n),
                "missed_multiplier_conf_low_share_lt_0_90": float((pred_conf.fillna(0.0) < 0.90).sum() / miss_n),
                "missed_pred_value_positive_share": float((pred_delta.fillna(np.inf) > 0.0).sum() / miss_n),
                "missed_nonoverlap_suppressed_share": float(nonoverlap.sum() / miss_n),
            }
        )
    return pd.DataFrame(rows)


def _summarize_transfer_reasons(run_dir: Path, arms: set[str] | None = None) -> pd.DataFrame:
    transfer = _read_transfer_diagnostics(run_dir)
    if transfer.empty or "arm" not in transfer.columns or "selector_transfer_reason" not in transfer.columns:
        return pd.DataFrame()
    if arms:
        transfer = transfer[transfer["arm"].isin(arms)].copy()
    if transfer.empty:
        return pd.DataFrame()
    work = transfer.copy()
    work["selector_transfer_reason"] = work["selector_transfer_reason"].fillna("unknown").astype(str)
    for col in ["y_intervene", "oracle_best_delta_full_J", "selected_delta_full_J"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
        else:
            work[col] = 0.0
    selected = work.get("selected")
    if isinstance(selected, pd.Series):
        selected_mask = selected.fillna(False).astype(bool)
    else:
        selected_mask = pd.Series(False, index=work.index)
    work["strict_oracle_miss"] = work["y_intervene"].gt(0.0) & ~selected_mask
    out = work.groupby(["arm", "selector_transfer_reason"], dropna=False).agg(
        rows=("selector_transfer_reason", "size"),
        strict_oracle_misses=("strict_oracle_miss", "sum"),
        oracle_best_delta_full_J_sum=("oracle_best_delta_full_J", "sum"),
        selected_delta_full_J_sum=("selected_delta_full_J", "sum"),
    ).reset_index()
    out.insert(0, "run", _run_label(run_dir))
    return out


def _summarize_oracle_miss_score_diagnostics(run_dir: Path, arms: set[str] | None = None) -> pd.DataFrame:
    """Compare score distributions for selected and missed oracle groups.

    The aggregate bottleneck report is useful for quick triage, but it does
    not show which score family rejects the missed oracle opportunities. This
    summary keeps the comparison at the strategy/timestamp group level and
    separates three economically different populations:

    * selected positives: interventions that helped;
    * selected false cuts: interventions that hurt or did not help;
    * missed oracle: exact-state oracle-positive groups the policy skipped.
    """

    transfer = _read_transfer_diagnostics(run_dir)
    if transfer.empty or "arm" not in transfer.columns:
        return pd.DataFrame()
    if "timestamp" in transfer.columns:
        transfer["timestamp"] = pd.to_datetime(transfer["timestamp"], utc=True, errors="coerce")
    if "strategy_id" in transfer.columns:
        transfer["strategy_id"] = transfer["strategy_id"].astype(str)

    key_cols = ["fold_id", "timestamp", "strategy_id"]
    if any(col not in transfer.columns for col in key_cols):
        return pd.DataFrame()

    oracle_arms = [a for a in transfer["arm"].dropna().unique() if str(a).startswith("C1_exact_state_oracle_full")]
    if not oracle_arms:
        return pd.DataFrame()
    oracle = _active_selected_rows(transfer, oracle_arms[0])
    if oracle.empty:
        return pd.DataFrame()
    oracle["oracle_delta_full_J"] = pd.to_numeric(oracle.get("selected_delta_full_J"), errors="coerce").fillna(0.0)
    oracle = oracle.loc[oracle["oracle_delta_full_J"] > 0.0, key_cols + ["oracle_delta_full_J"]].copy()
    if oracle.empty:
        return pd.DataFrame()
    oracle_index = pd.MultiIndex.from_frame(oracle[key_cols])

    candidate_arms = [a for a in transfer["arm"].dropna().unique() if not str(a).startswith(("C0_", "C1_"))]
    if arms:
        candidate_arms = [a for a in candidate_arms if a in arms]

    score_cols = [
        "selection_score",
        "p_intervene",
        "pred_delta_J",
        "cal_mean_delta_J",
        "cal_lcb_mean_delta_J",
        "cal_q25_delta_J",
        "cal_positive_rate",
        "p_action_positive",
        "ranker_score",
        "ranker_score_margin",
        "pred_delta_margin",
        "eligible_action",
    ]
    rows: list[dict[str, Any]] = []

    def append_summary(scope: str, arm: str, label: str, frame: pd.DataFrame, delta_col: str) -> None:
        if frame.empty:
            return
        row: dict[str, Any] = {
            "run": _run_label(run_dir),
            "scope": scope,
            "arm": arm,
            "group": label,
            "groups": int(len(frame)),
        }
        delta = pd.to_numeric(frame.get(delta_col), errors="coerce").fillna(0.0)
        row.update(
            {
                "delta_full_J_sum": float(delta.sum()),
                "delta_full_J_median": float(delta.median()) if len(delta) else 0.0,
                "delta_full_J_q25": float(delta.quantile(0.25)) if len(delta) else 0.0,
            }
        )
        for col in score_cols:
            if col not in frame.columns:
                continue
            values = pd.to_numeric(frame[col], errors="coerce").astype(float)
            finite = values[np.isfinite(values)]
            prefix = f"{col}"
            row[f"{prefix}_coverage"] = float(len(finite) / max(len(frame), 1))
            if len(finite):
                row[f"{prefix}_mean"] = float(finite.mean())
                row[f"{prefix}_q10"] = float(finite.quantile(0.10))
                row[f"{prefix}_median"] = float(finite.median())
                row[f"{prefix}_q90"] = float(finite.quantile(0.90))
            else:
                row[f"{prefix}_mean"] = np.nan
                row[f"{prefix}_q10"] = np.nan
                row[f"{prefix}_median"] = np.nan
                row[f"{prefix}_q90"] = np.nan
        rows.append(row)

    for arm in candidate_arms:
        arm_rows = transfer.loc[transfer["arm"].eq(arm)].copy()
        if arm_rows.empty:
            continue
        selected = _active_selected_rows(transfer, arm)
        selected_delta = (
            pd.to_numeric(selected["selected_delta_full_J"], errors="coerce").fillna(0.0)
            if "selected_delta_full_J" in selected.columns
            else pd.Series(0.0, index=selected.index, dtype=float)
        )
        selected_positive = selected.loc[selected_delta > 0.0].copy()
        selected_false = selected.loc[selected_delta <= 0.0].copy()

        selected_index = pd.MultiIndex.from_frame(selected[key_cols]) if not selected.empty else pd.MultiIndex.from_arrays([[], [], []])
        missed = oracle.loc[~oracle_index.isin(selected_index)].copy()
        if not missed.empty:
            arm_score_cols = [c for c in key_cols + score_cols if c in arm_rows.columns]
            missed = missed.merge(arm_rows[arm_score_cols].drop_duplicates(key_cols), on=key_cols, how="left")

        append_summary("all", arm, "selected_positive", selected_positive, "selected_delta_full_J")
        append_summary("all", arm, "selected_false", selected_false, "selected_delta_full_J")
        append_summary("all", arm, "missed_oracle", missed, "oracle_delta_full_J")

        for fold_id, fold_rows in arm_rows.groupby("fold_id", dropna=False):
            fold_selected = selected.loc[selected["fold_id"].eq(fold_id)].copy() if not selected.empty else pd.DataFrame()
            fold_delta = (
                pd.to_numeric(fold_selected["selected_delta_full_J"], errors="coerce").fillna(0.0)
                if "selected_delta_full_J" in fold_selected.columns
                else pd.Series(0.0, index=fold_selected.index, dtype=float)
            )
            fold_selected_positive = fold_selected.loc[fold_delta > 0.0].copy()
            fold_selected_false = fold_selected.loc[fold_delta <= 0.0].copy()
            fold_missed = missed.loc[missed["fold_id"].eq(fold_id)].copy() if not missed.empty else pd.DataFrame()
            scope = f"fold_{fold_id}"
            append_summary(scope, arm, "selected_positive", fold_selected_positive, "selected_delta_full_J")
            append_summary(scope, arm, "selected_false", fold_selected_false, "selected_delta_full_J")
            append_summary(scope, arm, "missed_oracle", fold_missed, "oracle_delta_full_J")

    return pd.DataFrame(rows)


def _summarize_promotion_gates(run_dir: Path, arms: set[str] | None = None) -> pd.DataFrame:
    """Summarize sparse-action promotion gates from existing run artifacts.

    The size-action learner is meant to be a sparse override, not a continuous
    allocator. These gates make that contract explicit so a high aggregate PnL
    does not hide broad exposure cuts, false interventions, or poor fold
    robustness.
    """

    quality = _summarize_quality(run_dir, arms)
    replay = _summarize_replay(run_dir, arms)
    bottlenecks = _summarize_bottlenecks(run_dir, arms)
    binding = _summarize_binding_opportunities(run_dir)
    promotion = _read_csv(run_dir / "size_action_promotion_summary.csv")
    deciles = _read_csv(run_dir / "size_action_predicted_benefit_deciles.csv")
    fold_quality = _read_csv(run_dir / "size_action_action_quality.csv")

    if quality.empty and replay.empty:
        return pd.DataFrame()

    work = quality.merge(
        replay,
        on=["run", "arm"],
        how="outer",
        suffixes=("_quality", "_replay"),
    )
    if not bottlenecks.empty:
        keep = [
            "run",
            "arm",
            "selected_false_groups",
            "selected_false_delta_sum",
            "missed_oracle_groups",
            "missed_oracle_gain_sum",
        ]
        work = work.merge(bottlenecks[[c for c in keep if c in bottlenecks.columns]], on=["run", "arm"], how="left")

    if not binding.empty:
        binding_total = binding.groupby("run", as_index=False).agg(
            binding_opportunity_groups=("binding_opportunity_groups", "sum"),
            scheduled_groups_from_panel=("scheduled_groups_from_panel", "sum"),
        )
        work = work.merge(binding_total, on="run", how="left")

    if not promotion.empty and "arm" in promotion.columns:
        promo = promotion.copy()
        promo.insert(0, "run", _run_label(run_dir))
        keep = [
            "run",
            "arm",
            "median_exposure_ratio",
            "median_multiplier",
            "positive_delta_net_pnl_share",
        ]
        work = work.merge(promo[[c for c in keep if c in promo.columns]], on=["run", "arm"], how="left")

    if not fold_quality.empty and "arm" in fold_quality.columns:
        fq = fold_quality.copy()
        if arms:
            fq = fq[fq["arm"].isin(arms)].copy()
        for col in ["intervention_count", "positive_action_count", "realized_delta_full_J_sum"]:
            fq[col] = pd.to_numeric(fq.get(col), errors="coerce").fillna(0.0)
        fq_active = fq.loc[fq["intervention_count"] > 0.0].copy()
        if not fq_active.empty:
            fq_active["fold_precision"] = fq_active["positive_action_count"] / fq_active["intervention_count"].replace(0.0, np.nan)
            active = fq_active.groupby("arm", dropna=False).agg(
                active_intervention_folds=("fold_id", "nunique"),
                min_active_fold_precision=("fold_precision", "min"),
                active_folds_positive_delta=("realized_delta_full_J_sum", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > MATERIAL_DELTA_EPS).sum())),
            ).reset_index()
            active.insert(0, "run", _run_label(run_dir))
            work = work.merge(active, on=["run", "arm"], how="left")

    if not deciles.empty and "arm" in deciles.columns:
        dd = deciles.copy()
        if arms:
            dd = dd[dd["arm"].isin(arms)].copy()
        if "score_column" not in dd.columns:
            dd["score_column"] = "score"
        for col in ["score_bucket", "mean_delta_full_J", "positive_delta_share", "rows"]:
            dd[col] = pd.to_numeric(dd.get(col), errors="coerce")
        top = dd.sort_values(["arm", "fold_id", "score_column", "score_bucket"]).groupby(
            ["arm", "fold_id", "score_column"], dropna=False
        ).tail(1)
        top_by_score = top.groupby(["arm", "score_column"], dropna=False).agg(
            top_bucket_rows=("rows", "sum"),
            top_bucket_mean_delta_full_J=("mean_delta_full_J", "mean"),
            top_bucket_positive_delta_share=("positive_delta_share", "mean"),
            top_bucket_positive_folds=("mean_delta_full_J", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > MATERIAL_DELTA_EPS).sum())),
        ).reset_index()
        top_by_score = top_by_score.sort_values(
            ["arm", "top_bucket_mean_delta_full_J", "top_bucket_positive_folds"],
            ascending=[True, False, False],
        )
        top_summary = top_by_score.groupby("arm", dropna=False).head(1).copy()
        top_summary = top_summary.rename(columns={"score_column": "top_bucket_score_column"})
        top_summary.insert(0, "run", _run_label(run_dir))
        work = work.merge(top_summary, on=["run", "arm"], how="left")

    defaults = {
        "interventions": 0.0,
        "scheduled_groups": 0.0,
        "binding_opportunity_groups": 0.0,
        "scheduled_groups_from_panel": 0.0,
        "precision_total": 0.0,
        "positive_delta_folds": 0.0,
        "folds_replay": np.nan,
        "folds_quality": np.nan,
        "q25_delta_net_pnl": 0.0,
        "median_delta_net_pnl": 0.0,
        "delta_net_pnl_sum": 0.0,
        "mean_multiplier": 1.0,
        "median_exposure_ratio": 1.0,
        "selected_false_delta_sum": 0.0,
        "oracle_gain_capture_ratio_sum": 0.0,
        "active_intervention_folds": 0.0,
        "min_active_fold_precision": 0.0,
        "top_bucket_mean_delta_full_J": 0.0,
        "top_bucket_positive_folds": 0.0,
    }
    for col, default in defaults.items():
        if col not in work.columns:
            work[col] = default
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)

    fold_count = work["folds_replay"].where(work["folds_replay"] > 0, work["folds_quality"])
    work["fold_count_for_gates"] = fold_count.fillna(0.0)
    work["intervention_rate_total"] = work["interventions"] / work["scheduled_groups"].replace(0.0, np.nan)
    work["intervention_rate_total"] = work["intervention_rate_total"].fillna(0.0)
    work["binding_intervention_rate_total"] = work["interventions"] / work["binding_opportunity_groups"].replace(0.0, np.nan)
    work["binding_intervention_rate_total"] = work["binding_intervention_rate_total"].fillna(work["intervention_rate_total"])
    work["oracle_capture_per_fold"] = work["oracle_gain_capture_ratio_sum"] / work["fold_count_for_gates"].replace(0.0, np.nan)
    work["oracle_capture_per_fold"] = work["oracle_capture_per_fold"].fillna(0.0)

    work["gate_sparse_intervention_rate"] = work["binding_intervention_rate_total"].between(0.0001, 0.15)
    work["gate_target_intervention_band_5_15pct"] = work["binding_intervention_rate_total"].between(0.05, 0.15)
    work["gate_exposure_retained"] = (work["median_exposure_ratio"] >= 0.98) | (work["mean_multiplier"] >= 0.98)
    work["gate_precision_total"] = work["precision_total"] >= 0.98
    work["gate_no_false_selected_delta"] = work["selected_false_delta_sum"] >= -1e-9
    work["gate_median_pnl_positive"] = work["median_delta_net_pnl"] > 0.0
    work["gate_q25_pnl_positive"] = work["q25_delta_net_pnl"] > 0.0
    work["gate_positive_most_folds"] = work["positive_delta_folds"] >= np.ceil(0.5 * work["fold_count_for_gates"].clip(lower=1.0))
    work["gate_active_fold_precision"] = (work["active_intervention_folds"] <= 0.0) | (work["min_active_fold_precision"] >= 0.98)
    work["gate_top_bucket_positive"] = work["top_bucket_mean_delta_full_J"] > MATERIAL_DELTA_EPS
    work["gate_oracle_capture_nonzero"] = work["oracle_capture_per_fold"] > 0.0

    gate_cols = [c for c in work.columns if c.startswith("gate_")]
    work["passed_gate_count"] = work[gate_cols].sum(axis=1).astype(int)
    work["failed_gates"] = work[gate_cols].apply(
        lambda row: ",".join(col.removeprefix("gate_") for col, passed in row.items() if not bool(passed)),
        axis=1,
    )
    work["promotion_ready"] = work[
        [
            "gate_sparse_intervention_rate",
            "gate_target_intervention_band_5_15pct",
            "gate_exposure_retained",
            "gate_precision_total",
            "gate_no_false_selected_delta",
            "gate_median_pnl_positive",
            "gate_q25_pnl_positive",
            "gate_active_fold_precision",
            "gate_top_bucket_positive",
        ]
    ].all(axis=1)
    return work


def _summarize_stage1_recall_tradeoff(run_dir: Path) -> pd.DataFrame:
    scores = _read_csv(run_dir / "size_action_stage1_group_scores.csv")
    if scores.empty:
        return pd.DataFrame()
    required = {"fold_id", "y_intervene", "best_gain"}
    if not required.issubset(scores.columns):
        return pd.DataFrame()
    work = scores.copy()
    for col in ["y_intervene", "best_gain", "group_can_bind"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    if "group_can_bind" in work.columns:
        work = work.loc[work["group_can_bind"] > 0.0].copy()
    if work.empty:
        return pd.DataFrame()

    score_cols = [
        "p_intervene",
        "bag_p_mean",
        "stage1_cal_positive_rate",
        "stage1_cal_mean_gain",
        "stage1_cal_lcb_mean_gain",
    ]
    score_cols = [c for c in score_cols if c in work.columns]
    if not score_cols:
        return pd.DataFrame()
    for col in score_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    total_positive = int((work["y_intervene"] > 0.0).sum())
    total_gain = float(work.loc[work["y_intervene"] > 0.0, "best_gain"].clip(lower=0.0).sum())
    rows: list[dict[str, Any]] = []
    top_fractions = [0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    for score_col in score_cols:
        for frac in top_fractions:
            selected_parts: list[pd.DataFrame] = []
            for _, fold_rows in work.groupby("fold_id", dropna=False):
                n = max(1, int(round(len(fold_rows) * float(frac))))
                selected_parts.append(fold_rows.sort_values(score_col, ascending=False).head(n))
            selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
            selected_count = int(len(selected))
            positives = int((selected.get("y_intervene", pd.Series(dtype=float)) > 0.0).sum()) if selected_count else 0
            selected_gain = float(selected.loc[selected.get("y_intervene", pd.Series(False, index=selected.index)) > 0.0, "best_gain"].clip(lower=0.0).sum()) if selected_count else 0.0
            rows.append(
                {
                    "run": _run_label(run_dir),
                    "score_column": score_col,
                    "top_fraction": float(frac),
                    "eligible_groups": int(len(work)),
                    "selected_groups": selected_count,
                    "total_positive_groups": total_positive,
                    "selected_positive_groups": positives,
                    "precision": float(positives / max(selected_count, 1)),
                    "recall": float(positives / max(total_positive, 1)),
                    "total_positive_gain": total_gain,
                    "selected_positive_gain": selected_gain,
                    "gain_capture": float(selected_gain / max(total_gain, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def _calibrated_group_hurdle_score(frame: pd.DataFrame) -> pd.Series:
    def col(name: str, default: float = 0.0) -> pd.Series:
        raw = frame[name] if name in frame.columns else pd.Series(default, index=frame.index)
        return pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

    return (
        (1.0 + col("p_intervene").clip(lower=0.0))
        * (0.25 + col("stage1_cal_positive_rate").clip(lower=0.0))
        * (0.25 + col("p_action_value_positive").clip(lower=0.0))
        * (0.25 + col("p_action_economic_positive").clip(lower=0.0))
        * (0.25 + col("cal_positive_rate").clip(lower=0.0))
        * (1.0 + col("ranker_score").clip(lower=0.0))
        * (1.0 + col("pred_delta_J").clip(lower=0.0) / 100.0)
    )


def _summarize_raw_candidate_recall(run_dir: Path) -> pd.DataFrame:
    """Audit whether raw eval action scores could have proposed missed oracle groups.

    This is intentionally independent of a specific arm. If raw non-baseline,
    binding candidates have useful score/risk characteristics for strict
    oracle groups, the next work should improve the secondary generator. If
    they do not, the bottleneck is upstream model scoring rather than schedule
    thresholding.
    """

    actions = _read_csv(run_dir / "size_action_eval_action_scores.csv")
    if actions.empty:
        return pd.DataFrame()
    required = {"fold_id", "timestamp", "strategy_id", "multiplier"}
    if not required.issubset(actions.columns):
        return pd.DataFrame()
    work = actions.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["strategy_id"] = work["strategy_id"].astype(str)
    for col in [
        "multiplier",
        "action_binds",
        "group_can_bind",
        "group_can_bind_x",
        "group_can_bind_y",
        "y_intervene",
        "y_intervene_x",
        "y_intervene_y",
        "best_gain",
        "best_gain_x",
        "best_gain_y",
        "pred_delta_J",
        "cal_mean_delta_J",
        "cal_lcb_mean_delta_J",
        "cal_q10_delta_J",
        "cal_q25_delta_J",
        "cal_positive_rate",
        "p_action_value_positive",
        "p_action_positive",
        "p_action_economic_positive",
        "p_intervene",
        "ranker_score",
        "strategy_rank_q90",
        "strategy_open_count",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "group_can_bind" not in work.columns:
        work["group_can_bind"] = work.get("group_can_bind_y", work.get("group_can_bind_x", 0.0))
    if "y_intervene" not in work.columns:
        work["y_intervene"] = work.get("y_intervene_y", work.get("y_intervene_x", 0.0))
    if "best_gain" not in work.columns:
        work["best_gain"] = work.get("best_gain_y", work.get("best_gain_x", 0.0))
    for col in ["p_intervene", "stage1_cal_positive_rate", "stage1_cal_mean_gain", "stage1_cal_lcb_mean_gain"]:
        if col not in work.columns:
            work[col] = 0.0
    if "stage1_cal_positive_rate" not in actions.columns or "stage1_cal_mean_gain" not in actions.columns:
        groups = _read_csv(run_dir / "size_action_stage1_group_scores.csv")
        if not groups.empty and {"fold_id", "timestamp", "strategy_id"}.issubset(groups.columns):
            groups = groups.copy()
            groups["timestamp"] = pd.to_datetime(groups["timestamp"], utc=True, errors="coerce")
            groups["strategy_id"] = groups["strategy_id"].astype(str)
            keep = [
                c
                for c in [
                    "fold_id",
                    "timestamp",
                    "strategy_id",
                    "p_intervene",
                    "stage1_cal_positive_rate",
                    "stage1_cal_mean_gain",
                    "stage1_cal_lcb_mean_gain",
                    "bag_vote_share",
                    "bag_p_mean",
                    "bag_p_std",
                ]
                if c in groups.columns
            ]
            suffix_cols = [c for c in keep if c not in {"fold_id", "timestamp", "strategy_id"}]
            work = work.drop(columns=[c for c in suffix_cols if c in work.columns], errors="ignore").merge(
                groups[keep].drop_duplicates(["fold_id", "timestamp", "strategy_id"]),
                on=["fold_id", "timestamp", "strategy_id"],
                how="left",
            )
            for col in suffix_cols:
                work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work["calibrated_group_hurdle_score"] = _calibrated_group_hurdle_score(work)
    nonbase = work.loc[
        (pd.to_numeric(work["multiplier"], errors="coerce").fillna(1.0) < 1.0)
        & (pd.to_numeric(work.get("group_can_bind"), errors="coerce").fillna(0.0) > 0.0)
        & (pd.to_numeric(work.get("action_binds"), errors="coerce").fillna(0.0) > 0.0)
    ].copy()
    if nonbase.empty:
        return pd.DataFrame()

    key_cols = ["fold_id", "timestamp", "strategy_id"]
    score_cols = [
        "calibrated_group_hurdle_score",
        "p_intervene",
        "stage1_cal_positive_rate",
        "stage1_cal_mean_gain",
        "stage1_cal_lcb_mean_gain",
        "pred_delta_J",
        "cal_mean_delta_J",
        "cal_lcb_mean_delta_J",
        "cal_q10_delta_J",
        "cal_q25_delta_J",
        "cal_positive_rate",
        "p_action_value_positive",
        "p_action_positive",
        "p_action_economic_positive",
        "ranker_score",
        "strategy_rank_q90",
    ]
    score_cols = [c for c in score_cols if c in nonbase.columns]
    best = (
        nonbase.sort_values(key_cols + ["calibrated_group_hurdle_score", "pred_delta_J", "multiplier"], ascending=[True, True, True, False, False, False])
        .drop_duplicates(key_cols)
        .copy()
    )
    best["strict_oracle"] = pd.to_numeric(best.get("y_intervene"), errors="coerce").fillna(0.0) > 0.0
    best["positive_value_floor_pass"] = (
        (pd.to_numeric(best.get("pred_delta_J"), errors="coerce").fillna(0.0) >= 0.0)
        & (pd.to_numeric(best.get("cal_mean_delta_J"), errors="coerce").fillna(0.0) >= 0.0)
    )
    best["rank_pressure_pass"] = (
        (pd.to_numeric(best.get("strategy_rank_q90"), errors="coerce").fillna(0.0) >= 0.70)
        & (pd.to_numeric(best.get("strategy_open_count"), errors="coerce").fillna(0.0) <= 2.0)
    )
    def best_col(name: str, default: float = 0.0) -> pd.Series:
        raw = best[name] if name in best.columns else pd.Series(default, index=best.index)
        return pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

    best["action_positive_pass"] = (
        (best_col("p_action_positive") >= 0.50)
        | (best_col("p_action_value_positive") >= 0.50)
    )

    rows: list[dict[str, Any]] = []
    for label, frame in [
        ("strict_oracle", best.loc[best["strict_oracle"]].copy()),
        ("non_oracle", best.loc[~best["strict_oracle"]].copy()),
    ]:
        if frame.empty:
            continue
        row: dict[str, Any] = {
            "run": _run_label(run_dir),
            "group": label,
            "groups": int(len(frame)),
            "best_gain_sum": float(pd.to_numeric(frame.get("best_gain"), errors="coerce").fillna(0.0).sum()),
            "positive_value_floor_pass_share": float(frame["positive_value_floor_pass"].mean()),
            "rank_pressure_pass_share": float(frame["rank_pressure_pass"].mean()),
            "action_positive_pass_share": float(frame["action_positive_pass"].mean()),
        }
        for col in score_cols:
            values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            row[f"{col}_median"] = float(values.median())
            row[f"{col}_q75"] = float(values.quantile(0.75))
            row[f"{col}_q90"] = float(values.quantile(0.90))
        rows.append(row)

    top_fractions = [0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    total_oracle = int(best["strict_oracle"].sum())
    total_oracle_gain = float(pd.to_numeric(best.loc[best["strict_oracle"], "best_gain"], errors="coerce").fillna(0.0).sum())
    for score_col in score_cols:
        for frac in top_fractions:
            selected_parts: list[pd.DataFrame] = []
            for _, fold_rows in best.groupby("fold_id", dropna=False):
                n = max(1, int(np.ceil(len(fold_rows) * float(frac))))
                selected_parts.append(fold_rows.sort_values(score_col, ascending=False).head(n))
            selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
            if selected.empty:
                continue
            selected_oracle = selected.loc[selected["strict_oracle"]].copy()
            selected_delta_raw = selected["delta_full_J"] if "delta_full_J" in selected.columns else pd.Series(0.0, index=selected.index)
            selected_delta = pd.to_numeric(selected_delta_raw, errors="coerce").fillna(0.0)
            rows.append(
                {
                    "run": _run_label(run_dir),
                    "group": f"top_{frac:g}_by_{score_col}",
                    "groups": int(len(selected)),
                    "selected_oracle_groups": int(len(selected_oracle)),
                    "total_oracle_groups": total_oracle,
                    "oracle_recall": float(len(selected_oracle) / max(total_oracle, 1)),
                    "oracle_precision": float(len(selected_oracle) / max(len(selected), 1)),
                    "selected_delta_full_J_sum": float(selected_delta.sum()),
                    "selected_delta_full_J_mean": float(selected_delta.mean()) if len(selected_delta) else 0.0,
                    "selected_delta_positive_share": float((selected_delta > MATERIAL_DELTA_EPS).mean()) if len(selected_delta) else 0.0,
                    "selected_oracle_gain": float(pd.to_numeric(selected_oracle.get("best_gain"), errors="coerce").fillna(0.0).sum()),
                    "total_oracle_gain": total_oracle_gain,
                    "oracle_gain_capture": float(
                        pd.to_numeric(selected_oracle.get("best_gain"), errors="coerce").fillna(0.0).sum() / max(total_oracle_gain, 1.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _manifest_rows(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {"run": _run_label(run_dir)}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"run": _run_label(run_dir)}
    return {
        "run": _run_label(run_dir),
        "fold_count": manifest.get("fold_count"),
        "max_train_timestamps": manifest.get("max_train_timestamps"),
        "max_eval_timestamps": manifest.get("max_eval_timestamps"),
        "epsilon_gain": manifest.get("epsilon_gain"),
        "epsilon_margin": manifest.get("epsilon_margin"),
        "panel_cache": manifest.get("panel_cache"),
        "requested_arms": manifest.get("requested_arms"),
    }


def _write_markdown(
    path: Path,
    manifests: pd.DataFrame,
    quality: pd.DataFrame,
    replay: pd.DataFrame,
    thresholds: pd.DataFrame,
    bottlenecks: pd.DataFrame,
    transfer_reasons: pd.DataFrame,
    promotion_gates: pd.DataFrame,
    stage1_recall: pd.DataFrame,
    raw_candidate_recall: pd.DataFrame,
    oracle_miss_scores: pd.DataFrame,
) -> None:
    lines: list[str] = ["# Size-Action Learning Run Comparison", ""]
    if not manifests.empty:
        lines.extend(["## Runs", "", manifests.to_markdown(index=False), ""])
    if not quality.empty:
        display_cols = [
            "run",
            "arm",
            "folds",
            "interventions",
            "positive_actions",
            "precision_total",
            "realized_delta_full_J_sum",
            "realized_delta_immediate_J_sum",
            "positive_folds",
            "mean_oracle_capture",
            "oracle_gain_capture_ratio_sum",
        ]
        display_cols = [c for c in display_cols if c in quality.columns]
        lines.extend(["## Action Quality", "", quality[display_cols].to_markdown(index=False), ""])
    if not replay.empty:
        replay_cols = [
            "run",
            "arm",
            "folds",
            "delta_net_pnl_sum",
            "median_delta_net_pnl",
            "q25_delta_net_pnl",
            "positive_delta_folds",
            "delta_trade_count_sum",
            "mean_delta_full_sl_rate",
            "mean_multiplier",
            "min_multiplier",
        ]
        replay_cols = [c for c in replay_cols if c in replay.columns]
        lines.extend(["## Sequential Replay Versus C0", "", replay[replay_cols].to_markdown(index=False), ""])
    if not thresholds.empty:
        threshold_cols = [
            "run",
            "fold_id",
            "arm",
            "threshold_source",
            "train_interventions",
            "train_precision",
            "train_delta_full_J_sum",
            "diagnostic_precondition_thresholds",
            "diagnostic_veto_thresholds",
        ]
        threshold_cols = [c for c in threshold_cols if c in thresholds.columns]
        lines.extend(["## Thresholds", "", thresholds[threshold_cols].to_markdown(index=False), ""])
    if not bottlenecks.empty:
        bottleneck_cols = [
            "run",
            "arm",
            "oracle_positive_groups",
            "selected_groups",
            "selected_positive_groups",
            "selected_false_groups",
            "selected_precision",
            "missed_oracle_groups",
            "missed_oracle_gain_sum",
            "missed_stage1_p_median",
            "missed_stage1_low_share_p_lt_0_40",
            "missed_multiplier_conf_low_share_lt_0_90",
            "missed_pred_value_positive_share",
            "missed_nonoverlap_suppressed_share",
        ]
        bottleneck_cols = [c for c in bottleneck_cols if c in bottlenecks.columns]
        lines.extend(
            [
                "## Missed Oracle Bottlenecks",
                "",
                bottlenecks[bottleneck_cols].to_markdown(index=False),
                "",
                "A high `missed_stage1_low_share_p_lt_0_40` means recall is blocked by the sparse intervention classifier, not just by the final action gate.",
                "",
            ]
        )
    if not transfer_reasons.empty:
        reason_cols = [
            "run",
            "arm",
            "selector_transfer_reason",
            "rows",
            "strict_oracle_misses",
            "oracle_best_delta_full_J_sum",
            "selected_delta_full_J_sum",
        ]
        reason_cols = [c for c in reason_cols if c in transfer_reasons.columns]
        lines.extend(
            [
                "## Selector Transfer Reasons",
                "",
                transfer_reasons[reason_cols].to_markdown(index=False),
                "",
                "These reason counts explain whether missed exact-state oracle opportunities were never proposed as secondary candidates, blocked by acceptance gates, blocked by positive-value floors, or selected with positive/nonpositive realized value.",
                "",
            ]
        )
    if not oracle_miss_scores.empty:
        score_cols = [
            "run",
            "scope",
            "arm",
            "group",
            "groups",
            "delta_full_J_sum",
            "p_intervene_median",
            "p_intervene_q90",
            "pred_delta_J_median",
            "cal_q25_delta_J_median",
            "cal_positive_rate_median",
            "p_action_positive_median",
            "ranker_score_median",
            "eligible_action_mean",
        ]
        score_cols = [c for c in score_cols if c in oracle_miss_scores.columns]
        all_scope = oracle_miss_scores.loc[oracle_miss_scores["scope"].eq("all"), score_cols].copy()
        lines.extend(
            [
                "## Oracle Miss Score Diagnostics",
                "",
                all_scope.to_markdown(index=False),
                "",
                "These rows compare selected helpful interventions, selected false interventions, and exact-state oracle-positive groups that the learned policy missed. They are intended to identify whether recall is blocked by Stage-1 probability, action-value calibration, ranker score, or eligibility.",
                "",
            ]
        )
    if not promotion_gates.empty:
        gate_cols = [
            "run",
            "arm",
            "promotion_ready",
            "passed_gate_count",
            "intervention_rate_total",
            "binding_intervention_rate_total",
            "binding_opportunity_groups",
            "precision_total",
            "selected_false_delta_sum",
            "delta_net_pnl_sum",
            "median_delta_net_pnl",
            "q25_delta_net_pnl",
            "median_exposure_ratio",
            "oracle_capture_per_fold",
            "top_bucket_score_column",
            "top_bucket_mean_delta_full_J",
            "failed_gates",
        ]
        gate_cols = [c for c in gate_cols if c in promotion_gates.columns]
        lines.extend(
            [
                "## Promotion Gates",
                "",
                promotion_gates[gate_cols].to_markdown(index=False),
                "",
                "`promotion_ready` is intentionally strict: it requires sparse intervention, retained exposure, no selected false delta, positive median and lower-quartile replay delta, active-fold precision, and positive top predicted-benefit bucket behavior.",
                "",
            ]
        )
    if not stage1_recall.empty:
        best = stage1_recall.sort_values(
            ["run", "top_fraction", "gain_capture", "precision"],
            ascending=[True, True, False, False],
        ).groupby(["run", "top_fraction"], dropna=False).head(1)
        recall_cols = [
            "run",
            "score_column",
            "top_fraction",
            "selected_groups",
            "selected_positive_groups",
            "precision",
            "recall",
            "gain_capture",
        ]
        recall_cols = [c for c in recall_cols if c in best.columns]
        lines.extend(
            [
                "## Stage1 Recall Tradeoff",
                "",
                best[recall_cols].to_markdown(index=False),
                "",
                "This table selects the best available stage1 score column at each top-fraction by gain capture. Low recall here means later gates cannot reach the target intervention band without admitting many false interventions.",
                "",
            ]
        )
    if not raw_candidate_recall.empty:
        summary = raw_candidate_recall.loc[raw_candidate_recall["group"].isin(["strict_oracle", "non_oracle"])].copy()
        summary_cols = [
            "run",
            "group",
            "groups",
            "best_gain_sum",
            "positive_value_floor_pass_share",
            "rank_pressure_pass_share",
            "action_positive_pass_share",
            "calibrated_group_hurdle_score_median",
            "p_intervene_median",
            "pred_delta_J_median",
            "cal_mean_delta_J_median",
        ]
        summary_cols = [c for c in summary_cols if c in summary.columns]
        top = raw_candidate_recall.loc[raw_candidate_recall["group"].astype(str).str.startswith("top_")].copy()
        top_cols = [
            "run",
            "group",
            "groups",
            "selected_oracle_groups",
            "total_oracle_groups",
            "oracle_recall",
            "oracle_precision",
            "selected_delta_full_J_sum",
            "selected_delta_positive_share",
            "oracle_gain_capture",
        ]
        top_cols = [c for c in top_cols if c in top.columns]
        if not top.empty and "oracle_gain_capture" in top.columns:
            top = top.sort_values(["run", "oracle_gain_capture", "oracle_precision"], ascending=[True, False, False]).groupby("run").head(12)
        lines.extend(["## Raw Candidate Recall", ""])
        if not summary.empty:
            lines.extend([summary[summary_cols].to_markdown(index=False), ""])
        if not top.empty:
            lines.extend(["Best raw score top-fraction recall rows:", "", top[top_cols].to_markdown(index=False), ""])
        lines.extend(
            [
                "This table audits all raw non-baseline binding action scores, before arm-specific schedule selection. If strict-oracle groups score poorly here, the secondary generator needs better Stage-1/value models rather than looser acceptance thresholds.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS), help="Comma-separated arms to include, or 'all'.")
    args = parser.parse_args()

    arms = _resolve_arms(args.arms)
    run_dirs = [p for p in args.run_dirs if p.exists()]
    if not run_dirs:
        raise SystemExit("No existing run directories supplied.")
    out_dir = args.out_dir or (run_dirs[-1] / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = pd.DataFrame([_manifest_rows(p) for p in run_dirs])
    quality = pd.concat([_summarize_quality(p, arms) for p in run_dirs], ignore_index=True)
    replay = pd.concat([_summarize_replay(p, arms) for p in run_dirs], ignore_index=True)
    thresholds = pd.concat([_threshold_summary(p, arms) for p in run_dirs], ignore_index=True)
    bottlenecks = pd.concat([_summarize_bottlenecks(p, arms) for p in run_dirs], ignore_index=True)
    transfer_reasons = pd.concat([_summarize_transfer_reasons(p, arms) for p in run_dirs], ignore_index=True)
    promotion_gates = pd.concat([_summarize_promotion_gates(p, arms) for p in run_dirs], ignore_index=True)
    stage1_recall = pd.concat([_summarize_stage1_recall_tradeoff(p) for p in run_dirs], ignore_index=True)
    raw_candidate_recall = pd.concat([_summarize_raw_candidate_recall(p) for p in run_dirs], ignore_index=True)
    oracle_miss_scores = pd.concat(
        [_summarize_oracle_miss_score_diagnostics(p, arms) for p in run_dirs],
        ignore_index=True,
    )

    manifests.to_csv(out_dir / "size_action_run_manifest_comparison.csv", index=False)
    quality.to_csv(out_dir / "size_action_quality_comparison.csv", index=False)
    replay.to_csv(out_dir / "size_action_replay_comparison.csv", index=False)
    thresholds.to_csv(out_dir / "size_action_threshold_comparison.csv", index=False)
    bottlenecks.to_csv(out_dir / "size_action_bottleneck_comparison.csv", index=False)
    transfer_reasons.to_csv(out_dir / "size_action_transfer_reason_comparison.csv", index=False)
    promotion_gates.to_csv(out_dir / "size_action_promotion_gate_comparison.csv", index=False)
    stage1_recall.to_csv(out_dir / "size_action_stage1_recall_tradeoff_comparison.csv", index=False)
    raw_candidate_recall.to_csv(out_dir / "size_action_raw_candidate_recall_comparison.csv", index=False)
    oracle_miss_scores.to_csv(out_dir / "size_action_oracle_miss_score_diagnostics.csv", index=False)
    _write_markdown(
        out_dir / "size_action_run_comparison.md",
        manifests,
        quality,
        replay,
        thresholds,
        bottlenecks,
        transfer_reasons,
        promotion_gates,
        stage1_recall,
        raw_candidate_recall,
        oracle_miss_scores,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "runs": len(run_dirs),
                "quality_rows": int(len(quality)),
                "replay_rows": int(len(replay)),
                "bottleneck_rows": int(len(bottlenecks)),
                "transfer_reason_rows": int(len(transfer_reasons)),
                "promotion_gate_rows": int(len(promotion_gates)),
                "stage1_recall_rows": int(len(stage1_recall)),
                "raw_candidate_recall_rows": int(len(raw_candidate_recall)),
                "oracle_miss_score_rows": int(len(oracle_miss_scores)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
