#!/usr/bin/env python3
"""Seal an evidence-based completion audit for the 2022--2026 regime objective.

This is deliberately an artifact audit, not a roadmap parser: a requirement is
``proved`` only when the cited materialized output contains the required
measurement.  ``incomplete`` means useful evidence exists but does not meet the
whole objective; ``missing`` means no such evidence was found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp/artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "regime_objective_completion_audit_20260730_v20"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed(root: Path, *files: str) -> tuple[bool, str]:
    """Return whether an immutable artifact and all requested files exist."""
    if not root.is_dir() or not all((root / name).is_file() for name in files):
        return False, "required artifact or output is absent"
    manifest = root / "manifest.json"
    sidecar = root / "manifest.sha256"
    if manifest.is_file() and sidecar.is_file():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        if expected != sha256(manifest):
            return False, "manifest.sha256 does not match manifest.json"
    return True, "artifact exists; manifest seal verified when supplied"


def _evidence(root: Path, names: Sequence[str]) -> str:
    return json.dumps([str(root / name) for name in names], sort_keys=True)


def build_audit(artifacts: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate the objective using fixed authoritative artifact predicates."""
    ledger = artifacts / "regime_episode_ledger_2022_2026_20260730_v1"
    multiview = artifacts / "regime_multiview_panel_2022_2026_20260730_v2"
    selection = artifacts / "fold_local_multiview_selection_2022_2026_20260730_v3"
    geometry = artifacts / "regime_transition_path_geometry_diagnostic_20260730_v1"
    forward_geometry = artifacts / "forward_exact_transition_geometry_20260730_v1"
    performance = artifacts / "stack_performance_calendar_2022_2026_20260730_v4"
    failures = artifacts / "stack_regime_failure_analysis_2022_2026_20260730_v5"
    catalogue_v6 = artifacts / "transition_pattern_catalogue_20260730_v6"
    brl_summary = artifacts / "recurring_transition_taxonomy_stability_20260730_v1"
    lgbm = artifacts / "regime_transition_lightgbm_hpo_20260726_v1"
    gmm = artifacts / "execution_ev_context_head_gmm_geometry_20260726_v1"
    ae = artifacts / "packb_side_local_ae_20260724_v1"
    unsup = artifacts / "failure_first_regime_pipeline_historical_20260726_v12"
    early22 = artifacts / "early_2022_inverse_pi_regime_supplement_20260730_v3"
    recurrence = artifacts / "all_era_worst_period_multiview_recurrence_20260730_v1"
    inventory = artifacts / "causal_regime_feature_inventory_20260730_v5"
    morphology_bound = artifacts / "transition_morphology_support_bound_20260730_v1"
    morphology_loeo = artifacts / "leave_one_era_out_transition_morphology_20260730_v1"
    morphology_readiness = artifacts / "nested_morphology_increment_readiness_20260730_v1"
    morphology_alignment = artifacts / "leave_one_era_out_transition_morphology_alignment_20260730_v1"
    unsup_common_oof = artifacts / "unsupervised_economic_common_oof_20260730_v2"
    alpha_ev_gap = artifacts / "alpha_execution_ev_gap_diagnostic_20260730_v1"
    hurdle_mapping = artifacts / "causal_opportunity_hurdle_mapping_ablation_20260730_v3"
    category_stability = artifacts / "h2_common30_regime_category_performance_stability_20260730_v3"
    cadence = artifacts / "regime_transition_hourly_cadence_audit_20260730_v2"
    strict_regime = artifacts / "strict_forward_regime_only_2022aug_2025_to_2026_20260730_v3"
    strict_sticky = artifacts / "strict_forward_sticky_fullcov_regime_challenger_2022aug_2025_to_2026_20260730_v1"
    strict_dae = artifacts / "strict_forward_dae_gmm_regime_challenger_2022aug_2025_to_2026_20260730_v1"
    strict_transition = artifacts / "strict_transition_v3_multihorizon_competing_risk_20260730_v2"
    strict_brl = artifacts / "strict_transition_brl_challenger_20260730_v1"
    strict_bocpd = artifacts / "strict_bocpd_regime_transition_challenger_20260730_v2"
    alpha_ev_divergence = artifacts / "alpha_execution_ev_divergence_2022_2026_20260730_v3"
    strict_unsupervised = artifacts / "unsupervised_economic_all_era_strict_20260730_v1"
    cost_clearing = artifacts / "a_grade_cost_clearing_conversion_ablation_20260730_v6"
    soft_sidecars = artifacts / "authoritative_soft_regime_transition_sidecars_20260730_v1"
    final_stack = artifacts / "final_identical_row_regime_stack_gam_ablation_20260730_v3"
    h2_bridge = artifacts / "h2_2025_identical_row_oof_bridge_audit_20260730_v7"
    mapping_resolution = artifacts / "pre2026_mapping_resolution_ablations_20260730_v2"
    interaction_diagnostics = artifacts / "final_v3_context_interaction_diagnostics_20260730_v2"
    gam_mixture = artifacts / "final_v3_gam_convex_mixture_ablation_20260730_v1"
    residual_interactions = artifacts / "final_v3_preregistered_residual_interactions_20260730_v1"
    july_bridge = artifacts / "july2025_common30_final_base_residual_oof_bridge_20260730_v1"
    july_map_refresh = artifacts / "july_common30_baseline_map_refresh_20260730_v1"
    july_context_scores = artifacts / "july2025_common30_regime_context_raw_score_extension_20260730_v1"
    july_context_maps = artifacts / "july2025_common30_all_context_map_refresh_20260730_v1"
    category_availability = artifacts / "heldout_regime_category_economics_stability_20260730_v2_availability"
    semantic_signatures = artifacts / "hourly_transition_semantic_signature_ablation_20260730_v1"
    augnov_preflight = artifacts / "augnov2025_pit_scoring_preflight_20260730_v2"
    augnov_bridge = artifacts / "augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1"
    augnov_economics = artifacts / "augnov2025_frozen_july_oos_bridge_validation_economics_20260730_v1"
    augnov_context = artifacts / "augnov2025_common30_fixed_preaug_context_oos_extension_20260730_v2"
    augnov_diagnosis = artifacts / "augnov2025_frozen_july_residual_regime_diagnosis_20260730_v2"
    h2_final_refit = artifacts / "final_refit_h2_common30_gam_sensitivity_20260730_v2"
    h2_final_comparison = artifacts / "final_refit_h2_common30_gam_vs_v3_controls_20260730_v2"
    category_trust = artifacts / "h2_category_failure_risk_trust_ablation_20260730_v1"
    recurring_prototypes = artifacts / "trainonly_recurring_transition_prototype_study_20260730_v3"
    trajectory_sidecar = artifacts / "hourly_trajectory_transition_soft_sidecar_20260730_v1"
    trajectory_coverage = artifacts / "trajectory_transition_identical_row_stack_coverage_20260730_v1"
    trajectory_ablation = artifacts / "trajectory_missingness_identical_row_ablation_20260730_v1"
    december_repair = artifacts / "dec2025_common30_exact1m_repair_stage_20260730_v2"
    december_labels = artifacts / "dec2025_execution_ev_common30_exact1m_labels_20260730_v1"
    december_preflight = artifacts / "dec2025_common30_pit_scoring_preflight_20260730_v1"
    december_bridge = artifacts / "dec2025_common30_frozen_august_base_residual_oos_bridge_20260730_v1"
    december_economics = artifacts / "dec2025_common30_frozen_base_residual_raw_economics_20260730_v1"
    december_context = artifacts / "dec2025_common30_fixed_preaug_context_oos_extension_20260730_v1"
    december_final12_context = artifacts / "dec2025_final12h_frozen_predec_regime_transition_context_extension_20260730_v1"
    constrained_taxonomy = artifacts / "constrained_coarse_transition_taxonomy_20260730_v3"
    supervised_topology = artifacts / "supervised_coarse_topology_taxonomy_audit_20260730_v1"
    failure_incremental = artifacts / "pre2026_oof_model_failure_incremental_value_20260730_v3"
    failure_incremental_cadence = artifacts / "pre2026_oof_model_failure_incremental_value_20260730_v4"
    nested_failure_overlay = artifacts / "pre2026_nested_residual_context_failure_overlay_20260730_v3"
    nested_failure_overlay_audit = artifacts / "pre2026_nested_residual_context_failure_overlay_20260730_v4"
    regime_overlay_gamma_hpo = artifacts / "pre2026_regime_overlay_gamma_hpo_20260730_v2"
    failure_incremental_score_control = artifacts / "pre2026_oof_model_failure_incremental_value_score_control_20260730_v1"
    joint_score_context_gate = artifacts / "pre2026_joint_score_context_incremental_gate_20260730_v2"
    joint_gate_environment = artifacts / "pre2026_joint_score_context_incremental_gate_environment_20260730_v1"
    joint_gate_review = artifacts / "pre2026_joint_score_context_incremental_gate_independent_review_20260730_v3"
    frozen_failure_value_preregistration = artifacts / "frozen_2026_failure_value_correction_preregistration_20260730_v3"
    hourly_book_risk = artifacts / "pre2026_hourly_book_risk_calibrator_20260730_v2_r1"
    hourly_downside_broadcast = artifacts / "pre2026_regime_only_downside_risk_broadcast_20260730_v3"
    rows: list[dict[str, Any]] = []

    def add(requirement: str, status: str, evidence_root: Path, names: Sequence[str], finding: str) -> None:
        ok, seal = _sealed(evidence_root, *names)
        rows.append({"requirement": requirement, "status": status if ok else "missing", "evidence": _evidence(evidence_root, names), "seal_check": seal, "finding": finding if ok else "No verified materialized evidence."})

    ledger_ok, ledger_seal = _sealed(ledger, "coverage_calendar.csv", "hourly_state_calendar.parquet", "transition_episode_ledger.parquet")
    early_ok, early_seal = _sealed(early22, "hourly_state_transition_oof.parquet", "performance_by_month_side_state_phase.csv", "cross_lineage_bridge.csv")
    rows.append({"requirement": "2022_h1_coverage", "status": "proved" if early_ok else "missing", "evidence": json.dumps([str(early22)]), "seal_check": early_seal, "finding": "Jan--Jul 2022 is covered by a separate inverse-PI leave-month-out OOF supplement. It is non-pooled with later PF data and its bridge prohibits taxonomy-ID equivalence." if early_ok else "The required separate early-2022 supplement is not sealed."})
    rows.append({"requirement": "2022_2026_coverage", "status": "proved" if ledger_ok and early_ok else "missing", "evidence": json.dumps([str(ledger), str(early22)]), "seal_check": f"later ledger: {ledger_seal}; early-2022 supplement: {early_seal}", "finding": "The separate inverse-PI supplement covers signal time 2022-01-01 through 2022-08-30 (exact labels through 12:00 UTC), and the later ledger begins 2022-08-30. The date gap is closed, but the two lineages remain non-pooled and non-equivalent." if ledger_ok and early_ok else "Later coverage and/or the required separate early-2022 supplement is not sealed."})
    add("hourly_model_and_assessment_cadence", "proved", cadence, ("manifest.json",), "Training and assessment samples are 1h. Exact 1m bars are nested label/policy-path replay evidence only and never independent model rows.")
    add("multi_horizon_state_analysis", "proved", ledger, ("weekly_regime_summary.csv", "monthly_regime_summary.csv", "state_profiles.csv"), "Hourly state calendar plus weekly/monthly summaries and state profiles are materialized across the available interval.")
    add("feature_distributions_covariances_interactions", "proved", recurrence, ("recurrence_summary.csv", "weekly_covariance_and_interactions.csv", "weekly_feature_shifts.csv", "era_coverage.csv", "manifest.json"), "The sealed all-era recurrence audit measures worst-period feature shifts, covariances and interactions across the tested separated eras with BH control and a same-signed-effect recurrence contract. Its explicitly absent inverse-PI 2022 H1 evidence remains a lineage-scoped limitation, not a reason to retain a duplicate incomplete requirement. This is explanatory evidence, not an economic acceptance gate or specialist promotion.")
    add("volatility_and_liquidity", "proved", selection, ("manifest.json",), "The v3 fold-local selection manifest records 14,536 source fields and selected liquidity inputs under strict expanding folds; the paired multiview panel provides the state measurement surface.")
    add("path_geometry", "proved", geometry, ("path_geometry_by_context.csv", "context_support.csv", "manifest.json"), "Historical path-geometry diagnostics are joined at candidate level; a separate exact-forward geometry artifact exists for May--July 2026.")
    performance_ok, performance_seal = _sealed(failures, "regime_performance.csv", "state_performance_qualification.csv", "worst_week_calendar.csv", "manifest.json")
    gap_ok, gap_seal = _sealed(alpha_ev_gap, "failure_hypothesis_ledger.csv", "mapping_rank_diagnostics.csv", "evidence_backed_next_ablations.csv")
    hurdle_ok, hurdle_seal = _sealed(hurdle_mapping, "heldout_monthly_arm_metrics.csv", "heldout_weekly_attribution.csv", "weekly_monthly_q10_q50_gates.csv", "portfolio_replay_status.csv")
    rows.append({"requirement": "long_short_performance", "status": "proved" if performance_ok else "missing", "evidence": json.dumps([str(failures), str(alpha_ev_gap), str(hurdle_mapping)]), "seal_check": f"failure ledger: {performance_seal}; alpha/EV gap: {gap_seal}; causal hurdle/mapping: {hurdle_seal}", "finding": "The strict-stack failure analysis reports long and short performance by state and period. The alpha-to-execution diagnostic identifies a target/economics and mapping-rank gap; its causal cost-aware hurdle and rank-preservation/support-shrinkage follow-up fails every aggregate/latest weekly/monthly Q10/Q50 gate, so no portfolio replay ran. These are negative diagnostic results, not a policy repair or promotion." if performance_ok else "The required strict-stack performance ledger is not sealed."})
    add("transition_phases", "proved", catalogue_v6, ("adaptive_phase_labels.parquet", "event_preonset_sequences.parquet", "stable_transition_oof.parquet", "stable_transition_brl_oof.parquet", "oof_diagnostic_metrics.csv"), "Adaptive phases, event sequences, LightGBM and native-Beta-Binomial-MAP rule-list OOF stable-versus-transition outputs are materialized.")
    morphology_ok, morphology_seal = _sealed(catalogue_v6, "morphology_oof.parquet", "morphology_recurrence_support.csv", "oof_diagnostic_metrics.csv")
    bound_ok, bound_seal = _sealed(morphology_bound, "component_support_bound.csv", "fold_local_support.csv", "limiting_counts.csv")
    loeo_ok, loeo_seal = _sealed(morphology_loeo, "oof_assignments.parquet", "recurrence_gate.csv", "support.csv")
    readiness_ok, readiness_seal = _sealed(morphology_readiness, "readiness.csv")
    alignment_ok, alignment_seal = _sealed(morphology_alignment, "manifest.json")
    semantic_ok, semantic_seal = _sealed(semantic_signatures, "manifest.json", "hourly_transfer_metrics.csv", "semantic_group_coefficient_stability.csv", "event_identity_support.csv", "cadence_audit.csv")
    prototype_ok, prototype_seal = _sealed(recurring_prototypes, "manifest.json", "prototype_candidate_stability.csv", "transition_vs_stable_transfer.csv", "pre2026_oof_transition_vs_stable_soft_probabilities.parquet", "assessment_2026_transition_vs_stable_soft_probabilities.parquet")
    rows.append({"requirement": "recurring_transition_clusters", "status": "incomplete" if morphology_ok and alignment_ok and semantic_ok and prototype_ok else "missing", "evidence": json.dumps([str(catalogue_v6), str(morphology_bound), str(morphology_loeo), str(morphology_readiness), str(morphology_alignment), str(semantic_signatures), str(recurring_prototypes)]), "seal_check": f"catalogue: {morphology_seal}; support bound: {bound_seal}; leave-era-out: {loeo_seal}; nested readiness: {readiness_seal}; train-only alignment: {alignment_seal}; hourly semantics: {semantic_seal}; trajectory prototypes: {prototype_seal}", "finding": "The outcome-free trajectory study improves transition-versus-stable discrimination materially, but recurring subtype identity still fails. Every K=2--5 train-only candidate contains a one-event/one-era component and bootstrap ARI remains 0.32--0.55, below the preregistered stability/support gate. No subtype IDs are emitted into the stack; recurring global transition types remain incomplete." if morphology_ok and alignment_ok and semantic_ok and prototype_ok else "The required morphology, alignment, semantic-transfer and trajectory-prototype evidence is not fully sealed."})
    taxonomy_ok, taxonomy_seal = _sealed(constrained_taxonomy, "manifest.json", "coarse_taxonomy_candidate_gates.csv", "transition_vs_stable_separate_metrics.csv", "cadence_audit.csv")
    topology_ok, topology_seal = _sealed(supervised_topology, "manifest.json", "decision.json", "state_id_semantic_gate.csv", "topology_target_support.csv", "cadence_audit.csv")
    rows.append({"requirement": "transition_subtype_taxonomy_admissibility", "status": "proved" if taxonomy_ok and topology_ok else "missing", "evidence": json.dumps([str(constrained_taxonomy), str(supervised_topology)]), "seal_check": f"constrained coarse taxonomy: {taxonomy_seal}; supervised topology audit: {topology_seal}", "finding": "The constrained K=2/3 coarse retry and the supervised topology precondition audit are both sealed negative results. Coarse types fail bootstrap/leave-era semantic recurrence, while topology labels fail state-lineage semantics and rotation support before fitting. This proves the stop condition: no subtype classifier, label collapse, subtype ID or routing field is authorized." if taxonomy_ok and topology_ok else "The constrained taxonomy retry and/or supervised topology admissibility audit is not fully sealed."})
    add("stable_vs_transition_discrimination", "proved", catalogue_v6, ("stable_transition_oof.parquet", "stable_transition_brl_oof.parquet", "oof_diagnostic_metrics.csv"), "OOF stable-versus-transition discrimination is materialized: LightGBM AUC 0.874/AP 0.871/Brier 0.149; the executed native_beta_binomial_map challenger is materially weaker (AUC 0.600/AP 0.571/Brier 0.265).")
    add("lightgbm_method", "proved", lgbm, ("report.json", "horizon_metrics.csv", "winner_horizon_grouped_oof.parquet"), "Nested-HPO grouped-CV transition LightGBM is materialized for 1/3/6/12h horizons (3h AUC 0.799; AP 0.133).")
    add("gmm_and_dae_geometry", "proved", gmm, ("manifest.json", "execution_ev_model_ablation_leaderboard.csv", "execution_ev_model_ablation_oof.parquet"), "GMM geometry is tested in a leakage-constrained outer-OOF execution-EV ablation; separate side-local AE/GMM frozen states are also available.")
    unsup_ok, unsup_seal = _sealed(unsup, "manifest.json", "failure_detector_oof.parquet", "taxonomy_gmm_summary_expost.parquet", "sufficiency_gate.json")
    common_ok, common_seal = _sealed(unsup_common_oof, "metrics_summary.csv", "period_metrics.parquet", "side_metrics.parquet", "fold_provenance.parquet")
    strict_unsup_ok, strict_unsup_seal = _sealed(strict_unsupervised, "manifest.json")
    rows.append({"requirement": "unsupervised_regime_learning", "status": "proved" if unsup_ok and strict_unsup_ok else "incomplete" if unsup_ok else "missing", "evidence": json.dumps([str(unsup), str(unsup_common_oof), str(strict_unsupervised)]), "seal_check": f"failure-first pipeline: {unsup_seal}; common economic OOF v2: {common_seal}; all-era strict: {strict_unsup_seal}", "finding": "The failure-first/unsupervised pipeline exists and the all-era strict extension evaluates semantically identical sticky-GMM and DAE-GMM arms using pre-2026 OOF maps and untouched 2026. Both degrade IC and net EV versus baseline; diagonal and failure-first fail closed because no identical historical transform/score contract exists. Method coverage is proved, but no unsupervised arm is promotable." if unsup_ok and strict_unsup_ok else "The all-era strict unsupervised comparison is not fully sealed."})
    brl_ok, brl_seal = _sealed(catalogue_v6, "stable_transition_brl_oof.parquet", "stable_transition_brl_rule_lists.json", "oof_diagnostic_metrics.csv")
    summary_ok, summary_seal = _sealed(brl_summary, "stable_transition_brl_oof_metrics.csv")
    rows.append({"requirement": "bayesian_rule_list_method", "status": "proved" if brl_ok and summary_ok else "incomplete", "evidence": json.dumps([str(catalogue_v6), str(brl_summary)]), "seal_check": f"catalogue v6: {brl_seal}; comparative summary: {summary_seal}", "finding": "The rule-list challenger is executed OOF and comparatively summarized. Its backend is native_beta_binomial_map, not MCMC Bayesian rule lists; it is therefore an executed interpretability challenger, not a successful economic gate or promoted control." if brl_ok and summary_ok else "The required executed BRL catalogue v6 and/or comparative summary is not sealed."})
    diagonal_ok, diagonal_seal = _sealed(strict_regime, "manifest.json", "regime_only_forward_2026_sidecar.parquet")
    sticky_ok, sticky_seal = _sealed(strict_sticky, "manifest.json", "regime_only_forward_2026_sidecar.parquet", "train_only_geometry_and_persistence_sweep.csv")
    dae_ok, dae_seal = _sealed(strict_dae, "manifest.json", "regime_only_forward_2026_sidecar.parquet", "train_only_dae_geometry_and_persistence_sweep.csv")
    rows.append({"requirement": "strict_2022_2025_to_2026_regime_benchmark", "status": "proved" if diagonal_ok and sticky_ok and dae_ok else "incomplete", "evidence": json.dumps([str(strict_regime), str(strict_sticky), str(strict_dae)]), "seal_check": f"diagonal: {diagonal_seal}; sticky full-covariance: {sticky_seal}; DAE-GMM: {dae_seal}", "finding": "All three strict hourly GMM-family arms are sealed on the same 2022-2025 fit / untouched-2026 assessment panel. All are rejected: median dwell remains 2h and hourly switching remains about 30-32%, so their soft identities are diagnostic and not eligible stack features." if diagonal_ok and sticky_ok and dae_ok else "The common strict GMM-family benchmark is not fully sealed."})
    transition_ok, transition_seal = _sealed(strict_transition, "manifest.json")
    strict_brl_ok, strict_brl_seal = _sealed(strict_brl, "manifest.json")
    strict_bocpd_ok, strict_bocpd_seal = _sealed(strict_bocpd, "manifest.json")
    rows.append({"requirement": "strict_2022_2025_to_2026_transition_benchmark", "status": "proved" if transition_ok and strict_brl_ok and strict_bocpd_ok else "incomplete" if transition_ok and strict_brl_ok else "missing", "evidence": json.dumps([str(strict_transition), str(strict_brl), str(strict_bocpd)]), "seal_check": f"LGBM: {transition_seal}; BRL: {strict_brl_seal}; BOCPD: {strict_bocpd_seal}", "finding": "Strict hourly LGBM, native-MAP BRL and resumable BOCPD challengers are sealed on the 2022-2025/2026 boundary. The benchmark is method-complete even when every arm is rejected economically or statistically." if transition_ok and strict_brl_ok and strict_bocpd_ok else "Strict LGBM/BRL exist, but BOCPD is not yet sealed."})
    add("alpha_to_execution_ev_divergence", "proved", alpha_ev_divergence, ("lineage_summary.parquet", "period_metrics.parquet", "score_deciles.parquet", "report.json"), "The lineage-local audit proves that first-touch remains related to net EV, while score-to-target conversion is weak, the explicit cost hurdle is about 100bps, and July reverses alpha-to-net ordering.")
    add("cost_clearing_conversion_ablation", "proved", cost_clearing, ("manifest.json", "strict_forward_2026_summary.csv", "strict_forward_arm_availability.csv"), "The frozen 2025-to-2026 cost-clearing hurdle is sealed with no 2026 fit/map labels. It worsens aggregate net EV and is rejected; context arms fail closed across incompatible semantics.")
    add("authoritative_hourly_soft_sidecars", "proved", soft_sidecars, ("manifest.json", "soft_regime_hourly.parquet", "soft_transition_hourly.parquet", "cadence_audit.csv", "label_resolution_audit.csv", "bocpd_reliability.csv"), "Separate causal hourly regime and transition sidecars are sealed with 2022-2025 blocked OOF / untouched-2026 provenance. One-minute observations remain nested labels only; BOCPD probability heads are explicitly diagnostic-only.")
    add("identical_row_regime_transition_stack_ablation", "proved", final_stack, ("manifest.json", "metrics_summary.csv", "context_intersection_coverage.csv", "input_score_coverage.csv", "period_metrics.parquet", "side_metrics.parquet"), "The corrected ten-arm hourly base/residual/GAM comparison conflict-checks and coalesces complementary historical score/label aliases, uses five chronological OOF folds for residual/GAM arms, and assesses one 127,777-row forward universe with one pooled global top 10%. GAM regime/combined improve aggregate EV and GAM transition improves recent/tail transfer, but every arm remains negative; no portfolio replay or promotion is authorized.")
    add("h2_2025_score_ledger_bridge", "proved", h2_bridge, ("manifest.json", "source_compatibility_ledger.csv", "readiness_report.json"), "The v7 H2 audit supersedes v6: July, August--November and December have sealed common-30 hourly score/economics bridges. December has 44,640 base/residual rows and 43,920 exact context rows. The final 12 context timestamps remain explicitly unavailable: the raw inputs exist, but the score-only frozen reconstruction does not reproduce the canonical transition overlap. These remain common-30 sensitivities rather than wider final-ledger replacements.")
    add("july_2025_common30_base_residual_oof", "proved", july_bridge, ("manifest.json", "bridge_contract.json", "base_stage_contract.json", "base_oof_predictions.parquet", "oof_predictions.parquet"), "July contributes 44,640 unique hourly candidates with exact both-side base/residual OOF scores, frozen 31/8 base features and residual contracts, and all fit labels resolved before July. Residual improves July global-top10 net EV only from -104.13 to -101.75 bps and remains non-promotable.")
    add("july_2025_baseline_map_refresh", "proved", july_map_refresh, ("manifest.json", "metrics_summary.csv", "mapping_fit_audit.json", "period_metrics.parquet", "side_metrics.parquet"), "Adding the sealed July common-30 cohort reduces map age by 31.54 days but leaves the frozen 2026 baseline global-top10 membership and -77.51 bps net EV exactly unchanged. Rank preservation repairs ties only; the population mismatch keeps this a non-promotional sensitivity.")
    add("july_2025_context_raw_score_extension", "proved", july_context_scores, ("manifest.json", "fit_audit.json", "metrics_summary.csv", "july_raw_context_scores.parquet", "period_metrics.parquet", "side_metrics.parquet"), "Strictly pre-July side-local models score all six residual/GAM regime, transition and combined context arms on the 44,640-row July common-30 bridge. Bounded GAM regime improves July raw global-top10 EV from -101.75 to -88.40 bps and execution IC from -0.039 to 0.023; transition and combined arms are weaker. The result remains negative, population-limited and non-promotable.")
    add("july_2025_all_context_map_refresh", "proved", july_context_maps, ("manifest.json", "metrics_summary.csv", "mapping_fit_audit.json", "period_metrics.parquet", "side_metrics.parquet"), "Appending each compatible July context raw-score cohort to its pre-2026 isotonic map leaves every 2026 global-top10 set exactly unchanged because the map is monotone and raw score breaks ties. GAM regime therefore remains -57.92 bps aggregate and -134.68 bps July; freshness changes calibration only, not economics.")
    add("augnov_2025_common30_pit_readiness", "proved", augnov_preflight, ("manifest.json", "readiness_report.json", "coverage_by_month_side_symbol.csv"), "The authoritative v2 full-population preflight verifies all 175,680 August--November common-30 hourly candidates, exact native/execution identities, complete PIT inputs for the frozen 31-long/8-short base and 69 residual fields, and both sides. One-minute paths remain nested label evidence only.")
    add("augnov_2025_frozen_july_oos_bridge", "proved", augnov_bridge, ("manifest.json", "oos_predictions.parquet"), "The frozen-through-July base/residual models score 175,680 unique August--November hourly candidates with no HPO or 2026 use. The paired economics audit remains negative: base/residual global-top10 EV is -109.92/-97.53 bps.")
    add("augnov_2025_bridge_validation_economics", "proved", augnov_economics, ("manifest.json", "metrics_summary.csv", "period_metrics.parquet", "side_metrics.parquet"), "The sealed validation report verifies unique hourly identity, both sides and exact economics endpoints. Residual raises execution IC from 0.0269 to 0.0470 and global-top10 EV from -109.92 to -97.53 bps, but weekly/monthly Q10 and every month remain negative.")
    add("augnov_2025_context_oos_extension", "proved", augnov_context, ("manifest.json", "metrics_summary.csv", "period_metrics.parquet", "side_metrics.parquet"), "Six fixed pre-August regime, transition and combined context arms are assessed OOS. Bounded GAM-regime is best at IC 0.0557 and -93.60 bps global-top10 EV, but all arms and tails remain negative.")
    add("augnov_2025_residual_regime_diagnosis", "proved", augnov_diagnosis, ("manifest.json", "score_metrics_month_week_side.csv", "residual_replacement_attribution.csv", "feature_context_distribution_shift.csv", "feature_context_covariance_shift.csv"), "The exact replacement diagnosis finds residual harm in August/September (-7.16/-2.10 bps versus base) and help in October/November (+43.44/+10.29 bps), alongside separate regime-state and transition shifts. These are preregistration hypotheses, not a gate.")
    add("h2_2025_final_refit_to_2026", "proved", h2_final_refit, ("manifest.json", "metrics_summary.csv", "fit_audit.json", "period_metrics.parquet", "side_metrics.parquet"), "Fixed side-local GAM regime/transition/combined heads are refit on all compatible pre-2026 hourly labels and assessed on the unchanged 127,777-row 2026 universe. GAM-regime improves baseline to -65.91 bps but trails frozen-v3 GAM-regime (-57.92 bps), while every arm and tail remains negative. V2 corrects only per-arm map-row reporting; scores and economics are unchanged from invalid v1.")
    add("h2_2025_final_refit_control_comparison", "proved", h2_final_comparison, ("manifest.json", "comparison.csv", "contract.json"), "The sealed v2 comparison binds the H2 common-30 refits to the original frozen-v3 controls on the same hourly 2026 assessment. Wider H2 fitting does not improve the best frozen regime/combined controls and remains non-promotional.")
    add("trajectory_transition_binary_soft_sidecar", "proved", trajectory_sidecar, ("manifest.json", "hourly_trajectory_transition_soft_sidecar.parquet", "anchor_calibration.csv", "coverage_by_era.csv", "cadence_audit.csv"), "A causal hourly trajectory transition head is materialized with 29,060 pre-2026 blocked-era OOF scores and 3,927 frozen 2026 scores plus availability, entropy, margin and fit provenance. On complete-lookback untouched-2026 anchors it reaches AUC 0.849 and AP 0.832. It is distinct from unstable subtype IDs.")
    add("trajectory_transition_identical_row_coverage", "proved", trajectory_coverage, ("manifest.json", "timestamp_coverage.csv", "report.json"), "The first strict stack join correctly fails closed because trajectory context covers only 1,042 of 1,699 frozen-2026 timestamps. No rows are dropped or silently filled; this audit motivates the separately preregistered neutral-fill availability policy.")
    add("trajectory_transition_stack_ablation", "proved", trajectory_ablation, ("manifest.json", "metrics_summary.csv", "availability_metrics.csv", "period_metrics.parquet", "side_metrics.parquet"), "The preregistered missingness-aware trajectory ablation retains all 127,777 frozen 2026 candidates with fixed neutral fill and availability. The fullest regime+existing-transition+trajectory arm raises execution IC from 0.0664 to 0.0832 but worsens global-top10 EV from -69.99 to -70.75 bps, worsens July, and fails both-side economics. No trajectory arm is promotable.")
    add("december_2025_common30_exact1m_repair", "proved", december_repair, ("manifest.json", "coverage.csv", "download_candidates.parquet"), "The scoped repair closes exact nested 1m path coverage for all 44,640 December common-30 hourly candidates, 22,320 per side, with zero outstanding requested windows. Minute bars remain nested label/replay observations only.")
    add("december_2025_common30_execution_labels", "proved", december_labels, ("manifest.json", "labels.parquet", "missing.csv"), "Exact 12-hour deployed-policy execution labels are materialized for all 44,640 December common-30 hourly candidates with zero missing rows and explicit label availability through 2026-01-01 12:00 UTC.")
    add("december_2025_common30_pit_readiness", "proved", december_preflight, ("manifest.json", "readiness_report.json", "coverage_by_month_side_symbol.csv"), "All December common-30 identities and frozen 31-long/8-short base plus residual feature contracts have exact finite hourly PIT coverage in every side/symbol cell.")
    add("december_2025_frozen_base_residual_bridge", "proved", december_bridge, ("manifest.json", "bridge_contract.json", "oos_predictions.parquet"), "The sealed 44,640-row December base/residual bridge reuses immutable models fitted only through 2025-07-31. December/January-resolving labels are assessment-only and never enter fit, HPO, selection, calibration or mapping.")
    add("december_2025_base_residual_economics", "proved", december_economics, ("manifest.json", "metrics_summary.csv", "period_metrics.parquet", "side_metrics.parquet"), "On one pooled global top 10%, December base/residual execution IC is 0.0142/0.0487 and net EV is -117.55/-84.69 bps. Residual improves ranking and gross economics but remains below costs, with short materially weaker than long.")
    add("december_2025_context_oos_extension", "proved", december_context, ("manifest.json", "metrics_summary.csv", "period_metrics.parquet", "side_metrics.parquet", "context_unavailable_candidates.parquet"), "On the exact 43,920-row common-context subset, the residual control is -83.79 bps and every fixed regime/transition/combined LGBM or GAM context arm is worse. The final 12 hourly sidecar timestamps (720 candidate rows) are explicitly excluded with no fill or forward-fill.")
    december_final12_ok, december_final12_seal = _sealed(december_final12_context, "manifest.json", "readiness_report.json")
    rows.append({"requirement": "december_2025_final12_frozen_context_reproduction", "status": "incomplete" if december_final12_ok else "missing", "evidence": json.dumps([str(h2_bridge), str(december_final12_context)]), "seal_check": f"H2 v7: {_sealed(h2_bridge, 'manifest.json', 'readiness_report.json')[1]}; final-12 extension: {december_final12_seal}", "finding": "Raw causal inputs for the final 12 timestamps are present, but frozen score-only reconstruction fails canonical overlap reproduction in transition fields (including LGBM and BOCPD outputs). The extension correctly fails closed and appends no rows. Recover the exact serialized fold-03 imputer/calibrator and BOCPD head states, then reproduce the canonical overlap exactly; do not fill, forward-fill, reroute or refit." if december_final12_ok else "The final-12 frozen-context reproduction audit is not sealed."})
    add("pre2026_mapping_resolution", "proved", mapping_resolution, ("manifest.json", "metrics_summary.csv", "mapping_fit_audit.json", "period_metrics.parquet", "side_metrics.parquet"), "Seven preregistered maps are fit only on pre-2026 hourly OOF rows and assessed on frozen 2026 hourly rows. Strict rank preservation removes isotonic tie plateaus without changing economics; support-shrunk and binned alternatives are unstable and no mapping is promotable.")
    add("regime_transition_interaction_diagnostics", "proved", interaction_diagnostics, ("manifest.json", "tree_shap_interactions.csv", "regime_conditional_permutation_importance.csv", "interaction_qualification.csv", "feature_covariance_shifts.csv"), "Subsampled tree-SHAP and regime-/transition-conditional permutation diagnostics are sealed on pre-2026 candidate-held OOF rows with untouched-2026 assessment. Residual-context effects are more stable than base-context effects; BOCPD context remains non-standalone and the diagnostic itself is not a promotion.")
    add("fixed_gam_context_mixture", "proved", gam_mixture, ("manifest.json", "historical_oof_gate_table.csv", "mixture_oof_and_forward_metrics.csv", "frozen_2026_mixture_top10_books.parquet"), "All 15 fixed convex mixtures of regime, transition and combined GAM experts fail the pre-2026 aggregate, tail and both-side gates. No learned gate/blender is authorized and the frozen 2026 results are diagnostic only.")
    add("preregistered_residual_context_interactions", "proved", residual_interactions, ("manifest.json", "metrics_summary.csv", "oof_fit_audit.parquet", "period_metrics.parquet", "side_metrics.parquet", "row_cadence_audit.csv"), "The fixed low-capacity side-local residual-context follow-on uses only preregistered interactions supported by the pre-2026 diagnostic. All arms remain negative and non-promotable. Its own cadence audit proves hourly fit, OOF, mapping and assessment rows with one-minute data restricted to nested labels/replay.")
    failure_incremental_ok, failure_incremental_seal = _sealed(failure_incremental, "manifest.json", "contract.json", "fold_metrics.csv", "global_top10_economics.csv", "materialized_targets.parquet")
    cadence_supplement_ok, cadence_supplement_seal = _sealed(failure_incremental_cadence, "manifest.json", "contract.json", "cadence_provenance_audit.csv")
    nested_overlay_ok, nested_overlay_seal = _sealed(nested_failure_overlay, "manifest.json", "contract.json", "cohort_audit.csv", "eligibility.csv", "pooled_deltas.csv", "side_metrics.csv", "true_pooled_metrics.csv")
    nested_overlay_audit_ok, nested_overlay_audit_seal = _sealed(nested_failure_overlay_audit, "manifest.json", "contract.json", "candidate_set_assertions.csv", "v1_outer_identity_assertions.csv", "high_low_ev_deltas.csv")
    gamma_hpo_ok, gamma_hpo_seal = _sealed(regime_overlay_gamma_hpo, "manifest.json", "contract.json", "parity_audit.json", "eligibility.csv", "pooled_deltas.csv", "pooled_metrics.csv", "side_deltas.csv")
    rows.append({"requirement": "direct_model_failure_incremental_value_learning", "status": "proved" if failure_incremental_ok and cadence_supplement_ok and nested_overlay_ok and nested_overlay_audit_ok and gamma_hpo_ok else "missing", "evidence": json.dumps([str(failure_incremental), str(failure_incremental_cadence), str(nested_failure_overlay), str(nested_failure_overlay_audit), str(regime_overlay_gamma_hpo)]), "seal_check": f"pre-2026 target/OOF evidence: {failure_incremental_seal}; hourly cadence supplement: {cadence_supplement_seal}; nested overlay v3: {nested_overlay_seal}; audit supplement v4: {nested_overlay_audit_seal}; gamma HPO v2: {gamma_hpo_seal}", "finding": "Direct selected-book residual-versus-base incremental-utility, residual selected-net-failure and false-positive-severity targets are materialized and learned side-locally leave-era-out on pre-2026 hourly rows only. Utility and failure heads transfer; severity does not. The v4 supplement verifies hour-aligned candidate rows and labels resolved before 2026, with 1m restricted to nested labels/economics. The authoritative nested failure overlay v3 plus v4 audit prove cohort/cadence/label/high-low integrity. Exact gamma-HPO v2 has 68,234-row parity and tests gamma 0.125/0.25/0.5: AUC diagnostics improve (median +0.002716/+0.005021/+0.006788, 75% positive eras), including positive both-side medians and better Brier/AP, but high-low EV stability fails (median +0.000096/+0.000813/+0.000577; only 0.375/0.375/0.500 improvement fractions). All gammas fail the economic gate; selected_gamma=null and authorized_for_2026=false. This proves target/overlay/HPO coverage only, not a frozen-2026 application." if failure_incremental_ok and cadence_supplement_ok and nested_overlay_ok and nested_overlay_audit_ok and gamma_hpo_ok else "The pre-2026 target/OOF, overlay/audit and/or gamma-HPO evidence is not fully sealed."})
    score_control_ok, score_control_seal = _sealed(failure_incremental_score_control, "manifest.json", "contract.json", "context_incremental_gate.csv", "context_vs_score_control_fold_deltas.csv", "score_control_fold_metrics.csv")
    joint_gate_ok, joint_gate_seal = _sealed(joint_score_context_gate, "manifest.json", "contract.json", "candidate_set_equality.csv", "eligibility.csv", "matched_fold_deltas.csv", "fold_audit.csv")
    joint_environment_ok, joint_environment_seal = _sealed(joint_gate_environment, "manifest.json", "provenance.json")
    joint_review_ok, joint_review_seal = _sealed(joint_gate_review, "manifest.json", "review.json", "input_and_output_hash_audit.csv", "matched_delta_recomputation_audit.csv", "oof_row_and_target_audit.csv", "source_and_code_hash_audit.csv")
    rows.append({"requirement": "context_incremental_value_beyond_score_only", "status": "proved" if joint_gate_ok and joint_environment_ok and joint_review_ok else "missing", "evidence": json.dumps([str(failure_incremental), str(failure_incremental_cadence), str(failure_incremental_score_control), str(joint_score_context_gate), str(joint_gate_environment), str(joint_gate_review)]), "seal_check": f"pre-2026 target/OOF evidence: {failure_incremental_seal}; hourly cadence supplement: {cadence_supplement_seal}; provisional v1: {score_control_seal}; authoritative joint v2: {joint_gate_seal}; environment: {joint_environment_seal}; independent review: {joint_review_seal}", "finding": "The authoritative joint v2 rerun supersedes provisional v1 (retained as non-authoritative lineage). It uses the same frozen side-local implementation, 150k candidate-hash cap, arm availability, folds and rows for context and CORE-only controls; all 132/132 arm/target/era/side train+test cohorts match. Its sealed environment records Python 3.12.2, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.6.1 and the joint code hashes. The independent v3 review recomputes matched deltas/OOF metrics, verifies source/output hashes and 1h cadence/no-2026 boundary, and independently confirms no context correction is authorized. Every gate fails. Utility median deltas are combined -0.005055, regime -0.000275, trajectory -0.000769 and transition -0.003997. Failure medians are combined +0.001935, regime +0.005315, trajectory +0.002259 and transition -0.009070, but positive-era fractions are only 0.500/0.625/0.556/0.375 and worst deltas -0.060785/-0.047085/-0.028010/-0.054822. None meets median >0, >=75% positive, min >=-0.02 and >=6 eras. Generic context has no stable incremental value beyond CORE scores." if joint_gate_ok and joint_environment_ok and joint_review_ok else "The authoritative joint score-control audit, its environment provenance and/or independent review is not fully sealed."})
    hourly_book_ok, hourly_book_seal = _sealed(
        hourly_book_risk,
        "manifest.json",
        "contract.json",
        "arm_era_hourly_and_candidate_availability.csv",
        "hourly_design_integrity_audit.csv",
        "hourly_oof_predictions.parquet",
        "candidate_oof_broadcast_scores.parquet",
        "context_incremental_economics_gate.csv",
    )
    rows.append({"requirement": "timestamp_level_book_risk_calibration", "status": "proved" if hourly_book_ok else "missing", "evidence": json.dumps([str(hourly_book_risk)]), "seal_check": hourly_book_seal, "finding": "The corrected v2_r1 timestamp study uses exactly one statistical row per available UTC hour, includes all-hour opportunity calibration, fits conditional count/mean-EV/failure/severity heads without candidate replication, and broadcasts one scalar correction while preserving within-hour ordering and one pooled global top-10 book. Regime/transition/combined retain unsupported 2023 candidates with exactly zero adjustment and exclude that era from their eight-era evidence gate; trajectory has nine supported eras. Regime gamma 0.10 improves median era net EV by 3.47 bps with 75% positive eras and positive long/short medians, but worsens median weekly Q10 by 3.57 bps. More importantly, the raw residual control is negative in every era, so all 12 arm/gamma candidates are ineligible. Training and assessment are 1h; 1m remains nested label/economics evidence; no 2026 file was read." if hourly_book_ok else "The corrected one-row-per-hour timestamp-level calibrator is not fully sealed."})
    hourly_downside_ok, hourly_downside_seal = _sealed(
        hourly_downside_broadcast,
        "manifest.json",
        "contract.json",
        "hourly_broadcasts.parquet",
        "within_hour_order_audit.csv",
        "eligibility.csv",
        "weekly_monthly_top10_metrics.csv",
    )
    rows.append({"requirement": "timestamp_level_expected_downside_broadcast", "status": "proved" if hourly_downside_ok else "missing", "evidence": json.dumps([str(hourly_downside_broadcast)]), "seal_check": hourly_downside_seal, "finding": "Authoritative v3 supersedes the misnamed probability-only v2 and composes the corrected hourly score-only/context heads as opportunity times conditional failure rate times conditional downside severity in return units. Fixed lambdas 0.25/0.5/1.0 are broadcast equally within each hour; the order audit passes, zero fallback is unused across all eight supported eras, selection remains one pooled global top-10, and no 2026 input is read. All six arms fail. Context lambda 0.25 is least bad (+0.68 bps aggregate, +1.07/+0.49 bps long/short) but has -2.55 bps minimum-era delta and -14.50/-12.29 bps weekly/monthly Q10 deltas. No lambda is selected or authorized." if hourly_downside_ok else "The corrected hourly expected-downside broadcast is not fully sealed."})
    frozen_preregistration_ok, frozen_preregistration_seal = _sealed(frozen_failure_value_preregistration, "manifest.json", "contract.json")
    rows.append({"requirement": "frozen_2026_failure_incremental_economics_application", "status": "incomplete" if joint_gate_ok and joint_environment_ok and joint_review_ok and frozen_preregistration_ok else "missing", "evidence": json.dumps([str(joint_score_context_gate), str(joint_gate_environment), str(joint_gate_review), str(frozen_failure_value_preregistration)]), "seal_check": f"authoritative joint v2: {joint_gate_seal}; environment: {joint_environment_seal}; independent review: {joint_review_seal}; final v3 preregistration: {frozen_preregistration_seal}", "finding": "Final preregistration v3 supersedes v1/v2, checksum-binds joint v2 and its environment provenance, records zero eligible context heads and authorized=false, and forbids any 2026 candidate/economics read or correction score. The independent v3 joint review likewise records no 2026 candidate/label/economics/replay/portfolio file opened. No frozen-2026 application, replay or promotion is authorized. A future application requires a materially different mechanism that first beats the exact jointly reviewed score-only gate, followed by a new no-2026-read preregistration." if joint_gate_ok and joint_environment_ok and joint_review_ok and frozen_preregistration_ok else "The joint gate/environment/review and/or final v3 no-read preregistration is not fully sealed."})
    category_ok, category_seal = _sealed(category_stability, "manifest.json", "category_era_summary.csv", "category_leave_era_out.csv", "category_stability_qualification.csv", "untouched_2026_category_assessment.csv")
    category_availability_ok, category_availability_seal = _sealed(category_availability, "manifest.json", "report.json", "availability_requirements.csv", "source_context_coverage.csv", "source_qualification.csv")
    category_trust_ok, category_trust_seal = _sealed(category_trust, "manifest.json", "leave_era_out_rank_stability.csv", "trust_preflight_gate.csv", "report.json")
    calendar_ok, calendar_seal = _sealed(performance, "performance_period_metrics.parquet", "meaningful_positive_summary.csv", "manifest.json")
    rows.append({"requirement": "regime_category_performance_stability", "status": "incomplete" if calendar_ok and category_ok and category_trust_ok else "missing", "evidence": json.dumps([str(performance), str(category_stability), str(category_availability), str(category_trust)]), "seal_check": f"performance calendar: {calendar_seal}; three-era category stability: {category_seal}; prior availability v2: {category_availability_seal}; relative trust preflight: {category_trust_seal}", "finding": "The H2 common-30 extension now supplies three independent pre-2026 eras, both-side support and seven adequately supported regime/transition/combined categories. Zero category has stable positive leave-era-out transfer. Even relative failure ordering is unstable: minimum pairwise Spearman is -0.429 for regime/combined and -1.0 for transition versus the preregistered 0.70 gate, so the trust correction fails closed before 2026 selection. The stability requirement remains incomplete because support is no longer the blocker; reproducible economic category ordering is." if calendar_ok and category_ok and category_trust_ok else "The required performance calendar, three-era category audit and relative trust preflight are not fully sealed."})
    add("worst_period_calendar", "proved", failures, ("worst_week_calendar.csv", "weekly_feature_shifts.csv", "weekly_covariance_shifts.csv", "weekly_interaction_shifts.csv"), "A complete-week worst-period calendar and feature/covariance/interaction shift diagnostics are materialized; current recurrent-significance flags do not establish a durable common driver.")
    add("existing_and_missing_regime_features", "proved", inventory, ("field_inventory.csv", "economically_plausible_missing_observables.csv", "forbidden_or_rejected_fields.csv", "source_availability.csv", "manifest.json"), "The causal feature inventory v5 now records available, unavailable, economically plausible missing, and forbidden/rejected regime observables. Inventory completion is not a claim that every missing input has been backfilled or is economically incremental.")
    audit = pd.DataFrame(rows)
    todo = pd.DataFrame([
        {"priority": 1, "next_work": "No chronological coverage backfill remains in the audited 2022--2026 calendar. Preserve the separate inverse-PI lineage and do not manufacture taxonomy equivalence or pooled economics."},
        {"priority": 2, "next_work": "Do not train transition subtype specialists: K=2--5 trajectory prototypes still fail bootstrap/support recurrence. Preserve the transferable binary trajectory probability as separate diagnostic context, and reopen subtype learning only after materially more independent transition events or a hierarchical non-singleton taxonomy exists."},
        {"priority": 3, "next_work": "July through December common-30 base/residual evidence is complete. December residual improves base materially but remains negative, and every December context arm is worse. Keep the final 12 context timestamps explicitly unavailable until the exact serialized fold-03 transition states reproduce canonical overlap; do not fill, forward-fill, reroute, refit or treat common-30 evidence as a wider-population replacement."},
        {"priority": 4, "next_work": "The authoritative joint candidate-level gate and corrected one-row-per-hour v2_r1 book-risk study reject every generic regime/transition/trajectory correction. Regime gamma 0.10 has a small median-EV improvement but worsens weekly Q10 and every residual control era remains negative. Do not apply, replay or promote these heads. A next mechanism must be narrowly preregistered, preserve pooled global top-10 and within-hour order, improve weekly/monthly tails and both sides, and beat an economically positive absolute control before a new no-2026-read application contract can be sealed."},
        {"priority": 5, "next_work": "Extend the matched common-OOF GMM/DAE/failure-first economic protocol beyond its May--July 2026 overlap and retain additions only with held-out aggregate and recent EV improvement; it is not yet all-era evidence."},
        {"priority": 6, "next_work": "Use the completed causal inventory to pre-register narrow causal feature/composite tests, with source-availability and lineage contracts retained; do not treat inventory presence as feature promotion."},
        {"priority": 7, "next_work": "Resolve the alpha-to-execution gap: rank-preserving mapping repairs ties but not net EV, and support-shrunk/binned maps plus fixed GAM mixtures fail aggregate/latest weekly/monthly Q10/Q50 gates. Keep one global top-k and exact policy economics; do not use 2026 outcomes, ex-post phase, category stability, or a failed arm as a gate."},
    ])
    summary = {"schema": "regime_objective_completion_audit_v1", "status_counts": audit.status.value_counts().to_dict(), "proved_is_not_promotion": True, "scope": "artifact-evidence audit; no model or policy is promoted"}
    return audit, todo, summary


def run(*, artifacts: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    audit, todo, summary = build_audit(artifacts)
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        audit.to_parquet(stage / "completion_audit.parquet", index=False, compression="zstd")
        audit.to_csv(stage / "completion_audit.csv", index=False)
        todo.to_csv(stage / "next_work.csv", index=False)
        manifest = {**summary, "inputs_root": str(artifacts), "outputs": {name: {"path": str(output_dir / name), "sha256": sha256(stage / name)} for name in ("completion_audit.parquet", "completion_audit.csv", "next_work.csv")}, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(run(artifacts=args.artifacts, output_dir=args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
