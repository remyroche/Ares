#!/usr/bin/env python3
"""Compare M0--M20 causal *meta-head input* ablations.

Unlike the prior overlay diagnostic, this runner never multiplies a final rank
by auxiliary predictions.  Every arm fits the same side/archetype-aware LGBM
meta head on the current soft target.  M1--M6 add only prior-fitted auxiliary
predictions or local priors as ordinary model inputs.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_negative_hit_residual_leaf_rules import _parse_months  # noqa: E402
from scripts.run_executable_quality_transition_ablation import (  # noqa: E402
    _breakdown,
    _causal_residual_target,
    _expected_ev,
    _fit_score_value_map,
    _fit_predict,
    _metrics,
    _negative_residual_event,
    _select_top10,
    _write_json,
)
from scripts.run_meta_residual_head_stack_ablation import (  # noqa: E402
    DEFAULT_INPUT,
    DEFAULT_LABELS_ROOT,
    STATE_CELLS,
    _conditional_path_target,
    _good_trade_target,
    _local_multiplier as _local_risk_prior_state,
    _apply_local_multiplier as _apply_local_risk_prior,
    _load_handoff_with_labels,
    _month_label_paths,
    _read_parquet,
    _residual_arch_context,
    _time_spread_sample,
)
from extreme_price_movements.static_feature_store import read_static_features  # noqa: E402
from extreme_price_movements.residual_event_archetypes import (  # noqa: E402
    EXECUTABLE_FAILURE_TARGETS,
    OUTCOME_COLUMNS as RESIDUAL_STATE_OUTCOME_COLUMNS,
    ResidualEventArchetypeConfig,
    ResidualEventArchetypeState,
    residual_event_distilled_feature_names,
    residual_event_quality_probability_feature_names,
)
from extreme_price_movements.global_residual_latent_state import (  # noqa: E402
    OUTCOME_COLUMNS as GLOBAL_STATE_OUTCOME_COLUMNS,
    PHASE_STATE_FEATURES,
    add_causal_phase_state_features,
)
from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    RESIDUAL_FEATURE_PREFIX,
    SEMANTIC_ARCHETYPES,
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
    strip_outcomes_for_oos,
)

DEFAULT_OUTPUT = ROOT / "data_perp/reports/meta_residual_head_feature_ablation_20260719_v1"
DEFAULT_FEATURE_CONTRACT = (
    ROOT
    / "data_perp/artifacts/20260717_meta_residual_state_mda95_hier_newaegmm_hpo150_v1"
    / "meta_model/staged_selection_hpo_manifest.json"
)
DEFAULT_FEATURE_STORE_ID = "20260711_070000"
DEFAULT_FULL_FEATURE_LEDGERS = (
    ROOT / "data_perp/reports/meta_v9_recovery_20260717/residual_state_mda95_hier_newaegmm_preapr_oof_v1/oos_predictions.parquet",
    ROOT / "data_perp/reports/meta_v9_recovery_20260717/residual_state_mda95_hier_newaegmm_frozenpremarch_v2/oos_predictions.parquet",
)
DERIVED_ANCHORS = {"base_score_rank_pct_train_prior"}
STATIC_FEATURE_BATCH = 96
STATIC_SYMBOL_BATCH = 24
# All ordinary arms must share one main-head seed per OOS fold.  Otherwise the
# number of added inputs changes LightGBM's bagging path and confounds an
# ablation with stochastic model variation, including when an input is gated
# to a constant zero value.
MAIN_HEAD_SEED_OFFSET = 90_000
ARM_FEATURES = {
    "M0_current_meta_head": [],
    "M1_good_trade_feature": ["meta_aux_good_trade_oof"],
    "M2_conditional_path_feature": ["meta_aux_good_trade_oof", "meta_aux_conditional_path_oof"],
    "M3_local_size_risk_feature": ["meta_aux_good_trade_oof", "meta_aux_conditional_path_oof", "meta_aux_local_size_risk_oof"],
    "M4_residual_state_features": [
        "meta_aux_good_trade_oof", "meta_aux_conditional_path_oof", "meta_aux_local_size_risk_oof",
        "meta_aux_negative_residual_oof",
        *[f"meta_aux_{name}_oof" for name in STATE_CELLS],
    ],
    "M5_residual_context_features": [
        "meta_aux_good_trade_oof", "meta_aux_conditional_path_oof", "meta_aux_local_size_risk_oof",
        "meta_aux_negative_residual_oof",
        *[f"meta_aux_{name}_oof" for name in STATE_CELLS],
        "meta_resid_arch_support_log1p", "meta_resid_arch_entropy",
        "meta_resid_arch_expected_hit_surprise", "meta_resid_arch_expected_dirty_positive",
    ],
    # M4/M5 confound the useful negative-residual signal with raw support and
    # entropy, which the leaf audit found harmful.  Test the causal failure
    # probability and the shrunk expected-hit-surprise context independently.
    "M12_good_trade_plus_negative_residual_probability": [
        "meta_aux_good_trade_oof",
        "meta_aux_negative_residual_oof",
    ],
    "M13_good_trade_plus_expected_hit_surprise": [
        "meta_aux_good_trade_oof",
        "meta_resid_arch_expected_hit_surprise",
    ],
    "M14_good_trade_negative_residual_plus_expected_hit_surprise": [
        "meta_aux_good_trade_oof",
        "meta_aux_negative_residual_oof",
        "meta_resid_arch_expected_hit_surprise",
    ],
    # M6 keeps the M1 good-trade probability and adds a wider set of frozen
    # side x archetype AE/GMM posterior priors.  These are state probabilities
    # for specific failure mechanisms, not post-score rank adjustments.
    "M6_good_trade_plus_residual_event_states": [
        "meta_aux_good_trade_oof",
        *residual_event_distilled_feature_names(include_market=False),
    ],
    # M7 is the selective, economic version of M6.  It deliberately omits
    # raw AE/GMM uncertainty, distances and temporal summaries: the meta head
    # receives only frozen probabilities for named executable mechanisms.
    "M7_good_trade_plus_residual_quality_probabilities": [
        "meta_aux_good_trade_oof",
        *residual_event_quality_probability_feature_names(),
    ],
}
# Each probability must earn its place against M1 independently.  A bundle can
# lose despite a useful individual state because the local priors are strongly
# correlated.  These arms are ordinary meta-head feature ablations, never
# score overlays or policy multipliers.
ARM_FEATURES.update(
    {
        f"M8q_{target}": [
            "meta_aux_good_trade_oof",
            f"resid_event_aegmm_expected_{target}",
        ]
        for target in EXECUTABLE_FAILURE_TARGETS
    }
)

# Expected downside magnitude is a complementary supervised residual feature.
# It is an OOF regression of causal score-conditioned EV shortfall, rather than
# a binary event flag.  The regular meta head can therefore rank a moderate
# risk state below a severe one without a brittle hard event threshold.
ARM_FEATURES["M11_residual_shortfall_regression"] = [
    "meta_aux_good_trade_oof",
    "meta_aux_residual_ev_shortfall_oof",
]

# M15 is a train-OOF-selected input-contract route, not a score blend.  It
# uses the M1 head by default and the M12 feature contract only for a
# side/archetype stream where prior chronological OOF selections show a
# material EV improvement without path-quality deterioration.
ARM_FEATURES["M15_oof_routed_m1_m12"] = [
    "meta_aux_good_trade_oof",
    "meta_aux_negative_residual_oof",
]

# These are fixed, causal lifecycle-state coordinates rather than outcome
# priors.  Their values are bounded phase intensities (effectively soft state
# probabilities) built from contemporaneous/past OI, funding, breadth and
# cross-asset geometry.  They are evaluated as ordinary meta inputs.
PHASE_AUX_FEATURES = tuple(f"meta_aux_{name}" for name in PHASE_STATE_FEATURES)
LOCAL_TRANSITION_RESIDUAL_FEATURE = "meta_aux_local_transition_negative_hit_oof"
AEGMM_TRANSITION_RESIDUAL_FEATURE = "meta_aux_aegmm_transition_negative_hit_oof"
AEGMM_TRANSITION_SOURCE_FEATURES = (
    "meta_aegmm_transition_posterior_tv_1h",
    "meta_aegmm_transition_posterior_tv_4h",
    "meta_aegmm_transition_cluster_switch_1h",
    "meta_aegmm_transition_cluster_switch_4h",
    "meta_aegmm_transition_entropy_delta_1h",
    "meta_aegmm_transition_entropy_delta_4h",
    "meta_aegmm_transition_posterior_max_delta_1h",
    "meta_aegmm_transition_posterior_max_delta_4h",
    "meta_aegmm_transition_reconstruction_delta_1h",
    "meta_aegmm_transition_reconstruction_delta_4h",
    "meta_aegmm_transition_mahal_delta_1h",
    "meta_aegmm_transition_mahal_delta_4h",
    "meta_aegmm_transition_ood_delta_1h",
    "meta_aegmm_transition_ood_delta_4h",
    "meta_aegmm_transition_latent_speed_1h",
    "meta_aegmm_transition_latent_speed_4h",
    "meta_aegmm_transition_market_breadth_1h",
    "meta_aegmm_transition_market_breadth_4h",
    "meta_aegmm_transition_market_entropy_delta_1h",
    "meta_aegmm_transition_market_ood_delta_1h",
)
AEGMM_COMPONENT_TRANSITION_SOURCE_FEATURES = tuple(
    feature
    for component in range(6)
    for feature in (
        f"meta_aegmm_transition_prob_{component}_delta_1h",
        f"meta_aegmm_transition_prob_{component}_delta_4h",
        f"meta_aegmm_transition_prob_{component}_enter_breadth_1h",
        f"meta_aegmm_transition_prob_{component}_exit_breadth_1h",
    )
)
AEGMM_DOMINANT_STATE_TRANSITION_SOURCE_FEATURES = (
    "meta_aegmm_transition_dominant_state_age_24h_norm",
    "meta_aegmm_transition_dominant_switch_count_4h",
    "meta_aegmm_transition_dominant_switch_count_8h",
    "meta_aegmm_transition_market_dominant_switch_breadth_1h",
    "meta_aegmm_transition_market_dominant_switch_breadth_4h",
    "meta_aegmm_transition_market_dominant_concentration",
    "meta_aegmm_transition_market_dominant_entropy",
)
AEGMM_COMPONENT_TRANSITION_RESIDUAL_FEATURE = (
    "meta_aux_aegmm_component_transition_negative_hit_oof"
)
AEGMM_DURABLE_TRANSITION_RESIDUAL_FEATURE = (
    "meta_aux_aegmm_durable_transition_negative_hit_oof"
)
TRANSITION_STATE_ANCHORS = (
    "score",
    "base_score_rank_pct_train_prior",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
)
ARM_FEATURES["M16_good_trade_plus_market_transition_states"] = [
    "meta_aux_good_trade_oof",
    *PHASE_AUX_FEATURES,
]
# A full lifecycle bundle can be redundant even when one transition provides
# incremental information.  Test each bounded state intensity independently
# against M1 before considering any aggregate phase representation.
ARM_FEATURES.update(
    {
        f"M17p_{name}": ["meta_aux_good_trade_oof", f"meta_aux_{name}"]
        for name in PHASE_STATE_FEATURES
    }
)
# Raw phase intensities can be helpful in one month and harmful in the next
# because their sign is not globally stable across archetypes.  This arm first
# learns an OOF local probability of *negative clean-hit residual* using only
# fixed base anchors plus causal transition coordinates, then supplies that
# probability as a normal meta-head input.  It is a supervised context feature,
# not an overlay or a hard gate.
ARM_FEATURES["M25_good_trade_plus_local_transition_negative_hit"] = [
    "meta_aux_good_trade_oof",
    LOCAL_TRANSITION_RESIDUAL_FEATURE,
]
# Frozen AE/GMM point states are row independent by contract.  M26 consumes
# separate causal 1h/4h transitions materialized on full symbol sequences and
# trains a side x archetype residual probability from them as a normal meta
# input.  It is explicitly not a policy overlay or a hard state gate.
ARM_FEATURES["M26_good_trade_plus_aegmm_transition_negative_hit"] = [
    "meta_aux_good_trade_oof",
    AEGMM_TRANSITION_RESIDUAL_FEATURE,
]
# M27 retains state-change direction: component entry and exit are distinct
# economic events, while M26's total variation treats them identically.
ARM_FEATURES["M27_good_trade_plus_aegmm_component_transition_negative_hit"] = [
    "meta_aux_good_trade_oof",
    AEGMM_COMPONENT_TRANSITION_RESIDUAL_FEATURE,
]
# M28 deliberately bypasses the compressed transition-risk probability.  The
# side-aware main meta head receives frozen, causal component direction and
# market participation directly, so it can learn that (for example) a given
# component entry is adverse for one archetype but useful for another.  These
# are observable state transitions, not outcomes, residuals, or a post-score
# policy overlay.
ARM_FEATURES["M28_good_trade_plus_aegmm_component_transition_inputs"] = [
    "meta_aux_good_trade_oof",
    *AEGMM_COMPONENT_TRANSITION_SOURCE_FEATURES,
]
# M29 makes the transition representation durable: state age, recent
# turnover, and market synchronization distinguish a genuine entry/exit into
# a state from transient hourly posterior noise.  As with M28, the main head
# learns the score x side x archetype interaction directly.
ARM_FEATURES["M29_good_trade_plus_aegmm_durable_transition_inputs"] = [
    "meta_aux_good_trade_oof",
    *AEGMM_DOMINANT_STATE_TRANSITION_SOURCE_FEATURES,
]
# M30 is a leakage-safe local contract selection, not an output blend.  It
# fits M1 and M29 independently, chooses M29 only for side x archetype streams
# where chronological OOF top-tail EV improves without material bad-MAE harm,
# then refits the selected contract on the entire permitted fold history.
ARM_FEATURES["M30_oof_routed_m1_m29_durable_transition"] = [
    "meta_aux_good_trade_oof",
    *AEGMM_DOMINANT_STATE_TRANSITION_SOURCE_FEATURES,
]
ARM_FEATURES["M31_good_trade_plus_aegmm_durable_transition_negative_hit"] = [
    "meta_aux_good_trade_oof",
    AEGMM_DURABLE_TRANSITION_RESIDUAL_FEATURE,
]

# A richer, semantic side x archetype recognizer was already available in the
# repository, but was never evaluated as a normal M1 meta-head input.  Unlike
# raw support/entropy, every retained coordinate has an economic meaning and
# is produced by a frozen recognizer from observable pre-entry inputs only.
# These are intentionally broader than the compact expected-hit-surprise prior:
# they distinguish dirty high-confidence rows, timeout-prone rows, bad-MAE
# false positives, high-variance uncertainty, and missed clean opportunities.
SEMANTIC_RESIDUAL_PROBABILITY_FEATURES = tuple(
    f"{RESIDUAL_FEATURE_PREFIX}prob__{name}" for name in SEMANTIC_ARCHETYPES
)
SEMANTIC_RESIDUAL_EXPECTED_FEATURES = (
    f"{RESIDUAL_FEATURE_PREFIX}expected_ev",
    f"{RESIDUAL_FEATURE_PREFIX}expected_bad_mae",
    f"{RESIDUAL_FEATURE_PREFIX}expected_timeout",
    f"{RESIDUAL_FEATURE_PREFIX}expected_dirty_positive",
)
SEMANTIC_RESIDUAL_FEATURES = (
    *SEMANTIC_RESIDUAL_PROBABILITY_FEATURES,
    *SEMANTIC_RESIDUAL_EXPECTED_FEATURES,
)
LEGACY_RESIDUAL_CONTEXT_FEATURES = (
    "meta_resid_arch_support_log1p",
    "meta_resid_arch_entropy",
    "meta_resid_arch_expected_hit_surprise",
    "meta_resid_arch_expected_dirty_positive",
)
ARM_FEATURES["M18_semantic_residual_state_priors"] = [
    "meta_aux_good_trade_oof",
    *SEMANTIC_RESIDUAL_FEATURES,
]
ARM_FEATURES["M20_semantic_state_plus_expected_hit_surprise"] = [
    "meta_aux_good_trade_oof",
    "meta_resid_arch_expected_hit_surprise",
    *SEMANTIC_RESIDUAL_FEATURES,
]
ARM_FEATURES.update(
    {
        f"M19s_{name.removeprefix(RESIDUAL_FEATURE_PREFIX)}": [
            "meta_aux_good_trade_oof",
            name,
        ]
        for name in SEMANTIC_RESIDUAL_FEATURES
    }
)

# The first individual-state screen shows a useful trade-off: dirty
# high-confidence probability improves top-tail EV/clean precision, while
# low-edge-noise is the only semantic state that reduces adverse-surprise
# persistence.  Keep these compact combinations as ordinary meta-head inputs
# so the head can learn their interaction rather than applying another policy
# blend or hard gate.
ARM_FEATURES.update(
    {
        "M21_dirty_high_confidence_plus_low_edge_noise": [
            "meta_aux_good_trade_oof",
            f"{RESIDUAL_FEATURE_PREFIX}prob__base_dirty_high_confidence",
            f"{RESIDUAL_FEATURE_PREFIX}prob__base_low_edge_noise",
        ],
        "M21_dirty_high_confidence_plus_slow_timeout": [
            "meta_aux_good_trade_oof",
            f"{RESIDUAL_FEATURE_PREFIX}prob__base_dirty_high_confidence",
            f"{RESIDUAL_FEATURE_PREFIX}prob__base_slow_timeout_positive",
        ],
        "M21_dirty_high_confidence_low_edge_timeout": [
            "meta_aux_good_trade_oof",
            f"{RESIDUAL_FEATURE_PREFIX}prob__base_dirty_high_confidence",
            f"{RESIDUAL_FEATURE_PREFIX}prob__base_low_edge_noise",
            f"{RESIDUAL_FEATURE_PREFIX}prob__base_slow_timeout_positive",
        ],
    }
)

# The raw auxiliary probabilities can be locally useful but harmful when the
# corresponding rare-event classifier is uninformative for another
# side/archetype.  M10g evaluates an OOF reliability-gated version as an
# ordinary meta input.  This is not an execution policy gate: it only removes
# a noisy auxiliary feature before the fitted meta head sees it.
RELIABILITY_GATED_MECHANISMS = (
    "top_tail_residual_timeout_loss",
    "top_tail_residual_local_adverse_episode_6h",
    # Candidate-conditioned false-positive risk is the first new mechanism
    # with a small but consistent EV/clean-path improvement.  Retain it only
    # where prior OOF separation verifies that the probability carries local
    # economic information; inactive streams receive a neutral zero input.
    "candidate_top20_residual_false_positive",
    "candidate_top20_cross_archetype_adverse_contagion",
    # The following candidate-conditioned targets extend the failure taxonomy
    # without forcing sparse event probabilities into every local stream.  They
    # are evaluated only as OOF-reliability-gated meta inputs: no side x
    # archetype receives a state feature unless its prior OOF high-probability
    # bucket has worse EV and a higher negative-EV rate than its low bucket.
    "candidate_top20_residual_first_touch_stop_loss",
    "candidate_top20_residual_timeout_loss",
    "candidate_top20_residual_systemic_loss",
    "candidate_top20_residual_local_adverse_episode_onset_6h",
    # Candidate-stream state labels with enough support to describe a shared
    # failure regime.  These deliberately use executable stop paths and a
    # longer episode bucket, rather than requiring every row in a 6h block to
    # be an unexpected terminal loss.
    "candidate_top20_cross_archetype_stop_contagion",
    "candidate_top20_residual_local_stop_episode_onset_12h",
    "candidate_top20_market_stop_pressure",
    "candidate_top20_reversal_after_initial_success",
    # The score-band contrast is useful in only a subset of local streams.
    # Gate the *input feature* by prior OOF economic separation, then let the
    # ordinary meta head decide how to use it.  This is not a policy gate.
    "top_tail_contrastive_executable_failure",
    # Transition-state inputs are the current priority.  A positive label
    # identifies a new episode using resolved train outcomes, but every
    # candidate in that onset bucket receives the historical state label.
    # This lets the auxiliary model learn an observable transition rather than
    # merely re-predicting the particular row that lost.
    "candidate_top20_local_loss_state_onset_6h",
    "candidate_top20_side_loss_state_onset_6h",
    "candidate_top20_cross_archetype_loss_state_onset_6h",
)
ARM_FEATURES.update(
    {
        f"M10g_{mechanism}": [
            "meta_aux_good_trade_oof",
            f"meta_aux_{mechanism}_reliability_gated_oof",
        ]
        for mechanism in RELIABILITY_GATED_MECHANISMS
    }
)

# Direct supervised residual-mechanism heads are deliberately separate from
# AE/GMM posterior priors.  They answer whether observable pre-entry features
# predict an *unexpected* top-tail failure conditional on score/archetype,
# then expose the cross-fitted probability to the normal meta head.
DIRECT_RESIDUAL_MECHANISMS = (
    # Broad residualized loss is deliberately included before narrower path
    # mechanisms.  It has materially higher support and asks the meta head to
    # resolve the operational ambiguity: an apparently strong top-tail row
    # that is unexpectedly negative after costs.
    "top_tail_residual_negative_ev",
    "top_tail_residual_false_positive",
    "top_tail_residual_clean_cost_fragile",
    "top_tail_residual_adverse_loss",
    "top_tail_residual_timeout_loss",
    # Path damage can be economically distinct from an immediate net loss.
    # These labels preserve that distinction and let the meta head learn its
    # own EV/path-quality trade-off instead of hard-blocking survivors.
    "top_tail_adverse_path_survivor",
    "top_tail_dirty_positive_survivor",
    "top_tail_timeout_positive_survivor",
    # A timestamp-side batch failure is a market-state target.  Its label is
    # constructed from resolved train outcomes, while its classifier sees only
    # causal pre-entry features at OOS/inference.
    "top_tail_residual_systemic_loss",
    # Episode labels reduce one-row outcome noise.  They are fitted from
    # resolved training outcomes only, then predicted from the same observable
    # market/AE-GMM/base-context feature space as the normal meta head.
    "top_tail_residual_local_loss_episode_6h",
    "top_tail_residual_local_adverse_episode_6h",
    "top_tail_residual_market_loss_episode_6h",
    # Onsets are much sparser than episode membership. They model the state
    # transition into a failure regime rather than relabelling every row in a
    # persistent bad period as a fresh signal.
    "top_tail_residual_local_loss_episode_onset_6h",
    "top_tail_residual_local_adverse_episode_onset_6h",
    "top_tail_residual_market_loss_episode_onset_6h",
    # Isolated local failures need not have the same observable mechanism as
    # a market-wide failure. This target keeps that microstructure/state case
    # separate from the systemic-loss head.
    "top_tail_residual_idiosyncratic_loss",
    # Candidate-conditioned versions remove the confound in the earlier
    # top-tail heads.  They are fitted only on the observable base top-20
    # population, so they answer whether a row is a failure *given that it is
    # already a plausible candidate*, rather than relearning the base rank.
    "candidate_top20_residual_negative_ev",
    "candidate_top20_residual_false_positive",
    "candidate_top20_residual_clean_cost_fragile",
    "candidate_top20_residual_adverse_loss",
    "candidate_top20_residual_first_touch_stop_loss",
    "candidate_top20_residual_timeout_loss",
    "candidate_top20_residual_systemic_loss",
    "candidate_top20_residual_local_adverse_episode_onset_6h",
    "candidate_top20_cross_archetype_stop_contagion",
    "candidate_top20_residual_local_stop_episode_onset_12h",
    "candidate_top20_market_stop_pressure",
    "candidate_top20_reversal_after_initial_success",
    "candidate_top20_adverse_path_survivor",
    # These are cross-archetype market-state labels. They distinguish a
    # symbol/local miss from a timestamp-side state where several independent
    # archetype streams simultaneously underperform.
    "candidate_top20_cross_archetype_loss_contagion",
    "candidate_top20_cross_archetype_adverse_contagion",
    # Candidate-stream transition states.  These are the next primary
    # experiments; direct false-positive and path-risk mechanisms remain in
    # the taxonomy for diagnostics but are deliberately not in this run.
    "candidate_top20_local_loss_state_onset_6h",
    "candidate_top20_side_loss_state_onset_6h",
    "candidate_top20_cross_archetype_loss_state_onset_6h",
    # Named side x archetype mechanisms from the leaf audit.  The labels use
    # only realized training outcomes; their classifiers are intentionally
    # allowed to discover whether AE/GMM OOD, posterior geometry, market state
    # or microstructure explains the failure from observable inputs.
    "long_mixed_latent_misfire",
    "short_mixed_off_manifold",
    "short_default_latent_uncertainty",
    "top_tail_reversal_after_initial_success",
    "long_mixed_reversal_after_initial_success",
    "short_mixed_reversal_after_initial_success",
    "long_breakout_overconfident_path_loss",
    "short_breakout_overconfident_path_loss",
    # Score-band-matched failure versus clean-executable contrast.  This is
    # not another global loss target: it removes the base-score explanation
    # before the auxiliary learner sees a row.
    "top_tail_contrastive_executable_failure",
    # This is the symmetric promotion target: clean, positive-EV rows just
    # below the current decision boundary.  It gives the meta head evidence
    # about states where the base under-ranks opportunities rather than only
    # evidence about trades to avoid.
    "near_tail_clean_executable",
)

# These labels describe a transition in the model's clean-hit reliability,
# conditional on the existing base candidate stream.  They are intentionally
# trained against a causal clean-execution residual rather than terminal EV:
# a market-state deterioration can first show up as lost clean hits before it
# becomes a cluster of fully negative outcomes.
TRANSITION_STATE_MECHANISMS = frozenset(
    {
        "candidate_top20_local_loss_state_onset_6h",
        "candidate_top20_side_loss_state_onset_6h",
        "candidate_top20_cross_archetype_loss_state_onset_6h",
    }
)
ARM_FEATURES.update(
    {
        f"M9s_{mechanism}": [
            "meta_aux_good_trade_oof",
            f"meta_aux_{mechanism}_oof",
        ]
        for mechanism in DIRECT_RESIDUAL_MECHANISMS
    }
)

# Transition states are evaluated independently first.  They must earn value
# against M1 one at a time before any correlated bundle is considered.
for _transition in (
    "candidate_top20_local_loss_state_onset_6h",
    "candidate_top20_side_loss_state_onset_6h",
    "candidate_top20_cross_archetype_loss_state_onset_6h",
):
    ARM_FEATURES[f"M24_transition_{_transition}"] = [
        "meta_aux_good_trade_oof",
        f"meta_aux_{_transition}_reliability_gated_oof",
    ]

# Cross-archetype adverse contagion improves EV and breadth of clean selection,
# while candidate false-positive risk is the only direct state that materially
# improves April's signed/negative surprise behaviour.  This is an ordinary
# two-input meta-head ablation, not an overlay or a policy gate.
ARM_FEATURES["M23_crossarch_adverse_plus_candidate_false_positive"] = [
    "meta_aux_good_trade_oof",
    "meta_aux_candidate_top20_cross_archetype_adverse_contagion_oof",
    "meta_aux_candidate_top20_residual_false_positive_oof",
]

RESIDUAL_STATE_FEATURES = residual_event_distilled_feature_names(include_market=False)
MIN_RESIDUAL_STATE_HISTORY = 2_500
RESIDUAL_STATE_FORBIDDEN = set(RESIDUAL_STATE_OUTCOME_COLUMNS).union(
    {
        "meta_target_soft",
        "first_touch_mae_to_sl",
        "first_touch_bad_mae_1r",
        "__first_touch_mae_to_sl__",
        "__first_touch_target_soft__",
        "__first_touch_policy_soft__",
    }
)


def _load_feature_contract(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_features")
    if not isinstance(selected, dict):
        raise ValueError(f"Missing side-specific selected_features in {path}")
    required = {"long", "short"}
    if not required.issubset(selected):
        raise ValueError(f"Feature contract must contain long and short features: {path}")
    return {
        side: list(dict.fromkeys(str(name) for name in selected[side]))
        for side in sorted(required)
    }


def _read_matching_label_rows(
    frame: pd.DataFrame,
    *,
    labels_root: Path,
    months: list[pd.Period],
    requested_columns: list[str],
) -> pd.DataFrame:
    """Batch-read only label rows that occur in a candidate ledger.

    Label shards are much wider than either model fitting or support analysis
    needs.  Materializing them whole creates a large, avoidable memory spike.
    This preserves the exact UTC/symbol/side join contract while streaming
    narrow batches and discarding non-candidate labels immediately.
    """

    import pyarrow.parquet as pq

    key_columns = ["__ts__", "__symbol__", "side_name"]
    required = list(dict.fromkeys([*key_columns, *requested_columns]))
    parts: list[pd.DataFrame] = []
    for month in months:
        start = pd.Timestamp(month.start_time, tz="UTC")
        end = pd.Timestamp((month + 1).start_time, tz="UTC")
        keys = frame.loc[
            frame["__ts__"].ge(start) & frame["__ts__"].lt(end), key_columns
        ].drop_duplicates()
        if keys.empty:
            continue
        for path in _month_label_paths(labels_root, month):
            parquet = pq.ParquetFile(path)
            columns = [name for name in required if name in parquet.schema_arrow.names]
            if not set(required).issubset(columns):
                continue
            for batch in parquet.iter_batches(columns=columns, batch_size=100_000):
                part = batch.to_pandas()
                part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
                part = part.loc[part["__ts__"].notna()]
                if part.empty:
                    continue
                matched = part.merge(
                    keys,
                    on=key_columns,
                    how="inner",
                    validate="many_to_one",
                    sort=False,
                )
                if not matched.empty:
                    parts.append(matched)
    if not parts:
        return pd.DataFrame(columns=required)
    return pd.concat(parts, ignore_index=True, copy=False).drop_duplicates(
        key_columns, keep="last"
    )


def _load_saved_full_feature_ledgers(
    paths: list[Path],
    *,
    labels_root: Path,
    months: list[pd.Period],
    features_by_side: dict[str, list[str]],
    extra_feature_names: list[str] | tuple[str, ...] = (),
    full_months: set[str] | None = None,
    max_rows_per_train_month: int = 0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the persisted residual_state_mda95 matrices and attach soft labels.

    These OOS ledgers are the exact side-specific model matrices produced by
    the residual_state_mda95 training path.  Reusing them avoids recomputing
    static features while retaining the same causal feature values that the
    incumbent meta model saw.
    """

    import pyarrow.parquet as pq

    selected = list(
        dict.fromkeys(
            [
                *(feature for values in features_by_side.values() for feature in values),
                *map(str, extra_feature_names),
            ]
        )
    )
    if not months:
        return pd.DataFrame(), {}
    full_months = set(full_months or (str(month) for month in months))
    first_month = min(months)
    after_last_month = max(months) + 1
    lower_bound = pd.Timestamp(first_month.start_time, tz="UTC")
    upper_bound = pd.Timestamp(after_last_month.start_time, tz="UTC")
    structural = [
        "__ts__", "__symbol__", "side_name", "archetype_policy_key", "score",
        "ev_after_1pct", "clean_exec", "dirty_positive", "full_path_bad_mae_1r", "timeout",
        "base_margin_to_cutoff", "base_margin_to_cutoff_z", "base_signal_zscore_within_archetype",
    ]
    # First pass: establish the exact top-20 candidate population from narrow
    # structural columns.  Loading the complete 95-column model matrix for
    # every prior month before this step can exceed memory even though model
    # fitting later keeps only a B/M/E train sample.  Evaluation months are
    # intentionally never sampled.
    structural_parts: list[pd.DataFrame] = []
    for path in paths:
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [name for name in structural if name in schema]
        file = pq.ParquetFile(path)
        for batch in file.iter_batches(columns=columns, batch_size=100_000):
            part = batch.to_pandas()
            part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
            part = part.loc[
                part["__ts__"].ge(lower_bound)
                & part["__ts__"].lt(upper_bound)
            ]
            if not part.empty:
                structural_parts.append(part)
    if not structural_parts:
        return pd.DataFrame(), {}
    structure = pd.concat(structural_parts, ignore_index=True, copy=False)
    structure = structure.loc[structure["__ts__"].dt.to_period("M").isin(months)].copy()
    structure["base_rank_pct_by_timestamp"] = structure.groupby(
        "__ts__", observed=True
    )["score"].rank(method="average", pct=True).astype(np.float32)
    structure = structure.loc[structure["base_rank_pct_by_timestamp"].ge(0.80)].copy()
    retained: list[pd.DataFrame] = []
    for period, part in structure.groupby(structure["__ts__"].dt.to_period("M"), observed=True, sort=True):
        if str(period) in full_months or max_rows_per_train_month <= 0:
            retained.append(part)
        else:
            retained.append(_time_spread_sample(part, int(max_rows_per_train_month)))
    keys = pd.concat(retained, ignore_index=True, copy=False).loc[
        :, ["__ts__", "__symbol__", "side_name", "base_rank_pct_by_timestamp"]
    ]
    keys = keys.drop_duplicates(["__ts__", "__symbol__", "side_name"], keep="last")

    # Second pass: the expensive model columns are loaded only for retained
    # train/OOS keys.  This preserves exact OOS rows while bounding historical
    # train memory by the deterministic sample already used downstream.
    key_columns = ["__ts__", "__symbol__", "side_name"]
    wide_parts: list[pd.DataFrame] = []
    key_index = pd.MultiIndex.from_frame(keys.loc[:, key_columns])
    for path in paths:
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [name for name in dict.fromkeys([*structural, *selected]) if name in schema]
        file = pq.ParquetFile(path)
        for batch in file.iter_batches(columns=columns, batch_size=25_000):
            part = batch.to_pandas()
            part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
            part = part.loc[
                part["__ts__"].ge(lower_bound)
                & part["__ts__"].lt(upper_bound)
            ]
            if part.empty:
                continue
            part_index = pd.MultiIndex.from_frame(part.loc[:, key_columns])
            keep = key_index.get_indexer(part_index) >= 0
            if keep.any():
                wide_parts.append(part.loc[keep].copy())
    if not wide_parts:
        return pd.DataFrame(), {}
    frame = pd.concat(wide_parts, ignore_index=True, copy=False).drop_duplicates(
        key_columns, keep="last"
    )
    frame = frame.merge(keys, on=key_columns, how="inner", validate="one_to_one")
    missing_selected = [name for name in selected if name not in frame]
    if missing_selected:
        raise RuntimeError("Saved residual_state_mda95 ledger is missing selected features: " + ", ".join(missing_selected))
    labels = _read_matching_label_rows(
        frame,
        labels_root=labels_root,
        months=months,
        requested_columns=[
            "__first_touch_target_soft__",
            "__first_touch_mae_to_sl__",
        ],
    )
    labels = labels.rename(
        columns={
            "__first_touch_target_soft__": "meta_target_soft",
            "__first_touch_mae_to_sl__": "first_touch_mae_to_sl",
        }
    )
    frame = frame.merge(labels, on=["__ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    frame = frame.loc[frame["meta_target_soft"].notna()].copy()
    frame["first_touch_bad_mae_1r"] = (
        pd.to_numeric(frame["first_touch_mae_to_sl"], errors="coerce").fillna(0.0) >= 1.0
    ).astype(np.float32)
    # ``base_rank_pct_by_timestamp`` was calculated before train-month
    # sampling.  Retaining that full-stream rank is critical: recomputing it
    # after selecting the top-20 population would turn every surviving row
    # into an apparent top-ranked row.
    coverage = {
        str(period): int(len(part))
        for period, part in frame.groupby(frame["__ts__"].dt.to_period("M"), observed=True)
    }
    return frame, coverage


def _load_saved_direct_mechanism_support_ledgers(
    paths: list[Path],
    *,
    labels_root: Path,
    months: list[pd.Period],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load only outcome/anchor columns needed by the target-support audit.

    The full residual-state matrices contain roughly one hundred feature
    columns.  Support diagnostics need none of them, so reading the full
    contract needlessly creates a wide pandas allocation before any model is
    fitted.  Keep this loader intentionally separate from the model path: it
    cannot be reused to train a meta head.
    """

    import pyarrow.parquet as pq

    if not months:
        return pd.DataFrame(), {}
    first_month = min(months)
    after_last_month = max(months) + 1
    lower_bound = pd.Timestamp(first_month.start_time, tz="UTC").to_pydatetime()
    upper_bound = pd.Timestamp(after_last_month.start_time, tz="UTC").to_pydatetime()
    columns_needed = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "score",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    parts: list[pd.DataFrame] = []
    filters = [("__ts__", ">=", lower_bound), ("__ts__", "<", upper_bound)]
    for path in paths:
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [name for name in columns_needed if name in schema]
        parts.append(pq.read_table(path, columns=columns, filters=filters).to_pandas())
    frame = pd.concat(parts, ignore_index=True, copy=False)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.loc[frame["__ts__"].dt.to_period("M").isin(months)].copy()
    # Labels are substantially wider than this audit needs.  Read their four
    # structural columns in batches and retain only rows represented in the
    # compact OOS ledger; loading every monthly label frame at once can consume
    # several GB before a single support statistic is computed.
    labels = _read_matching_label_rows(
        frame,
        labels_root=labels_root,
        months=months,
        requested_columns=["__first_touch_mae_to_sl__"],
    )
    frame = frame.merge(
        labels,
        on=["__ts__", "__symbol__", "side_name"],
        how="left",
        validate="one_to_one",
    )
    frame["first_touch_bad_mae_1r"] = (
        pd.to_numeric(frame["__first_touch_mae_to_sl__"], errors="coerce").fillna(0.0)
        >= 1.0
    ).astype(np.float32)
    frame = frame.drop(columns=["__first_touch_mae_to_sl__"])
    required = ["score", "ev_after_1pct", "clean_exec", "full_path_bad_mae_1r", "timeout"]
    frame = frame.loc[frame[required].notna().all(axis=1)].copy()
    frame["base_rank_pct_by_timestamp"] = frame.groupby(
        "__ts__", observed=True
    )["score"].rank(method="average", pct=True).astype(np.float32)
    coverage = {
        str(period): int(len(part))
        for period, part in frame.groupby(frame["__ts__"].dt.to_period("M"), observed=True)
    }
    return frame, coverage


def _require_month_coverage(
    coverage: dict[str, int],
    months: list[pd.Period],
    *,
    context: str,
) -> None:
    """Reject an expanding-window claim when source months are absent.

    A sparse historical source must not silently turn a requested six- or
    twelve-month rare-state study into a shorter study.  The caller may use a
    deliberately shorter ``--train-months`` window, but every requested month
    has to be materially present in the exact persisted candidate stream.
    """

    missing = [str(month) for month in months if int(coverage.get(str(month), 0)) <= 0]
    if missing:
        raise RuntimeError(
            f"{context} is missing required candidate months: {', '.join(missing)}. "
            "Materialize the corresponding base/meta OOS handoff before running "
            "a longer-history residual-state ablation."
        )


def _hydrate_static_features(
    frame: pd.DataFrame,
    *,
    features_by_side: dict[str, list[str]],
    feature_store_id: str,
    min_complete_coverage: float,
    extra_feature_names: list[str] | tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fill the selected model contract from the canonical static store only.

    The compact handoff is a candidate ledger, not the authoritative feature
    surface.  This uses the same read-only static endpoint as train/replay/live
    and preserves handoff values only where they are already finite.
    """

    out = frame.copy(deep=False)
    required = list(
        dict.fromkeys(
            feature
            for values in features_by_side.values()
            for feature in values
            if feature not in DERIVED_ANCHORS
        )
    )
    # Extra fields support an explicit ablation only.  They are not part of
    # the frozen parent contract and consequently do not participate in its
    # joint-complete coverage requirement below.
    requested = list(dict.fromkeys([*required, *map(str, extra_feature_names)]))
    # A compact candidate handoff can legitimately contain sparse *derived*
    # fields (for example ``score`` on an invalid base row).  Those fields are
    # not part of the static feature store and must never be "repaired" by
    # asking the static reader for a same-named column.  Besides being
    # semantically wrong, one unavailable handoff-owned field made a whole
    # feature batch look empty and aborted historical hydration.  Only absent
    # columns are static-store candidates.  Existing incomplete columns remain
    # incomplete and are handled by the joint contract coverage check below.
    needed = [feature for feature in requested if feature not in out]
    if needed:
        store_ts = pd.to_datetime(str(feature_store_id), format="%Y%m%d_%H%M%S", utc=True)
        row_ts = pd.DatetimeIndex(pd.to_datetime(out["__ts__"], utc=True))
        row_symbols = out["__symbol__"].astype(str).to_numpy()
        buffers: dict[str, np.ndarray] = {}
        for feature in needed:
            if feature in out:
                buffers[feature] = pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=np.float32)
            else:
                buffers[feature] = np.full(len(out), np.nan, dtype=np.float32)
        all_symbols = np.unique(row_symbols)
        for symbol_start in range(0, len(all_symbols), STATIC_SYMBOL_BATCH):
            symbol_batch = all_symbols[symbol_start : symbol_start + STATIC_SYMBOL_BATCH]
            row_idx = np.flatnonzero(np.isin(row_symbols, symbol_batch))
            if not len(row_idx):
                continue
            batch_ts = pd.DatetimeIndex(pd.unique(row_ts.take(row_idx))).sort_values()
            batch_symbol_index = pd.Index(symbol_batch.astype(str))
            ts_idx = batch_ts.get_indexer(row_ts.take(row_idx))
            symbol_idx = batch_symbol_index.get_indexer(row_symbols[row_idx])
            valid = (ts_idx >= 0) & (symbol_idx >= 0)
            for feature_start in range(0, len(needed), STATIC_FEATURE_BATCH):
                feature_batch = needed[feature_start : feature_start + STATIC_FEATURE_BATCH]
                # The shared reader logs each file/key group.  This ablation
                # deliberately chunks reads for memory safety, so suppress only
                # that repetitive progress output while retaining exceptions.
                with contextlib.redirect_stdout(io.StringIO()):
                    loaded = read_static_features(
                        feature_store_ts=store_ts,
                        data_root=ROOT / "data_perp",
                        feature_keys=feature_batch,
                        symbols=batch_symbol_index.tolist(),
                        start_ts=batch_ts.min(),
                        end_ts=batch_ts.max(),
                        output_layout="panels",
                    )
                    # A newer side-local model contract can contain columns
                    # absent from an older static-store generation.  Treat an
                    # all-missing batch as unavailable evidence, not a reader
                    # failure.  The joint side-contract coverage guard below
                    # then rejects the run with the actual coverage rate;
                    # this also lets diagnostics report exactly which columns
                    # prevent an extended-history study.
                    if loaded is None or not hasattr(loaded, "get"):
                        continue
                    for feature in feature_batch:
                        panel = loaded.get(feature)
                        if not isinstance(panel, pd.DataFrame) or panel.empty:
                            continue
                        panel = panel.copy(deep=False)
                        panel.index = pd.to_datetime(panel.index, utc=True, errors="coerce")
                        panel.columns = panel.columns.astype(str)
                        values = panel.reindex(index=batch_ts, columns=batch_symbol_index).to_numpy(dtype=np.float32, copy=False)
                        hydrated = values[ts_idx[valid], symbol_idx[valid]]
                        target_idx = row_idx[valid]
                        missing = ~np.isfinite(buffers[feature][target_idx])
                        buffers[feature][target_idx[missing]] = hydrated[missing]
        for feature, values in buffers.items():
            out[feature] = values
    coverage: dict[str, float] = {}
    for side, selected in features_by_side.items():
        mask = out["side_name"].astype(str).str.lower().eq(side).to_numpy()
        if not mask.any():
            continue
        observable = [feature for feature in selected if feature not in DERIVED_ANCHORS]
        values = out.loc[mask, observable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        complete = np.isfinite(values).all(axis=1) if len(values) else np.array([], dtype=bool)
        rate = float(complete.mean()) if len(complete) else 0.0
        coverage[side] = rate
        if rate < min_complete_coverage:
            feature_rates = np.isfinite(values).mean(axis=0) if len(values) else np.zeros(len(observable))
            blockers = [
                f"{feature}:{float(feature_rate):.0%}"
                for feature, feature_rate in zip(observable, feature_rates)
                if float(feature_rate) < 0.999
            ]
            raise RuntimeError(
                f"{side} selected-feature joint coverage {rate:.2%} is below "
                f"the required {min_complete_coverage:.0%} in static store {feature_store_id}; "
                f"incomplete columns: {', '.join(blockers[:16])}"
            )
    return out, coverage


def _add_train_prior_rank(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Materialize the protected rank anchor from each fold's train scores."""

    reference = np.sort(pd.to_numeric(train["score"], errors="coerce").dropna().to_numpy(dtype=np.float32))
    for frame in (train, test):
        score = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=np.float32)
        rank = np.full(len(frame), 0.5, dtype=np.float32)
        finite = np.isfinite(score)
        if len(reference):
            left = np.searchsorted(reference, score[finite], side="left")
            right = np.searchsorted(reference, score[finite], side="right")
            rank[finite] = ((left + right) / (2.0 * len(reference))).astype(np.float32)
        frame["base_score_rank_pct_train_prior"] = rank


def _assert_contract_coverage(
    frame: pd.DataFrame,
    *,
    features_by_side: dict[str, list[str]],
    min_complete_coverage: float,
    enforce: bool = True,
) -> dict[str, float]:
    """Require the complete side-specific model contract on usable rows."""

    result: dict[str, float] = {}
    for side, selected in features_by_side.items():
        mask = frame["side_name"].astype(str).str.lower().eq(side).to_numpy()
        if not mask.any():
            continue
        values = frame.loc[mask, selected].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        rate = float(np.isfinite(values).all(axis=1).mean())
        result[side] = rate
        if enforce and rate < min_complete_coverage:
            raise RuntimeError(
                f"{side} saved model matrix joint coverage {rate:.2%} is below "
                f"the required {min_complete_coverage:.0%}"
            )
    return result


def _probability_metrics(y: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float | None]:
    if len(y) == 0 or np.unique(y).size < 2:
        return {f"{prefix}_auc": None, f"{prefix}_ap": None}
    return {
        f"{prefix}_auc": float(roc_auc_score(y, pred)),
        f"{prefix}_ap": float(average_precision_score(y, pred)),
    }


def _lag1_autocorrelation(values: np.ndarray) -> float:
    """Return a finite lag-one autocorrelation without degenerate warnings."""

    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) < 4 or np.std(data[:-1]) <= 1e-10 or np.std(data[1:]) <= 1e-10:
        return float("nan")
    return float(np.corrcoef(data[:-1], data[1:])[0, 1])


def _selected_hit_surprise_metrics(
    selected: pd.DataFrame,
    train: pd.DataFrame,
) -> dict[str, float]:
    """Assess daily signed surprise using an expectation fit only on train.

    This is evaluation-only.  The selected population can differ by arm, but
    the expectation is frozen from the same pre-OOS train rows, so it cannot
    use selected or realized OOS outcomes to improve its own residual metric.
    """

    if selected.empty:
        return {
            "mean_hit_surprise": float("nan"),
            "signed_hit_surprise_ac": float("nan"),
            "negative_hit_surprise_ac": float("nan"),
            "positive_hit_surprise_ac": float("nan"),
        }
    # ``_select_top10`` retains source indices after timestamp sorting; the
    # hierarchical expectation mapper is positional by design.  Reindexing
    # here is evaluation-only and leaves the selected rows untouched.
    selected = selected.reset_index(drop=True)
    state = _fit_score_value_map(train, value_col="clean_exec")
    expected = _expected_ev(selected, state)
    actual = pd.to_numeric(selected["clean_exec"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    surprise = actual - np.asarray(expected, dtype=np.float32)
    day = pd.to_datetime(selected["__ts__"], utc=True, errors="coerce").dt.floor("D")
    daily = (
        pd.DataFrame({"day": day, "surprise": surprise})
        .dropna(subset=["day"])
        .groupby("day", observed=True, sort=True)["surprise"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    return {
        "mean_hit_surprise": float(np.mean(daily)) if len(daily) else float("nan"),
        "signed_hit_surprise_ac": _lag1_autocorrelation(daily),
        "negative_hit_surprise_ac": _lag1_autocorrelation(np.minimum(daily, 0.0)),
        "positive_hit_surprise_ac": _lag1_autocorrelation(np.maximum(daily, 0.0)),
    }


def _selected_ledger(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Persist the compact selected population needed for later parity audit."""

    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "score",
        "__selection_score__",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    out = frame.reindex(columns=[name for name in columns if name in frame]).copy()
    out["arm"] = str(arm)
    return out


def _fit_good(train: pd.DataFrame, test: pd.DataFrame, features: list[str] | dict[str, list[str]], seed: int) -> np.ndarray:
    return _fit_predict(train, test, features=features, target=_good_trade_target(train), seed=seed)[0]


def _fit_path(train: pd.DataFrame, test: pd.DataFrame, features: list[str] | dict[str, list[str]], seed: int) -> np.ndarray:
    plausible = train.loc[(pd.to_numeric(train["ev_after_1pct"], errors="coerce") > 0.0) | (_good_trade_target(train) > 0.0)]
    if len(plausible) < 1_500:
        return np.full(len(test), float(_conditional_path_target(train).mean()), dtype=np.float32)
    return _fit_predict(plausible.reset_index(drop=True), test, features=features, target=_conditional_path_target(plausible), seed=seed)[0]


def _fit_residual(train: pd.DataFrame, test: pd.DataFrame, features: list[str] | dict[str, list[str]], seed: int) -> np.ndarray:
    local, target, _ = _causal_residual_target(train, value_col="clean_exec", label_col="__negative_hit_residual_event__")
    return _fit_predict(local.reset_index(drop=True), test, features=features, target=target, seed=seed)[0]


def _mechanism_residual_target(
    train: pd.DataFrame,
    mechanism: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Build the causal residual reference appropriate to one mechanism."""

    if mechanism in TRANSITION_STATE_MECHANISMS:
        return _causal_residual_target(
            train,
            value_col="clean_exec",
            label_col="__negative_hit_residual_event__",
        )
    return _causal_residual_target(
        train,
        value_col="ev_after_1pct",
        label_col="__negative_ev_residual_event__",
    )


def _fit_residual_mechanism(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] | dict[str, list[str]],
    seed: int,
    mechanism: str,
) -> np.ndarray:
    """Fit a direct, causal classifier for one unexpected failure mechanism."""

    local, negative_residual, _ = _mechanism_residual_target(train, mechanism)
    if len(local) < 1_500:
        return np.full(len(test), 0.0, dtype=np.float32)
    sample_weight: np.ndarray | None = None
    if mechanism == "top_tail_contrastive_executable_failure":
        local, target, sample_weight = _contrastive_executable_failure_training_set(
            local,
            negative_residual,
        )
    else:
        target = _direct_residual_mechanism_target(local, negative_residual, mechanism)
        if mechanism.startswith("candidate_top20_"):
            # The base handoff is already candidate-filtered.  Restricting the
            # auxiliary fit to its top-20 score population prevents a sparse
            # event label from being dominated by the trivial score-tail
            # distinction.  The resulting probability is then a conditional
            # path/cost risk estimate that the normal meta head can combine
            # with its protected score anchors.
            rank = pd.to_numeric(
                local.get(
                    "base_rank_pct_by_timestamp",
                    pd.Series(0.0, index=local.index),
                ),
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=np.float32)
            eligible = rank >= 0.80
            local = local.loc[eligible].reset_index(drop=True)
            target = np.asarray(target, dtype=np.float32)[eligible]
    # Sparse rare-event heads should remain neutral rather than manufacture a
    # confident probability from a handful of positives.
    positive = int(np.sum(target > 0.5))
    negative = int(np.sum(target <= 0.5))
    if positive < 32 or negative < 128:
        return np.full(len(test), float(target.mean()), dtype=np.float32)
    return _fit_predict(
        local.reset_index(drop=True),
        test,
        features=features,
        target=target.astype(np.float32),
        seed=seed,
        sample_weight=sample_weight,
    )[0]


_PHASE_SOURCE_NAMES = (
    "mkt_median_oi_chg_4h_rz",
    "mkt_median_oi_chg_1h_rz",
    "mkt_pct_oi_chg_4h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus2",
    "mkt_oi_flush_breadth_accel_1h",
    "mkt_oi_flush_breadth_recovery_4h",
    "mkt_pct_price_down_oi_down_4h",
    "mkt_pct_price_up_oi_down_1h",
    "mkt_pct_price_up_oi_up_4h",
    "market_breadth_chg_1h",
    "market_breadth_recovery_from_6h_min",
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "mkt_leverage_rebuild_score",
    "market_pc1_variance_share_12h",
    "asset_flush_exhaustion_score",
    "asset_short_covering_score",
)
_PHASE_SOURCE_PREFIXES = (
    "",
    "full_universe__median__",
    "universe__median__",
    "selected__median__",
)
_MARKET_PHASE_SOURCE_NAMES = tuple(
    name
    for name in _PHASE_SOURCE_NAMES
    if name.startswith("mkt_") or name.startswith("market_")
)


def _hydrate_market_phase_sources(
    frame: pd.DataFrame,
    *,
    feature_store_id: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load market-wide lifecycle fields once, then broadcast by timestamp.

    Market OI/breadth/synchronization fields are identical across the tradable
    universe at a timestamp.  Hydrating every candidate symbol wastes memory
    and can bias a market state toward the currently selected candidates.  A
    small sentinel set is read through the canonical static-store endpoint,
    medianed at each timestamp, and joined back to every candidate row.
    """

    out = frame.copy(deep=False)
    timestamps = pd.DatetimeIndex(pd.to_datetime(out["__ts__"], utc=True, errors="coerce"))
    valid = timestamps.notna()
    if not valid.any():
        return out, {"status": "neutral_invalid_timestamps", "source_columns": []}
    observed_symbols = (
        out.get("__symbol__", pd.Series(dtype="object"))
        .astype(str)
        .value_counts()
        .index.tolist()
    )
    symbols = list(dict.fromkeys(["BTC/USD:USD", "ETH/USD:USD", *observed_symbols[:6]]))
    store_ts = pd.to_datetime(str(feature_store_id), format="%Y%m%d_%H%M%S", utc=True)
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = read_static_features(
            feature_store_ts=store_ts,
            data_root=ROOT / "data_perp",
            feature_keys=list(_MARKET_PHASE_SOURCE_NAMES),
            symbols=symbols,
            start_ts=timestamps[valid].min(),
            end_ts=timestamps[valid].max(),
            output_layout="panels",
        )
    if loaded is None or not hasattr(loaded, "get"):
        return out, {"status": "neutral_store_no_panels", "source_columns": []}
    coverage: dict[str, float] = {}
    for feature in _MARKET_PHASE_SOURCE_NAMES:
        panel = loaded.get(feature)
        if not isinstance(panel, pd.DataFrame) or panel.empty:
            coverage[feature] = 0.0
            continue
        panel = panel.copy(deep=False)
        panel.index = pd.to_datetime(panel.index, utc=True, errors="coerce")
        values = panel.reindex(index=timestamps[valid]).median(axis=1, skipna=True).to_numpy(dtype=np.float32)
        destination = np.full(len(out), np.nan, dtype=np.float32)
        destination[np.flatnonzero(valid)] = values
        # Preserve an existing finite store value if this experiment is fed a
        # wide ledger in the future; otherwise use the canonical sentinel map.
        existing = pd.to_numeric(out.get(feature, pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(dtype=np.float32)
        out[feature] = np.where(np.isfinite(existing), existing, destination).astype(np.float32)
        coverage[feature] = float(np.isfinite(out[feature]).mean())
    return out, {
        "status": "complete",
        "symbols": symbols,
        "source_columns": list(_MARKET_PHASE_SOURCE_NAMES),
        "coverage": coverage,
    }


def _causal_phase_state_context(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build order-invariant pre-entry lifecycle state inputs for a fold.

    Phase coordinates are market/side states, not asset-row indicators.  The
    source is therefore collapsed to one median panel row per side x timestamp
    before causal deltas are calculated, then broadcast back to candidates.
    Outcome and target fields are removed before the phase builder is called.
    """

    source_columns = ["__ts__", "side_name"]
    available = set(train.columns).union(test.columns)
    for name in _PHASE_SOURCE_NAMES:
        for prefix in _PHASE_SOURCE_PREFIXES:
            candidate = f"{prefix}{name}"
            if candidate in available:
                source_columns.append(candidate)
    source_columns = list(dict.fromkeys(source_columns))
    if len(source_columns) <= 2:
        neutral_train = pd.DataFrame(0.0, index=np.arange(len(train)), columns=PHASE_AUX_FEATURES, dtype=np.float32)
        neutral_test = pd.DataFrame(0.0, index=np.arange(len(test)), columns=PHASE_AUX_FEATURES, dtype=np.float32)
        return neutral_train, neutral_test, {"status": "neutral_missing_phase_sources", "source_columns": []}

    def _project(frame: pd.DataFrame, split: int) -> pd.DataFrame:
        projected = frame.reindex(columns=source_columns).copy(deep=False)
        projected["__ts__"] = pd.to_datetime(projected["__ts__"], utc=True, errors="coerce")
        projected["side_name"] = projected["side_name"].astype(str).str.lower()
        projected["__phase_split__"] = np.int8(split)
        projected["__phase_row__"] = np.arange(len(projected), dtype=np.int64)
        return projected

    combined = pd.concat([_project(train, 0), _project(test, 1)], ignore_index=True, copy=False)
    # Explicitly exclude the full outcome vocabulary as a defensive contract
    # check.  The selected source list contains market-state names only, but
    # this makes accidental future expansion safe.
    forbidden = set(GLOBAL_STATE_OUTCOME_COLUMNS).union(RESIDUAL_STATE_OUTCOME_COLUMNS).union(
        {"meta_target_soft", "__first_touch_target_soft__", "__target_soft__"}
    )
    source_columns = [name for name in source_columns if name not in forbidden]
    numeric = [name for name in source_columns if name not in {"__ts__", "side_name"}]
    panel = (
        combined.loc[:, ["__ts__", "side_name", *numeric]]
        .groupby(["__ts__", "side_name"], observed=True, as_index=False)
        .median(numeric_only=True)
        .sort_values(["side_name", "__ts__"], kind="stable")
        .reset_index(drop=True)
    )
    phase_panel, phase_manifest = add_causal_phase_state_features(panel)
    phase = phase_panel.loc[:, ["__ts__", "side_name", *PHASE_STATE_FEATURES]].copy()
    mapped = combined.loc[:, ["__ts__", "side_name", "__phase_split__", "__phase_row__"]].merge(
        phase,
        on=["__ts__", "side_name"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    mapped = mapped.sort_values(["__phase_split__", "__phase_row__"], kind="stable")
    output = pd.DataFrame(index=np.arange(len(mapped)))
    for name in PHASE_STATE_FEATURES:
        output[f"meta_aux_{name}"] = pd.to_numeric(mapped[name], errors="coerce").fillna(0.0).astype(np.float32)
    train_out = output.loc[mapped["__phase_split__"].to_numpy() == 0].reset_index(drop=True)
    test_out = output.loc[mapped["__phase_split__"].to_numpy() == 1].reset_index(drop=True)
    detail: dict[str, object] = {
        "status": "complete",
        "source_columns": numeric,
        "panel_rows": int(len(panel)),
        "nonzero_features": int((output.abs().sum(axis=0) > 1e-8).sum()),
        "manifest": phase_manifest,
    }
    return train_out, test_out, detail


def _contrastive_executable_failure_training_set(
    local: pd.DataFrame,
    negative_residual: np.ndarray,
    *,
    score_bands: int = 4,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build a score-band-matched executable-failure contrast population.

    The residual target alone remains somewhat correlated with parent rank.
    This helper keeps only unexpected high-tail losses and clean profitable
    controls, then balances the two classes within each side x archetype x
    score band.  It therefore asks the auxiliary model for information beyond
    ``score`` rather than for a second estimate of the same ranking signal.
    Outcomes define labels only; score bands are observable train covariates.
    """

    if len(local) != len(negative_residual):
        raise ValueError("negative_residual length must match local rows")
    rank = pd.to_numeric(
        local.get("base_rank_pct_by_timestamp", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    score = pd.to_numeric(
        local.get("score", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0)
    ev = pd.to_numeric(
        local.get("ev_after_1pct", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    clean = pd.to_numeric(
        local.get("clean_exec", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    bad_mae = pd.to_numeric(
        local.get("full_path_bad_mae_1r", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    timeout = pd.to_numeric(
        local.get("timeout", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    dirty = pd.to_numeric(
        local.get("dirty_positive", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    top_tail = rank >= 0.90
    top20 = rank >= 0.80
    unexpected_loss = (
        top_tail
        & (np.asarray(negative_residual, dtype=np.float32) > 0.5)
        & (ev <= 0.0)
    )
    clean_control = (
        top_tail
        & (clean > 0.5)
        & (ev > 0.0)
        & (bad_mae <= 0.5)
        & (timeout <= 0.5)
        & (dirty <= 0.5)
    )
    keep = unexpected_loss | clean_control
    if not keep.any():
        return local.iloc[0:0].copy(), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    work = local.loc[keep].reset_index(drop=True).copy()
    target = unexpected_loss[keep].astype(np.float32)
    work["__contrast_target__"] = target
    work["__contrast_score__"] = score.loc[keep].to_numpy(dtype=np.float32)
    # Bands are formed within local streams from observable parent scores.  A
    # stable percentile rank avoids comparing unlike long/short score scales.
    work["__contrast_band__"] = 0
    for _, idx in work.groupby(["side_name", "archetype_policy_key"], observed=True, sort=False).groups.items():
        positions = np.asarray(idx, dtype=np.int64)
        values = work.iloc[positions]["__contrast_score__"].rank(method="average", pct=True).to_numpy(dtype=np.float32)
        work.loc[work.index[positions], "__contrast_band__"] = np.minimum(
            int(score_bands) - 1,
            np.floor(values * int(score_bands)).astype(np.int8),
        )
    weight = np.ones(len(work), dtype=np.float32)
    for _, idx in work.groupby(
        ["side_name", "archetype_policy_key", "__contrast_band__"],
        observed=True,
        sort=False,
    ).groups.items():
        positions = np.asarray(idx, dtype=np.int64)
        values = target[positions] > 0.5
        positive = int(values.sum())
        negative = int((~values).sum())
        # A single-class band cannot teach the contrast.  Keep it at neutral
        # weight; other bands in the same local stream remain informative.
        if positive == 0 or negative == 0:
            continue
        support = float(len(positions))
        weight[positions[values]] = support / (2.0 * float(positive))
        weight[positions[~values]] = support / (2.0 * float(negative))
    weight = np.clip(weight, 0.25, 6.0)
    weight /= max(float(np.mean(weight)), 1e-6)
    return work.drop(columns=["__contrast_target__", "__contrast_score__", "__contrast_band__"]), target, weight.astype(np.float32)


def _causal_residual_shortfall_target(
    frame: pd.DataFrame,
    *,
    clip: float = 0.05,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a causal, non-negative score-conditioned EV shortfall target.

    Each chronological block is evaluated against a score-to-EV expectation
    fitted only on prior blocks.  The target has no OOS counterpart; only its
    prediction becomes a meta input.
    """

    ordered = frame.sort_values(["__ts__", "__symbol__", "side_name"], kind="stable").reset_index(drop=True)
    blocks = [part for part in np.array_split(np.arange(len(ordered), dtype=np.int64), 4) if len(part)]
    history: list[pd.DataFrame] = []
    labelled: list[pd.DataFrame] = []
    values: list[np.ndarray] = []
    for block_index, positions in enumerate(blocks):
        block = ordered.iloc[positions].reset_index(drop=True)
        if block_index == 0:
            history.append(block)
            continue
        reference = pd.concat(history, ignore_index=True, copy=False)
        state = _fit_score_value_map(reference, value_col="ev_after_1pct")
        actual = pd.to_numeric(block["ev_after_1pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        residual = actual - _expected_ev(block, state)
        labelled.append(block)
        values.append(np.clip(-residual, 0.0, float(clip)).astype(np.float32))
        history.append(block)
    if not labelled:
        return ordered.iloc[0:0].copy(), np.empty(0, dtype=np.float32)
    return pd.concat(labelled, ignore_index=True, copy=False), np.concatenate(values).astype(np.float32)


def _fit_predict_regressor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str] | dict[str, list[str]],
    target: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit shallow side/archetype-local robust regressors for an auxiliary input."""

    import lightgbm as lgb

    if isinstance(features, dict):
        features_by_side = {
            str(side).lower(): list(dict.fromkeys(str(feature) for feature in values))
            for side, values in features.items()
        }
        all_features = list(dict.fromkeys(feature for values in features_by_side.values() for feature in values))
    else:
        all_features = list(dict.fromkeys(str(feature) for feature in features))
        features_by_side = {}
    median = train.reindex(columns=all_features).median(numeric_only=True).reindex(all_features).fillna(0.0)
    global_prior = float(np.nanmean(target)) if len(target) else 0.0
    result = np.full(len(test), global_prior, dtype=np.float32)

    def _matrix(part: pd.DataFrame, selected: list[str]) -> np.ndarray:
        return (
            part.reindex(columns=selected)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(median.reindex(selected).fillna(0.0))
            .to_numpy(dtype=np.float32)
        )

    train_groups = train.groupby(["side_name", "archetype_policy_key"], observed=True, sort=False).groups
    side_groups = train.groupby("side_name", observed=True, sort=False).groups
    for (side, archetype), indices in test.groupby(["side_name", "archetype_policy_key"], observed=True, sort=False).groups.items():
        selected = features_by_side.get(str(side).lower(), all_features)
        local_indices = train_groups.get((side, archetype))
        if local_indices is None or len(local_indices) < 800:
            local_indices = side_groups.get(side)
        if local_indices is None or len(local_indices) < 1_500:
            continue
        local_indices = np.asarray(local_indices, dtype=np.int64)
        dataset = lgb.Dataset(
            _matrix(train.iloc[local_indices], selected),
            label=np.asarray(target, dtype=np.float32)[local_indices],
            feature_name=selected,
            free_raw_data=True,
        )
        model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "l1",
                "learning_rate": 0.035,
                "num_leaves": 15,
                "max_depth": 3,
                "min_data_in_leaf": 120,
                "lambda_l1": 1.0,
                "lambda_l2": 5.0,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "seed": int(seed),
                "num_threads": 2,
                "verbosity": -1,
            },
            dataset,
            num_boost_round=140,
        )
        result[np.asarray(indices, dtype=np.int64)] = model.predict(
            _matrix(test.iloc[indices], selected), num_iteration=model.best_iteration
        ).astype(np.float32)
        model.free_dataset()
        del model, dataset
    return np.nan_to_num(result, nan=global_prior, posinf=global_prior, neginf=0.0)


def _fit_residual_shortfall(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] | dict[str, list[str]],
    seed: int,
) -> np.ndarray:
    local, target = _causal_residual_shortfall_target(train)
    if len(local) < 1_500 or not np.isfinite(target).any():
        return np.full(len(test), float(np.nanmean(target)) if len(target) else 0.0, dtype=np.float32)
    return _fit_predict_regressor(local, test, features=features, target=target, seed=seed)


def _direct_residual_mechanism_target(
    local: pd.DataFrame,
    negative_residual: np.ndarray,
    mechanism: str,
) -> np.ndarray:
    """Return a labelled train-only mechanism target.

    ``negative_residual`` is itself causal within the supplied training frame:
    each chronological block uses a score-to-EV expectation fitted on earlier
    blocks.  The target is consequently suitable for an auxiliary classifier,
    but never accepted as a feature at inference.
    """

    def _numeric(name: str) -> np.ndarray:
        values = local[name] if name in local else pd.Series(0.0, index=local.index)
        return pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    rank = _numeric("base_rank_pct_by_timestamp")
    ev = _numeric("ev_after_1pct")
    clean = _numeric("clean_exec")
    timeout = _numeric("timeout")
    first_touch_bad_mae = _numeric("first_touch_bad_mae_1r")
    adverse = np.maximum(_numeric("full_path_bad_mae_1r"), timeout)
    dirty = _numeric("dirty_positive")
    top_tail = rank >= 0.90
    top20 = rank >= 0.80
    near_tail = (rank >= 0.80) & ~top_tail
    negative = np.asarray(negative_residual, dtype=np.float32) > 0.5
    clean_path = (clean > 0.5) & (adverse <= 0.5)
    negative_ev = ev <= 0.0
    reversal_after_initial_success = (
        top_tail
        & negative
        & (clean > 0.5)
        & negative_ev
        & ((adverse > 0.5) | (dirty > 0.5))
    )
    side = local.get("side_name", pd.Series("", index=local.index)).astype(str).str.lower().to_numpy()
    archetype = local.get(
        "archetype_policy_key", pd.Series("", index=local.index)
    ).astype(str).str.lower().to_numpy()

    def _episode(
        base: np.ndarray,
        group_columns: list[str],
        *,
        min_support: int,
        min_rate: float,
        hours: int = 6,
    ) -> np.ndarray:
        bucket = pd.to_datetime(local["__ts__"], utc=True, errors="coerce").dt.floor(
            f"{int(hours)}h"
        )
        work = pd.DataFrame(
            {
                "bucket": bucket,
                "base": base.astype(np.float32),
                **{
                    name: local[name].astype(str).to_numpy()
                    for name in group_columns
                },
            }
        )
        groups = work.groupby(["bucket", *group_columns], observed=True)["base"]
        support = groups.transform("size").to_numpy(dtype=np.int32)
        rate = groups.transform("mean").to_numpy(dtype=np.float32)
        return base & (support >= int(min_support)) & (rate >= float(min_rate))

    def _episode_onset(
        base: np.ndarray,
        group_columns: list[str],
        *,
        min_support: int,
        min_rate: float,
        hours: int = 6,
    ) -> np.ndarray:
        """Return the first bucket of each resolved six-hour failure episode.

        This constructs a train-only outcome label. The auxiliary classifier
        still consumes only pre-entry features when it produces OOF/OOS
        probabilities, so episode history never becomes an inference input.
        """

        bucket = pd.to_datetime(
            local["__ts__"], utc=True, errors="coerce"
        ).dt.floor(f"{int(hours)}h")
        work = pd.DataFrame(
            {
                "bucket": bucket,
                "base": base.astype(np.float32),
                **{
                    name: local[name].astype(str).to_numpy()
                    for name in group_columns
                },
            }
        )
        keys = [*group_columns, "bucket"]
        grouped = (
            work.groupby(keys, observed=True, sort=True)["base"]
            .agg(support="size", rate="mean")
            .reset_index()
        )
        grouped["active"] = (
            grouped["support"].ge(int(min_support))
            & grouped["rate"].ge(float(min_rate))
        )
        grouped = grouped.sort_values([*group_columns, "bucket"], kind="stable")
        previous_active = grouped.groupby(
            group_columns, observed=True, sort=False
        )["active"].shift(1, fill_value=False)
        previous_bucket = grouped.groupby(
            group_columns, observed=True, sort=False
        )["bucket"].shift(1)
        contiguous = grouped["bucket"].sub(previous_bucket).eq(
            pd.Timedelta(hours=int(hours))
        )
        grouped["onset"] = grouped["active"] & ~(previous_active & contiguous)
        onset = work.merge(
            grouped.loc[:, [*keys, "onset"]],
            on=keys,
            how="left",
            validate="many_to_one",
            sort=False,
        )["onset"]
        return base & onset.fillna(False).to_numpy(dtype=bool)

    def _episode_state_onset(
        base: np.ndarray,
        group_columns: list[str],
        *,
        min_support: int,
        min_rate: float,
        hours: int = 6,
    ) -> np.ndarray:
        """Label every candidate at a train-defined local-state onset.

        The active bucket is defined from resolved outcomes, but once active
        the target applies to all top-20 candidates in that side/archetype
        bucket.  The resulting OOF/OOS probability therefore models an
        observable transition state rather than a single losing row.
        """

        bucket = pd.to_datetime(
            local["__ts__"], utc=True, errors="coerce"
        ).dt.floor(f"{int(hours)}h")
        work = pd.DataFrame(
            {
                "bucket": bucket,
                "base": base.astype(np.float32),
                "candidate": top20.astype(bool),
                "__pos__": np.arange(len(local), dtype=np.int64),
                **{
                    name: local[name].astype(str).to_numpy()
                    for name in group_columns
                },
            }
        )
        keys = [*group_columns, "bucket"]
        # Both activation and output are conditional on the fixed base
        # candidate stream.  Including rows below the handoff cutoff here
        # would dilute the train-only failure rate and make a genuine
        # candidate-state onset impossible to activate.
        candidate_work = work.loc[work["candidate"]].copy()
        grouped = (
            candidate_work.groupby(keys, observed=True, sort=True)["base"]
            .agg(support="size", rate="mean")
            .reset_index()
        )
        grouped["active"] = (
            grouped["support"].ge(int(min_support))
            & grouped["rate"].ge(float(min_rate))
        )
        grouped = grouped.sort_values([*group_columns, "bucket"], kind="stable")
        previous_active = grouped.groupby(
            group_columns, observed=True, sort=False
        )["active"].shift(1, fill_value=False)
        previous_bucket = grouped.groupby(
            group_columns, observed=True, sort=False
        )["bucket"].shift(1)
        contiguous = grouped["bucket"].sub(previous_bucket).eq(
            pd.Timedelta(hours=int(hours))
        )
        grouped["onset"] = grouped["active"] & ~(previous_active & contiguous)
        onset = candidate_work.merge(
            grouped.loc[:, [*keys, "onset"]],
            on=keys,
            how="left",
            validate="many_to_one",
            sort=False,
        )["onset"].fillna(False).to_numpy(dtype=bool)
        target = np.zeros(len(local), dtype=bool)
        target[candidate_work["__pos__"].to_numpy(dtype=np.int64)] = onset
        return target

    def _cross_archetype_state_onset(
        base: np.ndarray,
        *,
        min_archetypes: int = 2,
        min_support: int = 3,
        min_rate: float = 0.45,
        hours: int = 6,
    ) -> np.ndarray:
        """Label a new side-wide episode shared by independent archetypes."""

        bucket = pd.to_datetime(
            local["__ts__"], utc=True, errors="coerce"
        ).dt.floor(f"{int(hours)}h")
        work = pd.DataFrame(
            {
                "bucket": bucket,
                "side_name": local["side_name"].astype(str).str.lower(),
                "archetype": local.get(
                    "archetype_policy_key", pd.Series("missing", index=local.index)
                ).astype(str),
                "base": base.astype(np.float32),
                "candidate": top20.astype(bool),
                "__pos__": np.arange(len(local), dtype=np.int64),
            }
        )
        candidate_work = work.loc[work["candidate"]].copy()
        cells = (
            candidate_work.groupby(["bucket", "side_name", "archetype"], observed=True)["base"]
            .agg(support="size", rate="mean")
            .reset_index()
        )
        cells["active"] = (
            cells["support"].ge(int(min_support))
            & cells["rate"].ge(float(min_rate))
        )
        active_counts = (
            cells.loc[cells["active"]]
            .groupby(["bucket", "side_name"], observed=True)["archetype"]
            .nunique()
            .rename("active_archetypes")
            .reset_index()
        )
        states = (
            candidate_work.loc[:, ["bucket", "side_name"]]
            .drop_duplicates()
            .merge(active_counts, on=["bucket", "side_name"], how="left", sort=False)
        )
        states["active"] = states["active_archetypes"].fillna(0).ge(
            int(min_archetypes)
        )
        states = states.sort_values(["side_name", "bucket"], kind="stable")
        previous_active = states.groupby("side_name", observed=True, sort=False)[
            "active"
        ].shift(1, fill_value=False)
        previous_bucket = states.groupby("side_name", observed=True, sort=False)[
            "bucket"
        ].shift(1)
        contiguous = states["bucket"].sub(previous_bucket).eq(
            pd.Timedelta(hours=int(hours))
        )
        states["onset"] = states["active"] & ~(previous_active & contiguous)
        onset = candidate_work.merge(
            states.loc[:, ["bucket", "side_name", "onset"]],
            on=["bucket", "side_name"],
            how="left",
            validate="many_to_one",
            sort=False,
        )["onset"].fillna(False).to_numpy(dtype=bool)
        target = np.zeros(len(local), dtype=bool)
        target[candidate_work["__pos__"].to_numpy(dtype=np.int64)] = onset
        return target

    def _cross_archetype_contagion(
        base: np.ndarray,
        *,
        min_archetypes: int = 2,
        min_support: int = 3,
        min_rate: float = 0.45,
    ) -> np.ndarray:
        """Mark outcome-defined market states shared across archetypes.

        A one-archetype failure can be a local miss.  This label requires at
        least ``min_archetypes`` independently supported side-archetype cells
        to fail at the same timestamp, producing a materially different
        train-only target for the observable market-state classifier.
        """

        work = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(local["__ts__"], utc=True, errors="coerce"),
                "side_name": local["side_name"].astype(str).str.lower(),
                "archetype": local.get(
                    "archetype_policy_key", pd.Series("missing", index=local.index)
                ).astype(str),
                "base": base.astype(np.float32),
            }
        )
        cells = (
            work.groupby(["timestamp", "side_name", "archetype"], observed=True)["base"]
            .agg(support="size", rate="mean")
            .reset_index()
        )
        cells["active"] = cells["support"].ge(int(min_support)) & cells["rate"].ge(float(min_rate))
        active_count = (
            cells.loc[cells["active"]]
            .groupby(["timestamp", "side_name"], observed=True)["archetype"]
            .nunique()
            .rename("active_archetypes")
            .reset_index()
        )
        mapped = work.merge(
            active_count,
            on=["timestamp", "side_name"],
            how="left",
            validate="many_to_one",
            sort=False,
        )["active_archetypes"].fillna(0.0).to_numpy(dtype=np.float32)
        return base & (mapped >= float(min_archetypes))

    if mechanism == "top_tail_residual_negative_ev":
        target = top_tail & negative & negative_ev
    elif mechanism == "top_tail_residual_false_positive":
        target = top_tail & negative & (clean <= 0.5) & negative_ev
    elif mechanism == "top_tail_residual_clean_cost_fragile":
        target = top_tail & negative & clean_path & negative_ev
    elif mechanism == "top_tail_residual_adverse_loss":
        target = top_tail & negative & (adverse > 0.5) & negative_ev
    elif mechanism == "top_tail_residual_timeout_loss":
        target = top_tail & negative & (timeout > 0.5) & negative_ev
    elif mechanism == "top_tail_adverse_path_survivor":
        target = top_tail & (adverse > 0.5) & (ev > 0.0)
    elif mechanism == "top_tail_dirty_positive_survivor":
        target = top_tail & (dirty > 0.5) & (ev > 0.0)
    elif mechanism == "top_tail_timeout_positive_survivor":
        target = top_tail & (timeout > 0.5) & (ev > 0.0)
    elif mechanism == "top_tail_residual_systemic_loss":
        base = top_tail & negative & negative_ev
        work = pd.DataFrame(
            {
                "__ts__": pd.to_datetime(local["__ts__"], utc=True, errors="coerce"),
                "side_name": local["side_name"].astype(str).str.lower(),
                "base": base.astype(np.float32),
            }
        )
        grouped = work.groupby(["__ts__", "side_name"], observed=True)["base"]
        support = grouped.transform("size").to_numpy(dtype=np.int32)
        rate = grouped.transform("mean").to_numpy(dtype=np.float32)
        target = base & (support >= 4) & (rate >= 0.50)
    elif mechanism == "top_tail_residual_local_loss_episode_6h":
        target = _episode(
            top_tail & negative & negative_ev,
            ["side_name", "archetype_policy_key"],
            min_support=5,
            min_rate=0.40,
        )
    elif mechanism == "top_tail_residual_local_adverse_episode_6h":
        target = _episode(
            top_tail & negative & (adverse > 0.5) & negative_ev,
            ["side_name", "archetype_policy_key"],
            min_support=5,
            min_rate=0.30,
        )
    elif mechanism == "top_tail_residual_market_loss_episode_6h":
        target = _episode(
            top_tail & negative & negative_ev,
            ["side_name"],
            min_support=12,
            min_rate=0.35,
        )
    elif mechanism == "top_tail_residual_local_loss_episode_onset_6h":
        target = _episode_onset(
            top_tail & negative & negative_ev,
            ["side_name", "archetype_policy_key"],
            min_support=5,
            min_rate=0.40,
        )
    elif mechanism == "top_tail_residual_local_adverse_episode_onset_6h":
        target = _episode_onset(
            top_tail & negative & (adverse > 0.5) & negative_ev,
            ["side_name", "archetype_policy_key"],
            min_support=5,
            min_rate=0.30,
        )
    elif mechanism == "top_tail_residual_market_loss_episode_onset_6h":
        target = _episode_onset(
            top_tail & negative & negative_ev,
            ["side_name"],
            min_support=12,
            min_rate=0.35,
        )
    elif mechanism == "top_tail_residual_idiosyncratic_loss":
        base = top_tail & negative & negative_ev
        work = pd.DataFrame(
            {
                "__ts__": pd.to_datetime(local["__ts__"], utc=True, errors="coerce"),
                "side_name": local["side_name"].astype(str).str.lower(),
                "base": base.astype(np.float32),
            }
        )
        grouped = work.groupby(["__ts__", "side_name"], observed=True)["base"]
        support = grouped.transform("size").to_numpy(dtype=np.int32)
        rate = grouped.transform("mean").to_numpy(dtype=np.float32)
        target = base & (support >= 6) & (rate <= 0.25)
    elif mechanism == "candidate_top20_residual_negative_ev":
        target = top20 & negative & negative_ev
    elif mechanism == "candidate_top20_residual_false_positive":
        target = top20 & negative & (clean <= 0.5) & negative_ev
    elif mechanism == "candidate_top20_residual_clean_cost_fragile":
        target = top20 & negative & clean_path & negative_ev
    elif mechanism == "candidate_top20_residual_adverse_loss":
        target = top20 & negative & (adverse > 0.5) & negative_ev
    elif mechanism == "candidate_top20_residual_first_touch_stop_loss":
        # This is intentionally first-touch, rather than full-path MAE: it
        # captures rows that would have failed the executable stop before a
        # later recovery can make them look harmless in terminal utility.
        target = top20 & negative & (first_touch_bad_mae > 0.5) & negative_ev
    elif mechanism == "candidate_top20_residual_timeout_loss":
        target = top20 & negative & (timeout > 0.5) & negative_ev
    elif mechanism == "candidate_top20_residual_systemic_loss":
        base = top20 & negative & negative_ev
        work = pd.DataFrame(
            {
                "__ts__": pd.to_datetime(local["__ts__"], utc=True, errors="coerce"),
                "side_name": local["side_name"].astype(str).str.lower(),
                "base": base.astype(np.float32),
            }
        )
        grouped = work.groupby(["__ts__", "side_name"], observed=True)["base"]
        support = grouped.transform("size").to_numpy(dtype=np.int32)
        rate = grouped.transform("mean").to_numpy(dtype=np.float32)
        target = base & (support >= 4) & (rate >= 0.50)
    elif mechanism == "candidate_top20_residual_local_adverse_episode_onset_6h":
        target = _episode_onset(
            top20 & negative & (adverse > 0.5) & negative_ev,
            ["side_name", "archetype_policy_key"],
            min_support=5,
            min_rate=0.30,
        )
    elif mechanism == "candidate_top20_local_loss_state_onset_6h":
        target = _episode_state_onset(
            top20 & negative,
            ["side_name", "archetype_policy_key"],
            min_support=5,
            min_rate=0.35,
            hours=6,
        )
    elif mechanism == "candidate_top20_side_loss_state_onset_6h":
        target = _episode_state_onset(
            top20 & negative,
            ["side_name"],
            min_support=12,
            min_rate=0.30,
            hours=6,
        )
    elif mechanism == "candidate_top20_cross_archetype_loss_state_onset_6h":
        target = _cross_archetype_state_onset(
            top20 & negative,
            min_archetypes=2,
            min_support=2,
            min_rate=0.30,
            hours=6,
        )
    elif mechanism == "candidate_top20_cross_archetype_stop_contagion":
        # The candidate stream frequently has fewer than three rows inside a
        # timestamp x archetype cell.  Require two independent archetypes to
        # show an executable stop instead of the earlier impossible
        # within-cell support threshold.  The state remains train-outcome
        # defined; its classifier sees only causal pre-entry inputs.
        target = _cross_archetype_contagion(
            top20 & (first_touch_bad_mae > 0.5) & negative_ev,
            min_archetypes=2,
            min_support=1,
            min_rate=0.50,
        )
    elif mechanism == "candidate_top20_residual_local_stop_episode_onset_12h":
        # Twelve hours provides enough local candidate support to identify an
        # onset without turning a single stopped row into a regime label.
        target = _episode_onset(
            top20 & (first_touch_bad_mae > 0.5) & negative_ev,
            ["side_name", "archetype_policy_key"],
            min_support=8,
            min_rate=0.15,
            hours=12,
        )
    elif mechanism == "candidate_top20_market_stop_pressure":
        base = top20 & (first_touch_bad_mae > 0.5) & negative_ev
        work = pd.DataFrame(
            {
                "__ts__": pd.to_datetime(local["__ts__"], utc=True, errors="coerce"),
                "side_name": local["side_name"].astype(str).str.lower(),
                "base": base.astype(np.float32),
            }
        )
        grouped = work.groupby(["__ts__", "side_name"], observed=True)["base"]
        support = grouped.transform("size").to_numpy(dtype=np.int32)
        rate = grouped.transform("mean").to_numpy(dtype=np.float32)
        # A 25% stopped share is a material market-pressure event while still
        # reachable in the top-20 candidate population.
        target = base & (support >= 4) & (rate >= 0.25)
    elif mechanism == "candidate_top20_reversal_after_initial_success":
        target = (
            top20
            & negative
            & (clean > 0.5)
            & negative_ev
            & ((adverse > 0.5) | (dirty > 0.5))
        )
    elif mechanism == "candidate_top20_adverse_path_survivor":
        target = top20 & (adverse > 0.5) & (ev > 0.0)
    elif mechanism == "candidate_top20_cross_archetype_loss_contagion":
        target = _cross_archetype_contagion(top20 & negative & negative_ev)
    elif mechanism == "candidate_top20_cross_archetype_adverse_contagion":
        target = _cross_archetype_contagion(top20 & negative & (adverse > 0.5))
    elif mechanism == "long_mixed_latent_misfire":
        target = (
            (side == "long")
            & (np.char.find(archetype.astype(str), "long_mixed") >= 0)
            & top_tail
            & negative
            & negative_ev
        )
    elif mechanism == "short_mixed_off_manifold":
        target = (
            (side == "short")
            & (np.char.find(archetype.astype(str), "short_mixed") >= 0)
            & top_tail
            & negative
            & negative_ev
            & ((clean <= 0.5) | (adverse > 0.5))
        )
    elif mechanism == "short_default_latent_uncertainty":
        target = (
            (side == "short")
            & (np.char.find(archetype.astype(str), "short_default") >= 0)
            & top_tail
            & negative
            & negative_ev
            & ((adverse > 0.5) | (timeout > 0.5))
        )
    elif mechanism == "top_tail_reversal_after_initial_success":
        target = reversal_after_initial_success
    elif mechanism == "long_mixed_reversal_after_initial_success":
        target = (
            reversal_after_initial_success
            & (side == "long")
            & (np.char.find(archetype.astype(str), "long_mixed") >= 0)
        )
    elif mechanism == "short_mixed_reversal_after_initial_success":
        target = (
            reversal_after_initial_success
            & (side == "short")
            & (np.char.find(archetype.astype(str), "short_mixed") >= 0)
        )
    elif mechanism == "long_breakout_overconfident_path_loss":
        target = (
            (side == "long")
            & (np.char.find(archetype.astype(str), "breakout") >= 0)
            & top_tail
            & negative
            & negative_ev
            & ((clean <= 0.5) | (adverse > 0.5))
        )
    elif mechanism == "short_breakout_overconfident_path_loss":
        target = (
            (side == "short")
            & (np.char.find(archetype.astype(str), "breakout") >= 0)
            & top_tail
            & negative
            & negative_ev
            & ((clean <= 0.5) | (adverse > 0.5))
        )
    elif mechanism == "top_tail_contrastive_executable_failure":
        # The fitter replaces this broad indicator with score-band-matched
        # loss/control rows and class-balanced train weights.
        target = top_tail & negative & negative_ev
    elif mechanism == "near_tail_clean_executable":
        target = near_tail & clean_path & (ev > 0.0)
    else:
        raise ValueError(f"Unknown direct residual mechanism: {mechanism}")
    return target.astype(np.float32)


def _direct_mechanism_support_rows(
    train: pd.DataFrame,
    *,
    fold_month: str,
    mechanism: str,
) -> list[dict[str, object]]:
    """Summarise a direct-state target before spending compute on a fit.

    Labels remain entirely train-derived: the residual expectation is causal
    within ``train`` and the output never touches the held-out month's
    outcomes.  Candidate mechanisms are measured on their actual top-20
    fitting population, while top-tail mechanisms retain their broader
    fitting population.  This report is deliberately descriptive only; it
    does not decide eligibility or modify a model contract.
    """

    local, negative_residual, _ = _mechanism_residual_target(train, mechanism)
    if local.empty:
        return []
    target = _direct_residual_mechanism_target(local, negative_residual, mechanism)
    if mechanism.startswith("candidate_top20_"):
        rank = pd.to_numeric(
            local.get("base_rank_pct_by_timestamp", pd.Series(0.0, index=local.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float32)
        keep = rank >= 0.80
        local = local.loc[keep].reset_index(drop=True)
        target = np.asarray(target, dtype=np.float32)[keep]
    if local.empty:
        return []
    work = local.loc[:, ["side_name", "archetype_policy_key"]].copy()
    work["target"] = np.asarray(target, dtype=np.float32)
    rows: list[dict[str, object]] = []
    group_specs = (
        ("global", []),
        ("side", ["side_name"]),
        ("side_archetype", ["side_name", "archetype_policy_key"]),
    )
    for level, keys in group_specs:
        grouped = [((), work)] if not keys else work.groupby(keys, observed=True, sort=True)
        for key, part in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            payload = dict(zip(keys, key))
            positives = int(part["target"].gt(0.5).sum())
            rows.append(
                {
                    "fold_month": str(fold_month),
                    "mechanism": str(mechanism),
                    "level": level,
                    "rows": int(len(part)),
                    "positive_rows": positives,
                    "positive_rate": float(positives / len(part)) if len(part) else 0.0,
                    "trainable": bool(positives >= 32 and (len(part) - positives) >= 128),
                    "side_name": str(payload.get("side_name", "__all__")),
                    "archetype_policy_key": str(payload.get("archetype_policy_key", "__all__")),
                }
            )
    return rows


def _fit_state(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str] | dict[str, list[str]], seed: int, side: str, archetype: str
) -> np.ndarray:
    local, target, _ = _causal_residual_target(train, value_col="clean_exec", label_col="__negative_hit_residual_event__")
    mask = local["side_name"].astype(str).str.lower().eq(side) & local["archetype_policy_key"].astype(str).eq(archetype)
    if int(mask.sum()) < 1_500:
        return np.zeros(len(test), dtype=np.float32)
    pred = _fit_predict(local.loc[mask].reset_index(drop=True), test, features=features, target=target[mask.to_numpy()], seed=seed)[0]
    test_mask = test["side_name"].astype(str).str.lower().eq(side) & test["archetype_policy_key"].astype(str).eq(archetype)
    return np.where(test_mask.to_numpy(), pred, 0.0).astype(np.float32)


def _fit_local_transition_negative_hit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Predict local negative clean-hit residuals from causal transitions.

    The target is fit only on prior realized rows.  Inputs are intentionally
    narrow: protected base-confidence anchors plus causal phase coordinates.
    This lets the local side x archetype model learn whether, for example, a
    late-continuation state is hazardous for that stream without relitigating
    the full meta feature universe.
    """

    local, target, _ = _causal_residual_target(
        train,
        value_col="clean_exec",
        label_col="__negative_hit_residual_event__",
    )
    features = {
        side: [
            column
            for column in (*TRANSITION_STATE_ANCHORS, *PHASE_AUX_FEATURES)
            if column in local.columns or column in test.columns
        ]
        for side in ("long", "short")
    }
    if not any(features.values()) or int(np.sum(target > 0.5)) < 32:
        return np.full(len(test), float(np.mean(target)) if len(target) else 0.5, dtype=np.float32)
    return _fit_predict(
        local.reset_index(drop=True),
        test.reset_index(drop=True),
        features=features,
        target=target,
        seed=seed,
    )[0]


def _fit_local_aegmm_transition_negative_hit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Predict local negative clean-hit residuals from frozen-state changes.

    This is deliberately narrower than the main meta universe.  It asks one
    question: conditional on the protected base-confidence anchors, has the
    observable AE/GMM state *just changed* in a way that historically makes a
    side x archetype stream miss its expected clean-hit rate?  All transition
    inputs are materialized causally from complete symbol panels before this
    function is called.
    """

    local, target, _ = _causal_residual_target(
        train,
        value_col="clean_exec",
        label_col="__negative_hit_residual_event__",
    )
    features = {
        side: [
            column
            for column in (*TRANSITION_STATE_ANCHORS, *AEGMM_TRANSITION_SOURCE_FEATURES)
            if column in local.columns or column in test.columns
        ]
        for side in ("long", "short")
    }
    if not any(features.values()) or int(np.sum(target > 0.5)) < 32:
        return np.full(
            len(test), float(np.mean(target)) if len(target) else 0.5, dtype=np.float32
        )
    return _fit_predict(
        local.reset_index(drop=True),
        test.reset_index(drop=True),
        features=features,
        target=target,
        seed=seed,
    )[0]


def _fit_local_aegmm_component_transition_negative_hit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Predict local negative clean-hit residuals from directed state changes.

    Unlike M26, entering a component and exiting it remain separate continuous
    coordinates.  This prevents, for instance, a transition into a rebound
    state from being conflated with a transition out of one.
    """

    local, target, _ = _causal_residual_target(
        train,
        value_col="clean_exec",
        label_col="__negative_hit_residual_event__",
    )
    features = {
        side: [
            column
            for column in (*TRANSITION_STATE_ANCHORS, *AEGMM_COMPONENT_TRANSITION_SOURCE_FEATURES)
            if column in local.columns or column in test.columns
        ]
        for side in ("long", "short")
    }
    if not any(features.values()) or int(np.sum(target > 0.5)) < 32:
        return np.full(
            len(test), float(np.mean(target)) if len(target) else 0.5, dtype=np.float32
        )
    return _fit_predict(
        local.reset_index(drop=True),
        test.reset_index(drop=True),
        features=features,
        target=target,
        seed=seed,
    )[0]


def _fit_local_aegmm_durable_transition_negative_hit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Predict adverse clean-hit residuals from durable causal transitions."""

    local, target, _ = _causal_residual_target(
        train,
        value_col="clean_exec",
        label_col="__negative_hit_residual_event__",
    )
    features = {
        side: [
            column
            for column in (
                *TRANSITION_STATE_ANCHORS,
                *AEGMM_DOMINANT_STATE_TRANSITION_SOURCE_FEATURES,
            )
            if column in local.columns or column in test.columns
        ]
        for side in ("long", "short")
    }
    if not any(features.values()) or int(np.sum(target > 0.5)) < 32:
        return np.full(len(test), float(np.mean(target)) if len(target) else 0.5, dtype=np.float32)
    return _fit_predict(
        local.reset_index(drop=True), test.reset_index(drop=True), features=features, target=target, seed=seed
    )[0]


def _oof_auxiliary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fit: Callable[[pd.DataFrame, pd.DataFrame, int], np.ndarray],
    *,
    neutral: float,
    seed: int,
    blocks: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce causal training OOF features and a frozen OOS feature."""

    result = np.full(len(train), neutral, dtype=np.float32)
    ordered_ts = np.sort(train["__ts__"].dropna().unique())
    partitions = np.array_split(ordered_ts, blocks)
    for block_idx, values in enumerate(partitions):
        if len(values) == 0:
            continue
        valid_mask = train["__ts__"].isin(values).to_numpy()
        history = train.loc[train["__ts__"] < values[0]].reset_index(drop=True)
        if len(history) < 1_500:
            continue
        result[np.flatnonzero(valid_mask)] = fit(history, train.loc[valid_mask].reset_index(drop=True), seed + block_idx)
    return result, fit(train, test, seed + 10_000)


def _oof_main_predictions(
    train: pd.DataFrame,
    *,
    features: dict[str, list[str]],
    target: np.ndarray,
    seed: int,
    blocks: int,
) -> np.ndarray:
    """Chronological OOF meta predictions used only for contract selection."""

    result = np.full(len(train), np.nan, dtype=np.float32)
    ordered_ts = np.sort(train["__ts__"].dropna().unique())
    for block_index, values in enumerate(np.array_split(ordered_ts, blocks)):
        if len(values) == 0:
            continue
        valid_mask = train["__ts__"].isin(values).to_numpy()
        history = train.loc[train["__ts__"] < values[0]].reset_index(drop=True)
        if len(history) < 1_500:
            continue
        prediction, _ = _fit_predict(
            history,
            train.loc[valid_mask].reset_index(drop=True),
            features=features,
            # The target is positional.  Keep the mask positional too: a
            # non-RangeIndex from a handoff must not silently align labels by
            # index when this helper is reused outside the current runner.
            target=target[train["__ts__"].lt(values[0]).to_numpy()],
            seed=int(seed + block_index),
        )
        result[np.flatnonzero(valid_mask)] = prediction
    return result


def _select_oof_feature_contract_groups(
    train: pd.DataFrame,
    m1_prediction: np.ndarray,
    m12_prediction: np.ndarray,
    *,
    min_selected_rows: int = 80,
    min_ev_improvement: float = 5e-4,
    max_bad_mae_worsening: float = 0.005,
) -> tuple[set[tuple[str, str]], list[dict[str, object]]]:
    """Select M12 only where its prior OOF selections improve M1 locally."""

    valid = np.isfinite(m1_prediction) & np.isfinite(m12_prediction)
    if not valid.any():
        return set(), []
    frame = train.loc[valid].reset_index(drop=True)
    m1_selected = _select_top10(frame, np.asarray(m1_prediction)[valid], np.ones(len(frame), dtype=bool))
    m12_selected = _select_top10(frame, np.asarray(m12_prediction)[valid], np.ones(len(frame), dtype=bool))
    metrics: dict[tuple[str, str], dict[str, tuple[int, float, float]]] = {}
    for label, selected in (("m1", m1_selected), ("m12", m12_selected)):
        for (side, archetype), part in selected.groupby(
            ["side_name", "archetype_policy_key"], observed=True, sort=True
        ):
            key = (str(side), str(archetype))
            metrics.setdefault(key, {})[label] = (
                int(len(part)),
                float(pd.to_numeric(part["ev_after_1pct"], errors="coerce").mean()),
                float(pd.to_numeric(part.get("full_path_bad_mae_1r", 0.0), errors="coerce").fillna(0.0).mean()),
            )
    active: set[tuple[str, str]] = set()
    diagnostics: list[dict[str, object]] = []
    for key, payload in sorted(metrics.items()):
        m1 = payload.get("m1", (0, float("nan"), float("nan")))
        m12 = payload.get("m12", (0, float("nan"), float("nan")))
        selected = (
            m1[0] >= int(min_selected_rows)
            and m12[0] >= int(min_selected_rows)
            and np.isfinite(m1[1])
            and np.isfinite(m12[1])
            and m12[1] >= m1[1] + float(min_ev_improvement)
            and m12[2] <= m1[2] + float(max_bad_mae_worsening)
        )
        if selected:
            active.add(key)
        diagnostics.append(
            {
                "side_name": key[0],
                "archetype_policy_key": key[1],
                "m1_selected_rows": m1[0],
                "m12_selected_rows": m12[0],
                "m1_mean_ev": m1[1],
                "m12_mean_ev": m12[1],
                "m12_minus_m1_ev": m12[1] - m1[1],
                "m1_bad_mae": m1[2],
                "m12_bad_mae": m12[2],
                "active": bool(selected),
            }
        )
    return active, diagnostics


def _gate_auxiliary_by_oof_reliability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_probability: np.ndarray,
    test_probability: np.ndarray,
    *,
    min_support: int = 320,
    min_ev_gap: float = 5e-4,
    min_negative_ev_gap: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Keep an auxiliary probability only where chronological OOF evidence supports it.

    The assessment uses rows already in the fold's training history and their
    auxiliary *OOF* probabilities.  It never inspects OOS outcomes.  The
    resulting state is a side x archetype feature-selection decision frozen
    before the OOS month.
    """

    train_prob = np.asarray(train_probability, dtype=np.float32)
    test_prob = np.asarray(test_probability, dtype=np.float32)
    train_out = np.zeros(len(train), dtype=np.float32)
    test_out = np.zeros(len(test), dtype=np.float32)
    accepted: set[tuple[str, str]] = set()
    diagnostics: list[dict[str, object]] = []
    work = train.loc[:, ["side_name", "archetype_policy_key", "ev_after_1pct"]].copy()
    work["probability"] = train_prob
    for (side, archetype), part in work.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        probability = pd.to_numeric(part["probability"], errors="coerce").to_numpy(dtype=np.float32)
        ev = pd.to_numeric(part["ev_after_1pct"], errors="coerce").to_numpy(dtype=np.float32)
        valid = np.isfinite(probability) & np.isfinite(ev)
        support = int(valid.sum())
        payload: dict[str, object] = {
            "side_name": str(side),
            "archetype_policy_key": str(archetype),
            "support": support,
            "active": False,
        }
        if support < int(min_support) or float(np.nanstd(probability[valid])) < 1e-5:
            diagnostics.append(payload)
            continue
        lo, hi = np.quantile(probability[valid], [0.25, 0.75])
        low = valid & (probability <= lo)
        high = valid & (probability >= hi)
        if int(low.sum()) < 32 or int(high.sum()) < 32:
            diagnostics.append(payload)
            continue
        ev_gap = float(np.mean(ev[high]) - np.mean(ev[low]))
        negative_gap = float(np.mean(ev[high] <= 0.0) - np.mean(ev[low] <= 0.0))
        active = ev_gap <= -float(min_ev_gap) and negative_gap >= float(min_negative_ev_gap)
        payload.update(
            {
                "active": bool(active),
                "high_minus_low_ev": ev_gap,
                "high_minus_low_negative_ev_rate": negative_gap,
            }
        )
        diagnostics.append(payload)
        if active:
            accepted.add((str(side), str(archetype)))
    if accepted:
        train_key = list(zip(train["side_name"].astype(str), train["archetype_policy_key"].astype(str)))
        test_key = list(zip(test["side_name"].astype(str), test["archetype_policy_key"].astype(str)))
        train_mask = np.fromiter((key in accepted for key in train_key), dtype=bool, count=len(train_key))
        test_mask = np.fromiter((key in accepted for key in test_key), dtype=bool, count=len(test_key))
        train_out[train_mask] = train_prob[train_mask]
        test_out[test_mask] = test_prob[test_mask]
    return train_out, test_out, diagnostics


def _oof_local_risk_prior(train: pd.DataFrame, test: pd.DataFrame, blocks: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return a causal side x archetype quality prior as a meta input.

    The underlying estimate is deliberately bounded, but it is not applied to
    the score here.  The fitted meta head receives it as an ordinary feature
    and decides whether, and how, it should affect the soft target.
    """

    out = np.ones(len(train), dtype=np.float32)
    ordered_ts = np.sort(train["__ts__"].dropna().unique())
    for values in np.array_split(ordered_ts, blocks):
        if len(values) == 0:
            continue
        valid_mask = train["__ts__"].isin(values).to_numpy()
        history = train.loc[train["__ts__"] < values[0]]
        if len(history) >= 1_500:
            state, _ = _local_risk_prior_state(history)
            out[np.flatnonzero(valid_mask)] = _apply_local_risk_prior(train.loc[valid_mask], state)
    state, _ = _local_risk_prior_state(train)
    return out, _apply_local_risk_prior(test, state)


def _oof_context(train: pd.DataFrame, test: pd.DataFrame, blocks: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [
        "meta_resid_arch_support_log1p", "meta_resid_arch_entropy",
        "meta_resid_arch_expected_hit_surprise", "meta_resid_arch_expected_dirty_positive",
    ]
    out = pd.DataFrame(0.0, index=np.arange(len(train)), columns=names, dtype=np.float32)
    ordered_ts = np.sort(train["__ts__"].dropna().unique())
    for values in np.array_split(ordered_ts, blocks):
        if len(values) == 0:
            continue
        valid_mask = train["__ts__"].isin(values).to_numpy()
        history = train.loc[train["__ts__"] < values[0]]
        if len(history) >= 1_500:
            values_frame = _residual_arch_context(
                history,
                train.loc[valid_mask].reset_index(drop=True),
            )[names]
            out.loc[valid_mask, names] = values_frame.to_numpy(dtype=np.float32, copy=False)
    test_context = _residual_arch_context(train, test)[names].astype(np.float32, copy=False)
    return out.reset_index(drop=True), test_context.reset_index(drop=True)


def _semantic_residual_state_config(seed: int) -> ResidualArchetypeConfig:
    """Return the bounded semantic-state recognizer contract for M18/M19.

    This recognizer deliberately does *not* refit an auxiliary AE/GMM.  Its
    observable feature basket already contains the frozen cycle AE/GMM fields
    from the parent model contract.  Reusing those fields preserves semantic
    parity with the base/meta cycle and avoids creating an independent latent
    coordinate system inside every OOF block.
    """

    return ResidualArchetypeConfig(
        score_col="score",
        min_side_rows=2_000,
        min_local_rows=900,
        min_cluster_rows=80,
        cluster_candidates=(3, 4, 5),
        max_cluster_fit_rows=24_000,
        max_recognizer_fit_rows=24_000,
        max_recognizer_features=64,
        mutual_info_rows=12_000,
        use_residual_ae_gmm=False,
        final_refit_all_rows=False,
        fit_local_models=True,
        allow_side_fallback=False,
        rank_scope="global",
        label_mode="economic_semantic",
        random_state=int(seed),
    )


def _semantic_residual_state_observable(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop every label/output field before a frozen semantic transform.

    ``ResidualArchetypeRecognizer`` rejects outcome columns itself.  The
    explicit ``meta_target_soft`` removal closes the one historical naming gap
    between the meta handoff and the recognizer's generic outcome vocabulary.
    """

    return strip_outcomes_for_oos(
        frame.drop(
            columns=(
                "meta_target_soft",
                "__negative_hit_residual_event__",
                "__negative_ev_residual_event__",
            ),
            errors="ignore",
        )
    ).copy(deep=False)


def _fit_semantic_residual_state_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features_by_side: dict[str, list[str]],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit side x archetype semantic states, then transform frozen OOS rows."""

    candidates = _residual_state_candidates(features_by_side)
    recognizer = ResidualArchetypeRecognizer(
        _semantic_residual_state_config(seed),
        candidate_features=candidates,
    ).fit(train)
    transformed = recognizer.transform_oos(_semantic_residual_state_observable(test))
    output = (
        transformed.reindex(columns=SEMANTIC_RESIDUAL_FEATURES, fill_value=0.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
        .reset_index(drop=True)
    )
    manifest = recognizer.manifest()
    return output, {
        "local_models": int(manifest.get("local_model_count", 0)),
        "side_models": int(manifest.get("side_model_count", 0)),
        "output_feature_count": int(output.shape[1]),
        "semantic_archetypes": list(manifest.get("semantic_archetypes", [])),
        "selected_feature_counts": {
            str(key): int(len(value))
            for key, value in dict(manifest.get("selected_features_by_model", {})).items()
        },
    }


def _oof_semantic_residual_state_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features_by_side: dict[str, list[str]],
    seed: int,
    blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Create chronological OOF semantic probabilities plus frozen OOS rows."""

    oof = pd.DataFrame(
        0.0,
        index=np.arange(len(train)),
        columns=SEMANTIC_RESIDUAL_FEATURES,
        dtype=np.float32,
    )
    diagnostics: list[dict[str, object]] = []
    ordered_ts = np.sort(train["__ts__"].dropna().unique())
    min_history = 2_500
    for block_index, values in enumerate(np.array_split(ordered_ts, blocks)):
        if len(values) == 0:
            continue
        valid_mask = train["__ts__"].isin(values).to_numpy()
        history = train.loc[train["__ts__"] < values[0]].reset_index(drop=True)
        if len(history) < min_history:
            diagnostics.append(
                {
                    "block": int(block_index),
                    "status": "neutral_insufficient_history",
                    "history_rows": int(len(history)),
                    "oof_rows": int(valid_mask.sum()),
                }
            )
            continue
        transformed, detail = _fit_semantic_residual_state_features(
            history,
            train.loc[valid_mask].reset_index(drop=True),
            features_by_side=features_by_side,
            seed=int(seed + block_index),
        )
        oof.loc[valid_mask, SEMANTIC_RESIDUAL_FEATURES] = transformed.to_numpy(
            dtype=np.float32,
            copy=False,
        )
        diagnostics.append(
            {
                "block": int(block_index),
                "status": "complete",
                "history_rows": int(len(history)),
                "oof_rows": int(valid_mask.sum()),
                **detail,
            }
        )
    if len(train) < min_history:
        oos = pd.DataFrame(
            0.0,
            index=np.arange(len(test)),
            columns=SEMANTIC_RESIDUAL_FEATURES,
            dtype=np.float32,
        )
        diagnostics.append(
            {
                "block": "oos",
                "status": "neutral_insufficient_train",
                "history_rows": int(len(train)),
                "oof_rows": int(len(test)),
            }
        )
    else:
        oos, detail = _fit_semantic_residual_state_features(
            train,
            test,
            features_by_side=features_by_side,
            seed=int(seed + 10_000),
        )
        diagnostics.append(
            {
                "block": "oos",
                "status": "complete",
                "history_rows": int(len(train)),
                "oof_rows": int(len(test)),
                **detail,
            }
        )
    return oof.reset_index(drop=True), oos.reset_index(drop=True), diagnostics


def _residual_state_observable(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove every outcome-derived field before a residual-state transform."""

    return frame.drop(
        columns=[name for name in RESIDUAL_STATE_FORBIDDEN if name in frame],
        errors="ignore",
    ).copy(deep=False)


def _residual_state_config(seed: int) -> ResidualEventArchetypeConfig:
    """Compute-bounded but fully causal local failure-state discovery config."""

    return ResidualEventArchetypeConfig(
        score_col="score",
        min_global_threshold_rows=1_500,
        min_local_threshold_rows=180,
        # The state fits only the train-fitted top-20 population.  This is a
        # representation/prior learner, not a high-capacity trading head, so
        # a 400-row local floor is adequate with posterior shrinkage.  Sparse
        # streams remain neutral; they are never pooled into a side state.
        min_local_state_rows=400,
        min_side_state_rows=800,
        min_event_class_rows=24,
        max_feature_candidates=128,
        max_features_after_mi=32,
        max_features_after_lgbm=20,
        mi_sample_rows=8_000,
        lgbm_min_rows=400,
        lgbm_num_boost_round=96,
        ae_gmm_max_train_rows=6_000,
        gmm_max_train_rows=18_000,
        ae_gmm_max_iter=72,
        ae_gmm_clusters=(3, 4, 5),
        ae_gmm_reg_covars=(1e-3, 3e-3),
        ae_gmm_covariance_types=("diag",),
        ae_gmm_smooth_lambdas=(0.0,),
        enable_market_secondary=False,
        random_state=int(seed),
    )


def _residual_state_candidates(
    features_by_side: dict[str, list[str]],
) -> list[str]:
    return list(
        dict.fromkeys(
            feature
            for selected in features_by_side.values()
            for feature in selected
            if feature not in DERIVED_ANCHORS
        )
    )


def _fit_residual_state_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features_by_side: dict[str, list[str]],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit a train-only residual AE/GMM state and transform an OOS frame."""

    state = ResidualEventArchetypeState(_residual_state_config(seed)).fit(
        train,
        candidate_features=_residual_state_candidates(features_by_side),
    )
    history = _residual_state_observable(train)
    # The longest emitted temporal feature uses a 96h lookback.  Retaining more
    # history cannot affect this transform and needlessly expands OOF memory.
    if not history.empty and "__ts__" in history:
        latest = pd.to_datetime(history["__ts__"], utc=True, errors="coerce").max()
        if pd.notna(latest):
            history = history.loc[
                pd.to_datetime(history["__ts__"], utc=True, errors="coerce").ge(
                    latest - pd.Timedelta(hours=96)
                )
            ]
    transformed = state.transform_oos_with_history(
        history,
        _residual_state_observable(test),
    )
    out = transformed.reindex(columns=RESIDUAL_STATE_FEATURES, fill_value=0.0)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    manifest = state.manifest()
    return out.reset_index(drop=True), {
        "local_models": int(manifest.get("local_model_count", 0)),
        "side_fallback_models": int(manifest.get("side_fallback_model_count", 0)),
        "state_feature_count": int(out.shape[1]),
        "failure_targets": list(manifest.get("executable_failure_targets", [])),
    }


def _oof_residual_state_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features_by_side: dict[str, list[str]],
    seed: int,
    blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Generate chronological OOF state priors plus a frozen OOS transform."""

    oof = pd.DataFrame(
        0.0,
        index=np.arange(len(train)),
        columns=RESIDUAL_STATE_FEATURES,
        dtype=np.float32,
    )
    diagnostics: list[dict[str, object]] = []
    ordered_ts = np.sort(train["__ts__"].dropna().unique())
    for block_index, values in enumerate(np.array_split(ordered_ts, blocks)):
        if len(values) == 0:
            continue
        valid_mask = train["__ts__"].isin(values).to_numpy()
        history = train.loc[train["__ts__"] < values[0]].reset_index(drop=True)
        if len(history) < MIN_RESIDUAL_STATE_HISTORY:
            diagnostics.append(
                {
                    "block": int(block_index),
                    "status": "neutral_insufficient_history",
                    "history_rows": int(len(history)),
                    "oof_rows": int(valid_mask.sum()),
                }
            )
            continue
        transformed, detail = _fit_residual_state_features(
            history,
            train.loc[valid_mask].reset_index(drop=True),
            features_by_side=features_by_side,
            seed=int(seed + block_index),
        )
        oof.loc[valid_mask, RESIDUAL_STATE_FEATURES] = transformed.to_numpy(
            dtype=np.float32, copy=False
        )
        diagnostics.append(
            {
                "block": int(block_index),
                "status": "complete",
                "history_rows": int(len(history)),
                "oof_rows": int(valid_mask.sum()),
                **detail,
            }
        )
    if len(train) < MIN_RESIDUAL_STATE_HISTORY:
        oos = pd.DataFrame(
            0.0,
            index=np.arange(len(test)),
            columns=RESIDUAL_STATE_FEATURES,
            dtype=np.float32,
        )
        diagnostics.append(
            {
                "block": "oos",
                "status": "neutral_insufficient_train",
                "history_rows": int(len(train)),
                "oof_rows": int(len(test)),
            }
        )
    else:
        oos, detail = _fit_residual_state_features(
            train,
            test,
            features_by_side=features_by_side,
            seed=int(seed + 10_000),
        )
        diagnostics.append(
            {
                "block": "oos",
                "status": "complete",
                "history_rows": int(len(train)),
                "oof_rows": int(len(test)),
                **detail,
            }
        )
    return oof.reset_index(drop=True), oos.reset_index(drop=True), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--max-train-rows-per-month", type=int, default=6_000)
    parser.add_argument(
        "--report-direct-mechanism-support-only",
        action="store_true",
        help=(
            "Write train-only target prevalence by fold/side/archetype and exit "
            "without fitting auxiliary or main meta models."
        ),
    )
    parser.add_argument(
        "--support-mechanisms",
        default=",".join(DIRECT_RESIDUAL_MECHANISMS),
        help="Comma-separated direct mechanism names for --report-direct-mechanism-support-only.",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARM_FEATURES),
        help="Comma-separated M0-M20 arms to run; each arm remains a separately fitted meta head.",
    )
    parser.add_argument("--aux-oof-blocks", type=int, default=4)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_FEATURE_CONTRACT)
    parser.add_argument("--feature-store-id", default=DEFAULT_FEATURE_STORE_ID)
    parser.add_argument("--min-complete-coverage", type=float, default=0.90)
    parser.add_argument(
        "--source-mode",
        choices=("saved_full_ledger", "static_handoff"),
        default="saved_full_ledger",
        help="Use the persisted residual_state_mda95 matrices or hydrate a compact handoff.",
    )
    parser.add_argument(
        "--full-feature-ledger",
        action="append",
        type=Path,
        default=None,
        help="Repeat to override the persisted full feature ledgers used by saved_full_ledger mode.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    requested_arms = [name.strip() for name in str(args.arms).split(",") if name.strip()]
    unknown_arms = sorted(set(requested_arms) - set(ARM_FEATURES))
    if unknown_arms:
        raise ValueError(f"Unknown meta-head ablation arms: {unknown_arms}")
    selected_arm_features = {name: ARM_FEATURES[name] for name in requested_arms}
    requested_features = set(
        feature for values in selected_arm_features.values() for feature in values
    )
    requires_phase_context = bool(
        set(requested_features).intersection(PHASE_AUX_FEATURES)
        or LOCAL_TRANSITION_RESIDUAL_FEATURE in requested_features
    )
    phase_source_features = list(_PHASE_SOURCE_NAMES) if requires_phase_context else []
    requires_aegmm_transition_context = (
        AEGMM_TRANSITION_RESIDUAL_FEATURE in requested_features
    )
    directed_transition_input_features = list(
        dict.fromkeys(
            [
                *(
                    AEGMM_COMPONENT_TRANSITION_SOURCE_FEATURES
                    if (
                        AEGMM_COMPONENT_TRANSITION_RESIDUAL_FEATURE
                        in requested_features
                        or set(AEGMM_COMPONENT_TRANSITION_SOURCE_FEATURES).intersection(
                            requested_features
                        )
                    )
                    else ()
                ),
                *(
                    AEGMM_DOMINANT_STATE_TRANSITION_SOURCE_FEATURES
                    if (
                        AEGMM_DURABLE_TRANSITION_RESIDUAL_FEATURE
                        in requested_features
                        or set(AEGMM_DOMINANT_STATE_TRANSITION_SOURCE_FEATURES).intersection(
                            requested_features
                        )
                    )
                    else ()
                ),
            ]
        )
    )
    requires_aegmm_component_transition_context = bool(directed_transition_input_features)
    transition_source_features = (
        list(AEGMM_TRANSITION_SOURCE_FEATURES)
        if requires_aegmm_transition_context
        else []
    )
    component_transition_source_features = directed_transition_input_features
    explicit_source_features = list(
        dict.fromkeys(
            [
                *phase_source_features,
                *transition_source_features,
                *component_transition_source_features,
            ]
        )
    )
    months = _parse_months(args.months)
    required_months = sorted({period for month in months for period in pd.period_range(month - args.train_months, month, freq="M")})
    base_features_by_side = _load_feature_contract(args.feature_contract)
    selected_union = list(dict.fromkeys(feature for values in base_features_by_side.values() for feature in values))
    static_coverage: dict[str, dict[str, float]] = {}
    if args.source_mode == "saved_full_ledger":
        if bool(args.report_direct_mechanism_support_only):
            data, coverage = _load_saved_direct_mechanism_support_ledgers(
                list(args.full_feature_ledger or DEFAULT_FULL_FEATURE_LEDGERS),
                labels_root=args.labels_root,
                months=required_months,
            )
        else:
            data, coverage = _load_saved_full_feature_ledgers(
                list(args.full_feature_ledger or DEFAULT_FULL_FEATURE_LEDGERS),
                labels_root=args.labels_root,
                months=required_months,
                features_by_side=base_features_by_side,
                extra_feature_names=explicit_source_features,
                full_months={str(month) for month in months},
                max_rows_per_train_month=int(args.max_train_rows_per_month),
            )
        _require_month_coverage(
            coverage,
            required_months,
            context="saved_full_ledger",
        )
    else:
        data, _, coverage = _load_handoff_with_labels(
            args.input_parquet,
            args.labels_root,
            required_months,
            extra_feature_names=list(dict.fromkeys([*selected_union, *explicit_source_features])),
            full_months={str(month) for month in months},
            max_rows_per_train_month=int(args.max_train_rows_per_month),
        )
        _require_month_coverage(
            coverage,
            required_months,
            context="static_handoff",
        )
    if (
        not bool(args.report_direct_mechanism_support_only)
        and ("meta_target_soft" not in data or data["meta_target_soft"].isna().any())
    ):
        raise RuntimeError("The meta soft target is required for feature ablation")
    if requires_aegmm_transition_context:
        missing_transition = [
            feature for feature in AEGMM_TRANSITION_SOURCE_FEATURES if feature not in data
        ]
        if missing_transition:
            raise RuntimeError(
                "M26 requires causal full-panel frozen-AE/GMM transition columns; "
                "materialize them with materialize_frozen_aegmm_transition_context.py first. "
                f"Missing={missing_transition[:8]}"
            )
        transition_complete = data.loc[:, list(AEGMM_TRANSITION_SOURCE_FEATURES)].apply(
            pd.to_numeric, errors="coerce"
        ).notna().all(axis=1)
        if float(transition_complete.mean()) < 0.90:
            raise RuntimeError(
                "M26 transition context coverage is below 90% on the loaded candidate stream: "
                f"{float(transition_complete.mean()):.2%}"
            )
    if requires_aegmm_component_transition_context:
        missing_component_transition = [
            feature for feature in directed_transition_input_features if feature not in data
        ]
        if missing_component_transition:
            raise RuntimeError(
                "directed transition arms require causal full-panel frozen-AE/GMM transition columns; "
                "materialize them with materialize_frozen_aegmm_transition_context.py first. "
                f"Missing={missing_component_transition[:8]}"
            )
        component_transition_complete = data.loc[:, directed_transition_input_features].apply(
            pd.to_numeric, errors="coerce"
        ).notna().all(axis=1)
        if float(component_transition_complete.mean()) < 0.90:
            raise RuntimeError(
                "directed transition context coverage is below 90% on the loaded candidate stream: "
                f"{float(component_transition_complete.mean()):.2%}"
            )
    if bool(args.report_direct_mechanism_support_only):
        requested_mechanisms = [
            value.strip()
            for value in str(args.support_mechanisms).split(",")
            if value.strip()
        ]
        unknown_mechanisms = sorted(
            set(requested_mechanisms).difference(DIRECT_RESIDUAL_MECHANISMS)
        )
        if unknown_mechanisms:
            raise ValueError(f"Unknown direct residual mechanisms: {unknown_mechanisms}")
        support_rows: list[dict[str, object]] = []
        for month in months:
            start = pd.Timestamp(month.start_time, tz="UTC")
            earliest = start - pd.DateOffset(months=int(args.train_months))
            train_support = data.loc[
                (data["__ts__"] >= earliest) & (data["__ts__"] < start)
            ].reset_index(drop=True)
            if len(train_support) < 1_500:
                continue
            for mechanism in requested_mechanisms:
                support_rows.extend(
                    _direct_mechanism_support_rows(
                        train_support,
                        fold_month=str(month),
                        mechanism=mechanism,
                    )
                )
        support = pd.DataFrame(support_rows)
        support.to_csv(args.output / "direct_mechanism_target_support.csv", index=False)
        _write_json(
            args.output / "manifest.json",
            {
                "schema": "meta_residual_direct_mechanism_support_v1",
                "months": [str(month) for month in months],
                "train_months": int(args.train_months),
                "mechanisms": requested_mechanisms,
                "source_mode": args.source_mode,
                "contract": (
                    "train-only causal residual targets; this report does not fit "
                    "or evaluate an OOS model"
                ),
            },
        )
        print(
            json.dumps(
                {
                    "event": "direct_mechanism_support_complete",
                    "rows": int(len(support)),
                    "trainable_side_archetype_rows": int(
                        support.loc[
                            support["level"].eq("side_archetype") & support["trainable"],
                        ].shape[0]
                    )
                    if not support.empty
                    else 0,
                }
            ),
            flush=True,
        )
        return
    if args.source_mode == "static_handoff":
        hydrated: list[pd.DataFrame] = []
        evaluation_months = {str(month) for month in months}
        for period, part in data.groupby(data["__ts__"].dt.to_period("M"), observed=True, sort=True):
            # The full-history handoff is used to extend state-model support,
            # not to fit a wider main meta model on every historical row.
            # Sample train-only months *before* static hydration, while every
            # OOS evaluation month remains intact for an exact top-k report.
            # This keeps memory bounded and preserves the B/M/E temporal
            # representation used by the normal training contract.
            source_part = part.reset_index(drop=True)
            if str(period) not in evaluation_months:
                source_part = _time_spread_sample(
                    source_part,
                    int(args.max_train_rows_per_month),
                )
            materialized, local_coverage = _hydrate_static_features(
                source_part,
                features_by_side=base_features_by_side,
                feature_store_id=str(args.feature_store_id),
                min_complete_coverage=float(args.min_complete_coverage),
                extra_feature_names=explicit_source_features,
            )
            hydrated.append(materialized)
            static_coverage[str(period)] = local_coverage
        data = pd.concat(hydrated, ignore_index=True, copy=False)
    else:
        # Selection-time contract coverage is the hard eligibility criterion.
        # Monthly OOS slices can contain early-history warm-up rows; retain
        # those with native LightGBM missing-value handling, matching the
        # incumbent model, and record their complete-case rate separately.
        _assert_contract_coverage(
            data,
            features_by_side=base_features_by_side,
            min_complete_coverage=float(args.min_complete_coverage),
            enforce=True,
        )
        for period, part in data.groupby(data["__ts__"].dt.to_period("M"), observed=True, sort=True):
            static_coverage[str(period)] = _assert_contract_coverage(
                part,
                features_by_side=base_features_by_side,
                min_complete_coverage=float(args.min_complete_coverage),
                enforce=False,
            )
    rows: list[dict[str, object]] = []
    detail: list[pd.DataFrame] = []
    selected_ledgers: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for fold, month in enumerate(months):
        start, end = pd.Timestamp(month.start_time, tz="UTC"), pd.Timestamp((month + 1).start_time, tz="UTC")
        earliest = start - pd.DateOffset(months=int(args.train_months))
        train_full = data.loc[(data["__ts__"] >= earliest) & (data["__ts__"] < start) & (pd.to_numeric(data["base_rank_pct_by_timestamp"], errors="coerce") >= 0.80)]
        test = data.loc[(data["__ts__"] >= start) & (data["__ts__"] < end) & (pd.to_numeric(data["base_rank_pct_by_timestamp"], errors="coerce") >= 0.80)].reset_index(drop=True)
        train = pd.concat([_time_spread_sample(part, args.max_train_rows_per_month) for _, part in train_full.groupby(train_full["__ts__"].dt.to_period("M"), observed=True, sort=True)], ignore_index=True)
        if len(train) < 8_000 or len(test) < 1_000:
            diagnostics.append(
                {
                    "month": str(month),
                    "status": "skipped_insufficient_candidate_rows",
                    "train_rows": int(len(train)),
                    "oos_rows": int(len(test)),
                    "required_train_rows": 8_000,
                    "required_oos_rows": 1_000,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "fold_skipped",
                        "month": str(month),
                        "reason": "insufficient_candidate_rows",
                        "train_rows": int(len(train)),
                        "oos_rows": int(len(test)),
                    }
                ),
                flush=True,
            )
            continue
        phase_static_coverage: dict[str, dict[str, object]] = {}
        if phase_source_features:
            # Phase states are market-wide by construction.  Read a small
            # canonical sentinel panel and broadcast it by timestamp instead
            # of materializing an unnecessary candidate-symbol x feature cube.
            train, phase_static_coverage["train"] = _hydrate_market_phase_sources(
                train,
                feature_store_id=str(args.feature_store_id),
            )
            test, phase_static_coverage["oos"] = _hydrate_market_phase_sources(
                test,
                feature_store_id=str(args.feature_store_id),
            )
        if "base_score_rank_pct_train_prior" not in train:
            _add_train_prior_rank(train, test)
        aux_good_train = np.full(len(train), 0.5, dtype=np.float32)
        aux_good_test = np.full(len(test), 0.5, dtype=np.float32)
        aux_path_train = np.full(len(train), 0.5, dtype=np.float32)
        aux_path_test = np.full(len(test), 0.5, dtype=np.float32)
        aux_resid_train = np.full(len(train), 0.5, dtype=np.float32)
        aux_resid_test = np.full(len(test), 0.5, dtype=np.float32)
        aux_shortfall_train = np.zeros(len(train), dtype=np.float32)
        aux_shortfall_test = np.zeros(len(test), dtype=np.float32)
        direct_residual_train = {
            mechanism: np.zeros(len(train), dtype=np.float32)
            for mechanism in DIRECT_RESIDUAL_MECHANISMS
        }
        direct_residual_test = {
            mechanism: np.zeros(len(test), dtype=np.float32)
            for mechanism in DIRECT_RESIDUAL_MECHANISMS
        }
        gated_direct_residual_train = {
            mechanism: np.zeros(len(train), dtype=np.float32)
            for mechanism in RELIABILITY_GATED_MECHANISMS
        }
        gated_direct_residual_test = {
            mechanism: np.zeros(len(test), dtype=np.float32)
            for mechanism in RELIABILITY_GATED_MECHANISMS
        }
        direct_reliability_diagnostics: dict[str, list[dict[str, object]]] = {}
        direct_oos_probability_diagnostics: dict[str, dict[str, float | None]] = {}
        size_train = np.ones(len(train), dtype=np.float32)
        size_test = np.ones(len(test), dtype=np.float32)
        if "meta_aux_good_trade_oof" in requested_features:
            aux_good_train, aux_good_test = _oof_auxiliary(train, test, lambda a, b, s: _fit_good(a, b, base_features_by_side, s), neutral=0.5, seed=args.seed + fold, blocks=args.aux_oof_blocks)
        if "meta_aux_conditional_path_oof" in requested_features:
            aux_path_train, aux_path_test = _oof_auxiliary(train, test, lambda a, b, s: _fit_path(a, b, base_features_by_side, s), neutral=0.5, seed=args.seed + 1_000 + fold, blocks=args.aux_oof_blocks)
        if "meta_aux_negative_residual_oof" in requested_features:
            aux_resid_train, aux_resid_test = _oof_auxiliary(train, test, lambda a, b, s: _fit_residual(a, b, base_features_by_side, s), neutral=0.5, seed=args.seed + 2_000 + fold, blocks=args.aux_oof_blocks)
        if "meta_aux_residual_ev_shortfall_oof" in requested_features:
            aux_shortfall_train, aux_shortfall_test = _oof_auxiliary(
                train,
                test,
                lambda a, b, s: _fit_residual_shortfall(a, b, base_features_by_side, s),
                neutral=0.0,
                seed=args.seed + 3_500 + fold,
                blocks=args.aux_oof_blocks,
            )
        for mechanism_index, mechanism in enumerate(DIRECT_RESIDUAL_MECHANISMS):
            feature_name = f"meta_aux_{mechanism}_oof"
            gated_feature_name = f"meta_aux_{mechanism}_reliability_gated_oof"
            if feature_name in requested_features or gated_feature_name in requested_features:
                (
                    direct_residual_train[mechanism],
                    direct_residual_test[mechanism],
                ) = _oof_auxiliary(
                    train,
                    test,
                    lambda a, b, s, mechanism=mechanism: _fit_residual_mechanism(
                        a, b, base_features_by_side, s, mechanism
                    ),
                    neutral=0.0,
                    seed=args.seed + 4_000 + mechanism_index * 100 + fold,
                    blocks=args.aux_oof_blocks,
                )
                if mechanism in RELIABILITY_GATED_MECHANISMS:
                    (
                        gated_direct_residual_train[mechanism],
                        gated_direct_residual_test[mechanism],
                        direct_reliability_diagnostics[mechanism],
                    ) = _gate_auxiliary_by_oof_reliability(
                        train,
                        test,
                        direct_residual_train[mechanism],
                        direct_residual_test[mechanism],
                    )
        # Evaluation only: derive the OOS event labels from a residual state
        # fitted on the preceding train window.  Test outcomes define the
        # diagnostic label but never enter auxiliary probabilities, reliability
        # gates, or main-head inputs.  Without this, a poor incremental result
        # cannot distinguish an unlearnable failure target from a state the
        # main meta model already absorbs.
        if any(
            f"meta_aux_{mechanism}_oof" in requested_features
            or f"meta_aux_{mechanism}_reliability_gated_oof" in requested_features
            for mechanism in DIRECT_RESIDUAL_MECHANISMS
        ):
            residual_eval_states: dict[str, tuple[dict[str, object], str]] = {}
            for mechanism in DIRECT_RESIDUAL_MECHANISMS:
                raw_name = f"meta_aux_{mechanism}_oof"
                gated_name = f"meta_aux_{mechanism}_reliability_gated_oof"
                if raw_name not in requested_features and gated_name not in requested_features:
                    continue
                residual_kind = "clean_exec" if mechanism in TRANSITION_STATE_MECHANISMS else "ev_after_1pct"
                if residual_kind not in residual_eval_states:
                    _, _, state = _causal_residual_target(
                        train,
                        value_col=residual_kind,
                        label_col=(
                            "__negative_hit_residual_event__"
                            if residual_kind == "clean_exec"
                            else "__negative_ev_residual_event__"
                        ),
                    )
                    residual_eval_states[residual_kind] = (state, residual_kind)
                residual_eval_state, value_col = residual_eval_states[residual_kind]
                test_negative_residual = _negative_residual_event(
                    test,
                    residual_eval_state,
                    value_col=value_col,
                )
                event = _direct_residual_mechanism_target(
                    test,
                    test_negative_residual,
                    mechanism,
                )
                if raw_name in requested_features:
                    direct_oos_probability_diagnostics[f"{mechanism}__raw"] = _probability_metrics(
                        event,
                        direct_residual_test[mechanism],
                        mechanism,
                    )
                if gated_name in requested_features:
                    direct_oos_probability_diagnostics[f"{mechanism}__reliability_gated"] = _probability_metrics(
                        event,
                        gated_direct_residual_test[mechanism],
                        mechanism,
                    )
        if "meta_aux_local_size_risk_oof" in requested_features:
            size_train, size_test = _oof_local_risk_prior(train, test, blocks=args.aux_oof_blocks)
        state_train: dict[str, np.ndarray] = {}
        state_test: dict[str, np.ndarray] = {}
        for idx, (name, (side, archetype)) in enumerate(STATE_CELLS.items()):
            state_train[name] = np.zeros(len(train), dtype=np.float32)
            state_test[name] = np.zeros(len(test), dtype=np.float32)
            if f"meta_aux_{name}_oof" in requested_features:
                state_train[name], state_test[name] = _oof_auxiliary(train, test, lambda a, b, s, side=side, archetype=archetype: _fit_state(a, b, base_features_by_side, s, side, archetype), neutral=0.0, seed=args.seed + 3_000 + idx * 100 + fold, blocks=args.aux_oof_blocks)
        context_train = pd.DataFrame(index=np.arange(len(train)))
        context_test = pd.DataFrame(index=np.arange(len(test)))
        # ``expected_dirty_positive`` exists in both the old compact prior and
        # the semantic recognizer.  When a semantic arm is active it must come
        # from the richer recognizer only; the remaining three legacy fields
        # are unambiguous and still trigger the compact context path for M5.
        legacy_only = set(LEGACY_RESIDUAL_CONTEXT_FEATURES).difference(
            SEMANTIC_RESIDUAL_FEATURES
        )
        if set(requested_features).intersection(legacy_only):
            context_train, context_test = _oof_context(train, test, blocks=args.aux_oof_blocks)
            # The semantic recognizer is authoritative for its own expected
            # dirty-path estimate.  The compact context remains useful for
            # expected hit surprise, but must not create duplicate labels.
            if set(requested_features).intersection(SEMANTIC_RESIDUAL_FEATURES):
                keep_context = [
                    name
                    for name in context_train.columns
                    if name in requested_features and name not in SEMANTIC_RESIDUAL_FEATURES
                ]
                context_train = context_train.reindex(columns=keep_context)
                context_test = context_test.reindex(columns=keep_context)
        phase_train = pd.DataFrame(0.0, index=np.arange(len(train)), columns=PHASE_AUX_FEATURES, dtype=np.float32)
        phase_test = pd.DataFrame(0.0, index=np.arange(len(test)), columns=PHASE_AUX_FEATURES, dtype=np.float32)
        phase_diagnostics: dict[str, object] = {"status": "not_requested"}
        if requires_phase_context:
            phase_train, phase_test, phase_diagnostics = _causal_phase_state_context(train, test)
        transition_negative_hit_train = np.full(len(train), 0.5, dtype=np.float32)
        transition_negative_hit_test = np.full(len(test), 0.5, dtype=np.float32)
        transition_state_diagnostics: dict[str, float | None] = {}
        if LOCAL_TRANSITION_RESIDUAL_FEATURE in requested_features:
            transition_train_frame = pd.concat(
                [train.reset_index(drop=True), phase_train], axis=1
            )
            transition_test_frame = pd.concat(
                [test.reset_index(drop=True), phase_test], axis=1
            )
            (
                transition_negative_hit_train,
                transition_negative_hit_test,
            ) = _oof_auxiliary(
                transition_train_frame,
                transition_test_frame,
                lambda a, b, s: _fit_local_transition_negative_hit(a, b, seed=s),
                neutral=0.5,
                seed=args.seed + 40_000 + fold,
                blocks=args.aux_oof_blocks,
            )
            transition_state_diagnostics = _probability_metrics(
                _negative_residual_event(
                    test,
                    _causal_residual_target(
                        train,
                        value_col="clean_exec",
                        label_col="__negative_hit_residual_event__",
                    )[2],
                    value_col="clean_exec",
                ),
                transition_negative_hit_test,
                "local_transition_negative_hit",
            )
        aegmm_transition_negative_hit_train = np.full(len(train), 0.5, dtype=np.float32)
        aegmm_transition_negative_hit_test = np.full(len(test), 0.5, dtype=np.float32)
        aegmm_transition_diagnostics: dict[str, float | None] = {}
        if AEGMM_TRANSITION_RESIDUAL_FEATURE in requested_features:
            (
                aegmm_transition_negative_hit_train,
                aegmm_transition_negative_hit_test,
            ) = _oof_auxiliary(
                train.reset_index(drop=True),
                test.reset_index(drop=True),
                lambda a, b, s: _fit_local_aegmm_transition_negative_hit(a, b, seed=s),
                neutral=0.5,
                seed=args.seed + 45_000 + fold,
                blocks=args.aux_oof_blocks,
            )
            aegmm_transition_diagnostics = _probability_metrics(
                _negative_residual_event(
                    test,
                    _causal_residual_target(
                        train,
                        value_col="clean_exec",
                        label_col="__negative_hit_residual_event__",
                    )[2],
                    value_col="clean_exec",
                ),
                aegmm_transition_negative_hit_test,
                "aegmm_transition_negative_hit",
            )
        aegmm_component_transition_negative_hit_train = np.full(
            len(train), 0.5, dtype=np.float32
        )
        aegmm_component_transition_negative_hit_test = np.full(
            len(test), 0.5, dtype=np.float32
        )
        aegmm_component_transition_diagnostics: dict[str, float | None] = {}
        if AEGMM_COMPONENT_TRANSITION_RESIDUAL_FEATURE in requested_features:
            (
                aegmm_component_transition_negative_hit_train,
                aegmm_component_transition_negative_hit_test,
            ) = _oof_auxiliary(
                train.reset_index(drop=True),
                test.reset_index(drop=True),
                lambda a, b, s: _fit_local_aegmm_component_transition_negative_hit(
                    a, b, seed=s
                ),
                neutral=0.5,
                seed=args.seed + 47_000 + fold,
                blocks=args.aux_oof_blocks,
            )
            aegmm_component_transition_diagnostics = _probability_metrics(
                _negative_residual_event(
                    test,
                    _causal_residual_target(
                        train,
                        value_col="clean_exec",
                        label_col="__negative_hit_residual_event__",
                    )[2],
                    value_col="clean_exec",
                ),
                aegmm_component_transition_negative_hit_test,
                "aegmm_component_transition_negative_hit",
            )
        aegmm_durable_transition_negative_hit_train = np.full(
            len(train), 0.5, dtype=np.float32
        )
        aegmm_durable_transition_negative_hit_test = np.full(
            len(test), 0.5, dtype=np.float32
        )
        aegmm_durable_transition_diagnostics: dict[str, float | None] = {}
        if AEGMM_DURABLE_TRANSITION_RESIDUAL_FEATURE in requested_features:
            (
                aegmm_durable_transition_negative_hit_train,
                aegmm_durable_transition_negative_hit_test,
            ) = _oof_auxiliary(
                train.reset_index(drop=True),
                test.reset_index(drop=True),
                lambda a, b, s: _fit_local_aegmm_durable_transition_negative_hit(
                    a, b, seed=s
                ),
                neutral=0.5,
                seed=args.seed + 49_000 + fold,
                blocks=args.aux_oof_blocks,
            )
            aegmm_durable_transition_diagnostics = _probability_metrics(
                _negative_residual_event(
                    test,
                    _causal_residual_target(
                        train,
                        value_col="clean_exec",
                        label_col="__negative_hit_residual_event__",
                    )[2],
                    value_col="clean_exec",
                ),
                aegmm_durable_transition_negative_hit_test,
                "aegmm_durable_transition_negative_hit",
            )
        residual_state_train = pd.DataFrame(
            0.0,
            index=np.arange(len(train)),
            columns=RESIDUAL_STATE_FEATURES,
            dtype=np.float32,
        )
        residual_state_test = pd.DataFrame(
            0.0,
            index=np.arange(len(test)),
            columns=RESIDUAL_STATE_FEATURES,
            dtype=np.float32,
        )
        residual_state_diagnostics: list[dict[str, object]] = []
        if any(name in requested_features for name in RESIDUAL_STATE_FEATURES):
            (
                residual_state_train,
                residual_state_test,
                residual_state_diagnostics,
            ) = _oof_residual_state_features(
                train,
                test,
                features_by_side=base_features_by_side,
                seed=args.seed + 20_000 + fold,
                blocks=args.aux_oof_blocks,
            )
        semantic_state_train = pd.DataFrame(
            0.0,
            index=np.arange(len(train)),
            columns=SEMANTIC_RESIDUAL_FEATURES,
            dtype=np.float32,
        )
        semantic_state_test = pd.DataFrame(
            0.0,
            index=np.arange(len(test)),
            columns=SEMANTIC_RESIDUAL_FEATURES,
            dtype=np.float32,
        )
        semantic_state_diagnostics: list[dict[str, object]] = []
        if any(name in requested_features for name in SEMANTIC_RESIDUAL_FEATURES):
            (
                semantic_state_train,
                semantic_state_test,
                semantic_state_diagnostics,
            ) = _oof_semantic_residual_state_features(
                train,
                test,
                features_by_side=base_features_by_side,
                seed=args.seed + 30_000 + fold,
                blocks=args.aux_oof_blocks,
            )
        extras_train = pd.DataFrame({"meta_aux_good_trade_oof": aux_good_train, "meta_aux_conditional_path_oof": aux_path_train, "meta_aux_local_size_risk_oof": size_train, "meta_aux_negative_residual_oof": aux_resid_train, "meta_aux_residual_ev_shortfall_oof": aux_shortfall_train, LOCAL_TRANSITION_RESIDUAL_FEATURE: transition_negative_hit_train, AEGMM_TRANSITION_RESIDUAL_FEATURE: aegmm_transition_negative_hit_train, AEGMM_COMPONENT_TRANSITION_RESIDUAL_FEATURE: aegmm_component_transition_negative_hit_train, AEGMM_DURABLE_TRANSITION_RESIDUAL_FEATURE: aegmm_durable_transition_negative_hit_train, **{f"meta_aux_{name}_oof": value for name, value in state_train.items()}, **{f"meta_aux_{name}_oof": value for name, value in direct_residual_train.items()}, **{f"meta_aux_{name}_reliability_gated_oof": value for name, value in gated_direct_residual_train.items()}})
        extras_test = pd.DataFrame({"meta_aux_good_trade_oof": aux_good_test, "meta_aux_conditional_path_oof": aux_path_test, "meta_aux_local_size_risk_oof": size_test, "meta_aux_negative_residual_oof": aux_resid_test, "meta_aux_residual_ev_shortfall_oof": aux_shortfall_test, LOCAL_TRANSITION_RESIDUAL_FEATURE: transition_negative_hit_test, AEGMM_TRANSITION_RESIDUAL_FEATURE: aegmm_transition_negative_hit_test, AEGMM_COMPONENT_TRANSITION_RESIDUAL_FEATURE: aegmm_component_transition_negative_hit_test, AEGMM_DURABLE_TRANSITION_RESIDUAL_FEATURE: aegmm_durable_transition_negative_hit_test, **{f"meta_aux_{name}_oof": value for name, value in state_test.items()}, **{f"meta_aux_{name}_oof": value for name, value in direct_residual_test.items()}, **{f"meta_aux_{name}_reliability_gated_oof": value for name, value in gated_direct_residual_test.items()}})
        extras_train = pd.concat(
            [
                extras_train,
                context_train,
                phase_train,
                residual_state_train,
                semantic_state_train,
            ],
            axis=1,
        )
        extras_test = pd.concat(
            [
                extras_test,
                context_test,
                phase_test,
                residual_state_test,
                semantic_state_test,
            ],
            axis=1,
        )
        # Direct transition inputs are an ordinary main-head feature path for
        # M28.  They remain untouched causal frozen-state values from the
        # full-panel materialization; unlike M27 they are not compressed into
        # a learned residual probability before the side/archetype meta model
        # can use them.
        if directed_transition_input_features:
            extras_train = pd.concat(
                [
                    extras_train,
                    train.loc[:, directed_transition_input_features]
                    .reset_index(drop=True),
                ],
                axis=1,
            )
            extras_test = pd.concat(
                [
                    extras_test,
                    test.loc[:, directed_transition_input_features]
                    .reset_index(drop=True),
                ],
                axis=1,
            )
        observed = {}
        routed_arm = next(
            (
                arm
                for arm in (
                    "M15_oof_routed_m1_m12",
                    "M30_oof_routed_m1_m29_durable_transition",
                )
                if arm in selected_arm_features
            ),
            None,
        )
        if routed_arm is not None:
            target = pd.to_numeric(train["meta_target_soft"], errors="coerce").fillna(0.0).to_numpy(np.float32)
            m1_features = {
                side: [*base_features_by_side[side], "meta_aux_good_trade_oof"]
                for side in base_features_by_side
            }
            alternate_extras = (
                ["meta_aux_good_trade_oof", "meta_aux_negative_residual_oof"]
                if routed_arm == "M15_oof_routed_m1_m12"
                else ["meta_aux_good_trade_oof", *AEGMM_DOMINANT_STATE_TRANSITION_SOURCE_FEATURES]
            )
            alternate_features = {
                side: [*base_features_by_side[side], *alternate_extras]
                for side in base_features_by_side
            }
            m1_train = pd.concat([train.reset_index(drop=True), extras_train.loc[:, ["meta_aux_good_trade_oof"]]], axis=1)
            m1_test = pd.concat([test.reset_index(drop=True), extras_test.loc[:, ["meta_aux_good_trade_oof"]]], axis=1)
            # Transition inputs can already be present on the loaded source
            # frame.  The fold-local ``extras`` values are authoritative, so
            # remove inherited copies before assembly just as the normal-arm
            # path does below.
            alternate_train = pd.concat(
                [
                    train.reset_index(drop=True).drop(columns=alternate_extras, errors="ignore"),
                    extras_train.loc[:, alternate_extras],
                ],
                axis=1,
            )
            alternate_test = pd.concat(
                [
                    test.reset_index(drop=True).drop(columns=alternate_extras, errors="ignore"),
                    extras_test.loc[:, alternate_extras],
                ],
                axis=1,
            )
            m1_oof = _oof_main_predictions(m1_train, features=m1_features, target=target, seed=args.seed + 50_000 + fold, blocks=args.aux_oof_blocks)
            alternate_oof = _oof_main_predictions(alternate_train, features=alternate_features, target=target, seed=args.seed + 60_000 + fold, blocks=args.aux_oof_blocks)
            active_groups, routing_diagnostics = _select_oof_feature_contract_groups(alternate_train, m1_oof, alternate_oof)
            # Reuse the arm seeds below.  This makes an empty routed set an
            # exact M1 replay, and a fully active routed set an exact M12
            # replay; the OOF routing decision is the only intended change.
            m1_prediction, m1_report = _fit_predict(
                m1_train,
                m1_test,
                features=m1_features,
                target=target,
                # Match the ordinary M1 arm exactly when no local alternate
                # contract is activated.  Contract routing must not create a
                # stochastic-model difference of its own.
                seed=args.seed + MAIN_HEAD_SEED_OFFSET + fold,
            )
            alternate_prediction, alternate_report = _fit_predict(
                alternate_train,
                alternate_test,
                features=alternate_features,
                target=target,
                seed=args.seed + MAIN_HEAD_SEED_OFFSET + 20_000 + fold,
            )
            keys = list(zip(test["side_name"].astype(str), test["archetype_policy_key"].astype(str)))
            route_mask = np.fromiter((key in active_groups for key in keys), dtype=bool, count=len(keys))
            routed_prediction = np.where(route_mask, alternate_prediction, m1_prediction).astype(np.float32)
            selected = _select_top10(test, routed_prediction, np.ones(len(test), dtype=bool))
            rows.append({
                "month": str(month),
                **_metrics(selected, routed_arm),
                **_selected_hit_surprise_metrics(selected, train),
            })
            detail.append(_breakdown(selected, routed_arm))
            selected_ledgers.append(_selected_ledger(selected, routed_arm))
            observed[routed_arm] = {
                "m1": m1_report,
                "alternate": alternate_report,
                "active_group_count": int(len(active_groups)),
                "routed_oos_rows": int(route_mask.sum()),
            }
        for arm, extra_names in selected_arm_features.items():
            if arm in {"M15_oof_routed_m1_m12", "M30_oof_routed_m1_m29_durable_transition"}:
                continue
            features = {
                side: [*base_features_by_side[side], *extra_names]
                for side in base_features_by_side
            }
            # A persisted parent ledger can carry an older diagnostic version
            # of a residual feature.  The fold-OOF auxiliary value is the
            # authoritative model input for this ablation; drop any inherited
            # names before concatenation so pandas cannot silently create
            # duplicate labels or let a stale field shadow the OOF one.
            inherited = list(dict.fromkeys(extra_names))
            x_train = pd.concat(
                [
                    train.reset_index(drop=True).drop(columns=inherited, errors="ignore"),
                    extras_train.loc[:, extra_names],
                ],
                axis=1,
            )
            x_test = pd.concat(
                [
                    test.reset_index(drop=True).drop(columns=inherited, errors="ignore"),
                    extras_test.loc[:, extra_names],
                ],
                axis=1,
            )
            duplicate_train = x_train.columns[x_train.columns.duplicated()].unique().tolist()
            duplicate_test = x_test.columns[x_test.columns.duplicated()].unique().tolist()
            if duplicate_train or duplicate_test:
                raise RuntimeError(
                    "Auxiliary input assembly produced duplicate feature names "
                    f"for {arm}: train={duplicate_train[:12]}, test={duplicate_test[:12]}"
                )
            prediction, report = _fit_predict(
                x_train,
                x_test,
                features=features,
                target=pd.to_numeric(train["meta_target_soft"], errors="coerce").fillna(0.0).to_numpy(np.float32),
                seed=args.seed + MAIN_HEAD_SEED_OFFSET + fold,
            )
            selected = _select_top10(x_test, prediction, np.ones(len(x_test), dtype=bool))
            rows.append({
                "month": str(month),
                **_metrics(selected, arm),
                **_selected_hit_surprise_metrics(selected, train),
            })
            detail.append(_breakdown(selected, arm))
            selected_ledgers.append(_selected_ledger(selected, arm))
            observed[arm] = report
        diagnostics.append({"month": str(month), "train_rows": int(len(train)), "oos_rows": int(len(test)), "aux_good": _probability_metrics(_good_trade_target(test), aux_good_test, "good_trade"), "aux_path": _probability_metrics(_conditional_path_target(test), aux_path_test, "conditional_path"), "direct_reliability": direct_reliability_diagnostics, "direct_oos_probability": direct_oos_probability_diagnostics, "local_transition_negative_hit": transition_state_diagnostics, "aegmm_transition_negative_hit": aegmm_transition_diagnostics, "aegmm_component_transition_negative_hit": aegmm_component_transition_diagnostics, "oof_input_contract_routing": routing_diagnostics if routed_arm is not None else [], "phase_static_coverage": phase_static_coverage, "phase_state": phase_diagnostics, "residual_state": residual_state_diagnostics, "semantic_residual_state": semantic_state_diagnostics, "main_models": observed})
        print(json.dumps({"event": "fold_complete", "month": str(month), "train_rows": len(train), "oos_rows": len(test)}), flush=True)
        del train, test, extras_train, extras_test
        gc.collect()
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(args.output / "oos_scorecard_by_month.csv", index=False)
    if scorecard.empty:
        pd.DataFrame(diagnostics).to_json(
            args.output / "head_diagnostics.json", orient="records", indent=2
        )
        raise RuntimeError(
            "No OOS folds were evaluated. Increase --max-train-rows-per-month "
            "before the top-20 candidate filter or provide a wider historical handoff."
        )
    aggregate = scorecard.groupby("arm", observed=True).mean(numeric_only=True).reset_index()
    if not aggregate.empty and aggregate["arm"].eq("M0_current_meta_head").any():
        baseline = aggregate.loc[aggregate["arm"].eq("M0_current_meta_head")].iloc[0]
        for col in ("mean_ev_after_1pct", "worst_week_ev", "worst_month_ev", "clean_exec_precision", "full_path_bad_mae_rate", "timeout_rate"):
            aggregate[f"delta_{col}_vs_M0"] = aggregate[col] - float(baseline[col])
    if not aggregate.empty and aggregate["arm"].eq("M1_good_trade_feature").any():
        baseline = aggregate.loc[aggregate["arm"].eq("M1_good_trade_feature")].iloc[0]
        for col in (
            "mean_ev_after_1pct",
            "worst_week_ev",
            "worst_month_ev",
            "clean_exec_precision",
            "negative_executable_ev_rate",
            "first_touch_bad_mae_rate",
            "full_path_bad_mae_rate",
            "timeout_rate",
            "dirty_positive_rate",
            "signed_hit_surprise_ac",
            "negative_hit_surprise_ac",
            "positive_hit_surprise_ac",
        ):
            if col in aggregate:
                aggregate[f"delta_{col}_vs_M1"] = aggregate[col] - float(baseline[col])
    aggregate.to_csv(args.output / "oos_scorecard_aggregate.csv", index=False)
    pd.concat(detail, ignore_index=True).to_csv(args.output / "oos_side_archetype_breakdown.csv", index=False)
    if selected_ledgers:
        pd.concat(selected_ledgers, ignore_index=True, copy=False).to_parquet(
            args.output / "oos_selected_ledger.parquet",
            index=False,
        )
    pd.DataFrame(diagnostics).to_json(args.output / "head_diagnostics.json", orient="records", indent=2)
    _write_json(args.output / "manifest.json", {"schema": "meta_residual_head_feature_ablation_v3", "months": [str(m) for m in months], "train_months": args.train_months, "arms": requested_arms, "aux_oof_blocks": int(args.aux_oof_blocks), "source_mode": args.source_mode, "full_feature_ledgers": [str(path) for path in (args.full_feature_ledger or DEFAULT_FULL_FEATURE_LEDGERS)], "feature_contract": str(args.feature_contract), "feature_store_id": str(args.feature_store_id), "base_feature_count_by_side": {side: len(values) for side, values in base_features_by_side.items()}, "base_features_by_side": base_features_by_side, "static_joint_complete_coverage_by_month_side": static_coverage, "outcome_joined_rows_by_month": coverage, "contract": "all auxiliary inputs are chronological OOF predictions or train-only priors; every arm is a fitted side-specific meta head on the full residual_state_mda95 contract, never a rank multiplier"})


if __name__ == "__main__":
    main()
