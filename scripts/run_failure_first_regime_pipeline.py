#!/usr/bin/env python3
"""Run the causal, failure-first regime discovery pipeline.

Descriptive health, membership, episode and event-window artifacts are always
published.  Taxonomy and detector fitting fail closed when the predeclared
support gate is not met.  The later forward cohort is audited separately and
is never mixed into discovery, feature selection, HPO, or taxonomy fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.unsupervised_regime_learning.failure_first_health import (  # noqa: E402
    FailureHealthConfig,
    build_causal_decision_health,
    group_failure_bins_into_episodes,
)
from extreme_price_movements.unsupervised_regime_learning.failure_first_pipeline import (  # noqa: E402
    DEFAULT_MARKET_FEATURES,
    DEFAULT_MODEL_HEALTH_FEATURES,
    FailureFirstSufficiencyConfig,
    attach_episode_window_coverage,
    build_failure_episode_profiles,
    build_hourly_failure_state_targets,
    build_hourly_observable_state,
    choose_taxonomy_fit_cutoff,
    evaluate_failure_detector_label_sufficiency,
    evaluate_failure_first_sufficiency,
    evaluate_taxonomy_bootstrap_stability,
    extract_failure_episode_outcomes,
    extract_failure_episode_windows,
    fit_frozen_failure_taxonomy,
    frame_fingerprint,
    prepare_failure_first_sources,
)
from extreme_price_movements.unsupervised_regime_learning.failure_first_detector import (  # noqa: E402
    FailureFirstDetectorConfig,
    add_causal_bocpd_features,
    chronological_failure_first_oof,
)


DEFAULT_LEDGER = Path(
    "data_perp/artifacts/"
    "execution_ev_context_clean_recent_mapping_forward_july19_20260726_v1/"
    "mapped_oof.parquet"
)
DEFAULT_STATE = Path(
    "data_perp/artifacts/"
    "raw_market_state_backward_recurrence_20260726_v1/"
    "weekly_raw_state_diagnostic_rows.parquet"
)
DEFAULT_RICH = Path(
    "data_perp/artifacts/"
    "execution_ev_repaired_heads_representation_handoff_20260726_v7/"
    "joined.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/failure_first_regime_pipeline_20260726_v2"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    time_columns = [
        name
        for name in ("execution_decision_utc", "__ts__")
        if name in frame.columns
    ]
    if time_columns:
        timestamp = pd.to_datetime(
            frame[time_columns[0]], utc=True, errors="coerce"
        )
        start = timestamp.min()
        end = timestamp.max()
    else:
        start = end = pd.NaT
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "start_utc": start,
        "end_utc": end,
    }


def _forward_coverage(
    ledger: pd.DataFrame,
    state_source: pd.DataFrame,
    *,
    active_score_valid_flag: str = (
        "causal_recent_side_isotonic_ev__is_oof"
    ),
) -> dict[str, Any]:
    flag = "causal_recent_side_isotonic_ev__is_forward_oos"
    if flag not in ledger:
        return {
            "status": "MISSING_FORWARD_PROVENANCE_FLAG",
            "forward_rows": 0,
            "raw_h0_matched_rows": 0,
            "detector_scoring_allowed": False,
        }
    forward = ledger.loc[ledger[flag].fillna(False).astype(bool)].copy()
    matched = (
        int(forward["candidate_id"].isin(state_source["candidate_id"]).sum())
        if len(forward)
        else 0
    )
    raw_fields = [
        name
        for name in state_source.columns
        if name.startswith("mkt_state__") and name.endswith("__h0")
    ]
    coverage = float(matched / len(forward)) if len(forward) else 0.0
    allowed = bool(len(forward) and matched == len(forward) and raw_fields)
    retired = bool(
        active_score_valid_flag
        != "causal_recent_side_isotonic_ev__is_oof"
        and active_score_valid_flag in forward
        and forward[active_score_valid_flag].fillna(False).astype(bool).all()
    )
    return {
        "status": (
            "RETIRED_RESOLVED_FORWARD_OOS_INCLUDED"
            if retired and allowed
            else "PASS"
            if allowed
            else "MISSING_CAUSAL_RAW_H0_COVERAGE"
        ),
        "forward_rows": int(len(forward)),
        "raw_h0_matched_rows": matched,
        "raw_h0_match_rate": coverage,
        "raw_h0_feature_count": int(len(raw_fields)),
        "detector_scoring_allowed": allowed,
        "policy": (
            "This already-opened resolved forward cohort is retired into "
            "strict model-OOS history for fitting a later detector. It is "
            "forbidden from evaluating any detector fitted on this history."
            if retired
            else
            "Forward rows are audit-only and cannot enter discovery, taxonomy, "
            "feature selection, HPO, or detector fitting."
        ),
    }


def _binary_metrics(
    frame: pd.DataFrame,
    *,
    target_col: str,
    probability_col: str,
) -> dict[str, Any]:
    local = frame.loc[
        frame[target_col].notna() & frame[probability_col].notna()
    ].copy()
    if local.empty:
        return {"rows": 0}
    target = pd.to_numeric(local[target_col], errors="coerce").astype(int)
    probability = pd.to_numeric(
        local[probability_col], errors="coerce"
    ).clip(1e-6, 1.0 - 1e-6)
    return {
        "rows": int(len(local)),
        "positive_rows": int(target.sum()),
        "positive_rate": float(target.mean()),
        "roc_auc": float(roc_auc_score(target, probability))
        if target.nunique() > 1
        else None,
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
    }


def _metric_slug(value: object) -> str:
    text = re.sub(
        r"[^a-zA-Z0-9]+", "_", str(value).strip()
    ).strip("_")
    return text.casefold() or "missing"


def _multiclass_metrics(
    frame: pd.DataFrame,
    *,
    target_col: str,
    predicted_col: str,
    probability_prefix: str,
) -> dict[str, Any]:
    local = frame.loc[
        frame[target_col].notna() & frame[predicted_col].notna()
    ].copy()
    if local.empty:
        return {"rows": 0}
    target = local[target_col].astype(str)
    predicted = local[predicted_col].astype(str)
    classes = sorted(target.unique())
    support = target.value_counts().reindex(classes, fill_value=0)
    recall = recall_score(
        target,
        predicted,
        labels=classes,
        average=None,
        zero_division=0,
    )
    report: dict[str, Any] = {
        "rows": int(len(local)),
        "classes": classes,
        "class_support": {
            name: int(value) for name, value in support.items()
        },
        "class_recall": {
            name: float(value)
            for name, value in zip(classes, recall, strict=True)
        },
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, predicted)
        ),
        "macro_f1": float(
            f1_score(
                target,
                predicted,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
    }
    probability_columns = [
        f"{probability_prefix}__{_metric_slug(name)}" for name in classes
    ]
    if all(name in local for name in probability_columns):
        probability = (
            local.loc[:, probability_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        valid = np.isfinite(probability).all(axis=1)
        if valid.any():
            probability = np.clip(probability[valid], 1e-8, 1.0)
            probability /= probability.sum(axis=1, keepdims=True)
            encoded = pd.Categorical(
                target.loc[valid], categories=classes
            ).codes
            one_hot = np.eye(len(classes), dtype=np.float64)[encoded]
            report["multiclass_log_loss"] = float(
                log_loss(
                    target.loc[valid],
                    probability,
                    labels=classes,
                )
            )
            report["multiclass_brier"] = float(
                np.mean(np.sum((probability - one_hot) ** 2, axis=1))
            )
    return report


def _detector_classification_report(predictions: pd.DataFrame) -> dict[str, Any]:
    timestamp = pd.to_datetime(
        predictions["execution_decision_utc"], utc=True, errors="raise"
    )
    latest_month = timestamp.dt.to_period("M").max()
    latest = predictions.loc[timestamp.dt.to_period("M").eq(latest_month)]

    def report(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "transition_within_3h": _binary_metrics(
                frame,
                target_col="target__transition_within_3h",
                probability_col="p_transition_within_3h",
            ),
            "active_transition": _binary_metrics(
                frame,
                target_col="target__active_transition",
                probability_col="p_active_transition",
            ),
            "current_failure_state": _multiclass_metrics(
                frame,
                target_col="target__current_failure_state",
                predicted_col="predicted_current_failure_state",
                probability_prefix="p_current_state",
            ),
            "destination_state_3h": _multiclass_metrics(
                frame,
                target_col="target__destination_state_3h",
                predicted_col="predicted_destination_state_3h",
                probability_prefix="p_destination",
            ),
        }

    return {
        "aggregate": report(predictions),
        "latest_month": str(latest_month),
        "latest_month_metrics": report(latest),
    }


def _one_global_top_decile(
    frame: pd.DataFrame,
    *,
    score_col: str,
) -> dict[str, Any]:
    local = frame.loc[
        pd.to_numeric(frame[score_col], errors="coerce").notna()
        & pd.to_numeric(frame["execution_net_ev_12h"], errors="coerce").notna()
    ].copy()
    if local.empty:
        return {"eligible_rows": 0, "selected_rows": 0}
    local["__score__"] = pd.to_numeric(local[score_col], errors="coerce")
    local["__candidate__"] = local["candidate_id"].astype(str)
    selected_rows = int(np.ceil(0.10 * len(local)))
    selected = local.sort_values(
        ["__score__", "__candidate__"],
        ascending=[False, True],
        kind="stable",
    ).head(selected_rows)
    return {
        "eligible_rows": int(len(local)),
        "selected_rows": selected_rows,
        "mean_net_ev": float(selected["execution_net_ev_12h"].mean()),
        "mean_net_ev_bps": float(
            10_000.0 * selected["execution_net_ev_12h"].mean()
        ),
        "positive_net_rate": float(
            selected["execution_net_ev_12h"].gt(0.0).mean()
        ),
    }


def _detector_economics_report(
    ledger: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    eligibility_flag: str = "causal_recent_side_isotonic_ev__is_oof",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hourly = predictions.loc[
        :,
        [
            "execution_decision_utc",
            "evaluation_origin",
            "p_failure_destination_3h",
        ],
    ].drop_duplicates(["execution_decision_utc", "evaluation_origin"])
    strict = ledger.loc[
        ledger[eligibility_flag]
        .fillna(False)
        .astype(bool)
    ].copy()
    covered = strict.merge(
        hourly,
        on=["execution_decision_utc", "evaluation_origin"],
        how="inner",
        validate="many_to_one",
    )
    covered["failure_trust_adjusted_score"] = (
        covered["causal_recent_side_isotonic_ev"]
        - covered["p_failure_destination_3h"]
        * covered["causal_recent_side_isotonic_ev"].abs()
    )
    timestamp = pd.to_datetime(
        covered["execution_decision_utc"], utc=True, errors="raise"
    )
    covered["evaluation_month"] = timestamp.dt.to_period("M").astype(str)
    scopes: dict[str, pd.DataFrame] = {"aggregate": covered}
    if len(covered):
        latest = covered["evaluation_month"].max()
        scopes[f"latest_month::{latest}"] = covered.loc[
            covered["evaluation_month"].eq(latest)
        ]
    report: dict[str, Any] = {}
    for name, local in scopes.items():
        report[name] = {
            "mapped_score": _one_global_top_decile(
                local, score_col="causal_recent_side_isotonic_ev"
            ),
            "failure_trust_adjusted_score": _one_global_top_decile(
                local, score_col="failure_trust_adjusted_score"
            ),
            "selection_contract": (
                "one pooled global top 10 percent across timestamps and sides"
            ),
        }
    return covered, report


def _detector_promotion_gate(
    classification: dict[str, Any],
    economics: dict[str, Any],
    *,
    minimum_rows: int,
    minimum_positive_events: int,
) -> dict[str, Any]:
    latest_classification = classification["latest_month_metrics"]
    latest_economics_key = next(
        name for name in economics if name.startswith("latest_month::")
    )
    latest_economics = economics[latest_economics_key]
    aggregate_economics = economics["aggregate"]
    criteria = {
        "latest_transition_rows": {
            "observed": latest_classification["transition_within_3h"].get(
                "rows", 0
            ),
            "required": int(minimum_rows),
        },
        "latest_transition_positives": {
            "observed": latest_classification["transition_within_3h"].get(
                "positive_rows", 0
            ),
            "required": int(minimum_positive_events),
        },
        "latest_active_positives": {
            "observed": latest_classification["active_transition"].get(
                "positive_rows", 0
            ),
            "required": int(minimum_positive_events),
        },
        "aggregate_transition_auc": {
            "observed": classification["aggregate"]["transition_within_3h"].get(
                "roc_auc"
            ),
            "required": 0.50,
        },
        "latest_transition_auc": {
            "observed": latest_classification["transition_within_3h"].get(
                "roc_auc"
            ),
            "required": 0.50,
        },
        "aggregate_adjusted_net_bps": {
            "observed": aggregate_economics[
                "failure_trust_adjusted_score"
            ].get("mean_net_ev_bps"),
            "required": 0.0,
        },
        "latest_adjusted_net_bps": {
            "observed": latest_economics[
                "failure_trust_adjusted_score"
            ].get("mean_net_ev_bps"),
            "required": 0.0,
        },
        "aggregate_incremental_bps": {
            "observed": (
                aggregate_economics["failure_trust_adjusted_score"].get(
                    "mean_net_ev_bps", np.nan
                )
                - aggregate_economics["mapped_score"].get(
                    "mean_net_ev_bps", np.nan
                )
            ),
            "required": 0.0,
        },
        "latest_incremental_bps": {
            "observed": (
                latest_economics["failure_trust_adjusted_score"].get(
                    "mean_net_ev_bps", np.nan
                )
                - latest_economics["mapped_score"].get(
                    "mean_net_ev_bps", np.nan
                )
            ),
            "required": 0.0,
        },
    }
    for name, item in criteria.items():
        observed = item["observed"]
        item["pass"] = bool(
            observed is not None
            and np.isfinite(observed)
            and float(observed) >= float(item["required"])
        )
    passed = all(bool(item["pass"]) for item in criteria.values())
    return {
        "detector_promotion_allowed": passed,
        "status": "PASS" if passed else "REJECT",
        "latest_economics_scope": latest_economics_key,
        "criteria": criteria,
        "policy": (
            "Training completion is not promotion. Latest coverage, transition "
            "discrimination, positive economics and incremental economics must "
            "all recur."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--state-source", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--rich-context", type=Path, default=DEFAULT_RICH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--score-valid-flag",
        default="causal_recent_side_isotonic_ev__is_oof",
        help=(
            "Explicit strict model-OOS provenance flag used for discovery and "
            "training. OOF remains the default; retired forward OOS requires "
            "a separately materialized combined flag."
        ),
    )
    parser.add_argument(
        "--market-feature-cols",
        default="",
        help=(
            "Optional comma-separated frozen H0 market feature contract. "
            "Use this for cross-era transfer; every named field must exist."
        ),
    )
    parser.add_argument("--minimum-cutoff-rows", type=int, default=4_000)
    parser.add_argument("--minimum-admitted-rows", type=int, default=20)
    parser.add_argument("--minimum-resolved-bins", type=int, default=20)
    parser.add_argument("--minimum-failure-episodes", type=int, default=40)
    parser.add_argument(
        "--minimum-complete-window-episodes", type=int, default=40
    )
    parser.add_argument("--minimum-failure-bins", type=int, default=40)
    parser.add_argument("--minimum-span-days", type=int, default=180)
    parser.add_argument("--minimum-observed-days", type=int, default=180)
    parser.add_argument("--maximum-calendar-gap-days", type=int, default=21)
    parser.add_argument("--minimum-profile-features", type=int, default=10)
    parser.add_argument("--minimum-detector-rows", type=int, default=1_000)
    parser.add_argument("--minimum-transitions", type=int, default=50)
    parser.add_argument(
        "--primary-taxonomy-method",
        choices=("gmm", "kmeans"),
        default="gmm",
    )
    parser.add_argument("--minimum-taxonomy-clusters", type=int, default=5)
    parser.add_argument("--maximum-taxonomy-clusters", type=int, default=8)
    parser.add_argument("--minimum-cluster-episodes", type=int, default=5)
    parser.add_argument(
        "--taxonomy-stability-repetitions", type=int, default=100
    )
    parser.add_argument(
        "--minimum-taxonomy-median-ari", type=float, default=0.80
    )
    parser.add_argument(
        "--minimum-taxonomy-q10-ari", type=float, default=0.50
    )
    parser.add_argument("--detector-eval-hours", type=int, default=168)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    ledger = pd.read_parquet(args.ledger)
    state_source = pd.read_parquet(args.state_source)
    rich = (
        pd.read_parquet(args.rich_context)
        if args.rich_context is not None and args.rich_context.exists()
        else None
    )
    score_valid_flag = str(
        getattr(
            args,
            "score_valid_flag",
            "causal_recent_side_isotonic_ev__is_oof",
        )
    )
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to mix failure-first artifacts in non-empty {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    source_manifest = {
        "contract": "failure_first_regime_pipeline_v1",
        "ledger": _source_manifest(Path(args.ledger), ledger),
        "state_source": _source_manifest(Path(args.state_source), state_source),
        "rich_context": _source_manifest(Path(args.rich_context), rich)
        if rich is not None
        else None,
        "forward": _forward_coverage(
            ledger,
            state_source,
            active_score_valid_flag=score_valid_flag,
        ),
    }
    _write_json(output / "source_manifest.json", source_manifest)

    available_h0 = sorted(
        name
        for name in state_source.columns
        if name.startswith("mkt_state__") and name.endswith("__h0")
    )
    frozen_market = [
        name.strip()
        for name in str(getattr(args, "market_feature_cols", "")).split(",")
        if name.strip()
    ]
    if frozen_market:
        missing_market = sorted(set(frozen_market).difference(available_h0))
        if missing_market:
            raise ValueError(
                "frozen market feature contract is missing: "
                + ", ".join(missing_market)
            )
        requested_market = list(dict.fromkeys(frozen_market))
    else:
        requested_market = [
            name for name in DEFAULT_MARKET_FEATURES if name in available_h0
        ]
        requested_market.extend(
            name
            for name in available_h0
            if name not in requested_market
        )
        requested_market = requested_market[:20]
    joined, observable_features, join_audit = prepare_failure_first_sources(
        ledger,
        state_source,
        rich_context=rich,
        requested_market_features=requested_market,
        requested_health_features=DEFAULT_MODEL_HEALTH_FEATURES,
        score_valid_flag=score_valid_flag,
    )
    health_config = FailureHealthConfig(
        minimum_cutoff_rows=int(args.minimum_cutoff_rows),
        minimum_admitted_rows=int(args.minimum_admitted_rows),
        minimum_resolved_bins=int(args.minimum_resolved_bins),
        score_oof_col=score_valid_flag,
    )
    health, membership = build_causal_decision_health(
        joined, config=health_config
    )
    episodes, episode_membership = group_failure_bins_into_episodes(
        health, membership, config=health_config
    )
    hourly_state, state_features = build_hourly_observable_state(
        joined, feature_columns=observable_features
    )
    windows = extract_failure_episode_windows(
        hourly_state,
        episodes,
        state_feature_columns=state_features,
    )
    outcomes = extract_failure_episode_outcomes(joined, windows)
    episodes = attach_episode_window_coverage(episodes, windows)
    profiles, profile_features = build_failure_episode_profiles(
        windows, state_feature_columns=state_features
    )
    incomplete_episodes = episodes.loc[
        ~episodes["complete_window_coverage"].fillna(False).astype(bool)
    ].copy()

    sufficiency_config = FailureFirstSufficiencyConfig(
        minimum_failure_episodes=int(args.minimum_failure_episodes),
        minimum_complete_window_episodes=int(
            args.minimum_complete_window_episodes
        ),
        minimum_failure_bins=int(args.minimum_failure_bins),
        minimum_span_days=int(args.minimum_span_days),
        minimum_observed_days=int(
            getattr(args, "minimum_observed_days", 180)
        ),
        maximum_calendar_gap_days=int(
            getattr(args, "maximum_calendar_gap_days", 21)
        ),
        minimum_profile_features=int(args.minimum_profile_features),
        minimum_detector_rows=int(
            getattr(args, "minimum_detector_rows", 1_000)
        ),
        minimum_transitions=int(getattr(args, "minimum_transitions", 50)),
    )
    gate = evaluate_failure_first_sufficiency(
        health,
        episodes,
        windows,
        profile_feature_count=len(profile_features),
        config=sufficiency_config,
    )
    gate["join_audit"] = join_audit
    gate["detector_promotion_allowed"] = False

    artifacts: dict[str, pd.DataFrame] = {
        "decision_health_6h.parquet": health,
        "candidate_membership_expost.parquet": membership,
        "failure_episodes.parquet": episodes,
        "episode_row_membership.parquet": episode_membership,
        "hourly_observable_state.parquet": hourly_state,
        "episode_window_state.parquet": windows,
        "episode_window_outcomes.parquet": outcomes,
        "failure_episode_profiles_expost.parquet": profiles,
        "excluded_incomplete_episodes.parquet": incomplete_episodes,
    }
    model_artifacts: dict[str, dict[str, Any]] = {}
    if gate["taxonomy_training_allowed"]:
        complete_ids = set(
            episodes.loc[
                episodes["complete_window_coverage"].fillna(False).astype(bool),
                "episode_id",
            ].astype(str)
        )
        taxonomy_episodes = episodes.loc[
            episodes["episode_id"].astype(str).isin(complete_ids)
        ].copy()
        taxonomy_profiles = profiles.loc[
            profiles["episode_id"].astype(str).isin(complete_ids)
        ].copy()
        cutoff = choose_taxonomy_fit_cutoff(
            taxonomy_episodes,
            minimum_failure_episodes=int(args.minimum_failure_episodes),
        )
        taxonomy_results: dict[str, Any] = {}
        taxonomy_errors: dict[str, str] = {}
        taxonomy_stability: dict[str, dict[str, Any]] = {}
        for method in ("gmm", "kmeans"):
            try:
                bundle, assignments, selection, summary = (
                    fit_frozen_failure_taxonomy(
                        taxonomy_profiles,
                        taxonomy_episodes,
                        profile_columns=profile_features,
                        fit_cutoff_utc=cutoff,
                        method=method,
                        min_clusters=int(
                            getattr(args, "minimum_taxonomy_clusters", 5)
                        ),
                        max_clusters=int(
                            getattr(args, "maximum_taxonomy_clusters", 8)
                        ),
                        minimum_cluster_episodes=int(
                            getattr(args, "minimum_cluster_episodes", 5)
                        ),
                    )
                )
            except ValueError as error:
                taxonomy_errors[method] = str(error)
                continue
            taxonomy_results[method] = (bundle, assignments)
            taxonomy_stability[method] = (
                evaluate_taxonomy_bootstrap_stability(
                    bundle,
                    taxonomy_profiles,
                    repetitions=int(
                        getattr(
                            args, "taxonomy_stability_repetitions", 100
                        )
                    ),
                    minimum_median_ari=float(
                        getattr(args, "minimum_taxonomy_median_ari", 0.80)
                    ),
                    minimum_q10_ari=float(
                        getattr(args, "minimum_taxonomy_q10_ari", 0.50)
                    ),
                )
            )
            artifacts[f"taxonomy_{method}_assignments_expost.parquet"] = (
                assignments
            )
            artifacts[f"taxonomy_{method}_selection_expost.parquet"] = selection
            artifacts[f"taxonomy_{method}_summary_expost.parquet"] = summary
            model_path = output / f"taxonomy_{method}_frozen.joblib"
            joblib.dump(bundle, model_path)
            model_artifacts[model_path.name] = {
                "sha256": _sha256(model_path),
                "method": method,
                "fit_cutoff_utc": cutoff,
                "train_episodes": int(len(bundle.train_episode_ids)),
                "selected_clusters": int(bundle.selected_clusters),
            }
        primary_method = str(
            getattr(args, "primary_taxonomy_method", "gmm")
        )
        gate["taxonomy_comparison"] = {
            "fit_cutoff_utc": cutoff,
            "completed_methods": sorted(taxonomy_results),
            "errors": taxonomy_errors,
            "primary_method": primary_method,
            "bootstrap_stability": taxonomy_stability,
        }
        if primary_method not in taxonomy_results:
            gate["taxonomy_training_allowed"] = False
            gate["detector_training_allowed"] = False
            gate["taxonomy_status"] = "FAILED_PRIMARY_TAXONOMY"
            gate["detector_status"] = "SKIPPED_PRIMARY_TAXONOMY_FAILED"
        else:
            gate["taxonomy_status"] = "FROZEN_TAXONOMY_COMPLETE"
            primary_bundle, primary_assignments = taxonomy_results[
                primary_method
            ]
            gate["taxonomy_stability_pass"] = bool(
                taxonomy_stability[primary_method]["pass"]
            )
            targets = build_hourly_failure_state_targets(
                health,
                episode_membership,
                primary_assignments,
                health_bin_hours=int(health_config.health_bin_hours),
                horizon_hours=3,
            )
            artifacts["failure_state_transition_targets.parquet"] = targets
            label_gate = evaluate_failure_detector_label_sufficiency(
                targets, config=sufficiency_config, horizon_hours=3
            )
            gate["detector_label_gate"] = label_gate
            gate["detector_training_allowed"] = bool(
                label_gate["detector_training_allowed"]
            )
            if not gate["detector_training_allowed"]:
                gate["detector_status"] = "SKIPPED_INSUFFICIENT_CLASS_SUPPORT"
            else:
                detector_panel = targets.merge(
                    hourly_state.drop(columns=["side_name"]),
                    on="execution_decision_utc",
                    how="inner",
                    validate="one_to_one",
                )
                bocpd_signals: list[str] = []
                for token in (
                    "volatility_of_volatility",
                    "market_breadth_4h",
                    "base_margin_to_cutoff_z",
                    "catboost_entropy",
                ):
                    match = next(
                        (name for name in state_features if token in name),
                        None,
                    )
                    if match is not None and match not in bocpd_signals:
                        bocpd_signals.append(match)
                if not bocpd_signals:
                    bocpd_signals = state_features[:3]
                detector_panel = add_causal_bocpd_features(
                    detector_panel,
                    signal_columns=bocpd_signals,
                    timestamp_col="execution_decision_utc",
                    group_columns=("side_name", "evaluation_origin"),
                )
                detector_features = [
                    *state_features,
                    "failure_bocpd_probability_max",
                    "failure_bocpd_break_count",
                    "failure_bocpd_break_intensity",
                ]
                if len(detector_features) > int(
                    sufficiency_config.maximum_detector_features
                ):
                    raise ValueError(
                        "detector feature contract exceeds the compact "
                        f"{sufficiency_config.maximum_detector_features}-field gate"
                    )
                first_eval = pd.Timestamp(cutoff).ceil("h")
                detector_config = FailureFirstDetectorConfig(
                    first_eval_time=first_eval.isoformat(),
                    eval_hours=int(
                        getattr(args, "detector_eval_hours", 168)
                    ),
                    min_train_rows=int(
                        sufficiency_config.minimum_detector_rows
                    ),
                    min_class_rows=int(
                        sufficiency_config.minimum_transitions
                    ),
                    max_features=int(
                        sufficiency_config.maximum_detector_features
                    ),
                    failure_state_labels=tuple(
                        sorted(
                            primary_assignments[
                                "expost__failure_taxonomy_label"
                            ]
                            .astype(str)
                            .unique()
                        )
                    ),
                )
                predictions, bundles = chronological_failure_first_oof(
                    detector_panel,
                    feature_columns=detector_features,
                    config=detector_config,
                )
                fully_fitted = all(
                    head.model is not None
                    for bundle in bundles
                    for head in (
                        bundle.transition_head,
                        bundle.active_head,
                        bundle.current_state_head,
                        bundle.destination_head,
                    )
                )
                if not fully_fitted:
                    gate["detector_training_allowed"] = False
                    gate["detector_status"] = (
                        "SKIPPED_CONSTANT_HEAD_IN_AT_LEAST_ONE_FOLD"
                    )
                else:
                    gate["detector_status"] = "CHRONOLOGICAL_OOF_COMPLETE"
                    artifacts["failure_detector_oof.parquet"] = predictions
                    artifacts["failure_detector_panel.parquet"] = detector_panel
                    classification = _detector_classification_report(
                        predictions
                    )
                    covered, economics = _detector_economics_report(
                        joined,
                        predictions,
                        eligibility_flag=score_valid_flag,
                    )
                    artifacts[
                        "failure_detector_candidate_overlay_oof.parquet"
                    ] = covered
                    _write_json(
                        output / "detector_classification_metrics.json",
                        classification,
                    )
                    _write_json(
                        output / "detector_global_top10_economics.json",
                        economics,
                    )
                    promotion_gate = _detector_promotion_gate(
                        classification,
                        economics,
                        minimum_rows=int(
                            sufficiency_config.minimum_detector_rows
                        ),
                        minimum_positive_events=int(
                            sufficiency_config.minimum_transitions
                        ),
                    )
                    stability_pass = bool(
                        gate.get("taxonomy_stability_pass", False)
                    )
                    promotion_gate["criteria"][
                        "taxonomy_bootstrap_stability"
                    ] = {
                        "observed": stability_pass,
                        "required": True,
                        "pass": stability_pass,
                    }
                    if not stability_pass:
                        promotion_gate["detector_promotion_allowed"] = False
                        promotion_gate["status"] = "REJECT"
                    gate["detector_promotion_gate"] = promotion_gate
                    gate["detector_promotion_allowed"] = bool(
                        promotion_gate["detector_promotion_allowed"]
                    )
                    gate["detector_status"] = (
                        "CHRONOLOGICAL_OOF_COMPLETE_PROMOTION_PASS"
                        if gate["detector_promotion_allowed"]
                        else "CHRONOLOGICAL_OOF_COMPLETE_PROMOTION_REJECT"
                    )
                    final_bundle_path = (
                        output / "failure_detector_latest_oof_fold.joblib"
                    )
                    joblib.dump(bundles[-1], final_bundle_path)
                    model_artifacts[final_bundle_path.name] = {
                        "sha256": _sha256(final_bundle_path),
                        "train_end_exclusive": bundles[-1].train_end_exclusive,
                        "train_rows": bundles[-1].train_rows,
                        "feature_count": len(detector_features),
                        "model_family": "CatBoostClassifier",
                    }
    else:
        gate["taxonomy_status"] = "SKIPPED_INSUFFICIENT_SUPPORT"
        gate["detector_status"] = "SKIPPED_INSUFFICIENT_SUPPORT"

    if gate["detector_promotion_allowed"]:
        gate["overall_status"] = "PROMOTION_GATE_PASS"
    elif gate["detector_training_allowed"]:
        gate["overall_status"] = "RESEARCH_COMPLETE_PROMOTION_REJECT"
    elif gate["taxonomy_training_allowed"]:
        gate["overall_status"] = "TAXONOMY_COMPLETE_DETECTOR_REJECT"
    else:
        gate["overall_status"] = str(gate["status"])
    _write_json(output / "sufficiency_gate.json", gate)
    artifact_manifest: dict[str, Any] = {}
    for name, frame in artifacts.items():
        path = output / name
        frame.to_parquet(path, index=False)
        artifact_manifest[name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "sha256": _sha256(path),
            "content_fingerprint": frame_fingerprint(
                frame, list(frame.columns)[: min(8, len(frame.columns))]
            ),
        }
    manifest = {
        "contract": "failure_first_regime_pipeline_v1",
        "status": gate["overall_status"],
        "health_config": asdict(health_config),
        "sufficiency_config": asdict(sufficiency_config),
        "observable_features": observable_features,
        "hourly_state_features": state_features,
        "artifacts": artifact_manifest,
        "model_artifacts": model_artifacts,
        "taxonomy_training_allowed": gate["taxonomy_training_allowed"],
        "detector_training_allowed": gate["detector_training_allowed"],
        "detector_promotion_allowed": gate["detector_promotion_allowed"],
        "score_valid_flag": score_valid_flag,
        "no_forward_rows_used": (
            score_valid_flag
            == "causal_recent_side_isotonic_ev__is_oof"
        ),
        "only_explicit_score_valid_rows_used": True,
        "no_h1_h3_h6_h12_fields_used": not any(
            name.endswith(("__h1", "__h3", "__h6", "__h12"))
            for name in observable_features
        ),
    }
    _write_json(output / "manifest.json", manifest)
    return {
        "output_dir": output,
        "status": gate["overall_status"],
        "failure_bins": int(health["model_failure_bin"].sum()),
        "episodes": int(len(episodes)),
        "taxonomy_training_allowed": bool(
            gate["taxonomy_training_allowed"]
        ),
        "detector_training_allowed": bool(gate["detector_training_allowed"]),
        "detector_promotion_allowed": bool(
            gate["detector_promotion_allowed"]
        ),
    }


def main() -> None:
    result = run(_parser().parse_args())
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
