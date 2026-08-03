#!/usr/bin/env python3
"""Materialize a research-only adaptive transition-pattern catalogue.

The input is the provenance-bound regime episode ledger.  This runner keeps
decision-time sequence fields separate from ex-post phase labels and produces
event-grouped, two-sided-purged OOF research diagnostics.  It is deliberately
not a policy gate and does not feed outputs into the trading stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.transition_pattern_catalogue import (
    TransitionPatternConfig,
    causal_predictor_columns,
    materialize_adaptive_transition_phases,
    sample_stable_vs_transition,
    summarize_event_preonset_sequences,
)
from extreme_price_movements.transition_pattern_models import (
    BayesianRuleListChallenger,
    TransitionClassifierAdapter,
    TransitionMorphologyConfig,
    TransitionMorphologyEmbedder,
    validate_preonset_sequence_columns,
)


DEFAULT_LEDGER = ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/transition_pattern_catalogue_20260730_v1"
SCHEMA = "transition_pattern_catalogue_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_ledger(ledger_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hourly_path = ledger_dir / "hourly_state_calendar.parquet"
    events_path = ledger_dir / "transition_episode_ledger.parquet"
    manifest_path = ledger_dir / "manifest.json"
    for path in (hourly_path, events_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    hourly = pd.read_parquet(hourly_path)
    events = pd.read_parquet(events_path)
    # The provenance-bound calendar deliberately renames the source segment to
    # distinguish it from its own calendar coverage segment.  The catalogue
    # APIs use the canonical short name internally, while retaining both source
    # columns in the original ledger artifact.
    if "segment_id" not in hourly and "source_segment_id" in hourly:
        hourly = hourly.copy()
        hourly["segment_id"] = hourly["source_segment_id"]
    if "segment_id" not in events and "source_segment_id" in events:
        events = events.copy()
        events["segment_id"] = events["source_segment_id"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return hourly, events, manifest


def _event_grouped_purged_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    purge_hours: int,
    group_column: str = "event_id",
    anchor_column: str = "anchor_source_utc",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Group events and remove train anchors within a two-sided time embargo."""

    required = {group_column, anchor_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"fold frame lacks {missing}")
    work = frame.reset_index(drop=True).copy()
    work[anchor_column] = pd.to_datetime(work[anchor_column], utc=True, errors="coerce")
    if work[anchor_column].isna().any() or work[group_column].isna().any():
        raise ValueError("event-grouped fold identity/anchor must be present")
    group = work[group_column].astype(str)
    groups = group.nunique()
    folds = min(int(n_splits), int(groups))
    if folds < 2:
        return [], pd.DataFrame(columns=["fold", "row", "role", "event_id", "anchor_source_utc"])
    splitter = GroupKFold(n_splits=folds)
    timestamp = work[anchor_column].to_numpy(dtype="datetime64[ns]")
    plans: list[tuple[np.ndarray, np.ndarray]] = []
    rows: list[dict[str, Any]] = []
    delta = np.timedelta64(int(purge_hours), "h")
    for fold, (train_index, validation_index) in enumerate(splitter.split(work, groups=group)):
        validation_time = timestamp[validation_index]
        # Candidates inside any validation event's full purge band are removed.
        distance = np.abs(timestamp[train_index, None] - validation_time[None, :])
        keep = (distance > delta).all(axis=1)
        purged_train = train_index[keep]
        train_groups = set(group.iloc[purged_train])
        validation_groups = set(group.iloc[validation_index])
        if train_groups.intersection(validation_groups):
            raise AssertionError("event group appears in both train and validation")
        plans.append((purged_train, validation_index))
        for role, indices in (("train", purged_train), ("validation", validation_index)):
            for index in indices:
                rows.append(
                    {
                        "fold": int(fold),
                        "row": int(index),
                        "role": role,
                        "event_id": str(group.iloc[index]),
                        "anchor_source_utc": work.iloc[index][anchor_column],
                    }
                )
    return plans, pd.DataFrame.from_records(rows)


def _usable_sequence_features(frame: pd.DataFrame) -> list[str]:
    candidates = [
        str(column)
        for column in frame.columns
        if str(column).startswith("sequence__") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    # A causal feature may have an unfortunate historical name such as
    # ``post_flush_leverage_rebuild``.  The model contract conservatively
    # rejects that ambiguous spelling.  Select every individually admissible
    # pre-onset summary rather than failing an otherwise valid materialization.
    features: list[str] = []
    for candidate in candidates:
        try:
            validate_preonset_sequence_columns(frame, [candidate])
        except ValueError:
            continue
        features.append(candidate)
    # Entirely unavailable fields make a train-fold imputer silently remove a
    # column.  Filter them once at the materialization boundary; all remaining
    # per-fold transforms are still fitted only on that fold.
    return [feature for feature in features if pd.to_numeric(frame[feature], errors="coerce").notna().any()]


def _causal_panel_feature_slice(hourly: pd.DataFrame, *, maximum: int) -> list[str]:
    """Bound the first catalogue slice without admitting a target-derived field."""

    if maximum < 1:
        raise ValueError("max_causal_features must be positive")
    candidates = causal_predictor_columns(hourly)
    # This is a deterministic schema-order cap, not outcome-driven selection.
    # It keeps the event materializer compact; later model comparison performs
    # fold-local selection/HPO over this and additional declared feature arms.
    return candidates[: int(maximum)]


def _stable_control_events(labeled: pd.DataFrame, *, count: int) -> pd.DataFrame:
    controls = labeled.loc[
        labeled["target__pattern_stable_eligible"].eq(1)
        & labeled["target__pattern_phase_available_utc"].notna(),
        [
            "source_utc",
            "segment_id",
            "target__pooled_state",
            "target__pattern_phase_available_utc",
        ],
    ].copy()
    if controls.empty or count <= 0:
        return pd.DataFrame()
    # Time-spread deterministic controls avoid a dense cluster of one calm
    # calendar interval without using outcome information.
    positions = np.unique(np.linspace(0, len(controls) - 1, num=min(count, len(controls)), dtype=int))
    controls = controls.iloc[positions].reset_index(drop=True)
    controls["event_id"] = [
        f"stable::{int(row.segment_id)}::{pd.Timestamp(row.source_utc).isoformat()}"
        for row in controls.itertuples(index=False)
    ]
    controls = controls.rename(columns={"source_utc": "anchor_source_utc", "target__pooled_state": "source_state"})
    controls["transition_end_utc"] = controls["anchor_source_utc"] + pd.Timedelta(hours=1)
    controls["destination_state"] = controls["source_state"]
    controls["target_available_utc"] = controls["target__pattern_phase_available_utc"]
    return controls[
        [
            "event_id",
            "segment_id",
            "anchor_source_utc",
            "transition_end_utc",
            "target_available_utc",
            "source_state",
            "destination_state",
        ]
    ]


def _merge_event_metadata(summary: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    fields = ["event_id", "target_available_utc"]
    result = summary.merge(events[fields], on="event_id", how="left", validate="one_to_one")
    result["target_available_utc"] = pd.to_datetime(result["target_available_utc"], utc=True, errors="coerce")
    return result


def _run_morphology_oof(
    events: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    morphology_config: TransitionMorphologyConfig,
    n_splits: int,
    purge_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train a GMM and a conditional morphology classifier inside each fold."""

    folds, fold_plan = _event_grouped_purged_folds(events, n_splits=n_splits, purge_hours=purge_hours)
    if not folds:
        return pd.DataFrame(), fold_plan, pd.DataFrame(), {"status": "insufficient_event_groups"}
    prediction: list[pd.DataFrame] = []
    support: list[pd.DataFrame] = []
    classifier_prediction: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(folds):
        train, validation = events.iloc[train_index].copy(), events.iloc[validation_index].copy()
        if len(train) < morphology_config.n_components:
            skipped.append({"fold": fold, "reason": "training_events_below_component_count"})
            continue
        try:
            embedder = TransitionMorphologyEmbedder(morphology_config).fit(
                train,
                feature_columns=feature_columns,
                era_column="era",
                bootstrap_draws=0,
            )
        except (ValueError, RuntimeError) as error:
            skipped.append({"fold": fold, "reason": f"morphology_fit:{type(error).__name__}"})
            continue
        local = validation[["event_id", "anchor_source_utc", "target_available_utc", "source_state", "destination_state"]].reset_index(drop=True)
        local = pd.concat([local, embedder.transform(validation).reset_index(drop=True)], axis=1)
        local["oof_fold"] = fold
        prediction.append(local)
        fold_support = embedder.support_table_.copy()
        fold_support["oof_fold"] = fold
        support.append(fold_support)
        # The supervised adapter learns the train-fold GMM's retained identity,
        # never a state-pair label.  Unsupported types are intentionally absent.
        train_type = embedder.transform(train)["morphology__component_id"]
        supported = train_type.ne("abstain")
        if supported.sum() < 4 or train_type.loc[supported].nunique() < 2:
            skipped.append({"fold": fold, "reason": "morphology_classifier_insufficient_supported_types"})
            continue
        classifier_train = train.loc[supported].copy()
        classifier_train["target__morphology_type"] = train_type.loc[supported].to_numpy()
        try:
            classifier = TransitionClassifierAdapter(random_state=morphology_config.random_state + fold).fit(
                classifier_train,
                target_column="target__morphology_type",
                feature_columns=feature_columns,
            )
            classifier_local = validation[["event_id", "anchor_source_utc"]].reset_index(drop=True)
            classifier_local = pd.concat([classifier_local, classifier.predict_proba(validation).reset_index(drop=True)], axis=1)
            classifier_local["oof_fold"] = fold
            classifier_prediction.append(classifier_local)
        except (ValueError, RuntimeError) as error:
            skipped.append({"fold": fold, "reason": f"morphology_classifier:{type(error).__name__}"})
    status = "complete" if prediction else "insufficient_supported_morphology_folds"
    return (
        pd.concat(prediction, ignore_index=True) if prediction else pd.DataFrame(),
        fold_plan,
        pd.concat(support, ignore_index=True) if support else pd.DataFrame(),
        {
            "status": status,
            "classifier_oof": pd.concat(classifier_prediction, ignore_index=True) if classifier_prediction else pd.DataFrame(),
            "skipped": pd.DataFrame(skipped),
        },
    )


def _run_stable_transition_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    n_splits: int,
    purge_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    folds, fold_plan = _event_grouped_purged_folds(frame, n_splits=n_splits, purge_hours=purge_hours)
    prediction: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(folds):
        train, validation = frame.iloc[train_index], frame.iloc[validation_index]
        if train["target__stable_vs_transition"].nunique() < 2:
            skipped.append({"fold": fold, "reason": "one_training_class"})
            continue
        try:
            model = TransitionClassifierAdapter(random_state=481 + fold).fit(
                train,
                target_column="target__stable_vs_transition",
                feature_columns=feature_columns,
            )
            local = validation[["event_id", "anchor_source_utc", "target_available_utc", "target__stable_vs_transition"]].reset_index(drop=True)
            local = pd.concat([local, model.predict_proba(validation).reset_index(drop=True)], axis=1)
            local["oof_fold"] = fold
            local["classifier_backend"] = model.backend
            prediction.append(local)
        except (ValueError, RuntimeError) as error:
            skipped.append({"fold": fold, "reason": f"stable_transition_classifier:{type(error).__name__}"})
    return (
        pd.concat(prediction, ignore_index=True) if prediction else pd.DataFrame(),
        fold_plan,
        pd.DataFrame(skipped),
        folds,
    )


def _run_stable_transition_brl_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score the BRL challenger on the exact LightGBM purged OOF row plan."""

    prediction: list[pd.DataFrame] = []
    rules: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(folds):
        train, validation = frame.iloc[train_index], frame.iloc[validation_index]
        if train["target__stable_vs_transition"].nunique() < 2:
            skipped.append({"fold": fold, "reason": "one_training_class"})
            continue
        try:
            model = BayesianRuleListChallenger(random_state=941 + fold).fit(
                train,
                target_column="target__stable_vs_transition",
                feature_columns=feature_columns,
            )
            local = validation[["event_id", "anchor_source_utc", "target_available_utc", "target__stable_vs_transition"]].reset_index(drop=True)
            local["brl__p_transition"] = model.predict_proba(validation).to_numpy(dtype=float)
            local["oof_fold"] = fold
            local["classifier_backend"] = model.backend
            prediction.append(local)
            rules.append({
                "oof_fold": fold,
                "backend": model.backend,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "feature_columns": list(feature_columns),
                "rule_list": model.describe(),
            })
        except (ValueError, RuntimeError) as error:
            skipped.append({"fold": fold, "reason": f"stable_transition_brl:{type(error).__name__}:{error}"})
    return (
        pd.concat(prediction, ignore_index=True) if prediction else pd.DataFrame(),
        pd.DataFrame(rules),
        pd.DataFrame(skipped),
    )


def _oof_metric_table(
    morphology_oof: pd.DataFrame,
    morphology_classifier_oof: pd.DataFrame,
    stable_oof: pd.DataFrame,
    stable_brl_oof: pd.DataFrame,
) -> pd.DataFrame:
    """Compact diagnostic metrics; no economic outcome is used here."""

    rows: list[dict[str, Any]] = []
    if not stable_oof.empty:
        columns = [name for name in stable_oof if name.startswith("classifier__p_")]
        positive = "classifier__p_1" if "classifier__p_1" in columns else None
        if positive is not None:
            y = pd.to_numeric(stable_oof["target__stable_vs_transition"], errors="coerce")
            p = pd.to_numeric(stable_oof[positive], errors="coerce")
            valid = y.notna() & p.notna()
            if valid.any() and y.loc[valid].nunique() == 2:
                rows.append(
                    {
                        "task": "stable_vs_transition",
                        "rows": int(valid.sum()),
                        "classes": 2,
                        "accuracy": float(accuracy_score(y.loc[valid], p.loc[valid].ge(0.5))),
                        "roc_auc": float(roc_auc_score(y.loc[valid], p.loc[valid])),
                        "average_precision": float(average_precision_score(y.loc[valid], p.loc[valid])),
                        "brier": float(brier_score_loss(y.loc[valid], p.loc[valid])),
                    }
                )
    if not stable_brl_oof.empty:
        y = pd.to_numeric(stable_brl_oof["target__stable_vs_transition"], errors="coerce")
        p = pd.to_numeric(stable_brl_oof["brl__p_transition"], errors="coerce")
        valid = y.notna() & p.notna()
        if valid.any() and y.loc[valid].nunique() == 2:
            rows.append(
                {
                    "task": "stable_vs_transition_brl",
                    "rows": int(valid.sum()),
                    "classes": 2,
                    "accuracy": float(accuracy_score(y.loc[valid], p.loc[valid].ge(0.5))),
                    "roc_auc": float(roc_auc_score(y.loc[valid], p.loc[valid])),
                    "average_precision": float(average_precision_score(y.loc[valid], p.loc[valid])),
                    "brier": float(brier_score_loss(y.loc[valid], p.loc[valid])),
                }
            )
    if not morphology_oof.empty and not morphology_classifier_oof.empty:
        key = ["event_id", "oof_fold"]
        actual = morphology_oof[[*key, "morphology__component_id"]]
        merged = morphology_classifier_oof.merge(actual, on=key, how="inner", validate="one_to_one")
        probability_columns = [name for name in merged if name.startswith("classifier__p_")]
        valid = merged["morphology__component_id"].ne("abstain") & merged[probability_columns].notna().all(axis=1)
        if valid.any() and probability_columns:
            probability = merged.loc[valid, probability_columns].to_numpy(float)
            predicted = np.asarray([column.removeprefix("classifier__p_") for column in probability_columns], dtype=object)[probability.argmax(axis=1)]
            y = merged.loc[valid, "morphology__component_id"].astype(str).to_numpy()
            rows.append(
                {
                    "task": "morphology_conditional_on_transition",
                    "rows": int(valid.sum()),
                    "classes": int(pd.Series(y).nunique()),
                    "accuracy": float(accuracy_score(y, predicted)),
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                    "brier": np.nan,
                }
            )
    return pd.DataFrame(rows)


def materialize_transition_pattern_catalogue(
    *,
    ledger_dir: Path = DEFAULT_LEDGER,
    output_dir: Path,
    pattern_config: TransitionPatternConfig = TransitionPatternConfig(),
    morphology_config: TransitionMorphologyConfig = TransitionMorphologyConfig(),
    n_splits: int = 5,
    purge_hours: int = 36,
    max_causal_features: int = 32,
) -> dict[str, Any]:
    """Build a provenance-bound research catalogue; never overwrite outputs."""

    ledger_dir, output_dir = Path(ledger_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    hourly, event_ledger, ledger_manifest = _require_ledger(ledger_dir)
    output_dir.mkdir(parents=True)
    labeled = materialize_adaptive_transition_phases(hourly, event_ledger, config=pattern_config)
    # The following two outputs are deliberately separate: the first is
    # outcome/phase labels, the second contains only pre-onset causal summaries
    # plus identity/state metadata for descriptive reporting.
    # Catalogue function keeps diagnostic event metadata in attrs for in-memory
    # callers.  Parquet metadata must remain JSON-serializable and this output
    # already has the canonical event ledger as a separate provenance source.
    labeled.attrs.clear()
    labeled.to_parquet(output_dir / "adaptive_phase_labels.parquet", index=False)
    causal_panel_features = _causal_panel_feature_slice(hourly, maximum=max_causal_features)
    event_summary = summarize_event_preonset_sequences(
        hourly,
        event_ledger,
        feature_columns=causal_panel_features,
        config=pattern_config,
    )
    event_summary = _merge_event_metadata(event_summary, event_ledger)
    event_summary["era"] = event_summary["anchor_source_utc"].dt.year.astype(str)
    feature_columns = _usable_sequence_features(event_summary)
    event_summary.to_parquet(output_dir / "event_preonset_sequences.parquet", index=False)

    morphology_oof, morphology_plan, recurrence, morphology_run = _run_morphology_oof(
        event_summary,
        feature_columns=feature_columns,
        morphology_config=morphology_config,
        n_splits=n_splits,
        purge_hours=purge_hours,
    )
    morphology_oof.to_parquet(output_dir / "morphology_oof.parquet", index=False)
    morphology_plan.to_parquet(output_dir / "morphology_fold_plan.parquet", index=False)
    recurrence.to_csv(output_dir / "morphology_recurrence_support.csv", index=False)
    morphology_run["classifier_oof"].to_parquet(output_dir / "morphology_classifier_oof.parquet", index=False)
    morphology_run["skipped"].to_csv(output_dir / "morphology_skipped.csv", index=False)

    # Stable controls are decision-time quiet state rows.  They are converted to
    # pseudo-events solely to reuse the exact same *pre-onset* sequence builder.
    # They are never fed to the morphology GMM and are not market-state labels.
    stable_rows = sample_stable_vs_transition(labeled, stable_to_transition_ratio=1.0)
    stable_events = _stable_control_events(
        stable_rows.loc[stable_rows["target__stable_vs_transition"].eq(0)],
        count=len(event_summary),
    )
    stable_summary = (
        summarize_event_preonset_sequences(
            hourly,
            stable_events,
            feature_columns=causal_panel_features,
            config=pattern_config,
        )
        if not stable_events.empty
        else pd.DataFrame()
    )
    if not stable_summary.empty:
        stable_summary = _merge_event_metadata(stable_summary, stable_events)
        stable_summary["target__stable_vs_transition"] = np.int8(0)
    event_classification = event_summary.copy()
    event_classification["target__stable_vs_transition"] = np.int8(1)
    classification = pd.concat([event_classification, stable_summary], ignore_index=True)
    classification_features = _usable_sequence_features(classification) if not classification.empty else []
    stable_oof, stable_plan, stable_skipped, stable_folds = _run_stable_transition_oof(
        classification,
        feature_columns=classification_features,
        n_splits=n_splits,
        purge_hours=purge_hours,
    ) if classification_features else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame({"reason": ["no_usable_sequence_features"]}), [])
    stable_brl_oof, stable_brl_rules, stable_brl_skipped = _run_stable_transition_brl_oof(
        classification,
        feature_columns=classification_features,
        folds=stable_folds,
    ) if classification_features else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame({"reason": ["no_usable_sequence_features"]}))
    if not stable_oof.empty and not stable_brl_oof.empty:
        lightgbm_keys = stable_oof.loc[:, ["event_id", "oof_fold"]].sort_values(["event_id", "oof_fold"]).reset_index(drop=True)
        brl_keys = stable_brl_oof.loc[:, ["event_id", "oof_fold"]].sort_values(["event_id", "oof_fold"]).reset_index(drop=True)
        if not lightgbm_keys.equals(brl_keys):
            raise RuntimeError("BRL OOF rows differ from the LightGBM purged fold plan")
    classification.to_parquet(output_dir / "stable_transition_sequence_inputs.parquet", index=False)
    stable_oof.to_parquet(output_dir / "stable_transition_oof.parquet", index=False)
    stable_plan.to_parquet(output_dir / "stable_transition_fold_plan.parquet", index=False)
    stable_skipped.to_csv(output_dir / "stable_transition_skipped.csv", index=False)
    stable_brl_oof.to_parquet(output_dir / "stable_transition_brl_oof.parquet", index=False)
    _write_json(output_dir / "stable_transition_brl_rule_lists.json", {"schema": "transition_brl_rule_lists_v1", "backend_contract": "native_beta_binomial_map is a dependency-free ordered MAP rule-list, not an MCMC BRL", "folds": stable_brl_rules.to_dict(orient="records")})
    stable_brl_skipped.to_csv(output_dir / "stable_transition_brl_skipped.csv", index=False)
    metrics = _oof_metric_table(morphology_oof, morphology_run["classifier_oof"], stable_oof, stable_brl_oof)
    metrics.to_csv(output_dir / "oof_diagnostic_metrics.csv", index=False)

    output_hashes = {
        path.name: _sha256(path)
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema": SCHEMA,
        "research_only": True,
        "promotion_eligible": False,
        "purpose": "adaptive transition-pattern discovery and OOF diagnostics; no policy or portfolio routing",
        "sources": {
            "ledger_dir": str(ledger_dir),
            "ledger_manifest_sha256": _sha256(ledger_dir / "manifest.json"),
            "ledger_schema": ledger_manifest.get("schema"),
            "hourly_sha256": _sha256(ledger_dir / "hourly_state_calendar.parquet"),
            "events_sha256": _sha256(ledger_dir / "transition_episode_ledger.parquet"),
        },
        "counts": {
            "hourly_rows": int(len(hourly)),
            "events": int(len(event_summary)),
            "preonset_feature_columns": int(len(feature_columns)),
            "causal_panel_feature_columns": int(len(causal_panel_features)),
            "stable_control_events": int(len(stable_summary)),
            "morphology_oof_rows": int(len(morphology_oof)),
            "stable_transition_oof_rows": int(len(stable_oof)),
            "stable_transition_brl_oof_rows": int(len(stable_brl_oof)),
            "oof_metric_rows": int(len(metrics)),
        },
        "field_contract": {
            "causal_sequence_fields": list(feature_columns),
            "causal_panel_fields": list(causal_panel_features),
            "descriptive_only": [
                "source_state",
                "destination_state",
                "target__pattern_phase",
                "target__pattern_event_id",
                "target__stable_vs_transition",
            ],
            "forbidden_model_inputs": "target/expost/outcome/post-onset fields, regime-state identity, event identity and availability fields",
            "morphology_identity": "fold-local train-only GMM component posterior; component state pair is never an input",
        },
        "validation_contract": {
            "fold": "GroupKFold by event/control identity with two-sided anchor purge",
            "purge_hours": int(purge_hours),
            "not_walk_forward": True,
            "morphology": morphology_run["status"],
            "stable_transition": "complete" if len(stable_oof) else "insufficient_support",
            "stable_transition_brl": "complete" if len(stable_brl_oof) else "insufficient_support",
        },
        "pattern_config": pattern_config.__dict__,
        "morphology_config": morphology_config.__dict__,
        "outputs_sha256": output_hashes,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--purge-hours", type=int, default=36)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--min-component-events", type=int, default=8)
    parser.add_argument("--max-causal-features", type=int, default=32)
    args = parser.parse_args()
    report = materialize_transition_pattern_catalogue(
        ledger_dir=args.ledger_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        purge_hours=args.purge_hours,
        max_causal_features=args.max_causal_features,
        morphology_config=TransitionMorphologyConfig(
            n_components=args.n_components,
            min_component_events=args.min_component_events,
        ),
    )
    print(json.dumps(_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
