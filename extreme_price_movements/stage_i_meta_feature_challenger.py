"""Sequential Stage-I meta feature-count challenger.

This module is deliberately separate from the base feature challenger.  It
consumes a *completed* side-local meta selection, freezes the exact same-side
base OOF handoff, and varies only the meta feature contract through the
predeclared automatic/20/30/40/60/full-input ladder.  It therefore answers the
specific over-pruning question without refitting or selecting a different base
model in every cell.

Materialising a plan never fits a model.  Execution remains callback-driven so
each target-specific meta head can retain its proper objective and strict OOF
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_nested_feature_challenger import (
    IDENTITY_COLUMNS,
    MDAFeatureRank,
    MetaTargetMetricSpec,
    NESTED_SET_NAMES,
    NESTED_SET_SIZES,
    NestedFeatureChallengerError,
    NestedFeatureSet,
    StrictOOFResult,
    _binary_metrics,
    _canonical_hash,
    _file_sha256,
    _finite,
    _multiclass_metrics,
    _normalise_family,
    _ordered_unique,
    _top_tail_metrics,
    _validate_strict,
)
from .stage_i_feature_selection import STAGE_I_CORRELATION_POLICIES


SCHEMA = "stage_i_meta_nested_feature_challenger_v1"
EVALUATION_SCHEMA = "stage_i_frozen_base_meta_feature_evaluation_v1"


@dataclass(frozen=True)
class CompletedMetaSelection:
    side: str
    selection_dir: Path
    selected_features: tuple[str, ...]
    input_features: tuple[str, ...]
    required_base_trust_features: tuple[str, ...]
    source_ranks: Mapping[str, MDAFeatureRank]
    manifest_sha256: str
    audit_sha256: str
    audit_path: Path
    selector_manifest_sha256: str
    selector_feature_contract_sha256: str
    frozen_base_manifest_sha256: str
    frozen_base_oof_sha256: str
    candidate_handoff_audit_sha256: str
    selector_meta_oof_sha256: str
    target_semantics: str
    correlation_policy: str
    base_correlation_policy: str


@dataclass(frozen=True)
class MetaFeatureChallengePlan:
    side: str
    source_manifest_sha256: str
    source_audit_sha256: str
    source_audit_path: str
    selector_manifest_sha256: str
    selector_feature_contract_sha256: str
    frozen_base_manifest_sha256: str
    frozen_base_oof_sha256: str
    candidate_handoff_audit_sha256: str
    selector_meta_oof_sha256: str
    target_semantics: str
    required_base_trust_features: tuple[str, ...]
    required_features: tuple[str, ...]
    protected_features: tuple[str, ...]
    feature_sets: tuple[NestedFeatureSet, ...]
    correlation_policy: str = ""
    base_correlation_policy: str = ""

    @property
    def plan_hash(self) -> str:
        return _canonical_hash(self.as_dict(include_hash=False))

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "side": self.side,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_audit_sha256": self.source_audit_sha256,
            "source_audit_path": self.source_audit_path,
            "selector_manifest_sha256": self.selector_manifest_sha256,
            "selector_feature_contract_sha256": self.selector_feature_contract_sha256,
            "frozen_base_manifest_sha256": self.frozen_base_manifest_sha256,
            "frozen_base_oof_sha256": self.frozen_base_oof_sha256,
            "candidate_handoff_audit_sha256": self.candidate_handoff_audit_sha256,
            "selector_meta_oof_sha256": self.selector_meta_oof_sha256,
            "target_semantics": self.target_semantics,
            "correlation_policy": self.correlation_policy,
            "base_correlation_policy": self.base_correlation_policy,
            "comparison_scope": (
                "sequential_frozen_same_side_base_oof; vary_meta_features_only; "
                "identical_candidate_rows_targets_folds_hpo_and_economic_gates"
            ),
            "required_base_trust_features": list(self.required_base_trust_features),
            "required_features": list(self.required_features),
            "protected_features": list(self.protected_features),
            "feature_sets": [item.as_dict() for item in self.feature_sets],
        }
        if include_hash:
            value["plan_sha256"] = _canonical_hash(value)
        return value


def _round_number(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("round_"))
    except ValueError as exc:
        raise NestedFeatureChallengerError(
            f"invalid immutable meta MDA round path: {path}"
        ) from exc


def _meta_audits(root: Path) -> tuple[Path, tuple[Path, ...]]:
    reports = sorted(root.glob("mda/**/mda_feature_selection_report.json"))
    if not reports:
        raise NestedFeatureChallengerError(
            f"{root}: no completed immutable meta MDA report was found"
        )
    final_report = max(reports, key=lambda path: (_round_number(path), str(path)))
    report = json.loads(final_report.read_text(encoding="utf-8"))
    configured = report.get("feature_audit_path")
    candidates = [final_report.parent / "mda_feature_audit.csv"]
    if isinstance(configured, str) and configured.strip():
        path = Path(configured)
        candidates.extend((path, final_report.parent / path.name))
    final_candidates = list(
        dict.fromkeys(path.resolve() for path in candidates if path.is_file())
    )
    if len(final_candidates) != 1:
        raise NestedFeatureChallengerError(
            f"{root}: completed meta MDA audit is missing or ambiguous"
        )
    final_audit = final_candidates[0]
    report_root = final_report.parent.parent
    audits = tuple(
        sorted(
            (path.resolve() for path in report_root.glob("round_*/mda_feature_audit.csv")),
            key=lambda path: (_round_number(path), str(path)),
        )
    )
    if not audits or final_audit not in audits:
        raise NestedFeatureChallengerError(
            f"{root}: final meta audit is not bound to its immutable round root"
        )
    return final_audit, audits


def _rank_meta_features(
    input_features: Sequence[str], audit_paths: Sequence[Path]
) -> tuple[dict[str, MDAFeatureRank], str]:
    records: dict[str, tuple[pd.Series, Path, int, str]] = {}
    for path in audit_paths:
        audit = pd.read_csv(path)
        if audit.empty or "feature" not in audit:
            raise NestedFeatureChallengerError(f"{path}: empty meta MDA audit")
        audit["feature"] = audit.feature.astype(str)
        if audit.feature.duplicated().any():
            raise NestedFeatureChallengerError(
                f"{path}: duplicate meta feature evidence in one immutable round"
            )
        round_number, digest = _round_number(path), _file_sha256(path)
        for _, row in audit.iterrows():
            observed = int(
                max(
                    0,
                    _finite(
                        row.get("mda_cohort_count", row.get("mda_n_folds", 0)), 0.0
                    ),
                )
            )
            repeats = int(max(0, _finite(row.get("mda_n_repeats", observed), 0.0)))
            if "mda_feature_evaluable" in row and not bool(row.get("mda_feature_evaluable")):
                continue
            if observed <= 0 and repeats <= 0:
                continue
            feature = str(row.feature)
            prior = records.get(feature)
            if prior is None or round_number > prior[2]:
                records[feature] = (row, path, round_number, digest)
            elif round_number == prior[2]:
                raise NestedFeatureChallengerError(
                    f"{path}: duplicate evaluated evidence for {feature!r}"
                )
    ordered = sorted(
        records.items(),
        key=lambda item: (
            -_finite(item[1][0].get("mda_median")),
            -_finite(item[1][0].get("mda_mean")),
            -_finite(item[1][0].get("mda_positive_cohort_rate")),
            -_finite(item[1][0].get("mda_latest_cohort_mda")),
            item[0],
        ),
    )
    ranks: dict[str, MDAFeatureRank] = {}
    for rank, (feature, (row, path, round_number, digest)) in enumerate(ordered, 1):
        group = row.get("mda_group_id")
        family = (
            str(group)
            if isinstance(group, str) and group.strip() and group.lower() != "nan"
            else _normalise_family(feature)
        )
        median, mean = _finite(row.get("mda_median")), _finite(row.get("mda_mean"))
        positive = _finite(row.get("mda_positive_cohort_rate"))
        worst, latest = _finite(row.get("mda_worst_cohort_mda")), _finite(
            row.get("mda_latest_cohort_mda")
        )
        cohorts = int(
            max(0, _finite(row.get("mda_cohort_count", row.get("mda_n_folds", 0)), 0.0))
        )
        stable = positive >= 0.5 and cohorts >= 2 and worst >= 0.0
        consistently_negative = (
            cohorts >= 2
            and median < 0.0
            and mean < 0.0
            and worst < 0.0
            and latest < 0.0
        )
        tier = (
            "strong_stable"
            if stable
            else "consistently_materially_negative_excluded"
            if consistently_negative
            else "borderline_or_uncertain"
        )
        ranks[feature] = MDAFeatureRank(
            feature=feature,
            source_rank=rank,
            family=family,
            mda_median=median,
            mda_mean=mean,
            positive_cohort_rate=positive,
            worst_cohort_mda=worst,
            latest_cohort_mda=latest,
            cohort_count=cohorts,
            confidence_label=str(row.get("confidence_label", "")),
            stable=stable,
            tier=tier,
            audit_observed=True,
            source_round=f"round_{round_number:02d}",
            source_audit_path=str(path),
            source_audit_sha256=digest,
        )
    for feature in input_features:
        if feature in ranks:
            continue
        ranks[feature] = MDAFeatureRank(
            feature=feature,
            source_rank=len(ranks) + 1,
            family=_normalise_family(feature),
            mda_median=0.0,
            mda_mean=0.0,
            positive_cohort_rate=0.0,
            worst_cohort_mda=0.0,
            latest_cohort_mda=0.0,
            cohort_count=0,
            confidence_label="not_present_in_completed_meta_mda_audit",
            stable=False,
            tier="untested_or_group_skipped",
            audit_observed=False,
            source_round="never_evaluated",
        )
    audit_hash = _canonical_hash(
        [{"path": str(path), "sha256": _file_sha256(path)} for path in audit_paths]
    )
    return ranks, audit_hash


def load_completed_stage_i_meta_selection(
    selection_dir: str | Path,
    *,
    side: str,
    selector_dir: str | Path,
    base_selection_dir: str | Path,
) -> CompletedMetaSelection:
    """Load and verify a completed meta selector and its frozen-base lineage."""
    side = str(side).lower()
    root, selector_root, base_root = (
        Path(selection_dir),
        Path(selector_dir),
        Path(base_selection_dir),
    )
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise NestedFeatureChallengerError(
            f"missing completed meta-selection manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "stage_i_meta_feature_selection_v1"
        or manifest.get("status") != "complete"
        or str(manifest.get("side", "")).lower() != side
        or side not in {"long", "short"}
    ):
        raise NestedFeatureChallengerError(
            f"{root}: not a completed Stage-I {side} meta selection"
        )
    selected = _ordered_unique(
        manifest.get("selected_feature_contract", manifest.get("selected_features", ()))
    )
    inputs = _ordered_unique(manifest.get("input_feature_contract", ()))
    required = _ordered_unique(
        manifest.get("required_same_side_base_oof_handoff_features", ())
    )
    if (
        not selected
        or not inputs
        or not required
        or not set(selected).issubset(inputs)
        or not set(required).issubset(selected)
    ):
        raise NestedFeatureChallengerError(
            f"{root}: meta selected/input/required feature contracts are incomplete"
        )

    selector_manifest = selector_root / "manifest.json"
    selector_contract = selector_root / "selector_feature_contract.json"
    base_manifest = base_root / side / "manifest.json"
    base_oof = base_root / side / "selector_base_oof.parquet"
    meta_oof = root / "selector_meta_oof.parquet"
    map_audit = root / "prequential_value_map_audit.parquet"
    candidate_audit = root / "base_candidate_handoff_audit.parquet"
    required_paths = (
        selector_manifest,
        selector_contract,
        base_manifest,
        base_oof,
        meta_oof,
        map_audit,
        candidate_audit,
    )
    if missing := [str(path) for path in required_paths if not path.is_file()]:
        raise NestedFeatureChallengerError(
            f"{root}: completed meta lineage artifacts are missing: {missing}"
        )
    expected = {
        "selector_sample_manifest_sha256": _file_sha256(selector_manifest),
        "selector_feature_contract_sha256": _file_sha256(selector_contract),
        "base_selector_manifest_sha256": _file_sha256(base_manifest),
        "base_selector_oof_sha256": _file_sha256(base_oof),
        "selector_meta_oof_sha256": _file_sha256(meta_oof),
        "prequential_value_map_audit_sha256": _file_sha256(map_audit),
        "base_candidate_handoff_audit_sha256": _file_sha256(candidate_audit),
    }
    drift = [key for key, digest in expected.items() if manifest.get(key) != digest]
    if drift:
        raise NestedFeatureChallengerError(
            f"{root}: completed meta lineage hash drift: {drift}"
        )
    base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    if (
        base_payload.get("schema") != "stage_i_base_feature_selection_v1"
        or base_payload.get("status") != "complete"
        or str(base_payload.get("side", "")).lower() != side
    ):
        raise NestedFeatureChallengerError(
            f"{side}: frozen base selection is not a completed same-side contract"
        )
    meta_policy = str(manifest.get("correlation_policy", ""))
    base_policy = str(base_payload.get("correlation_policy", ""))
    lineage_policy = str(
        (manifest.get("base_correlation_lineage") or {}).get(
            "correlation_policy", ""
        )
    )
    if (
        meta_policy not in STAGE_I_CORRELATION_POLICIES
        or base_policy not in STAGE_I_CORRELATION_POLICIES
        or meta_policy != base_policy
        or lineage_policy != base_policy
        or manifest.get("base_correlation_policy") != base_policy
    ):
        raise NestedFeatureChallengerError(
            f"{side}: meta/base correlation-policy lineage is missing or mismatched"
        )
    audit_path, audit_paths = _meta_audits(root)
    ranks, audit_hash = _rank_meta_features(inputs, audit_paths)
    return CompletedMetaSelection(
        side=side,
        selection_dir=root,
        selected_features=selected,
        input_features=inputs,
        required_base_trust_features=required,
        source_ranks=ranks,
        manifest_sha256=_file_sha256(manifest_path),
        audit_sha256=audit_hash,
        audit_path=audit_path,
        selector_manifest_sha256=expected["selector_sample_manifest_sha256"],
        selector_feature_contract_sha256=expected["selector_feature_contract_sha256"],
        frozen_base_manifest_sha256=expected["base_selector_manifest_sha256"],
        frozen_base_oof_sha256=expected["base_selector_oof_sha256"],
        candidate_handoff_audit_sha256=expected[
            "base_candidate_handoff_audit_sha256"
        ],
        selector_meta_oof_sha256=expected["selector_meta_oof_sha256"],
        target_semantics=str(
            manifest.get(
                "hpo_oof_score_semantics",
                "target_specific_meta_head_on_frozen_same_side_base_oof",
            )
        ),
        correlation_policy=meta_policy,
        base_correlation_policy=base_policy,
    )


def _family_counts(
    features: Sequence[str], ranks: Mapping[str, MDAFeatureRank]
) -> tuple[dict[str, str], dict[str, int]]:
    families = {feature: ranks[feature].family for feature in features}
    counts: dict[str, int] = {}
    for family in families.values():
        counts[family] = counts.get(family, 0) + 1
    return families, dict(sorted(counts.items()))


def _fixed_ladder(
    source: CompletedMetaSelection, mandatory: Sequence[str], requested: int
) -> tuple[str, ...]:
    existing, chosen = set(mandatory), []
    _, counts = _family_counts(mandatory, source.source_ranks)
    tier_order = {
        "strong_stable": 0,
        "borderline_or_uncertain": 1,
        "untested_or_group_skipped": 2,
    }
    candidates = [
        rank
        for feature, rank in source.source_ranks.items()
        if feature not in existing
        and feature in source.input_features
        and rank.tier != "consistently_materially_negative_excluded"
    ]
    needed = max(0, requested - len(mandatory))
    if len(candidates) < needed:
        raise NestedFeatureChallengerError(
            f"{source.side}: only {len(candidates)} eligible optional meta fields "
            f"are available for the top{requested} challenger"
        )
    while len(chosen) < needed:
        item = min(
            candidates,
            key=lambda rank: (
                tier_order[rank.tier],
                counts.get(rank.family, 0),
                rank.source_rank,
                rank.feature,
            ),
        )
        chosen.append(item.feature)
        counts[item.family] = counts.get(item.family, 0) + 1
        candidates.remove(item)
    return tuple(chosen)


def materialize_meta_feature_challenge(
    source: CompletedMetaSelection,
    *,
    required_features: Sequence[str] = (),
    protected_features: Sequence[str] = (),
) -> MetaFeatureChallengePlan:
    """Build the sequential meta automatic/20/30/40/60/full-input ladder."""
    required = _ordered_unique(required_features)
    protected = _ordered_unique(protected_features)
    mandatory = _ordered_unique(
        (*source.required_base_trust_features, *required, *protected)
    )
    if missing := sorted(set(mandatory).difference(source.input_features)):
        raise NestedFeatureChallengerError(
            f"{source.side}: mandatory meta fields escape the completed input contract: {missing}"
        )
    smallest = min(
        int(size) for size in NESTED_SET_SIZES.values() if size is not None
    )
    if len(mandatory) > smallest:
        raise NestedFeatureChallengerError(
            f"{source.side}: {len(mandatory)} mandatory meta fields cannot fit the "
            f"predeclared top{smallest} arm"
        )
    automatic = _ordered_unique((*source.selected_features, *mandatory))
    maximum = max(int(size) for size in NESTED_SET_SIZES.values() if size is not None)
    ladder = _fixed_ladder(source, mandatory, maximum)
    feature_sets: list[NestedFeatureSet] = []
    for name in NESTED_SET_NAMES:
        requested = NESTED_SET_SIZES[name]
        if name == "automatic_sparse":
            features, additions = automatic, ()
        elif name == "full_input_control":
            features, additions = source.input_features, ()
        else:
            additions = ladder[: max(0, int(requested) - len(mandatory))]
            features = _ordered_unique((*mandatory, *additions))
        families, family_composition = _family_counts(features, source.source_ranks)
        tier_composition: dict[str, int] = {}
        for feature in features:
            tier = (
                "mandatory_base_or_trust"
                if feature in mandatory
                else "selected_automatic_sparse"
                if name == "automatic_sparse"
                else f"full_input_control__{source.source_ranks[feature].tier}"
                if name == "full_input_control"
                else source.source_ranks[feature].tier
            )
            tier_composition[tier] = tier_composition.get(tier, 0) + 1
        rank_map = {
            feature: source.source_ranks[feature].source_rank for feature in features
        }
        source_hash = _canonical_hash(
            {
                "features": features,
                "source_ranks": rank_map,
                "meta_manifest_sha256": source.manifest_sha256,
                "meta_audit_sha256": source.audit_sha256,
                "frozen_base_oof_sha256": source.frozen_base_oof_sha256,
            }
        )
        control = (
            {
                "kind": "full_input_control",
                "source": "completed_stage_i_authorized_side_meta_input_feature_contract",
                "postscreen_bypass": True,
                "promotion_policy": (
                    "eligible_only_if_best_under_identical_strict_OOF_and_OOS_gates; "
                    "no_post_test_tuning"
                ),
            }
            if name == "full_input_control"
            else {}
        )
        feature_sets.append(
            NestedFeatureSet(
                side=source.side,
                name=name,
                requested_feature_count=requested,
                features=features,
                added_features=additions,
                source_ranks=rank_map,
                feature_families=families,
                family_composition=family_composition,
                tier_composition=dict(sorted(tier_composition.items())),
                source_hash=source_hash,
                control_provenance=control,
                promotion_eligible=True,
                source_rank_evidence={
                    feature: source.source_ranks[feature].as_dict()
                    for feature in features
                },
            )
        )
    return MetaFeatureChallengePlan(
        side=source.side,
        source_manifest_sha256=source.manifest_sha256,
        source_audit_sha256=source.audit_sha256,
        source_audit_path=str(source.audit_path),
        selector_manifest_sha256=source.selector_manifest_sha256,
        selector_feature_contract_sha256=source.selector_feature_contract_sha256,
        frozen_base_manifest_sha256=source.frozen_base_manifest_sha256,
        frozen_base_oof_sha256=source.frozen_base_oof_sha256,
        candidate_handoff_audit_sha256=source.candidate_handoff_audit_sha256,
        selector_meta_oof_sha256=source.selector_meta_oof_sha256,
        target_semantics=source.target_semantics,
        required_base_trust_features=source.required_base_trust_features,
        required_features=required,
        protected_features=protected,
        feature_sets=tuple(feature_sets),
        correlation_policy=source.correlation_policy,
        base_correlation_policy=source.base_correlation_policy,
    )


def checkpoint_meta_feature_plan(
    plan: MetaFeatureChallengePlan, output_dir: str | Path
) -> Path:
    """Persist an immutable, hash-bound meta ladder checkpoint."""
    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    payload = plan.as_dict()
    if destination.exists():
        sets_path = destination / "nested_meta_feature_sets.json"
        if not manifest_path.is_file() or not sets_path.is_file():
            raise NestedFeatureChallengerError(
                f"meta feature checkpoint exists but is incomplete: {destination}"
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("plan_sha256") != payload["plan_sha256"]
            or previous.get("nested_meta_feature_sets_sha256")
            != _file_sha256(sets_path)
        ):
            raise NestedFeatureChallengerError(
                f"meta feature checkpoint drift: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        sets_path = temporary / "nested_meta_feature_sets.json"
        sets_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest = {
            "schema": SCHEMA,
            "status": "materialized",
            "side": plan.side,
            "plan_sha256": payload["plan_sha256"],
            "source_manifest_sha256": plan.source_manifest_sha256,
            "frozen_base_manifest_sha256": plan.frozen_base_manifest_sha256,
            "frozen_base_oof_sha256": plan.frozen_base_oof_sha256,
            "nested_meta_feature_sets_sha256": _file_sha256(sets_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
    return destination


MetaFeatureHook = Callable[
    [NestedFeatureSet, pd.DataFrame, MetaTargetMetricSpec], StrictOOFResult
]


def _identity_key(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame.loc[:, list(IDENTITY_COLUMNS)])


def evaluate_frozen_base_meta_feature_challenge(
    plan: MetaFeatureChallengePlan,
    *,
    meta_selection_dir: str | Path,
    frozen_base_oof_path: str | Path,
    meta_hook: MetaFeatureHook,
    meta_specs: Sequence[MetaTargetMetricSpec],
) -> dict[str, Any]:
    """Evaluate only meta feature counts on one immutable base candidate stream.

    ``meta_hook`` owns chronological fitting for its target.  This adapter
    supplies the exact candidate population used by the completed selector and
    rejects any hook whose strict-OOF rows differ across feature-count arms.
    """
    if not callable(meta_hook) or not meta_specs:
        raise NestedFeatureChallengerError(
            "a target-specific meta hook and metric specs are required"
        )
    root, base_path = Path(meta_selection_dir), Path(frozen_base_oof_path)
    if _file_sha256(root / "manifest.json") != plan.source_manifest_sha256:
        raise NestedFeatureChallengerError("meta selection manifest drifted after plan freeze")
    if _file_sha256(base_path) != plan.frozen_base_oof_sha256:
        raise NestedFeatureChallengerError("frozen base OOF artifact drifted after plan freeze")
    candidate_path, meta_oof_path = (
        root / "base_candidate_handoff_audit.parquet",
        root / "selector_meta_oof.parquet",
    )
    if (
        _file_sha256(candidate_path) != plan.candidate_handoff_audit_sha256
        or _file_sha256(meta_oof_path) != plan.selector_meta_oof_sha256
    ):
        raise NestedFeatureChallengerError("frozen meta candidate population drifted")
    candidate, meta_population, base = (
        pd.read_parquet(candidate_path),
        pd.read_parquet(meta_oof_path),
        pd.read_parquet(base_path),
    )
    if "selected_base_candidate" not in candidate:
        raise NestedFeatureChallengerError("candidate handoff audit lacks its selection flag")
    selected_key = _identity_key(candidate.loc[candidate.selected_base_candidate.astype(bool)])
    population_key = _identity_key(meta_population)
    if not population_key.isin(selected_key).all() or population_key.duplicated().any():
        raise NestedFeatureChallengerError(
            "meta selector population escapes the frozen base top-candidate handoff"
        )
    indexed = base.set_index(list(IDENTITY_COLUMNS), drop=False)
    if indexed.index.duplicated().any() or not population_key.isin(indexed.index).all():
        raise NestedFeatureChallengerError(
            "frozen base OOF does not cover the exact meta selector population"
        )
    frozen = indexed.loc[population_key].reset_index(drop=True)
    if not _identity_key(frozen).equals(population_key):
        raise NestedFeatureChallengerError("frozen base/meta identity order differs")
    probability = frozen.loc[
        :, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if (
        probability.shape != (len(frozen), 3)
        or not np.isfinite(probability).all()
        or (probability < 0).any()
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise NestedFeatureChallengerError(
            "frozen same-side base candidate handoff is not a finite R3 simplex"
        )
    common_hash: str | None = None
    evaluations: list[dict[str, Any]] = []
    for feature_set in plan.feature_sets:
        heads: dict[str, Any] = {}
        for spec in meta_specs:
            result = meta_hook(feature_set, frozen.copy(), spec)
            if not isinstance(result, StrictOOFResult):
                raise NestedFeatureChallengerError(
                    "meta feature hook must return StrictOOFResult"
                )
            identity_hash = _validate_strict(result, side=plan.side, layer="meta")
            if not _identity_key(result.frame).equals(population_key):
                raise NestedFeatureChallengerError(
                    "meta feature arms do not share the frozen base candidate rows/order"
                )
            if common_hash is None:
                common_hash = identity_hash
            elif common_hash != identity_hash:
                raise NestedFeatureChallengerError(
                    "meta feature arms do not share identical strict OOF rows"
                )
            required_columns = {spec.target_column, *spec.prediction_columns}
            if missing := sorted(required_columns.difference(result.frame.columns)):
                raise NestedFeatureChallengerError(
                    f"{spec.name}: meta metric contract lacks {missing}"
                )
            target = pd.to_numeric(
                result.frame[spec.target_column], errors="coerce"
            ).to_numpy(float)
            if spec.family in {"reliability", "overestimate_veto"}:
                prediction = pd.to_numeric(
                    result.frame[spec.prediction_columns[0]], errors="coerce"
                ).to_numpy(float)
                metrics = _binary_metrics(target, prediction)
                if spec.family == "overestimate_veto":
                    veto, actual = prediction >= 0.5, target >= 0.5
                    metrics["veto_false_negative_rate"] = float(
                        (~veto & actual).sum() / max(1, actual.sum())
                    )
            elif spec.family in {"ordinal", "quantile_ordinal_residual"}:
                prediction = result.frame.loc[
                    :, list(spec.prediction_columns)
                ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                metrics = _multiclass_metrics(target, prediction)
                metrics["ordinal_expected_mae"] = float(
                    np.abs(prediction @ np.arange(prediction.shape[1]) - target).mean()
                )
            else:
                prediction = pd.to_numeric(
                    result.frame[spec.prediction_columns[0]], errors="coerce"
                ).to_numpy(float)
                if not np.isfinite(target).all() or not np.isfinite(prediction).all():
                    raise NestedFeatureChallengerError(
                        "clipped residual target/prediction must be finite"
                    )
                error = prediction - target
                metrics = {
                    "clipped_residual_mae": float(np.abs(error).mean()),
                    "clipped_residual_rmse": float(np.sqrt(np.square(error).mean())),
                    "clipped_residual_signed_bias": float(error.mean()),
                }
            if spec.ranking_score_column is not None:
                metrics.update(
                    _top_tail_metrics(
                        result.frame, score_column=spec.ranking_score_column
                    )
                )
            heads[spec.name] = {"family": spec.family, "metrics": metrics}
        evaluations.append(
            {
                "feature_set": feature_set.as_dict(),
                "frozen_base_oof_sha256": plan.frozen_base_oof_sha256,
                "strict_oof_identity_sha256": common_hash,
                "meta": heads,
            }
        )
    return {
        "schema": EVALUATION_SCHEMA,
        "status": "evaluated",
        "side": plan.side,
        "plan_sha256": plan.plan_hash,
        "frozen_base_oof_sha256": plan.frozen_base_oof_sha256,
        "strict_oof_identity_sha256": common_hash,
        "comparison_scope": "frozen_base_sequential_meta_feature_count_only",
        "evaluations": evaluations,
    }
