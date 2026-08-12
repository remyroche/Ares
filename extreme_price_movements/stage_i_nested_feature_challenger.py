"""Checkpointed Stage-I nested feature-set over-pruning challenger.

This is a predeclared challenger contract, not another post-hoc selector.
It consumes a *completed* side-local Stage-I base-selection manifest and its
MDA audit, expands the automatic sparse selection through predeclared nested
feature counts, and makes a coupled base+meta stack evaluate every arm on the
same strict-OOF population.  It never imports or changes ``lgbm_pipeline``.

The runner is intentionally callback-driven: wiring a fitted base/meta stack is
repository-specific and must be supplied explicitly.  In particular, this
module must not silently fit a convenient surrogate or reuse Huber residual
importance for non-residual meta targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_ranking import RANKING_POLICY, stable_stage_i_rank_frame


SCHEMA = "stage_i_nested_feature_challenger_v1"
IDENTITY_COLUMNS: tuple[str, ...] = ("candidate_id", "__ts__", "__symbol__")
NESTED_SET_NAMES: tuple[str, ...] = ("automatic_sparse", "full_input_control", "top20", "top30", "top40", "top60")
NESTED_SET_SIZES: Mapping[str, int | None] = {
    "automatic_sparse": None,
    "full_input_control": None,
    "top20": 20,
    "top30": 30,
    "top40": 40,
    "top60": 60,
}


class NestedFeatureChallengerError(ValueError):
    """Raised when a diagnostic contract is incomplete or non-comparable."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _strict_true(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and float(value) == 1.0)
    return False


def _finite(value: Any, default: float = float("-inf")) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _normalise_family(feature: str) -> str:
    """Conservative fallback family when the historical audit has no group id."""
    pieces = [piece for piece in str(feature).lower().split("_") if piece]
    if not pieces:
        return "ungrouped"
    # Keep enough of a prefix to distinguish price/flow/path variants while
    # folding only common lookback suffixes (``ret4h`` vs ``ret24h``).
    first = "".join(char for char in pieces[0] if not char.isdigit()) or pieces[0]
    if len(pieces) > 1 and pieces[1] not in {"1h", "2h", "4h", "6h", "8h", "12h", "24h", "48h", "120h"}:
        return f"{first}_{pieces[1]}"
    return first


@dataclass(frozen=True)
class MDAFeatureRank:
    feature: str
    source_rank: int
    family: str
    mda_median: float
    mda_mean: float
    positive_cohort_rate: float
    worst_cohort_mda: float
    latest_cohort_mda: float
    cohort_count: int
    confidence_label: str
    stable: bool
    tier: str
    audit_observed: bool
    source_round: str = ""
    source_audit_path: str = ""
    source_audit_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class CompletedBaseSelection:
    side: str
    selection_dir: Path
    selected_features: tuple[str, ...]
    input_features: tuple[str, ...]
    source_ranks: Mapping[str, MDAFeatureRank]
    manifest_sha256: str
    audit_sha256: str
    audit_path: Path
    stability_policy: Mapping[str, Any]


@dataclass(frozen=True)
class NestedFeatureSet:
    side: str
    name: str
    requested_feature_count: int | None
    features: tuple[str, ...]
    added_features: tuple[str, ...]
    source_ranks: Mapping[str, int | None]
    feature_families: Mapping[str, str]
    family_composition: Mapping[str, int]
    tier_composition: Mapping[str, int]
    source_hash: str
    control_provenance: Mapping[str, Any] = field(default_factory=dict)
    promotion_eligible: bool = True
    source_rank_evidence: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "name": self.name,
            "requested_feature_count": self.requested_feature_count,
            "feature_count": len(self.features),
            "features": list(self.features),
            "added_features": list(self.added_features),
            "source_ranks": dict(self.source_ranks),
            "feature_families": dict(self.feature_families),
            "family_composition": dict(self.family_composition),
            "tier_composition": dict(self.tier_composition),
            "source_hash": self.source_hash,
            "control_provenance": _jsonable(self.control_provenance),
            "promotion_eligible": self.promotion_eligible,
            "source_rank_evidence": _jsonable(self.source_rank_evidence),
        }


@dataclass(frozen=True)
class NestedFeatureChallengePlan:
    side: str
    source_manifest_sha256: str
    source_audit_sha256: str
    source_audit_path: str
    required_features: tuple[str, ...]
    protected_features: tuple[str, ...]
    stability_policy: Mapping[str, Any]
    feature_sets: tuple[NestedFeatureSet, ...]

    @property
    def plan_hash(self) -> str:
        return _canonical_hash(self.as_dict(include_hash=False))

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "side": self.side,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_audit_sha256": self.source_audit_sha256,
            "source_audit_path": self.source_audit_path,
            "required_features": list(self.required_features),
            "protected_features": list(self.protected_features),
            "stability_policy": _jsonable(self.stability_policy),
            "feature_sets": [item.as_dict() for item in self.feature_sets],
        }
        if include_hash:
            value["plan_sha256"] = _canonical_hash(value)
        return value


def _round_number(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("round_"))
    except ValueError as exc:
        raise NestedFeatureChallengerError(f"invalid immutable MDA round path: {path}") from exc


def _audit_paths(selection_dir: Path, manifest: Mapping[str, Any]) -> tuple[Path, tuple[Path, ...]]:
    report_paths = sorted(selection_dir.glob("mda/**/mda_feature_selection_report.json"))
    if not report_paths:
        raise NestedFeatureChallengerError(
            f"{selection_dir}: no completed immutable MDA round report was found"
        )
    # Each staged round writes a report.  The highest round is the manifest's
    # terminal report; all sibling round audits beneath its report root remain
    # immutable evidence for features removed by later rounds.
    final_report = max(report_paths, key=lambda path: (_round_number(path), str(path)))
    report = json.loads(final_report.read_text(encoding="utf-8"))
    raw = report.get("feature_audit_path")
    candidates: list[Path] = []
    if isinstance(raw, str) and raw.strip():
        configured = Path(raw)
        candidates.append(configured if configured.is_absolute() else Path.cwd() / configured)
        candidates.append(final_report.parent / configured.name)
    candidates.extend(final_report.parent.glob("mda_feature_audit.csv"))
    valid = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    unique = list(dict.fromkeys(valid))
    if len(unique) != 1:
        raise NestedFeatureChallengerError(
            f"{selection_dir}: completed MDA feature audit is missing or ambiguous"
        )
    final_audit = unique[0]
    report_root = final_report.parent.parent
    audits = tuple(sorted((path.resolve() for path in report_root.glob("round_*/mda_feature_audit.csv")), key=lambda path: (_round_number(path), str(path))))
    if not audits or final_audit not in audits:
        raise NestedFeatureChallengerError(f"{selection_dir}: final audit is not bound beneath its immutable report root")
    return final_audit, audits


def load_completed_stage_i_base_selection(
    selection_dir: str | Path, *, side: str | None = None,
    family_overrides: Mapping[str, str] | None = None,
    min_positive_cohort_rate: float = 0.50,
    min_observed_cohorts: int = 2,
    require_nonnegative_worst_cohort: bool = True,
) -> CompletedBaseSelection:
    """Load only a completed Stage-I base selection and its exact MDA audit.

    The stability rule is intentionally explicit and auditable.  It is not a
    production feature-count floor: if it cannot populate a requested nested
    arm, materialisation fails rather than backfilling a weak feature.
    """
    root = Path(selection_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise NestedFeatureChallengerError(f"missing completed base-selection manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_side = str(manifest.get("side", "")).lower()
    requested_side = observed_side if side is None else str(side).lower()
    if (
        manifest.get("schema") != "stage_i_base_feature_selection_v1"
        or manifest.get("status") != "complete"
        or requested_side not in {"long", "short"}
        or observed_side != requested_side
    ):
        raise NestedFeatureChallengerError(f"{root}: not a completed Stage-I {requested_side} base selection")
    selected = _ordered_unique(manifest.get("selected_feature_contract", manifest.get("selected_features", ())))
    input_features = _ordered_unique(manifest.get("input_feature_contract", ()))
    if not selected or not input_features or not set(selected).issubset(input_features):
        raise NestedFeatureChallengerError(f"{root}: selected/input feature contracts are incomplete")
    audit_path, audit_paths = _audit_paths(root, manifest)
    audit_records: dict[str, tuple[pd.Series, Path, int, str]] = {}
    for path in audit_paths:
        audit = pd.read_csv(path)
        if audit.empty or "feature" not in audit:
            raise NestedFeatureChallengerError(f"{path}: empty MDA feature audit")
        audit["feature"] = audit.feature.astype(str)
        if audit.feature.duplicated().any():
            raise NestedFeatureChallengerError(f"{path}: inconsistent duplicate feature evidence within immutable round")
        round_number = _round_number(path)
        digest = _file_sha256(path)
        for _, row in audit.iterrows():
            feature = str(row["feature"])
            if "mda_feature_evaluable" in row and not bool(row.get("mda_feature_evaluable")):
                evaluated = False
            elif "mda_n_repeats" in row:
                evaluated = _finite(row.get("mda_n_repeats"), 0.0) > 0.0
            elif "mda_n_folds" in row:
                evaluated = _finite(row.get("mda_n_folds"), 0.0) > 0.0
            else:
                evaluated = _finite(row.get("mda_cohort_count"), 0.0) > 0.0
            if not evaluated:
                continue
            prior = audit_records.get(feature)
            if prior is not None and prior[2] == round_number:
                raise NestedFeatureChallengerError(f"{path}: inconsistent duplicate evaluated evidence for {feature!r}")
            if prior is None or round_number > prior[2]:
                audit_records[feature] = (row, path, round_number, digest)
    family_overrides = {str(key): str(value) for key, value in (family_overrides or {}).items()}
    ordered = sorted(
        audit_records.items(),
        key=lambda item: (
            -_finite(item[1][0].get("mda_median")), -_finite(item[1][0].get("mda_mean")),
            -_finite(item[1][0].get("mda_positive_cohort_rate")), -_finite(item[1][0].get("mda_latest_cohort_mda")), item[0],
        ),
    )
    ranks: dict[str, MDAFeatureRank] = {}
    for rank, (feature, (row, source_path, source_round, source_digest)) in enumerate(ordered, start=1):
        group = row.get("mda_group_id")
        family = family_overrides.get(feature)
        if not family:
            family = str(group) if isinstance(group, str) and group.strip() and group.lower() != "nan" else _normalise_family(feature)
        positive = _finite(row.get("mda_positive_cohort_rate"))
        observed = int(max(0, _finite(row.get("mda_cohort_count", row.get("mda_n_folds", 0)), 0.0)))
        worst = _finite(row.get("mda_worst_cohort_mda"))
        stable = positive >= min_positive_cohort_rate and observed >= min_observed_cohorts
        if require_nonnegative_worst_cohort:
            stable = stable and worst >= 0.0
        latest = _finite(row.get("mda_latest_cohort_mda"))
        median = _finite(row.get("mda_median"))
        mean = _finite(row.get("mda_mean"))
        # This is a diagnostic expansion.  Only fields which are negative in
        # every valid-cohort summary are excluded; a zero-repeat/group-skipped
        # or unstable field is an explicit uncertainty tier, not a production
        # pruning decision repeated under another name.
        consistently_negative = (
            observed >= min_observed_cohorts and median < 0.0 and mean < 0.0
            and worst < 0.0 and latest < 0.0
        )
        if stable:
            tier = "strong_stable"
        elif observed == 0 or int(max(0, _finite(row.get("mda_n_repeats"), 0.0))) == 0:
            tier = "untested_or_group_skipped"
        elif consistently_negative:
            tier = "consistently_materially_negative_excluded"
        else:
            tier = "borderline_or_uncertain"
        ranks[feature] = MDAFeatureRank(
            feature=feature, source_rank=rank, family=family,
            mda_median=median, mda_mean=mean,
            positive_cohort_rate=positive, worst_cohort_mda=worst,
            latest_cohort_mda=latest, cohort_count=observed,
            confidence_label=str(row.get("confidence_label", "")), stable=stable, tier=tier,
            audit_observed=True, source_round=f"round_{source_round:02d}",
            source_audit_path=str(source_path), source_audit_sha256=source_digest,
        )
    # A completed selector can legitimately remove a field before MDA.  Such a
    # field is precisely relevant to an over-pruning diagnostic, but it must
    # never masquerade as MDA evidence.  Append it in the frozen input-contract
    # order behind audited ranks and retain an explicit untested provenance.
    for feature in input_features:
        if feature in ranks:
            continue
        ranks[feature] = MDAFeatureRank(
            feature=feature, source_rank=len(ranks) + 1,
            family=family_overrides.get(feature, _normalise_family(feature)),
            mda_median=0.0, mda_mean=0.0, positive_cohort_rate=0.0,
            worst_cohort_mda=0.0, latest_cohort_mda=0.0, cohort_count=0,
            confidence_label="not_present_in_completed_mda_audit", stable=False,
            tier="untested_or_group_skipped", audit_observed=False,
            source_round="never_evaluated", source_audit_path="", source_audit_sha256="",
        )
    consolidated_audit_hash = _canonical_hash([
        {"path": str(path), "sha256": _file_sha256(path)} for path in audit_paths
    ])
    return CompletedBaseSelection(
        side=requested_side, selection_dir=root, selected_features=selected, input_features=input_features,
        source_ranks=ranks, manifest_sha256=_file_sha256(manifest_path), audit_sha256=consolidated_audit_hash, audit_path=audit_path,
        stability_policy={
            "minimum_positive_cohort_rate": float(min_positive_cohort_rate),
            "minimum_observed_cohorts": int(min_observed_cohorts),
            "require_nonnegative_worst_cohort": bool(require_nonnegative_worst_cohort),
        },
    )


def _family_composition(features: Sequence[str], ranks: Mapping[str, MDAFeatureRank]) -> tuple[dict[str, str], dict[str, int]]:
    families = {feature: ranks[feature].family for feature in features}
    counts: dict[str, int] = {}
    for family in families.values():
        counts[family] = counts.get(family, 0) + 1
    return families, dict(sorted(counts.items()))


def _diverse_additions(
    *, source: CompletedBaseSelection, existing: Sequence[str], requested_count: int,
) -> tuple[str, ...]:
    """Add diagnostic tiers by MDA rank while preferring the least-used family."""
    needed = max(0, int(requested_count) - len(existing))
    if not needed:
        return ()
    existing_set = set(existing)
    candidates = [
        rank for feature, rank in source.source_ranks.items()
        if (
            feature not in existing_set and feature in source.input_features
            and rank.tier != "consistently_materially_negative_excluded"
        )
    ]
    if len(candidates) < needed:
        raise NestedFeatureChallengerError(
            f"{source.side}: only {len(candidates)} diagnostically eligible MDA-ranked additions are available for top{requested_count}; "
            "the remaining input fields are consistently materially negative or absent"
        )
    _families, counts = _family_composition(existing, source.source_ranks)
    selected: list[str] = []
    remaining = list(candidates)
    while len(selected) < needed:
        tier_order = {"strong_stable": 0, "borderline_or_uncertain": 1, "untested_or_group_skipped": 2}
        choice = min(remaining, key=lambda item: (tier_order[item.tier], counts.get(item.family, 0), item.source_rank, item.feature))
        selected.append(choice.feature)
        counts[choice.family] = counts.get(choice.family, 0) + 1
        remaining.remove(choice)
    return tuple(selected)


def materialize_nested_feature_challenge(
    source: CompletedBaseSelection,
    *, required_features: Sequence[str] = (), protected_features: Sequence[str] = (),
) -> NestedFeatureChallengePlan:
    """Build automatic sparse/top20/top30/top40/top60 sets without a size floor."""
    required = _ordered_unique(required_features)
    protected = _ordered_unique(protected_features)
    mandatory = _ordered_unique((*required, *protected))
    missing = sorted(set(mandatory).difference(source.input_features))
    if missing:
        raise NestedFeatureChallengerError(
            f"{source.side}: required/protected features escape the completed input contract: {missing}"
        )
    automatic = _ordered_unique((*source.selected_features, *mandatory))
    sets: list[NestedFeatureSet] = []
    policy = {
        "ranking": "mda_median_desc_then_mean_desc_then_positive_cohort_rate_desc_then_latest_desc",
        "stability": dict(source.stability_policy),
        "eligible_tiers": ["strong_stable", "borderline_or_uncertain", "untested_or_group_skipped"],
        "excluded_tier": "consistently_materially_negative_excluded",
        "family_diversity": "least_represented_family_then_source_rank",
        "production_feature_floor": None,
        "unstable_backfill": "forbidden",
    }
    # Fixed-count arms are independent of the selected-prefix control.  Compute
    # their one diversity-aware ladder from required/protected fields only,
    # then take its deterministic prefixes so top20 ⊂ top30 ⊂ top40 ⊂ top60.
    fixed_max = max(int(value) for value in NESTED_SET_SIZES.values() if value is not None)
    fixed_ladder = _diverse_additions(source=source, existing=mandatory, requested_count=fixed_max)
    for name in NESTED_SET_NAMES:
        requested = NESTED_SET_SIZES[name]
        if name == "automatic_sparse":
            anchor, additions, features = automatic, (), automatic
        elif name == "full_input_control":
            anchor, additions, features = source.input_features, (), source.input_features
        else:
            needed = max(0, int(requested) - len(mandatory))
            anchor, additions = mandatory, fixed_ladder[:needed]
            features = _ordered_unique((*anchor, *additions))
        families, composition = _family_composition(features, source.source_ranks)
        tier_composition: dict[str, int] = {}
        for feature in features:
            if name == "automatic_sparse" and feature in automatic:
                tier = "selected_automatic_sparse"
            elif name == "full_input_control":
                tier = f"full_input_control__{source.source_ranks[feature].tier}"
            elif feature in mandatory:
                tier = "mandatory_required_or_protected"
            else:
                tier = source.source_ranks[feature].tier
            tier_composition[tier] = tier_composition.get(tier, 0) + 1
        source_ranks = {feature: source.source_ranks[feature].source_rank for feature in features}
        source_rank_evidence = {feature: source.source_ranks[feature].as_dict() for feature in features}
        set_hash = _canonical_hash({"features": features, "source_ranks": source_ranks, "source_audit_sha256": source.audit_sha256})
        control_provenance = (
            {
                "kind": "full_input_control",
                "source": "completed_stage_i_authorized_side_input_feature_contract",
                "postscreen_bypass": True,
                "promotion_policy": "eligible_only_if_best_under_identical_strict_OOF_and_OOS_gates; no_post_test_tuning",
            }
            if name == "full_input_control" else {}
        )
        sets.append(NestedFeatureSet(
            side=source.side, name=name, requested_feature_count=requested, features=features,
            added_features=additions, source_ranks=source_ranks, feature_families=families,
            family_composition=composition, source_hash=set_hash,
            tier_composition=dict(sorted(tier_composition.items())),
            control_provenance=control_provenance,
            # The roadmap asks to keep the best ablation.  Full input is an
            # expensive control, but excluding it from promotion before seeing
            # the result would make that comparison non-decisive.  It remains
            # subject to the identical nested/OOS gates and cannot be tuned
            # after evaluation.
            promotion_eligible=True,
            source_rank_evidence=source_rank_evidence,
        ))
    return NestedFeatureChallengePlan(
        side=source.side, source_manifest_sha256=source.manifest_sha256, source_audit_sha256=source.audit_sha256,
        source_audit_path=str(source.audit_path), required_features=required, protected_features=protected,
        stability_policy=policy, feature_sets=tuple(sets),
    )


def checkpoint_nested_feature_plan(plan: NestedFeatureChallengePlan, output_dir: str | Path) -> Path:
    """Persist an immutable materialisation checkpoint; existing drift is fatal."""
    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    payload = plan.as_dict()
    if destination.exists():
        if not manifest_path.is_file():
            raise NestedFeatureChallengerError(f"checkpoint exists without manifest: {destination}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        sets_path = destination / "nested_feature_sets.json"
        if (
            previous.get("plan_sha256") != payload["plan_sha256"]
            or not sets_path.is_file()
            or previous.get("nested_feature_sets_sha256") != _file_sha256(sets_path)
        ):
            raise NestedFeatureChallengerError(f"checkpoint drift: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / "nested_feature_sets.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        manifest = {
            "schema": SCHEMA, "status": "materialized", "plan_sha256": payload["plan_sha256"],
            "nested_feature_sets_sha256": _file_sha256(temporary / "nested_feature_sets.json"),
            "side": plan.side, "source_manifest_sha256": plan.source_manifest_sha256,
            "source_audit_sha256": plan.source_audit_sha256,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return destination


@dataclass(frozen=True)
class StrictOOFResult:
    frame: pd.DataFrame
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class MetaTargetMetricSpec:
    name: str
    family: str
    target_column: str
    prediction_columns: tuple[str, ...]
    ranking_score_column: str | None = None
    clip_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.family not in {
            "reliability", "overestimate_veto", "ordinal",
            "quantile_ordinal_residual", "clipped_residual",
        }:
            raise NestedFeatureChallengerError(f"unsupported meta target family: {self.family}")
        if not self.name or not self.target_column or not self.prediction_columns:
            raise NestedFeatureChallengerError("meta metric specs require name, target, and prediction columns")
        if self.family in {"reliability", "overestimate_veto", "clipped_residual"} and len(self.prediction_columns) != 1:
            raise NestedFeatureChallengerError(f"{self.family} requires exactly one prediction column")
        if self.family in {"ordinal", "quantile_ordinal_residual"} and len(self.prediction_columns) < 2:
            raise NestedFeatureChallengerError("ordinal requires a probability simplex")


BaseStackHook = Callable[[NestedFeatureSet], StrictOOFResult]
MetaStackHook = Callable[[NestedFeatureSet, StrictOOFResult, MetaTargetMetricSpec], StrictOOFResult]


def _identity_hash(frame: pd.DataFrame) -> str:
    missing = sorted(set(IDENTITY_COLUMNS).difference(frame.columns))
    if missing or frame.loc[:, list(IDENTITY_COLUMNS)].isna().any().any():
        raise NestedFeatureChallengerError(f"strict OOF identity is incomplete: {missing}")
    identity = frame.loc[:, list(IDENTITY_COLUMNS)]
    if identity.duplicated().any():
        raise NestedFeatureChallengerError("strict OOF identity is not unique")
    return sha256(pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype=np.uint64).tobytes()).hexdigest()


def _validate_strict(result: StrictOOFResult, *, side: str, layer: str) -> str:
    if not isinstance(result.frame, pd.DataFrame) or result.frame.empty:
        raise NestedFeatureChallengerError(f"{layer}: strict OOF result is empty")
    if not _strict_true(result.provenance.get("strict_oof")):
        raise NestedFeatureChallengerError(f"{layer}: strict_oof=true provenance is required")
    if str(result.provenance.get("side", "")).lower() != side:
        raise NestedFeatureChallengerError(f"{layer}: side provenance mismatch")
    if str(result.provenance.get("layer", "")).lower() != layer:
        raise NestedFeatureChallengerError(f"{layer}: layer provenance mismatch")
    return _identity_hash(result.frame)


def _multiclass_metrics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if probabilities.ndim != 2 or len(target) != len(probabilities) or probabilities.shape[1] < 2:
        raise NestedFeatureChallengerError("multiclass metric input is malformed")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise NestedFeatureChallengerError("multiclass probabilities must be a finite simplex")
    target = np.asarray(target, dtype=np.int64)
    if (target < 0).any() or (target >= probabilities.shape[1]).any():
        raise NestedFeatureChallengerError("multiclass target escapes prediction support")
    p = np.clip(probabilities[np.arange(len(target)), target], 1e-15, 1.0)
    one_hot = np.eye(probabilities.shape[1], dtype=float)[target]
    return {
        "multiclass_log_loss": float(-np.log(p).mean()),
        "multiclass_brier": float(np.square(probabilities - one_hot).sum(axis=1).mean() / probabilities.shape[1]),
    }


def _binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isin(y, (0.0, 1.0)).all() or ((p < 0) | (p > 1)).any():
        raise NestedFeatureChallengerError("binary target/prediction must be finite in [0, 1]")
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    out = {"brier": float(np.square(p - y).mean()), "log_loss": float(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean())}
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for bucket in range(10):
        mask = bins == bucket
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean() - y[mask].mean()))
    out["ece_10"] = ece
    return out


def _top_tail_metrics(frame: pd.DataFrame, *, score_column: str) -> dict[str, float]:
    required = {score_column, "exact_net_bps"}
    if missing := sorted(required.difference(frame.columns)):
        raise NestedFeatureChallengerError(f"exact-net tail metrics lack columns: {missing}")
    work = frame.loc[:, list(IDENTITY_COLUMNS) + [score_column, "exact_net_bps"]].copy()
    score = pd.to_numeric(work[score_column], errors="coerce")
    net = pd.to_numeric(work.exact_net_bps, errors="coerce")
    if score.isna().any() or net.isna().any():
        raise NestedFeatureChallengerError("exact-net tail metrics require finite score and net")
    work[score_column], work["exact_net_bps"] = score, net
    order = stable_stage_i_rank_frame(work, score_column=score_column)
    out: dict[str, float] = {}
    for fraction in (0.01, 0.05, 0.10, 0.20):
        k = max(1, int(math.ceil(fraction * len(order))))
        out[f"exact_net_top_{int(fraction * 100):02d}_bps"] = float(order.head(k).exact_net_bps.mean())
        out[f"exact_net_top_{int(fraction * 100):02d}_rows"] = float(k)
    out["ranking_tie_policy"] = RANKING_POLICY
    return out


def evaluate_nested_feature_challenge(
    plan: NestedFeatureChallengePlan,
    *, base_hook: BaseStackHook, meta_hook: MetaStackHook,
    meta_specs: Sequence[MetaTargetMetricSpec],
) -> dict[str, Any]:
    """Evaluate only matched strict-OOF base+meta stack outputs for a plan."""
    if not callable(base_hook) or not callable(meta_hook):
        raise NestedFeatureChallengerError("explicit base and meta stack hooks are required")
    if not meta_specs:
        raise NestedFeatureChallengerError("at least one target-specific meta metric spec is required")
    evaluations: list[dict[str, Any]] = []
    common_hash: str | None = None
    for feature_set in plan.feature_sets:
        base = base_hook(feature_set)
        if not isinstance(base, StrictOOFResult):
            raise NestedFeatureChallengerError("base stack hook must return StrictOOFResult")
        base_hash = _validate_strict(base, side=plan.side, layer="base")
        if common_hash is None:
            common_hash = base_hash
        elif base_hash != common_hash:
            raise NestedFeatureChallengerError("nested base arms do not share identical strict OOF rows")
        required_base = {"r3_class", "r3_p_adverse", "r3_p_weak", "r3_p_clear", "exact_net_bps"}
        if missing := sorted(required_base.difference(base.frame.columns)):
            raise NestedFeatureChallengerError(f"base: R3 metric contract lacks {missing}")
        base_frame = base.frame.copy()
        r3 = pd.to_numeric(base_frame.r3_class, errors="coerce").to_numpy()
        probability = base_frame.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(r3).all():
            raise NestedFeatureChallengerError("base: R3 target is non-finite")
        base_metrics = _multiclass_metrics(r3, probability)
        tail_score = "base_score_bps" if "base_score_bps" in base_frame else "r3_opportunity_score"
        if tail_score not in base_frame:
            base_frame["r3_opportunity_score"] = probability[:, 2] - probability[:, 0]
        base_metrics.update(_top_tail_metrics(base_frame, score_column=tail_score))
        meta_results: dict[str, Any] = {}
        for spec in meta_specs:
            meta = meta_hook(feature_set, base, spec)
            if not isinstance(meta, StrictOOFResult):
                raise NestedFeatureChallengerError("meta stack hook must return StrictOOFResult")
            meta_hash = _validate_strict(meta, side=plan.side, layer="meta")
            if meta_hash != base_hash:
                raise NestedFeatureChallengerError("base/meta stack outputs are not on identical strict OOF rows")
            columns = {spec.target_column, *spec.prediction_columns}
            if missing := sorted(columns.difference(meta.frame.columns)):
                raise NestedFeatureChallengerError(f"{spec.name}: target-specific metric contract lacks {missing}")
            target = pd.to_numeric(meta.frame[spec.target_column], errors="coerce").to_numpy(float)
            if spec.family in {"reliability", "overestimate_veto"}:
                metrics = _binary_metrics(target, pd.to_numeric(meta.frame[spec.prediction_columns[0]], errors="coerce").to_numpy(float))
                if spec.family == "overestimate_veto":
                    veto = pd.to_numeric(meta.frame[spec.prediction_columns[0]], errors="coerce").to_numpy(float) >= 0.5
                    actual = target >= 0.5
                    metrics["veto_false_negative_rate"] = float((~veto & actual).sum() / max(1, actual.sum()))
            elif spec.family in {"ordinal", "quantile_ordinal_residual"}:
                probability = meta.frame.loc[:, list(spec.prediction_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                metrics = _multiclass_metrics(target, probability)
                expected = probability @ np.arange(probability.shape[1], dtype=float)
                metrics["ordinal_expected_mae"] = float(np.abs(expected - target).mean())
            else:
                prediction = pd.to_numeric(meta.frame[spec.prediction_columns[0]], errors="coerce").to_numpy(float)
                if not np.isfinite(target).all() or not np.isfinite(prediction).all():
                    raise NestedFeatureChallengerError("clipped residual metric requires finite target/prediction")
                if spec.clip_bounds is not None:
                    lower, upper = spec.clip_bounds
                    if (target < lower).any() or (target > upper).any():
                        raise NestedFeatureChallengerError("clipped residual target escapes declared clip bounds")
                error = prediction - target
                metrics = {"clipped_residual_mae": float(np.abs(error).mean()), "clipped_residual_rmse": float(np.sqrt(np.square(error).mean())), "clipped_residual_signed_bias": float(error.mean())}
            if spec.ranking_score_column is not None:
                metrics.update(_top_tail_metrics(meta.frame, score_column=spec.ranking_score_column))
            meta_results[spec.name] = {"family": spec.family, "metrics": metrics}
        evaluations.append({"feature_set": feature_set.as_dict(), "strict_oof_identity_sha256": base_hash, "base": base_metrics, "meta": meta_results})
    return {
        "schema": SCHEMA, "status": "evaluated", "side": plan.side, "plan_sha256": plan.plan_hash,
        "strict_oof_identity_sha256": common_hash, "evaluations": evaluations,
    }
