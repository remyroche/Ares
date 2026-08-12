"""Artifact-producing Stage-VI causal/path archetype execution.

This module turns the reusable representation primitives in
``stage_vi_archetypes`` into one bounded, auditable experiment.  It performs
no work on import and never chooses a live routing policy: every fitted arm is
side-local, discovery uses prior-resolved positive labels only, and realised
path memberships can reach model comparisons only through strict-OOF causal
recogniser probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_vi_archetypes import (
    ArchetypeConfig,
    ArchetypeDecisionConfig,
    ArchetypeView,
    ArchetypeWeightConfig,
    CURRENT_ARCHETYPE_TOKENS,
    FORBIDDEN_CAUSAL_TOKENS,
    STAGE_VI_SCHEMA,
    archetype_economic_separation,
    archetype_fold_stability,
    materialize_archetype_decision_matrix,
    remove_current_archetype_columns,
    run_matched_incremental_archetype_comparison,
    strict_oof_archetype_features,
)


RUNNER_SCHEMA = "stage_vi_artifact_runner_v1"
VIEW_SCHEMA = "stage_vi_cf_pf_view_contract_v1"
CAUSAL_COMPONENTS = (3, 4, 5, 6)
PATH_COMPONENTS = (3, 4, 5, 6, 8)
METHODS = ("kmeans", "gmm_diag", "gmm_full", "gmm_pca_diag", "ae_gmm_diag")
AW_CONTRACTS: Mapping[str, str] = {
    "AW0": "uniform",
    "AW1": "time_balanced",
    "AW2": "symbol_balanced",
    "AW3": "mandatory_side_local_fit",
    "AW4": "path_certainty",
    "AW5": "economic_diversity",
}


class StageVIRunnerError(ValueError):
    """Raised when an input or execution contract is incomplete."""


@dataclass(frozen=True)
class StageVIArmTrainingRequest:
    """One candidate-specific matched strict-OOF refit request."""

    candidate_id: str
    ledger: pd.DataFrame
    archetype_features: pd.DataFrame
    archetype_feature_columns: tuple[str, ...]
    candidate_ids: np.ndarray
    decision_timestamps: np.ndarray
    side_names: np.ndarray
    feature_sha256: str


@dataclass(frozen=True)
class StageVIArmTrainingResult:
    """Candidate-specific OOF scores returned by a downstream trainer."""

    candidate_ids: Sequence[Any]
    scores: Mapping[str, Sequence[float]]
    oof_flags: Mapping[str, Sequence[bool] | Sequence[int]]
    provenance: Mapping[str, Any]


ArmTrainer = Callable[[StageVIArmTrainingRequest], StageVIArmTrainingResult]


_CF1_TOKENS = (
    "atr", "barrier", "tp_", "sl_", "trend", "volatility", "rv_", "vol_",
    "volume", "breadth", "entry", "breakout", "range_", "dist_ema", "vwap",
)
_CF2_TOKENS = (
    "p_upper", "p_lower", "p_timeout", "p_adverse", "p_weak", "p_clear",
    "entropy", "top2_margin", "max_probability", "expected_net", "mapped_ev",
    "uncertainty", "cost_clear", "base_score", "opportunity_score", "trust",
)
_CF3_TOKENS = (
    "regime", "context", "market_", "breadth", "cross_asset", "cycle",
    "vol_state", "liquidity", "session", "hour_", "dow_", "spectral",
)

_PATH_ALIASES: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "PF0": (
        ("event_upper", "upper_touch", "upper_hit"),
        ("event_lower", "lower_touch", "lower_hit"),
        ("event_timeout", "timeout", "timeout_flag"),
        ("time_to_first_touch", "first_touch_time", "bars_to_first_touch"),
        ("same_bar_conflict", "same_bar_both_touch"),
        ("terminal_return", "exact_gross_bps", "gross_return"),
    ),
    "PF1": (
        ("mfe", "path_mfe", "mfe_atr", "mfe_12h_atr"),
        ("mae", "path_mae", "mae_atr", "mae_12h_atr"),
        ("mae_before_mfe", "mae_before_mfe_atr"),
        ("time_to_mfe", "bars_to_mfe"),
        ("time_to_mae", "bars_to_mae"),
    ),
    "PF2": (
        ("terminal_peak_ratio", "terminal_to_peak_ratio", "retention_ratio"),
        ("post_clear_change", "post_clear_return"),
        ("giveback_fraction", "giveback"),
        ("retained_positive_gross", "retained_gross"),
        ("retained_positive_net", "retained_net"),
    ),
    "PF3": (
        ("path_efficiency", "path_efficiency_12", "path_efficiency_24"),
        ("directional_consistency", "path_directional_consistency"),
        ("future_slope", "future_slope_atr_per_hour"),
        ("future_slope_r2", "slope_r2"),
        ("reversal_count", "path_reversal_count"),
        ("jump_concentration", "path_jump_concentration"),
        ("future_volatility", "path_future_volatility"),
    ),
}


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


def _expand_config_keys(config: Mapping[str, Any], roots: Sequence[str]) -> tuple[str, ...]:
    """Expand nested feature-key groups without interpreting arbitrary config."""

    result: list[str] = []
    active: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str) and value in config and value not in active:
            active.add(value)
            visit(config[value])
            active.remove(value)
        elif isinstance(value, str):
            result.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)

    for root in roots:
        visit(config.get(str(root), root))
    return _ordered_unique(result)


def _token_view(pool: Sequence[str], tokens: Sequence[str], *, cap: int) -> tuple[str, ...]:
    selected = [name for name in pool if any(token in name.lower() for token in tokens)]
    return tuple(selected[: int(cap)])


def _alias_view(frame: pd.DataFrame, groups: Sequence[Sequence[str]]) -> tuple[str, ...]:
    columns: list[str] = []
    for aliases in groups:
        match = next(
            (
                name for name in aliases
                if name in frame and pd.api.types.is_numeric_dtype(frame[name])
            ),
            None,
        )
        if match is not None:
            columns.append(match)
    return _ordered_unique(columns)


@dataclass(frozen=True)
class StageVIViewContract:
    causal_views: Mapping[str, tuple[str, ...]]
    path_views: Mapping[str, tuple[str, ...]]
    causal_recogniser_columns: tuple[str, ...]
    selected_causal_columns: tuple[str, ...]
    config_feature_roots: tuple[str, ...]
    multiview_sources: Mapping[str, tuple[str, ...]]

    def validate(self, frame: pd.DataFrame) -> None:
        if set(self.causal_views) != {f"CF{i}" for i in range(5)}:
            raise StageVIRunnerError("Stage-VI view contract requires exactly CF0-CF4")
        if set(self.path_views) != {f"PF{i}" for i in range(5)}:
            raise StageVIRunnerError("Stage-VI view contract requires exactly PF0-PF4")
        if not self.causal_recogniser_columns:
            raise StageVIRunnerError("path workstream requires causal recogniser columns")
        for name, columns in {**self.causal_views, **self.path_views}.items():
            if not columns:
                raise StageVIRunnerError(f"{name} resolved no source columns")
            missing = [column for column in columns if column not in frame]
            if missing:
                raise StageVIRunnerError(f"{name} source columns are absent: {missing[:8]}")
            if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in columns):
                raise StageVIRunnerError(f"{name} must contain numeric columns only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VIEW_SCHEMA,
            "causal_views": {key: list(value) for key, value in self.causal_views.items()},
            "path_views": {key: list(value) for key, value in self.path_views.items()},
            "causal_recogniser_columns": list(self.causal_recogniser_columns),
            "selected_causal_columns": list(self.selected_causal_columns),
            "config_feature_roots": list(self.config_feature_roots),
            "multiview_sources": {key: list(value) for key, value in self.multiview_sources.items()},
            "cf4_policy": "compact_regularized_multiview_union_for_fold_local_embedding",
            "pf4_policy": "compact_standardized_pf0_pf3_union",
        }


def materialize_stage_vi_view_contract(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    selected_causal_columns: Sequence[str],
    feature_roots: Sequence[str] = (
        "base_shared_feature_keys", "base_long_feature_keys", "base_short_feature_keys",
        "meta_shared_feature_keys", "meta_product_feature_keys",
    ),
    explicit_path_views: Mapping[str, Sequence[str]] | None = None,
    max_cf0_features: int = 64,
    max_per_causal_subview: int = 16,
    causal_recogniser_columns: Sequence[str] | None = None,
) -> StageVIViewContract:
    """Resolve the declared CF/PF views against config feature-key universes."""

    available_numeric = {
        str(column) for column in frame
        if pd.api.types.is_numeric_dtype(frame[column])
    }
    config_pool = _expand_config_keys(config, feature_roots)
    allowed = set(remove_current_archetype_columns(config_pool))
    selected = _ordered_unique(
        column for column in selected_causal_columns
        if column in allowed and column in available_numeric
        and not any(token in column.lower() for token in FORBIDDEN_CAUSAL_TOKENS)
        and not any(token in column.lower() for token in CURRENT_ARCHETYPE_TOKENS)
    )
    if not selected:
        raise StageVIRunnerError(
            "CF0 requires a compact non-empty training-only selection from config feature keys"
        )
    cf0 = selected[: int(max_cf0_features)]
    cf1 = _token_view(selected, _CF1_TOKENS, cap=max_per_causal_subview)
    cf2_pool = _ordered_unique([
        *selected,
        *sorted(
            column for column in available_numeric
            if not any(token in column.lower() for token in FORBIDDEN_CAUSAL_TOKENS)
            and not any(token in column.lower() for token in CURRENT_ARCHETYPE_TOKENS)
        ),
    ])
    cf2 = _token_view(cf2_pool, _CF2_TOKENS, cap=max_per_causal_subview)
    cf3 = _token_view(selected, _CF3_TOKENS, cap=max_per_causal_subview)
    if not cf1 or not cf2 or not cf3:
        raise StageVIRunnerError("CF1-CF3 require setup, trust/base-output, and regime support")
    cf4 = _ordered_unique([*cf1[:8], *cf2[:8], *cf3[:8]])

    path_views: dict[str, tuple[str, ...]] = {}
    explicit = dict(explicit_path_views or {})
    for name in ("PF0", "PF1", "PF2", "PF3"):
        path_views[name] = (
            _ordered_unique(explicit[name])
            if name in explicit
            else _alias_view(frame, _PATH_ALIASES[name])
        )
    path_views["PF4"] = (
        _ordered_unique(explicit["PF4"])
        if "PF4" in explicit
        else _ordered_unique(
            [column for name in ("PF0", "PF1", "PF2", "PF3") for column in path_views[name]]
        )
    )
    recogniser = _ordered_unique(causal_recogniser_columns or cf4)
    contract = StageVIViewContract(
        causal_views={"CF0": cf0, "CF1": cf1, "CF2": cf2, "CF3": cf3, "CF4": cf4},
        path_views=path_views,
        causal_recogniser_columns=recogniser,
        selected_causal_columns=selected,
        config_feature_roots=tuple(map(str, feature_roots)),
        multiview_sources={"setup": cf1, "base_trust": cf2, "regime": cf3},
    )
    contract.validate(frame)
    return contract


@dataclass(frozen=True)
class StageVIRunnerSpec:
    candidate_id_column: str = "candidate_id"
    symbol_column: str = "symbol"
    decision_ts_column: str = "decision_ts"
    label_available_ts_column: str = "label_available_ts"
    side_column: str = "side_name"
    exact_net_column: str = "exact_net_bps"
    exact_gross_column: str = "exact_gross_bps"
    path_certainty_column: str = "path_certainty"
    economic_bucket_column: str = "economic_bucket"
    causal_components: tuple[int, ...] = CAUSAL_COMPONENTS
    path_components: tuple[int, ...] = PATH_COMPONENTS
    methods: tuple[str, ...] = METHODS
    folds: int = 4
    min_side_rows: int = 250
    min_component_rows: int = 20
    random_state: int = 20260803
    full_grid: bool = False
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.10)
    outcome_columns: tuple[str, ...] = ("exact_gross_bps", "exact_net_bps")
    arm_score_columns: Mapping[str, str] = field(default_factory=lambda: {
        "control": "control_score", "base": "base_archetype_score",
        "meta": "meta_archetype_score", "both": "both_archetype_score",
    })
    arm_score_columns_by_candidate: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    arm_oof_flag_columns: Mapping[str, str] = field(default_factory=lambda: {
        "control": "control_is_strict_oof", "base": "base_is_strict_oof",
        "meta": "meta_is_strict_oof", "both": "both_is_strict_oof",
    })
    arm_oof_flag_columns_by_candidate: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if tuple(self.causal_components) != CAUSAL_COMPONENTS:
            raise StageVIRunnerError("causal Stage-VI grid must be exactly K={3,4,5,6}")
        if tuple(self.path_components) != PATH_COMPONENTS:
            raise StageVIRunnerError("path Stage-VI grid must be exactly K={3,4,5,6,8}")
        if set(self.arm_score_columns) != {"control", "base", "meta", "both"}:
            raise StageVIRunnerError("matched controls require control/base/meta/both")
        if set(self.arm_oof_flag_columns) != {"control", "base", "meta", "both"}:
            raise StageVIRunnerError("matched controls require four strict-OOF lineage flags")
        for candidate, columns in self.arm_score_columns_by_candidate.items():
            if not candidate or set(columns) != {"control", "base", "meta", "both"}:
                raise StageVIRunnerError(
                    "candidate-specific matched scores require control/base/meta/both"
                )
        for candidate, columns in self.arm_oof_flag_columns_by_candidate.items():
            if not candidate or set(columns) != {"control", "base", "meta", "both"}:
                raise StageVIRunnerError(
                    "candidate-specific strict-OOF flags require control/base/meta/both"
                )
        if int(self.folds) < 2:
            raise StageVIRunnerError("Stage-VI requires at least two strict chronological folds")
        if int(self.min_side_rows) < max(self.path_components) * int(self.min_component_rows):
            raise StageVIRunnerError(
                "min_side_rows must support every requested path component"
            )


@dataclass(frozen=True)
class StageVIRunResult:
    output_directory: Path
    candidate_audit: pd.DataFrame
    comparison: pd.DataFrame
    decision_matrix: pd.DataFrame
    manifest: Mapping[str, Any]


def _identity_sha256(frame: pd.DataFrame, spec: StageVIRunnerSpec) -> str:
    columns = [spec.candidate_id_column, spec.symbol_column, spec.decision_ts_column, spec.side_column]
    work = frame.loc[:, columns].copy()
    work[spec.decision_ts_column] = pd.to_datetime(
        work[spec.decision_ts_column], utc=True, errors="coerce"
    )
    if work.isna().any().any() or work.duplicated(columns).any():
        raise StageVIRunnerError("Stage-VI requires unique non-null immutable identities")
    work = work.astype("string").sort_values(columns, kind="stable")
    return sha256(pd.util.hash_pandas_object(work, index=False).values.tobytes()).hexdigest()


def _require_strict_oof_arm_lineage(
    frame: pd.DataFrame, columns: Mapping[str, str]
) -> None:
    for arm, column in columns.items():
        if column not in frame:
            raise StageVIRunnerError(f"matched {arm} control lacks strict-OOF flag {column}")
        values = frame[column].to_numpy(dtype=object)
        if not all(
            isinstance(value, (bool, np.bool_))
            or isinstance(value, (int, np.integer)) and int(value) in (0, 1)
            for value in values
        ):
            raise StageVIRunnerError(f"{column} must contain only bool or integer 0/1")
        if not np.asarray(values, dtype=bool).all():
            raise StageVIRunnerError(f"matched {arm} scores must be strict OOF on every row")


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Stable content hash including column order and values."""

    payload = json.dumps(list(map(str, frame.columns)), separators=(",", ":")).encode()
    hashed = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(
        dtype=np.uint64
    )
    return sha256(payload + hashed.tobytes()).hexdigest()


def _positive_discovery_support(
    frame: pd.DataFrame, spec: StageVIRunnerSpec
) -> pd.DataFrame:
    decision = pd.to_datetime(frame[spec.decision_ts_column], utc=True, errors="coerce")
    outcome = pd.to_numeric(frame[spec.exact_net_column], errors="coerce")
    side = frame[spec.side_column].astype(str).str.strip().str.lower()
    if decision.isna().any() or outcome.isna().any() or not set(side).issubset({"long", "short"}):
        raise StageVIRunnerError("positive discovery support requires finite time/outcome and exact sides")
    unique = np.sort(pd.unique(decision.to_numpy(dtype="datetime64[ns]")))
    bins = min(max(2, int(spec.folds)), len(unique))
    boundaries = np.linspace(0, len(unique), bins + 1, dtype=int)
    records: list[dict[str, Any]] = []
    for side_name in ("long", "short"):
        positive = side.eq(side_name) & outcome.gt(0.0)
        if int(positive.sum()) < int(spec.min_side_rows):
            raise StageVIRunnerError(
                f"{side_name} lacks minimum positive-label discovery support"
            )
        occupied = 0
        per_bin: list[int] = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            lower = pd.Timestamp(unique[start], tz="UTC")
            upper = (
                pd.Timestamp(unique[stop], tz="UTC")
                if stop < len(unique) else decision.max() + pd.Timedelta(nanoseconds=1)
            )
            count = int((positive & decision.ge(lower) & decision.lt(upper)).sum())
            per_bin.append(count)
            occupied += int(count > 0)
        if occupied != bins:
            raise StageVIRunnerError(
                f"{side_name} positive-label discovery rows are not spread across time"
            )
        records.append({
            "side": side_name, "positive_rows": int(positive.sum()),
            "positive_start_utc": decision.loc[positive].min().isoformat(),
            "positive_end_utc": decision.loc[positive].max().isoformat(),
            "time_bins": bins, "occupied_time_bins": occupied,
            "positive_rows_by_time_bin": json.dumps(per_bin),
        })
    return pd.DataFrame(records)


def _candidate_configs(
    views: StageVIViewContract, spec: StageVIRunnerSpec
) -> list[tuple[str, ArchetypeConfig]]:
    output: list[tuple[str, ArchetypeConfig]] = []
    for kind, mapping, components in (
        ("causal", views.causal_views, spec.causal_components),
        ("path", views.path_views, spec.path_components),
    ):
        weight_ids = ("AW0", "AW1", "AW2") if kind == "causal" else ("AW0", "AW1", "AW2", "AW4", "AW5")
        for view_name, columns in mapping.items():
            for method in spec.methods:
                for k in components:
                    for aw in weight_ids:
                        mode = AW_CONTRACTS[aw]
                        weights = ArchetypeWeightConfig(
                            mode=mode,  # type: ignore[arg-type]
                            timestamp_col=spec.decision_ts_column,
                            symbol_col=spec.symbol_column,
                            path_certainty_col=(spec.path_certainty_column if aw == "AW4" else None),
                            economic_bucket_col=(spec.economic_bucket_column if aw == "AW5" else None),
                        )
                        config = ArchetypeConfig(
                            view=ArchetypeView(view_name, tuple(columns), kind),  # type: ignore[arg-type]
                            method=method, components=k, side_col=spec.side_column,
                            decision_ts_col=spec.decision_ts_column,
                            label_available_ts_col=spec.label_available_ts_column,
                            positive_label_col=spec.exact_net_column,
                            min_side_rows=spec.min_side_rows,
                            min_component_rows=spec.min_component_rows,
                            random_state=spec.random_state, weights=weights,
                        )
                        config.validate()
                        output.append((f"{view_name}__{method}__k{k}__{aw}", config))
    return output


def stage_vi_candidate_grid(
    views: StageVIViewContract, spec: StageVIRunnerSpec = StageVIRunnerSpec()
) -> list[tuple[str, ArchetypeConfig]]:
    """Return the complete preregistered CF/PF × method × K × AW grid."""

    spec.validate()
    return _candidate_configs(views, spec)


def stage_vi_sequential_funnel_grid(
    views: StageVIViewContract, spec: StageVIRunnerSpec = StageVIRunnerSpec()
) -> list[tuple[str, ArchetypeConfig]]:
    """Return the bounded default funnel; exhaustive search is explicit opt-in.

    One baseline is fitted per view, then one axis at a time is varied on the
    compact CF4/PF4 views.  This cuts the default from 925 fits to 31 while
    retaining coverage of every method, legal K and applicable AW mode.
    """

    complete = dict(stage_vi_candidate_grid(views, spec))
    ids: list[str] = [
        f"{view}__gmm_diag__k4__AW0"
        for view in (*views.causal_views, *views.path_views)
    ]
    ids.extend(f"CF4__gmm_diag__k{k}__AW0" for k in spec.causal_components)
    ids.extend(f"PF4__gmm_diag__k{k}__AW0" for k in spec.path_components)
    ids.extend(f"CF4__{method}__k4__AW0" for method in spec.methods)
    ids.extend(f"PF4__{method}__k4__AW0" for method in spec.methods)
    ids.extend(f"CF4__gmm_diag__k4__{aw}" for aw in ("AW0", "AW1", "AW2"))
    ids.extend(
        f"PF4__gmm_diag__k4__{aw}"
        for aw in ("AW0", "AW1", "AW2", "AW4", "AW5")
    )
    selected = _ordered_unique(ids)
    return [(candidate_id, complete[candidate_id]) for candidate_id in selected]


def _validate_score_contracts(
    ledger: pd.DataFrame,
    candidates: Sequence[tuple[str, ArchetypeConfig]],
    spec: StageVIRunnerSpec,
    arm_trainer: ArmTrainer | None,
) -> None:
    if arm_trainer is not None:
        return
    candidate_ids = [candidate_id for candidate_id, _config in candidates]
    missing_scores = [
        candidate_id for candidate_id in candidate_ids
        if candidate_id not in spec.arm_score_columns_by_candidate
    ]
    missing_flags = [
        candidate_id for candidate_id in candidate_ids
        if candidate_id not in spec.arm_oof_flag_columns_by_candidate
    ]
    if missing_scores or missing_flags:
        raise StageVIRunnerError(
            "every candidate requires an explicit candidate-specific score and "
            "strict-OOF contract; generic scores would create false attribution"
        )
    if len(candidate_ids) > 1:
        augmented_mappings = [
            tuple(
                spec.arm_score_columns_by_candidate[candidate_id][arm]
                for arm in ("base", "meta", "both")
            )
            for candidate_id in candidate_ids
        ]
        if len(set(augmented_mappings)) != len(augmented_mappings):
            raise StageVIRunnerError(
                "candidate-specific base/meta/both score mappings must be distinct"
            )
    required: list[str] = []
    for candidate_id in candidate_ids:
        scores = spec.arm_score_columns_by_candidate[candidate_id]
        flags = spec.arm_oof_flag_columns_by_candidate[candidate_id]
        required.extend(scores.values())
        required.extend(flags.values())
    missing = [column for column in _ordered_unique(required) if column not in ledger]
    if missing:
        raise StageVIRunnerError(
            f"Stage-VI score contract references absent columns: {missing[:12]}"
        )
    for candidate_id in candidate_ids:
        flags = spec.arm_oof_flag_columns_by_candidate[candidate_id]
        _require_strict_oof_arm_lineage(ledger, flags)


def _trained_comparison_ledger(
    request: StageVIArmTrainingRequest,
    result: StageVIArmTrainingResult,
) -> tuple[pd.DataFrame, Mapping[str, str], Mapping[str, Any]]:
    if list(result.candidate_ids) != list(request.candidate_ids):
        raise StageVIRunnerError("arm trainer changed candidate identity or row order")
    arms = {"control", "base", "meta", "both"}
    if set(result.scores) != arms or set(result.oof_flags) != arms:
        raise StageVIRunnerError("arm trainer must return control/base/meta/both scores and flags")
    provenance = dict(result.provenance)
    expected_usage = {
        "control": "none", "base": "base", "meta": "meta",
        "both": "base_and_meta",
    }
    if (
        provenance.get("strict_oof") is not True
        or provenance.get("candidate_feature_sha256") != request.feature_sha256
        or provenance.get("archetype_feature_columns")
        != list(request.archetype_feature_columns)
        or provenance.get("arm_feature_usage") != expected_usage
    ):
        raise StageVIRunnerError(
            "arm trainer provenance must bind strict-OOF arm fits to candidate "
            "feature hash and column list"
        )
    output = request.ledger.copy()
    score_columns: dict[str, str] = {}
    flag_columns: dict[str, str] = {}
    for arm in ("control", "base", "meta", "both"):
        scores = np.asarray(result.scores[arm], dtype=float)
        flags = np.asarray(result.oof_flags[arm], dtype=object)
        if len(scores) != len(output) or not np.isfinite(scores).all():
            raise StageVIRunnerError(f"arm trainer returned invalid {arm} scores")
        score_column = f"__trained_{arm}_score"
        flag_column = f"__trained_{arm}_is_strict_oof"
        output[score_column] = scores
        output[flag_column] = flags
        score_columns[arm] = score_column
        flag_columns[arm] = flag_column
    _require_strict_oof_arm_lineage(output, flag_columns)
    return output, score_columns, provenance


def _normalized_candidate_evidence(
    *, candidate_id: str, kind: str, fold_audit: pd.DataFrame,
    economics: pd.DataFrame, stability: pd.DataFrame,
) -> dict[str, Any]:
    scored = fold_audit.loc[fold_audit.get("status", pd.Series(dtype=str)).eq("scored")]
    econ_columns = [column for column in economics if column.endswith("__mean")]
    separation = 0.0
    if econ_columns and not economics.empty:
        values = economics[econ_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        separation = float(np.tanh(np.std(finite) / 100.0)) if len(finite) else 0.0
    predictability = 0.0
    if "mean_membership_correlation" in scored and scored["mean_membership_correlation"].notna().any():
        predictability = float(np.clip(scored["mean_membership_correlation"].mean(), 0.0, 1.0))
    elif "membership_brier" in scored and scored["membership_brier"].notna().any():
        predictability = float(np.clip(1.0 - scored["membership_brier"].mean(), 0.0, 1.0))
    temporal = 0.0 if stability.empty else float(
        np.clip(1.0 / (1.0 + stability["mean_centroid_distance"].mean()), 0.0, 1.0)
    )
    concentration = 1.0
    if not economics.empty and "rows" in economics:
        support = economics.groupby("cluster", observed=True)["rows"].sum()
        concentration = float(support.max() / support.sum()) if support.sum() else 1.0
    return {
        "candidate_id": candidate_id, "view_kind": kind,
        "path_separation": separation if kind == "path" else 0.0,
        "economic_separation": separation,
        "causal_predictability": predictability,
        "temporal_stability": temporal,
        "concentration": concentration,
        "base_incremental_bps": 0.0, "meta_incremental_bps": 0.0,
        "hard_label_value": 0.0, "soft_membership_value": 1.0,
        "scored_folds": int(len(scored)),
    }


def run_stage_vi_archetype_funnel(
    ledger: pd.DataFrame,
    *,
    views: StageVIViewContract,
    output_directory: str | Path,
    spec: StageVIRunnerSpec = StageVIRunnerSpec(),
    decision_config: ArchetypeDecisionConfig = ArchetypeDecisionConfig(),
    candidate_ids: Sequence[str] | None = None,
    arm_trainer: ArmTrainer | None = None,
) -> StageVIRunResult:
    """Execute a bounded Stage-VI grid and publish an immutable artifact bundle."""

    spec.validate()
    ledger = ledger.reset_index(drop=True).copy()
    views.validate(ledger)
    output = Path(output_directory)
    if output.exists():
        raise StageVIRunnerError("Stage-VI output already exists; active outputs are immutable")
    required = [
        spec.candidate_id_column, spec.symbol_column, spec.decision_ts_column,
        spec.label_available_ts_column, spec.side_column, spec.exact_net_column,
        *spec.outcome_columns,
    ]
    missing = [column for column in dict.fromkeys(required) if column not in ledger]
    if missing:
        raise StageVIRunnerError(f"Stage-VI ledger is missing required columns: {missing[:12]}")
    identity_sha = _identity_sha256(ledger, spec)
    discovery_support = _positive_discovery_support(ledger, spec)
    if candidate_ids is not None:
        allow = set(map(str, candidate_ids))
        candidates = [
            item for item in stage_vi_candidate_grid(views, spec) if item[0] in allow
        ]
        known = {candidate_id for candidate_id, _config in candidates}
        if known != allow:
            raise StageVIRunnerError(
                f"unknown Stage-VI candidate IDs: {sorted(allow - known)[:8]}"
            )
    elif spec.full_grid:
        candidates = stage_vi_candidate_grid(views, spec)
    else:
        candidates = stage_vi_sequential_funnel_grid(views, spec)
    if not candidates:
        raise StageVIRunnerError("Stage-VI candidate filter selected no predeclared arms")
    _validate_score_contracts(ledger, candidates, spec, arm_trainer)
    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=staging_parent))
    audit_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    candidate_features: dict[str, pd.DataFrame] = {}
    candidate_feature_hashes: dict[str, str] = {}
    candidate_directories: dict[str, Path] = {}
    try:
        (stage / "candidates").mkdir()
        for ordinal, (candidate_id, config) in enumerate(candidates):
            recogniser = views.causal_recogniser_columns if config.view.kind == "path" else None
            result = strict_oof_archetype_features(
                ledger, config=config, causal_recogniser_columns=recogniser,
                folds=spec.folds,
            )
            probability_columns = [
                column for column in result.features
                if "prob__" in column and not column.endswith("unknown")
            ]
            available_column = next(
                column for column in result.features if column.endswith("available")
            )
            scored_mask = result.features[available_column].gt(0.5)
            economics = archetype_economic_separation(
                ledger.loc[scored_mask], result.features.loc[scored_mask, probability_columns],
                outcome_columns=spec.outcome_columns,
                timestamp_col=spec.decision_ts_column, side_col=spec.side_column,
                symbol_col=spec.symbol_column,
            )
            stability = archetype_fold_stability(result.catalog)
            candidate_dir = stage / "candidates" / f"{ordinal:05d}__{candidate_id}"
            candidate_dir.mkdir()
            identity_columns = [
                spec.candidate_id_column, spec.symbol_column,
                spec.decision_ts_column, spec.side_column,
            ]
            feature_bundle = pd.concat(
                [ledger.loc[:, identity_columns], result.features], axis=1
            )
            feature_bundle.to_parquet(
                candidate_dir / "strict_oof_features.parquet", index=False
            )
            feature_content_hash = _frame_sha256(
                feature_bundle.sort_values(identity_columns, kind="stable").reset_index(drop=True)
            )
            # Bind identical numeric memberships to their distinct declared
            # representation contract; candidate identity is part of the fit.
            feature_hash = sha256(
                f"{candidate_id}:{feature_content_hash}".encode("utf-8")
            ).hexdigest()
            candidate_features[candidate_id] = result.features.copy()
            candidate_feature_hashes[candidate_id] = feature_hash
            candidate_directories[candidate_id] = candidate_dir
            pd.concat(
                [ledger.loc[:, identity_columns], result.diagnostic_truth_memberships], axis=1
            ).to_parquet(
                candidate_dir / "diagnostic_truth_memberships.parquet", index=False
            )
            result.fold_audit.to_csv(candidate_dir / "fold_audit.csv", index=False)
            result.catalog.to_json(candidate_dir / "catalog.json", orient="records")
            economics.to_csv(candidate_dir / "economic_separation.csv", index=False)
            stability.to_csv(candidate_dir / "fold_stability.csv", index=False)
            candidate_manifest = {
                **result.manifest, "candidate_id": candidate_id,
                "method": config.method, "components": config.components,
                "weight_id": candidate_id.rsplit("__", 1)[-1],
                "aw3_side_local_enforced": True,
                "identity_sha256": identity_sha,
                "candidate_feature_sha256": feature_hash,
                "candidate_feature_content_sha256": feature_content_hash,
            }
            (candidate_dir / "manifest.json").write_text(
                json.dumps(candidate_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            evidence = _normalized_candidate_evidence(
                candidate_id=candidate_id, kind=config.view.kind,
                fold_audit=result.fold_audit, economics=economics, stability=stability,
            )
            evidence_rows.append(evidence)
            audit_rows.append({
                "candidate_id": candidate_id, "view": config.view.name,
                "view_kind": config.view.kind, "method": config.method,
                "components": config.components,
                "weight_id": candidate_id.rsplit("__", 1)[-1],
                "scored_rows": result.manifest["scored_rows"],
                "unknown_rows": result.manifest["unknown_rows"],
            })

        sort_columns = [
            spec.candidate_id_column, spec.decision_ts_column, spec.side_column
        ]
        comparison_order = ledger.sort_values(sort_columns, kind="stable").index
        comparison_ledger = ledger.loc[comparison_order].reset_index(drop=True)
        comparison_ids = comparison_ledger[spec.candidate_id_column].to_numpy(copy=True)
        comparisons: list[pd.DataFrame] = []
        delta_by_candidate: dict[str, dict[str, float]] = {}
        score_binding_rows: list[dict[str, Any]] = []
        for candidate_id, _config in candidates:
            feature_hash = candidate_feature_hashes[candidate_id]
            provenance: Mapping[str, Any]
            if arm_trainer is not None:
                features = candidate_features[candidate_id].loc[
                    comparison_order
                ].reset_index(drop=True)
                request = StageVIArmTrainingRequest(
                    candidate_id=candidate_id,
                    ledger=comparison_ledger.copy(),
                    archetype_features=features,
                    archetype_feature_columns=tuple(map(str, features.columns)),
                    candidate_ids=comparison_ids.copy(),
                    decision_timestamps=comparison_ledger[
                        spec.decision_ts_column
                    ].to_numpy(copy=True),
                    side_names=comparison_ledger[spec.side_column].astype(str).to_numpy(),
                    feature_sha256=feature_hash,
                )
                arm_ledger, arm_columns, provenance = _trained_comparison_ledger(
                    request, arm_trainer(request)
                )
                score_source = "arm_trainer"
            else:
                arm_ledger = comparison_ledger
                arm_columns = spec.arm_score_columns_by_candidate[candidate_id]
                provenance = {
                    "strict_oof": True,
                    "candidate_feature_sha256": feature_hash,
                    "archetype_feature_columns": list(
                        map(str, candidate_features[candidate_id].columns)
                    ),
                    "score_columns": dict(arm_columns),
                }
                score_source = "explicit_candidate_score_contract"
            score_bundle = pd.concat(
                [
                    arm_ledger.loc[:, sort_columns],
                    arm_ledger.loc[:, list(arm_columns.values())],
                ], axis=1,
            )
            score_hash = _frame_sha256(score_bundle)
            candidate_comparison = run_matched_incremental_archetype_comparison(
                arm_ledger,
                arm_score_columns=arm_columns,
                net_bps_col=spec.exact_net_column,
                gross_bps_col=spec.exact_gross_column,
                identity_columns=(
                    spec.candidate_id_column, spec.symbol_column,
                    spec.decision_ts_column, spec.side_column,
                ),
                top_fractions=spec.top_fractions,
            )
            candidate_comparison.insert(0, "candidate_id", candidate_id)
            comparisons.append(candidate_comparison)
            top10 = candidate_comparison.loc[candidate_comparison.tail_fraction.eq(0.10)]
            delta_by_candidate[candidate_id] = dict(
                zip(top10.arm, top10.delta_net_bps_per_trade_vs_control)
            )
            binding = {
                "candidate_id": candidate_id,
                "candidate_feature_sha256": feature_hash,
                "archetype_feature_columns": list(
                    map(str, candidate_features[candidate_id].columns)
                ),
                "score_sha256": score_hash,
                "score_source": score_source,
                "score_columns": dict(arm_columns),
                "provenance": dict(provenance),
            }
            score_binding_rows.append(binding)
            (candidate_directories[candidate_id] / "score_binding.json").write_text(
                json.dumps(binding, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
        comparison = pd.concat(comparisons, ignore_index=True)
        for row in evidence_rows:
            deltas = delta_by_candidate[str(row["candidate_id"])]
            row["base_incremental_bps"] = float(deltas.get("base", 0.0))
            row["meta_incremental_bps"] = float(deltas.get("meta", 0.0))
        evidence = pd.DataFrame(evidence_rows)
        decisions = materialize_archetype_decision_matrix(evidence, config=decision_config)
        candidate_audit = pd.DataFrame(audit_rows)
        candidate_audit.to_csv(stage / "candidate_audit.csv", index=False)
        discovery_support.to_csv(stage / "positive_discovery_support.csv", index=False)
        comparison.to_csv(stage / "matched_base_meta_both_comparison.csv", index=False)
        (stage / "score_bindings.json").write_text(
            json.dumps(score_binding_rows, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        decisions.to_csv(stage / "decision_matrix.csv", index=False)
        (stage / "view_contract.json").write_text(
            json.dumps(views.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema": RUNNER_SCHEMA, "representation_schema": STAGE_VI_SCHEMA,
            "identity_sha256": identity_sha, "rows": int(len(ledger)),
            "candidate_count": int(len(candidates)),
            "search_mode": (
                "explicit_candidate_filter" if candidate_ids is not None
                else "full_grid_explicit_opt_in" if spec.full_grid
                else "bounded_sequential_funnel"
            ),
            "full_grid_explicit_opt_in": bool(spec.full_grid),
            "causal_components": list(spec.causal_components),
            "path_components": list(spec.path_components),
            "methods": list(spec.methods), "aw_contracts": dict(AW_CONTRACTS),
            "positive_label_only_discovery": True,
            "positive_rows_spread_over_time": True,
            "positive_discovery_support": discovery_support.to_dict(orient="records"),
            "side_local_discovery": True, "strict_oof_causal_recognisers": True,
            "matched_controls": ["control", "base", "meta", "both"],
            "global_not_timestamp_local_ranking": True,
            "hard_routing": False, "local_trading_experts": False,
            "diagnostic_path_truth_never_model_input": True,
            "candidate_score_bindings": score_binding_rows,
            "spec": asdict(spec),
        }
        (stage / "run_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        checksums = {
            path.relative_to(stage).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        (stage / "checksums.json").write_text(
            json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        stage.replace(output)
        return StageVIRunResult(output, candidate_audit, comparison, decisions, manifest)
    except Exception:
        # Preserve the staging directory for forensic inspection.  It is never
        # mistaken for a published output because the requested path is absent.
        raise


__all__ = [
    "AW_CONTRACTS", "CAUSAL_COMPONENTS", "METHODS", "PATH_COMPONENTS",
    "RUNNER_SCHEMA", "VIEW_SCHEMA", "ArmTrainer", "StageVIArmTrainingRequest",
    "StageVIArmTrainingResult", "StageVIRunResult", "StageVIRunnerError",
    "StageVIRunnerSpec", "StageVIViewContract", "materialize_stage_vi_view_contract",
    "run_stage_vi_archetype_funnel", "stage_vi_candidate_grid",
    "stage_vi_sequential_funnel_grid",
]
