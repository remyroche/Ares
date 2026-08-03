"""Sequential Stage-III runner for one regime-aware residual expert.

The runner implements the ablation sequence declared in the Stage-III design:

``A target baseline -> B training robustness -> C conditioning -> D validity
context -> E calibration``.

It is deliberately not a factorial search.  Each round starts from the frozen
winner of the previous round and changes one declared axis.  Every model fit is
one shared model over both sides, validation folds are expanding chronological
environment blocks, and final tail selection is performed once over the pooled
OOF population after reconstruction/calibration into common bps.

The module has no CLI and performs no I/O on import.  Large experiments must be
started by an explicit caller that supplies an in-memory ledger and frozen
feature lists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .shared_regime_calibration import (
    fit_shared_bps_calibration,
    predict_shared_bps_calibration,
    prequential_shared_bps_calibration,
)
from .shared_regime_residual_expert import (
    SharedResidualColumns,
    SharedResidualExpertFit,
    SoftRegimeResidualConfig,
    fit_shared_regime_residual_expert,
    mild_environment_weights,
    prepare_shared_regime_residual_frame,
    reconstruct_shared_regime_expected_net_bps,
    robust_cross_era_selection_score,
)
from .stage_iii_pairwise_shared_expert import (
    PairSupportAudit,
    PairwiseSharedResidualColumns,
    PairwiseSharedResidualConfig,
    PairwiseSharedResidualExpertFit,
    TargetPreservingPairwiseAdapterFit,
    fit_pairwise_shared_residual_expert,
    fit_target_preserving_pairwise_adapter,
)
from .stage_iii_residual_target_challengers import PairConstructionConfig
from .stage_iii_robust_target_models import (
    OrdinalSharedRobustTargetFit,
    QuantileSharedRobustTargetFit,
    RobustTargetColumns,
    RobustTargetModelConfig,
    fit_ordinal_shared_robust_target,
    fit_quantile_shared_robust_target,
)


SCHEMA = "stage_iii_shared_expert_sequential_funnel_v1"


class StageIIISequentialRunnerError(ValueError):
    """Raised when the sequential or causal contract is violated."""


def stage_iii_feature_contract_sha256(feature_names: Sequence[str]) -> str:
    """Hash the exact ordered, de-duplicated Stage-III source feature contract."""
    payload = list(dict.fromkeys(str(x) for x in feature_names))
    return sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha256(path: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise StageIIISequentialRunnerError(f"frozen lineage artifact is missing: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_boolean(series: pd.Series, *, name: str) -> np.ndarray:
    """Parse only actual booleans or integer 0/1 values.

    Python truthiness is forbidden for lineage flags: strings such as
    ``"false"``, floats, missing values, and arbitrary non-zero integers all
    fail closed.
    """
    values: list[bool] = []
    for value in series.to_numpy(dtype=object):
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            values.append(bool(value))
        else:
            raise StageIIISequentialRunnerError(
                f"{name} must contain only canonical bool or integer 0/1 values"
            )
    return np.asarray(values, dtype=bool)


def _read_feature_contract(path: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageIIISequentialRunnerError("feature contract artifact must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise StageIIISequentialRunnerError("feature contract artifact must contain a JSON object")
    return payload


@dataclass(frozen=True)
class StageIIIInputLineageContract:
    """Serializable, row-verifiable predecessor contract.

    Content hashes freeze the actual predecessor artifacts.  The named
    evidence columns must additionally prove the contract row by row; boolean
    prose or a filename containing ``oof`` is intentionally insufficient.
    """

    schema: str = "stage_iii_shared_expert_input_lineage_v1"
    r3_artifact_sha256: str = ""
    r3_artifact_path: str = ""
    base_map_artifact_sha256: str = ""
    base_map_artifact_path: str = ""
    soft_regime_artifact_sha256: str = ""
    soft_regime_artifact_path: str = ""
    label_artifact_sha256: str = ""
    label_artifact_path: str = ""
    admission_artifact_sha256: str = ""
    admission_artifact_path: str = ""
    feature_contract_sha256: str = ""
    feature_contract_artifact_sha256: str = ""
    feature_contract_artifact_path: str = ""
    r3_oof_flag_column: str = "r3_is_strict_oof"
    r3_source_side_column: str = "r3_source_side"
    r3_fit_end_column: str = "r3_fit_end_ts"
    r3_semantics_column: str = "r3_score_semantics"
    r3_probability_columns: tuple[str, ...] = (
        "r3_p_adverse", "r3_p_weak", "r3_p_clear",
    )
    base_map_prequential_flag_column: str = "base_map_is_prequential"
    base_map_source_side_column: str = "base_map_source_side"
    base_map_max_label_available_column: str = "base_map_max_label_available_ts"
    regime_causal_flag_column: str = "soft_regime_is_causal_prequential"
    regime_fit_end_column: str = "soft_regime_fit_end_ts"
    admission_flag_column: str = "causal_21d_admitted"
    admission_prequential_flag_column: str = "causal_21d_admission_is_prequential"
    admission_source_side_column: str = "causal_21d_admission_source_side"
    admission_max_label_available_column: str = "causal_21d_admission_max_label_available_ts"
    admission_window_days_column: str = "causal_21d_admission_window_days"
    cost_atr_causal_flag_column: str = "cost_atr_is_causal"
    signal_timestamp_column: str = "signal_close_ts"
    cost_column: str = "total_cost_bps"
    geometry: Mapping[str, float] = field(default_factory=lambda: {
        "tp_atr": 6.0, "sl_atr": 4.0, "horizon_hours": 12.0,
    })
    total_cost_bps: float = 100.0
    cost_application_count: int = 1
    signal_to_entry_hours: float = 1.0
    signal_to_label_available_hours: float = 13.0
    r3_semantics: str = "same_side_direct_strict_oof_probabilities_without_conversion"
    ranking: str = "pooled_global_after_common_bps_mapping"
    routing: str = "one_shared_model_no_hard_routing"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageIIIInputLineageContract":
        return cls(**dict(value))

    @property
    def contract_sha256(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def validate(
        self, frame: pd.DataFrame, *, config: "StageIIIRunnerConfig",
        soft_regime_columns: Sequence[str], invariant_features: Sequence[str],
        regime_relative_features: Sequence[str],
        restricted_interaction_features: Sequence[str],
        validity_feature_groups: Mapping[str, Sequence[str]],
    ) -> None:
        if self.schema != "stage_iii_shared_expert_input_lineage_v1":
            raise StageIIISequentialRunnerError("unsupported Stage-III input-lineage schema")
        bindings = (
            ("r3_artifact_sha256", "r3_artifact_path"),
            ("base_map_artifact_sha256", "base_map_artifact_path"),
            ("soft_regime_artifact_sha256", "soft_regime_artifact_path"),
            ("label_artifact_sha256", "label_artifact_path"),
            ("admission_artifact_sha256", "admission_artifact_path"),
            ("feature_contract_artifact_sha256", "feature_contract_artifact_path"),
        )
        for name, path_name in bindings:
            digest = str(getattr(self, name))
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or len(set(digest)) == 1:
                raise StageIIISequentialRunnerError(f"{name} must be a non-placeholder SHA-256 digest")
            if _file_sha256(str(getattr(self, path_name))) != digest:
                raise StageIIISequentialRunnerError(f"frozen artifact hash mismatch for {path_name}")
        if dict(self.geometry) != {"tp_atr": 6.0, "sl_atr": 4.0, "horizon_hours": 12.0}:
            raise StageIIISequentialRunnerError("Stage-III labels must use exact TP6/SL4/H12 geometry")
        if self.total_cost_bps != 100.0 or self.cost_application_count != 1:
            raise StageIIISequentialRunnerError("Stage-III labels must apply 100 bps exactly once")
        if self.ranking != "pooled_global_after_common_bps_mapping":
            raise StageIIISequentialRunnerError("Stage-III selection must be pooled-global after common-bps mapping")
        if self.routing != "one_shared_model_no_hard_routing":
            raise StageIIISequentialRunnerError("local experts and hard routing are forbidden")
        evidence = [
            self.r3_oof_flag_column, self.r3_source_side_column, self.r3_fit_end_column,
            self.r3_semantics_column, self.base_map_prequential_flag_column,
            self.base_map_source_side_column, self.base_map_max_label_available_column,
            self.regime_causal_flag_column, self.regime_fit_end_column,
            self.admission_flag_column, self.admission_prequential_flag_column,
            self.admission_source_side_column, self.admission_max_label_available_column,
            self.admission_window_days_column,
            self.cost_atr_causal_flag_column,
            self.signal_timestamp_column, self.cost_column, *soft_regime_columns,
            *self.r3_probability_columns,
        ]
        missing = [name for name in evidence if name not in frame]
        if missing:
            raise StageIIISequentialRunnerError(
                f"naming-only lineage rejected; row evidence columns are missing: {missing[:12]}"
            )
        decision = _utc(frame, config.decision_timestamp_column)
        available = _utc(frame, config.label_available_timestamp_column)
        signal = _utc(frame, self.signal_timestamp_column)
        side = frame[config.side_column].astype(str).str.lower()
        if not _canonical_boolean(frame[self.r3_oof_flag_column], name=self.r3_oof_flag_column).all():
            raise StageIIISequentialRunnerError("every R3 row must be strict OOF")
        if not frame[self.r3_source_side_column].astype(str).str.lower().eq(side).all():
            raise StageIIISequentialRunnerError("R3 probabilities must be direct same-side outputs")
        if not frame[self.r3_semantics_column].astype(str).eq(self.r3_semantics).all():
            raise StageIIISequentialRunnerError("R3 outputs were converted or have unverifiable semantics")
        r3 = frame.loc[:, self.r3_probability_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if (
            len(self.r3_probability_columns) != 3 or not np.isfinite(r3).all()
            or (r3 < 0).any() or not np.allclose(r3.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise StageIIISequentialRunnerError("direct R3 adverse/weak/clear outputs must form a probability simplex")
        if not (_utc(frame, self.r3_fit_end_column) < decision).all():
            raise StageIIISequentialRunnerError("R3 fit end must precede every decision")
        if not _canonical_boolean(
            frame[self.base_map_prequential_flag_column], name=self.base_map_prequential_flag_column
        ).all():
            raise StageIIISequentialRunnerError("base expected-net map must be prequential")
        if not frame[self.base_map_source_side_column].astype(str).str.lower().eq(side).all():
            raise StageIIISequentialRunnerError("base bps map must use same-side R3 outputs")
        if not (_utc(frame, self.base_map_max_label_available_column) < decision).all():
            raise StageIIISequentialRunnerError("base bps map contains current/future resolved labels")
        if not _canonical_boolean(
            frame[self.regime_causal_flag_column], name=self.regime_causal_flag_column
        ).all():
            raise StageIIISequentialRunnerError("soft regime state lacks causal/prequential proof")
        if not (_utc(frame, self.regime_fit_end_column) < decision).all():
            raise StageIIISequentialRunnerError("soft regime state was not frozen before decision")
        _canonical_boolean(frame[self.admission_flag_column], name=self.admission_flag_column)
        if not _canonical_boolean(
            frame[self.admission_prequential_flag_column], name=self.admission_prequential_flag_column
        ).all():
            raise StageIIISequentialRunnerError("21-day admission map must be prequential")
        if not frame[self.admission_source_side_column].astype(str).str.lower().eq(side).all():
            raise StageIIISequentialRunnerError("21-day admission map must be side-local")
        if not (_utc(frame, self.admission_max_label_available_column) < decision).all():
            raise StageIIISequentialRunnerError("21-day admission uses current/future resolved labels")
        if not pd.to_numeric(frame[self.admission_window_days_column], errors="coerce").eq(21).all():
            raise StageIIISequentialRunnerError("admission map must use the frozen 21-day window")
        if not _canonical_boolean(
            frame[self.cost_atr_causal_flag_column], name=self.cost_atr_causal_flag_column
        ).all():
            raise StageIIISequentialRunnerError("cost-to-ATR context must be causal")
        p = frame.loc[:, soft_regime_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
            raise StageIIISequentialRunnerError("soft regime fields must be a finite probability simplex")
        if not np.allclose((decision - signal).dt.total_seconds() / 3600.0, self.signal_to_entry_hours):
            raise StageIIISequentialRunnerError("entry must be one hour after signal close")
        if not np.allclose((available - signal).dt.total_seconds() / 3600.0, self.signal_to_label_available_hours):
            raise StageIIISequentialRunnerError("label availability must be signal close +13h")
        cost = pd.to_numeric(frame[self.cost_column], errors="coerce").to_numpy(float)
        if not np.isfinite(cost).all() or not np.allclose(cost, self.total_cost_bps):
            raise StageIIISequentialRunnerError("row-level cost evidence must equal 100 bps")
        source_features = list(dict.fromkeys([
            *(str(x) for x in invariant_features), *(str(x) for x in soft_regime_columns),
            *(str(x) for x in regime_relative_features),
            *(str(x) for x in restricted_interaction_features),
            *(str(feature) for group in validity_feature_groups.values() for feature in group),
        ]))
        actual_feature_hash = stage_iii_feature_contract_sha256(source_features)
        if actual_feature_hash != self.feature_contract_sha256:
            raise StageIIISequentialRunnerError("feature contract hash does not match the supplied feature lists")
        artifact = _read_feature_contract(self.feature_contract_artifact_path)
        if artifact.get("schema") != "stage_iii_feature_admission_v1":
            raise StageIIISequentialRunnerError("runner requires a Stage-III feature-admission artifact")
        admitted = artifact.get("admitted_ordered_features")
        audit_rows = artifact.get("feature_audit")
        config_payload = artifact.get("config")
        if not isinstance(admitted, list) or not isinstance(audit_rows, list) or not isinstance(config_payload, Mapping):
            raise StageIIISequentialRunnerError("feature-admission artifact is structurally incomplete")
        if list(map(str, admitted)) != source_features:
            raise StageIIISequentialRunnerError(
                "feature-admission artifact must exactly match the ordered runner source contract"
            )
        minimum_coverage = config_payload.get("min_coverage")
        if not isinstance(minimum_coverage, (int, float)) or not 0.0 < float(minimum_coverage) <= 1.0:
            raise StageIIISequentialRunnerError("feature-admission min_coverage is invalid")
        audit_by_feature: dict[str, Mapping[str, Any]] = {}
        for row in audit_rows:
            if not isinstance(row, Mapping) or not str(row.get("feature_name", "")).strip():
                raise StageIIISequentialRunnerError("feature-admission audit contains an invalid row")
            name = str(row["feature_name"])
            if name in audit_by_feature:
                raise StageIIISequentialRunnerError("feature-admission audit has duplicate feature names")
            audit_by_feature[name] = row
        missing_sources = [name for name in source_features if name not in frame]
        if missing_sources:
            raise StageIIISequentialRunnerError(f"source feature columns are missing: {missing_sources[:12]}")
        invariant_set = set(map(str, invariant_features))
        for name in source_features:
            row = audit_by_feature.get(name)
            if row is None:
                raise StageIIISequentialRunnerError(f"feature admission evidence is missing for {name!r}")
            classification = str(row.get("classification", ""))
            if classification in {"REGIME_LOCAL_DIAGNOSTIC", "UNSTABLE", "REDUNDANT"}:
                raise StageIIISequentialRunnerError(
                    f"feature {name!r} has forbidden admission class {classification}"
                )
            allowed_classes = {"INVARIANT_CORE"} if name in invariant_set else {
                "INVARIANT_CORE", "REGIME_CONDITIONAL",
            }
            if classification not in allowed_classes or row.get("admitted") is not True:
                raise StageIIISequentialRunnerError(
                    f"feature {name!r} is not admitted in the appropriate class"
                )
            if row.get("live_parity") is not True or row.get("meta_allowed_key") is not True:
                raise StageIIISequentialRunnerError(
                    f"feature {name!r} lacks explicit allowed-meta/live-parity evidence"
                )
            declared = row.get("coverage")
            finite_fraction = row.get("finite_fraction")
            null_fraction = row.get("null_fraction")
            if not all(
                isinstance(value, (int, float)) and np.isfinite(float(value))
                for value in (declared, finite_fraction, null_fraction)
            ):
                raise StageIIISequentialRunnerError(f"feature coverage evidence is missing/non-finite for {name!r}")
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            actual = float(np.isfinite(values).mean())
            if (
                float(declared) < float(minimum_coverage)
                or float(finite_fraction) < float(minimum_coverage)
                or float(null_fraction) > 1.0 - float(minimum_coverage) + 1e-12
                or actual < float(minimum_coverage)
            ):
                raise StageIIISequentialRunnerError(
                    f"feature {name!r} fails declared/runtime coverage: declared={declared} actual={actual}"
                )


@dataclass(frozen=True)
class ExpandingEnvironmentFold:
    fold_id: int
    validation_environment: str
    validation_start_utc: pd.Timestamp
    validation_end_utc: pd.Timestamp
    train_positions: np.ndarray
    validation_positions: np.ndarray
    train_max_label_available_utc: pd.Timestamp


@dataclass(frozen=True)
class StageIIIStack:
    """The fully explicit state carried from one sequential round to the next."""

    baseline_mode: str = "A0_current"
    balance_mode: str = "natural"
    robust_hpo: bool = False
    conditioning_mode: str = "C0_no_regime_features"
    validity_mode: str = "D0_no_ood"
    calibration_mode: str = "E0_global"
    model_params_index: int = 0
    residual_target_arm: str = "T0_huber"
    residual_target_mode: str = "huber"
    residual_target_clip_bps: float = 400.0


@dataclass(frozen=True)
class StageIIIRunnerConfig:
    candidate_id_column: str = "candidate_id"
    symbol_column: str = "symbol"
    decision_timestamp_column: str = "decision_ts"
    label_available_timestamp_column: str = "label_available_ts"
    side_column: str = "side_name"
    exact_net_column: str = "exact_net_bps"
    exact_gross_column: str | None = "exact_gross_bps"
    base_expected_net_column: str = "prequential_base_expected_net_bps"
    environment_column: str = "environment"
    month_timestamp_column: str = "decision_ts"
    hard_regime_column: str | None = None
    min_train_environments: int = 2
    min_train_rows: int = 64
    min_rows_per_side: int = 1
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    primary_top_fraction: float = 0.10
    worst_penalty: float = 0.50
    dispersion_penalty: float = 0.25
    target_mode: Literal["huber", "clipped", "regime_standardized"] = "huber"
    calibration_min_rows: int = 32
    calibration_anchor: Literal["day", "timestamp"] = "day"
    admission_column: str = "causal_21d_admitted"
    cost_to_atr_column: str = "cost_to_atr"
    cost_atr_causal_flag_column: str = "cost_atr_is_causal"
    selection_admission_scope: Literal["without_21d", "with_21d"] = "without_21d"
    catastrophic_lift_bps: float = -100.0
    max_environment_dispersion_bps: float = 75.0
    min_positive_environment_fraction: float = 0.80
    run_transport_matrix: bool = True
    baseline_config: SoftRegimeResidualConfig = field(default_factory=SoftRegimeResidualConfig)
    hpo_param_candidates: tuple[Mapping[str, Any], ...] = (
        {"n_estimators": 200, "learning_rate": 0.035, "num_leaves": 15,
         "min_child_samples": 80, "verbosity": -1, "random_state": 17},
        {"n_estimators": 300, "learning_rate": 0.025, "num_leaves": 23,
         "min_child_samples": 120, "verbosity": -1, "random_state": 17},
        {"n_estimators": 160, "learning_rate": 0.050, "num_leaves": 11,
         "min_child_samples": 60, "verbosity": -1, "random_state": 17},
    )
    pairwise_config: PairwiseSharedResidualConfig = field(
        default_factory=PairwiseSharedResidualConfig
    )
    pair_construction_config: PairConstructionConfig = field(
        default_factory=PairConstructionConfig
    )
    robust_target_model_config: RobustTargetModelConfig = field(
        default_factory=RobustTargetModelConfig
    )
    target_gate_calibration_mode: str = "E0_global"

    def validate(self) -> None:
        if self.min_train_environments < 1 or self.min_train_rows < 1:
            raise StageIIISequentialRunnerError("training support thresholds must be positive")
        if self.min_rows_per_side < 1:
            raise StageIIISequentialRunnerError("min_rows_per_side must be positive")
        if not self.hpo_param_candidates:
            raise StageIIISequentialRunnerError("at least one frozen model parameter candidate is required")
        if not self.top_fractions or any(not 0.0 < x <= 1.0 for x in self.top_fractions):
            raise StageIIISequentialRunnerError("top fractions must lie in (0, 1]")
        if self.primary_top_fraction not in self.top_fractions:
            raise StageIIISequentialRunnerError("primary_top_fraction must be one of top_fractions")
        if self.selection_admission_scope not in {"without_21d", "with_21d"}:
            raise StageIIISequentialRunnerError("selection_admission_scope is invalid")
        if not 0.0 < self.min_positive_environment_fraction <= 1.0:
            raise StageIIISequentialRunnerError("min_positive_environment_fraction must be in (0, 1]")


@dataclass(frozen=True)
class StageIIIArmResult:
    arm: str
    round_name: str
    stack: StageIIIStack
    selected_params_index: int
    selected_params: Mapping[str, Any]
    selection_summary: Mapping[str, float]
    oof_predictions: pd.DataFrame
    metrics: pd.DataFrame
    fold_audit: pd.DataFrame
    calibration_audit: pd.DataFrame
    shared_model_fit_count: int
    model_feature_names: tuple[str, ...]
    model_feature_contract_sha256: str
    source_feature_contract_sha256: str
    predecessor_arm: str | None = None
    pair_support: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_audits: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class StageIIIFunnelResult:
    schema: str
    round_winners: Mapping[str, str]
    winner: StageIIIArmResult
    arms: tuple[StageIIIArmResult, ...]
    arm_summary: pd.DataFrame
    transport_matrix: pd.DataFrame
    advancement_gates: Mapping[str, Any]


ExpertFitter = Callable[..., SharedResidualExpertFit]
PairwiseExpertFitter = Callable[..., PairwiseSharedResidualExpertFit]
TargetPreservingPairwiseFitter = Callable[..., TargetPreservingPairwiseAdapterFit]
OrdinalTargetFitter = Callable[..., OrdinalSharedRobustTargetFit]
QuantileTargetFitter = Callable[..., QuantileSharedRobustTargetFit]


def _final_stack_identity(result: StageIIIArmResult) -> str:
    payload = {
        "arm": result.arm,
        "predecessor_arm": result.predecessor_arm,
        "stack": asdict(result.stack),
        "model_feature_contract_sha256": result.model_feature_contract_sha256,
        "source_feature_contract_sha256": result.source_feature_contract_sha256,
        "selected_params_index": result.selected_params_index,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_BASELINE_ARMS = {
    "A0": "A0_current",
    "A1": "A1_side_centered",
    "A2": "A2_side_hard_regime_centered",
    "A3": "A3_soft_regime_centered",
}
_TRAINING_ARMS = {
    "B0": ("natural", False),
    "B1": ("era", False),
    "B2": ("natural", True),
    "B3": ("era", True),
}
_TARGET_ARMS = {
    "T0": ("T0_huber", "huber", 400.0),
    "T1_200": ("T1_clipped_200", "clipped", 200.0),
    "T1_400": ("T1_clipped_400", "clipped", 400.0),
    "T2": ("T2_regime_standardized", "regime_standardized", 400.0),
    "T3": ("T3_ordinal", "ordinal", 400.0),
    "T4": ("T4_quantile_median", "quantile", 400.0),
}
_CONDITIONING_ARMS = {
    "C0": "C0_no_regime_features",
    "C1": "C1_soft_regime_probabilities",
    "C2": "C2_restricted_interactions",
    "C3": "C3_regime_relative_features",
}
_VALIDITY_ARMS = {
    "D0": "D0_no_ood",
    "D1": "D1_relationship_breaks",
    "D2": "D2_contribution_ood",
    "D3": "D3_active_failure_probability",
    "D4": "D4_compact_combination",
}
_CALIBRATION_ARMS = {
    "E0": "E0_global",
    "E1": "E1_side_local",
    "E2": "E2_side_soft_regime_hierarchical",
}
_PAIRWISE_ARMS = {
    "F0": "F0_pointwise",
    "F1": "F1_pairwise_50bps",
    "F2": "F2_pairwise_100bps",
}


def declared_sequential_arms() -> dict[str, tuple[str, ...]]:
    """Return the exact non-factorial round order."""
    return {
        "A_target_normalization": tuple(_BASELINE_ARMS),
        "T_residual_target": tuple(_TARGET_ARMS),
        "B_training_robustness": tuple(_TRAINING_ARMS),
        "C_conditioning": tuple(_CONDITIONING_ARMS),
        "D_model_validity": tuple(_VALIDITY_ARMS),
        "E_calibration": tuple(_CALIBRATION_ARMS),
        "F_pairwise_ranking": tuple(_PAIRWISE_ARMS),
    }


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise StageIIISequentialRunnerError(f"missing required column {column!r}")
    value = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if value.isna().any():
        raise StageIIISequentialRunnerError(f"column {column!r} contains invalid timestamps")
    return value


def _candidate_identity(frame: pd.DataFrame, config: StageIIIRunnerConfig) -> pd.Series:
    required = [config.candidate_id_column, config.symbol_column, config.side_column]
    missing = [name for name in required if name not in frame]
    if missing:
        raise StageIIISequentialRunnerError(f"immutable row identity columns are missing: {missing}")
    candidate = frame[config.candidate_id_column].astype("string")
    symbol = frame[config.symbol_column].astype("string")
    side = frame[config.side_column].astype("string").str.lower()
    if candidate.isna().any() or candidate.str.strip().eq("").any():
        raise StageIIISequentialRunnerError("candidate_id must be non-null and non-empty")
    if symbol.isna().any() or symbol.str.strip().eq("").any():
        raise StageIIISequentialRunnerError("symbol must be non-null and non-empty")
    decision = _utc(frame, config.decision_timestamp_column)
    identity = pd.Series(
        [
            json.dumps(
                [str(cid), str(sym), ts.isoformat(), str(side_name)],
                separators=(",", ":"), ensure_ascii=False,
            )
            for cid, sym, ts, side_name in zip(candidate, symbol, decision, side, strict=True)
        ],
        index=frame.index, dtype="string", name="__candidate_identity",
    )
    if identity.duplicated().any():
        raise StageIIISequentialRunnerError(
            "candidate_id + symbol + decision_ts + side row identity must be unique"
        )
    return identity


def _identity_set_sha256(values: Sequence[Any]) -> str:
    payload = sorted(str(value) for value in values)
    if len(payload) != len(set(payload)):
        raise StageIIISequentialRunnerError("candidate identity set contains duplicates")
    return sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_input(frame: pd.DataFrame, config: StageIIIRunnerConfig) -> None:
    config.validate()
    required = {
        config.decision_timestamp_column, config.label_available_timestamp_column,
        config.side_column, config.exact_net_column, config.base_expected_net_column,
        config.environment_column, config.admission_column,
        config.candidate_id_column, config.symbol_column,
        config.cost_to_atr_column, config.cost_atr_causal_flag_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise StageIIISequentialRunnerError(f"Stage-III ledger lacks required columns: {missing}")
    decision = _utc(frame, config.decision_timestamp_column)
    available = _utc(frame, config.label_available_timestamp_column)
    if (available <= decision).any():
        raise StageIIISequentialRunnerError("every target must resolve strictly after its decision")
    if not decision.is_monotonic_increasing:
        raise StageIIISequentialRunnerError("Stage-III input must be chronological by decision timestamp")
    _candidate_identity(frame, config)
    for name in (
        config.exact_net_column, config.base_expected_net_column, config.cost_to_atr_column,
    ):
        value = pd.to_numeric(frame[name], errors="coerce")
        if not np.isfinite(value).all():
            raise StageIIISequentialRunnerError(f"{name!r} must be finite common-bps values")
    if config.exact_gross_column is not None:
        if config.exact_gross_column not in frame:
            raise StageIIISequentialRunnerError(f"missing exact gross column {config.exact_gross_column!r}")
        gross = pd.to_numeric(frame[config.exact_gross_column], errors="coerce").to_numpy(float)
        net = pd.to_numeric(frame[config.exact_net_column], errors="coerce").to_numpy(float)
        if not np.isfinite(gross).all() or not np.allclose(gross - 100.0, net, atol=1e-5):
            raise StageIIISequentialRunnerError("exact gross minus 100 bps must equal exact net")
    side = frame[config.side_column].astype(str).str.lower()
    if not set(side.unique()).issubset({"long", "short"}) or side.nunique() != 2:
        raise StageIIISequentialRunnerError("one shared expert requires both long and short rows")


def build_expanding_environment_folds(
    frame: pd.DataFrame, *, config: StageIIIRunnerConfig
) -> tuple[ExpandingEnvironmentFold, ...]:
    """Build expanding folds whose validation environments are atomic blocks.

    An environment may not recur after a later environment begins.  This
    fail-closed rule prevents a convenient label from mixing train and
    validation periods or creating a pseudo walk-back fold.
    """
    _validate_input(frame, config)
    frame = frame.copy().reset_index(drop=True)
    frame["__candidate_identity"] = _candidate_identity(frame, config).to_numpy()
    decision = _utc(frame, config.decision_timestamp_column)
    available = _utc(frame, config.label_available_timestamp_column)
    environment = frame[config.environment_column].astype(str)
    if environment.str.strip().eq("").any():
        raise StageIIISequentialRunnerError("environment labels must be non-empty")
    ordered = pd.DataFrame({"ts": decision, "environment": environment}).sort_values("ts", kind="stable")
    runs = ordered["environment"].ne(ordered["environment"].shift()).cumsum()
    # Each environment must occupy exactly one chronological run.
    run_counts = pd.DataFrame({"environment": ordered.environment, "run": runs}).groupby(
        "environment", observed=True
    )["run"].nunique()
    repeated = run_counts[run_counts > 1].index.tolist()
    if repeated:
        raise StageIIISequentialRunnerError(
            f"environments must be contiguous chronological blocks; repeated={repeated[:8]}"
        )
    env_order = ordered.drop_duplicates("environment", keep="first")["environment"].tolist()
    folds: list[ExpandingEnvironmentFold] = []
    for env_pos, name in enumerate(env_order[config.min_train_environments :], start=config.min_train_environments):
        valid = environment.eq(name).to_numpy()
        validation_positions = np.flatnonzero(valid)
        validation_start = pd.Timestamp(decision.iloc[validation_positions].min())
        validation_end = pd.Timestamp(decision.iloc[validation_positions].max())
        earlier_environments = set(env_order[:env_pos])
        train = environment.isin(earlier_environments).to_numpy() & (available < validation_start).to_numpy()
        train_positions = np.flatnonzero(train)
        if len(train_positions) < config.min_train_rows:
            continue
        train_side = frame.iloc[train_positions][config.side_column].astype(str).str.lower()
        if any(int(train_side.eq(side).sum()) < config.min_rows_per_side for side in ("long", "short")):
            continue
        folds.append(
            ExpandingEnvironmentFold(
                fold_id=len(folds), validation_environment=str(name),
                validation_start_utc=validation_start, validation_end_utc=validation_end,
                train_positions=train_positions, validation_positions=validation_positions,
                train_max_label_available_utc=pd.Timestamp(available.iloc[train_positions].max()),
            )
        )
    if not folds:
        raise StageIIISequentialRunnerError("no expanding environment fold satisfies training support")
    return tuple(folds)


def _frozen_comparison_rows(
    frame: pd.DataFrame, *, folds: Sequence[ExpandingEnvironmentFold],
    config: StageIIIRunnerConfig, soft_regime_columns: Sequence[str],
) -> np.ndarray:
    """Freeze one conservative held-out row universe for every A--E arm."""
    columns = SharedResidualColumns(
        decision_timestamp=config.decision_timestamp_column,
        label_available_timestamp=config.label_available_timestamp_column,
        side=config.side_column, exact_net_bps=config.exact_net_column,
        base_expected_net_bps=config.base_expected_net_column,
    )
    valid_targets: dict[str, np.ndarray] = {}
    for mode in _BASELINE_ARMS.values():
        prepared, _ = prepare_shared_regime_residual_frame(
            frame, soft_regime_columns=soft_regime_columns,
            regime_relative_feature_names=(), restricted_interaction_feature_names=(),
            columns=columns, baseline_config=config.baseline_config, baseline_mode=mode,
            hard_regime_column=config.hard_regime_column,
        )
        valid_targets[mode] = prepared["candidate_residual_bps"].notna().to_numpy()
    rows: list[int] = []
    for fold in folds:
        fold_supported = True
        for valid in valid_targets.values():
            train_positions = fold.train_positions[valid[fold.train_positions]]
            train_side = frame.iloc[train_positions][config.side_column].astype(str).str.lower()
            if len(train_positions) < config.min_train_rows or any(
                int(train_side.eq(side).sum()) < config.min_rows_per_side for side in ("long", "short")
            ):
                fold_supported = False
                break
        if not fold_supported:
            continue
        keep = np.ones(len(fold.validation_positions), dtype=bool)
        for valid in valid_targets.values():
            keep &= valid[fold.validation_positions]
        rows.extend(fold.validation_positions[keep].tolist())
    if not rows:
        raise StageIIISequentialRunnerError("no common strict-OOF rows survive all residual baselines")
    return np.asarray(sorted(set(rows)), dtype=np.int64)


def _conditioning_features(
    stack: StageIIIStack,
    *, invariant_features: Sequence[str], soft_regime_columns: Sequence[str],
    relative_generated: Sequence[str], interaction_generated: Sequence[str],
    validity_feature_groups: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    names = list(dict.fromkeys(str(x) for x in invariant_features))
    names.append("shared_residual_side_is_long")
    if stack.conditioning_mode != "C0_no_regime_features":
        names.extend(soft_regime_columns)
        names.append("soft_regime_entropy")
    if stack.conditioning_mode in {"C2_restricted_interactions", "C3_regime_relative_features"}:
        names.extend(interaction_generated)
    if stack.conditioning_mode == "C3_regime_relative_features":
        names.extend(relative_generated)
    validity_keys = {
        "D0_no_ood": (),
        "D1_relationship_breaks": ("relationship_breaks",),
        "D2_contribution_ood": ("contribution_ood",),
        "D3_active_failure_probability": ("active_failure_probability",),
        "D4_compact_combination": (
            "relationship_breaks", "contribution_ood", "active_failure_probability",
        ),
    }[stack.validity_mode]
    for key in validity_keys:
        if key not in validity_feature_groups or not validity_feature_groups[key]:
            raise StageIIISequentialRunnerError(f"validity arm requires non-empty feature group {key!r}")
        names.extend(validity_feature_groups[key])
    return tuple(dict.fromkeys(names))


def _finite_model_feature_mask(
    frame: pd.DataFrame, feature_names: Sequence[str], *, context: str,
) -> np.ndarray:
    missing = [name for name in feature_names if name not in frame]
    if missing:
        raise StageIIISequentialRunnerError(f"{context} lacks model features: {missing[:12]}")
    matrix = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.any():
        raise StageIIISequentialRunnerError(f"{context} has no finite complete model-feature rows")
    return finite


def _stack_baseline_config(
    config: StageIIIRunnerConfig, stack: StageIIIStack,
) -> SoftRegimeResidualConfig:
    return replace(
        config.baseline_config, target_clip_bps=float(stack.residual_target_clip_bps)
    )


def _fit_round_f_model(
    train: pd.DataFrame,
    *,
    stack: StageIIIStack,
    pairwise_arm: str,
    feature_names: Sequence[str],
    soft_regime_columns: Sequence[str],
    fit_before_utc: object,
    columns: PairwiseSharedResidualColumns,
    config: StageIIIRunnerConfig,
    sample_weight: Sequence[float] | None,
    pairwise_expert_fitter: PairwiseExpertFitter,
    target_preserving_pairwise_fitter: TargetPreservingPairwiseFitter,
    ordinal_target_fitter: OrdinalTargetFitter,
    quantile_target_fitter: QuantileTargetFitter,
) -> PairwiseSharedResidualExpertFit | TargetPreservingPairwiseAdapterFit:
    """Fit Round F without changing the frozen Round-T target semantics.

    T0--T2 use the native pointwise/pairwise shared expert.  T3/T4 first refit
    their exact audited target model on the same prior-resolved ledger and then
    add the auxiliary rank head.  The adapter's F0 path is a strict no-op over
    that target model; no Huber fallback is permitted.
    """
    if stack.residual_target_mode in {"ordinal", "quantile"}:
        robust_columns = RobustTargetColumns(
            decision_timestamp=columns.decision_timestamp,
            label_available_timestamp=columns.label_available_timestamp,
            side=columns.side,
            candidate_id=columns.candidate_id,
            exact_net_bps=columns.exact_net_bps,
            base_expected_net_bps=columns.base_expected_net_bps,
        )
        if stack.residual_target_mode == "ordinal":
            base_model = ordinal_target_fitter(
                train,
                feature_names=feature_names,
                fit_before_utc=fit_before_utc,
                columns=robust_columns,
                config=config.robust_target_model_config,
                sample_weight=sample_weight,
            )
        else:
            base_model = quantile_target_fitter(
                train,
                feature_names=feature_names,
                fit_before_utc=fit_before_utc,
                columns=robust_columns,
                config=config.robust_target_model_config,
                sample_weight=sample_weight,
            )
        return target_preserving_pairwise_fitter(
            train,
            base_model=base_model,
            arm=pairwise_arm,
            feature_names=feature_names,
            soft_regime_columns=soft_regime_columns,
            fit_before_utc=fit_before_utc,
            columns=columns,
            pair_config=config.pair_construction_config,
            config=config.pairwise_config,
            sample_weight=sample_weight,
        )
    return pairwise_expert_fitter(
        train,
        arm=pairwise_arm,
        feature_names=feature_names,
        soft_regime_columns=soft_regime_columns,
        fit_before_utc=fit_before_utc,
        columns=columns,
        pair_config=config.pair_construction_config,
        config=config.pairwise_config,
        pointwise_target_mode=stack.residual_target_mode,
        pointwise_params=config.hpo_param_candidates[stack.model_params_index],
        sample_weight=sample_weight,
    )


def _robust_target_columns(config: StageIIIRunnerConfig) -> RobustTargetColumns:
    return RobustTargetColumns(
        decision_timestamp=config.decision_timestamp_column,
        label_available_timestamp=config.label_available_timestamp_column,
        side=config.side_column, candidate_id="__candidate_identity",
        exact_net_bps=config.exact_net_column,
        base_expected_net_bps=config.base_expected_net_column,
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def _huber_mean(error: np.ndarray, delta: float = 100.0) -> float:
    absolute = np.abs(error)
    return float(np.mean(np.where(absolute <= delta, 0.5 * error * error, delta * (absolute - 0.5 * delta))))


def _economic_lift_by_environment(
    predictions: pd.DataFrame, *, config: StageIIIRunnerConfig
) -> dict[str, float]:
    return {
        environment: values["lift_bps"]
        for environment, values in _environment_top_economics(predictions, config=config).items()
    }


def _environment_top_economics(
    predictions: pd.DataFrame, *, config: StageIIIRunnerConfig,
) -> dict[str, dict[str, float]]:
    population = predictions
    if config.selection_admission_scope == "with_21d":
        population = population.loc[_canonical_boolean(
            population[config.admission_column], name=config.admission_column
        )]
    decision = _utc(population, config.decision_timestamp_column)
    ordered_environments = (
        pd.DataFrame({"environment": population[config.environment_column].astype(str), "decision": decision})
        .groupby("environment", observed=True)["decision"].min().sort_values(kind="stable").index.tolist()
    )
    values: dict[str, dict[str, float]] = {}
    for environment in ordered_environments:
        local = population.loc[population[config.environment_column].astype(str).eq(environment)]
        k = max(1, int(np.ceil(config.primary_top_fraction * len(local))))
        model_top = local.sort_values(
            ["score_bps", "__candidate_identity"], ascending=[False, True], kind="stable"
        ).head(k)
        base_top = local.sort_values(
            [config.base_expected_net_column, "__candidate_identity"],
            ascending=[False, True], kind="stable"
        ).head(k)
        model_net = float(model_top[config.exact_net_column].mean())
        base_net = float(base_top[config.exact_net_column].mean())
        values[str(environment)] = {
            "model_top_net_bps": model_net,
            "base_top_net_bps": base_net,
            "lift_bps": model_net - base_net,
        }
    if len(values) < 2:
        raise StageIIISequentialRunnerError("robust selection requires at least two OOF environments")
    return values


def _paired_pooled_top_lift(predictions: pd.DataFrame, *, config: StageIIIRunnerConfig) -> float:
    population = predictions
    if config.selection_admission_scope == "with_21d":
        population = population.loc[_canonical_boolean(
            population[config.admission_column], name=config.admission_column
        )]
    if population.empty:
        raise StageIIISequentialRunnerError("selected admission population is empty")
    k = max(1, int(np.ceil(config.primary_top_fraction * len(population))))
    model_top = population.sort_values(
        ["score_bps", "__candidate_identity"], ascending=[False, True], kind="stable"
    ).head(k)
    base_top = population.sort_values(
        [config.base_expected_net_column, "__candidate_identity"], ascending=[False, True], kind="stable"
    ).head(k)
    return float(model_top[config.exact_net_column].mean() - base_top[config.exact_net_column].mean())


def _selection_summary(predictions: pd.DataFrame, *, config: StageIIIRunnerConfig) -> dict[str, float]:
    summary = robust_cross_era_selection_score(
        _economic_lift_by_environment(predictions, config=config),
        worst_penalty=config.worst_penalty, dispersion_penalty=config.dispersion_penalty,
    )
    summary["pooled_top_lift_bps"] = _paired_pooled_top_lift(predictions, config=config)
    return summary


def _common_oof_rows(frames: Sequence[pd.DataFrame]) -> tuple[str, ...]:
    if not frames:
        raise StageIIISequentialRunnerError("cannot align an empty OOF comparison")
    common = set(frames[0]["__candidate_identity"].astype(str).tolist())
    for frame in frames[1:]:
        common.intersection_update(frame["__candidate_identity"].astype(str).tolist())
    if not common:
        raise StageIIISequentialRunnerError("compared arms have no common strict-OOF rows")
    return tuple(sorted(common))


def _restrict_oof(frame: pd.DataFrame, common_rows: Sequence[str]) -> pd.DataFrame:
    order = {str(value): position for position, value in enumerate(common_rows)}
    out = frame.loc[frame["__candidate_identity"].astype(str).isin(order)].copy()
    out["__comparison_order"] = out["__candidate_identity"].astype(str).map(order)
    out = out.sort_values("__comparison_order", kind="stable").drop(columns="__comparison_order")
    if len(out) != len(common_rows):
        raise StageIIISequentialRunnerError("OOF row identity is duplicated or missing after alignment")
    return out


def _metrics_one_population(
    predictions: pd.DataFrame, *, config: StageIIIRunnerConfig, admission_scope: str,
) -> list[dict[str, Any]]:
    """Report pooled-global tails and their unchanged selected contributions."""
    work = predictions.copy()
    work["month"] = _utc(work, config.month_timestamp_column).dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    error = work[config.exact_net_column].to_numpy(float) - work["score_bps"].to_numpy(float)
    rows.append({
        "admission_scope": admission_scope,
        "scope": "pooled_oof", "top_fraction": np.nan, "month": "__all__",
        "side": "__all__", "environment": "__all__", "rows": len(work),
        "selected_rows": len(work), "net_bps_per_trade": float(work[config.exact_net_column].mean()),
        "residual_ic": _spearman(work["predicted_candidate_residual_bps"].to_numpy(float), work["candidate_residual_bps"].to_numpy(float)),
        "score_net_ic": _spearman(work["score_bps"].to_numpy(float), work[config.exact_net_column].to_numpy(float)),
        "mae_bps": float(np.mean(np.abs(error))), "huber_loss": _huber_mean(error),
        "calibration_slope": float(np.polyfit(work["score_bps"], work[config.exact_net_column], 1)[0]) if work["score_bps"].nunique() > 1 else np.nan,
        "calibration_intercept": float(np.polyfit(work["score_bps"], work[config.exact_net_column], 1)[1]) if work["score_bps"].nunique() > 1 else np.nan,
        "gross_bps_per_trade": (
            float(work[config.exact_gross_column].mean()) if config.exact_gross_column else np.nan
        ),
    })
    ordered = work.sort_values(
        ["score_bps", "__candidate_identity"], ascending=[False, True], kind="stable"
    )
    for fraction in config.top_fractions:
        k = max(1, int(np.ceil(fraction * len(ordered))))
        selected = ordered.head(k)
        common = {"top_fraction": fraction, "candidate_rows": len(work), "selected_global_rows": len(selected)}
        rows.append({
            **common, "admission_scope": admission_scope,
            "scope": "pooled_global_tail", "month": "__all__", "side": "__all__",
            "environment": "__all__", "rows": len(work), "selected_rows": len(selected),
            "net_bps_per_trade": float(selected[config.exact_net_column].mean()),
            "gross_bps_per_trade": (
                float(selected[config.exact_gross_column].mean()) if config.exact_gross_column else np.nan
            ),
        })
        for (month, side, environment), local in selected.groupby(
            ["month", config.side_column, config.environment_column], sort=True, observed=True
        ):
            rows.append({
                **common, "admission_scope": admission_scope,
                "scope": "pooled_global_selected_contribution", "month": str(month),
                "side": str(side), "environment": str(environment), "rows": len(work),
                "selected_rows": len(local), "net_bps_per_trade": float(local[config.exact_net_column].mean()),
                "gross_bps_per_trade": (
                    float(local[config.exact_gross_column].mean()) if config.exact_gross_column else np.nan
                ),
            })
    # Diagnostic within-environment quality; never used as a production admission/ranking rule.
    for environment, local in work.groupby(config.environment_column, sort=True, observed=True):
        local_error = local[config.exact_net_column].to_numpy(float) - local["score_bps"].to_numpy(float)
        rows.append({
            "admission_scope": admission_scope,
            "scope": "environment_diagnostic", "top_fraction": np.nan, "month": "__all__",
            "side": "__all__", "environment": str(environment), "rows": len(local),
            "selected_rows": len(local), "net_bps_per_trade": float(local[config.exact_net_column].mean()),
            "residual_ic": _spearman(local["predicted_candidate_residual_bps"].to_numpy(float), local["candidate_residual_bps"].to_numpy(float)),
            "score_net_ic": _spearman(local["score_bps"].to_numpy(float), local[config.exact_net_column].to_numpy(float)),
            "mae_bps": float(np.mean(np.abs(local_error))), "huber_loss": _huber_mean(local_error),
        })
    for dimension, column in (("month", "month"), ("side", config.side_column)):
        for value, local in work.groupby(column, sort=True, observed=True):
            local_error = local[config.exact_net_column].to_numpy(float) - local["score_bps"].to_numpy(float)
            rows.append({
                "admission_scope": admission_scope,
                "scope": f"{dimension}_diagnostic", "top_fraction": np.nan,
                "month": str(value) if dimension == "month" else "__all__",
                "side": str(value) if dimension == "side" else "__all__",
                "environment": "__all__", "rows": len(local), "selected_rows": len(local),
                "net_bps_per_trade": float(local[config.exact_net_column].mean()),
                "residual_ic": _spearman(local["predicted_candidate_residual_bps"].to_numpy(float), local["candidate_residual_bps"].to_numpy(float)),
                "score_net_ic": _spearman(local["score_bps"].to_numpy(float), local[config.exact_net_column].to_numpy(float)),
                "mae_bps": float(np.mean(np.abs(local_error))), "huber_loss": _huber_mean(local_error),
            })
    return rows


def _metrics(predictions: pd.DataFrame, *, config: StageIIIRunnerConfig) -> pd.DataFrame:
    rows = _metrics_one_population(
        predictions, config=config, admission_scope="without_21d",
    )
    admitted = predictions.loc[_canonical_boolean(
        predictions[config.admission_column], name=config.admission_column
    )].copy()
    if admitted.empty:
        raise StageIIISequentialRunnerError("causal 21-day admission rejects every OOF row")
    rows.extend(_metrics_one_population(admitted, config=config, admission_scope="with_21d"))
    return pd.DataFrame(rows)


def _calibration_library_mode(mode: str) -> str:
    return {
        "E0_global": "C0_global",
        "E1_side_local": "C1_side",
        "E2_side_soft_regime_hierarchical": "C2_side_soft_regime",
    }[mode]


def _run_oof_candidate(
    prepared: pd.DataFrame,
    *, stack: StageIIIStack, params: Mapping[str, Any], folds: Sequence[ExpandingEnvironmentFold],
    config: StageIIIRunnerConfig, feature_names: Sequence[str], soft_regime_columns: Sequence[str],
    expert_fitter: ExpertFitter,
    ordinal_target_fitter: OrdinalTargetFitter,
    quantile_target_fitter: QuantileTargetFitter,
    comparison_rows: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    prediction = np.full(len(prepared), np.nan, dtype=np.float32)
    fold_id = np.full(len(prepared), -1, dtype=np.int16)
    audits: list[dict[str, Any]] = []
    model_audits: list[dict[str, Any]] = []
    columns = SharedResidualColumns(
        decision_timestamp=config.decision_timestamp_column,
        label_available_timestamp=config.label_available_timestamp_column,
        side=config.side_column, exact_net_bps=config.exact_net_column,
        base_expected_net_bps=config.base_expected_net_column,
    )
    fit_count = 0
    feature_complete = _finite_model_feature_mask(prepared, feature_names, context="shared expert frame")
    for fold in folds:
        train_positions = fold.train_positions[feature_complete[fold.train_positions]]
        train = prepared.iloc[train_positions]
        train = train.loc[train["candidate_residual_bps"].notna()].copy()
        if len(train) < config.min_train_rows:
            continue
        train_side = train[config.side_column].astype(str).str.lower()
        if any(int(train_side.eq(side).sum()) < config.min_rows_per_side for side in ("long", "short")):
            continue
        weights = mild_environment_weights(
            train, environment_column=config.environment_column,
            soft_regime_columns=soft_regime_columns,
            balance=("era" if stack.balance_mode == "era" else "natural"),
        )
        if stack.residual_target_mode == "ordinal":
            fit = ordinal_target_fitter(
                train, feature_names=feature_names, fit_before_utc=fold.validation_start_utc,
                columns=_robust_target_columns(config), config=config.robust_target_model_config,
                sample_weight=weights,
            )
        elif stack.residual_target_mode == "quantile":
            fit = quantile_target_fitter(
                train, feature_names=feature_names, fit_before_utc=fold.validation_start_utc,
                columns=_robust_target_columns(config), config=config.robust_target_model_config,
                sample_weight=weights,
            )
        else:
            fit = expert_fitter(
                train, feature_names=feature_names, fit_before_utc=fold.validation_start_utc,
                columns=columns, target_mode=stack.residual_target_mode,
                sample_weight=weights, params=params,
            )
        fit_count += 1
        validation_positions = fold.validation_positions[feature_complete[fold.validation_positions]]
        valid = prepared.iloc[validation_positions]
        if valid.empty:
            continue
        correction = fit.predict_candidate_residual_bps(valid)
        prediction[validation_positions] = correction
        fold_id[validation_positions] = fold.fold_id
        exact_audit = getattr(fit, "audit", None)
        if exact_audit is not None and hasattr(exact_audit, "to_dict"):
            model_audits.append({"fold_id": fold.fold_id, **exact_audit.to_dict()})
            max_available = exact_audit.max_label_available_utc
        else:
            max_available = fit.max_label_available_utc
            model_audits.append({
                "fold_id": fold.fold_id, "arm": stack.residual_target_arm,
                "routing": "one_shared_both_side_model_no_local_or_hard_routing",
                "feature_names": list(feature_names),
                "feature_sha256": stage_iii_feature_contract_sha256(feature_names),
                "training_row_count": len(train),
                "training_candidate_ids_sha256": _identity_set_sha256(train["__candidate_identity"]),
                "training_cutoff_utc": fold.validation_start_utc,
                "max_label_available_utc": max_available,
                "target_mode": stack.residual_target_mode,
                "target_clip_bps": stack.residual_target_clip_bps,
            })
        audits.append({
            "fold_id": fold.fold_id, "validation_environment": fold.validation_environment,
            "validation_start_utc": fold.validation_start_utc, "validation_end_utc": fold.validation_end_utc,
            "train_rows": len(train), "validation_rows": len(valid),
            "train_side_count": int(train_side.nunique()),
            "train_max_label_available_utc": max_available,
            "shared_model_fit": True, "hard_routing": False,
            "train_candidate_identity_sha256": _identity_set_sha256(train["__candidate_identity"]),
            "validation_candidate_identity_sha256": _identity_set_sha256(valid["__candidate_identity"]),
        })
    keep = np.isfinite(prediction)
    comparison_mask = np.zeros(len(prepared), dtype=bool)
    comparison_mask[comparison_rows] = True
    keep &= comparison_mask
    if not keep.any():
        raise StageIIISequentialRunnerError("candidate produced no strict OOF predictions")
    if not np.array_equal(np.flatnonzero(keep), comparison_rows):
        missing = np.setdiff1d(comparison_rows, np.flatnonzero(keep))
        raise StageIIISequentialRunnerError(
            f"arm failed the frozen identical-OOF-row contract; missing={missing[:12].tolist()}"
        )
    out = prepared.loc[keep].copy()
    out["predicted_candidate_residual_bps"] = prediction[keep]
    out["fold_id"] = fold_id[keep]
    out["raw_shared_common_bps"] = reconstruct_shared_regime_expected_net_bps(
        out, prediction[keep], columns=columns,
    )
    out["score_bps"] = out["raw_shared_common_bps"].astype(np.float32)
    if "__candidate_identity" not in out or out["__candidate_identity"].duplicated().any():
        raise StageIIISequentialRunnerError("OOF prediction did not preserve immutable candidate identity")
    return out, pd.DataFrame(audits), fit_count, pd.DataFrame(model_audits)


def _evaluate_arm(
    frame: pd.DataFrame, *, arm: str, round_name: str, stack: StageIIIStack,
    config: StageIIIRunnerConfig, folds: Sequence[ExpandingEnvironmentFold],
    soft_regime_columns: Sequence[str], invariant_features: Sequence[str],
    regime_relative_features: Sequence[str], restricted_interaction_features: Sequence[str],
    validity_feature_groups: Mapping[str, Sequence[str]], expert_fitter: ExpertFitter,
    ordinal_target_fitter: OrdinalTargetFitter,
    quantile_target_fitter: QuantileTargetFitter,
    comparison_rows: np.ndarray,
    source_feature_contract_sha256: str,
) -> StageIIIArmResult:
    columns = SharedResidualColumns(
        decision_timestamp=config.decision_timestamp_column,
        label_available_timestamp=config.label_available_timestamp_column,
        side=config.side_column, exact_net_bps=config.exact_net_column,
        base_expected_net_bps=config.base_expected_net_column,
    )
    use_relative = stack.conditioning_mode == "C3_regime_relative_features"
    use_interactions = stack.conditioning_mode in {
        "C2_restricted_interactions", "C3_regime_relative_features",
    }
    prepared, generated = prepare_shared_regime_residual_frame(
        frame, soft_regime_columns=soft_regime_columns,
        regime_relative_feature_names=(regime_relative_features if use_relative else ()),
        restricted_interaction_feature_names=(restricted_interaction_features if use_interactions else ()),
        columns=columns, baseline_config=_stack_baseline_config(config, stack),
        baseline_mode=stack.baseline_mode,
        hard_regime_column=config.hard_regime_column,
    )
    relative_generated = [x for x in generated if x.startswith("__srre__")]
    interaction_generated = [x for x in generated if x.startswith("__srre_interaction__")]
    feature_names = _conditioning_features(
        stack, invariant_features=invariant_features, soft_regime_columns=soft_regime_columns,
        relative_generated=relative_generated, interaction_generated=interaction_generated,
        validity_feature_groups=validity_feature_groups,
    )
    missing = [name for name in feature_names if name not in prepared]
    if missing:
        raise StageIIISequentialRunnerError(f"arm {arm} lacks frozen causal features: {missing[:12]}")
    candidates: list[
        tuple[pd.DataFrame, pd.DataFrame, int, dict[str, float], pd.DataFrame]
    ] = []
    pointwise_compatible = stack.residual_target_mode in {"huber", "clipped", "regime_standardized"}
    if round_name == "B_training_robustness" and stack.robust_hpo and pointwise_compatible:
        param_indices = tuple(range(len(config.hpo_param_candidates)))
    else:
        param_indices = (stack.model_params_index,)
    param_set = tuple(config.hpo_param_candidates[index] for index in param_indices)
    for params in param_set:
        oof, audit, count, model_audit = _run_oof_candidate(
            prepared, stack=stack, params=params, folds=folds, config=config,
            feature_names=feature_names, soft_regime_columns=soft_regime_columns,
            expert_fitter=expert_fitter, comparison_rows=comparison_rows,
            ordinal_target_fitter=ordinal_target_fitter,
            quantile_target_fitter=quantile_target_fitter,
        )
        candidates.append((oof, audit, count, {}, model_audit))
    common_rows = _common_oof_rows([candidate[0] for candidate in candidates])
    candidates = [
        (
            _restrict_oof(oof, common_rows), audit, count,
            _selection_summary(_restrict_oof(oof, common_rows), config=config),
            model_audit,
        )
        for oof, audit, count, _, model_audit in candidates
    ]
    if round_name == "B_training_robustness" and stack.robust_hpo:
        selected_index = max(range(len(candidates)), key=lambda i: (candidates[i][3]["selection_score"], -i))
    else:
        selected_index = max(range(len(candidates)), key=lambda i: (candidates[i][3]["pooled_top_lift_bps"], -i))
    oof, fold_audit, fit_count, selection, model_audit = candidates[selected_index]
    return StageIIIArmResult(
        arm=arm, round_name=round_name,
        stack=replace(stack, model_params_index=param_indices[selected_index]),
        selected_params_index=param_indices[selected_index],
        selected_params=dict(param_set[selected_index]), selection_summary=selection,
        oof_predictions=oof, metrics=_metrics(oof, config=config), fold_audit=fold_audit,
        calibration_audit=pd.DataFrame(), shared_model_fit_count=fit_count,
        model_feature_names=tuple(feature_names),
        model_feature_contract_sha256=stage_iii_feature_contract_sha256(feature_names),
        source_feature_contract_sha256=source_feature_contract_sha256,
        model_audits=model_audit,
    )


def _winner(results: Sequence[StageIIIArmResult]) -> StageIIIArmResult:
    if not results:
        raise StageIIISequentialRunnerError("cannot select a winner from an empty round")
    # Worst-era robustness is primary; stable input order is the complexity tie-break.
    return max(results, key=lambda result: float(result.selection_summary["selection_score"]))


def _align_round_results(
    results: Sequence[StageIIIArmResult], *, config: StageIIIRunnerConfig,
) -> list[StageIIIArmResult]:
    common_rows = _common_oof_rows([result.oof_predictions for result in results])
    aligned: list[StageIIIArmResult] = []
    for result in results:
        oof = _restrict_oof(result.oof_predictions, common_rows)
        aligned.append(replace(
            result, oof_predictions=oof,
            selection_summary=_selection_summary(oof, config=config),
            metrics=_metrics(oof, config=config),
        ))
    return aligned


def run_train_test_transport_matrix(
    frame: pd.DataFrame, *, stack: StageIIIStack, config: StageIIIRunnerConfig,
    final_arm: str, final_predecessor_arm: str | None, final_stack_identity: str,
    model_feature_contract_sha256: str,
    soft_regime_columns: Sequence[str], invariant_features: Sequence[str],
    regime_relative_features: Sequence[str], restricted_interaction_features: Sequence[str],
    validity_feature_groups: Mapping[str, Sequence[str]],
    pairwise_expert_fitter: PairwiseExpertFitter = fit_pairwise_shared_residual_expert,
    target_preserving_pairwise_fitter: TargetPreservingPairwiseFitter = fit_target_preserving_pairwise_adapter,
    ordinal_target_fitter: OrdinalTargetFitter = fit_ordinal_shared_robust_target,
    quantile_target_fitter: QuantileTargetFitter = fit_quantile_shared_robust_target,
) -> pd.DataFrame:
    """Replay the exact final Round-F + E-calibration stack across eras.

    These fits are transport diagnostics only.  They neither route production
    rows nor replace the expanding shared expert selected by the funnel.
    """
    _validate_input(frame, config)
    if "__candidate_identity" not in frame:
        frame = frame.copy().reset_index(drop=True)
        frame["__candidate_identity"] = _candidate_identity(frame, config).to_numpy()
    columns = SharedResidualColumns(
        decision_timestamp=config.decision_timestamp_column,
        label_available_timestamp=config.label_available_timestamp_column,
        side=config.side_column, exact_net_bps=config.exact_net_column,
        base_expected_net_bps=config.base_expected_net_column,
    )
    use_relative = stack.conditioning_mode == "C3_regime_relative_features"
    use_interactions = stack.conditioning_mode in {"C2_restricted_interactions", "C3_regime_relative_features"}
    prepared, generated = prepare_shared_regime_residual_frame(
        frame, soft_regime_columns=soft_regime_columns,
        regime_relative_feature_names=(regime_relative_features if use_relative else ()),
        restricted_interaction_feature_names=(restricted_interaction_features if use_interactions else ()),
        columns=columns, baseline_config=_stack_baseline_config(config, stack),
        baseline_mode=stack.baseline_mode, hard_regime_column=config.hard_regime_column,
    )
    features = _conditioning_features(
        stack, invariant_features=invariant_features, soft_regime_columns=soft_regime_columns,
        relative_generated=[x for x in generated if x.startswith("__srre__")],
        interaction_generated=[x for x in generated if x.startswith("__srre_interaction__")],
        validity_feature_groups=validity_feature_groups,
    )
    feature_complete = _finite_model_feature_mask(
        prepared, features, context="transport model frame"
    )
    decision = _utc(prepared, config.decision_timestamp_column)
    available = _utc(prepared, config.label_available_timestamp_column)
    env = prepared[config.environment_column].astype(str)
    order = (
        pd.DataFrame({"environment": env, "decision": decision})
        .groupby("environment", observed=True)["decision"].min().sort_values().index.tolist()
    )
    records: list[dict[str, Any]] = []
    base_arm = str(final_arm).split("@", 1)[0]
    if base_arm not in _PAIRWISE_ARMS:
        raise StageIIISequentialRunnerError(f"transport final arm is not a Round-F arm: {final_arm!r}")
    pairwise_arm = _PAIRWISE_ARMS[base_arm]
    pairwise_columns = PairwiseSharedResidualColumns(
        decision_timestamp=config.decision_timestamp_column,
        label_available_timestamp=config.label_available_timestamp_column,
        side=config.side_column, candidate_id="__candidate_identity",
        exact_net_bps=config.exact_net_column,
        base_expected_net_bps=config.base_expected_net_column,
        cost_to_atr=config.cost_to_atr_column,
        base_map_prequential_flag="base_map_is_prequential",
        soft_regime_causal_flag="soft_regime_is_causal_prequential",
        cost_atr_causal_flag=config.cost_atr_causal_flag_column,
    )
    for test_position, test_environment in enumerate(order[1:], start=1):
        test_mask = env.eq(test_environment).to_numpy()
        test = prepared.loc[test_mask].copy()
        cutoff = pd.Timestamp(decision.loc[test_mask].min())
        for train_environment in order[:test_position]:
            train_mask = (
                env.eq(train_environment).to_numpy()
                & (available < cutoff).to_numpy()
                & feature_complete
            )
            train = prepared.loc[train_mask & prepared["candidate_residual_bps"].notna().to_numpy()].copy()
            side = train[config.side_column].astype(str).str.lower()
            if len(train) < config.min_train_rows or any(
                int(side.eq(name).sum()) < config.min_rows_per_side for name in ("long", "short")
            ):
                records.append({
                    "train_environment": str(train_environment), "test_environment": str(test_environment),
                    "status": "insufficient_train_support", "train_rows": len(train), "test_rows": len(test),
                })
                continue
            weights = mild_environment_weights(
                train, environment_column=config.environment_column,
                soft_regime_columns=soft_regime_columns,
                balance=("era" if stack.balance_mode == "era" else "natural"),
            )
            fit = _fit_round_f_model(
                train, stack=stack, pairwise_arm=pairwise_arm,
                feature_names=features, soft_regime_columns=soft_regime_columns,
                fit_before_utc=cutoff, columns=pairwise_columns, config=config,
                sample_weight=weights, pairwise_expert_fitter=pairwise_expert_fitter,
                target_preserving_pairwise_fitter=target_preserving_pairwise_fitter,
                ordinal_target_fitter=ordinal_target_fitter,
                quantile_target_fitter=quantile_target_fitter,
            )
            if not feature_complete[test_mask].all():
                raise StageIIISequentialRunnerError(
                    f"transport test environment {test_environment!r} has incomplete model features"
                )
            train_correction = fit.predict_candidate_residual_bps(train)
            train_raw = reconstruct_shared_regime_expected_net_bps(
                train, train_correction, columns=columns,
            )
            calibrator = fit_shared_bps_calibration(
                train, train_raw, train[config.exact_net_column].to_numpy(float),
                fit_before_utc=cutoff,
                mode=_calibration_library_mode(stack.calibration_mode),
                resolution_column=config.label_available_timestamp_column,
                side_column=config.side_column,
                soft_regime_columns=soft_regime_columns,
                min_global_rows=config.calibration_min_rows,
            )
            correction = fit.predict_candidate_residual_bps(test)
            raw_score = reconstruct_shared_regime_expected_net_bps(test, correction, columns=columns)
            score = predict_shared_bps_calibration(
                calibrator, test, raw_score,
                decision_timestamp_column=config.decision_timestamp_column,
                side_column=config.side_column,
            )
            test["score_bps"] = score
            test["predicted_candidate_residual_bps"] = correction
            error = test[config.exact_net_column].to_numpy(float) - np.asarray(score, dtype=float)
            slope, intercept = (np.nan, np.nan)
            if pd.Series(score).nunique() > 1:
                slope, intercept = np.polyfit(score, test[config.exact_net_column], 1)
            support = fit.audit.pair_support.to_dict()
            preserved = getattr(fit.audit, "preserved_base_target", None)
            ordered_model = test.sort_values(
                ["score_bps", "__candidate_identity"], ascending=[False, True], kind="stable"
            )
            ordered_base = test.sort_values(
                [config.base_expected_net_column, "__candidate_identity"],
                ascending=[False, True], kind="stable",
            )
            for fraction in (0.01, 0.05, 0.10, 0.20):
                k = max(1, int(np.ceil(float(fraction) * len(test))))
                selected = ordered_model.head(k)
                base_selected = ordered_base.head(k)
                by_side = {
                    str(side_name): local
                    for side_name, local in selected.groupby(config.side_column, observed=True)
                }
                records.append({
                    "train_environment": str(train_environment),
                    "test_environment": str(test_environment),
                    "status": "strict_prior_resolved_transport",
                    "top_fraction": float(fraction),
                    "is_primary_top_fraction": bool(np.isclose(fraction, config.primary_top_fraction)),
                    "train_rows": len(train), "test_rows": len(test), "selected_rows": len(selected),
                    "train_max_label_available_utc": pd.Timestamp(available.loc[train.index].max()),
                    "test_start_utc": cutoff,
                    "paired_top_lift_bps": float(
                        selected[config.exact_net_column].mean()
                        - base_selected[config.exact_net_column].mean()
                    ),
                    "top_net_bps": float(selected[config.exact_net_column].mean()),
                    "top_gross_bps": float(selected[config.exact_gross_column].mean()),
                    "mae_bps": float(np.mean(np.abs(error))), "huber_loss": _huber_mean(error),
                    "calibration_slope": float(slope), "calibration_intercept": float(intercept),
                    "score_net_ic": _spearman(score, test[config.exact_net_column].to_numpy(float)),
                    "residual_ic": _spearman(correction, test["candidate_residual_bps"].to_numpy(float)),
                    "long_selected_rows": len(by_side.get("long", ())),
                    "short_selected_rows": len(by_side.get("short", ())),
                    "long_net_bps": (
                        float(by_side["long"][config.exact_net_column].mean()) if "long" in by_side else np.nan
                    ),
                    "short_net_bps": (
                        float(by_side["short"][config.exact_net_column].mean()) if "short" in by_side else np.nan
                    ),
                    "long_gross_bps": (
                        float(by_side["long"][config.exact_gross_column].mean()) if "long" in by_side else np.nan
                    ),
                    "short_gross_bps": (
                        float(by_side["short"][config.exact_gross_column].mean()) if "short" in by_side else np.nan
                    ),
                    "shared_model": True, "hard_routing": False,
                    "round_f_arm": final_arm, "round_f_primitive": pairwise_arm,
                    "e_predecessor_arm": final_predecessor_arm,
                    "calibration_mode": stack.calibration_mode,
                    "final_stack_identity": final_stack_identity,
                    "model_feature_contract_sha256": model_feature_contract_sha256,
                    "pair_selected_pairs": support["selected_pairs"],
                    "pair_selected_ledger_sha256": support["selected_pair_ledger_sha256"],
                    "preserved_base_target_arm": (
                        None if preserved is None else preserved.base_target_arm
                    ),
                    "preserved_base_target_label_sha256": (
                        None if preserved is None else preserved.base_target_label_sha256
                    ),
                    "preserved_base_training_prediction_sha256": (
                        None if preserved is None else preserved.base_training_prediction_sha256
                    ),
                    "train_candidate_identity_sha256": _identity_set_sha256(train["__candidate_identity"]),
                    "test_candidate_identity_sha256": _identity_set_sha256(test["__candidate_identity"]),
                    "selected_candidate_identity_sha256": _identity_set_sha256(selected["__candidate_identity"]),
                    "base_selected_candidate_identity_sha256": _identity_set_sha256(base_selected["__candidate_identity"]),
                })
    return pd.DataFrame(records)


def stage_iii_advancement_gates(
    predictions: pd.DataFrame, transport_matrix: pd.DataFrame, *, config: StageIIIRunnerConfig,
    expected_transport_stack_identity: str,
) -> dict[str, Any]:
    """Evaluate explicit worst-era/latest/side/transport advancement gates."""
    environment_economics = _environment_top_economics(predictions, config=config)
    lifts = {name: value["lift_bps"] for name, value in environment_economics.items()}
    values = np.asarray(list(lifts.values()), dtype=float)
    model_environment_values = np.asarray(
        [value["model_top_net_bps"] for value in environment_economics.values()], dtype=float
    )
    base_environment_values = np.asarray(
        [value["base_top_net_bps"] for value in environment_economics.values()], dtype=float
    )
    latest_environment = list(lifts)[-1]
    side_lifts: dict[str, float] = {}
    for side, local in predictions.groupby(config.side_column, sort=True, observed=True):
        side_lifts[str(side)] = _paired_pooled_top_lift(local, config=config)
    required_transport_columns = {
        "status", "is_primary_top_fraction", "final_stack_identity", "paired_top_lift_bps",
    }
    if not required_transport_columns.issubset(transport_matrix.columns):
        transport = transport_matrix.iloc[0:0].copy()
        stack_identity_matches = False
    else:
        transport = transport_matrix.loc[
            transport_matrix.status.eq("strict_prior_resolved_transport")
            & transport_matrix.is_primary_top_fraction.eq(True)
        ]
        stack_identity_matches = bool(
            not transport.empty
            and transport.final_stack_identity.astype(str).eq(expected_transport_stack_identity).all()
        )
    transport_values = pd.to_numeric(transport.get("paired_top_lift_bps", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(float)
    required_positive = int(np.ceil(config.min_positive_environment_fraction * len(values)))
    gates = {
        "environment_lifts_bps": lifts,
        "environment_top_economics": environment_economics,
        "positive_environment_count": int((values > 0).sum()),
        "required_positive_environment_count": required_positive,
        "worst_environment_lift_bps": float(values.min()),
        "environment_dispersion_bps": float(values.std(ddof=0)),
        "model_environment_net_dispersion_bps": float(model_environment_values.std(ddof=0)),
        "frozen_base_environment_net_dispersion_bps": float(base_environment_values.std(ddof=0)),
        "model_worst_environment_net_bps": float(model_environment_values.min()),
        "frozen_base_worst_environment_net_bps": float(base_environment_values.min()),
        "catastrophic_environment_count": int((values <= config.catastrophic_lift_bps).sum()),
        "latest_environment": latest_environment,
        "latest_environment_lift_bps": float(lifts[latest_environment]),
        "side_lifts_bps": side_lifts,
        "side_failure_count": int(sum(value <= 0 for value in side_lifts.values())),
        "transport_cell_count": int(len(transport_values)),
        "positive_transport_cell_count": int((transport_values > 0).sum()),
        "positive_transport_cell_fraction": (
            float((transport_values > 0).mean()) if len(transport_values) else 0.0
        ),
        "worst_transport_lift_bps": float(transport_values.min()) if len(transport_values) else np.nan,
        "expected_transport_stack_identity": expected_transport_stack_identity,
        "transport_stack_identity_matches_final_winner": stack_identity_matches,
    }
    gates.update({
        "gate_positive_eras": gates["positive_environment_count"] >= required_positive,
        "gate_worst_era_improves": gates["worst_environment_lift_bps"] >= 0.0,
        "gate_dispersion": gates["environment_dispersion_bps"] <= config.max_environment_dispersion_bps,
        "gate_dispersion_improves_frozen_base": (
            gates["model_environment_net_dispersion_bps"]
            <= gates["frozen_base_environment_net_dispersion_bps"]
        ),
        "gate_worst_improves_frozen_base": (
            gates["model_worst_environment_net_bps"]
            >= gates["frozen_base_worst_environment_net_bps"]
        ),
        "gate_no_catastrophic_era": gates["catastrophic_environment_count"] == 0,
        "gate_latest_era": gates["latest_environment_lift_bps"] >= 0.0,
        "gate_no_side_failure": gates["side_failure_count"] == 0,
        "gate_transport_available": len(transport_values) > 0,
        "gate_transport_stack_identity": stack_identity_matches,
        "gate_transport_majority_positive": (
            len(transport_values) > 0
            and gates["positive_transport_cell_fraction"] >= config.min_positive_environment_fraction
        ),
        "gate_transport_no_catastrophe": (
            len(transport_values) > 0
            and gates["worst_transport_lift_bps"] > config.catastrophic_lift_bps
        ),
        "gate_transport_worst_positive": (
            len(transport_values) > 0 and gates["worst_transport_lift_bps"] > 0.0
        ),
    })
    gates["advances"] = bool(all(
        gates[name] for name in (
            "gate_positive_eras", "gate_worst_era_improves", "gate_dispersion",
            "gate_dispersion_improves_frozen_base", "gate_worst_improves_frozen_base",
            "gate_no_catastrophic_era", "gate_latest_era", "gate_no_side_failure",
            "gate_transport_available", "gate_transport_majority_positive",
            "gate_transport_stack_identity",
            "gate_transport_no_catastrophe",
            "gate_transport_worst_positive",
        )
    ))
    if gates["advances"]:
        terminal = "SHARED_RESIDUAL_EXPERT_TRANSPORTS"
    elif gates["gate_latest_era"] and gates["gate_no_side_failure"]:
        terminal = "SHARED_EXPERT_REQUIRES_REGIME_CONDITIONING"
    else:
        terminal = "SHARED_EXPERT_REMAINS_CROSS_ERA_UNSTABLE"
    gates["terminal_decision_code"] = terminal
    gates["regime_local_experts_decision"] = "REGIME_LOCAL_EXPERTS_NOT_JUSTIFIED"
    return gates


def _evaluate_calibration_arm(
    predecessor: StageIIIArmResult, *, arm: str, stack: StageIIIStack,
    config: StageIIIRunnerConfig, soft_regime_columns: Sequence[str],
) -> StageIIIArmResult:
    """Apply one prequential calibrator to the identical frozen raw OOF ledger."""
    oof = predecessor.oof_predictions.copy()
    oof["score_bps"] = oof["raw_shared_common_bps"].astype(np.float32)
    calibration_frame = oof.copy()
    calibration_frame["outcome_resolved_at"] = _utc(calibration_frame, config.label_available_timestamp_column)
    calibration_frame["__ts__"] = _utc(calibration_frame, config.decision_timestamp_column)
    calibrated, audit = prequential_shared_bps_calibration(
        calibration_frame, calibration_frame["raw_shared_common_bps"].to_numpy(float),
        calibration_frame[config.exact_net_column].to_numpy(float),
        mode=_calibration_library_mode(stack.calibration_mode),
        decision_timestamp_column="__ts__", resolution_column="outcome_resolved_at",
        side_column=config.side_column, soft_regime_columns=soft_regime_columns,
        anchor=config.calibration_anchor, min_global_rows=config.calibration_min_rows,
    )
    oof["score_bps"] = calibrated.astype(np.float32)
    selection = _selection_summary(oof, config=config)
    return StageIIIArmResult(
        arm=arm, round_name="E_calibration", stack=stack,
        selected_params_index=predecessor.selected_params_index,
        selected_params=predecessor.selected_params, selection_summary=selection,
        oof_predictions=oof, metrics=_metrics(oof, config=config),
        fold_audit=predecessor.fold_audit.copy(), calibration_audit=audit,
        shared_model_fit_count=0,
        model_feature_names=predecessor.model_feature_names,
        model_feature_contract_sha256=predecessor.model_feature_contract_sha256,
        source_feature_contract_sha256=predecessor.source_feature_contract_sha256,
    )


def _evaluate_pairwise_arm(
    frame: pd.DataFrame, *, predecessor: StageIIIArmResult, arm: str,
    pairwise_arm: str, config: StageIIIRunnerConfig,
    folds: Sequence[ExpandingEnvironmentFold], comparison_rows: np.ndarray,
    soft_regime_columns: Sequence[str], invariant_features: Sequence[str],
    regime_relative_features: Sequence[str], restricted_interaction_features: Sequence[str],
    validity_feature_groups: Mapping[str, Sequence[str]],
    pairwise_expert_fitter: PairwiseExpertFitter,
    target_preserving_pairwise_fitter: TargetPreservingPairwiseFitter,
    ordinal_target_fitter: OrdinalTargetFitter,
    quantile_target_fitter: QuantileTargetFitter,
) -> StageIIIArmResult:
    """Evaluate one Round-F shared pointwise/pairwise arm on frozen E lineage."""
    stack = predecessor.stack
    shared_columns = SharedResidualColumns(
        decision_timestamp=config.decision_timestamp_column,
        label_available_timestamp=config.label_available_timestamp_column,
        side=config.side_column, exact_net_bps=config.exact_net_column,
        base_expected_net_bps=config.base_expected_net_column,
    )
    use_relative = stack.conditioning_mode == "C3_regime_relative_features"
    use_interactions = stack.conditioning_mode in {
        "C2_restricted_interactions", "C3_regime_relative_features",
    }
    prepared, generated = prepare_shared_regime_residual_frame(
        frame, soft_regime_columns=soft_regime_columns,
        regime_relative_feature_names=(regime_relative_features if use_relative else ()),
        restricted_interaction_feature_names=(restricted_interaction_features if use_interactions else ()),
        columns=shared_columns, baseline_config=_stack_baseline_config(config, stack),
        baseline_mode=stack.baseline_mode, hard_regime_column=config.hard_regime_column,
    )
    feature_names = _conditioning_features(
        stack, invariant_features=invariant_features, soft_regime_columns=soft_regime_columns,
        relative_generated=[x for x in generated if x.startswith("__srre__")],
        interaction_generated=[x for x in generated if x.startswith("__srre_interaction__")],
        validity_feature_groups=validity_feature_groups,
    )
    if tuple(feature_names) != predecessor.model_feature_names:
        raise StageIIISequentialRunnerError("Round-F feature contract differs from its E predecessor")
    feature_complete = _finite_model_feature_mask(
        prepared, feature_names, context=f"Round-F {arm} frame"
    )
    prediction = np.full(len(prepared), np.nan, dtype=np.float32)
    fold_id = np.full(len(prepared), -1, dtype=np.int16)
    fold_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    columns = PairwiseSharedResidualColumns(
        decision_timestamp=config.decision_timestamp_column,
        label_available_timestamp=config.label_available_timestamp_column,
        side=config.side_column, candidate_id="__candidate_identity",
        exact_net_bps=config.exact_net_column,
        base_expected_net_bps=config.base_expected_net_column,
        cost_to_atr=config.cost_to_atr_column,
        base_map_prequential_flag="base_map_is_prequential",
        soft_regime_causal_flag="soft_regime_is_causal_prequential",
        cost_atr_causal_flag=config.cost_atr_causal_flag_column,
    )
    fit_count = 0
    for fold in folds:
        train_positions = fold.train_positions[feature_complete[fold.train_positions]]
        train = prepared.iloc[train_positions]
        train = train.loc[train["candidate_residual_bps"].notna()].copy()
        side = train[config.side_column].astype(str).str.lower()
        if len(train) < config.min_train_rows or any(
            int(side.eq(name).sum()) < config.min_rows_per_side for name in ("long", "short")
        ):
            continue
        validation_positions = fold.validation_positions[feature_complete[fold.validation_positions]]
        valid = prepared.iloc[validation_positions]
        if valid.empty:
            continue
        weights = mild_environment_weights(
            train, environment_column=config.environment_column,
            soft_regime_columns=soft_regime_columns,
            balance=("era" if stack.balance_mode == "era" else "natural"),
        )
        fit = _fit_round_f_model(
            train, stack=stack, pairwise_arm=pairwise_arm,
            feature_names=feature_names, soft_regime_columns=soft_regime_columns,
            fit_before_utc=fold.validation_start_utc, columns=columns, config=config,
            sample_weight=weights, pairwise_expert_fitter=pairwise_expert_fitter,
            target_preserving_pairwise_fitter=target_preserving_pairwise_fitter,
            ordinal_target_fitter=ordinal_target_fitter,
            quantile_target_fitter=quantile_target_fitter,
        )
        fit_count += 1
        correction = fit.predict_candidate_residual_bps(valid)
        prediction[validation_positions] = correction
        fold_id[validation_positions] = fold.fold_id
        support = fit.audit.pair_support
        preserved = getattr(fit.audit, "preserved_base_target", None)
        pair_records.append({
            "fold_id": fold.fold_id, "arm": arm, "pairwise_arm": pairwise_arm,
            **support.to_dict(),
            "preserved_base_target_arm": (
                None if preserved is None else preserved.base_target_arm
            ),
            "preserved_base_target_label_sha256": (
                None if preserved is None else preserved.base_target_label_sha256
            ),
            "preserved_base_training_prediction_sha256": (
                None if preserved is None else preserved.base_training_prediction_sha256
            ),
        })
        fold_records.append({
            "fold_id": fold.fold_id, "validation_environment": fold.validation_environment,
            "validation_start_utc": fold.validation_start_utc,
            "train_rows": len(train), "validation_rows": len(valid),
            "train_max_label_available_utc": fit.audit.max_label_available_utc,
            "shared_model_fit": True, "hard_routing": False,
            "train_candidate_identity_sha256": _identity_set_sha256(train["__candidate_identity"]),
            "validation_candidate_identity_sha256": _identity_set_sha256(valid["__candidate_identity"]),
        })
    comparison_mask = np.zeros(len(prepared), dtype=bool)
    comparison_mask[comparison_rows] = True
    keep = np.isfinite(prediction) & comparison_mask
    if not np.array_equal(np.flatnonzero(keep), comparison_rows):
        raise StageIIISequentialRunnerError(f"Round-F {arm} lost frozen OOF candidate identities")
    oof = prepared.loc[keep].copy()
    oof["predicted_candidate_residual_bps"] = prediction[keep]
    oof["fold_id"] = fold_id[keep]
    oof["raw_shared_common_bps"] = reconstruct_shared_regime_expected_net_bps(
        oof, prediction[keep], columns=shared_columns,
    )
    calibration_frame = oof.copy()
    calibration_frame["outcome_resolved_at"] = _utc(oof, config.label_available_timestamp_column)
    calibration_frame["__ts__"] = _utc(oof, config.decision_timestamp_column)
    calibrated, calibration_audit = prequential_shared_bps_calibration(
        calibration_frame, oof["raw_shared_common_bps"].to_numpy(float),
        oof[config.exact_net_column].to_numpy(float),
        mode=_calibration_library_mode(stack.calibration_mode),
        decision_timestamp_column="__ts__", resolution_column="outcome_resolved_at",
        side_column=config.side_column, soft_regime_columns=soft_regime_columns,
        anchor=config.calibration_anchor, min_global_rows=config.calibration_min_rows,
    )
    oof["score_bps"] = calibrated.astype(np.float32)
    pair_support = pd.DataFrame(pair_records)
    return StageIIIArmResult(
        arm=arm, round_name="F_pairwise_ranking", stack=stack,
        selected_params_index=predecessor.selected_params_index,
        selected_params=predecessor.selected_params,
        selection_summary=_selection_summary(oof, config=config),
        oof_predictions=oof, metrics=_metrics(oof, config=config),
        fold_audit=pd.DataFrame(fold_records), calibration_audit=calibration_audit,
        shared_model_fit_count=fit_count, model_feature_names=tuple(feature_names),
        model_feature_contract_sha256=stage_iii_feature_contract_sha256(feature_names),
        source_feature_contract_sha256=predecessor.source_feature_contract_sha256,
        predecessor_arm=predecessor.arm, pair_support=pair_support,
    )


def run_stage_iii_sequential_funnel(
    frame: pd.DataFrame, *, config: StageIIIRunnerConfig,
    input_lineage: StageIIIInputLineageContract,
    soft_regime_columns: Sequence[str], invariant_features: Sequence[str],
    regime_relative_features: Sequence[str], restricted_interaction_features: Sequence[str],
    validity_feature_groups: Mapping[str, Sequence[str]],
    expert_fitter: ExpertFitter = fit_shared_regime_residual_expert,
    pairwise_expert_fitter: PairwiseExpertFitter = fit_pairwise_shared_residual_expert,
    target_preserving_pairwise_fitter: TargetPreservingPairwiseFitter = fit_target_preserving_pairwise_adapter,
    ordinal_target_fitter: OrdinalTargetFitter = fit_ordinal_shared_robust_target,
    quantile_target_fitter: QuantileTargetFitter = fit_quantile_shared_robust_target,
) -> StageIIIFunnelResult:
    """Run A0-A3 through F0-F2 as a bounded sequential funnel.

    Arm selection uses OOF environment economics and a worst-era/dispersion
    penalty.  The returned winner remains a development winner; a later frozen
    OOS replay is still required before promotion.
    """
    _validate_input(frame, config)
    frame = frame.copy().reset_index(drop=True)
    frame["__candidate_identity"] = _candidate_identity(frame, config).to_numpy()
    if len(soft_regime_columns) < 2:
        raise StageIIISequentialRunnerError("at least two causal soft-regime probabilities are required")
    input_lineage.validate(
        frame, config=config, soft_regime_columns=soft_regime_columns,
        invariant_features=invariant_features,
        regime_relative_features=regime_relative_features,
        restricted_interaction_features=restricted_interaction_features,
        validity_feature_groups=validity_feature_groups,
    )
    folds = build_expanding_environment_folds(frame, config=config)
    comparison_rows = _frozen_comparison_rows(
        frame, folds=folds, config=config, soft_regime_columns=soft_regime_columns,
    )
    current = StageIIIStack()
    all_results: list[StageIIIArmResult] = []
    winners: dict[str, str] = {}

    def _run_round(
        round_name: str,
        arms: Mapping[str, Any],
        mutate: Callable[[StageIIIStack, Any], StageIIIStack],
    ) -> StageIIIArmResult:
        nonlocal current
        round_results = [
            _evaluate_arm(
                frame, arm=arm, round_name=round_name, stack=mutate(current, value),
                config=config, folds=folds, soft_regime_columns=soft_regime_columns,
                invariant_features=invariant_features,
                regime_relative_features=regime_relative_features,
                restricted_interaction_features=restricted_interaction_features,
                validity_feature_groups=validity_feature_groups, expert_fitter=expert_fitter,
                ordinal_target_fitter=ordinal_target_fitter,
                quantile_target_fitter=quantile_target_fitter,
                comparison_rows=comparison_rows,
                source_feature_contract_sha256=input_lineage.feature_contract_sha256,
            )
            for arm, value in arms.items()
        ]
        round_results = _align_round_results(round_results, config=config)
        best = _winner(round_results)
        current = best.stack
        winners[round_name] = best.arm
        all_results.extend(round_results)
        return best

    _run_round(
        "A_target_normalization", _BASELINE_ARMS,
        lambda stack, value: replace(stack, baseline_mode=value),
    )
    _run_round(
        "T_residual_target", _TARGET_ARMS,
        lambda stack, value: replace(
            stack,
            residual_target_arm=value[0],
            residual_target_mode=value[1],
            residual_target_clip_bps=float(value[2]),
        ),
    )
    for round_name, arms, mutate in (
        (
            "B_training_robustness", _TRAINING_ARMS,
            lambda stack, value: replace(
                stack, balance_mode=value[0], robust_hpo=value[1]
            ),
        ),
        (
            "C_conditioning", _CONDITIONING_ARMS,
            lambda stack, value: replace(stack, conditioning_mode=value),
        ),
        (
            "D_model_validity", _VALIDITY_ARMS,
            lambda stack, value: replace(stack, validity_mode=value),
        ),
    ):
        _run_round(round_name, arms, mutate)
    # Calibration is a matched replay of the frozen Round-D raw OOF ledger.
    calibration_results = [
        _evaluate_calibration_arm(
            next(result for result in reversed(all_results) if result.arm == winners["D_model_validity"]),
            arm=arm, stack=replace(current, calibration_mode=value), config=config,
            soft_regime_columns=soft_regime_columns,
        )
        for arm, value in _CALIBRATION_ARMS.items()
    ]
    calibration_results = _align_round_results(calibration_results, config=config)
    best_calibration = _winner(calibration_results)
    winners["E_calibration"] = best_calibration.arm
    current = best_calibration.stack
    all_results.extend(calibration_results)
    # Round F may compare at most the two strongest E calibrations.  Each F arm
    # selects its E predecessor on the same frozen OOF identities; then the
    # three F arms compete once.
    e_finalists = sorted(
        calibration_results,
        key=lambda result: float(result.selection_summary["selection_score"]),
        reverse=True,
    )[:2]
    pairwise_results: list[StageIIIArmResult] = []
    for arm, pairwise_arm in _PAIRWISE_ARMS.items():
        for predecessor in e_finalists:
            combination_id = f"{arm}@{predecessor.arm}"
            pairwise_results.append(_evaluate_pairwise_arm(
                frame, predecessor=predecessor, arm=arm, pairwise_arm=pairwise_arm,
                config=config, folds=folds, comparison_rows=comparison_rows,
                soft_regime_columns=soft_regime_columns,
                invariant_features=invariant_features,
                regime_relative_features=regime_relative_features,
                restricted_interaction_features=restricted_interaction_features,
                validity_feature_groups=validity_feature_groups,
                pairwise_expert_fitter=pairwise_expert_fitter,
                target_preserving_pairwise_fitter=target_preserving_pairwise_fitter,
                ordinal_target_fitter=ordinal_target_fitter,
                quantile_target_fitter=quantile_target_fitter,
            ))
            pairwise_results[-1] = replace(pairwise_results[-1], arm=combination_id)
    pairwise_results = _align_round_results(pairwise_results, config=config)
    best_pairwise = _winner(pairwise_results)
    winners["F_pairwise_ranking"] = best_pairwise.arm
    all_results.extend(pairwise_results)
    final = best_pairwise
    final_stack_identity = _final_stack_identity(final)
    if config.exact_gross_column is None:
        raise StageIIISequentialRunnerError("transport reporting requires exact_gross_column")
    transport_matrix = (
        run_train_test_transport_matrix(
            frame, stack=final.stack, config=config,
            final_arm=final.arm, final_predecessor_arm=final.predecessor_arm,
            final_stack_identity=final_stack_identity,
            model_feature_contract_sha256=final.model_feature_contract_sha256,
            soft_regime_columns=soft_regime_columns,
            invariant_features=invariant_features, regime_relative_features=regime_relative_features,
            restricted_interaction_features=restricted_interaction_features,
            validity_feature_groups=validity_feature_groups,
            pairwise_expert_fitter=pairwise_expert_fitter,
            target_preserving_pairwise_fitter=target_preserving_pairwise_fitter,
            ordinal_target_fitter=ordinal_target_fitter,
            quantile_target_fitter=quantile_target_fitter,
        )
        if config.run_transport_matrix else pd.DataFrame()
    )
    gates = stage_iii_advancement_gates(
        final.oof_predictions, transport_matrix, config=config,
        expected_transport_stack_identity=final_stack_identity,
    )
    summary = pd.DataFrame([
        {
            "round": result.round_name, "arm": result.arm,
            **asdict(result.stack), **dict(result.selection_summary),
            "selected_params_index": result.selected_params_index,
            "shared_model_fit_count": result.shared_model_fit_count,
            "model_feature_contract_sha256": result.model_feature_contract_sha256,
            "source_feature_contract_sha256": result.source_feature_contract_sha256,
        }
        for result in all_results
    ])
    return StageIIIFunnelResult(
        schema=SCHEMA, round_winners=winners, winner=final,
        arms=tuple(all_results), arm_summary=summary,
        transport_matrix=transport_matrix, advancement_gates=gates,
    )


__all__ = [
    "SCHEMA", "ExpandingEnvironmentFold", "StageIIIArmResult", "StageIIIFunnelResult",
    "StageIIIInputLineageContract", "StageIIIRunnerConfig", "StageIIISequentialRunnerError", "StageIIIStack",
    "build_expanding_environment_folds", "declared_sequential_arms",
    "run_stage_iii_sequential_funnel", "run_train_test_transport_matrix",
    "stage_iii_advancement_gates", "stage_iii_feature_contract_sha256",
]
