"""Stage-I feature-selection contracts for the base/meta LGBM stack.

The project already has a performant selector in :mod:`lgbm_pipeline`.  This
module deliberately does not implement a second selector.  It makes the Stage-I
experiment explicit and reproducible: callers must name each active head and
the layer-specific feature universe is resolved from ``config.CFG`` rather than
from whatever columns happen to be present in a panel.

Selection is run once for each ``side x layer x head`` contract.  The helper
uses the existing sequence:

``coverage -> univariate -> ReliefF rescue -> Spearman redundancy -> repeated
grouped MDA -> smallest-within-one-SE feature count``.

It is intentionally an infrastructure layer only.  It never guesses which
heads are promoted and therefore cannot accidentally start broad experiments
for retired auxiliary heads.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


STAGE_I_SELECTOR_SCHEMA = "stage_i_grouped_stability_mda_v2"
_VALID_LAYERS = frozenset({"base", "meta"})
_VALID_SIDES = frozenset({"long", "short"})
STAGE_I_LABEL_AVAILABILITY_HOURS = 13.0
STAGE_I_MDA_MAX_TRAIN_ROWS_PER_MODEL = 20_000
STAGE_I_MDA_COHORT_COUNT = 3
# These direct outputs come from the same-side chronological base OOF model.
# They are required residual inputs, not optional raw-store context.
STAGE_I_META_BASE_OOF_HANDOFF_FEATURES: tuple[str, ...] = (
    "r3_p_adverse",
    "r3_p_weak",
    "r3_p_clear",
    "r3_opportunity_score",
    "prequential_base_expected_net_bps",
)
_STAGE_I_CV_LOCK = RLock()


@dataclass(frozen=True)
class StageIHeadContract:
    """One explicit selector scope.

    ``head`` is intentionally free-form because the active winner inventory is
    experiment-dependent.  ``layer`` and ``side`` are constrained so artifacts
    from different model contracts cannot be silently mixed.
    """

    layer: str
    side: str
    head: str

    def __post_init__(self) -> None:
        if self.layer not in _VALID_LAYERS:
            raise ValueError(f"layer must be one of {sorted(_VALID_LAYERS)}, got {self.layer!r}")
        if self.side not in _VALID_SIDES:
            raise ValueError(f"side must be one of {sorted(_VALID_SIDES)}, got {self.side!r}")
        if not str(self.head).strip():
            raise ValueError("head must be a non-empty active-head identifier")

    @property
    def artifact_key(self) -> str:
        return f"{self.layer}__{self.side}__{self.head}"


# Stage I deliberately has only the four currently authorised cells.  The
# inventory step may replace these names, but a new head cannot hitch a ride on
# the expensive 2024--26 selector without an explicit contract update.
STAGE_I_ACTIVE_CONTRACTS: tuple[StageIHeadContract, ...] = (
    StageIHeadContract("base", "long", "R3_economic_simplex_b25"),
    StageIHeadContract("base", "short", "R3_economic_simplex_b25"),
    StageIHeadContract("meta", "long", "shared_exact_net_residual"),
    StageIHeadContract("meta", "short", "shared_exact_net_residual"),
)


def stage_i_active_contracts() -> tuple[StageIHeadContract, ...]:
    """Return the small, inventory-confirmed Stage-I execution matrix."""
    return STAGE_I_ACTIVE_CONTRACTS


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(v) for v in values if str(v).strip()))


def _require_explicit_true(value: Any, *, label: str) -> None:
    """Accept only a literal boolean or numeric 0/1 provenance flag."""
    if isinstance(value, (bool, np.bool_)):
        valid = bool(value)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        valid = bool(np.isfinite(value) and float(value) == 1.0)
    else:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be an explicit true boolean/1 flag")


def stage_i_feature_key_groups(
    layer: str, *, side: str | None = None, head: str | None = None
) -> tuple[str, ...]:
    """Return only the config key collections authorised for a layer."""
    if layer == "base":
        if side not in _VALID_SIDES:
            raise ValueError("base Stage I feature resolution requires side=long or side=short")
        return ("base_shared_feature_keys", f"base_{side}_feature_keys")
    if layer == "meta":
        if str(head) != "shared_exact_net_residual":
            raise ValueError(f"no active Stage I meta feature pool for head={head!r}")
        return (
            "meta_shared_feature_keys",
            "meta_product_feature_keys",
            "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS",
            "STAGE_I_REQUIRED_SAME_SIDE_BASE_OOF_HANDOFF_FEATURE_KEYS",
        )
    raise ValueError(f"unknown layer: {layer!r}")


def resolve_stage_i_feature_universe(
    cfg: Mapping[str, Any],
    *,
    layer: str,
    side: str | None = None,
    head: str | None = None,
    available_columns: Sequence[str] | None = None,
) -> list[str]:
    """Resolve the layer's *declared* universe, optionally intersected with a panel.

    Values in ``*_feature_keys`` are names of feature-list entries in ``CFG``.
    A small number of legacy configs place raw feature names directly in those
    lists; keeping those makes this adapter backwards-compatible without
    allowing the opposite layer's keys to leak in.
    """
    declared: list[str] = []
    def _expand(name: str, seen: set[str]) -> list[str]:
        if name in seen:
            return []
        nested = cfg.get(name)
        if not isinstance(nested, (list, tuple, set)):
            return [name]
        seen.add(name)
        out: list[str] = []
        for value in nested:
            out.extend(_expand(str(value), seen))
        return out

    for group in stage_i_feature_key_groups(layer, side=side, head=head):
        for key_or_feature in cfg.get(group, []) or []:
            name = str(key_or_feature)
            declared.extend(_expand(name, set()))
    declared = _ordered_unique(declared)
    if available_columns is None:
        return declared
    available = {str(column) for column in available_columns}
    return [feature for feature in declared if feature in available]


def stage_i_mda_config(
    contract: StageIHeadContract,
    *,
    report_root: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the promoted Stage-I selector recipe.

    The lower-level implementation owns memory details: float32 matrices,
    stratified row caps, path-gated prediction recomputation, and adaptive
    repeat stopping.  This recipe asks it to score exactly the global top 10%
    with signed trade correctness/economics, group correlated fields at 0.95,
    and use shuffled phantom features as the selection threshold.
    """
    root = Path(report_root) / contract.artifact_key
    config: dict[str, Any] = {
        "stage_i_schema": STAGE_I_SELECTOR_SCHEMA,
        "stage_i_contract": asdict(contract),
        "stage_i_label_availability_hours": STAGE_I_LABEL_AVAILABILITY_HOURS,
        "stage_i_rolling_stability_required": True,
        "stage_i_prefix_optimal_count_confirmation": "smallest_within_one_se",
        "stage_i_phantom_in_fit_threshold": True,
        "dedicated_chronological_cohorts_enabled": True,
        "cohort_policy": "disjoint_era_chunks_internal_chronological_split",
        "cohort_count": STAGE_I_MDA_COHORT_COUNT,
        # This is an aggregate evidence target, not a per-fit allowance.  Each
        # of the three independently refitted cohort models remains capped at
        # 20K training rows below.
        "mda_train_rows": 60_000,
        "mda_eval_rows": 20_000,
        "max_train_rows_per_mda_model": STAGE_I_MDA_MAX_TRAIN_ROWS_PER_MODEL,
        "require_disjoint_cohort_training_rows": True,
        "require_disjoint_cohort_evaluation_rows": True,
        "require_aggregate_training_support_over_single_model_cap_when_available": True,
        "enabled": True,
        "objective": "signed_top10_trade_economics",
        "require_exact_economic_outcomes": True,
        "require_prediction_offset": bool(contract.layer == "meta"),
        "require_economic_prefix_score": True,
        "prediction_offset_semantics": (
            "frozen_base_expected_net_bps"
            if contract.layer == "meta"
            else "none"
        ),
        "r3_multiclass_required": bool(
            contract.layer == "base" and str(contract.head).startswith("R3_")
        ),
        "economic_outcome_units": "bps",
        "topk_fracs": [0.10],
        "topk_frac_weights": [1.0],
        "use_sample_weight": True,
        # Pre-screens remain the established univariate + ReliefF sequence.
        # Redundancy is intentionally after those screens, as required.
        "correlation_pruning_before_prescreen": False,
        "correlation_threshold": 0.95,
        "group_mda_enabled": True,
        "group_permutation_style": "joint_row_shuffle",
        "permutation_mode": "path_gated_lgbm",
        "permutation_style": "row_shuffle",
        "min_repeats": 3,
        "max_repeats": 12,
        "repeat_batch_size": 2,
        "early_stop_strong_keep": True,
        "early_stop_null_drop": True,
        "shadow_null_enabled": True,
        "shadow_null_quantile": 0.95,
        "shadow_n_repeats": 4,
        "shadow_max_features": 48,
        "decision_default_for_borderline": "review",
        "report_dir": str(root),
    }
    if overrides:
        config.update(dict(overrides))
    max_train_rows = int(
        config.get(
            "max_train_rows_per_mda_model",
            STAGE_I_MDA_MAX_TRAIN_ROWS_PER_MODEL,
        )
    )
    if not 1 <= max_train_rows <= STAGE_I_MDA_MAX_TRAIN_ROWS_PER_MODEL:
        raise ValueError(
            "Stage-I MDA max_train_rows_per_mda_model must be in "
            f"[1, {STAGE_I_MDA_MAX_TRAIN_ROWS_PER_MODEL}]"
        )
    config["max_train_rows_per_mda_model"] = max_train_rows
    if int(config.get("cohort_count", 0)) < 2:
        raise ValueError("Stage-I MDA requires at least two disjoint era cohorts")
    config["dedicated_chronological_cohorts_enabled"] = True
    config["require_disjoint_cohort_training_rows"] = True
    config["require_disjoint_cohort_evaluation_rows"] = True
    # A caller may tune repeat counts, but may not downgrade the residual
    # contract to raw-residual ranking.
    if contract.layer == "meta":
        config["require_prediction_offset"] = True
        config["prediction_offset_semantics"] = "frozen_base_expected_net_bps"
        # Preserve the direct R3 simplex/contrast through cheap screens and
        # MDA.  The common-bps map remains the residual offset, not a
        # substitute for these causal model outputs.
        for key in ("pre_mda_bypass_features", "force_include_features"):
            config[key] = list(
                dict.fromkeys(
                    [
                        *config.get(key, []),
                        *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
                    ]
                )
            )
    return config


def validate_stage_i_label_availability(
    timestamps: Sequence[Any],
    label_available_timestamps: Sequence[Any],
    *,
    min_gap_hours: float = STAGE_I_LABEL_AVAILABILITY_HOURS,
) -> dict[str, float]:
    """Validate the input needed for the strict chronological selector.

    The LGBM selector purges at least ``min_gap_hours`` between a train decision
    time and a validation decision time.  This check proves that the supplied
    label availability timestamps are aligned to rows and do not extend beyond
    that declared gate.  It refuses a missing/non-UTC-like substrate rather
    than silently downgrading Stage I to shuffled folds.
    """
    decision = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    available = pd.to_datetime(
        pd.Series(label_available_timestamps), utc=True, errors="coerce")
    if len(decision) != len(available) or len(decision) == 0:
        raise ValueError("Stage I requires aligned decision and label-available timestamps")
    if decision.isna().any() or available.isna().any():
        raise ValueError("Stage I label-availability gate rejects missing timestamps")
    lag_hours = (available - decision).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0
    if not np.isfinite(lag_hours).all() or np.any(lag_hours < 0.0):
        raise ValueError("label_available_ts must be finite and no earlier than decision_ts")
    max_lag = float(np.max(lag_hours))
    if not np.allclose(lag_hours, float(min_gap_hours), rtol=0.0, atol=1e-9):
        raise ValueError(
            "Stage I requires the exact signal-close-to-H12 availability contract: "
            f"expected every row at +{float(min_gap_hours):.3f}h, observed "
            f"min={float(np.min(lag_hours)):.3f}h max={max_lag:.3f}h"
        )
    return {
        "rows": float(len(decision)),
        "label_availability_max_hours": max_lag,
        "label_availability_gate_hours": float(min_gap_hours),
    }


@contextmanager
def _stage_i_strict_cv_context() -> Iterator[None]:
    """Force chronological 13h-purged CV and prohibit shuffled fallback.

    ``lgbm_pipeline`` stores these knobs as module constants for compatibility
    with legacy CLIs.  The context restores them even when fitting fails.  A
    process lock prevents two Stage-I cells from temporarily changing the
    shared settings concurrently.
    """
    pipeline = import_module("extreme_price_movements.lgbm_pipeline")
    names = (
        "LGBM_CV_MODE",
        "LGBM_PURGE_HOURS",
        "LGBM_FORWARD_BURNIN_STRICT",
        "LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK",
        "LGBM_SELECTION_SE_MULT",
    )
    with _STAGE_I_CV_LOCK:
        old = {name: getattr(pipeline, name) for name in names}
        pipeline.LGBM_CV_MODE = "forward_burnin"
        pipeline.LGBM_PURGE_HOURS = max(
            float(getattr(pipeline, "LGBM_PURGE_HOURS")),
            STAGE_I_LABEL_AVAILABILITY_HOURS,
        )
        pipeline.LGBM_FORWARD_BURNIN_STRICT = True
        pipeline.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK = False
        pipeline.LGBM_SELECTION_SE_MULT = 1.0
        try:
            yield
        finally:
            for name, value in old.items():
                setattr(pipeline, name, value)


def validate_stage_i_contract_input(
    frame: pd.DataFrame,
    *,
    contract: StageIHeadContract,
    cfg: Mapping[str, Any],
) -> list[str]:
    """Return eligible columns or fail before a selector fits an invalid scope."""
    declared = resolve_stage_i_feature_universe(
        cfg,
        layer=contract.layer,
        side=contract.side,
        head=contract.head,
    )
    features = resolve_stage_i_feature_universe(
        cfg,
        layer=contract.layer,
        side=contract.side,
        head=contract.head,
        available_columns=list(frame.columns),
    )
    if len(features) < 2:
        raise ValueError(
            f"{contract.artifact_key}: only {len(features)} declared {contract.layer} "
            "features are present; refusing to fall back to all panel columns"
        )
    if contract.layer == "meta":
        required = list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES)
        absent_from_config = [feature for feature in required if feature not in declared]
        absent_from_frame = [feature for feature in required if feature not in frame.columns]
        absent_from_contract = [feature for feature in required if feature not in features]
        if absent_from_config or absent_from_frame or absent_from_contract:
            raise ValueError(
                f"{contract.artifact_key}: residual selection requires the exact "
                "same-side base OOF handoff fields; "
                f"absent_from_config={absent_from_config}, "
                f"absent_from_frame={absent_from_frame}, "
                f"absent_from_contract={absent_from_contract}"
            )
    return features


def run_stage_i_head_selection(
    frame: pd.DataFrame,
    target: Any,
    *,
    contract: StageIHeadContract,
    cfg: Mapping[str, Any],
    report_root: str | Path,
    train_candidate: Callable[..., Mapping[str, Any] | None],
    candidate_kwargs: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Run one contract through the existing LGBM candidate selector.

    ``train_candidate`` is injected to avoid an import cycle with
    ``lgbm_pipeline`` and to make the orchestration testable.  The function is
    deliberately side-local: callers must pass a frame already filtered to
    ``contract.side`` and retain side provenance in their outer ledger.
    """
    if contract not in set(STAGE_I_ACTIVE_CONTRACTS):
        raise ValueError(
            f"{contract.artifact_key} is not one of the four inventory-confirmed Stage-I cells"
        )
    features = validate_stage_i_contract_input(frame, contract=contract, cfg=cfg)
    kwargs = dict(candidate_kwargs or {})
    timestamps = kwargs.get("timestamps")
    label_available = kwargs.pop("label_available_timestamps", None)
    if timestamps is None or label_available is None:
        raise ValueError(
            "Stage I requires timestamps and label_available_timestamps; shuffled fallback is forbidden"
        )
    availability_diag = validate_stage_i_label_availability(
        timestamps, label_available
    )
    exact_net_bps = kwargs.pop("exact_net_bps", None)
    exact_net_units = str(kwargs.pop("exact_net_units", "")).strip().lower()
    if exact_net_bps is None or exact_net_units != "bps":
        raise ValueError(
            "Stage I signed MDA requires exact_net_bps with exact_net_units='bps'; "
            "target/correctness fallback is forbidden"
        )
    exact_net = np.asarray(exact_net_bps, dtype=np.float32).reshape(-1)
    if len(exact_net) != len(frame) or not np.isfinite(exact_net).all():
        raise ValueError("exact_net_bps must be finite and aligned to every Stage I row")
    existing_returns = kwargs.get("returns")
    if existing_returns is not None and not np.allclose(
        np.asarray(existing_returns, dtype=np.float32).reshape(-1), exact_net, equal_nan=False
    ):
        raise ValueError("Stage I returns must equal the declared exact_net_bps vector")
    kwargs["returns"] = exact_net
    if contract.layer == "base" and str(contract.head).startswith("R3_"):
        target_values = pd.to_numeric(
            pd.Series(np.asarray(target).reshape(-1)), errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isin(target_values[np.isfinite(target_values)], [0.0, 1.0, 2.0]).all():
            raise ValueError(
                "R3 economic-simplex Stage I requires integer class labels 0/1/2"
            )
        requested_mode = str(kwargs.get("mode", "multiclass3"))
        if requested_mode != "multiclass3":
            raise ValueError("R3 economic-simplex Stage I requires mode='multiclass3'")
        metric_target = kwargs.pop("r3_metric_target", kwargs.get("hard_labels"))
        if metric_target is None:
            raise ValueError(
                "R3 economic-simplex Stage I requires r3_metric_target: a predeclared ordinal/economic metric encoding"
            )
        metric_arr = np.asarray(metric_target, dtype=np.float32).reshape(-1)
        if len(metric_arr) != len(frame) or not np.isfinite(metric_arr).all():
            raise ValueError("r3_metric_target must be finite and aligned to every R3 row")
        kwargs["mode"] = "multiclass3"
        kwargs["hard_labels"] = metric_arr
    meta_provenance: dict[str, Any] | None = None
    frozen_base_expected_net: np.ndarray | None = None
    if contract.layer == "meta":
        raw_provenance = kwargs.pop("base_oof_provenance", None)
        if not isinstance(raw_provenance, Mapping):
            raise ValueError("Stage I meta selection requires same-side base OOF provenance")
        source_side = str(raw_provenance.get("side", "")).strip().lower()
        try:
            _require_explicit_true(
                raw_provenance.get("strict_oof", raw_provenance.get("is_oof", False)),
                label="base_oof_provenance.strict_oof",
            )
            strict_oof = True
        except ValueError:
            strict_oof = False
        if source_side != contract.side or not strict_oof:
            raise ValueError(
                "Stage I meta selection requires strict same-side OOF base predictions"
            )
        meta_provenance = dict(raw_provenance)
        raw_offset = kwargs.pop("frozen_base_expected_net_bps", None)
        offset_units = str(
            kwargs.pop("frozen_base_expected_net_units", "")
        ).strip().lower()
        if raw_offset is None or offset_units != "bps":
            raise ValueError(
                "Stage I residual meta selection requires frozen_base_expected_net_bps "
                "with frozen_base_expected_net_units='bps'"
            )
        frozen_base_expected_net = np.asarray(raw_offset, dtype=np.float32).reshape(-1)
        if (
            len(frozen_base_expected_net) != len(frame)
            or not np.isfinite(frozen_base_expected_net).all()
        ):
            raise ValueError(
                "frozen_base_expected_net_bps must be finite and aligned to every meta row"
            )
        if "prediction_offset" in kwargs:
            supplied = np.asarray(kwargs.pop("prediction_offset"), dtype=np.float32).reshape(-1)
            if not np.array_equal(supplied, frozen_base_expected_net):
                raise ValueError(
                    "meta prediction_offset must equal frozen_base_expected_net_bps exactly"
                )
        # This is deliberately passed separately from the residual target. The
        # lower selector carries it through every race/fold subsample and uses
        # `offset + predicted_residual` for both baseline and permutations.
        kwargs["prediction_offset"] = frozen_base_expected_net
    user_cfg = dict(kwargs.pop("cfg", {}) or {})
    if kwargs.get("preset_feature_names"):
        raise ValueError(
            "Stage I forbids preset_feature_names: every active cell must run "
            "coverage, univariate, Relief, Spearman and grouped MDA from its "
            "declared layer universe"
        )
    user_cfg["mda_config"] = stage_i_mda_config(
        contract,
        report_root=report_root,
        overrides=user_cfg.get("mda_config") if isinstance(user_cfg.get("mda_config"), Mapping) else None,
    )
    with _stage_i_strict_cv_context():
        result = train_candidate(
            frame.loc[:, features],
            target,
            cfg=user_cfg,
            **kwargs,
        )
    if result is None:
        return None
    output = dict(result)
    selected = list(map(str, output.get("selected_feature_names", []) or []))
    if contract.layer == "meta":
        missing_handoff = [
            feature for feature in STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
            if feature not in selected
        ]
        if missing_handoff:
            raise ValueError(
                f"{contract.artifact_key}: selected feature contract dropped required "
                f"same-side base OOF handoff fields: {missing_handoff}"
            )
        # LGBM sees columns in this exact ordered list during selection/HPO;
        # persist that order for the strict-OOS refit rather than letting an
        # unordered importance result define a different later contract.
        selected_set = set(selected)
        output["selected_feature_names"] = [
            feature for feature in features if feature in selected_set
        ]
    output["stage_i_contract"] = asdict(contract)
    output["stage_i_schema"] = STAGE_I_SELECTOR_SCHEMA
    output["stage_i_input_feature_count"] = int(len(features))
    output["stage_i_input_features"] = features
    output["stage_i_label_availability"] = availability_diag
    output["stage_i_exact_net_units"] = "bps"
    output["stage_i_exact_net_rows"] = int(len(exact_net))
    if meta_provenance is not None:
        output["stage_i_base_oof_provenance"] = meta_provenance
        output["stage_i_meta_target"] = (
            "exact_net_bps_minus_frozen_causal_base_expected_net_bps"
        )
        output["stage_i_mda_ranking_score"] = (
            "frozen_base_expected_net_bps_plus_predicted_residual_bps"
        )
        output["stage_i_frozen_base_expected_net_units"] = "bps"
        output["stage_i_frozen_base_expected_net_rows"] = int(
            len(frozen_base_expected_net) if frozen_base_expected_net is not None else 0
        )
        output["stage_i_required_same_side_base_oof_handoff_features"] = list(
            STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
        )
        output["stage_i_selected_feature_contract"] = list(
            output["selected_feature_names"]
        )
    else:
        output["stage_i_mda_ranking_score"] = "raw_base_prediction"
    output["stage_i_prefix_confirmation"] = stage_i_prefix_confirmation(output)
    return output


def stage_i_prefix_confirmation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the existing rolling one-SE prefix choice as an audit record.

    The lower selector already evaluates successive pruned prefixes on its
    chronological stability folds and chooses the smallest prefix within the
    configured one-SE band.  This function preserves the evidence in the
    Stage-I manifest; it does not treat a feature-count heuristic as an HPO
    result when a runner did not expose its pruning history.
    """
    history = list(result.get("pruning_history", []) or [])
    if not history:
        return {
            "available": False,
            "reason": "pruning_history_missing",
            "policy": "smallest_within_one_se",
        }
    score_key = "mda_economic_baseline_score_mean"
    se_key = "mda_economic_baseline_score_se"
    rows: list[Mapping[str, Any]] = []
    for row in history:
        if not isinstance(row, Mapping):
            raise ValueError("Stage I pruning_history must contain mapping records")
        score = float(row.get(score_key, np.nan))
        score_se = float(row.get(se_key, np.nan))
        if not np.isfinite(score) or not np.isfinite(score_se) or score_se < 0.0:
            raise ValueError(
                "Stage I prefix confirmation requires fold-level signed-economic "
                "MDA score mean and SE; legacy J_final/J_se is not valid"
            )
        rows.append(row)
    # Prefix choice is based on the same signed common-bps score as MDA, not
    # raw residual/model fit quality. This matters for residual heads where a
    # strong residual fit can still worsen the reconstructed live ranking.
    best = max(rows, key=lambda row: float(row[score_key]))
    best_score = float(best[score_key])
    one_se = max(float(best[se_key]), 0.0)
    # This is the literal predeclared one-standard-error rule.  Do not use the
    # legacy fractional-SE convenience setting in a Stage-I report.
    floor = best_score - one_se
    eligible = [row for row in rows if float(row[score_key]) >= floor]
    chosen = min(
        eligible,
        key=lambda row: int(row.get("n_features_end", row.get("n_features", 10**9))),
    )
    return {
        "available": True,
        "policy": "smallest_within_one_se",
        "rounds_scored": int(len(rows)),
        "score_key": score_key,
        "se_key": se_key,
        "best_score": best_score,
        "one_se": one_se,
        "score_floor": floor,
        "confirmed_prefix_feature_count": int(
            chosen.get("n_features_end", chosen.get("n_features", 0))
        ),
    }


def run_stage_i_active_matrix(
    jobs: Mapping[StageIHeadContract, Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
    report_root: str | Path,
    train_candidate: Callable[..., Mapping[str, Any] | None],
) -> dict[str, Mapping[str, Any] | None]:
    """Execute precisely the four confirmed side × layer × head cells.

    Each job must provide ``frame``, ``target`` and optionally
    ``candidate_kwargs``.  An incomplete matrix is rejected: silently
    comparing a long-only or base-only subset would not answer Stage I.
    """
    expected = set(STAGE_I_ACTIVE_CONTRACTS)
    supplied = set(jobs)
    if supplied != expected:
        missing = sorted(c.artifact_key for c in expected.difference(supplied))
        unexpected = sorted(c.artifact_key for c in supplied.difference(expected))
        raise ValueError(
            "Stage I active matrix must contain exactly four confirmed cells; "
            f"missing={missing}, unexpected={unexpected}"
        )
    outputs: dict[str, Mapping[str, Any] | None] = {}
    for contract in STAGE_I_ACTIVE_CONTRACTS:
        job = dict(jobs[contract])
        if "frame" not in job or "target" not in job:
            raise ValueError(f"{contract.artifact_key}: job requires frame and target")
        frame = job.pop("frame")
        target = job.pop("target")
        candidate_kwargs = job.pop("candidate_kwargs", None)
        if job:
            raise ValueError(f"{contract.artifact_key}: unsupported job fields {sorted(job)}")
        outputs[contract.artifact_key] = run_stage_i_head_selection(
            frame,
            target,
            contract=contract,
            cfg=cfg,
            report_root=report_root,
            train_candidate=train_candidate,
            candidate_kwargs=candidate_kwargs,
        )
    return outputs
