"""Offline pipeline for unsupervised regime primitive and operator selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.diagnostics import (
    FeatureSelectionResult,
    PreparedFrameContext,
    compute_feature_diagnostics,
    compute_quality_report,
    prepare_frame_context,
    select_primitive_features,
    select_representatives_by_spearman,
)
from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
    UNSUPERVISED_REGIME_LEARNING_DEFAULTS,
)
from extreme_price_movements.unsupervised_regime_learning.operators import (
    fit_transform_svd_knn_features,
    fit_transform_svd_knn_features_walk_forward,
    generate_autocorr_operator_features,
    generate_eigenvalue_summary_features,
    generate_pair_operator_features,
    generate_quantile_operator_features,
    make_mechanism_feature_groups,
    score_pair_candidates,
)
from extreme_price_movements.unsupervised_regime_learning.regime_models import (
    AdvancedRegimeLearningArtifact,
    AdvancedRegimeLearningConfig,
    fit_advanced_regime_learning,
    save_advanced_regime_learning_artifact,
)


@dataclass(frozen=True)
class OperatorSelectionResult:
    selected_operator_features: list[str]
    selected_pair_features: list[str]
    svd_knn_features: list[str]
    diagnostics: pd.DataFrame
    quality_report: pd.DataFrame
    pair_scores: pd.DataFrame
    spearman_threshold: float
    pair_spearman_threshold: float
    feature_frame: pd.DataFrame
    svd_state: dict[str, object]


@dataclass(frozen=True)
class UnsupervisedRegimeLearningResult:
    primitives: FeatureSelectionResult
    operators: OperatorSelectionResult
    final_feature_columns: list[str]
    regime_models: AdvancedRegimeLearningArtifact | None = None
    pipeline_steps: pd.DataFrame = field(default_factory=pd.DataFrame)


def _cfg_section(cfg: Mapping[str, object], key: str) -> dict[str, object]:
    value = cfg.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_int_or_none(cfg: Mapping[str, object], key: str) -> int | None:
    value = int(cfg.get(key, 0) or 0)
    return value if value > 0 else None


def _defaulted_config(cfg: Mapping[str, object] | None) -> dict[str, object]:
    out = dict(UNSUPERVISED_REGIME_LEARNING_DEFAULTS)
    if cfg:
        for key, value in cfg.items():
            if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
                nested = dict(out[key])
                nested.update(dict(value))
                out[key] = nested
            else:
                out[key] = value
    return out


def _advanced_regime_config(
    section: Mapping[str, object],
    *,
    timestamp_col: str,
    symbol_col: str,
) -> AdvancedRegimeLearningConfig:
    fields = set(AdvancedRegimeLearningConfig.__dataclass_fields__)
    defaults = AdvancedRegimeLearningConfig()
    values = {
        key: _coerce_advanced_regime_value(getattr(defaults, key), value)
        for key, value in dict(section).items()
        if key in fields
    }
    values["timestamp_col"] = timestamp_col
    values["symbol_col"] = symbol_col
    return AdvancedRegimeLearningConfig(**values)


def _coerce_advanced_regime_value(default: object, value: object) -> object:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except Exception:
            return int(default)
    if isinstance(default, float):
        try:
            return float(value)
        except Exception:
            return float(default)
    return value


def _pipeline_step_row(step: str, *, status: str = "completed", **metrics: object) -> dict[str, object]:
    row: dict[str, object] = {"step": str(step), "status": str(status)}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            row[str(key)] = value
        else:
            row[str(key)] = str(value)
    return row


def _pipeline_steps_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["step", "status"])
    out = pd.DataFrame([dict(row) for row in rows])
    order = ["step", "status"]
    remaining = [col for col in out.columns if col not in set(order)]
    return out.loc[:, order + remaining]


def select_unsupervised_primitives(
    frame: pd.DataFrame,
    *,
    cfg: Mapping[str, object] | None = None,
    feature_columns: Sequence[str] | None = None,
    context: PreparedFrameContext | None = None,
) -> FeatureSelectionResult:
    """Run quality filtering, diagnostics, and top-100 primitive selection."""

    active_cfg = _defaulted_config(cfg)
    quality_cfg = _cfg_section(active_cfg, "quality")
    primitive_cfg = _cfg_section(active_cfg, "primitive_selection")
    columns = list(feature_columns or active_cfg.get("primitive_feature_keys", []))
    return select_primitive_features(
        frame,
        columns,
        target_features=int(primitive_cfg.get("target_features", 100)),
        min_good_row_fraction=float(quality_cfg.get("min_good_row_fraction", 0.90)),
        warmup_rows=int(quality_cfg.get("warmup_rows", 0)),
        symbol_col=str(quality_cfg.get("symbol_col", "symbol")),
        timestamp_col=str(quality_cfg.get("timestamp_col", "timestamp")),
        initial_spearman_threshold=float(
            primitive_cfg.get("initial_spearman_threshold", 0.96)
        ),
        threshold_step=float(primitive_cfg.get("threshold_step", 0.005)),
        max_spearman_threshold=float(
            primitive_cfg.get("max_spearman_threshold", 0.999)
        ),
        block_hours=int(primitive_cfg.get("block_hours", 24 * 7)),
        min_block_rows=int(primitive_cfg.get("min_block_rows", 48)),
        autocorr_lag=int(primitive_cfg.get("autocorr_lag", 1)),
        treat_zero_as_low_quality=bool(
            quality_cfg.get("treat_zero_as_low_quality", True)
        ),
        spearman_max_corr_rows=_positive_int_or_none(
            primitive_cfg,
            "spearman_max_corr_rows",
        ),
        spearman_corr_time_bins=int(
            primitive_cfg.get("spearman_corr_time_bins", 24)
        ),
        spearman_max_candidates=_positive_int_or_none(
            primitive_cfg,
            "spearman_max_candidates",
        ),
        context=context,
    )


def build_operator_feature_frame(
    frame: pd.DataFrame,
    primitive_features: Sequence[str],
    *,
    cfg: Mapping[str, object] | None = None,
    context: PreparedFrameContext | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Generate quantile, autocorr, eigen, pair, SVD, and KNN candidates."""

    active_cfg = _defaulted_config(cfg)
    quality_cfg = _cfg_section(active_cfg, "quality")
    operator_cfg = _cfg_section(active_cfg, "operators")
    mechanisms = active_cfg.get("feature_mechanisms", {})
    mechanisms = mechanisms if isinstance(mechanisms, Mapping) else {}
    symbol_col = str(quality_cfg.get("symbol_col", "symbol"))
    timestamp_col = str(quality_cfg.get("timestamp_col", "timestamp"))
    min_periods = int(operator_cfg.get("min_periods", 0) or 0) or None
    quantile = generate_quantile_operator_features(
        frame,
        primitive_features,
        window=int(operator_cfg.get("quantile_window", 168)),
        min_periods=min_periods,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    autocorr = generate_autocorr_operator_features(
        frame,
        primitive_features,
        window=int(operator_cfg.get("autocorr_window", 168)),
        lag=int(operator_cfg.get("autocorr_lag", 1)),
        min_periods=min_periods,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    groups = make_mechanism_feature_groups(primitive_features, mechanisms)
    eigen = generate_eigenvalue_summary_features(
        frame,
        groups,
        window=int(operator_cfg.get("eigen_window", 168)),
        min_periods=min_periods,
        top_k=int(operator_cfg.get("eigen_top_k", 3)),
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    pair_scores = score_pair_candidates(
        frame,
        primitive_features,
        mechanisms=mechanisms,
        rolling_window=int(operator_cfg.get("pair_window", 168)),
        min_periods=min_periods,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        warmup_rows=int(quality_cfg.get("warmup_rows", 0)),
        min_good_row_fraction=float(quality_cfg.get("min_good_row_fraction", 0.90)),
        treat_zero_as_low_quality=bool(
            quality_cfg.get("treat_zero_as_low_quality", True)
        ),
        sparse_graph_enabled=bool(operator_cfg.get("sparse_graph_enabled", True)),
        sparse_graph_block_hours=int(
            operator_cfg.get(
                "sparse_graph_block_hours",
                operator_cfg.get("pair_window", 168),
            )
        ),
        sparse_graph_min_block_rows=int(
            operator_cfg.get("sparse_graph_min_block_rows", min_periods or 48)
        ),
        sparse_graph_alpha=float(operator_cfg.get("sparse_graph_alpha", 0.05)),
        sparse_graph_partial_corr_threshold=float(
            operator_cfg.get("sparse_graph_partial_corr_threshold", 1e-4)
        ),
        sparse_graph_max_iter=int(operator_cfg.get("sparse_graph_max_iter", 100)),
        sparse_graph_weight=float(operator_cfg.get("sparse_graph_weight", 0.50)),
        context=context,
    )
    operator_selection_cfg = _cfg_section(active_cfg, "operator_selection")
    pair_limit = int(operator_cfg.get("max_pair_candidates_for_generation", 0) or 0)
    if pair_limit <= 0:
        oversample = float(
            operator_cfg.get("pair_candidate_oversample_multiplier", 2.0)
        )
        target = int(operator_selection_cfg.get("target_features", 400))
        pair_limit = int(
            max(
                target,
                round(target * oversample),
            )
        )
    pair_scores_for_generation = pair_scores
    if pair_limit > 0 and len(pair_scores_for_generation) > pair_limit:
        pair_scores_for_generation = pair_scores_for_generation.head(pair_limit)
    pair_features = generate_pair_operator_features(
        frame,
        pair_scores_for_generation,
        window=int(operator_cfg.get("pair_window", 168)),
        min_periods=min_periods,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    svd_mode = str(operator_cfg.get("svd_mode", "walk_forward_prior_only"))
    if svd_mode == "walk_forward_prior_only":
        svd_knn, svd_state = fit_transform_svd_knn_features_walk_forward(
            frame,
            primitive_features,
            svd_components=operator_cfg.get("svd_components", [8, 16, 32]),
            knn_svd_components=int(operator_cfg.get("knn_svd_components", 16)),
            knn_neighbors=int(operator_cfg.get("knn_neighbors", 25)),
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
            block_hours=int(
                operator_cfg.get("svd_walk_forward_block_hours", 24 * 7)
            ),
            min_prior_rows=int(operator_cfg.get("svd_min_prior_rows", 500)),
            max_reference_rows=_positive_int_or_none(
                operator_cfg,
                "svd_max_reference_rows",
            ),
            knn_max_reference_rows=_positive_int_or_none(
                operator_cfg,
                "knn_max_reference_rows",
            ),
            sample_time_bins=int(operator_cfg.get("svd_sample_time_bins", 24)),
        )
    else:
        svd_knn, svd_state = fit_transform_svd_knn_features(
            frame,
            primitive_features,
            svd_components=operator_cfg.get("svd_components", [8, 16, 32]),
            knn_svd_components=int(operator_cfg.get("knn_svd_components", 16)),
            knn_neighbors=int(operator_cfg.get("knn_neighbors", 25)),
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
            max_reference_rows=_positive_int_or_none(
                operator_cfg,
                "svd_max_reference_rows",
            ),
            knn_max_reference_rows=_positive_int_or_none(
                operator_cfg,
                "knn_max_reference_rows",
            ),
            sample_time_bins=int(operator_cfg.get("svd_sample_time_bins", 24)),
        )
    derived = pd.concat(
        [quantile, autocorr, eigen, pair_features, svd_knn],
        axis=1,
    )
    return derived, pair_scores, svd_state


def select_operator_features(
    frame: pd.DataFrame,
    operator_frame: pd.DataFrame,
    pair_scores: pd.DataFrame,
    *,
    cfg: Mapping[str, object] | None = None,
    context: PreparedFrameContext | None = None,
) -> OperatorSelectionResult:
    """Select derived features per the operator rules.

    Quantile, autocorr, and eigenvalue features are pruned by
    ``data_quality * dynamics_score``. Pair covariance/correlation features are
    ordered by pair score and pruned separately. SVD/KNN features are retained.
    """

    active_cfg = _defaulted_config(cfg)
    quality_cfg = _cfg_section(active_cfg, "quality")
    operator_cfg = _cfg_section(active_cfg, "operator_selection")
    primitive_cfg = _cfg_section(active_cfg, "primitive_selection")
    symbol_col = str(quality_cfg.get("symbol_col", "symbol"))
    timestamp_col = str(quality_cfg.get("timestamp_col", "timestamp"))
    warmup_rows = int(quality_cfg.get("warmup_rows", 0))
    svd_cols = [
        col
        for col in operator_frame.columns
        if str(col).startswith("svd") or str(col).startswith("svd16_knn")
    ]
    pair_cols = [
        col
        for col in operator_frame.columns
        if str(col).startswith("cov_w") or str(col).startswith("corr_w")
    ]
    regular_cols = [
        col for col in operator_frame.columns if col not in set(svd_cols + pair_cols)
    ]
    quality = compute_quality_report(
        operator_frame,
        regular_cols,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        min_good_row_fraction=float(quality_cfg.get("min_good_row_fraction", 0.90)),
        treat_zero_as_low_quality=bool(
            quality_cfg.get("treat_zero_as_low_quality", True)
        ),
        context=context,
    )
    kept_regular = (
        quality.index[quality["keep"].to_numpy(dtype=bool, copy=False)]
        .astype(str)
        .tolist()
    )
    diagnostics = compute_feature_diagnostics(
        operator_frame,
        kept_regular,
        quality_report=quality,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        block_hours=int(primitive_cfg.get("block_hours", 24 * 7)),
        min_block_rows=int(primitive_cfg.get("min_block_rows", 48)),
        autocorr_lag=int(primitive_cfg.get("autocorr_lag", 1)),
        treat_zero_as_low_quality=bool(
            quality_cfg.get("treat_zero_as_low_quality", True)
        ),
        context=context,
    )
    selected_regular, threshold = select_representatives_by_spearman(
        operator_frame,
        diagnostics.index.tolist(),
        diagnostics["quality_dynamics_score"] if not diagnostics.empty else {},
        target_features=int(operator_cfg.get("target_features", 400)),
        initial_threshold=float(
            operator_cfg.get("initial_spearman_threshold", 0.95)
        ),
        threshold_step=float(operator_cfg.get("threshold_step", 0.005)),
        max_threshold=float(operator_cfg.get("max_spearman_threshold", 0.999)),
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        max_corr_rows=_positive_int_or_none(
            operator_cfg,
            "spearman_max_corr_rows",
        ),
        corr_time_bins=int(operator_cfg.get("spearman_corr_time_bins", 24)),
        max_candidates=_positive_int_or_none(
            operator_cfg,
            "max_regular_candidates_for_spearman",
        ),
        context=context,
    )
    pair_feature_scores: dict[str, float] = {}
    pair_window = int(_cfg_section(active_cfg, "operators").get("pair_window", 168))
    for row in pair_scores.itertuples(index=False):
        suffix = f"{str(row.feature_i)}__{str(row.feature_j)}"
        safe_suffix = (
            suffix.replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
        )
        pair_feature_scores[f"cov_w{pair_window}__{safe_suffix}"] = float(
            row.pair_score
        )
        pair_feature_scores[f"corr_w{pair_window}__{safe_suffix}"] = float(
            row.pair_score
        )
    selected_pair, pair_threshold = select_representatives_by_spearman(
        operator_frame,
        pair_cols,
        pair_feature_scores,
        target_features=int(operator_cfg.get("target_features", 400)),
        initial_threshold=float(
            operator_cfg.get("pair_initial_spearman_threshold", 0.96)
        ),
        threshold_step=float(operator_cfg.get("threshold_step", 0.005)),
        max_threshold=float(operator_cfg.get("max_spearman_threshold", 0.999)),
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        max_corr_rows=_positive_int_or_none(
            operator_cfg,
            "spearman_max_corr_rows",
        ),
        corr_time_bins=int(operator_cfg.get("spearman_corr_time_bins", 24)),
        max_candidates=_positive_int_or_none(
            operator_cfg,
            "max_pair_features_for_spearman",
        ),
        context=context,
    )
    selected_cols = selected_regular + selected_pair + svd_cols
    selected_frame = operator_frame.reindex(columns=selected_cols)
    return OperatorSelectionResult(
        selected_operator_features=selected_regular,
        selected_pair_features=selected_pair,
        svd_knn_features=svd_cols,
        diagnostics=diagnostics,
        quality_report=quality,
        pair_scores=pair_scores,
        spearman_threshold=threshold,
        pair_spearman_threshold=pair_threshold,
        feature_frame=selected_frame,
        svd_state={},
    )


def fit_unsupervised_regime_learning_features(
    frame: pd.DataFrame,
    *,
    cfg: Mapping[str, object] | None = None,
    feature_columns: Sequence[str] | None = None,
    regime_assessment_target: np.ndarray | pd.Series | None = None,
    regime_assessment_oof_pred: np.ndarray | pd.Series | None = None,
) -> UnsupervisedRegimeLearningResult:
    """Run the full primitive and operator feature-selection contract."""

    active_cfg = _defaulted_config(cfg)
    quality_cfg = _cfg_section(active_cfg, "quality")
    symbol_col = str(quality_cfg.get("symbol_col", "symbol"))
    timestamp_col = str(quality_cfg.get("timestamp_col", "timestamp"))
    context = prepare_frame_context(
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    )
    step_rows: list[dict[str, object]] = [
        _pipeline_step_row(
            "01_prepare_context",
            input_rows=int(len(frame)),
            requested_feature_count=int(
                len(feature_columns or active_cfg.get("primitive_feature_keys", []))
            ),
            symbol_col=symbol_col,
            timestamp_col=timestamp_col,
        )
    ]
    primitives = select_unsupervised_primitives(
        frame,
        cfg=active_cfg,
        feature_columns=feature_columns,
        context=context,
    )
    step_rows.append(
        _pipeline_step_row(
            "02_select_primitives",
            selected_feature_count=int(len(primitives.selected_features)),
            diagnostic_rows=int(len(primitives.diagnostics)),
            quality_report_rows=int(len(primitives.quality_report)),
            spearman_threshold=float(primitives.spearman_threshold),
        )
    )
    operator_frame, pair_scores, svd_state = build_operator_feature_frame(
        frame,
        primitives.selected_features,
        cfg=active_cfg,
        context=context,
    )
    step_rows.append(
        _pipeline_step_row(
            "03_generate_operator_features",
            operator_feature_count=int(operator_frame.shape[1]),
            pair_score_rows=int(len(pair_scores)),
            svd_mode=str(svd_state.get("mode", "")) if isinstance(svd_state, Mapping) else "",
            svd_generated_feature_count=int(len(svd_state.get("generated_features", [])))
            if isinstance(svd_state, Mapping)
            else 0,
        )
    )
    operators = select_operator_features(
        frame,
        operator_frame,
        pair_scores,
        cfg=active_cfg,
        context=context,
    )
    step_rows.append(
        _pipeline_step_row(
            "04_select_operator_features",
            selected_operator_feature_count=int(len(operators.selected_operator_features)),
            selected_pair_feature_count=int(len(operators.selected_pair_features)),
            svd_knn_feature_count=int(len(operators.svd_knn_features)),
            diagnostic_rows=int(len(operators.diagnostics)),
            quality_report_rows=int(len(operators.quality_report)),
            spearman_threshold=float(operators.spearman_threshold),
            pair_spearman_threshold=float(operators.pair_spearman_threshold),
        )
    )
    operators = OperatorSelectionResult(
        selected_operator_features=operators.selected_operator_features,
        selected_pair_features=operators.selected_pair_features,
        svd_knn_features=operators.svd_knn_features,
        diagnostics=operators.diagnostics,
        quality_report=operators.quality_report,
        pair_scores=operators.pair_scores,
        spearman_threshold=operators.spearman_threshold,
        pair_spearman_threshold=operators.pair_spearman_threshold,
        feature_frame=operators.feature_frame,
        svd_state=svd_state,
    )
    final_columns = (
        primitives.selected_features
        + operators.selected_operator_features
        + operators.selected_pair_features
        + operators.svd_knn_features
    )
    step_rows.append(
        _pipeline_step_row(
            "05_final_regime_learning_feature_set",
            final_feature_count=int(len(dict.fromkeys(final_columns))),
            primitive_feature_count=int(len(primitives.selected_features)),
            derived_feature_count=int(
                len(operators.selected_operator_features)
                + len(operators.selected_pair_features)
                + len(operators.svd_knn_features)
            ),
        )
    )
    regime_cfg = _cfg_section(active_cfg, "regime_models")
    regime_models = None
    if bool(regime_cfg.get("enabled", False)):
        model_frame = pd.concat([frame, operators.feature_frame], axis=1)
        model_features = list(dict.fromkeys(final_columns))
        regime_models = fit_advanced_regime_learning(
            model_frame,
            model_features,
            downstream_target=regime_assessment_target,
            base_oof_pred=regime_assessment_oof_pred,
            config=_advanced_regime_config(
                regime_cfg,
                timestamp_col=str(quality_cfg.get("timestamp_col", "timestamp")),
                symbol_col=str(quality_cfg.get("symbol_col", "symbol")),
            ),
        )
        output_dir = str(regime_cfg.get("artifact_output_dir", "") or "").strip()
        if output_dir:
            save_advanced_regime_learning_artifact(regime_models, output_dir)
        step_rows.append(
            _pipeline_step_row(
                "06_fit_advanced_regime_models",
                assessed_method_count=int(len(regime_models.regime_diagnostics)),
                model_regime_feature_count=int(regime_models.model_regime_features.shape[1]),
                pipeline_step_count=int(len(regime_models.pipeline_steps)),
                candidate_tier=str(
                    regime_models.diagnostics.get("model_regime_candidate_tier", "")
                    if isinstance(regime_models.diagnostics, Mapping)
                    else ""
                ),
                meaningful=bool(
                    regime_models.diagnostics.get("model_regime_package_meaningful", False)
                    if isinstance(regime_models.diagnostics, Mapping)
                    else False
                ),
            )
        )
    else:
        step_rows.append(
            _pipeline_step_row(
                "06_fit_advanced_regime_models",
                status="skipped",
                reason="regime_models.disabled",
            )
        )
    return UnsupervisedRegimeLearningResult(
        primitives=primitives,
        operators=operators,
        final_feature_columns=list(dict.fromkeys(final_columns)),
        regime_models=regime_models,
        pipeline_steps=_pipeline_steps_frame(step_rows),
    )
