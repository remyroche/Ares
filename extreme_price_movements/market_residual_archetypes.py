"""Leakage-safe broad-market residual states fitted after local archetypes.

The market layer is deliberately separate from side x archetype residual
recognition. It identifies synchronized favorable/adverse market states from
market-wide pre-entry features only and never consumes asset residuals at OOS
transform time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:  # pragma: no cover
    from sklearn.metrics import average_precision_score
    from sklearn.neural_network import MLPClassifier
except Exception:  # pragma: no cover
    average_precision_score = None
    MLPClassifier = None

from .features_gmm_ae import fit_ae_gmm_state, transform_ae_gmm_features
from .meta_residual_archetypes import (
    OUTCOME_COLUMNS,
    REFERENCE_DERIVED_COLUMNS,
    ResidualArchetypeConfig,
    _NativeMulticlassRecognizer,
    _prepare_numeric_matrix,
    _select_recognizer_features,
    _time_spread_indices,
    add_reference_surprise_targets,
)

MARKET_STATE_NAMES = ("neutral", "synchronized_adverse", "synchronized_favorable")
MARKET_PREFIX = "meta_resid_market_"
MARKET_ADVERSE_PREFIX = "meta_resid_market_arch_"
MARKET_ADVERSE_PROBABILITY_MODELS = ("aegmm", "mlp", "lgbm", "ensemble")
MARKET_ADVERSE_STATE_NAMES = (
    "neutral",
    "bad_mae_path",
    "slow_timeout",
    "dirty_overconfident",
    "negative_ev",
    "systemic_shock",
    "high_variance_uncertain",
)


def market_residual_feature_names() -> list[str]:
    return [f"{MARKET_PREFIX}prob__{name}" for name in MARKET_STATE_NAMES] + [
        f"{MARKET_PREFIX}expected_signed_surprise",
        f"{MARKET_PREFIX}expected_ev",
        f"{MARKET_PREFIX}entropy",
        f"{MARKET_PREFIX}confidence",
        f"{MARKET_PREFIX}support_log1p",
    ]


def market_archetype_adverse_feature_names() -> list[str]:
    """Stable continuous outputs from the per-archetype adverse-state layer."""

    return [
        *[
            f"{MARKET_ADVERSE_PREFIX}prob_adverse__{name}"
            for name in MARKET_ADVERSE_PROBABILITY_MODELS
        ],
        *[
            f"{MARKET_ADVERSE_PREFIX}state_prob__{name}"
            for name in MARKET_ADVERSE_STATE_NAMES
        ],
        f"{MARKET_ADVERSE_PREFIX}probability_disagreement",
        f"{MARKET_ADVERSE_PREFIX}binary_entropy",
        f"{MARKET_ADVERSE_PREFIX}expected_signed_surprise",
        f"{MARKET_ADVERSE_PREFIX}expected_ev",
        f"{MARKET_ADVERSE_PREFIX}expected_severity",
        f"{MARKET_ADVERSE_PREFIX}expected_bad_mae",
        f"{MARKET_ADVERSE_PREFIX}expected_timeout",
        f"{MARKET_ADVERSE_PREFIX}expected_dirty_positive",
        f"{MARKET_ADVERSE_PREFIX}expected_stop_or_adverse",
        f"{MARKET_ADVERSE_PREFIX}support_log1p",
        f"{MARKET_ADVERSE_PREFIX}episode_support_log1p",
        f"{MARKET_ADVERSE_PREFIX}local_model_available",
    ]


def market_wide_feature_columns(
    frame: pd.DataFrame, candidates: Iterable[str]
) -> list[str]:
    prefixes = (
        "mkt_",
        "market_",
        "cross_asset_",
        "return_dispersion_",
        "pct_assets_",
        "breadth_",
    )
    excluded_tokens = ("asset_minus", "symbol", "resid_asset")
    return [
        str(name)
        for name in candidates
        if str(name) in frame.columns
        and str(name) not in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
        and str(name).startswith(prefixes)
        and not any(token in str(name) for token in excluded_tokens)
        and pd.api.types.is_numeric_dtype(frame[str(name)])
    ]


@dataclass(frozen=True)
class MarketResidualConfig:
    score_col: str = "score_regime_calibrated"
    timestamp_col: str = "__ts__"
    ev_col: str = "ev_after_1pct"
    min_rows: int = 1_000
    max_fit_rows: int = 60_000
    max_features: int = 64
    max_cross_sectional_sources: int = 24
    tail_quantile: float = 0.80
    random_state: int = 20260712


@dataclass
class MarketResidualStateRecognizer:
    config: MarketResidualConfig = field(default_factory=MarketResidualConfig)
    candidate_features: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    feature_relevance: list[dict[str, Any]] = field(default_factory=list)
    medians: np.ndarray | None = None
    clip_low: np.ndarray | None = None
    clip_high: np.ndarray | None = None
    recognizer: Any = None
    state_priors: dict[int, dict[str, float]] = field(default_factory=dict)
    support_rows: int = 0
    train_start_: str | None = None
    train_end_: str | None = None
    explicit_market_features: list[str] = field(default_factory=list)
    cross_sectional_source_features: list[str] = field(default_factory=list)
    label_diagnostics: dict[str, Any] = field(default_factory=dict)

    def _configure_observable_features(self, frame: pd.DataFrame) -> None:
        explicit = market_wide_feature_columns(frame, self.candidate_features)
        self.explicit_market_features = [
            name
            for name in explicit
            if float(pd.to_numeric(frame[name], errors="coerce").notna().mean()) >= 0.05
            and float(pd.to_numeric(frame[name], errors="coerce").std()) > 1e-8
        ]
        tokens = (
            "oi_",
            "funding",
            "ret",
            "rv_",
            "vol",
            "shock",
            "liquidity",
            "spread",
            "pressure",
            "leverage",
            "dislocation",
            "breadth",
            "dispersion",
        )
        excluded = (
            OUTCOME_COLUMNS
            | REFERENCE_DERIVED_COLUMNS
            | {
                "score",
                "score_regime_calibrated",
                "score_meta_uncalibrated",
                "hit_probability",
            }
        )
        ranked: list[tuple[float, str]] = []
        for raw_name in self.candidate_features:
            name = str(raw_name)
            if (
                name in excluded
                or name in self.explicit_market_features
                or name not in frame.columns
                or not any(token in name.lower() for token in tokens)
                or not pd.api.types.is_numeric_dtype(frame[name])
            ):
                continue
            values = pd.to_numeric(frame[name], errors="coerce")
            coverage = float(values.notna().mean())
            variance = float(values.var())
            if coverage < 0.20 or not np.isfinite(variance) or variance <= 1e-8:
                continue
            ranked.append((coverage * math.log1p(variance), name))
        ranked.sort(reverse=True)
        self.cross_sectional_source_features = [
            name for _, name in ranked[: int(self.config.max_cross_sectional_sources)]
        ]

    def _collapse(self, frame: pd.DataFrame, *, training: bool) -> pd.DataFrame:
        ts = pd.to_datetime(frame[self.config.timestamp_col], utc=True, errors="coerce")
        work = frame.assign(_market_ts=ts)
        work.attrs = {}
        aggregations: dict[str, str] = {
            name: "median" for name in self.explicit_market_features
        }
        if training:
            aggregations.update(
                {
                    "market_signed_surprise": "mean",
                    self.config.ev_col: "mean",
                }
            )
        grouped = work.groupby("_market_ts", sort=True, observed=True)
        if aggregations:
            collapsed = grouped.agg(aggregations).reset_index()
        else:
            collapsed = pd.DataFrame(
                {"_market_ts": np.sort(work["_market_ts"].dropna().unique())}
            )
        if self.cross_sectional_source_features:
            source = grouped[self.cross_sectional_source_features]
            median = source.median().add_prefix("xs_median__")
            iqr = (source.quantile(0.75) - source.quantile(0.25)).add_prefix("xs_iqr__")
            collapsed = collapsed.merge(
                pd.concat([median, iqr], axis=1).reset_index(),
                on="_market_ts",
                how="left",
                validate="one_to_one",
                sort=False,
            )
        return collapsed

    def fit(self, train: pd.DataFrame) -> "MarketResidualStateRecognizer":
        if lgb is None:
            raise RuntimeError("LightGBM is required for market residual states")
        prepared = add_reference_surprise_targets(
            train,
            ResidualArchetypeConfig(
                score_col=self.config.score_col,
                rank_scope="global",
                allow_side_fallback=False,
            ),
        )
        prepared = prepared.loc[prepared["reference_rank_pct"].ge(0.80)]
        self._configure_observable_features(prepared)
        collapsed = self._collapse(prepared, training=True)
        if len(collapsed) < int(self.config.min_rows):
            raise ValueError(
                f"Insufficient market-state timestamps: {len(collapsed)} < {self.config.min_rows}"
            )
        timestamp = pd.to_datetime(collapsed["_market_ts"], utc=True, errors="coerce")
        self.train_start_ = str(timestamp.min())
        self.train_end_ = str(timestamp.max())
        surprise = pd.to_numeric(
            collapsed["market_signed_surprise"], errors="coerce"
        ).fillna(0.0)
        day = timestamp.dt.floor("D")
        daily = surprise.groupby(day, sort=False).transform("mean")
        daily_unique = pd.DataFrame({"day": day, "daily": daily}).drop_duplicates("day")
        daily_unique = daily_unique.sort_values("day", kind="stable")
        daily_unique["prior"] = daily_unique["daily"].shift(1)
        prior = daily_unique.set_index("day")["prior"].reindex(day).to_numpy()
        lower = float(daily.quantile(1.0 - float(self.config.tail_quantile)))
        upper = float(daily.quantile(float(self.config.tail_quantile)))
        center = float(daily.median())
        prior_centered = prior - center
        labels = np.zeros(len(collapsed), dtype=np.int32)
        labels[(daily.to_numpy() <= lower) & (prior_centered <= 0.0)] = 1
        labels[(daily.to_numpy() >= upper) & (prior_centered >= 0.0)] = 2
        minimum_state_rows = max(25, int(round(0.005 * len(collapsed))))
        persistent_counts = np.bincount(labels, minlength=3)
        fallback_states: list[str] = []
        if int(persistent_counts[1]) < minimum_state_rows:
            labels[daily.to_numpy() <= lower] = 1
            fallback_states.append("synchronized_adverse")
        if int(persistent_counts[2]) < minimum_state_rows:
            labels[daily.to_numpy() >= upper] = 2
            fallback_states.append("synchronized_favorable")
        label_counts = np.bincount(labels, minlength=3)
        self.label_diagnostics = {
            "daily_surprise_center": center,
            "daily_surprise_lower": lower,
            "daily_surprise_upper": upper,
            "persistent_label_counts": persistent_counts.astype(int).tolist(),
            "final_label_counts": label_counts.astype(int).tolist(),
            "tail_fallback_states": fallback_states,
            "label_basis": "train_relative_daily_surprise_with_prior_day_persistence",
        }
        collapsed["large_negative_surprise_label"] = (labels == 1).astype(np.int8)
        collapsed["large_positive_surprise_label"] = (labels == 2).astype(np.int8)
        collapsed["negative_autocorr_label"] = (labels == 1).astype(np.int8)
        collapsed["positive_autocorr_label"] = (labels == 2).astype(np.int8)

        screen_cfg = ResidualArchetypeConfig(
            max_recognizer_features=int(self.config.max_features),
            mutual_info_rows=min(len(collapsed), int(self.config.max_fit_rows)),
            feature_screen_mode="binned_mi_lgbm",
            random_state=int(self.config.random_state),
        )
        candidates = list(self.explicit_market_features) + [
            name
            for source in self.cross_sectional_source_features
            for name in (f"xs_median__{source}", f"xs_iqr__{source}")
        ]
        self.feature_columns, self.feature_relevance = _select_recognizer_features(
            collapsed, labels, candidates, screen_cfg, int(self.config.random_state)
        )
        if len(self.feature_columns) < 2:
            fallback_features: list[str] = []
            for name in candidates:
                if name not in collapsed.columns:
                    continue
                values = pd.to_numeric(collapsed[name], errors="coerce")
                if float(values.notna().mean()) < 0.05:
                    continue
                if not np.isfinite(float(values.var())) or float(values.var()) <= 1e-8:
                    continue
                fallback_features.append(name)
            self.feature_columns = fallback_features[: int(self.config.max_features)]
            self.label_diagnostics["feature_screen_fallback"] = True
        else:
            self.label_diagnostics["feature_screen_fallback"] = False
        if len(self.feature_columns) < 2 or np.unique(labels).size < 2:
            raise ValueError("Market residual labels/features are degenerate")
        x, self.medians, self.clip_low, self.clip_high = _prepare_numeric_matrix(
            collapsed, self.feature_columns
        )
        fit_idx = _time_spread_indices(len(collapsed), int(self.config.max_fit_rows))
        fit_labels = labels[fit_idx]
        fit_counts = np.bincount(fit_labels, minlength=3).astype(np.float64)
        class_weight = np.zeros(3, dtype=np.float32)
        present = fit_counts > 0
        class_weight[present] = (
            len(fit_labels) / (max(int(present.sum()), 1) * fit_counts[present])
        ).astype(np.float32)
        row_weight = class_weight[fit_labels]
        dataset = lgb.Dataset(
            x[fit_idx],
            label=fit_labels,
            weight=row_weight,
            free_raw_data=True,
        )
        booster = lgb.train(
            {
                "objective": "multiclass",
                "num_class": 3,
                "learning_rate": 0.04,
                "num_leaves": 7,
                "max_depth": 3,
                "min_data_in_leaf": 40,
                "feature_fraction": 0.80,
                "lambda_l1": 0.10,
                "lambda_l2": 4.0,
                "seed": int(self.config.random_state),
                "num_threads": 2,
                "verbosity": -1,
                "force_col_wise": True,
            },
            dataset,
            num_boost_round=120,
        )
        self.recognizer = _NativeMulticlassRecognizer(
            booster=booster, classes_=np.arange(3, dtype=np.int32)
        )
        ev = pd.to_numeric(collapsed[self.config.ev_col], errors="coerce").fillna(0.0)
        self.state_priors = {
            state: {
                "signed_surprise": float(surprise.loc[labels == state].mean()),
                "ev": float(ev.loc[labels == state].mean()),
            }
            for state in np.unique(labels)
        }
        self.support_rows = int(len(collapsed))
        return self

    def transform_oos(self, oos: pd.DataFrame) -> pd.DataFrame:
        forbidden = sorted(
            (OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS).intersection(oos.columns)
        )
        if forbidden:
            raise ValueError(
                f"OOS market residual transform received outcomes: {forbidden[:12]}"
            )
        if self.recognizer is None:
            raise RuntimeError("Market residual recognizer is not fitted")
        collapsed = self._collapse(oos, training=False)
        x, _, _, _ = _prepare_numeric_matrix(
            collapsed,
            self.feature_columns,
            medians=self.medians,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
        )
        probability = np.asarray(self.recognizer.predict_proba(x), dtype=np.float32)
        expected_surprise = np.zeros(len(collapsed), dtype=np.float32)
        expected_ev = np.zeros(len(collapsed), dtype=np.float32)
        for state in range(probability.shape[1]):
            prior = self.state_priors.get(state, {})
            expected_surprise += probability[:, state] * np.float32(
                prior.get("signed_surprise", 0.0)
            )
            expected_ev += probability[:, state] * np.float32(prior.get("ev", 0.0))
        entropy = -np.sum(
            probability * np.log(np.maximum(probability, 1e-8)), axis=1
        ) / math.log(3.0)
        generated = pd.DataFrame({"_market_ts": collapsed["_market_ts"]})
        for state, name in enumerate(MARKET_STATE_NAMES):
            generated[f"{MARKET_PREFIX}prob__{name}"] = probability[:, state]
        generated[f"{MARKET_PREFIX}expected_signed_surprise"] = expected_surprise
        generated[f"{MARKET_PREFIX}expected_ev"] = expected_ev
        generated[f"{MARKET_PREFIX}entropy"] = entropy.astype(np.float32)
        generated[f"{MARKET_PREFIX}confidence"] = probability.max(axis=1).astype(
            np.float32
        )
        generated[f"{MARKET_PREFIX}support_log1p"] = np.float32(
            np.log1p(self.support_rows)
        )
        row_ts = pd.to_datetime(
            oos[self.config.timestamp_col], utc=True, errors="coerce"
        )
        output = pd.DataFrame({"_market_ts": row_ts}, index=oos.index).merge(
            generated, on="_market_ts", how="left", sort=False, validate="many_to_one"
        )
        return output.drop(columns="_market_ts").set_axis(oos.index).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "market_residual_state_recognizer_v1",
            "train_start": self.train_start_,
            "train_end": self.train_end_,
            "support_timestamps": int(self.support_rows),
            "selected_features": list(self.feature_columns),
            "explicit_market_features": list(self.explicit_market_features),
            "cross_sectional_source_features": list(
                self.cross_sectional_source_features
            ),
            "label_diagnostics": dict(self.label_diagnostics),
            "feature_relevance": list(self.feature_relevance),
            "generated_features": market_residual_feature_names(),
            "fit_scope": "broad_market_second_stage",
            "leakage_contract": {
                "outcomes": "train-only market residual state labels",
                "oos": "market-wide pre-entry features and frozen model only",
                "asset_residuals_at_inference": False,
                "recent_hit_rate_inputs": False,
            },
        }


@dataclass(frozen=True)
class PerArchetypeMarketAdverseConfig:
    """Configuration for market states that explain one base archetype at a time."""

    score_col: str = "score_regime_calibrated"
    timestamp_col: str = "__ts__"
    archetype_col: str = "archetype_policy_key"
    ev_col: str = "ev_after_1pct"
    min_archetype_rows: int = 800
    min_selected_rows: int = 80
    min_adverse_days: int = 4
    max_fit_rows: int = 40_000
    max_features: int = 48
    max_cross_sectional_sources: int = 24
    adverse_day_quantile: float = 0.20
    risk_top_fraction: float = 0.10
    ae_gmm_max_rows: int = 3_000
    ae_max_iter: int = 64
    cluster_candidates: tuple[int, ...] = (3, 4, 5, 6, 7)
    random_state: int = 20260712


@dataclass
class _PerArchetypeAdverseModel:
    archetype: str
    feature_columns: list[str]
    medians: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray
    mlp_center: np.ndarray
    mlp_scale: np.ndarray
    lgbm_model: Any
    mlp_model: Any
    ae_gmm_state: dict[str, Any]
    ae_gmm_cluster_adverse_priors: np.ndarray
    ae_gmm_cluster_semantics: dict[int, str]
    ae_gmm_cluster_outcome_priors: dict[int, dict[str, float]]
    support_rows: int
    support_days: int
    adverse_days: int
    episode_count: int
    adverse_surprise_mean: float
    neutral_surprise_mean: float
    adverse_ev_mean: float
    neutral_ev_mean: float
    adverse_severity_mean: float
    neutral_severity_mean: float
    surprise_cut: float
    feature_relevance: list[dict[str, Any]] = field(default_factory=list)
    episode_catalog: list[dict[str, Any]] = field(default_factory=list)


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float32), 1e-7, 1.0 - 1e-7)
    return (
        -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
    ).astype(np.float32)


def _contiguous_episode_ids(
    day: pd.Series, adverse: pd.Series
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Assign episode IDs to contiguous adverse days without using row order."""

    ordered = pd.DataFrame(
        {
            "day": pd.to_datetime(day, utc=True, errors="coerce"),
            "adverse": pd.Series(adverse).fillna(False).astype(bool).to_numpy(),
        }
    ).sort_values("day", kind="stable")
    episode = np.full(len(ordered), -1, dtype=np.int32)
    current = -1
    previous: pd.Timestamp | None = None
    catalog: list[dict[str, Any]] = []
    for position, row in enumerate(ordered.itertuples(index=False)):
        if not bool(row.adverse):
            previous = None
            continue
        current_day = pd.Timestamp(row.day)
        if previous is None or current_day - previous > pd.Timedelta(days=1):
            current += 1
            catalog.append(
                {
                    "episode_id": int(current),
                    "start": str(current_day),
                    "end": str(current_day),
                    "days": 1,
                }
            )
        else:
            catalog[-1]["end"] = str(current_day)
            catalog[-1]["days"] = int(catalog[-1]["days"]) + 1
        episode[position] = current
        previous = current_day
    restored = np.full(len(ordered), -1, dtype=np.int32)
    restored[ordered.index.to_numpy(dtype=np.int64)] = episode
    return restored, catalog


def _adverse_daily_targets(
    prepared: pd.DataFrame,
    *,
    config: PerArchetypeMarketAdverseConfig,
    surprise_cut: float | None = None,
) -> tuple[pd.DataFrame, float, list[dict[str, Any]]]:
    """Build an adverse episode target from the frozen reachable-EV population."""

    timestamp = pd.to_datetime(
        prepared[config.timestamp_col], utc=True, errors="coerce"
    )
    selected = pd.to_numeric(
        prepared.get("reference_ev_equivalent_selected"), errors="coerce"
    ).fillna(0.0).ge(0.5)
    selected &= pd.to_numeric(
        prepared.get("reference_rank_pct"), errors="coerce"
    ).fillna(0.0).ge(0.80)
    work = prepared.loc[selected].copy(deep=False)
    if work.empty:
        return pd.DataFrame(), float("nan"), []
    work = work.copy()
    for target, candidates in {
        "_bad_mae": ("full_path_bad_mae_1r", "first_touch_bad_mae_1r"),
        "_timeout": ("timeout",),
        "_dirty": ("dirty_positive",),
        "_stop_or_adverse": ("stop_or_adverse", "full_stop_loss"),
    }.items():
        source = next((name for name in candidates if name in work.columns), None)
        work[target] = (
            pd.to_numeric(work[source], errors="coerce").fillna(0.0)
            if source is not None
            else np.float32(0.0)
        )
    work_day = pd.to_datetime(
        work[config.timestamp_col], utc=True, errors="coerce"
    ).dt.floor("D")
    daily = (
        work.assign(_day=work_day)
        .groupby("_day", sort=True, observed=True)
        .agg(
            selected_rows=(config.timestamp_col, "size"),
            signed_surprise=("hit_surprise", "mean"),
            mean_ev=(config.ev_col, "mean"),
            clean_rate=("clean_exec", "mean"),
            bad_mae_rate=("_bad_mae", "mean"),
            timeout_rate=("_timeout", "mean"),
            dirty_positive_rate=("_dirty", "mean"),
            stop_or_adverse_rate=("_stop_or_adverse", "mean"),
        )
        .reset_index()
        .rename(columns={"_day": "day"})
    )
    signed = pd.to_numeric(daily["signed_surprise"], errors="coerce").fillna(0.0)
    if surprise_cut is None or not np.isfinite(surprise_cut):
        surprise_cut = float(
            signed.quantile(float(np.clip(config.adverse_day_quantile, 0.05, 0.45)))
        )
    seed = signed.le(float(surprise_cut)) & signed.lt(0.0)
    adjacent_seed = seed | seed.shift(1, fill_value=False) | seed.shift(
        -1, fill_value=False
    )
    economically_bad = signed.lt(0.0) | pd.to_numeric(
        daily["mean_ev"], errors="coerce"
    ).fillna(0.0).lt(0.0)
    daily["market_adverse_label"] = (adjacent_seed & economically_bad).astype(
        np.int8
    )
    negative_scale = max(
        float(np.nanmedian(np.abs(signed.to_numpy(dtype=np.float64)))), 1e-4
    )
    ev_values = pd.to_numeric(daily["mean_ev"], errors="coerce").fillna(0.0)
    ev_scale = max(float(np.nanmedian(np.abs(ev_values.to_numpy()))), 1e-4)
    daily["market_adverse_severity"] = (
        np.maximum(-signed.to_numpy(dtype=np.float32) / negative_scale, 0.0)
        + np.maximum(-ev_values.to_numpy(dtype=np.float32) / ev_scale, 0.0)
    ).astype(np.float32)
    episode, catalog = _contiguous_episode_ids(
        daily["day"], daily["market_adverse_label"]
    )
    daily["market_adverse_episode_id"] = episode
    return daily, float(surprise_cut), catalog


def _predict_binary(model: Any, x: np.ndarray) -> np.ndarray | None:
    if model is None:
        return None
    if hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(x), dtype=np.float32)
        return probability[:, -1] if probability.ndim == 2 else probability
    probability = np.asarray(model.predict(x), dtype=np.float32)
    return probability[:, -1] if probability.ndim == 2 else probability


def _semantic_cluster_priors(
    posterior: np.ndarray,
    work: pd.DataFrame,
) -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """Map unstable GMM components to stable economic failure semantics."""

    posterior = np.asarray(posterior, dtype=np.float32)
    if posterior.ndim != 2 or posterior.shape[1] == 0:
        return {}, {}
    descriptors = {
        "adverse": pd.to_numeric(
            work["market_adverse_label"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32),
        "signed_surprise": pd.to_numeric(
            work["signed_surprise"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32),
        "ev": pd.to_numeric(work["mean_ev"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "bad_mae": pd.to_numeric(work["bad_mae_rate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "timeout": pd.to_numeric(work["timeout_rate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "dirty": pd.to_numeric(work["dirty_positive_rate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "stop_or_adverse": pd.to_numeric(
            work["stop_or_adverse_rate"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32),
        "severity": pd.to_numeric(
            work["market_adverse_severity"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32),
    }
    shock_columns = [
        name
        for name in (
            "mkt_systemic_deleveraging_score",
            "mkt_flush_exhaustion_score",
            "market_pc1_variance_share_12h",
            "cross_asset_downside_corr_1h",
        )
        if name in work.columns
    ]
    if shock_columns:
        descriptors["systemic_shock"] = (
            work[shock_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .mean(axis=1)
            .to_numpy(dtype=np.float32)
        )
    else:
        descriptors["systemic_shock"] = np.zeros(len(work), dtype=np.float32)
    global_prior = {name: float(np.mean(values)) for name, values in descriptors.items()}
    global_scale = {
        name: max(float(np.std(values)), 1e-4) for name, values in descriptors.items()
    }
    semantics: dict[int, str] = {}
    priors: dict[int, dict[str, float]] = {}
    alpha = 20.0
    for cluster in range(posterior.shape[1]):
        weight = posterior[:, cluster].astype(np.float64)
        support = float(weight.sum())
        local = {
            name: float(
                (np.dot(weight, values.astype(np.float64)) + alpha * global_prior[name])
                / max(support + alpha, 1e-6)
            )
            for name, values in descriptors.items()
        }
        local["support"] = support
        priors[cluster] = local
        if local["adverse"] < max(0.08, 0.75 * global_prior["adverse"]):
            semantics[cluster] = "neutral"
            continue
        scores = {
            "bad_mae_path": (local["bad_mae"] - global_prior["bad_mae"])
            / global_scale["bad_mae"],
            "slow_timeout": (local["timeout"] - global_prior["timeout"])
            / global_scale["timeout"],
            "dirty_overconfident": (local["dirty"] - global_prior["dirty"])
            / global_scale["dirty"],
            "negative_ev": (global_prior["ev"] - local["ev"])
            / global_scale["ev"],
            "systemic_shock": (
                local["systemic_shock"] - global_prior["systemic_shock"]
            )
            / global_scale["systemic_shock"],
        }
        semantic, semantic_score = max(scores.items(), key=lambda item: item[1])
        semantics[cluster] = (
            semantic if semantic_score > 0.15 else "high_variance_uncertain"
        )
    return semantics, priors


@dataclass
class PerArchetypeMarketAdverseRecognizer:
    """Predict adverse broad-market episodes separately for each archetype.

    Realized residuals define train labels only. OOS assignment consumes frozen
    market-wide features and emits continuous probabilities; no cluster ID is
    exposed to the downstream meta model.
    """

    config: PerArchetypeMarketAdverseConfig = field(
        default_factory=PerArchetypeMarketAdverseConfig
    )
    candidate_features: list[str] = field(default_factory=list)
    models: dict[str, _PerArchetypeAdverseModel] = field(default_factory=dict)
    explicit_market_features: list[str] = field(default_factory=list)
    cross_sectional_source_features: list[str] = field(default_factory=list)
    ev_equivalent_thresholds_: dict[tuple[str, str], float] = field(
        default_factory=dict
    )
    global_top10_ev_: float | None = None
    score_reference_values_: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    train_start_: str | None = None
    train_end_: str | None = None
    selection_source_: str = "static_train_ev_equivalent_threshold"

    def _configure_observable_features(self, frame: pd.DataFrame) -> None:
        helper = MarketResidualStateRecognizer(
            MarketResidualConfig(
                score_col=self.config.score_col,
                max_features=self.config.max_features,
                max_cross_sectional_sources=self.config.max_cross_sectional_sources,
                random_state=self.config.random_state,
            ),
            list(self.candidate_features),
        )
        helper._configure_observable_features(frame)
        self.explicit_market_features = list(helper.explicit_market_features)
        self.cross_sectional_source_features = list(
            helper.cross_sectional_source_features
        )

    def _collapse(self, frame: pd.DataFrame) -> pd.DataFrame:
        helper = MarketResidualStateRecognizer(
            MarketResidualConfig(
                score_col=self.config.score_col,
                timestamp_col=self.config.timestamp_col,
                ev_col=self.config.ev_col,
            ),
            [],
        )
        helper.explicit_market_features = self.explicit_market_features
        helper.cross_sectional_source_features = self.cross_sectional_source_features
        return helper._collapse(frame, training=False)

    def _prepared(self, frame: pd.DataFrame, *, frozen: bool) -> pd.DataFrame:
        kwargs: dict[str, Any] = {}
        if frozen:
            kwargs = {
                "ev_equivalent_thresholds": self.ev_equivalent_thresholds_,
                "global_top10_ev": self.global_top10_ev_,
                "score_reference_values": self.score_reference_values_,
            }
        return add_reference_surprise_targets(
            frame,
            ResidualArchetypeConfig(
                score_col=self.config.score_col,
                archetype_col=self.config.archetype_col,
                rank_scope="global",
                allow_side_fallback=False,
            ),
            **kwargs,
        )

    def fit(self, train: pd.DataFrame) -> "PerArchetypeMarketAdverseRecognizer":
        if lgb is None:
            raise RuntimeError("LightGBM is required for market adverse states")
        prepared = self._prepared(train, frozen=False)
        self.ev_equivalent_thresholds_ = dict(
            prepared.attrs.get("ev_equivalent_thresholds", {})
        )
        self.global_top10_ev_ = prepared.attrs.get("global_top10_ev")
        self.score_reference_values_ = np.sort(
            pd.to_numeric(prepared.get("_reference_score"), errors="coerce")
            .dropna()
            .to_numpy(dtype=np.float32, copy=True)
        )
        timestamp = pd.to_datetime(
            prepared[self.config.timestamp_col], utc=True, errors="coerce"
        )
        self.train_start_ = str(timestamp.min())
        self.train_end_ = str(timestamp.max())
        self._configure_observable_features(prepared)
        candidates = list(self.explicit_market_features) + [
            name
            for source in self.cross_sectional_source_features
            for name in (f"xs_median__{source}", f"xs_iqr__{source}")
        ]
        market_panel = self._collapse(prepared).sort_values(
            "_market_ts", kind="stable"
        )
        archetype = prepared.get(
            self.config.archetype_col,
            pd.Series("missing", index=prepared.index),
        ).astype(str)
        self.models = {}
        grouped = pd.Series(archetype, index=prepared.index).groupby(
            archetype, sort=True
        )
        for model_number, (arch, idx) in enumerate(grouped.groups.items()):
            group = prepared.loc[idx]
            if len(group) < int(self.config.min_archetype_rows):
                continue
            daily, surprise_cut, episode_catalog = _adverse_daily_targets(
                group, config=self.config
            )
            if (
                daily.empty
                or int(daily["selected_rows"].sum())
                < int(self.config.min_selected_rows)
                or int(daily["market_adverse_label"].sum())
                < int(self.config.min_adverse_days)
            ):
                continue
            group_timestamps = pd.DataFrame(
                {
                    "_market_ts": pd.to_datetime(
                        group[self.config.timestamp_col], utc=True, errors="coerce"
                    ).drop_duplicates()
                }
            )
            collapsed = group_timestamps.merge(
                market_panel,
                on="_market_ts",
                how="inner",
                validate="one_to_one",
                sort=False,
            )
            collapsed["day"] = pd.to_datetime(
                collapsed["_market_ts"], utc=True, errors="coerce"
            ).dt.floor("D")
            work = collapsed.merge(
                daily,
                on="day",
                how="inner",
                validate="many_to_one",
                sort=False,
            ).sort_values("_market_ts", kind="stable")
            labels = work["market_adverse_label"].to_numpy(dtype=np.int8)
            if len(work) < 200 or np.unique(labels).size < 2:
                continue
            screen_cfg = ResidualArchetypeConfig(
                max_recognizer_features=int(self.config.max_features),
                mutual_info_rows=min(len(work), int(self.config.max_fit_rows)),
                feature_screen_mode="binned_mi_lgbm",
                random_state=int(self.config.random_state + model_number * 101),
            )
            features, relevance = _select_recognizer_features(
                work,
                labels,
                candidates,
                screen_cfg,
                int(self.config.random_state + model_number * 101),
            )
            if len(features) < 2:
                continue
            x, medians, clip_low, clip_high = _prepare_numeric_matrix(work, features)
            fit_idx = _time_spread_indices(len(work), int(self.config.max_fit_rows))
            fit_labels = labels[fit_idx]
            day_count = (
                work["day"]
                .groupby(work["day"], sort=False)
                .transform("size")
                .to_numpy(dtype=np.float32)
            )
            weight = 1.0 / np.maximum(day_count, 1.0)
            positive_rate = float(np.mean(fit_labels))
            class_weight = np.where(
                labels > 0,
                0.5 / max(positive_rate, 1e-4),
                0.5 / max(1.0 - positive_rate, 1e-4),
            ).astype(np.float32)
            weight *= class_weight
            weight *= 1.0 + np.minimum(
                pd.to_numeric(
                    work["market_adverse_severity"], errors="coerce"
                ).fillna(0.0).to_numpy(dtype=np.float32),
                4.0,
            )
            weight /= max(float(np.mean(weight[fit_idx])), 1e-6)
            dataset = lgb.Dataset(
                x[fit_idx],
                label=fit_labels,
                weight=weight[fit_idx],
                free_raw_data=True,
            )
            lgbm_model = lgb.train(
                {
                    "objective": "binary",
                    "learning_rate": 0.035,
                    "num_leaves": 7,
                    "max_depth": 3,
                    "min_data_in_leaf": 40,
                    "feature_fraction": 0.80,
                    "lambda_l1": 0.10,
                    "lambda_l2": 5.0,
                    "seed": int(self.config.random_state + model_number * 101),
                    "num_threads": 2,
                    "verbosity": -1,
                    "force_col_wise": True,
                },
                dataset,
                num_boost_round=100,
            )
            mlp_center = np.median(x[fit_idx], axis=0).astype(np.float32)
            q25, q75 = np.percentile(x[fit_idx], [25.0, 75.0], axis=0)
            mlp_scale = np.maximum(q75 - q25, 1e-4).astype(np.float32)
            mlp_model: Any = None
            if MLPClassifier is not None:
                rng = np.random.default_rng(
                    int(self.config.random_state + model_number * 101 + 17)
                )
                positive = fit_idx[fit_labels > 0]
                negative = fit_idx[fit_labels <= 0]
                if len(positive) >= 20 and len(negative) >= 20:
                    negative_cap = min(len(negative), max(len(positive) * 4, 100))
                    negative = rng.choice(negative, negative_cap, replace=False)
                    mlp_idx = np.sort(np.concatenate([positive, negative]))
                    mlp_model = MLPClassifier(
                        hidden_layer_sizes=(24, 8),
                        activation="relu",
                        alpha=0.01,
                        batch_size=min(256, max(32, len(mlp_idx) // 10)),
                        learning_rate_init=0.002,
                        max_iter=100,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=10,
                        random_state=int(
                            self.config.random_state + model_number * 101 + 17
                        ),
                    )
                    mlp_model.fit(
                        (x[mlp_idx] - mlp_center) / mlp_scale,
                        labels[mlp_idx],
                    )
            ae_state = fit_ae_gmm_state(
                work.reindex(columns=features),
                economic_targets={
                    "adverse_episode": labels.astype(np.float32),
                    "negative_surprise": np.maximum(
                        -pd.to_numeric(
                            work["signed_surprise"], errors="coerce"
                        ).fillna(0.0).to_numpy(dtype=np.float32),
                        0.0,
                    ),
                    "returns": pd.to_numeric(
                        work["mean_ev"], errors="coerce"
                    ).fillna(0.0).to_numpy(dtype=np.float32),
                    "time_bucket": pd.to_datetime(
                        work["_market_ts"], utc=True
                    ).astype("int64").to_numpy(dtype=np.float64)
                    / float(7 * 24 * 60 * 60 * 1_000_000_000),
                },
                random_state=int(self.config.random_state + model_number * 101 + 31),
                max_train_rows=int(self.config.ae_gmm_max_rows),
                gmm_max_train_rows=int(self.config.ae_gmm_max_rows),
                ae_max_iter=int(self.config.ae_max_iter),
                cluster_candidates=self.config.cluster_candidates,
                reg_covar_candidates=(1e-4, 1e-3, 3e-3),
                smooth_lambda_candidates=(0.0,),
                path_aware_hpo=True,
                temporal_concentration_hpo=True,
            )
            cluster_priors = np.empty(0, dtype=np.float32)
            cluster_semantics: dict[int, str] = {}
            cluster_outcome_priors: dict[int, dict[str, float]] = {}
            if bool(ae_state.get("enabled", False)):
                ae_train = transform_ae_gmm_features(
                    work.reindex(columns=features),
                    ae_state,
                    index=work.index,
                    prefix="market_arch_ae_",
                )
                components = int(ae_state.get("gmm_n_components", 0) or 0)
                posterior = ae_train.reindex(
                    columns=[
                        f"market_arch_ae_gmm_prob_{i}" for i in range(components)
                    ],
                    fill_value=0.0,
                ).to_numpy(dtype=np.float32, copy=False)
                global_rate = float(np.mean(labels))
                alpha = 20.0
                support = posterior.sum(axis=0)
                cluster_priors = (
                    (posterior.T @ labels.astype(np.float32)) + alpha * global_rate
                ) / np.maximum(support + alpha, 1e-6)
                cluster_priors = cluster_priors.astype(np.float32)
                cluster_semantics, cluster_outcome_priors = _semantic_cluster_priors(
                    posterior, work
                )
            adverse = labels > 0
            surprise = pd.to_numeric(
                work["signed_surprise"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32)
            ev = pd.to_numeric(work["mean_ev"], errors="coerce").fillna(
                0.0
            ).to_numpy(dtype=np.float32)
            severity = pd.to_numeric(
                work["market_adverse_severity"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32)
            self.models[str(arch)] = _PerArchetypeAdverseModel(
                archetype=str(arch),
                feature_columns=list(features),
                medians=medians,
                clip_low=clip_low,
                clip_high=clip_high,
                mlp_center=mlp_center,
                mlp_scale=mlp_scale,
                lgbm_model=lgbm_model,
                mlp_model=mlp_model,
                ae_gmm_state=ae_state,
                ae_gmm_cluster_adverse_priors=cluster_priors,
                ae_gmm_cluster_semantics=cluster_semantics,
                ae_gmm_cluster_outcome_priors=cluster_outcome_priors,
                support_rows=int(len(work)),
                support_days=int(work["day"].nunique()),
                adverse_days=int(daily["market_adverse_label"].sum()),
                episode_count=int(len(episode_catalog)),
                adverse_surprise_mean=float(np.mean(surprise[adverse])),
                neutral_surprise_mean=float(np.mean(surprise[~adverse])),
                adverse_ev_mean=float(np.mean(ev[adverse])),
                neutral_ev_mean=float(np.mean(ev[~adverse])),
                adverse_severity_mean=float(np.mean(severity[adverse])),
                neutral_severity_mean=float(np.mean(severity[~adverse])),
                surprise_cut=float(surprise_cut),
                feature_relevance=list(relevance),
                episode_catalog=episode_catalog,
            )
        return self

    def _transform_model(
        self,
        frame: pd.DataFrame,
        model: _PerArchetypeAdverseModel,
        *,
        market_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        frame_timestamps = pd.DataFrame(
            {
                "_market_ts": pd.to_datetime(
                    frame[self.config.timestamp_col], utc=True, errors="coerce"
                ).drop_duplicates()
            }
        )
        collapsed = frame_timestamps.merge(
            market_panel,
            on="_market_ts",
            how="inner",
            validate="one_to_one",
            sort=False,
        ).sort_values("_market_ts", kind="stable")
        x, _, _, _ = _prepare_numeric_matrix(
            collapsed,
            model.feature_columns,
            medians=model.medians,
            clip_low=model.clip_low,
            clip_high=model.clip_high,
        )
        probabilities: dict[str, np.ndarray] = {}
        semantic_probability = {
            name: np.zeros(len(collapsed), dtype=np.float32)
            for name in MARKET_ADVERSE_STATE_NAMES
        }
        posterior_expected = {
            name: np.zeros(len(collapsed), dtype=np.float32)
            for name in (
                "bad_mae",
                "timeout",
                "dirty",
                "stop_or_adverse",
            )
        }
        lgbm_probability = _predict_binary(model.lgbm_model, x)
        if lgbm_probability is not None:
            probabilities["lgbm"] = np.clip(lgbm_probability, 0.0, 1.0)
        mlp_probability = _predict_binary(
            model.mlp_model, (x - model.mlp_center) / model.mlp_scale
        )
        if mlp_probability is not None:
            probabilities["mlp"] = np.clip(mlp_probability, 0.0, 1.0)
        if (
            bool(model.ae_gmm_state.get("enabled", False))
            and len(model.ae_gmm_cluster_adverse_priors) > 0
        ):
            ae_values = transform_ae_gmm_features(
                collapsed.reindex(columns=model.feature_columns),
                model.ae_gmm_state,
                index=collapsed.index,
                prefix="market_arch_ae_",
            )
            components = len(model.ae_gmm_cluster_adverse_priors)
            posterior = ae_values.reindex(
                columns=[f"market_arch_ae_gmm_prob_{i}" for i in range(components)],
                fill_value=0.0,
            ).to_numpy(dtype=np.float32, copy=False)
            probabilities["aegmm"] = np.clip(
                posterior @ model.ae_gmm_cluster_adverse_priors, 0.0, 1.0
            )
            for cluster in range(components):
                semantic = model.ae_gmm_cluster_semantics.get(
                    cluster, "high_variance_uncertain"
                )
                semantic_probability[semantic] += posterior[:, cluster]
                prior = model.ae_gmm_cluster_outcome_priors.get(cluster, {})
                for name in posterior_expected:
                    posterior_expected[name] += posterior[:, cluster] * np.float32(
                        prior.get(name, 0.0)
                    )
            semantic_sum = np.sum(
                np.column_stack(list(semantic_probability.values())), axis=1
            )
            missing_mass = np.maximum(1.0 - semantic_sum, 0.0)
            semantic_probability["high_variance_uncertain"] += missing_mass.astype(
                np.float32
            )
        if probabilities:
            matrix = np.column_stack(list(probabilities.values())).astype(np.float32)
            ensemble = matrix.mean(axis=1).astype(np.float32)
            disagreement = matrix.std(axis=1).astype(np.float32)
        else:
            ensemble = np.zeros(len(collapsed), dtype=np.float32)
            disagreement = np.zeros(len(collapsed), dtype=np.float32)
        generated = pd.DataFrame({"_market_ts": collapsed["_market_ts"]})
        for name in ("aegmm", "mlp", "lgbm"):
            generated[f"{MARKET_ADVERSE_PREFIX}prob_adverse__{name}"] = (
                probabilities.get(name, ensemble).astype(np.float32)
            )
        generated[f"{MARKET_ADVERSE_PREFIX}prob_adverse__ensemble"] = ensemble
        if not any(np.any(values) for values in semantic_probability.values()):
            semantic_probability["high_variance_uncertain"] = ensemble.copy()
            semantic_probability["neutral"] = (1.0 - ensemble).astype(np.float32)
        for name in MARKET_ADVERSE_STATE_NAMES:
            generated[f"{MARKET_ADVERSE_PREFIX}state_prob__{name}"] = np.clip(
                semantic_probability[name], 0.0, 1.0
            ).astype(np.float32)
        generated[f"{MARKET_ADVERSE_PREFIX}probability_disagreement"] = disagreement
        generated[f"{MARKET_ADVERSE_PREFIX}binary_entropy"] = _binary_entropy(
            ensemble
        )
        generated[f"{MARKET_ADVERSE_PREFIX}expected_signed_surprise"] = (
            ensemble * np.float32(model.adverse_surprise_mean)
            + (1.0 - ensemble) * np.float32(model.neutral_surprise_mean)
        )
        generated[f"{MARKET_ADVERSE_PREFIX}expected_ev"] = (
            ensemble * np.float32(model.adverse_ev_mean)
            + (1.0 - ensemble) * np.float32(model.neutral_ev_mean)
        )
        generated[f"{MARKET_ADVERSE_PREFIX}expected_severity"] = (
            ensemble * np.float32(model.adverse_severity_mean)
            + (1.0 - ensemble) * np.float32(model.neutral_severity_mean)
        )
        generated[f"{MARKET_ADVERSE_PREFIX}expected_bad_mae"] = posterior_expected[
            "bad_mae"
        ]
        generated[f"{MARKET_ADVERSE_PREFIX}expected_timeout"] = posterior_expected[
            "timeout"
        ]
        generated[f"{MARKET_ADVERSE_PREFIX}expected_dirty_positive"] = (
            posterior_expected["dirty"]
        )
        generated[f"{MARKET_ADVERSE_PREFIX}expected_stop_or_adverse"] = (
            posterior_expected["stop_or_adverse"]
        )
        generated[f"{MARKET_ADVERSE_PREFIX}support_log1p"] = np.float32(
            np.log1p(model.support_rows)
        )
        generated[f"{MARKET_ADVERSE_PREFIX}episode_support_log1p"] = np.float32(
            np.log1p(model.episode_count)
        )
        generated[f"{MARKET_ADVERSE_PREFIX}local_model_available"] = np.float32(1.0)
        row_ts = pd.to_datetime(
            frame[self.config.timestamp_col], utc=True, errors="coerce"
        )
        return (
            pd.DataFrame({"_market_ts": row_ts}, index=frame.index)
            .merge(
                generated,
                on="_market_ts",
                how="left",
                sort=False,
                validate="many_to_one",
            )
            .drop(columns="_market_ts")
            .set_axis(frame.index)
            .astype(np.float32)
        )

    def transform_oos(self, oos: pd.DataFrame) -> pd.DataFrame:
        forbidden = sorted(
            (OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS).intersection(oos.columns)
        )
        if forbidden:
            raise ValueError(
                f"OOS per-archetype market transform received outcomes: {forbidden[:12]}"
            )
        output = pd.DataFrame(
            0.0,
            index=oos.index,
            columns=market_archetype_adverse_feature_names(),
            dtype=np.float32,
        )
        archetype = oos.get(
            self.config.archetype_col,
            pd.Series("missing", index=oos.index),
        ).astype(str)
        market_panel = self._collapse(oos).sort_values("_market_ts", kind="stable")
        for arch, idx in pd.Series(archetype, index=oos.index).groupby(
            archetype, sort=False
        ).groups.items():
            model = self.models.get(str(arch))
            if model is None:
                continue
            transformed = self._transform_model(
                oos.loc[idx], model, market_panel=market_panel
            )
            output.loc[idx, transformed.columns] = transformed.to_numpy(
                dtype=np.float32, copy=False
            )
        return output.astype(np.float32, copy=False)

    def prepare_evaluation_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepared(frame, frozen=True)
        output = pd.DataFrame(index=prepared.index)
        output["market_adverse_label"] = np.int8(0)
        output["market_adverse_episode_id"] = np.int32(-1)
        output["market_adverse_severity"] = np.float32(0.0)
        archetype = prepared.get(
            self.config.archetype_col,
            pd.Series("missing", index=prepared.index),
        ).astype(str)
        for arch, idx in pd.Series(archetype, index=prepared.index).groupby(
            archetype, sort=False
        ).groups.items():
            model = self.models.get(str(arch))
            if model is None:
                continue
            group = prepared.loc[idx]
            daily, _, _ = _adverse_daily_targets(
                group,
                config=self.config,
                surprise_cut=model.surprise_cut,
            )
            if daily.empty:
                continue
            day = pd.to_datetime(
                group[self.config.timestamp_col], utc=True, errors="coerce"
            ).dt.floor("D")
            mapping = daily.set_index("day")
            output.loc[idx, "market_adverse_label"] = (
                day.map(mapping["market_adverse_label"])
                .fillna(0)
                .to_numpy(dtype=np.int8)
            )
            output.loc[idx, "market_adverse_episode_id"] = (
                day.map(mapping["market_adverse_episode_id"])
                .fillna(-1)
                .to_numpy(dtype=np.int32)
            )
            output.loc[idx, "market_adverse_severity"] = (
                day.map(mapping["market_adverse_severity"])
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        return output

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "per_archetype_market_adverse_recognizer_v1",
            "train_start": self.train_start_,
            "train_end": self.train_end_,
            "archetype_model_count": int(len(self.models)),
            "archetypes": sorted(self.models),
            "generated_features": market_archetype_adverse_feature_names(),
            "probability_models": list(MARKET_ADVERSE_PROBABILITY_MODELS),
            "semantic_failure_states": list(MARKET_ADVERSE_STATE_NAMES),
            "target": "negative reachable-EV residual episodes",
            "market_panel_scope": (
                "full timestamp universe first; identical market inputs routed to "
                "separate per-archetype learners"
            ),
            "selection_contract": (
                self.selection_source_
            ),
            "ranking_contract": "continuous adverse probabilities; no score nudge or calibration",
            "episode_contract": "contiguous adverse UTC days; episode-balanced training and assessment",
            "selected_features_by_archetype": {
                key: list(model.feature_columns) for key, model in self.models.items()
            },
            "feature_relevance_by_archetype": {
                key: list(model.feature_relevance) for key, model in self.models.items()
            },
            "episode_catalog_by_archetype": {
                key: list(model.episode_catalog) for key, model in self.models.items()
            },
            "cluster_semantics_by_archetype": {
                key: {
                    str(cluster): semantic
                    for cluster, semantic in model.ae_gmm_cluster_semantics.items()
                }
                for key, model in self.models.items()
            },
            "cluster_outcome_priors_by_archetype": {
                key: {
                    str(cluster): dict(prior)
                    for cluster, prior in model.ae_gmm_cluster_outcome_priors.items()
                }
                for key, model in self.models.items()
            },
            "support_by_archetype": {
                key: {
                    "rows": int(model.support_rows),
                    "days": int(model.support_days),
                    "adverse_days": int(model.adverse_days),
                    "episodes": int(model.episode_count),
                }
                for key, model in self.models.items()
            },
            "leakage_contract": {
                "outcomes": "train-only residual episode labels",
                "oos": "frozen market-wide pre-entry features only",
                "recent_performance_inputs": False,
                "hard_cluster_ids_exposed": False,
                "continuous_probabilities": True,
                "global_market_state_model": False,
                "fallback_model": "none; unsupported archetypes emit unavailable/zero context",
            },
        }


def adverse_episode_ranking_metrics(
    frame: pd.DataFrame,
    *,
    archetype_col: str = "archetype_policy_key",
    timestamp_col: str = "__ts__",
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    """Assess adverse-state ranking at episode level; row metrics are secondary."""

    required = {timestamp_col, archetype_col, "market_adverse_label"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    probability_columns = {
        name: f"{MARKET_ADVERSE_PREFIX}prob_adverse__{name}"
        for name in MARKET_ADVERSE_PROBABILITY_MODELS
        if f"{MARKET_ADVERSE_PREFIX}prob_adverse__{name}" in frame.columns
    }
    probability_columns.update(
        {
            f"state::{name}": f"{MARKET_ADVERSE_PREFIX}state_prob__{name}"
            for name in MARKET_ADVERSE_STATE_NAMES
            if name != "neutral"
            and f"{MARKET_ADVERSE_PREFIX}state_prob__{name}" in frame.columns
        }
    )
    if not probability_columns:
        return pd.DataFrame()
    timestamp = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    work = frame.assign(_metric_ts=timestamp)
    aggregations: dict[str, str] = {
        "market_adverse_label": "max",
        "market_adverse_episode_id": "max",
    }
    for name in (
        "ev_after_1pct",
        "full_path_bad_mae_1r",
        "first_touch_bad_mae_1r",
        "timeout",
        "dirty_positive",
    ):
        if name in work.columns:
            aggregations[name] = "mean"
    aggregations.update({column: "median" for column in probability_columns.values()})
    collapsed = (
        work.groupby([archetype_col, "_metric_ts"], observed=True, sort=True)
        .agg(aggregations)
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("__all__", collapsed)]
    scopes.extend(
        (str(arch), part)
        for arch, part in collapsed.groupby(archetype_col, observed=True, sort=True)
    )
    for arch, part in scopes:
        target = pd.to_numeric(
            part["market_adverse_label"], errors="coerce"
        ).fillna(0).to_numpy(dtype=np.int8)
        episode = pd.to_numeric(
            part["market_adverse_episode_id"], errors="coerce"
        ).fillna(-1).to_numpy(dtype=np.int32)
        if arch == "__all__":
            episode_key = np.asarray(
                [
                    f"{local_arch}::{episode_id}" if episode_id >= 0 else ""
                    for local_arch, episode_id in zip(
                        part[archetype_col].astype(str), episode, strict=False
                    )
                ],
                dtype=object,
            )
            adverse_episodes: np.ndarray = np.unique(episode_key[episode_key != ""])
        else:
            episode_key = episode
            adverse_episodes = np.unique(episode[episode >= 0])
        day = pd.to_datetime(part["_metric_ts"], utc=True).dt.floor("D")
        for model_name, column in probability_columns.items():
            score = pd.to_numeric(part[column], errors="coerce").fillna(0.0)
            score_is_degenerate = bool(
                score.nunique(dropna=True) < 2
                or not np.isfinite(float(score.std()))
                or float(score.std()) <= 1e-8
            )
            threshold = float(score.quantile(1.0 - float(top_fraction)))
            selected = (
                np.zeros(len(score), dtype=bool)
                if score_is_degenerate
                else score.ge(threshold).to_numpy(dtype=bool)
            )
            episode_hits: list[float] = []
            episode_coverage: list[float] = []
            detection_delay_hours: list[float] = []
            for episode_id in adverse_episodes:
                mask = episode_key == episode_id
                hits = selected[mask]
                episode_hits.append(float(hits.any()))
                episode_coverage.append(float(hits.mean()))
                if hits.any():
                    episode_ts = pd.to_datetime(
                        part.loc[mask, "_metric_ts"], utc=True
                    ).sort_values()
                    first_hit = pd.to_datetime(
                        part.loc[mask & selected, "_metric_ts"], utc=True
                    ).min()
                    detection_delay_hours.append(
                        float((first_hit - episode_ts.iloc[0]).total_seconds() / 3600.0)
                    )
            daily = pd.DataFrame(
                {"day": day, "selected": selected, "adverse": target > 0}
            ).groupby("day", sort=True).max()
            false_alarm = daily.loc[daily["selected"], "adverse"]
            ap = np.nan
            if (
                not score_is_degenerate
                and average_precision_score is not None
                and np.unique(target).size > 1
            ):
                ap = float(average_precision_score(target, score.to_numpy()))
            rows.append(
                {
                    "archetype_policy_key": arch,
                    "model": model_name,
                    "timestamps": int(len(part)),
                    "adverse_rate": float(np.mean(target)),
                    "risk_top_fraction": float(top_fraction),
                    "risk_threshold": threshold,
                    "score_status": (
                        "degenerate_unavailable"
                        if score_is_degenerate
                        else "available"
                    ),
                    "score_std": float(score.std()),
                    "active_probability_rate": float(score.gt(0.0).mean()),
                    "row_average_precision": ap,
                    "risk_top_precision": float(np.mean(target[selected]))
                    if selected.any()
                    else np.nan,
                    "episode_count": int(len(adverse_episodes)),
                    "episode_recall": float(np.mean(episode_hits))
                    if episode_hits
                    else np.nan,
                    "mean_episode_coverage": float(np.mean(episode_coverage))
                    if episode_coverage
                    else np.nan,
                    "median_detection_delay_hours": float(
                        np.median(detection_delay_hours)
                    )
                    if detection_delay_hours
                    else np.nan,
                    "false_alarm_day_rate": float(1.0 - false_alarm.mean())
                    if len(false_alarm)
                    else np.nan,
                    "risk_top_mean_ev": float(
                        pd.to_numeric(
                            part.loc[selected, "ev_after_1pct"], errors="coerce"
                        ).mean()
                    )
                    if selected.any() and "ev_after_1pct" in part.columns
                    else np.nan,
                    "risk_top_bad_mae_rate": float(
                        pd.to_numeric(
                            part.loc[
                                selected,
                                (
                                    "full_path_bad_mae_1r"
                                    if "full_path_bad_mae_1r" in part.columns
                                    else "first_touch_bad_mae_1r"
                                ),
                            ],
                            errors="coerce",
                        ).mean()
                    )
                    if selected.any()
                    and (
                        "full_path_bad_mae_1r" in part.columns
                        or "first_touch_bad_mae_1r" in part.columns
                    )
                    else np.nan,
                    "risk_top_timeout_rate": float(
                        pd.to_numeric(
                            part.loc[selected, "timeout"], errors="coerce"
                        ).mean()
                    )
                    if selected.any() and "timeout" in part.columns
                    else np.nan,
                    "risk_top_dirty_positive_rate": float(
                        pd.to_numeric(
                            part.loc[selected, "dirty_positive"], errors="coerce"
                        ).mean()
                    )
                    if selected.any() and "dirty_positive" in part.columns
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)
