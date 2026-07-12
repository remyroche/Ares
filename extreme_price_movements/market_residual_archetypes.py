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


def market_residual_feature_names() -> list[str]:
    return [f"{MARKET_PREFIX}prob__{name}" for name in MARKET_STATE_NAMES] + [
        f"{MARKET_PREFIX}expected_signed_surprise",
        f"{MARKET_PREFIX}expected_ev",
        f"{MARKET_PREFIX}entropy",
        f"{MARKET_PREFIX}confidence",
        f"{MARKET_PREFIX}support_log1p",
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
        labels = np.zeros(len(collapsed), dtype=np.int32)
        labels[(daily.to_numpy() <= lower) & (prior < 0.0)] = 1
        labels[(daily.to_numpy() >= upper) & (prior > 0.0)] = 2
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
        if len(self.feature_columns) < 2 or np.unique(labels).size < 2:
            raise ValueError("Market residual labels/features are degenerate")
        x, self.medians, self.clip_low, self.clip_high = _prepare_numeric_matrix(
            collapsed, self.feature_columns
        )
        fit_idx = _time_spread_indices(len(collapsed), int(self.config.max_fit_rows))
        dataset = lgb.Dataset(x[fit_idx], label=labels[fit_idx], free_raw_data=True)
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
