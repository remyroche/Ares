"""Causal side-specific heads for positive and negative meta surprise tails."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from .meta_residual_archetypes import (
    ResidualArchetypeConfig,
    _archetype,
    _time_spread_indices,
    add_reference_surprise_targets,
    inference_feature_columns,
)

SURPRISE_HEAD_OUTPUTS = (
    "meta_resid_signed_surprise_prediction",
    "meta_resid_negative_tail_probability",
    "meta_resid_positive_tail_probability",
    "meta_resid_surprise_head_net_probability",
    "meta_resid_surprise_head_support_log1p",
)


@dataclass
class _SideSurpriseHead:
    side: str
    feature_columns: list[str]
    archetype_categories: list[str]
    medians: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray
    signed_booster: Any
    negative_booster: Any
    positive_booster: Any
    tail_thresholds: dict[str, tuple[float, float]]
    fallback_thresholds: tuple[float, float]
    support_rows: int


@dataclass
class ResidualSurpriseHeadState:
    """Frozen side-specific surprise models; outcomes are used only during fit."""

    candidate_features: list[str]
    config: ResidualArchetypeConfig = field(default_factory=ResidualArchetypeConfig)
    max_fit_rows_per_side: int = 120_000
    side_models: dict[str, _SideSurpriseHead] = field(default_factory=dict)
    train_start_: str | None = None
    train_end_: str | None = None

    def _matrix(
        self,
        frame: pd.DataFrame,
        columns: Sequence[str],
        categories: Sequence[str],
        *,
        medians: np.ndarray | None = None,
        clip_low: np.ndarray | None = None,
        clip_high: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        numeric = (
            frame.reindex(columns=list(columns))
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        numeric[~np.isfinite(numeric)] = np.nan
        if medians is None:
            medians = np.nanmedian(numeric, axis=0).astype(np.float32)
            medians = np.nan_to_num(medians, nan=0.0)
        missing = ~np.isfinite(numeric)
        if missing.any():
            numeric = numeric.copy()
            numeric[missing] = np.take(medians, np.nonzero(missing)[1])
        if clip_low is None:
            clip_low = np.nanpercentile(numeric, 0.5, axis=0).astype(np.float32)
        if clip_high is None:
            clip_high = np.nanpercentile(numeric, 99.5, axis=0).astype(np.float32)
        np.clip(numeric, clip_low, clip_high, out=numeric)
        archetype = _archetype(frame, self.config.archetype_col).astype(str)
        mapping = {name: idx for idx, name in enumerate(categories)}
        positions = archetype.map(mapping).fillna(-1).to_numpy(dtype=np.int32)
        one_hot = np.zeros((len(frame), len(categories)), dtype=np.float32)
        valid = positions >= 0
        one_hot[np.flatnonzero(valid), positions[valid]] = 1.0
        return (
            np.concatenate([numeric, one_hot], axis=1).astype(np.float32, copy=False),
            medians,
            clip_low,
            clip_high,
        )

    @staticmethod
    def _fit_booster(x: np.ndarray, y: np.ndarray, objective: str, seed: int) -> Any:
        params = {
            "objective": objective,
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_data_in_leaf": 180,
            "bagging_fraction": 0.80,
            "bagging_freq": 1,
            "feature_fraction": 0.80,
            "lambda_l1": 0.10,
            "lambda_l2": 8.0,
            "seed": int(seed),
            "num_threads": 2,
            "verbosity": -1,
            "force_col_wise": True,
        }
        return lgb.train(
            params, lgb.Dataset(x, label=y, free_raw_data=True), num_boost_round=120
        )

    def fit(self, train: pd.DataFrame) -> "ResidualSurpriseHeadState":
        prepared = add_reference_surprise_targets(train, self.config)
        timestamp = pd.to_datetime(prepared["__ts__"], utc=True, errors="coerce")
        self.train_start_ = str(timestamp.min())
        self.train_end_ = str(timestamp.max())
        features = inference_feature_columns(prepared, self.candidate_features)
        self.side_models = {}
        side = prepared[self.config.side_col].astype(str).str.lower()
        for side_idx, (side_name, positions) in enumerate(
            side.groupby(side, sort=True).groups.items()
        ):
            group = prepared.loc[positions]
            group = group[group["reference_rank_pct"].ge(0.80)].copy()
            if len(group) < int(self.config.min_side_rows):
                continue
            archetype = _archetype(group, self.config.archetype_col).astype(str)
            categories = sorted(archetype.unique().tolist())
            surprise = pd.to_numeric(group["hit_surprise"], errors="coerce")
            thresholds: dict[str, tuple[float, float]] = {}
            for name, idx in archetype.groupby(archetype, sort=True).groups.items():
                values = surprise.loc[idx]
                thresholds[str(name)] = (
                    float(values.quantile(0.10)),
                    float(values.quantile(0.90)),
                )
            fallback = (float(surprise.quantile(0.10)), float(surprise.quantile(0.90)))
            low = archetype.map(
                {name: value[0] for name, value in thresholds.items()}
            ).fillna(fallback[0])
            high = archetype.map(
                {name: value[1] for name, value in thresholds.items()}
            ).fillna(fallback[1])
            negative = surprise.le(low).astype(np.float32).to_numpy()
            positive = surprise.ge(high).astype(np.float32).to_numpy()
            signed = surprise.to_numpy(dtype=np.float32)
            x, medians, clip_low, clip_high = self._matrix(group, features, categories)
            fit_idx = _time_spread_indices(len(group), self.max_fit_rows_per_side)
            seed = int(self.config.random_state + 1_000 + side_idx * 101)
            self.side_models[str(side_name)] = _SideSurpriseHead(
                side=str(side_name),
                feature_columns=list(features),
                archetype_categories=categories,
                medians=medians,
                clip_low=clip_low,
                clip_high=clip_high,
                signed_booster=self._fit_booster(
                    x[fit_idx], signed[fit_idx], "huber", seed
                ),
                negative_booster=self._fit_booster(
                    x[fit_idx], negative[fit_idx], "binary", seed + 1
                ),
                positive_booster=self._fit_booster(
                    x[fit_idx], positive[fit_idx], "binary", seed + 2
                ),
                tail_thresholds=thresholds,
                fallback_thresholds=fallback,
                support_rows=int(len(group)),
            )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(
            0.0, index=frame.index, columns=SURPRISE_HEAD_OUTPUTS, dtype=np.float32
        )
        side = (
            frame.get(
                self.config.side_col,
                pd.Series("missing", index=frame.index),
            )
            .astype(str)
            .str.lower()
        )
        for side_name, positions in side.groupby(side, sort=False).groups.items():
            model = self.side_models.get(str(side_name))
            if model is None:
                continue
            group = frame.loc[positions]
            x, _, _, _ = self._matrix(
                group,
                model.feature_columns,
                model.archetype_categories,
                medians=model.medians,
                clip_low=model.clip_low,
                clip_high=model.clip_high,
            )
            signed = np.asarray(model.signed_booster.predict(x), dtype=np.float32)
            negative = np.asarray(model.negative_booster.predict(x), dtype=np.float32)
            positive = np.asarray(model.positive_booster.predict(x), dtype=np.float32)
            output.loc[positions, "meta_resid_signed_surprise_prediction"] = signed
            output.loc[positions, "meta_resid_negative_tail_probability"] = negative
            output.loc[positions, "meta_resid_positive_tail_probability"] = positive
            output.loc[positions, "meta_resid_surprise_head_net_probability"] = (
                positive - negative
            )
            output.loc[positions, "meta_resid_surprise_head_support_log1p"] = (
                np.float32(np.log1p(model.support_rows))
            )
        return output.astype(np.float32, copy=False)

    def labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = add_reference_surprise_targets(frame, self.config)
        output = pd.DataFrame(index=frame.index)
        output["signed_surprise"] = pd.to_numeric(
            prepared["hit_surprise"], errors="coerce"
        )
        output["negative_tail"] = np.float32(0.0)
        output["positive_tail"] = np.float32(0.0)
        side = prepared[self.config.side_col].astype(str).str.lower()
        for side_name, positions in side.groupby(side, sort=False).groups.items():
            model = self.side_models.get(str(side_name))
            if model is None:
                continue
            archetype = _archetype(
                prepared.loc[positions], self.config.archetype_col
            ).astype(str)
            surprise = output.loc[positions, "signed_surprise"]
            low_map = {name: value[0] for name, value in model.tail_thresholds.items()}
            high_map = {name: value[1] for name, value in model.tail_thresholds.items()}
            low = archetype.map(low_map).fillna(model.fallback_thresholds[0])
            high = archetype.map(high_map).fillna(model.fallback_thresholds[1])
            output.loc[positions, "negative_tail"] = surprise.le(low).astype(np.float32)
            output.loc[positions, "positive_tail"] = surprise.ge(high).astype(
                np.float32
            )
        return output.astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "meta_residual_surprise_head_state_v1",
            "train_start": self.train_start_,
            "train_end": self.train_end_,
            "side_models": {
                side: {
                    "feature_count": len(model.feature_columns),
                    "archetype_categories": model.archetype_categories,
                    "support_rows": model.support_rows,
                    "tail_thresholds": model.tail_thresholds,
                }
                for side, model in self.side_models.items()
            },
            "outputs": list(SURPRISE_HEAD_OUTPUTS),
            "leakage_contract": (
                "Tail labels and thresholds are train-only; transform consumes only pre-entry "
                "features, side, and the existing base archetype."
            ),
        }
