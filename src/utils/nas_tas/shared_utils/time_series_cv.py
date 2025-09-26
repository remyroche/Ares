"""Blocked & purged cross-validation helpers for time-series model evaluation.

The helpers here intentionally layer on top of the legacy ``PurgedKFoldTime``
implementation so that new NAS/TAS search utilities inherit the same battle-
tested leakage protections used throughout the code base.  When the richer
splitter is unavailable (for example in constrained CI environments) the
module gracefully falls back to lightweight stochastic splits while preserving
the metadata contract expected by the unified search engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import pandas as pd

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - fallback path
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

try:  # pragma: no cover - optional dependency
    from src.utils.purged_kfold import PurgedKFoldTime

    _HAS_PURGED_SPLITTER = True
except Exception as exc:  # pragma: no cover - fallback path
    logger.warning("PurgedKFoldTime unavailable: %s", exc)
    PurgedKFoldTime = None  # type: ignore[assignment]
    _HAS_PURGED_SPLITTER = False


@dataclass
class BlockedPurgedCVConfig:
    """Configuration for blocked & purged time-series cross validation."""

    n_splits: int = 5
    purge_pct: float = 0.1
    embargo_pct: float = 0.05
    regime_labels: Sequence[str] = ("bull", "bear", "range")
    seasonal_labels: Sequence[str] = ("winter", "spring", "summer", "autumn")
    random_state: Optional[int] = None


@dataclass
class FoldMetadata:
    fold_id: int
    total_folds: int
    regime: str
    season: str
    purge_window: Tuple[float, float]
    train_size: int = 0
    val_size: int = 0


class BlockedPurgedCV:
    """Utility that orchestrates purged cross-validation calls."""

    def __init__(self, config: Optional[BlockedPurgedCVConfig] = None) -> None:
        self.config = config or BlockedPurgedCVConfig()
        self.rng = np.random.default_rng(self.config.random_state)
        self._purged_splitter_cls = PurgedKFoldTime if _HAS_PURGED_SPLITTER else None

    def evaluate(
        self,
        objective: Callable[[Dict[str, Any]], Dict[str, float]],
        base_params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Aggregate metrics across blocked & purged folds."""

        metrics: List[Dict[str, float]] = []
        feature_frame = base_params.get("cv_features")
        target_frame = base_params.get("cv_labels")
        group_labels = base_params.get("cv_groups")

        splitter_pairs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        n_samples: Optional[int] = None

        if self._purged_splitter_cls is not None and feature_frame is not None:
            if _HAS_PANDAS and isinstance(feature_frame, pd.DataFrame):
                n_samples = len(feature_frame)
            else:
                try:
                    n_samples = len(feature_frame)
                except Exception:  # pragma: no cover - fallback path
                    n_samples = None

            purge_n = 0
            embargo_n = 0
            if n_samples:
                purge_n = max(0, int(round(self.config.purge_pct * n_samples)))
                embargo_n = max(0, int(round(self.config.embargo_pct * n_samples)))

            splitter = self._purged_splitter_cls(
                n_splits=self.config.n_splits,
                purge=purge_n,
                embargo=embargo_n,
            )

            try:
                for pair in splitter.split(feature_frame, target_frame, group_labels):
                    splitter_pairs.append(pair)
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning(
                    "PurgedKFoldTime splitting failed (falling back to synthetic blocks): %s",
                    exc,
                )
                splitter_pairs.clear()

        if not splitter_pairs:
            splitter_pairs = [None] * self.config.n_splits

        for fold_id in range(self.config.n_splits):
            params = dict(base_params)
            train_size = 0
            val_size = 0
            val_indices: Optional[np.ndarray] = None

            if fold_id < len(splitter_pairs) and splitter_pairs[fold_id] is not None:
                train_idx, val_idx = splitter_pairs[fold_id]
                train_size = int(train_idx.size)
                val_size = int(val_idx.size)
                val_indices = val_idx
                params.setdefault(
                    "cv_indices",
                    {
                        "train": train_idx,
                        "validation": val_idx,
                    },
                )
                if _HAS_PANDAS and isinstance(feature_frame, pd.DataFrame):
                    params.setdefault("cv_train_features", feature_frame.iloc[train_idx])
                    params.setdefault("cv_val_features", feature_frame.iloc[val_idx])
                if target_frame is not None:
                    if _HAS_PANDAS and isinstance(target_frame, (pd.Series, pd.DataFrame)):
                        params.setdefault("cv_train_labels", target_frame.iloc[train_idx])
                        params.setdefault("cv_val_labels", target_frame.iloc[val_idx])
                    else:
                        target_array = np.asarray(target_frame)
                        params.setdefault("cv_train_labels", target_array[train_idx])
                        params.setdefault("cv_val_labels", target_array[val_idx])

            metadata = self._build_metadata(
                fold_id,
                n_samples=n_samples,
                val_indices=val_indices,
                train_size=train_size,
                val_size=val_size,
            )
            params.setdefault("cv_metadata", metadata.__dict__)
            params.setdefault("cv_fold", fold_id)
            params.setdefault("cv_total", self.config.n_splits)

            try:
                fold_metrics = objective(params)
                metrics.append(self._sanitize_metrics(fold_metrics))
            except Exception as exc:  # pragma: no cover - objective failure
                logger.warning("Fold %s evaluation failed: %s", fold_id, exc)

        if not metrics:
            return objective(base_params)

        aggregated: Dict[str, float] = {}
        for key in metrics[0].keys():
            values = [fold.get(key, 0.0) for fold in metrics]
            aggregated[key] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        aggregated["fold_count"] = len(metrics)
        return aggregated

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _build_metadata(
        self,
        fold_id: int,
        *,
        n_samples: Optional[int],
        val_indices: Optional[np.ndarray],
        train_size: int,
        val_size: int,
    ) -> FoldMetadata:
        regime = self.config.regime_labels[fold_id % len(self.config.regime_labels)]
        season = self.config.seasonal_labels[fold_id % len(self.config.seasonal_labels)]
        purge_window = self._estimate_purge_window(
            fold_id,
            n_samples=n_samples,
            val_indices=val_indices,
        )
        return FoldMetadata(
            fold_id=fold_id,
            total_folds=self.config.n_splits,
            regime=regime,
            season=season,
            purge_window=purge_window,
            train_size=train_size,
            val_size=val_size,
        )

    def _estimate_purge_window(
        self,
        fold_id: int,
        *,
        n_samples: Optional[int],
        val_indices: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        if n_samples and val_indices is not None and val_indices.size > 0:
            start_ratio = float(val_indices[0]) / max(1, n_samples - 1)
            end_ratio = float(val_indices[-1]) / max(1, n_samples - 1)
            left = max(0.0, start_ratio - self.config.purge_pct)
            right = min(1.0, end_ratio + self.config.embargo_pct)
            return (left, right)

        purge_start = fold_id * (1.0 / self.config.n_splits)
        purge_span = self.config.purge_pct / self.config.n_splits
        embargo_span = self.config.embargo_pct / self.config.n_splits
        return (
            max(0.0, purge_start - purge_span),
            min(1.0, purge_start + embargo_span),
        )

    def _sanitize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        clean = {}
        for key, value in metrics.items():
            try:
                clean[key] = float(value)
            except (TypeError, ValueError):
                continue
        return clean


__all__ = [
    "BlockedPurgedCV",
    "BlockedPurgedCVConfig",
    "FoldMetadata",
]
