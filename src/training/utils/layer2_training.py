"""Layer 2/3 committee training utilities.

This module wires together:
- Hive-partitioned base-layer predictions
- OHLCV data from KlinesParquetManager
- Layer2FeatureBuilder for feature construction
- Layer2Committee for model training and persistence

It provides a minimal training entrypoint that supports:
- Full-history retrain (from first available prediction to latest)
- Warm-start retrain (start from previous training end when metadata exists)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import pandas as pd

from src.utils.logger import system_logger
from src.utils.hive_partitioned_predictions import HivePartitionedReader
from src.utils.hive_partitioned_predictions.constants import LAYER_PATHS, MODELS_DIR
from src.training.utils.layer2_feature_builder import (
    Layer2FeatureBuilder,
    Layer2FeatureBuilderConfig,
)
from src.training.utils.layer2_committee import (
    Layer2Committee,
    Layer2CommitteeConfig,
)


logger = system_logger.getChild("Layer2Training")


@dataclass
class Layer2TrainingConfig:
    """Configuration for Layer 2/3 committee training.

    Notes:
    - `base_layer_name` controls which Hive layer we read predictions from
      (e.g., "specialists", "base_models", "meta_layer").
    - `committee_layer_name` controls where we persist the committee models
      under the Hive artifact hierarchy.
    - Labels are passed in by the caller as a DataFrame indexed by timestamp
      with a `target` column.
    """

    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    direction: str = "long"

    # Source of prediction features
    base_layer_name: str = "specialists"
    base_model_version: Optional[str] = None  # None -> latest

    # Where to store committee models
    committee_layer_name: str = "meta_layer"
    committee_model_version: str = "v1"

    # History control
    history_mode: str = "full"  # "full" or "from_last_n_days"
    last_n_days: Optional[int] = None

    # Warm-start: adjust training window start from previous metadata if available
    warm_start: bool = False

    # Optional overrides
    models_base_path: Optional[Path] = None
    committee_config: Optional[Layer2CommitteeConfig] = None


def _resolve_models_dir(cfg: Layer2TrainingConfig) -> Path:
    """Resolve directory where committee models + metadata are stored."""
    if cfg.models_base_path is not None:
        base = Path(cfg.models_base_path)
    else:
        if cfg.committee_layer_name not in LAYER_PATHS:
            raise ValueError(
                f"Unknown committee_layer_name={cfg.committee_layer_name}. "
                f"Available: {list(LAYER_PATHS.keys())}"
            )
        base = LAYER_PATHS[cfg.committee_layer_name] / MODELS_DIR

    # Ensure artifacts are always segmented per exchange, symbol, timeframe
    # and direction to avoid cross-contamination between markets.
    base = (
        base
        / f"exchange={cfg.exchange}"
        / f"symbol={cfg.symbol}"
        / f"timeframe={cfg.timeframe}"
        / f"direction={cfg.direction}"
    )

    return base / f"model_version={cfg.committee_model_version}"


def _load_previous_metadata(models_dir: Path) -> Optional[Dict[str, Any]]:
    """Load previous training metadata if present."""
    meta_path = models_dir / "training_metadata.json"
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load training metadata from {meta_path}: {exc}")
        return None


def _save_metadata(models_dir: Path, metadata: Dict[str, Any]) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    meta_path = models_dir / "training_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def _resolve_prediction_window(cfg: Layer2TrainingConfig) -> Tuple[datetime, datetime]:
    """Determine the raw prediction window from Hive storage."""
    reader = HivePartitionedReader(layer_name=cfg.base_layer_name)

    if cfg.history_mode == "from_last_n_days" and cfg.last_n_days is not None:
        end = datetime.now()
        start = end - timedelta(days=cfg.last_n_days)
        logger.info(
            "[Layer2Training] Using last_n_days window: %s to %s",
            start,
            end,
        )
        return start, end

    # Default: full history from predictions
    start, end = reader.get_date_range(
        model_version=cfg.base_model_version,
    )
    logger.info(
        "[Layer2Training] Using full prediction history: %s to %s",
        start,
        end,
    )
    return start, end


def _apply_warm_start(
    cfg: Layer2TrainingConfig,
    models_dir: Path,
    window: Tuple[datetime, datetime],
) -> Tuple[datetime, datetime]:
    """Adjust training window for warm-start if previous metadata exists.

    Warm-start semantics here are *data-window only*: if metadata exists and
    `warm_start=True`, the new training window's start is moved forward to the
    previous training end (when earlier than the raw window end).
    """
    if not cfg.warm_start:
        return window

    metadata = _load_previous_metadata(models_dir)
    if not metadata:
        logger.info("[Layer2Training] No previous metadata found; falling back to full window")
        return window

    prev_end_str = metadata.get("train_end")
    if not prev_end_str:
        return window

    try:
        prev_end = datetime.fromisoformat(prev_end_str)
    except Exception as exc:
        logger.warning(f"[Layer2Training] Could not parse previous train_end '{prev_end_str}': {exc}")
        return window

    start, end = window
    if prev_end >= end:
        logger.info(
            "[Layer2Training] Previous training already covers up to %s; "
            "no new data to train on",
            prev_end,
        )
        return window

    new_start = max(prev_end, start)
    logger.info(
        "[Layer2Training] Warm-start window adjusted from (%s, %s) to (%s, %s)",
        start,
        end,
        new_start,
        end,
    )
    return new_start, end


def train_layer2_committee(
    cfg: Layer2TrainingConfig,
    labels: pd.DataFrame,
) -> Dict[str, Any]:
    """Train a Layer 2/3 committee on the given labels.

    Args:
        cfg: Training configuration.
        labels: DataFrame indexed by timestamp with at least a `target` column.

    Returns:
        Dictionary with training summary and model path.
    """
    if "target" not in labels.columns:
        raise ValueError("labels DataFrame must contain a 'target' column")

    models_dir = _resolve_models_dir(cfg)

    # 1) Resolve prediction-based window and apply warm-start policy
    raw_window = _resolve_prediction_window(cfg)
    train_start, train_end = _apply_warm_start(cfg, models_dir, raw_window)

    # 2) Slice labels to training window and ensure datetime index
    if not isinstance(labels.index, pd.DatetimeIndex):
        labels = labels.copy()
        labels.index = pd.to_datetime(labels.index)

    labels_window = labels[(labels.index >= train_start) & (labels.index <= train_end)].copy()
    if labels_window.empty:
        raise ValueError(
            f"No labels available for training window {train_start} to {train_end}"
        )

    # 3) Build features
    fb_cfg = Layer2FeatureBuilderConfig(
        symbol=cfg.symbol,
        exchange=cfg.exchange,
        timeframe=cfg.timeframe,
        base_layer_name=cfg.base_layer_name,
    )
    builder = Layer2FeatureBuilder(fb_cfg)
    X, y = builder.build_features(train_start, train_end, labels_window)

    # 4) Train committee
    committee_cfg = cfg.committee_config or Layer2CommitteeConfig()
    committee = Layer2Committee(committee_cfg)
    committee.fit(X, y)

    # 5) Persist models + metadata
    committee.save(models_dir)

    metadata: Dict[str, Any] = {
        "symbol": cfg.symbol,
        "exchange": cfg.exchange,
        "timeframe": cfg.timeframe,
        "direction": cfg.direction,
        "base_layer_name": cfg.base_layer_name,
        "base_model_version": cfg.base_model_version,
        "committee_layer_name": cfg.committee_layer_name,
        "committee_model_version": cfg.committee_model_version,
        "history_mode": cfg.history_mode,
        "last_n_days": cfg.last_n_days,
        "warm_start": cfg.warm_start,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "n_samples": int(len(X)),
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_metadata(models_dir, metadata)

    logger.info(
        "[Layer2Training] Trained committee %s on %d samples (%s to %s)",
        cfg.committee_model_version,
        len(X),
        train_start,
        train_end,
    )

    return {
        "success": True,
        "models_dir": str(models_dir),
        "train_start": train_start,
        "train_end": train_end,
        "n_samples": int(len(X)),
        "metadata_path": str(models_dir / "training_metadata.json"),
    }
