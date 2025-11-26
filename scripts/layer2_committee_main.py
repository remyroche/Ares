"""CLI entrypoint for Layer 2 committee training.

This script wires together:
- Calendar-based retrain plan (Layer 1/2/3 scheduler)
- Layer 2 committee training utilities
- A small JSON config file under config/layer2_committee_training.json

Usage (from repo root):
    python -m scripts.layer2_committee_main \
        --config config/layer2_committee_training.json

Flags:
    --force-layer2   Ignore scheduler and always retrain Layer 2.
    --dry-run        Print the retrain plan and exit without training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.logger import system_logger
from src.training.steps.models_training.layer_committee_scheduler import (
    get_retrain_plan,
)
from src.training.utils.layer2_training import (
    Layer2TrainingConfig,
    train_layer2_committee,
)


logger = system_logger.getChild("Layer2CommitteeCLI")


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_labels(cfg: Dict[str, Any], repo_root: Path) -> pd.DataFrame:
    labels_cfg = cfg.get("labels") or {}
    rel_path = labels_cfg.get("path")
    if not rel_path:
        raise ValueError("Config must define labels.path")

    labels_path = (repo_root / rel_path).resolve()
    fmt = (labels_cfg.get("format") or "parquet").lower()
    target_col = labels_cfg.get("target_column") or "target"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    if fmt == "parquet":
        df = pd.read_parquet(labels_path)
    elif fmt == "csv":
        df = pd.read_csv(labels_path)
    else:
        raise ValueError(f"Unsupported labels format: {fmt}")

    if target_col not in df.columns:
        raise ValueError(
            f"Labels file {labels_path} does not contain target column '{target_col}'"
        )

    # Ensure a DatetimeIndex if there is a timestamp column
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = None
        for candidate in ("timestamp", "time", "datetime"):
            if candidate in df.columns:
                ts_col = candidate
                break
        if ts_col is not None:
            df.index = pd.to_datetime(df[ts_col])
        else:
            logger.warning(
                "Labels DataFrame has no explicit timestamp column; using existing index"
            )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "Labels DataFrame must have a DatetimeIndex or a timestamp/time/datetime column"
        )

    return df[[target_col]].rename(columns={target_col: "target"})


def _build_training_config(cfg: Dict[str, Any]) -> Layer2TrainingConfig:
    return Layer2TrainingConfig(
        symbol=cfg.get("symbol", "ETHUSDT"),
        exchange=cfg.get("exchange", "binance"),
        timeframe=cfg.get("timeframe", "15m"),
        direction=cfg.get("direction", "long"),
        base_layer_name=cfg.get("base_layer_name", "specialists"),
        base_model_version=cfg.get("base_model_version"),
        committee_layer_name=cfg.get("committee_layer_name", "meta_layer"),
        committee_model_version=cfg.get("committee_model_version", "v1"),
        history_mode=cfg.get("history_mode", "full"),
        last_n_days=cfg.get("last_n_days"),
        warm_start=bool(cfg.get("warm_start", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 2 committee training CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="config/layer2_committee_training.json",
        help="Path to Layer 2 committee training config JSON",
    )
    parser.add_argument(
        "--force-layer2",
        action="store_true",
        help="Ignore scheduler and always retrain Layer 2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print retrain plan and exit without training",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()

    cfg_dict = _load_config(config_path)

    # 1) Get calendar-based retrain plan
    plan = get_retrain_plan()
    logger.info("Retrain plan: %s", plan)

    if args.dry_run:
        return

    if not args.force_layer2 and not plan.get("layer2", False):
        logger.info("Layer 2 retrain not scheduled today and --force-layer2 not set; exiting")
        return

    # 2) Load labels
    labels_df = _load_labels(cfg_dict, repo_root)

    # 3) Build training config and run training
    train_cfg = _build_training_config(cfg_dict)
    result = train_layer2_committee(train_cfg, labels_df)

    logger.info("Layer 2 training result: %s", result)


if __name__ == "__main__":  # pragma: no cover
    main()
