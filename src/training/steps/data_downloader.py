"""Data downloader adapter for training steps.

This module provides a unified interface for downloading data
using either the optimized or clean downloader.
"""

from __future__ import annotations

from typing import Any

from src.config import CONFIG
from src.core.decorators import handles_errors
from src.utils.logger import system_logger


@handles_errors(fallback=False)
async def download_all_data_with_consolidation(
    symbol: str,
    exchange_name: str,
    interval: str = "1m",
    data_dir: str = None,
) -> bool:
    """Unified entrypoint used by training steps to download raw data.

    Tries the optimized downloader first, falls back to the clean downloader.
    Returns True on success, False otherwise.
    """
    logger = system_logger.getChild("DataDownloaderAdapter")

    lookback_years: int = 2
    try:
        if isinstance(CONFIG, dict):
            model_training_cfg: dict[str, Any] | None = CONFIG.get("MODEL_TRAINING")  # type: ignore[assignment]
            if model_training_cfg and isinstance(
                model_training_cfg.get("lookback_years"),
                int,
            ):
                lookback_years = int(model_training_cfg["lookback_years"])  # type: ignore[arg-type]
    except Exception:
        # Keep default lookback_years
        pass

    # Preferred path: optimized downloader
    try:
        from backtesting.ares_data_downloader_optimized import (
            DownloadConfig as OptimizedDownloadConfig,
        )
        from backtesting.ares_data_downloader_optimized import (
            OptimizedDataDownloader,
        )

        opt_cfg = OptimizedDownloadConfig(
            symbol=symbol,
            exchange=exchange_name,
            interval=interval,
            lookback_years=lookback_years,
            data_dir=data_dir,
        )
        optimized = OptimizedDataDownloader(opt_cfg)
        return await optimized.run_optimized_download()
    except Exception as e:
        logger.warning(
            f"Optimized downloader unavailable or failed, falling back to clean downloader: {e}",
        )

    # Fallback: clean downloader
    try:
        from backtesting.ares_data_downloader_clean import (
            CleanDataDownloader,
        )
        from backtesting.ares_data_downloader_clean import (
            DownloadConfig as CleanDownloadConfig,
        )

        clean_cfg = CleanDownloadConfig(
            symbol=symbol,
            exchange=exchange_name,
            interval=interval,
            lookback_years=lookback_years,
            data_dir=data_dir,
        )
        clean = CleanDataDownloader(clean_cfg)
        return await clean.run_clean_download()
    except Exception as e:
        logger.exception(f"All downloader backends failed: {e}")
        return False
