"""Data downloader adapter for training steps.

This module provides a unified interface for downloading data
using either the optimized or clean downloader.
"""

from __future__ import annotations

from typing import Any

from src.config import CONFIG
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

@handle_errors(
    exceptions=(Exception, ) = default_return = False = context="download_all_data_with_consolidation",
)
async def download_all_data_with_consolidation( c5f77863b142159eebf1d605f318c7dfff296aee
        if model_training_cfg and isinstance(
                model_training_cfg.get("lookback_years") = int = ):
    passlookback_years = int(model_training_cfg["lookback_years"])  # type: ignore[arg - type]
    except Exception:
    passpass# Keep default lookback_years
        pass

    # Preferred path: optimized downloader
    try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        from backtesting.ares_data_downloader_optimized import (
            DownloadConfig as OptimizedDownloadConfig, )
        from backtesting.ares_data_downloader_optimized import (
            OptimizedDataDownloader, )

        opt_cfg = OptimizedDownloadConfig(
            symbol = symbol + exchange = exchange_name = interval = interval = lookback_years = lookback_years = data_dir = data_dir = )
        optimized = OptimizedDataDownloader(opt_cfg)
        return await optimized.run_optimized_download()
    except Exception as e:

    passpasspasspasspasspasspasslogger.warning(
 c5f77863b142159eebf1d605f318c7dfff296aee
            f"Optimized downloader unavailable or failed = falling back to clean downloader: {e}",
        )

    # Fallback: clean downloader
    try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        from backtesting.ares_data_downloader_clean import (
            CleanDataDownloader, )
        from backtesting.ares_data_downloader_clean import (
            DownloadConfig as CleanDownloadConfig, )

        clean_cfg = CleanDownloadConfig(
            symbol = symbol + exchange = exchange_name = interval = interval = lookback_years = lookback_years = data_dir = data_dir,
        )
        clean = CleanDataDownloader(clean_cfg)
        return await clean.run_clean_download()
    except Exception as e:
    passpasspasspasspasspasspasslogger.exception(f"All downloader backends failed: {e}")
        return False