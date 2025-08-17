# src/training/steps/step3_hmm_regime_discovery_validator.py

import os
import pandas as pd
from typing import Any

from src.utils.logger import system_logger


def _exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def run_validator(
    training_input: dict[str, Any], pipeline_state: dict[str, Any]
) -> dict[str, Any]:
    logger = system_logger.getChild("Step3.Validator")
    logger.info("🔍 Starting Step 3 validation...")

    symbol = training_input.get("symbol", "ETHUSDT")
    exchange = training_input.get("exchange", "BINANCE")
    data_dir = training_input.get("data_dir", "data/training")

    logger.info(
        f"📋 Validation parameters: symbol={symbol}, exchange={exchange}, data_dir={data_dir}"
    )

    timeframes = ["1m", "5m", "15m", "30m"]
    found_any = False
    messages: list[str] = []
    validation_details = {
        "timeframes_checked": len(timeframes),
        "timeframes_passed": 0,
        "timeframes_failed": 0,
        "total_files_checked": 0,
        "total_files_found": 0,
        "file_sizes": {},
        "data_quality_checks": {},
    }

    for tf_idx, tf in enumerate(timeframes, 1):
        logger.info(f"🔄 Validating timeframe {tf_idx}/{len(timeframes)}: {tf}")

        block_path = os.path.join(
            data_dir, f"{exchange}_{symbol}_hmm_block_states_{tf}.parquet"
        )
        composite_path = os.path.join(
            data_dir, f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet"
        )
        intensity_path = os.path.join(
            data_dir, f"{exchange}_{symbol}_hmm_composite_intensity_{tf}.parquet"
        )
        meta_path = os.path.join(
            data_dir, f"{exchange}_{symbol}_hmm_composite_meta_{tf}.json"
        )

        validation_details["total_files_checked"] += 4

        # Use centralized HMM composite manager
        try:
            from src.utils.hmm_composite_manager import get_hmm_composite_manager

            hmm_manager = get_hmm_composite_manager()
            comp_df = hmm_manager.load_composite_clusters(
                exchange, symbol, tf, data_dir
            )
        except Exception as e:
            logger.error(f"❌ Failed to load HMM composite manager for {tf}: {e}")
            messages.append(f"ERROR {tf}: HMM manager failed - {e}")
            validation_details["timeframes_failed"] += 1
            continue

        tf_validation_passed = False
        tf_files_found = 0

        if _exists(block_path) and comp_df is not None:
            try:
                # Validate block states file
                df_b = pd.read_parquet(block_path)
                block_cols_ok = any(c.endswith("_state_id") for c in df_b.columns)
                block_data_ok = len(df_b) > 0 and not df_b.isnull().all().all()

                # Validate composite clusters
                df_c = comp_df
                composite_cols_ok = all(
                    c in df_c.columns
                    for c in ["combination_id", "composite_cluster_id"]
                )
                composite_data_ok = len(df_c) > 0 and not df_c.isnull().all().all()

                # Check file sizes
                block_size = os.path.getsize(block_path) if _exists(block_path) else 0
                validation_details["file_sizes"][f"block_{tf}"] = block_size

                # Data quality checks
                validation_details["data_quality_checks"][tf] = {
                    "block_rows": len(df_b),
                    "block_columns": len(df_b.columns),
                    "composite_rows": len(df_c),
                    "composite_columns": len(df_c.columns),
                    "block_has_states": block_cols_ok,
                    "composite_has_required_cols": composite_cols_ok,
                    "block_data_valid": block_data_ok,
                    "composite_data_valid": composite_data_ok,
                }

                if (
                    block_cols_ok
                    and composite_cols_ok
                    and block_data_ok
                    and composite_data_ok
                ):
                    found_any = True
                    tf_validation_passed = True
                    tf_files_found = 2  # block + composite
                    validation_details["total_files_found"] += tf_files_found
                    validation_details["timeframes_passed"] += 1
                    messages.append(
                        f"✅ {tf}: artifacts present, rows={len(df_b)}, clusters={len(df_c)}"
                    )
                    logger.info(f"✅ Timeframe {tf} validation passed")
                else:
                    validation_details["timeframes_failed"] += 1
                    issues = []
                    if not block_cols_ok:
                        issues.append("missing state columns")
                    if not composite_cols_ok:
                        issues.append("missing composite columns")
                    if not block_data_ok:
                        issues.append("invalid block data")
                    if not composite_data_ok:
                        issues.append("invalid composite data")
                    messages.append(f"❌ {tf}: validation failed - {', '.join(issues)}")
                    logger.warning(
                        f"❌ Timeframe {tf} validation failed: {', '.join(issues)}"
                    )
            except Exception as e:
                validation_details["timeframes_failed"] += 1
                messages.append(f"❌ {tf}: failed to read outputs - {e}")
                logger.error(f"❌ Timeframe {tf} validation error: {e}")
        else:
            validation_details["timeframes_failed"] += 1
            missing_files = []
            if not _exists(block_path):
                missing_files.append("block_states")
            if comp_df is None:
                missing_files.append("composite_clusters")
            messages.append(f"❌ {tf}: missing artifacts - {', '.join(missing_files)}")
            logger.warning(
                f"❌ Timeframe {tf} missing artifacts: {', '.join(missing_files)}"
            )

    # Final validation summary
    passed = found_any
    status = {
        "validation_passed": bool(passed),
        "messages": messages,
        "validation_details": validation_details,
        "summary": {
            "total_timeframes": len(timeframes),
            "passed_timeframes": validation_details["timeframes_passed"],
            "failed_timeframes": validation_details["timeframes_failed"],
            "success_rate": validation_details["timeframes_passed"] / len(timeframes)
            if timeframes
            else 0,
            "files_found": validation_details["total_files_found"],
            "files_checked": validation_details["total_files_checked"],
        },
    }

    if passed:
        logger.info("✅ Step 1_7 validation passed for at least one timeframe")
        logger.info(
            f"📊 Validation summary: {validation_details['timeframes_passed']}/{len(timeframes)} timeframes passed"
        )
    else:
        logger.error("❌ Step 1_7 validation failed - no valid artifacts found")
        logger.error(
            f"📊 Validation summary: {validation_details['timeframes_failed']}/{len(timeframes)} timeframes failed"
        )

    return status
