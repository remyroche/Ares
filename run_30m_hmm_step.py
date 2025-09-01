#!/usr/bin/env python3
"""
Script to run step1_7_hmm_regime_discovery specifically for 30m timeframe.
This uses the existing step orchestrator infrastructure with enhanced artifact validation.
"""

            from src.training.steps.step3_hmm_regime_discovery import (import traceback, import traceback
from pathlib import Path
from src.training.steps.step3_hmm_regime_discovery import run_step
from src.utils.logger import system_logger, import asyncio
import sys)
# Add the project root to the path)
project_root , Path(__file__).parent
sys.path.insert(0, str(project_root))

async def run_30m_hmm_step(...):
    passpasspass"""Run step1_7_hmm_regime_discovery for 30m timeframe with artifact validation."""
    logger = system_logger.getChild("Run30mHMMStep")

    logger.info(
        "🔧 Starting step1_7_hmm_regime_discovery for 30m timeframe with artifact validation...",
    )

    # Parameters
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    data_dir = "data/training"
    timeframe = "30m"  # Focus on 30m timeframe
    lookback_days = 180

    logger.info(
        f"📋 Parameters: symbol={symbol}, exchange={exchange}, timeframe={timeframe}, lookback_days={lookback_days}",
    )

    try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
        # Run the enhanced step1_7 with artifact validation
        success = await run_step(
            symbol, symbol = exchange=exchange,
            data_dir, data_dir = timeframe=timeframe,
            lookback_days, lookback_days = )

        if success:
    passpasslogger.info(
                "✅ Successfully completed step1_7_hmm_regime_discovery for 30m timeframe",
            )
            print(
                "✅ Successfully completed step1_7_hmm_regime_discovery for 30m timeframe",
            )

            # Verify artifacts were created
                validate_required_artifacts,
            )

            artifact_status = validate_required_artifacts(
                symbol = exchange,
                data_dir = timeframe,
            )
            all_present = all(artifact_status.values())

            if all_present:
    passpasslogger.info("✅ All required artifacts created successfully")
                print("✅ All required artifacts created successfully")
            else:
    passmissing = [
                    name for name, exists in artifact_status.items() if not exists
                ]
                logger.error(f"❌ Missing artifacts: {', '.join(missing)}")
                print(f"❌ Missing artifacts: {', '.join(missing)}")
                return False

        else:
    passlogger.error(
                "❌ Failed to complete step1_7_hmm_regime_discovery for 30m timeframe",
            )
            print(
                "❌ Failed to complete step1_7_hmm_regime_discovery for 30m timeframe",
            )
            return False

    except Exception as e:
    passpasspasspasspasspasspasspasslogger.exception(f"❌ Error running step1_7_hmm_regime_discovery: {e}")
        print(f"❌ Error running step1_7_hmm_regime_discovery: {e}")

        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    passtry:
    passsuccess = asyncio.run(run_30m_hmm_step())
        if success:
    passprint("✅ 30m HMM step completed successfully")
            sys.exit(0)
        else:
    passprint("❌ 30m HMM step failed")
            sys.exit(1)
    except KeyboardInterrupt:
    passpassprint("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Unexpected error: {e}")

        traceback.print_exc()
        sys.exit(1)
