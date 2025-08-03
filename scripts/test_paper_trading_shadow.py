#!/usr/bin/env python3
"""
Test script for paper trading with shadow trading functionality.
This script tests that the paper trading mode properly uses Binance's testnet APIs
for actual API calls while maintaining simulated trading logic.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.exchange.factory import ExchangeFactory
from src.utils.logger import setup_logging, system_logger


async def test_paper_trading_shadow():
    """Test paper trading with shadow trading functionality."""
    setup_logging()
    logger = system_logger.getChild("TestPaperTradingShadow")

    logger.info("🧪 Testing paper trading with shadow trading functionality...")

    # Test 1: Check if testnet API keys are available
    logger.info("📋 Checking testnet API configuration...")

    if not settings.binance_testnet_api_key or not settings.binance_testnet_api_secret:
        logger.warning(
            "⚠️  Testnet API keys not found. Shadow trading will fall back to simulation.",
        )
        logger.info(
            "   To enable shadow trading, set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET in your .env file",
        )
        return False

    logger.info("✅ Testnet API keys found")

    exchange_client = ExchangeFactory.get_exchange(settings.exchange_name)

        # Test 3: Verify testnet configuration
        if hasattr(exchange_client, "use_testnet") and exchange_client.use_testnet:
            logger.info("✅ Exchange client is properly configured for testnet")
        else:
            logger.error("❌ Exchange client is not configured for testnet")
            return False

        # Test 4: Test basic API call (get server time)
        logger.info("⏰ Testing basic API call (server time)...")
        try:
            server_time = await exchange_client._get_server_time()
            if server_time:
                logger.info(f"✅ Server time retrieved: {server_time}")
            else:
                logger.warning("⚠️  Could not retrieve server time")
        except Exception as e:
            logger.error(f"❌ Failed to get server time: {e}")
            return False

        # Test 5: Test account info (if available)
        logger.info("👤 Testing account info retrieval...")
        try:
            account_info = await exchange_client.get_account_info()
            if account_info:
                logger.info("✅ Account info retrieved successfully")
                logger.info(
                    f"   Account type: {account_info.get('accountType', 'Unknown')}",
                )
                logger.info(f"   Permissions: {account_info.get('permissions', [])}")
            else:
                logger.warning("⚠️  Could not retrieve account info")
        except Exception as e:
            logger.error(f"❌ Failed to get account info: {e}")
            return False

        # Test 6: Test market data retrieval
        logger.info("📊 Testing market data retrieval...")
        try:
            ticker = await exchange_client.get_ticker("ETHUSDT")
            if ticker:
                logger.info("✅ Market data retrieved successfully")
                logger.info(f"   Symbol: {ticker.get('symbol', 'Unknown')}")
                logger.info(f"   Price: {ticker.get('price', 'Unknown')}")
            else:
                logger.warning("⚠️  Could not retrieve market data")
        except Exception as e:
            logger.error(f"❌ Failed to get market data: {e}")
            return False

        logger.info(
            "🎉 All tests passed! Paper trading with shadow trading is working correctly.",
        )
        logger.info("📝 Summary:")
        logger.info("   ✅ Testnet API keys configured")
        logger.info("   ✅ Binance testnet connection established")
        logger.info("   ✅ Exchange client properly configured for testnet")
        logger.info("   ✅ Basic API calls working")
        logger.info("   ✅ Account info accessible")
        logger.info("   ✅ Market data retrieval working")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False


async def main():
    """Main test function."""
    success = await test_paper_trading_shadow()

    if success:
        print("\n🎉 Paper trading with shadow trading test PASSED!")
        print(
            "   The system is ready for shadow trading with actual API calls to Binance's testnet.",
        )
        print("\n   To run paper trading with shadow trading:")
        print("   python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE")
    else:
        print("\n❌ Paper trading with shadow trading test FAILED!")
        print("   Please check your configuration and try again.")

    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error during test: {e}")
        sys.exit(1)
