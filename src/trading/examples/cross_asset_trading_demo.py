from typing import Any, Dict
import asyncio

from ..cross_asset.cross_asset_trading_manager import start_cross_asset_trading

async def main():
    symbols = ["ETH", "BTC", "SOL", "ADA"]
    manager = await start_cross_asset_trading(
        symbols=symbols,
        trading_mode="paper",
        account_balance=10_000.0,
        orchestrator_base_config={
            "analyst_signals": {"confidence_threshold": 0.6},
            "tactician_signals": {"confidence_threshold": 0.6},
            "signal_combiner": {},
        },
    )

    try:
        for _ in range(5):
            stats = manager.get_manager_stats()
            print({"gate": stats.get("gate"), "symbols": stats.get("symbols")})
            await asyncio.sleep(5)

        report = await manager.generate_consolidated_report()
        print({"portfolio": report.get("portfolio", {})})

    finally:
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
