#!/usr/bin/env python3
"""
CLI to run Step03 debug suite and save a JSON report.

Usage:
  python scripts/run_step03_debug.py --symbol ETHUSDT --exchange BINANCE --timeframe 1m --data-dir data_cache --smoke
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from src.utils.logger import system_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description = 'Run Step03 debug suite')
    p.add_argument('--symbol', default = 'ETHUSDT')
    p.add_argument('--exchange', default = 'BINANCE')
    p.add_argument('--timeframe', default = '1m')
    p.add_argument('--data-dir', default = None)
    p.add_argument('--output-dir', default = 'results')
    p.add_argument('--smoke', action = 'store_true', help = 'Run a quick smoke test')
    p.add_argument('--timeout', type = float, default = 30.0, help = 'Smoke test timeout seconds')
    return p.parse_args()


async def _run(symbol: str, exchange: str, timeframe: str, data_dir: Optional[str], output_dir: str, smoke: bool, timeout: float) -> None:
    from src.training.steps.market_analysis.hmm_clustering.step03_debug_tools import run_debug_suite, save_report

    report = await run_debug_suite(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, smoke_test = smoke, smoke_timeout_seconds = timeout)
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    out_file = Path(output_dir) / f'step03_debug_report_{symbol}_{timeframe}.json'
    save_path = save_report(report, out_file)
    system_logger.info(f'📄 Step03 debug report saved to: {save_path}')


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args.symbol, args.exchange, args.timeframe, args.data_dir, args.output_dir, args.smoke, args.timeout))


if __name__ == '__main__':
    main()

