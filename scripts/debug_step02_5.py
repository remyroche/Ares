#!/usr/bin/env python3
"""
CLI: Thoroughly debug Step 2.5 (SR Optimization)

- Instruments the step with debug wrappers (no code changes to the step itself)
- Captures timings, function calls, exceptions, memory/CPU snapshots
- Saves a JSON debug report under src/training/reports/step02_5_debug/

Examples:
  python scripts/debug_step02_5.py --rows 20000
  python scripts/debug_step02_5.py --data data_cache/ethusdt_1m.parquet
  STEP025_MEMORY=1 STEP025_CPU=1 python scripts/debug_step02_5.py --rows 5000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import system_logger, log_dataframe_overview
from src.utils.step02_5_debug_tools import DebugConfig, DebugTracker, instrument_sr_step, summarize_result_for_console


def _create_synthetic_ohlcv(rows: int = 5000):
    import numpy as np
    import pandas as pd

    system_logger.info(f'🔧 Creating synthetic OHLCV with rows={rows} ...')
    rng = np.random.default_rng(42)
    base_price = 20000.0
    returns = rng.normal(loc=0.0, scale=0.001, size=rows)
    prices = base_price * (1 + returns).cumprod()

    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=rows, freq='1min'),
        'open': prices * (1 + rng.normal(0, 2e-4, size=rows)),
        'high': prices * (1 + abs(rng.normal(0, 5e-4, size=rows))),
        'low': prices * (1 - abs(rng.normal(0, 5e-4, size=rows))),
        'close': prices,
        'volume': rng.integers(1_000, 200_000, size=rows),
    })
    # Ensure OHLC is consistent
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    return df


def _load_dataframe(path: Path):
    import pandas as pd
    if not path.exists():
        raise FileNotFoundError(f'Data file not found: {path}')
    if path.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {'.csv'}:
        return pd.read_csv(path)
    raise ValueError(f'Unsupported data file format: {path.suffix}')


async def _run(args: argparse.Namespace) -> int:
    # Lazy import to avoid import-time overhead if not needed
    from src.training.steps.data_collection.data_preparation.step02_5_sr_optimization import SROptimizationStep

    config: Dict[str, Any] = {
        'sr_optimization': {
            'min_touches': 2,
            'tolerance_pct': 0.5,
            'lookback_periods': 100,
            'proximity_threshold': 0.002,
        }
    }

    # Prepare data
    if args.data is not None:
        df = _load_dataframe(Path(args.data))
    else:
        df = _create_synthetic_ohlcv(rows=args.rows)

    log_dataframe_overview(system_logger, df, name='debug_input_df')

    training_input: Dict[str, Any] = {
        'symbol': args.symbol,
        'exchange': args.exchange,
        'timeframe': args.timeframe,
        'data_dir': 'data_cache',
    }
    pipeline_state: Dict[str, Any] = {
        'dataframe': df,
        'data_info': {'symbol': args.symbol, 'exchange': args.exchange, 'timeframe': args.timeframe},
    }

    debug_config = DebugConfig.from_env()
    tracker = DebugTracker(debug_config, logger=system_logger)
    tracker.start_session(context={
        'symbol': args.symbol,
        'exchange': args.exchange,
        'timeframe': args.timeframe,
        'data_rows': int(getattr(df, 'shape', (0, 0))[0] or 0),
    })

    try:
        with tracker.section('instantiate_step'):
            step = SROptimizationStep(config)

        # Instrument step
        with tracker.section('instrumentation'):
            instrument_sr_step(step, tracker)

        with tracker.section('initialize'):
            await step.initialize()

        with tracker.section('execute'):
            result = await step.execute(training_input, pipeline_state)

        # Save report
        report_path = tracker.save_report()
        summary = summarize_result_for_console(result)
        system_logger.info(f'✅ Step02_5 debug completed | {summary}')
        print(f'JSON report: {report_path}')
        return 0

    except Exception as e:
        tracker.record_exception(e, context={'phase': 'debug_session'})
        traceback.print_exc()
        report_path = tracker.save_report()
        print(f'❌ Debug run failed. Report saved at: {report_path}')
        return 2
    finally:
        tracker.end_session()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Thoroughly debug Step 2.5 (SR Optimization)')
    p.add_argument('--symbol', default='ETHUSDT', help='Trading symbol')
    p.add_argument('--exchange', default='BINANCE', help='Exchange name')
    p.add_argument('--timeframe', default='1m', help='Timeframe')
    p.add_argument('--data', default=None, help='Path to CSV/Parquet OHLCV data')
    p.add_argument('--rows', type=int, default=10000, help='Synthetic rows if --data not provided')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    return asyncio.run(_run(args))


if __name__ == '__main__':
    raise SystemExit(main())

