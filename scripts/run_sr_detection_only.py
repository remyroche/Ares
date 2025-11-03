#!/usr/bin/env python3
"""
Simplified SR Detection Runner - Skips Parameter Optimization

This script runs ONLY the SR detection and clustering steps, using
default/pre-optimized parameters for faster execution and report generation.

Usage:
    python scripts/run_sr_detection_only.py --symbol ETHUSDT --exchange binance --timeframe 1h --lookback-days 2
"""

import asyncio
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector
from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent
from src.utils.data.real_data_loader import RealDataLoader


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run SR detection and clustering (no parameter optimization)")
    parser.add_argument('--symbol', type=str, default='ETHUSDT')
    parser.add_argument('--exchange', type=str, default='binance')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--lookback-days', type=int, default=2)
    parser.add_argument('--mode', type=str, default='light', choices=['light', 'full'])
    
    args = parser.parse_args()
    
    logger = system_logger.getChild('SRDetectionOnly')
    
    # Create outcomes directory
    outcomes_dir = Path('outcomes') / f"sr_detection_only_{args.symbol}_{args.timeframe}"
    outcomes_dir.mkdir(parents=True, exist_ok=True)
    datetime_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logger.info("=" * 80)
    logger.info("🚀 SR DETECTION & CLUSTERING (Fast Mode - No Parameter Optimization)")
    logger.info(f"   Symbol: {args.symbol}")
    logger.info(f"   Timeframe: {args.timeframe}")
    logger.info(f"   Lookback: {args.lookback_days} days")
    logger.info(f"   Mode: {args.mode}")
    logger.info("=" * 80)
    
    try:
        # Initialize detector with optimized config
        detector_config = {
            'min_touches': 2,
            'touch_proximity_threshold': 0.005,
            'min_strength': 0.15,
            'use_ml_model': True,
            'ml_model_path': 'models/sr_quality_model.lgb',
            
            # LIGHT mode optimizations
            'max_levels_per_method': 20,
            'max_fractal_levels': 20,
            'fractal_period': 5 if args.mode == 'light' else 3,
            'pivot_period': 5 if args.mode == 'light' else 4,
            'psychological_levels': (args.mode == 'full'),
            'fibonacci_levels': (args.mode == 'full'),
            'enable_fractal_caching': True,
        }
        
        detector = EnhancedSRDetector(detector_config)
        data_loader = RealDataLoader()
        
        # Load market data
        logger.info(f"\n📊 Loading market data for {args.symbol}...")
        market_data = await data_loader.load_market_data(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            start_date=None,
            end_date=None
        )
        
        if market_data is None or market_data.empty:
            logger.error("❌ Failed to load market data")
            return 1
        
        logger.info(f"✅ Loaded {len(market_data)} data points")
        
        # Detect SR levels
        logger.info("\n🔍 Detecting SR levels with ML model...")
        detected_levels = detector.detect_sr_levels(market_data)
        
        support_levels = [l for l in detected_levels if l.level_type == 'support']
        resistance_levels = [l for l in detected_levels if l.level_type == 'resistance']
        
        logger.info(f"✅ Detected {len(detected_levels)} SR levels")
        logger.info(f"   - Support: {len(support_levels)}")
        logger.info(f"   - Resistance: {len(resistance_levels)}")
        
        # Save detection results
        detection_result = {
            'total_levels': len(detected_levels),
            'support_levels': len(support_levels),
            'resistance_levels': len(resistance_levels),
            'levels': [
                {
                    'price': level.price,
                    'type': level.level_type,
                    'strength': level.strength,
                    'touches': level.touches,
                    'method': level.method,
                    'quality_score': getattr(level, 'quality_score', level.strength)
                } for level in detected_levels
            ],
            'metadata': {
                'symbol': args.symbol,
                'timeframe': args.timeframe,
                'data_points': len(market_data),
                'ml_model_used': True,
                'mode': args.mode
            }
        }
        
        # Save JSON report
        json_path = outcomes_dir / f"sr_detection_{args.symbol}_{args.timeframe}_{datetime_stamp}.json"
        with open(json_path, 'w') as f:
            json.dump(detection_result, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detection report saved: {json_path}")
        
        # Generate markdown report
        md_path = outcomes_dir / f"sr_detection_{args.symbol}_{args.timeframe}_{datetime_stamp}.md"
        with open(md_path, 'w') as f:
            f.write(f"# SR Detection Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Symbol**: {args.symbol}\\n")
            f.write(f"**Timeframe**: {args.timeframe}\\n")
            f.write(f"**Mode**: {args.mode}\\n\\n")
            f.write(f"---\\n\\n")
            f.write(f"## Summary\\n\\n")
            f.write(f"- **Total SR Levels**: {len(detected_levels)}\\n")
            f.write(f"- **Support Levels**: {len(support_levels)}\\n")
            f.write(f"- **Resistance Levels**: {len(resistance_levels)}\\n")
            f.write(f"- **ML Model Used**: Yes\\n")
            f.write(f"- **Data Points**: {len(market_data)}\\n\\n")
            f.write(f"## Top 10 Support Levels\\n\\n")
            for i, level in enumerate(sorted(support_levels, key=lambda x: x.strength, reverse=True)[:10], 1):
                f.write(f"{i}. **${level.price:.2f}** - Strength: {level.strength:.3f}, Method: {level.method}\\n")
            f.write(f"\\n## Top 10 Resistance Levels\\n\\n")
            for i, level in enumerate(sorted(resistance_levels, key=lambda x: x.strength, reverse=True)[:10], 1):
                f.write(f"{i}. **${level.price:.2f}** - Strength: {level.strength:.3f}, Method: {level.method}\\n")
        
        logger.info(f"📄 Markdown report saved: {md_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ SR DETECTION COMPLETED SUCCESSFULLY")
        logger.info(f"📁 Reports saved to: {outcomes_dir}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

