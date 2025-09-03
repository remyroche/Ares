#!/usr/bin/env python3
"""Example pipeline runner demonstrating the integration of Step 8."""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.logger import system_logger


async def run_pipeline_with_step08(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    start_step: str = "step06_feature_engineering"
):
    """
    Run the training pipeline with the new Step 8 integration.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        start_step: Step to start from
    """
    logger = system_logger.getChild("PipelineWithStep08")
    
    # Load configuration
    config = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        
        # Step 7 configuration
        "step07_enhanced_matrix_operations": {
            "output_dir": "data/matrix_operations",
            "target_features": 200,
            "removal_fraction": 0.33,
            "enable_regime_selection": True,
            "enable_shap_filtering": True
        },
        
        # Step 8 configuration
        "step08_advanced_feature_selection": {
            "output_dir": "data/selected_features",
            "phase1_target_features": 150,
            "enable_mrmr": True,
            "enable_rf_importance": True,
            "phase2_targets": [100, 80, 60],
            "boruta_max_iter": 100,
            "boruta_alpha": 0.05,
            "n_splits_ts": 5,
            "min_regime_samples": 100,
            "enable_shap": True,
            "enable_lime": True,
            "n_lime_samples": 10,
            "enable_redundancy_analysis": True,
            "min_redundancy_correlation": 0.7,
            "redundancy_groups_per_concept": 2,
            "n_jobs": -1  # Use all CPU cores
        },
        
        # Other step configurations...
        "training": {
            "start_step": start_step,
            "force_rerun": False,
            "validate_each_step": True
        }
    }
    
    # Initialize training manager
    training_manager = EnhancedTrainingManager(config)
    
    logger.info(f"🚀 Starting pipeline for {symbol} on {exchange}")
    logger.info(f"   Starting from: {start_step}")
    logger.info("\nPipeline Flow:")
    logger.info("  Step 6: Advanced Feature Engineering (~300+ features)")
    logger.info("  Step 7: Matrix Operations & Initial Filtering (~200 features)")
    logger.info("  Step 8: Advanced Feature Selection")
    logger.info("    - Phase 1: mRMR/RF selection (~150 features)")
    logger.info("    - Phase 2: Boruta with redundancy reduction")
    logger.info("      • 100 features (ensemble models)")
    logger.info("      • 80 features (neural networks)")
    logger.info("      • 60 features (linear models)")
    logger.info("  Step 9: HMM Model Training (uses Step 8 outputs)")
    
    # Run the pipeline
    try:
        success = await training_manager.train(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_step=start_step
        )
        
        if success:
            logger.info("\n✅ Pipeline completed successfully!")
            
            # Display feature selection results
            report_path = f"data/selected_features/{exchange}_{symbol}_{timeframe}_selection_report.json"
            if Path(report_path).exists():
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                logger.info("\n📊 Feature Selection Summary:")
                phase1 = report.get('phase1_summary', {})
                logger.info(f"  Phase 1: {phase1.get('input_features', 0)} → {phase1.get('output_features', 0)} features")
                logger.info(f"    - Consensus features: {phase1.get('consensus_features', 0)}")
                logger.info(f"    - Regime validated: {phase1.get('regime_validated', 0)}")
                
                phase2 = report.get('phase2_summary', {})
                for size, data in phase2.items():
                    logger.info(f"  Phase 2 ({size}): {data.get('features', 0)} features")
                    logger.info(f"    - TS validation score: {data.get('ts_score', 0):.4f}")
                    logger.info(f"    - Boruta confirmed: {data.get('boruta_confirmed', 0)}")
        else:
            logger.error("\n❌ Pipeline failed!")
            return False
            
    except Exception as e:
        logger.exception(f"❌ Pipeline error: {e}")
        return False
    
    return True


async def main():
    """Main function."""
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = "BTCUSDT"
    
    if len(sys.argv) > 2:
        exchange = sys.argv[2]
    else:
        exchange = "binance"
    
    if len(sys.argv) > 3:
        start_step = sys.argv[3]
    else:
        start_step = "step06_feature_engineering"
    
    success = await run_pipeline_with_step08(
        symbol=symbol,
        exchange=exchange,
        start_step=start_step
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    print("=" * 60)
    print("Training Pipeline with Advanced Feature Selection (Step 8)")
    print("=" * 60)
    print("\nUsage: python run_pipeline_with_step08.py [symbol] [exchange] [start_step]")
    print("Example: python run_pipeline_with_step08.py BTCUSDT binance step06_feature_engineering")
    print()
    
    asyncio.run(main())