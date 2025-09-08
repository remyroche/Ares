#!/usr/bin/env python3
"""
Example: Enhanced Regime-Aware Financial Logging Usage

This example demonstrates how to use the enhanced financial metrics logger
with per-HMM regime logging and fail-fast validation in training steps.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

# Import the enhanced financial logging components
from src.utils.enhanced_financial_metrics_logger import (
    get_enhanced_financial_metrics_logger,
    enhanced_financial_metrics_context,
    validate_and_log_regime_data
)

from src.utils.regime_aware_financial_logging_decorator import (
    regime_aware_financial_logging,
    auto_regime_aware_logging,
    is_post_hmm_step
)

from src.utils.financial_metrics_logger import (
import logging
import os
import time

    get_smart_financial_metrics_logger,
    log_financial_metric_with_regime_awareness
)


class ExampleTrainingStep:
    """Example training step demonstrating enhanced regime logging."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbol = config.get('symbol', 'ETHUSDT')
        self.exchange = config.get('exchange', 'BINANCE')
        self.timeframe = config.get('timeframe', '1m')
        
        # Initialize enhanced logger
        self.enhanced_logger = get_enhanced_financial_metrics_logger()
        self.smart_logger = get_smart_financial_metrics_logger(use_enhanced=True)
    
    @regime_aware_financial_logging(
        step_name="ExampleStep_Enhanced_Regime_Logging",
        enable_regime_validation=True,
        enable_fail_fast=True,
        min_regime_samples=100,
        max_regime_imbalance=0.8,
        regime_column='composite_cluster_id',
        min_data_quality=0.7
    )
    async def execute_with_decorator(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example step execution with automatic regime-aware logging decorator.
        
        The decorator automatically:
        1. Validates regime data
        2. Applies fail-fast validation
        3. Logs per-regime metrics
        4. Prevents empty running or degradation
        """
        # Simulate some training work
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Simulate regime-specific model training
        regime_models = {}
        if 'composite_cluster_id' in data.columns:
            unique_regimes = data['composite_cluster_id'].dropna().unique()
            
            for regime_id in unique_regimes:
                regime_data = data[data['composite_cluster_id'] == regime_id]
                
                # Simulate model training for this regime
                regime_models[str(regime_id)] = {
                    'accuracy': np.random.uniform(0.6, 0.9),
                    'precision': np.random.uniform(0.6, 0.9),
                    'recall': np.random.uniform(0.6, 0.9),
                    'f1_score': np.random.uniform(0.6, 0.9),
                    'training_samples': len(regime_data)
                }
        
        # Simulate execution results
        result = {
            'success': True,
            'execution_time': 120.5,
            'total_models_trained': len(regime_models),
            'regime_models': regime_models,
            'overall_accuracy': np.mean([m['accuracy'] for m in regime_models.values()]) if regime_models else 0.0
        }
        
        return result
    
    @auto_regime_aware_logging(
        enable_regime_validation=True,
        enable_fail_fast=True,
        min_regime_samples=100
    )
    async def execute_with_auto_decorator(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example step execution with auto regime-aware logging decorator.
        
        The auto decorator automatically detects if this is a post-HMM step
        and applies regime-aware logging only if needed.
        """
        # Simulate some training work
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Simulate model training results
        result = {
            'success': True,
            'execution_time': 95.2,
            'models_trained': 5,
            'accuracy': 0.85
        }
        
        return result
    
    async def execute_with_manual_logging(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example step execution with manual enhanced regime logging.
        
        This shows how to manually use the enhanced logging features.
        """
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Use enhanced financial metrics context
        with enhanced_financial_metrics_context(
            step_name="ExampleStep_Manual_Enhanced_Logging",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            data=data
        ) as enhanced_logger:
            
            # Validate regime data
            if not data.empty and 'composite_cluster_id' in data.columns:
                validation_success = validate_and_log_regime_data(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="ExampleStep_Manual_Enhanced_Logging",
                    data=data,
                    regime_column='composite_cluster_id'
                )
                
                if not validation_success:
                    raise RuntimeError("Regime validation failed")
            
            # Simulate regime-specific processing
            regime_metrics = {}
            if 'composite_cluster_id' in data.columns:
                unique_regimes = data['composite_cluster_id'].dropna().unique()
                
                for regime_id in unique_regimes:
                    regime_data = data[data['composite_cluster_id'] == regime_id]
                    
                    # Calculate regime-specific metrics
                    regime_metrics[str(regime_id)] = {
                        'sample_count': len(regime_data),
                        'accuracy': np.random.uniform(0.6, 0.9),
                        'volatility': regime_data['close'].std() if 'close' in regime_data.columns else 0.0,
                        'mean_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0
                    }
            
            # Log per-regime metrics
            if regime_metrics:
                success = enhanced_logger.log_per_regime_metrics(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="ExampleStep_Manual_Enhanced_Logging",
                    regime_metrics=regime_metrics,
                    data=data
                )
                
                if not success:
                    raise RuntimeError("Failed to log per-regime metrics")
            
            # Log individual metrics with regime awareness
            log_financial_metric_with_regime_awareness(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                metric_name="total_regimes_processed",
                metric_value=float(len(regime_metrics)),
                metric_type="regime",
                step_name="ExampleStep_Manual_Enhanced_Logging",
                data=data
            )
            
            # Simulate execution results
            result = {
                'success': True,
                'execution_time': 150.3,
                'regimes_processed': len(regime_metrics),
                'regime_metrics': regime_metrics
            }
            
            return result
    
    async def execute_with_smart_logger(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example step execution using the smart logger that automatically
        chooses between enhanced and base logging.
        """
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Use smart logger
        smart_logger = get_smart_financial_metrics_logger(use_enhanced=True)
        
        # Log step start
        smart_logger.log_step_start("ExampleStep_Smart_Logger", self.symbol, self.exchange, self.timeframe)
        
        try:
            # Simulate some work
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Log metrics using smart logger
            smart_logger.log_financial_metric(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                metric_name="processing_completed",
                metric_value=1.0,
                metric_type="performance",
                step_name="ExampleStep_Smart_Logger"
            )
            
            # Log regime-specific metrics if data is available
            if not data.empty and 'composite_cluster_id' in data.columns:
                unique_regimes = data['composite_cluster_id'].dropna().unique()
                
                for regime_id in unique_regimes:
                    regime_data = data[data['composite_cluster_id'] == regime_id]
                    
                    smart_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name=f"regime_{regime_id}_sample_count",
                        metric_value=float(len(regime_data)),
                        metric_type="regime",
                        step_name="ExampleStep_Smart_Logger",
                        regime_id=str(regime_id)
                    )
            
            # Log step end
            smart_logger.log_step_end("ExampleStep_Smart_Logger", self.symbol, self.exchange, self.timeframe, success=True)
            
            return {
                'success': True,
                'execution_time': 50.0,
                'regimes_processed': len(data['composite_cluster_id'].dropna().unique()) if 'composite_cluster_id' in data.columns else 0
            }
            
        except Exception as e:
            smart_logger.log_step_end("ExampleStep_Smart_Logger", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
            raise


def create_sample_data() -> pd.DataFrame:
    """Create sample data with regime information for testing."""
    np.random.seed(42)
    
    # Create sample time series data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 1000),
        'high': np.random.uniform(100, 200, 1000),
        'low': np.random.uniform(100, 200, 1000),
        'close': np.random.uniform(100, 200, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Add regime information (composite_cluster_id)
    # Simulate 3 different market regimes
    regime_pattern = np.tile([0, 1, 2], 334)[:1000]  # Repeat pattern
    np.random.shuffle(regime_pattern)  # Randomize order
    
    data['composite_cluster_id'] = regime_pattern
    data = data.set_index('timestamp')
    
    return data


async def run_examples():
    """Run all the example scenarios."""
    print("🚀 Running Enhanced Regime-Aware Financial Logging Examples")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_data()
    print(f"📊 Created sample data with {len(sample_data)} rows and {len(sample_data['composite_cluster_id'].unique())} regimes")
    
    # Create example step
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m'
    }
    
    step = ExampleTrainingStep(config)
    
    # Prepare pipeline state
    pipeline_state = {
        'dataframe': sample_data
    }
    
    training_input = {}
    
    print("\n1️⃣ Testing with regime-aware decorator...")
    try:
        result1 = await step.execute_with_decorator(training_input, pipeline_state)
        print(f"✅ Decorator execution successful: {result1['success']}")
    except Exception as e:
        print(f"❌ Decorator execution failed: {e}")
    
    print("\n2️⃣ Testing with auto regime-aware decorator...")
    try:
        result2 = await step.execute_with_auto_decorator(training_input, pipeline_state)
        print(f"✅ Auto decorator execution successful: {result2['success']}")
    except Exception as e:
        print(f"❌ Auto decorator execution failed: {e}")
    
    print("\n3️⃣ Testing with manual enhanced logging...")
    try:
        result3 = await step.execute_with_manual_logging(training_input, pipeline_state)
        print(f"✅ Manual logging execution successful: {result3['success']}")
    except Exception as e:
        print(f"❌ Manual logging execution failed: {e}")
    
    print("\n4️⃣ Testing with smart logger...")
    try:
        result4 = await step.execute_with_smart_logger(training_input, pipeline_state)
        print(f"✅ Smart logger execution successful: {result4['success']}")
    except Exception as e:
        print(f"❌ Smart logger execution failed: {e}")
    
    print("\n5️⃣ Testing fail-fast validation with empty data...")
    try:
        empty_pipeline_state = {'dataframe': pd.DataFrame()}
        result5 = await step.execute_with_decorator(training_input, empty_pipeline_state)
        print(f"⚠️ Empty data execution result: {result5['success']}")
    except Exception as e:
        print(f"🚨 Fail-fast validation triggered (expected): {e}")
    
    print("\n6️⃣ Testing step detection...")
    print(f"Step09 is post-HMM: {is_post_hmm_step('Step09_HMM_Based_Training')}")
    print(f"Step05 is post-HMM: {is_post_hmm_step('Step05_Labeling')}")
    print(f"Step15 is post-HMM: {is_post_hmm_step('Step15_Tactician_Training')}")
    
    print("\n📊 Getting regime summary...")
    try:
        enhanced_logger = get_enhanced_financial_metrics_logger()
        summary = enhanced_logger.get_regime_summary()
        print(f"Regimes tracked: {summary.get('total_regimes_tracked', 0)}")
        print(f"Total validations: {summary.get('total_validations', 0)}")
        print(f"Fail-fast checks: {summary.get('total_fail_fast_checks', 0)}")
    except Exception as e:
        print(f"Could not get regime summary: {e}")
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(run_examples())