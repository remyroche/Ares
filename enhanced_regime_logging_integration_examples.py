#!/usr/bin/env python3
"""
Enhanced Regime Logging Integration Examples

This file demonstrates how to integrate the enhanced regime logging system
into the main training step files. It shows multiple integration patterns
and best practices for implementing per-HMM regime logging and fail-fast validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

# Import the enhanced regime logging components
from src.utils.regime_aware_financial_logging_decorator import (
    auto_regime_aware_logging,
    regime_aware_financial_logging
)
from src.utils.enhanced_financial_metrics_logger import (
    enhanced_financial_metrics_context,
    validate_and_log_regime_data
)
from src.utils.financial_metrics_logger import (
    log_financial_metric_with_regime_awareness
)

# Import the enhanced financial loggers
from src.training.steps.model_training.step09_financial_logging import EnhancedStep09FinancialLogger
from src.training.steps.model_training.step10_financial_logging import EnhancedStep10FinancialLogger
from src.training.steps.model_training.step11_financial_logging import EnhancedStep11FinancialLogger
from src.training.steps.model_training.step12_financial_logging import EnhancedStep12FinancialLogger
from src.training.steps.model_training.step13_financial_logging import EnhancedStep13FinancialLogger
from src.training.steps.model_training.step14_financial_logging import EnhancedStep14FinancialLogger
from src.training.steps.model_training.step15_financial_logging import EnhancedStep15FinancialLogger
from src.training.steps.model_training.step16_financial_logging import EnhancedStep16FinancialLogger
from src.training.steps.backtesting.step18_financial_logging import EnhancedStep18FinancialLogger
from src.training.steps.backtesting.step19_financial_logging import EnhancedStep19FinancialLogger
from src.training.steps.backtesting.step20_financial_logging import EnhancedStep20FinancialLogger
from src.training.steps.optimisation.step17_financial_logging import EnhancedStep17FinancialLogger


class ExampleStep09HMMBasedTraining:
    """
    Example: Step09 HMM-Based Training with Enhanced Regime Logging
    
    This example shows how to integrate enhanced regime logging into Step09,
    which is the first step after HMM-based data splitting.
    """
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        
        # Initialize enhanced financial logger
        self.financial_logger = EnhancedStep09FinancialLogger(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            enable_enhanced_logging=True
        )
    
    @auto_regime_aware_logging(
        enable_regime_validation=True,
        enable_fail_fast=True,
        min_regime_samples=100,
        max_regime_imbalance=0.8,
        regime_column='composite_cluster_id',
        min_data_quality=0.7
    )
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step09 HMM-Based Training with automatic regime-aware logging.
        
        The @auto_regime_aware_logging decorator automatically:
        1. Detects that this is a post-HMM step (step number > 8)
        2. Validates regime data if available
        3. Applies fail-fast validation
        4. Logs regime-specific metrics
        """
        try:
            # Get data from pipeline state
            data = pipeline_state.get('dataframe', pd.DataFrame())
            
            # Your existing Step09 implementation here
            training_results = await self._perform_hmm_training(training_input, data)
            model_performance = await self._evaluate_model_performance(training_results, data)
            execution_data = await self._collect_execution_metadata()
            regime_models = await self._extract_regime_models(training_results)
            
            # Log with enhanced regime validation
            logging_success = self.financial_logger.log_step_execution(
                training_results=training_results,
                model_performance=model_performance,
                execution_data=execution_data,
                regime_models=regime_models,
                data=data  # This enables regime validation
            )
            
            if not logging_success:
                print("⚠️ Enhanced regime logging failed, but step completed")
            
            return {
                'success': True,
                'training_results': training_results,
                'model_performance': model_performance,
                'regime_models': regime_models,
                'logging_success': logging_success
            }
            
        except Exception as e:
            print(f"❌ Step09 execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _perform_hmm_training(self, training_input: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate HMM training process."""
        # Your existing HMM training logic here
        return {
            'total_models_trained': 5,
            'successful_trainings': 4,
            'training_time_seconds': 120.5
        }
    
    async def _evaluate_model_performance(self, training_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate model performance evaluation."""
        # Your existing performance evaluation logic here
        return {
            'overall_accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85
        }
    
    async def _collect_execution_metadata(self) -> Dict[str, Any]:
        """Collect execution metadata."""
        return {
            'execution_time': datetime.now().isoformat(),
            'memory_usage_mb': 512.3,
            'cpu_usage_percent': 75.2
        }
    
    async def _extract_regime_models(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract regime-specific model data."""
        # Your existing regime model extraction logic here
        return {
            'regime_0': {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.90, 'f1_score': 0.87, 'training_samples': 1500},
            'regime_1': {'accuracy': 0.83, 'precision': 0.80, 'recall': 0.86, 'f1_score': 0.83, 'training_samples': 1200},
            'regime_2': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1_score': 0.85, 'training_samples': 1800}
        }


class ExampleStep10UnifiedRegimeIntelligence:
    """
    Example: Step10 Unified Regime Intelligence with Enhanced Regime Logging
    
    This example shows how to integrate enhanced regime logging into Step10,
    which processes regime intelligence data.
    """
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        
        # Initialize enhanced financial logger
        self.financial_logger = EnhancedStep10FinancialLogger(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            enable_enhanced_logging=True
        )
    
    @auto_regime_aware_logging(
        enable_regime_validation=True,
        enable_fail_fast=True,
        min_regime_samples=50,
        max_regime_imbalance=0.9,
        regime_column='composite_cluster_id',
        min_data_quality=0.6
    )
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step10 Unified Regime Intelligence with automatic regime-aware logging.
        """
        try:
            # Get data from pipeline state
            data = pipeline_state.get('dataframe', pd.DataFrame())
            
            # Your existing Step10 implementation here
            analysis_results = await self._perform_multi_timeframe_analysis(training_input, data)
            prediction_results = await self._perform_intensity_based_prediction(training_input, data)
            integration_metrics = await self._integrate_tpsl_metrics(training_input, data)
            performance_data = await self._evaluate_unified_performance(analysis_results, prediction_results, data)
            
            # Log with enhanced regime validation
            logging_success = self.financial_logger.log_step_execution(
                analysis_results=analysis_results,
                prediction_results=prediction_results,
                integration_metrics=integration_metrics,
                performance_data=performance_data,
                data=data  # This enables regime validation
            )
            
            if not logging_success:
                print("⚠️ Enhanced regime logging failed, but step completed")
            
            return {
                'success': True,
                'analysis_results': analysis_results,
                'prediction_results': prediction_results,
                'integration_metrics': integration_metrics,
                'performance_data': performance_data,
                'logging_success': logging_success
            }
            
        except Exception as e:
            print(f"❌ Step10 execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _perform_multi_timeframe_analysis(self, training_input: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate multi-timeframe HMM analysis."""
        return {
            'temporal_consistency_score': 0.92,
            'cross_timeframe_regime_alignment': 0.88,
            'regime_transition_probability': 0.15
        }
    
    async def _perform_intensity_based_prediction(self, training_input: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate intensity-based prediction."""
        return {
            'intensity_based_confidence': 0.78,
            'total_trading_signals': 45,
            'signal_quality_score': 0.82
        }
    
    async def _integrate_tpsl_metrics(self, training_input: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate TPSL integration metrics."""
        return {
            'combined_tpsl_accuracy': 0.85,
            'profit_factor': 1.45,
            'sharpe_ratio': 1.23
        }
    
    async def _evaluate_unified_performance(self, analysis_results: Dict[str, Any], 
                                          prediction_results: Dict[str, Any], 
                                          data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate unified performance evaluation."""
        return {
            'overall_accuracy': 0.87,
            'multi_timeframe_consistency': 0.89,
            'regime_classification_accuracy': {
                'regime_0': 0.90,
                'regime_1': 0.85,
                'regime_2': 0.88
            }
        }


class ExampleStep11AnalystCreation:
    """
    Example: Step11 Analyst Creation with Enhanced Regime Logging
    
    This example shows how to integrate enhanced regime logging into Step11,
    which creates regime-specific analysts.
    """
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        
        # Initialize enhanced financial logger
        self.financial_logger = EnhancedStep11FinancialLogger(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            enable_enhanced_logging=True
        )
    
    @auto_regime_aware_logging(
        enable_regime_validation=True,
        enable_fail_fast=True,
        min_regime_samples=75,
        max_regime_imbalance=0.85,
        regime_column='composite_cluster_id',
        min_data_quality=0.65
    )
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step11 Analyst Creation with automatic regime-aware logging.
        """
        try:
            # Get data from pipeline state
            data = pipeline_state.get('dataframe', pd.DataFrame())
            
            # Your existing Step11 implementation here
            created_models_summary = await self._create_regime_analysts(training_input, data)
            execution_data = await self._collect_execution_metadata()
            performance_metrics = await self._evaluate_analyst_performance(created_models_summary, data)
            optimization_metrics = await self._collect_optimization_metrics(created_models_summary, data)
            
            # Log with enhanced regime validation
            logging_success = self.financial_logger.log_step_execution(
                created_models_summary=created_models_summary,
                execution_data=execution_data,
                performance_metrics=performance_metrics,
                optimization_metrics=optimization_metrics,
                data=data  # This enables regime validation
            )
            
            if not logging_success:
                print("⚠️ Enhanced regime logging failed, but step completed")
            
            return {
                'success': True,
                'created_models_summary': created_models_summary,
                'execution_data': execution_data,
                'performance_metrics': performance_metrics,
                'optimization_metrics': optimization_metrics,
                'logging_success': logging_success
            }
            
        except Exception as e:
            print(f"❌ Step11 execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _create_regime_analysts(self, training_input: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate regime analyst creation."""
        return {
            'total_analysts_created': 3,
            'regime_0_analyst': {'accuracy': 0.88, 'precision': 0.85, 'recall': 0.91},
            'regime_1_analyst': {'accuracy': 0.84, 'precision': 0.81, 'recall': 0.87},
            'regime_2_analyst': {'accuracy': 0.86, 'precision': 0.83, 'recall': 0.89}
        }
    
    async def _collect_execution_metadata(self) -> Dict[str, Any]:
        """Collect execution metadata."""
        return {
            'execution_time': datetime.now().isoformat(),
            'memory_usage_mb': 384.7,
            'cpu_usage_percent': 68.5
        }
    
    async def _evaluate_analyst_performance(self, created_models_summary: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate analyst performance evaluation."""
        return {
            'overall_analyst_accuracy': 0.86,
            'best_analyst_regime': 'regime_0',
            'worst_analyst_regime': 'regime_1'
        }
    
    async def _collect_optimization_metrics(self, created_models_summary: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate optimization metrics collection."""
        return {
            'optimization_time_seconds': 45.2,
            'hyperparameter_trials': 25,
            'best_hyperparameters': {'learning_rate': 0.001, 'batch_size': 32}
        }


class ExampleStep18WalkForwardValidation:
    """
    Example: Step18 Walk Forward Validation with Enhanced Regime Logging
    
    This example shows how to integrate enhanced regime logging into Step18,
    which performs walk-forward validation with regime awareness.
    """
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        
        # Initialize enhanced financial logger
        self.financial_logger = EnhancedStep18FinancialLogger(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            enable_enhanced_logging=True
        )
    
    @auto_regime_aware_logging(
        enable_regime_validation=True,
        enable_fail_fast=True,
        min_regime_samples=200,
        max_regime_imbalance=0.7,
        regime_column='composite_cluster_id',
        min_data_quality=0.8
    )
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step18 Walk Forward Validation with automatic regime-aware logging.
        """
        try:
            # Get data from pipeline state
            data = pipeline_state.get('dataframe', pd.DataFrame())
            
            # Your existing Step18 implementation here
            validation_results = await self._perform_walk_forward_validation(training_input, data)
            performance_metrics = await self._evaluate_validation_performance(validation_results, data)
            regime_analysis = await self._analyze_regime_performance(validation_results, data)
            execution_metadata = await self._collect_execution_metadata()
            
            # Log with enhanced regime validation
            logging_success = self.financial_logger.log_step_execution(
                validation_results=validation_results,
                performance_metrics=performance_metrics,
                regime_analysis=regime_analysis,
                execution_metadata=execution_metadata,
                data=data  # This enables regime validation
            )
            
            if not logging_success:
                print("⚠️ Enhanced regime logging failed, but step completed")
            
            return {
                'success': True,
                'validation_results': validation_results,
                'performance_metrics': performance_metrics,
                'regime_analysis': regime_analysis,
                'execution_metadata': execution_metadata,
                'logging_success': logging_success
            }
            
        except Exception as e:
            print(f"❌ Step18 execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _perform_walk_forward_validation(self, training_input: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate walk-forward validation."""
        return {
            'total_windows': 10,
            'successful_windows': 8,
            'average_window_performance': 0.82,
            'best_window_performance': 0.91,
            'worst_window_performance': 0.73
        }
    
    async def _evaluate_validation_performance(self, validation_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate validation performance evaluation."""
        return {
            'overall_validation_accuracy': 0.82,
            'consistency_score': 0.85,
            'stability_metric': 0.78
        }
    
    async def _analyze_regime_performance(self, validation_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate regime performance analysis."""
        return {
            'regime_0_performance': 0.85,
            'regime_1_performance': 0.79,
            'regime_2_performance': 0.82,
            'regime_transition_impact': 0.12
        }
    
    async def _collect_execution_metadata(self) -> Dict[str, Any]:
        """Collect execution metadata."""
        return {
            'execution_time': datetime.now().isoformat(),
            'memory_usage_mb': 1024.5,
            'cpu_usage_percent': 85.3
        }


class ExampleManualRegimeLogging:
    """
    Example: Manual Regime Logging Integration
    
    This example shows how to manually integrate regime logging without decorators,
    giving you full control over the logging process.
    """
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute with manual regime logging integration.
        """
        try:
            # Get data from pipeline state
            data = pipeline_state.get('dataframe', pd.DataFrame())
            
            # Manual regime validation
            if not data.empty and 'composite_cluster_id' in data.columns:
                validation_success = validate_and_log_regime_data(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Manual_Regime_Logging_Example",
                    data=data,
                    regime_column='composite_cluster_id'
                )
                
                if not validation_success:
                    print("🚨 Regime validation failed - stopping execution")
                    return {'success': False, 'error': 'Regime validation failed'}
            
            # Your existing implementation here
            results = await self._perform_manual_processing(training_input, data)
            
            # Manual regime-aware logging
            if not data.empty:
                # Log individual metrics with regime awareness
                log_financial_metric_with_regime_awareness(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="manual_processing_success",
                    metric_value=1.0,
                    metric_type="performance",
                    step_name="Manual_Regime_Logging_Example",
                    data=data
                )
                
                # Use enhanced context for comprehensive logging
                with enhanced_financial_metrics_context(
                    step_name="Manual_Regime_Logging_Example",
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    data=data
                ) as enhanced_logger:
                    # Log regime-specific metrics
                    regime_data = data['composite_cluster_id'].dropna()
                    regime_counts = regime_data.value_counts()
                    
                    regime_metrics = {}
                    for regime_id, count in regime_counts.items():
                        regime_metrics[str(regime_id)] = {
                            'sample_count': float(count),
                            'regime_processed': 1.0
                        }
                    
                    enhanced_logger.log_per_regime_metrics(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        step_name="Manual_Regime_Logging_Example",
                        regime_metrics=regime_metrics,
                        data=data
                    )
            
            return {
                'success': True,
                'results': results,
                'regime_validation_passed': not data.empty
            }
            
        except Exception as e:
            print(f"❌ Manual regime logging execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _perform_manual_processing(self, training_input: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate manual processing."""
        return {
            'processing_completed': True,
            'data_points_processed': len(data) if not data.empty else 0,
            'regimes_detected': len(data['composite_cluster_id'].unique()) if not data.empty and 'composite_cluster_id' in data.columns else 0
        }


async def main():
    """
    Main function demonstrating the enhanced regime logging integration examples.
    """
    print("🚀 Enhanced Regime Logging Integration Examples")
    print("=" * 60)
    
    # Example data with regime information
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
        'price': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000),
        'composite_cluster_id': np.random.choice(['regime_0', 'regime_1', 'regime_2'], 1000, p=[0.4, 0.35, 0.25])
    })
    
    pipeline_state = {
        'dataframe': sample_data,
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1h'
    }
    
    training_input = {
        'model_type': 'hmm_ensemble',
        'optimization_target': 'sharpe_ratio',
        'validation_method': 'walk_forward'
    }
    
    # Test Step09 with enhanced regime logging
    print("\n🔄 Testing Step09 HMM-Based Training...")
    step09 = ExampleStep09HMMBasedTraining('BTCUSDT', 'binance', '1h')
    result09 = await step09.execute(training_input, pipeline_state)
    print(f"✅ Step09 Result: {result09['success']}")
    
    # Test Step10 with enhanced regime logging
    print("\n🔄 Testing Step10 Unified Regime Intelligence...")
    step10 = ExampleStep10UnifiedRegimeIntelligence('BTCUSDT', 'binance', '1h')
    result10 = await step10.execute(training_input, pipeline_state)
    print(f"✅ Step10 Result: {result10['success']}")
    
    # Test Step11 with enhanced regime logging
    print("\n🔄 Testing Step11 Analyst Creation...")
    step11 = ExampleStep11AnalystCreation('BTCUSDT', 'binance', '1h')
    result11 = await step11.execute(training_input, pipeline_state)
    print(f"✅ Step11 Result: {result11['success']}")
    
    # Test Step18 with enhanced regime logging
    print("\n🔄 Testing Step18 Walk Forward Validation...")
    step18 = ExampleStep18WalkForwardValidation('BTCUSDT', 'binance', '1h')
    result18 = await step18.execute(training_input, pipeline_state)
    print(f"✅ Step18 Result: {result18['success']}")
    
    # Test manual regime logging
    print("\n🔄 Testing Manual Regime Logging...")
    manual = ExampleManualRegimeLogging('BTCUSDT', 'binance', '1h')
    result_manual = await manual.execute(training_input, pipeline_state)
    print(f"✅ Manual Result: {result_manual['success']}")
    
    print("\n" + "=" * 60)
    print("🎯 Enhanced Regime Logging Integration Examples Completed!")
    print("📋 All examples demonstrate:")
    print("   ✅ Automatic regime validation")
    print("   ✅ Fail-fast error handling")
    print("   ✅ Per-regime metrics logging")
    print("   ✅ Enhanced financial metrics tracking")
    print("   ✅ Backward compatibility")


if __name__ == "__main__":
    asyncio.run(main())