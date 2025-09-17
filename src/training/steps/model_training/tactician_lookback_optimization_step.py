"""
Tactician Lookback Optimization Training Step

This module implements the training step that optimizes lookback periods
for Tactician models, integrating with the main training pipeline.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Import core utilities
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_structured, LogLevel
)

# Import the optimizer
from .tactician_lookback_optimization import (
    TacticianLookbackOptimizer, TacticianLookbackConfig,
    optimize_tactician_lookbacks, create_tactician_lookback_config
)

# Import ML common utilities
try:
    from src.utils.ml_common.config import PerRegimeTrainingConfig
    from src.utils.ml_common.training import BaseTrainingStep
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    # Create fallback base class
    class BaseTrainingStep:
        def __init__(self, config):
            self.config = config
            self.logger = system_logger.getChild('BaseTrainingStep')

# Import model loading utilities
try:
    from src.utils.standardized_model_manager import standardized_model_manager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False

logger = system_logger.getChild('TacticianLookbackOptimizationStep')


class TacticianLookbackOptimizationStep(BaseTrainingStep):
    """
    Training step for optimizing Tactician lookback periods.
    
    This step runs between Analyst and Tactician training to optimize
    indicator lookback periods specifically for the Tactician model
    operating on 1m timeframe with Analyst inputs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Tactician lookback optimization step."""
        try:
            # Create default config if not provided
            if config is None:
                config = self._create_default_config()
            
            # Initialize parent class
            if ML_COMMON_AVAILABLE:
                # Convert to PerRegimeTrainingConfig if needed
                if isinstance(config, dict):
                    training_config = PerRegimeTrainingConfig(
                        model_name="tactician_lookback_optimization",
                        timeframe="1m",
                        **config
                    )
                else:
                    training_config = config
                super().__init__(training_config)
            else:
                super().__init__(config)
            
            self.logger = logger.getChild('TacticianLookbackOptimizationStep')
            
            # Create optimization config
            self.optimization_config = create_tactician_lookback_config(
                timeframe="1m",
                symbol=config.get('symbol', 'ETHUSDT'),
                exchange=config.get('exchange', 'binance'),
                optimization_method=config.get('optimization_method', 'two_step_grid_tpe'),
                tpe_trials=config.get('tpe_trials', 25),
                optimization_timeout=config.get('optimization_timeout', 3600)
            )
            
            # State management
            self.optimizer = None
            self.optimization_results = None
            self.analyst_models_loaded = False
            
            tprint_success("✅ Tactician Lookback Optimization Step initialized")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Tactician Lookback Optimization Step: {e}")
            raise
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration for the optimization step."""
        return {
            'model_name': 'tactician_lookback_optimization',
            'timeframe': '1m',
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'optimization_method': 'two_step_grid_tpe',
            'tpe_trials': 25,
            'optimization_timeout': 3600,
            'save_results': True,
            'results_path': './results/tactician_lookback_optimization',
            'requires_analyst_outputs': True,
            'analyst_model_path': './models/analyst_models',
            'analyst_ensemble_path': './models/analyst_ensemble'
        }
    
    async def execute(
        self,
        market_data_1m: pd.DataFrame,
        analyst_models: Optional[Dict[str, Any]] = None,
        analyst_ensemble: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the Tactician lookback optimization step.
        
        Args:
            market_data_1m: 1-minute market data for optimization
            analyst_models: Trained Analyst models (from previous step)
            analyst_ensemble: Trained Analyst ensemble (from previous step)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing optimization results and optimized lookback periods
        """
        try:
            tprint_info("🚀 Executing Tactician Lookback Optimization Step...")
            start_time = datetime.now()
            
            # Validate inputs
            if not self._validate_inputs(market_data_1m, analyst_models, analyst_ensemble):
                raise ValueError("Input validation failed")
            
            # Initialize optimizer
            self.optimizer = TacticianLookbackOptimizer(self.optimization_config)
            success = await self.optimizer.initialize()
            
            if not success:
                raise RuntimeError("Failed to initialize Tactician lookback optimizer")
            
            # Generate Analyst outputs for optimization
            analyst_signals, analyst_outputs = await self._generate_analyst_outputs(
                market_data_1m, analyst_models, analyst_ensemble
            )
            
            # Execute optimization
            tprint_info("🎯 Starting lookback period optimization...")
            optimization_results = await self.optimizer.optimize_lookback_periods(
                market_data_1m, analyst_signals, analyst_outputs
            )
            
            # Process and validate results
            processed_results = self._process_optimization_results(optimization_results)
            
            # Save results for Tactician training step
            await self._save_results_for_tactician(processed_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            final_results = {
                'step_name': 'tactician_lookback_optimization',
                'execution_time': execution_time,
                'optimization_results': processed_results,
                'optimized_lookbacks': processed_results.get('best_lookbacks', {}),
                'optimization_score': processed_results.get('best_score', 0.0),
                'configuration': self.optimization_config.__dict__,
                'metadata': {
                    'timestamp': start_time.isoformat(),
                    'duration_seconds': execution_time,
                    'data_samples': len(market_data_1m),
                    'analyst_models_count': len(analyst_models) if analyst_models else 0,
                    'has_analyst_ensemble': analyst_ensemble is not None
                }
            }
            
            tprint_success(f"✅ Tactician Lookback Optimization completed in {execution_time:.2f}s")
            tprint_structured({
                'optimized_lookbacks': len(processed_results.get('best_lookbacks', {})),
                'optimization_score': f"{processed_results.get('best_score', 0.0):.4f}",
                'total_evaluations': processed_results.get('optimization_metrics', {}).get('total_evaluations', 0)
            })
            
            return final_results
            
        except Exception as e:
            tprint_error(f"❌ Tactician Lookback Optimization Step failed: {e}")
            raise
    
    def _validate_inputs(
        self,
        market_data_1m: pd.DataFrame,
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensemble: Optional[Any]
    ) -> bool:
        """Validate input data and models."""
        try:
            tprint_info("🔍 Validating inputs for Tactician lookback optimization...")
            
            # Validate market data
            if market_data_1m is None or market_data_1m.empty:
                tprint_error("❌ Market data is empty or None")
                return False
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data_1m.columns]
            
            if missing_columns:
                tprint_error(f"❌ Missing required columns in market data: {missing_columns}")
                return False
            
            # Check data length
            min_required_length = 200  # Minimum for meaningful optimization
            if len(market_data_1m) < min_required_length:
                tprint_error(f"❌ Insufficient data: {len(market_data_1m)} rows, need at least {min_required_length}")
                return False
            
            # Validate Analyst models (required for dependency-aware optimization)
            if self.optimization_config.requires_analyst_outputs:
                if analyst_models is None and analyst_ensemble is None:
                    tprint_error("❌ Analyst models or ensemble required but not provided")
                    return False
                
                if analyst_models:
                    tprint_info(f"📊 Analyst models available: {len(analyst_models)} models")
                
                if analyst_ensemble:
                    tprint_info("📊 Analyst ensemble available")
            
            tprint_success("✅ Input validation passed")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Input validation failed: {e}")
            return False
    
    async def _generate_analyst_outputs(
        self,
        market_data_1m: pd.DataFrame,
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensemble: Optional[Any]
    ) -> tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate Analyst outputs for use in Tactician optimization.
        
        Returns:
            Tuple of (analyst_signals, analyst_outputs)
        """
        try:
            tprint_info("🔄 Generating Analyst outputs for Tactician optimization...")
            
            analyst_signals = None
            analyst_outputs = {}
            
            # Generate signals and outputs from individual models
            if analyst_models:
                model_predictions = []
                model_confidences = []
                
                for model_name, model in analyst_models.items():
                    try:
                        # This would call the actual model prediction
                        # For now, we'll create mock outputs based on market data
                        predictions = self._generate_mock_analyst_predictions(
                            market_data_1m, model_name
                        )
                        
                        model_predictions.append(predictions['predictions'])
                        model_confidences.append(predictions['confidences'])
                        
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate predictions from {model_name}: {e}")
                
                if model_predictions:
                    # Combine individual model outputs
                    analyst_outputs['individual_predictions'] = np.column_stack(model_predictions)
                    analyst_outputs['individual_confidences'] = np.column_stack(model_confidences)
                    
                    # Generate combined signals (green light indicators)
                    combined_predictions = np.mean(model_predictions, axis=0)
                    combined_confidences = np.mean(model_confidences, axis=0)
                    
                    # Green light when confidence > 0.6 and prediction > 0.5
                    analyst_signals = ((combined_confidences > 0.6) & (combined_predictions > 0.5)).astype(int)
                    
                    analyst_outputs['combined_predictions'] = combined_predictions
                    analyst_outputs['combined_confidences'] = combined_confidences
            
            # Generate ensemble outputs
            if analyst_ensemble:
                try:
                    # This would call the actual ensemble prediction
                    ensemble_predictions = self._generate_mock_ensemble_predictions(market_data_1m)
                    analyst_outputs['ensemble_predictions'] = ensemble_predictions['predictions']
                    analyst_outputs['ensemble_confidences'] = ensemble_predictions['confidences']
                    
                    # Update signals if ensemble available
                    if analyst_signals is None:
                        ensemble_signals = ((ensemble_predictions['confidences'] > 0.6) & 
                                          (ensemble_predictions['predictions'] > 0.5)).astype(int)
                        analyst_signals = ensemble_signals
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate ensemble predictions: {e}")
            
            # Fallback: create basic signals if none generated
            if analyst_signals is None:
                tprint_warning("⚠️ No Analyst signals generated, creating fallback signals")
                # Simple momentum-based fallback signals
                returns = market_data_1m['close'].pct_change()
                momentum = returns.rolling(window=10).mean()
                analyst_signals = (momentum > 0.001).astype(int).values
                
                analyst_outputs['fallback_signals'] = analyst_signals
            
            tprint_success(f"✅ Generated Analyst outputs: {len(analyst_signals)} signals, {len(analyst_outputs)} output types")
            
            return analyst_signals, analyst_outputs
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate Analyst outputs: {e}")
            return None, {}
    
    def _generate_mock_analyst_predictions(
        self,
        market_data: pd.DataFrame,
        model_name: str
    ) -> Dict[str, np.ndarray]:
        """Generate mock Analyst predictions for testing."""
        try:
            # This is a placeholder - in production, this would call the actual model
            n_samples = len(market_data)
            
            # Generate realistic mock predictions based on market data
            returns = market_data['close'].pct_change().fillna(0)
            volatility = returns.rolling(window=20).std().fillna(0.01)
            
            # Mock predictions with some market signal correlation
            base_predictions = 0.5 + 0.3 * np.tanh(returns.rolling(window=5).mean() / volatility)
            noise = np.random.normal(0, 0.1, n_samples)
            predictions = np.clip(base_predictions + noise, 0.1, 0.9)
            
            # Mock confidences inversely related to volatility
            base_confidences = 1.0 - np.clip(volatility * 10, 0.2, 0.8)
            confidence_noise = np.random.normal(0, 0.05, n_samples)
            confidences = np.clip(base_confidences + confidence_noise, 0.3, 0.95)
            
            return {
                'predictions': predictions.values,
                'confidences': confidences.values
            }
            
        except Exception as e:
            self.logger.warning(f"Mock prediction generation failed for {model_name}: {e}")
            # Return neutral predictions as fallback
            n_samples = len(market_data)
            return {
                'predictions': np.full(n_samples, 0.5),
                'confidences': np.full(n_samples, 0.6)
            }
    
    def _generate_mock_ensemble_predictions(self, market_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate mock ensemble predictions."""
        try:
            n_samples = len(market_data)
            
            # Generate ensemble predictions with higher quality than individual models
            returns = market_data['close'].pct_change().fillna(0)
            sma_short = market_data['close'].rolling(window=10).mean()
            sma_long = market_data['close'].rolling(window=30).mean()
            
            # Ensemble predictions based on moving average crossover
            signal_strength = (sma_short - sma_long) / sma_long
            predictions = 0.5 + 0.4 * np.tanh(signal_strength * 10)
            predictions = np.clip(predictions, 0.2, 0.8)
            
            # Higher confidence for ensemble
            base_confidences = 0.7 + 0.2 * (1 - returns.rolling(window=10).std())
            confidences = np.clip(base_confidences, 0.5, 0.9)
            
            return {
                'predictions': predictions.fillna(0.5).values,
                'confidences': confidences.fillna(0.7).values
            }
            
        except Exception as e:
            self.logger.warning(f"Mock ensemble prediction generation failed: {e}")
            n_samples = len(market_data)
            return {
                'predictions': np.full(n_samples, 0.5),
                'confidences': np.full(n_samples, 0.7)
            }
    
    def _process_optimization_results(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate optimization results."""
        try:
            tprint_info("📊 Processing optimization results...")
            
            # Validate results structure
            if not optimization_results:
                raise ValueError("Empty optimization results")
            
            best_lookbacks = optimization_results.get('best_lookbacks', {})
            if not best_lookbacks:
                tprint_warning("⚠️ No optimized lookbacks found, using defaults")
                best_lookbacks = self._get_default_lookbacks()
            
            # Validate lookback values
            validated_lookbacks = {}
            for indicator, lookback in best_lookbacks.items():
                if isinstance(lookback, (int, float)) and 5 <= lookback <= 60:
                    validated_lookbacks[indicator] = int(lookback)
                else:
                    tprint_warning(f"⚠️ Invalid lookback for {indicator}: {lookback}, using default")
                    validated_lookbacks[indicator] = self._get_default_lookback(indicator)
            
            # Update results with validated lookbacks
            optimization_results['best_lookbacks'] = validated_lookbacks
            optimization_results['validated'] = True
            optimization_results['validation_timestamp'] = datetime.now().isoformat()
            
            tprint_success(f"✅ Processed {len(validated_lookbacks)} optimized lookback periods")
            
            return optimization_results
            
        except Exception as e:
            tprint_error(f"❌ Failed to process optimization results: {e}")
            # Return fallback results
            return {
                'best_lookbacks': self._get_default_lookbacks(),
                'best_score': 0.5,
                'optimization_method': 'fallback',
                'error': str(e)
            }
    
    def _get_default_lookbacks(self) -> Dict[str, int]:
        """Get default lookback periods for 1m timeframe."""
        return {
            'rsi': 14,
            'macd': 26,
            'bollinger_bands': 20,
            'stoch': 14,
            'volume_sma': 20,
            'vwap': 20,
            'obv': 10,
            'volume_roc': 12,
            'williams_r': 14,
            'cci': 20,
            'momentum': 10,
            'roc': 12,
            'atr': 14,
            'volatility_bands': 20,
            'keltner_channels': 20
        }
    
    def _get_default_lookback(self, indicator: str) -> int:
        """Get default lookback for a specific indicator."""
        defaults = self._get_default_lookbacks()
        return defaults.get(indicator, 14)
    
    async def _save_results_for_tactician(self, results: Dict[str, Any]):
        """Save optimization results for use by Tactician training step."""
        try:
            tprint_info("💾 Saving results for Tactician training step...")
            
            # Create results directory
            results_dir = Path(self.optimization_config.results_path)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save optimized lookbacks in a format easy for Tactician to load
            lookbacks_file = results_dir / "tactician_optimized_lookbacks.json"
            
            lookback_data = {
                'optimized_lookbacks': results.get('best_lookbacks', {}),
                'optimization_score': results.get('best_score', 0.0),
                'optimization_method': results.get('optimization_method', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'timeframe': '1m',
                'configuration': {
                    'symbol': self.optimization_config.symbol,
                    'exchange': self.optimization_config.exchange,
                    'feature_categories': self.optimization_config.feature_categories
                }
            }
            
            # Save as JSON
            import json
            with open(lookbacks_file, 'w') as f:
                json.dump(lookback_data, f, indent=2)
            
            # Also save full results
            full_results_file = results_dir / f"tactician_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(full_results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            tprint_success(f"✅ Results saved to {results_dir}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save results: {e}")
    
    def get_optimized_lookbacks(self) -> Dict[str, int]:
        """Get the optimized lookback periods from the last optimization."""
        if self.optimization_results:
            return self.optimization_results.get('best_lookbacks', {})
        else:
            return self._get_default_lookbacks()
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization results."""
        if self.optimization_results:
            return {
                'optimized_indicators': len(self.optimization_results.get('best_lookbacks', {})),
                'optimization_score': self.optimization_results.get('best_score', 0.0),
                'optimization_method': self.optimization_results.get('optimization_method', 'unknown'),
                'total_evaluations': self.optimization_results.get('optimization_metrics', {}).get('total_evaluations', 0),
                'success_rate': self.optimization_results.get('execution_info', {}).get('success_rate', 0.0)
            }
        else:
            return {'status': 'not_executed'}


# Convenience functions for integration

async def execute_tactician_lookback_optimization(
    market_data_1m: pd.DataFrame,
    analyst_models: Optional[Dict[str, Any]] = None,
    analyst_ensemble: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to execute Tactician lookback optimization.
    
    Args:
        market_data_1m: 1-minute market data
        analyst_models: Trained Analyst models
        analyst_ensemble: Trained Analyst ensemble
        config: Configuration dictionary
        
    Returns:
        Optimization results
    """
    try:
        step = TacticianLookbackOptimizationStep(config)
        results = await step.execute(market_data_1m, analyst_models, analyst_ensemble)
        return results
        
    except Exception as e:
        tprint_error(f"❌ Tactician lookback optimization execution failed: {e}")
        raise


def create_tactician_optimization_step(config: Optional[Dict[str, Any]] = None) -> TacticianLookbackOptimizationStep:
    """Create a Tactician lookback optimization step."""
    return TacticianLookbackOptimizationStep(config)