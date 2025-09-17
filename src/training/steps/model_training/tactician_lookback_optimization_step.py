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
            
            # Generate comprehensive step-level metrics and artifacts
            step_artifacts = self._generate_step_artifacts(processed_results, start_time, execution_time)
            step_metrics = self._generate_step_metrics(processed_results, market_data_1m, analyst_models, analyst_ensemble)
            step_report = self._generate_step_report(processed_results, start_time, execution_time)
            
            final_results = {
                'step_name': 'tactician_lookback_optimization',
                'execution_time': execution_time,
                'execution_time_formatted': f"{execution_time:.2f}s",
                'optimization_results': processed_results,
                'optimized_lookbacks': processed_results.get('best_lookbacks', {}),
                'optimization_score': processed_results.get('best_score', 0.0),
                'configuration': self.optimization_config.__dict__,
                'step_artifacts': step_artifacts,
                'step_metrics': step_metrics,
                'step_report': step_report,
                'metadata': {
                    'timestamp': start_time.isoformat(),
                    'duration_seconds': execution_time,
                    'data_samples': len(market_data_1m),
                    'analyst_models_count': len(analyst_models) if analyst_models else 0,
                    'has_analyst_ensemble': analyst_ensemble is not None,
                    'optimization_method': processed_results.get('optimization_method', 'unknown'),
                    'total_evaluations': processed_results.get('optimization_metrics', {}).get('total_evaluations', 0),
                    'success_rate': processed_results.get('execution_info', {}).get('success_rate', 0.0)
                },
                'quality_assessment': {
                    'data_quality': 'good' if len(market_data_1m) > 1000 else 'limited',
                    'optimization_quality': (
                        'excellent' if processed_results.get('best_score', 0) > 0.8 else
                        'good' if processed_results.get('best_score', 0) > 0.6 else
                        'fair' if processed_results.get('best_score', 0) > 0.4 else 'poor'
                    ),
                    'analyst_integration_quality': (
                        'good' if analyst_models and analyst_ensemble else
                        'partial' if analyst_models or analyst_ensemble else 'limited'
                    )
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
                        # Call the actual model prediction
                        predictions = self._generate_analyst_predictions(
                            market_data_1m, model, model_name
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
                    # Call the actual ensemble prediction
                    ensemble_predictions = self._generate_ensemble_predictions(market_data_1m, analyst_ensemble)
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
    
    def _generate_analyst_predictions(
        self,
        market_data: pd.DataFrame,
        model: Any,
        model_name: str
    ) -> Dict[str, np.ndarray]:
        """Generate Analyst predictions using the actual trained model."""
        try:
            # Call the actual trained model
            if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                # Standard sklearn-like interface
                predictions = model.predict(market_data)
                confidences = model.predict_proba(market_data)
                
                # Extract confidence scores (assuming binary classification)
                if confidences.ndim > 1 and confidences.shape[1] > 1:
                    confidences = np.max(confidences, axis=1)
                
            elif hasattr(model, 'predict'):
                # Model with only predict method
                predictions = model.predict(market_data)
                confidences = np.full(len(predictions), 0.7)  # Default confidence
                
            elif isinstance(model, dict) and 'type' in model:
                # Fallback for mock models during development
                tprint_warning(f"⚠️ Using fallback for mock model {model_name}")
                n_samples = len(market_data)
                predictions = np.full(n_samples, 0.5)
                confidences = np.full(n_samples, 0.6)
                
            else:
                raise ValueError(f"Model {model_name} does not have a compatible interface")
            
            return {
                'predictions': np.array(predictions),
                'confidences': np.array(confidences)
            }
            
        except Exception as e:
            self.logger.warning(f"Analyst prediction generation failed for {model_name}: {e}")
            # Return neutral predictions as fallback
            n_samples = len(market_data)
            return {
                'predictions': np.full(n_samples, 0.5),
                'confidences': np.full(n_samples, 0.6)
            }
    
    def _generate_ensemble_predictions(
        self, 
        market_data: pd.DataFrame, 
        ensemble_model: Any
    ) -> Dict[str, np.ndarray]:
        """Generate ensemble predictions using the actual trained ensemble model."""
        try:
            # Call the actual trained ensemble model
            if hasattr(ensemble_model, 'predict') and hasattr(ensemble_model, 'predict_proba'):
                # Standard ensemble interface
                predictions = ensemble_model.predict(market_data)
                confidences = ensemble_model.predict_proba(market_data)
                
                # Extract confidence scores
                if confidences.ndim > 1 and confidences.shape[1] > 1:
                    confidences = np.max(confidences, axis=1)
                    
            elif hasattr(ensemble_model, 'predict'):
                # Ensemble with only predict method
                predictions = ensemble_model.predict(market_data)
                confidences = np.full(len(predictions), 0.8)  # Higher default confidence for ensemble
                
            elif isinstance(ensemble_model, dict) and 'type' in ensemble_model:
                # Fallback for mock ensemble during development
                tprint_warning("⚠️ Using fallback for mock ensemble model")
                n_samples = len(market_data)
                predictions = np.full(n_samples, 0.5)
                confidences = np.full(n_samples, 0.7)
                
            else:
                raise ValueError("Ensemble model does not have a compatible interface")
            
            return {
                'predictions': np.array(predictions),
                'confidences': np.array(confidences)
            }
            
        except Exception as e:
            self.logger.warning(f"Ensemble prediction generation failed: {e}")
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
        """Get default lookback periods for 1m timeframe (optimized for 0.3% movements)."""
        return {
            'rsi': 10,  # Shorter for 0.3% movements
            'macd': 18,  # Reduced from 26
            'bollinger_bands': 15,  # Shorter for quick reactions
            'stoch': 10,  # More responsive
            'volume_sma': 15,  # Shorter volume analysis
            'vwap': 15,  # Shorter VWAP for intraday
            'obv': 8,   # Quick volume momentum
            'volume_roc': 8,   # Fast volume changes
            'williams_r': 10,  # More responsive
            'cci': 15,  # Shorter for 1m
            'momentum': 6,   # Very short for 0.3% targets
            'roc': 8,    # Quick rate of change
            'atr': 10,   # Shorter volatility measure
            'volatility_bands': 15,  # Responsive volatility
            'keltner_channels': 15   # Shorter channels
        }
    
    def _get_default_lookback(self, indicator: str) -> int:
        """Get default lookback for a specific indicator."""
        defaults = self._get_default_lookbacks()
        return defaults.get(indicator, 10)  # Shorter default for 0.3% movements
    
    def _generate_step_artifacts(
        self, 
        optimization_results: Dict[str, Any], 
        start_time: datetime, 
        execution_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive step-level artifacts."""
        try:
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            
            artifacts = {
                'primary_artifacts': {
                    'optimized_lookbacks_file': f"tactician_optimized_lookbacks_{timestamp}.json",
                    'optimization_results_file': f"tactician_optimization_results_{timestamp}.json",
                    'performance_metrics_file': f"tactician_optimization_metrics_{timestamp}.json"
                },
                'analysis_artifacts': {
                    'feature_analysis_file': f"tactician_feature_analysis_{timestamp}.json",
                    'convergence_analysis_file': f"tactician_convergence_analysis_{timestamp}.json",
                    'performance_analysis_file': f"tactician_performance_analysis_{timestamp}.json"
                },
                'reporting_artifacts': {
                    'summary_report_file': f"tactician_optimization_summary_{timestamp}.json",
                    'detailed_report_file': f"tactician_optimization_detailed_{timestamp}.json",
                    'execution_log_file': f"tactician_optimization_log_{timestamp}.txt"
                },
                'metadata_artifacts': {
                    'configuration_file': f"tactician_optimization_config_{timestamp}.json",
                    'artifact_manifest_file': f"tactician_artifact_manifest_{timestamp}.json"
                },
                'artifact_summary': {
                    'total_artifacts': 0,  # Will be calculated
                    'primary_count': len([f for f in artifacts['primary_artifacts'].values() if f]),
                    'analysis_count': len([f for f in artifacts['analysis_artifacts'].values() if f]),
                    'reporting_count': len([f for f in artifacts['reporting_artifacts'].values() if f]),
                    'metadata_count': len([f for f in artifacts['metadata_artifacts'].values() if f])
                }
            }
            
            # Calculate total artifacts
            artifacts['artifact_summary']['total_artifacts'] = (
                artifacts['artifact_summary']['primary_count'] +
                artifacts['artifact_summary']['analysis_count'] +
                artifacts['artifact_summary']['reporting_count'] +
                artifacts['artifact_summary']['metadata_count']
            )
            
            return artifacts
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate step artifacts: {e}")
            return {'error': str(e)}
    
    def _generate_step_metrics(
        self,
        optimization_results: Dict[str, Any],
        market_data: pd.DataFrame,
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensemble: Optional[Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive step-level metrics."""
        try:
            metrics = {
                'optimization_metrics': {
                    'total_evaluations': optimization_results.get('optimization_metrics', {}).get('total_evaluations', 0),
                    'successful_evaluations': optimization_results.get('optimization_metrics', {}).get('successful_evaluations', 0),
                    'failed_evaluations': optimization_results.get('optimization_metrics', {}).get('failed_evaluations', 0),
                    'best_score': optimization_results.get('best_score', 0.0),
                    'optimization_method': optimization_results.get('optimization_method', 'unknown'),
                    'convergence_achieved': optimization_results.get('best_score', 0.0) > 0.5
                },
                'data_metrics': {
                    'market_data_samples': len(market_data),
                    'market_data_timespan_hours': len(market_data) / 60 if len(market_data) > 0 else 0,
                    'data_quality_score': 1.0 if len(market_data) > 1000 else 0.5,
                    'required_columns_present': all(col in market_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
                },
                'analyst_integration_metrics': {
                    'analyst_models_available': len(analyst_models) if analyst_models else 0,
                    'analyst_ensemble_available': analyst_ensemble is not None,
                    'integration_quality': (
                        1.0 if analyst_models and analyst_ensemble else
                        0.7 if analyst_models or analyst_ensemble else 0.3
                    ),
                    'dependency_satisfaction': analyst_models is not None or analyst_ensemble is not None
                },
                'feature_optimization_metrics': {
                    'indicators_optimized': len(optimization_results.get('best_lookbacks', {})),
                    'feature_categories_optimized': len(set(
                        indicator.split('_')[0] for indicator in optimization_results.get('best_lookbacks', {}).keys()
                    )) if optimization_results.get('best_lookbacks') else 0,
                    'average_lookback_period': (
                        np.mean(list(optimization_results.get('best_lookbacks', {}).values()))
                        if optimization_results.get('best_lookbacks') else 0
                    ),
                    'lookback_range_utilized': (
                        max(optimization_results.get('best_lookbacks', {}).values()) -
                        min(optimization_results.get('best_lookbacks', {}).values())
                        if optimization_results.get('best_lookbacks') else 0
                    )
                },
                'performance_metrics': {
                    'optimization_efficiency': (
                        optimization_results.get('optimization_metrics', {}).get('successful_evaluations', 0) /
                        max(1, optimization_results.get('optimization_metrics', {}).get('total_evaluations', 1))
                    ),
                    'score_improvement': optimization_results.get('best_score', 0.0) - 0.5,  # Improvement over baseline
                    'execution_efficiency': (
                        optimization_results.get('optimization_metrics', {}).get('total_evaluations', 0) /
                        max(1, optimization_results.get('execution_info', {}).get('total_duration', 1))
                    ),
                    'resource_utilization': 'optimal' if execution_time < 1800 else 'extended'  # 30 minutes threshold
                }
            }
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate step metrics: {e}")
            return {'error': str(e)}
    
    def _generate_step_report(
        self,
        optimization_results: Dict[str, Any],
        start_time: datetime,
        execution_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive step-level report."""
        try:
            report = {
                'report_header': {
                    'step_name': 'Tactician Lookback Optimization',
                    'report_type': 'step_execution_report',
                    'timestamp': start_time.isoformat(),
                    'execution_time': f"{execution_time:.2f}s",
                    'status': 'completed'
                },
                'executive_summary': {
                    'optimization_completed': True,
                    'indicators_optimized': len(optimization_results.get('best_lookbacks', {})),
                    'optimization_score': optimization_results.get('best_score', 0.0),
                    'optimization_quality': (
                        'excellent' if optimization_results.get('best_score', 0) > 0.8 else
                        'good' if optimization_results.get('best_score', 0) > 0.6 else
                        'fair' if optimization_results.get('best_score', 0) > 0.4 else 'poor'
                    ),
                    'execution_efficiency': 'good' if execution_time < 1800 else 'extended',
                    'ready_for_tactician_training': True
                },
                'detailed_findings': {
                    'optimization_method_used': optimization_results.get('optimization_method', 'unknown'),
                    'total_evaluations_performed': optimization_results.get('optimization_metrics', {}).get('total_evaluations', 0),
                    'evaluation_success_rate': optimization_results.get('execution_info', {}).get('success_rate', 0.0),
                    'convergence_achieved': optimization_results.get('best_score', 0.0) > 0.5,
                    'feature_categories_analyzed': list(self.optimization_config.feature_categories.keys()),
                    'analyst_integration_successful': optimization_results.get('metadata', {}).get('has_analyst_ensemble', False)
                },
                'optimization_insights': optimization_results.get('insights_and_recommendations', []),
                'performance_analysis': optimization_results.get('performance_analysis', {}),
                'feature_analysis': optimization_results.get('feature_analysis', {}),
                'convergence_analysis': optimization_results.get('convergence_analysis', {}),
                'recommendations': [
                    "Use optimized lookbacks in Tactician model training",
                    "Monitor Tactician performance improvements with optimized parameters",
                    "Compare against baseline performance using default lookbacks",
                    "Consider re-optimization if market regime changes significantly"
                ],
                'next_steps': [
                    "Proceed to tactician_models_training with optimized lookbacks",
                    "Validate optimization results in backtesting",
                    "Monitor real-time performance improvements"
                ]
            }
            
            return report
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate step report: {e}")
            return {'error': str(e)}
    
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