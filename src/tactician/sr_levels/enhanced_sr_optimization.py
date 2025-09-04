#!/usr/bin/env python3
"""Enhanced S/R Optimization Module.

This module integrates all enhanced S/R detection, validation, and confluence
components to provide comprehensive S/R optimization for step2_5.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import warnings
warnings.filterwarnings('ignore')

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger

# Import enhanced components
from .enhanced_sr_detection import EnhancedSRDetector, SRLevel
from .enhanced_sr_validation import EnhancedSRValidator, ValidationResult, BacktestResult
from .enhanced_sr_confluence import EnhancedSRConfluenceDetector, ConfluenceResult, ConfluenceLevel


@dataclass
class EnhancedOptimizationResult:
    """Enhanced S/R optimization result."""
    # Detection results
    detected_levels: List[SRLevel] = field(default_factory=list)
    confluence_levels: List[ConfluenceLevel] = field(default_factory=list)
    
    # Validation results
    validation_results: List[ValidationResult] = field(default_factory=list)
    backtest_result: Optional[BacktestResult] = None
    
    # Performance metrics
    optimization_score: float = 0.0
    detection_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    confluence_quality: float = 0.0
    
    # Configuration
    optimized_parameters: Dict[str, Any] = field(default_factory=dict)
    timeframe_weights: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    optimization_time: float = 0.0
    n_trials: int = 0
    market_regime: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EnhancedSROptimizer:
    """Enhanced S/R optimizer integrating all advanced components."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced S/R optimizer."""
        self.config = config
        self.logger = system_logger.getChild("EnhancedSROptimizer")
        
        # Initialize components
        self.detector = EnhancedSRDetector(config)
        self.validator = EnhancedSRValidator(config)
        self.confluence_detector = EnhancedSRConfluenceDetector(config)
        
        # Optimization parameters
        self.optimization_config = config.get("enhanced_sr_optimization", {})
        self.n_trials = self.optimization_config.get("n_trials", 50)
        self.cv_folds = self.optimization_config.get("cv_folds", 5)
        self.test_size = self.optimization_config.get("test_size", 0.2)
        
        # Performance thresholds
        self.performance_thresholds = self.optimization_config.get("performance_thresholds", {
            "min_optimization_score": 0.7,
            "min_detection_accuracy": 0.6,
            "min_validation_accuracy": 0.6,
            "min_confluence_quality": 0.5
        })
        
        # Results storage
        self.optimization_results: List[EnhancedOptimizationResult] = []
        self.best_result: Optional[EnhancedOptimizationResult] = None
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="initialize enhanced S/R optimizer"
    )
    async def initialize(self) -> bool:
        """Initialize the enhanced S/R optimizer."""
        try:
            self.logger.info("🚀 Initializing Enhanced S/R Optimizer...")
            
            # Validate configuration
            if not self._validate_configuration():
                return False
            
            self.logger.info("✅ Enhanced S/R Optimizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced S/R optimizer initialization failed: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate optimization configuration."""
        try:
            # Check required parameters
            required_params = ["n_trials", "cv_folds", "test_size"]
            for param in required_params:
                if param not in self.optimization_config:
                    self.logger.error(f"Missing required parameter: {param}")
                    return False
            
            # Validate parameter ranges
            if self.n_trials <= 0:
                self.logger.error("n_trials must be positive")
                return False
            
            if self.cv_folds < 2:
                self.logger.error("cv_folds must be at least 2")
                return False
            
            if not 0 < self.test_size < 1:
                self.logger.error("test_size must be between 0 and 1")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="run enhanced S/R optimization"
    )
    @traced(span_name="EnhancedSR.optimize")
    async def optimize_sr_detection(
        self,
        market_data: pd.DataFrame,
        multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Optional[EnhancedOptimizationResult]:
        """
        Run comprehensive enhanced S/R optimization.
        
        Args:
            market_data: Main market data for optimization
            multi_timeframe_data: Multi-timeframe data for confluence analysis
            
        Returns:
            Enhanced optimization result
        """
        try:
            self.logger.info("🎯 Starting Enhanced S/R Optimization...")
            start_time = datetime.now()
            
            # Step 1: Enhanced S/R Detection
            self.logger.info("🔍 Step 1: Enhanced S/R Detection...")
            detected_levels = self.detector.detect_sr_levels(market_data)
            
            if not detected_levels:
                self.logger.warning("No S/R levels detected")
                return None
            
            self.logger.info(f"✅ Detected {len(detected_levels)} S/R levels")
            
            # Step 2: Multi-timeframe Confluence Analysis
            confluence_levels = []
            if multi_timeframe_data:
                self.logger.info("🔗 Step 2: Multi-timeframe Confluence Analysis...")
                
                # Prepare multi-timeframe levels
                multi_timeframe_levels = {}
                for timeframe, data in multi_timeframe_data.items():
                    if len(data) > 100:  # Ensure sufficient data
                        tf_levels = self.detector.detect_sr_levels(data)
                        multi_timeframe_levels[timeframe] = tf_levels
                
                # Detect confluence
                confluence_result = self.confluence_detector.detect_confluence(multi_timeframe_levels)
                if confluence_result:
                    confluence_levels = confluence_result.confluence_levels
                    self.logger.info(f"✅ Detected {len(confluence_levels)} confluence levels")
                else:
                    self.logger.warning("No confluence levels detected")
            else:
                self.logger.info("⚠️ No multi-timeframe data provided, skipping confluence analysis")
            
            # Step 3: Enhanced Validation and Backtesting
            self.logger.info("✅ Step 3: Enhanced Validation and Backtesting...")
            validation_results = self.validator.validate_sr_levels(detected_levels, market_data)
            
            # Run backtest
            backtest_result = self.validator.run_comprehensive_backtest(detected_levels, market_data)
            
            # Step 4: Parameter Optimization
            self.logger.info("⚙️ Step 4: Parameter Optimization...")
            optimized_parameters = await self._optimize_parameters(
                detected_levels, validation_results, backtest_result
            )
            
            # Step 5: Calculate Performance Metrics
            self.logger.info("📊 Step 5: Calculate Performance Metrics...")
            performance_metrics = self._calculate_performance_metrics(
                detected_levels, validation_results, backtest_result, confluence_levels
            )
            
            # Step 6: Create Enhanced Result
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            result = EnhancedOptimizationResult(
                detected_levels=detected_levels,
                confluence_levels=confluence_levels,
                validation_results=validation_results,
                backtest_result=backtest_result,
                optimization_score=performance_metrics['optimization_score'],
                detection_accuracy=performance_metrics['detection_accuracy'],
                validation_accuracy=performance_metrics['validation_accuracy'],
                confluence_quality=performance_metrics['confluence_quality'],
                optimized_parameters=optimized_parameters,
                timeframe_weights=self.confluence_detector.timeframe_weights,
                optimization_time=optimization_time,
                n_trials=self.n_trials,
                market_regime=self._detect_market_regime(market_data)
            )
            
            # Store result
            self.optimization_results.append(result)
            if not self.best_result or result.optimization_score > self.best_result.optimization_score:
                self.best_result = result
            
            self.logger.info(f"✅ Enhanced S/R Optimization completed in {optimization_time:.2f}s")
            self.logger.info(f"📊 Optimization Score: {result.optimization_score:.4f}")
            self.logger.info(f"🎯 Detection Accuracy: {result.detection_accuracy:.4f}")
            self.logger.info(f"✅ Validation Accuracy: {result.validation_accuracy:.4f}")
            self.logger.info(f"🔗 Confluence Quality: {result.confluence_quality:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced S/R optimization failed: {e}")
            return None
    
    async def _optimize_parameters(
        self,
        detected_levels: List[SRLevel],
        validation_results: List[ValidationResult],
        backtest_result: Optional[BacktestResult]
    ) -> Dict[str, Any]:
        """Optimize S/R detection parameters."""
        try:
            self.logger.info("🔧 Optimizing S/R detection parameters...")
            
            # Analyze current performance
            current_performance = self._analyze_current_performance(
                detected_levels, validation_results, backtest_result
            )
            
            # Generate parameter suggestions
            parameter_suggestions = self._generate_parameter_suggestions(current_performance)
            
            # Run parameter optimization trials
            best_parameters = await self._run_parameter_trials(parameter_suggestions)
            
            return best_parameters
            
        except Exception as e:
            self.logger.warning(f"Parameter optimization failed: {e}")
            return {}
    
    def _analyze_current_performance(
        self,
        detected_levels: List[SRLevel],
        validation_results: List[ValidationResult],
        backtest_result: Optional[BacktestResult]
    ) -> Dict[str, float]:
        """Analyze current S/R detection performance."""
        try:
            performance = {
                'detection_rate': len(detected_levels) / 1000.0 if detected_levels else 0.0,
                'avg_strength': np.mean([level.strength for level in detected_levels]) if detected_levels else 0.0,
                'avg_validation_score': np.mean([r.validation_score for r in validation_results]) if validation_results else 0.0,
                'avg_bounce_rate': np.mean([r.bounce_rate for r in validation_results]) if validation_results else 0.0,
                'avg_false_breakout_rate': np.mean([r.false_breakout_rate for r in validation_results]) if validation_results else 0.0
            }
            
            if backtest_result:
                performance.update({
                    'sharpe_ratio': backtest_result.sharpe_ratio,
                    'win_rate': backtest_result.win_rate,
                    'profit_factor': backtest_result.profit_factor,
                    'max_drawdown': backtest_result.max_drawdown
                })
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Performance analysis failed: {e}")
            return {}
    
    def _generate_parameter_suggestions(self, performance: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate parameter optimization suggestions."""
        try:
            suggestions = []
            
            # Base parameters
            base_params = {
                'min_touches': 3,
                'touch_proximity_threshold': 0.002,
                'min_strength': 0.6,
                'volume_spike_threshold': 1.5,
                'fractal_period': 5,
                'pivot_period': 10
            }
            
            # Adjust parameters based on performance
            if performance.get('detection_rate', 0) < 0.1:
                # Too few detections, relax parameters
                suggestions.append({
                    **base_params,
                    'min_touches': 2,
                    'touch_proximity_threshold': 0.003,
                    'min_strength': 0.5
                })
            
            if performance.get('avg_false_breakout_rate', 0) > 0.4:
                # Too many false breakouts, tighten parameters
                suggestions.append({
                    **base_params,
                    'min_touches': 4,
                    'touch_proximity_threshold': 0.0015,
                    'min_strength': 0.7
                })
            
            if performance.get('avg_bounce_rate', 0) < 0.5:
                # Low bounce rate, adjust thresholds
                suggestions.append({
                    **base_params,
                    'bounce_threshold': 0.0008,
                    'breakout_threshold': 0.003
                })
            
            # Add random variations
            for _ in range(5):
                suggestion = base_params.copy()
                suggestion['min_touches'] = np.random.randint(2, 6)
                suggestion['touch_proximity_threshold'] = np.random.uniform(0.001, 0.004)
                suggestion['min_strength'] = np.random.uniform(0.5, 0.8)
                suggestion['volume_spike_threshold'] = np.random.uniform(1.2, 2.0)
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            self.logger.warning(f"Parameter suggestion generation failed: {e}")
            return [base_params]
    
    async def _run_parameter_trials(self, parameter_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run parameter optimization trials."""
        try:
            best_parameters = {}
            best_score = 0.0
            
            for i, params in enumerate(parameter_suggestions):
                try:
                    # Create temporary detector with new parameters
                    temp_config = self.config.copy()
                    temp_config.update(params)
                    temp_detector = EnhancedSRDetector(temp_config)
                    
                    # Test parameters (simplified evaluation)
                    score = self._evaluate_parameters(temp_detector, params)
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = params
                    
                    self.logger.info(f"Trial {i+1}/{len(parameter_suggestions)}: Score = {score:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Parameter trial {i+1} failed: {e}")
                    continue
            
            self.logger.info(f"✅ Best parameters found with score: {best_score:.4f}")
            return best_parameters
            
        except Exception as e:
            self.logger.warning(f"Parameter trials failed: {e}")
            return {}
    
    def _evaluate_parameters(self, detector: EnhancedSRDetector, params: Dict[str, Any]) -> float:
        """Evaluate parameter set performance."""
        try:
            # Simple evaluation based on parameter balance
            score = 0.0
            
            # Touch count balance
            min_touches = params.get('min_touches', 3)
            if 2 <= min_touches <= 4:
                score += 0.2
            
            # Proximity threshold balance
            proximity = params.get('touch_proximity_threshold', 0.002)
            if 0.001 <= proximity <= 0.003:
                score += 0.2
            
            # Strength threshold balance
            strength = params.get('min_strength', 0.6)
            if 0.5 <= strength <= 0.7:
                score += 0.2
            
            # Volume threshold balance
            volume = params.get('volume_spike_threshold', 1.5)
            if 1.2 <= volume <= 2.0:
                score += 0.2
            
            # Fractal period balance
            fractal = params.get('fractal_period', 5)
            if 3 <= fractal <= 7:
                score += 0.1
            
            # Pivot period balance
            pivot = params.get('pivot_period', 10)
            if 8 <= pivot <= 15:
                score += 0.1
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Parameter evaluation failed: {e}")
            return 0.0
    
    def _calculate_performance_metrics(
        self,
        detected_levels: List[SRLevel],
        validation_results: List[ValidationResult],
        backtest_result: Optional[BacktestResult],
        confluence_levels: List[ConfluenceLevel]
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            metrics = {}
            
            # Detection accuracy
            if detected_levels:
                avg_strength = np.mean([level.strength for level in detected_levels])
                high_quality_levels = len([l for l in detected_levels if l.strength > 0.7])
                detection_accuracy = (avg_strength + (high_quality_levels / len(detected_levels))) / 2
            else:
                detection_accuracy = 0.0
            
            # Validation accuracy
            if validation_results:
                avg_validation = np.mean([r.validation_score for r in validation_results])
                avg_bounce_rate = np.mean([r.bounce_rate for r in validation_results])
                avg_false_breakout = np.mean([r.false_breakout_rate for r in validation_results])
                validation_accuracy = (avg_validation + avg_bounce_rate + (1 - avg_false_breakout)) / 3
            else:
                validation_accuracy = 0.0
            
            # Confluence quality
            if confluence_levels:
                avg_confluence = np.mean([l.confluence_score for l in confluence_levels])
                multi_timeframe_levels = len([l for l in confluence_levels if len(l.timeframes) >= 3])
                confluence_quality = (avg_confluence + (multi_timeframe_levels / len(confluence_levels))) / 2
            else:
                confluence_quality = 0.0
            
            # Overall optimization score
            optimization_score = (detection_accuracy + validation_accuracy + confluence_quality) / 3
            
            # Add backtest metrics if available
            if backtest_result:
                optimization_score = (optimization_score + 
                                    (backtest_result.win_rate + 
                                     min(backtest_result.sharpe_ratio / 2, 0.5) + 
                                     min(backtest_result.profit_factor / 3, 0.5)) / 3) / 2
            
            metrics.update({
                'optimization_score': optimization_score,
                'detection_accuracy': detection_accuracy,
                'validation_accuracy': validation_accuracy,
                'confluence_quality': confluence_quality
            })
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Performance metrics calculation failed: {e}")
            return {
                'optimization_score': 0.0,
                'detection_accuracy': 0.0,
                'validation_accuracy': 0.0,
                'confluence_quality': 0.0
            }
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime."""
        try:
            if len(market_data) < 100:
                return "unknown"
            
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate trend
            sma_20 = market_data['close'].rolling(20).mean()
            sma_50 = market_data['close'].rolling(50).mean()
            trend = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            
            # Classify regime
            if volatility > 0.3:
                if abs(trend) > 0.05:
                    return "trending_high_vol"
                else:
                    return "ranging_high_vol"
            else:
                if abs(trend) > 0.05:
                    return "trending_low_vol"
                else:
                    return "ranging_low_vol"
                    
        except Exception as e:
            self.logger.warning(f"Market regime detection failed: {e}")
            return "unknown"
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        try:
            if not self.optimization_results:
                return {"message": "No optimization results available"}
            
            latest_result = self.optimization_results[-1]
            best_result = self.best_result
            
            summary = {
                "total_optimizations": len(self.optimization_results),
                "latest_optimization": {
                    "timestamp": latest_result.timestamp,
                    "optimization_score": latest_result.optimization_score,
                    "detection_accuracy": latest_result.detection_accuracy,
                    "validation_accuracy": latest_result.validation_accuracy,
                    "confluence_quality": latest_result.confluence_quality,
                    "detected_levels": len(latest_result.detected_levels),
                    "confluence_levels": len(latest_result.confluence_levels),
                    "optimization_time": latest_result.optimization_time
                },
                "best_optimization": {
                    "optimization_score": best_result.optimization_score,
                    "detection_accuracy": best_result.detection_accuracy,
                    "validation_accuracy": best_result.validation_accuracy,
                    "confluence_quality": best_result.confluence_quality,
                    "market_regime": best_result.market_regime
                } if best_result else None,
                "performance_thresholds": self.performance_thresholds
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Optimization summary generation failed: {e}")
            return {"error": str(e)}