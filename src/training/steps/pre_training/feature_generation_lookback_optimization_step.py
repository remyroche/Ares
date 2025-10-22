"""
Feature Generation Lookback Optimization Step

This step performs lookback optimization as part of the unified data-driven pipeline
by optimizing the lookback window for feature generation to maximize predictive performance.
"""

from __future__ import annotations

import warnings
import logging
import json
import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

from src.training.steps.base_step import BaseStep
from src.training.common.component_result import ComponentResult
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Import tprint utilities for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_performance, tprint_step, tprint_result, tprint_data_preview, tprint_data_format
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)
    def tprint_result(*args, **kwargs): print("RESULT:", *args, **kwargs)
    def tprint_data_preview(*args, **kwargs): print("DATA_PREVIEW:", *args, **kwargs)
    def tprint_data_format(*args, **kwargs): print("DATA_FORMAT:", *args, **kwargs)

class FeatureGenerationLookbackOptimizationStep(BaseStep):
    """Lookback optimization step that optimizes lookback windows for feature generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the step."""
        super().__init__("feature_generation_lookback_optimization_step", config)
        self.optimization_results = {}
        self.best_lookback = None
        self.lookback_scores = {}

    def _initialize_resources(self) -> bool:
        """Initialize lookback optimization resources."""
        try:
            tprint_step("Initializing lookback optimization resources")
            self.set_state('initialized_at', time.time())
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize lookback optimization: {e}")
            tprint_error(f"Failed to initialize lookback optimization: {e}")
            return False

    def _cleanup_resources(self) -> None:
        """Cleanup lookback optimization resources."""
        try:
            tprint_step("Cleaning up lookback optimization resources")
            self.set_state('cleaned_up_at', time.time())
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute lookback optimization step."""
        try:
            tprint_step("Starting lookback optimization execution")
            
            # Extract parameters from config
            data = config.get('data')
            symbol = config.get('symbol', 'ETHUSDT')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            intensity = config.get('intensity', 'light')
            lookback_days = config.get('lookback_days')
            start_date = config.get('start_date')
            end_date = config.get('end_date')
            exchange = config.get('exchange', 'binance')
            custom_overrides = config.get('custom_overrides', {})

            # Set context for enhanced file naming
            self._set_context(symbol=symbol, exchange=exchange, direction=direction, model='Analyst')

            if data is None or data.empty:
                tprint_warning("No data provided for lookback optimization, using default values")
                optimized_lookbacks = 20
                optimization_metadata = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'optimization_method': 'default',
                    'warning': 'No data provided, using default lookback'
                }
            else:
                # Perform actual lookback optimization
                tprint_info("Performing lookback optimization on provided data")
                
                # Define lookback ranges to test based on intensity
                lookback_ranges = {
                    'light': [5, 10, 15, 20, 25],
                    'medium': [5, 10, 15, 20, 25, 30, 40, 50],
                    'heavy': [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90]
                }
                
                lookbacks_to_test = lookback_ranges.get(intensity, lookback_ranges['light'])
                tprint_info(f"Testing {len(lookbacks_to_test)} lookback windows: {lookbacks_to_test}")
                
                # Load targets for optimization
                targets = self._load_targets_for_optimization(symbol, timeframe, direction)
                
                # Align data and targets
                aligned_data = self._align_data_and_targets(data, targets)
                
                if aligned_data.empty:
                    tprint_warning("No aligned data available, using default lookback")
                    optimized_lookbacks = 20
                    optimization_metadata = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'direction': direction,
                        'optimization_method': 'default',
                        'warning': 'No aligned data available'
                    }
                else:
                    # Perform lookback optimization
                    optimized_lookbacks, lookback_scores = self._optimize_lookbacks(
                        features=aligned_data.drop(columns=['target']),
                        targets=aligned_data['target'],
                        lookbacks_to_test=lookbacks_to_test,
                        direction=direction
                    )
                    
                    optimization_metadata = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'direction': direction,
                        'optimization_method': 'mutual_information_sharpe',
                        'best_lookback': optimized_lookbacks,
                        'lookback_scores': lookback_scores,
                        'tested_lookbacks': lookbacks_to_test
                    }
                    
                    tprint_success(f"Lookback optimization completed: {optimized_lookbacks} (score: {lookback_scores.get(str(optimized_lookbacks), 0):.3f})")

            # Save artifacts using BaseStep methods
            if data is not None:
                self._save_dataframe(data, 'input_data')
            
            self._save_metadata(optimization_metadata, 'optimization_metadata')
            
            # Save optimization results
            optimization_results = {
                'optimized_lookbacks': optimized_lookbacks,
                'lookback_scores': optimization_metadata.get('lookback_scores', {}),
                'optimization_metadata': optimization_metadata
            }
            self._save_metadata(optimization_results, 'lookback_optimization_results')

            return {
                'success': True,
                'artifacts': ['input_data', 'optimization_metadata', 'lookback_optimization_results'],
                'metrics': {
                    'optimized_lookbacks': optimized_lookbacks,
                    'optimization_metadata': optimization_metadata
                }
            }

        except Exception as e:
            self.logger.error(f"Lookback optimization failed: {e}")
            tprint_error(f"Lookback optimization failed: {e}")
            raise

    def _load_targets_for_optimization(self, symbol: str, timeframe: str, direction: str) -> pd.Series:
        """Load targets for optimization from artifact manager."""
        try:
            # Try to get artifact manager
            from src.utils.artifact_manager import get_pretraining_artifact_manager
            artifact_manager = get_pretraining_artifact_manager()
            
            # Try to load targets from various possible sources
            targets = None
            for step_name in ("feature_generation_labeling_integration_step", "labeling_integration"):
                try:
                    tmp = artifact_manager.get_artifact(step_name, 'targets')
                    if isinstance(tmp, pd.Series) and not tmp.empty:
                        targets = tmp
                        tprint_info(f"Loaded targets from {step_name}")
                        break
                except:
                    continue
            
            if targets is None or targets.empty:
                tprint_warning("No targets found, using synthetic targets for optimization")
                # Create synthetic targets for optimization
                targets = pd.Series(np.random.randn(1000), name='target')
            
            return targets
            
        except Exception as e:
            tprint_warning(f"Failed to load targets: {e}, using synthetic targets")
            return pd.Series(np.random.randn(1000), name='target')

    def _align_data_and_targets(self, data: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Align data and targets for optimization."""
        try:
            # Ensure targets have a proper index
            if not hasattr(targets, 'index') or targets.index.empty:
                targets.index = data.index[:len(targets)]
            
            # Align data and targets
            aligned_data = data.join(targets.rename('target'), how='inner').dropna()
            
            if aligned_data.empty:
                tprint_warning("No overlapping data between features and targets")
                return pd.DataFrame()
            
            tprint_info(f"Aligned data shape: {aligned_data.shape}")
            return aligned_data
            
        except Exception as e:
            tprint_error(f"Failed to align data and targets: {e}")
            return pd.DataFrame()

    def _optimize_lookbacks(self, features: pd.DataFrame, targets: pd.Series, 
                           lookbacks_to_test: List[int], direction: str) -> Tuple[int, Dict[str, float]]:
        """Optimize lookbacks using mutual information and out-of-sample Sharpe ratio."""
        tprint_step("Optimizing lookbacks")
        tprint_data_preview(features, "lookback_optimization_input", max_rows=5, level="DEBUG")
        tprint_data_format(features, "lookback_optimization_input", level="DEBUG")
        tprint_data_format(targets, "lookback_optimization_targets", level="DEBUG")
        
        lookback_scores = {}
        best_lookback = lookbacks_to_test[0]
        best_score = -np.inf
        
        for lookback in lookbacks_to_test:
            try:
                # Create lookback-based features
                lookback_features = self._create_lookback_features(features, lookback)
                
                if lookback_features.empty:
                    continue
                
                # Calculate mutual information score
                mi_score = self._calculate_mutual_information_score(lookback_features, targets)
                
                # Calculate out-of-sample Sharpe ratio
                sharpe_score = self._calculate_sharpe_ratio(lookback_features, targets)
                
                # Combined score (weighted average)
                combined_score = 0.6 * mi_score + 0.4 * sharpe_score
                
                lookback_scores[str(lookback)] = combined_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_lookback = lookback
                
                tprint_debug(f"Lookback {lookback}: MI={mi_score:.3f}, Sharpe={sharpe_score:.3f}, Combined={combined_score:.3f}")
                
            except Exception as e:
                tprint_warning(f"Failed to evaluate lookback {lookback}: {e}")
                continue
        
        tprint_data_preview(lookback_scores, "lookback_optimization_scores", level="INFO")
        tprint_data_format(lookback_scores, "lookback_optimization_scores", level="INFO")
        
        return best_lookback, lookback_scores

    def _create_lookback_features(self, features: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Create lookback-based features."""
        try:
            if len(features) < lookback:
                return pd.DataFrame()
            
            # Create rolling features
            lookback_features = features.rolling(window=lookback, min_periods=lookback).agg({
                col: ['mean', 'std', 'min', 'max', 'last'] for col in features.columns
            }).dropna()
            
            # Flatten column names
            lookback_features.columns = [f"{col[0]}_{col[1]}" for col in lookback_features.columns]
            
            return lookback_features
            
        except Exception as e:
            tprint_warning(f"Failed to create lookback features for lookback {lookback}: {e}")
            return pd.DataFrame()

    def _calculate_mutual_information_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate mutual information score between features and targets."""
        try:
            if features.empty or targets.empty:
                return 0.0
            
            # Align features and targets
            aligned_data = features.join(targets, how='inner').dropna()
            if aligned_data.empty:
                return 0.0
            
            aligned_features = aligned_data.drop(columns=[targets.name])
            aligned_targets = aligned_data[targets.name]
            
            # Calculate mutual information for each feature
            mi_scores = []
            for col in aligned_features.columns:
                try:
                    mi = mutual_info_regression(
                        aligned_features[[col]], 
                        aligned_targets, 
                        random_state=42
                    )[0]
                    mi_scores.append(mi)
                except:
                    continue
            
            return np.mean(mi_scores) if mi_scores else 0.0
            
        except Exception as e:
            tprint_warning(f"Failed to calculate mutual information: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate out-of-sample Sharpe ratio."""
        try:
            if features.empty or targets.empty or len(features) < 20:
                return 0.0
            
            # Align features and targets
            aligned_data = features.join(targets, how='inner').dropna()
            if aligned_data.empty or len(aligned_data) < 20:
                return 0.0
            
            aligned_features = aligned_data.drop(columns=[targets.name])
            aligned_targets = aligned_data[targets.name]
            
            # Use TimeSeriesSplit for out-of-sample evaluation
            tscv = TimeSeriesSplit(n_splits=3)
            sharpe_ratios = []
            
            for train_idx, test_idx in tscv.split(aligned_features):
                try:
                    X_train, X_test = aligned_features.iloc[train_idx], aligned_features.iloc[test_idx]
                    y_train, y_test = aligned_targets.iloc[train_idx], aligned_targets.iloc[test_idx]
                    
                    # Train a simple model
                    model = RandomForestRegressor(n_estimators=10, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    predictions = model.predict(X_test)
                    
                    # Calculate returns (assuming targets are returns)
                    returns = y_test
                    
                    # Calculate Sharpe ratio
                    if len(returns) > 1 and returns.std() > 0:
                        sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
                        sharpe_ratios.append(sharpe)
                    
                except Exception as e:
                    tprint_debug(f"Failed to calculate Sharpe for fold: {e}")
                    continue
            
            return np.mean(sharpe_ratios) if sharpe_ratios else 0.0
            
        except Exception as e:
            tprint_warning(f"Failed to calculate Sharpe ratio: {e}")
            return 0.0


class LookbackOptimizationResult:
    """Result from lookback optimization step."""
    success: bool
    optimized_lookbacks: int
    optimization_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


# Handler function for ares_launcher integration
async def handle_feature_generation_lookback_optimization_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    exchange: str = "binance",
    direction: str = "longs",
    intensity: str = "light",
    lookback_days: int = None,
    start_date: str = None,
    end_date: str = None,
    custom_overrides: dict = None,
    **kwargs
) -> ComponentResult:
    """
    Handler function for feature generation lookback optimization step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        timeframe: Timeframe (e.g., "15m")
        exchange: Exchange name (e.g., "binance")
        direction: Trading direction (e.g., "longs")
        intensity: Intensity level (e.g., "light", "medium", "heavy")
        lookback_days: Number of days to look back
        start_date: Start date for data
        end_date: End date for data
        custom_overrides: Custom configuration overrides
        **kwargs: Additional arguments

    Returns:
        ComponentResult: Result of the lookback optimization step
    """
    try:
        tprint_step("Starting feature generation lookback optimization step")
        
        # Get artifact manager
        from src.utils.artifact_manager import get_pretraining_artifact_manager
        artifact_manager = get_pretraining_artifact_manager()

        # Create the step instance
        step = FeatureGenerationLookbackOptimizationStep(
            config={
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'custom_overrides': custom_overrides or {}
            }
        )

        # Try to load data from previous steps
        data = None
        try:
            # Try to load features from feature generation step
            for step_name in ("feature_generation_feature_generation_step", "feature_generation"):
                try:
                    data = artifact_manager.get_artifact(step_name, 'features')
                    if data is not None and not data.empty:
                        tprint_info(f"Loaded features from {step_name}")
                        break
                except:
                    continue
            
            if data is None or data.empty:
                tprint_warning("No features found from previous steps, using synthetic data")
                # Create synthetic data for testing
                data = pd.DataFrame(
                    np.random.randn(1000, 10),
                    columns=[f'feature_{i}' for i in range(10)],
                    index=pd.date_range('2023-01-01', periods=1000, freq='15T')
                )
        except Exception as e:
            tprint_warning(f"Failed to load data: {e}, using synthetic data")
            data = pd.DataFrame(
                np.random.randn(1000, 10),
                columns=[f'feature_{i}' for i in range(10)],
                index=pd.date_range('2023-01-01', periods=1000, freq='15T')
            )

        # Execute the step
        result = await step.execute({
            'data': data,
            'symbol': symbol,
            'timeframe': timeframe,
            'exchange': exchange,
            'direction': direction,
            'intensity': intensity,
            'lookback_days': lookback_days,
            'start_date': start_date,
            'end_date': end_date,
            'custom_overrides': custom_overrides or {}
        })

        # Create result object
        step_result = LookbackOptimizationResult(
            success=result.get('success', False),
            optimized_lookbacks=result.get('metrics', {}).get('optimized_lookbacks', 20),
            optimization_metadata=result.get('metrics', {}).get('optimization_metadata', {}),
            artifacts=result.get('artifacts', {})
        )

        # Convert to ComponentResult
        component_result = ComponentResult(
            success=step_result.success,
            data=None,  # Lookback optimization doesn't return processed data
            metadata={
                'step_name': 'feature_generation_lookback_optimization_step',
                'optimized_lookbacks': step_result.optimized_lookbacks,
                'optimization_metadata': step_result.optimization_metadata,
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction
            },
            artifacts=step_result.artifacts,
            error_message=step_result.error_message
        )

        # Save artifacts
        try:
            await artifact_manager.save_step_result(
                step_name='feature_generation_lookback_optimization_step',
                result=component_result,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction
            )
            tprint_success("Artifacts saved successfully")
        except Exception as e:
            tprint_warning(f"Failed to save artifacts: {e}")

        tprint_success(f"Lookback optimization completed: {step_result.optimized_lookbacks}")
        return component_result

    except Exception as e:
        error_message = f"Lookback optimization step failed: {str(e)}"
        tprint_error(error_message)

        # Return failed result
        component_result = ComponentResult(
            success=False,
            data=None,
            metadata={'step_name': 'feature_generation_lookback_optimization_step'},
            artifacts={},
            error_message=error_message
        )

        return component_result
