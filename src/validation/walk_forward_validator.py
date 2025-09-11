
import pandas as pd
import numpy as np
"""Walk-Forward Validation System for preventing overfitting."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from ..utils.common_operations import ensure_directory, safe_json_dump, validate_finite, validate_positive, validate_range, safe_float, safe_int
from ..utils.logger import system_logger
from ..custom_types.validation import TypeValidator, RuntimeTypeError
import logging
import time

logger = system_logger.getChild('WalkForwardValidator')

@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {'window_id': self.window_id, 'train_start': self.train_start.isoformat(), 'train_end': self.train_end.isoformat(), 'test_start': self.test_start.isoformat(), 'test_end': self.test_end.isoformat(), 'train_days': (self.train_end - self.train_start).days, 'test_days': (self.test_end - self.test_start).days}

class WalkForwardValidator:
    """Implements walk-forward validation for trading strategies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        # Input validation
        if not isinstance(config, dict):
            raise RuntimeTypeError(dict, config, 'WalkForwardValidator config')
        
        self.config = config
        self.logger = system_logger.getChild('WalkForwardValidator')
        
        # Validate and set configuration parameters with bounds checking
        self.train_period_days = self._validate_positive_int(config.get('train_period_days', 365), 'train_period_days', min_val=30, max_val=3650)
        self.test_period_days = self._validate_positive_int(config.get('test_period_days', 30), 'test_period_days', min_val=1, max_val=365)
        self.step_days = self._validate_positive_int(config.get('step_days', 30), 'step_days', min_val=1, max_val=365)
        self.min_train_samples = self._validate_positive_int(config.get('min_train_samples', 1000), 'min_train_samples', min_val=10, max_val=100000)
        self.regime_aware = self._validate_bool(config.get('regime_aware', True), 'regime_aware')
        self.min_samples_per_regime = self._validate_positive_int(config.get('min_samples_per_regime', 500), 'min_samples_per_regime', min_val=5, max_val=50000)
        self.adaptive_windows = self._validate_bool(config.get('adaptive_windows', True), 'adaptive_windows')
        self.volatility_threshold = self._validate_positive_float(config.get('volatility_threshold', 0.03), 'volatility_threshold', min_val=0.001, max_val=1.0)
        self.max_acceptable_degradation = self._validate_positive_float(config.get('max_acceptable_degradation', 0.3), 'max_acceptable_degradation', min_val=0.0, max_val=1.0)
        self.min_out_sample_sharpe = self._validate_float(config.get('min_out_sample_sharpe', 0.5), 'min_out_sample_sharpe', min_val=-5.0, max_val=5.0)
        
        # Validate results directory
        results_dir_str = config.get('results_dir', 'validation_results')
        if not isinstance(results_dir_str, (str, Path)):
            raise RuntimeTypeError(Union[str, Path], results_dir_str, 'results_dir')
        self.results_dir = Path(results_dir_str)
        
        # Ensure results directory exists
        if not ensure_directory(self.results_dir):
            self.logger.warning(f"Failed to create results directory: {self.results_dir}")
        
        self.logger.info(f"WalkForwardValidator initialized with train_period={self.train_period_days}, test_period={self.test_period_days}, step_days={self.step_days}")
    
    def _validate_positive_int(self, value: Any, name: str, min_val: int = 1, max_val: int = None) -> int:
        """Validate positive integer parameter."""
        try:
            val = safe_int(value, 0)
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
            if val < min_val:
                raise ValueError(f"{name} must be >= {min_val}, got {val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{name} must be <= {max_val}, got {val}")
            return val
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")
    
    def _validate_positive_float(self, value: Any, name: str, min_val: float = 0.0, max_val: float = None) -> float:
        """Validate positive float parameter."""
        try:
            val = safe_float(value, 0.0)
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")
            if val < min_val:
                raise ValueError(f"{name} must be >= {min_val}, got {val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{name} must be <= {max_val}, got {val}")
            return val
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")
    
    def _validate_float(self, value: Any, name: str, min_val: float = None, max_val: float = None) -> float:
        """Validate float parameter with optional bounds."""
        try:
            val = safe_float(value, 0.0)
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
            if min_val is not None and val < min_val:
                raise ValueError(f"{name} must be >= {min_val}, got {val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{name} must be <= {max_val}, got {val}")
            return val
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")
    
    def _validate_bool(self, value: Any, name: str) -> bool:
        """Validate boolean parameter."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError(f"{name} must be a boolean, got {type(value)}")

    def generate_walk_forward_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """Generate walk-forward validation windows."""
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise RuntimeTypeError(pd.DataFrame, data, 'generate_walk_forward_windows data')
        
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
        
        # Check for required columns
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        try:
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            
            # Validate datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Data must have a datetime index or 'timestamp' column")
            
            start_date = data.index.min()
            end_date = data.index.max()
            
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            # Check if we have enough data for at least one window
            min_required_days = self.train_period_days + self.test_period_days
            available_days = (end_date - start_date).days
            if available_days < min_required_days:
                raise ValueError(f"Insufficient data: need at least {min_required_days} days, have {available_days}")
            
            windows = []
            window_id = 0
            train_end = start_date + timedelta(days=self.train_period_days)
            
            while train_end + timedelta(days=self.test_period_days) <= end_date:
                train_start = start_date
                test_start = train_end
                test_end = test_start + timedelta(days=self.test_period_days)
                
                if self.adaptive_windows:
                    try:
                        window_params = self._adjust_window_for_volatility(data, train_start, train_end)
                        if window_params:
                            train_start = window_params['train_start']
                            test_end = window_params['test_end']
                    except Exception as e:
                        self.logger.warning(f"Failed to adjust window for volatility: {e}")
                
                window = WalkForwardWindow(
                    train_start=train_start, 
                    train_end=train_end, 
                    test_start=test_start, 
                    test_end=test_end, 
                    window_id=window_id
                )
                windows.append(window)
                train_end += timedelta(days=self.step_days)
                window_id += 1
            
            if not windows:
                raise ValueError("No valid windows could be generated with the given parameters")
            
            self.logger.info(f'Generated {len(windows)} walk-forward windows')
            return windows
            
        except Exception as e:
            self.logger.error(f"Error generating walk-forward windows: {e}")
            raise

    def _adjust_window_for_volatility(self, data: pd.DataFrame, train_start: datetime, train_end: datetime) -> Optional[Dict[str, datetime]]:
        """Adjust window size based on market volatility."""
        try:
            # Input validation
            if not isinstance(data, pd.DataFrame):
                raise RuntimeTypeError(pd.DataFrame, data, '_adjust_window_for_volatility data')
            if not isinstance(train_start, datetime):
                raise RuntimeTypeError(datetime, train_start, '_adjust_window_for_volatility train_start')
            if not isinstance(train_end, datetime):
                raise RuntimeTypeError(datetime, train_end, '_adjust_window_for_volatility train_end')
            
            if train_start >= train_end:
                raise ValueError("train_start must be before train_end")
            
            train_data = data[train_start:train_end]
            if train_data.empty:
                self.logger.warning("No training data available for volatility adjustment")
                return None
            
            if 'close' not in train_data.columns:
                self.logger.warning("No 'close' column available for volatility calculation")
                return None
            
            returns = train_data['close'].pct_change().dropna()
            if len(returns) < 2:
                self.logger.warning("Insufficient data for volatility calculation")
                return None
            
            volatility = returns.std()
            if not np.isfinite(volatility):
                self.logger.warning("Invalid volatility value calculated")
                return None
            
            if volatility > self.volatility_threshold:
                new_train_days = max(1, int(self.train_period_days * 0.5))
                new_test_days = max(1, int(self.test_period_days * 0.5))
                
                # Ensure we don't create invalid date ranges
                adjusted_train_start = train_end - timedelta(days=new_train_days)
                adjusted_test_end = train_end + timedelta(days=new_test_days)
                
                if adjusted_train_start >= train_end or train_end >= adjusted_test_end:
                    self.logger.warning("Adjusted window parameters would create invalid date ranges")
                    return None
                
                return {
                    'train_start': adjusted_train_start, 
                    'test_end': adjusted_test_end
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error in volatility adjustment: {e}")
            return None

    async def validate_model(self, model_trainer: Callable, data: pd.DataFrame, regime_labels: Optional[np.ndarray]=None) -> Dict[str, Any]:
        """Run walk-forward validation on a model."""
        try:
            # Input validation
            if not callable(model_trainer):
                raise RuntimeTypeError(Callable, model_trainer, 'validate_model model_trainer')
            if not isinstance(data, pd.DataFrame):
                raise RuntimeTypeError(pd.DataFrame, data, 'validate_model data')
            if regime_labels is not None and not isinstance(regime_labels, np.ndarray):
                raise RuntimeTypeError(np.ndarray, regime_labels, 'validate_model regime_labels')
            
            self.logger.info('Starting walk-forward validation...')
            
            # Generate windows with error handling
            try:
                windows = self.generate_walk_forward_windows(data)
            except Exception as e:
                self.logger.error(f"Failed to generate walk-forward windows: {e}")
                return {
                    'windows': [],
                    'results': [],
                    'analysis': {
                        'total_windows': 0,
                        'successful_windows': 0,
                        'validation_passed': False,
                        'error': f"Window generation failed: {str(e)}"
                    }
                }
            
            # Validate regime labels if provided
            if self.regime_aware and regime_labels is not None:
                if len(regime_labels) != len(data):
                    raise ValueError(f"Regime labels length ({len(regime_labels)}) must match data length ({len(data)})")
                results = await self._validate_regime_aware(model_trainer, data, regime_labels, windows)
            else:
                results = await self._validate_standard(model_trainer, data, windows)
            
            # Analyze results with error handling
            try:
                analysis = self._analyze_validation_results(results)
            except Exception as e:
                self.logger.error(f"Failed to analyze validation results: {e}")
                analysis = {
                    'total_windows': len(results),
                    'successful_windows': 0,
                    'validation_passed': False,
                    'error': f"Analysis failed: {str(e)}"
                }
            
            # Save results with error handling
            try:
                self._save_validation_results(results, analysis)
            except Exception as e:
                self.logger.error(f"Failed to save validation results: {e}")
                # Don't fail the entire operation if saving fails
            
            return {
                'windows': [w.to_dict() for w in windows], 
                'results': results, 
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error in validate_model: {e}")
            return {
                'windows': [],
                'results': [],
                'analysis': {
                    'total_windows': 0,
                    'successful_windows': 0,
                    'validation_passed': False,
                    'error': f"Validation failed: {str(e)}"
                }
            }

    async def _validate_standard(self, model_trainer: Callable, data: pd.DataFrame, windows: List[WalkForwardWindow]) -> List[Dict[str, Any]]:
        """Standard walk-forward validation."""
        results = []
        max_workers = min(4, len(windows))  # Limit workers based on number of windows
        timeout_seconds = 3600  # 1 hour timeout per window
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # Submit all window validation tasks
                for window in windows:
                    try:
                        future = executor.submit(self._validate_single_window, model_trainer, data, window)
                        futures.append((window, future))
                    except Exception as e:
                        self.logger.error(f'Failed to submit window {window.window_id}: {e}')
                        results.append({
                            'window': window.to_dict(), 
                            'error': f'Submission failed: {str(e)}', 
                            'success': False
                        })
                
                # Collect results with timeout protection
                for window, future in futures:
                    try:
                        result = future.result(timeout=timeout_seconds)
                        results.append(result)
                    except TimeoutError:
                        self.logger.error(f'Window {window.window_id} timed out after {timeout_seconds} seconds')
                        results.append({
                            'window': window.to_dict(), 
                            'error': f'Timeout after {timeout_seconds} seconds', 
                            'success': False
                        })
                    except Exception as e:
                        self.logger.error(f'Window {window.window_id} failed: {e}')
                        results.append({
                            'window': window.to_dict(), 
                            'error': str(e), 
                            'success': False
                        })
                        
        except Exception as e:
            self.logger.error(f'Error in _validate_standard: {e}')
            # Return partial results if available
            if not results:
                results = [{
                    'window': {'window_id': -1},
                    'error': f'Standard validation failed: {str(e)}',
                    'success': False
                }]
        
        return results

    def _validate_single_window(self, model_trainer: Callable, data: pd.DataFrame, window: WalkForwardWindow) -> Dict[str, Any]:
        """Validate a single window."""
        try:
            # Input validation
            if not callable(model_trainer):
                return {'window': window.to_dict(), 'error': 'model_trainer must be callable', 'success': False}
            if not isinstance(data, pd.DataFrame):
                return {'window': window.to_dict(), 'error': 'data must be a DataFrame', 'success': False}
            if not isinstance(window, WalkForwardWindow):
                return {'window': window.to_dict(), 'error': 'window must be a WalkForwardWindow', 'success': False}
            
            # Extract training and test data
            try:
                train_data = data[window.train_start:window.train_end]
                test_data = data[window.test_start:window.test_end]
            except Exception as e:
                return {'window': window.to_dict(), 'error': f'Failed to extract data: {str(e)}', 'success': False}
            
            # Validate data availability
            if train_data.empty:
                return {'window': window.to_dict(), 'error': 'No training data available', 'success': False}
            if test_data.empty:
                return {'window': window.to_dict(), 'error': 'No test data available', 'success': False}
            
            if len(train_data) < self.min_train_samples:
                return {
                    'window': window.to_dict(), 
                    'error': f'Insufficient training samples: {len(train_data)} < {self.min_train_samples}', 
                    'success': False
                }
            
            # Train model with error handling
            try:
                model = model_trainer(train_data)
                if model is None:
                    return {'window': window.to_dict(), 'error': 'Model trainer returned None', 'success': False}
            except Exception as e:
                return {'window': window.to_dict(), 'error': f'Model training failed: {str(e)}', 'success': False}
            
            # Generate predictions with error handling
            try:
                train_predictions = model.predict(train_data)
                test_predictions = model.predict(test_data)
                
                if train_predictions is None or test_predictions is None:
                    return {'window': window.to_dict(), 'error': 'Model predictions returned None', 'success': False}
                
                if len(train_predictions) != len(train_data):
                    return {'window': window.to_dict(), 'error': 'Train predictions length mismatch', 'success': False}
                if len(test_predictions) != len(test_data):
                    return {'window': window.to_dict(), 'error': 'Test predictions length mismatch', 'success': False}
                    
            except Exception as e:
                return {'window': window.to_dict(), 'error': f'Prediction generation failed: {str(e)}', 'success': False}
            
            # Calculate metrics with error handling
            try:
                train_metrics = self._calculate_metrics(train_data, train_predictions, 'train')
                test_metrics = self._calculate_metrics(test_data, test_predictions, 'test')
                degradation = self._calculate_degradation(train_metrics, test_metrics)
            except Exception as e:
                return {'window': window.to_dict(), 'error': f'Metrics calculation failed: {str(e)}', 'success': False}
            
            # Extract model parameters safely
            model_params = {}
            try:
                if hasattr(model, 'get_params') and callable(model.get_params):
                    model_params = model.get_params()
                elif hasattr(model, '__dict__'):
                    model_params = {k: v for k, v in model.__dict__.items() if not k.startswith('_')}
            except Exception as e:
                self.logger.warning(f"Failed to extract model parameters: {e}")
            
            return {
                'window': window.to_dict(), 
                'train_metrics': train_metrics, 
                'test_metrics': test_metrics, 
                'degradation': degradation, 
                'model_params': model_params, 
                'success': True
            }
            
        except Exception as e:
            return {'window': window.to_dict(), 'error': f'Unexpected error: {str(e)}', 'success': False}

    async def _validate_regime_aware(self, model_trainer: Callable, data: pd.DataFrame, regime_labels: np.ndarray, windows: List[WalkForwardWindow]) -> List[Dict[str, Any]]:
        """Regime-aware walk-forward validation."""
        results = []
        for window in windows:
            window_results = {'window': window.to_dict(), 'regime_results': {}, 'success': True}
            train_mask = (data.index >= window.train_start) & (data.index <= window.train_end)
            test_mask = (data.index >= window.test_start) & (data.index <= window.test_end)
            train_data = data[train_mask]
            test_data = data[test_mask]
            train_regimes = regime_labels[train_mask]
            test_regimes = regime_labels[test_mask]
            for regime in ['bull', 'bear', 'sideways']:
                regime_result = await self._validate_regime_window(model_trainer, train_data, test_data, train_regimes, test_regimes, regime)
                window_results['regime_results'][regime] = regime_result
                if not regime_result['success']:
                    window_results['success'] = False
            results.append(window_results)
        return results

    async def _validate_regime_window(self, model_trainer: Callable, train_data: pd.DataFrame, test_data: pd.DataFrame, train_regimes: np.ndarray, test_regimes: np.ndarray, regime: str) -> Dict[str, Any]:
        """Validate a single regime within a window."""
        regime_map = {'bear': 0, 'sideways': 1, 'bull': 2}
        regime_num = regime_map.get(regime, 1)
        train_regime_data = train_data[train_regimes == regime_num]
        test_regime_data = test_data[test_regimes == regime_num]
        if len(train_regime_data) < self.min_samples_per_regime:
            return {'regime': regime, 'error': f'Insufficient {regime} training samples: {len(train_regime_data)}', 'success': False}
        if len(test_regime_data) < 10:
            return {'regime': regime, 'error': f'Insufficient {regime} test samples: {len(test_regime_data)}', 'success': False}
        try:
            model = model_trainer(train_regime_data, regime = regime)
            train_predictions = model.predict(train_regime_data)
            test_predictions = model.predict(test_regime_data)
            train_metrics = self._calculate_metrics(train_regime_data, train_predictions, f'train_{regime}')
            test_metrics = self._calculate_metrics(test_regime_data, test_predictions, f'test_{regime}')
            degradation = self._calculate_degradation(train_metrics, test_metrics)
            return {'regime': regime, 'train_samples': len(train_regime_data), 'test_samples': len(test_regime_data), 'train_metrics': train_metrics, 'test_metrics': test_metrics, 'degradation': degradation, 'success': True}
        except Exception as e:
            return {'regime': regime, 'error': str(e), 'success': False}

    def _calculate_metrics(self, data: pd.DataFrame, predictions: np.ndarray, prefix: str) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            # Input validation
            if not isinstance(data, pd.DataFrame):
                raise RuntimeTypeError(pd.DataFrame, data, '_calculate_metrics data')
            if not isinstance(predictions, np.ndarray):
                raise RuntimeTypeError(np.ndarray, predictions, '_calculate_metrics predictions')
            if not isinstance(prefix, str):
                raise RuntimeTypeError(str, prefix, '_calculate_metrics prefix')
            
            if data.empty:
                raise ValueError("Data cannot be empty")
            if len(predictions) == 0:
                raise ValueError("Predictions cannot be empty")
            
            # Ensure we have close prices
            if 'close' not in data.columns:
                raise ValueError("Data must contain 'close' column")
            
            # Calculate returns if not present
            if 'returns' not in data.columns:
                data = data.copy()  # Don't modify original data
                data['returns'] = data['close'].pct_change()
            
            # Align predictions with returns (skip first return which is NaN)
            if len(predictions) != len(data):
                # Handle length mismatch by taking the shorter length
                min_len = min(len(predictions), len(data))
                predictions = predictions[:min_len]
                data = data.iloc[:min_len]
            
            # Calculate strategy returns
            returns = data['returns'].values[1:]  # Skip first NaN
            pred_returns = predictions[:-1]  # Align with returns
            
            if len(returns) != len(pred_returns):
                min_len = min(len(returns), len(pred_returns))
                returns = returns[:min_len]
                pred_returns = pred_returns[:min_len]
            
            strategy_returns = returns * pred_returns
            
            # Validate strategy returns
            if len(strategy_returns) == 0:
                raise ValueError("No valid strategy returns calculated")
            
            # Calculate metrics with error handling
            metrics = {
                f'{prefix}_sharpe': self._calculate_sharpe(strategy_returns),
                f'{prefix}_sortino': self._calculate_sortino(strategy_returns),
                f'{prefix}_max_drawdown': self._calculate_max_drawdown(strategy_returns),
                f'{prefix}_win_rate': self._safe_mean(strategy_returns > 0),
                f'{prefix}_total_return': self._safe_sum(strategy_returns),
                f'{prefix}_volatility': self._safe_std(strategy_returns)
            }
            
            # Validate all metrics are finite
            for key, value in metrics.items():
                if not np.isfinite(value):
                    self.logger.warning(f"Non-finite metric {key}: {value}")
                    metrics[key] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {prefix}: {e}")
            # Return default metrics
            return {
                f'{prefix}_sharpe': 0.0,
                f'{prefix}_sortino': 0.0,
                f'{prefix}_max_drawdown': 0.0,
                f'{prefix}_win_rate': 0.0,
                f'{prefix}_total_return': 0.0,
                f'{prefix}_volatility': 0.0
            }
    
    def _safe_mean(self, array: np.ndarray) -> float:
        """Safely calculate mean."""
        try:
            if len(array) == 0:
                return 0.0
            return float(np.mean(array))
        except Exception:
            return 0.0
    
    def _safe_sum(self, array: np.ndarray) -> float:
        """Safely calculate sum."""
        try:
            if len(array) == 0:
                return 0.0
            return float(np.sum(array))
        except Exception:
            return 0.0
    
    def _safe_std(self, array: np.ndarray) -> float:
        """Safely calculate standard deviation."""
        try:
            if len(array) == 0:
                return 0.0
            return float(np.std(array))
        except Exception:
            return 0.0

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            if not isinstance(returns, np.ndarray):
                return 0.0
            if len(returns) == 0:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if not np.isfinite(mean_return) or not np.isfinite(std_return):
                return 0.0
            
            if std_return == 0:
                return 0.0
            
            sharpe = mean_return / std_return * np.sqrt(252)
            return float(sharpe) if np.isfinite(sharpe) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        try:
            if not isinstance(returns, np.ndarray):
                return 0.0
            if len(returns) == 0:
                return 0.0
            
            mean_return = np.mean(returns)
            if not np.isfinite(mean_return):
                return 0.0
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return float('inf') if mean_return > 0 else 0.0
            
            downside_std = np.std(downside_returns)
            if not np.isfinite(downside_std) or downside_std == 0:
                return float('inf') if mean_return > 0 else 0.0
            
            sortino = mean_return / downside_std * np.sqrt(252)
            return float(sortino) if np.isfinite(sortino) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            if not isinstance(returns, np.ndarray):
                return 0.0
            if len(returns) == 0:
                return 0.0
            
            # Check for infinite or NaN values
            if not np.all(np.isfinite(returns)):
                self.logger.warning("Non-finite values found in returns for drawdown calculation")
                returns = returns[np.isfinite(returns)]
                if len(returns) == 0:
                    return 0.0
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            
            # Handle division by zero
            drawdown = np.where(running_max == 0, 0, drawdown)
            
            max_dd = np.min(drawdown)
            return float(max_dd) if np.isfinite(max_dd) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating max drawdown: {e}")
            return 0.0

    def _calculate_degradation(self, train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance degradation from train to test."""
        try:
            # Input validation
            if not isinstance(train_metrics, dict):
                raise RuntimeTypeError(dict, train_metrics, '_calculate_degradation train_metrics')
            if not isinstance(test_metrics, dict):
                raise RuntimeTypeError(dict, test_metrics, '_calculate_degradation test_metrics')
            
            degradation = {}
            
            # Calculate Sharpe degradation
            train_sharpe = next((v for k, v in train_metrics.items() if 'sharpe' in k), 0.0)
            test_sharpe = next((v for k, v in test_metrics.items() if 'sharpe' in k), 0.0)
            
            if not np.isfinite(train_sharpe) or not np.isfinite(test_sharpe):
                degradation['sharpe_degradation'] = 0.0
            elif train_sharpe != 0:
                sharpe_degradation = (train_sharpe - test_sharpe) / abs(train_sharpe)
                degradation['sharpe_degradation'] = float(sharpe_degradation) if np.isfinite(sharpe_degradation) else 0.0
            else:
                degradation['sharpe_degradation'] = 0.0
            
            # Calculate win rate degradation
            train_wr = next((v for k, v in train_metrics.items() if 'win_rate' in k), 0.0)
            test_wr = next((v for k, v in test_metrics.items() if 'win_rate' in k), 0.0)
            
            if not np.isfinite(train_wr) or not np.isfinite(test_wr):
                degradation['win_rate_degradation'] = 0.0
            elif train_wr != 0:
                wr_degradation = (train_wr - test_wr) / train_wr
                degradation['win_rate_degradation'] = float(wr_degradation) if np.isfinite(wr_degradation) else 0.0
            else:
                degradation['win_rate_degradation'] = 0.0
            
            # Calculate overall degradation
            overall = (degradation['sharpe_degradation'] + degradation['win_rate_degradation']) / 2
            degradation['overall'] = float(overall) if np.isfinite(overall) else 0.0
            
            # Determine potential overfitting
            degradation['potential_overfitting'] = degradation['overall'] > self.max_acceptable_degradation
            
            return degradation
            
        except Exception as e:
            self.logger.error(f"Error calculating degradation: {e}")
            return {
                'sharpe_degradation': 0.0,
                'win_rate_degradation': 0.0,
                'overall': 0.0,
                'potential_overfitting': False
            }

    def _analyze_validation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward validation results."""
        try:
            # Input validation
            if not isinstance(results, list):
                raise RuntimeTypeError(list, results, '_analyze_validation_results results')
            
            analysis = {
                'total_windows': len(results),
                'successful_windows': 0,
                'average_degradation': 0.0,
                'overfitting_windows': 0,
                'regime_analysis': {} if self.regime_aware else None,
                'validation_passed': False
            }
            
            degradations = []
            
            # Analyze each result
            for result in results:
                if not isinstance(result, dict):
                    continue
                    
                if result.get('success', False):
                    analysis['successful_windows'] += 1
                    
                    if 'degradation' in result and isinstance(result['degradation'], dict):
                        degradation = result['degradation'].get('overall', 0.0)
                        if np.isfinite(degradation):
                            degradations.append(degradation)
                        
                        if result['degradation'].get('potential_overfitting', False):
                            analysis['overfitting_windows'] += 1
            
            # Calculate average degradation
            if degradations:
                avg_degradation = np.mean(degradations)
                analysis['average_degradation'] = float(avg_degradation) if np.isfinite(avg_degradation) else 0.0
            
            # Analyze regime-specific results
            if self.regime_aware:
                for regime in ['bull', 'bear', 'sideways']:
                    regime_degradations = []
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                            
                        if 'regime_results' in result and isinstance(result['regime_results'], dict):
                            regime_result = result['regime_results'].get(regime, {})
                            if isinstance(regime_result, dict) and regime_result.get('success', False):
                                if 'degradation' in regime_result and isinstance(regime_result['degradation'], dict):
                                    degradation = regime_result['degradation'].get('overall', 0.0)
                                    if np.isfinite(degradation):
                                        regime_degradations.append(degradation)
                    
                    if regime_degradations:
                        avg_regime_degradation = np.mean(regime_degradations)
                        analysis['regime_analysis'][regime] = {
                            'avg_degradation': float(avg_regime_degradation) if np.isfinite(avg_regime_degradation) else 0.0,
                            'windows_analyzed': len(regime_degradations)
                        }
            
            # Determine if validation passed
            successful_windows = max(analysis['successful_windows'], 1)
            overfitting_ratio = analysis['overfitting_windows'] / successful_windows
            analysis['validation_passed'] = (
                analysis['average_degradation'] <= self.max_acceptable_degradation and 
                overfitting_ratio < 0.3
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing validation results: {e}")
            return {
                'total_windows': len(results) if isinstance(results, list) else 0,
                'successful_windows': 0,
                'average_degradation': 0.0,
                'overfitting_windows': 0,
                'regime_analysis': {} if self.regime_aware else None,
                'validation_passed': False,
                'error': f"Analysis failed: {str(e)}"
            }

    def _save_validation_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
        """Save validation results to disk."""
        try:
            # Input validation
            if not isinstance(results, list):
                raise RuntimeTypeError(list, results, '_save_validation_results results')
            if not isinstance(analysis, dict):
                raise RuntimeTypeError(dict, analysis, '_save_validation_results analysis')
            
            # Ensure results directory exists
            if not ensure_directory(self.results_dir):
                raise RuntimeError(f"Failed to create results directory: {self.results_dir}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = self.results_dir / f'validation_results_{timestamp}.json'
            summary_path = self.results_dir / f'validation_summary_{timestamp}.json'
            
            # Prepare data for JSON serialization
            serializable_results = []
            for result in results:
                if isinstance(result, dict):
                    # Convert any non-serializable objects to strings
                    serializable_result = {}
                    for key, value in result.items():
                        try:
                            # Test if value is JSON serializable
                            import json
                            json.dumps(value)
                            serializable_result[key] = value
                        except (TypeError, ValueError):
                            serializable_result[key] = str(value)
                    serializable_results.append(serializable_result)
                else:
                    serializable_results.append(str(result))
            
            # Save results
            if not safe_json_dump(serializable_results, results_path, indent=2):
                raise RuntimeError(f"Failed to save results to {results_path}")
            
            # Save summary
            if not safe_json_dump(analysis, summary_path, indent=2):
                raise RuntimeError(f"Failed to save summary to {summary_path}")
            
            self.logger.info(f'Saved validation results to {results_path}')
            self.logger.info(f'Saved validation summary to {summary_path}')
            
        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")
            # Don't raise the exception to avoid failing the entire validation
    
    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Validate method compatible with BaseValidator interface.
        
        Args:
            data: The data to validate (should be a DataFrame)
            **kwargs: Additional validation parameters including:
                - model_trainer: Callable for training models
                - regime_labels: Optional regime labels for regime-aware validation
        
        Returns:
            Dict containing validation results
        """
        try:
            # Extract parameters from kwargs
            model_trainer = kwargs.get('model_trainer')
            regime_labels = kwargs.get('regime_labels', None)
            
            if model_trainer is None:
                return {
                    'valid': False,
                    'error': 'model_trainer is required for walk-forward validation',
                    'validation_passed': False
                }
            
            # Run validation synchronously (convert async to sync)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we need to handle this differently
                    # For now, return an error suggesting to use validate_model directly
                    return {
                        'valid': False,
                        'error': 'Walk-forward validation requires async context. Use validate_model() directly.',
                        'validation_passed': False
                    }
                else:
                    result = loop.run_until_complete(self.validate_model(model_trainer, data, regime_labels))
            except RuntimeError:
                # No event loop, create a new one
                result = asyncio.run(self.validate_model(model_trainer, data, regime_labels))
            
            # Convert result to BaseValidator format
            return {
                'valid': result['analysis'].get('validation_passed', False),
                'validation_passed': result['analysis'].get('validation_passed', False),
                'total_windows': result['analysis'].get('total_windows', 0),
                'successful_windows': result['analysis'].get('successful_windows', 0),
                'average_degradation': result['analysis'].get('average_degradation', 0.0),
                'overfitting_windows': result['analysis'].get('overfitting_windows', 0),
                'regime_analysis': result['analysis'].get('regime_analysis', {}),
                'results': result['results'],
                'windows': result['windows']
            }
            
        except Exception as e:
            self.logger.error(f"Error in validate method: {e}")
            return {
                'valid': False,
                'error': str(e),
                'validation_passed': False
            }

async def example_model_trainer(data: pd.DataFrame, regime: Optional[str]=None) -> Any:
    """Example model trainer for testing walk-forward validation."""

    class DummyModel:

        def __init__(self, regime: Any = None) -> None:
            self.regime = regime

        def predict(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
            return np.random.choice([-1, 0, 1], size = len(data))

        def get_params(self) -> Any:
            return {'regime': self.regime}
    return DummyModel(regime)
async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run walk-forward validator compatible with the validator orchestrator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
    
    Returns:
        Dictionary containing validation results
    """
    try:
        # Extract configuration from training_input
        config = training_input.get('walk_forward_config', {})
        if not config:
            # Use default configuration
            config = {
                'train_period_days': 365,
                'test_period_days': 30,
                'step_days': 30,
                'regime_aware': True,
                'adaptive_windows': True,
                'results_dir': 'validation_results'
            }
        
        # Create validator
        validator = WalkForwardValidator(config)
        
        # Extract data from pipeline state
        data = pipeline_state.get('data')
        if data is None:
            return {
                'validation_passed': False,
                'error': 'No data found in pipeline state for walk-forward validation'
            }
        
        # Extract model trainer from pipeline state or training input
        model_trainer = pipeline_state.get('model_trainer') or training_input.get('model_trainer')
        if model_trainer is None:
            # Use example model trainer for testing
            model_trainer = example_model_trainer
        
        # Extract regime labels if available
        regime_labels = pipeline_state.get('regime_labels')
        
        # Run validation
        result = await validator.validate_model(model_trainer, data, regime_labels)
        
        return {
            'validation_passed': result['analysis'].get('validation_passed', False),
            'total_windows': result['analysis'].get('total_windows', 0),
            'successful_windows': result['analysis'].get('successful_windows', 0),
            'average_degradation': result['analysis'].get('average_degradation', 0.0),
            'overfitting_windows': result['analysis'].get('overfitting_windows', 0),
            'regime_analysis': result['analysis'].get('regime_analysis', {}),
            'results': result['results'],
            'windows': result['windows']
        }
        
    except Exception as e:
        return {
            'validation_passed': False,
            'error': f'Walk-forward validation failed: {str(e)}'
        }

if __name__ == '__main__':

    async def main() -> None:
        config = {'train_period_days': 365, 'test_period_days': 30, 'step_days': 30, 'regime_aware': True, 'adaptive_windows': True}
        validator = WalkForwardValidator(config)
        print('Walk-forward validation system initialized')
    asyncio.run(main())