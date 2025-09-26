"""
Walk-Forward Analysis

Advanced walk-forward analysis for NAS-TAS models with regime-aware
validation and performance tracking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint_warning, tprint_debug
except ImportError:
    def tprint_warning(msg):
        logger.warning(msg)
    def tprint_debug(msg):
        logger.debug(msg)

logger = logging.getLogger(__name__)


class WalkForwardAnalyzerError(RuntimeError):
    """Base exception for walk-forward analyzer failures."""


class RegimeDetectionError(WalkForwardAnalyzerError):
    """Raised when regime detection fails for a fold."""


class ModelSelectionError(WalkForwardAnalyzerError):
    """Raised when no suitable model can be selected for a regime."""


class ModelRetrainingError(WalkForwardAnalyzerError):
    """Raised when model retraining fails."""


class ModelValidationError(WalkForwardAnalyzerError):
    """Raised when model validation fails."""


class PerformanceComputationError(WalkForwardAnalyzerError):
    """Raised when fold performance metrics cannot be computed."""


class WalkForwardMode(Enum):
    """Walk-forward analysis modes."""
    FIXED_WINDOW = "fixed_window"      # Fixed training window
    EXPANDING_WINDOW = "expanding_window"  # Expanding training window
    ADAPTIVE_WINDOW = "adaptive_window"    # Adaptive window based on regime changes
    ROLLING_WINDOW = "rolling_window"     # Rolling window


class ValidationMetric(Enum):
    """Validation metrics for walk-forward analysis."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    
    # Walk-forward settings
    mode: WalkForwardMode = WalkForwardMode.EXPANDING_WINDOW
    initial_training_size: int = 1000  # Initial training window size
    validation_size: int = 100         # Validation window size
    step_size: int = 50               # Step size for moving window
    
    # Regime-aware settings
    enable_regime_aware_validation: bool = True
    regime_change_threshold: float = 0.3  # Threshold for regime change detection
    min_regime_samples: int = 50      # Minimum samples per regime
    
    # Model retraining
    enable_model_retraining: bool = True
    retraining_frequency: int = 10    # Retrain every N steps
    enable_incremental_learning: bool = True
    incremental_learning_rate: float = 0.01
    
    # Performance tracking
    validation_metrics: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.ACCURACY,
        ValidationMetric.F1_SCORE,
        ValidationMetric.SHARPE_RATIO
    ])
    performance_threshold: float = 0.6  # Minimum performance threshold
    degradation_threshold: float = 0.1  # Performance degradation threshold
    
    # Data handling
    enable_data_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_data_validation: bool = True
    
    # Output settings
    save_results: bool = True
    results_path: str = "walk_forward_results"
    enable_detailed_logging: bool = True
    enable_visualization: bool = True
    
    # Advanced features
    enable_ensemble_validation: bool = True
    enable_uncertainty_quantification: bool = True
    enable_regime_transition_analysis: bool = True


@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis."""
    
    # Basic results
    success: bool
    execution_time: float
    total_folds: int
    successful_folds: int
    
    # Performance metrics
    overall_performance: Dict[str, float]
    fold_performance: List[Dict[str, Any]]
    regime_performance: Dict[int, Dict[str, float]]
    
    # Model evolution
    model_evolution: List[Dict[str, Any]]
    retraining_events: List[Dict[str, Any]]
    
    # Regime analysis
    regime_transitions: List[Dict[str, Any]]
    regime_stability: Dict[int, float]
    
    # Validation insights
    performance_trends: Dict[str, str]
    degradation_events: List[Dict[str, Any]]
    improvement_events: List[Dict[str, Any]]
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    configuration: Dict[str, Any] = field(default_factory=dict)
    data_statistics: Dict[str, Any] = field(default_factory=dict)


class WalkForwardAnalyzer:
    """
    Walk-forward analyzer for NAS-TAS models.
    
    Provides comprehensive walk-forward analysis with regime-aware validation,
    model evolution tracking, and performance degradation detection.
    """
    
    def __init__(self, config: WalkForwardConfig):
        """Initialize walk-forward analyzer.
        
        Args:
            config: Walk-forward configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Analysis state
        self.fold_results = []
        self.model_evolution = []
        self.regime_transitions = []
        self.performance_history = []
        
        # Model registry
        self.available_models = {}
        self.model_performance = {}
        
        self.logger.info("✅ Walk-Forward Analyzer initialized")
        self.logger.info(f"   Mode: {config.mode.value}")
        self.logger.info(f"   Initial training size: {config.initial_training_size}")
        self.logger.info(f"   Validation size: {config.validation_size}")
        self.logger.info(f"   Step size: {config.step_size}")
    
    def register_models(self, 
                       regime_models: Dict[int, Dict[str, Any]],
                       ensemble_models: Optional[Dict[str, Any]] = None):
        """
        Register models for walk-forward analysis.
        
        Args:
            regime_models: Dictionary of regime_id -> {model_type: model_info}
            ensemble_models: Optional ensemble models
        """
        self.logger.info("📝 Registering models for walk-forward analysis")
        
        try:
            # Register regime models
            for regime_id, models in regime_models.items():
                self.available_models[regime_id] = {}
                
                for model_type, model_info in models.items():
                    self.available_models[regime_id][model_type] = {
                        'model': model_info['model'],
                        'performance': model_info.get('val_metrics', {}),
                        'feature_importance': model_info.get('feature_importance', {}),
                        'hyperparameters': model_info.get('hyperparameters', {})
                    }
                    
                    # Initialize performance tracking
                    model_id = f"regime_{regime_id}_{model_type}"
                    self.model_performance[model_id] = {
                        'fold_performance': [],
                        'overall_performance': {},
                        'evolution_history': []
                    }
            
            # Register ensemble models
            if ensemble_models:
                self.available_models['ensemble'] = ensemble_models
            
            self.logger.info(f"✅ Registered models for {len(self.available_models)} regimes")
            
        except Exception as e:
            self.logger.error(f"❌ Model registration failed: {e}")
            raise
    
    def run_walk_forward_analysis(self, 
                                market_data: pd.DataFrame,
                                target_variable: str = 'close',
                                feature_columns: Optional[List[str]] = None) -> WalkForwardResult:
        """
        Run comprehensive walk-forward analysis.
        
        Args:
            market_data: Historical market data
            target_variable: Target variable for prediction
            feature_columns: List of feature columns
            
        Returns:
            WalkForwardResult with complete analysis results
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting walk-forward analysis")
        
        try:
            # Validate and prepare data
            self.logger.info("📊 Preparing data for walk-forward analysis...")
            prepared_data = self._prepare_data(market_data, target_variable, feature_columns)
            
            if not prepared_data['success']:
                return WalkForwardResult(
                    success=False,
                    execution_time=0.0,
                    total_folds=0,
                    successful_folds=0,
                    error_message=prepared_data['error']
                )
            
            data = prepared_data['data']
            data_statistics = prepared_data['statistics']
            
            # Initialize analysis state
            self._initialize_analysis_state()
            
            # Generate walk-forward folds
            self.logger.info("🔄 Generating walk-forward folds...")
            folds = self._generate_walk_forward_folds(data)
            
            if not folds:
                return WalkForwardResult(
                    success=False,
                    execution_time=0.0,
                    total_folds=0,
                    successful_folds=0,
                    error_message="No valid folds generated"
                )
            
            # Run walk-forward analysis
            self.logger.info(f"🔄 Running walk-forward analysis on {len(folds)} folds...")
            fold_results = self._run_walk_forward_folds(folds, data, target_variable)
            
            # Analyze results
            self.logger.info("📈 Analyzing walk-forward results...")
            analysis_results = self._analyze_walk_forward_results(fold_results)
            
            # Create result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = WalkForwardResult(
                success=True,
                execution_time=execution_time,
                total_folds=len(folds),
                successful_folds=len([f for f in fold_results if f['success']]),
                overall_performance=analysis_results['overall_performance'],
                fold_performance=fold_results,
                regime_performance=analysis_results['regime_performance'],
                model_evolution=self.model_evolution,
                retraining_events=analysis_results['retraining_events'],
                regime_transitions=self.regime_transitions,
                regime_stability=analysis_results['regime_stability'],
                performance_trends=analysis_results['performance_trends'],
                degradation_events=analysis_results['degradation_events'],
                improvement_events=analysis_results['improvement_events'],
                configuration=self._get_configuration_summary(),
                data_statistics=data_statistics
            )
            
            # Save results if requested
            if self.config.save_results:
                self.logger.info("💾 Saving walk-forward results...")
                self._save_walk_forward_results(result)
            
            self.logger.info(f"✅ Walk-forward analysis completed in {execution_time:.2f}s")
            self.logger.info(f"   Total folds: {result.total_folds}")
            self.logger.info(f"   Successful folds: {result.successful_folds}")
            self.logger.info(f"   Success rate: {result.successful_folds/result.total_folds:.2%}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Walk-forward analysis failed: {e}")
            
            return WalkForwardResult(
                success=False,
                execution_time=execution_time,
                total_folds=0,
                successful_folds=0,
                error_message=str(e)
            )
    
    def _prepare_data(self, 
                     market_data: pd.DataFrame,
                     target_variable: str,
                     feature_columns: Optional[List[str]]) -> Dict[str, Any]:
        """Prepare data for walk-forward analysis."""
        try:
            # Validate data
            if market_data.empty:
                return {'success': False, 'error': 'Empty dataset'}
            
            # Check required columns
            if target_variable not in market_data.columns:
                return {'success': False, 'error': f'Target variable {target_variable} not found'}
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in market_data.columns if col != target_variable]
            
            # Calculate data statistics
            statistics = {
                'total_records': len(market_data),
                'total_features': len(feature_columns),
                'date_range': (market_data.index[0], market_data.index[-1]) if hasattr(market_data.index, '__getitem__') else None,
                'missing_data_ratio': market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns)),
                'data_quality': 1.0 - (market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns)))
            }
            
            # Sort by index if datetime
            if hasattr(market_data.index, 'sort_values'):
                market_data = market_data.sort_index()
            
            return {
                'success': True,
                'data': market_data,
                'statistics': statistics
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _initialize_analysis_state(self):
        """Initialize analysis state."""
        self.fold_results = []
        self.model_evolution = []
        self.regime_transitions = []
        self.performance_history = []
    
    def _generate_walk_forward_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate walk-forward folds."""
        try:
            total_size = len(data)
            folds = []
            
            if self.config.mode == WalkForwardMode.FIXED_WINDOW:
                folds = self._generate_fixed_window_folds(data)
            elif self.config.mode == WalkForwardMode.EXPANDING_WINDOW:
                folds = self._generate_expanding_window_folds(data)
            elif self.config.mode == WalkForwardMode.ADAPTIVE_WINDOW:
                folds = self._generate_adaptive_window_folds(data)
            elif self.config.mode == WalkForwardMode.ROLLING_WINDOW:
                folds = self._generate_rolling_window_folds(data)
            else:
                raise ValueError(f"Unknown walk-forward mode: {self.config.mode}")
            
            self.logger.info(f"📊 Generated {len(folds)} walk-forward folds")
            return folds
            
        except Exception as e:
            self.logger.error(f"❌ Fold generation failed: {e}")
            return []
    
    def _generate_fixed_window_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate fixed window folds."""
        folds = []
        total_size = len(data)
        
        start_idx = self.config.initial_training_size
        end_idx = start_idx + self.config.validation_size
        
        while end_idx < total_size:
            fold = {
                'fold_id': len(folds),
                'training_start': 0,
                'training_end': start_idx,
                'validation_start': start_idx,
                'validation_end': end_idx,
                'training_data': data.iloc[:start_idx],
                'validation_data': data.iloc[start_idx:end_idx]
            }
            folds.append(fold)
            
            start_idx += self.config.step_size
            end_idx = start_idx + self.config.validation_size
        
        return folds
    
    def _generate_expanding_window_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate expanding window folds."""
        folds = []
        total_size = len(data)
        
        start_idx = self.config.initial_training_size
        end_idx = start_idx + self.config.validation_size
        
        while end_idx < total_size:
            fold = {
                'fold_id': len(folds),
                'training_start': 0,
                'training_end': start_idx,
                'validation_start': start_idx,
                'validation_end': end_idx,
                'training_data': data.iloc[:start_idx],
                'validation_data': data.iloc[start_idx:end_idx]
            }
            folds.append(fold)
            
            start_idx += self.config.step_size
            end_idx = start_idx + self.config.validation_size
        
        return folds
    
    def _generate_adaptive_window_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate adaptive window folds based on regime changes."""
        folds = []
        total_size = len(data)
        
        # Detect regime changes
        regime_changes = self._detect_regime_changes(data)
        
        start_idx = self.config.initial_training_size
        end_idx = start_idx + self.config.validation_size
        
        while end_idx < total_size:
            # Adjust window based on regime changes
            adjusted_start = self._adjust_window_for_regime_changes(
                start_idx, regime_changes, data
            )
            
            fold = {
                'fold_id': len(folds),
                'training_start': adjusted_start,
                'training_end': start_idx,
                'validation_start': start_idx,
                'validation_end': end_idx,
                'training_data': data.iloc[adjusted_start:start_idx],
                'validation_data': data.iloc[start_idx:end_idx],
                'regime_changes': [rc for rc in regime_changes if adjusted_start <= rc['index'] < end_idx]
            }
            folds.append(fold)
            
            start_idx += self.config.step_size
            end_idx = start_idx + self.config.validation_size
        
        return folds
    
    def _generate_rolling_window_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate rolling window folds."""
        folds = []
        total_size = len(data)
        
        start_idx = self.config.initial_training_size
        end_idx = start_idx + self.config.validation_size
        
        while end_idx < total_size:
            fold = {
                'fold_id': len(folds),
                'training_start': start_idx - self.config.initial_training_size,
                'training_end': start_idx,
                'validation_start': start_idx,
                'validation_end': end_idx,
                'training_data': data.iloc[start_idx - self.config.initial_training_size:start_idx],
                'validation_data': data.iloc[start_idx:end_idx]
            }
            folds.append(fold)
            
            start_idx += self.config.step_size
            end_idx = start_idx + self.config.validation_size
        
        return folds
    
    def _detect_regime_changes(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect regime changes in data."""
        try:
            # Simple regime change detection based on volatility
            if 'close' in data.columns:
                prices = data['close'].values
                returns = np.diff(prices) / prices[:-1]
                volatility = pd.Series(returns).rolling(window=20).std()
                
                # Detect significant changes in volatility
                volatility_changes = []
                for i in range(1, len(volatility)):
                    if abs(volatility.iloc[i] - volatility.iloc[i-1]) > self.config.regime_change_threshold:
                        volatility_changes.append({
                            'index': i,
                            'timestamp': data.index[i] if hasattr(data.index, '__getitem__') else None,
                            'change_type': 'volatility',
                            'magnitude': abs(volatility.iloc[i] - volatility.iloc[i-1])
                        })
                
                return volatility_changes
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Regime change detection failed: {e}")
            return []
    
    def _adjust_window_for_regime_changes(self, 
                                        start_idx: int,
                                        regime_changes: List[Dict[str, Any]],
                                        data: pd.DataFrame) -> int:
        """Adjust training window based on regime changes."""
        # Find the most recent regime change before start_idx
        recent_changes = [rc for rc in regime_changes if rc['index'] < start_idx]
        
        if recent_changes:
            # Adjust start to include the regime change
            latest_change = max(recent_changes, key=lambda x: x['index'])
            adjusted_start = max(0, latest_change['index'] - self.config.min_regime_samples)
            return adjusted_start
        
        return max(0, start_idx - self.config.initial_training_size)
    
    def _run_walk_forward_folds(self, 
                               folds: List[Dict[str, Any]],
                               data: pd.DataFrame,
                               target_variable: str) -> List[Dict[str, Any]]:
        """Run walk-forward analysis on all folds."""
        fold_results = []
        
        for fold in folds:
            try:
                self.logger.info(f"🔄 Processing fold {fold['fold_id']}...")
                
                # Detect regime for training data
                training_regime = self._detect_regime_for_data(fold['training_data'])
                
                # Select model for regime
                selected_model = self._select_model_for_regime(training_regime)
                
                if selected_model is None:
                    fold_results.append({
                        'fold_id': fold['fold_id'],
                        'success': False,
                        'error': 'No model available for regime'
                    })
                    continue
                
                # Train/retrain model if needed
                if self.config.enable_model_retraining:
                    retrained_model = self._retrain_model(
                        selected_model, fold['training_data'], target_variable
                    )
                    if retrained_model:
                        selected_model = retrained_model
                
                # Validate model on validation data
                validation_result = self._validate_model(
                    selected_model, fold['validation_data'], target_variable
                )
                
                # Calculate performance metrics
                performance_metrics = self._calculate_fold_performance(
                    validation_result, fold['validation_data'], target_variable
                )
                
                # Record fold result
                fold_result = {
                    'fold_id': fold['fold_id'],
                    'success': True,
                    'training_regime': training_regime,
                    'selected_model': selected_model['model_type'],
                    'performance_metrics': performance_metrics,
                    'validation_result': validation_result,
                    'regime_changes': fold.get('regime_changes', [])
                }
                
                fold_results.append(fold_result)
                
                # Update model evolution
                self._update_model_evolution(fold_result)
                
                # Update performance history
                self.performance_history.append({
                    'fold_id': fold['fold_id'],
                    'performance': performance_metrics,
                    'regime': training_regime,
                    'model': selected_model['model_type']
                })
                
                self.logger.info(f"   ✅ Fold {fold['fold_id']} completed - Performance: {performance_metrics.get('f1_score', 0):.3f}")
                
            except WalkForwardAnalyzerError as err:
                self.logger.error(f"   ❌ Fold {fold['fold_id']} failed: {err}")
                fold_results.append({
                    'fold_id': fold['fold_id'],
                    'success': False,
                    'error': str(err),
                    'error_type': err.__class__.__name__,
                })
            except Exception as e:
                self.logger.error(f"   ❌ Fold {fold['fold_id']} failed: {e}")
                fold_results.append({
                    'fold_id': fold['fold_id'],
                    'success': False,
                    'error': str(e),
                    'error_type': e.__class__.__name__,
                })
        
        return fold_results
    
    def _detect_regime_for_data(self, data: pd.DataFrame) -> int:
        """Detect regime for given data."""
        try:
            if 'close' not in data.columns or len(data) <= 1:
                raise RegimeDetectionError(
                    "Input data must contain a 'close' column with at least two observations"
                )

            # Simple regime detection based on volatility
            prices = data['close'].values
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)

            if volatility < 0.01:
                return 0  # Low volatility regime
            if volatility < 0.03:
                return 1  # Medium volatility regime
            return 2  # High volatility regime

        except RegimeDetectionError:
            raise
        except Exception as e:  # pragma: no cover - defensive logging path
            self.logger.warning(f"Regime detection failed: {e}")
            raise RegimeDetectionError(str(e)) from e
    
    def _select_model_for_regime(self, regime_id: int) -> Optional[Dict[str, Any]]:
        """Select model for regime."""
        try:
            if regime_id not in self.available_models:
                raise ModelSelectionError(f"No model available for regime {regime_id}")

            regime_models = self.available_models[regime_id]

            if not regime_models:
                raise ModelSelectionError(f"No candidate models registered for regime {regime_id}")

            # Select best performing model
            best_model_type = max(
                regime_models.keys(),
                key=lambda x: regime_models[x]['performance'].get('f1_score', 0.0)
            )

            return {
                'model': regime_models[best_model_type]['model'],
                'model_type': best_model_type,
                'regime_id': regime_id,
                'performance': regime_models[best_model_type]['performance']
            }

        except ModelSelectionError:
            raise
        except Exception as e:  # pragma: no cover - defensive path
            self.logger.warning(f"Model selection failed: {e}")
            raise ModelSelectionError(str(e)) from e
    
    def _retrain_model(self, 
                      selected_model: Dict[str, Any],
                      training_data: pd.DataFrame,
                      target_variable: str) -> Optional[Dict[str, Any]]:
        """Retrain model on new data."""
        try:
            if not self.config.enable_model_retraining:
                return selected_model

            # Simple retraining (in practice, this would be more sophisticated)
            model = selected_model['model']

            # Check if model supports incremental learning
            if hasattr(model, 'partial_fit'):
                # Incremental learning
                X = training_data.drop(columns=[target_variable]).values
                y = training_data[target_variable].values
                model.partial_fit(X, y)
            else:
                # Full retraining
                X = training_data.drop(columns=[target_variable]).values
                y = training_data[target_variable].values
                model.fit(X, y)

            return selected_model

        except Exception as e:
            self.logger.warning(f"Model retraining failed: {e}")
            raise ModelRetrainingError(str(e)) from e
    
    def _validate_model(self, 
                       selected_model: Dict[str, Any],
                       validation_data: pd.DataFrame,
                       target_variable: str) -> Dict[str, Any]:
        """Validate model on validation data."""
        try:
            model = selected_model['model']

            # Prepare validation data
            X_val = validation_data.drop(columns=[target_variable]).values
            y_val = validation_data[target_variable].values

            # Make predictions
            if not hasattr(model, 'predict'):
                raise ModelValidationError('Model does not support prediction')

            predictions = model.predict(X_val)

            # Calculate confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_val)
                    confidence = np.mean(np.max(proba, axis=1))
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to calculate model confidence: {e}")
                    tprint_debug("Using default confidence value of 0.5")
                    confidence = 0.5

            return {
                'success': True,
                'predictions': predictions,
                'confidence': confidence,
                'model_type': selected_model['model_type'],
                'regime_id': selected_model['regime_id']
            }

        except ModelValidationError:
            raise
        except Exception as e:
            self.logger.warning(f"Model validation failed: {e}")
            raise ModelValidationError(str(e)) from e
    
    def _calculate_fold_performance(self, 
                                   validation_result: Dict[str, Any],
                                   validation_data: pd.DataFrame,
                                   target_variable: str) -> Dict[str, float]:
        """Calculate performance metrics for a fold."""
        try:
            if not validation_result['success']:
                raise PerformanceComputationError(validation_result.get('error', 'Validation unsuccessful'))

            predictions = validation_result['predictions']
            y_true = validation_data[target_variable].values

            if len(predictions) != len(y_true):
                raise PerformanceComputationError('Predictions and targets have mismatched lengths')

            # Calculate basic metrics
            accuracy = float(np.mean(predictions == y_true))

            # Calculate precision, recall, F1
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = float(precision_score(y_true, predictions, average='weighted', zero_division=0))
            recall = float(recall_score(y_true, predictions, average='weighted', zero_division=0))
            f1 = float(f1_score(y_true, predictions, average='weighted', zero_division=0))

            # Calculate Sharpe ratio (simplified)
            returns = np.diff(validation_data[target_variable].values) / validation_data[target_variable].values[:-1]
            sharpe_ratio = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'sharpe_ratio': sharpe_ratio,
                'confidence': float(validation_result.get('confidence', 0.5)),
            }

        except PerformanceComputationError:
            raise
        except Exception as e:
            self.logger.warning(f"Performance calculation failed: {e}")
            raise PerformanceComputationError(str(e)) from e
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics when calculation fails."""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'sharpe_ratio': 0.0,
            'confidence': 0.0
        }
    
    def _update_model_evolution(self, fold_result: Dict[str, Any]):
        """Update model evolution tracking."""
        try:
            evolution_entry = {
                'fold_id': fold_result['fold_id'],
                'regime': fold_result['training_regime'],
                'model_type': fold_result['selected_model'],
                'performance': fold_result['performance_metrics'],
                'timestamp': datetime.now()
            }
            
            self.model_evolution.append(evolution_entry)
            
        except Exception as e:
            self.logger.warning(f"Model evolution update failed: {e}")
    
    def _analyze_walk_forward_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward results."""
        try:
            # Calculate overall performance
            successful_folds = [f for f in fold_results if f['success']]
            
            if not successful_folds:
                return self._get_default_analysis()
            
            # Aggregate performance metrics
            overall_performance = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'sharpe_ratio']:
                values = [f['performance_metrics'][metric] for f in successful_folds]
                overall_performance[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            # Analyze regime performance
            regime_performance = self._analyze_regime_performance(successful_folds)
            
            # Analyze model evolution
            retraining_events = self._analyze_retraining_events()
            
            # Analyze performance trends
            performance_trends = self._analyze_performance_trends(successful_folds)
            
            # Detect degradation and improvement events
            degradation_events = self._detect_degradation_events(successful_folds)
            improvement_events = self._detect_improvement_events(successful_folds)
            
            # Analyze regime stability
            regime_stability = self._analyze_regime_stability()
            
            return {
                'overall_performance': overall_performance,
                'regime_performance': regime_performance,
                'retraining_events': retraining_events,
                'performance_trends': performance_trends,
                'degradation_events': degradation_events,
                'improvement_events': improvement_events,
                'regime_stability': regime_stability
            }
            
        except Exception as e:
            self.logger.error(f"❌ Results analysis failed: {e}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis when calculation fails."""
        return {
            'overall_performance': {},
            'regime_performance': {},
            'retraining_events': [],
            'performance_trends': {},
            'degradation_events': [],
            'improvement_events': [],
            'regime_stability': {}
        }
    
    def _analyze_regime_performance(self, successful_folds: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        """Analyze performance by regime."""
        regime_performance = {}
        
        for fold in successful_folds:
            regime = fold['training_regime']
            performance = fold['performance_metrics']
            
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'folds': 0,
                    'accuracy': [],
                    'f1_score': [],
                    'sharpe_ratio': []
                }
            
            regime_performance[regime]['folds'] += 1
            regime_performance[regime]['accuracy'].append(performance['accuracy'])
            regime_performance[regime]['f1_score'].append(performance['f1_score'])
            regime_performance[regime]['sharpe_ratio'].append(performance['sharpe_ratio'])
        
        # Calculate averages
        for regime in regime_performance:
            perf = regime_performance[regime]
            regime_performance[regime] = {
                'folds': perf['folds'],
                'mean_accuracy': np.mean(perf['accuracy']),
                'mean_f1_score': np.mean(perf['f1_score']),
                'mean_sharpe_ratio': np.mean(perf['sharpe_ratio']),
                'std_accuracy': np.std(perf['accuracy']),
                'std_f1_score': np.std(perf['f1_score']),
                'std_sharpe_ratio': np.std(perf['sharpe_ratio'])
            }
        
        return regime_performance
    
    def _analyze_retraining_events(self) -> List[Dict[str, Any]]:
        """Analyze model retraining events."""
        retraining_events = []
        
        for i, evolution in enumerate(self.model_evolution):
            if i > 0:
                prev_evolution = self.model_evolution[i-1]
                
                # Check if model changed
                if evolution['model_type'] != prev_evolution['model_type']:
                    retraining_events.append({
                        'fold_id': evolution['fold_id'],
                        'from_model': prev_evolution['model_type'],
                        'to_model': evolution['model_type'],
                        'regime': evolution['regime'],
                        'performance_change': evolution['performance']['f1_score'] - prev_evolution['performance']['f1_score']
                    })
        
        return retraining_events
    
    def _analyze_performance_trends(self, successful_folds: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze performance trends."""
        trends = {}
        
        if len(successful_folds) < 3:
            return trends
        
        # Analyze F1 score trend
        f1_scores = [f['performance_metrics']['f1_score'] for f in successful_folds]
        early_f1 = np.mean(f1_scores[:len(f1_scores)//3])
        late_f1 = np.mean(f1_scores[-len(f1_scores)//3:])
        
        if late_f1 > early_f1 + 0.05:
            trends['f1_score'] = 'improving'
        elif late_f1 < early_f1 - 0.05:
            trends['f1_score'] = 'declining'
        else:
            trends['f1_score'] = 'stable'
        
        # Analyze accuracy trend
        accuracy_scores = [f['performance_metrics']['accuracy'] for f in successful_folds]
        early_acc = np.mean(accuracy_scores[:len(accuracy_scores)//3])
        late_acc = np.mean(accuracy_scores[-len(accuracy_scores)//3:])
        
        if late_acc > early_acc + 0.05:
            trends['accuracy'] = 'improving'
        elif late_acc < early_acc - 0.05:
            trends['accuracy'] = 'declining'
        else:
            trends['accuracy'] = 'stable'
        
        return trends
    
    def _detect_degradation_events(self, successful_folds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect performance degradation events."""
        degradation_events = []
        
        for i in range(1, len(successful_folds)):
            current_fold = successful_folds[i]
            prev_fold = successful_folds[i-1]
            
            current_f1 = current_fold['performance_metrics']['f1_score']
            prev_f1 = prev_fold['performance_metrics']['f1_score']
            
            if current_f1 < prev_f1 - self.config.degradation_threshold:
                degradation_events.append({
                    'fold_id': current_fold['fold_id'],
                    'metric': 'f1_score',
                    'current_value': current_f1,
                    'previous_value': prev_f1,
                    'degradation': prev_f1 - current_f1,
                    'regime': current_fold['training_regime']
                })
        
        return degradation_events
    
    def _detect_improvement_events(self, successful_folds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect performance improvement events."""
        improvement_events = []
        
        for i in range(1, len(successful_folds)):
            current_fold = successful_folds[i]
            prev_fold = successful_folds[i-1]
            
            current_f1 = current_fold['performance_metrics']['f1_score']
            prev_f1 = prev_fold['performance_metrics']['f1_score']
            
            if current_f1 > prev_f1 + self.config.degradation_threshold:
                improvement_events.append({
                    'fold_id': current_fold['fold_id'],
                    'metric': 'f1_score',
                    'current_value': current_f1,
                    'previous_value': prev_f1,
                    'improvement': current_f1 - prev_f1,
                    'regime': current_fold['training_regime']
                })
        
        return improvement_events
    
    def _analyze_regime_stability(self) -> Dict[int, float]:
        """Analyze regime stability."""
        regime_stability = {}
        
        for regime in set([f['training_regime'] for f in self.performance_history]):
            regime_folds = [f for f in self.performance_history if f['regime'] == regime]
            
            if len(regime_folds) < 2:
                regime_stability[regime] = 1.0
                continue
            
            # Calculate stability based on performance consistency
            f1_scores = [f['performance']['f1_score'] for f in regime_folds]
            stability = 1.0 - (np.std(f1_scores) / (np.mean(f1_scores) + 1e-8))
            regime_stability[regime] = max(0.0, min(1.0, stability))
        
        return regime_stability
    
    def _get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'mode': self.config.mode.value,
            'initial_training_size': self.config.initial_training_size,
            'validation_size': self.config.validation_size,
            'step_size': self.config.step_size,
            'enable_regime_aware_validation': self.config.enable_regime_aware_validation,
            'enable_model_retraining': self.config.enable_model_retraining,
            'validation_metrics': [m.value for m in self.config.validation_metrics],
            'performance_threshold': self.config.performance_threshold,
            'degradation_threshold': self.config.degradation_threshold
        }
    
    def _save_walk_forward_results(self, result: WalkForwardResult):
        """Save walk-forward results."""
        try:
            results_path = Path(self.config.results_path)
            results_path.mkdir(parents=True, exist_ok=True)
            
            # Save result summary
            result_summary = {
                'success': result.success,
                'execution_time': result.execution_time,
                'total_folds': result.total_folds,
                'successful_folds': result.successful_folds,
                'overall_performance': result.overall_performance,
                'regime_performance': result.regime_performance,
                'performance_trends': result.performance_trends,
                'configuration': result.configuration,
                'data_statistics': result.data_statistics
            }
            
            with open(results_path / "walk_forward_summary.json", 'w') as f:
                json.dump(result_summary, f, indent=2)
            
            # Save detailed results
            with open(results_path / "walk_forward_result.pkl", 'wb') as f:
                pickle.dump(result, f)
            
            self.logger.info(f"💾 Walk-forward results saved to {results_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save walk-forward results: {e}")
    
    def get_walk_forward_summary(self) -> Dict[str, Any]:
        """Get summary of walk-forward analysis."""
        if not self.fold_results:
            return {'error': 'No walk-forward data available'}
        
        successful_folds = [f for f in self.fold_results if f['success']]
        
        return {
            'total_folds': len(self.fold_results),
            'successful_folds': len(successful_folds),
            'success_rate': len(successful_folds) / len(self.fold_results),
            'model_evolution_events': len(self.model_evolution),
            'regime_transitions': len(self.regime_transitions),
            'performance_history': len(self.performance_history)
        }