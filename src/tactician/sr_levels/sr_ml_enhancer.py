from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import joblib
from pathlib import Path
import json
import logging
import time

# Core utilities
from ...utils.logger import system_logger
from ...core.sr_error_handlers import sr_error_handler, SROptimizationError, SRDataError

# ML Common utilities
from ...utils.ml_common.model_training import ModelTrainingManager
from ...utils.ml_common.feature_selection import FeatureSelector
from ...utils.ml_common.model_evaluation import ModelEvaluator
from ...utils.ml_common.ensemble_manager import EnsembleManager
from ...utils.ml_common.hpo_utils import HPOptimizer
from ...utils.ml_common.validation_utils import ValidationUtils
from ...utils.ml_common.confidence_metrics import calculate_confidence_metrics
from ...utils.ml_common.model_explanations import explain_model_with_shap_lime

# General utilities
from ...utils.performance_utils import PerformanceProfiler
from ...utils.parallel_processing_optimizer import ParallelProcessor
from ...utils.enhanced_error_handler import EnhancedErrorHandler
from ...utils.caching import CacheManager
from ...utils.intelligent_feature_cache import FeatureCache
from ...utils.validation import ValidationFramework
from ...utils.enhanced_data_operations import DataOperations
from ...utils.monitoring_utils import MonitoringUtils

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.inspection import permutation_importance
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print('Warning: scikit-learn not available, ML features disabled')

@dataclass
class MLFeatureSet:
    """Set of features for ML models."""
    features: np.ndarray
    feature_names: List[str]
    target: np.ndarray
    metadata: Dict[str, Any]

@dataclass
class MLModelResult:
    """Result of ML model prediction."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    confidence: float
    model_type: str
    feature_importance: Optional[Dict[str, float]]
    performance_metrics: Dict[str, float]

@dataclass
class SRQualityPrediction:
    """S/R level quality prediction."""
    level_id: str
    quality_score: float
    confidence: float
    features_used: List[str]
    prediction_reason: str

@dataclass
class BreakoutPrediction:
    """Breakout prediction result."""
    level_id: str
    breakout_probability: float
    confidence: float
    expected_direction: str
    time_to_breakout: Optional[int]
    features_used: List[str]

class SRMLEnhancer:
    """Machine learning enhancer for S/R detection and qualification using modern utilities."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ML enhancer with modern utilities."""
        self.config = config
        self.logger = system_logger.getChild('SRMLEnhancer')
        self.ml_config = config.get('ml_enhancement', {})
        
        # Initialize modern utilities
        self.error_handler = EnhancedErrorHandler('SRMLEnhancer')
        self.performance_profiler = PerformanceProfiler()
        self.parallel_processor = ParallelProcessor()
        self.cache_manager = CacheManager()
        self.feature_cache = FeatureCache()
        self.validation_framework = ValidationFramework()
        self.data_operations = DataOperations()
        self.monitoring_utils = MonitoringUtils()
        
        # Initialize ML utilities
        if ML_AVAILABLE:
            self.model_training_manager = ModelTrainingManager()
            self.feature_selector = FeatureSelector()
            self.model_evaluator = ModelEvaluator()
            self.ensemble_manager = EnsembleManager()
            self.hpo_optimizer = HPOptimizer()
            self.validation_utils = ValidationUtils()
            self.ml_enabled = self.ml_config.get('feature_engineering', {}).get('enable_ml_features', True)
        else:
            self.logger.warning('ML libraries not available, ML features disabled')
            self.ml_enabled = False
            
        # Model storage
        self.sr_quality_model = None
        self.breakout_prediction_model = None
        self.regime_classification_model = None
        self.feature_scaler = StandardScaler() if ML_AVAILABLE else None
        
        # Training data
        self.training_features = []
        self.training_targets = []
        self.feature_names = []
        
        # Performance tracking
        self.model_performance = {
            'sr_quality': {'accuracy': 0.0, 'last_update': None},
            'breakout_prediction': {'accuracy': 0.0, 'last_update': None},
            'regime_classification': {'accuracy': 0.0, 'last_update': None}
        }

    @self.error_handler.handle_errors(exceptions=(SROptimizationError, SRDataError), default_return=False, context='ML model training')
    async def train_models(self, market_data: pd.DataFrame, sr_levels: List[Dict[str, Any]], historical_performance: Optional[Dict[str, Any]] = None) -> bool:
        """Train all ML models using modern utilities."""
        with self.performance_profiler.profile('ml_model_training'):
            try:
                if not self.ml_enabled or not ML_AVAILABLE:
                    self.logger.info('ML training skipped - ML not available or disabled')
                    return False
                    
                self.logger.info('🤖 Starting ML model training with modern utilities...')
                
                # Validate inputs using validation framework
                if not self.validation_framework.validate_dataframe(market_data, required_columns=['open', 'high', 'low', 'close', 'volume']):
                    self.logger.error('Invalid market data provided')
                    return False
                    
                # Prepare training data with caching
                cache_key = f'training_data_{hash(str(market_data.index[-10:]))}_{len(sr_levels)}'
                training_data = await self.feature_cache.get_or_compute(
                    cache_key,
                    lambda: self._prepare_training_data(market_data, sr_levels, historical_performance)
                )
                
                if not training_data:
                    self.logger.warning('No training data available')
                    return False
                    
                # Optimize target weights using HPO
                await self.optimize_target_weights(market_data, sr_levels, historical_performance)
                
                # Train models in parallel using parallel processor
                training_tasks = [
                    self._train_sr_quality_model(training_data),
                    self._train_breakout_prediction_model(training_data),
                    self._train_regime_classification_model(market_data)
                ]
                
                results = await self.parallel_processor.run_parallel(training_tasks)
                
                # Monitor training results
                self.monitoring_utils.log_training_completion(results)
                
                self.logger.info('✅ ML model training completed with modern utilities')
                return True
                
            except Exception as e:
                self.logger.error(f'ML model training failed: {e}')
                return False

    async def _prepare_training_data(self, market_data: pd.DataFrame, sr_levels: List[Dict[str, Any]], historical_performance: Optional[Dict[str, Any]]) -> Optional[MLFeatureSet]:
        """Prepare training data for ML models with modern data operations."""
        with self.performance_profiler.profile('training_data_preparation'):
            try:
                # Use data operations for enhanced data processing
                market_data_processed = await self.data_operations.enhance_market_data(market_data)
                
                features = []
                targets = []
                
                # Extract step06 features with caching
                step06_cache_key = f'step06_features_{hash(str(market_data.index[-10:]))}'
                step06_features = await self.feature_cache.get_or_compute(
                    step06_cache_key,
                    lambda: self._extract_step06_features(market_data_processed)
                )
                
                # Process levels in parallel
                level_processing_tasks = []
                for level in sr_levels:
                    task = self._process_single_level(market_data_processed, level, step06_features, historical_performance)
                    level_processing_tasks.append(task)
                
                level_results = await self.parallel_processor.run_parallel(level_processing_tasks)
                
                # Collect results
                for level_features, target in level_results:
                    if level_features is not None and target is not None:
                        features.append(level_features)
                        targets.append(target)
                
                if not features:
                    return None
                    
                # Convert to arrays with validation
                features_array = np.array(features)
                targets_array = np.array(targets)
                
                # Validate data quality
                if not self.validation_framework.validate_feature_matrix(features_array):
                    self.logger.warning('Feature matrix validation failed, applying corrections')
                    features_array = self.data_operations.clean_feature_matrix(features_array)
                
                feature_names = await self._get_combined_feature_names()
                
                # Log comprehensive statistics
                self.logger.info(f'📊 Training data prepared: {len(features)} samples, {len(feature_names)} features')
                self.logger.info(f'   - S/R specific features: {len(await self._get_feature_names())} (47 features)')
                self.logger.info(f'   - Step06 features: {len(step06_features)} (200+ features)')
                self.logger.info(f'   - Data quality score: {self.data_operations.calculate_data_quality_score(features_array)}')
                
                return MLFeatureSet(
                    features=features_array,
                    feature_names=feature_names,
                    target=targets_array,
                    metadata={
                        'n_samples': len(features),
                        'n_features': len(feature_names),
                        'sr_features': len(await self._get_feature_names()),
                        'step06_features': len(step06_features),
                        'target_distribution': np.bincount(targets_array.astype(int)) if len(targets_array) > 0 else [],
                        'data_quality_score': self.data_operations.calculate_data_quality_score(features_array)
                    }
                )
                
            except Exception as e:
                self.logger.error(f'Training data preparation failed: {e}')
                return None
    
    async def _process_single_level(self, market_data: pd.DataFrame, level: Dict[str, Any], step06_features: List[float], historical_performance: Optional[Dict[str, Any]]) -> Tuple[Optional[List[float]], Optional[float]]:
        """Process a single S/R level for training data."""
        try:
            sr_features = await self._extract_level_features(market_data, level)
            if sr_features:
                combined_features = sr_features + step06_features
                target = await self._create_target_for_level(level, historical_performance)
                return combined_features, target
            return None, None
        except Exception as e:
            self.logger.error(f'Error processing level: {e}')
            return None, None

    async def _extract_step06_features(self, market_data: pd.DataFrame) -> List[float]:
        """Extract step06 features (200+ features)."""
        try:
            try:
                from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineeringRefactored
                step06_engineer = VectorizedAdvancedFeatureEngineeringRefactored()
                step06_result = await step06_engineer.engineer_features(market_data)
                all_features = []
                price_features = step06_result.get('price_features', {})
                for feature_name, feature_values in price_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                volume_features = step06_result.get('volume_features', {})
                for feature_name, feature_values in volume_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                microstructure_features = step06_result.get('microstructure_features', {})
                for feature_name, feature_values in microstructure_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                technical_features = step06_result.get('technical_features', {})
                penetration_features_count = 0
                for feature_name, feature_values in technical_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                        if 'penetration' in feature_name.lower() or 'wick' in feature_name.lower():
                            penetration_features_count += 1
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                        if 'penetration' in feature_name.lower() or 'wick' in feature_name.lower():
                            penetration_features_count += 1
                if penetration_features_count > 0:
                    self.logger.info(f'📊 Step06 penetration features extracted: {penetration_features_count} features')
                regime_features = step06_result.get('regime_features', {})
                for feature_name, feature_values in regime_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                wavelet_features = step06_result.get('wavelet_features', {})
                for feature_name, feature_values in wavelet_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                cross_timeframe_features = step06_result.get('cross_timeframe_features', {})
                for feature_name, feature_values in cross_timeframe_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                interaction_features = step06_result.get('interaction_features', {})
                for feature_name, feature_values in interaction_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                self.logger.info(f'✅ Step06 features extracted: {len(all_features)} features')
                return all_features
            except ImportError as e:
                self.logger.warning(f'Step06 feature engineering not available: {e}')
                return []
            except Exception as e:
                self.logger.warning(f'Step06 feature extraction failed: {e}')
                return []
        except Exception as e:
            self.logger.error(f'Step06 feature extraction failed: {e}')
            return []

    async def _get_combined_feature_names(self) -> List[str]:
        """Get combined feature names (S/R + step06)."""
        try:
            sr_feature_names = await self._get_feature_names()
            step06_feature_names = []
            step06_categories = ['price_features', 'volume_features', 'microstructure_features', 'technical_features', 'regime_features', 'wavelet_features', 'cross_timeframe_features', 'interaction_features']
            for category in step06_categories:
                for i in range(25):
                    step06_feature_names.append(f'{category}_{i}')
            combined_names = sr_feature_names + step06_feature_names
            self.logger.info(f'📊 Combined feature names: {len(combined_names)} total')
            self.logger.info(f'   - S/R features: {len(sr_feature_names)}')
            self.logger.info(f'   - Step06 features: {len(step06_feature_names)}')
            return combined_names
        except Exception as e:
            self.logger.error(f'Combined feature names failed: {e}')
            return await self._get_feature_names()

    async def _extract_level_features(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> Optional[List[float]]:
        """Extract S/R specific features for a specific S/R level."""
        try:
            features = []
            level_price = level.get('price', 0)
            features.extend([level.get('touch_count', 0), level.get('strength', 0.5), level.get('age_bars', 0), level.get('avg_bounce_ratio', 0), level.get('max_bounce_ratio', 0), level.get('volume_confirmation_score', 0.5), level.get('consistency_score', 0.5), level.get('failure_count', 0)])
            if level_price > 0:
                current_price = market_data['close'].iloc[-1]
                proximity = abs(current_price - level_price) / level_price
                features.append(proximity)
            else:
                features.append(1.0)
            advanced_features = await self._extract_advanced_sr_features(market_data, level)
            features.extend(advanced_features)
            hvn_features = await self._extract_hvn_features(market_data, level)
            features.extend(hvn_features)
            fibonacci_features = await self._extract_fibonacci_features(market_data, level)
            features.extend(fibonacci_features)
            psychological_features = await self._extract_psychological_features(market_data, level)
            features.extend(psychological_features)
            pivot_features = await self._extract_pivot_features(market_data, level)
            features.extend(pivot_features)
            trendline_features = await self._extract_trendline_features(market_data, level)
            features.extend(trendline_features)
            sr_specific_features = await self._extract_sr_specific_features(market_data, level)
            features.extend(sr_specific_features)
            return features
        except Exception as e:
            self.logger.error(f'S/R feature extraction failed for level: {e}')
            return None

    async def _extract_technical_features(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> List[float]:
        """Extract technical indicator features (15+ features)."""
        try:
            features = []
            rsi = self._calculate_rsi(market_data['close'], 14)
            features.append(rsi.iloc[-1] if not rsi.empty else 50.0)
            macd_line, macd_signal = self._calculate_macd(market_data['close'])
            features.extend([macd_line.iloc[-1] if not macd_line.empty else 0.0, macd_signal.iloc[-1] if not macd_signal.empty else 0.0])
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(market_data['close'])
            current_price = market_data['close'].iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if not bb_upper.empty else 0.5
            features.append(bb_position)
            atr = self._calculate_atr(market_data, 14)
            features.append(atr.iloc[-1] if not atr.empty else 0.0)
            volume_ma = market_data['volume'].rolling(window=20).mean()
            volume_ratio = market_data['volume'].iloc[-1] / volume_ma.iloc[-1] if not volume_ma.empty else 1.0
            features.append(volume_ratio)
            momentum = (market_data['close'].iloc[-1] - market_data['close'].iloc[-10]) / market_data['close'].iloc[-10] if len(market_data) >= 10 else 0.0
            features.append(momentum)
            stoch_k, stoch_d = self._calculate_stochastic(market_data)
            features.extend([stoch_k.iloc[-1] if not stoch_k.empty else 50.0, stoch_d.iloc[-1] if not stoch_d.empty else 50.0])
            williams_r = self._calculate_williams_r(market_data)
            features.append(williams_r.iloc[-1] if not williams_r.empty else -50.0)
            cci = self._calculate_cci(market_data)
            features.append(cci.iloc[-1] if not cci.empty else 0.0)
            adx = self._calculate_adx(market_data)
            features.append(adx.iloc[-1] if not adx.empty else 25.0)
            obv = self._calculate_obv(market_data)
            features.append(obv.iloc[-1] if not obv.empty else 0.0)
            doji = self._detect_doji_pattern(market_data)
            hammer = self._detect_hammer_pattern(market_data)
            features.extend([doji, hammer])
            vix_proxy = self._calculate_volatility_proxy(market_data)
            features.append(vix_proxy)
            return features
        except Exception as e:
            self.logger.error(f'Technical feature extraction failed: {e}')
            return [0.0] * 15

    async def _extract_advanced_sr_features(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> List[float]:
        """Extract advanced S/R features (6 features)."""
        try:
            features = []
            level_price = level.get('price', 0)
            if level_price > 0:
                features.append(1.0)
            else:
                features.append(0.0)
            features.append(level.get('confluence_score', 0.5))
            last_touch = level.get('last_touch_bar', 0)
            current_bar = len(market_data)
            time_since_touch = current_bar - last_touch if last_touch > 0 else current_bar
            features.append(time_since_touch)
            features.append(level.get('volume_at_touch', 1.0))
            features.append(level.get('price_action_score', 0.5))
            features.append(level.get('microstructure_score', 0.5))
            return features
        except Exception as e:
            self.logger.error(f'Advanced S/R feature extraction failed: {e}')
            return [0.5] * 6

    async def _extract_hvn_features(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> List[float]:
        """Extract HVN (High Volume Node) features (5 features)."""
        try:
            features = []
            level_price = level.get('price', 0)
            hvn_strength = level.get('hvn_strength', 0.5)
            features.append(hvn_strength)
            hvn_volume_ratio = level.get('hvn_volume_ratio', 1.0)
            features.append(hvn_volume_ratio)
            hvn_touch_count = level.get('hvn_touch_count', 0)
            features.append(hvn_touch_count)
            hvn_time_weight = level.get('hvn_time_weight', 0.5)
            features.append(hvn_time_weight)
            hvn_price_accuracy = level.get('hvn_price_accuracy', 0.5)
            features.append(hvn_price_accuracy)
            return features
        except Exception as e:
            self.logger.error(f'HVN feature extraction failed: {e}')
            return [0.5] * 5

    async def _extract_fibonacci_features(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> List[float]:
        """Extract Fibonacci retracement features (6 features)."""
        try:
            features = []
            fib_level_type = level.get('fib_level_type', 0.0)
            features.append(fib_level_type)
            fib_strength = level.get('fib_strength', 0.5)
            features.append(fib_strength)
            fib_confluence_count = level.get('fib_confluence_count', 0)
            features.append(fib_confluence_count)
            fib_timeframe_alignment = level.get('fib_timeframe_alignment', 0.5)
            features.append(fib_timeframe_alignment)
            fib_volume_confirmation = level.get('fib_volume_confirmation', 0.5)
            features.append(fib_volume_confirmation)
            fib_bounce_quality = level.get('fib_bounce_quality', 0.5)
            features.append(fib_bounce_quality)
            return features
        except Exception as e:
            self.logger.error(f'Fibonacci feature extraction failed: {e}')
            return [0.0] * 6

    async def _extract_psychological_features(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> List[float]:
        """Extract psychological level features (5 features)."""
        try:
            features = []
            level_price = level.get('price', 0)
            psychological_level_type = level.get('psychological_level_type', 0.0)
            if level_price > 0:
                if level_price % 100 == 0:
                    psychological_level_type = 1.0
                elif level_price % 50 == 0:
                    psychological_level_type = 0.8
                elif level_price % 10 == 0:
                    psychological_level_type = 0.6
            features.append(psychological_level_type)
            round_number_strength = level.get('round_number_strength', 0.5)
            features.append(round_number_strength)
            psychological_touch_count = level.get('psychological_touch_count', 0)
            features.append(psychological_touch_count)
            psychological_volume_spike = level.get('psychological_volume_spike', 1.0)
            features.append(psychological_volume_spike)
            psychological_bounce_ratio = level.get('psychological_bounce_ratio', 0.5)
            features.append(psychological_bounce_ratio)
            return features
        except Exception as e:
            self.logger.error(f'Psychological feature extraction failed: {e}')
            return [0.0] * 5

    async def _extract_pivot_features(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> List[float]:
        """Extract pivot point features (4 features)."""
        try:
            features = []
            pivot_type = level.get('pivot_type', 0.0)
            features.append(pivot_type)
            pivot_strength = level.get('pivot_strength', 0.5)
            features.append(pivot_strength)
            pivot_timeframe = level.get('pivot_timeframe', 0.5)
            features.append(pivot_timeframe)
            pivot_confluence = level.get('pivot_confluence', 0.5)
            features.append(pivot_confluence)
            return features
        except Exception as e:
            self.logger.error(f'Pivot feature extraction failed: {e}')
            return [0.0] * 4

    async def _extract_trendline_features(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> List[float]:
        """Extract trend line features (4 features)."""
        try:
            features = []
            trendline_type = level.get('trendline_type', 0.0)
            features.append(trendline_type)
            trendline_strength = level.get('trendline_strength', 0.5)
            features.append(trendline_strength)
            trendline_touch_count = level.get('trendline_touch_count', 0)
            features.append(trendline_touch_count)
            trendline_angle = level.get('trendline_angle', 0.0)
            features.append(trendline_angle)
            return features
        except Exception as e:
            self.logger.error(f'Trend line feature extraction failed: {e}')
            return [0.0] * 4

    async def _extract_sr_specific_features(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> List[float]:
        """Extract S/R specific features (8 features)."""
        try:
            features = []
            sr_type = level.get('sr_type', 0.5)
            features.append(sr_type)
            sr_timeframe_confluence = level.get('sr_timeframe_confluence', 0.5)
            features.append(sr_timeframe_confluence)
            sr_breakout_history = level.get('sr_breakout_history', 0.5)
            features.append(sr_breakout_history)
            sr_retest_success_rate = level.get('sr_retest_success_rate', 0.5)
            features.append(sr_retest_success_rate)
            sr_volume_profile_strength = level.get('sr_volume_profile_strength', 0.5)
            features.append(sr_volume_profile_strength)
            sr_market_structure_alignment = level.get('sr_market_structure_alignment', 0.5)
            features.append(sr_market_structure_alignment)
            avg_test_strength = await self._calculate_average_test_strength(level)
            features.append(avg_test_strength)
            avg_breakout_strength = await self._calculate_average_breakout_strength(level)
            features.append(avg_breakout_strength)
            return features
        except Exception as e:
            self.logger.error(f'S/R specific feature extraction failed: {e}')
            return [0.5] * 8

    async def _create_target_for_level(self, level: Dict[str, Any], historical_performance: Optional[Dict[str, Any]]) -> float:
        """Create optimized target variable for S/R level quality based on trading performance."""
        try:
            if historical_performance and level.get('id') in historical_performance:
                perf = historical_performance[level['id']]
                return perf.get('quality_score', 0.5)
            weights = self.ml_config.get('target_weights', {})
            bounce_rate = level.get('bounce_rate', 0.5)
            volume_qualified_bounce_rate = await self._calculate_volume_qualified_bounce_rate(level)
            bounce_weight = weights.get('bounce_rate', 0.2)
            target += volume_qualified_bounce_rate * bounce_weight
            false_breakout_rate = level.get('false_breakout_rate', 0.0)
            volume_qualified_false_breakout_rate = await self._calculate_volume_qualified_false_breakout_rate(level)
            false_breakout_weight = weights.get('false_breakout_rate', 0.15)
            target -= volume_qualified_false_breakout_rate * false_breakout_weight
            volume_confirmation = level.get('volume_confirmation_score', 0.5)
            volume_weight = weights.get('volume_confirmation', 0.1)
            target += volume_confirmation * volume_weight
            timeframe_consistency = level.get('timeframe_consistency', 0.5)
            timeframe_weight = weights.get('timeframe_consistency', 0.1)
            target += timeframe_consistency * timeframe_weight
            touch_count = level.get('touch_count', 0)
            touch_score = min(touch_count / 10.0, 1.0)
            touch_weight = weights.get('touch_count', 0.05)
            target += touch_score * touch_weight
            strength = level.get('strength', 0.5)
            strength_weight = weights.get('strength', 0.08)
            target += strength * strength_weight
            confluence_score = level.get('confluence_score', 0.5)
            confluence_weight = weights.get('confluence_score', 0.07)
            target += confluence_score * confluence_weight
            hvn_strength = level.get('hvn_strength', 0.5)
            hvn_weight = weights.get('hvn_strength', 0.05)
            target += hvn_strength * hvn_weight
            fib_confluence = level.get('fib_confluence_count', 0)
            fib_score = min(fib_confluence / 3.0, 1.0)
            fib_weight = weights.get('fib_confluence', 0.05)
            target += fib_score * fib_weight
            retest_success = level.get('sr_retest_success_rate', 0.5)
            retest_weight = weights.get('retest_success_rate', 0.06)
            target += retest_success * retest_weight
            market_structure_alignment = level.get('sr_market_structure_alignment', 0.5)
            market_structure_weight = weights.get('market_structure_alignment', 0.05)
            target += market_structure_alignment * market_structure_weight
            psychological_strength = level.get('psychological_level_type', 0.0)
            psychological_weight = weights.get('psychological_strength', 0.04)
            target += psychological_strength * psychological_weight
            return min(max(target, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f'Target creation failed: {e}')
            return 0.5

    async def _get_feature_names(self) -> List[str]:
        """Get feature names for ML models (S/R specific features only)."""
        core_features = ['touch_count', 'strength', 'age_bars', 'avg_bounce_ratio', 'max_bounce_ratio', 'volume_confirmation_score', 'consistency_score', 'failure_count', 'proximity_to_level', 'level_density', 'confluence_score', 'time_since_touch', 'volume_at_touch', 'price_action_score', 'microstructure_score']
        hvn_features = ['hvn_strength', 'hvn_volume_ratio', 'hvn_touch_count', 'hvn_time_weight', 'hvn_price_accuracy']
        fibonacci_features = ['fib_level_type', 'fib_strength', 'fib_confluence_count', 'fib_timeframe_alignment', 'fib_volume_confirmation', 'fib_bounce_quality']
        psychological_features = ['psychological_level_type', 'round_number_strength', 'psychological_touch_count', 'psychological_volume_spike', 'psychological_bounce_ratio']
        pivot_features = ['pivot_type', 'pivot_strength', 'pivot_timeframe', 'pivot_confluence']
        trendline_features = ['trendline_type', 'trendline_strength', 'trendline_touch_count', 'trendline_angle']
        sr_specific_features = ['sr_type', 'sr_timeframe_confluence', 'sr_breakout_history', 'sr_retest_success_rate', 'sr_volume_profile_strength', 'sr_market_structure_alignment', 'avg_test_strength', 'avg_breakout_strength']
        return core_features + hvn_features + fibonacci_features + psychological_features + pivot_features + trendline_features + sr_specific_features

    async def _train_sr_quality_model(self, training_data: MLFeatureSet) -> None:
        """Train S/R quality prediction model using modern ML utilities."""
        with self.performance_profiler.profile('sr_quality_model_training'):
            try:
                X = training_data.features
                y = training_data.target
                
                # Use modern feature selection
                if len(X) > 50:
                    feature_names = await self._get_feature_names()
                    
                    # Use the modern feature selector
                    selected_features, feature_importance = await self.feature_selector.select_features(
                        X, y, feature_names,
                        method='composite',
                        target_features=50,
                        sr_priority=True
                    )
                    
                    # Get feature indices
                    feature_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
                    X = X[:, feature_indices]
                    
                    self.logger.info(f'🎯 Feature selection completed: {len(selected_features)} features selected')
                
                # Scale features
                X_scaled = self.feature_scaler.fit_transform(X)
                
                # Use model training manager for training
                model_config = {
                    'type': 'random_forest',
                    'parameters': {
                        'n_estimators': 200,
                        'max_depth': 8,
                        'min_samples_split': 10,
                        'min_samples_leaf': 5,
                        'max_features': 'sqrt',
                        'bootstrap': True,
                        'random_state': 42
                    }
                }
                
                self.sr_quality_model = await self.model_training_manager.train_model(
                    X_scaled, y,
                    model_config=model_config,
                    task_type='regression'
                )
                
                # Evaluate model using modern evaluator
                if len(X) > 20:
                    evaluation_results = await self.model_evaluator.evaluate_model(
                        self.sr_quality_model, X_scaled, y,
                        task_type='regression',
                        cross_validation=True
                    )
                    
                    accuracy = evaluation_results.get('cv_score', 0.0)
                    self.model_performance['sr_quality']['accuracy'] = accuracy
                    self.model_performance['sr_quality']['last_update'] = datetime.now()
                    
                    # Log comprehensive evaluation results
                    self.logger.info(f'✅ S/R quality model trained. CV Score: {accuracy:.4f}')
                    self.logger.info(f'   - R² Score: {evaluation_results.get("r2_score", 0.0):.4f}')
                    self.logger.info(f'   - MAE: {evaluation_results.get("mae", 0.0):.4f}')
                    self.logger.info(f'   - RMSE: {evaluation_results.get("rmse", 0.0):.4f}')
                else:
                    self.logger.info('✅ S/R quality model trained (insufficient data for evaluation)')
                    
            except Exception as e:
                self.logger.error(f'S/R quality model training failed: {e}')

    async def _train_breakout_prediction_model(self, training_data: MLFeatureSet) -> None:
        """Train breakout prediction model using modern ML utilities."""
        with self.performance_profiler.profile('breakout_prediction_model_training'):
            try:
                X = training_data.features
                
                # Create more realistic breakout targets based on actual data
                y_breakout = await self._create_breakout_targets(training_data)
                
                # Scale features
                X_scaled = self.feature_scaler.fit_transform(X)
                
                # Use model training manager
                model_config = {
                    'type': 'random_forest',
                    'parameters': {
                        'n_estimators': 200,
                        'max_depth': 8,
                        'min_samples_split': 10,
                        'min_samples_leaf': 5,
                        'random_state': 42,
                        'class_weight': 'balanced'
                    }
                }
                
                self.breakout_prediction_model = await self.model_training_manager.train_model(
                    X_scaled, y_breakout,
                    model_config=model_config,
                    task_type='classification'
                )
                
                # Evaluate model
                if len(X) > 20:
                    evaluation_results = await self.model_evaluator.evaluate_model(
                        self.breakout_prediction_model, X_scaled, y_breakout,
                        task_type='classification',
                        cross_validation=True
                    )
                    
                    accuracy = evaluation_results.get('cv_score', 0.0)
                    self.model_performance['breakout_prediction']['accuracy'] = accuracy
                    self.model_performance['breakout_prediction']['last_update'] = datetime.now()
                    
                    self.logger.info(f'✅ Breakout prediction model trained. CV Score: {accuracy:.4f}')
                    self.logger.info(f'   - Precision: {evaluation_results.get("precision", 0.0):.4f}')
                    self.logger.info(f'   - Recall: {evaluation_results.get("recall", 0.0):.4f}')
                    self.logger.info(f'   - F1 Score: {evaluation_results.get("f1_score", 0.0):.4f}')
                else:
                    self.logger.info('✅ Breakout prediction model trained (insufficient data for evaluation)')
                    
            except Exception as e:
                self.logger.error(f'Breakout prediction model training failed: {e}')
    
    async def _create_breakout_targets(self, training_data: MLFeatureSet) -> np.ndarray:
        """Create realistic breakout targets based on S/R level characteristics."""
        try:
            # Create targets based on level strength and characteristics
            targets = []
            for i, target in enumerate(training_data.target):
                # Higher quality levels are more likely to break out
                # But also consider other factors like age, touch count, etc.
                if target > 0.7:  # High quality levels
                    breakout_prob = 0.4  # 40% chance of breakout
                elif target > 0.5:  # Medium quality levels
                    breakout_prob = 0.3  # 30% chance of breakout
                else:  # Low quality levels
                    breakout_prob = 0.2  # 20% chance of breakout
                
                # Add some randomness but keep it realistic
                breakout_target = 1 if np.random.random() < breakout_prob else 0
                targets.append(breakout_target)
            
            return np.array(targets)
            
        except Exception as e:
            self.logger.error(f'Breakout target creation failed: {e}')
            # Fallback to random targets
            return np.random.choice([0, 1], size=len(training_data.target), p=[0.7, 0.3])

    async def _train_regime_classification_model(self, market_data: pd.DataFrame) -> None:
        """Use step03 regime detection with LGBM model instead of training new model."""
        try:
            self.logger.info('Using step03 regime detection with LGBM model')
            try:
                from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineeringRefactored
                self.step03_engineer = VectorizedAdvancedFeatureEngineeringRefactored()
                self.logger.info('✅ Step03 regime detection loaded successfully')
            except ImportError as e:
                self.logger.warning(f'Step03 regime detection not available: {e}')
                self.step03_engineer = None
            self.regime_classification_model = None
            regime_features = await self._extract_regime_features(market_data)
            regime_targets = await self._create_regime_targets(market_data)
            if len(regime_features) > 10:
                accuracy = self._validate_regime_detection(regime_features, regime_targets)
                self.logger.info(f'✅ Regime detection validation completed. Accuracy: {accuracy:.4f}')
                if self.step03_engineer:
                    try:
                        step03_features = await self.step03_engineer.engineer_features(market_data)
                        regime_features_step03 = step03_features.get('regime_features', [])
                        if len(regime_features_step03) > 0:
                            self.logger.info(f'✅ Step03 regime features extracted: {len(regime_features_step03)} features')
                        else:
                            self.logger.warning('Step03 regime features not found')
                    except Exception as e:
                        self.logger.warning(f'Step03 regime detection test failed: {e}')
            else:
                self.logger.warning('Insufficient data for regime validation')
        except Exception as e:
            self.logger.error(f'Regime classification setup failed: {e}')

    def _validate_regime_detection(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Validate regime detection accuracy using simple rules."""
        try:
            correct_predictions = 0
            total_predictions = len(targets)
            for i, (feature, target) in enumerate(zip(features, targets)):
                sma_ratio, rsi, volatility, volume_ratio, momentum = feature
                if abs(sma_ratio - 1.0) > 0.02 and 30 < rsi < 70:
                    predicted_regime = 0
                elif abs(sma_ratio - 1.0) <= 0.02:
                    predicted_regime = 1
                else:
                    predicted_regime = 2
                if predicted_regime == target:
                    correct_predictions += 1
            return correct_predictions / total_predictions if total_predictions > 0 else 0.0
        except Exception as e:
            self.logger.error(f'Regime validation failed: {e}')
            return 0.0

    async def _extract_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime classification."""
        try:
            features = []
            sma_20 = market_data['close'].rolling(window=20).mean()
            sma_50 = market_data['close'].rolling(window=50).mean()
            rsi = self._calculate_rsi(market_data['close'], 14)
            atr = self._calculate_atr(market_data, 14)
            for i in range(50, len(market_data)):
                feature_vector = [sma_20.iloc[i] / sma_50.iloc[i] if not sma_50.empty else 1.0, rsi.iloc[i] if not rsi.empty else 50.0, atr.iloc[i] / market_data['close'].iloc[i] if not atr.empty else 0.0, market_data['volume'].iloc[i] / market_data['volume'].rolling(window=20).mean().iloc[i] if i >= 20 else 1.0, (market_data['close'].iloc[i] - market_data['close'].iloc[i - 10]) / market_data['close'].iloc[i - 10] if i >= 10 else 0.0]
                features.append(feature_vector)
            return np.array(features) if features else np.array([]).reshape(0, 5)
        except Exception as e:
            self.logger.error(f'Regime feature extraction failed: {e}')
            return np.array([]).reshape(0, 5)

    async def _create_regime_targets(self, market_data: pd.DataFrame) -> np.ndarray:
        """Create regime classification targets."""
        try:
            targets = []
            sma_20 = market_data['close'].rolling(window=20).mean()
            sma_50 = market_data['close'].rolling(window=50).mean()
            rsi = self._calculate_rsi(market_data['close'], 14)
            for i in range(50, len(market_data)):
                sma_ratio = sma_20.iloc[i] / sma_50.iloc[i] if not sma_50.empty else 1.0
                rsi_val = rsi.iloc[i] if not rsi.empty else 50.0
                if abs(sma_ratio - 1.0) > 0.02 and 30 < rsi_val < 70:
                    regime = 0
                elif abs(sma_ratio - 1.0) <= 0.02:
                    regime = 1
                else:
                    regime = 2
                targets.append(regime)
            return np.array(targets)
        except Exception as e:
            self.logger.error(f'Regime target creation failed: {e}')
            return np.array([])

    async def predict_sr_quality(self, market_data: pd.DataFrame, sr_levels: List[Dict[str, Any]]) -> List[SRQualityPrediction]:
        """Predict quality of S/R levels using ML with modern utilities."""
        with self.performance_profiler.profile('sr_quality_prediction'):
            try:
                if not self.ml_enabled or not self.sr_quality_model:
                    return await self._fallback_quality_prediction(sr_levels)
                
                # Validate inputs
                if not self.validation_framework.validate_dataframe(market_data):
                    self.logger.error('Invalid market data for prediction')
                    return await self._fallback_quality_prediction(sr_levels)
                
                predictions = []
                
                # Process levels in parallel for better performance
                prediction_tasks = []
                for level in sr_levels:
                    task = self._predict_single_level_quality(market_data, level)
                    prediction_tasks.append(task)
                
                level_predictions = await self.parallel_processor.run_parallel(prediction_tasks)
                
                # Collect valid predictions
                for prediction in level_predictions:
                    if prediction is not None:
                        predictions.append(prediction)
                
                # Calculate confidence metrics using modern utilities
                if predictions:
                    confidence_metrics = calculate_confidence_metrics([p.quality_score for p in predictions])
                    self.logger.info(f'📊 Quality prediction completed: {len(predictions)} predictions, avg confidence: {confidence_metrics.get("mean_confidence", 0.0):.3f}')
                
                return predictions
                
            except Exception as e:
                self.logger.error(f'S/R quality prediction failed: {e}')
                return await self._fallback_quality_prediction(sr_levels)
    
    async def _predict_single_level_quality(self, market_data: pd.DataFrame, level: Dict[str, Any]) -> Optional[SRQualityPrediction]:
        """Predict quality for a single S/R level."""
        try:
            features = await self._extract_level_features(market_data, level)
            if not features:
                return None
            
            X = np.array([features])
            
            # Apply feature scaling
            if self.feature_scaler:
                X_scaled = self.feature_scaler.transform(X)
            else:
                X_scaled = X
            
            # Make prediction
            quality_score = self.sr_quality_model.predict(X_scaled)[0]
            
            # Calculate confidence using modern utilities
            confidence = min(abs(quality_score - 0.5) * 2, 1.0)
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(self.sr_quality_model, 'feature_importances_'):
                feature_names = await self._get_feature_names()
                feature_importance = dict(zip(feature_names, self.sr_quality_model.feature_importances_))
            
            return SRQualityPrediction(
                level_id=level.get('id', 'unknown'),
                quality_score=float(quality_score),
                confidence=confidence,
                features_used=await self._get_feature_names(),
                prediction_reason=f'ML prediction with {confidence:.2%} confidence'
            )
            
        except Exception as e:
            self.logger.error(f'Error predicting quality for level: {e}')
            return None

    async def predict_breakouts(self, market_data: pd.DataFrame, sr_levels: List[Dict[str, Any]]) -> List[BreakoutPrediction]:
        """Predict breakouts using ML."""
        try:
            if not self.ml_enabled or not self.breakout_prediction_model:
                return await self._fallback_breakout_prediction(sr_levels)
            predictions = []
            for level in sr_levels:
                features = await self._extract_level_features(market_data, level)
                if not features:
                    continue
                X = np.array([features])
                breakout_prob = self.breakout_prediction_model.predict_proba(X)[0][1]
                level_price = level.get('price', 0)
                current_price = market_data['close'].iloc[-1]
                direction = 'up' if current_price > level_price else 'down'
                confidence = abs(breakout_prob - 0.5) * 2
                prediction = BreakoutPrediction(level_id=level.get('id', 'unknown'), breakout_probability=float(breakout_prob), confidence=confidence, expected_direction=direction, time_to_breakout=None, features_used=await self._get_feature_names())
                predictions.append(prediction)
            return predictions
        except Exception as e:
            self.logger.error(f'Breakout prediction failed: {e}')
            return await self._fallback_breakout_prediction(sr_levels)

    async def _fallback_quality_prediction(self, sr_levels: List[Dict[str, Any]]) -> List[SRQualityPrediction]:
        """Fallback quality prediction using rule-based approach."""
        predictions = []
        for level in sr_levels:
            quality_score = level.get('strength', 0.5)
            confidence = 0.5
            prediction = SRQualityPrediction(level_id=level.get('id', 'unknown'), quality_score=quality_score, confidence=confidence, features_used=['strength'], prediction_reason='Rule-based fallback prediction')
            predictions.append(prediction)
        return predictions

    async def _fallback_breakout_prediction(self, sr_levels: List[Dict[str, Any]]) -> List[BreakoutPrediction]:
        """Fallback breakout prediction using rule-based approach."""
        predictions = []
        for level in sr_levels:
            breakout_prob = 0.3
            confidence = 0.3
            prediction = BreakoutPrediction(level_id=level.get('id', 'unknown'), breakout_probability=breakout_prob, confidence=confidence, expected_direction='unknown', time_to_breakout=None, features_used=['rule_based'])
            predictions.append(prediction)
        return predictions

    def _calculate_rsi(self, prices: pd.Series, period: int=14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - 100 / (1 + rs)
            return rsi.fillna(50)
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int=12, slow: int=26, signal: int=9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            return (macd_line, macd_signal)
        except Exception:
            return (pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index))

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int=20, std_dev: float=2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + std * std_dev
            lower = sma - std * std_dev
            return (upper, sma, lower)
        except Exception:
            return (pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index))

    def _calculate_atr(self, market_data: pd.DataFrame, period: int=14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            return atr.fillna(0)
        except Exception:
            return pd.Series([0] * len(market_data), index=market_data.index)

    def _calculate_stochastic(self, market_data: pd.DataFrame, k_period: int=14, d_period: int=3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        try:
            low_min = market_data['low'].rolling(window=k_period).min()
            high_max = market_data['high'].rolling(window=k_period).max()
            k_percent = 100 * ((market_data['close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=d_period).mean()
            return (k_percent.fillna(50), d_percent.fillna(50))
        except Exception:
            return (pd.Series([50] * len(market_data), index=market_data.index), pd.Series([50] * len(market_data), index=market_data.index))

    def _calculate_williams_r(self, market_data: pd.DataFrame, period: int=14) -> pd.Series:
        """Calculate Williams %R."""
        try:
            high_max = market_data['high'].rolling(window=period).max()
            low_min = market_data['low'].rolling(window=period).min()
            williams_r = -100 * ((high_max - market_data['close']) / (high_max - low_min))
            return williams_r.fillna(-50)
        except Exception:
            return pd.Series([-50] * len(market_data), index=market_data.index)

    def _calculate_cci(self, market_data: pd.DataFrame, period: int=20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        try:
            typical_price = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci.fillna(0)
        except Exception:
            return pd.Series([0] * len(market_data), index=market_data.index)

    def _calculate_adx(self, market_data: pd.DataFrame, period: int=14) -> pd.Series:
        """Calculate Average Directional Index."""
        try:
            high_diff = market_data['high'].diff()
            low_diff = market_data['low'].diff()
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            plus_dm = pd.Series(plus_dm, index=market_data.index)
            minus_dm = pd.Series(minus_dm, index=market_data.index)
            atr = self._calculate_atr(market_data, period)
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            return adx.fillna(25)
        except Exception:
            return pd.Series([25] * len(market_data), index=market_data.index)

    def _calculate_obv(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        try:
            price_change = market_data['close'].diff()
            obv = np.where(price_change > 0, market_data['volume'], np.where(price_change < 0, -market_data['volume'], 0))
            obv = pd.Series(obv, index=market_data.index).cumsum()
            return obv.fillna(0)
        except Exception:
            return pd.Series([0] * len(market_data), index=market_data.index)

    def _detect_doji_pattern(self, market_data: pd.DataFrame) -> float:
        """Detect Doji candlestick pattern."""
        try:
            if len(market_data) < 1:
                return 0.0
            current = market_data.iloc[-1]
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            return 1.0 if body_size / total_range < 0.1 else 0.0
        except Exception:
            return 0.0

    def _detect_hammer_pattern(self, market_data: pd.DataFrame) -> float:
        """Detect Hammer candlestick pattern."""
        try:
            if len(market_data) < 1:
                return 0.0
            current = market_data.iloc[-1]
            body_size = abs(current['close'] - current['open'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            total_range = current['high'] - current['low']
            is_hammer = lower_shadow > 2 * body_size and upper_shadow < body_size and (body_size / total_range < 0.3)
            return 1.0 if is_hammer else 0.0
        except Exception:
            return 0.0

    def _calculate_volatility_proxy(self, market_data: pd.DataFrame, period: int=20) -> float:
        """Calculate volatility proxy (simplified VIX)."""
        try:
            if len(market_data) < period:
                return 0.0
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(window=period).std().iloc[-1]
            return float(volatility * 100) if not np.isnan(volatility) else 0.0
        except Exception:
            return 0.0

    def _calculate_feature_correlations(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate correlation between features and target."""
        try:
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            return np.array(correlations)
        except Exception:
            return np.zeros(X.shape[1])

    async def _calculate_shap_importance(self, model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate SHAP importance scores."""
        try:
            try:
                import shap
                SHAP_AVAILABLE = True
            except ImportError:
                SHAP_AVAILABLE = False
                self.logger.warning('SHAP not available, skipping SHAP analysis')
                return {name: 0.0 for name in feature_names}
            if not SHAP_AVAILABLE or len(X) < 100:
                return {name: 0.0 for name in feature_names}
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:100])
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            return dict(zip(feature_names, mean_shap_values))
        except Exception as e:
            self.logger.warning(f'SHAP calculation failed: {e}')
            return {name: 0.0 for name in feature_names}

    def _combine_feature_scores(self, rf_importance: np.ndarray, perm_scores: np.ndarray, correlation_scores: np.ndarray, shap_scores: Dict[str, float]) -> np.ndarray:
        """Combine different feature importance scores."""
        try:
            rf_norm = rf_importance / (np.max(rf_importance) + 1e-08)
            perm_norm = perm_scores / (np.max(perm_scores) + 1e-08)
            corr_norm = correlation_scores / (np.max(correlation_scores) + 1e-08)
            shap_array = np.array([shap_scores.get(f'feature_{i}', 0.0) for i in range(len(rf_importance))])
            shap_norm = shap_array / (np.max(shap_array) + 1e-08)
            combined = rf_norm * 0.3 + perm_norm * 0.3 + corr_norm * 0.2 + shap_norm * 0.2
            return combined
        except Exception as e:
            self.logger.error(f'Feature score combination failed: {e}')
            return rf_importance

    def _select_top_features(self, scores: np.ndarray, feature_names: List[str], top_k: int=20) -> List[str]:
        """Select top K features based on combined scores."""
        try:
            top_indices = np.argsort(scores)[-top_k:]
            return [feature_names[i] for i in top_indices]
        except Exception as e:
            self.logger.error(f'Top feature selection failed: {e}')
            return feature_names[:top_k]

    def _select_top_features_with_sr_priority(self, scores: np.ndarray, feature_names: List[str], top_k: int=50) -> List[str]:
        """Select top K features with S/R feature prioritization."""
        try:
            sr_feature_patterns = ['touch_count', 'strength', 'age_bars', 'avg_bounce_ratio', 'max_bounce_ratio', 'volume_confirmation_score', 'consistency_score', 'failure_count', 'proximity_to_level', 'level_density', 'confluence_score', 'time_since_touch', 'volume_at_touch', 'price_action_score', 'microstructure_score', 'hvn_strength', 'hvn_volume_ratio', 'hvn_touch_count', 'hvn_time_weight', 'hvn_price_accuracy', 'fib_level_type', 'fib_strength', 'fib_confluence_count', 'fib_timeframe_alignment', 'fib_volume_confirmation', 'fib_bounce_quality', 'psychological_level_type', 'round_number_strength', 'psychological_touch_count', 'psychological_volume_spike', 'psychological_bounce_ratio', 'pivot_type', 'pivot_strength', 'pivot_timeframe', 'pivot_confluence', 'trendline_type', 'trendline_strength', 'trendline_touch_count', 'trendline_angle', 'sr_type', 'sr_timeframe_confluence', 'sr_breakout_history', 'sr_retest_success_rate', 'sr_volume_profile_strength', 'sr_market_structure_alignment', 'avg_test_strength', 'avg_breakout_strength']
            sr_features = []
            non_sr_features = []
            for i, feature_name in enumerate(feature_names):
                is_sr_feature = any((pattern in feature_name.lower() for pattern in sr_feature_patterns))
                if is_sr_feature:
                    sr_features.append((i, feature_name, scores[i]))
                else:
                    non_sr_features.append((i, feature_name, scores[i]))
            sr_features.sort(key=lambda x: x[2], reverse=True)
            non_sr_features.sort(key=lambda x: x[2], reverse=True)
            selected_features = []
            min_sr_ratio = self.ml_config.get('feature_selection', {}).get('min_sr_ratio', 0.7)
            min_sr_count = int(top_k * min_sr_ratio)
            sr_count = max(min_sr_count, min(len(sr_features), top_k))
            for i in range(sr_count):
                selected_features.append(sr_features[i][1])
            remaining_count = top_k - len(selected_features)
            for i in range(min(remaining_count, len(non_sr_features))):
                selected_features.append(non_sr_features[i][1])
            if len(selected_features) < top_k:
                all_features = [(i, feature_names[i], scores[i]) for i in range(len(feature_names))]
                all_features.sort(key=lambda x: x[2], reverse=True)
                for i, feature_name, score in all_features:
                    if feature_name not in selected_features and len(selected_features) < top_k:
                        selected_features.append(feature_name)
            self.logger.info(f'🎯 Feature selection with S/R prioritization:')
            self.logger.info(f'   - S/R features selected: {sr_count} (minimum {min_sr_ratio * 100:.0f}% of total)')
            self.logger.info(f'   - Non-S/R features selected: {len(selected_features) - sr_count}')
            self.logger.info(f'   - Total features selected: {len(selected_features)}')
            self.logger.info(f'   - S/R feature categories: Core(15), HVN(5), Fibonacci(6), Psychological(5), Pivot(4), Trendline(4), S/R Specific(8)')
            self.logger.info(f'   - Selection strategy: Minimum {min_sr_ratio * 100:.0f}% S/R features, no upper limit')
            return selected_features
        except Exception as e:
            self.logger.error(f'S/R prioritized feature selection failed: {e}')
            return self._select_top_features(scores, feature_names, top_k)

    def _log_feature_analysis(self) -> None:
        """Log comprehensive feature analysis results."""
        try:
            if not hasattr(self, 'feature_importance') or not self.feature_importance:
                return
            combined_scores = self.feature_importance.get('combined_scores', {})
            selected_features = self.feature_importance.get('selected_features', [])
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            self.logger.info('🔍 Comprehensive Feature Analysis Results:')
            self.logger.info(f'📊 Total features analyzed: {len(combined_scores)}')
            self.logger.info(f'🎯 Selected features: {len(selected_features)}')
            self.logger.info('🏆 Top 25 Most Important Features:')
            for i, (feature, score) in enumerate(sorted_features[:25]):
                status = '✅ SELECTED' if feature in selected_features else '❌ NOT SELECTED'
                feature_type = '🎯 S/R' if any((pattern in feature.lower() for pattern in ['proximity', 'level', 'touch', 'bounce', 'strength', 'rsi', 'macd', 'bollinger', 'atr', 'stoch', 'williams', 'cci', 'adx', 'obv', 'doji', 'hammer', 'volatility'])) else '📊 STEP06'
                self.logger.info(f'  {i + 1:2d}. {feature:<30} {score:.4f} {feature_type} {status}')
            rf_importance = self.feature_importance.get('rf_importance', {})
            perm_importance = self.feature_importance.get('permutation_importance', {})
            if rf_importance and perm_importance:
                self.logger.info('📈 Feature Selection Statistics:')
                self.logger.info(f'   - Random Forest top feature: {max(rf_importance.items(), key=lambda x: x[1])}')
                self.logger.info(f'   - Permutation top feature: {max(perm_importance.items(), key=lambda x: x[1])}')
        except Exception as e:
            self.logger.error(f'Feature analysis logging failed: {e}')

    def save_models(self, model_dir: str) -> bool:
        """Save trained models to disk using modern utilities."""
        with self.performance_profiler.profile('model_saving'):
            try:
                model_path = Path(model_dir)
                model_path.mkdir(parents=True, exist_ok=True)
                
                # Use cache manager for model persistence
                models_to_save = {
                    'sr_quality_model': self.sr_quality_model,
                    'breakout_prediction_model': self.breakout_prediction_model,
                    'regime_classification_model': self.regime_classification_model,
                    'feature_scaler': self.feature_scaler
                }
                
                for model_name, model in models_to_save.items():
                    if model is not None:
                        model_file = model_path / f'{model_name}.pkl'
                        joblib.dump(model, model_file)
                        
                        # Cache the model for faster loading
                        self.cache_manager.cache_model(model_name, model)
                
                # Save model metadata
                metadata = {
                    'model_performance': self.model_performance,
                    'feature_names': self.feature_names,
                    'training_timestamp': datetime.now().isoformat(),
                    'ml_config': self.ml_config
                }
                
                with open(model_path / 'model_metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                self.logger.info(f'✅ Models saved to {model_path} with metadata')
                return True
                
            except Exception as e:
                self.logger.error(f'Model saving failed: {e}')
                return False

    def load_models(self, model_dir: str) -> bool:
        """Load trained models from disk using modern utilities."""
        with self.performance_profiler.profile('model_loading'):
            try:
                model_path = Path(model_dir)
                
                # Check cache first
                cached_models = self.cache_manager.get_cached_models([
                    'sr_quality_model', 'breakout_prediction_model',
                    'regime_classification_model', 'feature_scaler'
                ])
                
                # Load models from disk if not cached
                models_to_load = {
                    'sr_quality_model': 'sr_quality_model.pkl',
                    'breakout_prediction_model': 'breakout_prediction_model.pkl',
                    'regime_classification_model': 'regime_classification_model.pkl',
                    'feature_scaler': 'feature_scaler.pkl'
                }
                
                for model_name, filename in models_to_load.items():
                    if model_name in cached_models:
                        setattr(self, model_name, cached_models[model_name])
                    elif (model_path / filename).exists():
                        model = joblib.load(model_path / filename)
                        setattr(self, model_name, model)
                        # Cache for future use
                        self.cache_manager.cache_model(model_name, model)
                
                # Load metadata if available
                metadata_file = model_path / 'model_metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.model_performance = metadata.get('model_performance', self.model_performance)
                        self.feature_names = metadata.get('feature_names', [])
                        
                        # Validate loaded models
                        if not self.validation_framework.validate_loaded_models(self):
                            self.logger.warning('Model validation failed, models may not work correctly')
                
                self.logger.info(f'✅ Models loaded from {model_path} with caching')
                return True
                
            except Exception as e:
                self.logger.error(f'Model loading failed: {e}')
                return False

    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get current model performance metrics."""
        return self.model_performance.copy()

    async def optimize_target_weights(self, market_data: pd.DataFrame, sr_levels: List[Dict[str, Any]], historical_performance: Optional[Dict[str, Any]]=None) -> Dict[str, float]:
        """Optimize target weights through backtesting and performance analysis."""
        try:
            self.logger.info('🔧 Optimizing target weights through backtesting...')
            weight_ranges = {'bounce_rate': (0.15, 0.25), 'false_breakout_rate': (0.1, 0.2), 'volume_confirmation': (0.08, 0.15), 'timeframe_consistency': (0.08, 0.15), 'touch_count': (0.03, 0.08), 'strength': (0.05, 0.12), 'confluence_score': (0.05, 0.1), 'hvn_strength': (0.03, 0.08), 'fib_confluence': (0.03, 0.08), 'retest_success_rate': (0.04, 0.08), 'market_structure_alignment': (0.03, 0.07), 'psychological_strength': (0.02, 0.06)}
            best_weights = {'bounce_rate': 0.2, 'false_breakout_rate': 0.15, 'volume_confirmation': 0.1, 'timeframe_consistency': 0.1, 'touch_count': 0.05, 'strength': 0.08, 'confluence_score': 0.07, 'hvn_strength': 0.05, 'fib_confluence': 0.05, 'retest_success_rate': 0.06, 'market_structure_alignment': 0.05, 'psychological_strength': 0.04}
            best_score = 0.0
            for iteration in range(10):
                candidate_weights = {}
                for param, (min_val, max_val) in weight_ranges.items():
                    current_val = best_weights[param]
                    noise = (max_val - min_val) * 0.1
                    candidate_weights[param] = max(min_val, min(max_val, current_val + np.random.normal(0, noise)))
                total_weight = sum(candidate_weights.values())
                candidate_weights = {k: v / total_weight for k, v in candidate_weights.items()}
                score = await self._evaluate_target_weights(candidate_weights, market_data, sr_levels, historical_performance)
                if score > best_score:
                    best_score = score
                    best_weights = candidate_weights.copy()
                    self.logger.info(f'   Iteration {iteration + 1}: New best score {score:.4f}')
            self.logger.info(f'✅ Target weight optimization completed. Best score: {best_score:.4f}')
            self.logger.info(f'   Optimized weights: {best_weights}')
            if 'target_weights' not in self.ml_config:
                self.ml_config['target_weights'] = {}
            self.ml_config['target_weights'].update(best_weights)
            return best_weights
        except Exception as e:
            self.logger.error(f'Target weight optimization failed: {e}')
            return self.ml_config.get('target_weights', {})

    async def _evaluate_target_weights(self, weights: Dict[str, float], market_data: pd.DataFrame, sr_levels: List[Dict[str, Any]], historical_performance: Optional[Dict[str, Any]]) -> float:
        """Evaluate target weights by measuring correlation with actual trading performance."""
        try:
            original_weights = self.ml_config.get('target_weights', {})
            self.ml_config['target_weights'] = weights
            targets = []
            actual_performance = []
            for level in sr_levels:
                target = await self._create_target_for_level(level, historical_performance)
                targets.append(target)
                if historical_performance and level.get('id') in historical_performance:
                    perf = historical_performance[level['id']]
                    actual_perf = perf.get('actual_bounce_rate', 0.5)
                    actual_performance.append(actual_perf)
                else:
                    actual_perf = level.get('bounce_rate', 0.5)
                    actual_performance.append(actual_perf)
            self.ml_config['target_weights'] = original_weights
            if len(targets) < 5:
                return 0.0
            correlation = np.corrcoef(targets, actual_performance)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except Exception as e:
            self.logger.error(f'Target weight evaluation failed: {e}')
            return 0.0

    async def _calculate_volume_qualified_bounce_rate(self, level: Dict[str, Any]) -> float:
        """Calculate bounce rate qualified by volume and momentum strength."""
        try:
            base_bounce_rate = level.get('bounce_rate', 0.5)
            test_data = level.get('test_history', [])
            if not test_data:
                return base_bounce_rate
            qualified_bounces = 0
            total_tests = len(test_data)
            for test in test_data:
                volume_ratio = test.get('volume_ratio', 1.0)
                momentum_strength = test.get('momentum_strength', 0.5)
                test_duration = test.get('test_duration', 1)
                wick_penetration = test.get('wick_penetration', 0.0)
                step06_penetration_features = test.get('step06_penetration_features', None)
                test_strength = self._calculate_test_strength(volume_ratio, momentum_strength, test_duration, wick_penetration, step06_penetration_features)
                if test.get('bounced', False) and test_strength > 0.6:
                    qualified_bounces += 1
                elif test.get('bounced', False) and test_strength > 0.3:
                    qualified_bounces += 0.7
                elif test.get('bounced', False):
                    qualified_bounces += 0.3
            if total_tests == 0:
                return base_bounce_rate
            qualified_bounce_rate = qualified_bounces / total_tests
            return min(max(qualified_bounce_rate, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f'Volume qualified bounce rate calculation failed: {e}')
            return level.get('bounce_rate', 0.5)

    async def _calculate_volume_qualified_false_breakout_rate(self, level: Dict[str, Any]) -> float:
        """Calculate false breakout rate qualified by volume and momentum context."""
        try:
            base_false_breakout_rate = level.get('false_breakout_rate', 0.0)
            breakout_data = level.get('breakout_history', [])
            if not breakout_data:
                return base_false_breakout_rate
            qualified_false_breakouts = 0
            total_breakouts = len(breakout_data)
            for breakout in breakout_data:
                volume_ratio = breakout.get('volume_ratio', 1.0)
                momentum_strength = breakout.get('momentum_strength', 0.5)
                breakout_duration = breakout.get('breakout_duration', 1)
                retest_success = breakout.get('retest_success', False)
                breakout_strength = self._calculate_breakout_strength(volume_ratio, momentum_strength, breakout_duration)
                if not retest_success:
                    if breakout_strength > 0.7:
                        qualified_false_breakouts += 1.0
                    elif breakout_strength > 0.4:
                        qualified_false_breakouts += 0.7
                    else:
                        qualified_false_breakouts += 0.3
            if total_breakouts == 0:
                return base_false_breakout_rate
            qualified_false_breakout_rate = qualified_false_breakouts / total_breakouts
            return min(max(qualified_false_breakout_rate, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f'Volume qualified false breakout rate calculation failed: {e}')
            return level.get('false_breakout_rate', 0.0)

    def _calculate_test_strength(self, volume_ratio: float, momentum_strength: float, test_duration: int, wick_penetration: float, step06_penetration_features: Optional[Dict[str, float]]=None) -> float:
        """Calculate the strength of a test based on volume, momentum, duration, and penetration."""
        try:
            volume_score = min(volume_ratio / 2.0, 1.0)
            momentum_score = momentum_strength
            duration_score = min(test_duration / 5.0, 1.0)
            if step06_penetration_features:
                upper_wick_pen = step06_penetration_features.get('upper_wick_penetration', 0.0)
                lower_wick_pen = step06_penetration_features.get('lower_wick_penetration', 0.0)
                body_pen_ratio = step06_penetration_features.get('body_penetration_ratio', 0.0)
                combined_penetration = max(upper_wick_pen, lower_wick_pen) + body_pen_ratio * 0.5
                penetration_score = min(combined_penetration / 0.02, 1.0)
            else:
                penetration_score = min(wick_penetration / 0.02, 1.0)
            test_strength = volume_score * 0.4 + momentum_score * 0.3 + duration_score * 0.2 + penetration_score * 0.1
            return min(max(test_strength, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f'Test strength calculation failed: {e}')
            return 0.5

    def _calculate_breakout_strength(self, volume_ratio: float, momentum_strength: float, breakout_duration: int) -> float:
        """Calculate the strength of a breakout based on volume, momentum, and duration."""
        try:
            volume_score = min(volume_ratio / 1.5, 1.0)
            momentum_score = momentum_strength
            duration_score = min(breakout_duration / 3.0, 1.0)
            breakout_strength = volume_score * 0.5 + momentum_score * 0.35 + duration_score * 0.15
            return min(max(breakout_strength, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f'Breakout strength calculation failed: {e}')
            return 0.5

    async def _calculate_average_test_strength(self, level: Dict[str, Any]) -> float:
        """Calculate average test strength for the level."""
        try:
            test_data = level.get('test_history', [])
            if not test_data:
                return 0.5
            total_strength = 0.0
            valid_tests = 0
            for test in test_data:
                volume_ratio = test.get('volume_ratio', 1.0)
                momentum_strength = test.get('momentum_strength', 0.5)
                test_duration = test.get('test_duration', 1)
                wick_penetration = test.get('wick_penetration', 0.0)
                test_strength = self._calculate_test_strength(volume_ratio, momentum_strength, test_duration, wick_penetration)
                total_strength += test_strength
                valid_tests += 1
            if valid_tests == 0:
                return 0.5
            return total_strength / valid_tests
        except Exception as e:
            self.logger.error(f'Average test strength calculation failed: {e}')
            return 0.5

    async def _calculate_average_breakout_strength(self, level: Dict[str, Any]) -> float:
        """Calculate average breakout strength for the level."""
        try:
            breakout_data = level.get('breakout_history', [])
            if not breakout_data:
                return 0.5
            total_strength = 0.0
            valid_breakouts = 0
            for breakout in breakout_data:
                volume_ratio = breakout.get('volume_ratio', 1.0)
                momentum_strength = breakout.get('momentum_strength', 0.5)
                breakout_duration = breakout.get('breakout_duration', 1)
                breakout_strength = self._calculate_breakout_strength(volume_ratio, momentum_strength, breakout_duration)
                total_strength += breakout_strength
                valid_breakouts += 1
            if valid_breakouts == 0:
                return 0.5
            return total_strength / valid_breakouts
        except Exception as e:
            self.logger.error(f'Average breakout strength calculation failed: {e}')
            return 0.5