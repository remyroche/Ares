"""
Enhanced Profit Labeling System with Integrated Tools

This module implements a comprehensive profit labeling system that integrates:
- KlinesParquetManager for efficient data loading
- VectorBTRollingOptimizer for vectorized computations
- Feature bank and generation utilities
- Feature selection methods (mRMR, LASSO, RFE)
- Bayesian TPE optimization
- Hardware optimization utilities
- Serialization utilities

Author: AI Assistant
Date: 2025-01-10
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import gc

# Core imports
from ..consolidated_profit_labeler import ConsolidatedProfitLabeler
from ..enhanced_label_definitions import EnhancedLabelDefinitions
from ..multi_target_scheme import MultiTargetScheme
from ..volatility_modeling import VolatilityModeling
from ..quality_scoring import QualityScoring

# Data utilities
try:
    from src.utils.data.klines_parquet import KlinesParquetManager
    from src.utils.data.unified_data_utils import UnifiedDataUtils
    from src.utils.serialization_utils import JSONSerializer, PickleSerializer
    DATA_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Data utilities not available: {e}")
    DATA_UTILS_AVAILABLE = False

# Feature generation and selection
try:
    from src.feature_generation.core.feature_bank import FeatureBank
    from src.feature_generation.core.feature_generator import FeatureGenerator
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.features_common.transforms.vectorbt_scaler import VectorBTScaler
    from src.feature_selection.advanced.enhanced_advanced_selector import EnhancedAdvancedSelector
    from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
    FEATURE_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Feature utilities not available: {e}")
    FEATURE_UTILS_AVAILABLE = False

# ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    from src.utils.ml_common.explainability.shap_lime_integration import SHAPLimeIntegration
    from src.utils.ml_common.evaluation.unified_evaluator import UnifiedEvaluator
    ML_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML utilities not available: {e}")
    ML_UTILS_AVAILABLE = False

# Hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.optimization_decorators import performance_tracked, memory_optimized
    from src.utils.hardware.enhanced_cpu_optimizer import EnhancedCPUOptimizer
    HARDWARE_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware utilities not available: {e}")
    HARDWARE_UTILS_AVAILABLE = False

# Enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback functions
    def tprint(*args, **kwargs): print(f"[TPRINT] {' '.join(map(str, args))}")
    def tprint_info(*args, **kwargs): print(f"[INFO] {' '.join(map(str, args))}")
    def tprint_success(*args, **kwargs): print(f"[SUCCESS] {' '.join(map(str, args))}")
    def tprint_warning(*args, **kwargs): print(f"[WARNING] {' '.join(map(str, args))}")
    def tprint_error(*args, **kwargs): print(f"[ERROR] {' '.join(map(str, args))}")

logger = logging.getLogger(__name__)


@dataclass
class ProfitLabelingConfig:
    """Configuration for the enhanced profit labeling system."""
    
    # Data configuration
    data_path: str = "data/klines"
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    
    # Feature configuration
    feature_categories: List[str] = field(default_factory=lambda: [
        "volatility", "momentum", "volume", "trend", "oscillator"
    ])
    max_features: int = 1000
    feature_selection_method: str = "mrmr"  # mrmr, lasso, rfe, ensemble
    
    # Labeling configuration
    volatility_threshold: float = 0.02
    horizon_quantiles: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    target_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "small": 0.01, "medium": 0.02, "high": 0.05
    })
    
    # Optimization configuration
    enable_bayesian_optimization: bool = True
    n_trials: int = 100
    n_jobs: int = -1
    
    # Hardware configuration
    enable_gpu: bool = False
    enable_parallel: bool = True
    memory_efficient: bool = True
    
    # Quality configuration
    min_quality_score: float = 0.7
    enable_noise_gating: bool = True
    enable_leakage_detection: bool = True


class EnhancedProfitLabelingSystem:
    """
    Enhanced Profit Labeling System with integrated tools.
    
    This system provides a comprehensive profit labeling pipeline that integrates
    all the available utilities for optimal performance and functionality.
    """
    
    def __init__(self, config: ProfitLabelingConfig):
        """Initialize the enhanced profit labeling system."""
        self.config = config
        self.logger = logger
        
        # Initialize components
        self._initialize_data_managers()
        self._initialize_feature_system()
        self._initialize_ml_utilities()
        self._initialize_hardware_optimization()
        self._initialize_labeling_components()
        
        tprint_success("🚀 Enhanced Profit Labeling System initialized")
    
    def _initialize_data_managers(self):
        """Initialize data loading and management components."""
        if DATA_UTILS_AVAILABLE:
            self.klines_manager = KlinesParquetManager(
                data_path=self.config.data_path,
                enable_parallel=self.config.enable_parallel
            )
            self.data_utils = UnifiedDataUtils()
            self.serializer = JSONSerializer()
        else:
            self.klines_manager = None
            self.data_utils = None
            self.serializer = None
            tprint_warning("⚠️ Data utilities not available - using fallback methods")
    
    def _initialize_feature_system(self):
        """Initialize feature generation and selection components."""
        if FEATURE_UTILS_AVAILABLE:
            # Initialize VectorBT rolling optimizer
            self.rolling_optimizer = VectorBTRollingOptimizer(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient
            )
            
            # Initialize feature bank and generator
            self.feature_bank = FeatureBank(
                categories=self.config.feature_categories,
                rolling_optimizer=self.rolling_optimizer
            )
            self.feature_generator = FeatureGenerator(
                feature_bank=self.feature_bank,
                rolling_optimizer=self.rolling_optimizer
            )
            
            # Initialize scaler
            self.scaler = VectorBTScaler(
                use_optimizer=True,
                rolling_optimizer=self.rolling_optimizer
            )
            
            # Initialize feature selectors
            self._initialize_feature_selectors()
        else:
            self.rolling_optimizer = None
            self.feature_bank = None
            self.feature_generator = None
            self.scaler = None
            self.feature_selectors = {}
            tprint_warning("⚠️ Feature utilities not available - using fallback methods")
    
    def _initialize_feature_selectors(self):
        """Initialize feature selection methods."""
        self.feature_selectors = {}
        
        if self.config.feature_selection_method == "mrmr":
            self.feature_selectors["mrmr"] = VectorBTMRMRSelector(
                rolling_optimizer=self.rolling_optimizer
            )
        elif self.config.feature_selection_method == "ensemble":
            from src.feature_selection.advanced.enhanced_ensemble_selector import EnhancedEnsembleSelector
            self.feature_selectors["ensemble"] = EnhancedEnsembleSelector(
                methods=["mrmr", "lasso", "rfe"],
                rolling_optimizer=self.rolling_optimizer
            )
        else:
            self.feature_selectors["advanced"] = EnhancedAdvancedSelector(
                method=self.config.feature_selection_method,
                rolling_optimizer=self.rolling_optimizer
            )
    
    def _initialize_ml_utilities(self):
        """Initialize ML common utilities."""
        if ML_UTILS_AVAILABLE:
            # Initialize optimization
            if self.config.enable_bayesian_optimization:
                self.optimizer = BayesianTPEOptimizer(
                    n_trials=self.config.n_trials,
                    n_jobs=self.config.n_jobs
                )
            else:
                self.optimizer = None
            
            # Initialize vectorization manager
            self.vectorization_manager = UnifiedVectorizationManager(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel
            )
            
            # Initialize explainability
            self.explainability = SHAPLimeIntegration()
            
            # Initialize evaluator
            self.evaluator = UnifiedEvaluator()
        else:
            self.optimizer = None
            self.vectorization_manager = None
            self.explainability = None
            self.evaluator = None
            tprint_warning("⚠️ ML utilities not available - using fallback methods")
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        if HARDWARE_UTILS_AVAILABLE:
            self.hardware_manager = UnifiedHardwareManager()
            self.cpu_optimizer = EnhancedCPUOptimizer()
        else:
            self.hardware_manager = None
            self.cpu_optimizer = None
            tprint_warning("⚠️ Hardware utilities not available - using fallback methods")
    
    def _initialize_labeling_components(self):
        """Initialize profit labeling components."""
        # Initialize core labeling components
        self.label_definitions = EnhancedLabelDefinitions()
        self.volatility_modeling = VolatilityModeling()
        self.quality_scoring = QualityScoring()
        self.multi_target_scheme = MultiTargetScheme()
        
        # Initialize consolidated labeler
        self.profit_labeler = ConsolidatedProfitLabeler(
            volatility_threshold=self.config.volatility_threshold,
            horizon_quantiles=self.config.horizon_quantiles,
            target_thresholds=self.config.target_thresholds,
            enable_noise_gating=self.config.enable_noise_gating,
            enable_leakage_detection=self.config.enable_leakage_detection
        )
    
    @performance_tracked
    def load_data(self, symbols: Optional[List[str]] = None, 
                  timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load kline data using KlinesParquetManager.
        
        Args:
            symbols: List of symbols to load
            timeframes: List of timeframes to load
            
        Returns:
            Dictionary of DataFrames keyed by symbol_timeframe
        """
        symbols = symbols or self.config.symbols
        timeframes = timeframes or self.config.timeframes
        
        tprint_info(f"📊 Loading data for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        data = {}
        
        if self.klines_manager:
            try:
                for symbol in symbols:
                    for timeframe in timeframes:
                        key = f"{symbol}_{timeframe}"
                        tprint_info(f"Loading {key}...")
                        
                        df = self.klines_manager.load_klines(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=self.config.start_date,
                            end_date=self.config.end_date
                        )
                        
                        if df is not None and not df.empty:
                            data[key] = df
                            tprint_success(f"✅ Loaded {key}: {len(df)} bars")
                        else:
                            tprint_warning(f"⚠️ No data for {key}")
                            
            except Exception as e:
                tprint_error(f"❌ Error loading data: {e}")
                raise
        else:
            tprint_warning("⚠️ KlinesParquetManager not available - using mock data")
            # Generate mock data for testing
            for symbol in symbols:
                for timeframe in timeframes:
                    key = f"{symbol}_{timeframe}"
                    data[key] = self._generate_mock_data(symbol, timeframe)
        
        tprint_success(f"🎯 Loaded {len(data)} datasets")
        return data
    
    def _generate_mock_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate mock kline data for testing."""
        periods = 1000
        dates = pd.date_range(
            start=self.config.start_date,
            periods=periods,
            freq='1H' if timeframe == '1h' else '4H' if timeframe == '4h' else '1D'
        )
        
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = {
            'open': prices * (1 + np.random.normal(0, 0.001, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': np.random.exponential(1000, periods)
        }
        
        df = pd.DataFrame(data, index=dates)
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        return df
    
    @memory_optimized
    def generate_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate features using the feature bank and VectorBTRollingOptimizer.
        
        Args:
            data: Dictionary of DataFrames keyed by symbol_timeframe
            
        Returns:
            Dictionary of DataFrames with generated features
        """
        tprint_info("🔧 Generating features...")
        
        features = {}
        
        if self.feature_generator and self.rolling_optimizer:
            try:
                for key, df in data.items():
                    tprint_info(f"Generating features for {key}...")
                    
                    # Generate features using the feature generator
                    feature_df = self.feature_generator.generate_features(
                        df,
                        categories=self.config.feature_categories,
                        rolling_optimizer=self.rolling_optimizer
                    )
                    
                    if feature_df is not None and not feature_df.empty:
                        features[key] = feature_df
                        tprint_success(f"✅ Generated {len(feature_df.columns)} features for {key}")
                    else:
                        tprint_warning(f"⚠️ No features generated for {key}")
                        
            except Exception as e:
                tprint_error(f"❌ Error generating features: {e}")
                raise
        else:
            tprint_warning("⚠️ Feature generation not available - using basic features")
            # Generate basic features as fallback
            for key, df in data.items():
                features[key] = self._generate_basic_features(df)
        
        tprint_success(f"🎯 Generated features for {len(features)} datasets")
        return features
    
    def _generate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate basic features as fallback."""
        features = df.copy()
        
        # Basic technical indicators
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['volatility'] = features['returns'].rolling(20).std()
        features['sma_20'] = features['close'].rolling(20).mean()
        features['sma_50'] = features['close'].rolling(50).mean()
        features['rsi'] = self._calculate_rsi(features['close'])
        features['bb_upper'] = features['sma_20'] + 2 * features['volatility']
        features['bb_lower'] = features['sma_20'] - 2 * features['volatility']
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def select_features(self, features: Dict[str, pd.DataFrame], 
                       labels: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Select features using the configured feature selection method.
        
        Args:
            features: Dictionary of feature DataFrames
            labels: Dictionary of label DataFrames
            
        Returns:
            Dictionary of selected feature names for each dataset
        """
        tprint_info("🎯 Selecting features...")
        
        selected_features = {}
        
        if self.feature_selectors:
            try:
                for key in features.keys():
                    if key in labels:
                        tprint_info(f"Selecting features for {key}...")
                        
                        # Prepare data for feature selection
                        X = features[key].select_dtypes(include=[np.number]).fillna(0)
                        y = labels[key].select_dtypes(include=[np.number]).fillna(0)
                        
                        # Select features
                        selector = list(self.feature_selectors.values())[0]
                        selected = selector.select_features(
                            X, y, 
                            max_features=self.config.max_features
                        )
                        
                        selected_features[key] = selected
                        tprint_success(f"✅ Selected {len(selected)} features for {key}")
                        
            except Exception as e:
                tprint_error(f"❌ Error selecting features: {e}")
                raise
        else:
            tprint_warning("⚠️ Feature selection not available - using all features")
            # Use all numeric features as fallback
            for key, df in features.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                selected_features[key] = numeric_cols[:self.config.max_features]
        
        tprint_success(f"🎯 Selected features for {len(selected_features)} datasets")
        return selected_features
    
    def generate_labels(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate profit labels using the consolidated profit labeler.
        
        Args:
            data: Dictionary of DataFrames with kline data
            
        Returns:
            Dictionary of DataFrames with generated labels
        """
        tprint_info("🏷️ Generating profit labels...")
        
        labels = {}
        
        try:
            for key, df in data.items():
                tprint_info(f"Generating labels for {key}...")
                
                # Generate labels using the consolidated profit labeler
                label_df = self.profit_labeler.generate_labels(
                    df,
                    volatility_threshold=self.config.volatility_threshold,
                    horizon_quantiles=self.config.horizon_quantiles,
                    target_thresholds=self.config.target_thresholds
                )
                
                if label_df is not None and not label_df.empty:
                    labels[key] = label_df
                    tprint_success(f"✅ Generated {len(label_df.columns)} labels for {key}")
                else:
                    tprint_warning(f"⚠️ No labels generated for {key}")
                    
        except Exception as e:
            tprint_error(f"❌ Error generating labels: {e}")
            raise
        
        tprint_success(f"🎯 Generated labels for {len(labels)} datasets")
        return labels
    
    def optimize_hyperparameters(self, features: Dict[str, pd.DataFrame], 
                                labels: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian TPE optimization.
        
        Args:
            features: Dictionary of feature DataFrames
            labels: Dictionary of label DataFrames
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        if not self.optimizer:
            tprint_warning("⚠️ Bayesian optimization not available")
            return {}
        
        tprint_info("🔧 Optimizing hyperparameters...")
        
        try:
            # Define optimization objective
            def objective(trial):
                # Sample hyperparameters
                volatility_threshold = trial.suggest_float('volatility_threshold', 0.01, 0.05)
                small_threshold = trial.suggest_float('small_threshold', 0.005, 0.02)
                medium_threshold = trial.suggest_float('medium_threshold', 0.01, 0.03)
                high_threshold = trial.suggest_float('high_threshold', 0.02, 0.08)
                
                # Calculate objective (e.g., label quality score)
                total_score = 0
                count = 0
                
                for key in features.keys():
                    if key in labels:
                        # Generate labels with current hyperparameters
                        temp_config = self.config
                        temp_config.volatility_threshold = volatility_threshold
                        temp_config.target_thresholds = {
                            'small': small_threshold,
                            'medium': medium_threshold,
                            'high': high_threshold
                        }
                        
                        # Calculate quality score
                        score = self._calculate_label_quality(
                            features[key], labels[key], temp_config
                        )
                        total_score += score
                        count += 1
                
                return total_score / count if count > 0 else 0
            
            # Run optimization
            best_params = self.optimizer.optimize(objective)
            
            tprint_success(f"✅ Optimized hyperparameters: {best_params}")
            return best_params
            
        except Exception as e:
            tprint_error(f"❌ Error optimizing hyperparameters: {e}")
            return {}
    
    def _calculate_label_quality(self, features: pd.DataFrame, 
                                labels: pd.DataFrame, config: ProfitLabelingConfig) -> float:
        """Calculate label quality score."""
        try:
            # Use quality scoring if available
            if self.quality_scoring:
                quality_score = self.quality_scoring.calculate_quality_score(
                    features, labels, config
                )
                return quality_score
            else:
                # Simple quality metric as fallback
                return labels.select_dtypes(include=[np.number]).mean().mean()
        except:
            return 0.0
    
    def evaluate_labels(self, features: Dict[str, pd.DataFrame], 
                       labels: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Evaluate label quality and performance.
        
        Args:
            features: Dictionary of feature DataFrames
            labels: Dictionary of label DataFrames
            
        Returns:
            Dictionary of evaluation results
        """
        tprint_info("📊 Evaluating labels...")
        
        evaluation_results = {}
        
        try:
            for key in features.keys():
                if key in labels:
                    tprint_info(f"Evaluating labels for {key}...")
                    
                    # Calculate evaluation metrics
                    metrics = self._calculate_evaluation_metrics(
                        features[key], labels[key]
                    )
                    
                    evaluation_results[key] = metrics
                    tprint_success(f"✅ Evaluated labels for {key}")
                    
        except Exception as e:
            tprint_error(f"❌ Error evaluating labels: {e}")
            raise
        
        tprint_success(f"🎯 Evaluated labels for {len(evaluation_results)} datasets")
        return evaluation_results
    
    def _calculate_evaluation_metrics(self, features: pd.DataFrame, 
                                     labels: pd.DataFrame) -> Dict[str, float]:
        """Calculate evaluation metrics for labels."""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['label_count'] = len(labels)
            metrics['feature_count'] = len(features.columns)
            metrics['label_mean'] = labels.select_dtypes(include=[np.number]).mean().mean()
            metrics['label_std'] = labels.select_dtypes(include=[np.number]).std().mean()
            metrics['label_balance'] = self._calculate_balance_score(labels)
            metrics['label_stability'] = self._calculate_stability_score(labels)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating metrics: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def _calculate_balance_score(self, labels: pd.DataFrame) -> float:
        """Calculate label balance score."""
        try:
            numeric_labels = labels.select_dtypes(include=[np.number])
            if numeric_labels.empty:
                return 0.0
            
            # Calculate class balance
            balance_scores = []
            for col in numeric_labels.columns:
                value_counts = numeric_labels[col].value_counts()
                if len(value_counts) > 1:
                    balance = 1 - (value_counts.max() - value_counts.min()) / value_counts.sum()
                    balance_scores.append(balance)
            
            return np.mean(balance_scores) if balance_scores else 0.0
        except:
            return 0.0
    
    def _calculate_stability_score(self, labels: pd.DataFrame) -> float:
        """Calculate label stability score."""
        try:
            numeric_labels = labels.select_dtypes(include=[np.number])
            if numeric_labels.empty:
                return 0.0
            
            # Calculate stability as inverse of variance
            stability_scores = []
            for col in numeric_labels.columns:
                variance = numeric_labels[col].var()
                stability = 1 / (1 + variance) if variance > 0 else 1.0
                stability_scores.append(stability)
            
            return np.mean(stability_scores) if stability_scores else 0.0
        except:
            return 0.0
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save results using serialization utilities."""
        if self.serializer:
            try:
                success = self.serializer.save(results, filepath)
                if success:
                    tprint_success(f"✅ Results saved to {filepath}")
                else:
                    tprint_error(f"❌ Failed to save results to {filepath}")
            except Exception as e:
                tprint_error(f"❌ Error saving results: {e}")
        else:
            tprint_warning("⚠️ Serialization not available - results not saved")
    
    def run_full_pipeline(self, symbols: Optional[List[str]] = None, 
                         timeframes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete profit labeling pipeline.
        
        Args:
            symbols: List of symbols to process
            timeframes: List of timeframes to process
            
        Returns:
            Dictionary containing all pipeline results
        """
        tprint_info("🚀 Starting full profit labeling pipeline...")
        
        pipeline_results = {
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'data': {},
            'features': {},
            'labels': {},
            'selected_features': {},
            'evaluation': {},
            'optimization': {}
        }
        
        try:
            # Step 1: Load data
            tprint_info("📊 Step 1: Loading data...")
            data = self.load_data(symbols, timeframes)
            pipeline_results['data'] = {k: len(v) for k, v in data.items()}
            
            # Step 2: Generate features
            tprint_info("🔧 Step 2: Generating features...")
            features = self.generate_features(data)
            pipeline_results['features'] = {k: len(v.columns) for k, v in features.items()}
            
            # Step 3: Generate labels
            tprint_info("🏷️ Step 3: Generating labels...")
            labels = self.generate_labels(data)
            pipeline_results['labels'] = {k: len(v.columns) for k, v in labels.items()}
            
            # Step 4: Select features
            tprint_info("🎯 Step 4: Selecting features...")
            selected_features = self.select_features(features, labels)
            pipeline_results['selected_features'] = {k: len(v) for k, v in selected_features.items()}
            
            # Step 5: Optimize hyperparameters
            if self.config.enable_bayesian_optimization:
                tprint_info("🔧 Step 5: Optimizing hyperparameters...")
                optimization_results = self.optimize_hyperparameters(features, labels)
                pipeline_results['optimization'] = optimization_results
            
            # Step 6: Evaluate labels
            tprint_info("📊 Step 6: Evaluating labels...")
            evaluation_results = self.evaluate_labels(features, labels)
            pipeline_results['evaluation'] = evaluation_results
            
            # Step 7: Save results
            tprint_info("💾 Step 7: Saving results...")
            results_file = f"profit_labeling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_results(pipeline_results, results_file)
            
            tprint_success("🎉 Full pipeline completed successfully!")
            
        except Exception as e:
            tprint_error(f"❌ Pipeline failed: {e}")
            pipeline_results['error'] = str(e)
            raise
        
        return pipeline_results


def main():
    """Main function to demonstrate the enhanced profit labeling system."""
    # Create configuration
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1h", "4h"],
        max_features=500,
        enable_bayesian_optimization=True,
        n_trials=50
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    # Run pipeline
    results = system.run_full_pipeline()
    
    # Print summary
    tprint_info("📋 Pipeline Summary:")
    tprint_info(f"  - Datasets processed: {len(results['data'])}")
    tprint_info(f"  - Features generated: {sum(results['features'].values())}")
    tprint_info(f"  - Labels generated: {sum(results['labels'].values())}")
    tprint_info(f"  - Features selected: {sum(results['selected_features'].values())}")
    
    if results['optimization']:
        tprint_info(f"  - Optimization completed: {len(results['optimization'])} parameters")
    
    return results


if __name__ == "__main__":
    main()