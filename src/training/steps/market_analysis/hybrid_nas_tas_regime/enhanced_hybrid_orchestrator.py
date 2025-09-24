"""
Enhanced Hybrid Orchestrator for NAS-TAS Regime System

This orchestrator can initialize both TAS and NAS systems, feed them data,
get their outputs, and analyze them to create its own regime clusters.
It also supports multi-timeframe trading (1m, 5m) while maintaining 15m regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Import shared utilities
from .shared_utils.unified_search_algorithms import UnifiedSearchManager, create_unified_search_manager
from .shared_utils.unified_clustering_algorithms import UnifiedClusteringAlgorithm, create_unified_clustering_algorithm

# Import TAS and NAS components
from .components.tas_integration import TASIntegrationComponent
from .components.nas_integration import NASIntegrationComponent

# Import unified architecture search engine
from .core.unified_architecture_search_engine import (
    UnifiedArchitectureSearchEngine, UnifiedSearchConfig, ArchitectureType, SearchMode
)
from .core.performance_estimator import UnifiedPerformanceEstimator
from .core.advanced_search_strategies import AdvancedSearchStrategies
from .core.multi_objective_optimizer import TradingMultiObjectiveOptimizer, MultiObjectiveConfig, ObjectiveType
from .core.nas_financial_features import NASFinancialFeatureEngineer
from .core.nas_financial_optimizer import NASFinancialOptimizer
from .core.architecture_signal_generator import ArchitectureSignalGenerator, TradingSignal

# Import configuration
from .config.hybrid_regime_config import HybridRegimeConfig, RegimeCombinationStrategy

logger = logging.getLogger(__name__)


class TimeframeType(Enum):
    """Supported timeframe types."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"


@dataclass
class RegimeAnalysisResult:
    """Result from regime analysis."""
    regime_predictions: np.ndarray
    regime_probabilities: np.ndarray
    economic_significance_scores: np.ndarray
    trading_viability_scores: np.ndarray
    regime_stability_scores: np.ndarray
    transition_probabilities: np.ndarray
    tas_contributions: Dict[str, Any]
    nas_contributions: Dict[str, Any]
    hybrid_analysis: Dict[str, Any]
    timeframe_analysis: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class MultiTimeframeResult:
    """Result from multi-timeframe analysis."""
    regime_15m: RegimeAnalysisResult
    trading_1m: Optional[Dict[str, Any]] = None
    trading_5m: Optional[Dict[str, Any]] = None
    timeframe_correlation: Dict[str, float] = field(default_factory=dict)
    cross_timeframe_insights: Dict[str, Any] = field(default_factory=dict)


class EnhancedHybridOrchestrator:
    """
    Enhanced Hybrid Orchestrator that coordinates TAS and NAS systems.
    
    This orchestrator:
    1. Initializes both TAS and NAS systems
    2. Feeds them market data
    3. Gets their outputs and analyzes them
    4. Creates its own regime clusters using unified algorithms
    5. Supports multi-timeframe trading (1m, 5m) while maintaining 15m regime detection
    """
    
    def __init__(self, config: HybridRegimeConfig):
        """Initialize the enhanced hybrid orchestrator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize TAS and NAS integration components
        self.tas_integration = TASIntegrationComponent(config.tas_config)
        self.nas_integration = NASIntegrationComponent(config.nas_config)
        
        # Initialize unified algorithms
        self.search_manager = create_unified_search_manager(config.search_config)
        self.clustering_algorithm = create_unified_clustering_algorithm(config.clustering_config)
        
        # Multi-timeframe support
        self.enable_multi_timeframe = config.get('enable_multi_timeframe', True)
        self.primary_timeframe = TimeframeType.MINUTE_15  # Always 15m for regime detection
        self.trading_timeframes = [TimeframeType.MINUTE_1, TimeframeType.MINUTE_5]

        # Unified architecture search engine
        self.use_unified_search = config.get('use_unified_search', True)
        self.unified_search_engine = None

        # Signal generation system
        self.use_signal_generation = config.get('use_signal_generation', True)
        self.signal_generator = None

        # Results tracking
        self.regime_history = []
        self.tas_history = []
        self.nas_history = []
        self.hybrid_history = []
        
        self.logger.info("✅ Enhanced Hybrid Orchestrator initialized")
        self.logger.info(f"   TAS Integration: ✅ Enabled")
        self.logger.info(f"   NAS Integration: ✅ Enabled")
        self.logger.info(f"   Unified Search Engine: {'✅ Enabled' if self.use_unified_search else '❌ Disabled'}")
        self.logger.info(f"   Signal Generation: {'✅ Enabled' if self.use_signal_generation else '❌ Disabled'}")
        self.logger.info(f"   Multi-timeframe: {'✅ Enabled' if self.enable_multi_timeframe else '❌ Disabled'}")

        # Initialize unified search engine if enabled
        if self.use_unified_search:
            self._initialize_unified_search_engine()

        # Initialize signal generation system if enabled
        if self.use_signal_generation:
            self._initialize_signal_generator()

    def _initialize_unified_search_engine(self):
        """Initialize unified architecture search engine."""
        try:
            # Initialize unified search engine with financial objectives
            search_config = UnifiedSearchConfig(
                architecture_types=[ArchitectureType.NEURAL, ArchitectureType.TREE],
                search_mode=SearchMode.MULTI_OBJECTIVE,
                max_evaluations=1000,
                population_size=50,
                enable_trading_objectives=True,
                sharpe_weight=0.4,
                max_drawdown_weight=0.3,
                win_rate_weight=0.2,
                profit_factor_weight=0.1,
                enable_performance_estimation=True,
                enable_architecture_encoding=True,
                enable_constraint_validation=True
            )
            self.unified_search_engine = UnifiedArchitectureSearchEngine(search_config)

            self.logger.info("✅ Unified Architecture Search Engine initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize unified search engine: {e}")
            self.use_unified_search = False

    def _initialize_signal_generator(self):
        """Initialize architecture-based signal generation system."""
        try:
            from .core.architecture_signal_generator import ArchitectureSignalConfig

            # Create signal generator configuration
            signal_config = ArchitectureSignalConfig(
                signal_threshold=0.6,
                confidence_threshold=0.7,
                ensemble_method="weighted_average",
                enable_signal_validation=True,
                enable_real_time_processing=True
            )

            self.signal_generator = ArchitectureSignalGenerator(signal_config)

            # Mock neural and tree generators for demonstration
            # In practice, these would be trained architectures
            mock_neural = type('MockNeural', (), {})()
            mock_tree = type('MockTree', (), {})()

            self.signal_generator.add_neural_generator(mock_neural)
            self.signal_generator.add_tree_generator(mock_tree)
            self.signal_generator.create_ensemble_generator()

            self.logger.info("✅ Signal Generation System initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize signal generator: {e}")
            self.use_signal_generation = False

    def generate_architecture_signals(self,
                                    market_data: Union[pd.DataFrame, np.ndarray],
                                    regime_data: Optional[Dict[str, Any]] = None) -> List[TradingSignal]:
        """Generate trading signals from discovered architectures."""
        if not self.use_signal_generation or not self.signal_generator:
            self.logger.warning("Signal generation system not available")
            return []

        try:
            self.logger.info("🔄 Generating signals from architectures...")

            # Convert to numpy array if needed
            if isinstance(market_data, pd.DataFrame):
                market_array = market_data.values
            else:
                market_array = market_data

            # Generate signal using ensemble
            signal = self.signal_generator.generate_signal(market_array, regime_data)

            # Get recent signals for analysis
            recent_signals = self.signal_generator.get_recent_signals(5)

            self.logger.info(f"✅ Generated signal: {signal.signal_type.value} with confidence {signal.confidence:.3f}")
            return [signal] + recent_signals

        except Exception as e:
            self.logger.error(f"❌ Signal generation failed: {e}")
            return []

    def get_signal_quality_metrics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get signal quality metrics."""
        if not self.use_signal_generation or not self.signal_generator:
            return {'error': 'Signal generation system not available'}

        try:
            quality_metrics = self.signal_generator.evaluate_signal_quality(market_data)
            signal_stats = self.signal_generator.get_signal_statistics()

            return {
                'quality_metrics': {
                    'accuracy': quality_metrics.accuracy,
                    'precision': quality_metrics.precision,
                    'recall': quality_metrics.recall,
                    'f1_score': quality_metrics.f1_score,
                    'sharpe_ratio': quality_metrics.sharpe_ratio,
                    'max_drawdown': quality_metrics.max_drawdown,
                    'win_rate': quality_metrics.win_rate,
                    'profit_factor': quality_metrics.profit_factor
                },
                'signal_statistics': signal_stats
            }

        except Exception as e:
            return {'error': str(e)}

    def _convert_unified_result_to_regime_analysis(self, search_result) -> RegimeAnalysisResult:
        """Convert unified search result to regime analysis result."""
        try:
            # Extract architecture information
            best_architecture = search_result.best_architecture
            best_score = search_result.best_score
            trading_metrics = search_result.trading_metrics

            # Generate mock regime predictions based on architecture characteristics
            n_samples = 100  # Mock sample count
            n_regimes = 5

            # Generate regime predictions based on architecture type and score
            architecture_type = best_architecture.get('type', 'hybrid')
            if architecture_type == 'neural':
                base_regime = 1  # Neural architectures favor regime 1
            elif architecture_type == 'tree':
                base_regime = 2  # Tree architectures favor regime 2
            else:
                base_regime = 0  # Hybrid architectures favor regime 0

            # Add some variation based on performance score
            regime_variation = int((best_score - 0.5) * 10)  # Convert score to regime variation
            regime_variation = max(-2, min(2, regime_variation))  # Clamp to [-2, 2]

            regime_predictions = np.full(n_samples, base_regime + regime_variation)
            regime_probabilities = np.random.rand(n_samples, n_regimes)
            regime_probabilities = regime_probabilities / regime_probabilities.sum(axis=1, keepdims=True)

            # Generate trading metrics
            economic_scores = np.random.uniform(0.3, 0.9, n_samples)
            trading_scores = np.random.uniform(0.4, 0.8, n_samples)
            stability_scores = np.random.uniform(0.6, 0.95, n_samples)

            # Mock transition probabilities
            transition_probs = np.random.rand(n_regimes, n_regimes)
            transition_probs = transition_probs / transition_probs.sum(axis=1, keepdims=True)

            return RegimeAnalysisResult(
                regime_predictions=regime_predictions,
                regime_probabilities=regime_probabilities,
                economic_significance_scores=economic_scores,
                trading_viability_scores=trading_scores,
                regime_stability_scores=stability_scores,
                transition_probabilities=transition_probs,
                tas_contributions={'unified_search_used': True, 'architecture_type': architecture_type},
                nas_contributions={'unified_search_used': True, 'best_score': best_score},
                hybrid_analysis={'search_result': search_result.__dict__, 'confidence': best_score},
                timeframe_analysis={'primary_timeframe': '15m', 'analysis_type': 'unified_search'},
                execution_time=search_result.execution_time,
                metadata={'unified_analysis': True, 'search_mode': search_result.metadata.get('search_mode', 'unknown')}
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to convert unified result: {e}")
            # Return error result
            return RegimeAnalysisResult(
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                tas_contributions={},
                nas_contributions={},
                hybrid_analysis={'error': str(e)},
                timeframe_analysis={},
                execution_time=0.0,
                metadata={'error': str(e)}
            )

    def analyze_with_unified_search(self,
                                   market_data: Union[pd.DataFrame, np.ndarray],
                                   timestamps: Optional[np.ndarray] = None) -> RegimeAnalysisResult:
        """Analyze market regimes using unified architecture search engine."""
        if not self.use_unified_search:
            self.logger.warning("Unified search engine not available, falling back to traditional analysis")
            return self.analyze_market_regimes(market_data, timestamps, False)

        self.logger.info("🔍 Starting analysis with unified architecture search engine...")

        try:
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data_splits(market_data)

            # Use unified search engine
            if self.unified_search_engine:
                search_result = self.unified_search_engine.search(
                    train_data=(X_train, y_train),
                    validation_data=(X_val, y_val),
                    test_data=(X_test, y_test)
                )

                # Convert unified result to regime analysis result
                combined_result = self._convert_unified_result_to_regime_analysis(search_result)

                self.logger.info("✅ Unified search engine analysis completed")
                return combined_result
            else:
                raise ValueError("Unified search engine not properly initialized")

        except Exception as e:
            self.logger.error(f"❌ Unified search engine analysis failed: {e}")
            # Fallback to traditional analysis
            return self.analyze_market_regimes(market_data, timestamps, False)

    def _prepare_data_splits(self, market_data: Union[pd.DataFrame, np.ndarray]) -> Tuple:
        """Prepare data splits for enhanced engines."""
        # Simplified data preparation - in practice, would be more sophisticated
        if isinstance(market_data, pd.DataFrame):
            X = market_data.drop(columns=['target'], errors='ignore').values
            y = market_data.get('target', np.zeros(len(market_data))).values
        else:
            X = market_data
            y = np.random.randint(0, 3, len(market_data))  # Mock target for demonstration

        n_samples = len(X)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)

        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]

        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _combine_enhanced_results(self, nas_result: Any, tas_result: Any) -> RegimeAnalysisResult:
        """Combine results from enhanced NAS and TAS engines."""
        # Create mock regime analysis result based on enhanced engine outputs
        # In practice, this would be much more sophisticated

        n_samples = 100  # Mock sample count

        regime_predictions = np.random.randint(0, 5, n_samples)  # 5 regimes
        regime_probabilities = np.random.rand(n_samples, 5)
        regime_probabilities = regime_probabilities / regime_probabilities.sum(axis=1, keepdims=True)

        economic_scores = np.random.uniform(0.3, 0.9, n_samples)
        trading_scores = np.random.uniform(0.4, 0.8, n_samples)
        stability_scores = np.random.uniform(0.6, 0.95, n_samples)

        # Mock transition probabilities (5x5 matrix)
        transition_probs = np.random.rand(5, 5)
        transition_probs = transition_probs / transition_probs.sum(axis=1, keepdims=True)

        return RegimeAnalysisResult(
            regime_predictions=regime_predictions,
            regime_probabilities=regime_probabilities,
            economic_significance_scores=economic_scores,
            trading_viability_scores=trading_scores,
            regime_stability_scores=stability_scores,
            transition_probabilities=transition_probs,
            tas_contributions={'enhanced_engine_used': True, 'best_score': tas_result.best_score if tas_result else 0.0},
            nas_contributions={'enhanced_engine_used': True, 'best_score': nas_result.best_score if nas_result else 0.0},
            hybrid_analysis={'combination_method': 'enhanced_engines', 'confidence': 0.85},
            timeframe_analysis={'primary_timeframe': '15m', 'analysis_type': 'enhanced'},
            execution_time=0.0,
            metadata={'enhanced_analysis': True, 'engines_version': '2.0'}
        )
    
    def analyze_market_regimes(self,
                             market_data: Union[pd.DataFrame, np.ndarray],
                             timestamps: Optional[np.ndarray] = None,
                             enable_multi_timeframe: bool = True) -> Union[RegimeAnalysisResult, MultiTimeframeResult]:
        """Analyze market regimes using hybrid TAS-NAS approach."""
        try:
            self.logger.info("🚀 Starting enhanced hybrid regime analysis...")
            start_time = time.time()
            
            # Step 1: Preprocess market data
            processed_data = self._preprocess_market_data(market_data, timestamps)
            
            # Step 2: Run TAS and NAS systems
            tas_result = self._run_tas_analysis(processed_data)
            nas_result = self._run_nas_analysis(processed_data)
            
            # Step 3: Analyze outputs and create hybrid clusters
            hybrid_analysis = self._analyze_tas_nas_outputs(tas_result, nas_result, processed_data)
            hybrid_regimes = self._create_hybrid_regime_clusters(tas_result, nas_result, hybrid_analysis, processed_data)
            
            # Step 4: Multi-timeframe analysis if enabled
            timeframe_analysis = {}
            if enable_multi_timeframe and self.enable_multi_timeframe:
                timeframe_analysis = self._perform_multi_timeframe_analysis(processed_data, hybrid_regimes)
            
            # Step 5: Compile results
            execution_time = time.time() - start_time
            
            regime_result = RegimeAnalysisResult(
                regime_predictions=hybrid_regimes['regime_predictions'],
                regime_probabilities=hybrid_regimes['regime_probabilities'],
                economic_significance_scores=hybrid_regimes['economic_significance_scores'],
                trading_viability_scores=hybrid_regimes['trading_viability_scores'],
                regime_stability_scores=hybrid_regimes['regime_stability_scores'],
                transition_probabilities=hybrid_regimes['transition_probabilities'],
                tas_contributions=tas_result,
                nas_contributions=nas_result,
                hybrid_analysis=hybrid_analysis,
                timeframe_analysis=timeframe_analysis,
                execution_time=execution_time,
                metadata={
                    'orchestrator_version': '2.0.0',
                    'combination_strategy': self.config.combination_strategy.value,
                    'n_regimes': self.config.n_regimes,
                    'data_points': len(processed_data),
                    'timestamp': datetime.now().isoformat(),
                    'multi_timeframe_enabled': enable_multi_timeframe and self.enable_multi_timeframe
                }
            )
            
            # Return multi-timeframe result if enabled
            if enable_multi_timeframe and self.enable_multi_timeframe:
                multi_timeframe_result = MultiTimeframeResult(
                    regime_15m=regime_result,
                    trading_1m=timeframe_analysis.get('1m_trading', {}),
                    trading_5m=timeframe_analysis.get('5m_trading', {}),
                    timeframe_correlation=timeframe_analysis.get('correlations', {}),
                    cross_timeframe_insights=timeframe_analysis.get('insights', {})
                )
                
                self.logger.info("✅ Enhanced hybrid regime analysis completed with multi-timeframe support")
                return multi_timeframe_result
            else:
                self.logger.info("✅ Enhanced hybrid regime analysis completed")
                return regime_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Enhanced hybrid regime analysis failed: {e}")
            
            # Return error result
            error_result = RegimeAnalysisResult(
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                tas_contributions={},
                nas_contributions={},
                hybrid_analysis={},
                timeframe_analysis={},
                execution_time=execution_time,
                metadata={'error': str(e)}
            )
            
            return error_result
    
    def _preprocess_market_data(self, market_data: Union[pd.DataFrame, np.ndarray], timestamps: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Preprocess market data for analysis."""
        try:
            if isinstance(market_data, np.ndarray):
                columns = ['open', 'high', 'low', 'close', 'volume']
                if market_data.shape[1] >= 5:
                    market_data = pd.DataFrame(market_data[:, :5], columns=columns[:market_data.shape[1]])
                else:
                    market_data = pd.DataFrame(market_data, columns=columns[:market_data.shape[1]])
            
            if not isinstance(market_data, pd.DataFrame):
                raise ValueError("Market data must be pandas DataFrame or numpy array")
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in market_data.columns:
                    if col == 'volume':
                        market_data[col] = 1.0  # Default volume
                    else:
                        raise ValueError(f"Required column '{col}' not found in market data")
            
            # Add timestamps if provided
            if timestamps is not None:
                market_data['timestamp'] = timestamps
            elif 'timestamp' not in market_data.columns:
                market_data['timestamp'] = pd.date_range(
                    start=datetime.now().strftime('%Y-%m-%d'),
                    periods=len(market_data),
                    freq='15min'  # Default to 15m for regime detection
                )
            
            # Basic data cleaning
            market_data = market_data.dropna()
            market_data = market_data.replace([np.inf, -np.inf], np.nan).dropna()
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _run_tas_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run TAS regime detection analysis."""
        try:
            tas_features, tas_results = self.tas_integration.extract_features(market_data)
            
            self.tas_history.append({
                'timestamp': datetime.now().isoformat(),
                'features_shape': tas_features.shape,
                'results': tas_results
            })
            
            return {
                'features': tas_features,
                'results': tas_results,
                'method': 'tas_integration',
                'success': True
            }
            
        except Exception as e:
            self.logger.warning(f"TAS analysis failed: {e}")
            return {
                'features': np.array([]),
                'results': {'method': 'fallback', 'error': str(e)},
                'method': 'fallback',
                'success': False
            }
    
    def _run_nas_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run NAS regime detection analysis."""
        try:
            nas_features, nas_results = self.nas_integration.extract_features(market_data)
            
            self.nas_history.append({
                'timestamp': datetime.now().isoformat(),
                'features_shape': nas_features.shape,
                'results': nas_results
            })
            
            return {
                'features': nas_features,
                'results': nas_results,
                'method': 'nas_integration',
                'success': True
            }
            
        except Exception as e:
            self.logger.warning(f"NAS analysis failed: {e}")
            return {
                'features': np.array([]),
                'results': {'method': 'fallback', 'error': str(e)},
                'method': 'fallback',
                'success': False
            }
    
    def _analyze_tas_nas_outputs(self, tas_result: Dict[str, Any], nas_result: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze TAS and NAS outputs to create hybrid insights."""
        try:
            analysis = {
                'tas_contribution': 0.0,
                'nas_contribution': 0.0,
                'agreement_score': 0.0,
                'complementarity_score': 0.0,
                'hybrid_confidence': 0.0,
                'feature_correlation': 0.0,
                'regime_consistency': 0.0
            }
            
            if tas_result['success'] and nas_result['success']:
                tas_features = tas_result['features']
                nas_features = nas_result['features']
                
                # Calculate feature correlation
                if tas_features.size > 0 and nas_features.size > 0:
                    min_len = min(len(tas_features), len(nas_features))
                    tas_subset = tas_features[:min_len]
                    nas_subset = nas_features[:min_len]
                    
                    if tas_subset.ndim > 1 and nas_subset.ndim > 1:
                        tas_flat = tas_subset.flatten()
                        nas_flat = nas_subset.flatten()
                        
                        if len(tas_flat) == len(nas_flat):
                            correlation = np.corrcoef(tas_flat, nas_flat)[0, 1]
                            analysis['feature_correlation'] = abs(correlation) if not np.isnan(correlation) else 0.0
                
                # Calculate agreement and confidence scores
                tas_confidence = tas_result['results'].get('confidence', 0.5)
                nas_confidence = nas_result['results'].get('confidence', 0.5)
                analysis['agreement_score'] = min(tas_confidence, nas_confidence)
                analysis['hybrid_confidence'] = (tas_confidence + nas_confidence) / 2.0
                analysis['complementarity_score'] = 1.0 - analysis['feature_correlation']
                
                # Calculate contribution scores
                total_confidence = tas_confidence + nas_confidence
                if total_confidence > 0:
                    analysis['tas_contribution'] = tas_confidence / total_confidence
                    analysis['nas_contribution'] = nas_confidence / total_confidence
            
            self.hybrid_history.append({
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis
            })
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"TAS-NAS output analysis failed: {e}")
            return {
                'tas_contribution': 0.5,
                'nas_contribution': 0.5,
                'agreement_score': 0.0,
                'complementarity_score': 0.0,
                'hybrid_confidence': 0.0,
                'feature_correlation': 0.0,
                'regime_consistency': 0.0,
                'error': str(e)
            }
    
    def _create_hybrid_regime_clusters(self, tas_result: Dict[str, Any], nas_result: Dict[str, Any], hybrid_analysis: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Create hybrid regime clusters using unified algorithms."""
        try:
            # Combine TAS and NAS features
            combined_features = self._combine_tas_nas_features(tas_result, nas_result, hybrid_analysis)
            
            # Use unified clustering algorithm
            clustering_result = self.clustering_algorithm.cluster_features(
                features=combined_features,
                market_data=market_data,
                economic_weights=None
            )
            
            if not clustering_result.success:
                raise ValueError("Clustering failed")
            
            # Calculate economic significance and trading viability (simplified)
            n_regimes = len(set(clustering_result.labels))
            economic_scores = np.random.uniform(0.3, 0.9, n_regimes)  # Placeholder
            trading_scores = np.random.uniform(0.2, 0.8, n_regimes)  # Placeholder
            stability_scores = np.random.uniform(0.4, 0.9, n_regimes)  # Placeholder
            
            # Calculate transition probabilities
            transition_probs = self._calculate_transition_probabilities(clustering_result.labels, clustering_result.probabilities)
            
            hybrid_regimes = {
                'regime_predictions': clustering_result.labels,
                'regime_probabilities': clustering_result.probabilities,
                'economic_significance_scores': economic_scores,
                'trading_viability_scores': trading_scores,
                'regime_stability_scores': stability_scores,
                'transition_probabilities': transition_probs,
                'clustering_metrics': clustering_result.quality_metrics,
                'algorithm_used': clustering_result.algorithm_used
            }
            
            return hybrid_regimes
            
        except Exception as e:
            self.logger.error(f"Hybrid regime cluster creation failed: {e}")
            raise
    
    def _combine_tas_nas_features(self, tas_result: Dict[str, Any], nas_result: Dict[str, Any], hybrid_analysis: Dict[str, Any]) -> np.ndarray:
        """Combine TAS and NAS features based on analysis."""
        try:
            tas_features = tas_result.get('features', np.array([]))
            nas_features = nas_result.get('features', np.array([]))
            
            if tas_features.size == 0 and nas_features.size == 0:
                raise ValueError("No features available from TAS or NAS")
            
            # Get contribution weights
            tas_weight = hybrid_analysis.get('tas_contribution', 0.5)
            nas_weight = hybrid_analysis.get('nas_contribution', 0.5)
            
            # Normalize weights
            total_weight = tas_weight + nas_weight
            if total_weight > 0:
                tas_weight = tas_weight / total_weight
                nas_weight = nas_weight / total_weight
            else:
                tas_weight = nas_weight = 0.5
            
            # Combine features based on strategy
            if self.config.combination_strategy == RegimeCombinationStrategy.WEIGHTED_AVERAGE:
                if tas_features.size > 0 and nas_features.size > 0:
                    min_len = min(len(tas_features), len(nas_features))
                    tas_subset = tas_features[:min_len]
                    nas_subset = nas_features[:min_len]
                    combined_features = tas_weight * tas_subset + nas_weight * nas_subset
                elif tas_features.size > 0:
                    combined_features = tas_features
                else:
                    combined_features = nas_features
            else:  # Default to concatenation
                if tas_features.size > 0 and nas_features.size > 0:
                    min_len = min(len(tas_features), len(nas_features))
                    tas_subset = tas_features[:min_len]
                    nas_subset = nas_features[:min_len]
                    combined_features = np.hstack([tas_subset, nas_subset])
                elif tas_features.size > 0:
                    combined_features = tas_features
                else:
                    combined_features = nas_features
            
            return combined_features
            
        except Exception as e:
            self.logger.warning(f"Feature combination failed: {e}")
            if tas_result.get('features', np.array([])).size > 0:
                return tas_result['features']
            elif nas_result.get('features', np.array([])).size > 0:
                return nas_result['features']
            else:
                raise ValueError("No features available for combination")
    
    def _calculate_transition_probabilities(self, labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Calculate transition probabilities between regimes."""
        try:
            n_regimes = len(set(labels))
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                transition_matrix[current_regime, next_regime] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            transition_matrix = transition_matrix / row_sums
            
            return transition_matrix
            
        except Exception as e:
            self.logger.warning(f"Transition probability calculation failed: {e}")
            n_regimes = len(set(labels))
            return np.full((n_regimes, n_regimes), 1.0 / n_regimes)
    
    def _perform_multi_timeframe_analysis(self, market_data: pd.DataFrame, hybrid_regimes: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-timeframe analysis for trading."""
        try:
            timeframe_analysis = {
                '1m_trading': {},
                '5m_trading': {},
                'correlations': {},
                'insights': {}
            }
            
            # Analyze 1m trading timeframe
            if TimeframeType.MINUTE_1 in self.trading_timeframes:
                timeframe_analysis['1m_trading'] = self._analyze_trading_timeframe(market_data, hybrid_regimes, TimeframeType.MINUTE_1)
            
            # Analyze 5m trading timeframe
            if TimeframeType.MINUTE_5 in self.trading_timeframes:
                timeframe_analysis['5m_trading'] = self._analyze_trading_timeframe(market_data, hybrid_regimes, TimeframeType.MINUTE_5)
            
            # Calculate correlations and insights
            timeframe_analysis['correlations'] = self._calculate_timeframe_correlations(timeframe_analysis)
            timeframe_analysis['insights'] = self._generate_cross_timeframe_insights(hybrid_regimes, timeframe_analysis)
            
            return timeframe_analysis
            
        except Exception as e:
            self.logger.warning(f"Multi-timeframe analysis failed: {e}")
            return {
                '1m_trading': {},
                '5m_trading': {},
                'correlations': {},
                'insights': {'error': str(e)}
            }
    
    def _analyze_trading_timeframe(self, market_data: pd.DataFrame, hybrid_regimes: Dict[str, Any], timeframe: TimeframeType) -> Dict[str, Any]:
        """Analyze a specific trading timeframe."""
        try:
            # Simplified analysis for demonstration
            trading_analysis = {
                'timeframe': timeframe.value,
                'data_points': len(market_data),
                'regime_alignment': 0.7,  # Placeholder
                'trading_signals': {
                    'buy_signals': 1,
                    'sell_signals': 0,
                    'hold_signals': 0,
                    'signal_strength': 0.8,
                    'confidence': 0.75
                },
                'risk_metrics': {
                    'volatility': 0.02,
                    'max_drawdown': 0.05,
                    'sharpe_ratio': 1.5,
                    'var_95': -0.03
                },
                'opportunity_score': 0.75
            }
            
            return trading_analysis
            
        except Exception as e:
            self.logger.warning(f"Trading timeframe analysis failed for {timeframe.value}: {e}")
            return {
                'timeframe': timeframe.value,
                'error': str(e),
                'opportunity_score': 0.0
            }
    
    def _calculate_timeframe_correlations(self, timeframe_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlations between different timeframes."""
        try:
            correlations = {}
            
            trading_1m = timeframe_analysis.get('1m_trading', {})
            trading_5m = timeframe_analysis.get('5m_trading', {})
            
            if trading_1m and trading_5m:
                score_1m = trading_1m.get('opportunity_score', 0.0)
                score_5m = trading_5m.get('opportunity_score', 0.0)
                
                correlations['opportunity_score_correlation'] = min(abs(score_1m - score_5m), 1.0)
                
                signals_1m = trading_1m.get('trading_signals', {})
                signals_5m = trading_5m.get('trading_signals', {})
                
                strength_1m = signals_1m.get('signal_strength', 0.0)
                strength_5m = signals_5m.get('signal_strength', 0.0)
                
                correlations['signal_strength_correlation'] = min(abs(strength_1m - strength_5m), 1.0)
            
            return correlations
            
        except Exception as e:
            self.logger.warning(f"Timeframe correlation calculation failed: {e}")
            return {}
    
    def _generate_cross_timeframe_insights(self, hybrid_regimes: Dict[str, Any], timeframe_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights across different timeframes."""
        try:
            insights = {
                'optimal_timeframe': '15m',
                'trading_recommendations': [],
                'risk_assessment': 'medium',
                'market_conditions': 'normal'
            }
            
            trading_1m = timeframe_analysis.get('1m_trading', {})
            trading_5m = timeframe_analysis.get('5m_trading', {})
            
            if trading_1m and trading_5m:
                score_1m = trading_1m.get('opportunity_score', 0.0)
                score_5m = trading_5m.get('opportunity_score', 0.0)
                
                if score_1m > score_5m and score_1m > 0.6:
                    insights['optimal_timeframe'] = '1m'
                elif score_5m > score_1m and score_5m > 0.6:
                    insights['optimal_timeframe'] = '5m'
                
                if score_1m > 0.7:
                    insights['trading_recommendations'].append('High opportunity in 1m timeframe')
                if score_5m > 0.7:
                    insights['trading_recommendations'].append('High opportunity in 5m timeframe')
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Cross-timeframe insights generation failed: {e}")
            return {
                'optimal_timeframe': '15m',
                'trading_recommendations': [],
                'risk_assessment': 'medium',
                'market_conditions': 'normal',
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        try:
            status = {
                'orchestrator_version': '2.0.0',
                'tas_integration': {
                    'enabled': True,
                    'history_count': len(self.tas_history),
                    'last_run': self.tas_history[-1]['timestamp'] if self.tas_history else None
                },
                'nas_integration': {
                    'enabled': True,
                    'history_count': len(self.nas_history),
                    'last_run': self.nas_history[-1]['timestamp'] if self.nas_history else None
                },
                'hybrid_analysis': {
                    'history_count': len(self.hybrid_history),
                    'last_run': self.hybrid_history[-1]['timestamp'] if self.hybrid_history else None
                },
                'unified_search_engine': {
                    'enabled': self.use_unified_search,
                    'available': self.unified_search_engine is not None
                },
                'signal_generation_system': {
                    'enabled': self.use_signal_generation,
                    'available': self.signal_generator is not None
                },
                'multi_timeframe_support': self.enable_multi_timeframe,
                'available_algorithms': self.search_manager.get_available_algorithms() if self.search_manager else [],
                'clustering_algorithm': self.clustering_algorithm.algorithm_type if self.clustering_algorithm else 'none',
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.warning(f"System status retrieval failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def create_enhanced_hybrid_orchestrator(config: HybridRegimeConfig) -> EnhancedHybridOrchestrator:
    """Create an enhanced hybrid orchestrator instance."""
    return EnhancedHybridOrchestrator(config)